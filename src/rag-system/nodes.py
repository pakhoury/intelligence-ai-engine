import asyncio
import hashlib
import os
import re
from config import llm, db_connector, redis_client, load_prompt
from datetime import timedelta
from typing import Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from observability import logger, NodeTimer, CACHE_OPS, ROUTE_COUNT, REVIEW_SCORE, REFLECTION_COUNT
from security import sanitize_for_prompt, redact_pii, dlp_scan
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from llm_ops import context_manager
from metrics import MetricsCatalog
from metrics.loader import load_metrics_catalog
from metrics.llm_resolver import LLMMetricResolver
from metrics.registry import MetricDefinition
from circuit_breaker import llm_circuit_breaker, CircuitBreakerOpen
from validators import validate_sql_safety, run_all_validators

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_metrics_catalog: Optional[MetricsCatalog] = None
_metric_resolver: Optional[LLMMetricResolver] = None


def _get_metrics_catalog() -> MetricsCatalog:
    global _metrics_catalog
    if _metrics_catalog is None:
        _metrics_catalog = MetricsCatalog()
    return _metrics_catalog


def _get_metric_resolver() -> LLMMetricResolver:
    global _metric_resolver
    if _metric_resolver is None:
        registry = load_metrics_catalog()
        _metric_resolver = LLMMetricResolver(registry, _ainvoke_llm, load_prompt)
    return _metric_resolver


_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[PGVector] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_db = os.getenv("POSTGRES_DB", "rag_db")
        connection = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"
        _vector_store = PGVector(
            collection_name="compliance_docs",
            connection_string=connection,
            embedding_function=_get_embeddings(),
            use_jsonb=True,
        )
    return _vector_store


def _format_history(state: Dict) -> str:
    """Format conversation history with sanitization for safe prompt injection."""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    lines = []
    for turn in history[-5:]:
        q = sanitize_for_prompt(turn.get("question", ""), max_length=200)
        a = sanitize_for_prompt(turn.get("answer", ""), max_length=300)
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def _cache_key(question: str) -> str:
    """Normalize question into a stable cache key."""
    normalized = question.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return f"rag:answer:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"


def _parse_review_score(content: str) -> float:
    """Parse review score from LLM output with multiple fallback patterns."""
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", content)
    if match:
        return min(float(match.group(1)), 10.0)
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b", content)
    if match:
        return min(float(match.group(1)), 10.0)
    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
async def _ainvoke_llm(prompt: str) -> str:
    llm_circuit_breaker.before_call()
    try:
        response = await llm.ainvoke(prompt)
        llm_circuit_breaker.record_success()
        return response.content
    except CircuitBreakerOpen:
        raise
    except Exception as e:
        llm_circuit_breaker.record_failure()
        raise


async def cache_check(state: Dict):
    with NodeTimer("cache_check"):
        question = state["question"]
        key = _cache_key(question)
        try:
            cached = await asyncio.to_thread(redis_client.get, key)
            if cached:
                logger.info("Cache HIT", extra={"node": "cache_check", "cache_hit": True})
                CACHE_OPS.labels(result="hit").inc()
                return {"cache_hit": True, "final_answer": cached, "review_score": 10.0}
        except Exception as e:
            logger.warning(f"Redis error: {e}", extra={"node": "cache_check", "error": str(e)})
        logger.info("Cache MISS", extra={"node": "cache_check", "cache_hit": False})
        CACHE_OPS.labels(result="miss").inc()
        return {"cache_hit": False}


VALID_ROUTES = {"sql_only", "docs_only", "docs_then_sql", "sql_then_docs", "parallel"}


async def router_node(state: Dict):
    with NodeTimer("router"):
        history = _format_history(state)
        prompt = load_prompt("router.txt").format(
            question=state["question"],
            conversation_history=history or "No prior conversation."
        )
        content = await _ainvoke_llm(prompt)

        normalized = content.strip().upper().replace(" ", "_")

        if normalized in {v.upper() for v in VALID_ROUTES}:
            route = normalized.lower()
        elif "DOCS_THEN_SQL" in normalized:
            route = "docs_then_sql"
        elif "SQL_THEN_DOCS" in normalized:
            route = "sql_then_docs"
        elif "PARALLEL" in normalized or "BOTH" in normalized:
            route = "parallel"
        elif "SQL" in normalized and "DOC" not in normalized:
            route = "sql_only"
        elif "DOC" in normalized:
            route = "docs_only"
        else:
            route = "parallel"

        logger.info(f"Route decided: {route}", extra={"node": "router", "route": route})
        ROUTE_COUNT.labels(route=route).inc()
        return {"route": route}


async def clarification_node(state: Dict):
    with NodeTimer("clarify"):
        history = _format_history(state)
        prompt = load_prompt("clarification.txt").format(
            question=state["question"],
            conversation_history=history or "No prior conversation.",
        )
        content = await _ainvoke_llm(prompt)
        if "NO_CLARIFICATION_NEEDED" in content.upper():
            logger.info("No clarification needed", extra={"node": "clarify"})
            return {"needs_clarification": False}
        logger.info("Clarification needed", extra={"node": "clarify"})
        return {
            "needs_clarification": True,
            "final_answer": content.strip(),
            "review_score": 10.0
        }


EXTRACT_INSTRUCTIONS = {
    "docs_then_sql": (
        "Extract the formula, definition, calculation method, or criteria from the "
        "documents that is needed to answer the question using a database query."
    ),
    "sql_then_docs": (
        "Extract the key entity names, types, or specific terms from the data results "
        "that should be used to search for relevant policies or documents."
    ),
}


async def extract_node(state: Dict):
    with NodeTimer("extract"):
        route = state.get("route", "")
        question = state["question"]

        if route == "docs_then_sql":
            source_material = "\n".join(state.get("retrieved_docs", []))
        elif route == "sql_then_docs":
            source_material = state.get("sql_result", "")
        else:
            return {}

        if not source_material or source_material.startswith("Error"):
            logger.warning("Extract skipped: no usable source material", extra={"node": "extract"})
            return {"extracted_context": ""}

        instruction = EXTRACT_INSTRUCTIONS[route]
        prompt = load_prompt("extract.txt").format(
            question=question,
            source_material=source_material,
            instruction=instruction,
        )
        content = await _ainvoke_llm(prompt)

        if "EXTRACTION_FAILED" in content.upper():
            logger.warning("Extraction failed, proceeding without context", extra={"node": "extract"})
            return {"extracted_context": ""}

        logger.info(f"Extracted context: {content.strip()[:200]}", extra={"node": "extract"})
        return {"extracted_context": content.strip()}


def _build_metric_context(metric: MetricDefinition, extracted_params: Dict) -> str:
    effective_params = metric.get_default_params()
    effective_params.update(extracted_params)

    lines = [
        f"Metric: {metric.name} ({metric.qualified_id})",
        f"Formula: {metric.formula}",
        f"Unit: {metric.unit}",
        f"Tables: {', '.join(metric.tables)}",
    ]

    if metric.steps:
        lines.append("\nCalculation Steps:")
        for step in metric.steps:
            lines.append(f"  {step.get('step', '?')}. {step.get('action', '')}")
            if step.get("uses"):
                lines.append(f"     Uses: {step['uses']}")
            if step.get("note"):
                lines.append(f"     Note: {step['note']}")

    if effective_params:
        lines.append("\nParameters (extracted from user question):")
        for name, value in effective_params.items():
            source = "user" if name in extracted_params else "default"
            lines.append(f"  {name} = {value} ({source})")

    if metric.thresholds:
        lines.append("\nThresholds:")
        for level, t in metric.thresholds.items():
            lines.append(f"  {level}: {t.get('label', '')}")

    if metric.interpretation:
        lines.append(f"\nInterpretation: {metric.interpretation.strip()}")

    return "\n".join(lines)


async def metric_resolver_node(state: Dict):
    with NodeTimer("metric_resolver"):
        question = state["question"]
        resolver = _get_metric_resolver()
        resolved = await resolver.resolve(question)

        if resolved is None:
            logger.info(
                "No metric matched — falling back to catalog-driven SQL",
                extra={"node": "metric_resolver", "resolved": False},
            )
            return {"resolved_metric": None}

        metric_context = _build_metric_context(
            resolved.metric, resolved.extracted_params,
        )

        logger.info(
            f"Metric resolved: {resolved.metric.qualified_id} "
            f"(confidence={resolved.confidence:.2f})",
            extra={
                "node": "metric_resolver",
                "metric": resolved.metric.metric_id,
                "version": resolved.metric.version,
                "confidence": resolved.confidence,
                "params": resolved.extracted_params,
            },
        )
        return {
            "resolved_metric": resolved.metric.metric_id,
            "metric_version": resolved.metric.version,
            "metric_context": metric_context,
        }


def _strip_sql_fences(sql: str) -> str:
    """Remove markdown code fences from LLM-generated SQL."""
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


MAX_SQL_RETRIES = 2


async def sql_path(state: Dict):
    with NodeTimer("sql_path"):
        question = state["question"]

        logger.info(f"Generating SQL via LLM for: {question}", extra={"node": "sql_path"})

        try:
            table_prompt = db_connector.build_table_selection_prompt(question)
            table_response = await _ainvoke_llm(table_prompt)
            tables = db_connector.parse_table_selection(table_response)
        except Exception as e:
            logger.warning(
                f"LLM table selection failed, using keyword fallback: {e}",
                extra={"node": "sql_path", "error": str(e)},
            )
            tables = await asyncio.to_thread(db_connector.get_relevant_tables, question)
        schema = await asyncio.to_thread(db_connector.get_table_schema, tables)

        history = _format_history(state)

        metric_context = state.get("metric_context")
        if metric_context:
            metric_section = (
                "Resolved Metric Definition (follow this formula, "
                "adapting for any user modifiers):\n"
                f"{metric_context}"
            )
        else:
            metric_section = (
                "Known Metric Definitions:\n"
                f"{_get_metrics_catalog().build_all_context()}"
            )

        base_prompt = load_prompt("sql_agent.txt").format(
            question=question,
            schema_description=schema,
            metric_section=metric_section,
            conversation_history=history or "No prior conversation.",
        )

        extracted = state.get("extracted_context")
        if extracted:
            base_prompt += (
                f"\n\nA prior research step found this relevant definition/context:\n"
                f"{extracted}\n\n"
                f"Use this to inform your SQL query — it tells you which columns "
                f"and calculations to use."
            )

        last_error = None

        for attempt in range(MAX_SQL_RETRIES + 1):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = (
                    f"{base_prompt}\n\n"
                    f"Your previous SQL attempt failed:\n{last_error}\n\n"
                    f"Fix the issue and return ONLY the corrected SQL query."
                )

            content = await _ainvoke_llm(prompt)
            sql = _strip_sql_fences(content.strip())
            logger.info(
                f"Generated SQL (attempt {attempt + 1}): {sql}",
                extra={"node": "sql_path", "sql": sql},
            )

            if "NO_SQL_POSSIBLE" in sql.upper():
                return {"sql_result": "N/A - query cannot be answered with available tables"}

            safety = validate_sql_safety(sql)
            if not safety.passed:
                last_error = "; ".join(safety.failures)
                logger.warning(
                    f"SQL safety check failed (attempt {attempt + 1}): {last_error}",
                    extra={"node": "sql_path", "error": last_error},
                )
                if attempt < MAX_SQL_RETRIES:
                    continue
                return {"sql_result": f"Error: SQL blocked by safety validator — {last_error}"}

            col_error = await asyncio.to_thread(db_connector.validate_columns, sql, tables)
            if col_error:
                last_error = col_error
                logger.warning(
                    f"SQL column validation failed (attempt {attempt + 1}): {col_error}",
                    extra={"node": "sql_path", "error": col_error},
                )
                if attempt < MAX_SQL_RETRIES:
                    continue
                return {"sql_result": f"Error: {last_error}"}

            sql_result = await asyncio.to_thread(db_connector.execute_query, sql)

            if sql_result.startswith(("Execution Error:", "Error:")):
                last_error = sql_result
                logger.warning(
                    f"SQL execution failed (attempt {attempt + 1}): {sql_result}",
                    extra={"node": "sql_path", "error": sql_result},
                )
                if attempt < MAX_SQL_RETRIES:
                    continue
                return {"sql_result": sql_result}

            sql_result = redact_pii(sql_result)
            logger.info(f"SQL result: {sql_result[:200]}", extra={"node": "sql_path"})
            return {"sql_result": f"Query:\n{sql}\n\nResult:\n{sql_result}"}


async def vector_retrieval(state: Dict):
    with NodeTimer("vector_retrieval"):
        question = state["question"]
        extracted = state.get("extracted_context")
        search_query = extracted if extracted else question
        logger.info(f"Searching documents for: {search_query}", extra={"node": "vector_retrieval"})

        try:
            store = _get_vector_store()
            results = await asyncio.to_thread(store.similarity_search, search_query, k=4)
            docs = []
            for doc in results:
                source = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
                docs.append(f"[{source}]: {doc.page_content}")
            logger.info(f"Found {len(docs)} relevant chunks", extra={"node": "vector_retrieval"})
            return {"retrieved_docs": docs}
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}", extra={"node": "vector_retrieval", "error": str(e)})
            return {"retrieved_docs": []}


async def answer_generator(state: Dict):
    with NodeTimer("answer_generator"):
        route = state.get("route", "documents")
        attempt = state.get("reflection_attempt", 0)
        logger.info(f"Generating answer (route={route}, reflection={attempt})", extra={"node": "answer_generator", "route": route})

        if attempt > 0:
            REFLECTION_COUNT.inc()

        history = _format_history(state)
        prompt_template = load_prompt("final_answer.txt")
        sql_result = state.get("sql_result", "N/A - no SQL data retrieved")
        docs = "\n".join(state.get("retrieved_docs", [])) or "N/A - no documents retrieved"

        fitted = context_manager.truncate_to_fit(
            prompt_template=prompt_template,
            question=state["question"],
            sql_result=sql_result,
            docs=docs,
            history=history or "No prior conversation.",
        )

        prompt = prompt_template.format(
            question=state["question"],
            sql_result=fitted.get("sql_result", sql_result),
            context=fitted.get("docs", docs),
            conversation_history=fitted.get("history", history or "No prior conversation."),
        )

        metric_context = state.get("metric_context")
        if metric_context:
            prompt += (
                f"\n\nThis answer is based on a registered metric:\n"
                f"{metric_context}\n"
                f"Reference the metric name and version in your answer for auditability."
            )

        if attempt > 0 and state.get("reviewer_feedback"):
            prompt += (
                f"\n\nYour previous answer was reviewed and scored below the quality threshold.\n"
                f"Reviewer feedback:\n{state['reviewer_feedback']}\n\n"
                f"Please provide an improved answer addressing the reviewer's concerns."
            )

        content = await _ainvoke_llm(prompt)
        return {"final_answer": content}


async def reviewer_node(state: Dict):
    with NodeTimer("reviewer"):
        history = _format_history(state)
        prompt = load_prompt("reviewer.txt").format(
            question=state["question"],
            sql_result=state.get("sql_result", ""),
            docs=state.get("retrieved_docs", ""),
            answer=state.get("final_answer", ""),
            conversation_history=history or "No prior conversation.",
        )
        content = await _ainvoke_llm(prompt)
        score = _parse_review_score(content)

        sql_result = state.get("sql_result", "")
        sql_query = ""
        if sql_result and "Query:\n" in sql_result:
            sql_query = sql_result.split("Query:\n", 1)[1].split("\n\nResult:\n", 1)[0]

        _, validator_failures, score_cap = run_all_validators(
            sql=sql_query,
            sql_result=sql_result,
            answer=state.get("final_answer", ""),
            retrieved_docs=state.get("retrieved_docs", []),
            route=state.get("route", ""),
        )

        if validator_failures:
            logger.warning(
                f"Deterministic validation failures: {validator_failures}",
                extra={"node": "reviewer", "validator_failures": validator_failures},
            )
            score = min(score, score_cap)
            content += f"\n\n[Validator]: {'; '.join(validator_failures)}"

        result = {
            "review_score": score,
            "reflection_attempt": state.get("reflection_attempt", 0) + 1,
            "reviewer_feedback": content.strip(),
        }

        answer = state.get("final_answer", "")
        scan = dlp_scan(answer)
        if scan["pii_found"]:
            logger.warning(
                "DLP: PII detected in answer, redacting",
                extra={"node": "reviewer", "findings": scan["findings"]},
            )
            result["final_answer"] = scan["clean_text"]
            result["review_score"] = min(score, 6.0)

        logger.info(f"Review score: {result['review_score']}", extra={"node": "reviewer", "score": result["review_score"]})
        REVIEW_SCORE.observe(result["review_score"])
        return result


async def cache_write(state: Dict):
    with NodeTimer("cache_write"):
        if state.get("review_score", 0) >= 7.0 and state.get("final_answer"):
            key = _cache_key(state["question"])
            try:
                await asyncio.to_thread(
                    redis_client.set, key, state["final_answer"], timedelta(hours=24),
                )
                logger.info("Cached answer", extra={"node": "cache_write"})
            except Exception as e:
                logger.warning(f"Cache write failed: {e}", extra={"node": "cache_write", "error": str(e)})
        return {}
