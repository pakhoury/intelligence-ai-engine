import asyncio
import hashlib
import json
import os
import re
from datetime import timedelta

from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

from circuit_breaker import CircuitBreakerOpenError, llm_circuit_breaker
from config import db_connector, llm, llm_fallback, load_prompt, redis_client
from llm_ops import context_manager
from metrics.compiler import MetricCompiler
from metrics.llm_resolver import LLMMetricResolver
from metrics.loader import load_metrics_catalog
from metrics.registry import MetricDefinition
from observability import CACHE_OPS, REFLECTION_COUNT, REVIEW_SCORE, ROUTE_COUNT, NodeTimer, logger
from retrieval import lexical_search, rrf_fuse
from security import dlp_scan, redact_pii, sanitize_for_prompt
from validators import run_all_validators, validate_sql_safety, validate_sql_scope

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "compliance_docs"

# Hybrid retrieval: candidates fetched per retriever before fusion, and the
# final number of chunks passed downstream after RRF.
RETRIEVAL_CANDIDATES = 20
RETRIEVAL_TOP_N = 6

# Review score at or above which an answer passes without reflection and is
# eligible for caching. Shared with the workflow's after_review edge.
REVIEW_PASS_THRESHOLD = 7.0

_metric_resolver: LLMMetricResolver | None = None
_metric_registry = None


def _get_metric_registry():
    global _metric_registry
    if _metric_registry is None:
        _metric_registry = load_metrics_catalog()
    return _metric_registry


def _get_metric_resolver() -> LLMMetricResolver:
    global _metric_resolver
    if _metric_resolver is None:
        _metric_resolver = LLMMetricResolver(_get_metric_registry(), _ainvoke_llm, load_prompt)
    return _metric_resolver


_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: PGVector | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _pg_connection_string() -> str:
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_db = os.getenv("POSTGRES_DB", "rag_db")
    return f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"


def _get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=_pg_connection_string(),
            embedding_function=_get_embeddings(),
            use_jsonb=True,
        )
    return _vector_store


_pg_engine = None


def _get_pg_engine():
    """Lazy SQLAlchemy engine for lexical (full-text) queries."""
    global _pg_engine
    if _pg_engine is None:
        from sqlalchemy import create_engine
        _pg_engine = create_engine(_pg_connection_string(), pool_size=5, max_overflow=5)
    return _pg_engine


def _format_history(state: dict) -> str:
    """Format conversation history with sanitization for safe prompt injection."""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    lines = []
    for turn in history[-5:]:
        q = sanitize_for_prompt(turn.get("question", ""), max_length=200)
        a = sanitize_for_prompt(turn.get("answer", ""), max_length=500)
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
        context_parts = []
        if turn.get("resolved_metric"):
            version = turn.get("metric_version", "")
            mid = turn["resolved_metric"]
            context_parts.append(f"Metric: {mid}@{version}" if version else f"Metric: {mid}")
        if turn.get("route"):
            context_parts.append(f"Route: {turn['route']}")
        if turn.get("data_summary"):
            summary = sanitize_for_prompt(turn["data_summary"], max_length=200)
            context_parts.append(f"Data: {summary}")
        if context_parts:
            lines.append(f"[{' | '.join(context_parts)}]")
    return "\n".join(lines)


def _cache_key(question: str) -> str:
    """Build a stable cache key from the normalized question text.

    The key is computed from the raw first-turn question only. Follow-up
    questions (any request with conversation history) bypass the cache
    entirely — their meaning depends on context, so caching by text alone
    would leak answers across sessions. Metric params don't need to be in
    the key: they are extracted from the question text itself, so a
    different parameter implies different question text.
    """
    normalized = question.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return f"rag:answer:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"


def _cached_metric_is_stale(payload: dict) -> bool:
    """Check whether a cached answer was built from an outdated metric definition.

    A cached answer derived from a registered metric is only servable while
    the registry still carries the same version of that metric. On any doubt
    (metric gone, version changed, registry unreadable) treat the entry as
    stale — recomputing costs a few LLM calls; serving an outdated compliance
    figure is worse.
    """
    metric_id = payload.get("resolved_metric")
    if not metric_id:
        return False
    try:
        current = _get_metric_registry().get(metric_id)
    except Exception as e:
        logger.warning(
            f"Metric registry unavailable for cache validation: {e}",
            extra={"node": "cache_check", "error": str(e)},
        )
        return True
    if current is None:
        return True
    return current.version != payload.get("metric_version")


def _parse_review_score(content: str) -> float | None:
    """Parse review score from LLM output. Returns None if no score found.

    None (unparseable) must NOT map to 0.0: that would force the full
    reflection loop (2 extra generate+review LLM cycles) on every request
    whenever the reviewer's output format drifts.
    """
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", content)
    if match:
        return min(float(match.group(1)), 10.0)
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b", content)
    if match:
        return min(float(match.group(1)), 10.0)
    return None


async def _ainvoke_llm(prompt: str) -> str:
    # Note: transient-error retries are handled inside ChatGroq (max_retries).
    # A tenacity layer here previously retried only builtin TimeoutError/
    # ConnectionError, which the Groq SDK never raises — it was dead weight.
    try:
        llm_circuit_breaker.before_call()
    except CircuitBreakerOpenError:
        logger.warning("Circuit breaker open, trying fallback chain", extra={"node": "llm"})
        result = await asyncio.to_thread(llm_fallback.invoke_with_fallback, prompt)
        if result is not None:
            return result
        raise
    try:
        response = await llm.ainvoke(prompt)
        llm_circuit_breaker.record_success()
        return response.content
    except Exception:
        llm_circuit_breaker.record_failure()
        raise


async def cache_check(state: dict):
    with NodeTimer("cache_check"):
        if state.get("conversation_history"):
            # Follow-ups are context-dependent: identical text can mean
            # different things in different sessions. Never serve from cache.
            logger.info("Cache BYPASS (follow-up question)", extra={"node": "cache_check"})
            CACHE_OPS.labels(result="bypass").inc()
            return {"cache_hit": False}

        key = _cache_key(state["question"])
        try:
            cached = await asyncio.to_thread(redis_client.get, key)
            if cached:
                try:
                    payload = json.loads(cached)
                    if not isinstance(payload, dict) or "answer" not in payload:
                        payload = {"answer": cached}
                except ValueError:
                    payload = {"answer": cached}  # legacy plain-string entry

                if _cached_metric_is_stale(payload):
                    logger.info(
                        "Cache STALE (metric definition changed), invalidating",
                        extra={
                            "node": "cache_check",
                            "metric": payload.get("resolved_metric"),
                            "cached_version": payload.get("metric_version"),
                        },
                    )
                    CACHE_OPS.labels(result="stale").inc()
                    try:
                        await asyncio.to_thread(redis_client.delete, key)
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete stale cache entry: {e}",
                            extra={"node": "cache_check", "error": str(e)},
                        )
                    return {"cache_hit": False}

                logger.info("Cache HIT", extra={"node": "cache_check", "cache_hit": True})
                CACHE_OPS.labels(result="hit").inc()
                result = {
                    "cache_hit": True,
                    "final_answer": payload["answer"],
                    "review_score": payload.get("review_score", 10.0),
                    "confidence": payload.get("confidence", "unknown"),
                }
                for field in ("resolved_metric", "metric_version", "metric_owner",
                              "metric_approval", "route", "data_path"):
                    if payload.get(field):
                        result[field] = payload[field]
                return result
        except Exception as e:
            logger.warning(f"Redis error: {e}", extra={"node": "cache_check", "error": str(e)})
        logger.info("Cache MISS", extra={"node": "cache_check", "cache_hit": False})
        CACHE_OPS.labels(result="miss").inc()
        return {"cache_hit": False}


async def context_resolver(state: dict):
    """Rewrite follow-up questions into standalone queries using conversation history.

    On first turn (no history), passes through unchanged. On follow-ups, the LLM
    resolves references like "that", "those", "break it down" into a self-contained
    question that downstream nodes can process without needing conversation context.
    """
    with NodeTimer("context_resolver"):
        original = state["question"]
        history = state.get("conversation_history", [])

        if not history:
            return {"original_question": original}

        formatted = _format_history(state)
        prompt = load_prompt("context_resolver.txt").format(
            question=original,
            conversation_history=formatted,
        )
        try:
            rewritten = (await _ainvoke_llm(prompt)).strip()
            if not rewritten:
                rewritten = original

            if rewritten.lower() != original.lower():
                logger.info(
                    "Question rewritten for context",
                    extra={
                        "node": "context_resolver",
                        "original": original,
                        "rewritten": rewritten,
                    },
                )
            return {"original_question": original, "question": rewritten}
        except Exception as e:
            logger.warning(
                f"Context resolution failed, using original: {e}",
                extra={"node": "context_resolver", "error": str(e)},
            )
            return {"original_question": original}


VALID_ROUTES = {"sql_only", "docs_only", "docs_then_sql", "sql_then_docs", "parallel"}


async def router_node(state: dict):
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


async def clarification_node(state: dict):
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


async def extract_node(state: dict):
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


def _build_metric_context(metric: MetricDefinition, extracted_params: dict) -> str:
    effective_params = metric.get_default_params()
    effective_params.update(extracted_params)

    lines = [
        f"Metric: {metric.name} ({metric.qualified_id})",
        f"Formula: {metric.formula}",
        f"Unit: {metric.unit}",
        f"Tables: {', '.join(metric.tables)}",
    ]
    if metric.owner:
        lines.append(f"Owner: {metric.owner}" + (f" | Approval: {metric.approval}" if metric.approval else ""))

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


async def metric_resolver_node(state: dict):
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
            "metric_owner": resolved.metric.owner,
            "metric_approval": resolved.metric.approval,
            "metric_context": metric_context,
            "metric_params": resolved.extracted_params,
        }


def _strip_sql_fences(sql: str) -> str:
    """Remove markdown code fences from LLM-generated SQL."""
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


MAX_SQL_RETRIES = 2


_allowed_tables_cache: set[str] | None = None


def _allowed_tables() -> set[str]:
    """Catalog table allowlist for SQL scope validation, cached per process.

    An unreadable catalog returns an empty set (scope check is skipped)
    rather than blocking every query; the failure is not cached so the
    allowlist recovers as soon as the catalog does.
    """
    global _allowed_tables_cache
    if _allowed_tables_cache is None:
        try:
            _allowed_tables_cache = {t.upper() for t in db_connector.get_allowed_tables()}
        except Exception as e:
            logger.warning(
                f"Table allowlist unavailable, SQL scope check skipped: {e}",
                extra={"node": "sql_path", "error": str(e)},
            )
            return set()
    return _allowed_tables_cache


_compiler = MetricCompiler()


async def _try_compiled_metric(state: dict) -> dict | None:
    """Try deterministic metric compilation. Returns result dict or None to fall through to LLM."""
    resolved_metric_id = state.get("resolved_metric")
    if not resolved_metric_id:
        return None

    metric = _get_metric_registry().get(resolved_metric_id)
    if not metric or not metric.is_compilable:
        return None

    params = state.get("metric_params") or {}

    compiled = _compiler.compile(metric, params)
    if not compiled:
        return None

    logger.info(
        f"Compiled metric SQL deterministically ({compiled.compilation_mode}): {compiled.sql}",
        extra={
            "node": "sql_path",
            "metric": compiled.metric_id,
            "version": compiled.version,
            "owner": compiled.owner,
            "approval": compiled.approval,
            "mode": compiled.compilation_mode,
            "sql": compiled.sql,
        },
    )

    sql_result = await asyncio.to_thread(db_connector.execute_query, compiled.sql)

    if sql_result.startswith(("Execution Error:", "Error:")):
        logger.warning(
            f"Compiled SQL execution failed, falling back to LLM: {sql_result}",
            extra={"node": "sql_path", "error": sql_result},
        )
        return None

    sql_result = redact_pii(sql_result)
    return {
        "sql_result": f"Query:\n{compiled.sql}\n\nResult:\n{sql_result}",
        "compiled_metric": True,
    }


async def sql_path(state: dict):
    with NodeTimer("sql_path"):
        question = state["question"]

        compiled_result = await _try_compiled_metric(state)
        if compiled_result is not None:
            return compiled_result

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
                "No specific metric was identified for this question. "
                "Generate SQL based on the schema and question alone."
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

            scope = validate_sql_scope(sql, _allowed_tables())
            if not scope.passed:
                last_error = "; ".join(scope.failures)
                logger.warning(
                    f"SQL scope check failed (attempt {attempt + 1}): {last_error}",
                    extra={"node": "sql_path", "error": last_error},
                )
                if attempt < MAX_SQL_RETRIES:
                    continue
                return {"sql_result": f"Error: SQL blocked by scope validator — {last_error}"}

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


DOC_TYPE_HINTS = {
    "docs_then_sql": "policy",
    "sql_then_docs": "policy",
    "docs_only": None,
    "parallel": None,
}


def _build_doc_filter(state: dict) -> dict:
    """Build PGVector metadata filter from route context."""
    filters = {}
    route = state.get("route", "")
    doc_type = DOC_TYPE_HINTS.get(route)
    if doc_type:
        filters["doc_type"] = doc_type
    return filters


async def vector_retrieval(state: dict):
    """Hybrid retrieval: vector similarity + Postgres full-text, fused with RRF.

    Embeddings handle paraphrase; full-text handles exact identifiers
    (incident IDs, violation codes, regulation sections) that carry no
    embedding signal. Either retriever failing degrades gracefully to the
    other; both failing returns no docs, as before.
    """
    with NodeTimer("vector_retrieval"):
        question = state["question"]
        extracted = state.get("extracted_context")
        search_query = extracted if extracted else question
        logger.info(f"Searching documents for: {search_query}", extra={"node": "vector_retrieval"})

        metadata_filter = _build_doc_filter(state)
        search_kwargs = {"k": RETRIEVAL_CANDIDATES}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
            logger.info(
                f"Applying doc filter: {metadata_filter}",
                extra={"node": "vector_retrieval", "filter": str(metadata_filter)},
            )

        async def _vector():
            store = _get_vector_store()
            return await asyncio.to_thread(store.similarity_search, search_query, **search_kwargs)

        async def _lexical():
            return await asyncio.to_thread(
                lexical_search,
                _get_pg_engine(),
                search_query,
                COLLECTION_NAME,
                RETRIEVAL_CANDIDATES,
                metadata_filter.get("doc_type"),
            )

        vector_results, lexical_results = await asyncio.gather(
            _vector(), _lexical(), return_exceptions=True,
        )

        if isinstance(vector_results, BaseException):
            logger.error(
                f"Vector retrieval failed: {vector_results}",
                extra={"node": "vector_retrieval", "error": str(vector_results)},
            )
            vector_results = []
        if isinstance(lexical_results, BaseException):
            logger.warning(
                f"Lexical retrieval failed, using vector-only: {lexical_results}",
                extra={"node": "vector_retrieval", "error": str(lexical_results)},
            )
            lexical_results = []

        fused = rrf_fuse([vector_results, lexical_results], top_n=RETRIEVAL_TOP_N)

        docs = []
        for doc in fused:
            source = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
            docs.append(f"[{source}]: {doc.page_content}")
        logger.info(
            f"Hybrid retrieval: {len(vector_results)} vector + {len(lexical_results)} lexical "
            f"-> {len(docs)} fused chunks",
            extra={"node": "vector_retrieval"},
        )
        return {"retrieved_docs": docs}


async def answer_generator(state: dict):
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


def _compute_confidence(state: dict, review_score: float, validator_failures: list) -> str:
    compiled = state.get("compiled_metric", False)
    retries = state.get("reflection_attempt", 0)
    has_sql = bool(state.get("sql_result", "")) and not state.get("sql_result", "").startswith(("Error", "N/A"))
    has_docs = bool(state.get("retrieved_docs"))

    if compiled and review_score >= 8.0 and not validator_failures:
        return "high"
    if compiled and review_score >= 6.0:
        return "high"
    if review_score >= 8.0 and not validator_failures and (has_sql or has_docs):
        return "medium"
    if review_score >= 6.0 and retries == 0:
        return "medium"
    return "low"


async def reviewer_node(state: dict):
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
        parsed_score = _parse_review_score(content)
        if parsed_score is None:
            logger.warning(
                "Review score unparseable, defaulting to 7.0 (validators still apply)",
                extra={"node": "reviewer", "error": "unparseable_review_score"},
            )
            score = 7.0
        else:
            score = parsed_score

        sql_result = state.get("sql_result", "")
        sql_query = ""
        if sql_result and "Query:\n" in sql_result:
            sql_query = sql_result.split("Query:\n", 1)[1].split("\n\nResult:\n", 1)[0]

        _, validator_failures, score_cap, unfixable_cap = run_all_validators(
            sql=sql_query,
            sql_result=sql_result,
            answer=state.get("final_answer", ""),
            retrieved_docs=state.get("retrieved_docs", []),
            route=state.get("route", ""),
            resolved_metric=state.get("resolved_metric", ""),
            compiled_metric=state.get("compiled_metric", False),
            allowed_tables=_allowed_tables(),
        )

        if validator_failures:
            logger.warning(
                f"Deterministic validation failures: {validator_failures}",
                extra={"node": "reviewer", "validator_failures": validator_failures},
            )
            score = min(score, score_cap)
            content += f"\n\n[Validator]: {'; '.join(validator_failures)}"

        # A validator cap rooted in the data (SQL error, empty result, no
        # supporting evidence) re-applies on every reflection cycle, so if it
        # already pins the score below the pass threshold, regenerating the
        # answer can never succeed — skip the loop instead of burning cycles.
        skip_reflection = unfixable_cap < REVIEW_PASS_THRESHOLD
        if skip_reflection:
            logger.info(
                f"Skipping reflection: unfixable data failure caps score at {unfixable_cap}",
                extra={"node": "reviewer", "unfixable_cap": unfixable_cap},
            )

        result = {
            "review_score": score,
            "reflection_attempt": state.get("reflection_attempt", 0) + 1,
            "reviewer_feedback": content.strip(),
            "skip_reflection": skip_reflection,
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

        result["confidence"] = _compute_confidence(state, result["review_score"], validator_failures)

        logger.info(f"Review score: {result['review_score']}", extra={"node": "reviewer", "score": result["review_score"]})
        REVIEW_SCORE.observe(result["review_score"])
        return result


EXPLORATORY_NOTICE = (
    "\n\n---\n"
    "⚠️ Exploratory result: the figures above were produced by AI-generated SQL, "
    "not a governed metric definition. Verify independently before regulatory or audit use."
)


def _data_path(state: dict) -> str:
    """Classify the provenance of the answer's data, for labeling and audit.

    "governed"    — numbers came from a compiled, versioned metric definition
    "exploratory" — numbers came from LLM-generated SQL (lower trust)
    "documents"   — answer is grounded in retrieved documents only
    "none"        — no supporting data was retrieved
    """
    if state.get("compiled_metric"):
        return "governed"
    sql_result = state.get("sql_result", "")
    if sql_result and not sql_result.startswith(("Error", "Execution Error", "N/A")):
        return "exploratory"
    if state.get("retrieved_docs"):
        return "documents"
    return "none"


async def cache_write(state: dict):
    with NodeTimer("cache_write"):
        data_path = _data_path(state)
        updates: dict = {"data_path": data_path}

        answer = state.get("final_answer", "")
        if answer and data_path == "exploratory":
            # Numbers reached the user via the ungoverned LLM-SQL path —
            # the answer itself must say so, not just response metadata.
            answer = answer + EXPLORATORY_NOTICE
            updates["final_answer"] = answer

        if state.get("conversation_history"):
            # Mirror of cache_check: context-dependent answers are never cached.
            return updates
        if state.get("review_score", 0) >= REVIEW_PASS_THRESHOLD and answer:
            # Key on the ORIGINAL question text — the same key cache_check
            # computes on the next identical request.
            key = _cache_key(state.get("original_question") or state["question"])
            payload = json.dumps({
                "answer": answer,
                "review_score": state.get("review_score"),
                "confidence": state.get("confidence", "unknown"),
                "resolved_metric": state.get("resolved_metric"),
                "metric_version": state.get("metric_version"),
                "metric_owner": state.get("metric_owner"),
                "metric_approval": state.get("metric_approval"),
                "route": state.get("route"),
                "data_path": data_path,
            })
            try:
                await asyncio.to_thread(
                    redis_client.set, key, payload, timedelta(hours=24),
                )
                logger.info("Cached answer", extra={"node": "cache_write", "cache_key": key})
            except Exception as e:
                logger.warning(f"Cache write failed: {e}", extra={"node": "cache_write", "error": str(e)})
        return updates
