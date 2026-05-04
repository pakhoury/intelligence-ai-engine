import asyncio
import hashlib
import os
import re
from config import llm, db_connector, redis_client, load_prompt
from datetime import timedelta
from typing import Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from observability import logger, NodeTimer, CACHE_OPS, ROUTE_COUNT, REVIEW_SCORE
from security import sanitize_for_prompt, redact_pii, dlp_scan
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from llm_ops import context_manager

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
        connection = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"
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
    return 7.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
async def _ainvoke_llm(prompt: str) -> str:
    response = await llm.ainvoke(prompt)
    return response.content


def cache_check(state: Dict):
    with NodeTimer("cache_check"):
        question = state["question"]
        key = _cache_key(question)
        try:
            cached = redis_client.get(key)
            if cached:
                logger.info("Cache HIT", extra={"node": "cache_check", "cache_hit": True})
                CACHE_OPS.labels(result="hit").inc()
                return {"cache_hit": True, "final_answer": cached, "review_score": 10.0}
        except Exception as e:
            logger.warning(f"Redis error: {e}", extra={"node": "cache_check", "error": str(e)})
        logger.info("Cache MISS", extra={"node": "cache_check", "cache_hit": False})
        CACHE_OPS.labels(result="miss").inc()
        return {"cache_hit": False}


def router_node(state: Dict):
    with NodeTimer("router"):
        history = _format_history(state)
        prompt = load_prompt("router.txt").format(
            question=state["question"],
            conversation_history=history or "No prior conversation."
        )
        response = llm.invoke(prompt)
        content = response.content.strip().upper()

        if "BOTH" in content:
            route = "both"
        elif "SQL" in content:
            route = "sql"
        else:
            route = "documents"

        logger.info(f"Route decided: {route}", extra={"node": "router", "route": route})
        ROUTE_COUNT.labels(route=route).inc()
        return {"route": route}


def clarification_node(state: Dict):
    with NodeTimer("clarify"):
        prompt = load_prompt("clarification.txt").format(question=state["question"])
        response = llm.invoke(prompt)
        if "NO_CLARIFICATION_NEEDED" in response.content.upper():
            logger.info("No clarification needed", extra={"node": "clarify"})
            return {"needs_clarification": False}
        logger.info("Clarification needed", extra={"node": "clarify"})
        return {
            "needs_clarification": True,
            "final_answer": response.content.strip(),
            "review_score": 10.0
        }


def _strip_sql_fences(sql: str) -> str:
    """Remove markdown code fences from LLM-generated SQL."""
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


MAX_SQL_RETRIES = 2


def sql_path(state: Dict):
    with NodeTimer("sql_path"):
        question = state["question"]
        logger.info(f"Generating SQL for: {question}", extra={"node": "sql_path"})

        tables = db_connector.get_relevant_tables(question)
        schema = db_connector.get_table_schema(tables)
        base_prompt = load_prompt("sql_agent.txt").format(
            question=question,
            schema_description=schema,
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

            sql = _strip_sql_fences(llm.invoke(prompt).content.strip())
            logger.info(
                f"Generated SQL (attempt {attempt + 1}): {sql}",
                extra={"node": "sql_path", "sql": sql},
            )

            if "NO_SQL_POSSIBLE" in sql.upper():
                return {"sql_result": "N/A - query cannot be answered with available tables"}

            # Validate column references against catalog
            col_error = db_connector.validate_columns(sql, tables)
            if col_error:
                last_error = col_error
                logger.warning(
                    f"SQL column validation failed (attempt {attempt + 1}): {col_error}",
                    extra={"node": "sql_path", "error": col_error},
                )
                if attempt < MAX_SQL_RETRIES:
                    continue
                return {"sql_result": f"Error: {last_error}"}

            # Execute query
            sql_result = db_connector.execute_query(sql)

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
            return {"sql_result": sql_result}


def vector_retrieval(state: Dict):
    with NodeTimer("vector_retrieval"):
        question = state["question"]
        logger.info(f"Searching documents for: {question}", extra={"node": "vector_retrieval"})

        try:
            store = _get_vector_store()
            results = store.similarity_search(question, k=4)
            docs = []
            for doc in results:
                source = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
                docs.append(f"[{source}]: {doc.page_content}")
            logger.info(f"Found {len(docs)} relevant chunks", extra={"node": "vector_retrieval"})
            return {"retrieved_docs": docs}
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}", extra={"node": "vector_retrieval", "error": str(e)})
            return {"retrieved_docs": []}


def answer_generator(state: Dict):
    with NodeTimer("answer_generator"):
        route = state.get("route", "documents")
        logger.info(f"Generating answer (route={route})", extra={"node": "answer_generator", "route": route})

        history = _format_history(state)
        prompt_template = load_prompt("final_answer.txt")
        sql_result = state.get("sql_result", "N/A - no SQL data retrieved")
        docs = "\n".join(state.get("retrieved_docs", [])) or "N/A - no documents retrieved"

        # Truncate content to fit context window
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
        response = llm.invoke(prompt)
        return {"final_answer": response.content}


def reviewer_node(state: Dict):
    with NodeTimer("reviewer"):
        prompt = load_prompt("reviewer.txt").format(
            question=state["question"],
            sql_result=state.get("sql_result", ""),
            docs=state.get("retrieved_docs", ""),
            answer=state.get("final_answer", "")
        )
        response = llm.invoke(prompt)
        score = _parse_review_score(response.content)

        result = {"review_score": score}

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


def cache_write(state: Dict):
    with NodeTimer("cache_write"):
        if state.get("review_score", 0) >= 7.0 and state.get("final_answer"):
            key = _cache_key(state["question"])
            try:
                redis_client.set(key, state["final_answer"], ex=timedelta(hours=24))
                logger.info("Cached answer", extra={"node": "cache_write"})
            except Exception as e:
                logger.warning(f"Cache write failed: {e}", extra={"node": "cache_write", "error": str(e)})
        return {}
