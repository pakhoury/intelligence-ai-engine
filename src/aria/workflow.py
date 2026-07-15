import os
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes import (
    answer_generator,
    cache_check,
    cache_write,
    clarification_node,
    context_resolver,
    extract_node,
    metric_resolver_node,
    reviewer_node,
    router_node,
    sql_path,
    vector_retrieval,
)
from observability import logger


class AgentState(TypedDict, total=False):
    question: str
    original_question: str
    session_id: str
    conversation_history: list
    cache_hit: bool
    final_answer: str
    review_score: float
    route: str  # "sql_only", "docs_only", "docs_then_sql", "sql_then_docs", "parallel"
    needs_clarification: bool
    sql_result: str
    retrieved_docs: list
    reflection_attempt: int
    reviewer_feedback: str
    extracted_context: str
    resolved_metric: str
    metric_version: str
    metric_context: str
    metric_params: dict
    compiled_metric: bool
    confidence: str

MAX_REFLECTION_ATTEMPTS = 2

workflow = StateGraph(AgentState)

workflow.add_node("cache_check", cache_check)
workflow.add_node("context_resolver", context_resolver)
workflow.add_node("router", router_node)
workflow.add_node("clarify", clarification_node)
workflow.add_node("metric_resolver", metric_resolver_node)
workflow.add_node("sql_path", sql_path)
workflow.add_node("vector_retrieval", vector_retrieval)
workflow.add_node("extract", extract_node)
workflow.add_node("answer_generator", answer_generator)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("cache_write", cache_write)

workflow.set_entry_point("cache_check")

# Cache hit → END, miss → context_resolver → router
workflow.add_conditional_edges(
    "cache_check",
    lambda s: END if s.get("cache_hit") else "context_resolver",
    {END: END, "context_resolver": "context_resolver"}
)

workflow.add_edge("context_resolver", "router")

workflow.add_edge("router", "clarify")


def after_clarify(s):
    if s.get("needs_clarification", False):
        return END
    route = s.get("route", "docs_only")
    if route == "docs_only":
        return "vector_retrieval"
    return "metric_resolver"

workflow.add_conditional_edges(
    "clarify",
    after_clarify,
    {"metric_resolver": "metric_resolver", "vector_retrieval": "vector_retrieval", END: END}
)


def after_metric_resolver(s):
    route = s.get("route", "docs_only")
    if route in ("sql_only", "sql_then_docs", "parallel"):
        return "sql_path"
    if route == "docs_then_sql":
        return "vector_retrieval"
    return "sql_path"

workflow.add_conditional_edges(
    "metric_resolver",
    after_metric_resolver,
    {"sql_path": "sql_path", "vector_retrieval": "vector_retrieval"}
)


def after_sql(s):
    route = s.get("route")
    if route == "sql_only":
        return "answer_generator"
    if route == "sql_then_docs":
        return "extract"
    if route == "parallel":
        return "vector_retrieval"
    # docs_then_sql: sql_path was reached via extract → go to answer
    return "answer_generator"

workflow.add_conditional_edges(
    "sql_path",
    after_sql,
    {"answer_generator": "answer_generator", "extract": "extract", "vector_retrieval": "vector_retrieval"}
)


def after_vector(s):
    route = s.get("route")
    if route == "docs_only":
        return "answer_generator"
    if route == "docs_then_sql":
        return "extract"
    # parallel or sql_then_docs: vector was the last data step
    return "answer_generator"

workflow.add_conditional_edges(
    "vector_retrieval",
    after_vector,
    {"answer_generator": "answer_generator", "extract": "extract"}
)


def after_extract(s):
    route = s.get("route")
    if route == "docs_then_sql":
        return "sql_path"
    return "vector_retrieval"

workflow.add_conditional_edges(
    "extract",
    after_extract,
    {"sql_path": "sql_path", "vector_retrieval": "vector_retrieval"}
)

workflow.add_edge("answer_generator", "reviewer")


def after_review(s):
    score = s.get("review_score", 0)
    attempts = s.get("reflection_attempt", 0)
    if score >= 7.0 or attempts >= MAX_REFLECTION_ATTEMPTS:
        return "cache_write"
    return "answer_generator"

workflow.add_conditional_edges(
    "reviewer",
    after_review,
    {"cache_write": "cache_write", "answer_generator": "answer_generator"}
)

workflow.add_edge("cache_write", END)


def _create_checkpointer():
    """Create persistent PostgreSQL checkpointer.

    In production (ENVIRONMENT != 'development'), failure to connect is fatal —
    audit trail loss is unacceptable. In dev, falls back to in-memory.
    """
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_db = os.getenv("POSTGRES_DB", "rag_db")
    conn_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"
    is_dev = os.getenv("ENVIRONMENT", "development") == "development"

    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        conn = psycopg.connect(conn_string, autocommit=True)
        checkpointer = PostgresSaver(conn=conn)
        checkpointer.setup()
        logger.info("Using PostgreSQL checkpointer for persistent audit trail")
        return checkpointer
    except Exception as e:
        if not is_dev:
            raise RuntimeError(
                f"PostgreSQL checkpointer required in production but unavailable: {e}. "
                f"Audit trail cannot be persisted. Fix the connection or set "
                f"ENVIRONMENT=development to allow in-memory fallback."
            ) from e
        logger.warning(
            f"PostgreSQL checkpointer unavailable ({e}), falling back to in-memory. "
            f"Audit trail will NOT survive restarts.",
        )
        return MemorySaver()


checkpointer = _create_checkpointer()
app = workflow.compile(checkpointer=checkpointer)
