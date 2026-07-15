import asyncio
import json
import os
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import redis_client
from observability import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    get_prometheus_content_type,
    get_prometheus_metrics,
    logger,
    new_trace_id,
    trace_id_var,
    tracer,
)
from security import limiter, validate_question, verify_api_key
from workflow import app as workflow_app

app = FastAPI(
    title="ARIA — Analytical Risk Intelligence Agent",
    docs_url="/docs",
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]

# Wildcard only in development. In production an empty CORS_ALLOWED_ORIGINS
# means no cross-origin access (fail closed), not "allow everything".
if not ALLOWED_ORIGINS and os.getenv("ENVIRONMENT", "development") == "development":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # auth uses X-API-Key header, not cookies
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ── OpenTelemetry auto-instrumentation ────────────────────────────────
FastAPIInstrumentor.instrument_app(app)

HISTORY_TTL_SECONDS = 3600  # 1 hour
MAX_HISTORY_TURNS = 10
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))


class Query(BaseModel):
    question: str
    session_id: str | None = None


async def _load_history(session_id: str) -> list:
    """Load conversation history from Redis."""
    try:
        raw = await asyncio.to_thread(redis_client.get, f"history:{session_id}")
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Failed to load history: {e}", extra={"error": str(e)})
    return []


async def _save_history(session_id: str, history: list):
    """Save conversation history to Redis with TTL."""
    try:
        trimmed = history[-MAX_HISTORY_TURNS:]
        await asyncio.to_thread(
            redis_client.set,
            f"history:{session_id}",
            json.dumps(trimmed),
            HISTORY_TTL_SECONDS,
        )
    except Exception as e:
        logger.warning(f"Failed to save history: {e}", extra={"error": str(e)})


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    trace_id = new_trace_id()
    trace_id_var.set(trace_id)
    start = time.perf_counter()

    response = await call_next(request)

    latency_s = time.perf_counter() - start
    response.headers["X-Trace-ID"] = trace_id

    if request.url.path == "/query" and request.method == "POST":
        status = "success" if response.status_code < 500 else "error"
        REQUEST_COUNT.labels(status=status).inc()
        REQUEST_LATENCY.observe(latency_s)
        logger.info(
            "Request completed",
            extra={
                "route": request.url.path,
                "duration_ms": round(latency_s * 1000, 1),
            },
        )

    return response



@app.post("/query")
@limiter.limit("30/minute")
async def query(
    request: Request,
    body: Query,
    api_key: str = Depends(verify_api_key),
):
    # Validate and sanitize input
    question = validate_question(body.question)

    try:
        with tracer.start_as_current_span(
            "rag.query",
            attributes={"rag.question_length": len(question)},
        ):
            session_id = body.session_id or str(uuid.uuid4())
            history = await _load_history(session_id)

            result = await asyncio.wait_for(
                workflow_app.ainvoke(
                    {
                        "question": question,
                        "session_id": session_id,
                        "conversation_history": history,
                    },
                    config={"configurable": {"thread_id": session_id}},
                ),
                timeout=REQUEST_TIMEOUT,
            )

            answer = result.get("final_answer", "Sorry, I could not generate an answer.")

            if not result.get("cache_hit"):
                history_entry = {
                    "question": question,
                    "answer": answer,
                }
                if result.get("resolved_metric"):
                    history_entry["resolved_metric"] = result["resolved_metric"]
                    history_entry["metric_version"] = result.get("metric_version", "")
                if result.get("route"):
                    history_entry["route"] = result["route"]
                sql_result = result.get("sql_result", "")
                if sql_result and not sql_result.startswith(("Error", "N/A")):
                    if "\n\nResult:\n" in sql_result:
                        history_entry["data_summary"] = sql_result.split("\n\nResult:\n")[1][:300]
                    else:
                        history_entry["data_summary"] = sql_result[:300]
                history.append(history_entry)
                await _save_history(session_id, history)

            interpreted_as = result.get("question", question)

            response = {
                "answer": answer,
                "cache_hit": result.get("cache_hit", False),
                "session_id": session_id,
                "trace_id": trace_id_var.get("no-trace"),
                "metadata": {
                    "route": result.get("route"),
                    "resolved_metric": result.get("resolved_metric"),
                    "metric_version": result.get("metric_version"),
                    "compiled": result.get("compiled_metric", False),
                    "confidence": result.get("confidence", "unknown"),
                    "review_score": result.get("review_score"),
                },
            }
            if interpreted_as.lower() != question.lower():
                response["interpreted_as"] = interpreted_as
            return response
    except TimeoutError:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(
            f"Request timed out after {REQUEST_TIMEOUT}s",
            extra={"error": "timeout", "timeout_seconds": REQUEST_TIMEOUT},
        )
        return JSONResponse(
            status_code=504,
            content={
                "error": f"Request timed out after {REQUEST_TIMEOUT} seconds",
                "trace_id": trace_id_var.get("no-trace"),
            },
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Query failed: {e}", extra={"error": str(e)})
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "trace_id": trace_id_var.get("no-trace")},
        )


@app.get("/health")
async def health():
    """Liveness probe — app is running."""
    return {"status": "healthy"}


@app.get("/ready")
async def readiness():
    """Readiness probe — checks all downstream dependencies."""
    checks = {}

    # Redis
    try:
        await asyncio.to_thread(redis_client.ping)
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # PostgreSQL / PGVector
    try:
        from nodes import _get_vector_store
        _get_vector_store()
        checks["pgvector"] = "ok"
    except Exception as e:
        checks["pgvector"] = f"error: {e}"

    # Oracle
    try:
        from config import db_connector
        with db_connector.engine.connect() as conn:
            from sqlalchemy import text as sa_text
            conn.execute(sa_text(db_connector.get_readiness_query()))
        checks["oracle"] = "ok"
    except Exception as e:
        checks["oracle"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )


@app.get("/audit/{thread_id}")
async def audit_trail(thread_id: str, api_key: str = Depends(verify_api_key)):
    """Retrieve the full audit trail for a session — every node's input/output state."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        states = []
        for state in workflow_app.get_state_history(config):
            states.append({
                "step": state.metadata.get("step", -1),
                "node": state.metadata.get("source", "unknown"),
                "timestamp": state.metadata.get("created_at", ""),
                "state": {k: v for k, v in state.values.items() if k != "conversation_history"},
            })
        if not states:
            return JSONResponse(status_code=404, content={"error": "No audit trail found for this session"})
        return {"thread_id": thread_id, "total_steps": len(states), "trail": states}
    except Exception as e:
        logger.error(f"Audit trail retrieval failed: {e}", extra={"error": str(e)})
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve audit trail"})


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=get_prometheus_metrics(),
        media_type=get_prometheus_content_type(),
    )
