import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from authz import AuthzDecision, Principal, get_authz_client
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
import workflow
from security import API_KEY_HEADER, limiter, validate_question

_checkpointer_pool = None


async def _authorize(
    credential: str | None, action: str, resource: dict | None = None,
) -> Principal | None:
    """Policy Enforcement Point: hand the credential, action, and any resource
    attributes this service already owns (e.g. a thread's recorded owner) to
    the external RBAC service and enforce whatever it decides. Ownership and
    role/permission logic live entirely in that service, not here.
    """
    decision: AuthzDecision = await get_authz_client().authorize(credential, action, resource)
    if not decision.allowed:
        logger.warning(
            f"Authorization denied for action '{action}': {decision.reason}",
            extra={"error": "authz_denied", "action": action},
        )
        raise HTTPException(status_code=403, detail="Not authorized")
    return decision.principal


@asynccontextmanager
async def lifespan(app: FastAPI):
    # AsyncPostgresSaver must be constructed inside a running event loop, so
    # the persistent checkpointer is wired up here rather than at module
    # import time. `workflow.app` (accessed fresh at each call site) is
    # reassigned in place; anything that imported `workflow.app` before this
    # ran would keep the in-memory default, so request handlers below read
    # it via `workflow.app.*` rather than a captured binding.
    global _checkpointer_pool
    result = await workflow.create_persistent_app()
    if result is not None:
        workflow.app, _checkpointer_pool = result
    yield
    if _checkpointer_pool is not None:
        await _checkpointer_pool.close()


app = FastAPI(
    title="ARIA — Analytical Risk Intelligence Agent",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
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
    credential: str | None = Security(API_KEY_HEADER),
):
    principal = await _authorize(credential, action="query:submit")

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
                workflow.app.ainvoke(
                    {
                        "question": question,
                        "session_id": session_id,
                        "conversation_history": history,
                        "trace_id": trace_id_var.get("no-trace"),
                        "owner_id": principal.sub if principal else "unknown",
                        "owner_tenant_id": (principal.tenant_id if principal else None) or "",
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
                    "metric_owner": result.get("metric_owner"),
                    "metric_approval": result.get("metric_approval"),
                    "compiled": result.get("compiled_metric", False),
                    "confidence": result.get("confidence", "unknown"),
                    "data_path": result.get("data_path", "none"),
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
async def audit_trail(thread_id: str, credential: str | None = Security(API_KEY_HEADER)):
    """Retrieve the full audit trail for a session — every node's input/output state.

    Ownership is resolved from the thread's own checkpointed state (recorded
    at query time — see owner_id/owner_tenant_id in workflow.AgentState) and
    handed to the RBAC service as a resource attribute; the service decides
    whether this caller may read it (owner match, auditor role, admin role,
    etc. — that policy lives entirely in the external service). Authorization
    runs before the existence check below so a denied caller learns nothing
    about whether the thread exists.
    """
    config = {"configurable": {"thread_id": thread_id}}

    owner_id = None
    owner_tenant_id = None
    try:
        snapshot = await workflow.app.aget_state(config)
        if snapshot and snapshot.values:
            owner_id = snapshot.values.get("owner_id")
            owner_tenant_id = snapshot.values.get("owner_tenant_id")
    except Exception as e:
        logger.warning(f"Could not resolve thread owner for authz: {e}", extra={"error": str(e)})

    await _authorize(
        credential,
        action="audit:read",
        resource={"type": "thread", "id": thread_id, "owner_id": owner_id, "tenant_id": owner_tenant_id},
    )

    try:
        states = []
        async for state in workflow.app.aget_state_history(config):
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
