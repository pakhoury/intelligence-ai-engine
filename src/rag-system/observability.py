"""
Production observability module — OpenTelemetry tracing, Prometheus metrics,
structured JSON logging, and LLM call tracking.

Provides:
- OpenTelemetry distributed tracing with OTLP export
- Prometheus counters, histograms, and gauges for /metrics scraping
- JSON structured logger with trace IDs
- LangChain callback handler for LLM observability (tokens, latency, cost)
"""
import logging
import json
import os
import time
import uuid
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# ── Prometheus Metrics ────────────────────────────────────────────────
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of /query requests",
    ["status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

CACHE_OPS = Counter(
    "rag_cache_operations_total",
    "Cache hits and misses",
    ["result"],  # hit / miss
    registry=REGISTRY,
)

ROUTE_COUNT = Counter(
    "rag_route_total",
    "Queries by route type",
    ["route"],  # sql / documents / both
    registry=REGISTRY,
)

LLM_CALL_COUNT = Counter(
    "rag_llm_calls_total",
    "Total LLM calls",
    ["node", "status"],  # node=router|sql_path|..., status=success|error
    registry=REGISTRY,
)
LLM_LATENCY = Histogram(
    "rag_llm_call_duration_seconds",
    "LLM call latency in seconds",
    ["node"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "Total LLM tokens consumed",
    ["node", "type"],  # type=prompt|completion
    registry=REGISTRY,
)

NODE_LATENCY = Histogram(
    "rag_node_duration_seconds",
    "Workflow node execution time in seconds",
    ["node"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)
NODE_ERRORS = Counter(
    "rag_node_errors_total",
    "Node execution errors",
    ["node"],
    registry=REGISTRY,
)

REVIEW_SCORE = Histogram(
    "rag_review_score",
    "Answer review scores from reviewer node",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    registry=REGISTRY,
)

REFLECTION_COUNT = Counter(
    "rag_reflection_total",
    "Total reflection loop iterations",
    registry=REGISTRY,
)

LLM_COST = Counter(
    "rag_llm_cost_usd",
    "Estimated LLM cost in USD",
    ["node"],
    registry=REGISTRY,
)

APP_INFO = Info("rag_app", "Application metadata", registry=REGISTRY)
APP_INFO.info({
    "version": os.getenv("APP_VERSION", "1.0.0"),
    "llm_model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    "llm_provider": "groq",
})


def get_prometheus_metrics() -> bytes:
    return generate_latest(REGISTRY)


def get_prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

resource = Resource.create({
    "service.name": "compliance-rag",
    "service.version": os.getenv("APP_VERSION", "1.0.0"),
    "deployment.environment": os.getenv("ENVIRONMENT", "production"),
})

provider = TracerProvider(resource=resource)

# OTLP exporter if endpoint configured, otherwise console fallback in dev
if OTEL_ENDPOINT:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
elif os.getenv("ENVIRONMENT") == "development":
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("compliance-rag")


# ── Trace context ─────────────────────────────────────────────────────
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="no-trace")


def new_trace_id() -> str:
    return str(uuid.uuid4())[:12]


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "trace_id": trace_id_var.get("no-trace"),
            "node": getattr(record, "node", None),
            "message": record.getMessage(),
        }
        for key in ("duration_ms", "tokens", "model", "route", "score",
                     "cache_hit", "sql", "error", "llm_call"):
            val = getattr(record, key, None)
            if val is not None:
                log[key] = val
        return json.dumps(log)


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()


# ── LangChain LLM Callback Handler ───────────────────────────────────

class LLMObservabilityHandler(BaseCallbackHandler):
    """Tracks every LLM call with OTel spans and Prometheus metrics."""

    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self._spans: Dict[str, Any] = {}
        self._current_node: ContextVar[str] = ContextVar("current_node", default="unknown")

    @property
    def current_node(self) -> str:
        return self._current_node.get("unknown")

    @current_node.setter
    def current_node(self, value: str):
        self._current_node.set(value)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        run_id = str(kwargs.get("run_id", uuid.uuid4()))
        self._start_times[run_id] = time.perf_counter()

        model = serialized.get("kwargs", {}).get("model", "unknown")

        # Start OTel span
        span = tracer.start_span(
            f"llm.{self.current_node}",
            attributes={
                "llm.node": self.current_node,
                "llm.model": model,
                "llm.prompt_length": sum(len(p) for p in prompts),
            },
        )
        self._spans[run_id] = span

        logger.info(
            "LLM call started",
            extra={
                "node": self.current_node,
                "llm_call": "start",
                "model": model,
            },
        )

    def on_llm_end(self, response: LLMResult, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        start = self._start_times.pop(run_id, None)
        latency_s = (time.perf_counter() - start) if start else 0
        node = self.current_node

        # Extract token usage
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if not token_usage:
                token_usage = response.llm_output.get("usage", {})

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Record Prometheus metrics
        LLM_CALL_COUNT.labels(node=node, status="success").inc()
        LLM_LATENCY.labels(node=node).observe(latency_s)
        LLM_TOKENS.labels(node=node, type="prompt").inc(prompt_tokens)
        LLM_TOKENS.labels(node=node, type="completion").inc(completion_tokens)

        # Cost tracking
        try:
            from llm_ops import cost_tracker
            cost = cost_tracker.calculate_cost(prompt_tokens, completion_tokens)
            LLM_COST.labels(node=node).inc(cost)
        except Exception:
            pass

        # End OTel span
        span = self._spans.pop(run_id, None)
        if span:
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)
            span.set_attribute("llm.total_tokens", total_tokens)
            span.set_attribute("llm.duration_ms", round(latency_s * 1000, 1))
            span.end()

        logger.info(
            "LLM call completed",
            extra={
                "node": node,
                "llm_call": "end",
                "duration_ms": round(latency_s * 1000, 1),
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                },
            },
        )

    def on_llm_error(self, error: BaseException, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        self._start_times.pop(run_id, None)
        node = self.current_node

        LLM_CALL_COUNT.labels(node=node, status="error").inc()

        span = self._spans.pop(run_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error))
            span.end()

        logger.error(
            f"LLM call failed: {error}",
            extra={
                "node": node,
                "llm_call": "error",
                "error": str(error),
            },
        )


llm_handler = LLMObservabilityHandler()


class NodeTimer:
    """Context manager — times nodes with OTel spans and Prometheus histograms."""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.start = None
        self._span = None

    def __enter__(self):
        self.start = time.perf_counter()
        llm_handler.current_node = self.node_name
        self._span = tracer.start_span(
            f"node.{self.node_name}",
            attributes={"node.name": self.node_name},
        )
        logger.info("Node started", extra={"node": self.node_name})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_s = time.perf_counter() - self.start
        duration_ms = duration_s * 1000

        NODE_LATENCY.labels(node=self.node_name).observe(duration_s)

        if exc_type:
            NODE_ERRORS.labels(node=self.node_name).inc()
            if self._span:
                self._span.set_attribute("error", True)
                self._span.set_attribute("error.message", str(exc_val))
            logger.error(
                f"Node failed: {exc_val}",
                extra={"node": self.node_name, "duration_ms": round(duration_ms, 1), "error": str(exc_val)},
            )
        else:
            logger.info(
                "Node completed",
                extra={"node": self.node_name, "duration_ms": round(duration_ms, 1)},
            )

        if self._span:
            self._span.set_attribute("node.duration_ms", round(duration_ms, 1))
            self._span.end()

        return False
