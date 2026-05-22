# AI Runtime Platform

A hybrid Retrieval-Augmented Generation system that answers compliance questions by combining **structured data** (Oracle SQL) with **unstructured knowledge** (document semantic search). The system autonomously classifies each question, selects an execution strategy, chains data sources when one source's output informs another, synthesizes an answer, and self-reviews for quality before caching.

Built for a regulated financial services context where answers must be grounded in both quantitative records and policy documents.

## Why This Architecture

Traditional RAG systems retrieve documents and generate answers. Compliance teams need more: they ask questions that span structured data ("How many KYC violations this quarter?") and policy interpretation ("What does our framework say about VaR limits?") -- often in the same question. And frequently, the answer from one source is needed to query the other ("What is the compliance cost for 2024?" needs the cost formula from docs before it can generate the right SQL).

This system solves that with a **strategy-based hybrid pipeline**: an LLM classifies each query into one of five execution strategies, retrieves from the appropriate sources -- chaining them when needed -- and generates a grounded answer. A reviewer node scores every response; low-quality answers trigger a reflection loop for self-correction before caching.

### Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| LangGraph over LangChain agents | Deterministic state machine with explicit conditional edges. No autonomous tool-calling loops that could produce runaway LLM calls. | Less flexible than a ReAct agent, but predictable cost and latency per query. |
| 5 routing strategies | Beyond simple SQL/Docs/Both: `docs_then_sql` and `sql_then_docs` chain sources so one source's output informs the next query. An extract node bridges them. | More complex routing logic, but handles questions like "compliance cost" (needs formula from docs before SQL) or "top violation and its policy" (needs data before doc search). |
| LLM-driven table selection | The Oracle connector passes a business catalog to the LLM to pick relevant tables, rather than embedding table names or using keyword matching. | Costs one extra LLM call per SQL query, but dramatically improves SQL accuracy for natural language questions. |
| SQL validation + retry loop | Generated SQL is validated against the catalog schema before execution. Invalid column references are caught pre-execution and fed back to the LLM for correction (up to 2 retries). | Adds latency on malformed queries, but prevents cryptic Oracle errors from reaching the answer generator. |
| Persistent PostgreSQL checkpointer | LangGraph state is stored in PostgreSQL (via `langgraph-checkpoint-postgres`), surviving restarts and shared across workers. Falls back to in-memory for local dev. | Requires psycopg v3 alongside psycopg2 (used by PGVector). Enables durable audit trails for regulatory compliance. |
| Non-blocking Redis via `asyncio.to_thread` | Cache check, cache write, conversation history load/save all run in background threads to avoid blocking the async event loop. | Thread pool overhead vs native async Redis, but simpler migration and consistent with how Oracle calls are handled. |
| Request-level timeout | Every workflow invocation is wrapped in `asyncio.wait_for` with a configurable timeout (default 60s). | Prevents unbounded requests when LLM or database hangs. Returns 504 instead of hanging indefinitely. |
| Review score default of 0.0 | When the reviewer LLM returns unparseable output, the score defaults to 0.0 (fail) instead of 7.0 (pass). | Conservative: a non-parseable review triggers the reflection loop rather than silently passing a potentially bad answer into the cache. |
| Redis for both cache and history | Single dependency for response caching (24h TTL) and conversation history (1h TTL). | Acceptable for this scale. At higher throughput, separate the workloads or use a dedicated session store. |
| PII redaction + DLP gate | SQL results are scrubbed for PII (emails, SSNs, phone numbers, card numbers, IBANs) before reaching the answer generator. The reviewer node runs a second DLP scan on the final answer; any leaked PII is redacted and the review score is capped at 6.0, preventing caching. | Two-layer defense. Adds negligible latency (regex-based, no external API). |
| Local embeddings (MiniLM-L6-v2) | Embedding model runs in-process, no external API call for vector search. | ~80MB memory overhead, but eliminates a network dependency and per-request embedding cost. |

## System Architecture

```
                         User Question
                              |
                              v
                       +-------------+
                       |   FastAPI    |
                       |  POST /query|
                       +------+------+
                              |
                       [Auth + Rate Limit + Input Validation]
                              |
                              v
                     +--------+--------+
                     |  cache_check    |
                     |  (Redis, async) |
                     +--------+--------+
                       HIT /    \ MISS
                        |        |
                        v        v
                      [END]  +--------+
                             | router |  (LLM: 5 strategies)
                             +---+----+
                                 |
                                 v
                           +-----------+
                           |  clarify  |  (LLM detects ambiguity)
                           +-----+-----+
                       __________|__________
                      /    |    |    |      \
              sql_only  sql_  docs_  docs_  parallel
                        then  then   only
                        docs  sql
                        |     |     |     |       |
                        v     v     v     v       v
                 (see routing paths table below)
                              |
                     [Context Window Manager]
                              |
                              v
                    +---------+---------+
                    | answer_generator  |<------+
                    +---------+---------+       |
                              |           [reflection]
                              v                 |
                      +-------+-------+         |
                      |   reviewer    |---------+
                      |  (score + DLP)|   (if score < 7
                      +-------+-------+    and attempts < 2)
                              |
                              v
                      +-------+-------+
                      |  cache_write  |  (caches if score >= 7)
                      +-------+-------+
                              |
                              v
                           [END]
```

### Routing Strategies

| Strategy | Flow | Example | Typical Latency |
|----------|------|---------|-----------------|
| **sql_only** | clarify -> sql_path -> answer -> reviewer | "How many violations in Q4?" | 2-4s |
| **docs_only** | clarify -> vector_retrieval -> answer -> reviewer | "What is our access control policy?" | 2-3s |
| **docs_then_sql** | clarify -> vector_retrieval -> **extract** -> sql_path -> answer -> reviewer | "What is compliance cost for 2024?" (needs formula from docs first) | 4-6s |
| **sql_then_docs** | clarify -> sql_path -> **extract** -> vector_retrieval -> answer -> reviewer | "Top violation and its policy?" (needs data before doc search) | 4-6s |
| **parallel** | clarify -> sql_path -> vector_retrieval -> answer -> reviewer | "List violations and summarize the framework" | 3-5s |
| **Cache Hit** | cache_check -> END | Repeated question within 24h | <50ms |
| **Clarification** | router -> clarify -> END | "Show me the data" (vague) | ~1s |

### Chained Strategies: How Extract Works

The **extract node** bridges two data sources when one needs the other's output:

**docs_then_sql** ("What is the compliance cost for 2024?"):
1. `vector_retrieval` finds the cost formula in policy documents
2. `extract` pulls the formula: "SUM(fine_amount + remediation_cost + audit_fees)"
3. `sql_path` receives this formula as context and generates the correct SQL

**sql_then_docs** ("Most common violation and its policy?"):
1. `sql_path` queries the database and finds "ACCESS_CONTROL: 47 incidents"
2. `extract` pulls the key term: "ACCESS_CONTROL"
3. `vector_retrieval` uses "ACCESS_CONTROL" as the search query instead of the original question, finding the specific policy

If extraction fails (no usable content, or the LLM returns `EXTRACTION_FAILED`), the system falls back gracefully -- the downstream source receives the original question instead of extracted context.

### SQL Generation Pipeline

The SQL path includes a three-stage safety net:

1. **Catalog-aware generation** -- The LLM receives the full business catalog with table descriptions and column meanings, not raw DDL. This produces more accurate SQL because the LLM understands the domain semantics.

2. **Pre-execution validation** -- Before any SQL hits Oracle, column references are validated against the catalog. If the LLM references `SOE_ID` from a table that doesn't have it, the error is caught immediately with a descriptive message.

3. **Error-driven retry** -- If validation or execution fails, the error is fed back to the LLM with the original prompt. The LLM gets up to 2 retry attempts to produce correct SQL. This handles transient generation errors without failing the entire query.

Additionally, a SQL allowlist enforces that only `SELECT` statements execute. `DROP`, `DELETE`, `UPDATE`, statement chaining (`;`), comment injection (`--`, `/*`), and Oracle system packages (`UTL_`, `DBMS_`, `SYS.`) are all blocked.

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Llama 3.3 70B via Groq | Routing, SQL generation, extraction, answering, reviewing |
| Embeddings | all-MiniLM-L6-v2 (local) | Document embedding and semantic search |
| Orchestration | LangGraph | Stateful workflow with conditional routing and reflection |
| API | FastAPI + Uvicorn | REST API with auth, rate limiting, CORS |
| Structured Data | Oracle XE 21c | Compliance violations, audit findings, risk events, controls |
| Vector Store | PostgreSQL 16 + PGVector | Semantic search over compliance documents |
| Cache | Redis 7 | Response caching (24h TTL) + conversation history (1h TTL) |
| Checkpointer | PostgreSQL (via langgraph-checkpoint-postgres) | Persistent audit trail and conversation state across restarts |
| Metrics | Prometheus + Grafana | Request latency, LLM tokens/cost, cache hit rate, review scores |
| Ingestion | PyPDF + openpyxl | PDF and Excel document loading for vector store |
| Tracing | OpenTelemetry (OTLP) | Distributed tracing with per-node spans |
| Logging | Structured JSON | Trace IDs, node attribution, LLM call details |

## Project Structure

```
intelligence-ai-engine/
├── docker-compose.yml              # 6 services: Oracle, Postgres, Redis, App, Prometheus, Grafana
├── docker-compose.override.yml     # Dev override: enables --reload for hot reloading
├── Dockerfile                      # Multi-stage build, non-root user, 4 workers, healthcheck
├── pyproject.toml                  # Ruff, mypy, pytest configuration
├── requirement.txt
├── .env                            # Runtime configuration (API keys, DB credentials)
│
├── src/rag-system/                 # Application code
│   ├── main.py                     # FastAPI endpoints: /query, /health, /ready, /metrics, /audit
│   ├── config.py                   # LLM, DB connector, Redis client initialization
│   ├── workflow.py                 # LangGraph state machine: 5 strategies, extract node, reflection loop
│   ├── nodes.py                    # Workflow nodes (cache, router, extract, SQL, vector, answer, review)
│   ├── security.py                 # Auth, rate limiting, input validation, prompt injection guard, PII, DLP
│   ├── observability.py            # OpenTelemetry + Prometheus + structured JSON logging
│   ├── llm_ops.py                  # Cost tracking, context window management, model fallback
│   ├── metrics.py                  # Metrics catalog: loads business metric definitions from YAML
│   ├── metrics_catalog.yaml        # 7 defined metrics with formulas, steps, thresholds
│   ├── catalog.json                # Oracle table catalog with business-level descriptions
│   ├── database/
│   │   ├── base.py                 # Abstract DatabaseConnector interface
│   │   └── oracle.py               # Oracle: LLM table selection, SQL validation, column validation
│   ├── ingest/
│   │   └── ingest.py               # PDF + Excel ingestion, chunking, and PGVector storage
│   └── prompts/                    # LLM prompt templates (versioned in llm_ops.py)
│       ├── router.txt              # Query classification (5 strategies)
│       ├── clarification.txt       # Ambiguity detection (history-aware)
│       ├── extract.txt             # Context extraction between chained sources
│       ├── sql_agent.txt           # Oracle SQL generation with schema + history context
│       ├── final_answer.txt        # Answer synthesis from SQL + docs + history
│       └── reviewer.txt            # Quality scoring (0-10) with history consistency check
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (mock LLM, mock Redis, sample states)
│   ├── test_nodes.py               # Node-level unit tests (40 tests, mocked dependencies)
│   ├── test_workflow.py            # Workflow integration tests with mocked externals (10 tests)
│   ├── test_security.py            # Security tests: input validation, SQL injection, auth, PII, DLP (32 tests)
│   ├── test_metrics.py             # Metrics catalog + workflow metric routing tests (19 tests)
│   ├── test_ingest.py              # Excel ingestion tests (7 tests)
│   └── test_hybrid_rag.py          # End-to-end tests against live infrastructure (13 scenarios)
│
├── oracle-init/                    # Database initialization scripts (auto-run on first start)
│   ├── 01_create_schema.sql
│   ├── 02_create_tables.sql
│   └── 03_seed_data.sql
│
├── monitoring/
│   ├── prometheus.yml              # Scrape config: app:8000/metrics every 15s
│   └── grafana/
│       ├── provisioning/           # Auto-provisioned Prometheus datasource
│       └── dashboards/             # Pre-built 9-panel observability dashboard
│
├── documents/                      # Source compliance PDFs for ingestion
└── .github/workflows/ci.yml        # CI: lint -> test -> docker build
```

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine + Compose (Linux)
- Groq API key (free tier) from https://console.groq.com/keys

### 1. Configure

```bash
git clone <repo-url>
cd intelligence-ai-engine
```

Create `.env` (only `GROQ_API_KEY` is required -- all other values have defaults in `docker-compose.yml`):

```env
GROQ_API_KEY=gsk_your_key_here
```

When running **outside Docker** (e.g., local development against Docker services), override hosts to `localhost`:

```env
GROQ_API_KEY=gsk_your_key_here
REDIS_HOST=localhost
ORACLE_HOST=localhost
POSTGRES_HOST=localhost
```

### 2. Start

```bash
# Production (4 workers, no hot reload)
docker-compose -f docker-compose.yml up -d

# Development (hot reload enabled via override)
docker-compose up -d
```

Six containers start: Oracle (allow ~2 min for first boot), PostgreSQL + PGVector, Redis, the FastAPI app, Prometheus, and Grafana.

### 3. Ingest documents

```bash
docker exec rag-app python ingest/ingest.py
```

Loads compliance PDFs and Excel files from the `documents/` folder into PGVector. Excel files are converted row-by-row into `Column: Value` documents with sheet-level metadata.

### 4. Verify

```bash
# Liveness
curl http://localhost:8000/health

# Readiness (checks Oracle, Postgres, Redis)
curl http://localhost:8000/ready

# First query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many compliance violations per severity level?"}'
```

## API Reference

### POST /query

```json
// Request
{
  "question": "Show me all KYC violations and explain our compliance policy on KYC",
  "session_id": "optional-for-multi-turn"
}

// Response
{
  "answer": "Based on SQL results, there are 2 KYC violations... Section 5.1 requires...",
  "cache_hit": false,
  "session_id": "generated-or-provided-id",
  "trace_id": "a1b2c3d4e5f6"
}
```

**Headers:** `X-API-Key` (required when `API_KEYS` is configured). Response includes `X-Trace-ID`.

**Rate limit:** 30 requests/minute per API key or IP.

**Timeout:** Configurable via `REQUEST_TIMEOUT` env var (default 60s). Returns HTTP 504 on timeout.

### GET /health

Liveness probe. Returns 200 if the process is running.

### GET /ready

Readiness probe. Returns 200 if Oracle, PostgreSQL, and Redis are all reachable; 503 otherwise.

### GET /audit/{thread_id}

Returns the full state history for a session -- every node's input/output state, step number, and timestamp. Requires `X-API-Key` header. Backed by PostgreSQL checkpointer for persistence across restarts.

```json
// Response
{
  "thread_id": "session-uuid",
  "total_steps": 8,
  "trail": [
    {"step": 0, "node": "cache_check", "timestamp": "...", "state": {...}},
    {"step": 1, "node": "router", "timestamp": "...", "state": {...}}
  ]
}
```

### GET /metrics

Prometheus-format metrics. Scraped automatically by the Prometheus container.

## Multi-Turn Conversations

The system maintains per-session conversation history, enabling contextual follow-ups. All LLM nodes (router, clarification, SQL generation, answer generation, reviewer) receive conversation history for full context.

```bash
# Turn 1
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many compliance violations per severity level?", "session_id": "s1"}'
# -> "There are 2 Critical, 3 High, 2 Low, and 3 Medium violations."

# Turn 2 -- references Turn 1
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Break that down by department", "session_id": "s1"}'
# -> SQL node uses history to understand "that" refers to severity distribution

# Turn 3 -- references Turn 2
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does our policy say about those?", "session_id": "s1"}'
# -> "According to Section 4.3 of the Internal Trading Policy..."
```

History is stored in Redis (1h TTL, last 10 turns retained, last 5 sent to LLM). All history content is sanitized against prompt injection before inclusion in LLM prompts. Omit `session_id` for stateless single-turn queries.

## Security Model

The system defends against threats at multiple layers:

| Layer | Threat | Mitigation |
|-------|--------|------------|
| **Network** | Unauthorized access | API key authentication via `X-API-Key` header. Keys validated against `API_KEYS` env var (hashed for logging). Dev mode (no keys configured) disables auth. |
| **Network** | Abuse / DoS | Rate limiting at 30 req/min per API key or IP via `slowapi`. Request timeout (default 60s) prevents resource exhaustion. |
| **Input** | Prompt injection | 11 regex patterns detect injection attempts (`ignore previous instructions`, `<system>`, `[INST]`, etc.). Blocked with HTTP 400. |
| **Input** | Oversized input | Question length enforced: min 3, max 2,000 characters. |
| **Context** | History-based injection | Conversation history sanitized through `sanitize_for_prompt()` before LLM inclusion. Injection markers replaced with `[FILTERED]`, HTML tags stripped. |
| **Database** | SQL injection / destructive queries | Allowlist enforces `SELECT`-only. Blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, statement chaining (`;`), comment injection (`--`, `/*`), and Oracle system packages (`UTL_`, `DBMS_`, `SYS.`). |
| **Database** | Unbounded result sets | All queries appended with `FETCH FIRST 500 ROWS ONLY`. |
| **Output** | PII in SQL results | SQL results are scrubbed for PII (emails, SSNs, phones, card numbers, IBANs) via regex before reaching the answer generator. |
| **Output** | PII in final answer | DLP gate in the reviewer node scans every answer. Leaked PII is redacted and the review score is capped at 6.0, preventing caching of tainted responses. |
| **Container** | Privilege escalation | App runs as non-root `appuser` inside the container. |
| **CORS** | Cross-origin abuse | Configurable via `CORS_ALLOWED_ORIGINS`. Locked to explicit origins in production. |

## Observability

### Metrics (Prometheus)

The `/metrics` endpoint exposes 12 metric families (11 operational + 1 info). Prometheus scrapes every 15 seconds.

| Metric | Type | Labels | What to Watch |
|--------|------|--------|---------------|
| `rag_requests_total` | Counter | `status` | Error rate: `rate({status="error"}[5m]) / rate(total[5m])` |
| `rag_request_duration_seconds` | Histogram | -- | p95 latency: should stay under 5s for non-cache queries |
| `rag_cache_operations_total` | Counter | `result` | Hit rate below 20% suggests cache TTL is too short or question diversity is high |
| `rag_route_total` | Counter | `route` | Distribution shift may indicate prompt drift in the router |
| `rag_llm_calls_total` | Counter | `node`, `status` | Error spikes here often mean API key quota exhaustion |
| `rag_llm_call_duration_seconds` | Histogram | `node` | Groq p95 is typically 200-500ms; spikes suggest rate limiting |
| `rag_llm_tokens_total` | Counter | `node`, `type` | Track burn rate against daily quota (100K tokens/day on Groq free tier) |
| `rag_llm_cost_usd` | Counter | `node` | Cumulative cost tracking across providers |
| `rag_node_duration_seconds` | Histogram | `node` | Identifies bottleneck nodes in the pipeline |
| `rag_node_errors_total` | Counter | `node` | Elevated `sql_path` errors may indicate schema drift or LLM degradation |
| `rag_review_score` | Histogram | -- | Sustained low scores signal answer quality degradation |

### Dashboard (Grafana)

A 9-panel dashboard is auto-provisioned at http://localhost:3000 (admin/admin):

| Panel | Type | PromQL |
|-------|------|--------|
| Request Rate | Time series | `rate(rag_requests_total[5m])` |
| Request Latency (p50/p95/p99) | Time series | `histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))` |
| Cache Hit Rate | Gauge | `hit / (hit + miss) * 100` |
| Route Distribution | Pie chart | `rag_route_total` |
| LLM Latency by Node | Time series | `histogram_quantile(0.95, rate(rag_llm_call_duration_seconds_bucket[5m]))` |
| Token Consumption | Time series | `rate(rag_llm_tokens_total[5m])` |
| LLM Calls by Node | Bar chart | `rag_llm_calls_total` |
| Node Execution Time | Time series | `histogram_quantile(0.95, rate(rag_node_duration_seconds_bucket[5m]))` |
| Review Score Distribution | Histogram | `rag_review_score_bucket` |

### Structured Logs

Every log line is JSON with trace ID and node attribution:

```json
{"timestamp": "2026-05-03T10:15:21Z", "level": "INFO", "trace_id": "a1b2c3", "node": "router", "message": "Route decided: docs_then_sql", "route": "docs_then_sql"}
{"timestamp": "2026-05-03T10:15:21Z", "level": "INFO", "trace_id": "a1b2c3", "node": "extract", "message": "Extracted context: SUM(fine_amount + remediation_cost + audit_fees)"}
{"timestamp": "2026-05-03T10:15:22Z", "level": "INFO", "trace_id": "a1b2c3", "node": "sql_path", "message": "Generated SQL (attempt 1): SELECT SUM(fine_amount + remediation_cost) FROM COMPLIANCE_VIOLATIONS WHERE EXTRACT(YEAR FROM VIOLATION_DATE) = 2024"}
```

### Distributed Tracing (OpenTelemetry)

Every workflow node and LLM call is wrapped in an OTel span. To connect a trace backend:

```yaml
# docker-compose.yml
OTEL_EXPORTER_OTLP_ENDPOINT: http://jaeger:4317
```

Compatible with Jaeger, Grafana Tempo, Datadog, and any OTLP-compatible backend.

## Failure Modes

| Failure | Impact | Behavior |
|---------|--------|----------|
| **Redis down** | Cache miss on every request, no conversation history | Graceful degradation. Cache check catches the exception and proceeds as a miss. History loads return empty. |
| **Oracle down** | SQL-routed queries fail | `sql_path` returns an execution error string. The answer generator still produces a response noting the data is unavailable. |
| **PGVector down** | Document-routed queries return no context | `vector_retrieval` catches the exception and returns an empty doc list. Answer is generated without document context. |
| **Extract fails** | Chained strategy loses context bridge | Extract returns empty `extracted_context`. The downstream source (sql_path or vector_retrieval) falls back to using the original question. |
| **PostgreSQL checkpointer unavailable** | Audit trail not persisted | Falls back to in-memory `MemorySaver` with a warning log. Audit trail works within the process lifetime but is lost on restart. |
| **Groq API rate limit** | All LLM calls fail (router, SQL gen, answer, review) | `_ainvoke_llm` retries 3 times with exponential backoff for `TimeoutError` and `ConnectionError`. Sustained failures return 500. |
| **Request timeout** | LLM or database hangs | `asyncio.wait_for` kills the workflow after `REQUEST_TIMEOUT` seconds (default 60). Returns HTTP 504 with trace ID. |
| **LLM generates bad SQL** | Oracle returns an error | SQL validation catches column reference errors pre-execution. On execution errors, the error is fed back to the LLM for retry (up to 2 attempts). |
| **LLM reviewer returns garbage** | Score can't be parsed | Score defaults to 0.0 (fail), triggering the reflection loop. Prevents bad answers from silently passing review and being cached. |

## Data Sources

### Oracle Database (4 tables, 33 seed rows)

| Table | Records | Content |
|-------|---------|---------|
| `COMPLIANCE_VIOLATIONS` | 10 | Violations with severity, status, financial impact, employee attribution |
| `AUDIT_FINDINGS` | 7 | Internal/external audit findings with risk ratings |
| `CONTROL_MAPPINGS` | 8 | Controls mapped to compliance requirements |
| `RISK_EVENTS` | 8 | Operational, credit, and market risk incidents with loss amounts |

Table selection is LLM-driven: the Oracle connector passes a business-level catalog (`catalog.json`) to the LLM, which selects the most relevant tables for each query.

### Metrics Catalog (7 business metrics)

Defined in `metrics_catalog.yaml`, each metric includes a formula, calculation steps, required tables, parameters, and quality thresholds. The SQL agent receives all metric definitions in its prompt, enabling it to generate correct SQL for questions like "What is the compliance effectiveness score?" without needing a docs lookup.

### PGVector Document Store (4 PDFs + Excel)

The ingestion pipeline (`ingest/ingest.py`) supports both PDF and Excel files. PDFs are split into overlapping chunks (750 chars, 120 overlap). Excel files are converted row-by-row into `Column: Value` text documents with sheet-level metadata.

| Document | Format | Content |
|----------|--------|---------|
| Corporate Compliance Policy 2024 | PDF | AML, KYC, data privacy, code of conduct, whistleblower protection |
| Risk Management Framework v2.0 | PDF | Market/credit/operational/liquidity risk, VaR limits, stress testing |
| Q3 2024 Internal Audit Report | PDF | KYC gaps, trade surveillance, access control, reporting findings |
| SOE-45678 Incident Report | PDF | Unauthorized cross-trades, root cause analysis, remediation plan |
| *(any .xlsx/.xls in documents/)* | Excel | Auto-ingested: each row becomes a searchable document |

## Testing Strategy

### Unit Tests (108 tests, no infrastructure required)

```bash
pytest tests/ -v
```

All external dependencies (LLM, Oracle, PostgreSQL, Redis) are mocked. Tests validate:

- **Node logic (40 tests):** Cache hit/miss (async), routing decisions for all 5 strategies with fuzzy matching, extract node for both chaining directions, SQL path with retry/markdown stripping/extracted context, vector retrieval with extracted context and error handling, review score parsing (including 0.0 default), DLP redaction, cache write thresholds, history formatting and truncation.
- **Workflow integration (10 tests):** Full graph traversal for all 5 strategies (sql_only, docs_only, docs_then_sql, sql_then_docs, parallel), clarification early-exit, cache hit short-circuit, cache write conditions, reflection loop.
- **Security (32 tests):** Input validation boundaries, 11 prompt injection patterns, SQL allowlist (SELECT-only, blocks DROP/DELETE/chaining/comments/Oracle packages), API key auth flows, PII redaction (email/SSN/phone/card/IBAN), DLP scanning.
- **Metrics (19 tests):** Catalog loading, metric definitions, context building, SQL prompt integration, workflow metric routing.
- **Ingestion (7 tests):** Excel file loading, row-to-document conversion, sheet-aware metadata, empty/sparse file handling, graceful openpyxl fallback.

### End-to-End Tests (13 scenarios, requires live infrastructure)

```bash
python tests/test_hybrid_rag.py
```

Runs against real Oracle, PostgreSQL, Redis, and Groq API.

### CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main`/`develop` and on PRs:

1. **Lint** -- `ruff check src/rag-system/ tests/`
2. **Test** -- `pytest tests/test_security.py tests/test_nodes.py` with 60% minimum coverage
3. **Docker build** -- builds the image and verifies it starts

## Switching LLM Providers

The LLM is configured in `config.py` and controlled by the `LLM_MODEL` environment variable:

```python
# Groq (current -- free tier)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1200)

# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1200)

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=1200)

# Google
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_output_tokens=1200)
```

API key env vars: `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model identifier |
| `LLM_TIMEOUT` | `30` | LLM request timeout (seconds) |
| `LLM_MAX_RETRIES` | `3` | LLM client retry attempts |
| `REQUEST_TIMEOUT` | `60` | Overall request timeout for workflow execution (seconds) |
| `API_KEYS` | *(empty = auth disabled)* | Comma-separated valid API keys |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed origins |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | *(empty)* | Redis password |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_USER` | `postgres` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `postgres` | PostgreSQL password |
| `POSTGRES_DB` | `rag_db` | PostgreSQL database name |
| `ORACLE_HOST` | `localhost` | Oracle host |
| `ORACLE_PORT` | `1521` | Oracle listener port |
| `ORACLE_SERVICE` | `XEPDB1` | Oracle service name |
| `ORACLE_SCHEMA` | `COMPLIANCE_USER` | Oracle schema owner |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(empty)* | OpenTelemetry OTLP endpoint |
| `ENVIRONMENT` | `production` | Environment tag for tracing metadata |

## Operations

### Start / Stop

```bash
docker-compose up -d              # Development (hot reload via override)
docker-compose -f docker-compose.yml up -d   # Production (4 workers, no reload)
docker-compose down               # Stop (preserves data volumes)
docker-compose down -v            # Stop and destroy all data
```

### Logs

```bash
docker-compose logs -f app        # Structured JSON logs from the application
docker logs -f rag-app            # Same, via Docker directly
```

### Cache Management

```bash
docker exec rag-redis redis-cli KEYS "rag:answer:*"   # List cached answers
docker exec rag-redis redis-cli KEYS "history:*"       # List active sessions
docker exec rag-redis redis-cli FLUSHDB                # Clear all cache
```

### Service URLs

| URL | Service |
|-----|---------|
| http://localhost:8000/docs | FastAPI -- interactive API docs |
| http://localhost:8000/metrics | Prometheus metrics (raw) |
| http://localhost:9090 | Prometheus -- query and explore metrics |
| http://localhost:3000 | Grafana -- dashboards (admin/admin) |

## Known Limitations and Future Work

| Area | Current State | Production Path |
|------|--------------|-----------------|
| **PII patterns** | Regex-based (emails, SSNs, phones, cards, IBANs) -- no named entity recognition | Add spaCy NER or a dedicated PII service for names, addresses, dates of birth |
| **Auth** | API key validation against env var | Integrate with OAuth2 / OIDC provider; add RBAC for audit endpoint access |
| **Rate limiting** | In-memory via slowapi | Use Redis-backed rate limiter for multi-worker consistency |
| **Embedding model** | all-MiniLM-L6-v2 (~80MB, 384-dim) -- English-only, general-purpose | Evaluate domain-specific financial embeddings for compliance terminology |
| **Circuit breaker** | No circuit breaker on LLM or database calls | Add per-node circuit breakers to prevent cascading failures under sustained outages |
| **LLM retry scope** | Only retries on `TimeoutError` and `ConnectionError` | Add Groq-specific exceptions (`RateLimitError`, `APIError`) to retry filter |
| **Model fallback** | `ModelFallbackChain` class exists but is not wired into the main workflow | Configure fallback chain in `config.py` for production resilience |
| **Horizontal scaling** | Single-instance with 4 Uvicorn workers | Stateless app (Redis + Postgres backed) is ready for multi-instance deployment behind a load balancer |
