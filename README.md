# Compliance RAG Engine

A hybrid Retrieval-Augmented Generation system that answers compliance questions by combining **structured data** (Oracle SQL) with **unstructured knowledge** (document semantic search). The system autonomously classifies each question, routes it to the appropriate data source(s), synthesizes an answer, and self-reviews for quality before caching.

Built for a regulated financial services context where answers must be grounded in both quantitative records and policy documents.

## Why This Architecture

Traditional RAG systems retrieve documents and generate answers. Compliance teams need more: they ask questions that span structured data ("How many KYC violations this quarter?") and policy interpretation ("What does our framework say about VaR limits?") -- often in the same question.

This system solves that with a **router-based hybrid pipeline**: an LLM classifies each query into one of three paths (SQL, Documents, or Both), retrieves from the appropriate sources, and generates a grounded answer. A reviewer node scores every response; low-quality answers are not cached, creating a natural feedback loop that pushes for re-generation on retry.

### Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| LangGraph over LangChain agents | Deterministic state machine with explicit conditional edges. No autonomous tool-calling loops that could produce runaway LLM calls. | Less flexible than a ReAct agent, but predictable cost and latency per query. |
| LLM-driven table selection | The Oracle connector passes a business catalog to the LLM to pick relevant tables, rather than embedding table names or using keyword matching. | Costs one extra LLM call per SQL query, but dramatically improves SQL accuracy for natural language questions. |
| SQL validation + retry loop | Generated SQL is validated against the catalog schema before execution. Invalid column references are caught pre-execution and fed back to the LLM for correction (up to 2 retries). | Adds latency on malformed queries, but prevents cryptic Oracle errors from reaching the answer generator. |
| Synchronous workflow nodes | All nodes are synchronous despite FastAPI being async. LangGraph handles the async boundary at `ainvoke()`. | Simpler node code. The bottleneck is LLM latency (~200-500ms per call), not Python concurrency. |
| Redis for both cache and history | Single dependency for response caching (24h TTL) and conversation history (1h TTL). | Acceptable for this scale. At higher throughput, separate the workloads or use a dedicated session store. |
| Self-review before caching | The reviewer node scores every answer 0-10. Only answers scoring >= 7.0 are cached. | Prevents low-quality answers from being served repeatedly. Increases first-response latency by one LLM call. |
| PII redaction + DLP gate | SQL results are scrubbed for PII (emails, SSNs, phone numbers, card numbers, IBANs) before reaching the answer generator. The reviewer node runs a second DLP scan on the final answer; any leaked PII is redacted and the review score is capped at 6.0, preventing caching. | Two-layer defense. Adds negligible latency (regex-based, no external API). |
| LangGraph checkpointer | MemorySaver checkpointer records every node's state. `/audit/{thread_id}` endpoint exposes the full state history for any session. | Enables time-travel debugging and regulatory audit trails. In-memory storage; swap to PostgresSaver for persistence across restarts. |
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
                     |  (Redis)        |
                     +--------+--------+
                       HIT /    \ MISS
                        |        |
                        v        v
                      [END]  +--------+
                             | router |  (LLM: SQL / DOCUMENTS / BOTH)
                             +---+----+
                                 |
                                 v
                           +-----------+
                           |  clarify  |  (LLM detects ambiguity)
                           +-----+-----+
                          /      |       \
                    SQL  /    BOTH        \ DOCS
                        v       |          v
                  +-----------+ |   +-----------------+
                  | sql_path  | |   | vector_retrieval|
                  |  + retry  | |   +--------+--------+
                  +-----+-----+ |            |
                        |       v            |
                  [PII Redact]  +-----------+|
                        |       | sql_path  ||
                        |       |  + retry  ||
                        |       +-----+-----+|
                        |       [PII Redact] |
                        |             |      |
                        |             v      |
                        |    +---------------+
                        |    |vector_retrieval|
                        |    +--------+------+
                        |             |
                        +------+------+
                               |
                      [Context Window Manager]
                               |
                               v
                     +---------+---------+
                     | answer_generator  |
                     +---------+---------+
                               |
                               v
                       +-------+-------+
                       |   reviewer    |  (score 0-10 + DLP scan)
                       +-------+-------+
                               |
                               v
                       +-------+-------+
                       |  cache_write  |  (caches if score >= 7)
                       +-------+-------+
                               |
                               v
                            [END]
```

### Routing Paths

| Route | Trigger | Pipeline | Typical Latency |
|-------|---------|----------|-----------------|
| **SQL** | Counts, rankings, aggregates, specific records | cache -> router -> clarify -> sql_path -> answer -> reviewer -> cache_write | 2-4s |
| **Documents** | Policies, frameworks, procedures, definitions | cache -> router -> clarify -> vector_retrieval -> answer -> reviewer -> cache_write | 2-3s |
| **Both** | Questions needing data + policy context | cache -> router -> clarify -> sql_path -> vector_retrieval -> answer -> reviewer -> cache_write | 3-5s |
| **Cache Hit** | Repeated question (normalized) within 24h | cache -> END | <50ms |
| **Clarification** | Vague or ambiguous query | cache -> router -> clarify -> END | ~1s |

### SQL Generation Pipeline

The SQL path includes a three-stage safety net:

1. **Catalog-aware generation** -- The LLM receives the full business catalog with table descriptions and column meanings, not raw DDL. This produces more accurate SQL because the LLM understands the domain semantics.

2. **Pre-execution validation** -- Before any SQL hits Oracle, column references are validated against the catalog. If the LLM references `SOE_ID` from a table that doesn't have it, the error is caught immediately with a descriptive message.

3. **Error-driven retry** -- If validation or execution fails, the error is fed back to the LLM with the original prompt. The LLM gets up to 2 retry attempts to produce correct SQL. This handles transient generation errors without failing the entire query.

Additionally, a SQL allowlist enforces that only `SELECT` statements execute. `DROP`, `DELETE`, `UPDATE`, statement chaining (`;`), comment injection (`--`, `/*`), and Oracle system packages (`UTL_`, `DBMS_`, `SYS.`) are all blocked.

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Llama 3.3 70B via Groq | Routing, SQL generation, answering, reviewing |
| Embeddings | all-MiniLM-L6-v2 (local) | Document embedding and semantic search |
| Orchestration | LangGraph | Stateful workflow with conditional routing |
| API | FastAPI + Uvicorn | REST API with auth, rate limiting, CORS |
| Structured Data | Oracle XE 21c | Compliance violations, audit findings, risk events, controls |
| Vector Store | PostgreSQL 16 + PGVector | Semantic search over compliance documents |
| Cache | Redis 7 | Response caching (24h TTL) + conversation history (1h TTL) |
| Metrics | Prometheus + Grafana | Request latency, LLM tokens/cost, cache hit rate, review scores |
| Ingestion | PyPDF + openpyxl | PDF and Excel document loading for vector store |
| Tracing | OpenTelemetry (OTLP) | Distributed tracing with per-node spans |
| Logging | Structured JSON | Trace IDs, node attribution, LLM call details |

## Project Structure

```
intelligence-ai-engine/
├── docker-compose.yml              # 6 services: Oracle, Postgres, Redis, App, Prometheus, Grafana
├── Dockerfile                      # Multi-stage build, non-root user, healthcheck
├── pyproject.toml                  # Ruff, mypy, pytest configuration
├── requirement.txt
├── .env                            # Runtime configuration (API keys, DB credentials)
│
├── src/rag-system/                 # Application code
│   ├── main.py                     # FastAPI endpoints: /query, /health, /ready, /metrics, /audit
│   ├── config.py                   # LLM, DB connector, Redis client initialization
│   ├── workflow.py                 # LangGraph state machine definition
│   ├── nodes.py                    # Workflow node implementations (cache, router, SQL, vector, answer, review)
│   ├── security.py                 # Auth, rate limiting, input validation, prompt injection guard, PII redaction, DLP
│   ├── observability.py            # OpenTelemetry + Prometheus + structured JSON logging
│   ├── llm_ops.py                  # Cost tracking, context window management, model fallback
│   ├── catalog.json                # Oracle table catalog with business-level descriptions
│   ├── database/
│   │   ├── base.py                 # Abstract DatabaseConnector interface
│   │   └── oracle.py               # Oracle: LLM table selection, SQL validation, column validation
│   ├── ingest/
│   │   └── ingest.py               # PDF + Excel ingestion, chunking, and PGVector storage
│   └── prompts/                    # LLM prompt templates (versioned in llm_ops.py)
│       ├── router.txt              # Query classification (SQL / DOCUMENTS / BOTH)
│       ├── clarification.txt       # Ambiguity detection
│       ├── sql_agent.txt           # Oracle SQL generation with schema context
│       ├── final_answer.txt        # Answer synthesis from SQL + docs + history
│       └── reviewer.txt            # Quality scoring (0-10)
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (mock LLM, mock Redis, sample states)
│   ├── test_nodes.py               # Node-level unit tests (27 tests, mocked dependencies)
│   ├── test_security.py            # Security tests: input validation, SQL injection, auth, PII, DLP (32 tests)
│   ├── test_ingest.py              # Excel ingestion tests (7 tests)
│   ├── test_workflow.py            # Workflow integration tests with mocked externals (7 tests)
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

Create `.env` (only `GROQ_API_KEY` is required — all other values have defaults in `docker-compose.yml`):

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

### GET /health

Liveness probe. Returns 200 if the process is running.

### GET /ready

Readiness probe. Returns 200 if Oracle, PostgreSQL, and Redis are all reachable; 503 otherwise.

### GET /audit/{thread_id}

Returns the full state history for a session — every node's input/output state, step number, and timestamp. Requires `X-API-Key` header. Useful for regulatory audit trails and debugging.

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

The system maintains per-session conversation history, enabling contextual follow-ups:

```bash
# Turn 1
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many compliance violations per severity level?", "session_id": "s1"}'
# -> "There are 2 Critical, 3 High, 2 Low, and 3 Medium violations."

# Turn 2 -- references Turn 1
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me more about the critical ones", "session_id": "s1"}'
# -> "The 2 critical violations are SOE-45678 (Unauthorized Trading) and SOE-77890 (AML)..."

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
| **Network** | Abuse / DoS | Rate limiting at 30 req/min per API key or IP via `slowapi`. |
| **Input** | Prompt injection | 11 regex patterns detect injection attempts (`ignore previous instructions`, `<system>`, `[INST]`, etc.). Blocked with HTTP 400. |
| **Input** | Oversized input | Question length enforced: min 3, max 2,000 characters. |
| **Context** | History-based injection | Conversation history sanitized through `sanitize_for_prompt()` before LLM inclusion. Injection markers replaced with `[FILTERED]`, HTML tags stripped. |
| **Database** | SQL injection / destructive queries | Allowlist enforces `SELECT`-only. Blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, statement chaining (`;`), comment injection (`--`, `/*`), and Oracle system packages (`UTL_`, `DBMS_`, `SYS.`). |
| **Database** | Unbounded result sets | All queries appended with `FETCH FIRST 500 ROWS ONLY`. |
| **Output** | PII in SQL results | SQL results are scrubbed for PII (emails, SSNs, phones, card numbers, IBANs) via regex before reaching the answer generator. |
| **Output** | PII in final answer | DLP gate in the reviewer node scans every answer. Leaked PII is redacted and the review score is capped at 6.0, preventing caching of tainted responses. |
| **Container** | Privilege escalation | App runs as non-root `appuser` inside the container. |
| **CORS** | Cross-origin abuse | Configurable via `CORS_ALLOWED_ORIGINS`. Locked to explicit origins in production. |

## Data Protection (PII / DLP)

The system enforces two independent layers of PII protection, both regex-based with zero external dependencies:

**Layer 1 — SQL Result Redaction** (`nodes.py:sql_path`): Immediately after Oracle returns query results, the output is scrubbed before it enters the LangGraph state. This ensures PII never reaches the LLM prompt for answer generation. Patterns: emails, SSNs, phone numbers, card numbers (13-19 digits), IBANs.

**Layer 2 — DLP Gate in Reviewer** (`nodes.py:reviewer_node`): After the LLM generates the final answer, the reviewer node runs a second scan. If PII is detected:
1. The answer is redacted in-place (the user sees `[EMAIL]`, `[SSN]`, etc.)
2. The review score is capped at 6.0, which prevents the tainted answer from being cached (cache threshold is 7.0)
3. DLP findings are logged with type and count for audit

This two-layer approach catches PII at both the data ingress point (SQL results) and the egress point (final answer). The LLM itself may hallucinate PII patterns — Layer 2 catches those too.

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
{"timestamp": "2026-05-03T10:15:21Z", "level": "INFO", "trace_id": "a1b2c3", "node": "sql_path", "message": "Generated SQL (attempt 1): SELECT SEVERITY, COUNT(*) FROM COMPLIANCE_VIOLATIONS GROUP BY SEVERITY"}
{"timestamp": "2026-05-03T10:15:22Z", "level": "INFO", "trace_id": "a1b2c3", "node": "sql_path", "message": "SQL result: [('Critical', 2), ('High', 3), ('Low', 2), ('Medium', 3)]"}
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
| **Groq API rate limit** | All LLM calls fail (router, SQL gen, answer, review) | The Groq client's built-in retry (3 attempts with exponential backoff) handles transient 429s. Sustained rate limiting fails the request with a 500. |
| **Groq API down** | Complete outage | `ModelFallbackChain` in `llm_ops.py` supports configuring backup providers. Without fallback configured, requests fail. |
| **LLM generates bad SQL** | Oracle returns an error | SQL validation catches column reference errors pre-execution. On execution errors, the error is fed back to the LLM for retry (up to 2 attempts). |
| **LLM quality degrades** | Low review scores | Answers scoring below 7.0 are not cached. `rag_review_score` histogram in Prometheus surfaces the trend. |

## Data Sources

### Oracle Database (4 tables, 33 seed rows)

| Table | Records | Content |
|-------|---------|---------|
| `COMPLIANCE_VIOLATIONS` | 10 | Violations with severity, status, financial impact, employee attribution |
| `AUDIT_FINDINGS` | 7 | Internal/external audit findings with risk ratings |
| `CONTROL_MAPPINGS` | 8 | Controls mapped to compliance requirements |
| `RISK_EVENTS` | 8 | Operational, credit, and market risk incidents with loss amounts |

Table selection is LLM-driven: the Oracle connector passes a business-level catalog (`catalog.json`) to the LLM, which selects the most relevant tables for each query.

### PGVector Document Store (4 PDFs + Excel, ~19 chunks)

The ingestion pipeline (`ingest/ingest.py`) supports both PDF and Excel files. PDFs are split into overlapping chunks (750 chars, 120 overlap). Excel files are converted row-by-row into `Column: Value` text documents with sheet-level metadata.

| Document | Format | Content |
|----------|--------|---------|
| Corporate Compliance Policy 2024 | PDF | AML, KYC, data privacy, code of conduct, whistleblower protection |
| Risk Management Framework v2.0 | PDF | Market/credit/operational/liquidity risk, VaR limits, stress testing |
| Q3 2024 Internal Audit Report | PDF | KYC gaps, trade surveillance, access control, reporting findings |
| SOE-45678 Incident Report | PDF | Unauthorized cross-trades, root cause analysis, remediation plan |
| *(any .xlsx/.xls in documents/)* | Excel | Auto-ingested: each row becomes a searchable document |

## Testing Strategy

### Unit Tests (78 tests, no infrastructure required)

```bash
pytest tests/test_nodes.py tests/test_security.py tests/test_workflow.py tests/test_ingest.py -v
```

All external dependencies (LLM, Oracle, PostgreSQL, Redis) are mocked. Tests validate:

- **Node logic (27 tests):** Cache hit/miss, routing decisions, SQL path with markdown stripping and retry, vector retrieval error handling, review score parsing, DLP redaction in reviewer, cache write thresholds, history formatting and truncation.
- **Security (32 tests):** Input validation boundaries, 13 prompt injection patterns, SQL allowlist (SELECT-only, blocks DROP/DELETE/chaining/comments/Oracle packages), API key auth flows, PII redaction (email/SSN/phone/card/IBAN), DLP scanning.
- **Ingestion (7 tests):** Excel file loading, row-to-document conversion, sheet-aware metadata, empty/sparse file handling, graceful openpyxl fallback.
- **Workflow integration (7 tests):** Full graph traversal for each route (SQL, Documents, Both), clarification early-exit, cache hit short-circuit, cache write conditions.

### End-to-End Tests (13 scenarios, requires live infrastructure)

```bash
python tests/test_hybrid_rag.py
```

Runs against real Oracle, PostgreSQL, Redis, and Groq API. Flushes Redis cache before each run for deterministic results. Validates:

- 4 SQL-only queries (correct routing, SQL data present, no docs)
- 4 document-only queries (correct routing, docs present, no SQL)
- 3 hybrid queries (correct routing, both data sources hit)
- 1 clarification edge case (vague query triggers follow-up)
- 1 cache hit edge case (repeated query returns cached answer)

### CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main`/`develop` and on PRs:

1. **Lint** — `ruff check src/rag-system/ tests/`
2. **Test** — `pytest tests/test_security.py tests/test_nodes.py` with 60% minimum coverage
3. **Docker build** — builds the image and verifies it starts

Workflow tests (`test_workflow.py`, `test_ingest.py`) are excluded from CI because they import `workflow.py` which requires LangGraph state compilation. Run them locally with `PYTHONPATH=src/rag-system pytest tests/ -v`.

## Switching LLM Providers

The LLM is configured in `config.py` and controlled by the `LLM_MODEL` environment variable. Swap providers by changing the import and setting the corresponding API key:

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
docker-compose up -d          # Start all 6 services
docker-compose down            # Stop (preserves data volumes)
docker-compose down -v         # Stop and destroy all data
```

### Logs

```bash
docker-compose logs -f app     # Structured JSON logs from the application
docker logs -f rag-app         # Same, via Docker directly
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
| **Checkpointer** | In-memory (`MemorySaver`) — audit trails lost on restart | Swap to `PostgresSaver` for durable state |
| **PII patterns** | Regex-based (emails, SSNs, phones, cards, IBANs) — no named entity recognition | Add spaCy NER or a dedicated PII service for names, addresses, dates of birth |
| **Auth** | API key validation against env var | Integrate with OAuth2 / OIDC provider; add RBAC for audit endpoint access |
| **Rate limiting** | In-memory via slowapi | Use Redis-backed rate limiter for multi-worker consistency |
| **Embedding model** | all-MiniLM-L6-v2 (~80MB, 384-dim) — English-only, general-purpose | Evaluate domain-specific financial embeddings for compliance terminology |
| **Excel ingestion** | Row-per-document with column headers — no table-aware chunking | Add semantic grouping for multi-row records and cross-sheet references |
| **Model fallback** | `ModelFallbackChain` class exists but is not wired into the main workflow | Configure fallback chain in `config.py` for production resilience |
| **Horizontal scaling** | Single-instance with 4 Uvicorn workers | Stateless app (Redis-backed) is ready for multi-instance deployment behind a load balancer |
