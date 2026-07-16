# ARIA — Analytical Risk Intelligence Agent

A federated Retrieval-Augmented Generation platform that answers compliance and risk questions by combining **structured data** (Oracle SQL) with **unstructured knowledge** (hybrid document search: pgvector semantic + Postgres full-text, fused with Reciprocal Rank Fusion). The system classifies each question, selects an execution strategy, chains data sources when needed, compiles deterministic SQL for registered metrics, synthesizes an answer, and validates it through 5 deterministic validators + an LLM reviewer before caching.

Built for regulated financial services where answers must be grounded, auditable, and reproducible.

## Why This Architecture

Traditional RAG systems retrieve documents and generate answers. Compliance teams need more: questions span structured data ("How many KYC violations this quarter?") and policy interpretation ("What does our framework say about VaR limits?") — often in the same question. And frequently, the answer from one source is needed to query the other ("What is the compliance cost for 2024?" needs the cost formula from docs before generating the right SQL).

This system solves that with a **strategy-based hybrid pipeline**: an LLM classifies each query into one of five execution strategies, retrieves from the appropriate sources — chaining them when needed — and generates a grounded answer. Registered business metrics bypass the LLM entirely via a **deterministic SQL compiler**, eliminating hallucination risk for known KPIs. Every response includes a confidence score and full execution metadata.

### Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| LangGraph over LangChain agents | Deterministic state machine with explicit conditional edges. No autonomous tool-calling loops. | Less flexible than ReAct, but predictable cost and latency. |
| 5 routing strategies | `docs_then_sql` and `sql_then_docs` chain sources so one source's output informs the next. | More complex routing, but handles multi-source questions correctly. |
| Deterministic metric compiler | Registered metrics produce SQL from structured definitions (SELECT, WHERE, GROUP BY clauses). Zero LLM calls, same input = same SQL. | Only works for pre-registered metrics. Ad-hoc queries still use LLM. |
| Dual-path SQL execution | Compiler-first: try deterministic compilation, fall through to LLM only if metric is unresolved or non-compilable. | Slight code complexity, but eliminates LLM hallucination for all registered metrics. |
| Per-table catalog with on-demand loading | `database_metadata.json` index + one JSON per table. Tables loaded and cached only when needed. | More files to manage, but scales to 50+ tables without startup cost. |
| Model fallback chain | Primary LLM (llama-3.3-70b) + fallback (llama-3.1-8b). Circuit breaker triggers automatic fallback. | Fallback model is smaller (lower quality), but service stays up. |
| 5 deterministic validators | SQL safety, result sanity, answer grounding, number grounding, metric citation — each with independent score caps. | Adds review latency, but catches failures LLM reviewer misses. |
| Confidence scoring | `high/medium/low` computed from: compiled vs LLM path, review score, validator failures. | Requires tuning thresholds, but gives users actionable trust signal. |
| First-turn-only caching | Cache keyed on normalized question text; follow-ups (any request with history) bypass the cache entirely. Cached payload stores answer + review score + confidence + metric metadata. | Follow-ups never benefit from cache, but context-dependent answers can never leak across sessions. |
| MemorySaver fatal in production | PostgreSQL checkpointer failure raises `RuntimeError` unless `ENVIRONMENT=development`. | Crashes the app on PG failure, but audit trail loss is unacceptable for compliance. |
| SQL validation + retry loop | Column references validated against catalog pre-execution. Errors fed back to LLM (up to 2 retries). | Adds latency on malformed queries, but prevents cryptic Oracle errors. |
| Hybrid retrieval (vector + full-text, RRF fusion) | Embeddings miss exact identifiers (incident IDs, violation codes); Postgres full-text misses paraphrase. Both run concurrently (k=20 each), fused by Reciprocal Rank Fusion to top-6. Metadata (`doc_type`) filters apply to both sides. | One extra query per retrieval, but exact-ID questions ("incident SOE-45678") retrieve the right chunk. Lexical failure degrades gracefully to vector-only. |
| Persistent PostgreSQL checkpointer | LangGraph state stored in PostgreSQL, surviving restarts. | Requires PG dependency, but enables regulatory audit trails. |
| Non-blocking Redis via `asyncio.to_thread` | Cache, history operations run in background threads. | Thread pool overhead, but consistent with Oracle call pattern. |
| Request-level timeout | `asyncio.wait_for` with configurable timeout (default 60s). Returns 504 on hang. | Kills slow queries, but prevents unbounded resource consumption. |
| PII redaction + DLP gate | Two-layer defense: SQL results scrubbed pre-answer, final answer scanned post-review. | Regex-based (no NER), but catches emails, SSNs, phones, cards, IBANs with negligible latency. |

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
                      [END]  +------------------+
                             | context_resolver  |  (rewrites follow-ups)
                             +--------+---------+
                                      |
                                      v
                                 +--------+
                                 | router |  (LLM: 5 strategies)
                                 +---+----+
                                     |
                                     v
                               +-----------+
                               |  clarify  |  (LLM detects ambiguity)
                               +-----+-----+
                                     |
                                     v
                            +----------------+
                            | metric_resolver|  (LLM: match registered metrics)
                            +--------+-------+
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
                          | (LLM + 5     |   (if score < 7
                          |  validators) |    and attempts < 2)
                          +-------+-------+
                                  |
                                  v
                          +-------+-------+
                          |  cache_write  |  (enriched key, caches if score >= 7)
                          +-------+-------+
                                  |
                                  v
                               [END]
```

### Dual-Path SQL Execution

```
            metric_resolver matches?
                  /         \
               YES           NO
                |             |
                v             v
        MetricCompiler    LLM SQL Path
     (deterministic SQL)  (table selection +
                          SQL generation +
                          validation + retry)
                |             |
                v             v
           execute_query  execute_query
                \           /
                 v         v
              answer_generator
```

Registered metrics (7 currently) produce **identical SQL every time** — no LLM involved. Ad-hoc queries go through the LLM path with 3-layer validation.

### Routing Strategies

| Strategy | Flow | Example | Typical Latency |
|----------|------|---------|-----------------|
| **sql_only** | metric_resolver -> [compiler or LLM SQL] -> answer -> reviewer | "How many violations in Q4?" | 2-4s |
| **docs_only** | vector_retrieval -> answer -> reviewer | "What is our access control policy?" | 2-3s |
| **docs_then_sql** | vector_retrieval -> **extract** -> sql_path -> answer -> reviewer | "What is compliance cost for 2024?" (needs formula from docs first) | 4-6s |
| **sql_then_docs** | sql_path -> **extract** -> vector_retrieval -> answer -> reviewer | "Top violation and its policy?" (needs data before doc search) | 4-6s |
| **parallel** | sql_path -> vector_retrieval -> answer -> reviewer | "List violations and summarize the framework" | 3-5s |
| **Cache Hit** | cache_check -> END | Repeated question within 24h | <50ms |
| **Clarification** | router -> clarify -> END | "Show me the data" (vague) | ~1s |

### Chained Strategies: How Extract Works

**docs_then_sql** ("What is the compliance cost for 2024?"):
1. `vector_retrieval` finds the cost formula in policy documents
2. `extract` pulls the formula: "SUM(fine_amount + remediation_cost + audit_fees)"
3. `sql_path` receives this formula as context and generates the correct SQL

**sql_then_docs** ("Most common violation and its policy?"):
1. `sql_path` queries the database and finds "ACCESS_CONTROL: 47 incidents"
2. `extract` pulls the key term: "ACCESS_CONTROL"
3. `vector_retrieval` uses "ACCESS_CONTROL" as the search query instead of the original question

If extraction fails, the downstream source receives the original question instead of extracted context.

## Deterministic Metric Compiler

The biggest correctness improvement over typical RAG systems. Registered metrics bypass the LLM entirely:

```
Question: "What is the compliance effectiveness score?"
    |
    v
Metric Resolver: matches "compliance_effectiveness_score" (confidence: 1.0)
    |
    v
Metric Compiler: assembles SQL from structured definition
    |
    v
SQL:  SELECT
        COUNT(*) AS total_violations,
        SUM(CASE WHEN STATUS = 'Closed' THEN 1 ELSE 0 END) AS closed_violations,
        ROUND(SUM(CASE WHEN STATUS = 'Closed' THEN 1 ELSE 0 END) * 100.0
              / NULLIF(COUNT(*), 0), 2) AS effectiveness_score
      FROM COMPLIANCE_VIOLATIONS
      WHERE VIOLATION_DATE >= ADD_MONTHS(SYSDATE, -12)
      FETCH FIRST 500 ROWS ONLY
```

**Same input always produces same SQL.** Parameters are sanitized, row limits enforced, no LLM interpretation involved.

### Registered Metrics

| Metric | Unit | Compilation Mode | Tables |
|--------|------|------------------|--------|
| Compliance Effectiveness Score | % | clause | COMPLIANCE_VIOLATIONS |
| Violation Severity Distribution | count | clause | COMPLIANCE_VIOLATIONS |
| Audit Finding Resolution Rate | % | clause | AUDIT_FINDINGS |
| Regulatory Exposure Index | USD | clause | COMPLIANCE_VIOLATIONS |
| Department Risk Score | score | template | COMPLIANCE_VIOLATIONS, AUDIT_FINDINGS, RISK_EVENTS |
| Control Coverage Ratio | % | clause | CONTROL_MAPPINGS |
| Mean Time to Resolution | days | clause | COMPLIANCE_VIOLATIONS |

Each metric is defined in `metrics_catalog.yaml` with: formula, calculation steps, parameters (with defaults), thresholds, and interpretation text. The compiler supports two modes: **clause-based** (assembles SELECT/WHERE/GROUP BY from structured definitions) and **template-based** (substitutes parameters into pre-written SQL for complex multi-table queries).

## Correctness & Validation

### 5 Deterministic Validators

Every answer passes through 5 validators that run alongside the LLM reviewer. Each has an independent score cap:

| Validator | What It Catches | Score Cap |
|-----------|----------------|-----------|
| **SQL Safety** | DROP, DELETE, injection, unbalanced parens, statement chaining | 0.0 |
| **Result Sanity** | Empty results, execution errors, suspiciously large numbers | 3.0-6.0 |
| **Answer Grounding** | Answer claims data but no SQL/docs were retrieved | 3.0 |
| **Number Grounding** | Numeric claims in answer don't appear in SQL result | 5.0 |
| **Metric Citation** | Compiled metric answer doesn't reference the metric name | 6.0 |

### Confidence Scoring

Every response includes a confidence level computed from the execution path:

| Confidence | Criteria |
|------------|----------|
| **high** | Compiled metric + review score >= 8.0 + no validator failures |
| **medium** | LLM SQL path + good score + supporting data; or compiled with moderate score |
| **low** | Retries needed, validator failures, or low review score |

## Scalability

### Per-Table Catalog

```
catalog/
├── database_metadata.json          # Index: table name + description + file pointer
└── tables/
    ├── COMPLIANCE_VIOLATIONS.json   # Full column definitions
    ├── AUDIT_FINDINGS.json
    ├── CONTROL_MAPPINGS.json
    └── RISK_EVENTS.json
```

- **At startup:** only the lightweight index is loaded (~50 lines at 50 tables)
- **At query time:** per-table files loaded on demand and cached
- **Adding a table:** one JSON file + one line in the index

### Token Budget at Scale

| Stage | 4 Tables (current) | 50 Tables |
|-------|-------------------|-----------|
| Table selection prompt | ~400 tokens | ~2,000 tokens (1 line per table) |
| SQL generation prompt | ~1,000 tokens | ~1,500 tokens (selected tables only) |
| Compiled metric SQL | 0 LLM tokens | 0 LLM tokens |
| Vector retrieval | k=6, filtered | k=6, metadata-filtered |

### Additional Scaling Features

- **Synonym-enhanced keyword fallback** — 12 compliance domain synonyms (exposure -> RISK_EVENTS, penalty -> VIOLATIONS, etc.) for when LLM table selection fails
- **Model fallback chain** — primary LLM failure triggers automatic fallback to secondary model via circuit breaker
- **Table cap raised to 8** — supports cross-domain queries spanning multiple schemas
- **Metadata-filtered vector search** — PGVector JSONB filtering by `doc_type`/`category` based on route context

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Llama 3.3 70B via Groq + fallback (3.1 8B) | Routing, SQL generation, extraction, answering, reviewing |
| Embeddings | all-MiniLM-L6-v2 (local) | Document embedding and semantic search |
| Orchestration | LangGraph | Stateful workflow with conditional routing and reflection |
| API | FastAPI + Uvicorn | REST API with auth, rate limiting, CORS |
| Structured Data | Oracle XE 21c | Compliance violations, audit findings, risk events, controls |
| Vector Store | PostgreSQL 16 + PGVector | Semantic search with JSONB metadata filtering |
| Cache | Redis 7 | Response caching (24h TTL) + conversation history (1h TTL) |
| Checkpointer | PostgreSQL (langgraph-checkpoint-postgres) | Persistent audit trail (mandatory in production) |
| Metrics | Prometheus + Grafana | Request latency, LLM tokens/cost, cache hit rate, review scores |
| Tracing | OpenTelemetry (OTLP) | Distributed tracing with per-node spans |
| CI/CD | GitHub Actions | Lint -> All tests -> Docker build |

## Project Structure

```
intelligence-ai-engine/
├── docker-compose.yml              # 6 services: Oracle, Postgres, Redis, App, Prometheus, Grafana
├── docker-compose.override.yml     # Dev: hot reload
├── Dockerfile                      # Multi-stage, non-root, 4 workers, healthcheck
├── pyproject.toml                  # Ruff, mypy, pytest config
├── requirement.txt
├── .env                            # Runtime config (API keys, DB credentials)
│
├── src/aria/
│   ├── main.py                     # FastAPI: /query, /health, /ready, /metrics, /audit
│   ├── config.py                   # LLM (primary + fallback), DB, Redis init
│   ├── workflow.py                 # LangGraph: 11 nodes, 5 strategies, reflection
│   ├── nodes.py                    # All workflow nodes + compiled metric path
│   ├── security.py                 # Auth, rate limit, injection guard, PII, DLP
│   ├── observability.py            # OTel + Prometheus + structured JSON logging
│   ├── llm_ops.py                  # Cost tracking, context window, model fallback chain
│   ├── validators.py               # 5 deterministic validators
│   ├── circuit_breaker.py          # Circuit breaker for LLM calls
│   ├── metrics_catalog.yaml        # 7 metric definitions with formulas + thresholds
│   ├── catalog/                    # Per-table database schema catalog
│   │   ├── database_metadata.json  # Table index (lightweight)
│   │   └── tables/                 # One JSON per table (loaded on demand)
│   ├── database/
│   │   ├── base.py                 # Abstract DatabaseConnector (on-demand catalog loading)
│   │   └── oracle.py               # Oracle: LLM table selection, SQL validation
│   ├── metrics/
│   │   ├── registry.py             # Immutable MetricDefinition dataclasses
│   │   ├── loader.py               # YAML -> MetricRegistry
│   │   ├── resolver.py             # Deterministic keyword-based metric resolution
│   │   ├── llm_resolver.py         # LLM-based metric resolution (higher accuracy)
│   │   ├── compiler.py             # MetricDefinition -> deterministic SQL
│   │   ├── catalog.py              # Legacy catalog context builder
│   │   └── validator.py            # Metric definition validation against DB catalog
│   ├── ingest/
│   │   └── ingest.py               # PDF + Excel ingestion into PGVector
│   └── prompts/                    # LLM prompt templates
│       ├── router.txt              # Query classification (5 strategies)
│       ├── clarification.txt       # Ambiguity detection
│       ├── context_resolver.txt    # Follow-up question rewriting
│       ├── extract.txt             # Context extraction between chained sources
│       ├── sql_agent.txt           # Oracle SQL generation
│       ├── metric_resolver.txt     # LLM metric matching
│       ├── final_answer.txt        # Answer synthesis
│       └── reviewer.txt            # Quality scoring (0-10)
│
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_nodes.py               # ~60 node-level unit tests
│   ├── test_metric_compiler.py     # 39 golden tests: resolver -> compiler -> SQL
│   ├── test_validators.py          # 39 validator tests (all 5 validators)
│   ├── test_eval.py                # 56 evaluation tests (21 golden cases)
│   ├── golden_dataset.yaml         # 21 golden cases: all routes + edge cases
│   ├── test_workflow.py            # 10 workflow integration tests
│   ├── test_metrics.py             # ~140 metric pipeline tests
│   ├── test_security.py            # ~35 security tests
│   └── test_hybrid_rag.py          # E2E tests (requires live infrastructure)
│
├── oracle-init/                    # DB init scripts (auto-run on first start)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
└── .github/workflows/ci.yml        # CI: lint -> all tests -> Docker build
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

Create `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
```

For local development (outside Docker):

```env
GROQ_API_KEY=gsk_your_key_here
REDIS_HOST=localhost
ORACLE_HOST=localhost
POSTGRES_HOST=localhost
ENVIRONMENT=development
```

### 2. Start

```bash
# Production (4 workers, audit trail enforced)
docker-compose -f docker-compose.yml up -d

# Development (hot reload, MemorySaver allowed)
docker-compose up -d
```

### 3. Ingest documents

```bash
docker exec rag-app python ingest/ingest.py
```

### 4. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the compliance effectiveness score?"}'
```

## API Reference

### POST /query

```json
// Request
{
  "question": "What is the compliance effectiveness score for the Legal department?",
  "session_id": "optional-for-multi-turn"
}

// Response
{
  "answer": "The Compliance Effectiveness Score for Legal is 55.0%, rated as Warning...",
  "cache_hit": false,
  "session_id": "abc-123",
  "trace_id": "tr-9f8e7d",
  "metadata": {
    "route": "sql_only",
    "resolved_metric": "compliance_effectiveness_score",
    "metric_version": "1.0",
    "compiled": true,
    "confidence": "high",
    "review_score": 9.0
  }
}
```

**Headers:** `X-API-Key` (required when `API_KEYS` is configured). Response includes `X-Trace-ID`.

**Rate limit:** 30 requests/minute per API key or IP.

### GET /health

Liveness probe. Returns 200 if the process is running.

### GET /ready

Readiness probe. Returns 200 if Oracle, PostgreSQL, and Redis are all reachable; 503 with degraded status otherwise.

### GET /audit/{thread_id}

Full state history for a session — every node's input/output state, step number, timestamp.

### GET /metrics

Prometheus-format metrics (20+ metric families).

## Multi-Turn Conversations

```bash
# Turn 1
curl -X POST http://localhost:8000/query \
  -d '{"question": "What is the compliance effectiveness score?", "session_id": "s1"}'
# -> "The score is 85%..."

# Turn 2 — references Turn 1
curl -X POST http://localhost:8000/query \
  -d '{"question": "Break that down by department", "session_id": "s1"}'
# -> context_resolver rewrites to "Break down the compliance effectiveness score by department"

# Turn 3 — references Turn 2
curl -X POST http://localhost:8000/query \
  -d '{"question": "What does our policy say about those?", "session_id": "s1"}'
# -> Routes to sql_then_docs, finds policy for the departments mentioned
```

History stored in Redis (1h TTL, last 10 turns, last 5 sent to LLM). All history sanitized against prompt injection.

## Security Model

| Layer | Threat | Mitigation |
|-------|--------|------------|
| **Network** | Unauthorized access | API key authentication via `X-API-Key` header |
| **Network** | Abuse / DoS | Rate limiting at 30 req/min per API key or IP. Request timeout (default 60s) |
| **Input** | Prompt injection | 11 regex patterns detect injection attempts. HTTP 400 on match |
| **Input** | Oversized input | Min 3, max 2,000 characters |
| **Context** | History-based injection | `sanitize_for_prompt()` strips injection markers, HTML tags |
| **Database** | SQL injection | SELECT-only allowlist. Blocks DROP, DELETE, INSERT, UPDATE, statement chaining, comment injection, Oracle packages (UTL_, DBMS_, SYS.) |
| **Database** | Unbounded results | All queries appended with `FETCH FIRST 500 ROWS ONLY` |
| **Output** | PII in SQL results | Regex-based PII scrub before answer generation |
| **Output** | PII in final answer | DLP gate in reviewer. PII redacted, score capped at 6.0 |
| **Container** | Privilege escalation | Non-root `appuser` inside container |

## Observability

### Prometheus Metrics (20+ families)

| Metric | Type | Labels |
|--------|------|--------|
| `rag_requests_total` | Counter | `status` |
| `rag_request_duration_seconds` | Histogram | — |
| `rag_cache_operations_total` | Counter | `result` (hit/miss) |
| `rag_route_total` | Counter | `route` |
| `rag_llm_calls_total` | Counter | `node`, `status` |
| `rag_llm_call_duration_seconds` | Histogram | `node` |
| `rag_llm_tokens_total` | Counter | `node`, `type` |
| `rag_llm_cost_usd` | Counter | `node` |
| `rag_node_duration_seconds` | Histogram | `node` |
| `rag_node_errors_total` | Counter | `node` |
| `rag_review_score` | Histogram | — |
| `rag_reflection_total` | Counter | — |
| `rag_circuit_breaker_state` | Gauge | `name` |
| `rag_validator_failures_total` | Counter | `validator` |

### Grafana Dashboard

9-panel dashboard auto-provisioned at http://localhost:3000 (admin/admin): request rate, latency percentiles, cache hit rate, route distribution, LLM latency by node, token consumption, review score distribution.

### Structured Logs

```json
{"timestamp": "2026-06-03T10:15:21Z", "level": "INFO", "trace_id": "a1b2c3", "node": "sql_path", "message": "Compiled metric SQL deterministically (clause): SELECT..."}
```

### Distributed Tracing

Every node and LLM call wrapped in OTel spans. Set `OTEL_EXPORTER_OTLP_ENDPOINT` for Jaeger, Tempo, or Datadog.

## Testing

### 379 Tests, 7 Suites

```bash
pytest tests/ -v    # All tests, no infrastructure required
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_nodes.py` | ~60 | Every workflow node in isolation |
| `test_metric_compiler.py` | 39 | Resolver -> compiler -> SQL for all 7 metrics |
| `test_validators.py` | 39 | All 5 deterministic validators |
| `test_eval.py` | 56 | 21 golden cases: routing + answer quality + confidence |
| `test_workflow.py` | 10 | Full graph integration, all 5 routes |
| `test_metrics.py` | ~140 | Registry, loader, resolver, LLM resolver, compiler, validator |
| `test_security.py` | ~35 | Auth, rate limits, injection, PII, DLP |

### Golden Evaluation Dataset

21 calibrated test cases in `tests/golden_dataset.yaml` covering:
- All 5 routing strategies
- All metric types (compliance, audit, risk, governance, operations)
- Edge cases (empty results, vague questions, multi-table JOINs)
- SQL injection attempts (DROP, comment injection)
- Chained strategies (docs_then_sql, sql_then_docs)
- Confidence scoring validation

### CI Pipeline

GitHub Actions on every push to `main`/`develop`:
1. **Lint** — `ruff check src/aria/ tests/`
2. **Test** — `pytest tests/` with 60% minimum coverage
3. **Docker build** — builds image and verifies it starts

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Primary LLM model |
| `LLM_FALLBACK_MODEL` | `llama-3.1-8b-instant` | Fallback LLM model |
| `LLM_TIMEOUT` | `30` | LLM request timeout (seconds) |
| `REQUEST_TIMEOUT` | `60` | Workflow execution timeout (seconds) |
| `ENVIRONMENT` | `development` | `development` allows MemorySaver fallback; anything else enforces PostgreSQL checkpointer |
| `API_KEYS` | *(empty = auth disabled)* | Comma-separated valid API keys |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed origins |
| `REDIS_HOST` | `localhost` | Redis host |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `ORACLE_HOST` | `localhost` | Oracle host |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(empty)* | OpenTelemetry OTLP endpoint |

## Failure Modes

| Failure | Behavior |
|---------|----------|
| Redis down | Cache miss on every request. History loads return empty. Graceful degradation. |
| Oracle down | SQL queries fail. Answer notes data unavailable. |
| PGVector down | Document queries return no context. Answer generated without docs. |
| Groq rate limit | Circuit breaker opens. Fallback model (llama-3.1-8b) takes over automatically. |
| Groq + fallback both down | Retries exhausted. Returns 500. |
| PG checkpointer down (prod) | **App crashes with RuntimeError.** Audit trail loss is not tolerated. |
| PG checkpointer down (dev) | Falls back to MemorySaver with warning. Audit trail lost on restart. |
| LLM generates bad SQL | Column validation catches errors pre-execution. Error fed back for retry (up to 2). |
| LLM reviewer unparseable | Score defaults to 0.0, triggering reflection loop. |
| Request timeout | `asyncio.wait_for` kills workflow after `REQUEST_TIMEOUT`. Returns HTTP 504. |

## Operations

```bash
docker-compose up -d                              # Dev (hot reload)
docker-compose -f docker-compose.yml up -d         # Prod (4 workers)
docker-compose down                                # Stop (preserves volumes)
docker exec rag-redis redis-cli KEYS "rag:answer:*"  # List cached answers
docker exec rag-redis redis-cli FLUSHDB             # Clear cache
```

| URL | Service |
|-----|---------|
| http://localhost:8000/docs | FastAPI interactive docs |
| http://localhost:8000/metrics | Prometheus metrics |
| http://localhost:9090 | Prometheus UI |
| http://localhost:3000 | Grafana dashboards (admin/admin) |
