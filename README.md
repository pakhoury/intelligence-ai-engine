# Compliance RAG System - Hybrid Retrieval-Augmented Generation Engine

An AI-powered question-answering system that combines **Oracle SQL queries** with **document semantic search** (PGVector) to answer compliance-related questions. Built with LangGraph, FastAPI, and Llama 3.3 via Groq.

## Architecture

```
                        User Question
                             |
                             v
                      +-------------+
                      |   FastAPI    |
                      |  POST /query|
                      +------+------+
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
                            | router |  (LLM classifies query)
                            +---+----+
                                |
                                v
                          +-----------+
                          |  clarify  |  (asks follow-ups if vague)
                          +-----+-----+
                           /    |    \
                     SQL  /  BOTH  \  DOCS
                         /     |      \
                        v      v       v
                 +----------+  |  +-----------------+
                 | sql_path |  |  | vector_retrieval|
                 +----+-----+  |  +--------+--------+
                      |        v           |
                      |  +----------+      |
                      +->| sql_path |      |
                         +----+-----+      |
                              |            |
                              v            |
                       +-----------------+ |
                       |vector_retrieval |<+
                       +--------+--------+
                                |
                                v
                      +---------+---------+
                      | answer_generator  |  (combines SQL + docs)
                      +---------+---------+
                                |
                                v
                        +-------+-------+
                        |   reviewer    |  (scores 0-10)
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

## Routing Paths

| Route | When | Pipeline |
|-------|------|----------|
| **SQL** | Counts, rankings, aggregates, specific records | cache -> router -> clarify -> sql_path -> answer -> reviewer -> cache_write |
| **Documents** | Policies, explanations, definitions, procedures | cache -> router -> clarify -> vector_retrieval -> answer -> reviewer -> cache_write |
| **Both** | Questions needing data + policy context | cache -> router -> clarify -> sql_path -> vector_retrieval -> answer -> reviewer -> cache_write |
| **Cache Hit** | Repeated question within 24h | cache -> END |
| **Clarification** | Vague/ambiguous query | cache -> router -> clarify -> END |

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Llama 3.3 70B via [Groq](https://groq.com) (free tier) | Routing, SQL generation, answering, reviewing |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (local) | Document embedding and semantic search |
| **Orchestration** | LangGraph | Stateful workflow with conditional routing |
| **API** | FastAPI + Uvicorn | REST API |
| **Structured Data** | Oracle XE 21c | Compliance violations, audit findings, risk events, controls |
| **Vector Store** | PostgreSQL 16 + PGVector | Semantic search over compliance documents |
| **Cache** | Redis 7 | Response caching (24h TTL) |
| **Containerization** | Docker Compose | All services in one command |

## Project Structure

```
intelligence-ai-engine/
├── docker-compose.yml            # All 4 services (Oracle, Postgres, Redis, App)
├── Dockerfile                    # Python 3.11 app image
├── requirement.txt               # Python dependencies
├── .env                          # Environment variables (API keys, DB credentials)
├── .gitignore
├── .dockerignore
├── metadata_mapping.json         # PDF metadata for document ingestion
├── search.py                     # Standalone semantic search tool
├── generate_pdfs.py              # Script to generate sample compliance PDFs
│
├── documents/                    # Compliance PDF documents
│   ├── compliance-policy-2024.pdf
│   ├── risk-management-framework-v2.pdf
│   ├── q3-2024-audit-report.pdf
│   └── soe-45678-incident-report.pdf
│
├── oracle-init/                  # Oracle database initialization
│   ├── 01_create_schema.sql      # Create compliance_user
│   ├── 02_create_tables.sql      # Create 4 tables
│   └── 03_seed_data.sql          # 33 rows of sample data
│
├── src/rag-system/               # Main application
│   ├── main.py                   # FastAPI app (POST /query, GET /health)
│   ├── config.py                 # LLM, DB connector, Redis client setup
│   ├── workflow.py               # LangGraph StateGraph definition
│   ├── nodes.py                  # All workflow node functions
│   ├── catalog.json              # Oracle table catalog with business descriptions
│   │
│   ├── database/
│   │   ├── base.py               # Abstract DatabaseConnector
│   │   └── oracle.py             # Oracle implementation with LLM table selection
│   │
│   ├── ingest/
│   │   └── ingest.py             # PDF ingestion into PGVector
│   │
│   └── prompts/
│       ├── router.txt            # Query classification (SQL/DOCUMENTS/BOTH)
│       ├── clarification.txt     # Ambiguity detection
│       ├── sql_agent.txt         # Oracle SQL generation
│       ├── final_answer.txt      # Answer synthesis
│       └── reviewer.txt          # Quality scoring
│
└── tests/
    └── test_hybrid_rag.py        # End-to-end test suite (13 test cases)
```

## Prerequisites

- **Docker Desktop** (Windows/Mac) or Docker Engine (Linux)
- **Groq API Key** (free) - get one at https://console.groq.com/keys

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd intelligence-ai-engine
```

Create a `.env` file:

```env
REDIS_HOST=localhost
REDIS_PORT=6379

GROQ_API_KEY=gsk_your_groq_api_key_here

# Oracle Connection
ORACLE_USER=compliance_user
ORACLE_PASSWORD=compliance_pass
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE=XEPDB1
ORACLE_SCHEMA=COMPLIANCE_USER

# PostgreSQL Connection
POSTGRES_HOST=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag_db
```

### 2. Start all services

```bash
docker-compose up -d
```

This starts 4 containers:
- `rag-oracle` - Oracle XE 21c (takes ~2 min on first start)
- `rag-postgres` - PostgreSQL 16 with PGVector
- `rag-redis` - Redis 7
- `rag-app` - FastAPI application

### 3. Initialize Oracle database

The Oracle init scripts need to be run manually on first setup:

```bash
docker cp oracle-init/01_create_schema.sql rag-oracle:/tmp/
docker cp oracle-init/02_create_tables.sql rag-oracle:/tmp/
docker cp oracle-init/03_seed_data.sql rag-oracle:/tmp/

docker exec rag-oracle bash -c "sqlplus system/oracle_admin_pass@localhost:1521/XEPDB1 @/tmp/01_create_schema.sql"
docker exec rag-oracle bash -c "sqlplus system/oracle_admin_pass@localhost:1521/XEPDB1 @/tmp/02_create_tables.sql"
docker exec rag-oracle bash -c "sqlplus system/oracle_admin_pass@localhost:1521/XEPDB1 @/tmp/03_seed_data.sql"
```

### 4. Ingest documents into PGVector

```bash
docker exec rag-app python ingest/ingest.py
```

This loads 4 compliance PDFs, chunks them into 19 pieces, and stores embeddings in PostgreSQL.

### 5. Verify

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many compliance violations per severity level?"}'
```

## API Endpoints

### POST /query

Send a compliance question and get an AI-generated answer.

**Request:**
```json
{
  "question": "Show me all KYC violations and explain what our compliance policy says about KYC"
}
```

**Response:**
```json
{
  "answer": "Based on SQL results, there are 2 KYC violations... Our compliance policy Section 5.1 requires all customer accounts to undergo KYC verification before activation...",
  "cache_hit": false
}
```

### GET /health

Returns `{"status": "healthy"}`.

## Oracle Database Schema

The system uses 4 tables seeded with sample compliance data:

| Table | Records | Description |
|-------|---------|-------------|
| `COMPLIANCE_VIOLATIONS` | 10 | Violations with severity, status, financial impact |
| `AUDIT_FINDINGS` | 7 | Internal/external audit findings with risk ratings |
| `CONTROL_MAPPINGS` | 8 | Controls mapped to compliance requirements |
| `RISK_EVENTS` | 8 | Risk incidents with loss amounts |

Table selection is **LLM-driven** - the Oracle connector passes a business catalog (`catalog.json`) to the LLM, which picks the most relevant tables for each query.

## Document Store

4 sample compliance PDFs are ingested into PGVector:

| Document | Type | Content |
|----------|------|---------|
| Corporate Compliance Policy 2024 | Policy | AML, KYC, data privacy, code of conduct, whistleblower protection |
| Risk Management Framework v2.0 | Framework | Market/credit/operational/liquidity risk, VaR limits, stress testing |
| Q3 2024 Internal Audit Report | Report | 4 findings: KYC gaps, trade surveillance, access control, reporting |
| SOE-45678 Incident Report | Incident | Unauthorized cross-trades, root cause analysis, remediation plan |

## Semantic Search Tool

Search documents directly without going through the full RAG pipeline:

```bash
python search.py "KYC requirements and customer verification"
python search.py "unauthorized trading incident" --top 5
```

## Testing

Run the full test suite covering all routing scenarios:

```bash
cd src/rag-system
python ../../tests/test_hybrid_rag.py
```

### Test Scenarios

| Category | Tests | What it validates |
|----------|-------|-------------------|
| SQL-only (4) | Violation counts, dept rankings, risk events, audit findings | Routes to Oracle, returns structured data |
| Docs-only (4) | AML policy, whistleblower, risk framework, data privacy | Routes to PGVector, returns document chunks |
| Both (3) | KYC violations + policy, trading + incident, risk events + VaR | Hits both Oracle and PGVector |
| Edge cases (2) | Vague query (clarification), repeated query (cache hit) | Clarification flow, Redis caching |

## Example Queries

### SQL queries (structured data)
- "How many compliance violations do we have for each severity level?"
- "Which departments have the highest total financial impact from violations?"
- "List all risk events with loss amount greater than 100000"
- "How many high risk audit findings are currently open?"

### Document queries (policies and procedures)
- "What is the whistleblower protection policy?"
- "Explain the Anti-Money Laundering and KYC requirements"
- "What does the risk management framework say about stress testing?"
- "What are the data privacy requirements for customer PII?"

### Hybrid queries (both sources)
- "Show me all KYC violations and explain what our compliance policy says about KYC"
- "List unauthorized trading violations and describe the incident report for SOE-45678"
- "What are our market risk events and what does the framework say about VaR limits?"

## Monitoring

### Container logs
```bash
# All containers
docker-compose logs -f

# Specific container
docker logs -f rag-app

# Or use Docker Desktop: Containers -> click container -> Logs tab
```

### Container status
```bash
docker-compose ps
```

### Redis cache
```bash
# View cached keys
docker exec rag-redis redis-cli KEYS "*"

# Flush cache
docker exec rag-redis redis-cli FLUSHALL
```

### PGVector data
```bash
# Count stored chunks
docker exec rag-postgres psql -U postgres -d rag_db -c "
  SELECT c.name, COUNT(*) as chunks
  FROM langchain_pg_embedding e
  JOIN langchain_pg_collection c ON e.collection_id = c.uuid
  GROUP BY c.name;"

# Browse documents
docker exec rag-postgres psql -U postgres -d rag_db -c "
  SELECT e.cmetadata->>'title' AS title,
         e.cmetadata->>'category' AS category,
         LEFT(e.document, 100) AS preview
  FROM langchain_pg_embedding e
  JOIN langchain_pg_collection c ON e.collection_id = c.uuid
  WHERE c.name = 'compliance_docs';"
```

### Oracle data
```bash
docker exec rag-oracle bash -c "echo 'SELECT COUNT(*) FROM COMPLIANCE_VIOLATIONS;' | \
  sqlplus -s compliance_user/compliance_pass@localhost:1521/XEPDB1"
```

## Switching LLM Providers

The system is designed to swap LLM providers easily. Edit `config.py` and `requirement.txt`:

### Groq (current - free)
```python
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1200)
```

### OpenAI
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1200)
```

### Anthropic (Claude)
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=1200)
```

### Google Gemini
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_output_tokens=1200)
```

Set the corresponding API key in `.env`:
- Groq: `GROQ_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

## Stopping Services

```bash
# Stop all containers (preserves data)
docker-compose down

# Stop and remove all data (fresh start)
docker-compose down -v
```
# intelligence-ai-engine
