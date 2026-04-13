"""
End-to-end tests for the Hybrid RAG system.
Tests all three routing scenarios: SQL-only, Documents-only, and Both.

Prerequisites:
  - docker-compose up -d (Oracle, Postgres, Redis running)
  - python ingest/ingest.py (documents ingested into PGVector)
  - OPENAI_API_KEY set in .env

Usage:
  cd src/rag-system
  python -m pytest ../../tests/test_hybrid_rag.py -v -s
  OR
  python ../../tests/test_hybrid_rag.py
"""
import sys
import os
import asyncio
import json

# Add src/rag-system to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rag-system"))


def print_result(label: str, result: dict):
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{'='*70}")
    print(f"Route:     {result.get('route', 'N/A')}")
    print(f"Cache Hit: {result.get('cache_hit', False)}")
    print(f"SQL Data:  {str(result.get('sql_result', 'N/A'))[:200]}")
    print(f"Docs:      {len(result.get('retrieved_docs', []))} chunks retrieved")
    print(f"Score:     {result.get('review_score', 'N/A')}")
    print(f"Clarify:   {result.get('needs_clarification', False)}")
    print(f"----- Answer -----")
    print(result.get("final_answer", "NO ANSWER"))
    print(f"{'='*70}\n")


# ===================================================================
# TEST QUERIES - designed to exercise all routing paths
# ===================================================================

# --- SQL-only queries (structured data from Oracle) ---
SQL_QUERIES = [
    {
        "name": "SQL: Count violations by severity",
        "question": "How many compliance violations do we have for each severity level?",
        "expect_route": "sql",
        "expect_sql": True,
        "expect_docs": False,
    },
    {
        "name": "SQL: Top departments by financial impact",
        "question": "Which departments have the highest total financial impact from violations?",
        "expect_route": "sql",
        "expect_sql": True,
        "expect_docs": False,
    },
    {
        "name": "SQL: Open risk events",
        "question": "List all risk events with loss amount greater than 100000",
        "expect_route": "sql",
        "expect_sql": True,
        "expect_docs": False,
    },
    {
        "name": "SQL: Audit findings by risk rating",
        "question": "How many high risk audit findings are currently open?",
        "expect_route": "sql",
        "expect_sql": True,
        "expect_docs": False,
    },
]

# --- Document-only queries (policies from PGVector) ---
DOC_QUERIES = [
    {
        "name": "DOCS: AML policy explanation",
        "question": "What is our Anti-Money Laundering policy and what are the KYC requirements?",
        "expect_route": "documents",
        "expect_sql": False,
        "expect_docs": True,
    },
    {
        "name": "DOCS: Whistleblower protection",
        "question": "Tell me about the whistleblower protection policy",
        "expect_route": "documents",
        "expect_sql": False,
        "expect_docs": True,
    },
    {
        "name": "DOCS: Risk management framework",
        "question": "Explain our risk appetite statement and stress testing methodology",
        "expect_route": "documents",
        "expect_sql": False,
        "expect_docs": True,
    },
    {
        "name": "DOCS: Data privacy rules",
        "question": "What are the data privacy and protection requirements for customer PII?",
        "expect_route": "documents",
        "expect_sql": False,
        "expect_docs": True,
    },
]

# --- Both queries (need SQL data + document context) ---
BOTH_QUERIES = [
    {
        "name": "BOTH: KYC violations with policy context",
        "question": "Show me all KYC-related violations and explain what our compliance policy says about KYC requirements",
        "expect_route": "both",
        "expect_sql": True,
        "expect_docs": True,
    },
    {
        "name": "BOTH: Trading violations with incident context",
        "question": "List unauthorized trading violations from the database and describe the incident report findings for SOE-45678",
        "expect_route": "both",
        "expect_sql": True,
        "expect_docs": True,
    },
    {
        "name": "BOTH: Risk events with framework context",
        "question": "What are our market risk events and losses, and what does the risk management framework say about VaR limits?",
        "expect_route": "both",
        "expect_sql": True,
        "expect_docs": True,
    },
]

# --- Edge cases ---
EDGE_QUERIES = [
    {
        "name": "EDGE: Vague question needing clarification",
        "question": "Show me the data",
        "expect_clarification": True,
    },
    {
        "name": "EDGE: Cache hit (run same query twice)",
        "question": "How many compliance violations do we have for each severity level?",
        "expect_cache_hit": True,
    },
]

ALL_TEST_GROUPS = [
    ("SQL-ONLY ROUTING", SQL_QUERIES),
    ("DOCUMENT-ONLY ROUTING", DOC_QUERIES),
    ("HYBRID (BOTH) ROUTING", BOTH_QUERIES),
    ("EDGE CASES", EDGE_QUERIES),
]


async def run_single_test(workflow_app, test_case: dict) -> dict:
    """Run a single test and return the result with pass/fail info."""
    result = await workflow_app.ainvoke({"question": test_case["question"]})
    print_result(test_case["name"], result)

    checks = []

    # Check route
    if "expect_route" in test_case:
        actual_route = result.get("route", "")
        passed = actual_route == test_case["expect_route"]
        checks.append(("Route", passed, f"expected={test_case['expect_route']}, actual={actual_route}"))

    # Check SQL data present
    if "expect_sql" in test_case:
        has_sql = bool(result.get("sql_result")) and result.get("sql_result") != "N/A - no SQL data retrieved"
        passed = has_sql == test_case["expect_sql"]
        checks.append(("SQL Data", passed, f"expected={'present' if test_case['expect_sql'] else 'absent'}, actual={'present' if has_sql else 'absent'}"))

    # Check docs present
    if "expect_docs" in test_case:
        has_docs = len(result.get("retrieved_docs", [])) > 0
        passed = has_docs == test_case["expect_docs"]
        checks.append(("Doc Data", passed, f"expected={'present' if test_case['expect_docs'] else 'absent'}, actual={'present' if has_docs else 'absent'}"))

    # Check clarification
    if "expect_clarification" in test_case:
        needs_clarification = result.get("needs_clarification", False)
        passed = needs_clarification == test_case["expect_clarification"]
        checks.append(("Clarification", passed, f"expected={test_case['expect_clarification']}, actual={needs_clarification}"))

    # Check cache hit
    if "expect_cache_hit" in test_case:
        cache_hit = result.get("cache_hit", False)
        passed = cache_hit == test_case["expect_cache_hit"]
        checks.append(("Cache Hit", passed, f"expected={test_case['expect_cache_hit']}, actual={cache_hit}"))

    # Check answer exists (unless clarification expected)
    if not test_case.get("expect_clarification"):
        has_answer = bool(result.get("final_answer"))
        checks.append(("Has Answer", has_answer, f"answer={'present' if has_answer else 'MISSING'}"))

    return {
        "name": test_case["name"],
        "checks": checks,
        "all_passed": all(c[1] for c in checks),
    }


async def run_all_tests():
    from workflow import app as workflow_app

    all_results = []
    total_pass = 0
    total_fail = 0

    for group_name, queries in ALL_TEST_GROUPS:
        print(f"\n\n{'#'*70}")
        print(f"# {group_name}")
        print(f"{'#'*70}")

        for test_case in queries:
            try:
                result = await run_single_test(workflow_app, test_case)
                all_results.append(result)

                for check_name, passed, detail in result["checks"]:
                    status = "PASS" if passed else "FAIL"
                    if passed:
                        total_pass += 1
                    else:
                        total_fail += 1
                    print(f"  [{status}] {check_name}: {detail}")

            except Exception as e:
                print(f"  [ERROR] {test_case['name']}: {e}")
                total_fail += 1
                all_results.append({"name": test_case["name"], "checks": [], "all_passed": False, "error": str(e)})

    # Summary
    print(f"\n\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        status = "PASS" if r.get("all_passed") else "FAIL"
        error = f" (ERROR: {r['error']})" if "error" in r else ""
        print(f"  [{status}] {r['name']}{error}")

    print(f"\nTotal checks: {total_pass + total_fail}  |  Passed: {total_pass}  |  Failed: {total_fail}")
    print(f"{'='*70}")

    return total_fail == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
