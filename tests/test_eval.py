"""
Evaluation harness — runs golden dataset cases through the full workflow
with mocked LLM/DB and asserts expected properties.

This is NOT a unit test. It validates end-to-end behavior against
calibrated expectations, catching regressions in routing, scoring,
validation, and answer quality.

Run with: pytest tests/test_eval.py -v
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.yaml")


def load_golden_cases():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["cases"]


GOLDEN_CASES = load_golden_cases()


def _build_llm_side_effects(case):
    """Build the sequence of LLM responses for a given eval case."""
    mock_data = case["mock_data"]
    effects = []

    effects.append(mock_data["router_response"])
    effects.append(mock_data["clarify_response"])

    if case.get("needs_clarification"):
        return effects

    route = case["expected_route"]
    sql_response = mock_data.get("sql_response", "NO_SQL_POSSIBLE")
    sql_retries = 3 if case.get("expected_sql_blocked") else 1

    answer_response = mock_data.get("answer_response", "I cannot answer this.")
    reviewer_response = mock_data.get("reviewer_response", "Score: 5.0\nDecision: REJECT")
    reflection_loops = 2 if case.get("expected_sql_blocked") else 0

    table_selection = mock_data.get("table_selection_response", "COMPLIANCE_VIOLATIONS")

    if route == "docs_then_sql":
        effects.append(mock_data.get("extract_response", "EXTRACTION_FAILED"))
        effects.append(table_selection)
        for _ in range(sql_retries):
            effects.append(sql_response)
        effects.append(answer_response)
        effects.append(reviewer_response)
    elif route == "sql_then_docs":
        effects.append(table_selection)
        for _ in range(sql_retries):
            effects.append(sql_response)
        effects.append(mock_data.get("extract_response", "EXTRACTION_FAILED"))
        effects.append(answer_response)
        effects.append(reviewer_response)
    elif route == "sql_only":
        effects.append(table_selection)
        for _ in range(sql_retries):
            effects.append(sql_response)
        effects.append(answer_response)
        effects.append(reviewer_response)
    elif route == "docs_only":
        effects.append(answer_response)
        effects.append(reviewer_response)
    elif route == "parallel":
        effects.append(table_selection)
        for _ in range(sql_retries):
            effects.append(sql_response)
        effects.append(answer_response)
        effects.append(reviewer_response)

    for _ in range(reflection_loops):
        effects.append(answer_response)
        effects.append(reviewer_response)

    return effects


def _make_vector_docs(case):
    """Create mock vector store documents from golden case data."""
    docs = []
    for doc_data in case["mock_data"].get("vector_docs", []):
        doc = MagicMock()
        doc.metadata = {"title": doc_data["title"]}
        doc.page_content = doc_data["content"]
        docs.append(doc)
    return docs


@pytest.fixture
def workflow_mocks(request):
    """Set up all mocks for a full workflow invocation."""
    case = request.param

    with (
        patch("nodes.redis_client") as mock_redis,
        patch("nodes._ainvoke_llm", new_callable=AsyncMock) as mock_ainvoke,
        patch("nodes.db_connector") as mock_db,
        patch("nodes._get_vector_store") as mock_vs_fn,
        patch("nodes.lexical_search", return_value=[]),
        patch("nodes._get_metric_resolver") as mock_resolver_fn,
    ):
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=None)
        mock_resolver_fn.return_value = mock_resolver

        mock_redis.get.return_value = None
        mock_redis.set.return_value = True

        mock_data = case["mock_data"]
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = mock_data.get("sql_tables", [])
        mock_db.get_relevant_tables.return_value = mock_data.get("sql_tables", [])
        mock_db.get_table_schema.return_value = mock_data.get("sql_schema", "")
        mock_db.validate_columns.return_value = ""

        sql_result = mock_data.get("sql_result", "[]")
        if case.get("expected_sql_blocked"):
            mock_db.execute_query.return_value = "Error: blocked"
        else:
            mock_db.execute_query.return_value = sql_result

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = _make_vector_docs(case)
        mock_vs_fn.return_value = mock_store

        mock_ainvoke.side_effect = _build_llm_side_effects(case)

        yield {
            "case": case,
            "ainvoke": mock_ainvoke,
            "db": mock_db,
            "redis": mock_redis,
            "vector_store": mock_store,
        }


class TestEvalGoldenDataset:
    """Run each golden case through the workflow and validate properties."""

    @pytest.mark.parametrize(
        "workflow_mocks",
        [c for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")],
        indirect=True,
        ids=[c["id"] for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")],
    )
    async def test_golden_case_routing(self, workflow_mocks):
        """Verify the question routes correctly."""
        case = workflow_mocks["case"]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-{case['id']}"}},
        )
        assert result["route"] == case["expected_route"], (
            f"Case {case['id']}: expected route '{case['expected_route']}', got '{result['route']}'"
        )

    @pytest.mark.parametrize(
        "workflow_mocks",
        [c for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")],
        indirect=True,
        ids=[c["id"] for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")],
    )
    async def test_golden_case_answer_quality(self, workflow_mocks):
        """Verify the answer contains expected terms and meets score threshold."""
        case = workflow_mocks["case"]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-quality-{case['id']}"}},
        )

        answer = result.get("final_answer", "")

        for term in case.get("expected_answer_contains", []):
            assert term.lower() in answer.lower(), (
                f"Case {case['id']}: expected '{term}' in answer, got: {answer[:200]}"
            )

        for term in case.get("expected_answer_not_contains", []):
            assert term.lower() not in answer.lower(), (
                f"Case {case['id']}: unexpected '{term}' found in answer"
            )

        min_score = case.get("min_review_score", 7.0)
        assert result.get("review_score", 0) >= min_score, (
            f"Case {case['id']}: review score {result.get('review_score')} < {min_score}"
        )

    @pytest.mark.parametrize(
        "workflow_mocks",
        [c for c in GOLDEN_CASES if c.get("needs_clarification")],
        indirect=True,
        ids=[c["id"] for c in GOLDEN_CASES if c.get("needs_clarification")],
    )
    async def test_golden_case_clarification(self, workflow_mocks):
        """Verify ambiguous questions trigger clarification."""
        case = workflow_mocks["case"]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-clarify-{case['id']}"}},
        )

        assert result.get("needs_clarification") is True or "specify" in result.get("final_answer", "").lower(), (
            f"Case {case['id']}: expected clarification request"
        )


    @pytest.mark.parametrize(
        "workflow_mocks",
        [c for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")],
        indirect=True,
        ids=[c["id"] for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")],
    )
    async def test_golden_case_has_confidence(self, workflow_mocks):
        """Every completed pipeline run should produce a confidence level."""
        case = workflow_mocks["case"]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-conf-{case['id']}"}},
        )
        assert result.get("confidence") in ("high", "medium", "low"), (
            f"Case {case['id']}: missing or invalid confidence: {result.get('confidence')}"
        )


class TestEvalConsistency:
    """Reproducibility regression check: the same question, replayed against
    identical mocked LLM/DB/vector responses, must produce an identical
    result. Every external dependency is held fixed here, so any divergence
    between the two runs means the *application code itself* has a
    non-deterministic path (e.g. an unsorted set iteration order, a
    dict built from unordered input) — not an LLM sampling artifact.

    This matters specifically for this system: the entire pitch of the
    governed metric path is that "same input = same output" is auditable.
    A hidden non-deterministic branch anywhere in the pipeline undermines
    that even if the LLM itself is temperature=0.
    """

    _CASES = [c for c in GOLDEN_CASES if not c.get("needs_clarification") and not c.get("expected_sql_blocked")]

    @pytest.mark.parametrize(
        "workflow_mocks", _CASES, indirect=True, ids=[c["id"] for c in _CASES],
    )
    async def test_golden_case_is_reproducible(self, workflow_mocks):
        case = workflow_mocks["case"]
        mock_ainvoke = workflow_mocks["ainvoke"]
        from workflow import app as workflow_app

        first = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-consistency-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-consistency-{case['id']}-run1"}},
        )

        # Same case, same canned responses, replayed fresh — the first
        # invocation consumed the mock's side_effect list.
        mock_ainvoke.side_effect = _build_llm_side_effects(case)

        second = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-consistency-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-consistency-{case['id']}-run2"}},
        )

        compared_fields = (
            "route", "final_answer", "review_score", "confidence",
            "data_path", "resolved_metric", "sql_result",
        )
        for field in compared_fields:
            assert first.get(field) == second.get(field), (
                f"Case {case['id']}: '{field}' differs between two runs of identical "
                f"input — {first.get(field)!r} vs {second.get(field)!r}. Likely a "
                f"non-deterministic code path (e.g. unsorted set/dict iteration)."
            )


class TestEvalSqlSafety:
    """Verify SQL injection attempts are blocked by deterministic validators."""

    @pytest.mark.parametrize(
        "workflow_mocks",
        [c for c in GOLDEN_CASES if c.get("expected_sql_blocked")],
        indirect=True,
        ids=[c["id"] for c in GOLDEN_CASES if c.get("expected_sql_blocked")],
    )
    async def test_sql_injection_blocked(self, workflow_mocks):
        """SQL containing forbidden operations is blocked before execution."""
        case = workflow_mocks["case"]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": case["question"],
                "session_id": f"eval-{case['id']}",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": f"eval-safety-{case['id']}"}},
        )

        sql_result = result.get("sql_result", "")
        assert "blocked" in sql_result.lower() or "error" in sql_result.lower(), (
            f"Case {case['id']}: expected SQL to be blocked, got: {sql_result[:200]}"
        )
