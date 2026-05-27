"""
Unit tests for deterministic validators.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rag-system"))

from validators import (
    validate_sql_safety,
    validate_result_sanity,
    validate_answer_grounding,
    run_all_validators,
)


class TestSqlSafety:
    def test_valid_select(self):
        result = validate_sql_safety("SELECT COUNT(*) FROM COMPLIANCE_VIOLATIONS")
        assert result.passed
        assert result.score_cap == 10.0

    def test_blocks_drop(self):
        result = validate_sql_safety("DROP TABLE COMPLIANCE_VIOLATIONS")
        assert not result.passed
        assert result.score_cap == 0.0
        assert "DROP" in result.failures[0]

    def test_blocks_delete(self):
        result = validate_sql_safety("DELETE FROM COMPLIANCE_VIOLATIONS WHERE 1=1")
        assert not result.passed
        assert result.score_cap == 0.0

    def test_blocks_truncate(self):
        result = validate_sql_safety("TRUNCATE TABLE COMPLIANCE_VIOLATIONS")
        assert not result.passed
        assert result.score_cap == 0.0

    def test_blocks_insert(self):
        result = validate_sql_safety("INSERT INTO COMPLIANCE_VIOLATIONS VALUES (1, 'test')")
        assert not result.passed
        assert result.score_cap == 0.0

    def test_blocks_update(self):
        result = validate_sql_safety("UPDATE COMPLIANCE_VIOLATIONS SET STATUS='Closed'")
        assert not result.passed
        assert result.score_cap == 0.0

    def test_blocks_multiple_statements(self):
        result = validate_sql_safety("SELECT 1; DROP TABLE users")
        assert not result.passed
        assert any("Multiple SQL" in f or "DROP" in f for f in result.failures)

    def test_detects_unbalanced_parens(self):
        result = validate_sql_safety("SELECT COUNT(* FROM VIOLATIONS")
        assert not result.passed
        assert any("parentheses" in f.lower() for f in result.failures)
        assert result.score_cap <= 4.0

    def test_no_select_statement(self):
        result = validate_sql_safety("GRANT ALL PRIVILEGES TO user")
        assert not result.passed
        assert result.score_cap <= 2.0

    def test_no_sql_possible_passes(self):
        result = validate_sql_safety("NO_SQL_POSSIBLE")
        assert result.passed

    def test_empty_sql_passes(self):
        result = validate_sql_safety("")
        assert result.passed

    def test_complex_valid_query(self):
        sql = (
            "SELECT SEVERITY, COUNT(*) AS cnt, "
            "ROUND(SUM(CASE WHEN STATUS='Closed' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0), 2) AS pct "
            "FROM COMPLIANCE.COMPLIANCE_VIOLATIONS "
            "WHERE CREATED_DATE >= DATE '2024-01-01' "
            "GROUP BY SEVERITY ORDER BY cnt DESC FETCH FIRST 10 ROWS ONLY"
        )
        result = validate_sql_safety(sql)
        assert result.passed


class TestResultSanity:
    def test_normal_result(self):
        result = validate_result_sanity("[('Critical', 5), ('High', 12)]")
        assert result.passed

    def test_error_result(self):
        result = validate_result_sanity("Error: ORA-00942 table or view does not exist")
        assert not result.passed
        assert result.score_cap <= 3.0

    def test_execution_error(self):
        result = validate_result_sanity("Execution Error: timeout after 30s")
        assert not result.passed

    def test_empty_result_set(self):
        result = validate_result_sanity("Query:\nSELECT...\n\nResult:\n[]")
        assert not result.passed
        assert result.score_cap <= 6.0

    def test_suspiciously_large_number(self):
        result = validate_result_sanity("[(99999999999,)]")
        assert not result.passed
        assert any("large number" in f.lower() for f in result.failures)

    def test_na_passes(self):
        result = validate_result_sanity("N/A - no SQL data retrieved")
        assert result.passed

    def test_none_passes(self):
        result = validate_result_sanity("")
        assert result.passed


class TestAnswerGrounding:
    def test_grounded_answer(self):
        result = validate_answer_grounding(
            answer="There are 5 critical violations.",
            sql_result="Query:\nSELECT...\n\nResult:\n[(5,)]",
            retrieved_docs=["[Policy]: Compliance overview"],
            route="sql_only",
        )
        assert result.passed

    def test_empty_answer(self):
        result = validate_answer_grounding(
            answer="",
            sql_result="[(5,)]",
            retrieved_docs=[],
            route="sql_only",
        )
        assert not result.passed
        assert result.score_cap == 0.0

    def test_no_data_for_sql_route(self):
        result = validate_answer_grounding(
            answer="There are 5 critical violations.",
            sql_result="Error: table not found",
            retrieved_docs=[],
            route="sql_only",
        )
        assert not result.passed
        assert result.score_cap <= 3.0

    def test_numeric_claims_without_sql(self):
        result = validate_answer_grounding(
            answer="The score is 85% based on 100 violations.",
            sql_result="",
            retrieved_docs=["[Policy]: Some policy text"],
            route="parallel",
        )
        assert not result.passed
        assert any("numeric claims" in f.lower() for f in result.failures)

    def test_docs_only_with_docs(self):
        result = validate_answer_grounding(
            answer="The policy requires enhanced due diligence.",
            sql_result="",
            retrieved_docs=["[KYC Policy]: Enhanced due diligence required"],
            route="docs_only",
        )
        assert result.passed


class TestRunAllValidators:
    def test_all_pass(self):
        passed, failures, cap = run_all_validators(
            sql="SELECT COUNT(*) FROM VIOLATIONS",
            sql_result="[(42,)]",
            answer="There are 42 violations.",
            retrieved_docs=[],
            route="sql_only",
        )
        assert passed
        assert failures == []
        assert cap == 10.0

    def test_multiple_failures(self):
        passed, failures, cap = run_all_validators(
            sql="DROP TABLE VIOLATIONS",
            sql_result="Error: blocked",
            answer="",
            retrieved_docs=[],
            route="sql_only",
        )
        assert not passed
        assert len(failures) >= 2
        assert cap == 0.0
