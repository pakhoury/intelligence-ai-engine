"""
Unit tests for deterministic validators.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

from validators import (
    extract_sql_tables,
    run_all_validators,
    validate_answer_grounding,
    validate_metric_citation,
    validate_number_grounding,
    validate_result_sanity,
    validate_sql_safety,
    validate_sql_scope,
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


ALLOWED_TABLES = {
    "COMPLIANCE_VIOLATIONS", "AUDIT_FINDINGS", "RISK_EVENTS",
    "CONTROL_MAPPINGS", "DUAL",
}


class TestExtractSqlTables:
    def test_simple_from(self):
        assert extract_sql_tables("SELECT * FROM COMPLIANCE_VIOLATIONS") == {"COMPLIANCE_VIOLATIONS"}

    def test_alias_not_treated_as_table(self):
        tables = extract_sql_tables("SELECT cv.STATUS FROM COMPLIANCE_VIOLATIONS cv WHERE cv.STATUS = 'Open'")
        assert tables == {"COMPLIANCE_VIOLATIONS"}

    def test_joins_extracted(self):
        sql = (
            "SELECT * FROM COMPLIANCE_VIOLATIONS v "
            "LEFT JOIN AUDIT_FINDINGS a ON v.DEPARTMENT = a.DEPARTMENT"
        )
        assert extract_sql_tables(sql) == {"COMPLIANCE_VIOLATIONS", "AUDIT_FINDINGS"}

    def test_comma_join_extracted(self):
        sql = "SELECT * FROM COMPLIANCE_VIOLATIONS v, AUDIT_FINDINGS a WHERE v.DEPARTMENT = a.DEPARTMENT"
        assert extract_sql_tables(sql) == {"COMPLIANCE_VIOLATIONS", "AUDIT_FINDINGS"}

    def test_derived_table_inner_tables_extracted(self):
        sql = "SELECT * FROM (SELECT DEPARTMENT FROM RISK_EVENTS) r"
        assert extract_sql_tables(sql) == {"RISK_EVENTS"}

    def test_string_literals_ignored(self):
        sql = "SELECT * FROM COMPLIANCE_VIOLATIONS WHERE NOTE = 'copied FROM SECRET_TABLE'"
        assert extract_sql_tables(sql) == {"COMPLIANCE_VIOLATIONS"}


class TestSqlScope:
    def test_in_scope_passes(self):
        result = validate_sql_scope(
            "SELECT COUNT(*) FROM COMPLIANCE_VIOLATIONS", ALLOWED_TABLES,
        )
        assert result.passed

    def test_out_of_scope_table_fails(self):
        result = validate_sql_scope("SELECT * FROM EMPLOYEE_SALARIES", ALLOWED_TABLES)
        assert not result.passed
        assert result.score_cap == 0.0
        assert result.unfixable_cap == 0.0
        assert "EMPLOYEE_SALARIES" in result.failures[0]

    def test_out_of_scope_join_fails(self):
        sql = (
            "SELECT * FROM COMPLIANCE_VIOLATIONS v "
            "JOIN HR_RECORDS h ON v.DEPARTMENT = h.DEPARTMENT"
        )
        result = validate_sql_scope(sql, ALLOWED_TABLES)
        assert not result.passed
        assert "HR_RECORDS" in result.failures[0]

    def test_out_of_scope_subquery_fails(self):
        sql = "SELECT * FROM (SELECT * FROM PAYROLL) p"
        result = validate_sql_scope(sql, ALLOWED_TABLES)
        assert not result.passed
        assert "PAYROLL" in result.failures[0]

    def test_no_allowlist_skips_check(self):
        result = validate_sql_scope("SELECT * FROM ANYTHING", None)
        assert result.passed
        result = validate_sql_scope("SELECT * FROM ANYTHING", set())
        assert result.passed

    def test_no_sql_possible_skips_check(self):
        result = validate_sql_scope("NO_SQL_POSSIBLE", ALLOWED_TABLES)
        assert result.passed

    def test_case_insensitive(self):
        result = validate_sql_scope(
            "select * from compliance_violations", ALLOWED_TABLES,
        )
        assert result.passed

    def test_run_all_validators_blocks_out_of_scope(self):
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="SELECT * FROM HR_RECORDS",
            sql_result="Query:\nSELECT * FROM HR_RECORDS\n\nResult:\n[(1,)]",
            answer="There is 1 record.",
            retrieved_docs=[],
            route="sql_only",
            allowed_tables=ALLOWED_TABLES,
        )
        assert not passed
        assert cap == 0.0
        assert unfixable_cap == 0.0
        assert any("out-of-scope" in f for f in failures)

    def test_run_all_validators_skips_without_allowlist(self):
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="SELECT * FROM HR_RECORDS",
            sql_result="Query:\nSELECT * FROM HR_RECORDS\n\nResult:\n[(1,)]",
            answer="There is 1 record.",
            retrieved_docs=[],
            route="sql_only",
        )
        assert not any("out-of-scope" in f for f in failures)


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
        # >= 10^16 flags (cartesian-join scale); legitimate financial
        # aggregates in the billions/trillions must NOT flag.
        result = validate_result_sanity("[(99999999999999999,)]")
        assert not result.passed
        assert any("large number" in f.lower() for f in result.failures)

    def test_billions_are_legitimate(self):
        result = validate_result_sanity("[('total_penalties', 99999999999)]")
        assert result.passed

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

    def test_docs_only_with_docs(self):
        result = validate_answer_grounding(
            answer="The policy requires enhanced due diligence.",
            sql_result="",
            retrieved_docs=["[KYC Policy]: Enhanced due diligence required"],
            route="docs_only",
        )
        assert result.passed


class TestNumberGrounding:
    def test_all_numbers_grounded(self):
        result = validate_number_grounding(
            answer="There are 42 violations with a score of 85.5.",
            sql_result="[('count', 42), ('score', 85.5)]",
        )
        assert result.passed

    def test_many_ungrounded_numbers(self):
        result = validate_number_grounding(
            answer="Revenue was 5000, cost was 3200, profit was 1800, margin was 36%.",
            sql_result="[('total', 999)]",
        )
        assert not result.passed
        assert result.score_cap == 5.0

    def test_single_ungrounded_number_fails(self):
        """Zero tolerance: one hallucinated figure is one too many."""
        result = validate_number_grounding(
            answer="The score improved by 15 points to reach 85.",
            sql_result="[('score', 85)]",
        )
        assert not result.passed
        assert any("15" in f for f in result.failures)

    def test_trivial_numbers_ignored(self):
        result = validate_number_grounding(
            answer="The score is 0 out of 100 based on 1 violation.",
            sql_result="[('result', 55)]",
        )
        assert result.passed

    def test_no_data_at_all_fails(self):
        """A numeric claim with no SQL result and no docs is ungrounded by definition."""
        result = validate_number_grounding(
            answer="The score is 42.",
            sql_result="N/A - no SQL data retrieved",
        )
        assert not result.passed

    def test_error_result_does_not_ground(self):
        """An error string is not data; numbers 'quoted' from it are hallucinated."""
        result = validate_number_grounding(
            answer="The score is 42.",
            sql_result="Error: something went wrong",
        )
        assert not result.passed

    def test_no_numbers_in_answer(self):
        result = validate_number_grounding(
            answer="The compliance program is effective.",
            sql_result="[('score', 95)]",
        )
        assert result.passed

    def test_docs_ground_numbers(self):
        """docs_only answers are no longer exempt: numbers must appear in the chunks."""
        result = validate_number_grounding(
            answer="Access violations must be remediated within 48 hours.",
            sql_result="",
            retrieved_docs=["[Access Policy]: remediation required within 48 hours"],
        )
        assert result.passed

    def test_docs_do_not_ground_missing_numbers(self):
        result = validate_number_grounding(
            answer="The transaction limit is $25,000.",
            sql_result="",
            retrieved_docs=["[AML Policy]: transactions above the limit require review"],
        )
        assert not result.passed
        assert any("25000" in f for f in result.failures)

    def test_years_exempt(self):
        result = validate_number_grounding(
            answer="In Q1 2024 there were 12 violations.",
            sql_result="[(12,)]",
        )
        assert result.passed

    def test_derived_total_grounded(self):
        result = validate_number_grounding(
            answer="Critical (5), High (12), Medium (23), Low (8), totaling 48 violations.",
            sql_result="[('Critical', 5), ('High', 12), ('Medium', 23), ('Low', 8)]",
        )
        assert result.passed

    def test_derived_percentage_grounded(self):
        result = validate_number_grounding(
            answer="The resolution rate is 76% — 38 of 50 findings resolved.",
            sql_result="[(50, 38)]",
        )
        assert result.passed

    def test_scaled_units_grounded(self):
        result = validate_number_grounding(
            answer="Operational losses reached $500K against credit losses of $2M.",
            sql_result="[('Operational', 500000), ('Credit', 2000000)]",
        )
        assert result.passed

    def test_ratio_does_not_ground_large_counts(self):
        """Pairwise ratios only ground percent-like values, not arbitrary counts."""
        result = validate_number_grounding(
            answer="There were 150 violations.",
            sql_result="[(3, 2)]",  # 3/2*100 = 150, but 150 is a count, not a rate
        )
        assert not result.passed


class TestMetricCitation:
    def test_compiled_metric_cited(self):
        result = validate_metric_citation(
            answer="The compliance effectiveness score is 85%.",
            resolved_metric="compliance_effectiveness_score",
            compiled_metric=True,
        )
        assert result.passed

    def test_compiled_metric_not_cited(self):
        result = validate_metric_citation(
            answer="The result is 85%.",
            resolved_metric="compliance_effectiveness_score",
            compiled_metric=True,
        )
        assert not result.passed
        assert result.score_cap == 6.0

    def test_non_compiled_skipped(self):
        result = validate_metric_citation(
            answer="The result is 85%.",
            resolved_metric="compliance_effectiveness_score",
            compiled_metric=False,
        )
        assert result.passed

    def test_no_metric_skipped(self):
        result = validate_metric_citation(
            answer="The result is 85%.",
            resolved_metric="",
            compiled_metric=True,
        )
        assert result.passed

    def test_partial_match_passes(self):
        result = validate_metric_citation(
            answer="Based on the compliance effectiveness analysis, the score is 85%.",
            resolved_metric="compliance_effectiveness_score",
            compiled_metric=True,
        )
        assert result.passed


class TestRunAllValidators:
    def test_all_pass(self):
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="SELECT COUNT(*) FROM VIOLATIONS",
            sql_result="[(42,)]",
            answer="There are 42 violations.",
            retrieved_docs=[],
            route="sql_only",
        )
        assert passed
        assert failures == []
        assert cap == 10.0
        assert unfixable_cap == 10.0

    def test_multiple_failures(self):
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="DROP TABLE VIOLATIONS",
            sql_result="Error: blocked",
            answer="",
            retrieved_docs=[],
            route="sql_only",
        )
        assert not passed
        assert len(failures) >= 2
        assert cap == 0.0
        assert unfixable_cap == 0.0

    def test_new_validators_integrated(self):
        """New validators (number grounding + metric citation) run in the aggregate."""
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="SELECT 1 FROM T",
            sql_result="[('x', 42)]",
            answer="The result is 999.",
            retrieved_docs=[],
            route="sql_only",
            resolved_metric="test_metric",
            compiled_metric=True,
        )
        assert not passed
        assert any("test_metric" in f for f in failures)

    def test_sql_execution_error_is_unfixable(self):
        """A SQL execution error caps the score in a way reflection cannot lift."""
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="",
            sql_result="Execution Error: ORA-00942: table or view does not exist",
            answer="I could not retrieve the data.",
            retrieved_docs=["[Policy]: some doc"],
            route="docs_then_sql",
        )
        assert not passed
        assert unfixable_cap == 3.0

    def test_answer_only_failures_stay_fixable(self):
        """Failures in the answer text (metric citation) don't mark the run unfixable."""
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="SELECT 1 FROM T",
            sql_result="Query:\nSELECT 1 FROM T\n\nResult:\n[(42,)]",
            answer="The result is 42.",
            retrieved_docs=[],
            route="sql_only",
            resolved_metric="some_other_metric",
            compiled_metric=True,
        )
        assert not passed
        assert cap == 6.0
        assert unfixable_cap == 10.0

    def test_empty_result_set_is_unfixable(self):
        passed, failures, cap, unfixable_cap = run_all_validators(
            sql="SELECT 1 FROM T",
            sql_result="Query:\nSELECT 1 FROM T\n\nResult:\n[]",
            answer="No records were found.",
            retrieved_docs=[],
            route="sql_only",
        )
        assert not passed
        assert unfixable_cap == 6.0
