"""
Golden test suite — verifies the full resolver → compiler → SQL chain.

Each test case defines a question, the expected resolved metric, and
assertions on the deterministic SQL output. No database or LLM needed.
"""
import pytest

from metrics.compiler import MetricCompiler
from metrics.loader import load_metrics_catalog
from metrics.resolver import MetricResolver


@pytest.fixture(scope="module")
def registry():
    return load_metrics_catalog()


@pytest.fixture(scope="module")
def resolver(registry):
    return MetricResolver(registry)


@pytest.fixture(scope="module")
def compiler():
    return MetricCompiler()


# ── Golden test data ────────────────────────────────────────────────

GOLDEN_CASES = [
    {
        "id": "ces_basic",
        "question": "What is the compliance effectiveness score?",
        "expected_metric": "compliance_effectiveness_score",
        "sql_contains": ["COUNT(*)", "COMPLIANCE_VIOLATIONS", "STATUS = 'Closed'", "ADD_MONTHS(SYSDATE, -12)"],
    },
    {
        "id": "ces_with_lookback",
        "question": "What is the compliance effectiveness score for the last 6 months?",
        "expected_metric": "compliance_effectiveness_score",
        "sql_contains": ["ADD_MONTHS(SYSDATE, -6)", "COMPLIANCE_VIOLATIONS"],
    },
    {
        "id": "ces_with_department",
        "question": "What is the compliance effectiveness score for the Legal department?",
        "expected_metric": "compliance_effectiveness_score",
        "sql_contains": ["COMPLIANCE_VIOLATIONS", "'Legal'"],
    },
    {
        "id": "severity_distribution",
        "question": "Show me the violation severity breakdown",
        "expected_metric": "violation_severity_distribution",
        "sql_contains": ["SEVERITY", "GROUP BY", "COUNT(*)"],
    },
    {
        "id": "audit_resolution",
        "question": "What is the audit finding resolution rate?",
        "expected_metric": "audit_finding_resolution_rate",
        "sql_contains": ["AUDIT_FINDINGS", "COUNT(*)"],
    },
    {
        "id": "regulatory_exposure",
        "question": "What is our regulatory exposure?",
        "expected_metric": "regulatory_exposure_index",
        "sql_contains": ["REGULATORY_BODY", "FINANCIAL_IMPACT", "STATUS"],
    },
    {
        "id": "department_risk_template",
        "question": "What is the department risk score?",
        "expected_metric": "department_risk_score",
        "sql_contains": [
            "COMPLIANCE_VIOLATIONS", "AUDIT_FINDINGS", "RISK_EVENTS",
            "composite_risk_score",
        ],
    },
    {
        "id": "control_coverage",
        "question": "What is the control coverage ratio?",
        "expected_metric": "control_coverage_ratio",
        "sql_contains": ["CONTROL_MAPPINGS", "COMPLIANCE_REQUIREMENT"],
    },
    {
        "id": "mttr_basic",
        "question": "What is the mean time to resolution?",
        "expected_metric": "mean_time_to_resolution",
        "sql_contains": ["CLOSURE_DATE", "VIOLATION_DATE", "AVG"],
    },
    {
        "id": "mttr_critical",
        "question": "What is the mean time to resolution for critical violations?",
        "expected_metric": "mean_time_to_resolution",
        "sql_contains": ["'Critical'"],
    },
]


# ── Resolver tests ──────────────────────────────────────────────────

class TestMetricResolution:
    @pytest.mark.parametrize(
        "case",
        GOLDEN_CASES,
        ids=[c["id"] for c in GOLDEN_CASES],
    )
    def test_resolves_correct_metric(self, resolver, case):
        resolved = resolver.resolve(case["question"])
        assert resolved is not None, (
            f"Resolver returned None for: {case['question']}"
        )
        assert resolved.metric.metric_id == case["expected_metric"]

    def test_unrelated_question_returns_none(self, resolver):
        resolved = resolver.resolve("What is the weather today?")
        assert resolved is None


# ── Compiler tests ──────────────────────────────────────────────────

class TestMetricCompilation:
    @pytest.mark.parametrize(
        "case",
        GOLDEN_CASES,
        ids=[c["id"] for c in GOLDEN_CASES],
    )
    def test_compiles_to_valid_sql(self, resolver, compiler, case):
        resolved = resolver.resolve(case["question"])
        assert resolved is not None

        compiled = compiler.compile(resolved.metric, resolved.extracted_params)
        assert compiled is not None, (
            f"Compiler returned None for metric: {resolved.metric.metric_id}"
        )
        assert compiled.metric_id == case["expected_metric"]

        sql_upper = compiled.sql.upper()
        assert sql_upper.strip().startswith("SELECT"), (
            f"Compiled SQL doesn't start with SELECT: {compiled.sql[:80]}"
        )

    @pytest.mark.parametrize(
        "case",
        GOLDEN_CASES,
        ids=[c["id"] for c in GOLDEN_CASES],
    )
    def test_sql_contains_expected(self, resolver, compiler, case):
        resolved = resolver.resolve(case["question"])
        compiled = compiler.compile(resolved.metric, resolved.extracted_params)
        assert compiled is not None

        sql_upper = compiled.sql.upper()
        for fragment in case.get("sql_contains", []):
            assert fragment.upper() in sql_upper, (
                f"Expected '{fragment}' in SQL:\n{compiled.sql}"
            )

    @pytest.mark.parametrize(
        "case",
        [c for c in GOLDEN_CASES if "sql_not_contains" in c],
        ids=[c["id"] for c in GOLDEN_CASES if "sql_not_contains" in c],
    )
    def test_sql_excludes_unexpected(self, resolver, compiler, case):
        resolved = resolver.resolve(case["question"])
        compiled = compiler.compile(resolved.metric, resolved.extracted_params)
        assert compiled is not None

        sql_upper = compiled.sql.upper()
        for fragment in case["sql_not_contains"]:
            assert fragment.upper() not in sql_upper, (
                f"Unexpected '{fragment}' found in SQL:\n{compiled.sql}"
            )


# ── Determinism test ────────────────────────────────────────────────

class TestCompilerDeterminism:
    def test_same_input_same_output(self, resolver, compiler):
        """Compiling the same metric twice must produce identical SQL."""
        resolved = resolver.resolve("What is the compliance effectiveness score?")
        compiled1 = compiler.compile(resolved.metric, resolved.extracted_params)
        compiled2 = compiler.compile(resolved.metric, resolved.extracted_params)
        assert compiled1.sql == compiled2.sql


# ── Safety tests ────────────────────────────────────────────────────

class TestCompilerSafety:
    def test_no_dangerous_operations(self, registry, compiler):
        """No compiled SQL should contain dangerous operations."""
        import re
        dangerous = re.compile(
            r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT)\b",
            re.IGNORECASE,
        )
        for metric in registry.all_metrics:
            compiled = compiler.compile(metric)
            if compiled is None:
                continue
            assert not dangerous.search(compiled.sql), (
                f"Dangerous SQL in metric {metric.metric_id}: {compiled.sql[:100]}"
            )

    def test_all_compilable_metrics_have_row_limit(self, registry, compiler):
        """Every compiled query should include a row limit."""
        for metric in registry.all_metrics:
            compiled = compiler.compile(metric)
            if compiled is None:
                continue
            assert "FETCH FIRST" in compiled.sql.upper(), (
                f"No row limit in metric {metric.metric_id}: {compiled.sql[:100]}"
            )


# ── Param extraction tests ──────────────────────────────────────────

class TestParamExtraction:
    def test_extracts_lookback_months(self, resolver):
        resolved = resolver.resolve("compliance effectiveness score for the last 6 months")
        assert resolved is not None
        assert resolved.extracted_params.get("lookback_months") == "6"

    def test_extracts_department(self, resolver):
        resolved = resolver.resolve(
            "What is the compliance effectiveness score for the Legal department?"
        )
        assert resolved is not None
        assert "department" in resolved.extracted_params
        assert "Legal" in resolved.extracted_params["department"]

    def test_extracts_severity(self, resolver):
        resolved = resolver.resolve("mean time to resolution for critical violations")
        assert resolved is not None
        assert resolved.extracted_params.get("severity") == "Critical"

    def test_defaults_applied_when_no_params(self, resolver, compiler):
        resolved = resolver.resolve("What is the compliance effectiveness score?")
        compiled = compiler.compile(resolved.metric, resolved.extracted_params)
        assert compiled is not None
        assert "lookback_months" not in resolved.extracted_params
        assert "12" in compiled.parameters_used.get("lookback_months", "")
