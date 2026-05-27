"""
Golden SQL determinism tests — verifies that the metric pipeline
(resolver + compiler) produces the exact expected SQL for each
registered metric.

If any test here fails, it means a metric definition or compiler change
has broken SQL determinism. This is the regression gate.

Run with: pytest tests/eval/test_sql_golden.py -v
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "rag-system"))

from metrics.loader import load_metrics_catalog
from metrics.resolver import MetricResolver
from metrics.compiler import MetricCompiler

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "sql_golden_set.json")

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    GOLDEN_DATA = json.load(f)

GOLDEN_CASES = GOLDEN_DATA["cases"]


@pytest.fixture(scope="module")
def pipeline():
    registry = load_metrics_catalog()
    return {
        "registry": registry,
        "resolver": MetricResolver(registry),
        "compiler": MetricCompiler(),
    }


class TestSqlGoldenSet:
    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
    def test_resolves_to_expected_metric(self, case, pipeline):
        resolved = pipeline["resolver"].resolve(case["question"])
        assert resolved is not None, (
            f"Failed to resolve question: {case['question']}"
        )
        assert resolved.metric.metric_id == case["expected_metric"], (
            f"Expected metric '{case['expected_metric']}', "
            f"got '{resolved.metric.metric_id}'"
        )

    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
    def test_compiled_sql_contains_expected(self, case, pipeline):
        metric = pipeline["registry"].get(case["expected_metric"])
        compiled = pipeline["compiler"].compile(metric, case.get("params", {}))
        assert compiled is not None, f"Failed to compile: {case['expected_metric']}"

        for fragment in case.get("expected_sql_contains", []):
            assert fragment in compiled.sql, (
                f"Case {case['id']}: expected '{fragment}' in SQL.\n"
                f"Got:\n{compiled.sql}"
            )

        for fragment in case.get("expected_sql_not_contains", []):
            assert fragment not in compiled.sql, (
                f"Case {case['id']}: unexpected '{fragment}' in SQL.\n"
                f"Got:\n{compiled.sql}"
            )

    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
    def test_deterministic_across_runs(self, case, pipeline):
        """Compile the same metric 10 times and verify identical output."""
        metric = pipeline["registry"].get(case["expected_metric"])
        params = case.get("params", {})

        sqls = [
            pipeline["compiler"].compile(metric, params).sql
            for _ in range(10)
        ]
        assert len(set(sqls)) == 1, (
            f"Case {case['id']}: non-deterministic SQL across runs"
        )
