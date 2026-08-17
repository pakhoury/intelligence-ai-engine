"""
Unit tests for the observability module's per-node latency budget
mechanism — a soft SLO check, distinct from REQUEST_TIMEOUT (which is a
hard, request-level kill switch). Exceeding a budget must only ever be
observability (a Prometheus counter + a log line), never a failure.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

import observability
from observability import NODE_BUDGET_EXCEEDED, NODE_LATENCY_BUDGETS_MS, NodeTimer, _check_budget


def _counter_value(node: str) -> float:
    return NODE_BUDGET_EXCEEDED.labels(node=node)._value.get()


class TestCheckBudget:
    def test_within_budget_does_not_flag(self):
        before = _counter_value("router")
        assert _check_budget("router", NODE_LATENCY_BUDGETS_MS["router"] - 1) is False
        assert _counter_value("router") == before

    def test_exactly_at_budget_does_not_flag(self):
        """Boundary case: equal to budget is not 'exceeded'."""
        before = _counter_value("router")
        assert _check_budget("router", NODE_LATENCY_BUDGETS_MS["router"]) is False
        assert _counter_value("router") == before

    def test_over_budget_flags_and_increments_counter(self):
        before = _counter_value("router")
        assert _check_budget("router", NODE_LATENCY_BUDGETS_MS["router"] + 1) is True
        assert _counter_value("router") == before + 1

    def test_unconfigured_node_never_flags(self):
        """A node with no budget entry must never be reported as breaching
        one — silence here means 'not tracked', not 'always compliant'."""
        assert _check_budget("some_node_with_no_budget_entry", 10_000_000) is False

    def test_all_workflow_nodes_have_a_budget(self):
        """Every node wrapped in NodeTimer (see nodes.py) should have an
        explicit SLO. A node quietly missing from this map means its
        latency is invisible to budget tracking even though NodeTimer
        still measures it — this guards against that drifting silently."""
        expected_nodes = {
            "cache_check", "context_resolver", "router", "clarify",
            "metric_resolver", "sql_path", "vector_retrieval", "extract",
            "answer_generator", "reviewer", "cache_write",
        }
        assert expected_nodes <= NODE_LATENCY_BUDGETS_MS.keys()


class TestNodeTimerBudgetIntegration:
    def test_counter_increments_when_node_exceeds_budget(self):
        observability.NODE_LATENCY_BUDGETS_MS["_test_over_budget"] = 0
        before = _counter_value("_test_over_budget")
        try:
            with NodeTimer("_test_over_budget"):
                pass  # any measurable duration exceeds a 0ms budget
        finally:
            del observability.NODE_LATENCY_BUDGETS_MS["_test_over_budget"]
        assert _counter_value("_test_over_budget") == before + 1

    def test_no_exception_raised_when_over_budget(self):
        """Exceeding a budget must never fail the request — only observability."""
        observability.NODE_LATENCY_BUDGETS_MS["_test_no_raise"] = 0
        try:
            with NodeTimer("_test_no_raise"):
                pass
        finally:
            del observability.NODE_LATENCY_BUDGETS_MS["_test_no_raise"]
        # Reaching this line without an exception is the assertion.

    def test_node_exception_still_checked_against_budget(self):
        """A node that both fails AND overruns its budget should still be
        flagged — the exception path shouldn't bypass the budget check."""
        observability.NODE_LATENCY_BUDGETS_MS["_test_exc_and_budget"] = 0
        before = _counter_value("_test_exc_and_budget")
        try:
            with NodeTimer("_test_exc_and_budget"):
                raise ValueError("boom")
        except ValueError:
            pass
        finally:
            del observability.NODE_LATENCY_BUDGETS_MS["_test_exc_and_budget"]
        assert _counter_value("_test_exc_and_budget") == before + 1

    def test_node_under_budget_not_flagged(self):
        observability.NODE_LATENCY_BUDGETS_MS["_test_under_budget"] = 60_000
        before = _counter_value("_test_under_budget")
        try:
            with NodeTimer("_test_under_budget"):
                pass
        finally:
            del observability.NODE_LATENCY_BUDGETS_MS["_test_under_budget"]
        assert _counter_value("_test_under_budget") == before
