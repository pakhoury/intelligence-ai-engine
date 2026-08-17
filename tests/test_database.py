"""
Unit tests for the database connector base class — specifically the
keyword-based table-selection fallback used when the LLM table-selection
call fails (see nodes.sql_path's except branch).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

from database.oracle import OracleConnector


def _connector_with_catalog(table_names: list[str]) -> OracleConnector:
    """Build a connector with a synthetic catalog, without hitting a real DB."""
    connector = OracleConnector.__new__(OracleConnector)
    connector.catalog = {
        "schemas": {
            "TEST_SCHEMA": {
                "tables": {
                    name: {"description": f"{name} description"}
                    for name in table_names
                }
            }
        }
    }
    return connector


class TestKeywordFallbackDeterminism:
    """set() iteration order depends on Python's per-process string hash
    seed (PYTHONHASHSEED is randomized by default) — with more than 8
    matching tables, an unsorted return could silently select a different
    arbitrary subset on every process restart for the identical question.
    That's unacceptable in a system whose value proposition is
    reproducible, auditable answers.
    """

    def test_repeated_calls_return_identical_result(self):
        tables = [f"TABLE_{i}_VIOLATIONS" for i in range(15)]
        connector = _connector_with_catalog(tables)

        first = connector._keyword_fallback("violations")
        second = connector._keyword_fallback("violations")

        assert first == second

    def test_result_is_sorted(self):
        """Sorted output is itself the determinism guarantee — order can't
        depend on hash seed if it's always alphabetical."""
        tables = [f"TABLE_{i}_VIOLATIONS" for i in range(15)]
        connector = _connector_with_catalog(tables)

        result = connector._keyword_fallback("violations")

        assert result == sorted(result)

    def test_caps_at_eight_tables(self):
        tables = [f"TABLE_{i}_VIOLATIONS" for i in range(15)]
        connector = _connector_with_catalog(tables)

        result = connector._keyword_fallback("violations")

        assert len(result) == 8

    def test_matches_by_keyword_synonym(self):
        connector = _connector_with_catalog(["COMPLIANCE_VIOLATIONS", "RISK_EVENTS", "CONTROL_MAPPINGS"])

        result = connector._keyword_fallback("What is our exposure to penalties?")

        # EXPOSURE -> RISK_EVENTS, COMPLIANCE_VIOLATIONS; PENALTY -> COMPLIANCE_VIOLATIONS
        assert "RISK_EVENTS" in result
        assert "COMPLIANCE_VIOLATIONS" in result

    def test_no_match_returns_empty(self):
        connector = _connector_with_catalog(["COMPLIANCE_VIOLATIONS"])

        result = connector._keyword_fallback("xyzzy plugh nonsense")

        assert result == []
