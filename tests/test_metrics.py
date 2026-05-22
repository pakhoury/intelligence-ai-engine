"""
Tests for the metrics catalog — loading, context generation, and injection
into the sql_path node via the SQL agent prompt.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ── MetricsCatalog unit tests ────────────────────────────────────────


class TestMetricsCatalog:
    def test_loads_all_metrics(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        ids = catalog.metric_ids
        assert len(ids) >= 7
        assert "compliance_effectiveness_score" in ids
        assert "violation_severity_distribution" in ids
        assert "audit_finding_resolution_rate" in ids
        assert "regulatory_exposure_index" in ids
        assert "department_risk_score" in ids
        assert "control_coverage_ratio" in ids
        assert "mean_time_to_resolution" in ids

    def test_get_returns_metric(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        m = catalog.get("compliance_effectiveness_score")
        assert m is not None
        assert m["name"] == "Compliance Effectiveness Score"
        assert m["unit"] == "%"
        assert "COMPLIANCE_VIOLATIONS" in m["tables"]

    def test_metric_has_no_sql_template(self):
        """Metrics define steps and formulas, not SQL templates."""
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        for mid in catalog.metric_ids:
            m = catalog.get(mid)
            assert "sql_template" not in m, f"{mid} should not have sql_template"
            assert "steps" in m, f"{mid} must have calculation steps"
            assert "formula" in m, f"{mid} must have a formula"

    def test_get_unknown_returns_none(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        assert catalog.get("nonexistent_metric") is None

    def test_reload(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        count_before = len(catalog.metric_ids)
        catalog.reload()
        assert len(catalog.metric_ids) == count_before


class TestBuildAllContext:
    def test_includes_all_metrics(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Compliance Effectiveness Score" in ctx
        assert "Department Risk Score" in ctx
        assert "Mean Time to Resolution" in ctx
        assert "Control Coverage Ratio" in ctx

    def test_includes_formula_and_steps(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Formula:" in ctx
        assert "Steps:" in ctx
        assert "Step 1:" in ctx

    def test_includes_parameters(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Parameters:" in ctx
        assert "lookback_months" in ctx

    def test_includes_thresholds(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Thresholds:" in ctx
        assert "critical:" in ctx.lower() or "Critical" in ctx

    def test_includes_tables(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "COMPLIANCE_VIOLATIONS" in ctx
        assert "AUDIT_FINDINGS" in ctx
        assert "RISK_EVENTS" in ctx

    def test_empty_catalog_returns_empty(self):
        from metrics import MetricsCatalog
        import tempfile, os
        path = os.path.join(tempfile.gettempdir(), "empty_metrics.yaml")
        with open(path, "w") as f:
            f.write("metrics: {}")
        catalog = MetricsCatalog(path)
        assert catalog.build_all_context() == ""
        os.remove(path)


# ── sql_path uses metric context ─────────────────────────────────────


class TestSqlPathWithMetrics:
    @patch("nodes._get_metrics_catalog")
    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_sql_prompt_includes_metric_definitions(
        self, mock_ainvoke, mock_db, mock_catalog_fn, sample_state
    ):
        """The SQL agent prompt includes metric definitions from the catalog."""
        from metrics import MetricsCatalog
        mock_catalog_fn.return_value = MetricsCatalog()

        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS\nColumns: ..."
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.return_value = (
            "SELECT COUNT(*) AS total, "
            "ROUND(SUM(CASE WHEN STATUS='Closed' THEN 1 ELSE 0 END)*100.0"
            "/NULLIF(COUNT(*),0),2) AS score "
            "FROM COMPLIANCE.COMPLIANCE_VIOLATIONS "
            "FETCH FIRST 500 ROWS ONLY"
        )
        mock_db.execute_query.return_value = "[(100, 85.0)]"

        sample_state["question"] = "What is the compliance effectiveness score?"
        from nodes import sql_path
        await sql_path(sample_state)

        prompt_used = mock_ainvoke.call_args[0][0]
        assert "Compliance Effectiveness Score" in prompt_used
        assert "Formula:" in prompt_used
        assert "Steps:" in prompt_used

    @patch("nodes._get_metrics_catalog")
    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_non_metric_question_still_works(
        self, mock_ainvoke, mock_db, mock_catalog_fn, sample_state
    ):
        """Non-metric questions pass through sql_path normally."""
        from metrics import MetricsCatalog
        mock_catalog_fn.return_value = MetricsCatalog()

        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.return_value = "SELECT SEVERITY, COUNT(*) FROM COMPLIANCE_VIOLATIONS GROUP BY SEVERITY"
        mock_db.execute_query.return_value = "[('Critical', 2), ('High', 3)]"

        sample_state["question"] = "How many violations per severity?"
        from nodes import sql_path
        result = await sql_path(sample_state)

        assert "Critical" in result["sql_result"]


# ── Workflow integration with metric question via SQL route ──────────


class TestWorkflowMetricViaSqlRoute:
    @pytest.fixture
    def mock_deps(self):
        with (
            patch("nodes.redis_client") as mock_redis,
            patch("nodes._ainvoke_llm", new_callable=AsyncMock) as mock_ainvoke,
            patch("nodes.db_connector") as mock_db,
            patch("nodes._get_vector_store") as mock_vs_fn,
            patch("nodes._get_metrics_catalog") as mock_catalog_fn,
        ):
            from metrics import MetricsCatalog
            mock_catalog_fn.return_value = MetricsCatalog()

            mock_redis.get.return_value = None
            mock_redis.set.return_value = True
            mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
            mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
            mock_db.validate_columns.return_value = ""
            mock_db.execute_query.return_value = "[(100, 85, 85.0)]"

            mock_doc = MagicMock()
            mock_doc.metadata = {"title": "Policy"}
            mock_doc.page_content = "KYC requirements"
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = [mock_doc]
            mock_vs_fn.return_value = mock_store

            yield {
                "redis": mock_redis,
                "ainvoke": mock_ainvoke,
                "db": mock_db,
                "vector_store_fn": mock_vs_fn,
                "vector_store": mock_store,
            }

    async def test_metric_question_routes_through_sql(self, mock_deps):
        """A metric question is routed as SQL, and the LLM sees metric definitions."""
        mock_deps["ainvoke"].side_effect = [
            "SQL",                                                          # router
            "NO_CLARIFICATION_NEEDED",                                      # clarify
            "SELECT COUNT(*) AS total, "                                    # sql_path (LLM sees metrics)
            "ROUND(SUM(CASE WHEN STATUS='Closed' THEN 1 ELSE 0 END)*100.0"
            "/NULLIF(COUNT(*),0),2) AS score "
            "FROM COMPLIANCE.COMPLIANCE_VIOLATIONS "
            "FETCH FIRST 500 ROWS ONLY",
            "The compliance effectiveness score is 85%, rated as Good.",     # answer
            "Score: 9.0\nDecision: APPROVED",                               # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "What is the compliance effectiveness score?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-metric-via-sql"}},
        )
        assert result["route"] == "sql_only"
        assert "85%" in result["final_answer"]
        assert result["review_score"] == 9.0

        sql_prompt = mock_deps["ainvoke"].call_args_list[2][0][0]
        assert "Compliance Effectiveness Score" in sql_prompt
        assert "Formula:" in sql_prompt
