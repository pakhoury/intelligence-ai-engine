"""
Unit tests for workflow nodes with mocked LLM and database.
"""
from unittest.mock import AsyncMock, MagicMock, patch


class TestCacheCheck:
    @patch("nodes.redis_client")
    async def test_cache_hit(self, mock_redis, sample_state):
        mock_redis.get.return_value = "Cached answer text"
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is True
        assert result["final_answer"] == "Cached answer text"
        assert result["review_score"] == 10.0

    @patch("nodes.redis_client")
    async def test_cache_miss(self, mock_redis, sample_state):
        mock_redis.get.return_value = None
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is False
        assert "final_answer" not in result

    @patch("nodes.redis_client")
    async def test_cache_redis_error_returns_miss(self, mock_redis, sample_state):
        mock_redis.get.side_effect = ConnectionError("Redis down")
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is False


class TestRouterNode:
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_routes_to_sql_only(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "SQL_ONLY"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "sql_only"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_routes_to_docs_only(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "DOCS_ONLY"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "docs_only"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_routes_to_docs_then_sql(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "DOCS_THEN_SQL"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "docs_then_sql"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_routes_to_sql_then_docs(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "SQL_THEN_DOCS"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "sql_then_docs"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_routes_to_parallel(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "PARALLEL"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "parallel"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_fuzzy_both_maps_to_parallel(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "BOTH"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "parallel"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_fuzzy_sql_maps_to_sql_only(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "SQL"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "sql_only"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_fuzzy_documents_maps_to_docs_only(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "DOCUMENTS"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "docs_only"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_unknown_defaults_to_parallel(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "unknown response"
        from nodes import router_node
        result = await router_node(sample_state)
        assert result["route"] == "parallel"


class TestClarificationNode:
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_no_clarification_needed(self, mock_ainvoke, sample_state):
        mock_ainvoke.return_value = "NO_CLARIFICATION_NEEDED"
        from nodes import clarification_node
        result = await clarification_node(sample_state)
        assert result["needs_clarification"] is False

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_clarification_needed(self, mock_ainvoke, sample_state):
        sample_state["question"] = "Show me the data"
        mock_ainvoke.return_value = "Could you specify which data you'd like to see?"
        from nodes import clarification_node
        result = await clarification_node(sample_state)
        assert result["needs_clarification"] is True
        assert "specify" in result["final_answer"].lower()


class TestExtractNode:
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_extracts_from_docs_for_docs_then_sql(self, mock_ainvoke, sample_state):
        sample_state["route"] = "docs_then_sql"
        sample_state["retrieved_docs"] = ["[Policy]: Compliance cost = fines + remediation + audit fees"]
        mock_ainvoke.return_value = "SUM(fine_amount + remediation_cost + audit_fees)"
        from nodes import extract_node
        result = await extract_node(sample_state)
        assert "extracted_context" in result
        assert "SUM" in result["extracted_context"]

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_extracts_from_sql_for_sql_then_docs(self, mock_ainvoke, sample_state):
        sample_state["route"] = "sql_then_docs"
        sample_state["sql_result"] = "Query:\nSELECT...\n\nResult:\n[('ACCESS_CONTROL', 47)]"
        mock_ainvoke.return_value = "ACCESS_CONTROL"
        from nodes import extract_node
        result = await extract_node(sample_state)
        assert result["extracted_context"] == "ACCESS_CONTROL"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_extraction_failed_returns_empty(self, mock_ainvoke, sample_state):
        sample_state["route"] = "docs_then_sql"
        sample_state["retrieved_docs"] = ["[Policy]: General compliance overview"]
        mock_ainvoke.return_value = "EXTRACTION_FAILED"
        from nodes import extract_node
        result = await extract_node(sample_state)
        assert result["extracted_context"] == ""

    async def test_skips_when_no_source_material(self, sample_state):
        sample_state["route"] = "docs_then_sql"
        sample_state["retrieved_docs"] = []
        from nodes import extract_node
        result = await extract_node(sample_state)
        assert result["extracted_context"] == ""

    async def test_returns_empty_for_unknown_route(self, sample_state):
        sample_state["route"] = "sql_only"
        from nodes import extract_node
        result = await extract_node(sample_state)
        assert result == {}


class TestMetricResolverNode:
    @patch("nodes._get_metric_resolver")
    async def test_resolved_metric_builds_context(self, mock_resolver_fn, sample_state):
        """When a metric is resolved, metric_context contains the full definition."""
        from metrics.registry import MetricDefinition, MetricParameter, MetricSQL, SQLSelectExpr
        from metrics.resolver import ResolvedMetric

        metric = MetricDefinition(
            metric_id="compliance_effectiveness_score",
            name="Compliance Effectiveness Score",
            version="1.0",
            description="test",
            category="compliance",
            unit="%",
            formula="(closed / total) * 100",
            tables=("COMPLIANCE_VIOLATIONS",),
            sql=MetricSQL(base_table="COMPLIANCE_VIOLATIONS",
                          select=(SQLSelectExpr("COUNT(*)", "total"),),
                          filters=()),
            parameters=(
                MetricParameter(name="lookback_months", description="", type="integer", default="12"),
                MetricParameter(name="department", description="", type="string", optional=True),
            ),
            steps=(
                {"step": 1, "action": "Count total violations"},
                {"step": 2, "action": "Count closed violations"},
            ),
        )
        resolved = ResolvedMetric(
            metric=metric,
            confidence=0.95,
            matched_keyword="compliance effectiveness",
            extracted_params={"lookback_months": "6", "department": "Legal"},
        )
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=resolved)
        mock_resolver_fn.return_value = mock_resolver

        from nodes import metric_resolver_node
        result = await metric_resolver_node(sample_state)

        assert result["resolved_metric"] == "compliance_effectiveness_score"
        assert result["metric_version"] == "1.0"
        assert "compiled_sql" not in result
        assert "(closed / total) * 100" in result["metric_context"]
        assert "lookback_months = 6 (user)" in result["metric_context"]
        assert "department = Legal (user)" in result["metric_context"]
        assert "Count total violations" in result["metric_context"]

    @patch("nodes._get_metric_resolver")
    async def test_no_match_returns_none(self, mock_resolver_fn, sample_state):
        """When no metric matches, resolved_metric is None and no metric_context."""
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=None)
        mock_resolver_fn.return_value = mock_resolver

        from nodes import metric_resolver_node
        result = await metric_resolver_node(sample_state)

        assert result["resolved_metric"] is None
        assert "metric_context" not in result
        assert "compiled_sql" not in result


class TestBuildMetricContext:
    def test_includes_formula_and_tables(self):
        from metrics.registry import MetricDefinition, MetricParameter
        from nodes import _build_metric_context

        metric = MetricDefinition(
            metric_id="test_metric",
            name="Test Metric",
            version="1.0",
            description="test",
            category="test",
            unit="%",
            formula="(a / b) * 100",
            tables=("TABLE_A", "TABLE_B"),
            parameters=(
                MetricParameter(name="lookback_months", description="", type="integer", default="12"),
            ),
            steps=(
                {"step": 1, "action": "Count a", "uses": "TABLE_A"},
                {"step": 2, "action": "Count b", "note": "Handle zero"},
            ),
            thresholds={"good": {"label": "Good"}},
            interpretation="Higher is better.",
        )
        ctx = _build_metric_context(metric, {"lookback_months": "6"})

        assert "Test Metric (test_metric@1.0)" in ctx
        assert "(a / b) * 100" in ctx
        assert "TABLE_A, TABLE_B" in ctx
        assert "lookback_months = 6 (user)" in ctx
        assert "Count a" in ctx
        assert "Uses: TABLE_A" in ctx
        assert "Note: Handle zero" in ctx
        assert "good: Good" in ctx
        assert "Higher is better." in ctx

    def test_default_params_labeled_correctly(self):
        from metrics.registry import MetricDefinition, MetricParameter
        from nodes import _build_metric_context

        metric = MetricDefinition(
            metric_id="m", name="M", version="1.0", description="d",
            category="c", unit="u", formula="f",
            parameters=(
                MetricParameter(name="lookback_months", description="", type="integer", default="12"),
                MetricParameter(name="department", description="", type="string", optional=True),
            ),
        )
        ctx = _build_metric_context(metric, {})
        assert "lookback_months = 12 (default)" in ctx
        assert "department" not in ctx

    def test_extra_params_included(self):
        """Params not in the metric definition (e.g. business_key) still appear."""
        from metrics.registry import MetricDefinition, MetricParameter
        from nodes import _build_metric_context

        metric = MetricDefinition(
            metric_id="m", name="M", version="1.0", description="d",
            category="c", unit="u", formula="f",
            parameters=(
                MetricParameter(name="lookback_months", description="", type="integer", default="12"),
            ),
        )
        ctx = _build_metric_context(metric, {"lookback_months": "6", "business_key": "88"})
        assert "lookback_months = 6 (user)" in ctx
        assert "business_key = 88 (user)" in ctx

    def test_minimal_metric_no_optional_sections(self):
        """Metric with no steps, thresholds, or interpretation produces clean output."""
        from metrics.registry import MetricDefinition
        from nodes import _build_metric_context

        metric = MetricDefinition(
            metric_id="m", name="M", version="1.0", description="d",
            category="c", unit="u", formula="f", tables=("T",),
        )
        ctx = _build_metric_context(metric, {})
        assert "M (m@1.0)" in ctx
        assert "Calculation Steps" not in ctx
        assert "Parameters" not in ctx
        assert "Thresholds" not in ctx
        assert "Interpretation" not in ctx


class TestSqlPath:
    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_generates_and_executes_sql(self, mock_ainvoke, mock_db, sample_state):
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS\nColumns: ..."
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.side_effect = [
            "COMPLIANCE_VIOLATIONS",                                            # table selection
            "SELECT SEVERITY, COUNT(*) FROM COMPLIANCE_VIOLATIONS GROUP BY SEVERITY",  # sql_agent
        ]
        mock_db.execute_query.return_value = "[('Critical', 2), ('High', 3)]"

        from nodes import sql_path
        result = await sql_path(sample_state)
        assert "sql_result" in result
        assert "Critical" in result["sql_result"]

    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_strips_markdown_fences(self, mock_ainvoke, mock_db, sample_state):
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.side_effect = [
            "COMPLIANCE_VIOLATIONS",               # table selection
            "```sql\nSELECT 1 FROM DUAL\n```",    # sql_agent
        ]
        mock_db.execute_query.return_value = "[(1,)]"

        from nodes import sql_path
        await sql_path(sample_state)
        mock_db.execute_query.assert_called_once()
        called_sql = mock_db.execute_query.call_args[0][0]
        assert "```" not in called_sql

    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_uses_extracted_context(self, mock_ainvoke, mock_db, sample_state):
        sample_state["extracted_context"] = "SUM(fine_amount + remediation_cost)"
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.side_effect = [
            "COMPLIANCE_VIOLATIONS",                                                   # table selection
            "SELECT SUM(fine_amount + remediation_cost) FROM COMPLIANCE_VIOLATIONS",   # sql_agent
        ]
        mock_db.execute_query.return_value = "[(2400000,)]"

        from nodes import sql_path
        await sql_path(sample_state)
        sql_prompt = mock_ainvoke.call_args_list[1][0][0]
        assert "prior research step" in sql_prompt
        assert "SUM(fine_amount + remediation_cost)" in sql_prompt

    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_uses_resolved_metric_context(self, mock_ainvoke, mock_db, sample_state):
        """When metric_context is in state, the prompt uses it instead of the full catalog."""
        sample_state["metric_context"] = (
            "Metric: Test (test@1.0)\n"
            "Formula: (closed / total) * 100"
        )
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.side_effect = [
            "COMPLIANCE_VIOLATIONS",
            "SELECT COUNT(*) FROM COMPLIANCE_VIOLATIONS",
        ]
        mock_db.execute_query.return_value = "[(100,)]"

        from nodes import sql_path
        await sql_path(sample_state)
        sql_prompt = mock_ainvoke.call_args_list[1][0][0]
        assert "Resolved Metric Definition (follow this formula" in sql_prompt
        assert "(closed / total) * 100" in sql_prompt
        assert "Known Metric Definitions:\n" not in sql_prompt

    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_falls_back_to_clean_message_without_metric(self, mock_ainvoke, mock_db, sample_state):
        """Without metric_context, the prompt uses a clean fallback instead of dumping the catalog."""
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.side_effect = [
            "COMPLIANCE_VIOLATIONS",
            "SELECT COUNT(*) FROM COMPLIANCE_VIOLATIONS",
        ]
        mock_db.execute_query.return_value = "[(100,)]"

        from nodes import sql_path
        await sql_path(sample_state)
        sql_prompt = mock_ainvoke.call_args_list[1][0][0]
        assert "No specific metric was identified" in sql_prompt
        assert "Resolved Metric Definition (follow this formula" not in sql_prompt


class TestVectorRetrieval:
    @patch("nodes.lexical_search", return_value=[])
    @patch("nodes._get_vector_store")
    async def test_retrieves_documents(self, mock_store_fn, mock_lexical, sample_state):
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Compliance Policy 2024"}
        mock_doc.page_content = "KYC requirements state that..."
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc]
        mock_store_fn.return_value = mock_store

        from nodes import vector_retrieval
        result = await vector_retrieval(sample_state)
        assert len(result["retrieved_docs"]) == 1
        assert "KYC" in result["retrieved_docs"][0]

    @patch("nodes.lexical_search", return_value=[])
    @patch("nodes._get_vector_store")
    async def test_uses_extracted_context_for_search(self, mock_store_fn, mock_lexical, sample_state):
        sample_state["extracted_context"] = "ACCESS_CONTROL"
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Access Control Policy"}
        mock_doc.page_content = "Access control requires MFA..."
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc]
        mock_store_fn.return_value = mock_store

        from nodes import RETRIEVAL_CANDIDATES, vector_retrieval
        await vector_retrieval(sample_state)
        mock_store.similarity_search.assert_called_once_with("ACCESS_CONTROL", k=RETRIEVAL_CANDIDATES)
        # Lexical side also searches with the extracted context
        assert mock_lexical.call_args[0][1] == "ACCESS_CONTROL"

    @patch("nodes.lexical_search", return_value=[])
    @patch("nodes._get_vector_store")
    async def test_handles_pgvector_error(self, mock_store_fn, mock_lexical, sample_state):
        mock_store_fn.side_effect = Exception("PGVector connection failed")

        from nodes import vector_retrieval
        result = await vector_retrieval(sample_state)
        assert result["retrieved_docs"] == []


class TestAnswerGenerator:
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_generates_answer(self, mock_ainvoke, sample_state):
        sample_state["sql_result"] = "[('Critical', 2)]"
        sample_state["retrieved_docs"] = ["[Policy]: KYC policy details"]
        mock_ainvoke.return_value = "Based on the data, there are 2 critical violations."
        from nodes import answer_generator
        result = await answer_generator(sample_state)
        assert "critical" in result["final_answer"].lower()

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_includes_reflection_feedback(self, mock_ainvoke, sample_state):
        sample_state["sql_result"] = "[('Critical', 2)]"
        sample_state["retrieved_docs"] = ["[Policy]: KYC policy details"]
        sample_state["reflection_attempt"] = 1
        sample_state["reviewer_feedback"] = "Score: 4.0\nDecision: REJECT\nReason: Answer lacks specificity"
        mock_ainvoke.return_value = "Improved detailed answer."
        from nodes import answer_generator
        result = await answer_generator(sample_state)
        assert result["final_answer"] == "Improved detailed answer."
        prompt_used = mock_ainvoke.call_args[0][0]
        assert "reviewer's concerns" in prompt_used
        assert "lacks specificity" in prompt_used


class TestReviewerNode:
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_parses_score(self, mock_ainvoke, sample_state):
        sample_state["final_answer"] = "There are 2 critical violations."
        sample_state["sql_result"] = "Query:\nSELECT...\n\nResult:\n[('Critical', 2)]"
        sample_state["route"] = "sql_only"
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED\nReason: Accurate"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["review_score"] == 9.0
        assert result["reflection_attempt"] == 1
        assert "Accurate" in result["reviewer_feedback"]

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_parses_fraction_score(self, mock_ainvoke, sample_state):
        sample_state["final_answer"] = "Answer text"
        mock_ainvoke.return_value = "I give this 8/10"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["review_score"] == 8.0

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_defaults_score_on_parse_failure(self, mock_ainvoke, sample_state):
        """Unparseable reviewer output must not trigger the reflection loop (score 0)."""
        sample_state["final_answer"] = "Answer text"
        mock_ainvoke.return_value = "This looks good"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["review_score"] == 7.0

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_dlp_redacts_pii_in_answer(self, mock_ainvoke, sample_state):
        sample_state["final_answer"] = "Contact john@bank.com or call 555-123-4567 for violations."
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert "john@bank.com" not in result["final_answer"]
        assert "[EMAIL]" in result["final_answer"]
        assert result["review_score"] <= 6.0

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_dlp_clean_answer_unchanged(self, mock_ainvoke, sample_state):
        sample_state["final_answer"] = "There are 5 critical violations in Q1 2024."
        sample_state["sql_result"] = "Query:\nSELECT...\n\nResult:\n[('Critical', 5)]"
        sample_state["route"] = "sql_only"
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert "final_answer" not in result
        assert result["review_score"] == 9.0

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_skips_reflection_on_sql_execution_error(self, mock_ainvoke, sample_state):
        """A SQL execution error can't be fixed by regenerating the answer."""
        sample_state["final_answer"] = "I was unable to retrieve the requested data."
        sample_state["sql_result"] = "Execution Error: ORA-00942: table or view does not exist"
        sample_state["route"] = "docs_then_sql"
        sample_state["retrieved_docs"] = ["[Policy]: KYC thresholds apply."]
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["skip_reflection"] is True
        assert result["review_score"] <= 3.0

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_no_skip_on_clean_answer(self, mock_ainvoke, sample_state):
        sample_state["final_answer"] = "There are 2 critical violations."
        sample_state["sql_result"] = "Query:\nSELECT...\n\nResult:\n[('Critical', 2)]"
        sample_state["route"] = "sql_only"
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["skip_reflection"] is False

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_no_skip_on_answer_level_failure(self, mock_ainvoke, sample_state):
        """Failures in the answer text (missing metric citation) stay in the loop —
        a regenerated answer can fix them."""
        sample_state["final_answer"] = "The value is 42."
        sample_state["sql_result"] = "Query:\nSELECT...\n\nResult:\n[(42,)]"
        sample_state["route"] = "sql_only"
        sample_state["resolved_metric"] = "compliance_effectiveness_score"
        sample_state["compiled_metric"] = True
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["skip_reflection"] is False
        assert result["review_score"] <= 6.0


class TestCacheWrite:
    @patch("nodes.redis_client")
    async def test_caches_good_answer(self, mock_redis, sample_state):
        sample_state["review_score"] = 9.0
        sample_state["final_answer"] = "Good answer"
        from nodes import cache_write
        await cache_write(sample_state)
        mock_redis.set.assert_called_once()

    @patch("nodes.redis_client")
    async def test_skips_low_score(self, mock_redis, sample_state):
        sample_state["review_score"] = 3.0
        sample_state["final_answer"] = "Bad answer"
        from nodes import cache_write
        await cache_write(sample_state)
        mock_redis.set.assert_not_called()


class TestExploratoryLabeling:
    """Answers whose numbers came from LLM-generated SQL (not a compiled
    metric) must carry an explicit exploratory-path disclosure."""

    @patch("nodes.redis_client")
    async def test_llm_sql_answer_gets_labeled(self, mock_redis, sample_state):
        import json
        sample_state["review_score"] = 9.0
        sample_state["final_answer"] = "There are 48 open violations."
        sample_state["sql_result"] = "Query:\nSELECT COUNT(*)...\n\nResult:\n[(48,)]"
        sample_state["compiled_metric"] = False
        from nodes import EXPLORATORY_NOTICE, cache_write
        result = await cache_write(sample_state)
        assert result["data_path"] == "exploratory"
        assert result["final_answer"].endswith(EXPLORATORY_NOTICE)
        # The labeled answer is what gets cached, so cache hits carry it too.
        cached = json.loads(mock_redis.set.call_args[0][1])
        assert cached["answer"].endswith(EXPLORATORY_NOTICE)
        assert cached["data_path"] == "exploratory"

    @patch("nodes.redis_client")
    async def test_compiled_metric_answer_not_labeled(self, mock_redis, sample_state):
        sample_state["review_score"] = 9.0
        sample_state["final_answer"] = "The score is 55%."
        sample_state["sql_result"] = "Query:\nSELECT...\n\nResult:\n[(55.0,)]"
        sample_state["compiled_metric"] = True
        from nodes import EXPLORATORY_NOTICE, cache_write
        result = await cache_write(sample_state)
        assert result["data_path"] == "governed"
        assert "final_answer" not in result  # answer untouched
        assert EXPLORATORY_NOTICE not in mock_redis.set.call_args[0][1]

    @patch("nodes.redis_client")
    async def test_docs_only_answer_not_labeled(self, mock_redis, sample_state):
        sample_state["review_score"] = 9.0
        sample_state["final_answer"] = "The AML policy requires KYC checks."
        sample_state["retrieved_docs"] = ["[Policy]: KYC checks are required."]
        from nodes import cache_write
        result = await cache_write(sample_state)
        assert result["data_path"] == "documents"
        assert "final_answer" not in result

    @patch("nodes.redis_client")
    async def test_failed_sql_not_labeled_exploratory(self, mock_redis, sample_state):
        sample_state["review_score"] = 9.0
        sample_state["final_answer"] = "I could not retrieve the data."
        sample_state["sql_result"] = "Execution Error: ORA-00942"
        from nodes import cache_write
        result = await cache_write(sample_state)
        assert result["data_path"] == "none"
        assert "final_answer" not in result

    @patch("nodes.redis_client")
    async def test_followup_answer_still_labeled(self, mock_redis, sample_state_with_history):
        """Follow-ups bypass the cache but must still carry the disclosure."""
        sample_state_with_history["review_score"] = 9.0
        sample_state_with_history["final_answer"] = "There are 2 critical violations."
        sample_state_with_history["sql_result"] = "Query:\nSELECT...\n\nResult:\n[(2,)]"
        from nodes import EXPLORATORY_NOTICE, cache_write
        result = await cache_write(sample_state_with_history)
        assert result["data_path"] == "exploratory"
        assert result["final_answer"].endswith(EXPLORATORY_NOTICE)
        mock_redis.set.assert_not_called()

    @patch("nodes.redis_client")
    async def test_cache_hit_restores_data_path(self, mock_redis, sample_state):
        import json
        mock_redis.get.return_value = json.dumps({
            "answer": "There are 48 open violations.",
            "review_score": 9.0,
            "data_path": "exploratory",
        })
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is True
        assert result["data_path"] == "exploratory"


class TestCacheKeyNormalization:
    def test_same_question_different_case(self):
        from nodes import _cache_key
        key1 = _cache_key("How many violations?")
        key2 = _cache_key("how many violations?")
        assert key1 == key2

    def test_same_question_extra_whitespace(self):
        from nodes import _cache_key
        key1 = _cache_key("How many violations?")
        key2 = _cache_key("  How  many   violations?  ")
        assert key1 == key2

    def test_different_questions_different_keys(self):
        from nodes import _cache_key
        key1 = _cache_key("How many violations?")
        key2 = _cache_key("What is AML policy?")
        assert key1 != key2

    def test_check_and_write_keys_align(self):
        """Regression: cache_write must produce the same key cache_check reads."""
        from nodes import _cache_key
        # cache_check keys on the incoming question; cache_write keys on
        # original_question — for first-turn requests these are identical.
        assert _cache_key("How many violations?") == _cache_key("How many violations?")


class TestCacheContextIsolation:
    @patch("nodes.redis_client")
    async def test_cache_check_bypasses_followups(self, mock_redis, sample_state_with_history):
        """Follow-up questions must never be served from cache."""
        mock_redis.get.return_value = '{"answer": "Cached", "review_score": 9.0}'
        from nodes import cache_check
        result = await cache_check(sample_state_with_history)
        assert result["cache_hit"] is False
        mock_redis.get.assert_not_called()

    @patch("nodes.redis_client")
    async def test_cache_write_skips_followups(self, mock_redis, sample_state_with_history):
        sample_state_with_history["review_score"] = 9.0
        sample_state_with_history["final_answer"] = "Context-dependent answer"
        from nodes import cache_write
        await cache_write(sample_state_with_history)
        mock_redis.set.assert_not_called()

    @patch("nodes.redis_client")
    async def test_write_then_check_roundtrip(self, mock_redis, sample_state):
        """Answer cached for a metric question is found on the next identical request."""
        import json
        sample_state["review_score"] = 8.5
        sample_state["final_answer"] = "The score is 55%."
        sample_state["resolved_metric"] = "compliance_effectiveness_score"
        sample_state["metric_version"] = "1.0"
        sample_state["confidence"] = "high"

        from nodes import cache_check, cache_write
        await cache_write(sample_state)
        write_key, payload = mock_redis.set.call_args[0][0], mock_redis.set.call_args[0][1]

        mock_redis.get.return_value = payload
        result = await cache_check({"question": sample_state["question"], "conversation_history": []})
        read_key = mock_redis.get.call_args[0][0]

        assert write_key == read_key
        assert result["cache_hit"] is True
        assert result["final_answer"] == "The score is 55%."
        assert result["review_score"] == 8.5
        assert result["confidence"] == "high"
        assert result["resolved_metric"] == "compliance_effectiveness_score"
        assert json.loads(payload)["metric_version"] == "1.0"


class TestCacheMetricVersionCheck:
    """Cached answers built from a registered metric must be invalidated when
    the registry no longer carries the same version of that metric."""

    def _payload(self, version="1.0"):
        import json
        return json.dumps({
            "answer": "The score is 55%.",
            "review_score": 9.0,
            "resolved_metric": "compliance_effectiveness_score",
            "metric_version": version,
        })

    @patch("nodes._get_metric_registry")
    @patch("nodes.redis_client")
    async def test_version_mismatch_invalidates(self, mock_redis, mock_registry_fn, sample_state):
        mock_redis.get.return_value = self._payload(version="1.0")
        current = MagicMock()
        current.version = "2.0"
        mock_registry_fn.return_value.get.return_value = current
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is False
        mock_redis.delete.assert_called_once()

    @patch("nodes._get_metric_registry")
    @patch("nodes.redis_client")
    async def test_removed_metric_invalidates(self, mock_redis, mock_registry_fn, sample_state):
        mock_redis.get.return_value = self._payload()
        mock_registry_fn.return_value.get.return_value = None
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is False
        mock_redis.delete.assert_called_once()

    @patch("nodes._get_metric_registry")
    @patch("nodes.redis_client")
    async def test_matching_version_serves_hit(self, mock_redis, mock_registry_fn, sample_state):
        mock_redis.get.return_value = self._payload(version="1.0")
        current = MagicMock()
        current.version = "1.0"
        mock_registry_fn.return_value.get.return_value = current
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is True
        assert result["final_answer"] == "The score is 55%."
        mock_redis.delete.assert_not_called()

    @patch("nodes._get_metric_registry")
    @patch("nodes.redis_client")
    async def test_registry_error_invalidates(self, mock_redis, mock_registry_fn, sample_state):
        """If the registry can't be read, recompute rather than risk staleness."""
        mock_redis.get.return_value = self._payload()
        mock_registry_fn.side_effect = RuntimeError("catalog unreadable")
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is False

    @patch("nodes._get_metric_registry")
    @patch("nodes.redis_client")
    async def test_non_metric_answer_skips_registry(self, mock_redis, mock_registry_fn, sample_state):
        """Answers without a resolved metric are served without consulting the registry."""
        mock_redis.get.return_value = '{"answer": "Plain cached answer", "review_score": 9.0}'
        from nodes import cache_check
        result = await cache_check(sample_state)
        assert result["cache_hit"] is True
        mock_registry_fn.assert_not_called()


class TestContextResolver:
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_passes_through_without_history(self, mock_ainvoke, sample_state):
        sample_state["conversation_history"] = []
        from nodes import context_resolver
        result = await context_resolver(sample_state)
        assert result["original_question"] == sample_state["question"]
        assert "question" not in result
        mock_ainvoke.assert_not_called()

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_rewrites_followup_question(self, mock_ainvoke, sample_state_with_rich_history):
        mock_ainvoke.return_value = (
            "Break down the compliance effectiveness score by department"
        )
        from nodes import context_resolver
        result = await context_resolver(sample_state_with_rich_history)
        assert result["original_question"] == "Break that down by department"
        assert result["question"] == "Break down the compliance effectiveness score by department"

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_preserves_standalone_question(self, mock_ainvoke, sample_state_with_rich_history):
        sample_state_with_rich_history["question"] = "How many audit findings are open?"
        mock_ainvoke.return_value = "How many audit findings are open?"
        from nodes import context_resolver
        result = await context_resolver(sample_state_with_rich_history)
        assert result["original_question"] == "How many audit findings are open?"
        assert "interpreted_as" not in result or result.get("question") == result["original_question"]

    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_falls_back_on_llm_failure(self, mock_ainvoke, sample_state_with_rich_history):
        mock_ainvoke.side_effect = Exception("LLM timeout")
        from nodes import context_resolver
        result = await context_resolver(sample_state_with_rich_history)
        assert result["original_question"] == "Break that down by department"
        assert "question" not in result


class TestFormatHistory:
    def test_empty_history(self, sample_state):
        from nodes import _format_history
        result = _format_history(sample_state)
        assert result == ""

    def test_formats_history(self, sample_state_with_history):
        from nodes import _format_history
        result = _format_history(sample_state_with_history)
        assert "User:" in result
        assert "Assistant:" in result
        assert "violations" in result.lower()

    def test_formats_rich_history_with_metric_context(self, sample_state_with_rich_history):
        from nodes import _format_history
        result = _format_history(sample_state_with_rich_history)
        assert "User:" in result
        assert "Assistant:" in result
        assert "Metric: compliance_effectiveness_score@1.0" in result
        assert "Route: sql_only" in result
        assert "Data:" in result

    def test_handles_mixed_history_formats(self):
        """Old-format entries (no metadata) coexist with enriched entries."""
        from nodes import _format_history
        state = {
            "conversation_history": [
                {"question": "Old question", "answer": "Old answer"},
                {
                    "question": "New question",
                    "answer": "New answer",
                    "resolved_metric": "test_metric",
                    "route": "sql_only",
                },
            ]
        }
        result = _format_history(state)
        lines = result.split("\n")
        assert any("Old question" in line for line in lines)
        assert any("Metric: test_metric" in line for line in lines)
        assert sum(1 for line in lines if line.startswith("[")) == 1

    def test_limits_to_5_turns(self):
        from nodes import _format_history
        state = {
            "conversation_history": [
                {"question": f"Question {i}", "answer": f"Answer {i}"}
                for i in range(10)
            ]
        }
        result = _format_history(state)
        # Should only have last 5 turns (10 lines: 5 User + 5 Assistant)
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 10
        assert "Question 5" in result  # First of last 5
        assert "Question 0" not in result  # Should be trimmed
