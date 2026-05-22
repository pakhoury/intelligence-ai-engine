"""
Unit tests for workflow nodes with mocked LLM and database.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


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


class TestSqlPath:
    @patch("nodes._get_metrics_catalog")
    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_generates_and_executes_sql(self, mock_ainvoke, mock_db, mock_catalog_fn, sample_state):
        from metrics import MetricsCatalog
        mock_catalog_fn.return_value = MetricsCatalog()
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS\nColumns: ..."
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.return_value = "SELECT SEVERITY, COUNT(*) FROM COMPLIANCE_VIOLATIONS GROUP BY SEVERITY"
        mock_db.execute_query.return_value = "[('Critical', 2), ('High', 3)]"

        from nodes import sql_path
        result = await sql_path(sample_state)
        assert "sql_result" in result
        assert "Critical" in result["sql_result"]

    @patch("nodes._get_metrics_catalog")
    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_strips_markdown_fences(self, mock_ainvoke, mock_db, mock_catalog_fn, sample_state):
        from metrics import MetricsCatalog
        mock_catalog_fn.return_value = MetricsCatalog()
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.return_value = "```sql\nSELECT 1 FROM DUAL\n```"
        mock_db.execute_query.return_value = "[(1,)]"

        from nodes import sql_path
        result = await sql_path(sample_state)
        mock_db.execute_query.assert_called_once()
        called_sql = mock_db.execute_query.call_args[0][0]
        assert "```" not in called_sql


    @patch("nodes._get_metrics_catalog")
    @patch("nodes.db_connector")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    async def test_uses_extracted_context(self, mock_ainvoke, mock_db, mock_catalog_fn, sample_state):
        from metrics import MetricsCatalog
        mock_catalog_fn.return_value = MetricsCatalog()
        sample_state["extracted_context"] = "SUM(fine_amount + remediation_cost)"
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_ainvoke.return_value = "SELECT SUM(fine_amount + remediation_cost) FROM COMPLIANCE_VIOLATIONS"
        mock_db.execute_query.return_value = "[(2400000,)]"

        from nodes import sql_path
        result = await sql_path(sample_state)
        prompt_used = mock_ainvoke.call_args[0][0]
        assert "prior research step" in prompt_used
        assert "SUM(fine_amount + remediation_cost)" in prompt_used


class TestVectorRetrieval:
    @patch("nodes._get_vector_store")
    async def test_retrieves_documents(self, mock_store_fn, sample_state):
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

    @patch("nodes._get_vector_store")
    async def test_uses_extracted_context_for_search(self, mock_store_fn, sample_state):
        sample_state["extracted_context"] = "ACCESS_CONTROL"
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Access Control Policy"}
        mock_doc.page_content = "Access control requires MFA..."
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc]
        mock_store_fn.return_value = mock_store

        from nodes import vector_retrieval
        result = await vector_retrieval(sample_state)
        mock_store.similarity_search.assert_called_once_with("ACCESS_CONTROL", k=4)

    @patch("nodes._get_vector_store")
    async def test_handles_pgvector_error(self, mock_store_fn, sample_state):
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
        sample_state["final_answer"] = "Answer text"
        mock_ainvoke.return_value = "This looks good"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert result["review_score"] == 0.0

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
        mock_ainvoke.return_value = "Score: 9.0\nDecision: APPROVED"
        from nodes import reviewer_node
        result = await reviewer_node(sample_state)
        assert "final_answer" not in result
        assert result["review_score"] == 9.0


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
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 10
        assert "Question 5" in result  # First of last 5
        assert "Question 0" not in result  # Should be trimmed
