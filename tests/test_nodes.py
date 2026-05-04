"""
Unit tests for workflow nodes with mocked LLM and database.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestCacheCheck:
    @patch("nodes.redis_client")
    def test_cache_hit(self, mock_redis, sample_state):
        mock_redis.get.return_value = "Cached answer text"
        from nodes import cache_check
        result = cache_check(sample_state)
        assert result["cache_hit"] is True
        assert result["final_answer"] == "Cached answer text"
        assert result["review_score"] == 10.0

    @patch("nodes.redis_client")
    def test_cache_miss(self, mock_redis, sample_state):
        mock_redis.get.return_value = None
        from nodes import cache_check
        result = cache_check(sample_state)
        assert result["cache_hit"] is False
        assert "final_answer" not in result

    @patch("nodes.redis_client")
    def test_cache_redis_error_returns_miss(self, mock_redis, sample_state):
        mock_redis.get.side_effect = ConnectionError("Redis down")
        from nodes import cache_check
        result = cache_check(sample_state)
        assert result["cache_hit"] is False


class TestRouterNode:
    @patch("nodes.llm")
    def test_routes_to_sql(self, mock_llm, sample_state):
        mock_llm.invoke.return_value = MagicMock(content="SQL")
        from nodes import router_node
        result = router_node(sample_state)
        assert result["route"] == "sql"

    @patch("nodes.llm")
    def test_routes_to_documents(self, mock_llm, sample_state):
        mock_llm.invoke.return_value = MagicMock(content="DOCUMENTS")
        from nodes import router_node
        result = router_node(sample_state)
        assert result["route"] == "documents"

    @patch("nodes.llm")
    def test_routes_to_both(self, mock_llm, sample_state):
        mock_llm.invoke.return_value = MagicMock(content="BOTH")
        from nodes import router_node
        result = router_node(sample_state)
        assert result["route"] == "both"

    @patch("nodes.llm")
    def test_defaults_to_documents(self, mock_llm, sample_state):
        mock_llm.invoke.return_value = MagicMock(content="unknown response")
        from nodes import router_node
        result = router_node(sample_state)
        assert result["route"] == "documents"


class TestClarificationNode:
    @patch("nodes.llm")
    def test_no_clarification_needed(self, mock_llm, sample_state):
        mock_llm.invoke.return_value = MagicMock(content="NO_CLARIFICATION_NEEDED")
        from nodes import clarification_node
        result = clarification_node(sample_state)
        assert result["needs_clarification"] is False

    @patch("nodes.llm")
    def test_clarification_needed(self, mock_llm, sample_state):
        sample_state["question"] = "Show me the data"
        mock_llm.invoke.return_value = MagicMock(
            content="Could you specify which data you'd like to see?"
        )
        from nodes import clarification_node
        result = clarification_node(sample_state)
        assert result["needs_clarification"] is True
        assert "specify" in result["final_answer"].lower()


class TestSqlPath:
    @patch("nodes.db_connector")
    @patch("nodes.llm")
    def test_generates_and_executes_sql(self, mock_llm, mock_db, sample_state):
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS\nColumns: ..."
        mock_db.validate_columns.return_value = ""
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT SEVERITY, COUNT(*) FROM COMPLIANCE_VIOLATIONS GROUP BY SEVERITY"
        )
        mock_db.execute_query.return_value = "[('Critical', 2), ('High', 3)]"

        from nodes import sql_path
        result = sql_path(sample_state)
        assert "sql_result" in result
        assert "Critical" in result["sql_result"]

    @patch("nodes.db_connector")
    @patch("nodes.llm")
    def test_strips_markdown_fences(self, mock_llm, mock_db, sample_state):
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT 1 FROM DUAL\n```"
        )
        mock_db.execute_query.return_value = "[(1,)]"

        from nodes import sql_path
        result = sql_path(sample_state)
        mock_db.execute_query.assert_called_once()
        called_sql = mock_db.execute_query.call_args[0][0]
        assert "```" not in called_sql


class TestVectorRetrieval:
    @patch("nodes._get_vector_store")
    def test_retrieves_documents(self, mock_store_fn, sample_state):
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Compliance Policy 2024"}
        mock_doc.page_content = "KYC requirements state that..."
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc]
        mock_store_fn.return_value = mock_store

        from nodes import vector_retrieval
        result = vector_retrieval(sample_state)
        assert len(result["retrieved_docs"]) == 1
        assert "KYC" in result["retrieved_docs"][0]

    @patch("nodes._get_vector_store")
    def test_handles_pgvector_error(self, mock_store_fn, sample_state):
        mock_store_fn.side_effect = Exception("PGVector connection failed")

        from nodes import vector_retrieval
        result = vector_retrieval(sample_state)
        assert result["retrieved_docs"] == []


class TestAnswerGenerator:
    @patch("nodes.llm")
    def test_generates_answer(self, mock_llm, sample_state):
        sample_state["sql_result"] = "[('Critical', 2)]"
        sample_state["retrieved_docs"] = ["[Policy]: KYC policy details"]
        mock_llm.invoke.return_value = MagicMock(
            content="Based on the data, there are 2 critical violations."
        )
        from nodes import answer_generator
        result = answer_generator(sample_state)
        assert "critical" in result["final_answer"].lower()


class TestReviewerNode:
    @patch("nodes.llm")
    def test_parses_score(self, mock_llm, sample_state):
        sample_state["final_answer"] = "There are 2 critical violations."
        mock_llm.invoke.return_value = MagicMock(
            content="Score: 9.0\nDecision: APPROVED\nReason: Accurate"
        )
        from nodes import reviewer_node
        result = reviewer_node(sample_state)
        assert result["review_score"] == 9.0

    @patch("nodes.llm")
    def test_parses_fraction_score(self, mock_llm, sample_state):
        sample_state["final_answer"] = "Answer text"
        mock_llm.invoke.return_value = MagicMock(content="I give this 8/10")
        from nodes import reviewer_node
        result = reviewer_node(sample_state)
        assert result["review_score"] == 8.0

    @patch("nodes.llm")
    def test_defaults_score_on_parse_failure(self, mock_llm, sample_state):
        sample_state["final_answer"] = "Answer text"
        mock_llm.invoke.return_value = MagicMock(content="This looks good")
        from nodes import reviewer_node
        result = reviewer_node(sample_state)
        assert result["review_score"] == 7.0

    @patch("nodes.llm")
    def test_dlp_redacts_pii_in_answer(self, mock_llm, sample_state):
        sample_state["final_answer"] = "Contact john@bank.com or call 555-123-4567 for violations."
        mock_llm.invoke.return_value = MagicMock(
            content="Score: 9.0\nDecision: APPROVED"
        )
        from nodes import reviewer_node
        result = reviewer_node(sample_state)
        assert "john@bank.com" not in result["final_answer"]
        assert "[EMAIL]" in result["final_answer"]
        assert result["review_score"] <= 6.0

    @patch("nodes.llm")
    def test_dlp_clean_answer_unchanged(self, mock_llm, sample_state):
        sample_state["final_answer"] = "There are 5 critical violations in Q1 2024."
        mock_llm.invoke.return_value = MagicMock(
            content="Score: 9.0\nDecision: APPROVED"
        )
        from nodes import reviewer_node
        result = reviewer_node(sample_state)
        assert "final_answer" not in result  # no override when clean
        assert result["review_score"] == 9.0


class TestCacheWrite:
    @patch("nodes.redis_client")
    def test_caches_good_answer(self, mock_redis, sample_state):
        sample_state["review_score"] = 9.0
        sample_state["final_answer"] = "Good answer"
        from nodes import cache_write
        cache_write(sample_state)
        mock_redis.set.assert_called_once()

    @patch("nodes.redis_client")
    def test_skips_low_score(self, mock_redis, sample_state):
        sample_state["review_score"] = 3.0
        sample_state["final_answer"] = "Bad answer"
        from nodes import cache_write
        cache_write(sample_state)
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
