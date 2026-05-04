"""
Integration tests for the LangGraph workflow — verifies conditional routing
with all external dependencies mocked.
"""
import pytest
from unittest.mock import MagicMock, patch


def _mock_llm_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.fixture
def mock_deps():
    """Patch all external dependencies so the compiled graph can run."""
    with (
        patch("nodes.redis_client") as mock_redis,
        patch("nodes.llm") as mock_llm,
        patch("nodes.db_connector") as mock_db,
        patch("nodes._get_vector_store") as mock_vs_fn,
    ):
        # Defaults
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_db.execute_query.return_value = "[('Critical', 2)]"

        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Policy"}
        mock_doc.page_content = "KYC requirements"
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc]
        mock_vs_fn.return_value = mock_store

        yield {
            "redis": mock_redis,
            "llm": mock_llm,
            "db": mock_db,
            "vector_store_fn": mock_vs_fn,
            "vector_store": mock_store,
        }


class TestWorkflowCacheHitPath:
    def test_cache_hit_ends_immediately(self, mock_deps):
        mock_deps["redis"].get.return_value = "Cached answer"
        from workflow import app as workflow_app

        result = workflow_app.invoke(
            {
                "question": "How many violations?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-cache-hit"}},
        )
        assert result["cache_hit"] is True
        assert result["final_answer"] == "Cached answer"
        # LLM should never be called on cache hit
        mock_deps["llm"].invoke.assert_not_called()


class TestWorkflowSQLRoute:
    def test_sql_only_route(self, mock_deps):
        llm = mock_deps["llm"]
        llm.invoke.side_effect = [
            _mock_llm_response("SQL"),                                # router
            _mock_llm_response("NO_CLARIFICATION_NEEDED"),            # clarify
            _mock_llm_response("SELECT COUNT(*) FROM VIOLATIONS"),    # sql_agent
            _mock_llm_response("There are 2 critical violations."),   # answer
            _mock_llm_response("Score: 9.0\nDecision: APPROVED"),     # reviewer
        ]
        from workflow import app as workflow_app

        result = workflow_app.invoke(
            {
                "question": "How many violations?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-sql-route"}},
        )
        assert result["route"] == "sql"
        assert result["final_answer"] == "There are 2 critical violations."
        assert result["review_score"] == 9.0
        # Vector store should NOT be called for sql-only route
        mock_deps["vector_store"].similarity_search.assert_not_called()


class TestWorkflowDocumentsRoute:
    def test_documents_only_route(self, mock_deps):
        llm = mock_deps["llm"]
        llm.invoke.side_effect = [
            _mock_llm_response("DOCUMENTS"),                          # router
            _mock_llm_response("NO_CLARIFICATION_NEEDED"),            # clarify
            _mock_llm_response("The KYC policy requires..."),         # answer
            _mock_llm_response("Score: 8.0\nDecision: APPROVED"),     # reviewer
        ]
        from workflow import app as workflow_app

        result = workflow_app.invoke(
            {
                "question": "What is KYC policy?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-docs-route"}},
        )
        assert result["route"] == "documents"
        assert "KYC" in result["final_answer"]
        # SQL connector should NOT be called for documents-only route
        mock_deps["db"].execute_query.assert_not_called()


class TestWorkflowBothRoute:
    def test_both_route_hits_sql_then_docs(self, mock_deps):
        llm = mock_deps["llm"]
        llm.invoke.side_effect = [
            _mock_llm_response("BOTH"),                               # router
            _mock_llm_response("NO_CLARIFICATION_NEEDED"),            # clarify
            _mock_llm_response("SELECT * FROM VIOLATIONS"),           # sql_agent
            _mock_llm_response("Combined answer with data + policy"), # answer
            _mock_llm_response("Score: 8.5\nDecision: APPROVED"),     # reviewer
        ]
        from workflow import app as workflow_app

        result = workflow_app.invoke(
            {
                "question": "Show KYC violations and explain the policy",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-both-route"}},
        )
        assert result["route"] == "both"
        # Both data sources should have been called
        mock_deps["db"].execute_query.assert_called_once()
        mock_deps["vector_store"].similarity_search.assert_called_once()


class TestWorkflowClarificationPath:
    def test_clarification_ends_early(self, mock_deps):
        llm = mock_deps["llm"]
        llm.invoke.side_effect = [
            _mock_llm_response("DOCUMENTS"),                          # router
            _mock_llm_response("Could you be more specific?"),        # clarify (needs it)
        ]
        from workflow import app as workflow_app

        result = workflow_app.invoke(
            {
                "question": "Show me the data",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-clarify"}},
        )
        assert result["needs_clarification"] is True
        assert "specific" in result["final_answer"].lower()
        # Should NOT proceed to retrieval or answer generation
        mock_deps["db"].execute_query.assert_not_called()
        mock_deps["vector_store"].similarity_search.assert_not_called()


class TestWorkflowCacheWrite:
    def test_low_score_does_not_cache(self, mock_deps):
        llm = mock_deps["llm"]
        llm.invoke.side_effect = [
            _mock_llm_response("DOCUMENTS"),
            _mock_llm_response("NO_CLARIFICATION_NEEDED"),
            _mock_llm_response("Some answer"),
            _mock_llm_response("Score: 3.0\nDecision: REJECT"),
        ]
        from workflow import app as workflow_app

        workflow_app.invoke(
            {
                "question": "What is AML?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-low-score"}},
        )
        # Redis set should NOT be called for low-score answers
        mock_deps["redis"].set.assert_not_called()

    def test_high_score_caches(self, mock_deps):
        llm = mock_deps["llm"]
        llm.invoke.side_effect = [
            _mock_llm_response("DOCUMENTS"),
            _mock_llm_response("NO_CLARIFICATION_NEEDED"),
            _mock_llm_response("AML stands for Anti-Money Laundering."),
            _mock_llm_response("Score: 9.0\nDecision: APPROVED"),
        ]
        from workflow import app as workflow_app

        workflow_app.invoke(
            {
                "question": "What is AML?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-high-score"}},
        )
        mock_deps["redis"].set.assert_called_once()
