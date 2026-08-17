"""
Integration tests for the LangGraph workflow — verifies conditional routing,
chained strategies, and reflection loop with all external dependencies mocked.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_deps():
    """Patch all external dependencies so the compiled graph can run."""
    with (
        patch("nodes.redis_client") as mock_redis,
        patch("nodes._ainvoke_llm", new_callable=AsyncMock) as mock_ainvoke,
        patch("nodes.db_connector") as mock_db,
        patch("nodes._get_vector_store") as mock_vs_fn,
        patch("nodes.lexical_search", return_value=[]),
        patch("nodes._get_metric_resolver") as mock_resolver_fn,
    ):
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=None)
        mock_resolver_fn.return_value = mock_resolver

        # Defaults
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
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
            "ainvoke": mock_ainvoke,
            "db": mock_db,
            "vector_store_fn": mock_vs_fn,
            "vector_store": mock_store,
        }


class TestWorkflowCacheHitPath:
    async def test_cache_hit_ends_immediately(self, mock_deps):
        mock_deps["redis"].get.return_value = "Cached answer"
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "How many violations?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-cache-hit"}},
        )
        assert result["cache_hit"] is True
        assert result["final_answer"] == "Cached answer"
        mock_deps["ainvoke"].assert_not_called()


class TestWorkflowSQLRoute:
    async def test_sql_only_route(self, mock_deps):
        mock_deps["ainvoke"].side_effect = [
            "SQL_ONLY",                               # router
            "NO_CLARIFICATION_NEEDED",                # clarify
            "COMPLIANCE_VIOLATIONS",                   # table selection
            "SELECT COUNT(*) FROM VIOLATIONS",        # sql_agent
            "There are 2 critical violations.",        # answer
            "Score: 9.0\nDecision: APPROVED",          # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "How many violations?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-sql-route"}},
        )
        from nodes import EXPLORATORY_NOTICE
        assert result["route"] == "sql_only"
        # LLM-generated SQL (not a compiled metric) → answer must carry the
        # exploratory-path disclosure.
        assert result["final_answer"] == "There are 2 critical violations." + EXPLORATORY_NOTICE
        assert result["data_path"] == "exploratory"
        assert result["review_score"] == 9.0
        mock_deps["vector_store"].similarity_search.assert_not_called()


class TestWorkflowDocumentsRoute:
    async def test_documents_only_route(self, mock_deps):
        mock_deps["ainvoke"].side_effect = [
            "DOCS_ONLY",                              # router
            "NO_CLARIFICATION_NEEDED",                # clarify
            "The KYC policy requires...",              # answer
            "Score: 8.0\nDecision: APPROVED",          # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "What is KYC policy?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-docs-route"}},
        )
        assert result["route"] == "docs_only"
        assert "KYC" in result["final_answer"]
        mock_deps["db"].execute_query.assert_not_called()


class TestWorkflowParallelRoute:
    async def test_parallel_route_hits_sql_then_docs(self, mock_deps):
        mock_deps["ainvoke"].side_effect = [
            "PARALLEL",                               # router
            "NO_CLARIFICATION_NEEDED",                # clarify
            "COMPLIANCE_VIOLATIONS",                   # table selection
            "SELECT * FROM VIOLATIONS",               # sql_agent
            "Combined answer with data + policy",      # answer
            "Score: 8.5\nDecision: APPROVED",          # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "Show KYC violations and explain the policy",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-parallel-route"}},
        )
        assert result["route"] == "parallel"
        mock_deps["db"].execute_query.assert_called_once()
        mock_deps["vector_store"].similarity_search.assert_called_once()


class TestWorkflowDocsThenSqlRoute:
    async def test_docs_then_sql_chains_through_extract(self, mock_deps):
        # Data must support the mocked answer's "$2.4M" or the number
        # grounding validator triggers reflection.
        mock_deps["db"].execute_query.return_value = "[(2400000,)]"
        mock_deps["ainvoke"].side_effect = [
            "DOCS_THEN_SQL",                                          # router
            "NO_CLARIFICATION_NEEDED",                                # clarify
            # vector_retrieval doesn't call LLM — it just searches
            "SUM(fine_amount + remediation_cost + audit_fees)",       # extract
            "COMPLIANCE_VIOLATIONS",                                   # table selection
            "SELECT SUM(fine_amount) FROM COMPLIANCE_VIOLATIONS",     # sql_agent
            "The compliance cost for 2024 is $2.4M.",                 # answer
            "Score: 9.0\nDecision: APPROVED",                          # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "What is the compliance cost for 2024?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-docs-then-sql"}},
        )
        assert result["route"] == "docs_then_sql"
        assert result["extracted_context"] == "SUM(fine_amount + remediation_cost + audit_fees)"
        # Both sources must be called: docs first, then SQL
        mock_deps["vector_store"].similarity_search.assert_called_once()
        mock_deps["db"].execute_query.assert_called_once()
        assert "$2.4M" in result["final_answer"]


class TestWorkflowSqlThenDocsRoute:
    async def test_sql_then_docs_chains_through_extract(self, mock_deps):
        # Data must support the mocked answer's "47 incidents" or the number
        # grounding validator triggers reflection.
        mock_deps["db"].execute_query.return_value = "[('ACCESS_CONTROL', 47)]"
        mock_deps["ainvoke"].side_effect = [
            "SQL_THEN_DOCS",                                         # router
            "NO_CLARIFICATION_NEEDED",                               # clarify
            "COMPLIANCE_VIOLATIONS",                                   # table selection
            "SELECT type, COUNT(*) FROM VIOLATIONS GROUP BY type",   # sql_agent
            "ACCESS_CONTROL",                                         # extract
            # vector_retrieval doesn't call LLM — it just searches
            "ACCESS_CONTROL had 47 incidents. The policy states...",  # answer
            "Score: 9.0\nDecision: APPROVED",                         # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "Most common violation and its policy?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-sql-then-docs"}},
        )
        assert result["route"] == "sql_then_docs"
        assert result["extracted_context"] == "ACCESS_CONTROL"
        # Both sources must be called: SQL first, then docs
        mock_deps["db"].execute_query.assert_called_once()
        mock_deps["vector_store"].similarity_search.assert_called_once()
        # Vector search should use the extracted context, not the raw question
        search_query = mock_deps["vector_store"].similarity_search.call_args[0][0]
        assert search_query == "ACCESS_CONTROL"


class TestWorkflowClarificationPath:
    async def test_clarification_ends_early(self, mock_deps):
        mock_deps["ainvoke"].side_effect = [
            "DOCS_ONLY",                              # router
            "Could you be more specific?",            # clarify (needs it)
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "Show me the data",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-clarify"}},
        )
        assert result["needs_clarification"] is True
        assert "specific" in result["final_answer"].lower()
        mock_deps["db"].execute_query.assert_not_called()
        mock_deps["vector_store"].similarity_search.assert_not_called()


class TestWorkflowCacheWrite:
    async def test_low_score_does_not_cache(self, mock_deps):
        mock_deps["ainvoke"].side_effect = [
            "DOCS_ONLY",                                                # router
            "NO_CLARIFICATION_NEEDED",                                  # clarify
            "Some answer",                                              # answer (attempt 0)
            "Score: 3.0\nDecision: REJECT\nReason: Too vague",        # reviewer (attempt 1, reflect)
            "Slightly better answer",                                   # answer (reflection 1)
            "Score: 5.0\nDecision: REJECT\nReason: Still incomplete",  # reviewer (attempt 2, max)
        ]
        from workflow import app as workflow_app

        await workflow_app.ainvoke(
            {
                "question": "What is AML?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-low-score"}},
        )
        mock_deps["redis"].set.assert_not_called()

    async def test_high_score_caches(self, mock_deps):
        mock_deps["ainvoke"].side_effect = [
            "DOCS_ONLY",                              # router
            "NO_CLARIFICATION_NEEDED",                # clarify
            "AML stands for Anti-Money Laundering.",  # answer
            "Score: 9.0\nDecision: APPROVED",          # reviewer
        ]
        from workflow import app as workflow_app

        await workflow_app.ainvoke(
            {
                "question": "What is AML?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-high-score"}},
        )
        mock_deps["redis"].set.assert_called_once()


class TestWorkflowReflection:
    async def test_reflection_improves_answer(self, mock_deps):
        """Low score triggers reflection; improved answer passes on second attempt."""
        mock_deps["ainvoke"].side_effect = [
            "DOCS_ONLY",                                                # router
            "NO_CLARIFICATION_NEEDED",                                  # clarify
            "Weak answer",                                              # answer (attempt 0)
            "Score: 4.0\nDecision: REJECT\nReason: Too vague",        # reviewer (attempt 1, reflect)
            "Strong detailed answer about AML compliance",              # answer (reflection)
            "Score: 9.0\nDecision: APPROVED\nReason: Comprehensive",   # reviewer (attempt 2, pass)
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "What is AML?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-reflection"}},
        )
        assert result["final_answer"] == "Strong detailed answer about AML compliance"
        assert result["review_score"] == 9.0
        assert result["reflection_attempt"] == 2
        mock_deps["redis"].set.assert_called_once()

    async def test_sql_failure_skips_reflection(self, mock_deps):
        """An unfixable data failure (SQL execution error) must end the run after
        one review instead of burning reflection cycles that can never pass."""
        mock_deps["db"].execute_query.return_value = (
            "Execution Error: ORA-00942: table or view does not exist"
        )
        mock_deps["ainvoke"].side_effect = [
            "SQL_ONLY",                                     # router
            "NO_CLARIFICATION_NEEDED",                      # clarify
            "COMPLIANCE_VIOLATIONS",                        # table selection
            "SELECT COUNT(*) FROM VIOLATIONS",              # sql attempt 1
            "SELECT COUNT(*) FROM VIOLATIONS",              # sql attempt 2
            "SELECT COUNT(*) FROM VIOLATIONS",              # sql attempt 3
            "I was unable to retrieve the requested data.",  # answer
            "Score: 9.0\nDecision: APPROVED",               # reviewer
        ]
        from workflow import app as workflow_app

        result = await workflow_app.ainvoke(
            {
                "question": "How many violations?",
                "session_id": "s1",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-skip-reflection"}},
        )
        assert result["skip_reflection"] is True
        assert result["reflection_attempt"] == 1
        assert result["review_score"] <= 3.0
        # No reflection cycles ran: exactly the 8 mocked calls were consumed.
        assert mock_deps["ainvoke"].call_count == 8
        mock_deps["redis"].set.assert_not_called()


class TestAsyncOnlyCheckpointerCompatibility:
    """Regression test for a real deployment bug: AsyncPostgresSaver (the
    persistent-checkpointer backend used in production) implements only the
    async checkpoint API — its sync methods raise NotImplementedError. Any
    code path that calls the graph's sync state-history/invoke methods
    breaks against it, even though the same code works fine against
    MemorySaver (which implements both). This test compiles the real graph
    against an async-only checkpointer to catch that class of regression
    without needing a live Postgres instance."""

    async def test_aget_state_history_works_sync_does_not(self, mock_deps):
        from langgraph.checkpoint.memory import MemorySaver

        from workflow import workflow

        class AsyncOnlyCheckpointer(MemorySaver):
            """MemorySaver's own async methods just delegate to its sync
            methods (`aget_tuple` calls `self.get_tuple`), so naively
            overriding the sync methods to raise breaks the async path too.
            Bypass that delegation by calling the base implementation
            directly (`MemorySaver.get_tuple(self, ...)`), so the async
            surface stays fully functional while the sync surface — the
            thing this test needs disabled — is not.
            """

            async def aget_tuple(self, config):
                return MemorySaver.get_tuple(self, config)

            async def alist(self, config, **kwargs):
                for item in MemorySaver.list(self, config, **kwargs):
                    yield item

            async def aput(self, config, checkpoint, metadata, new_versions):
                return MemorySaver.put(self, config, checkpoint, metadata, new_versions)

            async def aput_writes(self, config, writes, task_id, task_path=""):
                return MemorySaver.put_writes(self, config, writes, task_id, task_path)

            def get_tuple(self, config):
                raise NotImplementedError

            def list(self, config, **kwargs):
                raise NotImplementedError

            def put(self, *args, **kwargs):
                raise NotImplementedError

            def put_writes(self, *args, **kwargs):
                raise NotImplementedError

        compiled = workflow.compile(checkpointer=AsyncOnlyCheckpointer())
        config = {"configurable": {"thread_id": "async-only-test"}}

        mock_deps["redis"].get.return_value = "Cached answer"
        await compiled.ainvoke(
            {"question": "How many violations?", "session_id": "s1", "conversation_history": []},
            config=config,
        )

        # The async interface (what main.py's /audit endpoint must use) works.
        states = [s async for s in compiled.aget_state_history(config)]
        assert len(states) > 0

        # The sync interface (the original bug) does not.
        with pytest.raises(NotImplementedError):
            list(compiled.get_state_history(config))
