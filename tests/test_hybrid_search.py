"""
Unit tests for hybrid retrieval — RRF fusion, lexical search, and the
hybrid vector_retrieval node.
"""
import os
import sys
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

from retrieval import lexical_search, rrf_fuse  # noqa: E402


def _doc(content: str, **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


class TestRRFFusion:
    def test_doc_in_both_lists_outranks_single_list_winner(self):
        """The SOE-45678 scenario: rank 8 in vector + rank 1 in lexical beats vector's rank 1."""
        target = _doc("Incident SOE-45678: unauthorized access to trading system")
        generic = [_doc(f"Generic incident framework text {i}") for i in range(7)]
        vector_results = generic + [target]  # target at vector rank 8
        lexical_results = [target]           # exact-ID match at lexical rank 1

        fused = rrf_fuse([vector_results, lexical_results], top_n=6)
        assert fused[0].page_content == target.page_content

    def test_agreement_beats_single_high_rank(self):
        a = _doc("chunk A")
        b = _doc("chunk B")
        # A is rank 1 in one list; B is rank 2 in both lists.
        # 1/61 = 0.0164 < 1/62 + 1/62 = 0.0323 -> B wins.
        fused = rrf_fuse([[a, b], [_doc("chunk C"), b]], top_n=3)
        assert fused[0].page_content == "chunk B"

    def test_deduplicates_identical_content(self):
        a1 = _doc("same text", title="from vector")
        a2 = _doc("same text", title="from lexical")
        fused = rrf_fuse([[a1], [a2]], top_n=6)
        assert len(fused) == 1
        # First occurrence's metadata is kept
        assert fused[0].metadata["title"] == "from vector"

    def test_top_n_limit(self):
        results = [_doc(f"chunk {i}") for i in range(10)]
        fused = rrf_fuse([results], top_n=6)
        assert len(fused) == 6

    def test_single_list_preserves_order(self):
        results = [_doc("first"), _doc("second"), _doc("third")]
        fused = rrf_fuse([results], top_n=6)
        assert [d.page_content for d in fused] == ["first", "second", "third"]

    def test_empty_lists(self):
        assert rrf_fuse([[], []], top_n=6) == []


class TestLexicalSearch:
    def _engine_returning(self, rows):
        engine = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = rows
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        return engine, conn

    def test_builds_documents_from_rows(self):
        engine, _ = self._engine_returning([
            ("Incident SOE-45678 report text", {"title": "Incident Report"}),
            ("Another chunk", None),
        ])
        docs = lexical_search(engine, "incident SOE-45678", "compliance_docs", k=20)
        assert len(docs) == 2
        assert docs[0].page_content == "Incident SOE-45678 report text"
        assert docs[0].metadata["title"] == "Incident Report"
        assert docs[1].metadata == {}

    def test_doc_type_filter_applied(self):
        engine, conn = self._engine_returning([])
        lexical_search(engine, "kyc policy", "compliance_docs", k=20, doc_type="policy")
        sql_text = str(conn.execute.call_args[0][0])
        params = conn.execute.call_args[0][1]
        assert "doc_type" in sql_text
        assert params["doc_type"] == "policy"
        assert params["collection"] == "compliance_docs"

    def test_no_doc_type_omits_clause(self):
        engine, conn = self._engine_returning([])
        lexical_search(engine, "kyc policy", "compliance_docs", k=20)
        params = conn.execute.call_args[0][1]
        assert "doc_type" not in params


class TestHybridVectorRetrievalNode:
    @patch("nodes.lexical_search")
    @patch("nodes._get_vector_store")
    async def test_lexical_hit_surfaces_exact_id_chunk(self, mock_store_fn, mock_lexical):
        """Vector alone would cut the SOE chunk at top-6; lexical rescues it."""
        target = _doc("Incident SOE-45678: root cause analysis", title="SOE-45678 Report")
        vector_results = [_doc(f"Generic incident doc {i}", title=f"Doc {i}") for i in range(7)]
        vector_results.append(target)  # vector rank 8 — outside top 6
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = vector_results
        mock_store_fn.return_value = mock_store
        mock_lexical.return_value = [target]

        from nodes import vector_retrieval
        result = await vector_retrieval({"question": "What happened in incident SOE-45678?"})

        assert len(result["retrieved_docs"]) == 6
        assert "SOE-45678" in result["retrieved_docs"][0]

    @patch("nodes.lexical_search", side_effect=ConnectionError("no postgres"))
    @patch("nodes._get_vector_store")
    async def test_lexical_failure_falls_back_to_vector_only(self, mock_store_fn, mock_lexical):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [_doc("KYC policy text", title="KYC Policy")]
        mock_store_fn.return_value = mock_store

        from nodes import vector_retrieval
        result = await vector_retrieval({"question": "What is the KYC policy?"})

        assert len(result["retrieved_docs"]) == 1
        assert "KYC policy text" in result["retrieved_docs"][0]

    @patch("nodes.lexical_search", return_value=[])
    @patch("nodes._get_vector_store")
    async def test_both_retrievers_empty_returns_no_docs(self, mock_store_fn, mock_lexical):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_store_fn.return_value = mock_store

        from nodes import vector_retrieval
        result = await vector_retrieval({"question": "anything"})
        assert result["retrieved_docs"] == []

    @patch("nodes.lexical_search", return_value=[])
    @patch("nodes._get_vector_store", side_effect=Exception("PGVector down"))
    async def test_vector_failure_degrades_gracefully(self, mock_store_fn, mock_lexical):
        from nodes import vector_retrieval
        result = await vector_retrieval({"question": "anything"})
        assert result["retrieved_docs"] == []
