"""
Hybrid retrieval — Postgres full-text (lexical) search plus Reciprocal Rank
Fusion for combining it with vector similarity results.

Why hybrid: embeddings capture paraphrase ("rules for verifying who a
customer is" ~ KYC policy) but carry almost no signal for exact identifiers
(incident IDs like SOE-45678, violation codes like V-203, regulation
sections). Postgres full-text has the opposite profile — exact tokens rank
highly, paraphrase matches nothing. Running both and fusing by rank gets the
best of each without having to normalize incompatible score scales.
"""
from langchain_core.documents import Document
from sqlalchemy import text as sa_text

# Standard RRF damping constant: differences among top ranks dominate,
# the tail flattens out.
RRF_K = 60

# Searches the chunk table maintained by langchain's PGVector store.
_LEXICAL_SQL = """
SELECT e.document, e.cmetadata
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = :collection
  AND to_tsvector('english', e.document) @@ websearch_to_tsquery('english', :query)
  {doc_type_clause}
ORDER BY ts_rank_cd(to_tsvector('english', e.document), websearch_to_tsquery('english', :query)) DESC
LIMIT :k
"""

FULLTEXT_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_pg_embedding_document_fts "
    "ON langchain_pg_embedding USING gin (to_tsvector('english', document))"
)


def lexical_search(
    engine,
    query: str,
    collection_name: str,
    k: int = 20,
    doc_type: str | None = None,
) -> list[Document]:
    """Full-text search over the PGVector chunk table. Returns ranked Documents.

    websearch_to_tsquery parses free-form user text safely (no tsquery syntax
    errors on quotes/operators), and hyphenated identifiers like SOE-45678
    are indexed both whole and as parts by the english config.
    """
    doc_type_clause = "AND e.cmetadata->>'doc_type' = :doc_type" if doc_type else ""
    sql = _LEXICAL_SQL.format(doc_type_clause=doc_type_clause)
    params = {"collection": collection_name, "query": query, "k": k}
    if doc_type:
        params["doc_type"] = doc_type
    with engine.connect() as conn:
        rows = conn.execute(sa_text(sql), params).fetchall()
    return [Document(page_content=row[0], metadata=row[1] or {}) for row in rows]


def rrf_fuse(result_lists: list[list[Document]], top_n: int = 6, k: int = RRF_K) -> list[Document]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank).

    Rank-based fusion means cosine similarity and ts_rank scores never need
    to be normalized against each other. A document appearing in multiple
    lists accumulates score, so agreement between retrievers wins.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = str(doc.page_content)
            if key not in docs_by_key:
                docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_key[key] for key, _ in ranked[:top_n]]


def ensure_fulltext_index(engine) -> None:
    """Create the GIN full-text index if missing. Idempotent; call at ingest time."""
    with engine.begin() as conn:
        conn.execute(sa_text(FULLTEXT_INDEX_DDL))
