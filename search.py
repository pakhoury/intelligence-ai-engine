"""
Direct semantic search on PGVector.

Usage:
  python search.py "your search query"
  python search.py "KYC requirements" --top 5
"""
import sys
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

pg_user = os.getenv("POSTGRES_USER", "postgres")
pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
pg_host = os.getenv("POSTGRES_HOST", "localhost")
pg_db = os.getenv("POSTGRES_DB", "rag_db")
connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

store = PGVector(
    collection_name="compliance_docs",
    connection_string=connection_string,
    embedding_function=embeddings,
    use_jsonb=True,
)

query = sys.argv[1] if len(sys.argv) > 1 else "compliance policy"
top_k = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--top" else 4

print(f"\nSearching for: \"{query}\" (top {top_k})\n")
print("=" * 70)

# Similarity search with scores
results = store.similarity_search_with_score(query, k=top_k)

for i, (doc, score) in enumerate(results, 1):
    title = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
    category = doc.metadata.get("category", "N/A")
    doc_type = doc.metadata.get("doc_type", doc.metadata.get("document_type", "N/A"))
    print(f"\n--- Result {i} | Score: {score:.4f} | {title} ---")
    print(f"Type: {doc_type} | Category: {category}")
    print(f"Content: {doc.page_content[:300]}...")
    print()
