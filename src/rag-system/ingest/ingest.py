import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from pathlib import Path
import json
import re

print("🚀 Starting Document Ingestion...\n")

# ===================== CONFIG =====================
DOCUMENT_FOLDER = "documents/"
METADATA_FILE = "metadata_mapping.json"
COLLECTION_NAME = "compliance_docs"

# Load metadata mapping
try:
    with open(METADATA_FILE, "r") as f:
        metadata_map = json.load(f)
    print(f"Loaded metadata for {len(metadata_map)} files")
except FileNotFoundError:
    print("metadata_mapping.json not found. Using empty metadata.")
    metadata_map = {}

# ===================== LOAD DOCUMENTS =====================
loader = PyPDFDirectoryLoader(DOCUMENT_FOLDER)
docs = loader.load()
print(f"Loaded {len(docs)} PDF documents")

# ===================== CHUNKING =====================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=120
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# ===================== ADD METADATA =====================
for chunk in chunks:
    source = chunk.metadata.get("source", "")
    filename = Path(source).name

    if filename in metadata_map:
        chunk.metadata.update(metadata_map[filename])
    else:
        chunk.metadata["category"] = "Compliance"
        chunk.metadata["doc_type"] = "policy" if "policy" in filename.lower() else "report"
        chunk.metadata["title"] = filename

    # Auto-detect year
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        chunk.metadata["year"] = year_match.group(1)

print("✅ Metadata applied to all chunks")

# ===================== STORE IN PGVECTOR =====================
print("Embedding and storing in PostgreSQL + PGVector...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pg_user = os.getenv("POSTGRES_USER", "postgres")
pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
pg_host = os.getenv("POSTGRES_HOST", "localhost")
pg_db = os.getenv("POSTGRES_DB", "rag_db")

connection_string = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"

vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=connection_string,
    use_jsonb=True,
)

print(f"\n🎉 SUCCESS! {len(chunks)} chunks have been stored in vector database.")
print("You can now run your RAG system.")