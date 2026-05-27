import os
import json
import re
from pathlib import Path

from langchain_core.documents import Document

DOCUMENT_FOLDER = "documents/"
METADATA_FILE = "metadata_mapping.json"
COLLECTION_NAME = "compliance_docs"


def load_excel_files(folder: str) -> list[Document]:
    """Load .xlsx and .xls files, converting each row to a Document.

    Each sheet becomes a group of documents. Row content is formatted as
    'Column: Value' pairs so the text is meaningful for embedding.
    """
    try:
        import openpyxl
    except ImportError:
        print("openpyxl not installed — skipping Excel files. pip install openpyxl")
        return []

    excel_docs = []
    folder_path = Path(folder)

    for ext in ("*.xlsx", "*.xls"):
        for filepath in folder_path.glob(ext):
            try:
                wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            except Exception as e:
                print(f"WARNING: Could not open {filepath.name}: {e}")
                continue

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))
                if len(rows) < 2:
                    continue

                headers = [str(h).strip() if h else f"Column_{i}" for i, h in enumerate(rows[0])]

                for row_idx, row in enumerate(rows[1:], start=2):
                    pairs = []
                    for header, value in zip(headers, row):
                        if value is not None and str(value).strip():
                            pairs.append(f"{header}: {value}")
                    if not pairs:
                        continue

                    text = "\n".join(pairs)
                    excel_docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(filepath),
                            "sheet": sheet_name,
                            "row": row_idx,
                            "file_type": "excel",
                        },
                    ))

            wb.close()
            print(f"  Excel: {filepath.name} — {len([d for d in excel_docs if d.metadata['source'] == str(filepath)])} rows")

    return excel_docs


def main():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import PGVector

    print("Starting Document Ingestion...\n")

    try:
        with open(METADATA_FILE, "r") as f:
            metadata_map = json.load(f)
        print(f"Loaded metadata for {len(metadata_map)} files")
    except FileNotFoundError:
        print("metadata_mapping.json not found. Using empty metadata.")
        metadata_map = {}

    # Load PDFs
    loader = PyPDFDirectoryLoader(DOCUMENT_FOLDER)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF files")

    # Load Excel files
    excel_docs = load_excel_files(DOCUMENT_FOLDER)
    docs.extend(excel_docs)
    print(f"Total documents after Excel: {len(docs)}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Add metadata
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        filename = Path(source).name

        if filename in metadata_map:
            chunk.metadata.update(metadata_map[filename])
        else:
            if chunk.metadata.get("file_type") != "excel":
                chunk.metadata["category"] = "Compliance"
                chunk.metadata["doc_type"] = "policy" if "policy" in filename.lower() else "report"
                chunk.metadata["title"] = filename
            else:
                chunk.metadata.setdefault("category", "Compliance")
                chunk.metadata.setdefault("doc_type", "data")
                chunk.metadata.setdefault("title", filename)

        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            chunk.metadata["year"] = year_match.group(1)

    print("Metadata applied to all chunks")

    # Store in PGVector
    print("Embedding and storing in PostgreSQL + PGVector...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_db = os.getenv("POSTGRES_DB", "rag_db")

    connection_string = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"

    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=connection_string,
        use_jsonb=True,
    )

    print(f"\nSUCCESS: {len(chunks)} chunks stored in vector database.")
    print(f"  PDF chunks: {len([c for c in chunks if c.metadata.get('file_type') != 'excel'])}")
    print(f"  Excel chunks: {len([c for c in chunks if c.metadata.get('file_type') == 'excel'])}")


if __name__ == "__main__":
    main()
