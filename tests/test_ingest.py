"""Tests for the Excel ingestion logic in ingest.py."""
import openpyxl
import pytest


def _create_test_excel(path, sheets: dict):
    """Create a test Excel file with given sheets and data.

    sheets: {"SheetName": [["Header1", "Header2"], ["val1", "val2"], ...]}
    """
    wb = openpyxl.Workbook()
    first = True
    for name, rows in sheets.items():
        ws = wb.active if first else wb.create_sheet(name)
        if first:
            ws.title = name
            first = False
        for row in rows:
            ws.append(row)
    wb.save(path)
    wb.close()


@pytest.fixture
def excel_dir(tmp_path):
    """Create a temp directory with sample Excel files."""
    _create_test_excel(
        tmp_path / "violations_2024.xlsx",
        {
            "Summary": [
                ["Violation ID", "Severity", "Description"],
                ["V001", "Critical", "KYC breach in onboarding"],
                ["V002", "High", "Late SAR filing"],
            ],
            "Details": [
                ["ID", "Department"],
                ["V001", "Compliance"],
            ],
        },
    )
    _create_test_excel(
        tmp_path / "empty.xlsx",
        {"Sheet1": [["Header1", "Header2"]]},  # header only, no data
    )
    return tmp_path


class TestLoadExcelFiles:
    def test_loads_rows_as_documents(self, excel_dir):
        from ingest.ingest import load_excel_files

        docs = load_excel_files(str(excel_dir))
        assert len(docs) >= 3  # 2 from Summary + 1 from Details

    def test_document_content_is_key_value_pairs(self, excel_dir):
        from ingest.ingest import load_excel_files

        docs = load_excel_files(str(excel_dir))
        summary_docs = [d for d in docs if d.metadata.get("sheet") == "Summary"]
        assert len(summary_docs) == 2
        assert "Violation ID: V001" in summary_docs[0].page_content
        assert "Severity: Critical" in summary_docs[0].page_content

    def test_metadata_includes_sheet_and_row(self, excel_dir):
        from ingest.ingest import load_excel_files

        docs = load_excel_files(str(excel_dir))
        doc = docs[0]
        assert doc.metadata["file_type"] == "excel"
        assert "sheet" in doc.metadata
        assert "row" in doc.metadata
        assert doc.metadata["row"] >= 2

    def test_skips_header_only_sheets(self, excel_dir):
        from ingest.ingest import load_excel_files

        docs = load_excel_files(str(excel_dir))
        empty_docs = [d for d in docs if "empty.xlsx" in d.metadata.get("source", "")]
        assert len(empty_docs) == 0

    def test_skips_empty_rows(self, excel_dir):
        _create_test_excel(
            excel_dir / "sparse.xlsx",
            {"Data": [["Col1", "Col2"], [None, None], ["A", "B"]]},
        )
        from ingest.ingest import load_excel_files

        docs = load_excel_files(str(excel_dir))
        sparse_docs = [d for d in docs if "sparse.xlsx" in d.metadata.get("source", "")]
        assert len(sparse_docs) == 1
        assert "Col1: A" in sparse_docs[0].page_content

    def test_handles_missing_openpyxl_gracefully(self, excel_dir, monkeypatch):
        import ingest.ingest as ingest_mod

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        docs = ingest_mod.load_excel_files(str(excel_dir))
        assert docs == []

    def test_no_excel_files_returns_empty(self, tmp_path):
        from ingest.ingest import load_excel_files

        docs = load_excel_files(str(tmp_path))
        assert docs == []
