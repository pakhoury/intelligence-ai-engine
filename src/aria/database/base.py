import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

DEFAULT_CATALOG_DIR = Path(__file__).parent.parent / "catalog"
LEGACY_CATALOG_PATH = Path(__file__).parent.parent / "catalog.json"

_SELECT_PATTERN = re.compile(r"^\s*SELECT\s", re.IGNORECASE | re.DOTALL)

_COMMON_DANGEROUS_PATTERNS = [
    re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|CREATE|EXEC|MERGE|CALL)\b", re.IGNORECASE),
    re.compile(r";\s*\w"),
    re.compile(r"--"),
    re.compile(r"/\*"),
]


class DatabaseConnector(ABC):

    def __init__(self, llm=None, catalog_path: Path = None):
        self.llm = llm
        self._catalog_dir: Path | None = None
        self._metadata: dict = {}
        self._table_cache: dict[str, dict] = {}
        self.catalog: dict = {}

        resolved = catalog_path or DEFAULT_CATALOG_DIR
        if resolved.is_dir():
            self._catalog_dir = resolved
            meta_file = resolved / "database_metadata.json"
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    self._metadata = json.load(f)
                self.catalog = self._metadata
        elif resolved.is_file():
            with open(resolved, encoding="utf-8") as f:
                self.catalog = json.load(f)
        elif LEGACY_CATALOG_PATH.exists():
            with open(LEGACY_CATALOG_PATH, encoding="utf-8") as f:
                self.catalog = json.load(f)

    def _load_table(self, table_name: str) -> dict | None:
        """Load a single table's column definitions on demand."""
        if table_name in self._table_cache:
            return self._table_cache[table_name]

        if self._catalog_dir:
            for schema_data in self._metadata.get("schemas", {}).values():
                entry = schema_data.get("tables", {}).get(table_name)
                if entry and "file" in entry:
                    table_file = self._catalog_dir / entry["file"]
                    if table_file.exists():
                        with open(table_file, encoding="utf-8") as f:
                            data = json.load(f)
                        self._table_cache[table_name] = data
                        return data

        for schema_data in self.catalog.get("schemas", {}).values():
            if table_name in schema_data.get("tables", {}):
                data = schema_data["tables"][table_name]
                self._table_cache[table_name] = data
                return data

        return None

    def _get_table_columns(self, table_name: str) -> set[str]:
        """Get column names for a table."""
        table = self._load_table(table_name)
        if table and "columns" in table:
            return set(table["columns"].keys())
        return set()

    # ── Catalog-based methods (shared across all dialects) ───────────

    def get_relevant_tables(self, question: str) -> list[str]:
        """Keyword-based table selection — the non-LLM fallback path.

        The primary (LLM) path is async and lives in the workflow: it uses
        build_table_selection_prompt() / parse_table_selection() below.
        """
        return self._keyword_fallback(question)

    def build_table_selection_prompt(self, question: str) -> str:
        """Build a prompt for LLM-based table selection."""
        catalog_summary = self._get_table_selection_summary()
        return (
            f"You are a senior {self.get_dialect()} database expert.\n\n"
            f"User Question: {question}\n\n"
            f"Available Tables and their business meaning:\n"
            f"{catalog_summary}\n\n"
            f"Select ONLY the tables that are needed to answer this question.\n"
            f"Return maximum 8 tables.\n"
            f"Return only table names separated by commas.\n\n"
            f"Example output: COMPLIANCE_VIOLATIONS, CONTROL_MAPPINGS, RISK_EVENTS"
        )

    def parse_table_selection(self, llm_response: str) -> list[str]:
        """Parse table names from an LLM response."""
        tables = [t.strip().upper() for t in llm_response.split(",") if t.strip()]
        return tables[:8]

    def get_table_schema(self, table_names: list[str]) -> str:
        if not table_names:
            return "No relevant tables found."
        schema_text = ""
        for table in table_names:
            info = self._load_table(table)
            if info:
                schema_text += f"\nTable: {table}\n"
                schema_text += f"Description: {info['description']}\n"
                schema_text += "Columns and Meaning:\n"
                for col, desc in info["columns"].items():
                    schema_text += f"  - {col}: {desc}\n"
        return schema_text.strip()

    def validate_columns(self, sql: str, table_names: list[str]) -> str:
        table_columns = {}
        for table in table_names:
            cols = self._get_table_columns(table)
            if cols:
                table_columns[table] = cols

        if not table_columns:
            return ""

        sql_keywords = {
            'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'INNER', 'LEFT', 'RIGHT',
            'OUTER', 'CROSS', 'FULL', 'FETCH', 'SET', 'AND', 'OR', 'NOT', 'IN',
            'BY', 'AS', 'SELECT', 'FROM', 'JOIN', 'WHEN', 'THEN', 'ELSE', 'END',
            'CASE', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'EXISTS', 'UNION', 'ALL',
            'DISTINCT', 'INTO', 'VALUES', 'ONLY', 'FIRST', 'ROWS', 'WITH', 'LIMIT',
        }

        alias_map = {}
        for table in table_columns:
            pattern = re.compile(r'\b' + re.escape(table) + r'\b\s+(?:AS\s+)?(\w+)', re.IGNORECASE)
            for match in pattern.finditer(sql):
                alias = match.group(1).upper()
                if alias not in sql_keywords:
                    alias_map[alias] = table
            alias_map[table] = table

        errors = []
        seen = set()
        for match in re.finditer(r'\b(\w+)\.(\w+)\b', sql.upper()):
            alias = match.group(1)
            column = match.group(2)
            if alias in alias_map:
                table = alias_map[alias]
                if column not in table_columns.get(table, set()):
                    key = (alias, column)
                    if key not in seen:
                        seen.add(key)
                        valid = ', '.join(sorted(table_columns.get(table, set())))
                        errors.append(
                            f"Column '{column}' does not exist in table '{table}' "
                            f"(referenced as {alias}.{column}). Valid columns: {valid}"
                        )

        return "; ".join(errors)

    # ── SQL safety validation (override in subclass for dialect-specific) ──

    def validate_sql(self, sql: str) -> str:
        if not _SELECT_PATTERN.match(sql):
            return "Only SELECT queries are allowed."
        for pattern in _COMMON_DANGEROUS_PATTERNS:
            match = pattern.search(sql)
            if match:
                return f"Forbidden SQL pattern detected: {match.group()}"
        return ""

    def apply_row_limit(self, sql: str, limit: int = 500) -> str:
        if re.search(r"\bLIMIT\b|\bFETCH FIRST\b|\bTOP\b", sql, re.IGNORECASE):
            return sql
        return sql.strip().rstrip(";") + f" {self.get_row_limit_clause(limit)}"

    # ── Private helpers ──────────────────────────────────────────────

    def _get_all_table_names(self) -> list[str]:
        """Return all table names from the index."""
        tables = []
        for schema_data in self.catalog.get("schemas", {}).values():
            tables.extend(schema_data.get("tables", {}).keys())
        return tables

    def _get_catalog_summary(self) -> str:
        summary = ""
        for table_name in self._get_all_table_names():
            info = self._load_table(table_name)
            if not info:
                continue
            summary += f"Table: {table_name}\n"
            summary += f"Description: {info['description']}\n"
            summary += "Columns:\n"
            for col, desc in info["columns"].items():
                summary += f"  - {col}: {desc}\n"
            summary += "\n"
        return summary

    def _get_table_selection_summary(self) -> str:
        """Slim summary for LLM table selection: name + description + column names only."""
        summary = ""
        for schema_data in self.catalog.get("schemas", {}).values():
            for table_name, entry in schema_data.get("tables", {}).items():
                desc = entry.get("description", "")
                info = self._load_table(table_name)
                if info:
                    cols = ", ".join(info.get("columns", {}).keys())
                    summary += f"- {table_name}: {desc} [{cols}]\n"
                else:
                    summary += f"- {table_name}: {desc}\n"
        return summary

    KEYWORD_SYNONYMS: dict = {
        "EXPOSURE": ["RISK_EVENTS", "COMPLIANCE_VIOLATIONS"],
        "PENALTY": ["COMPLIANCE_VIOLATIONS"],
        "FINE": ["COMPLIANCE_VIOLATIONS"],
        "LOSS": ["RISK_EVENTS"],
        "CONTROL": ["CONTROL_MAPPINGS"],
        "AUDIT": ["AUDIT_FINDINGS"],
        "FINDING": ["AUDIT_FINDINGS"],
        "INSPECTION": ["AUDIT_FINDINGS"],
        "BREACH": ["COMPLIANCE_VIOLATIONS"],
        "INCIDENT": ["RISK_EVENTS"],
        "KYC": ["COMPLIANCE_VIOLATIONS"],
        "AML": ["COMPLIANCE_VIOLATIONS"],
    }

    def _keyword_fallback(self, question: str) -> list[str]:
        words = set(question.upper().split())
        relevant = set()

        for word in words:
            if word in self.KEYWORD_SYNONYMS:
                relevant.update(self.KEYWORD_SYNONYMS[word])

        for schema_data in self.catalog.get("schemas", {}).values():
            for table_name, entry in schema_data.get("tables", {}).items():
                desc_upper = entry.get("description", "").upper()
                if any(word in table_name or word in desc_upper for word in words):
                    relevant.add(table_name)

        return list(relevant)[:8]

    # ── Abstract methods (each dialect must implement) ───────────────

    @abstractmethod
    def execute_query(self, sql: str) -> str:
        pass

    @abstractmethod
    def get_dialect(self) -> str:
        pass

    @abstractmethod
    def get_row_limit_clause(self, limit: int = 500) -> str:
        pass

    @abstractmethod
    def get_readiness_query(self) -> str:
        pass
