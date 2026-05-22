import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from observability import logger

DEFAULT_CATALOG_PATH = Path(__file__).parent.parent / "catalog.json"

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
        self.catalog = {}
        catalog_file = catalog_path or DEFAULT_CATALOG_PATH
        if catalog_file.exists():
            with open(catalog_file, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)

    # ── Catalog-based methods (shared across all dialects) ───────────

    def get_relevant_tables(self, question: str) -> List[str]:
        if not self.llm:
            return self._keyword_fallback(question)

        catalog_summary = self._get_catalog_summary()
        prompt = (
            f"You are a senior {self.get_dialect()} database expert.\n\n"
            f"User Question: {question}\n\n"
            f"Available Tables and their business meaning:\n{catalog_summary}\n\n"
            f"Select ONLY the tables that are needed to answer this question.\n"
            f"Return maximum 5 tables.\n"
            f"Return only table names separated by commas.\n\n"
            f"Example output: COMPLIANCE_VIOLATIONS, CONTROL_MAPPINGS, RISK_EVENTS"
        )
        try:
            response = self.llm.invoke(prompt)
            tables = [t.strip().upper() for t in response.content.split(",") if t.strip()]
            logger.info(
                f"LLM selected tables: {tables}",
                extra={"node": "sql_path", "sql": f"tables={tables}"},
            )
            return tables[:5]
        except Exception as e:
            logger.warning(
                f"LLM table selection failed: {e}",
                extra={"node": "sql_path", "error": str(e)},
            )
            return self._keyword_fallback(question)

    def get_table_schema(self, table_names: List[str]) -> str:
        if not table_names:
            return "No relevant tables found."
        schema_text = ""
        for table in table_names:
            for schema_data in self.catalog.get("schemas", {}).values():
                if table in schema_data.get("tables", {}):
                    info = schema_data["tables"][table]
                    schema_text += f"\nTable: {table}\n"
                    schema_text += f"Description: {info['description']}\n"
                    schema_text += "Columns and Meaning:\n"
                    for col, desc in info["columns"].items():
                        schema_text += f"  - {col}: {desc}\n"
                    break
        return schema_text.strip()

    def validate_columns(self, sql: str, table_names: List[str]) -> str:
        table_columns = {}
        for table in table_names:
            for schema_data in self.catalog.get("schemas", {}).values():
                if table in schema_data.get("tables", {}):
                    table_columns[table] = set(schema_data["tables"][table]["columns"].keys())
                    break

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

    def _get_catalog_summary(self) -> str:
        summary = ""
        for schema_name, schema_data in self.catalog.get("schemas", {}).items():
            for table_name, info in schema_data.get("tables", {}).items():
                summary += f"Table: {table_name}\n"
                summary += f"Description: {info['description']}\n"
                summary += "Columns:\n"
                for col, desc in info["columns"].items():
                    summary += f"  - {col}: {desc}\n"
                summary += "\n"
        return summary

    def _keyword_fallback(self, question: str) -> List[str]:
        words = set(question.upper().split())
        relevant = []
        for schema_data in self.catalog.get("schemas", {}).values():
            for table_name, info in schema_data.get("tables", {}).items():
                desc_upper = info["description"].upper()
                if any(word in table_name or word in desc_upper for word in words):
                    relevant.append(table_name)
        return relevant[:5]

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
