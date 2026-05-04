import os
import re
import json
from pathlib import Path
from sqlalchemy import create_engine, text
from .base import DatabaseConnector
from typing import List
from observability import logger

CATALOG_PATH = Path(__file__).parent.parent / "catalog.json"

# SQL allowlist: only SELECT statements permitted
_SELECT_PATTERN = re.compile(
    r"^\s*SELECT\s",
    re.IGNORECASE | re.DOTALL,
)

# Additional dangerous patterns to reject even inside SELECT
_DANGEROUS_PATTERNS = [
    re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|CREATE|EXEC|MERGE|CALL)\b", re.IGNORECASE),
    re.compile(r";\s*\w"),          # Statement chaining via semicolon
    re.compile(r"--"),               # SQL line comments
    re.compile(r"/\*"),              # SQL block comments
    re.compile(r"UTL_|DBMS_|SYS\."), # Oracle dangerous packages
]


class OracleConnector(DatabaseConnector):
    def __init__(self, llm=None):
        oracle_user = os.getenv("ORACLE_USER", "user")
        oracle_password = os.getenv("ORACLE_PASSWORD", "password")
        oracle_host = os.getenv("ORACLE_HOST", "localhost")
        oracle_port = os.getenv("ORACLE_PORT", "1521")
        oracle_service = os.getenv("ORACLE_SERVICE", "XEPDB1")
        self.engine = create_engine(
            f"oracle+oracledb://{oracle_user}:{oracle_password}@",
            connect_args={
                "host": oracle_host,
                "port": int(oracle_port),
                "service_name": oracle_service,
            },
            pool_size=10,
            max_overflow=20
        )
        self.schema_owner = os.getenv("ORACLE_SCHEMA", "COMPLIANCE")
        self.llm = llm

        # Load rich business catalog
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

    def get_relevant_tables(self, question: str) -> List[str]:
        """Use LLM to intelligently select tables from catalog."""
        if not self.llm:
            return self._keyword_fallback(question)

        catalog_summary = self._get_catalog_summary()

        prompt = f"""You are a senior Oracle database expert.

User Question: {question}

Available Tables and their business meaning:
{catalog_summary}

Select ONLY the tables that are needed to answer this question.
Return maximum 5 tables.
Return only table names separated by commas.

Example output: COMPLIANCE_VIOLATIONS, CONTROL_MAPPINGS, RISK_EVENTS"""

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

    def _get_catalog_summary(self) -> str:
        """Create rich summary for LLM."""
        summary = ""
        for schema_name, schema_data in self.catalog["schemas"].items():
            for table_name, info in schema_data["tables"].items():
                summary += f"Table: {table_name}\n"
                summary += f"Description: {info['description']}\n"
                summary += "Columns:\n"
                for col, desc in info["columns"].items():
                    summary += f"  - {col}: {desc}\n"
                summary += "\n"
        return summary

    def _keyword_fallback(self, question: str) -> List[str]:
        """Fallback if LLM fails."""
        words = set(question.upper().split())
        relevant = []
        for schema_data in self.catalog["schemas"].values():
            for table_name, info in schema_data["tables"].items():
                desc_upper = info["description"].upper()
                if any(word in table_name or word in desc_upper for word in words):
                    relevant.append(table_name)
        return relevant[:5]

    def get_table_schema(self, table_names: List[str]) -> str:
        """Return rich business context from catalog."""
        if not table_names:
            return "No relevant tables found."

        schema_text = ""
        for table in table_names:
            for schema_data in self.catalog["schemas"].values():
                if table in schema_data["tables"]:
                    info = schema_data["tables"][table]
                    schema_text += f"\nTable: {table}\n"
                    schema_text += f"Description: {info['description']}\n"
                    schema_text += "Columns and Meaning:\n"
                    for col, desc in info["columns"].items():
                        schema_text += f"  - {col}: {desc}\n"
                    break
        return schema_text.strip()

    def validate_columns(self, sql: str, table_names: List[str]) -> str:
        """Validate that column references in SQL match the catalog schema.
        Returns error description or empty string if valid."""
        table_columns = {}
        for table in table_names:
            for schema_data in self.catalog["schemas"].values():
                if table in schema_data["tables"]:
                    table_columns[table] = set(schema_data["tables"][table]["columns"].keys())
                    break

        if not table_columns:
            return ""

        sql_keywords = {
            'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'INNER', 'LEFT', 'RIGHT',
            'OUTER', 'CROSS', 'FULL', 'FETCH', 'SET', 'AND', 'OR', 'NOT', 'IN',
            'BY', 'AS', 'SELECT', 'FROM', 'JOIN', 'WHEN', 'THEN', 'ELSE', 'END',
            'CASE', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'EXISTS', 'UNION', 'ALL',
            'DISTINCT', 'INTO', 'VALUES', 'ONLY', 'FIRST', 'ROWS', 'WITH',
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

    def _validate_sql(self, sql: str) -> str:
        """Validate SQL is a safe SELECT query. Returns error message or empty string."""
        if not _SELECT_PATTERN.match(sql):
            return "Only SELECT queries are allowed."

        for pattern in _DANGEROUS_PATTERNS:
            match = pattern.search(sql)
            if match:
                return f"Forbidden SQL pattern detected: {match.group()}"

        return ""

    def execute_query(self, sql: str) -> str:
        # Allowlist validation
        error = self._validate_sql(sql)
        if error:
            logger.warning(
                f"SQL rejected: {error}",
                extra={"node": "sql_path", "sql": sql, "error": error},
            )
            return f"Error: {error}"

        if "FETCH FIRST" not in sql.upper():
            sql = sql.strip().rstrip(";") + " FETCH FIRST 500 ROWS ONLY"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return str(result.fetchall()[:30])
        except Exception as e:
            logger.error(
                f"SQL execution error: {e}",
                extra={"node": "sql_path", "sql": sql, "error": str(e)},
            )
            return f"Execution Error: {str(e)}"

    def get_dialect(self) -> str:
        return "oracle"
