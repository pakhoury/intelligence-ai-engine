import os
import json
from pathlib import Path
from sqlalchemy import create_engine, text
from .base import DatabaseConnector
from typing import List

CATALOG_PATH = Path(__file__).parent.parent / "catalog.json"


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
        """Use LLM to intelligently select tables from catalog"""
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
            print(f"LLM selected tables: {tables}")
            return tables[:5]
        except Exception as e:
            print(f"LLM table selection failed: {e}")
            return self._keyword_fallback(question)

    def _get_catalog_summary(self) -> str:
        """Create rich summary for LLM"""
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
        """Fallback if LLM fails"""
        words = set(question.upper().split())
        relevant = []
        for schema_data in self.catalog["schemas"].values():
            for table_name, info in schema_data["tables"].items():
                desc_upper = info["description"].upper()
                if any(word in table_name or word in desc_upper for word in words):
                    relevant.append(table_name)
        return relevant[:5]

    def get_table_schema(self, table_names: List[str]) -> str:
        """Return rich business context from catalog"""
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

    def execute_query(self, sql: str) -> str:
        sql_upper = sql.upper()
        if any(cmd in sql_upper for cmd in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT", "CREATE", "EXEC"]):
            return "Error: Only SELECT queries allowed."

        if "FETCH FIRST" not in sql_upper:
            sql = sql.strip().rstrip(";") + " FETCH FIRST 500 ROWS ONLY"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return str(result.fetchall()[:30])
        except Exception as e:
            return f"Execution Error: {str(e)}"

    def get_dialect(self) -> str:
        return "oracle"
