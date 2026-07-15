import os
import re
from pathlib import Path

from sqlalchemy import create_engine, text

from observability import logger

from .base import DatabaseConnector

CATALOG_PATH = Path(__file__).parent.parent / "catalog"

_ORACLE_DANGEROUS_PATTERNS = [
    re.compile(r"UTL_|DBMS_|SYS\.", re.IGNORECASE),
]


class OracleConnector(DatabaseConnector):
    """Oracle dialect. Catalog-driven schema/table logic lives in the base class."""

    def __init__(self, llm=None):
        super().__init__(llm=llm, catalog_path=CATALOG_PATH)
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

    def _validate_sql(self, sql: str) -> str:
        """Validate SQL safety — base checks + Oracle-specific dangerous packages."""
        error = self.validate_sql(sql)
        if error:
            return error
        for pattern in _ORACLE_DANGEROUS_PATTERNS:
            match = pattern.search(sql)
            if match:
                return f"Forbidden Oracle pattern detected: {match.group()}"
        return ""

    def execute_query(self, sql: str) -> str:
        error = self._validate_sql(sql)
        if error:
            logger.warning(
                f"SQL rejected: {error}",
                extra={"node": "sql_path", "sql": sql, "error": error},
            )
            return f"Error: {error}"

        sql = self.apply_row_limit(sql)

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

    def get_row_limit_clause(self, limit: int = 500) -> str:
        return f"FETCH FIRST {limit} ROWS ONLY"

    def get_readiness_query(self) -> str:
        return "SELECT 1 FROM DUAL"
