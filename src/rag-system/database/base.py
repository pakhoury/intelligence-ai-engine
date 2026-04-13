from abc import ABC, abstractmethod
from typing import List


class DatabaseConnector(ABC):

    @abstractmethod
    def get_relevant_tables(self, keyword: str) -> List:
        pass

    @abstractmethod
    def get_table_schema(self, table_names: List) -> str:
        pass

    @abstractmethod
    def execute_query(self, sql: str) -> str:
        pass

    @abstractmethod
    def get_dialect(self) -> str:
        pass