"""
src/adapters/duckdb_adapter.py - In-Memory & Parquet Lakehouse DuckDB Adapter.
Implements AnalyticalStorageProtocol via vectorized DuckDB execution,
enabling sub-second OLAP queries across partitioned Silver/Gold layers.
"""

from typing import Any, Optional
import duckdb
import pandas as pd
from src.domain.contracts import AnalyticalStorageProtocol


class DuckDBStorageAdapter(AnalyticalStorageProtocol):
    """
    Concrete analytical storage adapter implementing AnalyticalStorageProtocol.
    Provides local-first vectorized execution over Hive-partitioned Parquet datasets.
    """

    def __init__(self, database: str = ":memory:"):
        self.conn = duckdb.connect(database)
        # Enable multi-threaded execution
        self.conn.execute("SET threads = 4;")
        self.conn.execute("SET preserve_insertion_order = false;")

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Executes an analytical SQL statement and returns the resulting DataFrame.
        """
        return self.conn.execute(query).df()

    def table_exists(self, table_name: str) -> bool:
        """
        Checks catalog metadata to verify if a view or table exists.
        """
        res = self.conn.execute(
            f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}';"
        ).fetchall()
        return len(res) > 0

    def register_view(self, view_name: str, query_or_df: Any) -> None:
        """
        Registers a virtual view either from a SQL string or an in-memory DataFrame.
        """
        if isinstance(query_or_df, str):
            self.conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {query_or_df};")
        elif isinstance(query_or_df, pd.DataFrame):
            self.conn.register(view_name, query_or_df)
        else:
            raise TypeError(f"Unsupported type for view registration: {type(query_or_df)}")

    def close(self) -> None:
        """Closes the underlying DuckDB connection."""
        self.conn.close()
