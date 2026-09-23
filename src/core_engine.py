"""
src/core_engine.py - Core Analytical & Algorithmic Engine for mirante_tecnologia_data_scientist_bridge_project.
Implements Dependency Inversion Principle (DIP) over AnalyticalStorageProtocol.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import duckdb
import pandas as pd

# Path resolution for standalone script invocation
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.contracts import AnalyticalStorageProtocol, ExecutionContext


class DuckDBStorageAdapter:
    """Concrete infrastructure adapter for in-memory DuckDB OLAP."""
    def __init__(self, database: str = ":memory:"):
        self.conn = duckdb.connect(database)

    def execute_query(self, query: str) -> pd.DataFrame:
        return self.conn.execute(query).df()

    def scan_dataset(self, base_path: str) -> pd.DataFrame:
        return self.conn.execute(f"SELECT * FROM read_parquet('{base_path}');").df()


class DomainAnalyticsEngine:
    """
    Decoupled analytical engine.
    Depends strictly on AnalyticalStorageProtocol abstraction (Anti-Buried Dependencies).
    """
    def __init__(
        self,
        storage: AnalyticalStorageProtocol,
        data_path: str = "data/raw_dataset.parquet",
    ):
        self.storage = storage
        self.data_path = data_path

    def execute_analysis(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        numeric_vars = [
            v.name for v in [
                {"name": "primary_metric", "type": "continuous"},
                {"name": "volume_count", "type": "discrete"},
                {"name": "is_active", "type": "boolean"},
                {"name": "status_category", "type": "categorical"},
            ] if v["type"] in ["continuous", "discrete"]
        ]
        
        first_num = numeric_vars[0] if numeric_vars else "1"

        query = f"""
            SELECT 
                COUNT(*) as total_records,
                ROUND(AVG({first_num}), 4) as mean_primary_metric,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {first_num}), 4) as p95_metric,
                ROUND(MIN({first_num}), 4) as min_metric,
                ROUND(MAX({first_num}), 4) as max_metric
            FROM read_parquet('{self.data_path}');
        """
        return self.storage.execute_query(query)


def create_engine(data_path: str = "data/raw_dataset.parquet") -> DomainAnalyticsEngine:
    """Composition Root."""
    adapter = DuckDBStorageAdapter()
    return DomainAnalyticsEngine(storage=adapter, data_path=data_path)


if __name__ == "__main__":
    from src.data_generator import generate_domain_dataset

    path = "data/raw_dataset.parquet"
    if not os.path.exists(path):
        print(f"[Core Engine] Bootstrapping dataset at {path}...")
        generate_domain_dataset(num_records=10000, output_path=path)

    engine = create_engine(data_path=path)
    res = engine.execute_analysis()
    print("\n" + "="*70)
    print("  MIRANTE_TECNOLOGIA_DATA_SCIENTIST_BRIDGE_PROJECT - ANALYTICAL EXECUTION (DIP)")
    print("="*70)
    print(res.to_string(index=False))
    print("="*70 + "\n")