"""
tests/test_duckdb_adapter.py - Unit & Integration Tests for DuckDB Storage Adapter.
Verifies SQL query execution, view registration, catalog inspection, and Parquet scanning.
"""

import os
import pytest
import pandas as pd
from src.adapters.duckdb_adapter import DuckDBStorageAdapter
from src.domain.contracts import AnalyticalStorageProtocol


@pytest.fixture
def duckdb_adapter():
    adapter = DuckDBStorageAdapter(database=":memory:")
    yield adapter
    adapter.close()


def test_duckdb_adapter_protocol_conformance(duckdb_adapter):
    """Ensures DuckDBStorageAdapter strictly conforms to AnalyticalStorageProtocol."""
    assert isinstance(duckdb_adapter, AnalyticalStorageProtocol)


def test_duckdb_basic_query_and_aggregation(duckdb_adapter):
    """Validates arbitrary vectorized mathematical execution."""
    df = duckdb_adapter.execute_query("""
        SELECT 
            42 as answer,
            SQRT(144) as twelve,
            LN(EXP(1.0)) as e_one
    """)
    assert len(df) == 1
    assert df["answer"].iloc[0] == 42
    assert df["twelve"].iloc[0] == 12.0
    assert abs(df["e_one"].iloc[0] - 1.0) < 1e-6


def test_duckdb_view_registration_and_table_exists(duckdb_adapter):
    """Validates virtual view registration from SQL and catalog existence checks."""
    assert not duckdb_adapter.table_exists("v_sample_metrics")

    duckdb_adapter.register_view(
        "v_sample_metrics",
        "SELECT 'run-101' as run_id, 3200.0 as duration_sec, false as breach"
    )

    assert duckdb_adapter.table_exists("v_sample_metrics")
    df = duckdb_adapter.execute_query("SELECT * FROM v_sample_metrics")
    assert len(df) == 1
    assert df["run_id"].iloc[0] == "run-101"
    assert df["duration_sec"].iloc[0] == 3200.0


def test_duckdb_parquet_scan_with_predicate_pushdown(tmp_path, duckdb_adapter):
    """Verifies reading local Parquet datasets with SQL predicate filters."""
    p_file = tmp_path / "sample_runs.parquet"
    test_df = pd.DataFrame({
        "run_id": [f"run-{i:03d}" for i in range(100)],
        "duration_sec": [1000.0 + i * 50.0 for i in range(100)],
        "domain": ["Pix" if i % 2 == 0 else "Bacen" for i in range(100)],
    })
    test_df.to_parquet(str(p_file), index=False)

    parquet_path = str(p_file).replace("\\", "/")
    res = duckdb_adapter.execute_query(f"""
        SELECT domain, COUNT(*) as count, AVG(duration_sec) as avg_dur
        FROM read_parquet('{parquet_path}')
        WHERE domain = 'Pix'
        GROUP BY domain;
    """)

    assert len(res) == 1
    assert res["domain"].iloc[0] == "Pix"
    assert res["count"].iloc[0] == 50
