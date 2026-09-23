"""
tests/test_suite.py - End-to-End Integration Test Suite for Fase 1.
Validates the end-to-end pipeline: telemetry generation -> physical Parquet persistence
-> vectorized DuckDB query scan -> DIP domain entities.
"""

import os
import pytest
import pandas as pd
from src.data_generator import generate_lakehouse_telemetry
from src.adapters.duckdb_adapter import DuckDBStorageAdapter
from src.domain.contracts import AnalyticalStorageProtocol


@pytest.fixture(scope="module")
def e2e_lakehouse(tmp_path_factory):
    base_dir = str(tmp_path_factory.mktemp("e2e_lakehouse"))
    df_runs, df_snaps = generate_lakehouse_telemetry(
        num_runs=200,
        base_output_dir=base_dir,
        seed=999,
    )
    adapter = DuckDBStorageAdapter(database=":memory:")
    yield base_dir, adapter
    adapter.close()


def test_end_to_end_lakehouse_query_integration(e2e_lakehouse):
    """
    Verifies that DuckDB can query the Hive-partitioned Parquet directory directly
    and aggregate metrics across business domains.
    """
    base_dir, adapter = e2e_lakehouse
    runs_glob = os.path.join(base_dir, "silver", "fact_pipeline_execution", "**", "*.parquet").replace("\\", "/")

    query = f"""
        SELECT 
            business_domain,
            COUNT(*) as total_runs,
            AVG(observed_duration_sec) as mean_duration,
            SUM(CASE WHEN event_breach THEN 1 ELSE 0 END) as breaches,
            ROUND(AVG(CASE WHEN event_breach THEN 1.0 ELSE 0.0 END) * 100.0, 1) as breach_rate_pct
        FROM read_parquet('{runs_glob}', hive_partitioning = true)
        GROUP BY business_domain
        ORDER BY total_runs DESC;
    """
    df = adapter.execute_query(query)

    assert len(df) == 4
    assert "business_domain" in df.columns
    assert "breach_rate_pct" in df.columns
    assert df["total_runs"].sum() == 200


def test_snapshot_telemetry_scan_integration(e2e_lakehouse):
    """
    Verifies vectorized scan over observation snapshots for high-pressure runs.
    """
    base_dir, adapter = e2e_lakehouse
    snaps_glob = os.path.join(base_dir, "silver", "fact_pipeline_observation", "**", "*.parquet").replace("\\", "/")

    query = f"""
        SELECT 
            business_domain,
            COUNT(*) as total_snapshots,
            ROUND(AVG(skew_duration_ratio), 2) as mean_skew,
            ROUND(AVG(gc_pressure), 3) as mean_gc_pressure,
            ROUND(SUM(disk_spilled_gib), 1) as total_disk_spill_gib
        FROM read_parquet('{snaps_glob}', hive_partitioning = true)
        GROUP BY business_domain;
    """
    df = adapter.execute_query(query)

    assert len(df) == 4
    assert df["total_snapshots"].sum() > 2000
    assert df["mean_skew"].min() > 1.0