"""
tests/test_data_generator.py - Unit & Integration Tests for Calibrated Telemetry Generator.
Verifies dataset completeness, Hive-partitioned directory creation, AFT physics,
and strict anti-leakage invariants across time-varying snapshots.
"""

import os
import pytest
import pandas as pd
from src.data_generator import generate_lakehouse_telemetry, DOMAIN_SPECS


@pytest.fixture(scope="module")
def sample_lakehouse(tmp_path_factory):
    base_dir = str(tmp_path_factory.mktemp("lakehouse_fixture"))
    df_runs, df_snaps = generate_lakehouse_telemetry(
        num_runs=300,
        base_output_dir=base_dir,
        seed=123,
    )
    return base_dir, df_runs, df_snaps


def test_lakehouse_generation_completeness(sample_lakehouse):
    """Verifies that runs and snapshot counts meet mathematical expectations."""
    _, df_runs, df_snaps = sample_lakehouse

    assert len(df_runs) == 300
    assert len(df_snaps) >= 4000  # Multiple snapshots per run
    assert df_runs["run_id"].nunique() == 300
    assert set(df_runs["business_domain"].unique()) == set(DOMAIN_SPECS.keys())


def test_zero_label_leakage_invariant(sample_lakehouse):
    """
    CRITICAL INVARIANT: Strict anti-leakage test.
    An observation snapshot at as_of_seconds t_c must NEVER exceed the final observed duration!
    """
    _, df_runs, df_snaps = sample_lakehouse

    # Join snapshots with runs to compare timestamps
    merged = df_snaps.merge(df_runs[["run_id", "observed_duration_sec"]], on="run_id")
    leakage = merged[merged["as_of_seconds"] >= merged["observed_duration_sec"]]

    assert len(leakage) == 0, (
        f"Found {len(leakage)} snapshots with as_of_seconds >= observed_duration_sec! "
        "Label leakage detected."
    )


def test_physical_spark_telemetry_distributions(sample_lakehouse):
    """Verifies that shuffle spill, skew, GC, and OCC contention exhibit non-trivial variance."""
    _, _, df_snaps = sample_lakehouse

    # Skew ratio must show normal vs extreme tails
    assert df_snaps["skew_duration_ratio"].min() >= 1.0
    assert df_snaps["skew_duration_ratio"].max() > 2.5

    # Memory and disk spill must be positive in high-pressure runs
    assert (df_snaps["disk_spilled_gib"] > 0).sum() > 0

    # GC pressure must be within physically realistic bounds [0.0, 1.0]
    assert 0.0 <= df_snaps["gc_pressure"].mean() <= 0.35

    # Fraction completed strictly in [0, 1)
    assert df_snaps["fraction_tasks_completed"].min() >= 0.0
    assert df_snaps["fraction_tasks_completed"].max() <= 1.0


def test_hive_partitioned_parquet_directory_structure(sample_lakehouse):
    """Verifies physical Hive partition directory layout on disk."""
    base_dir, _, _ = sample_lakehouse

    runs_dir = os.path.join(base_dir, "silver", "fact_pipeline_execution")
    snaps_dir = os.path.join(base_dir, "silver", "fact_pipeline_observation")

    assert os.path.exists(runs_dir)
    assert os.path.exists(snaps_dir)

    # Check for domain partitions in fact_pipeline_execution
    subdirs = os.listdir(runs_dir)
    domain_subdirs = [d for d in subdirs if d.startswith("business_domain=")]
    assert len(domain_subdirs) == 4
