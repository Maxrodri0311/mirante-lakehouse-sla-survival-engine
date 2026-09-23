"""
tests/test_medallion_marts.py - Unit & Integration Tests for Medallion Gold Data Marts.
Validates:
1. Dimensional aggregation over Silver Parquet partitions in DuckDB.
2. Executive Risk Mart schema, metric bounds, and bottleneck diagnosis.
3. FinOps Remediation Mart cost-benefit ROI calculation and financial accounting.
4. Export to physical Gold Parquet layer.
"""

import os
import pytest
import pandas as pd

from src.adapters.duckdb_adapter import DuckDBStorageAdapter
from src.marts.medallion_marts import MedallionMartBuilder
from src.domain.entities import CausalDecision
from src.data_generator import generate_lakehouse_telemetry


@pytest.fixture(scope="module")
def lakehouse_with_marts(tmp_path_factory):
    base_dir = str(tmp_path_factory.mktemp("marts_lakehouse"))
    generate_lakehouse_telemetry(num_runs=250, base_output_dir=base_dir, seed=777)
    storage = DuckDBStorageAdapter(database=":memory:")
    builder = MedallionMartBuilder(storage=storage, base_lakehouse_dir=base_dir)
    yield base_dir, storage, builder
    storage.close()


def test_executive_risk_mart_generation(lakehouse_with_marts):
    """Verifies mart_sla_executive_risk aggregation across all domains."""
    base_dir, storage, builder = lakehouse_with_marts

    df_exec = builder.build_executive_risk_mart(export_parquet=True)

    assert isinstance(df_exec, pd.DataFrame)
    assert len(df_exec) >= 4  # 4 business domains
    assert "business_domain" in df_exec.columns
    assert "total_pipeline_runs" in df_exec.columns
    assert "total_breaches" in df_exec.columns
    assert "observed_breach_rate_pct" in df_exec.columns
    assert "total_penalty_liability_usd" in df_exec.columns
    assert "primary_operational_bottleneck" in df_exec.columns

    # Check bounds
    assert df_exec["total_pipeline_runs"].sum() == 250
    assert (df_exec["observed_breach_rate_pct"] >= 0.0).all()
    assert (df_exec["observed_breach_rate_pct"] <= 100.0).all()
    assert (df_exec["total_penalty_liability_usd"] >= 0.0).all()

    # Verify DuckDB view registration
    assert storage.table_exists("mart_sla_executive_risk")

    # Verify physical Gold Parquet export
    gold_file = os.path.join(base_dir, "gold", "mart_sla_executive_risk.parquet")
    assert os.path.exists(gold_file)


def test_finops_remediation_mart_generation(lakehouse_with_marts):
    """Verifies mart_finops_remediation calculation and ROI ratios."""
    base_dir, storage, builder = lakehouse_with_marts

    sample_decisions = [
        CausalDecision(
            decision_id="d1",
            run_id="r1",
            as_of_seconds=2400.0,
            action="AQE_COALESCE_SKEW_JOIN",
            reason="Skew mitigation",
            prob_breach_without_action=0.60,
            prob_breach_with_action=0.10,
            action_cost_usd=18.0,
            contract_penalty_usd=50000.0,
            expected_loss_without_action_usd=30000.0,
            expected_loss_with_action_usd=5018.0,
            expected_net_savings_usd=24982.0,
            should_intervene=True,
        ),
        CausalDecision(
            decision_id="d2",
            run_id="r2",
            as_of_seconds=3000.0,
            action="SCALE_WORKERS_MEMORY",
            reason="Disk spill remediation",
            prob_breach_without_action=0.70,
            prob_breach_with_action=0.20,
            action_cost_usd=72.0,
            contract_penalty_usd=40000.0,
            expected_loss_without_action_usd=28000.0,
            expected_loss_with_action_usd=8072.0,
            expected_net_savings_usd=19928.0,
            should_intervene=True,
        ),
        CausalDecision(
            decision_id="d3",
            run_id="r3",
            as_of_seconds=1800.0,
            action="NO_OP",
            reason="Low risk nominal",
            prob_breach_without_action=0.05,
            prob_breach_with_action=0.05,
            action_cost_usd=0.0,
            contract_penalty_usd=25000.0,
            expected_loss_without_action_usd=1250.0,
            expected_loss_with_action_usd=1250.0,
            expected_net_savings_usd=0.0,
            should_intervene=False,
        ),
    ]

    df_finops = builder.build_finops_remediation_mart(sample_decisions, export_parquet=True)

    assert isinstance(df_finops, pd.DataFrame)
    assert len(df_finops) == 3
    assert "action" in df_finops.columns
    assert "total_action_cost_usd" in df_finops.columns
    assert "net_financial_savings_usd" in df_finops.columns
    assert "mean_roi_ratio" in df_finops.columns

    # Check that interventions show high ROI
    aqe_row = df_finops[df_finops["action"] == "AQE_COALESCE_SKEW_JOIN"].iloc[0]
    assert aqe_row["mean_roi_ratio"] > 100.0  # Massive risk reduction for $18 overhead
    assert aqe_row["net_financial_savings_usd"] > 20000.0

    # Verify physical Gold Parquet export
    gold_file = os.path.join(base_dir, "gold", "mart_finops_remediation.parquet")
    assert os.path.exists(gold_file)
