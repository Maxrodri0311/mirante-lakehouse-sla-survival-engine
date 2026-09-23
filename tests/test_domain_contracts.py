"""
tests/test_domain_contracts.py - Unit Tests for Pure Domain Entities and Protocols.
Validates Pydantic schema validation, domain entity bounds, and Dependency Inversion Principle (DIP).
Guarantees sub-5ms isolated testing without touching disk, network, or external databases.
"""

import pytest
from pydantic import ValidationError
import pandas as pd
from src.domain.entities import (
    PipelineRun,
    PipelineObservationSnapshot,
    RiskEstimate,
    CausalDecision,
)
from src.domain.contracts import AnalyticalStorageProtocol


def test_pipeline_run_entity_instantiation():
    """Verifies that PipelineRun validates required fields and types."""
    run = PipelineRun(
        run_id="run-000001",
        job_id="job-pix-001",
        job_name="BRL_Pix_Clearing_ETL",
        business_domain="BRL_Pix_Clearing",
        cluster_id="cluster-mem-01",
        contract_sla_sec=7200.0,
        scheduled_start_ts="2026-01-01T00:00:00",
        actual_start_ts="2026-01-01T00:00:00",
        observed_duration_sec=5400.0,
        final_status="SUCCESS_ON_TIME",
        event_breach=False,
        is_censored=False,
        total_compute_cost_usd=54.0,
        penalty_fee_usd=50000.0,
    )
    assert run.run_id == "run-000001"
    assert run.final_status == "SUCCESS_ON_TIME"
    assert not run.event_breach
    assert run.observed_duration_sec < run.contract_sla_sec


def test_pipeline_run_validation_error_on_invalid_status():
    """Ensures Pydantic rejects invalid final_status literals."""
    with pytest.raises(ValidationError):
        PipelineRun(
            run_id="run-000002",
            job_id="job-pix-002",
            job_name="BRL_Pix_Clearing_ETL",
            business_domain="BRL_Pix_Clearing",
            cluster_id="cluster-mem-01",
            contract_sla_sec=7200.0,
            scheduled_start_ts="2026-01-01T00:00:00",
            actual_start_ts="2026-01-01T00:00:00",
            observed_duration_sec=7500.0,
            final_status="UNKNOWN_STATUS",  # Invalid
            event_breach=True,
            is_censored=False,
        )


def test_pipeline_observation_snapshot_bounds():
    """Verifies valid bounds on snapshot telemetries."""
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-000001-001",
        run_id="run-000001",
        job_name="BRL_Pix_Clearing_ETL",
        business_domain="BRL_Pix_Clearing",
        as_of_seconds=1800.0,
        contract_sla_seconds=7200.0,
        remaining_to_sla_seconds=5400.0,
        fraction_tasks_completed=0.25,
        tasks_active=32,
        shuffle_read_gib=120.5,
        memory_spilled_gib=15.0,
        disk_spilled_gib=8.5,
        skew_duration_ratio=2.1,
        tail_amplification=2.8,
        gc_pressure=0.04,
        occ_conflict_retries=1,
    )
    assert snapshot.as_of_seconds == 1800.0
    assert snapshot.remaining_to_sla_seconds == 5400.0
    assert 0.0 <= snapshot.fraction_tasks_completed <= 1.0


def test_risk_estimate_and_causal_decision_schemas():
    """Verifies RiskEstimate and CausalDecision schema integrity."""
    risk = RiskEstimate(
        run_id="run-000001",
        as_of_seconds=3600.0,
        breach_probability=0.35,
        survival_to_sla=0.65,
        survival_to_now=0.98,
        p50_completion_seconds=5800.0,
        p90_completion_seconds=7800.0,
        dominant_risk_mechanism="SKEW",
        mechanism_severity_score=0.72,
    )
    assert risk.breach_probability == 0.35
    assert risk.dominant_risk_mechanism == "SKEW"

    decision = CausalDecision(
        decision_id="dec-000001-01",
        run_id="run-000001",
        as_of_seconds=3600.0,
        action="AQE_COALESCE_SKEW_JOIN",
        reason="Severe partition skew detected (R_skew=3.8). Triggering AQE.",
        prob_breach_without_action=0.35,
        prob_breach_with_action=0.08,
        action_cost_usd=12.0,
        contract_penalty_usd=50000.0,
        expected_loss_without_action_usd=0.35 * 50000.0,
        expected_loss_with_action_usd=12.0 + (0.08 * 50000.0),
        expected_net_savings_usd=(0.35 * 50000.0) - (12.0 + (0.08 * 50000.0)),
        should_intervene=True,
    )
    assert decision.action == "AQE_COALESCE_SKEW_JOIN"
    assert decision.should_intervene
    assert decision.expected_net_savings_usd > 0.0


def test_dependency_inversion_in_memory_mock():
    """
    Validates that domain logic can interact with an in-memory mock storage adapter
    satisfying AnalyticalStorageProtocol without importing duckdb or touching disk.
    """
    class InMemoryMockStorage:
        def __init__(self):
            self.views = {}

        def execute_query(self, query: str) -> pd.DataFrame:
            return pd.DataFrame([{"total_runs": 100, "breach_rate": 0.22}])

        def table_exists(self, table_name: str) -> bool:
            return table_name in self.views

        def register_view(self, view_name: str, query_or_df) -> None:
            self.views[view_name] = query_or_df

    mock_storage = InMemoryMockStorage()
    assert isinstance(mock_storage, AnalyticalStorageProtocol)

    mock_storage.register_view("v_runs", pd.DataFrame())
    assert mock_storage.table_exists("v_runs")

    df = mock_storage.execute_query("SELECT * FROM v_runs")
    assert len(df) == 1
    assert df["total_runs"].iloc[0] == 100
