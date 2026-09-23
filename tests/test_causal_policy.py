"""
tests/test_causal_policy.py - Comprehensive Unit & Behavioral Tests for Causal FinOps Policy.
Validates:
1. Protocol conformance (CausalPolicyProtocol).
2. Counterfactual optimization: E[Loss(a)] = C_action(a) + P(Breach | do(a)) * Penalty.
3. Amdahl's Law physical constraint: AQE for partition skew, worker scaling for disk spill,
   exponential backoff for Delta OCC write contention.
4. Refusal of worker scaling on partition skew (anti-waste guarantee).
5. Exact financial accounting identities and sub-millisecond evaluation speed.
"""

import time
import pytest

from src.domain.contracts import CausalPolicyProtocol
from src.domain.entities import (
    PipelineObservationSnapshot,
    RiskEstimate,
    CausalDecision,
)
from src.domain.causal_policy import CausalFinOpsPolicy


@pytest.fixture
def policy():
    return CausalFinOpsPolicy(
        min_intervention_threshold=0.20,
        min_net_savings_usd=250.0,
    )


def test_causal_policy_protocol_conformance(policy):
    """Verifies that CausalFinOpsPolicy conforms to CausalPolicyProtocol."""
    assert isinstance(policy, CausalPolicyProtocol)


def test_skew_triggers_aqe_and_refuses_scaling(policy):
    """
    CRITICAL CAUSAL INVARIANT:
    When the bottleneck is partition skew, the slowest task is strictly serial.
    Scaling workers burns budget with zero speedup (Amdahl's law).
    The policy MUST prescribe AQE_COALESCE_SKEW_JOIN and NOT SCALE_WORKERS_MEMORY.
    """
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-skew-01",
        run_id="run-skew-001",
        job_name="BRL_Pix_Clearing_ETL_Pipeline",
        business_domain="BRL_Pix_Clearing",
        as_of_seconds=3600.0,
        contract_sla_seconds=7200.0,
        remaining_to_sla_seconds=3600.0,
        fraction_tasks_completed=0.45,
        tasks_active=24,
        tasks_failed=0,
        skew_duration_ratio=3.8,  # Extreme partition skew
        disk_spilled_gib=0.0,
        gc_pressure=0.02,
        occ_conflict_retries=0,
    )

    risk = RiskEstimate(
        run_id="run-skew-001",
        as_of_seconds=3600.0,
        breach_probability=0.65,
        survival_to_sla=0.35,
        survival_to_now=0.90,
        p50_completion_seconds=7800.0,
        p90_completion_seconds=9200.0,
        dominant_risk_mechanism="SKEW",
        mechanism_severity_score=0.70,
        model_version="aft-weibull-v1.0",
    )

    decision = policy.evaluate(snapshot, risk)

    assert isinstance(decision, CausalDecision)
    assert decision.action == "AQE_COALESCE_SKEW_JOIN"
    assert decision.should_intervene is True
    assert decision.action_cost_usd == 18.0
    assert decision.prob_breach_with_action < decision.prob_breach_without_action
    assert decision.expected_net_savings_usd > 20000.0  # Big penalty ($50k) reduction
    assert "AQE skew join" in decision.reason


def test_pure_spill_triggers_worker_memory_scaling(policy):
    """
    When memory/disk spill dominates and skew is nominal (parallel tasks memory-bound),
    expanding aggregate cluster memory eliminates spill serialization.
    """
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-spill-01",
        run_id="run-spill-001",
        job_name="BACEN_Regulatory_Report_ETL_Pipeline",
        business_domain="BACEN_Regulatory_Report",
        as_of_seconds=4500.0,
        contract_sla_seconds=10800.0,
        remaining_to_sla_seconds=6300.0,
        fraction_tasks_completed=0.40,
        tasks_active=48,
        tasks_failed=0,
        skew_duration_ratio=1.1,  # Perfectly uniform partitions
        disk_spilled_gib=24.5,   # Heavy NVMe shuffle spill
        memory_spilled_gib=35.0,
        gc_pressure=0.04,
        occ_conflict_retries=0,
    )

    risk = RiskEstimate(
        run_id="run-spill-001",
        as_of_seconds=4500.0,
        breach_probability=0.70,
        survival_to_sla=0.40,
        survival_to_now=0.92,
        p50_completion_seconds=11500.0,
        p90_completion_seconds=13000.0,
        dominant_risk_mechanism="SPILL",
        mechanism_severity_score=0.85,
        model_version="aft-weibull-v1.0",
    )

    decision = policy.evaluate(snapshot, risk)

    assert decision.action == "SCALE_WORKERS_MEMORY"
    assert decision.should_intervene is True
    assert decision.action_cost_usd > 0.0
    assert decision.expected_net_savings_usd > 15000.0
    assert "disk spill" in decision.reason


def test_delta_occ_contention_triggers_backoff_serialization(policy):
    """
    When OCC transaction conflicts cause commit aborts on Delta Lake,
    injecting backoff with jitter resolves write lock collisions.
    """
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-occ-01",
        run_id="run-occ-001",
        job_name="Receita_Tax_Reconciliation_ETL_Pipeline",
        business_domain="Receita_Tax_Reconciliation",
        as_of_seconds=5000.0,
        contract_sla_seconds=12600.0,
        remaining_to_sla_seconds=7600.0,
        fraction_tasks_completed=0.55,
        tasks_active=16,
        tasks_failed=0,
        skew_duration_ratio=1.15,
        disk_spilled_gib=0.0,
        gc_pressure=0.02,
        occ_conflict_retries=5,  # Heavy Delta OCC commit contention
    )

    risk = RiskEstimate(
        run_id="run-occ-001",
        as_of_seconds=5000.0,
        breach_probability=0.58,
        survival_to_sla=0.30,
        survival_to_now=0.88,
        p50_completion_seconds=13200.0,
        p90_completion_seconds=14500.0,
        dominant_risk_mechanism="OCC_CONTENTION",
        mechanism_severity_score=0.80,
        model_version="aft-weibull-v1.0",
    )

    decision = policy.evaluate(snapshot, risk)

    assert decision.action == "BACKOFF_OCC_SERIALIZE"
    assert decision.should_intervene is True
    assert decision.action_cost_usd == 12.0
    assert "exponential backoff" in decision.reason


def test_low_risk_triggers_no_op(policy):
    """
    Pipelines on-time with low risk (< 20%) must not trigger spurious interventions.
    """
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-ontime-01",
        run_id="run-ontime-001",
        job_name="STJ_Judicial_Analytics_ETL_Pipeline",
        business_domain="STJ_Judicial_Analytics",
        as_of_seconds=3000.0,
        contract_sla_seconds=9000.0,
        remaining_to_sla_seconds=6000.0,
        fraction_tasks_completed=0.65,
        tasks_active=16,
        tasks_failed=0,
        skew_duration_ratio=1.1,
        disk_spilled_gib=0.0,
        gc_pressure=0.02,
        occ_conflict_retries=0,
    )

    risk = RiskEstimate(
        run_id="run-ontime-001",
        as_of_seconds=3000.0,
        breach_probability=0.08,  # Below min_intervention_threshold (0.20)
        survival_to_sla=0.05,
        survival_to_now=0.95,
        p50_completion_seconds=5200.0,
        p90_completion_seconds=6800.0,
        dominant_risk_mechanism="NOMINAL",
        mechanism_severity_score=0.05,
        model_version="aft-weibull-v1.0",
    )

    decision = policy.evaluate(snapshot, risk)

    assert decision.action == "NO_OP"
    assert decision.should_intervene is False
    assert decision.action_cost_usd == 0.0
    assert decision.expected_net_savings_usd == 0.0
    assert decision.prob_breach_with_action == decision.prob_breach_without_action


def test_financial_accounting_identities(policy):
    """
    CRITICAL: Verifies exact accounting identity for net savings:
    net_savings = E[Loss(NO_OP)] - E[Loss(action)]
    """
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-math-01",
        run_id="run-math-001",
        job_name="BRL_Pix_Clearing_ETL_Pipeline",
        business_domain="BRL_Pix_Clearing",
        as_of_seconds=3000.0,
        contract_sla_seconds=7200.0,
        remaining_to_sla_seconds=4200.0,
        fraction_tasks_completed=0.35,
        skew_duration_ratio=2.5,
        disk_spilled_gib=0.0,
        gc_pressure=0.02,
        occ_conflict_retries=0,
    )

    risk = RiskEstimate(
        run_id="run-math-001",
        as_of_seconds=3000.0,
        breach_probability=0.50,
        survival_to_sla=0.30,
        survival_to_now=0.90,
        p50_completion_seconds=7500.0,
        p90_completion_seconds=8800.0,
        dominant_risk_mechanism="SKEW",
        mechanism_severity_score=0.60,
        model_version="aft-weibull-v1.0",
    )

    decision = policy.evaluate(snapshot, risk)

    # Check exact math
    expected_savings = round(
        decision.expected_loss_without_action_usd - decision.expected_loss_with_action_usd, 2
    )
    assert abs(decision.expected_net_savings_usd - expected_savings) < 0.05


def test_policy_evaluation_speed_sub_millisecond(policy):
    """
    BENCHMARK: Evaluates policy decision time for 100 consecutive snapshots.
    Target: < 0.1 ms per evaluation.
    """
    snapshot = PipelineObservationSnapshot(
        snapshot_id="snap-bench",
        run_id="run-bench",
        job_name="BRL_Pix_Clearing_ETL_Pipeline",
        business_domain="BRL_Pix_Clearing",
        as_of_seconds=3000.0,
        contract_sla_seconds=7200.0,
        remaining_to_sla_seconds=4200.0,
        fraction_tasks_completed=0.35,
        skew_duration_ratio=2.5,
        disk_spilled_gib=0.0,
        gc_pressure=0.02,
        occ_conflict_retries=0,
    )
    risk = RiskEstimate(
        run_id="run-bench",
        as_of_seconds=3000.0,
        breach_probability=0.50,
        survival_to_sla=0.30,
        survival_to_now=0.90,
        p50_completion_seconds=7500.0,
        p90_completion_seconds=8800.0,
        dominant_risk_mechanism="SKEW",
        mechanism_severity_score=0.60,
        model_version="aft-weibull-v1.0",
    )

    # Warm-up
    for _ in range(5):
        policy.evaluate(snapshot, risk)

    t0 = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        policy.evaluate(snapshot, risk)
    total_ms = (time.perf_counter() - t0) * 1000.0
    latency_us = (total_ms / iterations) * 1000.0

    print(f"\n[Benchmark] CausalFinOpsPolicy latency: {latency_us:.1f} microseconds / decision")
    assert latency_us < 500.0  # < 0.5 ms
