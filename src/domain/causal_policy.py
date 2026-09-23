"""
src/domain/causal_policy.py - Causal FinOps Decision & Remediation Policy for Spark Lakehouses.
Evaluates prescriptive counterfactual interventions:
    E[Loss(a | H_t)] = C_action(a) + P(Breach | do(a), H_t) * ContractPenalty
Selects:
    a* = argmin_{a in A} E[Loss(a | H_t)]
Enforces strict Amdahl's Law physical constraints:
- Refuses worker scaling on partition skew (straggler task is serial).
- Refuses interventions when action cost exceeds risk reduction value.
- Prescribes AQE skew joins for skew, worker scaling for memory spill, and backoff for Delta OCC.
"""

from typing import Dict, List, Optional, Tuple, Any
import uuid
import numpy as np

from src.domain.entities import (
    PipelineObservationSnapshot,
    RiskEstimate,
    CausalDecision,
    InterventionActionType,
    RiskMechanismType,
)
from src.domain.contracts import CausalPolicyProtocol


# Standard domain SLA penalty mapping (mirrors Brazilian enterprise contracts)
DEFAULT_DOMAIN_PENALTIES: Dict[str, float] = {
    "BRL_Pix_Clearing": 50000.0,
    "BACEN_Regulatory_Report": 40000.0,
    "STJ_Judicial_Analytics": 25000.0,
    "Receita_Tax_Reconciliation": 35000.0,
}


class CausalFinOpsPolicy:
    """
    Prescriptive FinOps remediation engine implementing CausalPolicyProtocol.
    Optimizes operational interventions to minimize expected enterprise financial loss.
    """

    def __init__(
        self,
        domain_penalties: Optional[Dict[str, float]] = None,
        min_intervention_threshold: float = 0.20,
        min_net_savings_usd: float = 250.0,
    ):
        self.penalties = domain_penalties or DEFAULT_DOMAIN_PENALTIES
        self.min_intervention_threshold = min_intervention_threshold
        self.min_net_savings_usd = min_net_savings_usd

    def _get_contract_penalty(self, domain: str) -> float:
        return self.penalties.get(domain, 30000.0)

    def _estimate_counterfactual_outcomes(
        self,
        snapshot: PipelineObservationSnapshot,
        risk: RiskEstimate,
        penalty: float,
    ) -> List[Tuple[InterventionActionType, float, float, str]]:
        """
        Estimates (action, action_cost_usd, prob_breach_with_action, reason)
        for all candidate counterfactual interventions do(a).
        """
        candidates: List[Tuple[InterventionActionType, float, float, str]] = []
        p_base = risk.breach_probability
        remaining_hours = max(0.1, snapshot.remaining_to_sla_seconds / 3600.0)
        burn_rate = snapshot.hourly_cluster_cost_usd

        # 1. Action: NO_OP (Baseline)
        candidates.append((
            "NO_OP",
            0.0,
            p_base,
            "Continue nominal execution without operational intervention.",
        ))

        # 2. Action: AQE_COALESCE_SKEW_JOIN
        # Highly effective for partition skew; minimal overhead
        if snapshot.skew_duration_ratio >= 1.5 or risk.dominant_risk_mechanism == "SKEW":
            aqe_cost = 18.0  # Minor AQE dynamic shuffle recalculation overhead
            # Skew resolution reduces probability of breach by ~75%
            aqe_p = float(np.clip(p_base * 0.22, 0.02, 1.0))
            candidates.append((
                "AQE_COALESCE_SKEW_JOIN",
                aqe_cost,
                aqe_p,
                f"Enable AQE skew join splitting (skew ratio: {snapshot.skew_duration_ratio:.2f}x). "
                f"Eliminates partition stragglers with $18 runtime overhead.",
            ))

        # 3. Action: SCALE_WORKERS_MEMORY
        # Effective for memory/disk spill with high active tasks
        # STRICT CAUSAL INVARIANT: Ineffective if bottleneck is single-partition skew (Amdahl's Law)
        if snapshot.disk_spilled_gib > 1.0 or snapshot.memory_spilled_gib > 5.0 or risk.dominant_risk_mechanism == "SPILL":
            # If skew is severe (ratio > 3.0), scaling workers yields nearly 0 benefit because straggler is serial
            is_skew_bound = snapshot.skew_duration_ratio >= 2.8
            scaling_cost = round(remaining_hours * burn_rate * 1.5, 2)  # +150% cluster capacity
            
            if is_skew_bound:
                # Amdahl's Law penalty: workers added do not accelerate the straggler task!
                scale_p = float(np.clip(p_base * 0.90, 0.05, 1.0))
                reason = (
                    f"Scale 16 workers & double memory. CAUTION: High skew ({snapshot.skew_duration_ratio:.1f}x) "
                    "limits parallelism gains under Amdahl's Law."
                )
            else:
                scale_p = float(np.clip(p_base * 0.28, 0.03, 1.0))
                reason = (
                    f"Scale 16 workers and double executor memory. Eliminates {snapshot.disk_spilled_gib:.1f} GiB "
                    "disk spill and restores in-memory shuffle throughput."
                )
            candidates.append((
                "SCALE_WORKERS_MEMORY",
                scaling_cost,
                scale_p,
                reason,
            ))

        # 4. Action: BACKOFF_OCC_SERIALIZE
        # Highly effective for Delta Lake optimistic concurrency control contention
        if snapshot.occ_conflict_retries >= 2 or risk.dominant_risk_mechanism == "OCC_CONTENTION":
            backoff_cost = 12.0  # Scheduling backoff delay cost
            backoff_p = float(np.clip(p_base * 0.25, 0.03, 1.0))
            candidates.append((
                "BACKOFF_OCC_SERIALIZE",
                backoff_cost,
                backoff_p,
                f"Inject exponential backoff with full jitter on Delta table '{snapshot.delta_table_name}'. "
                f"Resolves {snapshot.occ_conflict_retries} OCC conflict retries.",
            ))

        # 5. Action: RESTART_CLEAN
        # Only feasible if early in execution (<20% progress) and breach is already near certain (>80%)
        if snapshot.fraction_tasks_completed < 0.20 and p_base > 0.75:
            restart_cost = round((snapshot.as_of_seconds / 3600.0) * burn_rate + 25.0, 2)
            restart_p = 0.35  # Clean slate restart with tuned config
            candidates.append((
                "RESTART_CLEAN",
                restart_cost,
                restart_p,
                "Terminate stalled execution early and restart with pre-allocated shuffle memory.",
            ))

        return candidates

    def evaluate(
        self,
        snapshot: PipelineObservationSnapshot,
        risk: RiskEstimate,
    ) -> CausalDecision:
        """
        Evaluates minimum expected financial loss and prescribes prescriptive remediation.
        """
        penalty = self._get_contract_penalty(snapshot.business_domain)
        p_base = risk.breach_probability
        e_loss_noop = p_base * penalty

        # If residual breach risk is negligible, prescribe NO_OP immediately
        if p_base < self.min_intervention_threshold:
            return CausalDecision(
                decision_id=f"dec-{uuid.uuid4().hex[:8]}",
                run_id=snapshot.run_id,
                as_of_seconds=snapshot.as_of_seconds,
                action="NO_OP",
                reason=f"Pipeline is operating on-time with low breach probability ({p_base*100:.1f}% < {self.min_intervention_threshold*100:.0f}% threshold).",
                prob_breach_without_action=p_base,
                prob_breach_with_action=p_base,
                action_cost_usd=0.0,
                contract_penalty_usd=penalty,
                expected_loss_without_action_usd=round(e_loss_noop, 2),
                expected_loss_with_action_usd=round(e_loss_noop, 2),
                expected_net_savings_usd=0.0,
                should_intervene=False,
            )

        candidates = self._estimate_counterfactual_outcomes(snapshot, risk, penalty)

        best_action: InterventionActionType = "NO_OP"
        best_cost = 0.0
        best_p = p_base
        best_loss = e_loss_noop
        best_savings = 0.0
        best_reason = "No candidate intervention achieves positive expected net financial savings."

        for action, cost, p_counterfactual, reason in candidates:
            # Expected loss: E[Loss(a)] = Cost(a) + P(Breach | do(a)) * Penalty
            e_loss_action = cost + (p_counterfactual * penalty)
            savings = e_loss_noop - e_loss_action

            if e_loss_action < best_loss:
                best_loss = e_loss_action
                best_action = action
                best_cost = cost
                best_p = p_counterfactual
                best_savings = savings
                best_reason = reason

        should_intervene = (
            best_action != "NO_OP"
            and best_savings >= self.min_net_savings_usd
        )

        # If optimal savings below minimum threshold, fall back to NO_OP
        if not should_intervene and best_action != "NO_OP":
            best_reason = (
                f"Candidate action '{best_action}' projected savings (${best_savings:.2f}) "
                f"below minimum financial threshold (${self.min_net_savings_usd:.2f})."
            )
            best_action = "NO_OP"
            best_cost = 0.0
            best_p = p_base
            best_loss = e_loss_noop
            best_savings = 0.0

        return CausalDecision(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id=snapshot.run_id,
            as_of_seconds=round(snapshot.as_of_seconds, 1),
            action=best_action,
            reason=best_reason,
            prob_breach_without_action=round(p_base, 4),
            prob_breach_with_action=round(best_p, 4),
            action_cost_usd=round(best_cost, 2),
            contract_penalty_usd=round(penalty, 2),
            expected_loss_without_action_usd=round(e_loss_noop, 2),
            expected_loss_with_action_usd=round(best_loss, 2),
            expected_net_savings_usd=round(best_savings, 2),
            should_intervene=should_intervene,
        )
