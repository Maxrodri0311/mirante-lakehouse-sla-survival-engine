"""
src/marts/medallion_marts.py - Medallion Architecture Gold Data Marts (Kimball Dimensional Modeling).
Transforms Silver partitioned Parquet layers into C-Level analytical data marts using DuckDB:
1. mart_sla_executive_risk: Monthly risk exposure, breach rates, compute spend, and penalty liabilities.
2. mart_finops_remediation: Remediation action audit, cost-benefit analysis, gross risk reduction, and ROI.
Follows Dependency Inversion Principle (DIP): operates strictly over AnalyticalStorageProtocol.
"""

import os
from typing import Optional, List, Dict, Any
import pandas as pd

from src.domain.contracts import AnalyticalStorageProtocol
from src.domain.entities import CausalDecision


class MedallionMartBuilder:
    """
    Transforms Silver Lakehouse Parquet partitions into executive Gold Data Marts.
    Decoupled via AnalyticalStorageProtocol.
    """

    def __init__(self, storage: AnalyticalStorageProtocol, base_lakehouse_dir: str = "data/lakehouse"):
        self.storage = storage
        self.base_lakehouse_dir = base_lakehouse_dir
        self.silver_runs_path = os.path.join(
            base_lakehouse_dir, "silver", "fact_pipeline_execution", "**", "*.parquet"
        ).replace("\\", "/")
        self.silver_snaps_path = os.path.join(
            base_lakehouse_dir, "silver", "fact_pipeline_observation", "**", "*.parquet"
        ).replace("\\", "/")
        self.gold_dir = os.path.join(base_lakehouse_dir, "gold")

    def build_executive_risk_mart(self, export_parquet: bool = False) -> pd.DataFrame:
        """
        Builds mart_sla_executive_risk:
        Aggregates runs and time-varying telemetry to quantify monthly SLA breach liabilities
        and dominant physical bottlenecks per business domain.
        """
        query = f"""
            WITH runs AS (
                SELECT 
                    business_domain,
                    year,
                    month,
                    run_id,
                    contract_sla_sec,
                    observed_duration_sec,
                    event_breach,
                    total_compute_cost_usd,
                    penalty_fee_usd
                FROM read_parquet('{self.silver_runs_path}', hive_partitioning = true)
            ),
            snaps AS (
                SELECT 
                    run_id,
                    AVG(skew_duration_ratio) as mean_skew,
                    SUM(disk_spilled_gib) as run_disk_spill,
                    AVG(gc_pressure) as mean_gc,
                    MAX(occ_conflict_retries) as max_occ
                FROM read_parquet('{self.silver_snaps_path}', hive_partitioning = true)
                GROUP BY run_id
            ),
            joined AS (
                SELECT 
                    r.*,
                    COALESCE(s.mean_skew, 1.0) as mean_skew,
                    COALESCE(s.run_disk_spill, 0.0) as run_disk_spill,
                    COALESCE(s.mean_gc, 0.02) as mean_gc,
                    COALESCE(s.max_occ, 0) as max_occ
                FROM runs r
                LEFT JOIN snaps s ON r.run_id = s.run_id
            )
            SELECT 
                business_domain,
                year,
                month,
                COUNT(*) as total_pipeline_runs,
                SUM(CASE WHEN event_breach THEN 1 ELSE 0 END) as total_breaches,
                ROUND(AVG(CASE WHEN event_breach THEN 1.0 ELSE 0.0 END) * 100.0, 2) as observed_breach_rate_pct,
                ROUND(AVG(observed_duration_sec) / 3600.0, 2) as mean_duration_hours,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY observed_duration_sec) / 3600.0, 2) as p95_duration_hours,
                ROUND(SUM(total_compute_cost_usd), 2) as total_compute_spend_usd,
                ROUND(SUM(CASE WHEN event_breach THEN penalty_fee_usd ELSE 0.0 END), 2) as total_penalty_liability_usd,
                ROUND(AVG(mean_skew), 2) as domain_avg_skew_ratio,
                ROUND(SUM(run_disk_spill), 1) as domain_total_disk_spill_gib,
                CASE 
                    WHEN AVG(mean_skew) > 1.8 THEN 'PARTITION_SKEW'
                    WHEN SUM(run_disk_spill) > 500.0 THEN 'SHUFFLE_DISK_SPILL'
                    WHEN AVG(mean_gc) > 0.08 THEN 'JVM_GC_PAUSE'
                    WHEN MAX(max_occ) >= 3 THEN 'DELTA_OCC_CONTENTION'
                    ELSE 'NOMINAL_CAPACITY'
                END as primary_operational_bottleneck
            FROM joined
            GROUP BY business_domain, year, month
            ORDER BY total_penalty_liability_usd DESC, observed_breach_rate_pct DESC;
        """
        df = self.storage.execute_query(query)
        self.storage.register_view("mart_sla_executive_risk", df)

        if export_parquet:
            os.makedirs(self.gold_dir, exist_ok=True)
            out_file = os.path.join(self.gold_dir, "mart_sla_executive_risk.parquet")
            df.to_parquet(out_file, index=False)

        return df

    def build_finops_remediation_mart(
        self,
        decisions: List[CausalDecision],
        export_parquet: bool = False,
    ) -> pd.DataFrame:
        """
        Builds mart_finops_remediation:
        Aggregates causal prescriptive interventions to evaluate financial efficacy and ROI.
        """
        if not decisions:
            # Empty fallback schema
            return pd.DataFrame(columns=[
                "action",
                "recommendation_count",
                "total_action_cost_usd",
                "gross_penalty_avoided_usd",
                "net_financial_savings_usd",
                "mean_roi_ratio",
                "intervention_rate_pct",
            ])

        records = [d.model_dump() for d in decisions]
        df_decisions = pd.DataFrame(records)
        self.storage.register_view("v_decisions", df_decisions)

        query = """
            SELECT 
                action,
                COUNT(*) as recommendation_count,
                ROUND(SUM(action_cost_usd), 2) as total_action_cost_usd,
                ROUND(SUM((prob_breach_without_action - prob_breach_with_action) * contract_penalty_usd), 2) as gross_penalty_avoided_usd,
                ROUND(SUM(expected_net_savings_usd), 2) as net_financial_savings_usd,
                ROUND(
                    CASE 
                        WHEN SUM(action_cost_usd) > 0 
                        THEN SUM((prob_breach_without_action - prob_breach_with_action) * contract_penalty_usd) / SUM(action_cost_usd)
                        ELSE 0.0 
                    END, 2
                ) as mean_roi_ratio,
                ROUND(AVG(CASE WHEN should_intervene THEN 1.0 ELSE 0.0 END) * 100.0, 1) as intervention_rate_pct
            FROM v_decisions
            GROUP BY action
            ORDER BY net_financial_savings_usd DESC;
        """
        df = self.storage.execute_query(query)
        self.storage.register_view("mart_finops_remediation", df)

        if export_parquet:
            os.makedirs(self.gold_dir, exist_ok=True)
            out_file = os.path.join(self.gold_dir, "mart_finops_remediation.parquet")
            df.to_parquet(out_file, index=False)

        return df
