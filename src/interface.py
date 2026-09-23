"""
src/interface.py - Rich Terminal User Interface (CLI_TUI Paradigm).
Mirante Lakehouse SLA Survival Engine - Interactive Executive & Telemetry Console.
Visualizes real-time pipeline survival curves, physical bottleneck diagnostics,
and prescriptive causal FinOps remediation decisions across Brazilian tier-1 institutions.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adapters.duckdb_adapter import DuckDBStorageAdapter
from src.domain.entities import PipelineObservationSnapshot
from src.domain.survival_models import SurvivalRiskEngine
from src.domain.causal_policy import CausalFinOpsPolicy
from src.marts.medallion_marts import MedallionMartBuilder

console = Console()


def render_dashboard(base_lakehouse_dir: str = "data/lakehouse"):
    # 1. Header Banner
    banner_text = (
        "[bold white]MIRANTE LAKEHOUSE SLA SURVIVAL ENGINE[/bold white]\n"
        "[dim]Causal & Survival Lifecycle Analytics for Mission-Critical Spark & Delta Lake Pipelines[/dim]\n"
        "[cyan]Institutions: BRL Pix Clearing | BACEN Regulatory | STJ Judicial | Receita Federal[/cyan]"
    )
    console.print(Panel(banner_text, title="[bold green]System Operational Status: ONLINE (DIP Engine)[/bold green]", border_style="cyan"))

    # 2. Ingest sample telemetry & initialize models
    storage = DuckDBStorageAdapter(database=":memory:")
    runs_path = os.path.join(base_lakehouse_dir, "silver", "fact_pipeline_execution", "**", "*.parquet").replace("\\", "/")
    snaps_path = os.path.join(base_lakehouse_dir, "silver", "fact_pipeline_observation", "**", "*.parquet").replace("\\", "/")

    if not os.path.exists(os.path.join(base_lakehouse_dir, "silver", "fact_pipeline_execution")):
        console.print("[yellow]Notice: Generating initial lakehouse dataset for telemetry demonstration...[/yellow]")
        from src.data_generator import generate_lakehouse_telemetry
        generate_lakehouse_telemetry(num_runs=300, base_output_dir=base_lakehouse_dir, seed=42)

    # Train model on representative sample
    query_train = f"""
        SELECT 
            r.observed_duration_sec,
            CASE WHEN r.final_status in ('SUCCESS_ON_TIME', 'RUNNING_PAST_DEADLINE') THEN 1 ELSE 0 END as completed,
            s.as_of_seconds,
            s.fraction_tasks_completed,
            s.skew_duration_ratio,
            s.disk_spilled_gib,
            s.gc_pressure,
            s.occ_conflict_retries
        FROM read_parquet('{runs_path}', hive_partitioning = true) r
        JOIN read_parquet('{snaps_path}', hive_partitioning = true) s ON r.run_id = s.run_id
        USING SAMPLE 2500 ROWS;
    """
    df_train = storage.execute_query(query_train)
    engine = SurvivalRiskEngine(model_type="weibull", penalizer=0.01)
    engine.fit(df_train)

    policy = CausalFinOpsPolicy()

    # 3. Live Pipeline Telemetry & Prognosis Table
    live_samples = [
        PipelineObservationSnapshot(
            snapshot_id="snap-pix-live",
            run_id="run-pix-0842",
            job_name="BRL_Pix_Clearing_ETL",
            business_domain="BRL_Pix_Clearing",
            as_of_seconds=4200.0,
            contract_sla_seconds=7200.0,
            remaining_to_sla_seconds=3000.0,
            fraction_tasks_completed=0.42,
            tasks_active=32,
            tasks_failed=0,
            skew_duration_ratio=3.85,  # Extreme Skew
            disk_spilled_gib=0.0,
            gc_pressure=0.03,
            occ_conflict_retries=0,
        ),
        PipelineObservationSnapshot(
            snapshot_id="snap-bacen-live",
            run_id="run-bacen-0129",
            job_name="BACEN_Regulatory_ETL",
            business_domain="BACEN_Regulatory_Report",
            as_of_seconds=5400.0,
            contract_sla_seconds=10800.0,
            remaining_to_sla_seconds=5400.0,
            fraction_tasks_completed=0.38,
            tasks_active=48,
            tasks_failed=0,
            skew_duration_ratio=1.12,
            disk_spilled_gib=26.4,     # Heavy Spill
            memory_spilled_gib=38.0,
            gc_pressure=0.04,
            occ_conflict_retries=0,
        ),
        PipelineObservationSnapshot(
            snapshot_id="snap-receita-live",
            run_id="run-receita-0551",
            job_name="Receita_Tax_ETL",
            business_domain="Receita_Tax_Reconciliation",
            as_of_seconds=6000.0,
            contract_sla_seconds=12600.0,
            remaining_to_sla_seconds=6600.0,
            fraction_tasks_completed=0.52,
            tasks_active=16,
            tasks_failed=0,
            skew_duration_ratio=1.15,
            disk_spilled_gib=0.0,
            gc_pressure=0.02,
            occ_conflict_retries=5,    # Delta OCC contention
        ),
        PipelineObservationSnapshot(
            snapshot_id="snap-stj-live",
            run_id="run-stj-0914",
            job_name="STJ_Judicial_ETL",
            business_domain="STJ_Judicial_Analytics",
            as_of_seconds=3600.0,
            contract_sla_seconds=9000.0,
            remaining_to_sla_seconds=5400.0,
            fraction_tasks_completed=0.68,
            tasks_active=16,
            tasks_failed=0,
            skew_duration_ratio=1.08,
            disk_spilled_gib=0.0,
            gc_pressure=0.02,
            occ_conflict_retries=0,    # Nominal on-time
        ),
    ]

    telemetry_table = Table(title="[bold yellow][LIVE] Live Pipeline Telemetry & Survival Prognosis (t_c)[/bold yellow]", border_style="yellow")
    telemetry_table.add_column("Domain", style="cyan", no_wrap=True)
    telemetry_table.add_column("Run ID", style="dim")
    telemetry_table.add_column("Elapsed / SLA", justify="center")
    telemetry_table.add_column("Tasks %", justify="right")
    telemetry_table.add_column("Skew", justify="right")
    telemetry_table.add_column("Spill (GiB)", justify="right")
    telemetry_table.add_column("OCC", justify="right")
    telemetry_table.add_column("P(Breach)", justify="right")
    telemetry_table.add_column("p50 Est (h)", justify="right")
    telemetry_table.add_column("Primary Bottleneck", justify="center")

    decisions = []

    for snap in live_samples:
        risk = engine.predict_risk(snap)
        decision = policy.evaluate(snap, risk)
        decisions.append(decision)

        p_breach_color = "red" if risk.breach_probability > 0.50 else ("yellow" if risk.breach_probability > 0.20 else "green")
        elapsed_str = f"{snap.as_of_seconds/60:.0f}m / {snap.contract_sla_seconds/60:.0f}m"
        p50_hours = f"{risk.p50_completion_seconds/3600:.2f}h"

        telemetry_table.add_row(
            snap.business_domain,
            snap.run_id,
            elapsed_str,
            f"{snap.fraction_tasks_completed*100:.1f}%",
            f"{snap.skew_duration_ratio:.2f}x",
            f"{snap.disk_spilled_gib:.1f}",
            str(snap.occ_conflict_retries),
            f"[{p_breach_color}]{risk.breach_probability*100:.1f}%[/{p_breach_color}]",
            p50_hours,
            f"[bold]{risk.dominant_risk_mechanism}[/bold]",
        )

    console.print(telemetry_table)

    # 4. Prescriptive Causal FinOps Remediation Table
    finops_table = Table(title="[bold green][FINOPS] Prescriptive Causal FinOps Remediation Ledger[/bold green]", border_style="green")
    finops_table.add_column("Run ID", style="dim")
    finops_table.add_column("Prescribed Action", style="bold cyan")
    finops_table.add_column("Action Cost", justify="right")
    finops_table.add_column("Risk Before", justify="right")
    finops_table.add_column("Risk After", justify="right")
    finops_table.add_column("Net Savings ($)", justify="right", style="bold green")
    finops_table.add_column("Status", justify="center")
    finops_table.add_column("Causal Justification", style="dim")

    for d in decisions:
        action_color = "bold magenta" if d.should_intervene else "dim"
        status_text = "[bold green]INTERVENE[/bold green]" if d.should_intervene else "[dim]MONITOR[/dim]"
        savings_text = f"${d.expected_net_savings_usd:,.2f}" if d.expected_net_savings_usd > 0 else "$0.00"

        finops_table.add_row(
            d.run_id,
            f"[{action_color}]{d.action}[/{action_color}]",
            f"${d.action_cost_usd:.2f}",
            f"{d.prob_breach_without_action*100:.1f}%",
            f"{d.prob_breach_with_action*100:.1f}%",
            savings_text,
            status_text,
            d.reason[:75] + ("..." if len(d.reason) > 75 else ""),
        )

    console.print(finops_table)

    # 5. Executive Gold Data Mart Summary
    builder = MedallionMartBuilder(storage=storage, base_lakehouse_dir=base_lakehouse_dir)
    df_exec = builder.build_executive_risk_mart(export_parquet=False)

    exec_table = Table(title="[bold magenta][GOLD MART] Medallion Gold Mart: SLA Executive Risk Exposure[/bold magenta]", border_style="magenta")
    exec_table.add_column("Business Domain", style="bold white")
    exec_table.add_column("Total Runs", justify="right")
    exec_table.add_column("Breaches", justify="right")
    exec_table.add_column("Breach Rate", justify="right")
    exec_table.add_column("p95 Dur (h)", justify="right")
    exec_table.add_column("Compute Spend ($)", justify="right")
    exec_table.add_column("Penalty Exposure ($)", justify="right", style="bold red")
    exec_table.add_column("Primary Bottleneck", justify="center")

    for _, row in df_exec.head(4).iterrows():
        exec_table.add_row(
            str(row["business_domain"]),
            f"{int(row['total_pipeline_runs']):,}",
            f"{int(row['total_breaches']):,}",
            f"{row['observed_breach_rate_pct']:.1f}%",
            f"{row['p95_duration_hours']:.2f}h",
            f"${row['total_compute_spend_usd']:,.2f}",
            f"${row['total_penalty_liability_usd']:,.2f}",
            str(row["primary_operational_bottleneck"]),
        )

    console.print(exec_table)

    storage.close()
    console.print("\n[bold green][OK] All lakehouse partitions, survival models, and causal policies executed successfully.[/bold green]\n")


if __name__ == "__main__":
    render_dashboard()