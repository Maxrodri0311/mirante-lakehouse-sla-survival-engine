"""
src/data_generator.py - Calibrated Stochastic Telemetry & Execution Generator.
Simulates real-world Apache Spark & Delta Lake pipeline telemetry for Mirante Tecnologia Lakehouse.
Generates physical pipeline execution logs and time-varying observation snapshots (as-of t_c)
governed by Accelerated Failure Time (AFT) physics, partition skew, shuffle spill,
JVM GC pauses, and Delta Lake OCC concurrency contention.
Zero toy placeholders: variables, types, bounds, and distributions reflect production lakehouses.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# Ensure project root is available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DOMAIN_SPECS = {
    "BRL_Pix_Clearing": {
        "sla_seconds": 7200.0,      # 2.0 hours SLA
        "nominal_lambda": 4200.0,   # Nominal mean ~1.17 hrs
        "penalty_usd": 50000.0,
        "base_hourly_cost": 36.0,
        "cluster_type": "databricks-memory-r5d",
        "weight": 0.30,
    },
    "BACEN_Regulatory_Report": {
        "sla_seconds": 10800.0,     # 3.0 hours SLA
        "nominal_lambda": 6800.0,   # Nominal mean ~1.89 hrs
        "penalty_usd": 40000.0,
        "base_hourly_cost": 48.0,
        "cluster_type": "databricks-standard-i3en",
        "weight": 0.30,
    },
    "STJ_Judicial_Analytics": {
        "sla_seconds": 9000.0,      # 2.5 hours SLA
        "nominal_lambda": 5200.0,   # Nominal mean ~1.44 hrs
        "penalty_usd": 25000.0,
        "base_hourly_cost": 28.0,
        "cluster_type": "databricks-compute-c5d",
        "weight": 0.25,
    },
    "Receita_Tax_Reconciliation": {
        "sla_seconds": 12600.0,     # 3.5 hours SLA
        "nominal_lambda": 8000.0,   # Nominal mean ~2.22 hrs
        "penalty_usd": 35000.0,
        "base_hourly_cost": 42.0,
        "cluster_type": "databricks-serverless-xl",
        "weight": 0.15,
    },
}


def generate_lakehouse_telemetry(
    num_runs: int = 2500,
    base_output_dir: str = "data/lakehouse",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic enterprise lakehouse datasets:
    1. fact_pipeline_execution: Run-level physical lifecycle and outcomes.
    2. fact_pipeline_observation: Time-varying observation snapshots (as-of t_c).
    Total snapshots: ~50,000+ records.
    """
    start_time = time.time()
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[Data Generator] Simulating {num_runs:,} Spark pipeline runs for Mirante Tecnologia...")

    domains = list(DOMAIN_SPECS.keys())
    domain_weights = [DOMAIN_SPECS[d]["weight"] for d in domains]
    assigned_domains = rng.choice(domains, size=num_runs, p=domain_weights)

    runs_data = []
    snapshots_data = []
    base_date = datetime(2026, 1, 1, 0, 0, 0)

    for i in range(1, num_runs + 1):
        domain = assigned_domains[i - 1]
        spec = DOMAIN_SPECS[domain]
        run_id = f"run-{i:06d}"
        job_id = f"job-{domain[:4].lower()}-{(i % 40) + 1:03d}"
        job_name = f"{domain}_ETL_Pipeline"
        cluster_id = f"cluster-{spec['cluster_type']}-{rng.integers(100, 999)}"
        sla_sec = spec["sla_seconds"]
        penalty_usd = spec["penalty_usd"]
        hourly_rate = spec["base_hourly_cost"]

        # Latent operational shock coefficients (The Physics of Spark Execution)
        # Latent skew severity [0, 1]
        latent_skew = float(rng.beta(a=1.5, b=5.0))
        # Latent memory pressure [0, 1]
        latent_mem_pressure = float(rng.beta(a=1.8, b=4.5))
        # Latent OCC concurrency contention [0, 1]
        latent_occ = float(rng.beta(a=1.2, b=6.0)) if domain == "Receita_Tax_Reconciliation" else float(rng.beta(a=1.0, b=10.0))

        # 1. Accelerated Failure Time (AFT) Weibull Physics
        # k = 2.0 (aging effect: pipeline risk accumulates as late stages struggle)
        k_shape = 1.9
        lambda_0 = spec["nominal_lambda"]

        # Covariate acceleration factors:
        # T_complete = lambda_0 * exp(beta^T X) * (-ln U)^(1/k)
        beta_skew = 0.55
        beta_spill = 0.65
        beta_gc = 0.40
        beta_occ = 0.30

        acceleration_index = (
            beta_skew * latent_skew
            + beta_spill * latent_mem_pressure
            + beta_gc * (latent_mem_pressure * 0.8)
            + beta_occ * latent_occ
        )

        weibull_rand = float(-np.log(max(1e-6, rng.uniform(0.01, 0.99)))) ** (1.0 / k_shape)
        nominal_duration = lambda_0 * np.exp(acceleration_index) * weibull_rand
        nominal_duration = max(900.0, nominal_duration)  # minimum 15 minutes

        # Stochastic Failure (OOM / Node loss) or Human Cancellation
        is_terminal_failure = bool(latent_mem_pressure > 0.85 and rng.uniform() < 0.35)
        is_cancelled = bool(not is_terminal_failure and nominal_duration > (sla_sec * 1.3) and rng.uniform() < 0.20)

        if is_terminal_failure:
            actual_duration = round(nominal_duration * rng.uniform(0.40, 0.80), 1)
            final_status = "FAILED_TERMINAL"
            is_censored = False
            event_breach = True
        elif is_cancelled:
            actual_duration = round(nominal_duration * rng.uniform(0.60, 0.90), 1)
            final_status = "CANCELLED"
            is_censored = True  # Censored: terminal completion was interrupted
            event_breach = actual_duration > sla_sec
        elif nominal_duration > sla_sec:
            actual_duration = round(nominal_duration, 1)
            final_status = "RUNNING_PAST_DEADLINE"
            is_censored = False  # Event observed: breach
            event_breach = True
        else:
            actual_duration = round(nominal_duration, 1)
            final_status = "SUCCESS_ON_TIME"
            is_censored = False  # Success observed cleanly
            event_breach = False

        start_offset_sec = float(rng.uniform(0, 180 * 86400))  # 6-month historical span
        start_dt = base_date + timedelta(seconds=start_offset_sec)
        completion_dt = start_dt + timedelta(seconds=actual_duration)

        compute_cost_usd = round((actual_duration / 3600.0) * hourly_rate, 2)

        runs_data.append({
            "run_id": run_id,
            "job_id": job_id,
            "job_name": job_name,
            "business_domain": domain,
            "cluster_id": cluster_id,
            "spark_version": "3.5.0",
            "contract_sla_sec": sla_sec,
            "scheduled_start_ts": start_dt.isoformat(),
            "actual_start_ts": start_dt.isoformat(),
            "completion_ts": completion_dt.isoformat(),
            "observed_duration_sec": actual_duration,
            "final_status": final_status,
            "event_breach": event_breach,
            "is_censored": is_censored,
            "total_compute_cost_usd": compute_cost_usd,
            "penalty_fee_usd": penalty_usd,
            "year": start_dt.year,
            "month": start_dt.month,
        })

        # 2. Time-Varying Observation Snapshots Generation (as-of t_c)
        # Snapshots every 300 to 600 seconds until completion or breach
        step_sec = 600.0 if actual_duration > 7200.0 else 300.0
        t_c = step_sec
        snap_idx = 1

        while t_c < actual_duration:
            progress_ratio = t_c / actual_duration
            remaining_to_sla = sla_sec - t_c

            # Dynamic Task Progress (S-curve)
            frac_completed = float(np.clip(progress_ratio ** 1.3 + rng.normal(0, 0.02), 0.0, 0.99))
            active_tasks = int(max(4, np.round(64 * (1.0 - frac_completed) + rng.integers(-4, 5))))
            failed_tasks = int(max(0, np.round(latent_mem_pressure * 4 + rng.integers(-1, 2)))) if is_terminal_failure else 0

            # Dynamic Shuffle & Spill Telemetry
            # Spills manifest after stage 2 (progress > 30%)
            spill_active = progress_ratio > 0.30
            shuff_read_gib = round(float(25.0 + 200.0 * progress_ratio * (1.0 + latent_skew)), 2)
            shuff_write_gib = round(float(shuff_read_gib * 0.85), 2)

            if spill_active and latent_mem_pressure > 0.40:
                mem_spilled_gib = round(float((progress_ratio - 0.3) * 120.0 * latent_mem_pressure), 2)
                disk_spilled_gib = round(float(mem_spilled_gib * (0.6 + 0.5 * latent_mem_pressure)), 2)
            else:
                mem_spilled_gib = 0.0
                disk_spilled_gib = 0.0

            spill_disk_to_mem = round(disk_spilled_gib / (mem_spilled_gib + 1e-4), 4)
            spill_out = round(disk_spilled_gib / (shuff_write_gib + 1e-4), 4)
            fetch_wait = round(float(0.02 + 0.25 * latent_skew * progress_ratio), 4)

            # Dynamic Skew & Tail Amplification (increases dramatically in skewed jobs)
            skew_dur = round(float(1.1 + 4.5 * latent_skew * (progress_ratio ** 0.8)), 3)
            skew_cv = round(float(0.1 + 1.2 * latent_skew * progress_ratio), 3)
            tail_amp = round(float(1.15 + 5.2 * latent_skew * progress_ratio), 3)

            # Dynamic JVM GC & CPU
            cpu_time_ms = round(t_c * 1000.0 * 16 * float(rng.uniform(0.65, 0.92)), 1)
            gc_time_ms = round(float(cpu_time_ms * (0.02 + 0.25 * latent_mem_pressure * progress_ratio)), 1)
            gc_press = round(gc_time_ms / (t_c * 1000.0 * 16 + 1e-4), 4)
            gc_cpu = round(gc_time_ms / (cpu_time_ms + 1e-4), 4)

            # Dynamic Delta Lake OCC
            occ_retries = int(max(0, np.round(latent_occ * 6 * progress_ratio + rng.integers(-1, 2))))
            occ_ratio = round(occ_retries / (occ_retries + 5.0), 3)
            small_files = round(float(0.04 + 0.30 * latent_occ), 3)

            snapshots_data.append({
                "snapshot_id": f"snap-{i:06d}-{snap_idx:03d}",
                "run_id": run_id,
                "job_name": job_name,
                "business_domain": domain,
                "as_of_seconds": round(t_c, 1),
                "contract_sla_seconds": sla_sec,
                "remaining_to_sla_seconds": round(remaining_to_sla, 1),
                "fraction_tasks_completed": round(frac_completed, 4),
                "tasks_active": active_tasks,
                "tasks_failed": failed_tasks,
                "shuffle_read_gib": shuff_read_gib,
                "shuffle_write_gib": shuff_write_gib,
                "memory_spilled_gib": mem_spilled_gib,
                "disk_spilled_gib": disk_spilled_gib,
                "spill_ratio_disk_to_mem": spill_disk_to_mem,
                "spill_ratio_output": spill_out,
                "fetch_wait_ratio": fetch_wait,
                "skew_duration_ratio": skew_dur,
                "skew_bytes_cv": skew_cv,
                "tail_amplification": tail_amp,
                "jvm_gc_time_ms": gc_time_ms,
                "executor_cpu_time_ms": cpu_time_ms,
                "executor_run_time_ms": round(t_c * 1000.0, 1),
                "gc_pressure": gc_press,
                "gc_cpu_ratio": gc_cpu,
                "executor_count": 16,
                "hourly_cluster_cost_usd": hourly_rate,
                "delta_table_name": f"gold_{domain.lower()}",
                "delta_version": 12 + (snap_idx % 5),
                "occ_conflict_retries": occ_retries,
                "commit_duration_ms": round(150.0 + 350.0 * occ_ratio, 1),
                "occ_contention_ratio": occ_ratio,
                "small_file_ratio": small_files,
            })

            t_c += step_sec
            snap_idx += 1

    df_runs = pd.DataFrame(runs_data)
    df_snapshots = pd.DataFrame(snapshots_data)

    # 3. Physical Persistence to Silver Lakehouse Parquet
    runs_dir = os.path.join(base_output_dir, "silver", "fact_pipeline_execution")
    snapshots_dir = os.path.join(base_output_dir, "silver", "fact_pipeline_observation")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)

    # Write runs partitioned by business_domain / year / month
    table_runs = pa.Table.from_pandas(df_runs)
    ds.write_dataset(
        data=table_runs,
        base_dir=runs_dir,
        format="parquet",
        partitioning=["business_domain", "year", "month"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
    )

    # Write snapshots partitioned by business_domain
    table_snaps = pa.Table.from_pandas(df_snapshots)
    ds.write_dataset(
        data=table_snaps,
        base_dir=snapshots_dir,
        format="parquet",
        partitioning=["business_domain"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
    )

    elapsed = time.time() - start_time
    breach_rate = df_runs["event_breach"].mean() * 100.0
    print(
        f"[Data Generator] Successfully generated {len(df_runs):,} runs and "
        f"{len(df_snapshots):,} observation snapshots in {elapsed:.2f}s.\n"
        f"                 - Overall SLA Breach Rate: {breach_rate:.1f}%\n"
        f"                 - Terminal Failures:       {(df_runs['final_status'] == 'FAILED_TERMINAL').mean()*100:.1f}%\n"
        f"                 - Success On-Time:         {(df_runs['final_status'] == 'SUCCESS_ON_TIME').mean()*100:.1f}%\n"
        f"                 - Hive Lake Root:          {base_output_dir}/silver"
    )

    return df_runs, df_snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mirante Lakehouse Telemetry Dataset.")
    parser.add_argument("--runs", type=int, default=2500, help="Number of physical pipeline runs to simulate")
    parser.add_argument("--output", type=str, default="data/lakehouse", help="Output lakehouse root directory")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    args = parser.parse_args()

    generate_lakehouse_telemetry(
        num_runs=args.runs,
        base_output_dir=args.output,
        seed=args.seed,
    )