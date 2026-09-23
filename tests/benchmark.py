"""
tests/benchmark.py - Quantitative Latency & Memory Benchmark.
Measures p50, p95, and p99 latency over 50 iterations.
Enforces SLA constraints: Survival Inference < 5.0 ms, Causal Policy < 1.0 ms.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adapters.duckdb_adapter import DuckDBStorageAdapter
from src.domain.entities import PipelineObservationSnapshot
from src.domain.survival_models import SurvivalRiskEngine
from src.domain.causal_policy import CausalFinOpsPolicy


def run_benchmarks(iterations: int = 50):
    print("=" * 70)
    print("  MIRANTE LAKEHOUSE SLA SURVIVAL ENGINE - QUANTITATIVE BENCHMARK")
    print("=" * 70)

    # 1. Setup sample data and model
    n_sample = 500
    df_train = pd.DataFrame({
        "observed_duration_sec": np.random.weibull(2.0, n_sample) * 6000.0 + 1200.0,
        "completed": np.random.binomial(1, 0.92, n_sample),
        "as_of_seconds": np.random.uniform(300.0, 4000.0, n_sample),
        "fraction_tasks_completed": np.random.uniform(0.1, 0.9, n_sample),
        "skew_duration_ratio": 1.0 + np.random.exponential(0.5, n_sample),
        "disk_spilled_gib": np.random.exponential(2.0, n_sample),
        "gc_pressure": np.random.beta(2, 20, n_sample),
        "occ_conflict_retries": np.random.poisson(0.8, n_sample),
    })

    print("[1/3] Fitting SurvivalRiskEngine (AFT Weibull)...")
    engine = SurvivalRiskEngine(model_type="weibull", penalizer=0.01)
    engine.fit(df_train)

    policy = CausalFinOpsPolicy()

    snapshot = PipelineObservationSnapshot(
        snapshot_id="bench-snap-01",
        run_id="bench-run-001",
        job_name="BRL_Pix_Clearing_ETL_Pipeline",
        business_domain="BRL_Pix_Clearing",
        as_of_seconds=2400.0,
        contract_sla_seconds=7200.0,
        remaining_to_sla_seconds=4800.0,
        fraction_tasks_completed=0.40,
        tasks_active=32,
        tasks_failed=0,
        skew_duration_ratio=2.8,
        disk_spilled_gib=4.5,
        gc_pressure=0.03,
        occ_conflict_retries=1,
    )

    # Warmup
    for _ in range(10):
        r = engine.predict_risk(snapshot)
        policy.evaluate(snapshot, r)

    # 2. Benchmark Survival Risk Inference
    print(f"[2/3] Benchmarking Survival Risk Inference ({iterations} iterations)...")
    risk_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.predict_risk(snapshot)
        risk_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 3. Benchmark Causal Policy Decision
    print(f"[3/3] Benchmarking Causal FinOps Policy ({iterations} iterations)...")
    risk = engine.predict_risk(snapshot)
    policy_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        policy.evaluate(snapshot, risk)
        policy_latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_risk = float(np.percentile(risk_latencies, 50))
    p95_risk = float(np.percentile(risk_latencies, 95))
    p99_risk = float(np.percentile(risk_latencies, 99))

    p50_pol = float(np.percentile(policy_latencies, 50))
    p95_pol = float(np.percentile(policy_latencies, 95))
    p99_pol = float(np.percentile(policy_latencies, 99))

    print("\n" + "=" * 70)
    print("  QUANTITATIVE LATENCY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Survival Risk Inference p50:  {p50_risk:.4f} ms")
    print(f"  Survival Risk Inference p95:  {p95_risk:.4f} ms (Target: < 5.0 ms)")
    print(f"  Survival Risk Inference p99:  {p99_risk:.4f} ms")
    print("  " + "-" * 66)
    print(f"  Causal Policy Decision p50:   {p50_pol:.4f} ms")
    print(f"  Causal Policy Decision p95:   {p95_pol:.4f} ms (Target: < 1.0 ms)")
    print(f"  Causal Policy Decision p99:   {p99_pol:.4f} ms")
    print("=" * 70 + "\n")

    assert p95_risk < 5.0, f"Risk p95 latency {p95_risk} exceeded 5.0 ms SLA!"
    assert p95_pol < 1.0, f"Policy p95 latency {p95_pol} exceeded 1.0 ms SLA!"
    print("[SUCCESS] All architectural latency constraints satisfied.")


if __name__ == "__main__":
    run_benchmarks()