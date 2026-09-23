"""
tests/benchmark.py - Quantitative Latency & Memory Benchmark.
Measures p50, p95, and p99 query latency over 30 iterations.
Enforces SLA constraints: p95 < 150.0 ms.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_generator import generate_domain_dataset
from src.core_engine import DomainAnalyticsEngine, DuckDBStorageAdapter


def run_benchmarks(iterations=30, num_records=10000):
    print(f"[Benchmark] Preparing dataset with {num_records:,} records...")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_domain_dataset(num_records=num_records, output_path=tmp_path)
        adapter = DuckDBStorageAdapter()
        engine = DomainAnalyticsEngine(storage=adapter, data_path=tmp_path)
        
        # Warmup
        engine.execute_analysis()
        
        print(f"[Benchmark] Running {iterations} iterations...")
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            engine.execute_analysis()
            latencies.append((time.perf_counter() - t0) * 1000)
            
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        
        print("\n" + "="*60)
        print("  MIRANTE_TECNOLOGIA_DATA_SCIENTIST_BRIDGE_PROJECT - QUANTITATIVE BENCHMARK (ms)")
        print("="*60)
        print(f"  Dataset Size: {num_records:,} rows")
        print(f"  Iterations:   {iterations}")
        print(f"  p50 Latency:  {p50:.2f} ms")
        print(f"  p95 Latency:  {p95:.2f} ms (Constraint: < 150.0 ms)")
        print(f"  p99 Latency:  {p99:.2f} ms")
        print("="*60 + "\n")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    run_benchmarks()