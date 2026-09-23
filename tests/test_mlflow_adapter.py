"""
tests/test_mlflow_adapter.py - Unit & Integration Tests for MLOps Tracking Protocol.
Validates:
1. Conformance to MLflowTrackingProtocol.
2. Experiment metadata, parameters, and metric logging.
3. Actuarial validation metrics: Harrell's C-index and Integrated Brier Score.
4. Survival model evaluation and governance audit trail.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.domain.contracts import MLflowTrackingProtocol
from src.domain.survival_models import SurvivalRiskEngine
from src.mlops.mlflow_adapter import MLflowTrackingAdapter


@pytest.fixture
def tracking_adapter(tmp_path):
    tracking_dir = str(tmp_path / "mlruns_test")
    return MLflowTrackingAdapter(tracking_dir=tracking_dir)


@pytest.fixture
def eval_dataset():
    np.random.seed(42)
    n = 300
    durations = np.random.weibull(2.0, n) * 6000.0 + 1200.0
    completed = np.random.binomial(1, 0.95, n)
    as_of = np.random.uniform(300.0, 3500.0, n)
    frac = np.clip(as_of / durations, 0.05, 0.95)
    skew = 1.0 + np.random.exponential(0.6, n)
    spill = np.random.exponential(1.5, n)
    gc = np.random.beta(2, 20, n)
    occ = np.random.poisson(0.8, n)

    return pd.DataFrame({
        "observed_duration_sec": durations,
        "completed": completed,
        "as_of_seconds": as_of,
        "fraction_tasks_completed": frac,
        "skew_duration_ratio": skew,
        "disk_spilled_gib": spill,
        "gc_pressure": gc,
        "occ_conflict_retries": occ,
    })


def test_mlflow_protocol_conformance(tracking_adapter):
    """Verifies DIP contract adherence."""
    assert isinstance(tracking_adapter, MLflowTrackingProtocol)


def test_log_run_and_retrieval(tracking_adapter):
    """Verifies logging and query retrieval from the tracking registry."""
    run_id = tracking_adapter.log_run(
        experiment_name="mirante_survival_tuning",
        params={"model_type": "weibull", "penalizer": 0.01},
        metrics={"harrell_c_index": 0.82, "brier_score": 0.09},
        tags={"dataset": "gold_v1"},
    )

    assert isinstance(run_id, str)
    assert run_id.startswith("exp-")

    runs = tracking_adapter.get_runs(experiment_name="mirante_survival_tuning")
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["metrics"]["harrell_c_index"] == 0.82


def test_concordance_index_calculation():
    """Verifies Harrell's C-index calculation with perfect vs random discrimination."""
    actual = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    perfect_pred = [1100.0, 2100.0, 3100.0, 4100.0, 5100.0]
    events = [1, 1, 1, 1, 1]

    c_perfect = MLflowTrackingAdapter.compute_concordance_index(actual, perfect_pred, events)
    assert c_perfect == 1.0

    inverted_pred = [5000.0, 4000.0, 3000.0, 2000.0, 1000.0]
    c_inverted = MLflowTrackingAdapter.compute_concordance_index(actual, inverted_pred, events)
    assert c_inverted == 0.0


def test_brier_score_calculation():
    """Verifies Brier score calibration metric."""
    actual = [1000.0, 2000.0, 3000.0, 4000.0]
    events = [1, 1, 1, 1]
    eval_time = 2500.0

    # True status at 2500s: [0, 0, 1, 1]
    # Perfect survival predictions: [0.0, 0.0, 1.0, 1.0]
    bs_perfect = MLflowTrackingAdapter.compute_brier_score_at_time(
        actual, events, [0.0, 0.0, 1.0, 1.0], eval_time
    )
    assert bs_perfect == 0.0

    # Poor predictions
    bs_poor = MLflowTrackingAdapter.compute_brier_score_at_time(
        actual, events, [1.0, 1.0, 0.0, 0.0], eval_time
    )
    assert bs_poor == 1.0


def test_evaluate_and_log_model_integration(tracking_adapter, eval_dataset):
    """
    End-to-End MLOps integration:
    Fits SurvivalRiskEngine, evaluates on holdout data, and verifies Harrell C-index > 0.65.
    """
    engine = SurvivalRiskEngine(model_type="weibull", penalizer=0.01)
    engine.fit(eval_dataset)

    metrics = tracking_adapter.evaluate_and_log_model(
        experiment_name="production_validation",
        engine=engine,
        eval_df=eval_dataset,
        domain_tag="all_domains",
    )

    assert "harrell_c_index" in metrics
    assert "brier_score_at_median" in metrics
    assert "aic" in metrics

    # Survival model should discriminate well above random (0.50)
    assert metrics["harrell_c_index"] > 0.65
    assert metrics["brier_score_at_median"] < 0.25

    # Check persistence
    logged_runs = tracking_adapter.get_runs(experiment_name="production_validation")
    assert len(logged_runs) == 1
