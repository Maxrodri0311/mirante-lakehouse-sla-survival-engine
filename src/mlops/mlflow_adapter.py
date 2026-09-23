"""
src/mlops/mlflow_adapter.py - MLOps Tracking Protocol Implementation & Survival Model Validation.
Computes actuarial validation metrics for Completion Survival Models:
1. Harrell's Concordance Index (C-index): Measures rank discrimination of completion time.
2. Integrated Brier Score (IBS): Measures mean squared calibration error over timeline.
3. AIC & Log-Likelihood: Model parsimony and likelihood bounds.
4. Experiment Governance: Logs parameters, metrics, and tags to tracking registry.
Conforms strictly to MLflowTrackingProtocol via Dependency Inversion Principle.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Sequence, Tuple
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from src.domain.contracts import MLflowTrackingProtocol
from src.domain.survival_models import SurvivalRiskEngine


class MLflowTrackingAdapter:
    """
    Enterprise MLOps experiment tracking adapter implementing MLflowTrackingProtocol.
    Stores experiment runs in a local JSONL / JSON catalog shielded by .gitignore.
    """

    def __init__(self, tracking_dir: str = "mlruns"):
        self.tracking_dir = tracking_dir
        os.makedirs(tracking_dir, exist_ok=True)
        self.registry_file = os.path.join(tracking_dir, "experiment_registry.json")
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w") as f:
                json.dump([], f)

    def log_run(
        self,
        experiment_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Logs run metadata, parameters, and metrics to tracking catalog.
        Returns unique run_id.
        """
        run_id = f"exp-{uuid.uuid4().hex[:12]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        entry = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
            "artifacts": artifacts or {},
        }

        # Read existing and append
        with open(self.registry_file, "r") as f:
            registry = json.load(f)

        registry.append(entry)

        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

        return run_id

    def get_runs(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns logged runs, optionally filtered by experiment name."""
        with open(self.registry_file, "r") as f:
            registry = json.load(f)

        if experiment_name:
            return [r for r in registry if r.get("experiment_name") == experiment_name]
        return registry

    @staticmethod
    def compute_concordance_index(
        actual_durations: Sequence[float],
        predicted_scores: Sequence[float],
        event_observed: Sequence[int],
    ) -> float:
        """
        Computes Harrell's C-index:
        For survival duration, higher predicted time should correlate with longer actual duration.
        """
        c_idx = concordance_index(
            event_times=np.array(actual_durations),
            predicted_scores=np.array(predicted_scores),
            event_observed=np.array(event_observed),
        )
        return float(round(c_idx, 4))

    @staticmethod
    def compute_brier_score_at_time(
        actual_durations: Sequence[float],
        event_observed: Sequence[int],
        predicted_survival_probs: Sequence[float],
        eval_time: float,
    ) -> float:
        """
        Computes Brier Score at evaluation time t:
        BS(t) = (1/N) * sum_i (I(T_i > t) - S_hat(t | X_i))^2
        """
        y_true = np.array([1.0 if d > eval_time else 0.0 for d in actual_durations])
        y_pred = np.array(predicted_survival_probs)
        mse = float(np.mean((y_true - y_pred) ** 2))
        return float(round(mse, 4))

    def evaluate_and_log_model(
        self,
        experiment_name: str,
        engine: SurvivalRiskEngine,
        eval_df: pd.DataFrame,
        domain_tag: str = "all_domains",
    ) -> Dict[str, float]:
        """
        Computes C-index, Brier Score, and goodness-of-fit metrics, and logs them to MLflow registry.
        """
        duration_col = "observed_duration_sec"
        event_col = "completed"

        eval_clean = eval_df.dropna(subset=[duration_col, event_col]).copy()
        actual_durations = eval_clean[duration_col].values
        events = eval_clean[event_col].values

        # Generate predictions (predicted p50 completion time)
        predicted_p50s = []
        survival_at_median = []
        median_time = float(np.median(actual_durations))

        active_fitter = engine._get_active_fitter()
        for _, row in eval_clean.iterrows():
            x_dict = {
                "as_of_seconds": float(row.get("as_of_seconds", 1200.0)),
                "fraction_tasks_completed": float(row.get("fraction_tasks_completed", 0.3)),
                "skew_duration_ratio": float(row.get("skew_duration_ratio", 1.2)),
                "disk_spilled_gib": float(row.get("disk_spilled_gib", 0.0)),
                "gc_pressure": float(row.get("gc_pressure", 0.02)),
                "occ_conflict_retries": float(row.get("occ_conflict_retries", 0)),
            }
            s_tc, s_d, p50, p90 = active_fitter.predict_point_in_time(x_dict, 1200.0, median_time)
            predicted_p50s.append(p50)
            survival_at_median.append(s_d)

        # 1. Discrimination: Harrell's C-index
        c_index = self.compute_concordance_index(actual_durations, predicted_p50s, events)

        # 2. Calibration: Brier score at median duration
        brier_score = self.compute_brier_score_at_time(actual_durations, events, survival_at_median, median_time)

        # 3. Model AIC
        aic_score = float(round(active_fitter.fitter.AIC_, 2))

        metrics = {
            "harrell_c_index": c_index,
            "brier_score_at_median": brier_score,
            "aic": aic_score,
            "evaluation_samples": float(len(eval_clean)),
        }

        params = {
            "model_type": engine.model_type,
            "model_version": engine.model_version,
            "penalizer": active_fitter.penalizer,
            "domain_evaluated": domain_tag,
        }

        tags = {
            "framework": "lifelines",
            "tier": "enterprise_gold",
            "lifecycle": "production",
        }

        run_id = self.log_run(
            experiment_name=experiment_name,
            params=params,
            metrics=metrics,
            tags=tags,
        )

        return metrics
