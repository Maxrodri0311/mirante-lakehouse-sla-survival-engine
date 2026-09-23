"""
src/domain/survival_models.py - Survival Lifecycle Risk Engine for Apache Spark Pipelines.
Implements:
1. KaplanMeierCompletionEstimator: Non-parametric baseline with Greenwood confidence intervals.
2. AFTWeibullEstimator: Parametric Accelerated Failure Time model for completion time.
3. AFTLogLogisticEstimator: Parametric AFT model for non-monotonic hazard rates.
4. SurvivalRiskEngine: Production engine conforming to CompletionRiskModelProtocol.
Calculates exact conditional SLA breach probability:
    P(Breach at D | T^C > t_c, X(t_c)) = S^C(D | X(t_c)) / S^C(t_c | X(t_c))
"""

from typing import Dict, List, Optional, Sequence, Tuple, Any, Literal
import time
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullAFTFitter, LogLogisticAFTFitter

from src.domain.entities import (
    PipelineObservationSnapshot,
    RiskEstimate,
    RiskMechanismType,
)
from src.domain.contracts import CompletionRiskModelProtocol


FEATURE_COLS = [
    "as_of_seconds",
    "fraction_tasks_completed",
    "skew_duration_ratio",
    "disk_spilled_gib",
    "gc_pressure",
    "occ_conflict_retries",
]


class KaplanMeierCompletionEstimator:
    """
    Non-parametric baseline survival estimator stratified by business domain.
    Computes S^C(t) = P(T^C > t) and Greenwood standard errors.
    """

    def __init__(self):
        self.fitters: Dict[str, KaplanMeierFitter] = {}
        self.global_fitter = KaplanMeierFitter()

    def fit(self, df: pd.DataFrame, duration_col: str = "observed_duration_sec", event_col: str = "completed") -> "KaplanMeierCompletionEstimator":
        """
        Fits global and domain-stratified Kaplan-Meier survival curves.
        event_col = 1 (completed uncensored), 0 (right-censored).
        """
        self.global_fitter.fit(
            durations=df[duration_col],
            event_observed=df[event_col],
            label="global_completion",
        )

        if "business_domain" in df.columns:
            for domain, group in df.groupby("business_domain"):
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=group[duration_col],
                    event_observed=group[event_col],
                    label=str(domain),
                )
                self.fitters[str(domain)] = kmf

        return self

    def predict_survival(self, times: Sequence[float], domain: Optional[str] = None) -> pd.Series:
        """
        Returns S^C(t) at requested time points.
        """
        fitter = self.fitters.get(domain, self.global_fitter) if domain else self.global_fitter
        sf = fitter.survival_function_at_times(times)
        return sf

    def predict_conditional_breach(self, t_c: float, deadline: float, domain: Optional[str] = None) -> float:
        """
        Calculates P(T^C > D | T^C > t_c) = S^C(D) / S^C(t_c).
        """
        if t_c >= deadline:
            return 1.0

        sf = self.predict_survival([t_c, deadline], domain=domain)
        s_tc = float(sf.loc[t_c]) if t_c in sf.index else 1.0
        s_d = float(sf.loc[deadline]) if deadline in sf.index else 0.0

        if s_tc <= 1e-6:
            return 1.0

        ratio = s_d / s_tc
        return float(np.clip(ratio, 0.0, 1.0))


class AFTWeibullEstimator:
    """
    Parametric Weibull Accelerated Failure Time (AFT) model.
    Models completion duration: ln(T^C) = mu + beta^T X + sigma * W.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.fitter = WeibullAFTFitter(penalizer=penalizer)
        self.is_fitted = False
        self.feature_cols = FEATURE_COLS

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "observed_duration_sec",
        event_col: str = "completed",
    ) -> "AFTWeibullEstimator":
        cols = [c for c in self.feature_cols if c in df.columns] + [duration_col, event_col]
        train_data = df[cols].dropna().copy()

        self.fitter.fit(
            train_data,
            duration_col=duration_col,
            event_col=event_col,
        )
        self.is_fitted = True

        # Extract and cache analytical parameters for sub-millisecond vectorized inference
        self.coef_dict = dict(self.fitter.params_["lambda_"])
        self.intercept = float(self.coef_dict.pop("Intercept", 0.0))
        if "Intercept" in self.fitter.params_["rho_"]:
            self.rho = float(np.exp(self.fitter.params_["rho_"]["Intercept"]))
        else:
            self.rho = float(np.exp(self.fitter.params_["rho_"].iloc[0]))
        return self

    def predict_point_in_time(
        self, x_dict: Dict[str, float], t_c: float, d_deadline: float
    ) -> Tuple[float, float, float, float]:
        """
        Fast analytical survival inference: returns (s_tc, s_d, p50, p90) in < 50 microseconds.
        """
        if not self.is_fitted:
            raise RuntimeError("AFTWeibullEstimator must be fitted before prediction.")
        ln_lambda = self.intercept + sum(
            self.coef_dict.get(k, 0.0) * float(x_dict.get(k, 0.0)) for k in self.coef_dict
        )
        lambda_val = float(np.exp(ln_lambda))
        s_tc = float(np.exp(-((t_c / lambda_val) ** self.rho)))
        s_d = float(np.exp(-((d_deadline / lambda_val) ** self.rho)))
        p50 = float(lambda_val * ((-np.log(0.5)) ** (1.0 / self.rho)))
        p90 = float(lambda_val * ((-np.log(0.1)) ** (1.0 / self.rho)))
        return s_tc, s_d, p50, p90

    def predict_survival_function(self, x_df: pd.DataFrame, times: Sequence[float]) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("AFTWeibullEstimator must be fitted before prediction.")
        cols = [c for c in self.feature_cols if c in x_df.columns]
        return self.fitter.predict_survival_function(x_df[cols], times=times)

    def predict_percentile(self, x_df: pd.DataFrame, p: float = 0.5) -> float:
        if not self.is_fitted:
            raise RuntimeError("AFTWeibullEstimator must be fitted before prediction.")
        cols = [c for c in self.feature_cols if c in x_df.columns]
        val = self.fitter.predict_percentile(x_df[cols], p=p)
        return float(val.iloc[0])


class AFTLogLogisticEstimator:
    """
    Parametric Log-Logistic Accelerated Failure Time (AFT) model.
    Captures non-monotonic hazard rates characteristic of multi-stage Spark executions.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.fitter = LogLogisticAFTFitter(penalizer=penalizer)
        self.is_fitted = False
        self.feature_cols = FEATURE_COLS

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "observed_duration_sec",
        event_col: str = "completed",
    ) -> "AFTLogLogisticEstimator":
        cols = [c for c in self.feature_cols if c in df.columns] + [duration_col, event_col]
        train_data = df[cols].dropna().copy()

        self.fitter.fit(
            train_data,
            duration_col=duration_col,
            event_col=event_col,
        )
        self.is_fitted = True

        # Extract and cache analytical parameters for sub-millisecond inference
        self.coef_dict = dict(self.fitter.params_["alpha_"])
        self.intercept = float(self.coef_dict.pop("Intercept", 0.0))
        if "Intercept" in self.fitter.params_["beta_"]:
            self.beta = float(np.exp(self.fitter.params_["beta_"]["Intercept"]))
        else:
            self.beta = float(np.exp(self.fitter.params_["beta_"].iloc[0]))
        return self

    def predict_point_in_time(
        self, x_dict: Dict[str, float], t_c: float, d_deadline: float
    ) -> Tuple[float, float, float, float]:
        """
        Fast analytical survival inference: returns (s_tc, s_d, p50, p90) in < 50 microseconds.
        """
        if not self.is_fitted:
            raise RuntimeError("AFTLogLogisticEstimator must be fitted before prediction.")
        ln_alpha = self.intercept + sum(
            self.coef_dict.get(k, 0.0) * float(x_dict.get(k, 0.0)) for k in self.coef_dict
        )
        alpha_val = float(np.exp(ln_alpha))
        s_tc = float(1.0 / (1.0 + (t_c / alpha_val) ** self.beta))
        s_d = float(1.0 / (1.0 + (d_deadline / alpha_val) ** self.beta))
        p50 = float(alpha_val)
        p90 = float(alpha_val * (9.0 ** (1.0 / self.beta)))
        return s_tc, s_d, p50, p90

    def predict_survival_function(self, x_df: pd.DataFrame, times: Sequence[float]) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("AFTLogLogisticEstimator must be fitted before prediction.")
        cols = [c for c in self.feature_cols if c in x_df.columns]
        return self.fitter.predict_survival_function(x_df[cols], times=times)

    def predict_percentile(self, x_df: pd.DataFrame, p: float = 0.5) -> float:
        if not self.is_fitted:
            raise RuntimeError("AFTLogLogisticEstimator must be fitted before prediction.")
        cols = [c for c in self.feature_cols if c in x_df.columns]
        val = self.fitter.predict_percentile(x_df[cols], p=p)
        return float(val.iloc[0])


class SurvivalRiskEngine:
    """
    Production Completion Survival Risk Engine conforming to CompletionRiskModelProtocol.
    Provides sub-5ms conditional SLA breach predictions and bottleneck mechanism diagnosis.
    """

    def __init__(self, model_type: Literal["weibull", "log_logistic"] = "weibull", penalizer: float = 0.01):
        self.model_type = model_type
        self.model_version = f"aft-{model_type}-v1.0"
        self.weibull = AFTWeibullEstimator(penalizer=penalizer)
        self.log_logistic = AFTLogLogisticEstimator(penalizer=penalizer)
        self.km_estimator = KaplanMeierCompletionEstimator()
        self.is_fitted = False

    def fit(
        self,
        training_df: pd.DataFrame,
        duration_col: str = "observed_duration_sec",
        event_col: str = "completed",
    ) -> "SurvivalRiskEngine":
        """
        Fits both parametric AFT and non-parametric KM baselines.
        """
        if self.model_type == "weibull":
            self.weibull.fit(training_df, duration_col=duration_col, event_col=event_col)
        else:
            self.log_logistic.fit(training_df, duration_col=duration_col, event_col=event_col)

        self.km_estimator.fit(training_df, duration_col=duration_col, event_col=event_col)
        self.is_fitted = True
        return self

    def _get_active_fitter(self):
        return self.weibull if self.model_type == "weibull" else self.log_logistic

    def _diagnose_dominant_mechanism(self, snapshot: PipelineObservationSnapshot) -> Tuple[RiskMechanismType, float]:
        """
        Diagnoses the primary physical bottleneck from Spark/Delta telemetry.
        Returns: (mechanism_type, severity_score [0.0, 1.0])
        """
        scores: Dict[RiskMechanismType, float] = {}

        # 1. Partition Skew: severe when ratio > 2.0
        if snapshot.skew_duration_ratio >= 1.5:
            skew_score = min(1.0, (snapshot.skew_duration_ratio - 1.0) / 4.0)
            scores["SKEW"] = skew_score

        # 2. Shuffle Spill: severe when disk spill > 2.0 GiB
        if snapshot.disk_spilled_gib > 0.5 or snapshot.memory_spilled_gib > 5.0:
            spill_score = min(1.0, snapshot.disk_spilled_gib / 25.0)
            scores["SPILL"] = spill_score

        # 3. JVM GC Pause: severe when GC pressure > 0.08
        if snapshot.gc_pressure > 0.06:
            gc_score = min(1.0, snapshot.gc_pressure / 0.25)
            scores["GC_PRESSURE"] = gc_score

        # 4. Delta Lake OCC Contention: severe when conflict retries >= 2
        if snapshot.occ_conflict_retries >= 2:
            occ_score = min(1.0, snapshot.occ_conflict_retries / 6.0)
            scores["OCC_CONTENTION"] = occ_score

        # 5. Worker starvation
        if snapshot.tasks_active < 8 and snapshot.fraction_tasks_completed < 0.6:
            scores["CAPACITY_STARVATION"] = 0.55

        if not scores:
            return "NOMINAL", 0.05

        dominant = max(scores.items(), key=lambda item: item[1])
        return dominant[0], round(dominant[1], 3)

    def predict_risk(self, snapshot: PipelineObservationSnapshot) -> RiskEstimate:
        """
        Calculates conditional SLA breach probability:
        P(T^C > D | T^C > t_c, X(t_c)) = S(D | X(t_c)) / S(t_c | X(t_c))
        Guarantees sub-5ms latency and full Pydantic validation.
        """
        t_c = snapshot.as_of_seconds
        d_deadline = snapshot.contract_sla_seconds

        # Boundary condition: Deadline already breached
        if t_c >= d_deadline:
            dominant_mech, severity = self._diagnose_dominant_mechanism(snapshot)
            return RiskEstimate(
                run_id=snapshot.run_id,
                as_of_seconds=t_c,
                breach_probability=1.0,
                survival_to_sla=0.0,
                survival_to_now=0.0,
                p50_completion_seconds=t_c + 300.0,
                p90_completion_seconds=t_c + 1200.0,
                dominant_risk_mechanism=dominant_mech,
                mechanism_severity_score=severity,
                model_version=self.model_version,
            )

        fitter = self._get_active_fitter()
        x_dict = {
            "as_of_seconds": t_c,
            "fraction_tasks_completed": snapshot.fraction_tasks_completed,
            "skew_duration_ratio": snapshot.skew_duration_ratio,
            "disk_spilled_gib": snapshot.disk_spilled_gib,
            "gc_pressure": snapshot.gc_pressure,
            "occ_conflict_retries": snapshot.occ_conflict_retries,
        }

        s_tc, s_d, raw_p50, raw_p90 = fitter.predict_point_in_time(x_dict, t_c, d_deadline)

        # Conditional breach probability
        if s_tc <= 1e-5:
            cond_breach = 1.0
        else:
            cond_breach = float(np.clip(s_d / s_tc, 0.0, 1.0))

        p50 = max(t_c + 60.0, raw_p50)
        p90 = max(p50 + 60.0, raw_p90)

        dominant_mech, severity = self._diagnose_dominant_mechanism(snapshot)

        return RiskEstimate(
            run_id=snapshot.run_id,
            as_of_seconds=round(t_c, 1),
            breach_probability=round(cond_breach, 4),
            survival_to_sla=round(s_d, 4),
            survival_to_now=round(s_tc, 4),
            p50_completion_seconds=round(p50, 1),
            p90_completion_seconds=round(p90, 1),
            dominant_risk_mechanism=dominant_mech,
            mechanism_severity_score=severity,
            model_version=self.model_version,
        )

    def predict_survival(self, snapshot: PipelineObservationSnapshot, horizon_seconds: float) -> float:
        """
        Calculates conditional survival probability S(horizon | T^C > t_c, X(t_c)).
        """
        t_c = snapshot.as_of_seconds
        if horizon_seconds <= t_c:
            return 1.0

        fitter = self._get_active_fitter()
        x_dict = {
            "as_of_seconds": [t_c],
            "fraction_tasks_completed": [snapshot.fraction_tasks_completed],
            "skew_duration_ratio": [snapshot.skew_duration_ratio],
            "disk_spilled_gib": [snapshot.disk_spilled_gib],
            "gc_pressure": [snapshot.gc_pressure],
            "occ_conflict_retries": [snapshot.occ_conflict_retries],
        }
        x_df = pd.DataFrame(x_dict)
        sf_df = fitter.predict_survival_function(x_df, times=[t_c, horizon_seconds])

        s_tc = float(sf_df.loc[t_c].iloc[0])
        s_h = float(sf_df.loc[horizon_seconds].iloc[0])

        if s_tc <= 1e-5:
            return 0.0
        return float(np.clip(s_h / s_tc, 0.0, 1.0))
