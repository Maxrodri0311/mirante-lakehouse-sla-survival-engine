"""
src/domain/contracts.py - Abstract Domain Protocols (Dependency Inversion Principle - DIP).
Enforces strict decoupling: domain models, survival estimators, and decision policies depend
exclusively on these abstract interfaces, never on concrete DuckDB, Spark, Azure, or Databricks SDKs.
"""

from typing import runtime_checkable, Protocol, Optional, List, Dict, Any, Sequence
import pandas as pd
from src.domain.entities import (
    PipelineRun,
    PipelineObservationSnapshot,
    RiskEstimate,
    CausalDecision,
)


@runtime_checkable
class AnalyticalStorageProtocol(Protocol):
    """
    Abstract contract for high-throughput OLAP query execution (DuckDB, Athena, In-Memory Mock).
    """
    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes arbitrary SQL query and returns a pandas DataFrame."""
        ...

    def table_exists(self, table_name: str) -> bool:
        """Returns True if the view or table exists in the analytical catalog."""
        ...

    def register_view(self, view_name: str, query_or_df: Any) -> None:
        """Registers a virtual view or table over external Parquet/Delta sources."""
        ...


@runtime_checkable
class TelemetrySourceProtocol(Protocol):
    """
    Abstract contract for loading execution logs and snapshot observations.
    """
    def read_runs(self, domain_filter: Optional[str] = None) -> pd.DataFrame:
        """Loads physical pipeline execution records."""
        ...

    def read_snapshots(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """Loads runtime observation snapshots as-of t_c."""
        ...


@runtime_checkable
class CompletionRiskModelProtocol(Protocol):
    """
    Abstract contract for Completion Survival Risk Models (AFT Weibull, Log-Logistic, Cox PH).
    Estimates P(T_complete > D | T_complete > t_c, X(t_c)).
    """
    def predict_risk(self, snapshot: PipelineObservationSnapshot) -> RiskEstimate:
        """Calculates conditional breach probability and identifies dominant risk mechanism."""
        ...

    def predict_survival(self, snapshot: PipelineObservationSnapshot, horizon_seconds: float) -> float:
        """Calculates survival probability S(horizon | X(t_c))."""
        ...


@runtime_checkable
class CausalPolicyProtocol(Protocol):
    """
    Abstract contract for Causal FinOps Remediation Policies.
    Evaluates feasible interventions: NO_OP, AQE_COALESCE, SCALE_WORKERS, BACKOFF_OCC.
    """
    def evaluate(self, snapshot: PipelineObservationSnapshot, risk: RiskEstimate) -> CausalDecision:
        """Evaluates minimum expected financial loss and prescribes action."""
        ...


@runtime_checkable
class MLflowTrackingProtocol(Protocol):
    """
    Abstract contract for MLOps tracking and experiment governance.
    """
    def log_run(
        self,
        experiment_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> str:
        """Logs run metadata, metrics, and artifact references."""
        ...