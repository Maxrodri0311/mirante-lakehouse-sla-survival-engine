"""
src/domain/entities.py - Pure Domain Entities for Mirante Lakehouse SLA Survival Engine.
Defines immutable data models for Spark/Delta Lake telemetry snapshots, pipeline execution lifecycle,
actuarial risk estimates, and causal FinOps intervention decisions.
Strictly decoupled: Zero external I/O or vendor SDKs imported here.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


FinalStatusType = Literal[
    "SUCCESS_ON_TIME",
    "RUNNING_PAST_DEADLINE",
    "FAILED_TERMINAL",
    "CANCELLED",
]

RiskMechanismType = Literal[
    "NOMINAL",
    "SKEW",
    "SPILL",
    "GC_PRESSURE",
    "OCC_CONTENTION",
    "CAPACITY_STARVATION",
]

InterventionActionType = Literal[
    "NO_OP",
    "AQE_COALESCE_SKEW_JOIN",
    "SCALE_WORKERS_MEMORY",
    "BACKOFF_OCC_SERIALIZE",
    "RESTART_CLEAN",
]


class PipelineRun(BaseModel):
    """
    Physical execution record of a distributed Spark pipeline run on Delta Lake.
    """
    run_id: str = Field(..., description="Unique physical run identifier")
    job_id: str = Field(..., description="Logical Spark/Databricks job identifier")
    job_name: str = Field(..., description="Human-readable pipeline name")
    business_domain: str = Field(..., description="Business domain (e.g. Banking, Regulatory, Legal)")
    cluster_id: str = Field(..., description="Cluster identifier hosting the run")
    spark_version: str = Field(default="3.5.0", description="Spark engine version")
    contract_sla_sec: float = Field(..., description="Contractual SLA deadline D in seconds from start")
    scheduled_start_ts: str = Field(..., description="Scheduled execution timestamp (ISO 8601)")
    actual_start_ts: str = Field(..., description="Actual execution start timestamp")
    completion_ts: Optional[str] = Field(default=None, description="Completion timestamp if terminated")
    observed_duration_sec: float = Field(..., description="Total elapsed runtime or terminal duration")
    final_status: FinalStatusType = Field(..., description="Terminal state of the execution")
    event_breach: bool = Field(..., description="True if run breached contractual SLA (duration > D or terminal failure)")
    is_censored: bool = Field(..., description="True if duration is right-censored (e.g. active run, administrative cutoff)")
    total_compute_cost_usd: float = Field(default=0.0, description="Realized compute cost in USD")
    penalty_fee_usd: float = Field(default=25000.0, description="Contractual penalty if SLA breached")


class PipelineObservationSnapshot(BaseModel):
    """
    As-of observation snapshot of an active Spark pipeline at time t_c.
    Contains strictly available runtime telemetry without label leakage.
    """
    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    run_id: str = Field(..., description="Foreign key to PipelineRun")
    job_name: str = Field(..., description="Pipeline name for contextual routing")
    business_domain: str = Field(..., description="Business domain")
    as_of_seconds: float = Field(..., description="Elapsed runtime t_c at snapshot observation time")
    contract_sla_seconds: float = Field(..., description="Contractual deadline D")
    remaining_to_sla_seconds: float = Field(..., description="Remaining time to contractual deadline (D - t_c)")
    
    # 1. Task Execution & Progress Metrics
    fraction_tasks_completed: float = Field(default=0.0, ge=0.0, le=1.0)
    tasks_active: int = Field(default=32, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    
    # 2. Shuffle & Memory Spilling Telemetry
    shuffle_read_gib: float = Field(default=0.0, ge=0.0)
    shuffle_write_gib: float = Field(default=0.0, ge=0.0)
    memory_spilled_gib: float = Field(default=0.0, ge=0.0)
    disk_spilled_gib: float = Field(default=0.0, ge=0.0)
    spill_ratio_disk_to_mem: float = Field(default=0.0, description="diskSpilled / (memorySpilled + eps)")
    spill_ratio_output: float = Field(default=0.0, description="diskSpilled / (shuffleWriteBytes + eps)")
    fetch_wait_ratio: float = Field(default=0.0, description="fetchWaitTime / (executorRunTime + eps)")

    # 3. Partition Skew & Tail Amplification Telemetry
    skew_duration_ratio: float = Field(default=1.0, description="max(taskDuration) / median(taskDuration)")
    skew_bytes_cv: float = Field(default=0.1, description="Coefficient of variation in partition bytes")
    tail_amplification: float = Field(default=1.0, description="p99(taskDuration) / p50(taskDuration)")

    # 4. JVM, CPU & Memory Contention
    jvm_gc_time_ms: float = Field(default=0.0, ge=0.0)
    executor_cpu_time_ms: float = Field(default=0.0, ge=0.0)
    executor_run_time_ms: float = Field(default=1.0, ge=0.0)
    gc_pressure: float = Field(default=0.02, description="jvmGCTime / executorRunTime (GC/wall)")
    gc_cpu_ratio: float = Field(default=0.03, description="jvmGCTime / executorCpuTime (GC/CPU)")
    executor_count: int = Field(default=16, ge=1)
    hourly_cluster_cost_usd: float = Field(default=24.0, description="Cluster operating burn rate ($/hr)")

    # 5. Delta Lake ACID & Concurrency Telemetry
    delta_table_name: str = Field(default="gold_transactions")
    delta_version: int = Field(default=10, ge=0)
    occ_conflict_retries: int = Field(default=0, ge=0)
    commit_duration_ms: float = Field(default=120.0, ge=0.0)
    occ_contention_ratio: float = Field(default=0.0, description="conflictRetries / commitAttempts")
    small_file_ratio: float = Field(default=0.05, description="smallFiles / totalFiles")


class RiskEstimate(BaseModel):
    """
    Output of the Completion Survival Risk Model at as-of time t_c.
    Answers: What is the residual probability that this pipeline breaches the SLA deadline D?
    """
    run_id: str
    as_of_seconds: float
    breach_probability: float = Field(..., ge=0.0, le=1.0, description="P(T_complete > D | T_complete > t_c, X(t_c))")
    survival_to_sla: float = Field(..., ge=0.0, le=1.0, description="S(D | X(t_c))")
    survival_to_now: float = Field(..., ge=0.0, le=1.0, description="S(t_c | X(t_c))")
    p50_completion_seconds: float = Field(..., description="Estimated median time of completion")
    p90_completion_seconds: float = Field(..., description="Estimated 90th percentile completion time")
    dominant_risk_mechanism: RiskMechanismType = Field(default="NOMINAL")
    mechanism_severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    model_version: str = Field(default="aft-weibull-v1.0")


class CausalDecision(BaseModel):
    """
    Output of the Causal FinOps Decision Policy.
    Minimizes expected loss: E[Loss(a)] = C_action(a) + P(Breach | do(a)) * C_breach.
    """
    decision_id: str
    run_id: str
    as_of_seconds: float
    action: InterventionActionType
    reason: str
    prob_breach_without_action: float = Field(..., ge=0.0, le=1.0)
    prob_breach_with_action: float = Field(..., ge=0.0, le=1.0)
    action_cost_usd: float = Field(..., ge=0.0)
    contract_penalty_usd: float = Field(..., ge=0.0)
    expected_loss_without_action_usd: float = Field(..., ge=0.0)
    expected_loss_with_action_usd: float = Field(..., ge=0.0)
    expected_net_savings_usd: float
    should_intervene: bool = Field(..., description="True if action net savings > 0 and feasibility criteria met")