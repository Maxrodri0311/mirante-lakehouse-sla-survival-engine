# 📐 SPEC & BLUEPRINT: Mirante Lakehouse SLA Survival Engine

**Target Company:** Mirante Tecnologia (Brazil Enterprise Public & Financial Consulting)  
**Target Role:** Data Scientist  
**Delivery Paradigm:** `EXPLAINABLE_ANALYTICS & CLI_TUI`  
**Core Algorithm:** `Accelerated Failure Time (Weibull / Log-Logistic) & Time-Varying Hazard Modeling + Causal FinOps Decision Engine`  
**Repository:** `https://github.com/Maxrodri0311/mirante-lakehouse-sla-survival-engine.git`  

---

## 🏛️ 1. The Core Business Bottleneck
Mirante Tecnologia manages mission-critical distributed data pipelines on Apache Spark & Delta Lake for Brazilian tier-1 institutions (Pix Clearing, Central Bank BACEN regulatory reporting, Superior Court STJ analytics, Receita Federal tax reconciliation). 
Traditional monitoring relies on naive static thresholds (e.g. alerts when a job runs > 80% SLA), which causes:
1. **False Alarms & Alert Fatigue:** High-throughput jobs naturally vary; static alarms trigger unnecessary operational disruptions.
2. **Late Breach Detection:** Silent operational degradation (partition skew, shuffle disk spill, JVM GC pauses, Delta Lake OCC write contention) manifests early in execution ($t_c$) but is only detected when the deadline is already mathematically unrecoverable.
3. **Counter-Productive Interventions:** Naively adding executors to a skewed join straggler aggravates cluster cost without reducing runtime, whereas adaptive query execution (AQE) skew-join salting resolves it at sub-5% cost.

---

## ⚖️ 2. Domain Entities & Telemetry Variables

### Domain Entities
- **`PipelineRun`**: Physical lifecycle record of an executed batch pipeline (`run_id`, `business_domain`, `contract_sla_sec`, `observed_duration_sec`, `final_status`, `event_breach`, `is_censored`, `total_compute_cost_usd`, `penalty_fee_usd`).
- **`PipelineObservationSnapshot`**: Point-in-time runtime telemetry snapshot observed strictly as-of $t_c$ (`snapshot_id`, `run_id`, `as_of_seconds`, `remaining_to_sla_seconds`, `fraction_tasks_completed`, `tasks_active`, `skew_duration_ratio`, `disk_spilled_gib`, `gc_pressure`, `occ_conflict_retries`). Zero label leakage guaranteed.
- **`RiskEstimate`**: Dynamic survival prediction at $t_c$ estimating conditional breach probability:
  $$P(\text{Breach} \mid T^C > t_c, X(t_c)) = \frac{S^C(D \mid X(t_c))}{S^C(t_c \mid X(t_c))}$$
- **`CausalDecision`**: Prescriptive intervention minimizing expected loss:
  $$\arg\min_{a \in \mathcal{A}} \mathbb{E}[L(a \mid H_t)] = C_{\text{action}}(a) + P(\text{Breach} \mid do(a), H_t) \cdot \text{Penalty}$$

---

## 🎙️ 3. Architectural Defense & Interview Edge

### ❓ Question 1: Why model completion time $T^C$ with AFT survival models instead of a standard binary classifier?
> **💡 Strategic Answer:**  
> *"A binary classifier predicts breach at an arbitrary fixed cutoff, collapsing temporal dynamics and discarding right-censored runs (jobs cancelled or running at administrative cutoffs). By formulating execution under Accelerated Failure Time (AFT Weibull/Log-Logistic), we model the entire survival curve $S^C(t \mid X(t_c))$ and calculate the exact conditional breach probability $\frac{S^C(D)}{S^C(t_c)}$ dynamically at any observation snapshot $t_c$, while maintaining full statistical rigor with Greenwood variance bounds."*

### ❓ Question 2: Why separate prognosis from causal action $P(\text{Breach} \mid do(a))$?
> **💡 Strategic Answer:**  
> *"A survival model diagnoses the prognosis (probability of breach under current trajectory), but prescribing an action requires causal inference. For instance, if the root cause is extreme partition skew (a single straggler partition), scaling cluster workers has zero causal acceleration effect because the remaining work is purely serial. Our causal policy matches the intervention mechanism to the bottleneck: AQE skew join salting for skew, worker scaling for memory/disk spill with high CPU, and exponential backoff with jitter for Delta Lake OCC write contention."*