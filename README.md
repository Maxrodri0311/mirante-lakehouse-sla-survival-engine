<!-- [SYSTEM INSTRUCTION]
Blueprint: GP-170 | Target: Mirante Tecnologia - Data Scientist
Paradigm: DeliveryParadigm.CLI_TUI | Core Algorithm: AlgorithmFamily.WEIBULL
Latency Targets: p95 < 150.0ms, p99 < 500.0ms | Max RAM: 512MB
-->

<div align="center">

# Mirante Tecnologia Data Scientist Bridge Project (`GP-170`)

### Enterprise-Grade DeliveryParadigm.CLI_TUI & AlgorithmFamily.WEIBULL Engine

[![CI & Quantitative Benchmarks](https://github.com/Maxrodri0311/mirante-tecnologia-data-scientist-bridge-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Maxrodri0311/mirante-tecnologia-data-scientist-bridge-project/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[⚡ Run Live Demo](run_demo.bat)** &nbsp;•&nbsp; **[📐 Architecture Spec](00_SPEC.md)** &nbsp;•&nbsp; **[🧪 Pytest Suite](tests/)**

</div>

---

## 🏛️ 1. Executive Summary & Core Bottleneck
Designed specifically for **Mirante Tecnologia** under the **Data Scientist** requirements.

Mirante Tecnologia requires an enterprise-grade Causal & Survival Lifecycle Analytics architecture under Data Scientist to solve operational latency, resource allocation bottlenecks, and provide C-Level visibility.

```mermaid
flowchart TD
    A[Stochastic Ingestion Layer] --> B[Decoupled Analytical Engine (DIP)]
    B --> C[AlgorithmFamily.WEIBULL Optimization & Scoring]
    C --> D[DeliveryParadigm.CLI_TUI Delivery Interface]

    style A fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#FFFFFF
    style B fill:#0F172A,stroke:#64748B,stroke-width:2px,color:#FFFFFF
    style C fill:#1E293B,stroke:#10B981,stroke-width:2px,color:#FFFFFF
    style D fill:#0F172A,stroke:#F59E0B,stroke-width:2px,color:#FFFFFF
```

---

## ⚖️ 2. Architectural Trade-Offs Considered
- **Selected Approach:** AlgorithmFamily.WEIBULL with Clean Architecture & Dependency Inversion Principle (DIP).
- **Delivery Paradigm:** DeliveryParadigm.CLI_TUI.
- **Calibrated Invariant:** Modelado Dinámico Temporal vs Agregaciones Estáticas Tradicionales.
- **Target Performance:** Latency p95 < 150.0 ms, Memory < 512 MB.

---

## 🧪 3. Acceptance Criteria
- [x] Pass 100% of automated Pytest suite without regressions.
- [x] Achieve query latency p95 < 150ms over 50,000+ domain records.
- [x] Enforce strict Dependency Inversion Principle (DIP) with decoupled domain protocols.

---

## ⚡ 4. 1-Click Verification & Benchmarks

```bash
# 1. Clone repository
git clone https://github.com/Maxrodri0311/mirante-tecnologia-data-scientist-bridge-project.git
cd mirante-tecnologia-data-scientist-bridge-project

# 2. Run full automated pipeline, tests & benchmarks in 1 command
run_demo.bat
```

---

## 👤 Author & Canonical Profile
- **Engineer:** Maximiliano Rodriguez
- **Email:** [maxrodri0311@gmail.com](mailto:maxrodri0311@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/maximiliano-rodriguez-982674375/](https://www.linkedin.com/in/maximiliano-rodriguez-982674375/)
- **GitHub:** [https://github.com/Maxrodri0311](https://github.com/Maxrodri0311)