# `benchmarks/`: RiskGate on real, logged agent decisions from public benchmarks

`examples/` evaluates RiskGate on simulated domains with scripted doers. `benchmarks/` evaluates the **same library**
on **public benchmark data**: logged trajectories of real LLM agents where they exist (τ²-bench, AgentDojo,
ITBench), and real telemetry or benchmark labels where no agent runs are published (RCAEval, Refund Arena, TEP).
No LLM is called anywhere: the gate is LLM-free by design (rule D1), so it can be evaluated offline on the logs.

## Protocol (every experiment follows it; deviations are stated in its report)

| # | Rule |
|---|---|
| P1 | **No leakage.** Gold actions, hidden user instructions, injection identities and fault labels are used only as labels (and, for the plausible-set level, as calibration targets), never as gate inputs. |
| P2 | **Exchangeable units are groups** (task, scenario, incident, ticket pair), never single steps. Splits are drawn by group; conformal risk control runs on each group's mean loss (`Gate.calibrate(groups=...)`). Guarantee: `E[share of a new group's decisions that are certified and unsafe] <= alpha`. |
| P3 | **Frozen design.** Hypothesis/score design and any hand-set weights are fixed on a development subset that is named in the report, before the held-out data are evaluated. |
| P4 | **Baselines.** `none` (execute everything), the domain's natural deterministic rule (`fixed`, no calibration), top-1 reading + CRC, RiskGate (CRC), RiskGate (LTT, `delta'=0.1`). |
| P5 | **Validity.** Fixed split (50/50 by group, seed 0) with Wilson CIs, 200 random group re-splits (mean ± sd of the test unsafe rate, share of splits above alpha), alpha sweep. |
| P6 | **Efficiency.** Ask rate, safe-auto rate (correct actions certified), avoidable asks, run-level unsafe rate. |
| P7 | **Transfer.** Calibrate on one doer, test on another (policy-mode guarantees are doer-specific; report what happens). |
| P8 | **Honesty.** Each report separates what is real (agent behaviour, telemetry, benchmark labels), what is derived by rule, and what is simulated, and lists licences. Model outputs are not redistributed. |
| P9 | **Library path.** Scores come from a real `riskgate.Gate` (`gate.score_episode`), and each run asserts that `Gate.calibrate(cal_eps, groups=..., tau_fraction=0)` returns the harness's λ̂ on the fixed split. |

## Layout

```text
benchmarks/
├── common.py            grouped-CRC harness (Decision records, run_protocol, per_doer, auroc, dump)
├── tests/test_common.py harness == riskgate.Gate.calibrate(groups=...) (CRC and LTT)
├── tau2/                τ²-bench: 2026 leaderboard frontier models + 2025 models (retail, airline)
├── agentdojo/           AgentDojo: prompt-injection safety, 29 logged model/defence configurations
├── sre/                 ITBench-Trajectories (real SRE agent) and RCAEval (real telemetry)
├── refund_arena/        Refund Arena: refund gate over candidate writes
├── tep/                 Tennessee Eastman: fault-hypothesis sets and humility on plant data
└── results/             <name>_results.json and <name>_report.md per experiment
```

Data are cached outside the package in `../public_data/` (see each module's docstring for sources and licences).
Run everything with the experiment environment: `.venv-public/bin/python benchmarks/<name>/run.py`.
