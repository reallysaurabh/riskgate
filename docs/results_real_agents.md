# Results on real agents (public benchmarks)

> [!IMPORTANT]
> These are **offline replays** of logs published with public benchmarks. Each logged consequential action is one
> decision. Its context is the conversation and environment state before the action, and its label is the
> benchmark's ground truth.
> - What would have happened after an `ASK` is not observed.
> - Customers in τ²-bench are LLM-simulated.
> - Some components are synthetic or rule-derived; each report says which.
>
> The gate never calls an LLM. Protocol: [benchmarks/README.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/README.md).

## Protocol in one paragraph

Tasks (scenarios, incidents) are the exchangeable units, so calibration and test data are split by task and
calibration runs on task-mean losses (`Gate.calibrate(groups=...)`). The guarantee then bounds the expected share of a
new task's decisions that are certified and unsafe. Hypothesis families and weights are frozen on a named development
subset before the held-out logs are evaluated. Baselines:

- no checker;
- the domain's own deterministic rule;
- top-1 hypothesis + CRC;
- RiskGate with CRC and with LTT (δ' = 0.1).

Validity is reported as the mean over 200 random task re-splits.

## τ²-bench (retail, airline): eight 2026 frontier agents and four 2025 agents

Source: the τ²-bench leaderboard's public trajectories (Sierra Research, MIT; model outputs remain subject to their
providers' terms). 9,759 database writes, each replayed in the benchmark's own environment with zero mismatches.

| | Retail (2026 agents) | Airline (2026 agents) |
|---|---|---|
| Wrong writes (match no gold write, final database wrong) | 5.0–9.6% | 8.9–26.5% |
| Wrong writes right after an explicit customer "yes" | 355 of 363 | — |
| RiskGate α = 10%: unsafe / ask, per agent | 7.3–8.9% / 6–28% | 4.6–5.9% / 35–80% |
| RiskGate α = 5%: unsafe / ask, per agent | 3.2–4.2% / 53–70% | infeasible (< 19 calibration tasks) |
| AUROC of the LLM-free gate | 0.49–0.65 | 0.34–0.78 |
| Exploratory: readings from other agents' runs, AUROC / ask at α = 10% | 0.92–0.96 / 1–5% | 0.97–1.00 / 11–34% |

Key observations:

- **Validity held for every agent.**
- **Top-1 + CRC asks about everything.** This is Proposition 1 seen on real logs.
- **The policy's "explicit yes" rule removes almost nothing.** Task-averaged unsafe goes only from 13.8% to 13.3%.
- **Recalibrate on agent upgrades.** A threshold calibrated on 2025 agents gives 14.6% unsafe on 2026 agents in airline at α = 10%. Recalibrating gives 6.4%.
- **The committee readings are optimistic.** Each logged run had its own conversation with the simulated customer.

Full report: [tau2_report.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/results/tau2_report.md).

## AgentDojo: prompt-injection safety across 28 model and defence pipelines

Source: logged runs published by the AgentDojo authors (ETH Zurich, MIT; model outputs remain subject to providers' terms). 35,390 non-DoS runs replayed in the AgentDojo environment, yielding 20,833 state-changing tool calls on the main evaluation (`important_instructions` + benign runs across 28 evaluable pipelines of 22 base models) and 35,603 state-changing calls including 9 held-out attack types.

- **Label agreement:** per-call rule labels agree with AgentDojo's own run-level `security` flag in **97.1%** of 18,337 evaluated runs and label the completing attack call unsafe in **99.9%** of successful attacks.
- **Validity across all 27 held-out pipelines:** at α = 5%, grouped CRC (`(suite, user_task)` units) keeps the 200-resplit mean unsafe-certified rate at **0.2–3.1%** (median **2.0%**, against a median base unsafe rate of **29.3%** and up to **57.3%**), with median score AUROC **0.88** (0.79–0.96).
- **Run-level safety and autonomy:** assuming an `ASK`ed unsafe call is refused, median attack success drops from **12.3%** to **1.5%** at **100%** median benign-run autonomy. A fixed rule that blocks every call with an untrusted entity stops attacks about as well (median 1.6%) but lets only 68% of benign runs through without a human. On `Meta-SecAlign-70B`, attack success drops from 2.2% to **0.3%** at 100% benign autonomy (8.6% ask rate).
- **Shift and transfer:** calibrating on `important_instructions` transfers to all six template variants (0.1–2.4% unsafe), whereas shorter attacks (`direct`, `ignore_previous`, `injecagent`) carry fewer instruction cues and exceed α on the Llama-3.3-70B and GPT-4o pipelines (7.0–13.1% unsafe; Meta-SecAlign-70B stays at 3.6–4.6%) until recalibrated (1.0–2.6% unsafe). Swapping the agent without recalibrating exceeds α in 35% of cross-pipeline pairs.

Full report: [agentdojo_report.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/results/agentdojo_report.md).

## SRE

- **ITBench-Trajectories** (real GPT-OSS-120B agent, CC BY-NC).
  - Root-cause claims are right in 17.6% of 85 runs, and re-running and voting does not help.
  - An evidence-consistency gate separates right from wrong claims well (held-out AUROC 0.90).
  - 35 scenarios are too few for α ≤ 10%. At α = 20%, unsafe is 10.8% and asks are 76%.
- **RCAEval RE1-OB** (real fault-injection telemetry).
  - The conformal set over 55 (service, fault) hypotheses keeps the true cause 96.2% of the time at δ = 5% (mean size 4.2).
  - Coverage drops under fault-type shift (packet loss: 65%).
  - A remediation gate with rule-derived labels gives 3.6% unsafe at α = 5%.

Reports: [itbench_report.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/results/itbench_report.md),
[rcaeval_report.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/results/rcaeval_report.md).

## Refunds: Refund Arena (synthetic, Sep 2026)

- The database settles every policy clause: the dry-run reproduces all 445 gold labels.
- A deterministic policy engine with the top-1 reading is therefore already exact on held-out tickets.
- With 15 calibration groups the conformal layer adds a guarantee and little else.
- A severity-weighted margin let CRC certify cheap forbidden refunds on one split. When labels are binary, use binary
  margins.

Report: [refund_arena_report.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/results/refund_arena_report.md).

## Plant data: Tennessee Eastman (simulated plant)

Rieth et al. 2017 provide 500 independent runs per fault (public domain), so runs are the exchangeable units.

- An LLM-free RDA fault scorer gets 80.0% top-1 accuracy over 21 classes.
- The conformal plausible set covers 95.0% ± 0.3% of held-out windows at δ = 5% (mean size 2.1), and 90.0% at δ = 10% (size 1.75).
- For the undetectable IDV 3, 9 and 15 it widens to about 4.7 classes at δ = 5% (3.5 at δ = 10%), and stays near a singleton for IDV 1, 2, 6 and 7.
- Marginal coverage is not conditional: hard faults are covered 78.1% at δ = 10%.
- A small, partly rule-derived decision layer gives 1.3% unsafe at α = 10% (top-1 twin gate: 1.6%), at 9.3% versus 8.0% asks.

Report: [tep_report.md](https://github.com/reallysaurabh/riskgate/blob/main/benchmarks/results/tep_report.md).

## Reproduce

```bash
.venv-public/bin/python benchmarks/tau2/extract.py && .venv-public/bin/python benchmarks/tau2/run.py full && .venv-public/bin/python benchmarks/tau2/report.py
.venv-public/bin/python benchmarks/sre/run.py itbench && .venv-public/bin/python benchmarks/sre/run.py rcaeval
.venv-public/bin/python benchmarks/refund_arena/run.py
```

Data sources and licences are listed in each module's docstring; the public files are cached outside the package.
