# ITBench-Trajectories: a real LLM SRE agent's root-cause claims under an LLM-free evidence gate

> **Licence:** ITBench-Trajectories: CC BY-NC 4.0 (non-commercial research use; outputs not redistributed). Only aggregate numbers, entity names and signal classes are reported; no model output is redistributed.

## What is real, what is derived, what is simulated

| | |
|---|---|
| **Real** | 105 logged runs of a GPT-OSS-120B ReAct agent (ITBench-SRE-Agent) on 35 ITBench Kubernetes incident snapshots (3 runs each): its tool calls, the tool outputs it saw, and its final JSON diagnosis; the ground truth and the LLM judge's (gemini-2.5-pro) per-entity verdicts. |
| **Derived by rule (LLM-free)** | Root-cause hypotheses extracted from the agent's own tool outputs by a deterministic parser (signal classes: injected-fault object, spec change, quota, workload failure, symptom); the consequence model 'a remediation works iff it targets the true cause's entity'; an LLM-free name matcher used to cross-check the judge. |
| **Simulated** | Nothing. There is no remediation in the data: the decision we gate is the agent's root-cause claim, i.e. what a remediation would act on. |

## What the data contain

- The agent works on an **offline snapshot** and writes a diagnosis: root-cause entities (`contributing_factor=true`), impacted entities, a propagation chain, alert explanations. **It issues no remediation commands** (no kubectl writes), so 'each consequential command' does not exist; the consequential output is the root-cause claim.
- The judge grades root-cause entity precision/recall/F1, reasoning, propagation chain, component-level fault localisation and proximity metrics.
- 105 runs: 15 produced no diagnosis at all (crashed / empty sessions), 5 named no root cause, **85 runs made a root-cause claim** (mean 1.52 entities, max 15).
- Tool use was unreliable: 2888 tool calls (27.5 per run), of which 679 (23.5%) timed out and 319 (11.0%) failed on malformed arguments. 413 outputs only replayed the agent's own text (its patches, `jq . agent_output.json`) and are excluded from the evidence (otherwise the check would be circular).

## 1. Shadow replay (primary result: measuring the problem, no calibration)

| quantity | value |
|---|---|
| root-cause claim correct (entity level, judge) | **15/85 = 17.6%** (95% Wilson 11.0%–27.1%) |
| ... counting failed / empty runs as wrong | 15/105 = 14.3% |
| every claimed root cause correct (precision = 1) | 12/85 |
| right component, any entity type (judge fault localisation) | 49/85 |
| scenario pass@1 / pass@3 (35 scenarios) | 14.3% / 34.3% |
| scenarios where all 3 runs are right | 1 of 35 |
| claims naming an entity that never appears in the agent's own (non-echo) evidence | 12/85 (correct: 0) |

**By ground-truth category** (category read off the ground-truth ids, used for reporting only):

| category | scenarios | claims | entity correct | component correct |
|---|---|---|---|---|
| feature-flag config (flagd) | 13 | 32 | 0 | 15 |
| injected fault (chaos-mesh object) | 12 | 31 | 6 | 23 |
| other infrastructure (namespace, HPA, network policy) | 2 | 4 | 1 | 1 |
| workload (deployment/pod) | 7 | 18 | 8 | 10 |

**Run-to-run consistency** (first claimed root cause, normalised to the entity a remediation would target):

- 33 scenarios have >= 2 claiming runs; the first claim is unanimous in only 4 (1 unanimously right, **3 unanimously wrong**: Scenario-7 (`workload:frontend` × 2, truth ['flagd-config']), Scenario-21 (`workload:valkey-cart` × 3, truth ['otel-demo-valkey-memory-stress']), Scenario-83 (`workload:checkout` × 3, truth ['otel-demo-email-checkout-network-partition'])); 18/69 run pairs agree.
- Correctness across the runs of a scenario (scenarios with >= 1 claim): all claiming runs right: 1, all claiming runs wrong: 22, mixed: 11.
- Self-consistency is **not** a correctness signal here: runs that agree with another run are correct 18.5% of the time, runs that disagree 17.5% (n = 84). Re-running the agent and voting would not have made its claims safe.

**Judge agreement** (the labels are the judge's; we can only cross-check its matching step):

- An LLM-free name matcher (strip pod/replica-set hashes and numeric suffixes, map workload kinds and chaos object names) agrees with the judge on 97.7% of 129 claimed entities (Cohen's κ = 0.89; judge positives 15, matcher positives 16).
- Disagreements: `Scenario-24/1` Service/checkout vs ['checkout-deployment'] (judge ✗, matcher ✓); `Scenario-29/3` JVMChaos/otel-demo-ad-jvm-return-6rznj vs ['jvm-return-otel-demo-ad-1'] (judge ✓, matcher ✗); `Scenario-105/2` Pod/product-catalog-fcf7564cf-p8wgh vs ['product-catalog-deployment'] (judge ✗, matcher ✓). The judge is stricter on entity kind (a Pod or Service claim does not match a Deployment-level ground truth) and more lenient on reordered chaos-object names. The judge's entity list equals the agent's claim in 100% of runs.

## 2. Exploratory gate: LLM-free consistency of the claim with the agent's own evidence

**Design** (frozen on 10 development scenarios: Scenario-2, Scenario-4, Scenario-5, Scenario-6, Scenario-14, Scenario-23, Scenario-24, Scenario-34, Scenario-81, Scenario-102; the 25 others are held out). Every entity with a cause-type signal in the agent's tool outputs is a hypothesis with evidence score E (weights: inject 3, change 2.5, quota 2.5, fault 2, symptom 1; + a small volume term), nonconformity r = E_max − E, plus an `unobserved` sentinel (r = E_max + 1: the true cause left no trace). The claim's remediation works under h iff it targets h's entity (chaos object, namespace, or the faulty service's workload). Scores (all from `riskgate.Gate.score_episode`):
- **twin (top-1 evidence, fixed)**: certify iff the claim covers the best-supported hypothesis (λ = 0, no calibration);
- **top-1 + CRC**: + (lead of the top hypothesis over the runner-up) if the claim covers it, else −1 − lead;
- **RiskGate**: evidence gap = r(best hypothesis the claim does NOT cover, incl. `unobserved`) − r(best one it covers); −∞ if the claim covers nothing (→ REJECT: 'your root cause is not supported by your own evidence'). There is no hypothesis-level ground truth to calibrate a plausible set, so τ = ∞ (all hypotheses plausible).

**Data for the gate:** 59 claims in 24 held-out scenarios (groups); base unsafe rate 84.7% pooled, 85.4% as a mean over scenarios (the guaranteed quantity is the scenario-mean). The 50/50 split gives 28 calibration claims in 12 scenarios.

> [!IMPORTANT]
> CRC needs at least ⌈1/α⌉ calibration **groups** (scenarios), and with n groups the smallest achievable risk bound is 1/(n+1). With 12 calibration scenarios, **α = 5% is infeasible** (needs 20), **α = 10% is borderline** (it requires zero calibration losses at λ̂) and LTT at α = 10% cannot certify anything (Hoeffding–Bentkus with 12 groups cannot reject R > 0.1 at δ' = 0.1). These are exploratory numbers.

**Discrimination (no calibration):** AUROC of each score for 'claim correct'

| score | dev claims | held-out claims (95% scenario-bootstrap CI) | held-out: component-level label | held-out: precision=1 label |
|---|---|---|---|---|
| twin (±1) | 0.90 | **0.87** [0.74, 0.96] | 0.59 | 0.84 |
| top-1 confidence | 0.90 | **0.93** [0.85, 0.99] | 0.55 | 0.92 |
| RiskGate evidence gap | 0.85 | **0.90** [0.72, 1.00] | 0.67 | 0.86 |

**Common protocol, α = 10%** (held-out scenarios, groups = scenarios, fixed 50/50 split seed 0 + 200 group re-splits):

| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |
|---|---|---|---|---|---|---|---|---|---|
| none | −∞ | 77.8% [60.2%, 88.6%] | 0.0% | 22.2% | 0.0% | – | – | – | – |
| twin (top-1 evidence, fixed) | 0.000 | 9.7% [3.3%, 24.9%] | 68.1% | 22.2% | 0.0% | 10.9% ± 4.2% | 56% | 75.6% | 13.5% |
| top-1 + CRC | ∞ (certify nothing) | 0.0% [0.0%, 11.0%] | 100.0% | 0.0% | 22.2% | 2.3% ± 3.9% | 4% | 95.2% | 2.4% |
| riskgate (CRC) | ∞ (certify nothing) | 0.0% [0.0%, 11.0%] | 100.0% | 0.0% | 22.2% | 2.0% ± 2.9% | 0% | 94.2% | 3.8% |
| riskgate (LTT) | ∞ (certify nothing) | 0.0% [0.0%, 11.0%] | 100.0% | 0.0% | 22.2% | 0.0% ± 0.0% | 0% | 100.0% | 0.0% |

**Common protocol, α = 20%** (held-out scenarios, groups = scenarios, fixed 50/50 split seed 0 + 200 group re-splits):

| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |
|---|---|---|---|---|---|---|---|---|---|
| none | −∞ | 77.8% [60.2%, 88.6%] | 0.0% | 22.2% | 0.0% | – | – | – | – |
| twin (top-1 evidence, fixed) | 0.000 | 9.7% [3.3%, 24.9%] | 68.1% | 22.2% | 0.0% | 10.9% ± 4.2% | 0% | 75.6% | 13.5% |
| top-1 + CRC | 0.081 | 0.0% [0.0%, 11.0%] | 84.7% | 15.3% | 6.9% | 10.7% ± 8.9% | 14% | 77.3% | 12.0% |
| riskgate (CRC) | 0.081 | 0.0% [0.0%, 11.0%] | 77.8% | 22.2% | 0.0% | 10.8% ± 8.3% | 14% | 75.7% | 13.5% |
| riskgate (LTT) | ∞ (certify nothing) | 0.0% [0.0%, 11.0%] | 100.0% | 0.0% | 22.2% | 2.0% ± 2.9% | 0% | 94.2% | 3.8% |

Reading the tables:
- The natural deterministic rule (*twin*: certify iff the claim names the best-supported entity) cuts the unsafe rate from ~78–85% to 10.9% ± 4.2% on average, but it is uncalibrated: it exceeds α = 10% in 56% of the re-splits.
- At α = 10% the calibrated methods refuse to certify on the fixed split (λ̂ = ∞) and certify ~2–4% of claims on average over re-splits: the bound is respected, the gate is nearly useless — the group count, not the score, is the binding constraint. At α = 20% CRC lands on average at the twin's operating point with a guarantee; LTT stays (almost) silent.
- The top-1 confidence score discriminates as well as the RiskGate gap here (AUROC table); with one candidate per decision and binary 'targets the cause or not' effects the two scores are near-monotone transforms of each other whenever the claim covers the best-supported hypothesis.

At α = 20% (fixed-split λ̂ = 0.081), on the test half of the fixed split, by ground-truth category: feature-flag config (flagd): 0/8 certified (0 unsafe); injected fault (chaos-mesh object): 4/16 certified (0 unsafe); other infrastructure (namespace, HPA, network policy): 0/2 certified (0 unsafe); workload (deployment/pod): 3/5 certified (0 unsafe).

**P9 (library path):** `Gate.calibrate(cal_eps, groups=scenarios, tau_fraction=0)` reproduces the harness's λ̂ on the fixed split for riskgate (CRC) @ alpha=0.1 (λ̂ = ∞ (certify nothing)), riskgate (LTT) @ alpha=0.1 (λ̂ = ∞ (certify nothing)), top-1 + CRC @ alpha=0.1 (λ̂ = ∞ (certify nothing)), riskgate (CRC) @ alpha=0.2 (λ̂ = 0.081), riskgate (LTT) @ alpha=0.2 (λ̂ = ∞ (certify nothing)), top-1 + CRC @ alpha=0.2 (λ̂ = 0.081).

## Examples (held-out)

- **Scenario-22/2** (injected fault (chaos-mesh object)): claim `otel-demo/StressChaos/otel-demo-ad-memory-stress`; judge: correct; RiskGate score 4.080. Best-supported hypotheses in the agent's evidence: `inject:otel-demo-ad-memory-stress` (inject, r = 0.00), `unobserved` (none, r = 4.08).
- **Scenario-21/3** (injected fault (chaos-mesh object)): claim `otel-demo/Pod/valkey-cart-58df56c79c-4cvft, otel-demo/Deployment/valkey-cart`; judge: wrong; RiskGate score 3.132. Best-supported hypotheses in the agent's evidence: `workload:valkey-cart` (fault, r = 0.00), `unobserved` (none, r = 3.13).
- **Scenario-1/1** (feature-flag config (flagd)): claim `otel-demo/Deployment/cart`; judge: wrong; RiskGate score −∞. Best-supported hypotheses in the agent's evidence: `unobserved` (none, r = 0.00).

Reading: the top example is the typical certified claim (the agent named the chaos-mesh object whose events it had seen). The highest-scoring wrong claim is the gate's blind spot: the agent saw only workload-failure signals of the right service and named the workload, while the ground truth is the fault-injection object it never looked at — no consistency check on the agent's own evidence can see a cause the agent never observed; only the calibrated threshold (and its group count) protects against it. The −∞ example is a claim with no support at all in the agent's evidence (REJECT).

## Caveats

- **Few groups.** 24 held-out scenarios with claims; every CRC/LTT number above rests on 12 calibration scenarios. The fixed-split results at α = 20% (zero unsafe) and the re-split means are consistent with the guarantee, but the spread is large; this is a feasibility result, not a validated deployment.
- **Design not blind to label types.** A per-run label overview of all scenarios was printed once before the parser was written (which entity types tend to be right: injected-fault objects, deployments). Signal classes come from generic SRE triage practice and were checked on the 10 development scenarios only; categories that only occur in held-out ground truth (network policies, HPAs, feature-flag config) have no parser rule.
- **Benchmark artefact.** The strongest evidence class is the chaos-mesh object of the injected fault, visible in the snapshot. Production incidents have no such label; change events (rollouts, config edits) are the analogue. 13/35 scenarios are feature-flag (flagd) faults that the agent never got right and never collected evidence about: the gate can only refuse such claims when their support is weak (symptom-level), and a symptom-level claim that narrowly leads the evidence ranking can still clear a low λ̂; it can never name the true cause.
- **Labels are LLM-judge verdicts** against ground-truth ids with alias groups we cannot see. The name matcher agrees with the judge (κ above) but shares the ground truth, so it checks the matching step only.
- **One doer.** Policy-mode guarantees are specific to this agent (GPT-OSS-120B, this scaffold); the two other models announced in the dataset README are not released, so no transfer test is possible.
- The parser is heuristic (nearest-entity attribution, truncated tool outputs, agent-written scripts whose outputs we treat as evidence).
