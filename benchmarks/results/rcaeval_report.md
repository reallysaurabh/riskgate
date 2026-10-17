# RCAEval RE1-OB: RiskGate's hypothesis sets on real fault-injection telemetry, and a rule-labelled remediation gate

> **Licences:** RCAEval: HF mirror MIT; original data Zenodo 14590730 CC BY 4.0. No LLM is called; every gate score comes from `riskgate.Gate.score_episode`.

## What is real, what is rule-derived, what is scripted

| | |
|---|---|
| **Real** | The telemetry of 125 fault-injection cases on Online Boutique (per-second CPU, memory, latency, load of 11 services), their injection times and ground-truth (service, fault). **Section 1 (hypothesis sets) is entirely a real-data result.** |
| **Rule-derived** | The remediation labels in Section 2: an action-effect table written from SRE practice, applied to the TRUE cause. Nothing was executed on the testbed: labels are not observed outcomes. The same table is the gate's consequence model, so the consequence model is exact by construction and the only error source left is the diagnosis (which is real). |
| **Scripted** | The doers: an LLM-like dashboard reader and a restart-first variant (no LLM). |

## Data

- RE1-OB: 5 root-cause services × 5 faults (cpu, mem, disk, delay, loss) × 5 repetitions. Excluded: `re1ob_productcatalogservice_cpu_3` (telemetry ends before the injection (no faulty period)). Repaired: `re1ob_currencyservice_loss_1` (truncated inject_time (8 digits) repaired to time_start + 360; development subset only).
- **Development subset** (P3): repetition 1 of every family (25 cases): signature weights, the doers and the gate's plausible-set threshold τ were fixed on it. **Held out**: repetitions 2–5 (99 cases).
- Harmonised features: the two RE1 schema variants differ (cpu/mem cases: `_latency`, `_load`; delay/disk/loss cases: `_latency-50`, `_latency-90`, `_workload`). Only metrics present in both are used (p90 is dropped), over the same ±300 s window around the injection, so the fault type cannot be read off the schema or the case length.
- Hypotheses: **55** = 11 services × 5 faults (every service with metrics, not only the 5 the benchmark injects into). Scorer: BARO-style robust z-scores (median after vs before, IQR-based sd with a 5% floor), log-compressed, combined by a hand-set signature per fault type (CPU-dominant; memory-dominant; disk = moderate CPU + memory + own latency; network = own latency not explained by an elevated callee nor by own CPU/memory, delay vs loss by the size of the p50 shift). Nonconformity r(h) = E_max − E(h).

## 1. Hypothesis sets on real telemetry (held-out cases)

| | dev (25) | held-out (99) |
|---|---|---|
| top-1 (service, fault) | 72.0% | **72.7%** |
| service@1 / service@3 | 80.0% / 96.0% | 84.8% / 99.0% |
| fault type of the top-1 | 92.0% | 86.9% |
| mean rank of the truth (of 55) | 2.12 | 1.54 |
| reference: BARO-style service@1 / (service, metric→fault)@1 | 72.0% / 40.0% | 80.8% / 52.5% |

Held-out top-1 by fault type: cpu 89% (service@1 100%), delay 85% (service@1 85%), disk 80% (service@1 100%), loss 35% (service@1 40%), mem 75% (service@1 100%).

**Split conformal by case** (group = case; τ from a random half of the held-out cases, coverage and size on the other half; fixed split seed 0, then 200 re-splits):

| δ | target coverage | fixed split: coverage / mean set size | re-splits: coverage mean ± sd | share of splits below target | re-splits: set size mean ± sd |
|---|---|---|---|---|---|
| 0.05 | 95% | 96.0% / 4.54 | **96.2% ± 3.6%** | 29% | 4.24 ± 1.17 |
| 0.1 | 90% | 94.0% / 2.72 | **89.8% ± 6.1%** | 37% | 2.22 ± 0.46 |
| 0.2 | 80% | 84.0% / 1.26 | **80.0% ± 8.1%** | 40% | 1.28 ± 0.25 |

**Shift checks** (calibrate on the other groups, test on the held-out group; exchangeability does not hold across groups, so per-group coverage is not guaranteed):

| δ | leave-(service, fault)-family-out: pooled coverage / families below target (of 25) / worst family / mean size | leave-fault-type-out: pooled coverage / worst type / mean size |
|---|---|---|
| 0.05 | 96.0% / 2 / adservice/delay 25% / 4.32 | 96.0% / delay 85% / 3.35 |
| 0.1 | 88.9% / 3 / adservice/disk 0% / 2.34 | 85.9% / loss 65% / 1.85 |
| 0.2 | 79.8% / 8 / adservice/disk 0% / 1.17 | 77.8% / loss 35% / 1.08 |

Per fault type (leave-fault-type-out, δ = 0.1): cpu 100% (size 1.2), delay 85% (size 1.1), disk 80% (size 1.4), loss 65% (size 3.6), mem 100% (size 1.8).

## 2. Remediation gate (rule-derived labels, scripted doers)

**Action-effect table** (consequence model = label rule; an action on any other service never mitigates):

| fault on service s | restart s | scale out s ×2 | roll back s | rate-limit frontend | rationale |
|---|---|---|---|---|---|
| cpu | ✓ | ✓ | ✓ | ✗ | CPU hog in the pod: restart kills it; scale-out x2 adds capacity and halves the hogged replica's share; a rollback replaces the pods (= restart); rate-limiting the frontend sheds users and fixes nothing |
| mem | ✓ | ✗ | ✓ | ✗ | memory leak / stress: restart frees the memory (rollback replaces the pods too); scale-out only delays the leaking replica's degradation |
| disk | ✓ | ✗ | ✓ | ✗ | disk I/O stress (runaway writer in the pod): restart / rollback kill the writer; scale-out leaves the saturated replica serving slow requests |
| delay | ✗ | ✗ | ✗ | ✗ | network delay on the service's path: neither restart nor scale-out repairs the network -> escalate |
| loss | ✗ | ✗ | ✗ | ✗ | packet loss on the service's path: neither restart nor scale-out repairs the network -> escalate |

**Doers (scripted):** *LLM-like*: acts on the service with the largest jump on its dashboard (the largest robust z-score of CPU, memory or latency, after vs the previous 5 min; no call graph, no fault signatures), with the 'most common fix' (CPU → scale out, memory → restart, latency → restart), then the other fix, then the runner-up service; 15% of cases swap the first two (sampling noise). *Restart-first*: same reading, restart → roll back → restart runner-up. Each case is one decision with 3 ranked candidates (policy mode: the first certified candidate is executed).

| doer | first choice unsafe | some candidate safe | first choice unsafe by fault |
|---|---|---|---|
| llm-like (dashboard) | 60.6% | 55.6% | cpu 0%, mem 25%, disk 75%, delay 100%, loss 100% |
| restart-first | 47.5% | 59.6% | cpu 0%, mem 0%, disk 35%, delay 100%, loss 100% |

**Methods** (scores from `riskgate.Gate`): *twin gate*: certify iff the action works under the top-1 hypothesis (λ = 0, uncalibrated); *top-1 + CRC*: + (top-1 lead over the runner-up) if the action works under the top-1, else −1 − lead, CRC-calibrated; *plausible set only*: certify iff the action works under every hypothesis of the δ = 0.1 conformal set (τ = 1.648 from the development subset; held-out coverage 97.0%, mean size 4.86), no λ calibration; *RiskGate*: evidence gap over the same plausible set = r(best plausible hypothesis under which the action FAILS, capped at τ) − r(best one under which it works), −∞ if none works, calibrated by CRC or LTT (δ' = 0.1). With binary effects the per-hypothesis margin carries no magnitude, so the gap (a `score_fn` variant of the min-over-plausible-hypotheses score) is what gives CRC a graded score to threshold.

**AUROC of the first choice's score for 'first choice unsafe'**: llm-like (dashboard): twin 0.941, top1 0.974, set 0.744, riskgate 0.991; restart-first: twin 0.989, top1 1.000, set 0.942, riskgate 1.000.

**α = 5%, doer = llm-like (dashboard)** (99 cases; fixed split 50 calibration / 49 test cases):

| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |
|---|---|---|---|---|---|---|---|---|---|
| none | −∞ | 61.2% [47.2%, 73.6%] | 0.0% | 38.8% | 0.0% | – | – | – | – |
| twin gate (top-1, fixed) | 0.000 | 2.0% [0.4%, 10.7%] | 46.9% | 51.0% | 0.0% | 4.1% ± 2.0% | 29% | 43.2% | 52.7% |
| top-1 + CRC | 0.911 | 0.0% [0.0%, 7.3%] | 65.3% | 34.7% | 16.3% | 3.4% ± 3.3% | 22% | 56.7% | 39.9% |
| plausible set only (fixed, delta=0.1) | 0.000 | 0.0% [0.0%, 7.3%] | 55.1% | 44.9% | 6.1% | 0.0% ± 0.0% | 0% | 53.7% | 46.3% |
| riskgate (CRC) | 0.911 | 0.0% [0.0%, 7.3%] | 51.0% | 49.0% | 2.0% | 3.6% ± 3.5% | 28% | 47.3% | 49.2% |
| riskgate (LTT) | 1.154 | 0.0% [0.0%, 7.3%] | 53.1% | 46.9% | 4.1% | 1.8% ± 2.6% | 12% | 50.1% | 48.2% |

**α = 5%, doer = restart-first** (99 cases; fixed split 50 calibration / 49 test cases):

| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |
|---|---|---|---|---|---|---|---|---|---|
| none | −∞ | 49.0% [35.6%, 62.5%] | 0.0% | 51.0% | 0.0% | – | – | – | – |
| twin gate (top-1, fixed) | 0.000 | 2.0% [0.4%, 10.7%] | 40.8% | 57.1% | 0.0% | 1.0% ± 1.0% | 0% | 39.0% | 60.0% |
| top-1 + CRC | -1.031 | 2.0% [0.4%, 10.7%] | 40.8% | 57.1% | 0.0% | 2.5% ± 2.7% | 20% | 37.9% | 59.6% |
| plausible set only (fixed, delta=0.1) | 0.000 | 0.0% [0.0%, 7.3%] | 55.1% | 44.9% | 12.2% | 0.0% ± 0.0% | 0% | 53.7% | 46.3% |
| riskgate (CRC) | -1.185 | 4.1% [1.1%, 13.7%] | 38.8% | 57.1% | 0.0% | 2.1% ± 2.6% | 11% | 38.0% | 59.9% |
| riskgate (LTT) | 0.382 | 0.0% [0.0%, 7.3%] | 44.9% | 55.1% | 2.0% | 0.0% ± 0.0% | 0% | 42.1% | 57.9% |

**α = 10%, doer = llm-like (dashboard)** (99 cases; fixed split 50 calibration / 49 test cases):

| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |
|---|---|---|---|---|---|---|---|---|---|
| none | −∞ | 61.2% [47.2%, 73.6%] | 0.0% | 38.8% | 0.0% | – | – | – | – |
| twin gate (top-1, fixed) | 0.000 | 2.0% [0.4%, 10.7%] | 46.9% | 51.0% | 0.0% | 4.1% ± 2.0% | 0% | 43.2% | 52.7% |
| top-1 + CRC | -1.031 | 2.0% [0.4%, 10.7%] | 46.9% | 51.0% | 0.0% | 8.3% ± 5.1% | 37% | 39.1% | 52.6% |
| plausible set only (fixed, delta=0.1) | 0.000 | 0.0% [0.0%, 7.3%] | 55.1% | 44.9% | 6.1% | 0.0% ± 0.0% | 0% | 53.7% | 46.3% |
| riskgate (CRC) | -0.748 | 6.1% [2.1%, 16.5%] | 46.9% | 46.9% | 0.0% | 8.8% ± 5.5% | 38% | 42.0% | 49.2% |
| riskgate (LTT) | 0.911 | 0.0% [0.0%, 7.3%] | 51.0% | 49.0% | 2.0% | 3.6% ± 3.5% | 6% | 47.3% | 49.2% |

**α = 10%, doer = restart-first** (99 cases; fixed split 50 calibration / 49 test cases):

| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |
|---|---|---|---|---|---|---|---|---|---|
| none | −∞ | 49.0% [35.6%, 62.5%] | 0.0% | 51.0% | 0.0% | – | – | – | – |
| twin gate (top-1, fixed) | 0.000 | 2.0% [0.4%, 10.7%] | 40.8% | 57.1% | 0.0% | 1.0% ± 1.0% | 0% | 39.0% | 60.0% |
| top-1 + CRC | -1.180 | 10.2% [4.4%, 21.8%] | 32.7% | 57.1% | 0.0% | 8.5% ± 5.3% | 38% | 31.5% | 60.0% |
| plausible set only (fixed, delta=0.1) | 0.000 | 0.0% [0.0%, 7.3%] | 55.1% | 44.9% | 12.2% | 0.0% ± 0.0% | 0% | 53.7% | 46.3% |
| riskgate (CRC) | -1.637 | 14.3% [7.1%, 26.7%] | 30.6% | 55.1% | 0.0% | 7.9% ± 4.7% | 44% | 33.7% | 58.4% |
| riskgate (LTT) | -1.185 | 4.1% [1.1%, 13.7%] | 38.8% | 57.1% | 0.0% | 2.1% ± 2.6% | 2% | 38.0% | 59.9% |

**Alpha sweep (fixed split, RiskGate CRC, LLM-like doer):** α=0.01: unsafe 0.0%, ask 100.0%, α=0.02: unsafe 0.0%, ask 53.1%, α=0.05: unsafe 0.0%, ask 51.0%, α=0.1: unsafe 6.1%, ask 46.9%, α=0.15: unsafe 14.3%, ask 40.8%, α=0.2: unsafe 20.4%, ask 34.7%.

**Transfer (P7):** λ̂ calibrated on one doer's calibration cases, applied to the other doer's test cases (RiskGate CRC; unsafe / ask):

| α | calibrated on → tested on | unsafe | ask | base unsafe |
|---|---|---|---|---|
| 0.05 | llm-like (dashboard) → llm-like (dashboard) | 0.0% | 51.0% | 61.2% |
| 0.05 | llm-like (dashboard) → restart-first | 0.0% | 46.9% | 49.0% |
| 0.05 | restart-first → llm-like (dashboard) | 14.3% | 40.8% | 61.2% |
| 0.05 | restart-first → restart-first | 4.1% | 38.8% | 49.0% |
| 0.1 | llm-like (dashboard) → llm-like (dashboard) | 6.1% | 46.9% | 61.2% |
| 0.1 | llm-like (dashboard) → restart-first | 2.0% | 40.8% | 49.0% |
| 0.1 | restart-first → llm-like (dashboard) | 26.5% | 30.6% | 61.2% |
| 0.1 | restart-first → restart-first | 14.3% | 30.6% | 49.0% |

**P9 (library path):** `Gate.calibrate(cal_eps, groups=cases, tau_fraction=0)` reproduces the harness λ̂ on every fixed split (12 checks: both doers × CRC / LTT / top-1 CRC × α ∈ {5%, 10%}).

## Examples (held-out, LLM-like doer)

- **re1ob_adservice_disk_3** (truth `adservice:disk`): plausible set (τ = 1.65): `adservice:cpu` (r = 0.00), `adservice:disk` (r = 0.80). Candidates (doer's order): `scale_out({"service":"adservice"})` → unsafe, RiskGate 0.800, top-1 0.80, twin +1; `restart({"service":"adservice"})` → safe, RiskGate 1.648, top-1 0.80, twin +1; `restart({"service":"checkoutservice"})` → unsafe, RiskGate −∞, top-1 -1.80, twin -1. At α = 5% (fixed split; λ̂ = 0.911 for RiskGate, 0.911 for top-1 + CRC): *twin gate (top-1, fixed)* certifies `scale_out({"service":"adservice"})` (UNSAFE); *top-1 + CRC* asks; *plausible set only* certifies `restart({"service":"adservice"})` (safe); *riskgate (CRC)* certifies `restart({"service":"adservice"})` (safe).
- **re1ob_adservice_delay_3** (truth `adservice:delay`): plausible set (τ = 1.65): `checkoutservice:delay` (r = 0.00), `checkoutservice:loss` (r = 0.18), `adservice:loss` (r = 1.15). Candidates (doer's order): `restart({"service":"checkoutservice"})` → unsafe, RiskGate −∞, top-1 -1.18, twin -1; `scale_out({"service":"checkoutservice"})` → unsafe, RiskGate −∞, top-1 -1.18, twin -1; `restart({"service":"adservice"})` → unsafe, RiskGate −∞, top-1 -1.18, twin -1. At α = 5% (fixed split; λ̂ = 0.911 for RiskGate, 0.911 for top-1 + CRC): *twin gate (top-1, fixed)* asks; *top-1 + CRC* asks; *plausible set only* asks; *riskgate (CRC)* rejects (no plausible hypothesis under which any candidate works: escalate).

## What the experiment shows, and caveats

- **Real-data content = Section 1.** On held-out real telemetry the 55-hypothesis conformal sets reach their nominal coverage on average across random case splits (marginal guarantee; individual splits vary), at small set sizes; coverage degrades under family or fault-type shift (packet loss is the weak spot of a metrics-only scorer), which is exactly what exchangeability predicts.
- **The hypothesis set alone already bounds action risk.** Certifying only actions that work under every hypothesis of the δ = 0.1 set (τ fixed on the development subset) gave 0.0% unsafe over all re-splits for both doers, auto-approving 46.3% of cases: if the truth is in the set, such an action is safe, so P(certified and unsafe) <= P(truth outside the set) <= δ for exchangeable cases (here the dev-calibrated set is conservative: held-out coverage 97.0%).
- **Section 2 is a rule-labelled simulation on real diagnoses.** Because the consequence model equals the label rule, the gate is only as wrong as its hypothesis set; the experiment isolates the value of handling diagnosis uncertainty, not of the consequence model.
- **RiskGate vs single-diagnosis CRC.** For the LLM-like doer at α = 5%, RiskGate (CRC) auto-approves 49.2% of cases vs 39.9% for top-1 + CRC at a similar unsafe rate (3.6% vs 3.4%): the single diagnosis cannot tell 'restart is safe under both cpu and disk' from 'scale-out is safe only under cpu' (first example). For the restart-first doer the two are tied (59.9% vs 59.6%), and at α = 10% top-1 + CRC is slightly more efficient (52.6% vs 49.2%; 60.0% vs 58.4%). RiskGate's score discriminates best (AUROC above).
- **The uncalibrated twin gate is strong here** (4.1% unsafe on average for the LLM-like doer, 1.0% for restart-first) because the scorer is good (73% top-1, 99% service@3) and the consequence model is exact by construction; it carries no guarantee and nothing adapts it when the diagnoser degrades.
- **CRC bounds the expected risk over calibration draws**; with ~50 test cases per split, individual splits exceed α often (see 'share > α'; e.g. the restart-first fixed split at α = 10%). LTT is the high-probability alternative: its δ' = 10% bounds the probability that the TRUE risk exceeds α, and the empirical rate on ~50 test cases adds sampling noise on top (share > α between 0% and 12%).
- **Transfer:** a λ̂ calibrated on the restart-first doer is too permissive for the LLM-like doer (whose first choices are unsafe more often): policy-mode guarantees are doer-specific, as the paper states.
- Groups are single cases (one decision each); repetitions of the same family are treated as exchangeable cases (the family-out analysis shows what happens when they are not).
- The doers are scripted and deterministic given the telemetry, and they read the same robust z-scores as the gate's scorer (without its call graph and signatures), so doer and gate errors are correlated; a real LLM reading raw dashboards may err differently. The action-effect table is a stylised SRE rule, not a measurement of what restarts or scale-outs did on this testbed (e.g. a pod restart may also clear a tc/netem rule injected into the pod's network namespace; the table models the fault classes, not the injection mechanism).
