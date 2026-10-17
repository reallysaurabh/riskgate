# End-to-end tutorial: an SRE agent, from calibration to drift

This tutorial walks through the full lifecycle of a gate on the Kubernetes incident-response example:
build → calibrate → sign → deploy → report outcomes → detect drift → update the model → recalibrate.
Every output below is **real output** of [`examples/end_to_end_sre.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/end_to_end_sre.py)
(full transcript: [`examples/e2e_transcript.txt`](https://github.com/reallysaurabh/riskgate/blob/main/examples/e2e_transcript.txt)).

```bash
cd examples && PYTHONPATH=.. python end_to_end_sre.py            # about 5 minutes
cd examples && PYTHONPATH=.. python end_to_end_sre.py --quick    # about 1 minute, smaller numbers
```

> **Simulated data.** The service, the incidents and the "agent" are simulated (a queueing model of a checkout
> service and a scripted SRE assistant that reproduces typical LLM habits). Set `RISKGATE_LLM_API_KEY` (and
> optionally `RISKGATE_LLM_BASE_URL`, `RISKGATE_LLM_MODEL`) to let a real LLM rank the remediation actions instead;
> the gate code does not change.

## The story in plain words

A web shop's `checkout-api` gets paged: latency is up. An AI on-call assistant proposes fixes: add servers, roll
back yesterday's release, rate-limit traffic, and so on. Some of these fixes make things *worse* depending on what
is really wrong (for example, adding servers when the database is the bottleneck exhausts its connections).

The gate's job: let a fix through only if it keeps the service within its promise (95% of requests under 800 ms and
fewer than 2% errors, every minute for the next half hour) under **every cause the evidence still allows**; otherwise
suggest a quick diagnostic, explain why the fix is unsafe, or page a human. Then, when the world changes, notice it
and stop trusting itself until it is recalibrated.

## Stage 1: build the gate from domain components

```python
gate = K.make_gate(audit=rg.AuditLog(OUT / "decisions.jsonl", extra={"service": "checkout-api"}),
                   monitor=rg.Monitor(alarm_level=20),          # false-alarm probability <= 1/20 (Ville)
                   doer_id="scripted-sre-v1")                    # policy mode: the agent is part of the guarantee
```

`K.make_gate` (in [`k8s_incident.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/k8s_incident.py)) wires four domain components into `rg.Gate`:

| Component | Here | In your system |
|---|---|---|
| consequence model | minute-level Erlang-C queueing twin of pods, workers, Postgres connection limits and a Redis cache | load model, `kubectl --dry-run` + capacity model, staging replay |
| hypotheses | 13 causes × magnitudes (traffic surge, slow query, cache outage, bad release, none), scored by how well they explain the last 10 minutes of metrics | your incident classifier / RCA tool, with a nonconformity score |
| spec | `Upper("p95_ms", 800)`, `Upper("error_rate", 0.02)`, each minute from t+4 to t+30 | your SLOs |
| probes | `pg_stat_statements_top`, `redis_info_stats`, `canary_version_compare` (read-only, 2 minutes) | any read-only diagnostic |

```text
  consequence model : k8s-twin-v1 (Erlang-C queueing twin, Postgres connection limits)
  hypotheses        : k8s-diag-v1 (13 incident causes x magnitudes)
  spec              : ['p95<=800ms', 'errors<=2%'] (each minute, t+4..t+30)
  probes            : ['pg_stat_statements_top', 'redis_info_stats', 'canary_version_compare']   alpha=0.05  mode=policy
  fingerprint       : aa85e0ef19d53254   status before calibration: void
```

Before calibration the status is `VOID`, so every check would return `ASK`.

## Stage 2: collect labelled incidents

An episode is one past (or simulated) decision point: the context the gate would have seen, the actions the agent
proposed in its order of preference, and whether each action would have breached the SLO.

```text
  example record: {"incident_id": "INC-010000", "alert": "LatencySLOBurnRate", "recent_deploy": true,
                   "agent_ranked_actions": ["noop", "scale_out_1_5x", "failover_reads", ...],
                   "ground_truth_cause": {"kind": "slow_db_query", "magnitude": 0.637}}
  doer's first choice unsafe in 33.8% of calibration incidents
```

The agent's first choice would breach the SLO in about a third of incidents. That is the problem the gate solves.
In production, labels come from staging replays, sandboxed dry-runs or audited postmortems
(see [Operations](operations.md)).

## Stage 3: calibrate, sign, inspect

```python
cal = gate.calibrate(cal_eps)                         # 1200 labelled incidents
cal.save(OUT / "checkout-api.cal.json", key=KEY)      # HMAC-signed; KEY comes from your secret store
riskgate_cli(["report", str(OUT / "checkout-api.cal.json"), "--key", KEY])
```

```text
riskgate calibration 9fe91033e2d6  (schema riskgate.calibration/v1, ...)
  guarantee   : expectation  alpha=0.05
  mode        : policy    episodes for risk control n=840  (tau split n=360)
  plausible   : tau=4.554 (target coverage 0.95), mean |H|=3.93, tau_ood=5.813
  threshold   : lambda=-1.398  calibration ask rate=0.217  calibration risk=0.029
  probing     : lambda1=-0.2818 lambda2=-3.775 share=0.6 probed episodes=45
  fingerprint : aa85e0ef19d53254
```

How to read it:
- **360 incidents** fixed the plausible-set threshold τ (the true cause is kept in about 95% of cases; on average
  3.9 causes remain plausible). The other **840** chose the thresholds. Keeping these apart is what makes the
  guarantee exact.
- **λ = −1.398** is the single-stage threshold (margins are normalised: −1 means "predicted to exceed the SLO by
  one SLO-width"). A negative value means actions with slightly negative predicted margins still turned out safe
  often enough: the twin is somewhat pessimistic, and calibration measured by how much. Nobody guessed it.
- **Probing is calibrated**: stage 1 certifies only above λ₁ = −0.28 (risk budget 0.6 × 5%); after a probe, stage 2
  certifies above λ₂ = −3.78 (budget 0.4 × 5%, counting any harm the probe itself does). Total ≤ 5%.

## Stage 4: deploy

The loop per incident (simplified from `handle()` in the script):

```python
d = gate.check_ranked(ranked_actions, snapshot)                 # agent's proposals, best first
if d.verdict is rg.Verdict.PROBE:
    run(d.probe)                                                 # read-only diagnostic
    d = gate.check_ranked(ranked_actions, new_snapshot, stage=2)
if d.verdict is rg.Verdict.CERTIFY:
    execute(d.action)
    gate.record_outcome(d, slo_breached)                         # later, when the outcome is known
else:
    page_on_call(d.reason)                                       # ASK or REJECT, with the gate's explanation
```

Three real incidents from the run:

```text
  --- example: certified directly ---
  incident INC-050000: rps=920 p95=268ms errors=0.1% pod CPU=63% DB CPU=54% recent_deploy=False
    hidden truth   : {'kind': 'traffic_surge', 'magnitude': 1.334}
    agent proposes : ['scale_out_2x', 'scale_out_1_5x', 'rollback_and_scale'] ...
    gate (stage 1) : CERTIFY - robust margin above calibrated threshold
    plausible      : ['none', 'traffic_surge:1.4', 'slow_db_query:0.75', 'slow_db_query:0.55', 'bad_pod_version:1.5']
    executed       : failover_reads -> SLO held
```

Five causes were still plausible, and none of them made `failover_reads` breach the SLO, so it was certified even
though it was not the agent's first choice: none of the agent's first three choices was safe under all five.

```text
  --- example: probe, then decide ---
  incident INC-050020: rps=1152 p95=2041ms errors=14.6% pod CPU=76% DB CPU=62% recent_deploy=False
    hidden truth   : {'kind': 'traffic_surge', 'magnitude': 1.704}
    gate (stage 1) : PROBE - plausible hypotheses disagree about safety; probe 'pg_stat_statements_top' is
                     expected to resolve 100% of the disagreements
    plausible      : ['traffic_surge:1.4', 'traffic_surge:1.8', 'cache_outage:0.4', 'bad_pod_version:2']
    probe result   : pg_stat_statements_top -> 2.958 (2 minutes later)
    gate (stage 2) : CERTIFY - robust margin above calibrated threshold
    executed       : scale_out_2x -> SLO held
```

Scaling out is right for a traffic surge and wrong for some of the other causes. Query latency tells them apart, so
the gate asked for that first.

```text
  --- example: escalated to a human ---
  incident INC-050005: rps=1002 p95=2037ms errors=93.5% pod CPU=40% DB CPU=96% recent_deploy=True
    hidden truth   : {'kind': 'cache_outage', 'magnitude': 0.421}
    agent proposes : ['rollback_deploy', 'rollback_and_scale', 'scale_out_2x'] ...
    gate (stage 1) : PROBE - ... probe 'redis_info_stats' is expected to resolve 100% of the disagreements
    probe result   : redis_info_stats -> 0.409 (2 minutes later)
    gate (stage 2) : ASK - cannot certify: best robust margin -5.000 < threshold -3.775; worst case under hypothesis 'cache_outage:0.4'
    -> paged the on-call engineer with the gate's explanation
```

There was a deploy yesterday, so the agent wanted to roll back. The probe showed the cache had collapsed; no proposed
action keeps the SLO under that cause, so a human gets the page, with the reason.

Over 300 live incidents:

```text
  live summary: certified-and-unsafe 2.0% | asked/rejected 17.3% | probed 12.3% |
                agent's first choice would have been unsafe 29.0% | n=300
```

## Stage 5: outcomes and audit

```text
  monitor: observed risk on certified actions = 0.0242, martingale = 0.01, status = valid
{"decisions": 337, "verdicts": {"certify": 248, "probe": 37, "ask": 20, "reject": 32},
 "status": {"valid": 337}, "latency_ms": {"p50": 9.96, "p95": 17.46, "p99": 22.66}}
```

- `record_outcome(decision, unsafe)` matches late-arriving labels to decisions by `decision_id`.
- The drift **martingale** stays near zero: deployment looks like calibration.
- `riskgate audit-summary` reads the JSONL decision log. There are 337 decisions for 300 incidents because each probe
  produces a second (stage-2) decision. Latency is dominated by the queueing twin (about 10 ms); the gate itself adds
  about 0.1 ms.

## Stage 6: the world changes

The database is moved to a smaller instance class (−40% capacity). Nobody tells the twin.

```text
  the twin still assumes the old capacity: its predictions are now optimistic
  drift alarm after 227 incidents (martingale 22 >= 20) -> status DEGRADED; the online threshold may only tighten
  drift summary: certified-and-unsafe 8.3% | asked/rejected 51.7% | probed 10.0% |
                 agent's first choice would have been unsafe 52.7% | n=300
  status degraded; threshold in use 1.934 vs calibrated -0.282; observed risk on certified actions 0.079
```

This stage shows a limitation, on purpose:
- **The guarantee no longer applies** once deployment stops looking like calibration. Over these 300 incidents
  8.3% of incidents ended with a certified action that breached the SLO, above α = 5%.
- **Detection takes time.** The monitor tests whether the gate's *inputs* (the best robust score per decision) still
  look like calibration. A capacity change shows up there only indirectly, so the alarm fired after 227 incidents.
  Watch the outcome feed as well: here the observed risk among certified actions (7.9%) was visibly above α.
- **After the alarm the gate becomes more cautious, never less**: the status is `DEGRADED`, the threshold rises
  (to 1.93 here) as SLO breaches are reported, and the alarm stays latched until you recalibrate.

## Stage 7: update the model, recalibrate

The team updates the twin to the new capacity. Its version string changes, so the fingerprint changes:

```text
  new twin version k8s-twin-v1-cap2880: fingerprint 25983401ea5167e0 != artefact aa85e0ef19d53254 -> status void
  any check now -> ASK: guarantee void: gate not calibrated or a component changed since calibration
  recalibrated on 900 labelled incidents from the new regime: artefact 20e6d804081e, status valid
  after recalibration: certified-and-unsafe 3.3% | asked/rejected 43.7% | probed 29.3% |
                       agent's first choice would have been unsafe 53.7% | n=300 | status valid
```

- The old artefact cannot be used with the new model by accident: the gate asks until it is recalibrated.
- After recalibration the unsafe rate is back under α. The ask rate stays high because the smaller database leaves
  fewer safe options: with half of the agent's first choices unsafe, asking is the honest answer.

## What to take away

1. The gate never trusts the agent's ranking; it trusts calibrated consequence predictions under every plausible
   cause.
2. The threshold is learned from labelled outcomes, and the artefact records exactly which components it is valid
   for.
3. When the world drifts, the gate detects it (with delay), tightens, and tells you; it never quietly loosens.
4. Recalibration is a routine operation: collect labelled incidents from the new regime, `recalibrate`, sign, deploy.

Next: [Bring your own domain](byo_domain.md) · [Operations](operations.md) · [Calibration & guarantees](calibration.md)
