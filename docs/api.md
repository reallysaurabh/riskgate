# API reference

This page documents the public API of riskgate `1.0.0rc1`. Every name below is importable as written. The top-level package re-exports the core:

```py
import riskgate as rg
rg.Gate, rg.Calibration, rg.Episode, rg.Action, rg.Hypothesis, rg.Decision, rg.Verdict, rg.GuaranteeStatus
rg.Spec, rg.Upper, rg.Lower, rg.Envelope, rg.Predicate, rg.Custom, rg.HardRule
rg.ConsequenceModel, rg.HypothesisProvider, rg.SingleHypothesis, rg.CallableModel
rg.Monitor, rg.AuditLog, rg.MemoryAudit, rg.MultiSink, rg.summarize_audit
rg.guard, rg.ActionRejected, rg.ApprovalRequired, rg.SignatureError
rg.conformal_quantile, rg.crc_threshold, rg.__version__
```

Optional modules are imported explicitly:

| Module | Contents | Extra dependency |
|---|---|---|
| [`riskgate.config`](#riskgateconfig) | `load_config`, `spec_from_config`, `rule_from_config`, `gate_from_config` | `pyyaml` for YAML only |
| [`riskgate.telemetry`](#riskgatetelemetryotelsink) | `OTelSink` | `opentelemetry-api` (+ SDK/exporter) |
| [`riskgate.doers`](#riskgatedoersllmranker) | `LLMRanker` | none (stdlib HTTP) |
| [`riskgate.conformal`](#riskgateconformal) | CRC primitives | none |
| [`riskgate.stl`](#riskgatestl) | signal temporal logic constraints | none |
| [`riskgate.multirisk`](#riskgatemultirisk) | Learn-then-Test p-values and thresholds | none |
| [`riskgate.weighted`](#riskgateweighted) | weighted conformal methods under covariate shift | none |
| [`riskgate.adapters.*`](#framework-adapters) | ADK, LangGraph, OpenAI Agents SDK, MCP | the framework |

> [!NOTE]
> **How to read the examples.** Blocks marked as Python run top to bottom in one session. Each was executed against `1.0.0rc1`, and the `# ->` comments show its real output. Signature blocks are for reference only. They all use the toy domain defined in [Setup](#setup-used-on-this-page).

> [!NOTE]
> The use-case building blocks (`riskgate.worlds`, `riskgate.provenance`, `riskgate.replay`, `riskgate.rules`)
> are documented with examples in [Use cases](use_cases.md).

## Setup used on this page

A tank holds `level` litres. The action `add(v)` adds `v` litres. A hidden leak (0, 10 or 25 L) drains it, and we only see noisy `evidence` of the leak. The level must stay between 10 L and 100 L.

```python
import numpy as np
import riskgate as rg


class Tank:
    """Consequence model: the level after adding v litres, under a hypothesised leak."""
    version = "tank-v1"                      # part of the fingerprint: bump it whenever the model changes

    def predict(self, action, hyp, ctx):
        return {"level": ctx["level"] + action.args.get("v", 0.0) - hyp.params["leak"],
                "observation": [hyp.params["leak"] / 5]}     # what a flowmeter probe would read (noise units)


class LeakHyps:
    """Hypothesis provider: which leak explains the evidence? Lower score = better fit."""
    version = "leak-v1"

    def hypotheses(self, ctx):
        return [rg.Hypothesis(f"leak{l}", family=f"leak{l}", params={"leak": l},
                              score=abs(ctx["evidence"] - l) / 5) for l in (0, 10, 25)]


ACTIONS = [rg.Action("add", {"v": v}) for v in (40, 20, 0, -10)]


def episode(rng):
    """One labelled decision point: context, the doer's ranked proposals, counterfactual labels."""
    leak, level = int(rng.choice([0, 10, 25])), float(rng.uniform(30, 70))
    ctx = {"level": level, "evidence": leak + rng.normal(0, 6)}
    ranked = list(ACTIONS)
    rng.shuffle(ranked)                      # whatever order the agent proposed
    unsafe = {a.key: not 10 <= level + a.args["v"] - leak <= 100 for a in ranked}
    return rg.Episode(ctx, ranked, unsafe, true_family=f"leak{leak}")


rng = np.random.default_rng(0)
episodes = [episode(rng) for _ in range(400)]
spec = rg.Spec(rg.Envelope("level", lo=10, hi=100, scale=10))    # margin in units of 10 L
```

---

## Gate

`riskgate.Gate` is the gate. It turns hypotheses into consequences, consequences into margins, and margins into a calibrated verdict.

```py
Gate(consequence, spec, hypotheses=None, *, alpha=0.05, mode="policy", delta=0.05,
     hard_rules=(), probes=(), noop=None, noop_tol=0.0, min_separation=1.0, min_resolved=0.3,
     ood_alpha=0.01, audit=None, monitor=None, require_calibration=True, doer_id=None,
     timeout_s=None, cache_size=0, context_key=None, parallel=0, weight_fn=None, score_fn=None)
```

### Constructor parameters

| Parameter | Default | Meaning |
|---|---|---|
| `consequence` | required | [`ConsequenceModel`](#consequencemodel-protocol): `version` plus `predict(action, hypothesis, context)`, and optionally `predict_batch(actions, hypotheses, context)`. |
| `spec` | required | [`Spec`](#spec) that turns a predicted outcome into a signed margin. |
| `hypotheses` | `None` | [`HypothesisProvider`](#hypothesisprovider-protocol). `None` means `SingleHypothesis()`: one hypothesis with score 0. |
| `alpha` | `0.05` | Default risk level for `calibrate()`. Decisions always use the threshold of the **loaded artefact**, so changing `gate.alpha` has no effect until you recalibrate. |
| `mode` | `"policy"` | `"policy"` or `"agnostic"` loss (see [Calibration & guarantees](calibration.md)). Any other value raises `ValueError`. Part of the fingerprint. `calibrate(mode=...)` and `load_calibration` set it to the artefact's mode. |
| `delta` | `0.05` | Default plausible-set miscoverage for `calibrate()`. τ keeps the true hypothesis family with probability ≥ 1 − δ. |
| `hard_rules` | `()` | [`HardRule`](#hardrule)s, checked before any prediction. Each rule's name, verdict and logic (or its `version`) are part of the fingerprint. |
| `probes` | `()` | Read-only `Action`s that the gate may return with a `PROBE` verdict. Their keys are part of the fingerprint. |
| `noop` | `None` | The "do nothing" action. A probe is *admissible* if, under every plausible hypothesis, it is certifiable or its margin is ≥ the noop margin − `noop_tol`. Without `noop`, a probe must be certifiable under every plausible hypothesis. |
| `noop_tol` | `0.0` | Tolerance for that comparison, in margin units. |
| `min_separation` | `1.0` | A probe *resolves* a pair of hypotheses if their predicted `observation`s differ by at least this much (max-norm, in noise units). |
| `min_resolved` | `0.3` | Minimum share of disagreeing hypothesis pairs that a probe must resolve before the gate proposes it. |
| `ood_alpha` | `0.01` | τ_ood, set by `calibrate()`, is the conformal (1 − `ood_alpha`) quantile of the best hypothesis score. |
| `audit` | `None` | Sink with `write(decision)`, called for every decision ([`AuditLog`](#auditlog), [`MultiSink`](#multisink), [`OTelSink`](#riskgatetelemetryotelsink), ...). |
| `monitor` | `None` | [`Monitor`](#monitor) for drift detection, the tighten-only online threshold and delayed outcomes. |
| `require_calibration` | `True` | If the status is `VOID`, return `ASK` without evaluating anything. With `False` the gate still evaluates (margins, plausible set, reasons) for analysis, but while the status is `VOID` it **never returns `CERTIFY` or `PROBE`**: those become `ASK`, with the would-be verdict in `Decision.would_verdict` (and in the audit record). `REJECT` stays `REJECT`. An uncalibrated gate uses threshold +∞. For a shadow deployment, record decisions next to the humans without enforcing them; do not rely on this flag. |
| `doer_id` | `None` | Identifier of the proposing agent (model, prompt version, temperature). Calibrating in policy mode without it emits `RiskGateWarning`. In policy mode it is part of the fingerprint, because the policy-mode guarantee only covers the calibrated doer. |
| `timeout_s` | `None` | Wall-clock budget per check. If it is exceeded, the check returns `ASK` and `reason` starts with `timeout:`. |
| `cache_size` | `0` | Size of an LRU cache of margin matrices, keyed by `(context_key(context), action keys, hypothesis names)`. `0` disables the cache. A positive value **requires** `context_key`, otherwise `ValueError`. |
| `context_key` | `None` | `context -> hashable` cache key, derived from the context's **content** (for example an incident id plus a revision number). `id()` is unsafe: a mutated or recycled context object would get stale margins. |
| `parallel` | `0` | Worker threads for models **without** `predict_batch` (`0` = sequential). |
| `weight_fn` | `None` | `context -> w(x)`, the likelihood ratio for weighted CRC under a known covariate shift. Use it with `calibrate(weights=...)`. Content-hashed into the fingerprint. |
| `score_fn` | `None` | `score_fn(margins_row, plausible_hypotheses) -> float`. Replaces the default aggregation (the minimum over plausible hypotheses). Content-hashed into the fingerprint. |

> [!NOTE]
> **Audit failures fail closed.** The sink is called once per decision. If `audit.write` raises, the error is logged on the `riskgate` logger and a `CERTIFY` is downgraded to `ASK` (`fail-closed: audit sink error (...); decision not recorded`). Other verdicts are returned unchanged. A [`MultiSink`](#multisink) swallows errors from its members, so put sinks that are allowed to fail (telemetry) inside one, and keep the authoritative audit log outside it.

```python
gate = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1)
print(gate.status())                     # -> GuaranteeStatus.VOID  (not calibrated yet)
cal = gate.calibrate(episodes)
print(gate.status(), f"lambda={cal.lam:.3f} tau={cal.tau:.3f} n={cal.n}")
# -> GuaranteeStatus.VALID lambda=-1.499 tau=2.259 n=280
d = gate.check(rg.Action("add", {"v": 20}), {"level": 50, "evidence": 1.0})
print(d.verdict, f"score={d.score:.2f}", d.plausible, "|", d.reason)
# -> Verdict.CERTIFY score=3.00 ['leak0', 'leak10'] | robust margin above calibrated threshold
```

λ is measured in margin units and can be negative, as here. The robust score is a worst case over the plausible hypotheses, and calibration learns how much of that pessimism the realised outcomes allow. Never read a margin as "safe" or "unsafe" without the calibrated λ.

### `Gate.check` and `Gate.check_ranked`

```py
gate.check(action, context, *, stage=1) -> Decision           # == check_ranked([action], context, stage=stage)
gate.check_ranked(candidates, context, *, stage=1) -> Decision
```

`check_ranked` receives the doer's proposals in rank order (first = preferred). It returns the **first** candidate that it can certify, or one of `REJECT`, `PROBE` or `ASK`:

1. **Status.** If the status is `VOID` and `require_calibration=True`, return `ASK` (`guarantee void: ...`). With `require_calibration=False` the steps below run, and a final check turns `CERTIFY` and `PROBE` into `ASK` while `VOID` (`guarantee void: would certify, ...`; `would_verdict` is set).
2. **Hard rules**, candidate by candidate. If a REJECT rule fires, the candidate is skipped. If an ASK rule fires on the first remaining candidate, return `ASK`. If it fires on a lower-ranked candidate, that candidate is skipped: an action that needs approval is never certified automatically. If no candidate is left, return `REJECT` (`blocked by hard rules (...)`).
3. **Hypotheses.** If the best hypothesis score is above τ_ood, return `ASK` (out of distribution). The plausible set is every hypothesis with score ≤ τ. If it is empty, return `ASK`.
4. **Margins.** Compute `margin_matrix(viable, plausible)`. The robust score is the minimum over plausible hypotheses, or `score_fn`. NaN scores count as −∞, and +∞ is clipped to 1e300, so λ = +∞ never certifies anything.
5. **Threshold λ.** With probes configured and probing calibrated, use λ₁ at stage 1 and λ₂ at stage 2. Otherwise, with a `weight_fn` and weighted calibration data, use the per-context weighted λ. Otherwise use λ̂. While `DEGRADED`, use `max(λ, monitor.online_lambda)`.
6. If a candidate has robust score ≥ λ, return `CERTIFY` for the first such candidate.
7. At stage 1, with calibrated probing, return `PROBE` if [`select_probe`](#gateselect_probe-and-gatemargin_matrix) finds an admissible probe.
8. If every viable candidate violates the spec under every plausible hypothesis, return `REJECT` with the binding constraint. Otherwise return `ASK` with the best margin, the threshold and the worst-case hypothesis.

Any exception in the hypotheses, the model or the spec returns `ASK` (`fail-closed: ...`). So does an exceeded `timeout_s` (`timeout: ...`). After a `PROBE`, execute the probe, refresh the context and call `check_ranked(..., stage=2)`, which never probes again.

```python
d = gate.check_ranked([rg.Action("add", {"v": 80}), rg.Action("add", {"v": 20})], {"level": 50, "evidence": 1.0})
print(d.verdict, d.action.key)           # -> Verdict.CERTIFY add({"v":20})
for row in d.considered:
    print("  ", row)
# ->    {'action': 'add({"v":80})', 'score': -3.0, 'certified': False}
# ->    {'action': 'add({"v":20})', 'score': 3.0, 'certified': True}
```

### `Gate.calibrate`

```py
gate.calibrate(episodes, *, alpha=None, mode=None, delta=None, probe_share=0.6, probe_split=0.5,
               tau_fraction=0.3, seed=0, guarantee="expectation", delta_ltt=0.1,
               extra_losses=None, weights=None, groups=None) -> Calibration
```

Calibrates τ (plausible set), τ_ood, λ̂ and, if probes are configured, λ₁ and λ₂. The new artefact is loaded into the gate and bound to the monitor, then returned.

| Parameter | Default | Meaning |
|---|---|---|
| `episodes` | required | Sequence of [`Episode`](#episode). |
| `alpha` | `gate.alpha` | Risk level α. |
| `mode` | `gate.mode` | Loss mode. Passing it also sets `gate.mode`, so the fingerprint, which binds `doer_id` in policy mode, matches what was calibrated. |
| `delta` | `gate.delta` | Plausible-set miscoverage δ. |
| `probe_share` | `0.6` | Stage-1 level as a share of α: λ₁ is calibrated at `probe_share · alpha`. |
| `probe_split` | `0.5` | Share of the risk-control episodes used for λ₁. The rest are used for λ₂. |
| `tau_fraction` | `0.3` | Share of all episodes drawn at random for the τ split. Only drawn episodes that have `true_family` are used for τ. The rest stay in risk control. |
| `seed` | `0` | Seed for the random splits. |
| `guarantee` | `"expectation"` | `"expectation"`: E[risk] ≤ α (conformal risk control). `"high_probability"`: P(risk ≤ α) ≥ 1 − `delta_ltt` (Learn-then-Test, [`riskgate.multirisk`](#riskgatemultirisk)). |
| `delta_ltt` | `0.1` | Family-wise error level for `"high_probability"`. |
| `extra_losses` | `None` | `{name: (fn(episode, executed_action) -> value in [0, 1], alpha_j)}`. These risks are controlled together with safety, each at its own level. λ̂ is the strictest of the per-risk thresholds (expectation), or the joint LTT threshold. Values may be fractional in both modes. |
| `weights` | `None` | One likelihood ratio w(x) per episode, for weighted CRC under covariate shift. Use it with `Gate(weight_fn=...)`. |
| `groups` | `None` | One hashable label per episode (conversation, task, incident). Episodes that share a label are dependent. The τ split is drawn by group, and risk control runs on each group's mean loss, so the guarantee is E[share of a new group's decisions certified and unsafe] ≤ α. `cal.n` counts groups. Cannot be combined with `weights`; disables two-stage probing. See [calibration.md](calibration.md#dependent-decisions-grouped-calibration). |

What it computes:

1. **τ split.** A random `tau_fraction` share of the episodes is drawn. Those with a `true_family` set τ (conformal 1 − δ quantile of the true family's best score) and τ_ood (conformal 1 − `ood_alpha` quantile of the best score). If no episode has a `true_family`, or `tau_fraction=0`, then τ = τ_ood = +∞: every hypothesis is always plausible and there is no OOD guard.
2. **Risk control** runs on the remaining, disjoint episodes (`n` of them). Every candidate is scored exactly as the deployed gate would score it. Candidates removed or escalated by hard rules, out-of-distribution contexts and evaluation errors get −∞ (never certified). Errors are counted in `diagnostics["evaluation_failures"]` and logged. Each episode's loss is a function of λ over the exact grid of observed scores ∪ {+∞}, and λ̂ is the conformal risk control threshold (or the LTT threshold).
3. **Two-stage probing**, only if `gate.probes` is non-empty, some episode has `after_probe`, and the run uses `guarantee="expectation"` without `extra_losses` or `weights`. Otherwise `diagnostics["probing_disabled"]` explains why. The risk-control episodes are split by `probe_split` into A and B.
    - λ₁ is CRC on A at `probe_share · alpha`: the risk of certifying before any probe.
    - λ₂ is CRC on **all** of B at `(1 − probe_share) · alpha`. An episode's loss is 0 if the gate would not probe it. If it would, the loss is the harm caused by the probe (`meta["probe_unsafe"]`) plus the loss of the post-probe decision. A probe without post-probe data counts as a loss.
    - Together, stage 1 and stage 2 keep the risk ≤ α. Probing is enabled (`cal.probing`) only if B has at least ⌈1/((1 − probe_share)·α)⌉ episodes, at least one is probed, and λ₂ is finite.

Raises `ValueError` for an unknown `guarantee` or `mode`, for `weights` whose length differs from `episodes`, and for fewer than ⌈1/α⌉ risk-control episodes.

```python
big_move = lambda ep, a: float(abs(a.args["v"]) >= 20)          # second risk: large moves
g2 = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1)
c_hp = g2.calibrate(episodes, guarantee="high_probability", delta_ltt=0.1)
c_mr = g2.calibrate(episodes, extra_losses={"big_move": (big_move, 0.3)})
print(f"expectation lambda={cal.lam:.3f}  high-probability lambda={c_hp.lam:.3f}  with extra risk lambda={c_mr.lam:.3f}")
# -> expectation lambda=-1.499  high-probability lambda=-1.499  with extra risk lambda=3.736
print(c_mr.extra_alphas, round(c_mr.diagnostics["calibration_risk_big_move"], 3))
# -> {'big_move': 0.3} 0.296

# weighted CRC: deployment sees low tank levels 10x more often than calibration did
w = lambda ctx: 10.0 if ctx["level"] < 40 else 1.0
gw = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1, weight_fn=w)
gw.calibrate(episodes, weights=[w(ep.context) for ep in episodes])
low = gw.check(rg.Action("add", {"v": 0}), {"level": 35, "evidence": 1.0})
high = gw.check(rg.Action("add", {"v": 0}), {"level": 60, "evidence": 1.0})
print(f"{low.threshold:.3f} {high.threshold:.3f}")   # the threshold is recomputed per context
# -> -1.292 -1.499       (stricter where the deployment weight is large)
```

### `Gate.record_outcome` and `Gate.recalibrate`

```py
gate.record_outcome(decision_or_decision_id, unsafe: bool) -> bool
gate.recalibrate(episodes, **calibrate_kwargs) -> Calibration
```

`record_outcome` reports the realised outcome of an executed decision. The label can arrive later, and the decision can be identified by its `decision_id`. It returns `True` if the monitor knew the decision. It returns `False` if there is no monitor, the decision is unknown (only `CERTIFY` decisions are registered), or the outcome was already reported. Outcomes feed `Monitor.observed_risk` and the tighten-only online threshold.

`recalibrate` calls `calibrate(episodes, **kw)` and rebinds the monitor, which resets the martingale. Use it after a drift alarm, with fresh labelled episodes.

```python
gm = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1, monitor=rg.Monitor(alarm_level=20))
gm.calibrate(episodes)
d = gm.check(rg.Action("add", {"v": 0}), {"level": 50, "evidence": 1.0})
print(d.verdict, gm.record_outcome(d.decision_id, unsafe=False), gm.record_outcome(d, unsafe=False))
# -> Verdict.CERTIFY True False        (the second report is a duplicate)
print(gm.monitor.observed_risk)          # -> 0.0
gm.recalibrate([episode(rng) for _ in range(400)])
print(gm.status())                       # -> GuaranteeStatus.VALID
```

### `Gate.select_probe` and `Gate.margin_matrix`

```py
gate.margin_matrix(actions, hyps, context) -> (np.ndarray[n_actions, n_hyps], list[list[str]])
gate.select_probe(actions, plausible, margins, context, lam) -> (Action | None, resolved_share)
```

`margin_matrix` returns the margins and the name of the binding constraint for every action × hypothesis pair. It uses `predict_batch` if the model has one, otherwise `parallel` threads or a sequential loop, and the LRU cache if `cache_size > 0`.

`select_probe` implements the probe rule ([SPEC §5.4.2](SPEC.md)):

- *Disagreeing pairs* are pairs of plausible hypotheses for which some candidate is certifiable (margin ≥ `lam`) under one and not the other.
- A probe is *admissible* if, under every plausible hypothesis, its margin is ≥ `lam` or ≥ the `noop` margin − `noop_tol`.
- A pair is *resolved* if the probe's predicted `observation`s differ by ≥ `min_separation`.

It returns the admissible probe that resolves the largest share of pairs (ties broken by mean capped separation), provided that share is ≥ `min_resolved`, otherwise `(None, 0.0)`. Probe outcomes are evaluated by the spec too, so they must contain the spec's keys as well as `observation`.

```python
gp = rg.Gate(Tank(), spec, LeakHyps(), probes=[rg.Action("read_flowmeter", read_only=True)],
             noop=rg.Action("add", {"v": 0}), min_separation=0.5)
ctx = {"level": 28, "evidence": 8.0}
plausible = [h for h in LeakHyps().hypotheses(ctx) if h.score <= cal.tau]
print([h.name for h in plausible])       # -> ['leak0', 'leak10']
cands = [rg.Action("add", {"v": 0}), rg.Action("add", {"v": 5})]
m, binding = gp.margin_matrix(cands, plausible, ctx)
print(np.round(m, 2).tolist())           # -> [[1.8, 0.8], [2.3, 1.3]]
print(gp.select_probe(cands, plausible, m, ctx, lam=1.0))   # "hold" is certifiable under leak0 only
# -> (Action(name='read_flowmeter', args={}, read_only=True), 1.0)
```

### `Gate.fingerprint`, `Gate.status`, `Gate.load_calibration`

```py
gate.fingerprint() -> str                       # 16 hex characters
gate.status() -> GuaranteeStatus
gate.load_calibration(cal_or_path, key=None, require_signature=False) -> Calibration
```

The **fingerprint** hashes:

- the spec fingerprint, including a content hash of the functions inside `Custom`/`Predicate` constraints;
- `consequence.version` and `hypotheses.version` (class names if absent);
- the probe keys;
- each hard rule's name, verdict and logic (`HardRule.version` if set, otherwise a content hash of `fires`);
- the mode, and `doer_id` in policy mode;
- content hashes of `score_fn` and `weight_fn`.

**Status** is `VOID` if no calibration is loaded or the artefact's fingerprint differs from the gate's. It is `DEGRADED` if `monitor.alarm` is set, and `VALID` otherwise. `load_calibration` accepts a `Calibration` or a path. With `key`, the HMAC signature is verified (see [`Calibration.load`](#calibration)). Loading also sets `gate.mode` to the artefact's mode, so a policy-mode artefact always binds `doer_id`.

> [!IMPORTANT]
> Callables are hashed with `riskgate.types.callable_digest`: their source code (bytecode if the source is unavailable), default values and simple closure values. The hash is deliberately conservative, so editing the function's source, even a comment on the same line, voids the artefact. To pin a callable's identity explicitly, set `fn.riskgate_version = "..."` (or `HardRule(version=...)`). Model and hypothesis-provider *classes* are **not** hashed: bump their `version` strings whenever their behaviour changes.

```python
print(gate.fingerprint(), gate.status())     # -> 476aaff52ab03d1c GuaranteeStatus.VALID
Tank.version = "tank-v2"                     # e.g. the model was retrained
print(gate.fingerprint(), gate.status())     # -> e67ca5ec81092636 GuaranteeStatus.VOID
print(gate.check(rg.Action("add", {"v": 0}), {"level": 50, "evidence": 1.0}).reason)
# -> guarantee void: gate not calibrated or a component changed since calibration
Tank.version = "tank-v1"
print(gate.status())                         # -> GuaranteeStatus.VALID
```

### Attributes

The constructor arguments are stored as attributes with the same names: `consequence`, `spec`, `hypotheses`, `alpha`, `mode`, `delta`, `hard_rules`, `probes`, `noop`, `noop_tol`, `min_separation`, `min_resolved`, `ood_alpha`, `audit`, `monitor`, `require_calibration`, `doer_id`, `timeout_s`, `parallel`, `weight_fn`, `score_fn` and `cache_size`. `gate.calibration` holds the loaded `Calibration`, or `None`. The status is recomputed on every check, so changing a fingerprinted attribute (for example `gate.doer_id`) takes effect immediately.

---

## Data types

### `Action`

```py
@dataclass(frozen=True)
class Action:
    name: str
    args: Mapping[str, Any] = {}
    read_only: bool = False          # marks information-gathering actions usable as probes
    key: str                         # property: "name" or "name({canonical JSON of args})"
    def to_dict(self) -> dict
```

`key` identifies an action in labels (`Episode.unsafe`, `Episode.after_probe`), caches and the fingerprint (probe keys). Arguments are serialised as canonical JSON with sorted keys, so `{"v": 20}` and `{"v": 20.0}` give **different** keys. Actions are not hashable, because `args` is a dict. Use `action.key` as the dictionary key.

### `Hypothesis`

```py
@dataclass(frozen=True)
class Hypothesis:
    name: str                        # unique within one hypotheses() call
    family: str | None = None        # calibration label (e.g. all magnitudes of one fault type)
    params: Mapping[str, Any] = {}   # whatever your consequence model needs
    score: float = 0.0               # nonconformity: lower = explains the evidence better
    fam: str                         # property: family or name
```

```python
a = rg.Action("add", {"v": 20})
print(a.key, rg.Action("add", {"v": 20.0}).key, a.to_dict())
# -> add({"v":20}) add({"v":20.0}) {'name': 'add', 'args': {'v': 20}, 'read_only': False}
print(rg.Hypothesis("leak10", params={"leak": 10}).fam)          # -> leak10
```

### `Episode`

```py
@dataclass
class Episode:
    context: Any                              # what your model and hypothesis provider receive
    candidates: list[Action]                  # the doer's proposals, in rank order
    unsafe: Mapping[str, bool]                # action.key -> True if executing it violated (would violate) the spec
    true_family: str | None = None            # label of the true hypothesis family (calibrates tau, tau_ood)
    after_probe: Mapping[str, Episode] = {}   # probe.key -> the episode as it looks after running that probe
    meta: dict = {}                           # free-form; meta["probe_unsafe"]=True marks harm caused by a probe
```

Candidates without a label count as **unsafe**, which is conservative. Agnostic mode needs a label for every candidate. See [Bring your own domain](byo_domain.md#5-build-labelled-episodes) for how to build episodes.

### `Decision`

| Field | Type | Meaning |
|---|---|---|
| `verdict` | `Verdict` | `CERTIFY`, `REJECT`, `PROBE` or `ASK`. |
| `action` | `Action` or `None` | For `CERTIFY`, the certified candidate. For other verdicts after scoring, the top viable candidate. For an ASK rule, the candidate it fired on. For a void status, a hard-rule `REJECT`, a timeout or an error, the first candidate. |
| `score` | `float` | Robust margin of `action`. NaN if no margins were computed (void, hard rules, OOD, empty plausible set, timeout, error). |
| `top_score` | `float` | Best robust score among the viable candidates. −∞ for hard-rule exits, OOD and an empty plausible set; NaN if not computed. This is the drift monitor's statistic. |
| `threshold` | `float` | λ in use, or NaN if the check stopped before scoring. |
| `margins` | `dict[str, float]` | Margin of `action` under each plausible hypothesis. |
| `binding` | `dict[str, str]` | Binding (minimum-margin) constraint under each plausible hypothesis. |
| `plausible` | `list[str]` | Names of the plausible hypotheses. |
| `probe` | `Action` or `None` | The read-only action to run first (`PROBE` only). |
| `reason` | `str` | Human-readable explanation (prefixes below). |
| `status` | `GuaranteeStatus` | Guarantee status at the start of the check. |
| `calibration_id` | `str` or `None` | Id of the artefact in use. |
| `considered` | `list[dict]` | Every candidate looked at, in rank order: `{"action", "score", "certified"}` or `{"action", "verdict", "rule"}`. |
| `stage` | `int` | `1`, or `2` after a probe. |
| `latency_ms` | `float` | Wall-clock time of the check. |
| `agent_reason` | `str` | Numbers-free text the **agent** may see (no robust margins, no threshold, no exception internals). `""` means `reason` is safe to show as is. Adapters send `agent_reason or reason`; `reason` stays in audit records, `on_ask` and approver prompts. |
| `would_verdict` | `str \| None` | Set when a `VOID` artefact (with `require_calibration=False`) turned `CERTIFY` or `PROBE` into `ASK`: the verdict it would have given. |
| `decision_id` | `str` | Random 16-hex id. Pass it to `record_outcome`. |
| `approved` | `bool` (property) | `verdict is Verdict.CERTIFY`. |
| `to_dict()` | method | JSON-ready dict: enums as strings, actions as dicts (NaN and ±inf stay floats). |

| Verdict | `reason` starts with |
|---|---|
| `CERTIFY` | `robust margin above calibrated threshold` |
| `PROBE` | `plausible hypotheses disagree about safety; probe '<name>' is expected to resolve N% of the disagreements` |
| `REJECT` | `blocked by hard rules (...)` · `unsafe under every plausible hypothesis (binding constraint: <name>)` |
| `ASK` | `guarantee void: ...` · `hard rule '<name>': <message>` · `evidence unlike anything seen in calibration (out of distribution)` · `no hypothesis explains the evidence` · `cannot certify: best robust margin ... < threshold ...` · `timeout: ...` · `fail-closed: <Exception>: ...` |

Adapters never send the `cannot certify` numbers, or exception details, to the agent. They send `Decision.agent_reason` when it is set, for example `cannot certify: not safe enough under every plausible explanation of the situation (worst case: '<hypothesis>')`. The full `reason` stays in the audit log. A calibrated threshold and margins in front of a manipulated agent would show it how far to push.

```python
d = gate.check(rg.Action("add", {"v": 45}), {"level": 50, "evidence": 1.0})
print(d.verdict, d.approved, d.binding, "|", d.reason)
# -> Verdict.CERTIFY True {'leak0': '10<=level<=100', 'leak10': '10<=level<=100'} | robust margin above calibrated threshold
print(sorted(d.to_dict()))
# -> ['action', 'agent_reason', 'binding', 'calibration_id', 'considered', 'decision_id', 'latency_ms', 'margins',
#     'plausible', 'probe', 'reason', 'score', 'stage', 'status', 'threshold', 'top_score', 'verdict', 'would_verdict']
```

### `Verdict` and `GuaranteeStatus`

Both are `str` enums, so `Verdict("certify") is Verdict.CERTIFY` and `d.verdict == "certify"`.

| `Verdict` | Value | Meaning |
|---|---|---|
| `CERTIFY` | `"certify"` | Execute: the calibrated risk bound covers this decision. |
| `REJECT` | `"reject"` | Do not execute: unsafe under every plausible hypothesis, or blocked by a hard rule. The agent should re-plan. |
| `PROBE` | `"probe"` | First run the read-only `decision.probe`, then re-check with `stage=2`. |
| `ASK` | `"ask"` | Escalate to a human or a safe fallback. |

| `GuaranteeStatus` | Value | Meaning |
|---|---|---|
| `VALID` | `"valid"` | Calibrated, fingerprints match, no drift alarm. |
| `DEGRADED` | `"degraded"` | The drift alarm is raised. The threshold can only tighten, and the formal bound is not guaranteed. |
| `VOID` | `"void"` | Not calibrated, or a component changed since calibration. |

What to do in each status: [Operations](operations.md#guarantee-status-what-it-means-and-what-to-do).

---

## `Calibration`

A dataclass, the calibration artefact (JSON schema `riskgate.calibration/v1`).

| Field | Meaning |
|---|---|
| `alpha`, `mode`, `delta` | Risk level, loss mode and plausible-set miscoverage used. |
| `tau`, `tau_ood` | Plausible-set and out-of-distribution thresholds (+∞ without a τ split). |
| `lam` | Single-stage threshold λ̂. +∞ means "certify nothing". |
| `fingerprint` | Gate fingerprint at calibration time. |
| `n` | Number of risk-control episodes (excludes the τ split). |
| `lam_probe1`, `lam_probe2` | Two-stage thresholds λ₁ and λ₂ (`None` unless probing was calibrated). |
| `probe_share`, `n_probe` | Stage-1 share of α, and number of split-B episodes the gate would probe. |
| `created` | ISO-8601 UTC timestamp. |
| `guarantee`, `delta_ltt` | `"expectation"` or `"high_probability"`, and the LTT level (high-probability only). |
| `extra_alphas` | `{name: alpha_j}` of the jointly controlled extra risks. |
| `curve` | About 200 points `[λ, empirical safety risk, ask rate]`. |
| `reference_scores` | Best robust score of each risk-control episode (at most 5,000). The monitor's reference sample. |
| `weighted` | Stored grid, weights and losses for weighted CRC, or `None`. |
| `diagnostics` | Always: `tau_coverage_target`, `calibration_ask_rate_at_lambda`, `calibration_risk_at_lambda`, `mean_plausible_hypotheses`, `n_tau`, `evaluation_failures`. Plus `calibration_risk_<name>` per extra risk. With probes: `probe_episodes`, `probe_episodes_missing_post_probe_data`, `probe_split_b`, or `probing_disabled` (reason). |

```py
cal.id -> str                    # 12 hex characters: digest of (fingerprint, created, lam, n)
cal.probing -> bool              # both probe thresholds set and lambda_2 finite
cal.to_dict() -> dict            # adds "id" and "schema"; +-inf encoded as "inf" / "-inf"
cal.save(path, key=None)         # JSON; with key: adds an HMAC-SHA256 "signature"
Calibration.load(path, key=None, require_signature=False) -> Calibration     # staticmethod
```

`Calibration.load` checks the signature whenever `key` is given or `require_signature=True`. It raises [`SignatureError`](#errors) if the artefact is unsigned, if no key is given, or if the signature does not match (tampered file or wrong key). Without a key and without `require_signature`, any signature is ignored. The HMAC gives tamper evidence, not secrecy: the artefact itself is plain JSON.

```python
cal.save("cal.json", key="s3cret")
again = rg.Calibration.load("cal.json", key="s3cret", require_signature=True)
print(again.id == cal.id, again.lam == cal.lam, sorted(cal.diagnostics))
# -> True True ['calibration_ask_rate_at_lambda', 'calibration_risk_at_lambda', 'evaluation_failures',
#               'mean_plausible_hypotheses', 'n_tau', 'tau_coverage_target']
try:
    rg.Calibration.load("cal.json", key="wrong")
except rg.SignatureError as e:
    print("SignatureError:", e)
# -> SignatureError: calibration artefact signature does not match (tampered or wrong key)
```

---

## Specs

Everything in this section lives in `riskgate.spec` and is re-exported by `riskgate`. A **margin** is positive when a constraint holds and negative when it is violated. Its magnitude is in units of `scale`, so that the margins of different constraints are comparable.

### Constraints

```py
Upper(key, limit, scale=1.0, name=None)      # (limit - max(x)) / scale;            default name "key<=limit"
Lower(key, limit, scale=1.0, name=None)      # (min(x) - limit) / scale;            default name "key>=limit"
Envelope(key, lo, hi, scale=1.0, name=None)  # min(hi - max(x), min(x) - lo) / scale; default name "lo<=key<=hi"
Predicate(name, fn, safe_margin=1.0, unsafe_margin=-1.0)   # fn(outcome) -> bool (True = satisfied)
Custom(name, fn)                                            # fn(outcome) -> float (positive = safe)
```

`x = outcome[key]` for a mapping, or `getattr(outcome, key)` for an object. It can be a scalar or a trajectory: `max`/`min` run over all samples. For temporal properties use [`riskgate.stl`](#riskgatestl).

### `Spec`

```py
Spec(*constraints, floor=None)
spec.evaluate(outcome) -> (margin, binding_constraint_name)    # min over constraints
spec.margin(outcome) -> float
spec.violated(outcome) -> bool                                  # margin < 0
spec.fingerprint() -> str
```

A `Spec` is the conjunction of its constraints, so its margin is the minimum over them. A NaN margin counts as −∞, so the gate fails closed. `floor` optionally clips margins from below (for example `-5.0`) to keep scores finite. Calibration sees the same clipped scores, so validity is unaffected, but candidates below the floor become indistinguishable. Keep `floor` negative. `Spec()` with no constraints raises `ValueError`.

```python
s = rg.Spec(rg.Upper("p95_ms", 800, scale=800), rg.Upper("error_rate", 0.02, scale=0.02),
            rg.Lower("level", 10, scale=10, name="min level"))
print(s.evaluate({"p95_ms": [420, 610, 980], "error_rate": 0.004, "level": 55}))   # -> (-0.225, 'p95_ms<=800')
print(s.evaluate({"p95_ms": 500, "error_rate": 0.01, "level": float("nan")}))      # -> (-inf, 'min level')
print(rg.Spec(rg.Predicate("no open chargeback", lambda o: not o["chargeback"])).margin({"chargeback": True}))  # -> -1.0
```

### Writing your own constraint

Subclass `riskgate.spec.Constraint`, set `name` and implement `margin(outcome) -> float`. `describe()` feeds the spec fingerprint. By default it contains the class name, every attribute, and a content hash for attributes that are functions. Override `describe()` if an attribute is large or not JSON-serialisable.

```python
from riskgate.spec import Constraint


class Budget(Constraint):
    """Total spend must stay below the budget; margin in units of 10 % of the budget."""

    def __init__(self, budget: float):
        self.name, self.budget = f"spend<={budget:g}", float(budget)

    def margin(self, outcome) -> float:
        return (self.budget - float(sum(outcome["spend"]))) / (0.1 * self.budget)


print(rg.Spec(Budget(1000)).evaluate({"spend": [300, 450]}))        # -> (2.5, 'spend<=1000')
```

### `HardRule`

```py
@dataclass(frozen=True)
class HardRule:
    name: str
    fires: Callable[[Action, Any], bool]       # fires(action, context) -> True when the rule applies
    verdict: Verdict = Verdict.REJECT          # REJECT (never execute) or ASK (needs a human)
    message: str = ""
    version: str | None = None                 # pins the rule's identity; default: content hash of `fires`
    def describe(self) -> list                 # [name, verdict, version or callable_digest(fires)]: fingerprint material
```

Hard rules are deterministic, action-level checks that run before any prediction (see [`check_ranked`](#gatecheck-and-gatecheck_ranked), step 2). Only `REJECT` and `ASK` have an effect: a rule can never certify. Editing a rule's logic or verdict changes the gate fingerprint and voids the artefact. Calibration applies the same screening as deployment. Rules can also be declared in configuration files ([`rule_from_config`](#riskgateconfig)).

```python
rules = [rg.HardRule("no draining", lambda a, c: a.args.get("v", 0) < 0, rg.Verdict.REJECT, "never drain"),
         rg.HardRule("big adds need approval", lambda a, c: a.args.get("v", 0) >= 45, rg.Verdict.ASK, ">= 45 L")]
gr = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1, hard_rules=rules)
gr.calibrate(episodes)
for v in (-10, 45):
    d = gr.check(rg.Action("add", {"v": v}), {"level": 50, "evidence": 1.0})
    print(v, d.verdict, "|", d.reason)
# -> -10 Verdict.REJECT | blocked by hard rules (add({"v":-10}): no draining)
# -> 45 Verdict.ASK | hard rule 'big adds need approval': >= 45 L
d = gr.check_ranked([rg.Action("add", {"v": 0}), rg.Action("add", {"v": 45})], {"level": 0, "evidence": 1.0})
print(d.verdict, d.considered)   # add(45) would be safe, but it needs approval: never certified automatically
# -> Verdict.REJECT [{'action': 'add({"v":45})', 'verdict': 'ask', 'rule': 'big adds need approval'},
#                    {'action': 'add({"v":0})', 'score': -2.0, 'certified': False}]
```

---

## Component protocols and helpers

### `ConsequenceModel` (protocol)

```py
class ConsequenceModel(Protocol):
    version: str                                                        # part of the fingerprint
    def predict(self, action, hypothesis, context) -> Outcome: ...
    # optional, strongly recommended for speed:
    def predict_batch(self, actions, hypotheses, context) -> list[list[Outcome]]: ...   # [action][hypothesis]
```

An *Outcome* is any mapping or object that the spec's constraints can read. Probe outcomes must also include `observation`: a vector in noise units (see [Bring your own domain](byo_domain.md#1-write-a-consequence-model)). The protocol is `runtime_checkable`, so `isinstance(model, rg.ConsequenceModel)` works.

### `HypothesisProvider` (protocol)

```py
class HypothesisProvider(Protocol):
    version: str
    def hypotheses(self, context) -> list[Hypothesis]: ...     # scored; lower score = more plausible
```

### `SingleHypothesis`

`SingleHypothesis(name="nominal")`: for domains without latent-state ambiguity. It always returns one hypothesis with score 0 (`version = "single-v1"`).

### `CallableModel`

`CallableModel(fn, version="callable-v1", batch_fn=None)` wraps plain functions as a consequence model. `fn(action, hypothesis, context) -> outcome`. `batch_fn(actions, hypotheses, context) -> [[outcome]]` becomes `predict_batch`.

```python
model = rg.CallableModel(lambda a, h, ctx: {"level": ctx["level"] + a.args["v"]}, version="no-leak-v1")
g1 = rg.Gate(model, spec, rg.SingleHypothesis(), alpha=0.1)
print(isinstance(model, rg.ConsequenceModel), g1.hypotheses.hypotheses(None))
# -> True [Hypothesis(name='nominal', family=None, params={}, score=0.0)]
```

---

## `Monitor`

`riskgate.Monitor` provides online drift detection and conservative threshold adaptation. See [Operations → Monitoring](operations.md#monitoring-drift-alarm-and-tighten-only-threshold) for how to operate it.

```py
Monitor(alarm_level=100.0, eta=None, seed=0, latch=True)
```

| Member | Description |
|---|---|
| `alarm_level` | Alarm threshold for the martingale. By Ville's inequality the false-alarm probability is about ≤ 1/`alarm_level` under exchangeability. |
| `eta` | Step size of the online threshold. `None` means 0.05 × (90th − 10th percentile) of the reference scores, re-derived at every `bind`. |
| `latch` | With `True` (default), once the martingale crosses `alarm_level` the alarm stays on (status `DEGRADED`) until `bind()` or `reset()`. A falling martingale is not evidence that the drift has ended. |
| `bind(calibration)` | Loads `reference_scores` and `alpha`, sets `online_lambda = calibration.lam` (or `None` if infinite), clears `pending` and `outcomes`, and calls `reset()`. The gate calls it from `load_calibration`, `calibrate` and `recalibrate`. |
| `reset()` | Restarts the drift test: martingale back to 1, `history` and `n_seen` cleared, latched alarm cleared. Outcome counters are kept. |
| `update_score(score)` | Turns one score into a conformal p-value against the reference scores and updates the martingale. A small p means an unusually *low* margin. The gate calls it with `decision.top_score`, the best robust score among the viable candidates (the same statistic as the reference scores), for every stage-1 decision while the status is not `VOID`. |
| `martingale`, `log_martingale` | Current value of the mixture power martingale, and its logarithm (properties). `martingale` returns `inf` instead of overflowing. |
| `alarm` | `True` once bound and either latched or `martingale >= alarm_level` (property). |
| `online_lambda` | ACI-style online threshold. The gate uses `max(λ, online_lambda)` while `DEGRADED`. |
| `observe_outcome(certified, unsafe)` | Immediate outcome feedback. For certified actions, `online_lambda += eta * (1{unsafe} - alpha)`. |
| `register(decision)` | Called by the gate for every `CERTIFY`: remembers `decision_id` for a later outcome. |
| `report_outcome(decision_id, unsafe)` | Delayed outcome. Returns `False` if the id is unknown, was already reported, or was certified under a previous calibration. |
| `observed_risk` | Unsafe share among certified decisions with reported outcomes since the last `bind`, or `None` (property). |
| `history`, `n_seen`, `pending`, `max_pending`, `outcomes` | Martingale path since the last reset; scores seen; `decision_id -> time` of open decisions (at most `max_pending = 100_000`, oldest evicted); outcome counts. |

```python
mon = rg.Monitor(alarm_level=20)
gd = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1, monitor=mon)
gd.calibrate(episodes)
for _ in range(40):                       # deployment looks far more dangerous than calibration
    gd.check(rg.Action("add", {"v": 40}), {"level": 75, "evidence": 25.0})
print(gd.status(), f"martingale={mon.martingale:.3g}", mon.alarm)
# -> GuaranteeStatus.DEGRADED martingale=4.62e+20 True
for _ in range(400):                      # benign traffic afterwards: the martingale falls ...
    gd.check(rg.Action("add", {"v": 0}), {"level": 55, "evidence": -5.0})
print(gd.status(), f"martingale={mon.martingale:.3g}")   # ... but the alarm stays latched
# -> GuaranteeStatus.DEGRADED martingale=4.68e-05
mon.reset()                               # explicit decision by an operator (or recalibrate)
print(gd.status())                        # -> GuaranteeStatus.VALID
```

---

## Audit sinks

Any object with `write(decision, **fields)` can be passed as `Gate(audit=...)`. The gate calls `audit.write(decision)` once per decision.

### `AuditLog`

`AuditLog(path, extra=None, redact=None)` writes an append-only JSONL file, creating parent directories. Each line is `{"ts": ..., **extra, **fields, **decision.to_dict()}`, passed through `redact(record) -> record` if given. Decision fields win over `extra` keys with the same name. Writes are protected by a lock. The `context` is never logged, but action arguments are, so use `redact` for sensitive values.

### `MemoryAudit`

`MemoryAudit()` keeps records in the `.records` list (for tests and notebooks).

### `MultiSink`

`MultiSink(*sinks)` fans each decision out to several sinks. An exception in one sink is swallowed and never blocks the others or the decision.

### `summarize_audit`

`summarize_audit(path) -> dict` returns verdict and status counts and latency percentiles (p50/p95/p99) of a JSONL audit log. The CLI equivalent is `riskgate audit-summary`.

```python
import hashlib, json

def redact(rec):
    rec = dict(rec, margins=None)                                  # drop per-hypothesis margins
    if rec.get("action"):
        digest12 = hashlib.sha256(json.dumps(rec["action"]["args"], sort_keys=True).encode()).hexdigest()[:12]
        rec["action"] = dict(rec["action"], args=digest12)        # hash the tool arguments
    return rec

mem = rg.MemoryAudit()
ga = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1,
             audit=rg.MultiSink(rg.AuditLog("logs/decisions.jsonl", extra={"service": "tank-1"}, redact=redact), mem))
ga.load_calibration(cal)
for v in (0, 20, 80):
    ga.check(rg.Action("add", {"v": v}), {"level": 50, "evidence": 1.0})
print(len(mem.records), json.loads(open("logs/decisions.jsonl").readline())["action"])
# -> 3 {'name': 'add', 'args': '50057fe3c90a', 'read_only': False}
print(rg.summarize_audit("logs/decisions.jsonl")["verdicts"])     # -> {'certify': 2, 'reject': 1}
```

---

## Tool adapter: `guard`

```py
guard(gate, *, context=lambda: None, name=None, on_ask=None, raise_on_reject=False, stage=lambda: 1, feedback=None)
```

`guard` is a decorator factory for any Python tool function, sync or `async`. On each call it binds the arguments (defaults included) to `Action(name or fn.__name__, arguments)`. It then calls `gate.check(action, context(), stage=stage())` and routes the verdict. If `context()` raises, the call is treated as `ASK` (`fail-closed: context unavailable (...)`).

| Verdict | Result |
|---|---|
| `CERTIFY` | The tool runs and its return value is returned. |
| `ASK` + `on_ask` given | `on_ask(decision)` decides. `True` runs the tool. `False` returns `{"status": "declined", "reason", "guarantee", "executed": false}`. |
| `ASK`, no `on_ask` | Raises `ApprovalRequired(decision)`. With `raise_on_ask=False`, returns `{"status": "needs_approval", ...}`: nobody could be asked. |
| `REJECT` | Returns `{"status": "rejected", "reason", "guarantee", "executed": false}`, or raises `ActionRejected(decision)` if `raise_on_reject=True`. |
| `PROBE` | Returns `{"status": "probe_requested", "reason", "guarantee", "probe": {...}}`. |

`feedback` formats the dict that is returned for blocked calls, for example `feedback=json.dumps` for frameworks that expect string tool results. The wrapper exposes the gate as `wrapper.__riskgate__`. For `async` tools the gate check runs synchronously inside the coroutine. `ActionRejected` and `ApprovalRequired` are `RuntimeError` subclasses that carry `.decision`.

```python
state = {"level": 50, "evidence": 1.0}

@rg.guard(gate, context=lambda: state, on_ask=lambda d: False)   # on_ask: your human-approval hook
def add(v: float) -> str:
    state["level"] += v
    return f"added {v} L"

print(add(20))       # -> added 20 L
print(add(80))
# -> {'status': 'rejected', 'reason': 'unsafe under every plausible hypothesis (binding constraint: 10<=level<=100)',
#     'guarantee': 'valid'}

@rg.guard(gate, context=lambda: state, name="add")              # no on_ask: ASK raises
def add_strict(v: float) -> str:
    return "ran"

state["evidence"] = 60.0                                        # a glitching sensor: unlike any calibration episode
try:
    add_strict(5)
except rg.ApprovalRequired as e:
    print("ApprovalRequired:", e.decision.verdict, "|", e)
# -> ApprovalRequired: Verdict.ASK | evidence unlike anything seen in calibration (out of distribution)
```

## Errors

| Exception | Base | Raised by |
|---|---|---|
| `SignatureError` | `ValueError` | `Calibration.load` / `Gate.load_calibration`: missing, unverifiable or wrong HMAC signature. |
| `ActionRejected` | `RuntimeError` | `guard(..., raise_on_reject=True)` on `REJECT`. `.decision` holds the decision. |
| `ApprovalRequired` | `RuntimeError` | `guard(...)` without `on_ask` on `ASK`. `.decision` holds the decision. |
| `ValueError` | built-in | `Gate(mode=...)` with an unknown mode, or `cache_size > 0` without `context_key`. `calibrate` with too few episodes, an unknown `guarantee`/`mode` or mismatched `weights`. `Spec()` with no constraints. Invalid configuration files. |

The gate itself never raises during `check`. Model, hypothesis and spec errors and timeouts become `ASK` decisions, and an audit sink error downgrades a `CERTIFY` to `ASK`.

---

## `riskgate.config`

Declarative configuration in JSON, TOML or YAML. No code is ever evaluated from configuration files. The full schema is in [Configuration](configuration.md).

```py
load_config(path) -> dict                         # .json, .toml, .yaml/.yml (YAML needs pyyaml); else ValueError
spec_from_config(items, floor=None) -> Spec       # items: [{"type": "upper"|"lower"|"envelope", ...}]
rule_from_config(item) -> HardRule                # {"name", "action"?, "arg"?, "op"?, "value"?, "verdict"?, "message"?}
gate_from_config(cfg, consequence, hypotheses=None, **overrides) -> Gate
```

`gate_from_config` reads these sections:

- `gate`;
- `spec` (required) and `spec_floor`;
- `hard_rules`, `monitor` and `audit`;
- `calibration`, `calibration_key_env` and `require_signature`.

All of them except `gate` may also be nested under `gate`. The artefact named by `calibration` is verified with the HMAC key read from the environment variable named by `calibration_key_env`. Keyword `overrides` win over the file. Use them for objects that a file cannot express (`context_key`, `weight_fn`, `score_fn`, a custom `audit` sink). Rules built by `rule_from_config` get `version = "cfg:<digest of the rule table>"`, so editing a rule in the file voids the artefact.

```python
from riskgate.config import gate_from_config, rule_from_config, spec_from_config

cfg = {"gate": {"alpha": 0.1, "cache_size": 64},
       "spec": [{"type": "envelope", "key": "level", "lo": 10, "hi": 100, "scale": 10}],
       "hard_rules": [{"name": "no draining", "action": "add", "arg": "v", "op": "<", "value": 0}]}
gc = gate_from_config(cfg, Tank(), LeakHyps(), context_key=lambda c: (c["level"], c["evidence"]))
print(gc.alpha, gc.cache_size, [r.name for r in gc.hard_rules], gc.spec.constraints[0].name)
# -> 0.1 64 ['no draining'] 10<=level<=100
print(rule_from_config(cfg["hard_rules"][0]).fires(rg.Action("add", {"v": -5}), None))   # -> True
```

## `riskgate.telemetry.OTelSink`

```py
OTelSink(tracer=None, meter=None, name="riskgate")
```

An audit sink that emits one span `riskgate.check` per decision and four instruments:

| Instrument | Kind | Attributes / unit |
|---|---|---|
| `riskgate.decisions` | counter | `verdict`, `status` |
| `riskgate.latency_ms` | histogram | unit `ms` |
| `riskgate.margin` | histogram | non-negative robust margins (finite only), attribute `verdict` |
| `riskgate.margin_deficit` | histogram | −margin for negative robust margins, attribute `verdict`. OTel histograms accept only non-negative values, so each decision contributes to exactly one of the two. |

Span attributes:

- always: `riskgate.verdict`, `riskgate.status`, `riskgate.action` (tool name), `riskgate.stage`, `riskgate.calibration_id`, `riskgate.decision_id`, `riskgate.probe`;
- when finite: `riskgate.score` and `riskgate.threshold`;
- one `riskgate.extra.<field>` per extra `write()` field.

Without `tracer`/`meter`, the global OpenTelemetry API is used, which raises `ImportError` if `opentelemetry-api` is missing. Combine it with an audit log through `MultiSink`.

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from riskgate.telemetry import OTelSink

spans, reader = InMemorySpanExporter(), InMemoryMetricReader()
tp = TracerProvider()
tp.add_span_processor(SimpleSpanProcessor(spans))
sink = OTelSink(tracer=tp.get_tracer("riskgate"), meter=MeterProvider(metric_readers=[reader]).get_meter("riskgate"))
go = rg.Gate(Tank(), spec, LeakHyps(), alpha=0.1, audit=rg.MultiSink(rg.MemoryAudit(), sink))
go.load_calibration(cal)
go.check(rg.Action("add", {"v": 20}), {"level": 50, "evidence": 1.0})
span = spans.get_finished_spans()[0]
print(span.name, span.attributes["riskgate.verdict"], span.attributes["riskgate.status"])   # -> riskgate.check certify valid
print(sorted(m.name for rm in reader.get_metrics_data().resource_metrics for sm in rm.scope_metrics for m in sm.metrics))
# -> ['riskgate.decisions', 'riskgate.latency_ms', 'riskgate.margin']
```

## `riskgate.doers.LLMRanker`

```py
LLMRanker(base_url=None, api_key=None, model=None, timeout=60.0, temperature=0.0, system_prompt=DEFAULT_SYSTEM)
ranker.rank(situation: str, candidates: Sequence[Action], policy: str = "") -> list[Action]
ranker.doer_id -> str            # "llm:<model>:t<temperature>:<digest of system prompt>"
ranker.last_rationale, ranker.last_error
```

A *doer* that ranks candidate actions with any OpenAI-compatible `/chat/completions` endpoint, using only the standard library. It sits outside the trust boundary, so every proposal still goes through the gate. Defaults come from `RISKGATE_LLM_BASE_URL` (default `https://api.openai.com/v1`), `RISKGATE_LLM_API_KEY` and `RISKGATE_LLM_MODEL` (default `gpt-4o-mini`). Localhost endpoints bypass HTTP proxies. On any failure, `rank` returns the candidates in the given order and sets `last_error`.

In policy mode, pass `Gate(doer_id=ranker.doer_id)`. `doer_id` covers the model, the temperature and the system prompt. It does not cover the `policy` text passed to `rank()` or the endpoint, so add those yourself if they change.

```python
from riskgate.doers import LLMRanker
from riskgate.types import digest

RUNBOOK = "Keep the tank between 10 and 100 L. Prefer the smallest move."
ranker = LLMRanker(base_url="http://127.0.0.1:9/v1", api_key="x", model="local-model", timeout=0.5)   # nothing listens here
print([a.key for a in ranker.rank("level 50 L, flow 1 L/min", ACTIONS[:3], policy=RUNBOOK)])
# -> ['add({"v":40})', 'add({"v":20})', 'add({"v":0})']     (failure: the given order is kept)
print(ranker.last_error is not None, ranker.doer_id)          # -> True llm:local-model:t0.0:01be3f88
gate_llm = rg.Gate(Tank(), spec, LeakHyps(), doer_id=f"{ranker.doer_id}:{digest(RUNBOOK, 8)}")
```

---

## `riskgate.conformal`

The distribution-free primitives that `Gate.calibrate` uses (`conformal_quantile` and `crc_threshold` are also exported by `riskgate`).

```py
conformal_quantile(scores, alpha) -> float          # ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small
lambda_grid(score_lists) -> np.ndarray              # distinct finite scores + [+inf]
agnostic_losses(scores, unsafe, grid) -> np.ndarray # L_i(lam) = 1{some unsafe candidate has score >= lam}
policy_losses(scores, unsafe, grid, envelope=True)  # L_i(lam) = loss in [0, 1] of the doer's first candidate with score >= lam; upper envelope
crc_threshold(losses, grid, alpha, bound=1.0) -> float   # smallest lam with n/(n+1)*mean L + B/(n+1) <= alpha
ask_rate(scores, grid) -> np.ndarray                # share of episodes with no certified candidate, per lam
```

Each loss matrix has shape `(n_episodes, n_grid)`, and its rows must be non-increasing along the ascending grid, with L(+∞) = 0.

```python
from riskgate.conformal import agnostic_losses, crc_threshold, lambda_grid

scores = [np.array([0.2, 1.5]), np.array([-0.4, 0.9]), np.array([2.0, 0.1])] * 10
unsafe = [np.array([True, False]), np.array([False, False]), np.array([False, True])] * 10
grid = lambda_grid(scores)
print(rg.conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.2), crc_threshold(agnostic_losses(scores, unsafe, grid), grid, 0.2))
# -> 8.0 0.9
```

## `riskgate.stl`

Signal temporal logic with quantitative robustness for discrete-time signals. The robustness ρ is a signed margin, so `STLConstraint` puts a formula straight into a `Spec`. Read the module docstring for the full semantics: truncated windows on finite signals, empty windows, and NaN propagation.

```py
Pred(key, op, threshold, scale=1.0)       # op ">=": (x - c)/scale;  op "<=": (c - x)/scale
Not(phi); And(*phis); Or(*phis); Implies(phi, psi)          # also: phi & psi, phi | psi, ~phi
Always(phi, a=0, b=None); Eventually(phi, a=0, b=None)      # [a, b] in integer steps; b=None: to the end
Until(phi, psi, a=0, b=None, *, strict=False)
formula.rho(signals) -> np.ndarray        # robustness trace for t = 0..T-1
formula.robustness(signals, t=0) -> float
formula.keys() -> frozenset[str]; formula.horizon -> int | None
STLConstraint(name, formula)              # Constraint with margin = robustness at t = 0 (NaN -> -inf)
```

```python
from riskgate.stl import Always, Eventually, Pred, STLConstraint

band = Always(Pred("T", "<=", 362, scale=10) & Pred("T", ">=", 340, scale=10))     # stay in the band
settle = Eventually(Always(Pred("T", "<=", 352, scale=10), 0, 3), 0, 6)            # within 6 steps, hold <= 352 for 4 steps
traj = {"T": [355, 358, 356, 351, 350, 349, 350, 351, 350, 349, 350]}
print(round(band.robustness(traj), 3), round(settle.robustness(traj), 3), settle.horizon)   # -> 0.4 0.1 9
s_stl = rg.Spec(STLConstraint("340<=T<=362", band), STLConstraint("settles", settle))
print(s_stl.evaluate(traj))                                                         # -> (0.1, 'settles')
```

`STLConstraint.describe()` embeds the exact `repr` of the formula, so editing a formula changes the spec fingerprint and voids old artefacts. Predicted trajectories should be at least `formula.horizon + 1` samples long: a window that runs past the end of the signal is truncated, and an empty one makes `Always` vacuously +∞.

## `riskgate.multirisk`

Learn-then-Test (Angelopoulos et al., 2021): high-probability control of one or several risks with a single threshold. `Gate.calibrate(guarantee="high_probability")` uses `ltt_threshold` with Hoeffding–Bentkus p-values.

```py
binomial_pvalue(risk_hat, n, alpha)       # exact; binary losses only
hoeffding_pvalue(risk_hat, n, alpha)      # any loss in [0, 1]
hb_pvalue(risk_hat, n, alpha)             # Hoeffding-Bentkus (default)
PVALUES = {"binomial": ..., "hoeffding": ..., "hb": ...}
ltt_pvalues(loss_mats, alphas, pvalue="hb", *, bounds=1.0) -> np.ndarray          # (n_risks, n_grid)
ltt_threshold(loss_mats, grid, alphas, delta, pvalue="hb", *, bounds=1.0) -> float
```

`ltt_threshold` runs fixed-sequence testing from the most conservative grid point down. It returns the smallest λ rejected before the first non-rejection, or +∞. With i.i.d. episodes, P(R_j(λ̂) ≤ α_j for all j) ≥ 1 − δ.

```python
from riskgate.multirisk import hb_pvalue, ltt_threshold

safety = agnostic_losses(scores, unsafe, grid)
print(f"{hb_pvalue(0.02, 300, 0.1):.1e}", ltt_threshold([safety], grid, [0.3], delta=0.1))   # -> 1.1e-07 0.9
```

## `riskgate.weighted`

Conformal calibration under covariate shift with a known likelihood ratio w(x) = dQ/dP (Tibshirani et al., 2019). The threshold depends on the test context through w(x_test), so it is recomputed for every decision. Weights only need to be known up to a common factor.

```py
weighted_conformal_quantile(scores, w_cal, w_test, alpha) -> float | np.ndarray
WeightedCRC(losses, grid, alpha, w_cal, bound=1.0).threshold(w_test) -> float | np.ndarray   # O(log n_grid) per call
weighted_crc_threshold(losses, grid, alpha, w_cal, w_test, bound=1.0) -> float | np.ndarray
```

With equal weights they reduce exactly to `conformal_quantile` and `crc_threshold`. A large test weight makes the threshold more conservative, up to +∞ (certify nothing).

```python
from riskgate.weighted import WeightedCRC

L = agnostic_losses(scores, unsafe, grid)
wcrc = WeightedCRC(L, grid, 0.2, w_cal=np.ones(len(scores)))
print(wcrc.threshold(1.0) == crc_threshold(L, grid, 0.2), wcrc.threshold(1.0), wcrc.threshold(5.0), wcrc.threshold(10.0))
# -> True 0.9 0.9 inf
```

---

## Utilities (`riskgate.types`)

```py
canonical(obj) -> str            # deterministic JSON (sorted keys); callables become "callable:<digest>"
digest(obj, n=16) -> str         # first n hex characters of sha256(canonical(obj))
callable_digest(fn) -> str       # content hash of a function: fn.riskgate_version if set, else source/bytecode,
                                 # defaults and simple closure values (used by fingerprints)
```

`Action.key`, fingerprints and artefact ids are built from these. Set `fn.riskgate_version = "v2"` on a function (rule, constraint, `score_fn`, `weight_fn`) to pin its identity, so that cosmetic edits do not void artefacts.

---

## Framework adapters

These adapters are thin and duck-typed. Each module imports its framework lazily, and the names are also importable lazily from `riskgate.adapters` (for example `from riskgate.adapters import gated_tool_node`). Usage, verified versions and human-in-the-loop flows are in [Integrations](integrations.md).

| Module | Function | Purpose |
|---|---|---|
| `riskgate.adapters.tool` | `guard` | Any Python tool function ([above](#tool-adapter-guard)). |
| `riskgate.adapters.frameworks` | `adk_before_tool_callback(gate, *, context, on_ask=None, confirm=True, passthrough=())` | Google ADK `before_tool_callback`. Returning a dict skips the tool. ASK uses ADK tool confirmation. |
| | `adk_plugin(gate, *, context, on_ask=None, confirm=True, passthrough=(), name="riskgate")` | ADK `BasePlugin` that gates every agent of a Runner. |
| | `langgraph_gate_node(gate, *, context, interrupt=None, on_ask=None, messages_key="messages", decisions_key="riskgate", message_format="auto")` and `route_after_gate(state, tools_node="tools", model_node="agent", ...)` | LangGraph gate node between the model and the `ToolNode`, plus its conditional edge. |
| | `langgraph_tool_wrapper`, `langgraph_async_tool_wrapper`, `gated_tool_node(gate, tools, *, context, interrupt=None, on_ask=None, **tool_node_kwargs)` | `ToolNode` `wrap_tool_call` hooks / a gated `ToolNode`. |
| `riskgate.adapters.openai_agents` | `guarded_function_tool(gate, fn, *, context, on_ask=None, name=None, **function_tool_kwargs)` | Agents SDK `FunctionTool` over `guard(fn)`. |
| | `riskgate_tool_input_guardrail(gate, *, context, name="riskgate", halt_on=())` | Native `ToolInputGuardrail`. |
| | `riskgate_needs_approval(gate, tool_name, *, context)` | `needs_approval` callable: True exactly when the gate says ASK. |
| `riskgate.adapters.mcp_proxy` | `build_proxy_server(gate, upstream_session, *, context, name, on_ask, passthrough, version, instructions)`, `serve_stdio(gate, upstream_command, upstream_args=(), *, context, env, cwd, name, on_ask, passthrough)`, `main(argv)` | MCP proxy that gates `tools/call` of an upstream MCP server (see also [`riskgate serve-mcp`](cli.md#riskgate-serve-mcp)). |
