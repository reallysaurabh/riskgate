# Bring your own domain

> [!TIP]
> Have agent traces and outcome labels but no simulator? Start with [Start from your traces](from_traces.md): it learns
> the plausible worlds from the traces and calibrates a gate without any of the code below. Come back here when you
> want ranked alternatives, probes with a consequence model, or a digital twin.

This guide walks through connecting riskgate to a new domain, step by step. Each step comes with a template you can copy. The running example is a **reservoir operator agent**: every hour it proposes a water release for the next 12 hours, and riskgate certifies, probes, rejects or asks.

| You provide | Interface | Step |
|---|---|---|
| A **consequence model**: simulator, digital twin or dry-run | `version`, `predict(action, hypothesis, context)`, optionally `predict_batch` | [1](#1-write-a-consequence-model) |
| A **hypothesis provider**: what could explain the evidence | `version`, `hypotheses(context) -> [Hypothesis]` | [2](#2-write-a-hypothesis-provider) |
| A **spec**: what "unsafe" means, as signed margins | `Spec(...)` of constraints, optionally STL | [3](#3-write-the-spec) |
| **Hard rules**: deterministic action-level policy | `HardRule(name, fires, verdict, message)` | [4](#4-add-hard-rules) |
| **Labelled episodes**: past or simulated decision points with ground truth | `Episode(context, candidates, unsafe, true_family, after_probe)` | [5](#5-build-labelled-episodes) |
| The **risk level** and the amount of data | `alpha`, `delta`, `n` | [6](#6-choose-alpha-delta-and-n) |

Before writing code, fix four things on paper:

- the **decision point**: when the agent proposes and riskgate checks;
- the **actions**: tool calls with arguments;
- the **context**: the snapshot available at decision time, and nothing from the future;
- **unsafe**: the outcome you want to happen at most α of the time, over a stated horizon.

Every Python block on this page runs top to bottom in one session, and the `# ->` comments show real output.

## 1. Write a consequence model

The consequence model answers: *if we execute this action and the world is in this state, what happens?* Riskgate's guarantee does not depend on its accuracy ([why](calibration.md)): a poor model costs you `ASK`s, not safety. Rules for the model:

- **Version it.** `version` is part of the gate fingerprint. Bump it whenever the model, its parameters or its code change, or old artefacts stay `VALID` for a model they were never calibrated on.
- **Return outcomes the spec can read**: a mapping (or an object with attributes) whose values are scalars or trajectories (1-D arrays).
- **Implement `predict_batch(actions, hypotheses, context)`** if you can vectorise. It returns a list per action of outcomes per hypothesis (`out[i][j]`). Without it, the gate calls `predict` for every pair, optionally in `parallel` threads.
- **Probes** are read-only actions (`read_only=True`). Their outcome needs the spec's keys too (what happens to the system *while* the probe runs, usually the same as doing nothing), plus `observation`: a vector of what the probe would measure, **in noise units**. For example `log(value) / relative_noise`, or `value / sd`. Then `min_separation=2.0` means "the hypotheses' predicted readings are at least two noise standard deviations apart".
- **No side effects, deterministic.** Dry-runs must hit isolated copies (database clones or branches, restored snapshots, staging namespaces) or be read-only rewrites. A savepoint on the live database is not isolation; see [Safe database dry-runs](operations.md#safe-database-dry-runs). Exceptions are fine: the check fails closed (`ASK`). NaN outputs count as violations.

```python
import math

import numpy as np

import riskgate as rg

H = 12                                   # hourly steps in the prediction horizon
K = 3600.0 / 1.0e6                       # metres of level per (m3/s of net inflow) per hour; surface 1 km2
GAUGE_NOISE, PROBE_NOISE = 0.25, 0.05    # log-normal noise of the routine gauge and of the precise ridge gauge
PROBE = rg.Action("read_upstream_gauge", {"station": "ridge-2"}, read_only=True)
HOLD = rg.Action("hold")                 # keep the current release (the "do nothing" action)


class ReservoirTwin:
    """Level trajectory for a release q under a hypothesised inflow (mass balance)."""

    version = "reservoir-twin-v1"        # bump on every change

    def predict_batch(self, actions, hyps, ctx):
        q = np.array([ctx["q_now"] if a.name in ("hold", PROBE.name) else a.args["q"] for a in actions])
        inflow = np.array([h.params["inflow"] for h in hyps])
        t = np.arange(1, H + 1)
        level = ctx["level"] + K * (inflow[None, :, None] - q[:, None, None]) * t      # (actions, hyps, H)
        out = []
        for i, a in enumerate(actions):
            row = []
            for j, h in enumerate(hyps):
                o = {"level": level[i, j], "downstream": float(q[i])}
                if a.read_only:                                   # what the precise gauge would read, in noise units
                    o["observation"] = [math.log(h.params["inflow"]) / PROBE_NOISE]
                row.append(o)
            out.append(row)
        return out

    def predict(self, action, hyp, ctx):
        return self.predict_batch([action], [hyp], ctx)[0][0]
```

## 2. Write a hypothesis provider

Hypotheses are the explanations of the current state that the evidence cannot rule out: fault types, user intents, load regimes. The gate keeps every hypothesis whose score is at most τ, and takes the **worst case** over them.

- **Cover the space.** Use a grid over the latent variables (here, inflow regimes × magnitudes; the k8s example has 13 incident causes × magnitudes). Include the nominal "nothing is wrong" case.
- **Score = nonconformity: lower is a better fit, and comparable across contexts.** Two good choices:
    - *normalised residuals*: the RMS of (observed − predicted) / noise over the evidence;
    - *negative log posterior*.

    τ is a single global threshold, so a score must mean the same thing whether you have one reading or ten. RMS rather than a sum is what makes this work.
- **Families** group hypotheses of the same kind (all magnitudes of a "flood"). Calibration sets τ from the best score within the *true family*, so `Episode.true_family` must use exactly the same strings as `Hypothesis.family`.
- **Version it**, like the model.
- An LLM may *propose extra hypotheses*. That can only enlarge the plausible set and make the gate more cautious ([rule R1](concepts.md)). It must never remove hypotheses or override scores.
- If there is no hidden state, use `rg.SingleHypothesis()`.

```python
GRID = [("normal", 80.0), ("normal", 120.0), ("high", 200.0), ("high", 280.0), ("flood", 400.0)]


class InflowDiagnoser:
    """Which inflow explains the gauge readings? Score = RMS of log-residuals in noise units."""

    version = "inflow-diagnoser-v1"

    def hypotheses(self, ctx):
        hyps = []
        for family, inflow in GRID:
            z = [math.log(ctx["gauge"] / inflow) / GAUGE_NOISE]
            if ctx.get("probe_gauge") is not None:                # after the probe: a much sharper reading
                z.append(math.log(ctx["probe_gauge"] / inflow) / PROBE_NOISE)
            hyps.append(rg.Hypothesis(f"{family}:{inflow:g}", family=family, params={"inflow": inflow},
                                      score=math.sqrt(sum(v * v for v in z) / len(z))))
        return hyps


print([(h.name, round(h.score, 2)) for h in InflowDiagnoser().hypotheses({"gauge": 160.0})])
# -> [('normal:80', 2.77), ('normal:120', 1.15), ('high:200', 0.89), ('high:280', 2.24), ('flood:400', 3.67)]
```

## 3. Write the spec

The spec turns an outcome into a **signed margin**: positive is safe, negative is a violation, and the magnitude says by how much. A `Spec` is the conjunction of its constraints, so its margin is the **minimum** over them, and units must be comparable.

- **Normalise.** Choose each `scale` so that one margin unit means the same severity everywhere. Here 1 unit is 20% of the allowed range: 1 m of level against a 5 m half-band, and 50 m³/s against a 250 m³/s limit. Common choices are `scale=limit` (relative margins) or a tolerance.
- **Trajectories.** `Upper`, `Lower` and `Envelope` take the max/min over every sample.
- **Temporal properties.** Use STL (`riskgate.stl`). For example, "never outside 95–105 m, and back inside the 96–104 m operating band from hour 6 on" cannot be written as a plain envelope. STL robustness *is* a signed margin ([API](api.md#riskgatestl)).
- **Booleans and anything else.** Use `Predicate(name, fn, safe_margin, unsafe_margin)`, with margins on the same scale as the other constraints, or `Custom(name, fn)`. Their functions are content-hashed into the fingerprint.
- **`floor`** (for example `-10`) keeps scores finite. Keep it negative.

```python
from riskgate.stl import Always, Pred, STLConstraint

safe_band = Always(Pred("level", ">=", 95.0, scale=1.0) & Pred("level", "<=", 105.0, scale=1.0))
operating_band = Always(Pred("level", ">=", 96.0, scale=1.0) & Pred("level", "<=", 104.0, scale=1.0), 5)  # samples 5.. = hours 6..12
spec = rg.Spec(
    STLConstraint("95 m <= level <= 105 m", safe_band),
    STLConstraint("96 m <= level <= 104 m from hour 6", operating_band),
    rg.Upper("downstream", 250.0, scale=50.0, name="downstream <= 250 m3/s"),
    floor=-10.0,
)
margin, binding = spec.evaluate({"level": np.linspace(103.0, 104.8, H), "downstream": 100.0})   # rises to 104.8 m
print(round(margin, 3), "|", binding)
# -> -0.8 | 96 m <= level <= 104 m from hour 6
```

## 4. Add hard rules

Hard rules are deterministic, action-level checks that run **before** any prediction:

- `REJECT` means never execute: limits, deny-lists, protected resources.
- `ASK` means a human must approve: large or irreversible changes.

A rule sees the action and the context. Calibration applies the same screening as deployment. A rule's name, verdict and logic are part of the fingerprint, so editing one voids the artefact (pin a rule's identity with `HardRule(version=...)`). Simple rules can also live in [configuration files](configuration.md#hard_rules).

```python
rules = [
    rg.HardRule("spillway rating", lambda a, ctx: a.args.get("q", 0.0) > 350.0,
                rg.Verdict.REJECT, "the spillway is rated for 350 m3/s"),
    rg.HardRule("large changes need the duty engineer",
                lambda a, ctx: abs(a.args.get("q", ctx["q_now"]) - ctx["q_now"]) > 150.0,
                rg.Verdict.ASK, "release changes above 150 m3/s are approved by a human"),
]
```

## 5. Build labelled episodes

An [`Episode`](api.md#episode) is one decision point with ground truth:

| Field | What to put in it |
|---|---|
| `context` | Exactly what the gate will see at decision time. No information from after the decision. |
| `candidates` | The doer's proposals in its rank order. In **policy** mode they must come from the doer you will deploy. |
| `unsafe` | `{action.key: bool}`: would executing this candidate have violated the spec? These are **counterfactual** labels for every candidate, from a source of truth that is *not* your consequence model: a higher-fidelity simulator with randomised parameters, a sandbox replay, or expert review. Unlabelled candidates count as unsafe. |
| `true_family` | The true hypothesis family, if known. It calibrates τ and the OOD guard. Without it the plausible set is always every hypothesis, which is valid but conservative. |
| `after_probe` | `{probe.key: Episode}`: the same decision as it looks after running the probe (new context, labels at that time). Set `meta["probe_unsafe"] = True` on it if the probe or its delay caused harm. |

Label with **the same spec**, applied to the true outcome (`spec.violated(true_outcome)`), so that labels and margins can never disagree about what "unsafe" means.

| Requirement | Policy mode | Agnostic mode |
|---|---|---|
| Candidates | the deployed doer's real ranked proposals (`doer_id` is fingerprinted) | any candidate set, for example every action |
| Labels | every candidate the doer could end up executing: in practice all of them | every candidate |
| Guarantee | for this doer (model, prompt, temperature) | for any doer that only executes certified actions |
| Efficiency | tighter: fewer `ASK`s | more conservative |

```python
FAMILIES = {"normal": (60.0, 140.0), "high": (170.0, 300.0), "flood": (350.0, 450.0)}
RELEASES = [50.0, 100.0, 200.0, 300.0]


def true_outcome(level, inflow, q, rng):
    """The 'truth' simulator: continuous inflow and an uncertain surface area (not the twin's grid)."""
    k = K * rng.uniform(0.9, 1.1)
    return {"level": level + k * (inflow - q) * np.arange(1, H + 1), "downstream": q}


def doer_rank(ctx, rng):
    """Stand-in for the operator assistant: release about what flows in, with some noise."""
    return sorted(RELEASES, key=lambda q: abs(q - min(ctx["gauge"], 260.0)) + rng.normal(0, 30.0))


def episode(rng):
    family = str(rng.choice(list(FAMILIES), p=[0.6, 0.3, 0.1]))
    inflow = float(rng.uniform(*FAMILIES[family]))
    level, q_now = float(rng.uniform(97.0, 103.0)), float(rng.choice(RELEASES[:3]))
    ctx = {"level": level, "q_now": q_now, "gauge": inflow * math.exp(rng.normal(0, GAUGE_NOISE)), "probe_gauge": None}
    cands = [rg.Action("release", {"q": q}) for q in doer_rank(ctx, rng)]
    unsafe = {a.key: spec.violated(true_outcome(level, inflow, a.args["q"], rng)) for a in cands}   # same spec
    ep = rg.Episode(ctx, cands, unsafe, true_family=family)
    after = dict(ctx, probe_gauge=inflow * math.exp(rng.normal(0, PROBE_NOISE)))    # the probe is instantaneous here
    ep.after_probe = {PROBE.key: rg.Episode(after, cands, unsafe, true_family=family)}
    return ep


rng = np.random.default_rng(1)
episodes = [episode(rng) for _ in range(1000)]
first_unsafe = np.mean([ep.unsafe[ep.candidates[0].key] for ep in episodes])
print(episodes[0].candidates[0].key, f"doer's first choice unsafe in {first_unsafe:.1%} of episodes")
# -> release({"q":200.0}) doer's first choice unsafe in 26.8% of episodes
```

If the probe takes time, simulate the delay. Build the post-probe context and labels at the later decision time, and flag harm done in the meantime with `meta["probe_unsafe"]`.

## 6. Choose alpha, delta and n

- **α** is the tolerated rate of *certified and unsafe* executions, marginal over decision points. Pick it from the cost of a violation ("Choosing α and n" in the [calibration guide](calibration.md)). For catastrophic or irreversible harm, rely on hard rules and independent safety systems, not on statistics alone.
- **n**: conformal risk control needs `n/(n+1)·risk + 1/(n+1) ≤ α`, so at least ⌈1/α⌉ risk-control episodes (enforced). With so few, λ̂ is often +∞ and the gate certifies nothing. Plan for about 10/α, plus the τ split (`tau_fraction=0.3` of the episodes):

    | α | Minimum risk-control n | Recommended n | Total episodes (with the 30% τ split) |
    |---|---|---|---|
    | 0.10 | 10 | 100 | ≈ 145 |
    | 0.05 | 20 | 200 | ≈ 290 |
    | 0.02 | 50 | 500 | ≈ 715 |
    | 0.01 | 100 | 1,000 | ≈ 1,430 |

    Two-stage probing also needs split B (`probe_split` of the risk-control set) to hold at least ⌈1/((1 − `probe_share`)·α)⌉ episodes. `guarantee="high_probability"` needs at least log δ / log(1 − α) episodes before it can certify anything at all (45 at α = 0.05, δ = 0.1), and many more in practice.
- **δ** (plausible-set miscoverage) is an *efficiency* knob, not a validity knob. A smaller δ means larger plausible sets, more conservative scores and more `ASK`s. A larger δ makes the true hypothesis drop out more often, which calibration accounts for. 0.05–0.1 is typical. Check `mean_plausible_hypotheses` in the report: if it approaches the total number of hypotheses, the scores do not discriminate.

```python
gate = rg.Gate(ReservoirTwin(), spec, InflowDiagnoser(), alpha=0.1, delta=0.05, hard_rules=rules,
               probes=[PROBE], noop=HOLD, min_separation=2.0, doer_id="operator-assistant:v3")
cal = gate.calibrate(episodes)
print(f"lambda={cal.lam:.3f} lambda1={cal.lam_probe1:.3f} lambda2={cal.lam_probe2:.3f} probing={cal.probing} n={cal.n}")
# -> lambda=-1.575 lambda1=-0.990 lambda2=-0.883 probing=True n=700
print({k: cal.diagnostics[k] for k in ("mean_plausible_hypotheses", "calibration_ask_rate_at_lambda", "evaluation_failures", "probe_episodes")})
# -> {'mean_plausible_hypotheses': 2.0742857142857143, 'calibration_ask_rate_at_lambda': 0.2257142857142857,
#     'evaluation_failures': 0, 'probe_episodes': 66}
```

## 7. Use it, and check it

Every verdict appears in this domain:

```python
def show(ctx, q, stage=1):
    d = gate.check(rg.Action("release", {"q": q}), ctx, stage=stage)
    print(f"release {q:g}: {d.verdict.value.upper()} plausible={d.plausible} | {d.reason}")
    return d

normal = {"level": 100.0, "q_now": 100.0, "gauge": 105.0, "probe_gauge": None}
show(normal, 100)
# -> release 100: CERTIFY plausible=['normal:80', 'normal:120'] | robust margin above calibrated threshold
show(normal, 300)
# -> release 300: ASK plausible=[] | hard rule 'large changes need the duty engineer': release changes above 150 m3/s
#    are approved by a human
rising = {"level": 103.5, "q_now": 100.0, "gauge": 160.0, "probe_gauge": None}
d = show(rising, 100)
# -> release 100: PROBE plausible=['normal:120', 'high:200'] | plausible hypotheses disagree about safety; probe
#    'read_upstream_gauge' is expected to resolve 100% of the disagreements
show(dict(rising, probe_gauge=118.0), 100, stage=2)  # the precise gauge read 118 m3/s
# -> release 100: CERTIFY plausible=['normal:120'] | robust margin above calibrated threshold
show({"level": 104.0, "q_now": 100.0, "gauge": 420.0, "probe_gauge": None}, 50)
# -> release 50: REJECT plausible=['high:280', 'flood:400'] | unsafe under every plausible hypothesis (binding
#    constraint: 96 m <= level <= 104 m from hour 6)
```

Before trusting a gate, replay **held-out** episodes, with the probe loop, and compare the realised rate of certified-and-unsafe actions with α:

```python
held_out, executed_unsafe = [episode(rng) for _ in range(1000)], 0
for ep in held_out:
    d = gate.check_ranked(ep.candidates, ep.context)
    if d.verdict is rg.Verdict.PROBE:                       # run the probe, then re-check at stage 2
        ep = ep.after_probe[d.probe.key]
        d = gate.check_ranked(ep.candidates, ep.context, stage=2)
    executed_unsafe += d.verdict is rg.Verdict.CERTIFY and ep.unsafe[d.action.key]
print(f"certified-and-unsafe rate {executed_unsafe / len(held_out):.3f} (alpha = {gate.alpha})")
# -> certified-and-unsafe rate 0.042 (alpha = 0.1)
```

From here, sign and store the artefact, wire the gate into your agent framework, and set up monitoring: see [Operations](operations.md), [Integrations](integrations.md) and the [CLI](cli.md).

## Checklist of common mistakes

- [ ] **Forgetting to bump `version`.** Model and hypothesis-provider classes are not hashed. If you retrain or edit one without bumping `version`, the old artefact stays `VALID` for a component it never saw.
- [ ] **Mixing units across constraints.** The spec takes the minimum over constraints, so one unnormalised constraint (milliseconds next to a fraction) dominates every decision. Normalise every `scale` to a comparable severity.
- [ ] **Reusing calibration data.** Episodes used to fit or tune the model, the hypothesis scores, the spec scales or α are not exchangeable with deployment. Recalibrating repeatedly on the same set until λ "looks right" is selection. Keep a held-out calibration set and refresh it after every change.
- [ ] **Leaving candidates unlabelled.** Unlabelled candidates count as unsafe. That is valid, but the ask rate goes up. Label every candidate (agnostic mode needs it), or at least every candidate the doer could execute.
- [ ] **Label keys that do not match `Action.key`.** Build `unsafe` from the candidates themselves (`{a.key: ... for a in candidates}`). `{"q": 100}` and `{"q": 100.0}` are different keys.
- [ ] **Labelling with the gate's own model.** The labels must come from reality or a higher-fidelity truth, never from the consequence model you are calibrating.
- [ ] **Labels that disagree with the spec.** Compute them with `spec.violated(true_outcome)`, over the same horizon.
- [ ] **Policy mode with a different doer.** The policy-mode guarantee covers the calibrated doer only. Set `doer_id` (model, prompt, temperature) so that a swap makes the artefact `VOID`, or use agnostic mode.
- [ ] **Broken probe outcomes.** Probe outcomes need the spec keys and an `observation` in noise units, otherwise `min_separation` is meaningless and probes are never chosen, or chosen wrongly.
- [ ] **Hypothesis scores that are not comparable across contexts.** For example, raw likelihoods summed over evidence windows of different lengths. Normalise (RMS), or τ will be wrong for most contexts.
- [ ] **Family names that do not match.** `Episode.true_family` must equal a `Hypothesis.family` string, otherwise τ becomes +∞.
- [ ] **Future information in the context.** The calibration context must contain only what is known at decision time.
- [ ] **Unstable cache keys.** Contexts mutated in place while `cache_size > 0`: `context_key` must change whenever the content does.
- [ ] **Side effects in `predict`.** Dry-runs must never touch production.
- [ ] **Too few episodes.** λ̂ = +∞ means everything is `ASK`. See the table in [step 6](#6-choose-alpha-delta-and-n).
- [ ] **STL trajectories that are too short.** A bounded formula needs at least `formula.horizon + 1` samples.
- [ ] **Expecting per-situation guarantees.** The bound is marginal over decision points, not conditional on a given situation.
