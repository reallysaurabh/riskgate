# Calibration & guarantees

## What you need: labelled episodes

An `Episode` is one decision point:

```python
rg.Episode(
    context=snapshot,                  # whatever your models need
    candidates=[a1, a2, a3],           # what the agent proposed, in its order of preference
    unsafe={a1.key: True, a2.key: False, a3.key: False},   # would executing it violate the spec?
    true_family="slow_db_query",       # optional: the true hypothesis family (calibrates the plausible set)
    after_probe={probe.key: episode_after_probe},          # optional: two-stage probing
)
```

Where labels come from:

| Source | Gives labels for | Typical domains |
|---|---|---|
| High-fidelity simulator or digital twin | **every** candidate (counterfactual) | process plants, robotics, energy |
| Sandbox replay (staging namespace, database clone, payment test mode) | every candidate | infrastructure, transactional software |
| Audited production logs and postmortems | executed actions only | everything else; use policy mode, or label alternatives by expert review |

Unlabelled candidates are treated as **unsafe**. This is conservative and never invalidates the bound.

Calibration scores every episode with **exactly the deployed decision path**: hard rules are applied first (a
candidate removed by a `REJECT` rule, or skipped because an `ASK` rule fires, can never be certified), the
out-of-distribution check runs, and if the model or a rule raises an exception the episode is treated as "the gate
would ask" (loss 0), just as `check()` fails closed at run time. Failures are counted in
`calibration.diagnostics["evaluation_failures"]` and logged. So the calibrated risk is the risk of the gate you deploy.

## The two guarantee modes

Let s(a) be the robust score and λ the threshold. A candidate is *certified* if s(a) ≥ λ.

| Mode | Episode loss L(λ) | Guarantee | Needs |
|---|---|---|---|
| `agnostic` | 1 if **any** certified candidate is unsafe | P(executed action unsafe) ≤ α for **any** doer that only executes certified actions, even after you swap the LLM | labels for every candidate |
| `policy` (default) | 1 if the action the doer would actually execute (its first certified candidate) is unsafe; made monotone by its upper envelope sup over λ′ ≥ λ of L(λ′) | the same bound for the **calibrated** doer: fewer asks, but recalibrate when the model, prompt or temperature changes | labels along the doer's ranking |

Set `Gate(doer_id=...)` in policy mode so that a doer change voids the calibration automatically. Calibrating in policy
mode without one emits a `RiskGateWarning`: on public real-agent logs, a threshold calibrated on 2025 agents gave
14.6% unsafe on 2026 agents (τ²-bench airline, α = 10%), and swapping AgentDojo pipelines without recalibrating
exceeded α in 35% of pairs ([results on real agents](results_real_agents.md)).

## Conformal risk control (expectation guarantee)

With n exchangeable risk-control episodes and the grid Λ of distinct scores ∪ {+∞},

$$\hat\lambda=\min\Big\{\lambda\in\Lambda:\ \tfrac{n}{n+1}\hat R_n(\lambda)+\tfrac{1}{n+1}\le\alpha\Big\},\qquad \mathbb E[L_{n+1}(\hat\lambda)]\le\alpha .$$

This holds whatever the score function, consequence model or hypothesis provider (Angelopoulos et al., *Conformal Risk Control*, ICLR 2024). A good model buys **efficiency** (fewer `ASK`s), not validity.

> **Read this twice.** The expectation is over calibration draws. One calibrated gate can have a true risk above α,
> and by more when the risk curve has a *cliff* near λ̂ (many candidates with almost the same score and a high unsafe
> rate). In our reactor example the mean over random re-splits is below α, but one particular split showed about 9%
> on its test set. If you need the bound to hold for *your* artefact, use the high-probability guarantee below.

**Minimum n:** ⌈1/α⌉ risk-control episodes is enforced. Use at least 10/α (500 at α = 5%) for a usable threshold.

## High-probability guarantee (Learn-then-Test)

`gate.calibrate(episodes, guarantee="high_probability", delta_ltt=0.1)` returns λ̂ such that **P(risk(λ̂) ≤ α) ≥ 1 − δ** over the calibration draw. This is useful when you want the bound to hold for *this* artefact, not only on average. It uses fixed-sequence testing from the most conservative threshold downwards with Hoeffding–Bentkus p-values (`riskgate.multirisk`; Angelopoulos et al., *Learn then Test*, 2021). Expect a stricter threshold when the empirical risk is close to α; when the risk curve jumps from well below α to well above it (as in our refunds and Kubernetes examples) both guarantees pick the same threshold.

The p-values assume **i.i.d.** episodes (CRC only needs exchangeability). Probing is calibrated for the expectation guarantee only: with `guarantee="high_probability"` the gate does not probe (`diagnostics["probing_disabled"]` says why).

## Several risks at once

```python
cal = gate.calibrate(episodes, extra_losses={
    "cost_overrun": (lambda ep, a: float(a.args.get("factor", 1) >= 2), 0.20),   # big scale-outs at most 20%
})
```

Each extra loss is a function of the episode and the executed action, with values in [0, 1] (real values are kept, not rounded), plus its own level. In expectation mode every risk is controlled at its level, using the strictest of the individual CRC thresholds. In high-probability mode they are controlled jointly at family-wise level δ: a threshold is accepted only if *every* risk's p-value is below δ (an intersection–union test, so no multiplicity correction across risks is needed).

## Covariate shift with a known likelihood ratio

If you know how deployment differs from calibration, for example "night traffic is 3× more common in production", pass per-episode weights w(x) ∝ p_test(x)/p_cal(x) and a `weight_fn`:

```python
gate = rg.Gate(model, spec, hyps, weight_fn=lambda ctx: 3.0 if ctx.is_night else 1.0)
gate.calibrate(episodes, weights=[3.0 if ep.context.is_night else 1.0 for ep in episodes])
```

At each decision the threshold is recomputed with weighted CRC (Tibshirani et al., 2019; `riskgate.weighted`). Contexts that are rare in calibration but common in deployment get stricter thresholds.

## Dependent decisions: grouped calibration

One agent run often contains several consequential tool calls, and the same task is often run more than once (for
example several conversations, or retries). These decision points are **not exchangeable** with each other. What is
exchangeable is the unit they belong to: the conversation, the task or the incident. Pass one label per episode:

```python
cal = gate.calibrate(episodes, groups=[ep.meta["task_id"] for ep in episodes])
```

What happens:

- The τ split is drawn by group.
- Conformal risk control (or Learn-then-Test) runs on each group's **mean** loss, L_g(λ) = (1/|g|) Σ_{i∈g} L_i(λ).
  Each L_g is still non-increasing in λ, bounded by 1 and equal to 0 at +∞, so the usual argument applies to groups.
- The guarantee becomes **E[share of a new group's decisions that are certified and unsafe] ≤ α**: the risk
  averaged over tasks, not over individual calls.
- `cal.n` counts groups. You need at least ⌈1/α⌉ risk-control groups, which is more demanding than counting
  episodes: 50 tasks split in half cannot support α = 5%.
- `weights` cannot be combined with `groups`, and two-stage probing is disabled (`diagnostics["probing_disabled"]`).

**How many groups to collect.** Count independent tasks, conversations or incidents, not tool calls:

| Target | Risk-control groups needed | Why |
|---|---|---|
| CRC at α (anything certified at all) | ⌈1/α⌉: 10 at 10%, 20 at 5% | the finite-sample term 1/(n+1) must fit under α |
| CRC with a usable ask rate | several times more | with n ≈ 20 at α = 5%, 1/(n+1) ≈ 4.8% leaves almost no budget |
| Learn-then-Test at α, δ' (high probability) | ln δ' / ln(1 − α) loss-free groups: 45 at α = 5%, δ' = 0.1 | the most conservative threshold must already be rejected |

Add the τ split on top (default `tau_fraction=0.3`): about 29 groups in total for the CRC minimum at α = 5%. On public
benchmarks this was the binding constraint more than once: τ²-bench airline (37 tasks with writes, α = 5% infeasible),
ITBench (35 scenarios, α ≤ 10% impractical) and AgentDojo (about 48 groups per pipeline: LTT asks about everything).

`benchmarks/` uses this throughout. Its offline harness, `benchmarks/common.py`, is tested to return exactly the
threshold computed by `Gate.calibrate(groups=...)`.

## The plausible set and the OOD guard

A share `tau_fraction` (default 30%) of the labelled episodes is held out to set:

- τ, the conformal (1 − δ) quantile of the *true family's* best score, so that the true family is kept with probability ≥ 1 − δ;
- τ_ood, the 99% quantile of the best score. Evidence that fits worse than anything seen in calibration yields `ASK`.

The split matters. If the same episodes chose τ and λ, the score would depend on the risk-control data, and validity could break. Our first prototype did this and showed a 7% unsafe rate at α = 5% in one domain.

δ affects **efficiency only**: a larger plausible set means more conservative scores and more `ASK`s, but never invalidity.

## Probing (two-stage calibration)

With probes configured and episodes carrying `after_probe`:

1. Split the risk-control episodes into A and B (`probe_split`, default 0.5).
2. λ₁ is CRC on A at level ρα (`probe_share` ρ, default 0.6). This bounds the risk of certifying *before* any probe.
3. λ₂ is CRC on **all** of B at level (1 − ρ)α. An episode's stage-2 loss is 0 if stage 1 would certify it or would
   not probe. Otherwise it is the harm caused by the probe itself (`meta["probe_unsafe"]` on the post-probe episode;
   this does not depend on λ₂) plus the loss of the post-probe decision. A probe the episode has no `after_probe`
   data for counts as a loss of 1 (conservative).
4. Total risk: P(unsafe after stage 1) + P(probe harm or unsafe after stage 2) ≤ ρα + (1 − ρ)α = **α**.

τ comes from its own split, so nothing is reused across stages. If B is smaller than ⌈1/((1 − ρ)α)⌉ episodes, or no
λ₂ meets its budget, probing is **disabled** in that artefact and the single-stage threshold is used.

> An earlier version calibrated λ₂ only on the probed episodes at level α. That bounds the total risk by
> ρα + α·P(probe), which exceeds α when most episodes are probed; in our reactor example (about 80% probed) it
> exceeded α in 9 of 10 random splits. Fixed in 1.0.0rc1.

## What is *not* guaranteed

- Conditional (per-situation) risk. The bound is marginal over episodes.
- Validity under distribution shift. The drift monitor detects it, the gate only tightens, and you should recalibrate.
- Multi-step closed-loop guarantees. Each decision point is calibrated on its own.
- Correct labels. Garbage labels produce a garbage bound.

## Choosing α and n

| Situation | Suggestion |
|---|---|
| Costly but recoverable harm (cloud cost, minor SLO breach) | α = 0.05–0.10, expectation mode |
| Serious harm (money out of the door, equipment) | α = 0.01–0.02, high-probability mode, plus hard rules for catastrophic cases |
| Catastrophic, irreversible harm | do not rely on statistics alone: hard rules and independent certified safety systems |
| n | ≥ 10/α risk-control episodes plus ≈ 30% for τ; probing needs split B ≥ 1/((1 − ρ)α) episodes, and many more than that before it beats single-stage gating |
