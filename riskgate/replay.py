"""Replay: run labelled history through a gate and report what it would have done (the onboarding "replay" step).

    report = replay(gate, held_out_episodes)          # gate already calibrated on OTHER episodes
    print(report)                                     # unsafe 1.2% (agent alone 41.0%) · autonomous 72.5% · ...

    study = resplit(make_gate, episodes, alpha=0.05, n_splits=50, groups=[ep.meta["task"] for ep in episodes])
    print(study["unsafe_mean"], study["frac_above_alpha"])

A decision counts as **unsafe** when the gate certified an action that was labelled unsafe (or a probe that caused
harm, ``meta["probe_unsafe"]``); REJECT and ASK never execute anything. PROBE verdicts are followed through
``Episode.after_probe`` exactly as in deployment (stage 2); a probe with no post-probe episode counts as ASK.

Replay on the episodes you calibrated on is optimistic: always hold out (``resplit`` does it for you).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .gate import Episode, Gate
from .types import Verdict

__all__ = ["ReplayOutcome", "ReplayReport", "replay", "replay_episode", "resplit", "sweep"]


@dataclass
class ReplayOutcome:
    """What the gate did with one episode."""

    verdict: str                     # final verdict: certify, reject, ask (a probe is followed to its stage-2 verdict)
    action: str | None               # key of the executed action (certify only)
    unsafe: bool                     # an unsafe action (or harmful probe) was executed
    probed: str | None = None        # key of the probe run first, if any
    first_choice_unsafe: bool = False
    safe_option_existed: bool = True
    decision_id: str = ""


def replay_episode(gate: Gate, ep: Episode) -> ReplayOutcome:
    first = ep.candidates[0].key if ep.candidates else None
    base = {"first_choice_unsafe": bool(ep.unsafe.get(first, True)) if first is not None else False,
            "safe_option_existed": any(not ep.unsafe.get(a.key, True) for a in ep.candidates)}
    d = gate.check_ranked(ep.candidates, ep.context)
    probed = None
    harm = False
    if d.verdict is Verdict.PROBE and d.probe is not None:
        probed = d.probe.key
        ep2 = ep.after_probe.get(probed)
        if ep2 is None:
            return ReplayOutcome("ask", None, False, probed, decision_id=d.decision_id, **base)
        harm = bool(ep2.meta.get("probe_unsafe", False))
        d = gate.check_ranked(ep2.candidates, ep2.context, stage=2)
        labels = ep2.unsafe
    else:
        labels = ep.unsafe
    if d.verdict is Verdict.CERTIFY and d.action is not None:
        return ReplayOutcome("certify", d.action.key, bool(labels.get(d.action.key, True)) or harm, probed,
                             decision_id=d.decision_id, **base)
    verdict = "ask" if d.verdict is Verdict.PROBE else d.verdict.value      # a stage-2 PROBE cannot happen; be safe
    return ReplayOutcome(verdict, None, harm, probed, decision_id=d.decision_id, **base)


def _rate(x: Sequence[bool]) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class ReplayReport:
    """Rates over the replayed episodes (shares of episodes, in [0, 1])."""

    n: int
    unsafe: float                    # certified and unsafe (what alpha bounds)
    unsafe_ci: tuple[float, float]   # 95% Wilson interval
    agent_alone_unsafe: float        # the agent's first choice, executed unchecked
    certified: float                 # executed autonomously (with or without a probe first)
    rejected: float
    asked: float                     # went to a person
    probed: float                    # a read-only probe ran first
    avoidable_ask: float             # asked or rejected although a labelled-safe candidate existed
    alpha: float | None = None
    status: str = ""
    outcomes: list[ReplayOutcome] = field(default_factory=list, repr=False)

    def to_dict(self, outcomes: bool = False) -> dict:
        d = asdict(self)
        if not outcomes:
            d.pop("outcomes")
        return d

    def __str__(self) -> str:
        pct = lambda x: f"{100 * x:.1f}%"   # noqa: E731
        head = f"replayed {self.n} episodes" + (f" at alpha={self.alpha:g}" if self.alpha is not None else "")
        return (f"{head} [{self.status}]: unsafe {pct(self.unsafe)} (95% CI {pct(self.unsafe_ci[0])}-"
                f"{pct(self.unsafe_ci[1])}; agent alone {pct(self.agent_alone_unsafe)}) · autonomous {pct(self.certified)}"
                f" · asked {pct(self.asked)} · rejected {pct(self.rejected)} · probed first {pct(self.probed)}")


def replay(gate: Gate, episodes: Sequence[Episode]) -> ReplayReport:
    """Replay `episodes` through `gate` (as deployed: hard rules, OOD check, probing, fail-closed)."""
    outs = [replay_episode(gate, ep) for ep in episodes]
    n = len(outs)
    unsafe = _rate([o.unsafe for o in outs])
    cal = gate.calibration
    return ReplayReport(
        n=n, unsafe=unsafe, unsafe_ci=_wilson(unsafe, n) if n else (0.0, 1.0),
        agent_alone_unsafe=_rate([o.first_choice_unsafe for o in outs]),
        certified=_rate([o.verdict == "certify" for o in outs]), rejected=_rate([o.verdict == "reject" for o in outs]),
        asked=_rate([o.verdict == "ask" for o in outs]), probed=_rate([o.probed is not None for o in outs]),
        avoidable_ask=_rate([o.verdict != "certify" and o.safe_option_existed for o in outs]),
        alpha=cal.alpha if cal is not None else None, status=gate.status().value, outcomes=outs)


def resplit(make_gate: Callable[[], Gate], episodes: Sequence[Episode], *, alpha: float = 0.05,
            n_splits: int = 20, cal_fraction: float = 0.6, groups: Sequence[Hashable] | None = None,
            seed: int = 0, **calibrate_kw: Any) -> dict:
    """Validity study: for each of `n_splits` random splits, calibrate a fresh gate on `cal_fraction` of the
    episodes (by group if `groups` is given) and replay the rest.

    Conformal risk control bounds the unsafe rate in expectation over such splits, so ``unsafe_mean`` should be at
    most alpha (up to sampling noise in the held-out sets); ``frac_above_alpha`` shows how often one split exceeds it.
    """
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError(f"cal_fraction must be in (0, 1), got {cal_fraction}")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    episodes = list(episodes)
    if groups is not None and len(groups) != len(episodes):
        raise ValueError("groups must have one entry per episode")
    labels = list(groups) if groups is not None else list(range(len(episodes)))
    units = list(dict.fromkeys(labels))
    rows = []
    for r in range(n_splits):
        perm = np.random.default_rng(seed + r).permutation(len(units))
        cal_units = {units[i] for i in perm[: int(round(cal_fraction * len(units)))]}
        cal_idx = [i for i, g in enumerate(labels) if g in cal_units]
        test_idx = [i for i, g in enumerate(labels) if g not in cal_units]
        gate = make_gate()
        gate.calibrate([episodes[i] for i in cal_idx], alpha=alpha, seed=seed + r,
                       groups=[labels[i] for i in cal_idx] if groups is not None else None, **calibrate_kw)
        rep = replay(gate, [episodes[i] for i in test_idx])
        rows.append(rep)
    u = np.array([x.unsafe for x in rows])
    return {"alpha": alpha, "n_splits": n_splits, "n_test_mean": float(np.mean([x.n for x in rows])),
            "unsafe_mean": float(u.mean()), "unsafe_sd": float(u.std()), "frac_above_alpha": float(np.mean(u > alpha)),
            "agent_alone_unsafe": float(np.mean([x.agent_alone_unsafe for x in rows])),
            "certified_mean": float(np.mean([x.certified for x in rows])),
            "asked_mean": float(np.mean([x.asked for x in rows])),
            "rejected_mean": float(np.mean([x.rejected for x in rows])),
            "probed_mean": float(np.mean([x.probed for x in rows])),
            "splits": [x.to_dict() for x in rows]}


def sweep(make_gate: Callable[[], Gate], calibration: Sequence[Episode], test: Sequence[Episode],
          alphas: Sequence[float] = (0.02, 0.05, 0.10), **calibrate_kw: Any) -> list[dict]:
    """The error budget as a dial: calibrate at each alpha on `calibration`, replay `test`, and report how the work
    moves between autonomous actions, probes and people (and which actions run: tightening alpha typically moves
    work from irreversible to reversible actions when the agent ranks both)."""
    rows = []
    for a in alphas:
        gate = make_gate()
        gate.calibrate(list(calibration), alpha=a, **calibrate_kw)
        rep = replay(gate, test)
        executed: dict[str, int] = {}
        for o in rep.outcomes:
            if o.action is not None:
                name = o.action.split("(", 1)[0]
                executed[name] = executed.get(name, 0) + 1
        rows.append({"alpha": a, "lambda": gate.calibration.lam if gate.calibration else None, **rep.to_dict(),
                     "executed": {k: v / rep.n for k, v in sorted(executed.items())} if rep.n else {}})
    return rows
