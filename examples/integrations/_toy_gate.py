"""Toy domain shared by the integration examples: a tank with a hidden leak.

Same domain as ``tests/test_core.py`` and ``tests/contract``. The agent's only world-changing tool is
``add(v)`` (add ``v`` litres); the safe envelope is 10 <= level <= 100; a hidden leak drains 0, 10 or 25
litres and is only noisily observed. With ``STATE = {"level": 50, "evidence": 0}`` the calibrated gate says:

    add(v=10) -> CERTIFY   add(v=50) -> ASK (operator approval rule)   add(v=90) -> REJECT

``PrintAudit`` prints every decision as the gate makes it, so each example shows what the gate did.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import riskgate as rg

STATE: dict[str, float] = {"level": 50.0, "evidence": 0.0}


class Tank:
    """Consequence model: predicted level after the action under a leak hypothesis."""

    version = "tank-v1"

    def predict(self, action: rg.Action, h: rg.Hypothesis, ctx: dict) -> dict:
        return {"level": ctx["level"] + float(action.args.get("v", 0.0)) - h.params["leak"],
                "observation": [h.params["leak"] / 5]}


class LeakHyps:
    """Hypotheses about the hidden leak, scored against the noisy evidence."""

    version = "leak-v1"

    def hypotheses(self, ctx: dict) -> list[rg.Hypothesis]:
        return [rg.Hypothesis(f"leak{l}", family=f"leak{l}", params={"leak": l}, score=abs(ctx["evidence"] - l) / 5)
                for l in (0, 10, 25)]


class PrintAudit:
    """Audit sink that prints each decision (use ``riskgate.AuditLog(path)`` for JSONL in production)."""

    def write(self, decision: rg.Decision, **fields: Any) -> None:
        action = decision.action.key if decision.action is not None else "-"
        print(f"    [riskgate] {action:<14} -> {decision.verdict.value.upper():<7} {decision.reason}")


def _episode(rng: np.random.Generator) -> rg.Episode:
    actions = [rg.Action("add", {"v": v}) for v in (40.0, 20.0, 0.0, -10.0)]
    leak = rng.choice([0, 10, 25])
    level = rng.uniform(30, 70)
    ctx = {"level": level, "evidence": leak + rng.normal(0, 6)}
    rng.shuffle(actions)
    unsafe = {a.key: not (10 <= level + a.args["v"] - leak + rng.normal(0, 2) <= 100) for a in actions}
    return rg.Episode(ctx, actions, unsafe, true_family=f"leak{leak}")


def make_gate(verbose: bool = True) -> rg.Gate:
    """A gate calibrated (alpha = 0.1) on 200 simulated episodes, with one ASK hard rule."""
    gate = rg.Gate(Tank(), rg.Spec(rg.Envelope("level", 10, 100, scale=10)), LeakHyps(), alpha=0.1,
                   audit=PrintAudit() if verbose else None,
                   hard_rules=[rg.HardRule("big adds need approval", lambda a, c: 40 <= a.args.get("v", 0) <= 60,
                                           rg.Verdict.ASK, "adds of 40-60 L need operator approval")])
    gate.calibrate([_episode(np.random.default_rng(4)) for _ in range(200)])
    return gate


def snapshot() -> dict:
    """The gate context: in a real deployment, read the plant/DB/cluster state here."""
    return dict(STATE)
