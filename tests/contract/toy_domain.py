"""Toy domain shared by the adapter contract tests (and the stdio MCP proxy test subprocess).

The tank from ``tests/test_core.py``: action ``add(v)`` adds ``v`` litres, a hidden leak drains 0/10/25
litres, and the spec keeps the level within [10, 100]. With the calibrated gate below and the context
``STATE = {"level": 50, "evidence": 0}``:

* ``add(v=10)``  -> CERTIFY  (robust margin 4.0 above the calibrated threshold)
* ``add(v=50)``  -> ASK      (hard rule: adds of 40-60 L need operator approval)
* ``add(v=90)``  -> REJECT   (unsafe under every plausible leak hypothesis)

``StubGate`` returns a fixed verdict (used for PROBE and for counting calls).
"""
from __future__ import annotations

from typing import Any

import numpy as np

import riskgate as rg

STATE = {"level": 50.0, "evidence": 0.0}
CERTIFIED, ASKED, REJECTED = 10.0, 50.0, 90.0


class Tank:
    version = "tank-v1"

    def predict(self, action: rg.Action, h: rg.Hypothesis, ctx: dict) -> dict:
        return {"level": ctx["level"] + action.args.get("v", 0.0) - h.params["leak"],
                "observation": [h.params["leak"] / 5]}


class LeakHyps:
    version = "leak-v1"

    def hypotheses(self, ctx: dict) -> list[rg.Hypothesis]:
        return [rg.Hypothesis(f"leak{l}", family=f"leak{l}", params={"leak": l}, score=abs(ctx["evidence"] - l) / 5)
                for l in (0, 10, 25)]


ACTIONS = [rg.Action("add", {"v": v}) for v in (40.0, 20.0, 0.0, -10.0)]


def _episode(rng: np.random.Generator) -> rg.Episode:
    leak = rng.choice([0, 10, 25])
    level = rng.uniform(30, 70)
    ctx = {"level": level, "evidence": leak + rng.normal(0, 6)}
    ranked = list(ACTIONS)
    rng.shuffle(ranked)
    unsafe = {a.key: not (10 <= level + a.args["v"] - leak + rng.normal(0, 2) <= 100) for a in ranked}
    return rg.Episode(ctx, ranked, unsafe, true_family=f"leak{leak}")


def make_toy_gate(audit: Any = None) -> rg.Gate:
    gate = rg.Gate(Tank(), rg.Spec(rg.Envelope("level", 10, 100, scale=10)), LeakHyps(), alpha=0.1, audit=audit,
                   hard_rules=[rg.HardRule("big adds need approval", lambda a, c: 40 <= a.args.get("v", 0) <= 60,
                                           rg.Verdict.ASK, "adds of 40-60 L need operator approval")])
    gate.calibrate([_episode(np.random.default_rng(4)) for _ in range(200)])
    return gate


class StubGate:
    """Returns a fixed verdict for every action and records what it was asked."""

    def __init__(self, verdict: rg.Verdict, probe: rg.Action | None = None) -> None:
        self.verdict, self.probe, self.calls = verdict, probe, []

    def check(self, action: rg.Action, context: Any, *, stage: int = 1) -> rg.Decision:
        self.calls.append((action, context))
        return rg.Decision(self.verdict, action, reason=f"stub says {self.verdict.value}", probe=self.probe,
                           status=rg.GuaranteeStatus.VALID)


class PrintAudit:
    """Audit sink that prints to stdout (a stdio MCP server must keep such output off the wire)."""

    def write(self, decision: rg.Decision, **fields: Any) -> None:
        print(f"[riskgate] {decision.action.key if decision.action else '-'} -> {decision.verdict.value}")


def proxy_factory() -> tuple[rg.Gate, Any]:
    """``(gate, context_fn)`` for ``python -m riskgate.adapters.mcp_proxy --gate toy_domain.py:proxy_factory``."""
    return make_toy_gate(audit=PrintAudit()), lambda: dict(STATE)
