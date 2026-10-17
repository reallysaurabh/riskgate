import asyncio
import math

import numpy as np
import pytest

import riskgate as rg
from riskgate.adapters.frameworks import adk_before_tool_callback, langgraph_gate_node, route_after_gate
from riskgate.conformal import lambda_grid, policy_losses


# --------------------------------------------------------------------------- toy domain
# A tank at level L; action add(v) adds v litres; a hidden leak drains `leak` litres.
# Safe envelope: 10 <= level <= 100. Hypotheses: leak in {0, 10, 25}; noisy evidence about the leak.
class Tank:
    version = "tank-v1"

    def predict(self, action, h, ctx):
        return {"level": ctx["level"] + action.args.get("v", 0.0) - h.params["leak"], "observation": [h.params["leak"] / 5]}


class LeakHyps:
    version = "leak-v1"

    def hypotheses(self, ctx):
        return [rg.Hypothesis(f"leak{l}", family=f"leak{l}", params={"leak": l}, score=abs(ctx["evidence"] - l) / 5)
                for l in (0, 10, 25)]


ACTIONS = [rg.Action("add", {"v": v}) for v in (40, 20, 0, -10)]


def episode(rng):
    leak = rng.choice([0, 10, 25])
    level = rng.uniform(30, 70)
    ctx = {"level": level, "evidence": leak + rng.normal(0, 6)}
    ranked = list(ACTIONS)
    rng.shuffle(ranked)
    unsafe = {a.key: not (10 <= level + a.args["v"] - leak + rng.normal(0, 2) <= 100) for a in ranked}
    return rg.Episode(ctx, ranked, unsafe, true_family=f"leak{leak}")


def episodes(seed, n):
    """n independent episodes from one random stream."""
    rng = np.random.default_rng(seed)
    return [episode(rng) for _ in range(n)]


def make_gate(**kw):
    return rg.Gate(Tank(), rg.Spec(rg.Envelope("level", 10, 100, scale=10)), LeakHyps(), **kw)


# --------------------------------------------------------------------------- tests
def test_spec_margins_and_binding():
    spec = rg.Spec(rg.Upper("p95_ms", 500, scale=500), rg.Lower("level", 10, scale=10, name="min level"))
    m, b = spec.evaluate({"p95_ms": [200, 600], "level": 50})
    assert m == pytest.approx(-0.2) and b == "p95_ms<=500"
    assert spec.evaluate({"p95_ms": 100, "level": float("nan")})[0] == -math.inf   # NaN fails closed


def test_policy_losses_envelope_monotone_and_dominates():
    scores = [np.array([1.0, 3.0, 2.0])]
    unsafe = [np.array([True, False, True])]
    grid = lambda_grid(scores)
    raw = policy_losses(scores, unsafe, grid, envelope=False)
    env = policy_losses(scores, unsafe, grid, envelope=True)
    assert np.all(np.diff(env[0]) <= 0) and np.all(env >= raw)
    assert env[0, -1] == 0.0                                  # lambda = +inf: nothing certified


def test_crc_controls_risk_in_both_modes():
    rng = np.random.default_rng(0)
    for mode in ("policy", "agnostic"):
        risks = []
        for _rep in range(40):
            g = make_gate(mode=mode, alpha=0.1)
            g.calibrate([episode(rng) for _ in range(150)])
            test = [episode(rng) for _ in range(150)]
            bad = 0
            for ep in test:
                d = g.check_ranked(ep.candidates, ep.context)
                bad += d.verdict is rg.Verdict.CERTIFY and ep.unsafe[d.action.key]
            risks.append(bad / len(test))
        assert np.mean(risks) <= 0.1 + 0.02, (mode, np.mean(risks))


def test_verdicts_hard_rules_and_fail_closed():
    g = make_gate(alpha=0.1, hard_rules=[
        rg.HardRule("no draining", lambda a, c: a.args.get("v", 0) < 0, rg.Verdict.REJECT, "never drain"),
        rg.HardRule("big adds need approval", lambda a, c: a.args.get("v", 0) >= 40, rg.Verdict.ASK, "> 40 L"),
    ])
    assert g.check(ACTIONS[0], {"level": 50, "evidence": 0}).verdict is rg.Verdict.ASK     # uncalibrated -> void -> ask
    g.calibrate(episodes(1, 200))
    assert g.status() is rg.GuaranteeStatus.VALID
    d = g.check(rg.Action("add", {"v": -10}), {"level": 50, "evidence": 0})
    assert d.verdict is rg.Verdict.REJECT and "hard rules" in d.reason
    assert g.check(rg.Action("add", {"v": 40}), {"level": 20, "evidence": 0}).verdict is rg.Verdict.ASK
    # confidently unsafe -> reject; error inside the model -> ask (fail closed)
    assert g.check(rg.Action("add", {"v": 20}), {"level": 99, "evidence": 0}).verdict is rg.Verdict.REJECT
    assert g.check(rg.Action("add", {"v": 20}), {"evidence": 0}).verdict is rg.Verdict.ASK


def test_fingerprint_change_voids_guarantee_and_roundtrip(tmp_path):
    g = make_gate(alpha=0.1)
    cal = g.calibrate(episodes(2, 200))
    cal.save(tmp_path / "cal.json")
    g2 = make_gate(alpha=0.1)
    g2.load_calibration(tmp_path / "cal.json")
    assert g2.status() is rg.GuaranteeStatus.VALID and g2.calibration.lam == cal.lam
    g3 = rg.Gate(Tank(), rg.Spec(rg.Envelope("level", 5, 100, scale=10)), LeakHyps())   # spec changed
    g3.load_calibration(tmp_path / "cal.json")
    assert g3.status() is rg.GuaranteeStatus.VOID
    assert g3.check(ACTIONS[2], {"level": 50, "evidence": 0}).verdict is rg.Verdict.ASK


def ambiguous_episode(rng):
    """Low level, noisy evidence: plausible leaks disagree about whether holding is safe."""
    leak = rng.choice([0, 10, 25])
    level = rng.uniform(22, 38)
    ctx = {"level": level, "evidence": leak + rng.normal(0, 7)}
    ranked = [rg.Action("add", {"v": 0}), rg.Action("add", {"v": 5})]
    unsafe = {a.key: not (10 <= level + a.args["v"] - leak <= 100) for a in ranked}
    ep = rg.Episode(ctx, ranked, unsafe, true_family=f"leak{leak}")
    post = dict(ctx, evidence=leak + rng.normal(0, 1))          # the flowmeter reveals the leak
    ep.after_probe = {"read_flowmeter": rg.Episode(post, ranked, unsafe, f"leak{leak}")}
    return ep


def test_probe_two_stage_calibration_and_flow():
    g = make_gate(alpha=0.1, probes=[rg.Action("read_flowmeter", read_only=True)], noop=rg.Action("add", {"v": 0}),
                  min_separation=0.5)
    rng = np.random.default_rng(3)
    cal = g.calibrate([ambiguous_episode(rng) for _ in range(600)])
    assert cal.n_probe > 0 and cal.probing
    probed = certified_after = bad = 0
    for _ in range(300):
        ep = ambiguous_episode(rng)
        d = g.check_ranked(ep.candidates, ep.context)
        if d.verdict is rg.Verdict.PROBE:
            probed += 1
            assert d.probe.read_only
            post = ep.after_probe[d.probe.key]
            d = g.check_ranked(post.candidates, post.context, stage=2)
            certified_after += d.verdict is rg.Verdict.CERTIFY
        if d.verdict is rg.Verdict.CERTIFY:
            bad += ep.unsafe[d.action.key]
    assert probed > 0 and certified_after > 0
    assert bad / 300 <= 0.1 + 0.05


def test_guard_decorator_sync_async_and_adapters():
    g = make_gate(alpha=0.1)
    g.calibrate(episodes(4, 200))
    state = {"level": 50, "evidence": 0}

    @rg.guard(g, context=lambda: state, name="add", on_ask=lambda d: False)
    def add(v: float) -> str:
        return f"added {v}"

    assert add(20) == "added 20"
    blocked = add(80)
    assert isinstance(blocked, dict) and blocked["status"] in ("rejected", "needs_approval")

    @rg.guard(g, context=lambda: state, name="add")
    async def aadd(v: float) -> str:
        return "ok"

    assert asyncio.run(aadd(10)) == "ok"

    @rg.guard(make_gate(alpha=0.1), context=lambda: state, name="add")    # uncalibrated -> void -> ASK
    async def aadd_void(v: float) -> str:
        return "ok"

    with pytest.raises(rg.ApprovalRequired):
        asyncio.run(aadd_void(10))

    cb = adk_before_tool_callback(g, context=lambda tc: state)
    tool = type("T", (), {"name": "add"})()
    assert cb(tool, {"v": 10}) is None and isinstance(cb(tool, {"v": 90}), dict)

    node = langgraph_gate_node(g, context=lambda s: state)
    upd = node({"messages": [{"tool_calls": [{"id": "1", "name": "add", "args": {"v": 10}}]}]})
    assert route_after_gate(upd) == "tools"
    upd = node({"messages": [{"tool_calls": [{"id": "2", "name": "add", "args": {"v": 90}}]}]})
    assert route_after_gate(upd) == "agent" and upd["messages"][0]["tool_call_id"] == "2"


def test_monitor_flags_drift_and_only_tightens():
    g = make_gate(alpha=0.1, monitor=rg.Monitor(alarm_level=20))
    g.calibrate(episodes(5, 300))
    lam0 = g._threshold(1)
    for _ in range(60):   # deployment where everything looks far more dangerous than in calibration
        g.check(ACTIONS[0], {"level": 75, "evidence": 25})
    assert g.status() is rg.GuaranteeStatus.DEGRADED
    for _ in range(20):
        g.monitor.observe_outcome(certified=True, unsafe=True)
    assert g._threshold(1) >= lam0
