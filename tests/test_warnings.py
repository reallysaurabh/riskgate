"""Policy-mode calibration without a doer_id warns: the guarantee is doer-specific, but nothing would void it on a swap.

Motivation (benchmarks/results): a policy-mode threshold calibrated on 2025 agents gave 14.6% unsafe on 2026 agents in
tau2-bench airline at alpha = 10%, and swapping AgentDojo pipelines without recalibrating exceeded alpha in 35% of pairs.
"""
import warnings

import numpy as np
import pytest

import riskgate as rg


class Tank:
    version = "tank-v1"

    def predict(self, action, h, ctx):
        return {"level": ctx["level"] + action.args["v"] - h.params["leak"]}


class Leaks:
    version = "leaks-v1"

    def hypotheses(self, ctx):
        return [rg.Hypothesis(f"leak={k}", family=f"leak{k}", params={"leak": k}, score=abs(ctx["flow"] - k) / 8)
                for k in (0, 15, 30)]


SPEC = rg.Spec(rg.Envelope("level", lo=10, hi=100, scale=10))
CANDIDATES = [rg.Action("add", {"v": v}) for v in (40, 20, 5, 0)]


def episodes(n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        leak, level = rng.choice([0, 15, 30]), rng.uniform(25, 90)
        ranked = list(rng.permutation(CANDIDATES))
        unsafe = {a.key: not 10 <= level + a.args["v"] - leak <= 100 for a in ranked}
        out.append(rg.Episode({"level": level, "flow": leak + rng.normal(0, 8)}, ranked, unsafe, true_family=f"leak{leak}"))
    return out


def test_policy_mode_without_doer_id_warns():
    with pytest.warns(rg.RiskGateWarning, match="doer_id"):
        rg.Gate(Tank(), SPEC, Leaks(), alpha=0.1).calibrate(episodes())


@pytest.mark.parametrize("kw", [{"doer_id": "scripted-ranker-v1"}, {"mode": "agnostic"}])
def test_no_warning_with_doer_id_or_in_agnostic_mode(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", rg.RiskGateWarning)
        cal = rg.Gate(Tank(), SPEC, Leaks(), alpha=0.1, **kw).calibrate(episodes())
    assert cal.n > 0


def test_warning_does_not_change_the_calibration():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rg.RiskGateWarning)
        a = rg.Gate(Tank(), SPEC, Leaks(), alpha=0.1).calibrate(episodes())
    b = rg.Gate(Tank(), SPEC, Leaks(), alpha=0.1, doer_id="scripted-ranker-v1").calibrate(episodes())
    assert a.lam == b.lam and a.tau == b.tau and a.fingerprint != b.fingerprint
