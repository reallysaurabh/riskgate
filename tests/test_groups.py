"""Grouped calibration: several dependent decision points per conversation / task / incident.

Real agent logs contain many tool calls per run, and runs of the same task are correlated. Conformal risk control is
then applied to exchangeable GROUPS with their mean loss (Gate.calibrate(groups=...)).
"""
import math

import numpy as np
import pytest

import riskgate as rg


class Det:
    """Consequence model: the outcome margin is stored in the context (score = what the gate would compute)."""

    version = "det-v1"

    def predict(self, action, h, ctx):
        return {"m": ctx["score"][action.name]}


SPEC = rg.Spec(rg.Custom("m", lambda o: o["m"]))


def make_gate(**kw):
    return rg.Gate(Det(), SPEC, **kw)


def grouped_episodes(rng, n_groups, size=4, rho=0.8):
    """Each group shares a latent difficulty z: its decisions are unsafe together and their scores move together."""
    eps, groups = [], []
    for g in range(n_groups):
        z = rng.normal()
        for _ in range(int(rng.integers(1, size + 1))):
            e = rho * z + math.sqrt(1 - rho * rho) * rng.normal()
            s = -e + rng.normal(0, 0.7)                      # higher score = safer, imperfectly
            unsafe = bool(e > 0.9)
            eps.append(rg.Episode({"score": {"a": s}}, [rg.Action("a")], {"a": unsafe}))
            groups.append(f"g{g}")
    return eps, groups


def test_groups_none_is_unchanged():
    eps, _ = grouped_episodes(np.random.default_rng(0), 120)
    a = make_gate(alpha=0.1).calibrate(eps, seed=3)
    b = make_gate(alpha=0.1).calibrate(eps, seed=3, groups=None)
    assert a.lam == b.lam and a.n == b.n and a.tau == b.tau


def test_singleton_groups_equal_ungrouped():
    eps, _ = grouped_episodes(np.random.default_rng(1), 150)
    a = make_gate(alpha=0.1).calibrate(eps, seed=5)
    b = make_gate(alpha=0.1).calibrate(eps, seed=5, groups=list(range(len(eps))))
    assert a.lam == b.lam and a.n == b.n
    assert b.diagnostics["grouped"] and b.diagnostics["n_groups"] == b.n == b.diagnostics["n_episodes"]


def test_group_counts_and_mean_loss_threshold():
    rng = np.random.default_rng(2)
    eps, groups = grouped_episodes(rng, 200)
    cal = make_gate(alpha=0.1).calibrate(eps, groups=groups, tau_fraction=0.0)
    assert cal.n == 200 and cal.diagnostics["n_episodes"] == len(eps)
    # the threshold must satisfy the CRC inequality on GROUP-mean losses, and be the smallest grid point that does
    s = np.array([ep.context["score"]["a"] for ep in eps])
    u = np.array([ep.unsafe["a"] for ep in eps], float)
    gid = np.array([int(g[1:]) for g in groups])

    def group_risk(lam):
        per = np.zeros(200)
        np.add.at(per, gid, u * (s >= lam))
        return float(np.mean(per / np.bincount(gid, minlength=200)))

    assert 200 / 201 * group_risk(cal.lam) + 1 / 201 <= 0.1 + 1e-12
    below = s[s < cal.lam]
    if below.size:
        assert 200 / 201 * group_risk(below.max()) + 1 / 201 > 0.1


def test_groups_validation():
    eps, groups = grouped_episodes(np.random.default_rng(3), 50)
    with pytest.raises(ValueError, match="one entry per episode"):
        make_gate(alpha=0.1).calibrate(eps, groups=groups[:-1])
    with pytest.raises(ValueError, match="cannot be combined"):
        make_gate(alpha=0.1).calibrate(eps, groups=groups, weights=[1.0] * len(eps))
    with pytest.raises(ValueError, match="risk-control groups"):
        make_gate(alpha=0.01).calibrate(eps, groups=groups)          # 50 groups < 1/alpha
    g = make_gate(alpha=0.2, probes=[rg.Action("look", read_only=True)])
    cal = g.calibrate(eps, groups=groups)
    assert "groups" in cal.diagnostics["probing_disabled"] and not cal.probing


def test_tau_split_is_by_group():
    eps, groups = grouped_episodes(np.random.default_rng(4), 100)
    for ep in eps:                                                   # labelled families force a tau split
        ep.true_family = "nominal"
    cal = make_gate(alpha=0.1).calibrate(eps, groups=groups, tau_fraction=0.3)
    assert cal.diagnostics["n_tau_groups"] == 30 and cal.n == 70


@pytest.mark.parametrize("guarantee", ["expectation", "high_probability"])
def test_grouped_validity_over_redraws(guarantee):
    """Mean (over calibration draws) of the realised group-mean risk on fresh groups stays <= alpha."""
    alpha, risks = 0.1, []
    for r in range(150):
        rng = np.random.default_rng(1000 + r)
        cal_eps, cal_g = grouped_episodes(rng, 60)
        test_eps, test_g = grouped_episodes(rng, 400)
        g = make_gate(alpha=alpha)
        g.calibrate(cal_eps, groups=cal_g, tau_fraction=0.0, guarantee=guarantee, delta_ltt=0.1)
        lam = g.calibration.lam
        per: dict = {}
        for ep, gg in zip(test_eps, test_g):
            per.setdefault(gg, []).append(float(ep.unsafe["a"] and ep.context["score"]["a"] >= lam))
        risks.append(np.mean([np.mean(v) for v in per.values()]))
    risks = np.array(risks)
    assert risks.mean() <= alpha + 0.01                              # expectation guarantee (+ MC noise)
    if guarantee == "high_probability":
        assert np.mean(risks > alpha) <= 0.1 + 0.06                  # P(risk > alpha) <= delta_ltt (+ MC noise)
