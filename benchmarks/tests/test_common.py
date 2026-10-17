"""The fast harness must reproduce riskgate.Gate.calibrate(groups=...) exactly (same lambda-hat)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

import riskgate as rg  # noqa: E402


class Det:
    version = "det-v1"

    def predict(self, action, h, ctx):
        return {"m": ctx["s"][action.name]}


def make(n_groups, k, rng):
    eps, groups, decs, scores = [], [], [], []
    for g in range(n_groups):
        z = rng.normal()
        for t in range(int(rng.integers(1, 5))):
            names = [f"a{j}" for j in range(k)]
            s = {a: float(np.round(rng.normal() - 0.8 * z, 2)) for a in names}
            u = {a: bool(rng.random() < 1 / (1 + np.exp(2 * s[a] - 1))) for a in names}
            eps.append(rg.Episode({"s": s}, [rg.Action(a) for a in names], u))
            groups.append(f"g{g}")
            decs.append(C.Decision(group=f"g{g}", traj=f"g{g}-{t}", doer="d", unsafe=[u[a] for a in names]))
            scores.append([s[a] for a in names])
    return eps, groups, decs, scores


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("guarantee,kind", [("expectation", "crc"), ("high_probability", "ltt")])
def test_harness_matches_library(k, guarantee, kind):
    rng = np.random.default_rng(7 + k)
    for alpha in (0.05, 0.1, 0.2):
        eps, groups, decs, scores = make(120, k, rng)
        g = rg.Gate(Det(), rg.Spec(rg.Custom("m", lambda o: o["m"])), alpha=alpha)
        cal = g.calibrate(eps, groups=groups, tau_fraction=0.0, guarantee=guarantee, delta_ltt=C.DELTA_LTT)
        lam = C.lam_for("x", kind, list(range(len(decs))), scores, decs, alpha)
        assert lam == cal.lam, (alpha, lam, cal.lam)
        # and the library's own scores are the ones the harness uses
        assert np.allclose([s for ep in eps for s in g.score_episode(ep)], [s for row in scores for s in row])


def test_breakpoint_envelope_matches_policy_losses():
    rng = np.random.default_rng(3)
    _, _, decs, scores = make(60, 4, rng)
    grid = np.concatenate([np.unique(np.concatenate([np.array(s) for s in scores])), [np.inf]])
    L = rg.conformal.policy_losses([np.array(s) for s in scores], [np.array(d.unsafe, float) for d in decs], grid)
    b = C.breakpoints(scores, decs)
    assert np.array_equal(L, (b[:, None] >= grid[None, :]).astype(float))
