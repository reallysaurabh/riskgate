"""Tests for riskgate.weighted: weighted conformal quantile and weighted CRC under covariate shift."""
import math

import numpy as np
import pytest

from riskgate.conformal import conformal_quantile, crc_threshold
from riskgate.weighted import WeightedCRC, weighted_conformal_quantile, weighted_crc_threshold

_erf = np.vectorize(math.erf, otypes=[float])


def Phi(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


# Covariate shift: contexts x ~ N(0, 1) during calibration and N(MU, 1) in deployment; the outcome law
# given x is unchanged: u | x ~ N(x, 1), where u is the score of the most dangerous unsafe candidate,
# so the episode loss is L(lambda) = 1{lambda <= u}. Likelihood ratio dQ/dP(x) = exp(MU x - MU^2 / 2).
MU = 1.0


def lr(x):
    return np.exp(MU * np.asarray(x) - MU**2 / 2)


def calibration_set(rng, n):
    x = rng.standard_normal(n)
    u = x + rng.standard_normal(n)
    grid = np.concatenate([np.unique(u), [np.inf]])
    L = (grid[None, :] <= u[:, None]).astype(float)
    return x, u, grid, L


def test_unit_weights_reduce_to_unweighted_methods():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(0, 60))
        alpha = float(rng.choice([0.05, 0.1, 0.2, 0.33, 0.5]))
        s = rng.normal(size=n).round(1)                        # ties on purpose
        for c in (1.0, 0.25, 8.0):                             # a common factor does not matter
            assert weighted_conformal_quantile(s, np.full(n, c), c, alpha) == conformal_quantile(s, alpha)
        if n == 0:
            continue
        u, v = rng.normal(size=n), rng.uniform(size=n)         # bounded, non-binary, monotone losses
        grid = np.concatenate([np.unique(u), [np.inf]])
        L = (grid[None, :] <= u[:, None]) * v[:, None]
        assert weighted_crc_threshold(L, grid, alpha, np.ones(n), 1.0) == crc_threshold(L, grid, alpha)


def test_weighted_crc_restores_validity_under_covariate_shift():
    rng = np.random.default_rng(1)
    alpha, n, reps, n_test = 0.1, 300, 200, 100
    risk_w, risk_u = [], []
    for _ in range(reps):
        x, u, grid, L = calibration_set(rng, n)
        xt = MU + rng.standard_normal(n_test)                              # deployment contexts
        lam_w = weighted_crc_threshold(L, grid, alpha, lr(x), lr(xt))       # one threshold per context
        lam_u = crc_threshold(L, grid, alpha)
        # risk given the context (Rao-Blackwellised): P(u_test >= lambda | x_test) = 1 - Phi(lambda - x_test)
        risk_w.append(np.mean(1 - Phi(lam_w - xt)))
        risk_u.append(np.mean(1 - Phi(lam_u - xt)))
    assert np.mean(risk_u) > alpha + 0.1           # unweighted CRC is badly invalid under the shift
    assert np.mean(risk_w) <= alpha + 0.01          # weighted CRC: E[L_test] <= alpha
    assert np.mean(risk_w) >= alpha - 0.04          # ...without being vacuous (observed: about 0.078)


def test_weighted_conformal_quantile_covers_under_covariate_shift():
    rng = np.random.default_rng(2)
    alpha, n, reps, n_test = 0.1, 300, 200, 100
    cov_w, cov_u = [], []
    for _ in range(reps):
        x, u, _, _ = calibration_set(rng, n)
        xt = MU + rng.standard_normal(n_test)
        q_w = weighted_conformal_quantile(u, lr(x), lr(xt), alpha)
        q_u = conformal_quantile(u, alpha)
        cov_w.append(np.mean(Phi(q_w - xt)))      # P(s_test <= q | x_test)
        cov_u.append(np.mean(Phi(q_u - xt)))
    assert np.mean(cov_u) < 1 - alpha - 0.1
    assert np.mean(cov_w) >= 1 - alpha - 0.01


def test_edge_cases_vectorisation_and_validation():
    rng = np.random.default_rng(3)
    x, u, grid, L = calibration_set(rng, 200)
    w = lr(x)
    # a context that is (extremely) rare under calibration -> certify nothing
    assert weighted_crc_threshold(L, grid, 0.1, w, 1e12) == math.inf
    assert weighted_crc_threshold(L, grid, 0.1, w, math.inf) == math.inf
    assert weighted_conformal_quantile(u, w, 1e12, 0.1) == math.inf
    assert weighted_conformal_quantile(u, w, math.inf, 0.1) == math.inf
    # thresholds only get more conservative as the test weight grows; array == scalar calls == WeightedCRC
    wt = np.linspace(0.0, 60.0, 121)
    th = weighted_crc_threshold(L, grid, 0.1, w, wt)
    q = weighted_conformal_quantile(u, w, wt, 0.1)
    assert th.shape == q.shape == wt.shape and th[-1] == math.inf
    assert np.all(th[1:] >= th[:-1]) and np.all(q[1:] >= q[:-1])
    crc = WeightedCRC(L, grid, 0.1, w)
    for i in range(0, wt.size, 10):
        assert th[i] == weighted_crc_threshold(L, grid, 0.1, w, wt[i]) == crc.threshold(wt[i])
        assert q[i] == weighted_conformal_quantile(u, w, wt[i], 0.1)
    # zero test weight: weighted empirical risk <= alpha, no finite-sample correction
    assert weighted_crc_threshold(L, grid, 0.1, w, 0.0) == grid[np.flatnonzero((w @ L) / w.sum() <= 0.1)[0]]
    # bound B: losses in [0, 2] at level 0.2 == losses in [0, 1] at level 0.1
    assert weighted_crc_threshold(2 * L, grid, 0.2, w, 1.0, bound=2.0) == weighted_crc_threshold(L, grid, 0.1, w, 1.0)
    # degenerate weights and sizes -> +inf (fail closed)
    assert weighted_crc_threshold(L, grid, 0.1, np.zeros(200), 1.0) == math.inf
    assert weighted_crc_threshold(L, grid, 0.1, np.zeros(200), 0.0) == math.inf
    assert weighted_conformal_quantile(u, np.zeros(200), 1.0, 0.1) == math.inf
    assert weighted_conformal_quantile(u, np.zeros(200), 0.0, 0.1) == math.inf
    assert weighted_conformal_quantile([], [], 1.0, 0.1) == math.inf
    assert weighted_crc_threshold(np.zeros((0, 3)), np.array([0.0, 1.0, np.inf]), 0.1, [], 1.0) == math.inf
    # invalid inputs
    for bad in (
        lambda: weighted_crc_threshold(L, grid, 0.1, -w, 1.0),
        lambda: weighted_crc_threshold(L, grid, 0.1, w[:-1], 1.0),
        lambda: weighted_crc_threshold(L, grid, 0.1, w, np.nan),
        lambda: weighted_crc_threshold(L, grid, 0.1, w, -1.0),
        lambda: weighted_crc_threshold(2 * L, grid, 0.1, w, 1.0),        # losses above the bound
        lambda: weighted_crc_threshold(L, grid[:-1], 0.1, w, 1.0),
        lambda: weighted_crc_threshold(L, grid[::-1], 0.1, w, 1.0),
        lambda: weighted_crc_threshold(L, grid, 0.1, w, 1.0, bound=0.0),
        lambda: weighted_conformal_quantile(u, w, 1.0, 0.0),
        lambda: weighted_conformal_quantile(u, w, 1.0, 1.0),
        lambda: weighted_conformal_quantile(np.append(u[:-1], np.nan), w, 1.0, 0.1),
        lambda: weighted_conformal_quantile(u, np.append(w[:-1], np.inf), 1.0, 0.1),
    ):
        with pytest.raises(ValueError):
            bad()
