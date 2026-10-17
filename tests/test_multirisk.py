"""Tests for riskgate.multirisk (Learn then Test): p-values and Monte-Carlo validity of ltt_threshold."""
import math

import numpy as np
import pytest

from riskgate.multirisk import binomial_pvalue, hb_pvalue, hoeffding_pvalue, ltt_pvalues, ltt_threshold

_erf = np.vectorize(math.erf, otypes=[float])


def Phi(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def binom_cdf_bruteforce(k, n, p):
    return math.fsum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


# --------------------------------------------------------------------------- p-values
def test_binomial_pvalue_matches_bruteforce_sum():
    for n in range(1, 31):
        for alpha in (0.01, 0.1, 0.37, 0.5, 0.9):
            ks = np.arange(n + 1)
            want = [binom_cdf_bruteforce(int(k), n, alpha) for k in ks]
            np.testing.assert_allclose(binomial_pvalue(ks / n, n, alpha), want, rtol=1e-10, atol=1e-300)
            k = n // 3
            assert binomial_pvalue(k / n, n, alpha) == pytest.approx(want[k], rel=1e-10)   # scalar path


def test_pvalue_known_values():
    assert binomial_pvalue(0.2, 10, 0.5) == pytest.approx(56 / 1024, rel=1e-12)
    assert binomial_pvalue(0.0, 20, 0.1) == pytest.approx(0.9**20, rel=1e-12)
    assert binomial_pvalue(1.0, 20, 0.1) == 1.0
    assert binomial_pvalue(0.3, 0, 0.1) == 1.0 and hb_pvalue(0.0, 0, 0.1) == 1.0       # no data, no evidence
    n, r, a = 100, 0.02, 0.1
    kl = r * math.log(r / a) + (1 - r) * math.log((1 - r) / (1 - a))
    assert hoeffding_pvalue(r, n, a) == pytest.approx(math.exp(-n * kl), rel=1e-12)
    assert hoeffding_pvalue(r, n, a) <= math.exp(-2 * n * (a - r) ** 2)                 # KL form is tighter
    assert hoeffding_pvalue(0.0, n, a) == pytest.approx((1 - a) ** n, rel=1e-12)
    assert hoeffding_pvalue(0.1, n, a) == 1.0 and hoeffding_pvalue(0.3, n, a) == 1.0
    assert hb_pvalue(0.0, n, a) == pytest.approx((1 - a) ** n, rel=1e-12)              # Hoeffding wins at 0
    for r in (0.03, 0.05, 0.08, 0.12):
        bentkus = math.e * binom_cdf_bruteforce(math.ceil(n * r), n, a)
        assert hb_pvalue(r, n, a) == pytest.approx(min(hoeffding_pvalue(r, n, a), bentkus, 1.0), rel=1e-10)


def test_pvalues_stable_and_monotone_for_large_n():
    n, a = 100_000, 0.05
    r = np.linspace(0.0, 0.08, 801)
    for fn in (binomial_pvalue, hoeffding_pvalue, hb_pvalue):
        p = fn(r, n, a)
        assert p.shape == r.shape and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)), fn.__name__
        assert np.all(np.diff(p) >= 0), fn.__name__              # more observed risk -> weaker evidence
    assert binomial_pvalue(0.05, n, a) == pytest.approx(0.5, abs=0.01)
    assert 0.0 < binomial_pvalue(0.047, n, a) < 1e-4
    assert binomial_pvalue(np.array([[0.0, 0.05]]), n, a).shape == (1, 2)


def test_pvalues_are_superuniform_under_the_null():
    """P(p <= u) <= u at the least favourable null (risk exactly alpha)."""
    rng = np.random.default_rng(0)
    n, a = 40, 0.2
    us = np.array([0.01, 0.05, 0.1, 0.25, 0.5])

    def check(fn, r):
        freq = (fn(r, n, a)[None, :] <= us[:, None]).mean(axis=1)
        assert np.all(freq <= us + 3 * np.sqrt(us * (1 - us) / r.size)), (fn.__name__, freq)

    r_binary = rng.binomial(n, a, size=200_000) / n                       # Bernoulli(alpha) losses
    for fn in (binomial_pvalue, hoeffding_pvalue, hb_pvalue):
        check(fn, r_binary)
    r_bounded = rng.beta(0.5, 0.5 * (1 - a) / a, size=(20_000, n)).mean(axis=1)   # [0, 1] losses, mean alpha
    for fn in (hoeffding_pvalue, hb_pvalue):
        check(fn, r_bounded)


def test_pvalue_input_validation():
    for bad in (lambda: binomial_pvalue(0.1, 10, 0.0), lambda: hoeffding_pvalue(0.1, 10, 1.0),
                lambda: hb_pvalue(0.1, -1, 0.1), lambda: hb_pvalue(float("nan"), 10, 0.1)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError):
        binomial_pvalue(0.1, 10.5, 0.1)


# --------------------------------------------------------------------------- fixed-sequence logic
def test_fixed_sequence_stops_at_first_non_rejection_and_uses_max_p():
    grid = np.array([1.0, 2.0, 3.0, 4.0, np.inf])

    def ident(r, n, a):            # p-value := the empirical risk, to script the p-value sequence
        return r

    def thr(*rows, delta=0.05):
        return ltt_threshold([np.array([row]) for row in rows], grid, [0.5] * len(rows), delta, pvalue=ident)

    assert thr([0.0, 0.0, 0.0, 0.0, 0.0]) == 1.0          # every hypothesis rejected
    assert thr([0.0, 0.9, 0.0, 0.0, 0.0]) == 3.0          # stops at lambda=2 although lambda=1 would pass
    assert thr([0.0, 0.0, 0.0, 0.0, 0.2]) == math.inf     # the first, most conservative test fails
    assert thr([0.0, 0.0, 0.0, 0.06, 0.0]) == math.inf    # only lambda=+inf rejected
    assert thr([0.0, 0.0, 0.0, 0.06, 0.0], delta=0.1) == 1.0
    # several risks: a lambda is certified only if EVERY risk is (max of the p-values)
    assert thr([0.0, 0.0, 0.01, 0.0, 0.0], [0.04, 0.3, 0.0, 0.0, 0.0]) == 3.0
    assert thr([0.0, 0.0, 0.01, 0.0, 0.0], [0.04, 0.0, 0.0, 0.0, 0.0]) == 1.0


# --------------------------------------------------------------------------- Monte-Carlo validity
ALPHAS, DELTA = (0.10, 0.15), 0.10


def two_risk_episodes(rng, n):
    """Per episode: u1 = max score of an unsafe candidate (none w.p. 0.3), u2 = score above which a
    certified action overruns its cost budget. Losses L_j(lambda) = 1{lambda <= u_j}: monotone, 0 at +inf."""
    z = rng.standard_normal(n)
    u1 = np.where(rng.random(n) < 0.7, z + 0.5 * rng.standard_normal(n), -np.inf)
    u2 = z + rng.standard_normal(n) - 0.3
    grid = np.concatenate([np.unique(np.concatenate([u1[np.isfinite(u1)], u2])), [np.inf]])
    L1 = (grid[None, :] <= u1[:, None]).astype(float)
    L2 = (grid[None, :] <= u2[:, None]).astype(float)
    return grid, L1, L2


def true_risks(lam):
    if not math.isfinite(lam):
        return 0.0, 0.0
    return float(0.7 * (1 - Phi(lam / math.sqrt(1.25)))), float(1 - Phi((lam + 0.3) / math.sqrt(2.0)))


def violates(lam, alphas):
    r1, r2 = true_risks(lam)
    return r1 > alphas[0] or r2 > alphas[1]


# (0.10, 0.15): both budgets bind at almost the same lambda (~1.19), where the max-p test is conservative.
# (0.10, 0.30): only the safety budget binds; the exact binomial LTT then violates close to delta.
@pytest.mark.parametrize("alphas", [(0.10, 0.15), (0.10, 0.30)], ids=["both-binding", "one-binding"])
def test_ltt_controls_two_risks_with_high_probability(alphas):
    rng = np.random.default_rng(7)
    n, reps = 300, 300
    viol = {"binomial": 0, "hb": 0, "hoeffding": 0}
    risk1 = {k: [] for k in viol}
    naive_viol = 0
    for _ in range(reps):
        grid, L1, L2 = two_risk_episodes(rng, n)
        for pv in viol:
            lam = ltt_threshold([L1, L2], grid, alphas, DELTA, pvalue=pv)
            assert math.isfinite(lam)                                     # not vacuous at n = 300
            viol[pv] += violates(lam, alphas)
            risk1[pv].append(true_risks(lam)[0])
        ok = (L1.mean(axis=0) <= alphas[0]) & (L2.mean(axis=0) <= alphas[1])   # plug-in rule, no guarantee
        naive_viol += violates(grid[np.flatnonzero(ok)[0]], alphas)
    for pv, v in viol.items():
        assert v / reps <= DELTA + 0.04, (pv, v / reps)
    assert naive_viol / reps > 0.2                  # the experiment does detect invalid thresholds
    assert np.mean(risk1["binomial"]) > 0.06        # the exact test spends a good part of the budget
    assert np.mean(risk1["hoeffding"]) <= np.mean(risk1["hb"]) <= np.mean(risk1["binomial"])


def test_ltt_input_validation_bounds_and_small_samples():
    rng = np.random.default_rng(3)
    grid, L1, L2 = two_risk_episodes(rng, 100)
    for bad in (
        lambda: ltt_threshold([L1, L2[:, :-1]], grid, ALPHAS, DELTA),          # shapes differ
        lambda: ltt_threshold([L1], grid, ALPHAS, DELTA),                      # one alpha per risk
        lambda: ltt_threshold([L1], grid[::-1], (0.1,), DELTA),                # grid must ascend
        lambda: ltt_threshold([L1], grid[:-1], (0.1,), DELTA),                 # grid / columns mismatch
        lambda: ltt_threshold([L1], grid, (0.1,), 1.5),                        # delta in (0, 1)
        lambda: ltt_threshold([L1], grid, (1.0,), DELTA),                      # alpha < bound
        lambda: ltt_threshold([2 * L1], grid, (0.1,), DELTA),                  # losses above the bound
        lambda: ltt_threshold([0.5 * L1], grid, (0.1,), DELTA, pvalue="binomial"),   # exact test needs binary
        lambda: ltt_threshold([L1], grid, (0.1,), DELTA, pvalue="wsr"),
        lambda: ltt_threshold([], grid, (), DELTA),
    ):
        with pytest.raises(ValueError):
            bad()
    # losses in [0, B]: identical to rescaled losses (B a power of two keeps the arithmetic exact)
    soft = L2 * rng.uniform(0, 1, size=(L2.shape[0], 1))
    assert ltt_threshold([L1, 4 * soft], grid, (0.1, 0.4), DELTA, bounds=(1.0, 4.0)) == \
        ltt_threshold([L1, soft], grid, (0.1, 0.1), DELTA)
    # +inf (all losses 0) is rejected only once (1 - alpha)^n <= delta
    g3 = np.array([0.0, 1.0, np.inf])
    assert ltt_threshold([np.zeros((0, 3))], g3, (0.1,), 0.1) == math.inf     # no episodes
    assert ltt_threshold([np.zeros((5, 3))], g3, (0.1,), 0.1) == math.inf     # 0.9^5 = 0.59 > 0.1
    assert ltt_threshold([np.zeros((30, 3))], g3, (0.1,), 0.1) == 0.0         # 0.9^30 = 0.04 <= 0.1
    p = ltt_pvalues([L1, L2], ALPHAS)
    assert p.shape == (2, grid.size)
    assert p[0, -1] == pytest.approx(0.9**100) and p[1, -1] == pytest.approx(0.85**100)
