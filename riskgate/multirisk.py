"""Learn then Test (LTT): high-probability control of several risks with one threshold.

Conformal risk control (`riskgate.conformal.crc_threshold`) bounds ONE risk IN EXPECTATION. Learn
then Test (Angelopoulos, Bates, Candès, Jordan & Lei, 2021) certifies instead, with probability at
least 1 - delta over the calibration episodes, that SEVERAL risks are below their budgets at the
selected threshold at the same time (e.g. safety violations, cost overruns, privacy leaks; SPEC.md
§12):

    P( R_j(lambda_hat) <= alpha_j for every j ) >= 1 - delta,      R_j(lambda) = E[L_j(lambda)].

Recipe: for every grid point lambda, test the null H_lambda "some risk exceeds its budget" with a
valid p-value, and select lambda among the rejected nulls with a procedure that controls the
family-wise error rate (FWER) at delta.

Why fixed-sequence over lambda with a max-p over risks is valid
---------------------------------------------------------------
1. Several risks at one lambda: intersection-union test (Berger, 1982). H_lambda is the union of
   H_{lambda,j}: R_j(lambda) > alpha_j. If every p_j(lambda) is super-uniform under H_{lambda,j},
   then p(lambda) = max_j p_j(lambda) is super-uniform under H_lambda: when H_lambda holds, some j*
   has R_j*(lambda) > alpha_j*, and P(max_j p_j <= u) <= P(p_j* <= u) <= u. Rejecting a union of
   nulls requires rejecting each of them, so no Bonferroni correction over the m risks is needed.
2. Many lambdas: fixed-sequence testing (Maurer, Hothorn & Lehmacher, 1995; one of the procedures
   recommended in LTT). The order lambda_(G) = +inf > lambda_(G-1) > ... > lambda_(1) (most
   conservative first) is fixed before the losses are seen, each H_lambda is tested at the full
   level delta, and testing stops at the first non-rejection. A true null can only be rejected if
   the FIRST true null in this order, lambda*, is rejected, so FWER <= P(p(lambda*) <= delta) <= delta.
   lambda_hat (the last rejection) is a rejected null, hence all its risks are within budget with
   probability >= 1 - delta. This needs no monotonicity; non-increasing loss rows (as produced by
   agnostic_losses / policy_losses) only make the procedure powerful, because the empirical risks
   then decrease along the sequence and the testing does not stop early by accident.
3. Data-driven grid (distinct calibration scores ∪ {+inf}, `riskgate.conformal.lambda_grid`). If
   each L_i(lambda) depends on episode i only and changes value only at that episode's own scores,
   every empirical risk is constant on each interval (g_{k-1}, g_k], so the procedure coincides with
   fixed-sequence testing over the continuum lambda ∈ R ∪ {+inf}. Its rejected set is the upper
   interval [lambda_hat, +inf], and {lambda_hat <= l} ⊂ {p(l) <= delta} for every fixed l. If the
   supremum lambda* of the true nulls is a null, the error event lies in {p(lambda*) <= delta};
   otherwise it is the increasing union over true nulls l_m ↑ lambda* of {lambda_hat <= l_m}, whose
   probability is the limit of numbers <= delta. Either way FWER <= delta.

The p-values are concentration bounds and assume i.i.d. episodes (stronger than the exchangeability
CRC needs). For a loss in [0, 1] with empirical mean R_hat over n episodes, testing R > alpha:

    binomial   P(Binom(n, alpha) <= n R_hat)                 exact, binary {0, 1} losses only
    hoeffding  exp(-n h1(min(R_hat, alpha), alpha))          Hoeffding (1963, Thm 1), any [0, 1] loss
    hb         min(hoeffding, e P(Binom(n, alpha) <= ceil(n R_hat)))   Hoeffding-Bentkus (Bentkus 2004;
                                                              Bates et al. 2021; LTT), any [0, 1] loss

with h1(a, b) = a log(a / b) + (1 - a) log((1 - a) / (1 - b)). Losses in [0, B] are rescaled by B.

References
----------
A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei. Learn then Test: calibrating
    predictive algorithms to achieve risk control. arXiv:2110.01052, 2021.
S. Bates, A. N. Angelopoulos, L. Lei, J. Malik, M. I. Jordan. Distribution-free, risk-controlling
    prediction sets. Journal of the ACM 68(6), 2021.
R. L. Berger. Multiparameter hypothesis testing and acceptance sampling. Technometrics 24(4), 1982.
G. Maurer, L. A. Hothorn, W. Lehmacher. Multiple comparisons in drug clinical trials and preclinical
    assays: a-priori ordered hypotheses. Biometrie in der chemisch-pharmazeutischen Industrie 6, 1995.
W. Hoeffding. Probability inequalities for sums of bounded random variables. JASA 58(301), 1963.
V. Bentkus. On Hoeffding's inequalities. Annals of Probability 32(2), 2004.
"""
from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt

__all__ = ["PVALUES", "binomial_pvalue", "hb_pvalue", "hoeffding_pvalue", "ltt_pvalues", "ltt_threshold"]

PValueFn = Callable[[Any, int, float], Any]   # (risk_hat, n, alpha) -> p-value(s)
"""p-value function: (risk_hat array, n, alpha) -> array of p-values for H: risk > alpha."""

_TOL = 1e-9  # absorbs floating error in n * risk_hat (e.g. 2.9999999999 losses -> 3)


# --------------------------------------------------------------------------------------------
# p-values
# --------------------------------------------------------------------------------------------
def _check_n_alpha(n: int, alpha: float) -> tuple[int, float]:
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return n, alpha


def _risks(risk_hat: npt.ArrayLike) -> tuple[np.ndarray, bool]:
    r = np.asarray(risk_hat, dtype=float)
    if np.isnan(r).any():
        raise ValueError("risk_hat contains NaN")
    return np.clip(np.atleast_1d(r), 0.0, 1.0), r.ndim == 0


def _ret(p: np.ndarray, scalar: bool) -> Union[float, np.ndarray]:
    return float(p.reshape(-1)[0]) if scalar else p


@functools.lru_cache(maxsize=128)
def _binom_logcdf(n: int, p: float) -> np.ndarray:
    """log P(Binom(n, p) <= k) for k = 0..n, from log-pmfs (math.lgamma) and a log-sum-exp scan."""
    k = np.arange(n + 1, dtype=float)
    log_fact = np.array([math.lgamma(i + 1.0) for i in range(n + 1)])  # log(i!)
    log_pmf = log_fact[n] - log_fact - log_fact[::-1] + k * math.log(p) + (n - k) * math.log1p(-p)
    out = np.minimum(np.logaddexp.accumulate(log_pmf), 0.0)
    out.flags.writeable = False
    return out


def _binom_cdf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """P(Binom(n, p) <= k) elementwise for integer-valued k (k < 0 -> 0, k >= n -> 1)."""
    table = _binom_logcdf(n, p)
    out = np.exp(table[np.clip(k, 0, n).astype(np.int64)])
    return np.where(k < 0, 0.0, np.where(k >= n, 1.0, out))


def binomial_pvalue(risk_hat: npt.ArrayLike, n: int, alpha: float) -> Union[float, np.ndarray]:
    """Exact binomial p-value P(Binom(n, alpha) <= n * risk_hat) for H: risk > alpha.

    Valid (super-uniform under the null) for i.i.d. BINARY losses, where n * risk_hat is the number
    of losses: under risk >= alpha the count is stochastically larger than Binom(n, alpha). It is NOT
    valid for general [0, 1]-valued losses (a constant loss equal to its mean defeats it); use
    `hb_pvalue` for those. Computed in log space with math.lgamma (stable for large n).
    `risk_hat` may be a scalar (returns float) or an array (returns an array of the same shape).
    """
    n, alpha = _check_n_alpha(n, alpha)
    r, scalar = _risks(risk_hat)
    if n == 0:
        return _ret(np.ones_like(r), scalar)
    return _ret(_binom_cdf(np.floor(n * r + _TOL), n, alpha), scalar)


def hoeffding_pvalue(risk_hat: npt.ArrayLike, n: int, alpha: float) -> Union[float, np.ndarray]:
    """Hoeffding p-value exp(-n h1(min(risk_hat, alpha), alpha)) for H: risk > alpha.

    h1(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b)) is the Bernoulli relative entropy; this is
    Hoeffding's (1963, Thm 1) bound, valid for i.i.d. losses in [0, 1] and never larger than the
    familiar exp(-2 n (alpha - risk_hat)^2) (Pinsker). Equals 1 when risk_hat >= alpha.
    """
    n, alpha = _check_n_alpha(n, alpha)
    r, scalar = _risks(risk_hat)
    a = np.minimum(r, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log(a / alpha), 0.0)
    h1 = np.maximum(t1 + (1.0 - a) * (np.log1p(-a) - math.log1p(-alpha)), 0.0)
    return _ret(np.exp(-n * h1), scalar)


def hb_pvalue(risk_hat: npt.ArrayLike, n: int, alpha: float) -> Union[float, np.ndarray]:
    """Hoeffding-Bentkus p-value min(Hoeffding, e * P(Binom(n, alpha) <= ceil(n * risk_hat))).

    The p-value recommended by LTT (Angelopoulos et al., 2021) and RCPS (Bates et al., 2021) for
    i.i.d. losses in [0, 1]; the second term is Bentkus' (2004) inequality.
    """
    n, alpha = _check_n_alpha(n, alpha)
    r, scalar = _risks(risk_hat)
    if n == 0:
        return _ret(np.ones_like(r), scalar)
    hoef = np.asarray(hoeffding_pvalue(r, n, alpha))
    bent = math.e * _binom_cdf(np.ceil(n * r - _TOL), n, alpha)
    return _ret(np.minimum(np.minimum(hoef, bent), 1.0), scalar)


PVALUES: dict[str, PValueFn] = {"binomial": binomial_pvalue, "hoeffding": hoeffding_pvalue, "hb": hb_pvalue}


# --------------------------------------------------------------------------------------------
# Learn then Test
# --------------------------------------------------------------------------------------------
def _scaled_losses(loss_mats: Sequence[npt.ArrayLike], alphas: Sequence[float],
                   bounds: float | Sequence[float]) -> tuple[list[np.ndarray], list[float]]:
    mats = [np.asarray(L, dtype=float) for L in loss_mats]
    m = len(mats)
    if m == 0:
        raise ValueError("need at least one loss matrix")
    alphas = [float(a) for a in alphas]
    if len(alphas) != m:
        raise ValueError(f"got {m} loss matrices but {len(alphas)} alphas")
    b = np.asarray(bounds, dtype=float).reshape(-1)
    if b.size == 1:
        b = np.full(m, b[0])
    if b.size != m:
        raise ValueError(f"bounds must be a scalar or have one entry per risk ({m}), got {b.size}")
    shape = mats[0].shape
    scaled = []
    for j, L in enumerate(mats):
        if L.ndim != 2 or L.shape != shape:
            raise ValueError(f"loss matrix {j} has shape {L.shape}; all must be 2-D (n_episodes, n_grid) "
                             f"and equal to {shape}")
        if not (math.isfinite(b[j]) and b[j] > 0):
            raise ValueError(f"bound {j} must be positive and finite, got {b[j]}")
        if np.isnan(L).any() or L.min(initial=0.0) < -_TOL or L.max(initial=0.0) > b[j] * (1 + _TOL):
            raise ValueError(f"loss matrix {j} must take values in [0, {b[j]:g}] (no NaN)")
        if not 0.0 < alphas[j] < b[j]:
            raise ValueError(f"alpha {j} must be in (0, bound) = (0, {b[j]:g}), got {alphas[j]}")
        scaled.append(np.clip(L / b[j], 0.0, 1.0))
    return scaled, [a / float(bj) for a, bj in zip(alphas, b)]


def ltt_pvalues(loss_mats: Sequence[npt.ArrayLike], alphas: Sequence[float], pvalue: str | PValueFn = "hb", *,
                bounds: float | Sequence[float] = 1.0) -> np.ndarray:
    """Per-risk p-values p[j, k] for H_{k,j}: R_j(grid[k]) > alphas[j]; shape (n_risks, n_grid).

    loss_mats[j] has shape (n_episodes, n_grid) with values in [0, bounds[j]]. `pvalue` is 'hb'
    (default), 'hoeffding', 'binomial' (binary losses only; checked) or a callable
    fn(risk_hat, n, alpha) for losses rescaled to [0, 1]. With no episodes every p-value is 1.
    """
    mats, alphas_s = _scaled_losses(loss_mats, alphas, bounds)
    if callable(pvalue):
        fn = pvalue
    elif pvalue in PVALUES:
        fn = PVALUES[pvalue]
    else:
        raise ValueError(f"pvalue must be one of {sorted(PVALUES)} or a callable, got {pvalue!r}")
    if pvalue == "binomial" and any(np.any((L != 0.0) & (L != 1.0)) for L in mats):
        raise ValueError("exact binomial p-values need binary losses (0 or the bound); use 'hb' for bounded losses")
    n, g = mats[0].shape
    if n == 0:
        return np.ones((len(mats), g))
    return np.vstack([np.asarray(fn(L.mean(axis=0), n, a), dtype=float).reshape(g) for L, a in zip(mats, alphas_s)])


def ltt_threshold(loss_mats: Sequence[npt.ArrayLike], grid: npt.ArrayLike, alphas: Sequence[float], delta: float,
                  pvalue: str | PValueFn = "hb", *, bounds: float | Sequence[float] = 1.0) -> float:
    """Learn-then-Test threshold controlling several risks with probability >= 1 - delta.

    Args:
        loss_mats: one (n_episodes, n_grid) loss matrix per risk, L_j[i, k] = loss j of episode i at
            lambda = grid[k], with values in [0, bounds[j]] (e.g. agnostic_losses / policy_losses).
        grid: ascending candidate thresholds (typically distinct scores plus +inf; larger lambda
            certifies fewer actions, i.e. is more conservative).
        alphas: risk budgets, one per risk (0 < alpha_j < bound_j).
        delta: tolerated probability that some budget is violated (FWER level).
        pvalue: 'hb' (default, any bounded loss), 'hoeffding', 'binomial' (exact, binary losses) or
            a callable fn(risk_hat, n, alpha) on losses rescaled to [0, 1].
        bounds: upper bound(s) of the losses (scalar or one per risk).

    Fixed-sequence testing from the largest grid point downwards: H_lambda ("some risk exceeds its
    budget") is rejected when max_j p_j(lambda) <= delta, and testing stops at the first
    non-rejection. Returns the smallest lambda rejected before that, or +inf (certify nothing) if
    the very first, most conservative point is not rejected. With i.i.d. episodes,

        P( R_j(lambda_hat) <= alpha_j for all j ) >= 1 - delta

    (see the module docstring for the proof). At lambda = +inf all losses are 0, which is rejected
    as soon as (1 - alpha_j / bound_j)^n <= delta for every j, i.e. n >= log(delta)/log(1 - alpha).
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or np.isnan(grid).any() or np.any(np.diff(grid) < 0):
        raise ValueError("grid must be a non-empty, ascending 1-D array")
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    p = ltt_pvalues(loss_mats, alphas, pvalue, bounds=bounds)
    if p.shape[1] != grid.size:
        raise ValueError(f"loss matrices have {p.shape[1]} columns but the grid has {grid.size} points")
    rejected = p.max(axis=0) <= delta                # intersection-union test at every lambda
    stops = np.flatnonzero(~rejected)
    k = int(stops[-1]) + 1 if stops.size else 0      # first non-rejection, scanning from the top
    return float("inf") if k >= grid.size else float(grid[k])
