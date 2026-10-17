"""Conformal calibration under covariate shift with a known likelihood ratio.

Split conformal and CRC (riskgate.conformal) assume that calibration and deployment episodes are
exchangeable. Under *covariate shift* (Tibshirani, Barber, Candès & Ramdas, 2019) the context
distribution changes from P (calibration) to Q (deployment), dQ/dP(x) = w(x) is known, and the
conditional law of the outcome given the context is unchanged (e.g. a planned new plant mode, a
new traffic mix with a known mixture ratio). Reweighting the calibration episodes restores validity:

    p_i = w(x_i) / (sum_k w(x_k) + w(x_test)),   p_test = w(x_test) / (sum_k w(x_k) + w(x_test)).

* Weighted conformal quantile: q_hat = Quantile(1 - alpha; sum_i p_i delta_{s_i} + p_test delta_{+inf})
  gives P(s_test <= q_hat) >= 1 - alpha (Tibshirani et al., 2019).
* Weighted CRC: lambda_hat = min{lambda in grid : sum_i p_i L_i(lambda) + p_test B <= alpha} gives
  E[L_test(lambda_hat)] <= alpha for non-increasing losses bounded by B (Angelopoulos, Bates, Fisch,
  Lei & Schuster, 2024, covariate-shift extension of CRC).

Both thresholds depend on the TEST context through w(x_test): recompute them for every decision
(`WeightedCRC` precomputes the calibration side, so each threshold costs O(log n_grid)). A large test
weight (a context that is rare under calibration) makes them conservative, up to +inf (certify
nothing), which is the desired fail-closed behaviour (rule R3). The weights only need to be known up
to a common positive factor. With all weights equal they reduce exactly to
`conformal_quantile` and `crc_threshold`.

References
----------
R. J. Tibshirani, R. F. Barber, E. J. Candès, A. Ramdas. Conformal prediction under covariate
    shift. NeurIPS 2019.
A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster. Conformal risk control. ICLR 2024.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import numpy.typing as npt

__all__ = ["WeightedCRC", "weighted_conformal_quantile", "weighted_crc_threshold"]


def _cal_weights(w_cal: npt.ArrayLike, n: int) -> np.ndarray:
    w = np.asarray(w_cal, dtype=float)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"w_cal must be a 1-D array with one weight per calibration point ({n}), got shape {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("calibration weights must be finite and >= 0")
    return w


def _test_weights(w_test: npt.ArrayLike) -> tuple[np.ndarray, bool]:
    wt = np.asarray(w_test, dtype=float)
    scalar = wt.ndim == 0
    wt = np.atleast_1d(wt)
    if wt.ndim != 1:
        raise ValueError(f"w_test must be a scalar or a 1-D array, got shape {wt.shape}")
    if np.isnan(wt).any() or np.any(wt < 0):
        raise ValueError("test weights must be >= 0 (+inf allowed, NaN not)")
    return wt, scalar


def _ret(x: np.ndarray, scalar: bool) -> Union[float, np.ndarray]:
    return float(x[0]) if scalar else x


def weighted_conformal_quantile(scores: npt.ArrayLike, w_cal: npt.ArrayLike, w_test: npt.ArrayLike,
                                alpha: float) -> Union[float, np.ndarray]:
    """Weighted split-conformal quantile (Tibshirani et al., 2019).

    Returns the smallest calibration score s with sum_{i: s_i <= s} w_i >= (1 - alpha) (sum_i w_i + w_test),
    i.e. the (1 - alpha)-quantile of sum_i p_i delta_{s_i} + p_test delta_{+inf}; +inf when the
    calibration mass is insufficient (few points, or a large test weight). Under covariate shift
    with likelihood ratio w, P(s_test <= q_hat) >= 1 - alpha.

    `w_test` may be a scalar (returns float) or a 1-D array of test weights (returns an array).
    Unit weights give exactly `riskgate.conformal.conformal_quantile`.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = s.shape[0]
    w = _cal_weights(w_cal, n)
    wt, scalar = _test_weights(w_test)
    out = np.full(wt.shape, np.inf)
    fin = np.isfinite(wt)
    if n > 0 and fin.any():
        order = np.argsort(s, kind="stable")
        s_sorted, cum = s[order], np.cumsum(w[order])
        total = cum[-1] + wt[fin]
        idx = np.searchsorted(cum, (1.0 - alpha) * total, side="left")   # first index with enough mass
        ok = (idx < n) & (total > 0)
        vals = np.full(idx.shape, np.inf)
        vals[ok] = s_sorted[idx[ok]]
        out[fin] = vals
    return _ret(out, scalar)


class WeightedCRC:
    """Weighted conformal risk control with the calibration side precomputed.

    `WeightedCRC(losses, grid, alpha, w_cal, bound).threshold(w_test)` equals
    `weighted_crc_threshold(losses, grid, alpha, w_cal, w_test, bound)` and costs O(log n_grid) per
    test weight, so the threshold can be recomputed for every decision.
    """

    def __init__(self, losses: npt.ArrayLike, grid: npt.ArrayLike, alpha: float, w_cal: npt.ArrayLike,
                 bound: float = 1.0):
        L = np.asarray(losses, dtype=float)
        self.grid = np.asarray(grid, dtype=float)
        if L.ndim != 2:
            if L.size:
                raise ValueError(f"losses must be 2-D (n_episodes, n_grid), got shape {L.shape}")
            L = L.reshape(0, self.grid.shape[0] if self.grid.ndim == 1 else 0)
        n, g = L.shape
        if self.grid.ndim != 1 or self.grid.shape[0] != g:
            raise ValueError(f"grid must be 1-D with one point per loss column ({g}), got shape {self.grid.shape}")
        if np.isnan(self.grid).any() or np.any(np.diff(self.grid) < 0):
            raise ValueError("grid must be ascending")
        self.alpha, self.bound = float(alpha), float(bound)
        if not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if not (math.isfinite(self.bound) and self.bound > 0):
            raise ValueError(f"bound must be positive and finite, got {bound}")
        if np.isnan(L).any() or L.max(initial=-np.inf) > self.bound * (1 + 1e-12):
            raise ValueError(f"losses must be <= bound = {self.bound:g} (CRC requires bounded losses) and not NaN")
        w = _cal_weights(w_cal, n)
        self.n = n
        self.w_sum = float(w.sum())
        self.weighted_loss = w @ L if n else np.zeros(g)                    # S(lambda) = sum_i w_i L_i(lambda)
        self._run_min = np.minimum.accumulate(self.weighted_loss) if g else self.weighted_loss

    def threshold(self, w_test: npt.ArrayLike) -> Union[float, np.ndarray]:
        """Smallest grid lambda with S(lambda) + w_test * B <= alpha * (sum w + w_test); +inf if none."""
        wt, scalar = _test_weights(w_test)
        out = np.full(wt.shape, np.inf)
        g = self.grid.shape[0]
        if self.n == 0 or g == 0:
            return _ret(out, scalar)
        fin = np.isfinite(wt)
        wf = wt[fin]
        # S + w_t B <= alpha (W + w_t)  <=>  S <= alpha W + w_t (alpha - B). The first grid index where S
        # satisfies it is the first index where the running minimum of S does (monotone -> binary search).
        cap = self.alpha * self.w_sum + wf * (self.alpha - self.bound)
        idx = np.searchsorted(-self._run_min, -cap, side="left")
        ok = (idx < g) & (self.w_sum + wf > 0)
        vals = np.full(idx.shape, np.inf)
        vals[ok] = self.grid[idx[ok]]
        out[fin] = vals
        if self.bound <= self.alpha:            # w_test = +inf: p_test = 1, condition is B <= alpha
            out[~fin] = self.grid[0]
        return _ret(out, scalar)


def weighted_crc_threshold(losses: npt.ArrayLike, grid: npt.ArrayLike, alpha: float, w_cal: npt.ArrayLike,
                           w_test: npt.ArrayLike, bound: float = 1.0) -> Union[float, np.ndarray]:
    """Conformal risk control under covariate shift with a known likelihood ratio.

    Smallest lambda in the ascending `grid` with

        sum_i p_i L_i(lambda) + p_test * bound <= alpha,
        p_i = w_cal[i] / (sum(w_cal) + w_test),   p_test = w_test / (sum(w_cal) + w_test),

    or +inf if no grid point qualifies. `losses` is (n_episodes, n_grid) with rows non-increasing
    along the grid, L(+inf) = 0 and L <= bound (as for `crc_threshold`); w_cal[i] = w(x_i) and
    w_test = w(x_test) for the likelihood ratio w = dQ/dP of the contexts. Then
    E[L_test(lambda_hat)] <= alpha under covariate shift (Angelopoulos et al., 2024, after
    Tibshirani et al., 2019). The threshold depends on the test context: recompute it per decision
    (see `WeightedCRC`). `w_test` may be a scalar (returns float) or a 1-D array (returns an array).
    Unit weights give exactly `riskgate.conformal.crc_threshold`.
    """
    return WeightedCRC(losses, grid, alpha, w_cal, bound).threshold(w_test)
