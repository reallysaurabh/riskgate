"""Distribution-free calibration primitives.

* split-conformal quantiles (hypothesis-set threshold, OOD threshold)
* conformal risk control (CRC, Angelopoulos et al., ICLR 2024) over a finite grid
* episode loss curves for the two guarantee modes:
    - "agnostic": loss = 1{any candidate with score >= lambda is unsafe}. Valid for ANY doer
      that only executes certified actions (requires labels for every candidate).
    - "policy":   loss = 1{the action the doer's ranking actually executes is unsafe}. Tighter,
      valid for the calibrated doer only. Not monotone in lambda, so CRC is applied to its upper
      envelope sup_{lambda' >= lambda} L(lambda'), which is monotone and dominates it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score; +inf when n is too small.

    For exchangeable data, P(s_new <= q) >= 1 - alpha.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.sort(np.asarray(scores, dtype=float))
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    n = s.shape[0]
    k = max(1, math.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n or n == 0 else float(s[k - 1])


def lambda_grid(score_lists: Sequence[Any]) -> np.ndarray:
    """All distinct finite scores plus +inf. Thresholding 'score >= lambda' only changes at these
    values, so restricting CRC to this grid loses nothing (no artificial floor or ceiling)."""
    vals = np.unique(np.concatenate([np.asarray(s, float) for s in score_lists if len(s)] or [np.array([])]))
    vals = vals[np.isfinite(vals)]
    return np.concatenate([vals, [np.inf]])


def agnostic_losses(scores: Sequence[np.ndarray], unsafe: Sequence[np.ndarray], grid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(scores), grid.shape[0]))
    for i, (s, u) in enumerate(zip(scores, unsafe)):
        s, u = np.asarray(s, float), np.asarray(u, bool)
        if u.any():
            out[i] = grid <= s[u].max()          # some unsafe action is certified iff lambda <= its score
    return out


def policy_losses(scores: Sequence[np.ndarray], unsafe: Sequence[np.ndarray], grid: np.ndarray,
                  envelope: bool = True) -> np.ndarray:
    """Policy-mode losses: L_i(lambda) = loss of the FIRST candidate (in the doer's rank order) whose score is
    >= lambda, 0 if none. `unsafe` holds per-candidate losses in [0, 1] (0/1 labels for safety; any bounded
    value for extra risks such as cost overruns). With envelope=True the running maximum from the right makes
    each row non-increasing in lambda (a conservative upper bound, as conformal risk control requires)."""
    out = np.zeros((len(scores), grid.shape[0]))
    for i, (s, u) in enumerate(zip(scores, unsafe)):
        prev = -np.inf
        for sj, uj in zip(np.asarray(s, float), np.clip(np.asarray(u, float), 0.0, 1.0)):
            if sj > prev:                         # candidate j is the first certified for lambda in (prev, sj]
                out[i, (grid > prev) & (grid <= sj)] = float(uj)
                prev = sj
    if envelope:
        out = np.maximum.accumulate(out[:, ::-1], axis=1)[:, ::-1]
    return out


def crc_threshold(losses: np.ndarray, grid: np.ndarray, alpha: float, bound: float = 1.0) -> float:
    """Smallest lambda in `grid` with n/(n+1) * mean_i L_i(lambda) + B/(n+1) <= alpha.

    Requires each row to be non-increasing along the (ascending) grid and L(+inf) = 0.
    Guarantees E[L_{n+1}(lambda_hat)] <= alpha for exchangeable episodes.
    """
    alpha, bound = float(alpha), float(bound)
    if not (math.isfinite(bound) and bound > 0.0):
        raise ValueError(f"bound must be positive and finite, got {bound}")
    if not 0.0 < alpha < bound:
        raise ValueError(f"alpha must be in (0, bound) = (0, {bound:g}), got {alpha}")
    losses = np.asarray(losses, dtype=float)
    if np.isnan(losses).any():
        raise ValueError("losses contain NaN")
    n = losses.shape[0]
    if n == 0:
        return float("inf")
    risk = losses.mean(axis=0)
    ok = n / (n + 1) * risk + bound / (n + 1) <= alpha
    idx = np.flatnonzero(ok)
    return float(grid[idx[0]]) if idx.size else float("inf")


def ask_rate(scores: Sequence[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Fraction of episodes in which no candidate is certified, as a function of lambda."""
    best = np.array([np.max(s) if len(s) else -np.inf for s in scores])
    return (best[:, None] < grid[None, :]).mean(axis=0)
