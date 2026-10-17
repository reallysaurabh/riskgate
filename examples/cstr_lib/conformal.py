"""Split conformal prediction and conformal risk control (CRC)."""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score: P(s_new <= q) >= 1 - alpha under exchangeability."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def crc_threshold(loss_fn, lambdas: np.ndarray, alpha: float, n: int) -> float:
    """Conformal risk control (Angelopoulos et al., ICLR 2024).

    loss_fn(lam) -> array of per-calibration-unit losses in [0, 1], non-increasing in lam.
    Returns the smallest lam with  n/(n+1) * mean(L(lam)) + 1/(n+1) <= alpha,
    which guarantees E[L_new(lam_hat)] <= alpha for exchangeable units.
    """
    for lam in np.sort(lambdas):
        risk = np.mean(loss_fn(lam))
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return float(lam)
    return float("inf")
