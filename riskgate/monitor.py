"""Online operation: drift detection and conservative threshold adaptation.

* A mixture power martingale over conformal p-values of the gate's top-candidate scores tests
  whether deployment scores still look like calibration scores. By Ville's inequality the false
  alarm probability is <= 1/alarm_level under exchangeability (approximately, since the p-values
  are computed against a fixed calibration sample).
* When drift is flagged, an adaptive-conformal-style update (Gibbs & Candes, 2021) adjusts the
  threshold from observed outcomes of certified actions. The gate only ever uses
  max(calibrated lambda, online lambda): online adaptation can tighten, never loosen.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np


class Monitor:
    """Drift alarm + delayed-outcome bookkeeping for one gate.

    alarm_level  Ville threshold: false-alarm probability <= 1/alarm_level under exchangeability
    eta          step size of the online (tighten-only) threshold; default 5% of the calibration score spread
    latch        once raised, the alarm stays on (status DEGRADED) until the gate is recalibrated (bind) or
                 reset() is called explicitly -- a falling martingale is not evidence that drift has ended
    """

    def __init__(self, alarm_level: float = 100.0, eta: float | None = None, seed: int = 0, latch: bool = True,
                 max_history: int = 10_000):
        if not (math.isfinite(alarm_level) and alarm_level > 1.0):
            raise ValueError(f"alarm_level must be finite and > 1, got {alarm_level}")
        if eta is not None and not (math.isfinite(eta) and eta >= 0.0):
            raise ValueError(f"eta must be finite and >= 0, got {eta}")
        self.alarm_level = float(alarm_level)
        self._eta_user = float(eta) if eta is not None else None
        self.eta = self._eta_user
        self.latch = latch
        self._latched = False
        self.eps = np.linspace(0.02, 0.98, 49)
        self.log_m = np.zeros_like(self.eps)
        self.ref: np.ndarray | None = None
        self.alpha = 0.05
        self.online_lambda: float | None = None
        self.n_seen = 0
        self.rng = np.random.default_rng(seed)
        self.history: list[float] = []
        self.max_history = max(1, int(max_history))
        self.pending: dict[str, float] = {}      # decision_id -> time certified (labels may arrive later)
        self.max_pending = 100_000
        self.outcomes = {"certified_with_outcome": 0, "unsafe": 0}
        self._lock = threading.RLock()

    def bind(self, calibration: Any) -> None:
        with self._lock:
            ref = np.asarray([s for s in calibration.reference_scores], float)
            ref[np.isneginf(ref)] = -1e12
            self.ref = np.sort(ref)
            self.alpha = calibration.alpha
            self.online_lambda = calibration.lam if math.isfinite(calibration.lam) else None
            if self._eta_user is not None:
                self.eta = self._eta_user
            else:
                finite = self.ref[np.isfinite(self.ref) & (self.ref > -1e11) & (self.ref < 1e299)]
                spread = float(np.subtract(*np.percentile(finite, [90, 10]))) if finite.size > 10 else 1.0
                self.eta = 0.05 * max(spread, 1e-6)
            self.pending.clear()                       # decisions certified under the previous calibration
            self.outcomes = {"certified_with_outcome": 0, "unsafe": 0}
            self.reset()

    def reset(self) -> None:
        """Restart the drift test (martingale, history, latched alarm). Outcome counters are kept."""
        with self._lock:
            self.log_m[:] = 0.0
            self.n_seen = 0
            self.history.clear()
            self._latched = False

    @property
    def log_martingale(self) -> float:
        with self._lock:
            mx = float(np.max(self.log_m))
            return mx + math.log(float(np.mean(np.exp(self.log_m - mx))))

    @property
    def martingale(self) -> float:
        with self._lock:
            lm = self.log_martingale
            return math.exp(lm) if lm < 700 else math.inf

    @property
    def alarm(self) -> bool:
        with self._lock:
            return self.ref is not None and (self._latched or self.martingale >= self.alarm_level)

    def update_score(self, score: float) -> None:
        with self._lock:
            if self.ref is None or self.ref.size == 0:
                return
            s = -1e12 if score == -math.inf else min(float(score), 1e300)
            n = self.ref.shape[0]
            lo = np.searchsorted(self.ref, s, side="left")
            hi = np.searchsorted(self.ref, s, side="right")
            p = (lo + self.rng.random() * (hi - lo + 1)) / (n + 1)       # small p = unusually LOW margin
            p = min(max(p, 1e-12), 1.0)
            self.log_m += np.log(self.eps) + (self.eps - 1.0) * math.log(p)
            self.n_seen += 1
            m = self.martingale
            if len(self.history) >= self.max_history:
                del self.history[: len(self.history) - self.max_history + 1]
            self.history.append(m)
            if self.latch and m >= self.alarm_level:
                self._latched = True

    def observe_outcome(self, certified: bool, unsafe: bool) -> None:
        """Feed back the realised outcome of an executed decision (ACI-style update of the online threshold)."""
        if not certified:
            return
        with self._lock:
            self.outcomes["certified_with_outcome"] += 1
            self.outcomes["unsafe"] += int(bool(unsafe))
            if self.online_lambda is None or self.eta is None:
                return
            self.online_lambda += self.eta * ((1.0 if unsafe else 0.0) - self.alpha)

    # -------------------------------------------------- delayed labels
    def register(self, decision: Any) -> None:
        """Called by the gate for every CERTIFY decision; outcomes can be reported later by decision_id."""
        import time as _t
        with self._lock:
            if len(self.pending) >= self.max_pending:
                self.pending.pop(next(iter(self.pending)))
            self.pending[decision.decision_id] = _t.time()

    def report_outcome(self, decision_id: str, unsafe: bool) -> bool:
        """Report a (possibly delayed) outcome. Returns False if the decision is unknown or already reported."""
        with self._lock:
            if self.pending.pop(decision_id, None) is None:
                return False
            self.observe_outcome(certified=True, unsafe=unsafe)
            return True

    @property
    def observed_risk(self) -> float | None:
        with self._lock:
            n = self.outcomes["certified_with_outcome"]
            return self.outcomes["unsafe"] / n if n else None
