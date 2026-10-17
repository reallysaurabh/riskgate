"""Safety specifications: constraints that turn a predicted outcome into a signed margin.

A margin is positive when the constraint is satisfied and negative when it is violated;
its magnitude says by how much (in units of `scale`, so constraints are comparable).
A Spec is the conjunction of its constraints: margin = min over constraints.
Hard rules are deterministic, action-level checks evaluated before any prediction.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import Action, Verdict, callable_digest, digest


def _get(outcome: Any, key: str) -> Any:
    if isinstance(outcome, Mapping):
        return outcome[key]
    return getattr(outcome, key)


def _arr(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


class Constraint:
    name: str = "constraint"

    def margin(self, outcome: Any) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        """Fingerprint material: type, parameters and a content hash of any function (see callable_digest)."""
        return {"type": type(self).__name__,
                **{k: (f"callable:{callable_digest(v)}" if callable(v) and not isinstance(v, type) else v)
                   for k, v in vars(self).items()}}


def _check_scale(scale: float) -> float:
    s = float(scale)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(f"scale must be positive and finite, got {scale}")
    return s


class Upper(Constraint):
    """max(outcome[key]) <= limit."""

    def __init__(self, key: str, limit: float, scale: float = 1.0, name: str | None = None):
        limit = float(limit)
        if not math.isfinite(limit):
            raise ValueError(f"limit must be finite, got {limit}")
        self.key, self.limit, self.scale = key, limit, _check_scale(scale)
        self.name = name or f"{key}<={limit:g}"

    def margin(self, outcome: Any) -> float:
        return float((self.limit - _arr(_get(outcome, self.key)).max()) / self.scale)


class Lower(Constraint):
    """min(outcome[key]) >= limit."""

    def __init__(self, key: str, limit: float, scale: float = 1.0, name: str | None = None):
        limit = float(limit)
        if not math.isfinite(limit):
            raise ValueError(f"limit must be finite, got {limit}")
        self.key, self.limit, self.scale = key, limit, _check_scale(scale)
        self.name = name or f"{key}>={limit:g}"

    def margin(self, outcome: Any) -> float:
        return float((_arr(_get(outcome, self.key)).min() - self.limit) / self.scale)


class Envelope(Constraint):
    """lo <= outcome[key] <= hi at every point of a trajectory (or for a scalar)."""

    def __init__(self, key: str, lo: float, hi: float, scale: float = 1.0, name: str | None = None):
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError(f"envelope bounds must be finite with lo <= hi (got lo={lo}, hi={hi})")
        self.key, self.lo, self.hi, self.scale = key, lo, hi, _check_scale(scale)
        self.name = name or f"{lo:g}<={key}<={hi:g}"

    def margin(self, outcome: Any) -> float:
        x = _arr(_get(outcome, self.key))
        return float(min(self.hi - x.max(), x.min() - self.lo) / self.scale)


class Predicate(Constraint):
    """Boolean constraint; fn(outcome) -> True when satisfied."""

    def __init__(self, name: str, fn: Callable[[Any], bool], safe_margin: float = 1.0, unsafe_margin: float = -1.0):
        safe_margin, unsafe_margin = float(safe_margin), float(unsafe_margin)
        if not (math.isfinite(safe_margin) and math.isfinite(unsafe_margin) and safe_margin > unsafe_margin):
            raise ValueError(f"safe_margin ({safe_margin}) must be finite and > unsafe_margin ({unsafe_margin})")
        self.name, self.fn, self.safe_margin, self.unsafe_margin = name, fn, safe_margin, unsafe_margin

    def margin(self, outcome: Any) -> float:
        return self.safe_margin if self.fn(outcome) else self.unsafe_margin


class Custom(Constraint):
    """Any signed-margin function fn(outcome) -> float (positive = safe)."""

    def __init__(self, name: str, fn: Callable[[Any], float]):
        self.name, self.fn = name, fn

    def margin(self, outcome: Any) -> float:
        return float(self.fn(outcome))


class Spec:
    """Conjunction of constraints. NaN margins are treated as -inf (fail closed)."""

    def __init__(self, *constraints: Constraint, floor: float | None = None):
        if not constraints:
            raise ValueError("a Spec needs at least one constraint")
        self.constraints = list(constraints)
        self.floor = floor   # optional lower clip for margins (keeps scores finite, does not affect decisions)

    def evaluate(self, outcome: Any) -> tuple[float, str]:
        best, name = math.inf, ""
        for c in self.constraints:
            m = c.margin(outcome)
            if m != m:  # NaN
                m = -math.inf
            if m < best:
                best, name = m, c.name
        if self.floor is not None:
            best = max(best, self.floor)
        return best, name

    def margin(self, outcome: Any) -> float:
        return self.evaluate(outcome)[0]

    def violated(self, outcome: Any) -> bool:
        return self.margin(outcome) < 0

    def fingerprint(self) -> str:
        return digest([c.describe() for c in self.constraints] + [self.floor])


@dataclass(frozen=True)
class HardRule:
    """Deterministic, action-level rule evaluated before any prediction.

    `fires(action, context)` returns True when the rule applies; the gate then returns
    `verdict` (REJECT: never execute; ASK: needs human approval) regardless of calibration.
    """

    name: str
    fires: Callable[[Action, Any], bool]
    verdict: Verdict = Verdict.REJECT
    message: str = ""
    version: str | None = None      # pins the rule's identity; default: a content hash of `fires`

    def describe(self) -> list:
        """Fingerprint material: editing the rule's logic (or its verdict) voids the calibration."""
        return [self.name, self.verdict.value, self.version or callable_digest(self.fires)]
