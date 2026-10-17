"""Signal Temporal Logic (STL) with quantitative robustness for discrete-time signals.

The *robustness* rho(phi, x, t) of a formula is a signed margin (Donzé & Maler, 2010; Fainekos &
Pappas, 2009): positive when the signal satisfies the formula at time t, negative when it violates
it, and |rho| says how much the signal may be perturbed (in units of the predicates' `scale`) before
the verdict flips. That is exactly the contract of a riskgate `Constraint` (docs/SPEC.md §5.2), so
`STLConstraint` puts a formula into a `Spec`, and robust scores, CRC and LTT work on top unchanged.

Semantics (discrete time; a signal has T samples t = 0..T-1; [a, b] are integer step bounds with
0 <= a <= b, and b=None means "until the end of the signal")::

    Pred(k, '>=', c, s)     rho(t) = (x_k[t] - c) / s
    Pred(k, '<=', c, s)     rho(t) = (c - x_k[t]) / s
    Not(phi)                -rho_phi(t)
    And(phi_1, ..)          min_i rho_i(t)
    Or(phi_1, ..)           max_i rho_i(t)
    Implies(phi, psi)       max(-rho_phi(t), rho_psi(t))
    Always(phi, a, b)       min_{t' in W(t)} rho_phi(t')            W(t) = [t+a, t+b] ∩ [0, T-1]
    Eventually(phi, a, b)   max_{t' in W(t)} rho_phi(t')
    Until(phi, psi, a, b)   max_{t' in W(t)} min(rho_psi(t'), min_{t <= t'' <= t'} rho_phi(t''))

Conventions for finite signals
------------------------------
* Windows are *truncated* at the end of the signal (W(t) above).
* Empty windows: the min over an empty set is +inf and the max is -inf. So `Always` is vacuously
  satisfied (+inf) and `Eventually` / `Until` are unsatisfiable (-inf) at every t with t + a > T - 1.
  This keeps the dualities exact at every t, e.g. Always(phi) == Not(Eventually(Not(phi))).
  A +inf margin at t = 0 therefore means that the formula looks beyond the signal: make predicted
  trajectories at least `formula.horizon + 1` samples long.
* `Until` is non-strict by default: phi must hold on the closed interval [t, t'] (Maler & Nickovic,
  2004; Donzé & Maler, 2010). With `strict=True` phi must hold on [t, t') only (the LTL convention:
  psi holding now satisfies the Until whatever phi does).
* NaN samples propagate through every operator; `STLConstraint` maps a NaN margin to -inf (fail
  closed, rule R3), as `Spec` does.
* All 1-D signals read by a formula must have the same length; 0-d scalars are broadcast as constant
  signals.

Complexity: O(T) per Pred/Not/And/Or/Implies/Always/Eventually node (van Herk / Gil-Werman sliding
min/max) and O(T log T) per Until node (associative scan of clamp maps; bounded windows through the
exact identity rho(phi U[0,c] psi) = min(rho(phi U psi), rho(F[0,c] psi)), cf. Donzé, Ferrère &
Maler, 2013).

References
----------
O. Maler, D. Nickovic. Monitoring temporal properties of continuous signals. FORMATS/FTRTFT 2004.
A. Donzé, O. Maler. Robust satisfaction of temporal logic over real-valued signals. FORMATS 2010.
G. Fainekos, G. Pappas. Robustness of temporal logic specifications for continuous-time signals.
    Theoretical Computer Science 410(42), 2009.
A. Donzé, T. Ferrère, O. Maler. Efficient robust monitoring for STL. CAV 2013.
M. van Herk. A fast algorithm for local minimum and maximum filters on rectangular and octagonal
    kernels. Pattern Recognition Letters 13(7), 1992. J. Gil, M. Werman. Computing 2-D min, median,
    and max filters. IEEE TPAMI 15(5), 1993.
"""
from __future__ import annotations

import functools
import math
import operator
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .spec import Constraint, _get

__all__ = ["Formula", "Pred", "Not", "And", "Or", "Implies", "Always", "Eventually", "Until", "STLConstraint"]

_OPS = (">=", "<=")


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _formula(x: Any) -> Formula:
    if not isinstance(x, Formula):
        raise TypeError(f"expected an STL Formula, got {type(x).__name__}")
    return x


def _bounds(a: int, b: int | None) -> tuple[int, int | None]:
    """Validate integer step bounds 0 <= a <= b (b=None: until the end of the signal)."""
    a = operator.index(a)
    if a < 0:
        raise ValueError(f"lower bound a must be >= 0, got {a}")
    if b is not None:
        b = operator.index(b)
        if b < a:
            raise ValueError(f"upper bound b must be >= a (got a={a}, b={b})")
    return a, b


def _hmax(hs: Iterable[int | None]) -> int | None:
    out = 0
    for h in hs:
        if h is None:
            return None
        out = max(out, h)
    return out


def _ivl(a: int, b: int | None) -> str:
    return f"[{a},{'end' if b is None else b}]"


def _signals(signals: Any, keys: frozenset[str]) -> dict[str, np.ndarray]:
    """Read the signals a formula needs as float arrays of one common length T >= 1."""
    arrs: dict[str, np.ndarray] = {}
    consts: dict[str, float] = {}
    for k in sorted(keys):
        x = np.asarray(_get(signals, k), dtype=float)
        if x.ndim == 0:
            consts[k] = float(x)
        elif x.ndim == 1:
            arrs[k] = x
        else:
            raise ValueError(f"signal {k!r} must be a scalar or a 1-D array, got shape {x.shape}")
    lengths = {x.shape[0] for x in arrs.values()}
    if len(lengths) > 1:
        raise ValueError(f"signals have different lengths: { {k: x.shape[0] for k, x in arrs.items()} }")
    T = lengths.pop() if lengths else 1
    if T == 0:
        raise ValueError("empty signal: robustness is undefined without samples")
    for k, v in consts.items():
        arrs[k] = np.full(T, v)
    return arrs


def _sliding_min(x: np.ndarray, w: int) -> np.ndarray:
    """m[i] = min(x[i:i+w]) for i = 0..len(x)-w, in O(len(x)) (van Herk / Gil-Werman)."""
    n = x.shape[0]
    if w <= 1:
        return x.copy()
    nb = -(-n // w)
    blk = np.full(nb * w, np.inf)
    blk[:n] = x
    blk = blk.reshape(nb, w)
    pre = np.minimum.accumulate(blk, axis=1).ravel()                   # min from block start to i
    suf = np.minimum.accumulate(blk[:, ::-1], axis=1)[:, ::-1].ravel()  # min from i to block end
    i = np.arange(n - w + 1)
    return np.minimum(suf[i], pre[i + w - 1])


def _win_min(r: np.ndarray, a: int, b: int | None) -> np.ndarray:
    """out[t] = min of r over [t+a, t+b] ∩ [0, T-1]; +inf where that window is empty."""
    T = r.shape[0]
    out = np.full(T, np.inf)
    if a >= T:
        return out
    n = T - a                       # times whose window is non-empty
    y = r[a:]
    if b is None or b - a >= n - 1:  # every window runs to the end of the signal
        out[:n] = np.minimum.accumulate(y[::-1])[::-1]
        return out
    w = b - a + 1
    ext = np.full(n + w - 1, np.inf)
    ext[:n] = y
    out[:n] = _sliding_min(ext, w)
    return out


def _win_max(r: np.ndarray, a: int, b: int | None) -> np.ndarray:
    """out[t] = max of r over [t+a, t+b] ∩ [0, T-1]; -inf where that window is empty."""
    return -_win_min(-r, a, b)


def _until(phi: np.ndarray, psi: np.ndarray, a: int, b: int | None, strict: bool) -> np.ndarray:
    """Robustness trace of Until(phi, psi, a, b) from the traces of phi and psi.

    1. The unbounded until from s, U(s) = max_{s' >= s} min(psi(s'), min_{s <= s'' <= s'} phi(s'')),
       satisfies U(s) = f_s(U(s+1)), U(T) = -inf, with the clamp map f_s(x) = max(lo_s, min(hi_s, x)),
       lo = min(phi, psi), hi = phi (non-strict) or lo = psi, hi = max(phi, psi) (strict). Clamps are
       closed under composition (f o g has lo = f(lo_g), hi = f(hi_g)), so U(s) is the `lo` of the
       suffix composition f_s o ... o f_{T-1}, computed with a log-depth (Hillis-Steele) scan.
    2. Window [0, c]: U_c(s) = min(U(s), max_{s <= s' <= s+c} psi(s')). Exact, because the running
       minimum of phi is non-increasing in s' (the argmax beyond s+c can be traded for one inside).
    3. Offset a: rho(t) = min(min_{t <= t'' < t+a} phi(t''), U_{b-a}(t+a)), and -inf if t + a > T-1.
    """
    T = phi.shape[0]
    out = np.full(T, -np.inf)
    if a >= T:
        return out
    if strict:
        lo, hi = psi.copy(), np.maximum(phi, psi)
    else:
        lo, hi = np.minimum(phi, psi), phi.copy()
    d = 1
    while d < T:                    # position s now holds f_s o ... o f_{min(s+d-1, T-1)}
        lo_f, hi_f = lo[:-d], hi[:-d]
        new_lo = np.maximum(lo_f, np.minimum(hi_f, lo[d:]))
        new_hi = np.maximum(lo_f, np.minimum(hi_f, hi[d:]))
        lo[:-d], hi[:-d] = new_lo, new_hi
        d *= 2
    u = lo
    if b is not None and b - a < T - 1:
        u = np.minimum(u, _win_max(psi, 0, b - a))
    n = T - a
    res = u[a:]
    if a > 0:
        res = np.minimum(res, _win_min(phi, 0, a - 1)[:n])
    out[:n] = res
    return out


# --------------------------------------------------------------------------------------------
# formulas
# --------------------------------------------------------------------------------------------
class Formula:
    """Base class of STL formulas.

    Subclasses implement `keys` (signal names read), `horizon` (look-ahead in steps) and `_eval`
    (robustness trace on prepared signals). Formulas compose with `&` (And), `|` (Or) and `~` (Not).
    `repr` is exact and eval-able (used for fingerprints); `str` is a compact infix rendering.
    """

    def keys(self) -> frozenset[str]:
        """Names of the signals the formula reads."""
        raise NotImplementedError

    @property
    def horizon(self) -> int | None:
        """Upper bound on how many steps beyond t the formula reads (None = unbounded)."""
        raise NotImplementedError

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def rho(self, signals: Any) -> np.ndarray:
        """Robustness trace rho(t) for t = 0..T-1 (float array of length T).

        `signals` maps every key in `keys()` to a 1-D array (all of one length T) or to a scalar
        (constant signal). Objects exposing the keys as attributes are accepted too.
        """
        return self._eval(_signals(signals, self.keys()))

    def robustness(self, signals: Any, t: int = 0) -> float:
        """Robustness at time step t (default 0: the time at which the decision is taken)."""
        t = operator.index(t)
        r = self.rho(signals)
        if not 0 <= t < r.shape[0]:
            raise IndexError(f"t={t} is outside the signal (length {r.shape[0]})")
        return float(r[t])

    def __and__(self, other: Formula) -> And:
        return And(self, other)

    def __or__(self, other: Formula) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and repr(self) == repr(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Pred(Formula):
    """Atomic predicate `x_key >= threshold` (op '>=') or `x_key <= threshold` (op '<=').

    Robustness is the signed distance to the threshold in units of `scale`: (x - c) / scale for
    '>=' and (c - x) / scale for '<='. Pick `scale` so that the margins of different constraints
    are comparable (SPEC.md §5.2 normalisation rule).
    """

    def __init__(self, key: str, op: str, threshold: float, scale: float = 1.0):
        if op not in _OPS:
            raise ValueError(f"op must be one of {_OPS}, got {op!r}")
        threshold, scale = float(threshold), float(scale)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f"scale must be positive and finite, got {scale}")
        self.key, self.op, self.threshold, self.scale = str(key), op, threshold, scale

    def keys(self) -> frozenset[str]:
        return frozenset((self.key,))

    @property
    def horizon(self) -> int | None:
        return 0

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        x = sig[self.key]
        return (x - self.threshold) / self.scale if self.op == ">=" else (self.threshold - x) / self.scale

    def __repr__(self) -> str:
        s = f", scale={self.scale!r}" if self.scale != 1.0 else ""
        return f"Pred({self.key!r}, {self.op!r}, {self.threshold!r}{s})"

    def __str__(self) -> str:
        s = f"{self.key} {self.op} {self.threshold:g}"
        return f"({s})/{self.scale:g}" if self.scale != 1.0 else s


class Not(Formula):
    """Negation: rho = -rho_phi."""

    def __init__(self, phi: Formula):
        self.phi = _formula(phi)

    def keys(self) -> frozenset[str]:
        return self.phi.keys()

    @property
    def horizon(self) -> int | None:
        return self.phi.horizon

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return -self.phi._eval(sig)

    def __repr__(self) -> str:
        return f"Not({self.phi!r})"

    def __str__(self) -> str:
        return f"!({self.phi})"


class _NAry(Formula):
    _sym = ""

    def __init__(self, *phis: Formula):
        if not phis:
            raise ValueError(f"{type(self).__name__} needs at least one operand")
        self.phis = tuple(_formula(p) for p in phis)

    def keys(self) -> frozenset[str]:
        return frozenset().union(*(p.keys() for p in self.phis))

    @property
    def horizon(self) -> int | None:
        return _hmax(p.horizon for p in self.phis)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(repr(p) for p in self.phis)})"

    def __str__(self) -> str:
        return "(" + f" {self._sym} ".join(str(p) for p in self.phis) + ")"


class And(_NAry):
    """Conjunction: rho = min_i rho_i."""

    _sym = "&"

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return functools.reduce(np.minimum, (p._eval(sig) for p in self.phis))


class Or(_NAry):
    """Disjunction: rho = max_i rho_i."""

    _sym = "|"

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return functools.reduce(np.maximum, (p._eval(sig) for p in self.phis))


class Implies(Formula):
    """Implication phi -> psi, i.e. Or(Not(phi), psi): rho = max(-rho_phi, rho_psi)."""

    def __init__(self, phi: Formula, psi: Formula):
        self.phi, self.psi = _formula(phi), _formula(psi)

    def keys(self) -> frozenset[str]:
        return self.phi.keys() | self.psi.keys()

    @property
    def horizon(self) -> int | None:
        return _hmax((self.phi.horizon, self.psi.horizon))

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.maximum(-self.phi._eval(sig), self.psi._eval(sig))

    def __repr__(self) -> str:
        return f"Implies({self.phi!r}, {self.psi!r})"

    def __str__(self) -> str:
        return f"({self.phi} -> {self.psi})"


class Always(Formula):
    """G[a,b] phi: rho(t) = min of rho_phi over [t+a, t+b] ∩ [0, T-1]; +inf if that window is empty."""

    def __init__(self, phi: Formula, a: int = 0, b: int | None = None):
        self.phi = _formula(phi)
        self.a, self.b = _bounds(a, b)

    def keys(self) -> frozenset[str]:
        return self.phi.keys()

    @property
    def horizon(self) -> int | None:
        h = self.phi.horizon
        return None if self.b is None or h is None else self.b + h

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return _win_min(self.phi._eval(sig), self.a, self.b)

    def __repr__(self) -> str:
        return f"Always({self.phi!r}, {self.a}, {self.b})"

    def __str__(self) -> str:
        return f"G{_ivl(self.a, self.b)}({self.phi})"


class Eventually(Formula):
    """F[a,b] phi: rho(t) = max of rho_phi over [t+a, t+b] ∩ [0, T-1]; -inf if that window is empty."""

    def __init__(self, phi: Formula, a: int = 0, b: int | None = None):
        self.phi = _formula(phi)
        self.a, self.b = _bounds(a, b)

    def keys(self) -> frozenset[str]:
        return self.phi.keys()

    @property
    def horizon(self) -> int | None:
        h = self.phi.horizon
        return None if self.b is None or h is None else self.b + h

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return _win_max(self.phi._eval(sig), self.a, self.b)

    def __repr__(self) -> str:
        return f"Eventually({self.phi!r}, {self.a}, {self.b})"

    def __str__(self) -> str:
        return f"F{_ivl(self.a, self.b)}({self.phi})"


class Until(Formula):
    """phi U[a,b] psi: psi holds at some t' in [t+a, t+b] and phi holds from t up to t'.

    rho(t) = max_{t' in [t+a, t+b] ∩ [0, T-1]} min(rho_psi(t'), min_{t <= t'' <= t'} rho_phi(t''))
    (-inf if the window is empty). Note that phi is required from t itself, also before the window
    opens at t+a. With `strict=True` the inner minimum runs over t <= t'' < t' instead.
    """

    def __init__(self, phi: Formula, psi: Formula, a: int = 0, b: int | None = None, *, strict: bool = False):
        self.phi, self.psi = _formula(phi), _formula(psi)
        self.a, self.b = _bounds(a, b)
        self.strict = bool(strict)

    def keys(self) -> frozenset[str]:
        return self.phi.keys() | self.psi.keys()

    @property
    def horizon(self) -> int | None:
        h = _hmax((self.phi.horizon, self.psi.horizon))
        return None if self.b is None or h is None else self.b + h

    def _eval(self, sig: Mapping[str, np.ndarray]) -> np.ndarray:
        return _until(self.phi._eval(sig), self.psi._eval(sig), self.a, self.b, self.strict)

    def __repr__(self) -> str:
        s = ", strict=True" if self.strict else ""
        return f"Until({self.phi!r}, {self.psi!r}, {self.a}, {self.b}{s})"

    def __str__(self) -> str:
        return f"({self.phi} U{'s' if self.strict else ''}{_ivl(self.a, self.b)} {self.psi})"


# --------------------------------------------------------------------------------------------
# riskgate constraint
# --------------------------------------------------------------------------------------------
class STLConstraint(Constraint):
    """A riskgate `Constraint` whose signed margin is the STL robustness at t = 0.

    `outcome` maps each signal name to a 1-D trajectory (or a scalar), as a Mapping or as
    attributes. Example::

        band = Always(Pred("T", "<=", 362, scale=10) & Pred("T", ">=", 340, scale=10))
        spec = Spec(STLConstraint("340 <= T <= 362", band), Upper("p95_ms", 800, scale=800))

    NaN margins become -inf (fail closed). `describe()` embeds the exact `repr` of the formula, so
    the Spec fingerprint, and with it the validity of a stored calibration, changes whenever the
    formula changes.
    """

    def __init__(self, name: str, formula: Formula):
        if not isinstance(formula, Formula):
            raise TypeError(f"formula must be an STL Formula, got {type(formula).__name__}")
        self.name, self.formula = str(name), formula

    def margin(self, outcome: Any) -> float:
        m = self.formula.robustness(outcome, 0)
        return -math.inf if math.isnan(m) else m

    def describe(self) -> dict:
        return {"type": type(self).__name__, "name": self.name, "formula": repr(self.formula)}
