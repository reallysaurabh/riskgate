"""Tests for riskgate.stl: robustness vs. a brute-force reference, finite-signal conventions, Spec integration."""
import math
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from riskgate import stl
from riskgate.spec import Envelope, Lower, Spec, Upper
from riskgate.stl import Always, And, Eventually, Formula, Implies, Not, Or, Pred, STLConstraint, Until

INF = math.inf
KEYS = ("x", "y")


# --------------------------------------------------------------------------- brute-force references
def ref_trace(f, sig, T, memo=None):
    """Robustness trace straight from the definitions (explicit loops; empty min = +inf, empty max = -inf)."""
    memo = {} if memo is None else memo
    if id(f) in memo:
        return memo[id(f)]

    def sub(g):
        return ref_trace(g, sig, T, memo)

    def window(t, a, b):
        hi = T - 1 if b is None else min(t + b, T - 1)
        return range(t + a, hi + 1)

    if isinstance(f, Pred):
        x = sig[f.key]
        out = [((x[t] - f.threshold) if f.op == ">=" else (f.threshold - x[t])) / f.scale for t in range(T)]
    elif isinstance(f, Not):
        out = [-v for v in sub(f.phi)]
    elif isinstance(f, And):
        cs = [sub(g) for g in f.phis]
        out = [min(c[t] for c in cs) for t in range(T)]
    elif isinstance(f, Or):
        cs = [sub(g) for g in f.phis]
        out = [max(c[t] for c in cs) for t in range(T)]
    elif isinstance(f, Implies):
        p, q = sub(f.phi), sub(f.psi)
        out = [max(-p[t], q[t]) for t in range(T)]
    elif isinstance(f, Always):
        c = sub(f.phi)
        out = [min((c[k] for k in window(t, f.a, f.b)), default=INF) for t in range(T)]
    elif isinstance(f, Eventually):
        c = sub(f.phi)
        out = [max((c[k] for k in window(t, f.a, f.b)), default=-INF) for t in range(T)]
    elif isinstance(f, Until):
        p, q = sub(f.phi), sub(f.psi)
        out = []
        for t in range(T):
            best = -INF
            for tp in window(t, f.a, f.b):
                upto = tp if f.strict else tp + 1           # phi on [t, t') if strict else [t, t']
                inner = min((p[k] for k in range(t, upto)), default=INF)
                best = max(best, min(q[tp], inner))
            out.append(best)
    else:  # pragma: no cover
        raise TypeError(f)
    memo[id(f)] = out
    return out


def bool_trace(f, sig, T):
    """Boolean (qualitative) semantics with the same finite-trace conventions."""
    def window(t, a, b):
        hi = T - 1 if b is None else min(t + b, T - 1)
        return range(t + a, hi + 1)

    if isinstance(f, Pred):
        x = sig[f.key]
        return [bool(x[t] >= f.threshold) if f.op == ">=" else bool(x[t] <= f.threshold) for t in range(T)]
    if isinstance(f, Not):
        return [not v for v in bool_trace(f.phi, sig, T)]
    if isinstance(f, And):
        cs = [bool_trace(g, sig, T) for g in f.phis]
        return [all(c[t] for c in cs) for t in range(T)]
    if isinstance(f, Or):
        cs = [bool_trace(g, sig, T) for g in f.phis]
        return [any(c[t] for c in cs) for t in range(T)]
    if isinstance(f, Implies):
        p, q = bool_trace(f.phi, sig, T), bool_trace(f.psi, sig, T)
        return [(not p[t]) or q[t] for t in range(T)]
    if isinstance(f, Always):
        c = bool_trace(f.phi, sig, T)
        return [all(c[k] for k in window(t, f.a, f.b)) for t in range(T)]
    if isinstance(f, Eventually):
        c = bool_trace(f.phi, sig, T)
        return [any(c[k] for k in window(t, f.a, f.b)) for t in range(T)]
    if isinstance(f, Until):
        p, q = bool_trace(f.phi, sig, T), bool_trace(f.psi, sig, T)
        return [any(q[tp] and all(p[k] for k in range(t, tp if f.strict else tp + 1)) for tp in window(t, f.a, f.b))
                for t in range(T)]
    raise TypeError(f)  # pragma: no cover


# --------------------------------------------------------------------------- random generators
def rand_pred(rng):
    return Pred(KEYS[rng.integers(2)], (">=", "<=")[rng.integers(2)], float(rng.integers(-3, 4)),
                scale=float(rng.choice([1.0, 0.5, 4.0])))


def rand_bounds(rng):
    a = int(rng.integers(0, 4))
    b = None if rng.random() < 0.3 else a + int(rng.integers(0, 7))
    return a, b


def rand_formula(rng, depth):
    if depth == 0 or rng.random() < 0.2:
        return rand_pred(rng)
    k = int(rng.integers(9))

    def sub():
        return rand_formula(rng, depth - 1)

    if k == 0:
        return Not(sub())
    if k == 1:
        return And(*[sub() for _ in range(rng.integers(1, 4))])
    if k == 2:
        return Or(*[sub() for _ in range(rng.integers(1, 4))])
    if k == 3:
        return Implies(sub(), sub())
    if k == 4:
        return Always(sub(), *rand_bounds(rng))
    if k == 5:
        return Eventually(sub(), *rand_bounds(rng))
    return Until(sub(), sub(), *rand_bounds(rng), strict=bool(rng.integers(2)))


def rand_signals(rng, T):
    return {k: rng.integers(-4, 5, size=T).astype(float) for k in KEYS}


def assert_matches_reference(f, sig):
    T = len(sig["x"])
    got = f.rho(sig)
    assert got.shape == (T,)
    np.testing.assert_array_equal(got, np.array(ref_trace(f, sig, T)), err_msg=repr(f))


# --------------------------------------------------------------------------- tests: reference agreement
OPERATORS = {
    "pred": lambda rng: rand_pred(rng),
    "not": lambda rng: Not(rand_formula(rng, 1)),
    "and": lambda rng: And(*[rand_formula(rng, 1) for _ in range(3)]),
    "or": lambda rng: Or(*[rand_formula(rng, 1) for _ in range(3)]),
    "implies": lambda rng: Implies(rand_formula(rng, 1), rand_formula(rng, 1)),
    "always": lambda rng: Always(rand_formula(rng, 1), *rand_bounds(rng)),
    "eventually": lambda rng: Eventually(rand_formula(rng, 1), *rand_bounds(rng)),
    "until": lambda rng: Until(rand_formula(rng, 1), rand_formula(rng, 1), *rand_bounds(rng)),
    "until_strict": lambda rng: Until(rand_formula(rng, 1), rand_formula(rng, 1), *rand_bounds(rng), strict=True),
}


@pytest.mark.parametrize("op", sorted(OPERATORS))
def test_each_operator_matches_bruteforce(op):
    rng = np.random.default_rng(zlib.crc32(op.encode()))
    for _ in range(150):
        f = OPERATORS[op](rng)
        assert_matches_reference(f, rand_signals(rng, int(rng.integers(1, 15))))


def test_random_nested_formulas_match_bruteforce_and_boolean_semantics():
    rng = np.random.default_rng(0)
    for _ in range(400):
        f = rand_formula(rng, 3)
        T = int(rng.integers(1, 15))
        sig = rand_signals(rng, T)
        assert_matches_reference(f, sig)
        r = f.rho(sig)
        b = np.array(bool_trace(f, sig, T))
        # soundness (Fainekos & Pappas): positive robustness => satisfied, negative => violated
        assert np.all(b[r > 0]) and not np.any(b[r < 0]), repr(f)
        t = int(rng.integers(T))
        assert f.robustness(sig, t) == r[t]


def test_long_signals_match_naive_windows():
    """Exercise the O(T) sliding min/max (many blocks) against a direct windowed computation."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    sig = {"x": x}
    for a, b in [(0, 0), (0, 1), (7, 300), (3, 4999), (0, 6000), (250, 251), (4999, 5005), (6000, None)]:
        hi = [len(x) - 1 if b is None else min(t + b, len(x) - 1) for t in range(len(x))]
        want_min = np.array([x[t + a: hi[t] + 1].min() if t + a <= hi[t] else INF for t in range(len(x))])
        want_max = np.array([x[t + a: hi[t] + 1].max() if t + a <= hi[t] else -INF for t in range(len(x))])
        np.testing.assert_array_equal(Always(Pred("x", ">=", 0.0), a, b).rho(sig), want_min)
        np.testing.assert_array_equal(Eventually(Pred("x", ">=", 0.0), a, b).rho(sig), want_max)
    # Until on a longer signal (7 levels of the doubling scan) against the cubic-time reference
    y = rng.normal(size=70)
    sig2 = {"x": x[:70], "y": y}
    for strict in (False, True):
        for a, b in [(0, None), (5, 40), (0, 69), (17, 17), (3, None)]:
            f = Until(Pred("x", ">=", -1.0), Pred("y", ">=", 1.0), a, b, strict=strict)
            np.testing.assert_array_equal(f.rho(sig2), np.array(ref_trace(f, sig2, 70)))


# --------------------------------------------------------------------------- tests: conventions
def test_dualities_hold_exactly_including_empty_windows():
    rng = np.random.default_rng(2)
    for _ in range(200):
        phi = rand_formula(rng, 2)
        a, b = rand_bounds(rng)
        sig = rand_signals(rng, int(rng.integers(1, 12)))
        np.testing.assert_array_equal(Always(phi, a, b).rho(sig), Not(Eventually(Not(phi), a, b)).rho(sig))
        np.testing.assert_array_equal(Implies(phi, phi & phi).rho(sig), Or(Not(phi), phi).rho(sig))


def test_empty_window_conventions():
    p, q = Pred("x", ">=", 0.0), Pred("y", ">=", 0.0)
    sig = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([5.0, 5.0, 5.0])}
    assert np.all(Always(p, 3, 5).rho(sig) == INF)             # vacuously true
    assert np.all(Eventually(p, 3, None).rho(sig) == -INF)     # unsatisfiable
    assert np.all(Until(p, q, 4, 9).rho(sig) == -INF)
    np.testing.assert_array_equal(Always(p, 1, 1).rho(sig), [2.0, 3.0, INF])      # truncated, then empty
    np.testing.assert_array_equal(Eventually(p, 2, 9).rho(sig), [3.0, -INF, -INF])
    assert Always(p, 0, None).horizon is None and Always(Eventually(p, 0, 3), 2, 5).horizon == 8
    assert Until(p, Always(q, 1, 2), 0, 4).horizon == 6 and Not(p).horizon == 0


def test_until_edge_cases():
    # psi holds at t but phi does not: the non-strict Until needs phi at t' = t too, the strict one does not
    sig = {"p": np.array([-1.0, -1.0, -1.0]), "q": np.array([2.0, -3.0, -3.0])}
    phi, psi = Pred("p", ">=", 0.0), Pred("q", ">=", 0.0)
    assert Until(phi, psi, 0, 2).robustness(sig) == -1.0
    assert Until(phi, psi, 0, 2, strict=True).robustness(sig) == 2.0

    # phi is required from t itself, also before the window [t+a, t+b] opens
    sig = {"p": np.array([1.0, -1.0, 1.0, 1.0, 1.0]), "q": np.array([-1.0, -1.0, -1.0, 3.0, -1.0])}
    assert Until(phi, psi, 2, 4).robustness(sig, 0) == -1.0         # phi fails at t=1 < t+a
    assert Until(phi, psi, 1, 3).robustness(sig, 2) == 1.0          # from t=2 phi holds and psi at 3
    assert Until(phi, psi, 2, 2).robustness(sig, 1) == -1.0         # exact time: psi(3)=3 but phi(1) < 0
    assert Until(phi, psi, 2, 2).robustness(sig, 0) == -1.0         # psi(2) < 0

    # truncation: b beyond the end equals b=None; a == b; windows past the end are -inf
    rng = np.random.default_rng(3)
    for _ in range(50):
        T = int(rng.integers(1, 10))
        s = {"p": rng.normal(size=T), "q": rng.normal(size=T)}
        for strict in (False, True):
            a = int(rng.integers(0, 4))
            np.testing.assert_array_equal(Until(phi, psi, a, T + 5, strict=strict).rho(s),
                                          Until(phi, psi, a, None, strict=strict).rho(s))
            r = Until(phi, psi, a, a, strict=strict).rho(s)
            for t in range(T):
                if t + a >= T:
                    assert r[t] == -INF
                else:
                    seg = s["p"][t: t + a + (0 if strict else 1)]
                    want = min(s["q"][t + a], seg.min() if seg.size else INF)
                    assert r[t] == want

    # phi trivially true => Until reduces to Eventually (both conventions)
    s = {"p": np.full(6, 100.0), "q": np.array([-2.0, 0.5, -1.0, 4.0, -3.0, 1.0])}
    for a, b in [(0, None), (1, 3), (2, 2), (5, 9)]:
        for strict in (False, True):
            np.testing.assert_array_equal(Until(phi, psi, a, b, strict=strict).rho(s), Eventually(psi, a, b).rho(s))


def test_signals_scalars_objects_and_validation():
    phi = Always(Pred("T", "<=", 10.0)) & Pred("mode", ">=", 1.0)
    T = np.array([1.0, 5.0, 9.0])
    assert phi.robustness({"T": T, "mode": 3}) == 1.0            # scalar = constant signal
    assert phi.robustness(SimpleNamespace(T=T, mode=1.5)) == 0.5  # attribute access
    assert (~phi).robustness({"T": T, "mode": 3}) == -1.0
    assert (Pred("T", ">=", 4.0) | Pred("T", ">=", 8.0)).rho({"T": T}).tolist() == [-3.0, 1.0, 5.0]
    with pytest.raises(ValueError):
        phi.rho({"T": T, "mode": np.array([1.0, 2.0])})          # different lengths
    with pytest.raises(ValueError):
        Pred("T", ">=", 0).rho({"T": np.zeros((2, 2))})
    with pytest.raises(ValueError):
        Pred("T", ">=", 0).rho({"T": np.array([])})
    with pytest.raises(IndexError):
        Pred("T", ">=", 0).robustness({"T": T}, 3)
    with pytest.raises(KeyError):
        Pred("missing", ">=", 0).rho({"T": T})
    for bad in (lambda: Pred("T", ">", 0), lambda: Pred("T", ">=", 0, scale=0), lambda: Pred("T", ">=", math.inf),
                lambda: Always(Pred("T", ">=", 0), 3, 1), lambda: Eventually(Pred("T", ">=", 0), -1), lambda: And()):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError):
        Always(Pred("T", ">=", 0), 1.5)
    with pytest.raises(TypeError):
        And(Pred("T", ">=", 0), "T >= 0")
    with pytest.raises(TypeError):
        STLConstraint("x", lambda o: 1.0)


def test_repr_is_exact_and_evaluable():
    rng = np.random.default_rng(4)
    ns = {k: getattr(stl, k) for k in stl.__all__}
    for _ in range(100):
        f = rand_formula(rng, 3)
        g = eval(repr(f), ns)  # noqa: S307 - test of the documented eval-able repr
        assert g == f and hash(g) == hash(f) and isinstance(g, Formula)
        sig = rand_signals(rng, 6)
        np.testing.assert_array_equal(g.rho(sig), f.rho(sig))
    assert str(Until(Pred("x", ">=", 1), Pred("y", "<=", 2, scale=4), 0, None, strict=True)) == \
        "(x >= 1 Us[0,end] (y <= 2)/4)"


# --------------------------------------------------------------------------- tests: Spec integration
def test_always_band_equals_spec_envelope():
    rng = np.random.default_rng(5)
    band = Always(And(Pred("T", "<=", 362), Pred("T", ">=", 340)), 0, None)
    band_scaled = Always(Pred("T", "<=", 362, scale=10) & Pred("T", ">=", 340, scale=10))
    env, env_scaled = Envelope("T", 340, 362), Envelope("T", 340, 362, scale=10)
    for _ in range(300):
        traj = rng.uniform(330, 372, size=int(rng.integers(1, 60)))
        out = {"T": traj}
        assert STLConstraint("band", band).margin(out) == env.margin(out)
        assert STLConstraint("band", band_scaled).margin(out) == env_scaled.margin(out)
        assert Always(Pred("T", "<=", 362, scale=3)).robustness(out) == Upper("T", 362, scale=3).margin(out)
        assert Always(Pred("T", ">=", 340, scale=3)).robustness(out) == Lower("T", 340, scale=3).margin(out)
    assert STLConstraint("band", band).margin({"T": 350.0}) == env.margin({"T": 350.0})   # scalar outcome


def test_stl_constraint_in_spec_fails_closed_and_fingerprints():
    band = Always(Pred("T", "<=", 362, scale=10) & Pred("T", ">=", 340, scale=10))
    settle = Eventually(Always(Pred("T", ">=", 348, scale=10) & Pred("T", "<=", 352, scale=10), 0, 3), 0, 10)
    spec = Spec(STLConstraint("band", band), STLConstraint("settles", settle), Upper("p95_ms", 800, scale=800))
    traj = np.concatenate([np.linspace(360, 350, 8), np.full(12, 350.0)])
    m, binding = spec.evaluate({"T": traj, "p95_ms": 200.0})
    assert binding == "band" and m == pytest.approx(0.2)
    m, binding = spec.evaluate({"T": np.full(20, 345.0), "p95_ms": 200.0})
    assert binding == "settles" and m == pytest.approx(-0.3)
    bad = traj.copy()
    bad[5] = np.nan
    assert STLConstraint("band", band).margin({"T": bad}) == -INF          # NaN fails closed
    assert spec.evaluate({"T": bad, "p95_ms": 200.0})[0] == -INF
    # deterministic fingerprints that track the formula exactly
    same = Always(Pred("T", "<=", 362.0, scale=10.0) & Pred("T", ">=", 340.0, scale=10.0))
    moved = Always(Pred("T", "<=", 362.0000001, scale=10.0) & Pred("T", ">=", 340.0, scale=10.0))
    fp = Spec(STLConstraint("band", band)).fingerprint()
    assert fp == Spec(STLConstraint("band", same)).fingerprint()
    assert fp != Spec(STLConstraint("band", moved)).fingerprint()
    assert STLConstraint("band", band).describe()["formula"] == repr(band)
