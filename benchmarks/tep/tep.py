"""Tennessee Eastman process (TEP): calibrated fault-hypothesis sets ("humility") and a small RiskGate decision layer.

Data (cached in ../public_data, never redistributed)
  * Braatz TEP data (tep_braatz/): one closed-loop simulation run per fault (IDV 1-21) plus normal operation (IDV 0).
    Training files dXX.dat (480 x 52; d00.dat is stored TRANSPOSED, 52 x 500), test files dXX_te.dat (960 x 52, the
    fault enters after sample 160 = 8 h, 3-min sampling). The repository licence covers "software and documentation";
    the data licence is unclear, so they are used for research only. Observed fact: the faulty TRAINING files deviate
    from normal operation from (almost) the first sample (e.g. d07 from sample 1), i.e. the training onset is at or
    near sample 1, not after sample 20; samples 1-20 of each faulty training file are dropped as an ambiguous
    transient.
  * Rieth, Amsel, Tran & Cook (2017), "Additional Tennessee Eastman process simulation data for anomaly detection
    evaluation", Harvard Dataverse doi:10.7910/DVN/6C3JR1 (public-domain dedication): 500 independent runs per
    fault (IDV 1-20) and fault-free. Training runs: 500 samples, onset after sample 20; testing runs: 960 samples,
    onset after sample 160. Runs are the exchangeable units (rule P2), which the Braatz data cannot provide.

Hypothesis/humility component (LLM-free)
  a window of the last W = 40 samples (2 h) is summarised by the mean and log standard deviation of each of the 52
  standardised variables (104 features); a regularised Gaussian classifier (RDA: class covariance (1-g) S_k + g S_pooled,
  g = 0.5, equal priors) fitted on training data only gives a posterior over the fault classes with temperature T = 16.
  The nonconformity of class h is r(h) = -log p(h | window); the plausible set is {h : r(h) <= tau} with tau the
  split-conformal (1 - delta) quantile of r(true class) on calibration windows. W, g, T and the features were chosen
  by time-block cross-validation on the Braatz TRAINING files only (scratch/dev_cv.py, scratch/dev_temp.py).

Decision layer (small and partly rule-derived)
  candidate operator responses to an upset: hold (keep set-points, base-layer control continues), rcw_up (raise the
  reactor cooling-water flow set-point), feed_cut (reduce the A/C feed, stream 4); escalate is the gate's ASK.
  * hold: label DATA-derived from the recorded run (which is the no-intervention outcome): unsafe iff the reactor
    pressure exceeds the Downs & Vogel (1993) normal operating limit of 2895 kPa within the next hour. The gate's
    consequence model predicts it from TRAINING runs only: p_now + q95 of the 1-h pressure rise observed under the
    hypothesised class.
  * rcw_up, feed_cut: RULE-derived effect table from the IDV definitions (UNSAFE_ACTIVE, justifications below);
    labels and the gate's model use the same table, so only the fault class is uncertain for these actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import ledoit_wolf

import riskgate as rg

PUBLIC = Path(__file__).resolve().parents[3] / "public_data"
BRAATZ, RIETH = PUBLIC / "tep_braatz", PUBLIC / "tep_rieth"
CACHE = Path(__file__).resolve().parent / "cache"
W, STRIDE_TEST, STRIDE_TRAIN = 40, 20, 5          # window (2 h), decision every hour, training-window stride
ONSET_TEST = 160                                    # test runs: fault active from sample 161 (index 160)
HORIZON = 20                                        # hold-safety horizon: 1 h
P_LIMIT = 2895.0                                    # reactor pressure normal operating limit, kPa (shutdown: 3000)
P_COL = 6                                           # XMEAS(7) reactor pressure
GAMMA, TEMP = 0.5, 16.0
HARD, EASY = (3, 9, 15), (1, 2, 4, 5, 6, 7)
ACTIONS = ("hold", "rcw_up", "feed_cut")
UNSAFE_ACTIVE = {
    # rule-derived (IDV definitions): an active response is unsafe under these classes
    "rcw_up": {6, 14, 16, 17, 18, 19, 20},
    "feed_cut": {6, 7, 16, 17, 18, 19, 20, 21},
}
RULE_NOTES = {
    6: "A feed loss: reactor pressure runs into its limit in the recorded runs; extra cooling slows the gas-consuming "
       "reactions (pressure up) and cutting stream 4 removes the remaining A supply",
    7: "C header pressure loss: stream-4 supply is already reduced; cutting it further starves the reactor",
    14: "reactor cooling-water valve sticking: moving its set-point provokes stick-slip jumps in cooling",
    21: "stream-4 valve fixed: a feed cut cannot be executed, the plant would be believed mitigated when it is not",
    16: "IDV 16-20 are of unknown type: no validated response exists, so any active intervention counts as unsafe",
}


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def _braatz(name: str) -> np.ndarray:
    a = np.loadtxt(BRAATZ / name)
    return a.T if a.shape[0] == 52 else a               # d00.dat is stored transposed (52 x 500)


def load_braatz() -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    """train[c] / test[c]: lists of runs (one per class here). Faulty training runs keep samples 21-480 only."""
    train = {c: [_braatz(f"d{c:02d}.dat")[(0 if c == 0 else 20):]] for c in range(22)}
    test = {c: [_braatz(f"d{c:02d}_te.dat")] for c in range(22)}
    return train, test


def onset_check_braatz() -> dict:
    """Mean |z| (vs d00) of samples 1-20, 21-40 and 41-480 of the faulty training files: is sample 1-20 fault-free?"""
    d00 = _braatz("d00.dat")
    mu, sd = d00.mean(0), d00.std(0, ddof=1)
    out = {"normal_d00": float(np.abs((d00 - mu) / sd).mean())}
    for c in range(1, 22):
        z = np.abs((_braatz(f"d{c:02d}.dat") - mu) / sd).mean(1)
        out[f"IDV{c}"] = [round(float(z[:20].mean()), 2), round(float(z[20:40].mean()), 2), round(float(z[40:].mean()), 2),
                          round(float(z[0]), 2)]
    return out


def build_rieth_cache(r_train: int = 50, r_test: int = 200) -> None:
    """Convert the four Rieth RData files to per-class float32 arrays (runs x samples x 52), one file at a time."""
    import pyreadr  # only needed once, to build the cache

    CACHE.mkdir(exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")
    for fname, kind, n_runs in (("TEP_FaultFree_Training.RData", "train", r_train), ("TEP_Faulty_Training.RData", "train", r_train),
                                ("TEP_FaultFree_Testing.RData", "test", r_test), ("TEP_Faulty_Testing.RData", "test", r_test)):
        df = next(iter(pyreadr.read_r(str(RIETH / fname)).values()))
        cols = list(df.columns)
        assert cols[:3] == ["faultNumber", "simulationRun", "sample"] and cols[3] == "xmeas_1" and cols[-1] == "xmv_11", cols[:5]
        fn = df["faultNumber"].to_numpy().astype(int)
        run = df["simulationRun"].to_numpy().astype(int)
        smp = df["sample"].to_numpy().astype(int)
        vals = df.iloc[:, 3:].to_numpy(dtype=np.float32)
        del df
        n_s = int(smp.max())
        for c in np.unique(fn):
            sel = np.flatnonzero((fn == c) & (run <= n_runs))
            arr = np.full((n_runs, n_s, 52), np.nan, np.float32)
            arr[run[sel] - 1, smp[sel] - 1] = vals[sel]
            complete = ~np.isnan(arr).any(axis=(1, 2))
            np.save(CACHE / f"rieth_{kind}_c{int(c):02d}.npy", arr[complete])
        del vals, fn, run, smp


def load_rieth(kind: str) -> dict[int, list[np.ndarray]]:
    if not any(CACHE.glob(f"rieth_{kind}_c*.npy")):
        build_rieth_cache()
    out = {}
    for p in sorted(CACHE.glob(f"rieth_{kind}_c*.npy")):
        arr = np.load(p).astype(float)
        out[int(p.stem[-2:])] = list(arr)
    return out


# ----------------------------------------------------------------------------------------------
# Features and the fault-hypothesis scorer
# ----------------------------------------------------------------------------------------------
def window_features(z: np.ndarray, ends) -> np.ndarray:
    """Mean and log standard deviation of each standardised variable over the W samples ending (exclusive) at e."""
    out = np.empty((len(ends), 2 * z.shape[1]))
    for i, e in enumerate(ends):
        x = z[e - W:e]
        out[i, : z.shape[1]] = x.mean(0)
        out[i, z.shape[1]:] = np.log(x.std(0, ddof=1) + 1e-3)
    return out


class RDA:
    """Regularised Gaussian discriminant: class covariance (1 - g) S_k + g S_pooled (Ledoit-Wolf pooled), equal priors."""

    def __init__(self, g: float = GAMMA):
        self.g = g

    def fit(self, X: np.ndarray, y: np.ndarray) -> RDA:
        self.classes_ = np.unique(y)
        self.mu = np.array([X[y == c].mean(0) for c in self.classes_])
        Sp, _ = ledoit_wolf(X - self.mu[np.searchsorted(self.classes_, y)])
        self.P, self.ld = [], []
        for k, c in enumerate(self.classes_):
            R = X[y == c] - self.mu[k]
            S = (1 - self.g) * (R.T @ R / max(len(R) - 1, 1)) + self.g * Sp
            S += 1e-6 * np.trace(S) / len(S) * np.eye(len(S))
            self.P.append(np.linalg.inv(S))
            self.ld.append(np.linalg.slogdet(S)[1])
        return self

    def loglik(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            d = X - self.mu[k]
            out[:, k] = -0.5 * (np.einsum("ij,jk,ik->i", d, self.P[k], d) + self.ld[k])
        return out


class FaultScorer:
    """Fit on training runs only; nll(X)[i, k] = -log p(class k | window i) with temperature TEMP."""

    def __init__(self, gamma: float = GAMMA, temp: float = TEMP):
        self.gamma, self.temp = gamma, temp

    def fit(self, train: dict[int, list[np.ndarray]], onset: dict[int, int]) -> FaultScorer:
        normal = np.vstack(train[0])
        self.mu, self.sd = normal.mean(0), normal.std(0, ddof=1)
        self.sd[self.sd < 1e-9] = 1.0
        Xs, ys = [], []
        for c, runs in train.items():
            for run in runs:
                ends = np.arange(onset[c] + W, len(run) + 1, STRIDE_TRAIN)
                Xs.append(window_features((run - self.mu) / self.sd, ends))
                ys.append(np.full(len(ends), c))
        self.model = RDA(self.gamma).fit(np.vstack(Xs), np.concatenate(ys))
        self.classes = [int(c) for c in self.model.classes_]
        return self

    def features(self, run: np.ndarray, ends) -> np.ndarray:
        return window_features((run - self.mu) / self.sd, ends)

    def nll(self, X: np.ndarray) -> np.ndarray:
        L = self.model.loglik(X) / self.temp
        return -(L - logsumexp(L, axis=1, keepdims=True))


@dataclass
class WindowSet:
    """Test windows: per window the true class, run id, end index t (exclusive), time block, NLL vector, p_now."""

    cls: np.ndarray
    run: np.ndarray
    t: np.ndarray
    block: np.ndarray
    nll: np.ndarray
    p_now: np.ndarray
    hold_unsafe: np.ndarray          # recorded pressure > P_LIMIT within the next HORIZON samples (nan-free: last window dropped)
    has_future: np.ndarray
    classes: list[int] = field(default_factory=list)


def make_windows(scorer: FaultScorer, test: dict[int, list[np.ndarray]], block_len: int = 3) -> WindowSet:
    """Windows ending at 200, 220, ..., 960 (post-onset only), for every test run of every class."""
    ends = np.arange(ONSET_TEST + W, 961, STRIDE_TEST)
    rows: dict[str, list] = {k: [] for k in ("cls", "run", "t", "block", "p_now", "hold_unsafe", "has_future")}
    feats = []
    rid = 0
    for c in sorted(test):
        for run in test[c]:
            feats.append(scorer.features(run, ends))
            p = run[:, P_COL]
            for j, e in enumerate(ends):
                fut = p[e:e + HORIZON]
                rows["cls"].append(c)
                rows["run"].append(rid)
                rows["t"].append(e)
                rows["block"].append(f"r{rid}-b{j // block_len}")
                rows["p_now"].append(p[e - 1])
                rows["hold_unsafe"].append(bool(fut.size and fut.max() > P_LIMIT))
                rows["has_future"].append(bool(fut.size))
            rid += 1
    X = np.vstack(feats)
    return WindowSet(np.array(rows["cls"]), np.array(rows["run"]), np.array(rows["t"]), np.array(rows["block"]), scorer.nll(X),
                     np.array(rows["p_now"]), np.array(rows["hold_unsafe"]), np.array(rows["has_future"]), scorer.classes)


# ----------------------------------------------------------------------------------------------
# Conformal plausible sets
# ----------------------------------------------------------------------------------------------
def true_scores(ws: WindowSet) -> np.ndarray:
    idx = np.searchsorted(ws.classes, ws.cls)
    return ws.nll[np.arange(len(ws.cls)), idx]


def stratified_group_split(groups: np.ndarray, strata: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of calibration windows: within every stratum (class), a `frac` share of its groups."""
    cal = np.zeros(len(groups), bool)
    for s in np.unique(strata):
        g = np.unique(groups[strata == s])
        pick = set(rng.permutation(g)[: int(round(frac * len(g)))])
        cal |= (strata == s) & np.isin(groups, list(pick))
    return cal


def plausible_mask(nll: np.ndarray, tau) -> np.ndarray:
    """Membership matrix; tau is a scalar (marginal conformal) or one threshold per class (class-conditional)."""
    t = np.asarray(tau, float)
    return nll <= (t if t.ndim == 0 else t[None, :])


# ----------------------------------------------------------------------------------------------
# Decision layer
# ----------------------------------------------------------------------------------------------
def pressure_rise_q(train: dict[int, list[np.ndarray]], onset: dict[int, int], q: float = 0.95) -> dict[int, float]:
    """Per class: q-quantile over training runs and times of max(P[t:t+H]) - P[t-1] (the 1-h pressure rise)."""
    out = {}
    for c, runs in train.items():
        rises = []
        for run in runs:
            p = run[:, P_COL]
            for t in range(onset[c] + 1, len(p) - HORIZON + 1):
                rises.append(p[t:t + HORIZON].max() - p[t - 1])
        out[c] = float(np.quantile(rises, q))
    return out


def action_unsafe(action: str, cls: int, hold_unsafe: bool) -> bool:
    if action == "hold":
        return bool(hold_unsafe)
    return int(cls) in UNSAFE_ACTIVE[action]


@dataclass(frozen=True)
class PlantContext:
    """What the gate sees: the scorer's NLL per class for the current window and the current reactor pressure."""

    key: str
    nll: tuple
    classes: tuple
    p_now: float


class PlausibleFaults:
    """Hypothesis provider: fault classes with NLL <= tau (tau frozen in, fitted by split conformal on a tau split)."""

    def __init__(self, tau: float, tag: str):
        self.tau = float(tau)
        self.version = f"tep-rda-{tag}|tau={self.tau:.6g}"

    def hypotheses(self, ctx: PlantContext) -> list[rg.Hypothesis]:
        return [rg.Hypothesis(f"IDV{c}", family=str(c), params={"cls": c}, score=float(s))
                for c, s in zip(ctx.classes, ctx.nll) if s <= self.tau]


class TopFault:
    """Baseline: the single most likely fault (a 'twin' that trusts its diagnosis)."""

    def __init__(self, tag: str):
        self.version = f"tep-rda-{tag}|top1"

    def hypotheses(self, ctx: PlantContext) -> list[rg.Hypothesis]:
        k = int(np.argmin(ctx.nll))
        return [rg.Hypothesis(f"IDV{ctx.classes[k]}", family=str(ctx.classes[k]), params={"cls": ctx.classes[k]}, score=float(ctx.nll[k]))]


class PlantModel:
    """Consequence model. hold: predicted 1-h peak pressure p_now + rise_q[h] from TRAINING runs vs the 2895 kPa limit
    (margin in 100 kPa); active responses: +1 / -1 from the rule-derived effect table."""

    def __init__(self, rise_q: dict[int, float], tag: str):
        self.rise_q = dict(rise_q)
        self.version = f"tep-plant-{tag}-v1"

    def predict(self, action: rg.Action, h: rg.Hypothesis, ctx: PlantContext) -> dict:
        c = int(h.params["cls"])
        if action.name == "hold":                      # rounded to 1 kPa: keeps the lambda grid small, no effect on decisions
            return {"margin": round((P_LIMIT - (ctx.p_now + self.rise_q.get(c, 0.0))) / 100.0, 2)}
        return {"margin": -1.0 if c in UNSAFE_ACTIVE[action.name] else 1.0}


SPEC = rg.Spec(rg.Custom("pressure limit / effect table", lambda o: o["margin"]))
ACT = {a: rg.Action(a) for a in ACTIONS}


def doer_playbook(top: int) -> list[str]:
    """SCRIPTED operator assistant: the playbook response for its own top-1 diagnosis first, then the others."""
    if top in (4, 11, 14):                   # reactor cooling problems -> more reactor cooling
        first = "rcw_up"
    elif top in (1, 2, 6, 7, 8, 12, 13, 18):  # feed / pressure problems -> cut the A/C feed
        first = "feed_cut"
    else:
        first = "hold"
    return [first] + [a for a in ACTIONS if a != first]


def make_gate(model: PlantModel, provider, mode: str, alpha: float = 0.1) -> rg.Gate:
    return rg.Gate(model, SPEC, provider, alpha=alpha, mode=mode)


def hours_since_onset(t: np.ndarray) -> np.ndarray:
    return (np.asarray(t) - ONSET_TEST) * 3.0 / 60.0
