"""Shared evaluation harness for RiskGate on REAL, logged agent decisions from public benchmarks.

Every benchmark adapter produces a list of `Decision`s (one per consequential action proposed by a real agent, with
its ground-truth label) and, per method, a score for every candidate (higher = safer, -inf = never certified). The
harness then applies the protocol used throughout `benchmarks/`:

  * exchangeable units are GROUPS (a task, scenario or incident), never single steps: the calibration/test split is
    drawn by group and conformal risk control runs on each group's MEAN loss (riskgate.Gate.calibrate(groups=...));
    the guarantee reads  E[share of a new group's decisions that are certified and unsafe] <= alpha;
  * methods: "none" (execute everything), "fixed" (deterministic rule: certify iff score >= 0, no calibration),
    "crc" (expectation guarantee) and "ltt" (Learn-then-Test, P(risk <= alpha) >= 1 - delta');
  * metrics on held-out groups, validity over random group re-splits, an alpha sweep and cross-doer transfer.

Policy-mode losses with binary labels reduce to one breakpoint per decision: with the monotone envelope,
L_i(lambda) = 1{lambda <= b_i}, where b_i is the largest score of an unsafe candidate that would be the first
certified one for some lambda (see riskgate.conformal.policy_losses). This makes hundreds of re-splits cheap and is
checked against riskgate.Gate.calibrate on the fixed split by the adapters.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from riskgate.multirisk import ltt_threshold

INF = float("inf")
DELTA_LTT = 0.1


@dataclass
class Decision:
    """One consequential action proposed by a real agent (or a ranked list of candidates)."""

    group: str                       # exchangeable unit: task / scenario / incident id
    traj: str                        # run id (conversation / episode) for run-level metrics
    doer: str                        # agent identifier (model, prompt, scaffold)
    unsafe: list[bool]               # per candidate, rank order (first = the agent's choice)
    meta: dict = field(default_factory=dict)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def breakpoints(scores: Sequence[Sequence[float]], decisions: Sequence[Decision]) -> np.ndarray:
    """b_i: the policy-mode envelope loss is 1{lambda <= b_i} (-inf if no unsafe candidate is ever executed)."""
    out = np.full(len(decisions), -INF)
    for i, (s, d) in enumerate(zip(scores, decisions)):
        run = -INF
        for sj, uj in zip(s, d.unsafe):
            if sj > run:
                if uj:
                    out[i] = max(out[i], sj)
                run = sj
    return out


def _executed(s: Sequence[float], u: Sequence[bool], lam: float) -> tuple[bool, bool]:
    """(certified?, executed action unsafe?) for one decision at threshold lam (first candidate with s >= lam)."""
    for sj, uj in zip(s, u):
        if sj >= lam:
            return True, bool(uj)
    return False, False


def unit_weights(idx: Sequence[int], decisions: Sequence[Decision]) -> np.ndarray:
    """w_i = 1 / (n_groups * |group(i)|): group-mean losses as a weighted sum over decisions."""
    cnt: dict[str, int] = defaultdict(int)
    for i in idx:
        cnt[decisions[i].group] += 1
    g = len(cnt)
    return np.array([1.0 / (g * cnt[decisions[i].group]) for i in idx])


def calibrate_lambda(b: np.ndarray, all_scores: np.ndarray, gid: np.ndarray, alpha: float, method: str) -> float:
    """lambda-hat from calibration decisions: b = envelope breakpoints, all_scores = every finite candidate score
    (the same grid as riskgate.conformal.lambda_grid), gid = group index of each decision."""
    n_groups = int(gid.max()) + 1
    grid = np.concatenate([np.unique(all_scores[np.isfinite(all_scores)]), [INF]])
    if method == "crc":
        cnt_g = np.bincount(gid, minlength=n_groups).astype(float)
        w = 1.0 / (n_groups * cnt_g[gid])
        order = np.argsort(-b)
        bs, cw = b[order], np.cumsum(w[order])
        k = np.searchsorted(-bs, -grid, side="right")            # decisions with b_i >= lambda
        risk = np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)   # group-mean risk R(lambda)
        # identical to riskgate.conformal.crc_threshold on the (n_groups, n_grid) group-mean loss matrix
        ok = n_groups / (n_groups + 1) * risk + 1.0 / (n_groups + 1) <= alpha
        idx = np.flatnonzero(ok)
        return float(grid[idx[0]]) if idx.size else INF
    if method == "ltt":
        cnt_g = np.bincount(gid, minlength=n_groups).astype(float)
        mat = np.zeros((n_groups, grid.size))
        np.add.at(mat, gid, (b[:, None] >= grid[None, :]).astype(float) / cnt_g[gid][:, None])
        return float(ltt_threshold([mat], grid, [alpha], DELTA_LTT))
    raise ValueError(method)


def evaluate_at(lam: float, idx: Sequence[int], scores: Sequence[Sequence[float]], decisions: Sequence[Decision]) -> dict:
    """Metrics of the rule 'execute the first candidate with score >= lam' on the decisions `idx`."""
    n = len(idx)
    cert = np.zeros(n, bool)
    uns = np.zeros(n, bool)
    for k, i in enumerate(idx):
        cert[k], uns[k] = _executed(scores[i], decisions[i].unsafe, lam)
    w = unit_weights(idx, decisions)
    first_unsafe = np.array([decisions[i].unsafe[0] for i in idx])
    exists_safe = np.array([not all(decisions[i].unsafe) for i in idx])
    trajs: dict[str, bool] = defaultdict(bool)
    for k, i in enumerate(idx):
        trajs[decisions[i].traj] |= bool(uns[k])
    p = float(uns.mean()) if n else 0.0
    lo, hi = wilson(p, n)
    return {
        "n": n, "n_groups": len({decisions[i].group for i in idx}),
        "unsafe": float(np.sum(w * uns)),                  # group-mean (the guaranteed quantity)
        "unsafe_pooled": p, "unsafe_ci": [lo, hi],
        "ask": float(np.sum(w * ~cert)), "ask_pooled": float((~cert).mean()) if n else 0.0,
        "safe_auto": float(np.sum(w * (cert & ~uns))),
        "avoidable_ask": float(np.sum(w * (~cert & exists_safe))),
        "unsafe_runs": float(np.mean(list(trajs.values()))) if trajs else 0.0,
        "base_unsafe": float(np.sum(w * first_unsafe)),
    }


def split_groups(decisions: Sequence[Decision], frac: float, seed: int, subset: Sequence[int] | None = None):
    idx = list(range(len(decisions))) if subset is None else list(subset)
    groups = sorted({decisions[i].group for i in idx})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    k = int(round(frac * len(groups)))
    cal_g = {groups[j] for j in perm[:k]}
    return [i for i in idx if decisions[i].group in cal_g], [i for i in idx if decisions[i].group not in cal_g]


def lam_for(method: str, kind: str, cal_idx: Sequence[int], scores, decisions, alpha: float) -> float:
    if kind == "none":
        return -INF
    if kind == "fixed":
        return 0.0
    b = breakpoints([scores[i] for i in cal_idx], [decisions[i] for i in cal_idx])
    allsc = np.array([s for i in cal_idx for s in scores[i]], float)
    gk = {g: j for j, g in enumerate(sorted({decisions[i].group for i in cal_idx}))}
    gid = np.array([gk[decisions[i].group] for i in cal_idx], dtype=int)
    return calibrate_lambda(b, allsc, gid, alpha, kind)


def run_protocol(decisions: Sequence[Decision], methods: dict[str, tuple[str, Sequence[Sequence[float]]]], *,
                 alpha: float = 0.05, cal_frac: float = 0.5, seed: int = 0, n_resplits: int = 200,
                 alphas: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20),
                 transfer_methods: Sequence[str] = ("riskgate",), subset: Sequence[int] | None = None) -> dict:
    """methods: name -> (kind in {"none","fixed","crc","ltt"}, per-decision candidate scores)."""
    cal_idx, test_idx = split_groups(decisions, cal_frac, seed, subset)
    out: dict[str, Any] = {"alpha": alpha, "cal_frac": cal_frac, "n_decisions": len(cal_idx) + len(test_idx),
                           "n_groups": len({decisions[i].group for i in cal_idx + test_idx}),
                           "n_cal": len(cal_idx), "n_test": len(test_idx), "fixed_split": {}, "resplits": {},
                           "alpha_sweep": {}, "transfer": {}}
    for name, (kind, sc) in methods.items():
        lam = lam_for(name, kind, cal_idx, sc, decisions, alpha)
        r = evaluate_at(lam, test_idx, sc, decisions)
        r["lambda"] = lam
        out["fixed_split"][name] = r
    # validity over random group re-splits (the quantity CRC bounds in expectation)
    for name, (kind, sc) in methods.items():
        if kind not in ("crc", "ltt"):
            continue
        us, asks, safe = [], [], []
        for r in range(n_resplits):
            c_i, t_i = split_groups(decisions, cal_frac, 1000 + r, subset)
            lam = lam_for(name, kind, c_i, sc, decisions, alpha)
            m = evaluate_at(lam, t_i, sc, decisions)
            us.append(m["unsafe"])
            asks.append(m["ask"])
            safe.append(m["safe_auto"])
        out["resplits"][name] = {"n": n_resplits, "unsafe_mean": float(np.mean(us)), "unsafe_sd": float(np.std(us)),
                                 "ask_mean": float(np.mean(asks)), "safe_auto_mean": float(np.mean(safe)),
                                 "frac_above_alpha": float(np.mean(np.array(us) > alpha))}
    # alpha sweep (fixed split, CRC methods)
    for name, (kind, sc) in methods.items():
        if kind != "crc":
            continue
        out["alpha_sweep"][name] = []
        for a in alphas:
            lam = lam_for(name, kind, cal_idx, sc, decisions, a)
            m = evaluate_at(lam, test_idx, sc, decisions)
            out["alpha_sweep"][name].append({"alpha": a, "lambda": lam, "unsafe": m["unsafe"], "ask": m["ask"],
                                             "safe_auto": m["safe_auto"]})
    # cross-doer transfer: calibrate on doer A's calibration groups, test on doer B's test groups
    doers = sorted({decisions[i].doer for i in cal_idx + test_idx})
    if len(doers) > 1:
        for name in transfer_methods:
            if name not in methods or methods[name][0] != "crc":
                continue
            sc = methods[name][1]
            tab = {}
            for da in doers:
                ca = [i for i in cal_idx if decisions[i].doer == da]
                if len({decisions[i].group for i in ca}) < math.ceil(1 / alpha):
                    continue
                lam = lam_for(name, "crc", ca, sc, decisions, alpha)
                tab[da] = {}
                for db in doers:
                    tb = [i for i in test_idx if decisions[i].doer == db]
                    if tb:
                        m = evaluate_at(lam, tb, sc, decisions)
                        tab[da][db] = {"unsafe": m["unsafe"], "ask": m["ask"], "base_unsafe": m["base_unsafe"]}
            out["transfer"][name] = tab
    return out


def per_doer(decisions: Sequence[Decision], methods: dict, **kw) -> dict:
    """run_protocol separately for each doer (each agent gets its own calibration: policy mode)."""
    res = {}
    for d in sorted({x.doer for x in decisions}):
        sub = [i for i, x in enumerate(decisions) if x.doer == d]
        if len({decisions[i].group for i in sub}) < 2 * math.ceil(1 / kw.get("alpha", 0.05)):
            res[d] = {"skipped": "too few groups for this alpha"}
            continue
        res[d] = run_protocol(decisions, methods, subset=sub, transfer_methods=(), **kw)
    return res


def auroc(scores: Sequence[float], unsafe: Sequence[bool]) -> float:
    """P(score of a random safe decision > score of a random unsafe one); ties count 1/2."""
    s, u = np.asarray(scores, float), np.asarray(unsafe, bool)
    s = np.where(np.isfinite(s), s, np.where(s > 0, 1e300, -1e300))
    pos, neg = s[~u], s[u]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size)
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < allv.size:                                   # average ranks for ties
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def dump(obj: Any, path: Path) -> None:
    """JSON with +-inf as the strings "inf"/"-inf" and NaN (e.g. an undefined AUROC) as null."""
    def enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return clean(o.item())
        return str(o)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            o = o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None if o != o else ("inf" if o > 0 else "-inf")
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1, default=enc))
