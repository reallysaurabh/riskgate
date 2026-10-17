"""TEP experiment: calibrated fault-hypothesis sets and a small RiskGate decision layer (see tep.py for the design).

Run from the repository root:  .venv-public/bin/python benchmarks/tep/run.py

Two datasets, two exchangeability regimes:
  * Braatz (one simulation run per fault): windows of one run are strongly dependent, so the calibration/evaluation
    split is drawn over contiguous TIME BLOCKS (3 windows = 3 h of decisions) within each run, stratified by class.
    Calibration and evaluation windows of a class still come from the SAME run, so every coverage number on Braatz is
    optimistic for a new run of that fault (prominent caveat in the report).
  * Rieth et al. 2017 (500 runs per fault): the scorer is refitted on Rieth TRAINING runs only, and calibration /
    evaluation windows are split by RUN (exchangeable units), which is the study the Braatz data cannot support.

Protocol notes (benchmarks/README.md):
  * P3: window length, features, classifier, regularisation and temperature were fixed by time-block cross-validation
    on the Braatz training files (scratch/dev_cv.py, scratch/dev_temp.py) before any test file was scored, and reused
    unchanged for Rieth. The decision layer's plausible-set level delta = 0.05 and the effect table were fixed before
    the decision results were computed.
  * P9: agnostic and policy lambda-hats on the fixed split are asserted equal to Gate.calibrate(..., tau_fraction=0).
    Agnostic mode goes through the policy-mode harness by putting each window's highest-scoring unsafe candidate first
    (then the harness's losses are exactly the agnostic losses).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tep as P  # noqa: E402

import riskgate as rg  # noqa: E402
from riskgate.conformal import conformal_quantile  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
DELTAS = (0.05, 0.10, 0.20)
DELTA_GATE = 0.05
ALPHAS = (0.05, 0.10)
N_RESPLITS = 200
INF = float("inf")
TIME_BINS = ((2, 4), (4, 8), (8, 16), (16, 41))     # hours since onset (window end)


# ----------------------------------------------------------------------------------------------
# harness helpers (same logic as benchmarks/refund_arena/run.py)
# ----------------------------------------------------------------------------------------------
def agnostic_view(scores, labels, groups, trajs, doer="agnostic"):
    """Highest-scoring unsafe candidate first: the policy-mode harness then computes agnostic losses exactly."""
    decs, sc = [], []
    for s, u, g, tr in zip(scores, labels, groups, trajs):
        order = list(range(len(u)))
        bad = [j for j in order if u[j]]
        if bad:
            a = max(bad, key=lambda j: s[j])
            order = [a] + [j for j in order if j != a]
        decs.append(C.Decision(group=g, traj=tr, doer=doer, unsafe=[bool(u[j]) for j in order]))
        sc.append([float(s[j]) for j in order])
    return decs, sc


def library_lambda(gate, eps, groups, alpha, mode, guarantee) -> float:
    try:
        return float(gate.calibrate(eps, groups=groups, tau_fraction=0.0, mode=mode, alpha=alpha, guarantee=guarantee,
                                    delta_ltt=C.DELTA_LTT).lam)
    except ValueError as exc:
        if "risk-control groups" in str(exc):
            return INF
        raise


def p9_check(name, gate, eps, decs, scores, cal_idx, alpha, mode) -> dict:
    out = {}
    for kind, guarantee in (("crc", "expectation"), ("ltt", "high_probability")):
        lam_h = C.lam_for(name, kind, cal_idx, scores, decs, alpha)
        lam_l = library_lambda(gate, [eps[i] for i in cal_idx], [decs[i].group for i in cal_idx], alpha, mode, guarantee)
        assert lam_h == lam_l, (name, kind, alpha, lam_h, lam_l)
        out[kind] = lam_h
    return out


def ms(x) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(np.mean(x)), "sd": float(np.std(x))}


def add_uncalibrated_resplits(res: dict, decs, methods: dict, alpha: float, n_resplits: int, subset=None) -> None:
    """The harness re-splits only calibrated methods; evaluate 'none' and 'fixed' on the SAME test halves (seeds
    1000 + r) so that every row of a table is a mean over the same 200 re-splits."""
    for name, (kind, sc) in methods.items():
        if kind not in ("none", "fixed"):
            continue
        lam = -INF if kind == "none" else 0.0
        us, asks, safe = [], [], []
        for r in range(n_resplits):
            _, t_i = C.split_groups(decs, 0.5, 1000 + r, subset)
            m = C.evaluate_at(lam, t_i, sc, decs)
            us.append(m["unsafe"])
            asks.append(m["ask"])
            safe.append(m["safe_auto"])
        res["resplits"][name] = {"n": n_resplits, "unsafe_mean": float(np.mean(us)), "unsafe_sd": float(np.std(us)),
                                 "ask_mean": float(np.mean(asks)), "safe_auto_mean": float(np.mean(safe)),
                                 "frac_above_alpha": float(np.mean(np.array(us) > alpha)), "uncalibrated": True}


# ----------------------------------------------------------------------------------------------
# Part 1: calibrated fault-hypothesis sets
# ----------------------------------------------------------------------------------------------
def conformal_study(ws: P.WindowSet, groups: np.ndarray, n_splits: int = N_RESPLITS, seed: int = 0) -> dict:
    r = P.true_scores(ws)
    cls, classes = ws.cls, list(ws.classes)
    k_true = np.searchsorted(classes, cls)
    nC = len(classes)
    top1 = np.array(classes)[ws.nll.argmin(1)]
    hours = P.hours_since_onset(ws.t)
    others = tuple(c for c in classes if c not in P.HARD + P.EASY + (0,))
    fam = {"normal (IDV 0)": (0,), "hard (IDV 3, 9, 15)": P.HARD, "easy (IDV 1, 2, 4, 5, 6, 7)": P.EASY, "other faults": others}
    out: dict = {"n_windows": int(len(cls)), "n_groups": int(len(np.unique(groups))), "n_classes": nC,
                 "top1_acc": float(np.mean(top1 == cls)),
                 "top1_acc_by_family": {k: float(np.mean(top1[np.isin(cls, v)] == cls[np.isin(cls, v)])) for k, v in fam.items()},
                 "top1_acc_by_class": {int(c): float(np.mean(top1[cls == c] == c)) for c in classes}, "marginal": {}, "class_conditional": {}}
    rng = np.random.default_rng(seed)
    acc = {d: defaultdict(list) for d in DELTAS}
    accm = {d: defaultdict(list) for d in DELTAS}
    per = {d: np.zeros((nC, 3)) for d in DELTAS}                 # covered, size, n
    perm = {d: np.zeros((nC, 3)) for d in DELTAS}
    member = {d: np.zeros((nC, nC)) for d in DELTAS}            # member[true, k]: windows of `true` whose set contains k
    for _ in range(n_splits):
        cal = P.stratified_group_split(groups, cls, 0.5, rng)
        ev = ~cal
        for d in DELTAS:
            tau = conformal_quantile(r[cal], d)
            M = ws.nll[ev] <= tau
            cov, size = r[ev] <= tau, M.sum(1)
            taus = np.array([conformal_quantile(r[cal & (cls == c)], d) for c in classes])
            Mm = ws.nll[ev] <= taus[None, :]
            covm, sizem = Mm[np.arange(ev.sum()), k_true[ev]], Mm.sum(1)
            for dst, cv, sz in ((acc[d], cov, size), (accm[d], covm, sizem)):
                dst["coverage"].append(cv.mean())
                dst["size"].append(sz.mean())
                dst["empty"].append(np.mean(sz == 0))
                dst["singleton_correct"].append(np.mean((sz == 1) & cv))
                for name, grp in fam.items():
                    m = np.isin(cls[ev], grp)
                    dst[f"coverage|{name}"].append(cv[m].mean())
                    dst[f"size|{name}"].append(sz[m].mean())
                for lo, hi in TIME_BINS:
                    m = (hours[ev] >= lo) & (hours[ev] < hi) & (cls[ev] != 0)
                    dst[f"coverage|faults {lo}-{hi} h"].append(cv[m].mean())
                    dst[f"size|faults {lo}-{hi} h"].append(sz[m].mean())
            acc[d]["tau"].append(tau)
            for k in range(nC):
                m = k_true[ev] == k
                per[d][k] += [cov[m].sum(), size[m].sum(), m.sum()]
                perm[d][k] += [covm[m].sum(), sizem[m].sum(), m.sum()]
                member[d][k] += M[m].sum(0)
    for d in DELTAS:
        out["marginal"][f"delta={d:g}"] = {k: ms(v) for k, v in acc[d].items()}
        out["class_conditional"][f"delta={d:g}"] = {k: ms(v) for k, v in accm[d].items()}
        out["marginal"][f"delta={d:g}"]["by_class"] = {int(c): {"coverage": float(per[d][k, 0] / per[d][k, 2]),
                                                                "size": float(per[d][k, 1] / per[d][k, 2])} for k, c in enumerate(classes)}
        out["class_conditional"][f"delta={d:g}"]["by_class"] = {int(c): {"coverage": float(perm[d][k, 0] / perm[d][k, 2]),
                                                                         "size": float(perm[d][k, 1] / perm[d][k, 2])}
                                                                for k, c in enumerate(classes)}
        # what the plausible set of a hard / easy fault window typically contains (share of windows, top members)
        comp = {}
        for c in P.HARD + (1, 6, 0):
            if c not in classes:
                continue
            k = classes.index(c)
            share = member[d][k] / per[d][k, 2]
            top = np.argsort(-share)[:6]
            comp[int(c)] = {f"IDV{classes[j]}": round(float(share[j]), 3) for j in top if share[j] > 0.005}
        out["marginal"][f"delta={d:g}"]["set_members"] = comp
    return out


def temporal_split(ws: P.WindowSet) -> dict:
    """Braatz non-stationarity check: calibrate on the first half of each run's post-onset windows, evaluate on the
    second half, and vice versa (no randomness)."""
    r = P.true_scores(ws)
    first = ws.t <= np.median(ws.t)
    out = {}
    for name, cal in (("calibrate early, evaluate late", first), ("calibrate late, evaluate early", ~first)):
        row = {}
        for d in DELTAS:
            tau = conformal_quantile(r[cal], d)
            ev = ~cal
            row[f"delta={d:g}"] = {"coverage": float(np.mean(r[ev] <= tau)), "size": float(np.mean((ws.nll[ev] <= tau).sum(1)))}
        out[name] = row
    return out


# ----------------------------------------------------------------------------------------------
# Part 2: decision layer
# ----------------------------------------------------------------------------------------------
def decision_layer(ws: P.WindowSet, rise_q: dict, groups: np.ndarray, tag: str, n_resplits: int = N_RESPLITS) -> dict:
    r = P.true_scores(ws)
    tau_cal = P.stratified_group_split(groups, ws.cls, 0.3, np.random.default_rng(12345))
    tau = conformal_quantile(r[tau_cal], DELTA_GATE)
    idx = np.flatnonzero(~tau_cal & ws.has_future)
    classes = tuple(int(c) for c in ws.classes)
    model, prov, top = P.PlantModel(rise_q, tag), P.PlausibleFaults(tau, tag), P.TopFault(tag)
    ctxs = [P.PlantContext(f"{tag}-{i}", tuple(float(x) for x in ws.nll[i]), classes, float(ws.p_now[i])) for i in idx]
    lab = [[P.action_unsafe(a, ws.cls[i], ws.hold_unsafe[i]) for a in P.ACTIONS] for i in idx]
    grp = [str(groups[i]) for i in idx]
    trajs = [f"run{ws.run[i]}" for i in idx]
    out: dict = {"tau": tau, "delta": DELTA_GATE, "tau_coverage_heldout": float(np.mean(r[idx] <= tau)),
                 "plausible_size_heldout": float(np.mean((ws.nll[idx] <= tau).sum(1))), "n_decisions": len(idx),
                 "n_groups": len(set(grp)), "label_rates": {a: float(np.mean([x[j] for x in lab])) for j, a in enumerate(P.ACTIONS)},
                 "hold_unsafe_by_class": {int(c): float(np.mean(ws.hold_unsafe[idx][ws.cls[idx] == c])) for c in classes},
                 "rise_q95_kPa": {int(c): round(v, 1) for c, v in rise_q.items()}, "agnostic": {}, "policy": {}, "by_class": {}}
    cands = [P.ACT[a] for a in P.ACTIONS]
    a_eps = [rg.Episode(c, cands, {P.ACT[a].key: lab[n][j] for j, a in enumerate(P.ACTIONS)}) for n, c in enumerate(ctxs)]
    gates = {"riskgate": P.make_gate(model, prov, "agnostic"), "top1": P.make_gate(model, top, "agnostic")}
    raw = {k: [g.score_episode(ep).tolist() for ep in a_eps] for k, g in gates.items()}
    views = {k: agnostic_view(raw[k], lab, grp, trajs) for k in raw}
    out["auroc"] = {k: C.auroc([s for row in raw[k] for s in row], [u for row in lab for u in row]) for k in raw}
    for alpha in ALPHAS:
        blk = {}
        for k in gates:
            decs, sc = views[k]
            methods = {"none": ("none", sc), "fixed": ("fixed", sc), "crc": ("crc", sc), "ltt": ("ltt", sc)}
            res = C.run_protocol(decs, methods, alpha=alpha, n_resplits=n_resplits, transfer_methods=())
            add_uncalibrated_resplits(res, decs, methods, alpha, n_resplits)
            cal_idx, _ = C.split_groups(decs, 0.5, 0)
            res["p9_lambda"] = p9_check(k, gates[k], a_eps, decs, sc, cal_idx, alpha, "agnostic")
            blk[k] = res
        out["agnostic"][f"alpha={alpha:g}"] = blk
    out["examples"] = examples(gates["riskgate"], prov, views["riskgate"][0], a_eps, ctxs, lab, grp, ws, idx)

    # policy mode: the scripted operator assistant ranks the responses from its own top-1 diagnosis
    pg = {"riskgate": P.make_gate(model, prov, "policy"), "top1": P.make_gate(model, top, "policy")}
    p_eps, p_decs, p_raw = [], [], {k: [] for k in pg}
    for n, (i, c) in enumerate(zip(idx, ctxs)):
        top_cls = classes[int(np.argmin(ws.nll[i]))]
        rank = P.doer_playbook(top_cls)
        u = {a: lab[n][P.ACTIONS.index(a)] for a in P.ACTIONS}
        ep = rg.Episode(c, [P.ACT[a] for a in rank], {P.ACT[a].key: u[a] for a in rank})
        p_eps.append(ep)
        p_decs.append(C.Decision(group=grp[n], traj=trajs[n], doer="playbook (scripted)", unsafe=[u[a] for a in rank],
                                 meta={"cls": int(ws.cls[i]), "first": rank[0]}))
        for k, g in pg.items():
            p_raw[k].append(g.score_episode(ep).tolist())
    out["policy"]["base"] = {"first_choice_unsafe": float(np.mean([d.unsafe[0] for d in p_decs])),
                             "first_choice_counts": {a: int(sum(d.meta["first"] == a for d in p_decs)) for a in P.ACTIONS}}
    methods = {"none": ("none", p_raw["top1"]), "fixed": ("fixed", p_raw["top1"]), "top1_crc": ("crc", p_raw["top1"]),
               "riskgate": ("crc", p_raw["riskgate"]), "riskgate_ltt": ("ltt", p_raw["riskgate"])}
    for alpha in ALPHAS:
        res = C.run_protocol(p_decs, methods, alpha=alpha, n_resplits=n_resplits, transfer_methods=())
        add_uncalibrated_resplits(res, p_decs, methods, alpha, n_resplits)
        cal_idx, _ = C.split_groups(p_decs, 0.5, 0)
        res["p9_lambda"] = {k: p9_check(k, pg[k], p_eps, p_decs, p_raw[k], cal_idx, alpha, "policy") for k in pg}
        out["policy"][f"alpha={alpha:g}"] = res
    out["by_class"] = per_class_outcomes(views, raw, lab, ws, idx, alpha=0.10)
    return out


def examples(gate, prov, decs, eps, ctxs, lab, grp, ws, idx, alpha=0.10):
    """Worked examples: the agnostic gate calibrated on the fixed split's calibration groups, checked on mid-run test
    windows of a hard fault (IDV 3, 15), the pressure-limited IDV 6 and the sticking reactor valve IDV 14."""
    cal_idx, test_idx = C.split_groups(decs, 0.5, 0)
    gate.calibrate([eps[i] for i in cal_idx], groups=[grp[i] for i in cal_idx], tau_fraction=0.0, mode="agnostic", alpha=alpha)
    out = []
    for want in (3, 15, 6, 14):
        cand = [n for n in test_idx if ws.cls[idx[n]] == want]
        if not cand:
            continue
        n = cand[len(cand) // 2]
        i = idx[n]
        rows = []
        for j, a in enumerate(P.ACTIONS):
            d = gate.check(P.ACT[a], ctxs[n])
            rows.append({"action": a, "unsafe": bool(lab[n][j]), "verdict": d.verdict.value, "score": d.score,
                         "threshold": d.threshold, "reason": d.reason})
        out.append({"class": want, "hours_since_onset": float(P.hours_since_onset(ws.t[i])), "p_now_kPa": float(ws.p_now[i]),
                    "plausible": {h.name: round(h.score, 3) for h in prov.hypotheses(ctxs[n])}, "verdicts": rows})
    return out


def per_class_outcomes(views, raw, lab, ws, idx, alpha, n_resplits=100):
    """Agnostic outcomes per true class at alpha, averaged over group re-splits (test halves only)."""
    out = {}
    for k in ("riskgate", "top1"):
        decs, sc = views[k]
        acc = defaultdict(lambda: np.zeros(5))     # n, unsafe certified, safe certified (some), ask, hold certified
        for rr in range(n_resplits):
            c_i, t_i = C.split_groups(decs, 0.5, 1000 + rr)
            lam = C.lam_for(k, "crc", c_i, sc, decs, alpha)
            for n in t_i:
                s, u = np.array(raw[k][n]), np.array(lab[n], bool)
                cert = s >= lam
                acc[int(ws.cls[idx[n]])] += [1, bool(np.any(cert & u)), bool(np.any(cert & ~u)), not bool(np.any(cert)), bool(cert[0])]
        out[k] = {c: {"unsafe_certified": float(v[1] / v[0]), "safe_certified": float(v[2] / v[0]), "ask": float(v[3] / v[0]),
                      "hold_certified": float(v[4] / v[0])} for c, v in sorted(acc.items())}
    return out


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    res: dict = {"meta": {
        "braatz": "Braatz TEP data (one run per fault); repository licence covers software and documentation, data licence "
                  "unclear: research use only, not redistributed",
        "rieth": "Rieth, Amsel, Tran, Cook (2017), Harvard Dataverse doi:10.7910/DVN/6C3JR1, public-domain dedication",
        "design": {"window_samples": P.W, "decision_stride": P.STRIDE_TEST, "features": "mean + log sd of 52 standardised variables",
                   "classifier": f"RDA gamma={P.GAMMA}, temperature={P.TEMP}", "horizon_samples": P.HORIZON,
                   "pressure_limit_kPa": P.P_LIMIT, "unsafe_active": {k: sorted(v) for k, v in P.UNSAFE_ACTIVE.items()},
                   "rule_notes": P.RULE_NOTES, "delta_gate": DELTA_GATE}}}
    # ---------------- Braatz
    train, test = P.load_braatz()
    res["braatz_onset_check"] = P.onset_check_braatz()
    sc_b = P.FaultScorer().fit(train, {c: 0 for c in train})
    ws_b = P.make_windows(sc_b, test, block_len=3)
    res["braatz"] = {"hypothesis_sets": conformal_study(ws_b, ws_b.block), "temporal_split": temporal_split(ws_b),
                     "decision": decision_layer(ws_b, P.pressure_rise_q(train, {c: 0 for c in train}), ws_b.block, "braatz")}
    print(f"braatz done {time.time() - t0:.0f}s", flush=True)
    # ---------------- Rieth: scorer on training runs 1-50; hypothesis sets on testing runs 1-100; decisions on runs 101-130
    tr_r = P.load_rieth("train")
    te_r = P.load_rieth("test")
    res["rieth_onset_check"] = rieth_onset_check(tr_r)
    onset_r = {c: (0 if c == 0 else 20) for c in tr_r}
    sc_r = P.FaultScorer().fit(tr_r, onset_r)
    res["rieth_counts"] = {"classes": sorted(tr_r), "train_runs_per_class": len(tr_r[1]), "test_runs_cached_per_class": len(te_r[1])}
    ws_r = P.make_windows(sc_r, {c: v[:100] for c, v in te_r.items()})
    res["rieth"] = {"hypothesis_sets": conformal_study(ws_r, ws_r.run)}
    print(f"rieth sets done {time.time() - t0:.0f}s", flush=True)
    ws_rd = P.make_windows(sc_r, {c: v[100:130] for c, v in te_r.items()})
    res["rieth"]["decision"] = decision_layer(ws_rd, P.pressure_rise_q(tr_r, onset_r), ws_rd.run, "rieth")
    # transfer of the Braatz-fitted scorer to Rieth testing runs (does a scorer fitted on one run per fault generalise?)
    ws_x = P.make_windows(sc_b, {c: v[:100] for c, v in te_r.items() if c in sc_b.classes})
    top_x = np.array(ws_x.classes)[ws_x.nll.argmin(1)]
    res["rieth"]["braatz_scorer_top1_acc"] = float(np.mean(top_x == ws_x.cls))
    print(f"rieth decisions done {time.time() - t0:.0f}s", flush=True)
    res["paper_summary"] = paper_summary(res)
    return res


def rieth_onset_check(tr: dict) -> dict:
    """Mean |z| of samples 1-20 vs 21-40 of Rieth faulty training runs (onset documented after sample 20)."""
    normal = np.vstack(tr[0])
    mu, sd = normal.mean(0), normal.std(0, ddof=1)
    sd[sd < 1e-9] = 1.0
    out = {"normal": float(np.mean([np.abs((r - mu) / sd).mean() for r in tr[0][:20]]))}
    for c in (1, 2, 6, 7):
        z = np.mean([np.abs((r - mu) / sd).mean(1) for r in tr[c][:20]], axis=0)
        out[f"IDV{c}"] = [round(float(z[:20].mean()), 2), round(float(z[20:40].mean()), 2), round(float(z[40:].mean()), 2)]
    return out


def rieth_step_check() -> dict:
    """Post-hoc data check behind the IDV 4 / 5 confusions: mean reactor (XMV 10) and condenser (XMV 11) cooling-water
    flows after onset in 50 Rieth testing runs of IDV 0, 4 and 5 (the steps are clearly present in the data)."""
    te = P.load_rieth("test")
    out = {}
    for c in (0, 4, 5):
        runs = te[c][:50]
        out[f"IDV{c}"] = {"XMV10_mean": round(float(np.mean([r[P.ONSET_TEST:, 50].mean() for r in runs])), 2),
                          "XMV11_mean": round(float(np.mean([r[P.ONSET_TEST:, 51].mean() for r in runs])), 2)}
    normal = np.vstack([r[P.ONSET_TEST:] for r in te[0][:50]])
    out["sample_sd_normal"] = {"XMV10": round(float(normal[:, 50].std()), 2), "XMV11": round(float(normal[:, 51].std()), 2)}
    return out


def paper_summary(res: dict) -> dict:
    hb = res["braatz"]["hypothesis_sets"]
    hr = res["rieth"]["hypothesis_sets"]
    pct = lambda x: f"{100 * x:.1f}\\%"
    mr = hr["marginal"]["delta=0.1"]
    dr = res["rieth"]["decision"]["agnostic"]["alpha=0.1"]
    rg_rs, t1_rs = dr["riskgate"]["resplits"]["crc"], dr["top1"]["resplits"]["crc"]
    text = (f"Tennessee Eastman (simulated): an LLM-free RDA fault scorer has top-1 accuracy {pct(hb['top1_acc'])} (Braatz, one run per "
            f"fault) and {pct(hr['top1_acc'])} (Rieth, runs as units); split-conformal plausible sets at $\\delta=0.1$ cover "
            f"{pct(mr['coverage']['mean'])} of held-out Rieth windows with mean size {mr['size']['mean']:.2f}, growing to "
            f"{mr['size|hard (IDV 3, 9, 15)']['mean']:.1f} classes for the undetectable IDV 3/9/15 (covered "
            f"{pct(mr['coverage|hard (IDV 3, 9, 15)']['mean'])}) versus {mr['size|easy (IDV 1, 2, 4, 5, 6, 7)']['mean']:.1f} for easy faults; "
            f"in the partly rule-derived decision layer, agnostic CRC at $\\alpha=0.1$ certifies an unsafe response in "
            f"{pct(rg_rs['unsafe_mean'])} of decisions with RiskGate versus {pct(t1_rs['unsafe_mean'])} with a top-1 twin "
            f"(mean over 200 run re-splits), at {pct(rg_rs['ask_mean'])} versus {pct(t1_rs['ask_mean'])} asks.")
    return {"n_decisions": f"Braatz {hb['n_windows']} windows; Rieth {hr['n_windows']} windows (sets) and "
                           f"{res['rieth']['decision']['n_decisions']} decisions",
            "text": text, "braatz_coverage_delta0.1": hb["marginal"]["delta=0.1"]["coverage"]["mean"]}


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        out = main()
        out["rieth_step_check"] = rieth_step_check()
        C.dump(out, OUT / "tep_results.json")
        print("results written", flush=True)
    import json

    from report import write_report  # noqa: E402

    res = json.loads((OUT / "tep_results.json").read_text())
    if "--report-only" in sys.argv:                    # refresh derived fields without recomputing the experiment
        res.setdefault("rieth_step_check", rieth_step_check())
        res["paper_summary"] = paper_summary(res)
        C.dump(res, OUT / "tep_results.json")
    write_report(res, OUT / "tep_report.md")
    print("report written")
