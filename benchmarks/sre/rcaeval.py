"""RCAEval RE1-OB: REAL fault-injection telemetry from Google's Online Boutique, used for (1) the hypothesis-set
component of RiskGate on real data and (2) a rule-labelled remediation gate with a scripted doer.

Data: `../public_data/rcaeval-hf/` (HF mirror of RCAEval, MIT; original Zenodo record 14590730, CC BY 4.0). RE1-OB has
125 cases = 5 root-cause services x 5 fault types (cpu, mem, disk, delay, loss) x 5 repetitions; each case has
per-second metrics (`<service>_<metric>` columns) and the injection time. Two data quirks are handled explicitly:
  * re1ob_productcatalogservice_cpu_3 has 63 s of telemetry that END before the injection (no faulty period): excluded;
  * re1ob_currencyservice_loss_1 has a truncated inject_time (16933142, 8 digits); every other loss case injects 360 s
    after the start of its 721 s window and the truncated value is a prefix of that time, so it is repaired to
    time_start + 360 (this case is in the development subset only).

What is REAL, what is RULE-DERIVED, what is SCRIPTED:
  * real: the telemetry, the injection times and the ground-truth (service, fault) of every case. The hypothesis-set
    results (top-1 accuracy, coverage and size of conformal plausible sets) are entirely real-data results;
  * rule-derived: the remediation labels. An action-effect table written from SRE practice (EFFECT below) says which
    action mitigates which fault class; applied to the TRUE cause it gives the label, applied to each plausible
    hypothesis it is the gate's consequence model. No remediation was ever executed on the testbed: labels are NOT
    observed outcomes, and the gate's consequence model is exactly correct by construction (the realistic error source
    left is the diagnosis);
  * scripted: the doer, an LLM-like dashboard reader (largest relative change -> "most common fix"), plus a
    restart-first variant for the transfer check. No LLM is called.

Hypotheses (LLM-free scorer, BARO/N-sigma style): for 11 services x 5 fault types = 55 hypotheses (all services with
metrics, not only the 5 the benchmark injects into: the scorer must not know the benchmark design), robust z-scores of
the median over 300 s after vs before the injection, for CPU, memory, p50 latency and load (the only metrics present in
both RE1 schema variants: cpu/mem cases have `_latency`/`_load`, the others `_latency-50`/`_workload` and an extra p90
that we drop so the fault type cannot be read off the schema), then a hand-set signature per fault type. Nonconformity
r(h) = E_max - E(h). The signature weights were set on the DEVELOPMENT subset (repetition 1 of every family, 25 cases)
and frozen before the held-out cases (repetitions 2-5, 99 cases) were scored.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import riskgate as rg

DATA = Path(__file__).resolve().parents[3] / "public_data" / "rcaeval-hf"
LICENCE = "RCAEval: HF mirror MIT; original data Zenodo 14590730 CC BY 4.0"
SERVICES = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice",
            "productcatalogservice", "recommendationservice", "redis", "shippingservice"]
FAULTS = ["cpu", "mem", "disk", "delay", "loss"]
# Online Boutique call graph (public architecture): caller -> callees
CALLEES = {"frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice", "productcatalogservice",
                        "recommendationservice", "shippingservice"],
           "checkoutservice": ["cartservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice",
                               "shippingservice"],
           "recommendationservice": ["productcatalogservice"], "cartservice": ["redis"]}
WINDOW = 300          # seconds before / after the injection
EXCLUDED = {"re1ob_productcatalogservice_cpu_3": "telemetry ends before the injection (no faulty period)"}
REPAIRED = {"re1ob_currencyservice_loss_1": "truncated inject_time (8 digits) repaired to time_start + 360"}


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
@dataclass
class Case:
    case: str
    service: str
    fault: str
    rep: int
    inject: int
    z: dict = field(default_factory=dict)       # (service, metric) -> robust z (post vs pre)

    @property
    def family(self) -> str:
        return f"{self.service}/{self.fault}"

    @property
    def truth(self) -> str:
        return f"{self.service}:{self.fault}"


def _metric_kind(col: str) -> tuple[str, str] | None:
    s, _, k = col.rpartition("_")
    if s not in SERVICES:
        return None
    k = {"load": "load", "workload": "load", "latency": "lat", "latency-50": "lat", "cpu": "cpu", "mem": "mem"}.get(k)
    return (s, k) if k else None


def robust_z(pre: np.ndarray, post: np.ndarray) -> float:
    """BARO-style robust standardisation: (median(post) - median(pre)) / robust sd(pre); the sd has a 5 % relative
    floor so that near-constant series (flat memory) do not produce infinite z-scores."""
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 30 or len(post) < 30:
        return 0.0
    med = float(np.median(pre))
    iqr = float(np.subtract(*np.percentile(pre, [75, 25])))
    sd = max(iqr / 1.349, 0.05 * abs(med), 1e-9)
    return float((np.median(post) - med) / sd)


def load_cases() -> tuple[list[Case], dict]:
    idx = pd.read_parquet(DATA / "cases.parquet")
    ob = idx[idx.dataset == "RE1-OB"].sort_values("case")
    cases, notes = [], {"excluded": dict(EXCLUDED), "repaired": dict(REPAIRED)}
    for _, row in ob.iterrows():
        if row.case in EXCLUDED:
            continue
        m = pd.read_parquet(DATA / row.case / "metrics.parquet")
        t = m["time"].to_numpy()
        inj = int((DATA / row.case / "inject_time.txt").read_text().strip())
        if row.case in REPAIRED:
            inj = int(t.min()) + 360
        pre, post = (t >= inj - WINDOW) & (t < inj), (t >= inj) & (t < inj + WINDOW)
        c = Case(row.case, row.root_cause_service, row.fault, int(row.repetition), inj)
        for col in m.columns:
            sk = _metric_kind(col)
            if sk:
                x = m[col].to_numpy(float)
                c.z[sk] = robust_z(x[pre], x[post])
        cases.append(c)
    return cases, notes


# ----------------------------------------------------------------------------------------------
# LLM-free hypothesis scorer (signatures fixed on the development subset)
# ----------------------------------------------------------------------------------------------
W_DOM = 0.5            # penalty when the other resource metric dominates (cpu vs mem)
W_CALLEE = 0.5         # network faults: an elevated callee explains a caller's latency
W_RES = 0.5            # network faults do not raise the service's own CPU / memory
W_CPU_LAT = 0.3        # a CPU hog barely moves the service's own p50 latency (disk and memory stress do)
DISK = (0.6, 0.4, 0.4, 4.0, 3.0, 5.0)   # disk: w_cpu, w_mem, w_lat, mem cap, lat cap, mem level beyond which it is mem
LAT_SPLIT = 4.0        # p50 latency shift (log1p z) separating "delay" (large) from "loss" (moderate)
W_SPLIT = 0.3
FAULT_OF_METRIC = {"cpu": "cpu", "mem": "mem", "lat": "delay"}


def _a(z: float) -> float:
    return math.copysign(math.log1p(abs(z)), z)


def evidence(c: Case) -> dict[str, float]:
    """E(h) for every hypothesis h = 'service:fault' (higher = better explained), from log-compressed robust z-scores
    a = sign(z) log(1 + |z|) of the service's CPU, memory and p50 latency (and its callees' latency)."""
    a = {k: _a(v) for k, v in c.z.items()}

    def g(s, k):
        return a.get((s, k), 0.0)

    wc, wm, wl, mcap, lcap, mlvl = DISK
    out = {}
    for s in SERVICES:
        cpu, mem, lat = g(s, "cpu"), g(s, "mem"), g(s, "lat")
        callee = max([max(0.0, g(x, "lat")) for x in CALLEES.get(s, [])] or [0.0])
        net = lat - W_CALLEE * callee - W_RES * max(0.0, cpu) - W_RES * max(0.0, mem)
        out[f"{s}:cpu"] = cpu - W_DOM * max(0.0, mem - cpu) - W_CPU_LAT * max(0.0, lat)
        out[f"{s}:mem"] = mem - W_DOM * max(0.0, cpu - mem)
        out[f"{s}:disk"] = wc * cpu + wm * min(mem, mcap) + wl * min(max(0.0, lat), lcap) - W_DOM * max(0.0, mem - mlvl)
        out[f"{s}:delay"] = net + W_SPLIT * (lat - LAT_SPLIT)
        out[f"{s}:loss"] = net - W_SPLIT * (lat - LAT_SPLIT)
    return out


def nonconformity(c: Case) -> dict[str, float]:
    e = evidence(c)
    top = max(e.values())
    return {h: top - v for h, v in e.items()}


def baro_rank(c: Case) -> list[str]:
    """Reference ranking (BARO-style): services by their largest robust |z| over CPU, memory, latency; fault type
    read off the argmax metric (cpu -> cpu, mem -> mem, latency -> delay). Not used by the gate."""
    best = []
    for s in SERVICES:
        vals = {k: abs(c.z.get((s, k), 0.0)) for k in ("cpu", "mem", "lat")}
        k = max(vals, key=vals.get)
        best.append((vals[k], f"{s}:{FAULT_OF_METRIC[k]}"))
    return [h for _, h in sorted(best, reverse=True)]


# ----------------------------------------------------------------------------------------------
# Remediation: action-effect table (SRE practice) = consequence model AND label rule (applied to the true cause)
# ----------------------------------------------------------------------------------------------
KINDS = ("restart", "scale_out", "rollback", "rate_limit")
EFFECT = {   # does the action, applied to the FAULTY service, mitigate this fault class?
    "cpu": {"restart": True, "scale_out": True, "rollback": True, "rate_limit": False},
    "mem": {"restart": True, "scale_out": False, "rollback": True, "rate_limit": False},
    "disk": {"restart": True, "scale_out": False, "rollback": True, "rate_limit": False},
    "delay": {"restart": False, "scale_out": False, "rollback": False, "rate_limit": False},
    "loss": {"restart": False, "scale_out": False, "rollback": False, "rate_limit": False},
}
EFFECT_RATIONALE = {
    "cpu": "CPU hog in the pod: restart kills it; scale-out x2 adds capacity and halves the hogged replica's share; a "
           "rollback replaces the pods (= restart); rate-limiting the frontend sheds users and fixes nothing",
    "mem": "memory leak / stress: restart frees the memory (rollback replaces the pods too); scale-out only delays the "
           "leaking replica's degradation",
    "disk": "disk I/O stress (runaway writer in the pod): restart / rollback kill the writer; scale-out leaves the "
            "saturated replica serving slow requests",
    "delay": "network delay on the service's path: neither restart nor scale-out repairs the network -> escalate",
    "loss": "packet loss on the service's path: neither restart nor scale-out repairs the network -> escalate",
    "other service": "any action on a service that is not the root cause leaves the incident in place and disrupts a "
                     "healthy service",
    "rollback": "RE1 faults are not deploy-related; a rollback only helps through the pod replacement it implies",
}


def act(kind: str, service: str = "frontend") -> rg.Action:
    return rg.Action(kind, {"service": "frontend" if kind == "rate_limit" else service})


def effective(a: rg.Action, service: str, fault: str) -> bool:
    if a.name == "rate_limit":
        return False
    return a.args["service"] == service and EFFECT[fault][a.name]


# ----------------------------------------------------------------------------------------------
# Scripted doers (clearly scripted, LLM-like; no LLM is called)
# ----------------------------------------------------------------------------------------------
PRIMARY = {"cpu": ("scale_out", "restart"), "mem": ("restart", "scale_out"), "lat": ("restart", "scale_out")}


def _dashboard(c: Case) -> list[tuple[float, str, str]]:
    """What an LLM agent reading dashboards sees: per service, its largest positive jump (robust z, after vs the
    previous 5 min) among CPU, memory and latency. No call graph and no fault signatures (unlike the gate's scorer)."""
    out = []
    for s in SERVICES:
        vals = {k: c.z.get((s, k), 0.0) for k in ("cpu", "mem", "lat")}
        k = max(vals, key=vals.get)
        out.append((vals[k], s, k))
    return sorted(out, reverse=True)


def doer_llm_like(c: Case) -> list[rg.Action]:
    """Acts on the service with the biggest jump, with the 'most common fix' for what jumped (CPU -> scale out,
    memory -> restart, latency -> restart), then the other fix, then the runner-up service; 15 % of the time the
    first two are swapped (sampling noise), seeded by the case name."""
    d = _dashboard(c)
    (_, s1, k1), (_, s2, k2) = d[0], d[1]
    cands = [act(PRIMARY[k1][0], s1), act(PRIMARY[k1][1], s1), act(PRIMARY[k2][0], s2)]
    if np.random.default_rng(zlib.crc32(c.case.encode())).random() < 0.15:
        cands[0], cands[1] = cands[1], cands[0]
    return cands


def doer_restart_first(c: Case) -> list[rg.Action]:
    """Same reading, always restarts first (then rolls back, then restarts the runner-up)."""
    d = _dashboard(c)
    return [act("restart", d[0][1]), act("rollback", d[0][1]), act("restart", d[1][1])]


DOERS = {"llm-like (dashboard)": doer_llm_like, "restart-first": doer_restart_first}


# ----------------------------------------------------------------------------------------------
# Gate components
# ----------------------------------------------------------------------------------------------
OUTSIDE = "outside-plausible-set"   # sentinel at score tau: the truth may lie outside the set (every action fails)


class SignatureHypotheses:
    """The scorer's hypotheses with r <= tau (the delta-calibrated plausible set), optionally with the OUTSIDE
    sentinel (needed by the evidence-gap score), or only the top-1 hypothesis (single-diagnosis baselines)."""

    def __init__(self, tau: float = math.inf, sentinel: bool = False, top1: bool = False):
        self.tau, self.sentinel, self.top1 = float(tau), sentinel, top1
        self.version = f"re1ob-signatures-v1:tau={tau:.6g}:sentinel={sentinel}:top1={top1}"

    def hypotheses(self, ctx: dict) -> list[rg.Hypothesis]:
        hs = sorted(ctx["r"].items(), key=lambda kv: (kv[1], kv[0]))
        mk = [rg.Hypothesis(h, family=h, params={"service": h.split(":")[0], "fault": h.split(":")[1]}, score=float(v))
              for h, v in hs]
        if self.top1:
            gap = float(hs[1][1] - hs[0][1])
            return [rg.Hypothesis(mk[0].name, family=mk[0].family, params={**mk[0].params, "gap": gap}, score=0.0)]
        out = [h for h in mk if h.score <= self.tau]
        if self.sentinel and math.isfinite(self.tau):
            out.append(rg.Hypothesis(OUTSIDE, family=OUTSIDE, params={}, score=self.tau))
        return out


class EffectTable:
    """Consequence model: the action-effect table applied to hypothesis (service, fault)."""

    version = "re1ob-effect-table-v1"

    def predict(self, a: rg.Action, h: rg.Hypothesis, ctx: dict) -> dict:
        if h.name == OUTSIDE:
            return {"effective": False}
        return {"effective": effective(a, h.params["service"], h.params["fault"])}


SPEC = rg.Spec(rg.Predicate("action_mitigates_the_fault", lambda o: bool(o["effective"])))


def evidence_gap(m: np.ndarray, plaus) -> float:
    """Robust score for binary effects: r of the best-fitting plausible hypothesis under which the action FAILS
    (capped at tau by the OUTSIDE sentinel) minus r of the best one under which it works; -inf if none works."""
    r = np.array([h.score for h in plaus], float)
    m = np.asarray(m, float)
    ok, bad = r[m > 0], r[m < 0]
    if ok.size == 0:
        return -math.inf
    return float((bad.min() if bad.size else 1e6) - ok.min())


def top1_confidence(m: np.ndarray, plaus) -> float:
    """Single-diagnosis reading: + (top-1 lead over the runner-up) if the action works under the top-1 hypothesis."""
    gap = float(plaus[0].params.get("gap", 1.0))
    return gap if float(m[0]) > 0 else -1.0 - gap


def make_gates(tau: float, doer: str) -> dict[str, rg.Gate]:
    return {
        "twin": rg.Gate(EffectTable(), SPEC, SignatureHypotheses(top1=True), doer_id=doer),
        "top1": rg.Gate(EffectTable(), SPEC, SignatureHypotheses(top1=True), score_fn=top1_confidence, doer_id=doer),
        "set": rg.Gate(EffectTable(), SPEC, SignatureHypotheses(tau), doer_id=doer),
        "riskgate": rg.Gate(EffectTable(), SPEC, SignatureHypotheses(tau, sentinel=True), score_fn=evidence_gap, doer_id=doer),
    }


def episode_of(c: Case, doer) -> rg.Episode:
    cands = doer(c)
    return rg.Episode({"case": c.case, "r": nonconformity(c)}, cands,
                      {a.key: not effective(a, c.service, c.fault) for a in cands}, true_family=c.truth,
                      meta={"case": c.case, "truth": c.truth})


# ----------------------------------------------------------------------------------------------
# B.1  Hypothesis sets on real telemetry: accuracy, split conformal by case, family-out
# ----------------------------------------------------------------------------------------------
def ranking_metrics(cases: list[Case]) -> dict:
    top1 = svc1 = svc3 = fault1 = baro_svc1 = baro_top1 = 0
    ranks, by_fault = [], {}
    for c in cases:
        r = nonconformity(c)
        order = sorted(r, key=lambda h: (r[h], h))
        svcs = list(dict.fromkeys(h.split(":")[0] for h in order))
        ranks.append(order.index(c.truth) + 1)
        top1 += order[0] == c.truth
        svc1 += svcs[0] == c.service
        svc3 += c.service in svcs[:3]
        fault1 += order[0].split(":")[1] == c.fault
        b = baro_rank(c)
        baro_svc1 += b[0].split(":")[0] == c.service
        baro_top1 += b[0] == c.truth
        f = by_fault.setdefault(c.fault, {"n": 0, "top1": 0, "svc1": 0})
        f["n"] += 1
        f["top1"] += order[0] == c.truth
        f["svc1"] += svcs[0] == c.service
    n = len(cases)
    return {"n": n, "top1": top1 / n, "service@1": svc1 / n, "service@3": svc3 / n, "fault@1": fault1 / n,
            "mean_rank_of_truth": float(np.mean(ranks)), "n_hypotheses": len(SERVICES) * len(FAULTS),
            "baro_style_service@1": baro_svc1 / n, "baro_style_top1": baro_top1 / n,
            "by_fault": {f: {"n": v["n"], "top1": v["top1"] / v["n"], "service@1": v["svc1"] / v["n"]}
                         for f, v in sorted(by_fault.items())}}


def _rtrue(cases: list[Case]) -> np.ndarray:
    return np.array([nonconformity(c)[c.truth] for c in cases])


def _sizes(cases: list[Case], tau: float) -> np.ndarray:
    return np.array([sum(v <= tau for v in nonconformity(c).values()) for c in cases])


def conformal_sets(cases: list[Case], deltas=(0.05, 0.1, 0.2), n_resplits: int = 200, seed: int = 0) -> dict:
    """Split conformal by case: tau = conformal (1 - delta) quantile of r(true) on a random half, coverage and set
    size on the other half (fixed split seed 0, then `n_resplits` random splits)."""
    rt = _rtrue(cases)
    allr = [np.array(list(nonconformity(c).values())) for c in cases]
    n = len(cases)
    out = {}
    for d in deltas:
        res = {}
        covs, sizes = [], []
        for k in range(n_resplits + 1):
            rng = np.random.default_rng(seed if k == 0 else 1000 + k)
            perm = rng.permutation(n)
            cal, test = perm[: n // 2], perm[n // 2:]
            tau = rg.conformal_quantile(rt[cal], d)
            cov = float(np.mean(rt[test] <= tau))
            size = float(np.mean([np.sum(allr[i] <= tau) for i in test]))
            if k == 0:
                res["fixed_split"] = {"tau": tau, "coverage": cov, "mean_set_size": size, "n_cal": len(cal),
                                      "n_test": len(test)}
            else:
                covs.append(cov)
                sizes.append(size)
        res["resplits"] = {"n": n_resplits, "coverage_mean": float(np.mean(covs)), "coverage_sd": float(np.std(covs)),
                           "frac_below_target": float(np.mean(np.array(covs) < 1 - d)),
                           "set_size_mean": float(np.mean(sizes)), "set_size_sd": float(np.std(sizes))}
        out[str(d)] = res
    return out


def group_out(cases: list[Case], key, deltas=(0.05, 0.1, 0.2)) -> dict:
    """Leave-one-group-out conformal: tau from all other groups, coverage on the held-out group (a shift test:
    exchangeability does not hold across groups, so coverage is NOT guaranteed per group)."""
    groups = sorted({key(c) for c in cases})
    out = {}
    for d in deltas:
        per = {}
        for g in groups:
            cal = [c for c in cases if key(c) != g]
            test = [c for c in cases if key(c) == g]
            tau = rg.conformal_quantile(_rtrue(cal), d)
            per[g] = {"n": len(test), "coverage": float(np.mean(_rtrue(test) <= tau)), "mean_set_size": float(np.mean(_sizes(test, tau)))}
        covs = np.array([v["coverage"] for v in per.values()])
        w = np.array([v["n"] for v in per.values()], float)
        out[str(d)] = {"pooled_coverage": float(np.sum(covs * w) / w.sum()), "min_group_coverage": float(covs.min()),
                       "groups_below_target": int(np.sum(covs < 1 - d)), "n_groups": len(groups),
                       "mean_set_size": float(np.sum(np.array([v["mean_set_size"] for v in per.values()]) * w) / w.sum()),
                       "per_group": per}
    return out


# ----------------------------------------------------------------------------------------------
# B.2  Remediation gate with the common harness
# ----------------------------------------------------------------------------------------------
def gate_experiment(cases: list[Case], dev: list[Case], common, alphas=(0.05, 0.1), delta: float = 0.1,
                    n_resplits: int = 200) -> dict:
    tau = rg.conformal_quantile(_rtrue(dev), delta)            # plausible set fixed on the development subset
    res: dict = {"delta": delta, "tau_from_dev": tau, "n_dev": len(dev), "n_heldout": len(cases),
                 "plausible_set_heldout": {"coverage": float(np.mean(_rtrue(cases) <= tau)),
                                           "mean_size": float(np.mean(_sizes(cases, tau)))}}
    decs, scores = [], {k: [] for k in ("twin", "top1", "set", "riskgate")}
    eps_by_doer: dict[str, list] = {}
    for dname, doer in DOERS.items():
        gates = make_gates(tau, dname)
        eps = [episode_of(c, doer) for c in cases]
        eps_by_doer[dname] = eps
        for c, ep in zip(cases, eps):
            decs.append(common.Decision(group=c.case, traj=f"{c.case}:{dname}", doer=dname,
                                        unsafe=[bool(ep.unsafe[a.key]) for a in ep.candidates],
                                        meta={"truth": c.truth, "cands": [a.key for a in ep.candidates]}))
            for k, g in gates.items():
                scores[k].append([float(x) for x in g.score_episode(ep)])
    methods = {"none": ("none", scores["riskgate"]),
               "twin gate (top-1, fixed)": ("fixed", scores["twin"]),
               "top-1 + CRC": ("crc", scores["top1"]),
               "plausible set only (fixed, delta=0.1)": ("fixed", scores["set"]),
               "riskgate (CRC)": ("crc", scores["riskgate"]),
               "riskgate (LTT)": ("ltt", scores["riskgate"])}
    res["per_doer"], res["transfer"] = {}, {}
    for a in alphas:
        res["per_doer"][str(a)] = common.per_doer(decs, methods, alpha=a, n_resplits=n_resplits,
                                                  alphas=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20))
        res["transfer"][str(a)] = common.run_protocol(decs, methods, alpha=a, n_resplits=1,
                                                      transfer_methods=("riskgate (CRC)", "top-1 + CRC"))["transfer"]
    # discrimination of each score for "the doer's first choice is unsafe"
    res["auroc_first_choice"] = {}
    for dname in DOERS:
        idx = [i for i, d in enumerate(decs) if d.doer == dname]
        res["auroc_first_choice"][dname] = {k: common.auroc([scores[k][i][0] for i in idx], [decs[i].unsafe[0] for i in idx])
                                            for k in scores}
    # P9: Gate.calibrate (library) == harness lambda-hat on the fixed split of each doer
    p9 = {}
    for dname in DOERS:
        sub = [i for i, d in enumerate(decs) if d.doer == dname]
        cal_idx, _ = common.split_groups(decs, 0.5, 0, sub)
        pos = {i: k for k, i in enumerate(sub)}
        for a in alphas:
            for mname, gname, kind, guar in (("riskgate (CRC)", "riskgate", "crc", "expectation"),
                                             ("riskgate (LTT)", "riskgate", "ltt", "high_probability"),
                                             ("top-1 + CRC", "top1", "crc", "expectation")):
                g = make_gates(tau, dname)[gname]
                cal = g.calibrate([eps_by_doer[dname][pos[i]] for i in cal_idx], groups=[decs[i].group for i in cal_idx],
                                  tau_fraction=0.0, alpha=a, guarantee=guar, delta_ltt=common.DELTA_LTT)
                lam = common.lam_for(mname, kind, cal_idx, methods[mname][1], decs, a)
                assert lam == cal.lam, (dname, mname, a, lam, cal.lam)
                p9[f"{dname} | {mname} @ alpha={a}"] = {"harness_lambda": lam, "gate_lambda": cal.lam, "n_groups": cal.n}
    res["p9_library_check"] = p9
    # what the doers do, and the label distribution (rule-derived)
    res["doer_profile"] = {}
    for dname in DOERS:
        idx = [i for i, d in enumerate(decs) if d.doer == dname]
        res["doer_profile"][dname] = {
            "first_choice_unsafe": float(np.mean([decs[i].unsafe[0] for i in idx])),
            "some_candidate_safe": float(np.mean([not all(decs[i].unsafe) for i in idx])),
            "first_choice_unsafe_by_fault": {f: float(np.mean([decs[i].unsafe[0] for i in idx if decs[i].meta["truth"].endswith(":" + f)]))
                                             for f in FAULTS}}
    res["_decisions"] = decs
    res["_scores"] = scores
    return res
