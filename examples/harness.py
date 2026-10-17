"""Shared evaluation harness for the riskgate example domains.

Every domain provides:  make_gate(**kw) -> Gate,  sample(seed, shift=False) -> Episode (with after_probe),
and record(ep) -> dict (a realistic JSON log line). The harness compares five decision policies on
identical episodes and identical agent proposals:

  no_checker     execute the agent's first choice
  twin_gate      deterministic check: margin >= 0 under the single best-fitting hypothesis (no calibration)
  top1_crc       conformal risk control on the single best hypothesis' margin
  riskgate       conformal risk control on the hypothesis-robust margin (all plausible hypotheses)
  riskgate_hp    riskgate with the high-probability guarantee (Learn-then-Test, delta = 0.1)
  riskgate_probe riskgate + certified probing with two-stage calibration
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import riskgate as rg
from riskgate.gate import Calibration

DELTA_LTT = 0.1


class TopOne:
    """Wrap a hypothesis provider to expose only its best-fitting hypothesis (baseline)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.version = f"top1({getattr(inner, 'version', 'x')})"

    def hypotheses(self, ctx: Any) -> list[rg.Hypothesis]:
        hs = self.inner.hypotheses(ctx)
        return [min(hs, key=lambda h: h.score)] if hs else []


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class Outcome:
    verdict: str
    action: str | None
    unsafe: bool
    probed: str | None = None


def run_policy(gate: rg.Gate | None, ep: rg.Episode, name: str) -> Outcome:
    if gate is None:                                                    # no checker
        a = ep.candidates[0]
        return Outcome("execute", a.key, bool(ep.unsafe[a.key]))
    d = gate.check_ranked(ep.candidates, ep.context)
    probed = None
    if d.verdict is rg.Verdict.PROBE:
        probed = d.probe.key
        ep2 = ep.after_probe.get(d.probe.key)
        if ep2 is None:
            return Outcome("ask", None, False, probed)
        d = gate.check_ranked(ep2.candidates, ep2.context, stage=2)
        harm_probe = bool(ep2.meta.get("probe_unsafe", False))
        if d.verdict is rg.Verdict.CERTIFY:
            return Outcome("execute", d.action.key, bool(ep2.unsafe[d.action.key]) or harm_probe, probed)
        return Outcome(d.verdict.value, None, harm_probe, probed)
    if d.verdict is rg.Verdict.CERTIFY:
        return Outcome("execute", d.action.key, bool(ep.unsafe[d.action.key]))
    return Outcome(d.verdict.value, None, False)


def summarise(outs: list[Outcome], eps: list[rg.Episode]) -> dict:
    n = len(outs)
    unsafe = np.array([o.unsafe for o in outs])
    ask = np.array([o.verdict in ("ask", "reject") for o in outs])
    safe_exec = np.array([o.verdict == "execute" and not o.unsafe for o in outs])
    exists_safe = np.array([not all(e.unsafe.values()) for e in eps])
    lo, hi = wilson(unsafe.mean(), n)
    return {"n": n, "unsafe": float(unsafe.mean()), "unsafe_ci": [lo, hi], "ask": float(ask.mean()),
            "safe_execution": float(safe_exec.mean()), "avoidable_ask": float((ask & exists_safe).mean()),
            "probe": float(np.mean([o.probed is not None for o in outs]))}


def evaluate_domain(domain: Any, out_dir: Path, n_cal: int, n_test: int, n_shift: int, alpha: float = 0.05,
                    alphas: tuple = (0.02, 0.05, 0.10, 0.15, 0.20), seed0: int = 0, write_data: bool = True,
                    n_resplits: int = 20, resplit_probe: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cal_eps = [domain.sample(seed0 + 10_000 + i) for i in range(n_cal)]
    test_eps = [domain.sample(seed0 + 50_000 + i) for i in range(n_test)]
    shift_eps = [domain.sample(seed0 + 70_000 + i, shift=True) for i in range(n_shift)] if n_shift else []
    print(f"[{domain.NAME}] sampled {n_cal}+{n_test}+{n_shift} episodes in {time.time() - t0:.0f}s")

    gates: dict[str, rg.Gate | None] = {"no_checker": None}
    g_twin = domain.make_gate(hypotheses=TopOne(domain.hypotheses()), probes=())
    g_twin.load_calibration(Calibration(alpha=alpha, mode="policy", delta=1.0, tau=math.inf, tau_ood=math.inf,
                                        lam=0.0, fingerprint=g_twin.fingerprint(), n=0))
    gates["twin_gate"] = g_twin
    g_top = domain.make_gate(hypotheses=TopOne(domain.hypotheses()), probes=())
    g_top.calibrate(cal_eps, alpha=alpha)
    gates["top1_crc"] = g_top
    g_rob = domain.make_gate(probes=())
    cal_rob = g_rob.calibrate(cal_eps, alpha=alpha)
    gates["riskgate"] = g_rob
    g_hp = domain.make_gate(probes=())
    cal_hp = g_hp.calibrate(cal_eps, alpha=alpha, guarantee="high_probability", delta_ltt=DELTA_LTT)
    gates["riskgate_hp"] = g_hp
    g_probe = domain.make_gate()
    cal_probe = g_probe.calibrate(cal_eps, alpha=alpha)
    gates["riskgate_probe"] = g_probe
    print(f"[{domain.NAME}] calibrated in {time.time() - t0:.0f}s: tau={cal_rob.tau:.3g} lambda={cal_rob.lam:.3g} "
          f"top1 lambda={g_top.calibration.lam:.3g} probe lambdas=({cal_probe.lam_probe1}, {cal_probe.lam_probe2}) "
          f"n_probe={cal_probe.n_probe}")
    cal_rob.save(out_dir / f"{domain.NAME}_calibration_artifact.json")

    results: dict[str, Any] = {"domain": domain.NAME, "alpha": alpha, "n_cal": n_cal,
                               "calibration": {"tau": cal_rob.tau, "lambda": cal_rob.lam,
                                               "lambda_top1": g_top.calibration.lam,
                                               "lambda_hp": cal_hp.lam, "delta_ltt": DELTA_LTT,
                                               "lambda_probe": [cal_probe.lam_probe1, cal_probe.lam_probe2],
                                               "n_probe": cal_probe.n_probe,
                                               "mean_plausible": cal_rob.diagnostics["mean_plausible_hypotheses"]}}
    for cond, eps in (("nominal", test_eps), ("shift", shift_eps)):
        if not eps:
            continue
        results[cond] = {}
        outs_all = {}
        for name, g in gates.items():
            outs = [run_policy(g, ep, name) for ep in eps]
            outs_all[name] = outs
            results[cond][name] = summarise(outs, eps)
        print(f"[{domain.NAME}] {cond}:")
        for name, s in results[cond].items():
            print(f"   {name:15s} unsafe {100 * s['unsafe']:5.1f}% [{100 * s['unsafe_ci'][0]:4.1f},{100 * s['unsafe_ci'][1]:4.1f}]"
                  f"  ask {100 * s['ask']:5.1f}%  safe-exec {100 * s['safe_execution']:5.1f}%  probe {100 * s['probe']:4.1f}%")
        if cond == "nominal":
            results["_outs_nominal"] = outs_all

    # validity over random re-splits of the pooled (calibration + test) episodes: CRC guarantees the risk in
    # expectation over calibration draws, so a single split can land above alpha by chance.
    pool = cal_eps + test_eps
    rs: dict[str, list] = {"riskgate": [], "riskgate_hp": [], "top1_crc": [], "riskgate_probe": []}
    for r in range(n_resplits):
        perm = np.random.default_rng(1000 + r).permutation(len(pool))
        c_eps = [pool[i] for i in perm[:n_cal]]
        t_eps = [pool[i] for i in perm[n_cal:]]
        for name, factory in (("riskgate", lambda: domain.make_gate(probes=())),
                              ("riskgate_hp", lambda: domain.make_gate(probes=())),
                              ("top1_crc", lambda: domain.make_gate(hypotheses=TopOne(domain.hypotheses()), probes=())),
                              ("riskgate_probe", lambda: domain.make_gate())):
            if name == "riskgate_probe" and not resplit_probe:
                continue
            g = factory()
            if name == "riskgate_hp":
                g.calibrate(c_eps, alpha=alpha, seed=r, guarantee="high_probability", delta_ltt=DELTA_LTT)
            else:
                g.calibrate(c_eps, alpha=alpha, seed=r)
            s_ = summarise([run_policy(g, ep, name) for ep in t_eps], t_eps)
            rs[name].append((s_["unsafe"], s_["ask"]))
    results["resplits"] = {k: {"n": len(v), "unsafe_mean": float(np.mean([a for a, _ in v])),
                               "unsafe_sd": float(np.std([a for a, _ in v])),
                               "ask_mean": float(np.mean([b for _, b in v])),
                               "frac_above_alpha": float(np.mean([a > alpha for a, _ in v]))}
                           for k, v in rs.items() if v}
    print(f"[{domain.NAME}] resplits: " + "  ".join(f"{k}: unsafe {100 * v['unsafe_mean']:.1f}+-{100 * v['unsafe_sd']:.1f}% "
                                                  f"ask {100 * v['ask_mean']:.0f}%" for k, v in results["resplits"].items()))

    # validity across target levels (riskgate, no probing)
    sweep = []
    for a in alphas:
        g = domain.make_gate(probes=())
        g.calibrate(cal_eps, alpha=a)
        outs = [run_policy(g, ep, "riskgate") for ep in test_eps]
        s = summarise(outs, test_eps)
        sweep.append({"alpha": a, "lambda": g.calibration.lam, "unsafe": s["unsafe"], "ask": s["ask"]})
    results["alpha_sweep"] = sweep
    print(f"[{domain.NAME}] alpha sweep: " + "  ".join(f"a={s['alpha']:.2f}: unsafe {100 * s['unsafe']:.1f}% "
                                                     f"ask {100 * s['ask']:.0f}%" for s in sweep))

    if write_data:
        with (out_dir / f"{domain.NAME}_calibration.jsonl").open("w") as f:
            f.write(json.dumps({"_meta": domain.META}) + "\n")
            for ep in cal_eps:
                f.write(json.dumps(domain.record(ep), default=_json_default) + "\n")
        with (out_dir / f"{domain.NAME}_decisions_sample.jsonl").open("w") as f:
            for ep in test_eps[:40]:
                d = g_probe.check_ranked(ep.candidates, ep.context)
                rec = domain.record(ep)
                rec["riskgate_decision"] = {k: v for k, v in d.to_dict().items() if k not in ("considered",)}
                f.write(json.dumps(rec, default=_json_default) + "\n")
    results.pop("_outs_nominal", None)
    clean = json.loads(json.dumps(results, default=_json_default))
    (out_dir / f"{domain.NAME}_results.json").write_text(json.dumps(clean, indent=1))
    print(f"[{domain.NAME}] done in {time.time() - t0:.0f}s")
    return clean


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)
