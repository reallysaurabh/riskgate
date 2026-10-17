"""Ablations and latency benchmark (PLAN.md phase 4).

    cd examples && PYTHONPATH=.. python ablations.py [--quick]

A1 mode: policy vs agnostic
A2 doer swap: calibrate with the default doer, deploy with a different (randomised-order) doer
A3 plausible-set coverage delta
A4 calibration size n (mean +- sd over random re-splits)
A5 guarantee type: expectation (CRC) vs high probability (Learn-then-Test)
A6 probe share rho (two-stage probing)
L  latency: gate overhead without a model, and per-domain check latency
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import harness
import k8s_incident as K
import numpy as np
import refunds_agent as R

import riskgate as rg

OUT = Path(__file__).resolve().parent / "results"


def evaluate(gate: rg.Gate, eps: list[rg.Episode], name: str = "riskgate") -> dict:
    s = harness.summarise([harness.run_policy(gate, ep, name) for ep in eps], eps)
    return {"unsafe": s["unsafe"], "ask": s["ask"], "probe": s["probe"], "safe_execution": s["safe_execution"]}


def shuffled(eps: list[rg.Episode], seed: int) -> list[rg.Episode]:
    """Same situations and labels, proposals re-ordered by a different (random) doer."""
    rng = np.random.default_rng(seed)
    out = []
    for ep in eps:
        order = list(rng.permutation(len(ep.candidates)))
        cands = [ep.candidates[i] for i in order]
        ap = {k: rg.Episode(v.context, [v.candidates[i] for i in order] if len(v.candidates) == len(order) else v.candidates,
                            v.unsafe, v.true_family, meta=v.meta) for k, v in ep.after_probe.items()}
        out.append(rg.Episode(ep.context, cands, ep.unsafe, ep.true_family, ap, ep.meta))
    return out


def run_domain(dom, n_cal: int, n_test: int, quick: bool) -> dict:
    t0 = time.time()
    cal = [dom.sample(10_000 + i) for i in range(n_cal)]
    test = [dom.sample(50_000 + i) for i in range(n_test)]
    print(f"[{dom.NAME}] sampled {n_cal}+{n_test} in {time.time() - t0:.0f}s")
    res: dict = {}
    swapped = shuffled(test, 7)
    # A1 + A2
    for mode in ("policy", "agnostic"):
        g = dom.make_gate(probes=(), mode=mode)
        g.calibrate(cal)
        res[f"A1_{mode}"] = evaluate(g, test)
        res[f"A2_{mode}_doer_swap"] = evaluate(g, swapped)
    # A3
    for d in (0.01, 0.05, 0.2, 0.5):
        g = dom.make_gate(probes=(), delta=d)
        g.calibrate(cal)
        res[f"A3_delta_{d}"] = {**evaluate(g, test), "mean_plausible": g.calibration.diagnostics["mean_plausible_hypotheses"]}
    # A4
    pool = cal + test
    for n in ([150, 300, 600] if quick else [150, 300, 600, 1200]):
        if n > len(pool) - 200:
            continue
        vals = []
        for r in range(5 if quick else 10):
            perm = np.random.default_rng(500 + r).permutation(len(pool))
            c, t = [pool[i] for i in perm[:n]], [pool[i] for i in perm[n:n + 600]]
            g = dom.make_gate(probes=())
            g.calibrate(c, seed=r)
            vals.append(evaluate(g, t))
        res[f"A4_n_{n}"] = {"unsafe_mean": float(np.mean([v["unsafe"] for v in vals])),
                            "unsafe_sd": float(np.std([v["unsafe"] for v in vals])),
                            "ask_mean": float(np.mean([v["ask"] for v in vals]))}
    # A5
    try:
        g = dom.make_gate(probes=())
        g.calibrate(cal, guarantee="high_probability", delta_ltt=0.1)
        res["A5_high_probability"] = evaluate(g, test)
        res["A5_expectation"] = res["A1_policy"]
    except ImportError as exc:
        res["A5_high_probability"] = {"skipped": str(exc)}
    # A6
    for rho in (0.4, 0.6, 0.8):
        g = dom.make_gate()
        g.calibrate(cal, probe_share=rho)
        res[f"A6_probe_share_{rho}"] = {**evaluate(g, test, "riskgate_probe"), "probing_calibrated": g.calibration.probing}
    # latency per check (excluding sampling)
    g = dom.make_gate(probes=())
    g.calibrate(cal)
    lat = []
    for ep in test[:300]:
        t = time.perf_counter()
        g.check_ranked(ep.candidates, ep.context)
        lat.append((time.perf_counter() - t) * 1e3)
    res["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    print(f"[{dom.NAME}] done in {time.time() - t0:.0f}s")
    for k, v in res.items():
        print(f"   {k:28s} {json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})}")
    return res


def overhead() -> dict:
    """Gate overhead with a trivial model: 4 actions x 3 hypotheses."""
    class M:
        version = "m"

        def predict(self, a, h, c):
            return {"x": c + a.args["v"] - h.params["d"]}

    class H:
        version = "h"

        def hypotheses(self, c):
            return [rg.Hypothesis(str(d), params={"d": d}, score=0.0) for d in (0, 1, 2)]

    g = rg.Gate(M(), rg.Spec(rg.Envelope("x", 0, 10)), H(), require_calibration=False)
    acts = [rg.Action("a", {"v": v}) for v in (1, 2, 3, 4)]
    lat = []
    for _ in range(3000):
        t = time.perf_counter()
        g.check_ranked(acts, 5.0)
        lat.append((time.perf_counter() - t) * 1e3)
    return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = {"overhead_ms": overhead()}
    print("gate overhead (no model):", out["overhead_ms"])
    out["refunds_agent"] = run_domain(R, 1500 if args.quick else 4000, 600 if args.quick else 1500, args.quick)
    out["k8s_incident"] = run_domain(K, 600 if args.quick else 1500, 400 if args.quick else 1000, args.quick)
    OUT.mkdir(exist_ok=True)
    (OUT / "ablations.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "ablations.json")


if __name__ == "__main__":
    main()
