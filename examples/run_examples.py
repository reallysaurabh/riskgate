"""Run all example domains: calibrate, evaluate, export realistic calibration data and figures.

    cd examples && PYTHONPATH=.. python run_examples.py            # full (CSTR takes the longest)
    cd examples && PYTHONPATH=.. python run_examples.py --quick
    cd examples && PYTHONPATH=.. python run_examples.py --only k8s_incident
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import harness
import numpy as np

HERE = Path(__file__).resolve().parent
SIZES = {"cstr_reactor": (800, 400, 200), "k8s_incident": (1500, 1000, 600), "refunds_agent": (4000, 1500, 600)}
RESPLITS = {"cstr_reactor": 10, "k8s_incident": 20, "refunds_agent": 20}


def figures(all_results: dict, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["no_checker", "twin_gate", "top1_crc", "riskgate", "riskgate_hp", "riskgate_probe"]
    labels = ["no\nchecker", "twin\ngate", "top-1\nCRC", "riskgate", "riskgate\nLTT", "riskgate\n+probe"]
    doms_all = [d for d in all_results if all(m in all_results[d]["nominal"] for m in methods)]
    if len(doms_all) < len(all_results):             # older result files without the LTT policy
        methods = [m for m in methods if m != "riskgate_hp"]
        labels = [lb for lb in labels if "LTT" not in lb]
    doms = [d for d in ("cstr_reactor", "k8s_incident", "refunds_agent") if d in all_results]
    titles = {"cstr_reactor": "Chemical reactor (CSTR)", "k8s_incident": "Kubernetes incident (SRE agent)",
              "refunds_agent": "Refunds (support agent)"}
    fig, axs = plt.subplots(1, len(doms), figsize=(5.0 * len(doms), 3.6), squeeze=False)
    for ax, d in zip(axs[0], doms):
        r = all_results[d]["nominal"]
        x = np.arange(len(methods))
        u = [r[m]["unsafe"] for m in methods]
        lo = [max(0.0, r[m]["unsafe"] - r[m]["unsafe_ci"][0]) for m in methods]
        hi = [max(0.0, r[m]["unsafe_ci"][1] - r[m]["unsafe"]) for m in methods]
        ax.bar(x - 0.2, u, 0.4, yerr=[lo, hi], color="#c0392b", capsize=2, label="unsafe executed")
        ax.bar(x + 0.2, [r[m]["ask"] for m in methods], 0.4, color="#d98e04", label="asked / rejected")
        ax.text(0.01, 0.97, "simulated data", transform=ax.transAxes, fontsize=6.5, va="top", color="#666")
        ax.axhline(all_results[d]["alpha"], ls="--", c="k", lw=0.8)
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_title(titles[d], fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "methods_all_domains.png", dpi=170)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    amax = 0.22
    ax.plot([0, amax], [0, amax], "k--", lw=0.8, label="y = alpha")
    for d, c in zip(doms, ["#1a3d66", "#8a63d2", "#1f9d55"]):
        sw = all_results[d]["alpha_sweep"]
        ax.plot([s["alpha"] for s in sw], [s["unsafe"] for s in sw], "o-", c=c, label=titles[d].split(" (")[0])
    ax.set_xlabel("target risk level alpha")
    ax.set_ylabel("unsafe executed (test)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "alpha_sweep_all_domains.png", dpi=170)

    # validity over random re-splits (what CRC bounds: the mean over calibration draws)
    rs_methods = [("riskgate", "riskgate"), ("riskgate_hp", "riskgate\nLTT"), ("riskgate_probe", "riskgate\n+probe"),
                  ("top1_crc", "top-1\nCRC")]
    fig, axs = plt.subplots(1, len(doms), figsize=(3.6 * len(doms), 3.2), squeeze=False)
    for ax, d in zip(axs[0], doms):
        rs = all_results[d].get("resplits", {})
        ms = [(m, lb) for m, lb in rs_methods if m in rs]
        x = np.arange(len(ms))
        means = [rs[m]["unsafe_mean"] for m, _ in ms]
        ax.bar(x, means, 0.55, yerr=[rs[m]["unsafe_sd"] for m, _ in ms], color="#c0392b", capsize=3,
               label="unsafe executed (mean ± sd)")
        for xi, (m, _), mu in zip(x, ms, means):
            ax.text(xi, mu + rs[m]["unsafe_sd"] + 0.003, f"ask {100 * rs[m]['ask_mean']:.0f}%", ha="center", fontsize=6.5,
                    color="#8a5a00")
        ax.axhline(all_results[d]["alpha"], ls="--", c="k", lw=0.8, label="alpha")
        ax.set_xticks(x, [lb for _, lb in ms], fontsize=7.5)
        ax.set_ylim(0, 0.12)
        n = max((rs[m]["n"] for m, _ in ms), default=0)
        ax.set_title(f"{titles[d].split(' (')[0]} ({n} re-splits)", fontsize=9.5)
        ax.text(0.01, 0.97, "simulated data", transform=ax.transAxes, fontsize=6.5, va="top", color="#666")
        ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "validity_resplits.png", dpi=170)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--ncal", type=int, default=None, help="override the number of calibration episodes")
    ap.add_argument("--figures-only", action="store_true", help="rebuild all_results.json and figures from results/")
    args = ap.parse_args()
    out = HERE / "results"
    data = HERE / "data"
    out.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    ap_figs = args.figures_only
    for name in ("k8s_incident", "refunds_agent", "cstr_reactor"):
        if ap_figs:
            break
        if args.only and name != args.only:
            continue
        mod = __import__(name)
        n_cal, n_test, n_shift = (100, 100, 60) if args.quick else SIZES[name]
        if args.ncal:
            n_cal = args.ncal
        harness.evaluate_domain(mod, data, n_cal, n_test, n_shift,
                                      n_resplits=3 if args.quick else RESPLITS[name])
        for f in data.glob(f"{name}_results.json"):
            f.replace(out / f.name)
    all_results = {p.stem.replace("_results", ""): json.loads(p.read_text()) for p in sorted(out.glob("*_results.json"))
                   if p.name != "all_results.json"}
    (out / "all_results.json").write_text(json.dumps(all_results, indent=1))
    figures(all_results, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
