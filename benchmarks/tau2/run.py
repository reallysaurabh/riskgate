"""Score every tau2 write with RiskGate and baselines, then run the common protocol (benchmarks/common.py).

    .venv-public/bin/python benchmarks/tau2/run.py dev        # 2025 runs only: design / sanity check (AUROC, rates)
    .venv-public/bin/python benchmarks/tau2/run.py full       # frozen design on all runs -> results/tau2_*.json, report

Design freeze: the hypothesis families and evidence weights in gate.py were set while looking only at the 2025
runs (`dev`); the 2026 leaderboard runs are evaluated once with the frozen version (VERSION in gate.py).
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

CACHE = HERE / "cache"
RES = HERE.parent / "results"
INF = float("inf")


def lite(domain: str) -> list[dict]:
    p = CACHE / f"decisions_{domain}_lite.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = []
    with (CACHE / f"decisions_{domain}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["error"]:
                continue
            r["prefix"] = [{"role": x["role"], "text": x.get("text", "")} for x in r["prefix"] if x["role"] in ("user", "assistant")]
            out.append(r)
    p.write_bytes(pickle.dumps(out))
    return out


SCORES = ("lr", "nearest", "mass", "top1")


def _score_chunk(recs: list[dict]) -> list[tuple]:
    from gate import Ctx, make_gate

    import riskgate as rg
    gates = {s: make_gate(score=s) for s in ("lr", "nearest", "mass")}
    gates["top1"] = make_gate(top1=True, score="nearest")
    out = []
    for r in recs:
        ctx = Ctx(r)
        ep = rg.Episode(ctx, [rg.Action(r["name"], r["args"])], {})
        row = [float(gates[s].score_episode(ep, tau=INF, tau_ood=INF)[0]) for s in SCORES]
        hs = gates["lr"].hypotheses.hypotheses(ctx)
        out.append(tuple(row) + (len(hs),))
    return out


def score_all(recs: list[dict], workers: int = 20) -> np.ndarray:
    chunks = [recs[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_score_chunk, chunks))
    out = np.zeros((len(recs), len(SCORES) + 1))
    for w, part in enumerate(parts):
        for j, row in enumerate(part):
            out[w + j * workers] = row
    return out


def decisions(recs: list[dict]) -> list[C.Decision]:
    return [C.Decision(group=f"{r['domain']}:{r['task_id']}", traj=str(r["sim_id"]), doer=r["model"], unsafe=[bool(r["unsafe"])],
                       meta={"name": r["name"], "year": r["year"]}) for r in recs]


def dev(domain: str) -> None:
    recs = [r for r in lite(domain) if r["year"] == "2025"]
    sc = score_all(recs)
    u = np.array([r["unsafe"] for r in recs])
    conf = np.array([r["confirmed_kw"] for r in recs], float)
    print(f"[dev {domain}] n={len(recs)} unsafe={u.mean():.3f}  " + "  ".join(f"AUROC {s}={C.auroc(sc[:, j], u):.3f}" for j, s in enumerate(SCORES))
          + f"  confirm={C.auroc(conf, u):.3f}  mean |H|={sc[:, -1].mean():.1f}")
    for name in sorted({r["name"] for r in recs}):
        m = np.array([r["name"] == name for r in recs])
        if u[m].any() and (~u[m]).any():
            print(f"    {name:32s} n={m.sum():4d} unsafe={u[m].mean():.2f} " +
                  " ".join(f"{s}={C.auroc(sc[m, j], u[m]):.2f}" for j, s in enumerate(SCORES)))
    decs = decisions(recs)
    methods = {"none": ("none", [[0.0]] * len(recs)), "confirm": ("fixed", [[1.0 if c else -1.0] for c in conf]),
               "top1_crc": ("crc", [[s] for s in sc[:, 3]])}
    for j, s in enumerate(SCORES[:3]):
        methods[f"riskgate_{s}"] = ("crc", [[x] for x in sc[:, j]])
    for alpha in (0.05, 0.10):
        res = C.run_protocol(decs, methods, alpha=alpha, n_resplits=50, transfer_methods=())
        print(f"  alpha={alpha}: " + "  ".join(f"{k}: {100 * v['unsafe']:.1f}%/ask {100 * v['ask']:.0f}%"
                                               for k, v in res["fixed_split"].items()))
        print("     resplits: " + "  ".join(f"{k}: {100 * v['unsafe_mean']:.1f}+-{100 * v['unsafe_sd']:.1f}% ask {100 * v['ask_mean']:.0f}%"
                                            for k, v in res["resplits"].items()))


def methods_for(recs: list[dict], sc: np.ndarray, cs: np.ndarray | None = None) -> dict:
    conf = [r["confirmed_kw"] for r in recs]
    col = {s: j for j, s in enumerate(SCORES)}
    m = {"none": ("none", [[0.0]] * len(recs)),
         "confirm_rule": ("fixed", [[1.0 if c else -1.0] for c in conf]),
         "top1_crc": ("crc", [[s] for s in sc[:, col["top1"]]]),
         "riskgate": ("crc", [[s] for s in sc[:, col["mass"]]]),
         "riskgate_ltt": ("ltt", [[s] for s in sc[:, col["mass"]]]),
         "riskgate_worstcase": ("crc", [[s] for s in sc[:, col["nearest"]]]),
         "riskgate_lr": ("crc", [[s] for s in sc[:, col["lr"]]])}
    if cs is not None:                       # exploratory: hypotheses proposed by other agent runs (committee.py)
        m["selfconsistency"] = ("crc", [[s] for s in cs[:, 0]])
        m["committee"] = ("crc", [[s] for s in cs[:, 1]])
        m["committee_ltt"] = ("ltt", [[s] for s in cs[:, 1]])
    return m


def p9_check(recs: list[dict], decs: list[C.Decision], sc: np.ndarray) -> dict:
    """The harness lambda-hat equals riskgate.Gate.calibrate(groups=...) on the fixed split (primary score)."""
    from gate import Ctx, make_gate

    import riskgate as rg
    cal_idx, _ = C.split_groups(decs, 0.5, 0)
    n_groups = len({decs[i].group for i in cal_idx})
    alpha = 0.05 if n_groups >= 20 else 0.10
    g = make_gate(alpha=alpha)
    eps = [rg.Episode(Ctx(recs[i]), [rg.Action(recs[i]["name"], recs[i]["args"])],
                      {rg.Action(recs[i]["name"], recs[i]["args"]).key: bool(recs[i]["unsafe"])}) for i in cal_idx]
    cal = g.calibrate(eps, groups=[decs[i].group for i in cal_idx], tau_fraction=0.0)
    lam_h = C.lam_for("riskgate", "crc", cal_idx, [[s] for s in sc[:, SCORES.index("mass")]], decs, alpha)
    assert cal.lam == lam_h, (cal.lam, lam_h)
    return {"alpha": alpha, "lambda_library": cal.lam, "lambda_harness": lam_h, "n_groups": cal.n,
            "n_decisions": len(cal_idx), "fingerprint": cal.fingerprint}


def examples(recs: list[dict], sc: np.ndarray, lam: float, k: int = 2) -> list[dict]:
    from gate import Ctx, IntentHypotheses
    hp = IntentHypotheses()
    s = sc[:, SCORES.index("mass")]
    out = []
    picks = [("stopped: wrong write the gate does not certify", lambda i: recs[i]["unsafe"] and s[i] < lam),
             ("certified: correct write executed automatically", lambda i: (not recs[i]["unsafe"]) and s[i] >= lam),
             ("missed: wrong write the gate certifies", lambda i: recs[i]["unsafe"] and s[i] >= lam)]
    rng = np.random.default_rng(0)
    for title, f in picks:
        cand = [i for i in range(len(recs)) if recs[i]["year"] == "2026" and f(i)]
        for i in rng.choice(cand, size=min(k, len(cand)), replace=False) if cand else []:
            r = recs[int(i)]
            hs = sorted(hp.hypotheses(Ctx(r)), key=lambda h: h.score)
            users = [p["text"] for p in r["prefix"] if p["role"] == "user"]
            out.append({"kind": title, "model": r["model"], "task": r["task_id"], "trial": r["trial"], "score": float(s[int(i)]),
                        "call": {"name": r["name"], "args": r["args"]}, "gold": r["gold_writes"],
                        "first_user": users[0][:500] if users else "", "last_user": users[-1][:300] if users else "",
                        "top_hypotheses": [(h.name, round(float(np.exp(-h.score)), 3)) for h in hs[:5]]})
    return out


def full(domain: str) -> dict:
    import committee
    recs = lite(domain)
    order = {m: k for k, m in enumerate(sorted({(r["year"], r["model"]) for r in recs}, key=lambda x: (x[0], x[1]), reverse=True))}
    sc = score_all(recs)
    cs = committee.scores(domain, recs)
    decs = decisions(recs)
    meth = methods_for(recs, sc, cs)
    u = np.array([r["unsafe"] for r in recs])
    out: dict = {"domain": domain, "gate_version": __import__("gate").VERSION, "committee_version": committee.VERSION,
                 "n_writes": len(recs), "models": {}, "pooled": {}, "auroc": {}, "p9": {}, "examples": []}
    for (year, model), _ in sorted(order.items(), key=lambda x: x[1]):
        idx = [i for i, r in enumerate(recs) if r["model"] == model]
        out["auroc"][model] = {s: C.auroc(sc[idx, j], u[idx]) for j, s in enumerate(SCORES)}
        out["auroc"][model]["confirm_rule"] = C.auroc([float(recs[i]["confirmed_kw"]) for i in idx], u[idx])
        out["auroc"][model]["selfconsistency"] = C.auroc(cs[idx, 0], u[idx])
        out["auroc"][model]["committee"] = C.auroc(cs[idx, 1], u[idx])
        out["models"][model] = {"year": year, "n": len(idx), "n_tasks": len({recs[i]["task_id"] for i in idx}),
                                "unsafe_pooled": float(u[idx].mean()),
                                "unsafe_taskavg": C.evaluate_at(-INF, idx, [[0.0]] * len(recs), decs)["base_unsafe"],
                                "unsafe_upper": float(np.mean([recs[i]["unsafe_upper"] for i in idx])),
                                "unsafe_confirmed": int(sum(recs[i]["unsafe"] and recs[i]["confirmed_kw"] for i in idx))}
        for alpha in (0.05, 0.10):
            out["models"][model][f"a{int(alpha * 100)}"] = C.run_protocol(decs, meth, alpha=alpha, subset=idx, transfer_methods=())
        a5, a10 = out["models"][model]["a5"]["resplits"], out["models"][model]["a10"]["resplits"]
        print(f"[{domain}] {model}: n={len(idx)} unsafe={u[idx].mean():.3f} AUROC mass={out['auroc'][model]['mass']:.3f} "
              f"committee={out['auroc'][model]['committee']:.3f} self={out['auroc'][model]['selfconsistency']:.3f} | "
              f"rg@5% {a5['riskgate']['unsafe_mean']:.3f}/{a5['riskgate']['ask_mean']:.2f} "
              f"@10% {a10['riskgate']['unsafe_mean']:.3f}/{a10['riskgate']['ask_mean']:.2f} | "
              f"committee@5% {a5['committee']['unsafe_mean']:.3f}/{a5['committee']['ask_mean']:.2f} "
              f"@10% {a10['committee']['unsafe_mean']:.3f}/{a10['committee']['ask_mean']:.2f}")
    for year in ("2026", "2025", "all"):
        idx = [i for i, r in enumerate(recs) if year == "all" or r["year"] == year]
        for alpha in (0.05, 0.10):
            out["pooled"][f"{year}_a{int(alpha * 100)}"] = C.run_protocol(decs, meth, alpha=alpha, subset=idx,
                                                                          transfer_methods=("riskgate", "committee") if year != "all" else ())
    # agent upgrade: calibrate on ALL 2025 runs (old agents), deploy on 2026 runs (new agents) of held-out tasks
    old = [i for i, r in enumerate(recs) if r["year"] == "2025"]
    new = [i for i, r in enumerate(recs) if r["year"] == "2026"]
    s_main = [[s] for s in sc[:, SCORES.index("mass")]]
    out["upgrade"] = {}
    for alpha in (0.05, 0.10):
        rows = []
        for rr in range(200):
            c_i, t_i = C.split_groups(decs, 0.5, 1000 + rr)
            cg, tg = {decs[i].group for i in c_i}, {decs[i].group for i in t_i}
            lam = C.lam_for("riskgate", "crc", [i for i in old if decs[i].group in cg], s_main, decs, alpha)
            m_new = C.evaluate_at(lam, [i for i in new if decs[i].group in tg], s_main, decs)
            lam2 = C.lam_for("riskgate", "crc", [i for i in new if decs[i].group in cg], s_main, decs, alpha)
            m_re = C.evaluate_at(lam2, [i for i in new if decs[i].group in tg], s_main, decs)
            rows.append((m_new["unsafe"], m_new["ask"], m_re["unsafe"], m_re["ask"]))
        a = np.array(rows)
        out["upgrade"][f"a{int(alpha * 100)}"] = {
            "calibrated_on_2025": {
                "unsafe_mean": float(a[:, 0].mean()), "unsafe_sd": float(a[:, 0].std()),
                "ask_mean": float(a[:, 1].mean()), "frac_above_alpha": float(np.mean(a[:, 0] > alpha)),
            },
            "recalibrated_on_2026": {
                "unsafe_mean": float(a[:, 2].mean()), "unsafe_sd": float(a[:, 2].std()),
                "ask_mean": float(a[:, 3].mean()), "frac_above_alpha": float(np.mean(a[:, 2] > alpha)),
            },
        }
    new_recs = [recs[i] for i in new]
    out["p9"] = p9_check(new_recs, [decs[i] for i in new], sc[new])
    lam_pool = out["pooled"]["2026_a10"]["fixed_split"]["riskgate"]["lambda"]
    out["examples"] = examples(recs, sc, float(lam_pool) if not isinstance(lam_pool, str) else INF)
    # tool-level view (2026 runs)
    out["by_tool_2026"] = {}
    for name in sorted({recs[i]["name"] for i in new}):
        idx = [i for i in new if recs[i]["name"] == name]
        out["by_tool_2026"][name] = {"n": len(idx), "unsafe": float(u[idx].mean()), "auroc": C.auroc(sc[idx, SCORES.index("mass")], u[idx]),
                                     "auroc_committee": C.auroc(cs[idx, 1], u[idx])}
    C.dump(out, RES / f"tau2_{domain}_results.json")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    doms = sys.argv[2:] or ["retail", "airline"]
    for d in doms:
        if mode == "dev":
            dev(d)
        else:
            full(d)
