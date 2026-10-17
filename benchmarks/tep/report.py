"""Markdown report for the TEP experiment (reads results/tep_results.json; numbers are never typed by hand except the
design-CV table, which reproduces the output of scratch/dev_cv.py and scratch/dev_temp.py)."""
from __future__ import annotations

import math
from pathlib import Path

DESIGN_CV = """| window | LDA (shrinkage) | RDA g = 0.5 | RDA g = 0.8 |
|---|---|---|---|
| 10 samples (30 min) | 0.649 | 0.767 | 0.760 |
| 20 samples (1 h) | 0.683 | 0.809 | 0.798 |
| 40 samples (2 h) | 0.731 | **0.866** | 0.847 |

Time-block cross-validated top-1 accuracy on the Braatz TRAINING files (4 contiguous blocks per run), features = mean and
log sd of the 52 standardised variables. Exact CV log-loss of RDA (g = 0.5, 40 samples) by softmax temperature
T = 1 / 4 / 8 / 16 / 32: 5.64 / 1.79 / 1.34 / **1.30** / 1.48, hence T = 16."""


def _f(x):
    if isinstance(x, str):
        return float(x.replace("inf", "Infinity")) if "inf" in x else float(x)
    return float(x)


def pct(x, nd=1):
    if x is None:
        return "n/a"
    x = _f(x)
    if math.isnan(x):
        return "n/a"
    return f"{100 * x:.{nd}f}%"


def num(x, nd=2):
    x = _f(x)
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:.{nd}f}"


def pm(d, nd=1):
    return f"{pct(d['mean'], nd)} +- {pct(d['sd'], nd)}"


def sets_section(L, hs, title, unit_note):
    w = L.append
    w(f"### {title}\n")
    w(unit_note + "\n")
    w(f"{hs['n_windows']} windows, {hs['n_groups']} groups, {hs['n_classes']} classes. Top-1 accuracy of the scorer: "
      f"**{pct(hs['top1_acc'])}** (" + ", ".join(f"{k}: {pct(v)}" for k, v in hs["top1_acc_by_family"].items()) + ").\n")
    w("| delta | sets | coverage (mean +- sd over re-splits) | mean set size | empty sets | correct singletons | "
      "coverage hard (3, 9, 15) | size hard | coverage easy (1, 2, 4-7) | size easy | coverage normal | size normal |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kind, lab in (("marginal", "marginal split conformal"), ("class_conditional", "class-conditional (per-class tau)")):
        for dk, m in hs[kind].items():
            w(f"| {dk.split('=')[1]} | {lab} | {pm(m['coverage'])} | {num(m['size']['mean'])} | {pct(m['empty']['mean'])} | "
              f"{pct(m['singleton_correct']['mean'])} | {pct(m['coverage|hard (IDV 3, 9, 15)']['mean'])} | "
              f"{num(m['size|hard (IDV 3, 9, 15)']['mean'])} | {pct(m['coverage|easy (IDV 1, 2, 4, 5, 6, 7)']['mean'])} | "
              f"{num(m['size|easy (IDV 1, 2, 4, 5, 6, 7)']['mean'])} | {pct(m['coverage|normal (IDV 0)']['mean'])} | "
              f"{num(m['size|normal (IDV 0)']['mean'])} |")
    w("")
    w("Class-conditional (Mondrian) sets need ceil((n_c + 1)(1 - delta)) <= n_c calibration windows per class; with one run per "
      "class (Braatz: 18-21 calibration windows per class) delta = 0.05 is infeasible and the set is every class.\n")
    m = hs["marginal"]["delta=0.1"]
    w("Marginal sets at delta = 0.1 by time since the fault entered (fault windows only): " + ", ".join(
        f"{k.split('|faults ')[1]}: coverage {pct(v['mean'])}, size {num(m[k.replace('coverage', 'size')]['mean'])}"
        for k, v in m.items() if k.startswith("coverage|faults")) + ".\n")
    w("What the marginal delta = 0.1 set contains for windows of a given true class (share of its windows):\n")
    for c, comp in m["set_members"].items():
        w(f"* IDV {c}: " + ", ".join(f"{k} {pct(v, 0)}" for k, v in comp.items()))
    w("")
    w("<details><summary>Per-class table (top-1 accuracy; marginal and class-conditional sets at delta = 0.1)</summary>\n")
    w("| IDV | top-1 acc | marginal coverage | marginal size | class-cond. coverage | class-cond. size |\n|---|---|---|---|---|---|")
    cc = hs["class_conditional"]["delta=0.1"]["by_class"]
    for c, v in m["by_class"].items():
        w(f"| {c} | {pct(hs['top1_acc_by_class'][c])} | {pct(v['coverage'])} | {num(v['size'])} | {pct(cc[c]['coverage'])} | {num(cc[c]['size'])} |")
    w("\n</details>\n")


def decision_section(L, dl, title):
    w = L.append
    w(f"### {title}\n")
    w(f"tau = {num(dl['tau'], 3)} (split-conformal, delta = {dl['delta']}, fitted on a separate 30% tau split of the groups); on "
      f"the decision windows it covers the true class on {pct(dl['tau_coverage_heldout'])} with mean plausible-set size "
      f"{num(dl['plausible_size_heldout'])}. {dl['n_decisions']} decisions in {dl['n_groups']} groups. Unsafe-label rates: "
      + ", ".join(f"{a} {pct(v)}" for a, v in dl["label_rates"].items()) + ". Hold is unsafe (recorded pressure > 2895 kPa "
      "within the next hour) in windows of " + ", ".join(f"IDV {c} ({pct(v, 0)})" for c, v in dl["hold_unsafe_by_class"].items() if _f(v) > 0)
      + f". Score AUROC (unsafe vs safe candidates): RiskGate {num(dl['auroc']['riskgate'], 3)}, top-1 {num(dl['auroc']['top1'], 3)}.\n")
    w("**Agnostic mode** (all three responses are candidates; unsafe = some certified response is unsafe):\n")
    w("| alpha | score | method | lambda | unsafe | 95% CI (pooled) | ask | safe-auto | re-splits: unsafe mean +- sd | share > alpha | "
      "re-splits: ask |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for ak, blk in dl["agnostic"].items():
        for k in ("top1", "riskgate"):
            r = blk[k]
            for m in ("none", "fixed", "crc", "ltt"):
                if k == "riskgate" and m == "none":
                    continue
                f, rs = r["fixed_split"][m], r["resplits"].get(m)
                w(f"| {ak.split('=')[1]} | {'top-1 twin' if k == 'top1' else 'RiskGate'} | {m} | {num(f['lambda'])} | {pct(f['unsafe'])} | "
                  f"[{pct(f['unsafe_ci'][0])}, {pct(f['unsafe_ci'][1])}] | {pct(f['ask'])} | {pct(f['safe_auto'])} | "
                  + (f"{pct(rs['unsafe_mean'])} +- {pct(rs['unsafe_sd'])} | {pct(rs['frac_above_alpha'], 0)} | {pct(rs['ask_mean'])} |"
                     if rs else " | | |"))
    w("")
    sw = {k: dl["agnostic"]["alpha=0.1"][k]["alpha_sweep"]["crc"] for k in ("top1", "riskgate")}
    w("Alpha sweep (agnostic CRC, fixed split; unsafe / ask): " + "; ".join(
        f"alpha {a['alpha']:g}: top-1 {pct(a['unsafe'])} / {pct(a['ask'])}, RiskGate {pct(b['unsafe'])} / {pct(b['ask'])}"
        for a, b in zip(sw["top1"], sw["riskgate"])) + ".\n")
    b = dl["policy"]["base"]
    w(f"**Policy mode** (scripted playbook doer; its first choice is unsafe in {pct(b['first_choice_unsafe'])} of decisions; first "
      f"choices: {b['first_choice_counts']}):\n")
    w("| alpha | method | lambda | unsafe | ask | safe-auto | re-splits: unsafe mean +- sd | share > alpha |\n|---|---|---|---|---|---|---|---|")
    for ak in ("alpha=0.05", "alpha=0.1"):
        r = dl["policy"][ak]
        for m in ("none", "fixed", "top1_crc", "riskgate", "riskgate_ltt"):
            f, rs = r["fixed_split"][m], r["resplits"].get(m)
            w(f"| {ak.split('=')[1]} | {m} | {num(f['lambda'])} | {pct(f['unsafe'])} | {pct(f['ask'])} | {pct(f['safe_auto'])} | "
              + (f"{pct(rs['unsafe_mean'])} +- {pct(rs['unsafe_sd'])} | {pct(rs['frac_above_alpha'], 0)} |" if rs else " | |"))
    w("")
    w("Per true class (agnostic CRC, alpha = 0.1, 100 re-splits, test halves; RiskGate / top-1 twin):\n")
    w("| IDV | unsafe certified | some safe response certified | hold certified | ask |\n|---|---|---|---|---|")
    rgc, t1 = dl["by_class"]["riskgate"], dl["by_class"]["top1"]
    for c in rgc:
        a, t = rgc[c], t1[c]
        w(f"| {c} | {pct(a['unsafe_certified'])} / {pct(t['unsafe_certified'])} | {pct(a['safe_certified'])} / {pct(t['safe_certified'])} | "
          f"{pct(a['hold_certified'])} / {pct(t['hold_certified'])} | {pct(a['ask'])} / {pct(t['ask'])} |")
    w("")
    if dl.get("examples"):
        w("Worked examples (agnostic RiskGate calibrated on the fixed split's calibration groups, alpha = 0.1):\n")
        for ex in dl["examples"]:
            w(f"* **IDV {ex['class']}**, {ex['hours_since_onset']:.0f} h after onset, reactor pressure {ex['p_now_kPa']:.0f} kPa; plausible "
              f"set (NLL): `{ex['plausible']}`")
            for r in ex["verdicts"]:
                w(f"  * `{r['action']}` ({'unsafe' if r['unsafe'] else 'safe'} in truth): **{r['verdict'].upper()}**, robust score "
                  f"{num(r['score'])} vs lambda {num(r['threshold'])}; {r['reason']}")
        w("")


def summary(L, res):
    """Headline tables and findings (all numbers from the results JSON)."""
    w = L.append
    hr, hb = res["rieth"]["hypothesis_sets"], res["braatz"]["hypothesis_sets"]
    w("## Summary\n")
    w("**Fault-hypothesis sets** (marginal split conformal; Rieth = runs as units, Braatz = time blocks of one run per fault):\n")
    w("| dataset | top-1 acc | delta | coverage | mean size | hard IDV 3/9/15: coverage / size | easy IDV 1/2/4/5/6/7: coverage / size | "
      "first 2-4 h after onset: coverage |\n|---|---|---|---|---|---|---|---|")
    for name, hs in (("Rieth", hr), ("Braatz", hb)):
        for dk in ("delta=0.05", "delta=0.1", "delta=0.2"):
            m = hs["marginal"][dk]
            w(f"| {name} | {pct(hs['top1_acc'])} | {dk.split('=')[1]} | {pm(m['coverage'])} | {num(m['size']['mean'])} | "
              f"{pct(m['coverage|hard (IDV 3, 9, 15)']['mean'])} / {num(m['size|hard (IDV 3, 9, 15)']['mean'])} | "
              f"{pct(m['coverage|easy (IDV 1, 2, 4, 5, 6, 7)']['mean'])} / {num(m['size|easy (IDV 1, 2, 4, 5, 6, 7)']['mean'])} | "
              f"{pct(m['coverage|faults 2-4 h']['mean'])} |")
    w("")
    w("**Decision layer** (agnostic mode, alpha = 0.1; every row is a mean over the same 200 group re-splits' test halves; "
      "uncalibrated rows need no calibration half but are evaluated on the same test halves):\n")
    w("| method | Braatz unsafe | Braatz ask | Rieth unsafe | Rieth ask |\n|---|---|---|---|---|")
    rows = (("none (certify every response)", "top1", "none"), ("top-1 twin gate (certify iff safe under the top-1 fault)", "top1", "fixed"),
            ("top-1 + CRC", "top1", "crc"), ("robust set, no calibration (lambda = 0)", "riskgate", "fixed"),
            ("RiskGate CRC", "riskgate", "crc"), ("RiskGate LTT (delta' = 0.1)", "riskgate", "ltt"))
    for label, k, m in rows:
        cells = []
        for ds in ("braatz", "rieth"):
            r = res[ds]["decision"]["agnostic"]["alpha=0.1"][k]
            rs = r["resplits"].get(m)
            cells += [pct(rs["unsafe_mean"]), pct(rs["ask_mean"])] if rs else [pct(r["fixed_split"][m]["unsafe"]), pct(r["fixed_split"][m]["ask"])]
        w(f"| {label} | " + " | ".join(cells) + " |")
    w("")
    mr = hr["marginal"]["delta=0.1"]
    cr = hr["class_conditional"]["delta=0.1"]
    dr = res["rieth"]["decision"]["agnostic"]
    sw = {k: {x["alpha"]: x for x in dr["alpha=0.1"][k]["alpha_sweep"]["crc"]} for k in ("top1", "riskgate")}
    st = res.get("rieth_step_check", {})
    w("**Findings.**\n")
    w("1. With runs as the exchangeable units (Rieth), marginal split-conformal coverage is on target: "
      + ", ".join(f"{pm(hr['marginal'][dk]['coverage'])} at delta = {dk.split('=')[1]}" for dk in ("delta=0.05", "delta=0.1", "delta=0.2"))
      + ". The Braatz time-block numbers look similar but rest on one run per fault.")
    w(f"2. Humility where the data are silent: for the undetectable IDV 3, 9 and 15 the delta = 0.1 set holds "
      f"{num(mr['size|hard (IDV 3, 9, 15)']['mean'], 1)} classes on average (typically {{0, 3, 9, 15}}, often IDV 4), against near-singletons "
      f"for IDV 1, 2, 6, 7 (per-class sizes {', '.join(num(mr['by_class'][c]['size']) for c in ('1', '2', '6', '7'))}).")
    w(f"3. Marginal coverage is not conditional coverage: at delta = 0.1 the hard faults are covered "
      f"{pct(mr['coverage|hard (IDV 3, 9, 15)']['mean'])}, "
      f"normal operation {pct(mr['coverage|normal (IDV 0)']['mean'])}, IDV 5 {pct(mr['by_class']['5']['coverage'])}, the first 2-4 h after "
      f"onset {pct(mr['coverage|faults 2-4 h']['mean'])}. Class-conditional calibration restores ~90% per class "
      f"(hard {pct(cr['coverage|hard (IDV 3, 9, 15)']['mean'])}) at the price of larger sets (hard {num(cr['size|hard (IDV 3, 9, 15)']['mean'], 1)} "
      "classes).")
    if st:
        w(f"4. IDV 4 and 5 are 'easy' detection faults, but this frozen scorer confuses them with normal operation on part of the Rieth "
          f"runs (top-1 {pct(hr['top1_acc_by_class']['4'])} and {pct(hr['top1_acc_by_class']['5'])}; Braatz: "
          f"{pct(hb['top1_acc_by_class']['4'])} and {pct(hb['top1_acc_by_class']['5'])}). The steps are in the data (post-hoc check, "
          f"50 runs: reactor cooling-water flow XMV 10 {st['IDV0']['XMV10_mean']} -> {st['IDV4']['XMV10_mean']} under IDV 4, condenser "
          f"flow XMV 11 {st['IDV0']['XMV11_mean']} -> {st['IDV5']['XMV11_mean']} under IDV 5; sample sd {st['sample_sd_normal']['XMV10']} and "
          f"{st['sample_sd_normal']['XMV11']}), so this is an efficiency loss of the scorer, which conformal calibration turns into "
          "larger sets rather than wrong certainties.")
    w(f"5. Decision layer (Rieth, alpha = 0.1): certifying everything is unsafe in "
      f"{pct(dr['alpha=0.1']['top1']['fixed_split']['none']['unsafe'])} of decisions; every gate stays far below alpha. RiskGate "
      f"lowers the unsafe rate relative to top-1 + CRC ({pct(dr['alpha=0.1']['riskgate']['resplits']['crc']['unsafe_mean'])} vs "
      f"{pct(dr['alpha=0.1']['top1']['resplits']['crc']['unsafe_mean'])}) and pays with asks "
      f"({pct(dr['alpha=0.1']['riskgate']['resplits']['crc']['ask_mean'])} vs {pct(dr['alpha=0.1']['top1']['resplits']['crc']['ask_mean'])}): "
      f"a trade-off, not a dominance. At alpha = 0.01 both calibrations move lambda-hat to ~1 and ask on "
      f"{pct(sw['riskgate'][0.01]['ask'])} (RiskGate) and {pct(sw['top1'][0.01]['ask'])} (top-1) of the fixed split's test decisions. "
      "Residual unsafe certifications come from the hold model under-predicting the 1-h pressure rise of IDV 6/18 runs and "
      "from misidentified IDV 16/18 windows (per-class table below).\n")


def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    w = L.append
    meta = res["meta"]
    w("# Tennessee Eastman process: calibrated fault-hypothesis sets and a small RiskGate decision layer\n")
    w("Generated by `benchmarks/tep/run.py` from `results/tep_results.json`.\n")
    w("## What is real, what is rule-derived, what is simulated\n")
    w("| Component | Status |\n|---|---|")
    w("| Plant data | **simulated** closed-loop Tennessee Eastman process (Downs & Vogel model), not a real plant. Braatz data: one run "
      "per fault (IDV 0-21). Rieth et al. 2017: 500 independent runs per fault (IDV 0-20). Fault labels are the simulator's. |")
    w("| Fault-hypothesis scorer | LLM-free regularised Gaussian discriminant (RDA) on 2-h window statistics, fitted on TRAINING "
      "files only; design fixed by cross-validation on the Braatz training files |")
    w("| Plausible sets | split conformal on held-out test windows (Braatz: time blocks; Rieth: runs) |")
    w("| `hold` labels | **data-derived**: the recorded run is the no-intervention outcome; unsafe iff reactor pressure exceeds the "
      "Downs & Vogel normal operating limit (2895 kPa) within the next hour |")
    w("| `rcw_up`, `feed_cut` labels and effects | **rule-derived** effect table from the IDV definitions (below) |")
    w("| Doer (policy mode) | **scripted** playbook on the scorer's top-1 diagnosis |\n")
    w(f"Licences: {meta['braatz']}. {meta['rieth']}. No data are redistributed (the Rieth cache under `benchmarks/tep/cache/` is "
      "git-ignored).\n")
    w("> [!IMPORTANT]\n> **Exchangeability.** The Braatz data contain ONE run per fault: calibration and evaluation windows of a fault "
      "come from the same run (split by contiguous time blocks), so Braatz coverage numbers are optimistic for a new occurrence of "
      "that fault. The Rieth study splits by run (500 runs per fault available; 100 used for the sets, 30 for decisions) and is "
      "the one to quote.\n")
    w("## Data facts checked\n")
    oc = res["braatz_onset_check"]
    w("* `d00.dat` is stored transposed (52 x 500); all other Braatz files are samples x 52.")
    w("* Braatz training files: the fault is active from (almost) the first sample, not after sample 20. Mean |z| vs normal "
      f"operation (normal run: {num(oc['normal_d00'])}) for samples 1-20 / 21-40 / 41-480 / sample 1: " + "; ".join(
          f"{k} {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f} / {v[3]:.2f}" for k, v in oc.items() if k in ("IDV1", "IDV2", "IDV6", "IDV7", "IDV5"))
      + ". Samples 1-20 of each faulty training file are dropped as an ambiguous transient.")
    ro = res["rieth_onset_check"]
    w("* Rieth training runs: onset after sample 20 as documented (mean |z|, samples 1-20 / 21-40 / 41-500, 20 runs: " + "; ".join(
        f"{k} {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}" for k, v in ro.items() if k != "normal") + f"; normal {num(ro['normal'])}).")
    w("* Test runs (both datasets): onset after sample 160 confirmed (mean |z| jumps between samples 140-160 and 160-180).\n")
    summary(L, res)
    w("## Design (fixed before any test file was scored)\n")
    w(DESIGN_CV + "\n")
    d = meta["design"]
    w(f"Final: window {d['window_samples']} samples, a decision every {d['decision_stride']} samples (1 h), {d['features']}, "
      f"{d['classifier']}; nonconformity r(h) = -log p(h | window). Hypothesis set = {{h : r(h) <= tau}}.\n")
    w("## 1. Calibrated fault-hypothesis sets (the real-data content)\n")
    sets_section(L, res["braatz"]["hypothesis_sets"], "Braatz: one run per fault, splits by 3-h time blocks (200 stratified re-splits, 50/50)",
                 "Caveat: blocks of one run are not exchangeable with a new run; adjacent windows overlap by 1 h.")
    ts = res["braatz"]["temporal_split"]
    w("Non-stationarity check (Braatz, no randomness): " + "; ".join(
        f"{k}: " + ", ".join(f"delta {dk.split('=')[1]} coverage {pct(v['coverage'])} size {num(v['size'])}" for dk, v in row.items())
        for k, row in ts.items()) + ".\n")
    sets_section(L, res["rieth"]["hypothesis_sets"],
                 "Rieth: runs as exchangeable units (testing runs 1-100 per class, 200 stratified run re-splits, 50/50)",
                 "Scorer refitted on Rieth training runs 1-50 per class with the same design; IDV 21 is not in this dataset.")
    w(f"Transfer check: the scorer fitted on the Braatz training files (one run per fault) reaches top-1 accuracy "
      f"{pct(res['rieth']['braatz_scorer_top1_acc'])} on the Rieth testing runs.\n")

    w("## 2. Decision layer (small, partly rule-derived)\n")
    w("Candidate responses to an upset: `hold` (keep set-points), `rcw_up` (raise the reactor cooling-water flow set-point), "
      "`feed_cut` (reduce the A/C feed, stream 4); `escalate` is the gate's ASK and is never gated. RiskGate certifies a response "
      "only if its robust margin (worst case over the plausible fault set) clears the calibrated threshold.\n")
    w("| response | unsafe under (rule-derived unless stated) | gate's consequence model |\n|---|---|---|")
    w("| hold | **data**: recorded reactor pressure > 2895 kPa within 1 h | p_now + q95 of the 1-h pressure rise under the hypothesised "
      "class in the TRAINING runs, vs 2895 kPa (margin in 100 kPa) |")
    for a, cl in d["unsafe_active"].items():
        w(f"| {a} | IDV {', '.join(map(str, cl))} | same table (+1 safe / -1 unsafe) |")
    w("")
    for c, note in d["rule_notes"].items():
        w(f"* IDV {c}: {note}.")
    w("")
    decision_section(L, res["braatz"]["decision"], "Braatz (groups = 3-h time blocks)")
    decision_section(L, res["rieth"]["decision"], "Rieth (groups = runs; testing runs 101-130 per class)")
    w("P9: every fixed-split lambda-hat above (agnostic and policy, CRC and LTT) was asserted equal to "
      "`riskgate.Gate.calibrate(cal_eps, groups=..., tau_fraction=0, mode=...)` when the run was produced.\n")
    w("## Caveats\n")
    w("* **Simulated plant.** Both datasets come from the Tennessee Eastman simulator; fault magnitudes, sensor noise and the "
      "control structure are the simulator's. Nothing here is evidence about a real plant.")
    w("* **Braatz exchangeability.** One run per fault; time blocks of one run are dependent and non-stationary (transients "
      "after onset), adjacent windows share samples. Rieth results (runs as units) are the defensible ones.")
    w("* **Conditional coverage.** Marginal conformal coverage is an average over classes and times: hard faults (IDV 3, 9, 15) "
      "and the first hours after onset are under-covered unless sets are calibrated per class, which inflates the sets of "
      "normal-looking windows to several classes (the honest answer: these faults are not identifiable from these data).")
    w("* **Decision layer is illustrative.** Active-response effects are a rule table, not a simulation; only `hold` uses "
      "recorded outcomes. Because the table makes most uncertainty fall between benign classes (0, 3, 9, 15), the top-1 twin "
      "gate is already competitive here; the layer shows how the calibrated set enters the decision, not a plant-ready policy.")
    w("* **Design reuse.** Hyper-parameters were chosen on the Braatz training files and reused for Rieth (refitted on Rieth "
      "training runs); no tuning on test files.")
    path.write_text("\n".join(L) + "\n")
