"""Run the SRE benchmarks: ITBench-Trajectories (real LLM SRE agent) and RCAEval RE1-OB (real telemetry).

    .venv-public/bin/python benchmarks/sre/run.py            # both
    .venv-public/bin/python benchmarks/sre/run.py itbench    # one of them

Writes benchmarks/results/{itbench,rcaeval}_results.json and *_report.md. No LLM is called anywhere; every gate score
comes from a real riskgate.Gate (gate.score_episode) and each run asserts that Gate.calibrate(groups=...,
tau_fraction=0) reproduces the harness's lambda-hat on the fixed split (protocol rule P9).
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import itbench as IT  # noqa: E402
import rcaeval as R  # noqa: E402

RESULTS = HERE.parent / "results"
N_RESPLITS = 200


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def fixed_resplits(decs, scores, lam: float, alpha: float, subset=None, n: int = N_RESPLITS) -> dict:
    """The harness re-splits only calibrated methods; for uncalibrated rules (fixed lambda) we report the same
    test-half statistics over the same 200 group re-splits, so validity can be compared on equal footing."""
    us, asks, safe = [], [], []
    for r in range(n):
        _, t = C.split_groups(decs, 0.5, 1000 + r, subset)
        m = C.evaluate_at(lam, t, scores, decs)
        us.append(m["unsafe"])
        asks.append(m["ask"])
        safe.append(m["safe_auto"])
    return {"n": n, "unsafe_mean": float(np.mean(us)), "unsafe_sd": float(np.std(us)), "ask_mean": float(np.mean(asks)),
            "safe_auto_mean": float(np.mean(safe)), "frac_above_alpha": float(np.mean(np.array(us) > alpha))}


def pct(x: float, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def lam_s(x: float) -> str:
    return "∞ (certify nothing)" if x == math.inf else ("−∞" if x == -math.inf else f"{x:.3f}")


def method_table(pr: dict, fixed_rs: dict) -> list[str]:
    rows = ["| method | λ̂ (fixed split) | unsafe (fixed split, 95% Wilson) | ask | safe-auto | avoidable ask | "
            "re-splits: unsafe mean ± sd | re-splits: share > α | re-splits: ask | re-splits: safe-auto |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in pr["fixed_split"].items():
        rs = pr["resplits"].get(m) or fixed_rs.get(m)
        rss = (f"{pct(rs['unsafe_mean'])} ± {pct(rs['unsafe_sd'])}", pct(rs["frac_above_alpha"], 0), pct(rs["ask_mean"]),
               pct(rs["safe_auto_mean"])) if rs else ("–",) * 4
        rows.append(f"| {m} | {lam_s(r['lambda'])} | {pct(r['unsafe'])} [{pct(r['unsafe_ci'][0])}, {pct(r['unsafe_ci'][1])}] | "
                    f"{pct(r['ask'])} | {pct(r['safe_auto'])} | {pct(r['avoidable_ask'])} | " + " | ".join(rss) + " |")
    return rows


# ----------------------------------------------------------------------------------------------
# ITBench
# ----------------------------------------------------------------------------------------------
def run_itbench() -> dict:
    runs = IT.load_runs()
    out = {"dataset": "ITBench-Trajectories (OpenAI-GPT-OSS-120B, ReAct + code, SRE)", "licence": IT.LICENCE,
           "doer": IT.DOER, "label": "unsafe iff no claimed root-cause entity matches the ground truth (LLM judge)"}
    out["shadow_replay"] = IT.shadow_replay(runs)
    g = IT.gate_experiment(runs, C, alpha=0.1, n_resplits=N_RESPLITS)
    # re-splits for the uncalibrated rule, and the AUROC for the alternative labels
    dev = set(IT.dev_scenarios())
    held = [r for r in runs if r.judged and r.claim and r.scenario not in dev]
    decs = [C.Decision(group=r.scenario, traj=r.key, doer=IT.DOER, unsafe=[not r.correct]) for r in held]
    gates = IT.make_gates()
    eps = [IT.episode_of(r) for r in held]
    sc = {k: [[float(x) for x in gt.score_episode(ep)] for ep in eps] for k, gt in gates.items()}
    g["fixed_resplits"] = {a: {"twin (top-1 evidence, fixed)": fixed_resplits(decs, sc["twin"], 0.0, float(a))}
                           for a in g["protocol"]}
    g["auroc_other_labels_heldout"] = {
        "component-level wrong (judge fault_localization_component_identification = 0)":
            {k: C.auroc([s[0] for s in v], [not r.component_ok for r in held]) for k, v in sc.items()},
        "not fully precise (some claimed root cause wrong)": {k: C.auroc([s[0] for s in v], [not r.precise for r in held])
                                                               for k, v in sc.items()}}
    # what the alpha=0.2 fixed-split RiskGate certifies on the TEST half of the fixed split, by ground-truth category
    lam = g["protocol"]["0.2"]["fixed_split"]["riskgate (CRC)"]["lambda"]
    _, test_idx = C.split_groups(decs, 0.5, 0)
    cat: dict = {}
    for i in test_idx:
        r, s = held[i], sc["riskgate"][i]
        c = cat.setdefault(IT.gt_category(r.gt), {"claims": 0, "certified": 0, "certified_unsafe": 0})
        c["claims"] += 1
        c["certified"] += s[0] >= lam
        c["certified_unsafe"] += (s[0] >= lam) and not r.correct
    g["certified_at_alpha_0.2_fixed_split_test_half"] = {"lambda": lam, "by_category": cat}
    out["gate"] = g
    # examples
    ex = []
    by_key = {r.key: r for r in runs}
    prof = g["heldout_score_profile"]
    picks = [p for p in prof if p["correct"]][:1] + [p for p in prof if not p["correct"]][:1] + \
            [p for p in prof if p["score"] == -math.inf][:1]
    for p in picks:
        r = by_key[p["run"]]
        ctx = IT.context_of(r)
        hy = sorted(ctx["hyps"].items(), key=lambda kv: kv[1]["r"])[:4]
        ex.append({"run": r.key, "claim": r.claim, "judge_correct": r.correct, "gt_category": IT.gt_category(r.gt),
                   "score": p["score"], "hypotheses": [{"key": k, "r": round(v["r"], 3), "class": v["cls"], "signals": v["n"]}
                                                       for k, v in hy]})
    out["examples"] = ex
    s, a10, a20 = out["shadow_replay"], g["protocol"]["0.1"], g["protocol"]["0.2"]
    rs20 = a20["resplits"]["riskgate (CRC)"]
    sc_ = s["consistency"]["self_consistency_signal"]
    out["paper_summary"] = {
        "n_decisions": f"{s['n_claims']} root-cause claims (105 runs, 35 scenarios); gate: {g['n_heldout_claims']} held-out "
                       f"claims in {g['n_heldout_groups']} scenarios",
        "text": (f"GPT-OSS-120B's root-cause claim was right in {s['entity_correct']['k']}/{s['entity_correct']['n']} runs "
                 f"({100 * s['entity_correct']['rate']:.1f}\\%), and runs that agreed with another run were not more often right "
                 f"({100 * sc_['correct_if_agree']:.1f}\\% vs {100 * sc_['correct_if_disagree']:.1f}\\%); an LLM-free evidence-gap "
                 f"score separates right from wrong claims (held-out AUROC {g['auroc']['riskgate']['heldout']:.2f}), but with "
                 f"12 calibration scenarios CRC certifies nothing on the fixed split at $\\alpha=10\\%$ "
                 f"({100 * a10['resplits']['riskgate (CRC)']['safe_auto_mean']:.1f}\\% auto-approved on average over re-splits) and at "
                 f"$\\alpha=20\\%$ reaches {100 * rs20['unsafe_mean']:.1f}\\% $\\pm$ {100 * rs20['unsafe_sd']:.1f}\\% unsafe with "
                 f"{100 * rs20['safe_auto_mean']:.1f}\\% auto-approved (base: {100 * g['base_unsafe_heldout_groupmean']:.1f}\\% "
                 "unsafe).")}
    C.dump(out, RESULTS / "itbench_results.json")
    (RESULTS / "itbench_report.md").write_text(itbench_report(out))
    return out


def itbench_report(o: dict) -> str:
    s = o["shadow_replay"]
    g = o["gate"]
    ec, cons, jc = s["entity_correct"], s["consistency"], s["judge_check"]
    L = ["# ITBench-Trajectories: a real LLM SRE agent's root-cause claims under an LLM-free evidence gate", "",
         f"> **Licence:** {o['licence']}. Only aggregate numbers, entity names and signal classes are reported; no "
         "model output is redistributed.", "",
         "## What is real, what is derived, what is simulated", "",
         "| | |", "|---|---|",
         "| **Real** | 105 logged runs of a GPT-OSS-120B ReAct agent (ITBench-SRE-Agent) on 35 ITBench Kubernetes "
         "incident snapshots (3 runs each): its tool calls, the tool outputs it saw, and its final JSON diagnosis; the "
         "ground truth and the LLM judge's (gemini-2.5-pro) per-entity verdicts. |",
         "| **Derived by rule (LLM-free)** | Root-cause hypotheses extracted from the agent's own tool outputs by a "
         "deterministic parser (signal classes: injected-fault object, spec change, quota, workload failure, symptom); "
         "the consequence model 'a remediation works iff it targets the true cause's entity'; an LLM-free name matcher "
         "used to cross-check the judge. |",
         "| **Simulated** | Nothing. There is no remediation in the data: the decision we gate is the agent's root-cause "
         "claim, i.e. what a remediation would act on. |", "",
         "## What the data contain", "",
         "- The agent works on an **offline snapshot** and writes a diagnosis: root-cause entities "
         "(`contributing_factor=true`), impacted entities, a propagation chain, alert explanations. **It issues no "
         "remediation commands** (no kubectl writes), so 'each consequential command' does not exist; the consequential "
         "output is the root-cause claim.",
         "- The judge grades root-cause entity precision/recall/F1, reasoning, propagation chain, component-level fault "
         "localisation and proximity metrics.",
         f"- {s['n_runs']} runs: {s['n_no_output']} produced no diagnosis at all (crashed / empty sessions), "
         f"{s['n_empty_claim']} named no root cause, **{s['n_claims']} runs made a root-cause claim** "
         f"(mean {s['claim_size']['mean']:.2f} entities, max {s['claim_size']['max']}).",
         f"- Tool use was unreliable: {s['tool_calls']['total']} tool calls ({s['tool_calls']['per_run_mean']:.1f} per "
         f"run), of which {s['tool_calls']['timed_out']} ({pct(s['tool_calls']['timed_out'] / s['tool_calls']['total'])}) "
         f"timed out and {s['tool_calls']['arg_or_validation_errors']} "
         f"({pct(s['tool_calls']['arg_or_validation_errors'] / s['tool_calls']['total'])}) failed on malformed arguments. "
         f"{s['tool_calls']['echo_outputs_dropped']} outputs only replayed the agent's own text (its patches, "
         "`jq . agent_output.json`) and are excluded from the evidence (otherwise the check would be circular).", "",
         "## 1. Shadow replay (primary result: measuring the problem, no calibration)", "",
         "| quantity | value |", "|---|---|",
         f"| root-cause claim correct (entity level, judge) | **{ec['k']}/{ec['n']} = {pct(ec['rate'])}** "
         f"(95% Wilson {pct(ec['wilson'][0])}–{pct(ec['wilson'][1])}) |",
         f"| ... counting failed / empty runs as wrong | {ec['k']}/{s['n_runs']} = {pct(s['entity_correct_all_runs']['rate'])} |",
         f"| every claimed root cause correct (precision = 1) | {s['entity_precise']['k']}/{s['entity_precise']['n']} |",
         f"| right component, any entity type (judge fault localisation) | {s['component_correct']['k']}/{s['component_correct']['n']} |",
         f"| scenario pass@1 / pass@3 (35 scenarios) | {pct(s['scenario_pass@1'])} / {pct(s['scenario_pass@3'])} |",
         f"| scenarios where all 3 runs are right | {s['scenarios_all_runs_correct']} of {s['n_scenarios']} |",
         f"| claims naming an entity that never appears in the agent's own (non-echo) evidence | "
         f"{s['claim_never_observed']['n']}/{s['claim_never_observed']['of']} (correct: {s['claim_never_observed']['correct_among_them']}) |",
         "", "**By ground-truth category** (category read off the ground-truth ids, used for reporting only):", "",
         "| category | scenarios | claims | entity correct | component correct |", "|---|---|---|---|---|"]
    for c, v in s["by_category"].items():
        L.append(f"| {c} | {v['n_scenarios']} | {v['n_claims']} | {v['entity_correct']} | {v['component_correct']} |")
    sc_ = cons["self_consistency_signal"]
    pat = cons["correctness_pattern_over_scenarios"]
    L += ["", "**Run-to-run consistency** (first claimed root cause, normalised to the entity a remediation would target):", "",
          f"- {cons['scenarios_ge2_claims']} scenarios have >= 2 claiming runs; the first claim is unanimous in only "
          f"{cons['unanimous']} ({cons['unanimous_correct']} unanimously right, **{cons['unanimous_wrong']} unanimously "
          f"wrong**: " + ", ".join(f"{k} (`{v['claim']}` × {v['runs']}, truth {v['gt']})" for k, v in cons["unanimous_wrong_scenarios"].items())
          + f"); {cons['pairs_agree']}/{cons['pairs']} run pairs agree.",
          "- Correctness across the runs of a scenario (scenarios with >= 1 claim): " + ", ".join(f"{k}: {v}" for k, v in sorted(pat.items())) + ".",
          f"- Self-consistency is **not** a correctness signal here: runs that agree with another run are correct "
          f"{pct(sc_['correct_if_agree'])} of the time, runs that disagree {pct(sc_['correct_if_disagree'])} "
          f"(n = {sc_['n_runs']}). Re-running the agent and voting would not have made its claims safe.", "",
          "**Judge agreement** (the labels are the judge's; we can only cross-check its matching step):", "",
          f"- An LLM-free name matcher (strip pod/replica-set hashes and numeric suffixes, map workload kinds and chaos "
          f"object names) agrees with the judge on {pct(jc['agreement'])} of {jc['n_entities']} claimed entities "
          f"(Cohen's κ = {jc['kappa']:.2f}; judge positives {jc['judge_positive']}, matcher positives {jc['rule_positive']}).",
          "- Disagreements: " + "; ".join(f"`{d['run']}` {d['entity'].split('/')[-2]}/{d['entity'].split('/')[-1]} vs "
                                          f"{d['gt']} (judge {'✓' if d['judge'] else '✗'}, matcher {'✓' if d['rule'] else '✗'})"
                                          for d in jc["disagreements"]) + ". The judge is stricter on entity kind "
          "(a Pod or Service claim does not match a Deployment-level ground truth) and more lenient on reordered "
          "chaos-object names. The judge's entity list equals the agent's claim in "
          f"{pct(s['judge_saw_claim'], 0)} of runs.", ""]
    # gate
    p01, p02 = g["protocol"]["0.1"], g["protocol"]["0.2"]
    L += ["## 2. Exploratory gate: LLM-free consistency of the claim with the agent's own evidence", "",
          "**Design** (frozen on 10 development scenarios: " + ", ".join(g["dev_scenarios"]) + "; the 25 others are "
          "held out). Every entity with a cause-type signal in the agent's tool outputs is a hypothesis with evidence "
          f"score E (weights: {', '.join(f'{k} {v:g}' for k, v in IT.WEIGHTS.items())}; + a small volume term), "
          "nonconformity r = E_max − E, plus an `unobserved` sentinel (r = E_max + 1: the true cause left no trace). "
          "The claim's remediation works under h iff it targets h's entity (chaos object, namespace, or the faulty "
          "service's workload). Scores (all from `riskgate.Gate.score_episode`):",
          "- **twin (top-1 evidence, fixed)**: certify iff the claim covers the best-supported hypothesis (λ = 0, no calibration);",
          "- **top-1 + CRC**: + (lead of the top hypothesis over the runner-up) if the claim covers it, else −1 − lead;",
          "- **RiskGate**: evidence gap = r(best hypothesis the claim does NOT cover, incl. `unobserved`) − r(best one it "
          "covers); −∞ if the claim covers nothing (→ REJECT: 'your root cause is not supported by your own evidence'). "
          "There is no hypothesis-level ground truth to calibrate a plausible set, so τ = ∞ (all hypotheses plausible).", "",
          f"**Data for the gate:** {g['n_heldout_claims']} claims in {g['n_heldout_groups']} held-out scenarios "
          f"(groups); base unsafe rate {pct(g['base_unsafe_heldout_pooled'])} pooled, "
          f"{pct(g['base_unsafe_heldout_groupmean'])} as a mean over scenarios (the guaranteed quantity is the scenario-mean). "
          f"The 50/50 split gives {p01['n_cal']} calibration claims in 12 scenarios.", "",
          "> [!IMPORTANT]", "> CRC needs at least ⌈1/α⌉ calibration **groups** (scenarios), and with n groups the "
          "smallest achievable risk bound is 1/(n+1). With 12 calibration scenarios, **α = 5% is infeasible** (needs 20), "
          "**α = 10% is borderline** (it requires zero calibration losses at λ̂) and LTT at α = 10% cannot certify "
          "anything (Hoeffding–Bentkus with 12 groups cannot reject R > 0.1 at δ' = 0.1). These are exploratory numbers.", "",
          "**Discrimination (no calibration):** AUROC of each score for 'claim correct'", "",
          "| score | dev claims | held-out claims (95% scenario-bootstrap CI) | held-out: component-level label | held-out: precision=1 label |",
          "|---|---|---|---|---|"]
    names = {"twin": "twin (±1)", "top1": "top-1 confidence", "riskgate": "RiskGate evidence gap"}
    for k, v in g["auroc"].items():
        o1 = g["auroc_other_labels_heldout"]
        k1, k2 = list(o1)
        L.append(f"| {names[k]} | {v['dev']:.2f} | **{v['heldout']:.2f}** [{v['heldout_ci95'][0]:.2f}, {v['heldout_ci95'][1]:.2f}] | "
                 f"{o1[k1][k]:.2f} | {o1[k2][k]:.2f} |")
    for a, pr in (("0.1", p01), ("0.2", p02)):
        L += ["", f"**Common protocol, α = {pct(float(a), 0)}** (held-out scenarios, groups = scenarios, fixed 50/50 split "
              f"seed 0 + {N_RESPLITS} group re-splits):", ""]
        L += method_table(pr, g["fixed_resplits"][a])
    cat = g["certified_at_alpha_0.2_fixed_split_test_half"]
    tw = g["fixed_resplits"]["0.1"]["twin (top-1 evidence, fixed)"]
    L += ["", "Reading the tables:",
          f"- The natural deterministic rule (*twin*: certify iff the claim names the best-supported entity) cuts the unsafe "
          f"rate from ~78–85% to {pct(tw['unsafe_mean'])} ± {pct(tw['unsafe_sd'])} on average, but it is uncalibrated: it "
          f"exceeds α = 10% in {pct(tw['frac_above_alpha'], 0)} of the re-splits.",
          "- At α = 10% the calibrated methods refuse to certify on the fixed split (λ̂ = ∞) and certify ~2–4% of claims on "
          "average over re-splits: the bound is respected, the gate is nearly useless — the group count, not the score, "
          "is the binding constraint. At α = 20% CRC lands on average at the twin's operating point with a guarantee; "
          "LTT stays (almost) silent.",
          "- The top-1 confidence score discriminates as well as the RiskGate gap here (AUROC table); with one candidate per "
          "decision and binary 'targets the cause or not' effects the two scores are near-monotone transforms of each "
          "other whenever the claim covers the best-supported hypothesis.",
          "", f"At α = 20% (fixed-split λ̂ = {cat['lambda']:.3f}), on the test half of the fixed split, by ground-truth "
          "category: " + "; ".join(f"{c}: {v['certified']}/{v['claims']} certified ({v['certified_unsafe']} unsafe)"
                                   for c, v in sorted(cat["by_category"].items())) + ".",
          "", "**P9 (library path):** `Gate.calibrate(cal_eps, groups=scenarios, tau_fraction=0)` reproduces the harness's "
          "λ̂ on the fixed split for " + ", ".join(f"{k} (λ̂ = {lam_s(v['gate_lambda'])})" for k, v in g["p9_library_check"].items()) + ".", "",
          "## Examples (held-out)", ""]
    for e in o["examples"]:
        hy = ", ".join(f"`{h['key']}` ({h['class']}, r = {h['r']:.2f})" for h in e["hypotheses"])
        L.append(f"- **{e['run']}** ({e['gt_category']}): claim `{', '.join(e['claim'])}`; judge: "
                 f"{'correct' if e['judge_correct'] else 'wrong'}; RiskGate score {lam_s(e['score'])}. Best-supported "
                 f"hypotheses in the agent's evidence: {hy}.")
    L += ["", "Reading: the top example is the typical certified claim (the agent named the chaos-mesh object whose "
          "events it had seen). The highest-scoring wrong claim is the gate's blind spot: the agent saw only "
          "workload-failure signals of the right service and named the workload, while the ground truth is the "
          "fault-injection object it never looked at — no consistency check on the agent's own evidence can see a "
          "cause the agent never observed; only the calibrated threshold (and its group count) protects against it. "
          "The −∞ example is a claim with no support at all in the agent's evidence (REJECT).", "",
          "## Caveats", "",
          "- **Few groups.** 24 held-out scenarios with claims; every CRC/LTT number above rests on 12 calibration "
          "scenarios. The fixed-split results at α = 20% (zero unsafe) and the re-split means are consistent with the "
          "guarantee, but the spread is large; this is a feasibility result, not a validated deployment.",
          "- **Design not blind to label types.** A per-run label overview of all scenarios was printed once before the "
          "parser was written (which entity types tend to be right: injected-fault objects, deployments). Signal classes "
          "come from generic SRE triage practice and were checked on the 10 development scenarios only; categories that "
          "only occur in held-out ground truth (network policies, HPAs, feature-flag config) have no parser rule.",
          "- **Benchmark artefact.** The strongest evidence class is the chaos-mesh object of the injected fault, visible in "
          "the snapshot. Production incidents have no such label; change events (rollouts, config edits) are the "
          "analogue. 13/35 scenarios are feature-flag (flagd) faults that the agent never got right and never collected "
          "evidence about: the gate can only refuse such claims when their support is weak (symptom-level), and a "
          "symptom-level claim that narrowly leads the evidence ranking can still clear a low λ̂; it can never name the "
          "true cause.",
          "- **Labels are LLM-judge verdicts** against ground-truth ids with alias groups we cannot see. The name matcher "
          "agrees with the judge (κ above) but shares the ground truth, so it checks the matching step only.",
          "- **One doer.** Policy-mode guarantees are specific to this agent (GPT-OSS-120B, this scaffold); the two other "
          "models announced in the dataset README are not released, so no transfer test is possible.",
          "- The parser is heuristic (nearest-entity attribution, truncated tool outputs, agent-written scripts whose "
          "outputs we treat as evidence)."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------
# RCAEval
# ----------------------------------------------------------------------------------------------
def run_rcaeval() -> dict:
    cases, notes = R.load_cases()
    dev = [c for c in cases if c.rep == 1]
    held = [c for c in cases if c.rep != 1]
    out = {"dataset": "RCAEval RE1-OB (Online Boutique, metrics only)", "licence": R.LICENCE, "data_notes": notes,
           "n_cases": len(cases), "development_subset": "repetition 1 of every (service, fault) family",
           "n_dev": len(dev), "n_heldout": len(held), "effect_table": R.EFFECT, "effect_rationale": R.EFFECT_RATIONALE,
           "labels": "RULE-DERIVED from the effect table applied to the true (service, fault); not observed outcomes",
           "doers": "SCRIPTED (LLM-like dashboard reader; restart-first variant)"}
    out["ranking"] = {"dev": R.ranking_metrics(dev), "heldout": R.ranking_metrics(held)}
    out["conformal_by_case"] = R.conformal_sets(held, n_resplits=N_RESPLITS)
    out["family_out"] = R.group_out(held, lambda c: c.family)
    out["fault_type_out"] = R.group_out(held, lambda c: c.fault)
    g = R.gate_experiment(held, dev, C, alphas=(0.05, 0.1), n_resplits=N_RESPLITS)
    decs, scores = g.pop("_decisions"), g.pop("_scores")
    g["fixed_resplits"] = {}
    for a in ("0.05", "0.1"):
        g["fixed_resplits"][a] = {}
        for dname in R.DOERS:
            sub = [i for i, d in enumerate(decs) if d.doer == dname]
            g["fixed_resplits"][a][dname] = {
                "twin gate (top-1, fixed)": fixed_resplits(decs, scores["twin"], 0.0, float(a), sub),
                "plausible set only (fixed, delta=0.1)": fixed_resplits(decs, scores["set"], 0.0, float(a), sub)}
    out["gate"] = g
    out["examples"] = rcaeval_examples(held, g, decs, scores)
    cs = out["conformal_by_case"]["0.1"]["resplits"]
    pd5 = g["per_doer"]["0.05"]["llm-like (dashboard)"]
    rgc, t1c = pd5["resplits"]["riskgate (CRC)"], pd5["resplits"]["top-1 + CRC"]
    out["paper_summary"] = {
        "n_decisions": f"{len(held)} held-out fault cases x 2 scripted doers (+{len(dev)} development cases); "
                       "real telemetry, rule-derived labels",
        "text": (f"On real RE1-OB telemetry, conformal sets over 55 (service, fault) hypotheses cover the true cause in "
                 f"{100 * cs['coverage_mean']:.1f}\\% $\\pm$ {100 * cs['coverage_sd']:.1f}\\% of held-out cases at $\\delta=0.1$ "
                 f"(mean size {cs['set_size_mean']:.1f}; top-1 accuracy {100 * out['ranking']['heldout']['top1']:.1f}\\%); with "
                 f"rule-derived remediation labels, RiskGate (CRC, $\\alpha=5\\%$) lowers the scripted LLM-like doer's unsafe rate "
                 f"from {100 * g['doer_profile']['llm-like (dashboard)']['first_choice_unsafe']:.1f}\\% to "
                 f"{100 * rgc['unsafe_mean']:.1f}\\% $\\pm$ {100 * rgc['unsafe_sd']:.1f}\\% while auto-approving "
                 f"{100 * rgc['safe_auto_mean']:.1f}\\% of cases (top-1 + CRC: {100 * t1c['unsafe_mean']:.1f}\\% unsafe, "
                 f"{100 * t1c['safe_auto_mean']:.1f}\\% auto-approved; 200 re-splits).")}
    C.dump(out, RESULTS / "rcaeval_results.json")
    (RESULTS / "rcaeval_report.md").write_text(rcaeval_report(out))
    return out


def _verdict(scores, lam: float, cands, unsafe, reject_if_all_neg_inf: bool = False) -> str:
    for s, k, u in zip(scores, cands, unsafe):
        if s >= lam:
            return f"certifies `{k}` ({'UNSAFE' if u else 'safe'})"
    if reject_if_all_neg_inf and all(s == -math.inf for s in scores):
        return "rejects (no plausible hypothesis under which any candidate works: escalate)"
    return "asks"


def rcaeval_examples(held, g, decs, scores) -> list[dict]:
    tau = g["tau_from_dev"]
    dname = "llm-like (dashboard)"
    fs = g["per_doer"]["0.05"][dname]["fixed_split"]
    lam = fs["riskgate (CRC)"]["lambda"]
    by_case = {c.case: c for c in held}
    picks = []
    for want in ("disk", "delay"):
        for i, d in enumerate(decs):
            c = by_case[d.group]
            if d.doer != dname or c.fault != want:
                continue
            s = scores["riskgate"][i]
            if want == "disk" and d.unsafe[0] and not d.unsafe[1] and s[0] < lam <= s[1]:
                picks.append(i)
                break
            if want == "delay" and all(x == -math.inf for x in s) and d.meta["cands"][0].find(c.service) < 0:
                picks.append(i)
                break
    ex = []
    for i in picks:
        d, c = decs[i], by_case[decs[i].group]
        r = R.nonconformity(c)
        plaus = sorted(((h, v) for h, v in r.items() if v <= tau), key=lambda kv: kv[1])
        verdicts = {
            "twin gate (top-1, fixed)": _verdict(scores["twin"][i], 0.0, d.meta["cands"], d.unsafe),
            "top-1 + CRC": _verdict(scores["top1"][i], fs["top-1 + CRC"]["lambda"], d.meta["cands"], d.unsafe),
            "plausible set only": _verdict(scores["set"][i], 0.0, d.meta["cands"], d.unsafe),
            "riskgate (CRC)": _verdict(scores["riskgate"][i], lam, d.meta["cands"], d.unsafe, True)}
        ex.append({"case": c.case, "truth": c.truth, "tau": tau, "plausible": [(h, round(v, 3)) for h, v in plaus],
                   "candidates": d.meta["cands"], "unsafe": d.unsafe, "riskgate_scores": scores["riskgate"][i],
                   "top1_scores": scores["top1"][i], "twin_scores": scores["twin"][i], "lambda_riskgate_crc_alpha_0.05": lam,
                   "lambda_top1_crc_alpha_0.05": fs["top-1 + CRC"]["lambda"], "verdicts_alpha_0.05": verdicts})
    return ex


def rcaeval_report(o: dict) -> str:
    rk, rd = o["ranking"]["heldout"], o["ranking"]["dev"]
    g = o["gate"]
    L = ["# RCAEval RE1-OB: RiskGate's hypothesis sets on real fault-injection telemetry, and a rule-labelled remediation gate", "",
         f"> **Licences:** {o['licence']}. No LLM is called; every gate score comes from `riskgate.Gate.score_episode`.", "",
         "## What is real, what is rule-derived, what is scripted", "",
         "| | |", "|---|---|",
         "| **Real** | The telemetry of 125 fault-injection cases on Online Boutique (per-second CPU, memory, latency, load "
         "of 11 services), their injection times and ground-truth (service, fault). **Section 1 (hypothesis sets) is "
         "entirely a real-data result.** |",
         "| **Rule-derived** | The remediation labels in Section 2: an action-effect table written from SRE practice, "
         "applied to the TRUE cause. Nothing was executed on the testbed: labels are not observed outcomes. The same "
         "table is the gate's consequence model, so the consequence model is exact by construction and the only error "
         "source left is the diagnosis (which is real). |",
         "| **Scripted** | The doers: an LLM-like dashboard reader and a restart-first variant (no LLM). |", "",
         "## Data", "",
         "- RE1-OB: 5 root-cause services × 5 faults (cpu, mem, disk, delay, loss) × 5 repetitions. Excluded: "
         + "; ".join(f"`{k}` ({v})" for k, v in o["data_notes"]["excluded"].items()) + ". Repaired: "
         + "; ".join(f"`{k}` ({v}; development subset only)" for k, v in o["data_notes"]["repaired"].items()) + ".",
         f"- **Development subset** (P3): repetition 1 of every family ({o['n_dev']} cases): signature weights, the doers and "
         f"the gate's plausible-set threshold τ were fixed on it. **Held out**: repetitions 2–5 ({o['n_heldout']} cases).",
         "- Harmonised features: the two RE1 schema variants differ (cpu/mem cases: `_latency`, `_load`; delay/disk/loss "
         "cases: `_latency-50`, `_latency-90`, `_workload`). Only metrics present in both are used (p90 is dropped), "
         "over the same ±300 s window around the injection, so the fault type cannot be read off the schema or the "
         "case length.",
         "- Hypotheses: **55** = 11 services × 5 faults (every service with metrics, not only the 5 the benchmark injects "
         "into). Scorer: BARO-style robust z-scores (median after vs before, IQR-based sd with a 5% floor), log-compressed, "
         "combined by a hand-set signature per fault type (CPU-dominant; memory-dominant; disk = moderate CPU + memory + "
         "own latency; network = own latency not explained by an elevated callee nor by own CPU/memory, delay vs loss by "
         "the size of the p50 shift). Nonconformity r(h) = E_max − E(h).", "",
         "## 1. Hypothesis sets on real telemetry (held-out cases)", "",
         "| | dev (25) | held-out (99) |", "|---|---|---|",
         f"| top-1 (service, fault) | {pct(rd['top1'])} | **{pct(rk['top1'])}** |",
         f"| service@1 / service@3 | {pct(rd['service@1'])} / {pct(rd['service@3'])} | {pct(rk['service@1'])} / {pct(rk['service@3'])} |",
         f"| fault type of the top-1 | {pct(rd['fault@1'])} | {pct(rk['fault@1'])} |",
         f"| mean rank of the truth (of 55) | {rd['mean_rank_of_truth']:.2f} | {rk['mean_rank_of_truth']:.2f} |",
         f"| reference: BARO-style service@1 / (service, metric→fault)@1 | {pct(rd['baro_style_service@1'])} / "
         f"{pct(rd['baro_style_top1'])} | {pct(rk['baro_style_service@1'])} / {pct(rk['baro_style_top1'])} |", "",
         "Held-out top-1 by fault type: " + ", ".join(f"{f} {pct(v['top1'], 0)} (service@1 {pct(v['service@1'], 0)})"
                                                      for f, v in rk["by_fault"].items()) + ".", "",
         "**Split conformal by case** (group = case; τ from a random half of the held-out cases, coverage and size on "
         f"the other half; fixed split seed 0, then {N_RESPLITS} re-splits):", "",
         "| δ | target coverage | fixed split: coverage / mean set size | re-splits: coverage mean ± sd | "
         "share of splits below target | re-splits: set size mean ± sd |",
         "|---|---|---|---|---|---|"]
    for d, v in o["conformal_by_case"].items():
        f, rs = v["fixed_split"], v["resplits"]
        L.append(f"| {d} | {pct(1 - float(d), 0)} | {pct(f['coverage'])} / {f['mean_set_size']:.2f} | "
                 f"**{pct(rs['coverage_mean'])} ± {pct(rs['coverage_sd'])}** | {pct(rs['frac_below_target'], 0)} | "
                 f"{rs['set_size_mean']:.2f} ± {rs['set_size_sd']:.2f} |")
    L += ["", "**Shift checks** (calibrate on the other groups, test on the held-out group; exchangeability does not hold "
          "across groups, so per-group coverage is not guaranteed):", "",
          "| δ | leave-(service, fault)-family-out: pooled coverage / families below target (of 25) / worst family / mean size | "
          "leave-fault-type-out: pooled coverage / worst type / mean size |", "|---|---|---|"]
    for d in o["family_out"]:
        fo, ft = o["family_out"][d], o["fault_type_out"][d]
        worst_f = min(fo["per_group"].items(), key=lambda kv: kv[1]["coverage"])
        worst_t = min(ft["per_group"].items(), key=lambda kv: kv[1]["coverage"])
        L.append(f"| {d} | {pct(fo['pooled_coverage'])} / {fo['groups_below_target']} / {worst_f[0]} {pct(worst_f[1]['coverage'], 0)} / "
                 f"{fo['mean_set_size']:.2f} | {pct(ft['pooled_coverage'])} / {worst_t[0]} {pct(worst_t[1]['coverage'], 0)} / "
                 f"{ft['mean_set_size']:.2f} |")
    L += ["", "Per fault type (leave-fault-type-out, δ = 0.1): " + ", ".join(
        f"{t} {pct(v['coverage'], 0)} (size {v['mean_set_size']:.1f})" for t, v in o["fault_type_out"]["0.1"]["per_group"].items()) + ".",
        "", "## 2. Remediation gate (rule-derived labels, scripted doers)", "",
        "**Action-effect table** (consequence model = label rule; an action on any other service never mitigates):", "",
        "| fault on service s | restart s | scale out s ×2 | roll back s | rate-limit frontend | rationale |", "|---|---|---|---|---|---|"]
    for f, row in R.EFFECT.items():
        L.append(f"| {f} | " + " | ".join("✓" if row[k] else "✗" for k in R.KINDS) + f" | {R.EFFECT_RATIONALE[f]} |")
    L += ["", "**Doers (scripted):** *LLM-like*: acts on the service with the largest jump on its dashboard (the largest "
          "robust z-score of CPU, memory or latency, after vs the previous 5 min; no call graph, no fault signatures), "
          "with the 'most common fix' (CPU → scale out, memory → restart, latency → restart), then the other fix, then "
          "the runner-up service; 15% of cases swap the first two (sampling noise). *Restart-first*: same reading, "
          "restart → roll back → restart runner-up. Each case is one decision with 3 ranked candidates (policy mode: "
          "the first certified candidate is executed).", "",
          "| doer | first choice unsafe | some candidate safe | first choice unsafe by fault |", "|---|---|---|---|"]
    for dn, p in g["doer_profile"].items():
        L.append(f"| {dn} | {pct(p['first_choice_unsafe'])} | {pct(p['some_candidate_safe'])} | " +
                 ", ".join(f"{f} {pct(v, 0)}" for f, v in p["first_choice_unsafe_by_fault"].items()) + " |")
    L += ["", f"**Methods** (scores from `riskgate.Gate`): *twin gate*: certify iff the action works under the top-1 "
          f"hypothesis (λ = 0, uncalibrated); *top-1 + CRC*: + (top-1 lead over the runner-up) if the action works under "
          f"the top-1, else −1 − lead, CRC-calibrated; *plausible set only*: certify iff the action works under every "
          f"hypothesis of the δ = 0.1 conformal set (τ = {g['tau_from_dev']:.3f} from the development subset; held-out "
          f"coverage {pct(g['plausible_set_heldout']['coverage'])}, mean size {g['plausible_set_heldout']['mean_size']:.2f}), "
          "no λ calibration; *RiskGate*: evidence gap over the same plausible set = r(best plausible hypothesis under which "
          "the action FAILS, capped at τ) − r(best one under which it works), −∞ if none works, calibrated by CRC or LTT "
          "(δ' = 0.1). With binary effects the per-hypothesis margin carries no magnitude, so the gap (a `score_fn` "
          "variant of the min-over-plausible-hypotheses score) is what gives CRC a graded score to threshold.", "",
          "**AUROC of the first choice's score for 'first choice unsafe'**: " + "; ".join(
              f"{dn}: " + ", ".join(f"{k} {v:.3f}" for k, v in a.items()) for dn, a in g["auroc_first_choice"].items()) + ".", ""]
    for a in ("0.05", "0.1"):
        for dn in R.DOERS:
            pr = g["per_doer"][a][dn]
            L += [f"**α = {pct(float(a), 0)}, doer = {dn}** ({pr['n_groups']} cases; fixed split {pr['n_cal']} calibration / "
                  f"{pr['n_test']} test cases):", ""]
            L += method_table(pr, g["fixed_resplits"][a][dn])
            L.append("")
    L += ["**Alpha sweep (fixed split, RiskGate CRC, LLM-like doer):** " + ", ".join(
        f"α={x['alpha']:g}: unsafe {pct(x['unsafe'])}, ask {pct(x['ask'])}"
        for x in g["per_doer"]["0.05"]["llm-like (dashboard)"]["alpha_sweep"]["riskgate (CRC)"]) + ".", "",
        "**Transfer (P7):** λ̂ calibrated on one doer's calibration cases, applied to the other doer's test cases "
        "(RiskGate CRC; unsafe / ask):", "", "| α | calibrated on → tested on | unsafe | ask | base unsafe |", "|---|---|---|---|---|"]
    for a, t in g["transfer"].items():
        for meth, tab in t.items():
            if meth != "riskgate (CRC)":
                continue
            for da, row in tab.items():
                for db, m in row.items():
                    L.append(f"| {a} | {da} → {db} | {pct(m['unsafe'])} | {pct(m['ask'])} | {pct(m['base_unsafe'])} |")
    L += ["", "**P9 (library path):** `Gate.calibrate(cal_eps, groups=cases, tau_fraction=0)` reproduces the harness λ̂ "
          f"on every fixed split ({len(g['p9_library_check'])} checks: both doers × CRC / LTT / top-1 CRC × α ∈ {{5%, 10%}}).", "",
          "## Examples (held-out, LLM-like doer)", ""]
    for e in o["examples"]:
        pl = ", ".join(f"`{h}` (r = {v:.2f})" for h, v in e["plausible"])
        cands = "; ".join(f"`{k}` → {'unsafe' if u else 'safe'}, RiskGate {lam_s(s)}, top-1 {t:.2f}, twin {w:+.0f}"
                          for k, u, s, t, w in zip(e["candidates"], e["unsafe"], e["riskgate_scores"], e["top1_scores"], e["twin_scores"]))
        vd = "; ".join(f"*{m}* {v}" for m, v in e["verdicts_alpha_0.05"].items())
        L.append(f"- **{e['case']}** (truth `{e['truth']}`): plausible set (τ = {e['tau']:.2f}): {pl}. Candidates (doer's "
                 f"order): {cands}. At α = 5% (fixed split; λ̂ = {e['lambda_riskgate_crc_alpha_0.05']:.3f} for RiskGate, "
                 f"{lam_s(e['lambda_top1_crc_alpha_0.05'])} for top-1 + CRC): {vd}.")
    def rs(a, dn, m):
        r = g["per_doer"][a][dn]["resplits"].get(m) or g["fixed_resplits"][a][dn][m]
        return r

    llm, rf = "llm-like (dashboard)", "restart-first"
    L += ["", "## What the experiment shows, and caveats", "",
          "- **Real-data content = Section 1.** On held-out real telemetry the 55-hypothesis conformal sets reach their "
          "nominal coverage on average across random case splits (marginal guarantee; individual splits vary), at small "
          "set sizes; coverage degrades under family or fault-type shift (packet loss is the weak spot of a metrics-only "
          "scorer), which is exactly what exchangeability predicts.",
          "- **The hypothesis set alone already bounds action risk.** Certifying only actions that work under every "
          "hypothesis of the δ = 0.1 set (τ fixed on the development subset) gave "
          f"{pct(rs('0.05', llm, 'plausible set only (fixed, delta=0.1)')['unsafe_mean'])} unsafe over all re-splits for "
          f"both doers, auto-approving {pct(rs('0.05', llm, 'plausible set only (fixed, delta=0.1)')['safe_auto_mean'])} "
          "of cases: if the truth is in the set, such an action is safe, so P(certified and unsafe) <= P(truth outside "
          "the set) <= δ for exchangeable cases (here the dev-calibrated set is conservative: held-out coverage "
          f"{pct(g['plausible_set_heldout']['coverage'])}).",
          "- **Section 2 is a rule-labelled simulation on real diagnoses.** Because the consequence model equals the label "
          "rule, the gate is only as wrong as its hypothesis set; the experiment isolates the value of handling diagnosis "
          "uncertainty, not of the consequence model.",
          f"- **RiskGate vs single-diagnosis CRC.** For the LLM-like doer at α = 5%, RiskGate (CRC) auto-approves "
          f"{pct(rs('0.05', llm, 'riskgate (CRC)')['safe_auto_mean'])} of cases vs {pct(rs('0.05', llm, 'top-1 + CRC')['safe_auto_mean'])} "
          f"for top-1 + CRC at a similar unsafe rate ({pct(rs('0.05', llm, 'riskgate (CRC)')['unsafe_mean'])} vs "
          f"{pct(rs('0.05', llm, 'top-1 + CRC')['unsafe_mean'])}): the single diagnosis cannot tell 'restart is safe under "
          f"both cpu and disk' from 'scale-out is safe only under cpu' (first example). For the restart-first doer the two "
          f"are tied ({pct(rs('0.05', rf, 'riskgate (CRC)')['safe_auto_mean'])} vs {pct(rs('0.05', rf, 'top-1 + CRC')['safe_auto_mean'])}), "
          f"and at α = 10% top-1 + CRC is slightly more efficient ({pct(rs('0.1', llm, 'top-1 + CRC')['safe_auto_mean'])} vs "
          f"{pct(rs('0.1', llm, 'riskgate (CRC)')['safe_auto_mean'])}; {pct(rs('0.1', rf, 'top-1 + CRC')['safe_auto_mean'])} vs "
          f"{pct(rs('0.1', rf, 'riskgate (CRC)')['safe_auto_mean'])}). RiskGate's score discriminates best (AUROC above).",
          f"- **The uncalibrated twin gate is strong here** ({pct(rs('0.05', llm, 'twin gate (top-1, fixed)')['unsafe_mean'])} "
          f"unsafe on average for the LLM-like doer, {pct(rs('0.05', rf, 'twin gate (top-1, fixed)')['unsafe_mean'])} for "
          "restart-first) because the scorer is good (73% top-1, 99% service@3) and the consequence model is exact by "
          "construction; it carries no guarantee and nothing adapts it when the diagnoser degrades.",
          "- **CRC bounds the expected risk over calibration draws**; with ~50 test cases per split, individual splits "
          "exceed α often (see 'share > α'; e.g. the restart-first fixed split at α = 10%). LTT is the high-probability "
          "alternative: its δ' = 10% bounds the probability that the TRUE risk exceeds α, and the empirical rate on "
          "~50 test cases adds sampling noise on top (share > α between "
          + f"{pct(min(rs(a, d, 'riskgate (LTT)')['frac_above_alpha'] for a in ('0.05', '0.1') for d in (llm, rf)), 0)} and "
          + f"{pct(max(rs(a, d, 'riskgate (LTT)')['frac_above_alpha'] for a in ('0.05', '0.1') for d in (llm, rf)), 0)}).",
          "- **Transfer:** a λ̂ calibrated on the restart-first doer is too permissive for the LLM-like doer (whose first "
          "choices are unsafe more often): policy-mode guarantees are doer-specific, as the paper states.",
          "- Groups are single cases (one decision each); repetitions of the same family are treated as exchangeable "
          "cases (the family-out analysis shows what happens when they are not).",
          "- The doers are scripted and deterministic given the telemetry, and they read the same robust z-scores as the "
          "gate's scorer (without its call graph and signatures), so doer and gate errors are correlated; a real LLM "
          "reading raw dashboards may err differently. The action-effect table is a stylised SRE "
          "rule, not a measurement of what restarts or scale-outs did on this testbed (e.g. a pod restart may also clear "
          "a tc/netem rule injected into the pod's network namespace; the table models the fault classes, not the "
          "injection mechanism)."]
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    which = argv[1:] or ["itbench", "rcaeval"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if "itbench" in which:
        o = run_itbench()
        s, g = o["shadow_replay"], o["gate"]
        print(f"ITBench: {s['entity_correct']['k']}/{s['entity_correct']['n']} claims correct; held-out AUROC "
              f"{g['auroc']['riskgate']['heldout']:.2f}; P9 checks {len(g['p9_library_check'])} ok")
    if "rcaeval" in which:
        o = run_rcaeval()
        print(f"RCAEval: held-out top-1 {o['ranking']['heldout']['top1']:.3f}; P9 checks {len(o['gate']['p9_library_check'])} ok")


if __name__ == "__main__":
    main(sys.argv)
