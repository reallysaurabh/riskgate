"""End-to-end: the full riskgate lifecycle on the Kubernetes incident-response domain.

    cd examples && PYTHONPATH=.. python end_to_end_sre.py            # ~5 min
    cd examples && PYTHONPATH=.. python end_to_end_sre.py --quick    # ~1 min

Stages
  1  build the gate from domain components (twin, diagnoser, SLO spec, probes, monitor, audit log)
  2  collect labelled calibration incidents (here: simulator; in production: replays, staging, postmortems)
  3  calibrate, sign and save the artefact; inspect it with the CLI
  4  deploy: an incident-response loop where the agent proposes and the gate certifies / probes / asks
  5  report outcomes back (labels arrive later) and summarise the audit log
  6  drift: the database is moved to a smaller instance class the twin does not know about -> the monitor
     raises DEGRADED and the online threshold tightens as SLO breaches are reported
  7  the team updates the twin (new model version) -> the old calibration is VOID -> recalibrate -> VALID
Set RISKGATE_LLM_API_KEY (and optionally RISKGATE_LLM_BASE_URL / RISKGATE_LLM_MODEL) to let a real LLM rank
the remediation actions instead of the scripted SRE assistant.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import k8s_incident as K
import numpy as np

import riskgate as rg
from riskgate.cli import main as riskgate_cli

OUT = Path(__file__).resolve().parent / "e2e_output"
KEY = "demo-signing-key"              # in production: a secret from your vault


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 100}\n[{n}] {title}\n{'=' * 100}")


def describe(ep: rg.Episode) -> str:
    w = ep.context.window
    return (f"rps={w['rps'][-1]:.0f} p95={w['p95_ms'][-1]:.0f}ms errors={100 * w['error_rate'][-1]:.1f}% "
            f"pod CPU={100 * w['cpu'][-1]:.0f}% DB CPU={100 * w['db_cpu'][-1]:.0f}% recent_deploy={ep.context.deploy_seen}")


def maybe_llm_rank(ep: rg.Episode, ranker) -> list[rg.Action]:
    if ranker is None:
        return ep.candidates
    policy = ("SLO: p95 <= 800 ms and errors <= 2% every minute. Postgres max_connections=200 with 10 per pod. "
              "Prefer the least disruptive action that restores the SLO.")
    return ranker.rank(describe(ep), ep.candidates, policy)


def handle(gate: rg.Gate, ep: rg.Episode, ranked: list[rg.Action]) -> dict:
    """One incident: the agent proposes `ranked`, the gate decides, the world responds. Returns a log with a
    human-readable transcript."""
    t = [f"  incident INC-{ep.meta['seed']:06d}: {describe(ep)}",
         f"    hidden truth   : {ep.meta['truth']}",
         f"    agent proposes : {[K.NAMES.get(a.key, a.name) for a in ranked[:3]]} ..."]
    d = gate.check_ranked(ranked, ep.context)
    log = {"verdict": d.verdict.value, "probe": None, "executed": None, "unsafe": False,
           "first_choice_unsafe": bool(ep.unsafe[ranked[0].key])}
    t += [f"    gate (stage 1) : {d.verdict.value.upper()} - {d.reason}", f"    plausible      : {d.plausible}"]
    if d.verdict is rg.Verdict.PROBE:
        log["probe"] = d.probe.name
        ep2 = ep.after_probe[d.probe.key]                  # the world after running the read-only probe
        t.append(f"    probe result   : {d.probe.name} -> {ep2.meta.get('probe_value')} (2 minutes later)")
        d = gate.check_ranked(ep2.candidates, ep2.context, stage=2)
        t.append(f"    gate (stage 2) : {d.verdict.value.upper()} - {d.reason}")
        ep = ep2
    if d.verdict is rg.Verdict.CERTIFY:
        log["executed"] = K.NAMES.get(d.action.key, d.action.name)
        log["unsafe"] = bool(ep.unsafe[d.action.key])
        gate.record_outcome(d, log["unsafe"])              # in reality the label arrives minutes later
        t.append(f"    executed       : {log['executed']} -> SLO {'BREACHED' if log['unsafe'] else 'held'}")
    else:
        t.append("    -> paged the on-call engineer with the gate's explanation")
    log["verdict_final"] = d.verdict.value
    log["transcript"] = "\n".join(t)
    return log


def summary(logs: list[dict]) -> str:
    return (f"certified-and-unsafe {100 * np.mean([l['unsafe'] for l in logs]):.1f}% | "
            f"asked/rejected {100 * np.mean([l['verdict_final'] != 'certify' for l in logs]):.1f}% | "
            f"probed {100 * np.mean([l['probe'] is not None for l in logs]):.1f}% | "
            f"agent's first choice would have been unsafe {100 * np.mean([l['first_choice_unsafe'] for l in logs]):.1f}% | "
            f"n={len(logs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_cal, n_live, n_drift, n_recal = (300, 120, 150, 300) if args.quick else (1200, 300, 300, 900)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.jsonl"):
        f.unlink()
    ranker = None
    if os.environ.get("RISKGATE_LLM_API_KEY"):
        from riskgate.doers import LLMRanker
        ranker = LLMRanker()

    banner(1, "Build the gate from domain components")
    gate = K.make_gate(audit=rg.AuditLog(OUT / "decisions.jsonl", extra={"service": "checkout-api"}),
                       monitor=rg.Monitor(alarm_level=20),          # false-alarm probability <= 1/20 (Ville)
                       doer_id=ranker.doer_id if ranker else "scripted-sre-v1")
    print(f"  consequence model : {gate.consequence.version} (Erlang-C queueing twin, Postgres connection limits)")
    print(f"  hypotheses        : {gate.hypotheses.version} ({len(K.GRID)} incident causes x magnitudes)")
    print(f"  spec              : {[c.name for c in gate.spec.constraints]} (each minute, t+4..t+30)")
    print(f"  probes            : {[p.name for p in gate.probes]}   alpha={gate.alpha}  mode={gate.mode}")
    print(f"  fingerprint       : {gate.fingerprint()}   status before calibration: {gate.status().value}")

    banner(2, f"Collect {n_cal} labelled calibration incidents")
    cal_eps = [K.sample(10_000 + i) for i in range(n_cal)]
    r = K.record(cal_eps[0])
    print("  example record:", json.dumps({k: r[k] for k in ("incident_id", "alert", "recent_deploy",
                                                              "agent_ranked_actions", "ground_truth_cause")})[:400])
    print(f"  doer's first choice unsafe in {100 * np.mean([e.unsafe[e.candidates[0].key] for e in cal_eps]):.1f}% "
          "of calibration incidents")

    banner(3, "Calibrate, sign and save the artefact")
    cal = gate.calibrate(cal_eps)
    cal.save(OUT / "checkout-api.cal.json", key=KEY)
    riskgate_cli(["report", str(OUT / "checkout-api.cal.json"), "--key", KEY])

    banner(4, f"Deploy: {n_live} live incidents")
    logs = []
    for i in range(n_live):
        ep = K.sample(50_000 + i)
        logs.append(handle(gate, ep, maybe_llm_rank(ep, ranker)))
    shown = set()
    for want, pick in (("certified directly", lambda l: l["verdict"] == "certify"),
                       ("probe, then decide", lambda l: l["probe"] is not None and l["verdict_final"] == "certify"),
                       ("escalated to a human", lambda l: l["verdict_final"] in ("ask", "reject"))):
        hit = next((l for l in logs if pick(l) and id(l) not in shown), None)
        if hit:
            shown.add(id(hit))
            print(f"\n  --- example: {want} ---\n{hit['transcript']}")
    print(f"\n  live summary: {summary(logs)}")

    banner(5, "Outcomes and audit")
    print(f"  monitor: observed risk on certified actions = {gate.monitor.observed_risk}, martingale = {gate.monitor.martingale:.2f}, "
          f"status = {gate.status().value}")
    riskgate_cli(["audit-summary", str(OUT / "decisions.jsonl")])

    banner(6, f"Drift: the database moved to a smaller instance class (-40% capacity); {n_drift} incidents")
    print("  the twin still assumes the old capacity: its predictions are now optimistic")
    alarm_at = None
    drift_logs = []
    for i in range(n_drift):
        ep = K.sample(70_000 + i, db_cap_factor=0.6)
        drift_logs.append(handle(gate, ep, maybe_llm_rank(ep, ranker)))
        if alarm_at is None and gate.status() is rg.GuaranteeStatus.DEGRADED:
            alarm_at = i
            print(f"  drift alarm after {i + 1} incidents (martingale {gate.monitor.martingale:.0f} >= "
                  f"{gate.monitor.alarm_level:.0f}) -> status DEGRADED; the online threshold may only tighten")
    if alarm_at is None:
        print(f"  no drift alarm (martingale {gate.monitor.martingale:.1f})")
    base_lam = cal.lam_probe1 if cal.probing else cal.lam
    print(f"  drift summary: {summary(drift_logs)}\n  status {gate.status().value}; threshold in use "
          f"{gate._threshold(1):.3f} vs calibrated {base_lam:.3f}; observed risk on certified actions "
          f"{gate.monitor.observed_risk:.3f}")

    banner(7, "Update the twin to the new capacity -> VOID -> recalibrate -> VALID")
    gate2 = K.make_gate(db_cap=K.DB_CAP * 0.6, audit=rg.AuditLog(OUT / "decisions.jsonl", extra={"service": "checkout-api"}),
                        monitor=rg.Monitor(alarm_level=20), doer_id=gate.doer_id)
    gate2.load_calibration(OUT / "checkout-api.cal.json", key=KEY)
    print(f"  new twin version {gate2.consequence.version}: fingerprint {gate2.fingerprint()} != artefact "
          f"{gate2.calibration.fingerprint} -> status {gate2.status().value}")
    probe_ep = K.sample(75_000, db_cap_factor=0.6)
    d = gate2.check_ranked(probe_ep.candidates, probe_ep.context)
    print(f"  any check now -> {d.verdict.value.upper()}: {d.reason}")
    recal_eps = [K.sample(80_000 + i, db_cap_factor=0.6) for i in range(n_recal)]
    cal2 = gate2.recalibrate(recal_eps)
    cal2.save(OUT / "checkout-api.v2.cal.json", key=KEY)
    print(f"  recalibrated on {n_recal} labelled incidents from the new regime: artefact {cal2.id}, status {gate2.status().value}")
    post = []
    for i in range(n_drift):
        ep = K.sample(90_000 + i, db_cap_factor=0.6)
        post.append(handle(gate2, ep, maybe_llm_rank(ep, ranker)))
    print(f"  after recalibration: {summary(post)} | status {gate2.status().value}")
    print(f"\n  files: {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
