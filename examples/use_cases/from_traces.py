"""Onboarding from traces: the customer sends agent traces and an outcome signal; we build and calibrate the gate.

The traces are simulated refund-agent runs (read tools: get_order, check_refund_ledger; write: issue_refund) and the
outcomes are "was the refund reversed or duplicated later?", joined by trace id. Nothing is labelled by a judge.

    1. infer which tools are reads and which are writes            infer_tool_kinds
    2. one labelled decision point per write                       decision_points
    3. seal a group-disjoint fit / cal / eval split                seal_split
    4. on FIT only: learn the worlds from what the agent had seen  DiscreteWorlds.fit
    5. on CAL: calibrate the frozen gate, by run                   Gate.calibrate(groups=...)
    6. on EVAL: report                                             replay

    python from_traces.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import riskgate as rg
from riskgate.traces import decision_points, infer_tool_kinds, load_outcomes, load_traces, seal_split

WORLDS = ("owed", "already_refunded", "second_bag")
MIX = (0.5, 0.3, 0.2)                                                         # simulation assumption, not data


def simulate(n: int, out: Path) -> None:
    """Write traces.jsonl and outcomes.csv the way a customer export would look."""
    rng = np.random.default_rng(1)
    with (out / "traces.jsonl").open("w") as tf, (out / "outcomes.csv").open("w") as of:
        of.write("trace_id,tool,unsafe,source\n")
        for i in range(n):
            w = WORLDS[int(rng.choice(3, p=MIX))]
            order = {"id": f"O{i}", "bags": 2 if (w == "second_bag" or rng.random() < 0.3) else 1,
                     "unavailable_notice": bool(rng.random() < (0.4 if w == "already_refunded" else 0.05))}
            calls = [{"name": "get_order", "args": {"id": order["id"]}, "output": order}]
            if rng.random() < 0.35:                                         # the agent sometimes reads the ledger
                calls.append({"name": "check_refund_ledger", "args": {"id": order["id"]},
                              "output": {"refunded": w == "already_refunded"}})
            calls.append({"name": "issue_refund", "args": {"order": order["id"], "amount": 49.0}})
            tf.write(json.dumps({"trace_id": f"t{i}", "group": f"cust{i // 2}", "doer_id": "refund-agent:v12",
                                 "calls": calls}) + "\n")
            of.write(f"t{i},issue_refund,{str(w != 'owed').lower()},{'reversal' if w == 'second_bag' else 'ledger_audit'}\n")


def features(p) -> dict:
    """What the agent had seen before the refund (computed only from the history, no look-ahead)."""
    order = p.context.last_output("get_order") or {}
    ledger = p.context.last_output("check_refund_ledger")
    return {"two_bags": order.get("bags", 1) >= 2, "unavailable_notice": bool(order.get("unavailable_notice")),
            "ledger_refunded": None if ledger is None else bool(ledger["refunded"])}


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        simulate(3000, Path(d))
        traces, outcomes = load_traces(Path(d) / "traces.jsonl"), load_outcomes(Path(d) / "outcomes.csv")
    kinds = infer_tool_kinds(traces)
    points = decision_points(traces, outcomes, kinds)
    print("tool kinds:", kinds)
    print("decision points:", decision_points.last_report)
    split = seal_split(points, seed=7)
    print("sealed split:", split.summary())

    # 4. FIT split only: which world was each refund in? The outcome signal says unsafe or not; the reason for an
    #    unsafe refund (already refunded vs. delivered in a second bag) comes from the label source here.
    world_of = lambda p: "owed" if not p.unsafe else ("second_bag" if p.source == "reversal" else "already_refunded")  # noqa: E731
    worlds = rg.DiscreteWorlds.fit([features(p) for p in split.fit], [world_of(p) for p in split.fit], name="refund-worlds")
    table = rg.OutcomeTable({"issue_refund": ["owed"]})
    gate = rg.Gate(table, table.spec, worlds, score_fn=rg.unsafe_mass, doer_id="refund-agent:v12")

    # 5. CAL split: calibrate once, by customer (several runs of one customer are dependent)
    gate.calibrate(split.episodes("cal", context=features), alpha=0.05, groups=split.groups("cal"))
    # 6. EVAL split: what it would have done
    report = rg.replay(gate, split.episodes("eval", context=features))
    print(report)
    # where the asks come from: refunds issued without reading the ledger cannot tell "owed" from "already refunded"
    for read, label in ((True, "agent read the ledger first"), (False, "agent skipped the ledger")):
        sel = [o for p, o in zip(split.eval, report.outcomes) if p.context.called("check_refund_ledger") is read]
        auto = sum(o.verdict == "certify" for o in sel) / max(1, len(sel))
        print(f"  {label:28s}: {len(sel):4d} refunds, autonomous {100 * auto:.0f}%")
    # Even after a ledger read, a refund on a two-bag order can be for an item that is still on its way (it shows up
    # later as a reversal), so those still go to a person. The report tells the customer what would buy autonomy:
    print("  -> next steps: make the ledger read a probe the gate can request, and add a rider/delivery-status read")


if __name__ == "__main__":
    main()
