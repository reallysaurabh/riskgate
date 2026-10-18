"""Check a trace export before building a gate: tool kinds, label coverage, groups, and which alpha it supports.

    pip install --pre riskgate
    python inspect_traces.py traces.jsonl [outcomes.csv] [--read get_x,list_y] [--write do_z] [--json]

traces.jsonl is riskgate's trace format (one run per line, see references/formats.md); outcomes are CSV or JSONL
joined by trace id. Prints a report and exits non-zero if the data cannot support a calibrated gate yet.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys

from riskgate.traces import decision_points, infer_tool_kinds, load_outcomes, load_traces, seal_split

ALPHAS = (0.10, 0.05, 0.02)
CAL_SHARE = 0.5                      # seal_split default (0.2, 0.5, 0.3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traces")
    ap.add_argument("outcomes", nargs="?")
    ap.add_argument("--read", default="", help="comma-separated tools to treat as read-only")
    ap.add_argument("--write", default="", help="comma-separated tools to treat as consequential")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)

    traces = load_traces(a.traces)
    kinds = infer_tool_kinds(traces, read=[t for t in a.read.split(",") if t], write=[t for t in a.write.split(",") if t])
    calls = collections.Counter(c.name for t in traces for c in t.calls)
    with_output = collections.Counter(c.name for t in traces for c in t.calls if c.output is not None)
    groups = {t.unit for t in traces}
    doers = collections.Counter(t.doer_id or "(none)" for t in traces)
    rep: dict = {
        "traces": len(traces), "groups": len(groups), "doer_ids": dict(doers),
        "tools": {n: {"kind": kinds[n], "calls": calls[n], "with_output": with_output[n]} for n in kinds},
        "warnings": [], "supports_alpha": {},
    }
    if len(doers) > 1 or "(none)" in doers:
        rep["warnings"].append("traces mix agent versions or lack doer_id: calibrate per agent version (policy mode)")
    for n, k in kinds.items():
        if k == "read" and with_output[n] == 0:
            rep["warnings"].append(f"read tool {n!r} has no outputs in the traces: it cannot serve as evidence")
    if len(groups) == len(traces):
        rep["warnings"].append("no 'group' field: every trace is its own group; set group if runs share a ticket/customer")

    if a.outcomes:
        outcomes = load_outcomes(a.outcomes)
        points = decision_points(traces, outcomes, kinds)
        rep["labels"] = dict(decision_points.last_report)
        rep["label_sources"] = dict(collections.Counter(o["source"] or "(none)" for o in outcomes))
        rep["unsafe_rate"] = sum(p.unsafe for p in points) / len(points) if points else None
        rep["labelled_writes_by_tool"] = dict(collections.Counter(p.action.name for p in points))
        lab = rep["labels"]
        if lab["writes"] and lab["labelled"] / lab["writes"] < 0.8:
            rep["warnings"].append(f"only {lab['labelled']} of {lab['writes']} writes are labelled: unlabelled writes are "
                                   "dropped, which is honest only if labels are missing at random")
        if "(none)" in rep["label_sources"]:
            rep["warnings"].append("some outcomes have no 'source': record where each label came from")
        n_groups = len({p.group for p in points})
        cal_groups = int(round(CAL_SHARE * n_groups))
        for alpha in ALPHAS:
            need = math.ceil(1 / alpha)
            rep["supports_alpha"][f"{alpha:g}"] = {"cal_groups": cal_groups, "hard_floor": need,
                                                    "comfortable": cal_groups >= 10 * need, "possible": cal_groups >= need}
        if points:
            rep["sealed_split"] = seal_split(points).summary()
        ok = bool(points) and rep["supports_alpha"]["0.1"]["possible"]
    else:
        rep["warnings"].append("no outcomes given: traces alone cannot say which actions were unsafe")
        ok = False

    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(f"{rep['traces']} traces in {rep['groups']} groups; agent versions: {rep['doer_ids']}")
        print("tools (kind, calls, calls with output):")
        for n, t in rep["tools"].items():
            print(f"  {t['kind']:5s} {n:32s} {t['calls']:6d} {t['with_output']:6d}")
        if "labels" in rep:
            print(f"labels: {rep['labels']}")
            print(f"label sources: {rep['label_sources']}; unsafe rate {rep['unsafe_rate']}")
            for al, s in rep["supports_alpha"].items():
                verdict = "comfortable" if s["comfortable"] else "possible (thin)" if s["possible"] else "NOT enough"
                print(f"  alpha={al}: {s['cal_groups']} calibration groups (floor {s['hard_floor']}) -> {verdict}")
        for w in rep["warnings"]:
            print("WARNING:", w)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
