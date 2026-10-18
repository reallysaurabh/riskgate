# Formats and an end-to-end script

## Traces: JSONL, one run per line

```json
{"trace_id": "t-1", "group": "ticket-88", "doer_id": "refund-agent:v12", "prompt": "My butter didn't come",
 "calls": [{"name": "get_order", "args": {"id": "A1"}, "output": {"bags": 2, "total": 247.0}},
           {"name": "check_refund_ledger", "args": {"id": "A1"}, "output": {"refunded": false}},
           {"name": "issue_refund", "args": {"order": "A1", "amount": 49.0}}]}
```

- **`args`** may be a JSON string. `arguments` and `result` are accepted as aliases.
- **`group`**: runs that share it (one ticket, customer or incident) are kept together.
- **`trace_id`** values must be unique.

## Outcomes: CSV or JSONL

```text
trace_id,tool,call,unsafe,source
t-1,issue_refund,,true,chargeback
t-2,,,false,qa_audit
```

- **Required:** `trace_id` and `unsafe` (true/false; empty means unknown and the write is left out).
- **`tool`** labels every call of that tool in the trace. **`call`** is a 0-based index into `calls`.
- **Neither given:** the label applies to every write in the trace.
- **`source`:** sources naming an LLM judge or an approval are refused by `load_outcomes`.

## Converting common exports

**OpenTelemetry spans** (one JSON span per line, GenAI semantic conventions). Group the spans by `trace_id`, keep
the tool-call spans in start-time order, and read the tool name, arguments and result from the attributes. Attribute
names vary by instrumentation library: print a few spans and adjust.

```python
import collections, json
runs = collections.defaultdict(list)
for line in open("spans.jsonl"):
    s = json.loads(line); a = s.get("attributes", {})
    name = a.get("gen_ai.tool.name") or a.get("tool.name")
    if name:
        runs[s["trace_id"]].append((s["start_time"], {"name": name, "args": a.get("gen_ai.tool.call.arguments", "{}"),
                                                      "output": a.get("gen_ai.tool.call.result")}))
with open("traces.jsonl", "w") as f:
    for tid, calls in runs.items():
        f.write(json.dumps({"trace_id": tid, "calls": [c for _, c in sorted(calls, key=lambda x: x[0])]}) + "\n")
```

**LangSmith**: export runs, take each root run, and use its child runs with `run_type == "tool"` in `start_time`
order: `name`, `inputs` as args, `outputs` as output.

## End to end

```python
import riskgate as rg
from riskgate.traces import decision_points, infer_tool_kinds, load_outcomes, load_traces, seal_split

traces, outcomes = load_traces("traces.jsonl"), load_outcomes("outcomes.csv")
kinds = infer_tool_kinds(traces, read=[], write=[])          # review; pass explicit lists to correct it
points = decision_points(traces, outcomes, kinds)
print(decision_points.last_report)                           # labelled / unlabelled / conflicting writes
split = seal_split(points, ratios=(0.2, 0.5, 0.3), seed=7)
print(split.summary())                                       # record split.digest in the report

def features(p):                                             # only what the agent had seen before the write
    order = p.context.last_output("get_order") or {}
    ledger = p.context.last_output("check_refund_ledger")
    return {"two_bags": order.get("bags", 1) >= 2,
            "ledger_refunded": None if ledger is None else bool(ledger["refunded"])}

def world_of(p):                                             # name worlds from the outcome reason (fit split only)
    return "owed" if not p.unsafe else ("second_bag" if p.source == "reversal" else "already_refunded")

worlds = rg.DiscreteWorlds.fit([features(p) for p in split.fit], [world_of(p) for p in split.fit])
table = rg.OutcomeTable({"issue_refund": ["owed"]})
gate = rg.Gate(table, table.spec, worlds, score_fn=rg.unsafe_mass, doer_id="refund-agent:v12")
cal = gate.calibrate(split.episodes("cal", context=features), alpha=0.05, groups=split.groups("cal"))
report = rg.replay(gate, split.episodes("eval", context=features))
print(report)
for read in (True, False):                                   # where the asks come from
    sel = [o for p, o in zip(split.eval, report.outcomes) if p.context.called("check_refund_ledger") is read]
    print("ledger read" if read else "no ledger read", len(sel), sum(o.verdict == "certify" for o in sel) / max(1, len(sel)))
cal.save("gate.cal.json")                                    # pass key=... to sign it
```
