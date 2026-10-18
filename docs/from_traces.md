# Start from your traces

The fastest way to a calibrated gate is not to write a simulator first. Send two things:

1. **Agent traces**: one record per run, with its tool calls in order, their arguments and, for read tools, what
   they returned. OpenTelemetry spans, LangSmith runs and MCP logs all convert to this shape.
2. **An outcome signal**: what happened after each consequential action, joined by trace id. Examples are
   reversals, chargebacks, incident tickets, QA audits, SLO breaches, or a deterministic validator run on the state
   after the action.

From those, riskgate classifies the tools, builds one labelled decision point per write, and seals a group-disjoint
split. It learns the plausible worlds on the first part, calibrates on the second and reports on the third. A domain
expert then reviews what was learnt (the worlds, the probes, the rules) before anything is enforced.

```python
import riskgate as rg
from riskgate.traces import decision_points, infer_tool_kinds, load_outcomes, load_traces, seal_split

traces = load_traces("traces.jsonl")
outcomes = load_outcomes("outcomes.csv")
kinds = infer_tool_kinds(traces)                       # read tools (evidence, probes) vs write tools (decisions)
points = decision_points(traces, outcomes, kinds)      # one per labelled write; see decision_points.last_report
split = seal_split(points, ratios=(0.2, 0.5, 0.3), seed=7)

def features(p):                                       # what the agent had seen before the write (no look-ahead)
    order = p.context.last_output("get_order") or {}
    ledger = p.context.last_output("check_refund_ledger")
    return {"two_bags": order.get("bags", 1) >= 2, "ledger_refunded": None if ledger is None else ledger["refunded"]}

worlds = rg.DiscreteWorlds.fit([features(p) for p in split.fit], [world_of(p) for p in split.fit])   # FIT only
table = rg.OutcomeTable({"issue_refund": ["owed"]})
gate = rg.Gate(table, table.spec, worlds, score_fn=rg.unsafe_mass, doer_id="refund-agent:v12")
gate.calibrate(split.episodes("cal", context=features), alpha=0.05, groups=split.groups("cal"))       # CAL, once
print(rg.replay(gate, split.episodes("eval", context=features)))                                    # EVAL
```

The runnable version, with simulated traces, is
[`examples/use_cases/from_traces.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/from_traces.py).

## The three rules that keep the guarantee

> [!IMPORTANT]
> Traces make onboarding easy, but they can also quietly turn the bound into something weaker. These three rules
> are what keep it a bound on real harm.

**1. Outcomes, not opinions.**
- **The problem.** A wrong refund usually returns `200 OK`, and the customer often says "thanks". In the public
  τ²-bench logs, 355 of 363 wrong retail writes came right after the customer said "yes". The trace alone rarely
  shows harm.
- **What the label must be.** What happened afterwards, or a deterministic check on the state after the action.
- **What it must not be.** An LLM judging the trace: the bound would then be about disagreement with the judge.
  Nor a human approving the action: an approval authorises execution, it does not show the outcome was safe.
- **How the code enforces it.** `load_outcomes` refuses label sources that name a judge or an approval.
  `decision_points` leaves out unlabelled writes and counts them, rather than guessing.

**2. Learn on one split, calibrate on another.**
- **What counts as learnt.** Anything taken from the traces: which worlds exist, their probabilities, feature
  weights, which read tool to use as a probe, the limits of rules.
- **Where to learn it.** On `split.fit` only. Freeze it before calibration touches `split.cal`, and keep
  `split.eval` for the report.
- **Why.** Learning and calibrating on the same traces breaks exchangeability, the assumption the bound rests on.
- **Calibrate once.** Recalibrating on the same split until the numbers look good is p-hacking. `SealedSplit.digest`
  names exactly which data a threshold was calibrated on.

**3. Count runs, not tool calls.**
- **The problem.** Calls inside one run, or several runs of one customer case, are dependent.
- **What to do.** `seal_split` splits by `group`, and `calibrate(groups=...)` controls the share of a new group's
  decisions that are certified and unsafe.
- **What it needs.** At least ⌈1/α⌉ independent groups in the calibration split (20 at α = 5%).

## What can be learnt from traces, and what cannot

| Component | From traces? | How |
|---|---|---|
| Which tools are reads and which are writes | yes, then review | `infer_tool_kinds`: read prefixes (`get_`, `list_`, ...) are reads. Unknown tools count as writes (fail closed). |
| Plausible worlds and their probabilities | yes, on the fit split | `DiscreteWorlds.fit` on features of what the agent had seen. Also use the outcome's reason (reversal, duplicate, delivered) to name the worlds. |
| Missing-evidence worlds | yes | `p.context.called("check_refund_ledger")`: a refund issued without reading the ledger leaves "already refunded" plausible. |
| Read-vs-write mismatches | yes | compare earlier read outputs with the write's arguments: `amount <= order.paid`, the order id is one of the customer's orders. For frontier agents these relational checks matter more than wording. |
| Probes | yes | read tools whose output separates the worlds (the `reveals` of an `OutcomeTable`). |
| Hard-rule limits | yes, on the fit split | e.g. `rules.amount_above(<p99 of past refunds>)`, or `needs_person` for actions that label people. |
| Whether an action was unsafe | **no** | needs the outcome signal (rule 1). |
| What an action the agent did *not* take would have done | **no** | traces label only the action taken, so each episode has one candidate and the gate runs in **policy mode**. The bound covers this agent: set `doer_id`, and collect new traces after a model or prompt change. Ranked alternatives need a sandbox or simulator (see [Bring your own domain](byo_domain.md)). |

A rough world model learnt from traces is still valid: a weaker model costs autonomy (more `ASK`), not safety. The
report says where the asks come from. In the example, refunds issued after a ledger read are certified much more
often than refunds issued without one, which tells the customer what to change next.

## Formats

**Traces** (JSONL, one run per line). `args` may be a JSON string, and `arguments` / `result` are accepted as aliases.

```json
{"trace_id": "t-1", "group": "ticket-88", "doer_id": "refund-agent:v12", "prompt": "My butter didn't come",
 "calls": [{"name": "get_order", "args": {"id": "A1"}, "output": {"bags": 2, "items": ["..."]}},
           {"name": "issue_refund", "args": {"order": "A1", "amount": 49.0}}]}
```

**Outcomes** (CSV or JSONL):

- **Required:** `trace_id`, `unsafe` (true / false; empty means unknown and the write is left out).
- **Optional:** `call` (the index of the call in the trace) or `tool` (every call of that tool), and `source`.
- **No `call` or `tool`:** the label applies to every write in the trace.

```text
trace_id,tool,unsafe,source
t-1,issue_refund,true,chargeback
t-2,issue_refund,false,qa_audit
```

## How many traces

You need enough labelled decisions in the calibration split, counted in groups. Roughly 300 labelled writes for
α = 10% and 500 for α = 5%, split 20 / 50 / 30. At the bare minimum the threshold is often infinite and nothing is
certified. How to count groups is in [Calibration & guarantees](calibration.md).
