# Use cases

The RiskGate site tells sixteen short stories. Each one is the same mechanism (list the worlds the evidence cannot
rule out, play the action forward in each, run it only if it is safe in all of them, look first or ask otherwise)
applied to a different decision. This page maps every story to the parts of the package that build it, and to a
runnable script.

> [!NOTE]
> The scripts in [`examples/use_cases/`](https://github.com/reallysaurabh/riskgate/tree/main/examples/use_cases/)
> use **simulated** cases with assumed world mixes. They show how to build each gate; they are not evidence about any
> real platform, bank or cloud. Results on real, public agent logs are in [Results on real agents](results_real_agents.md).

## Building blocks (new in 1.0.0rc3)

| Module | What it gives you | Use it when |
|---|---|---|
| [`riskgate.worlds`](#discrete-worlds) | `DiscreteWorlds` (a naive-Bayes hypothesis provider over a few named worlds), `OutcomeTable` (which action is safe in which world, and what each read-only lookup reveals), `nearest_violation` and `unsafe_mass` score functions | the decision hinges on one hidden fact with a handful of values: *was it paid, refunded, a hold, dormant?* |
| [`riskgate.provenance`](#provenance) | `AgentContext`, `ToolOutput`, `ProvenanceWorlds`, `ProvenanceModel`, `instruction_score`, `entities` | instructions hidden in files, e-mails or web pages must not be able to move money (prompt injection) |
| [`riskgate.replay`](#replay) | `replay`, `resplit`, `sweep`, `ReplayReport` | onboarding: run labelled history through a gate; validity over re-splits; the error budget as a dial |
| [`riskgate.rules`](#rules) | `needs_person`, `never`, `amount_above`, `irreversible` | deterministic policy: labels on people go to a person, refunds above a limit need a supervisor |

None of them changes the guarantee: it still comes from calibration on labelled episodes. Wrong tables or a weak
instruction scorer cost autonomy (more `ASK`), not validity.

## The sixteen stories

| Story (site) | Decision | Worlds | Lookup (PROBE) | Building blocks | Script |
|---|---|---|---|---|---|
| RiskGate in 45 seconds | scale out on a HighErrorRate alert | 13 causes, 4 still fit | `pg_stat_statements` | core `Gate`, queueing twin | [`end_to_end_sre.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/end_to_end_sre.py), [`k8s_incident.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/k8s_incident.py) |
| "Yes." isn't a check | refund to which payment method | readings of the conversation | — | intent readings, `unsafe_mass`, `calibrate(groups=...)` | [`benchmarks/tau2`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/tau2) |
| Which booking? | cancel which reservation | the customer's bookings | — | as above | [`benchmarks/tau2`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/tau2) |
| Non-refundable, unless… | refund under which policy clause | 24 h, airline-cancelled, business, insured, none | read the reservation | `DiscreteWorlds`-style clause prior, probing | [`benchmarks/tau2`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/tau2) |
| The butter never came | refund a missing item | never packed, already refunded, second bag | pick log + refund ledger | `DiscreteWorlds`, `OutcomeTable(safe=callable)`, `rules.amount_above` | [`missing_item_refund.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/missing_item_refund.py) |
| The trimmer is broken | refund or replace a defective item | first defect, repeat defect, already refunded | return ledger | `DiscreteWorlds`, `OutcomeTable`, ranked remedies | pattern of `missing_item_refund.py` |
| Money deducted, order not placed | refund, confirm or explain | paid late, failed (auto-reversal), pending, paid twice | payment switch status | `DiscreteWorlds`, `OutcomeTable(reveals=...)`, `rules.amount_above`, `replay`, `resplit` | [`payments_upi.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/payments_upi.py) |
| "I don't recognise this payment" | block the card or explain | auth hold, subscription, add-on card, fraud | transaction lookup | `DiscreteWorlds` (message cues + lookup fields), `rules.amount_above` | [`card_dispute.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/card_dispute.py) |
| "Fraudulent user?" | block an account | fraud ring, hostel, bad batch, account takeover | return context | `rules.needs_person` (labels on people), reversible limit certified | pattern of `card_dispute.py` |
| Home or office? | checkout to which address | home, office, already ordered | ask the user once | MCP proxy (`riskgate serve-mcp`), `OutcomeTable` | [`integrations/mcp_proxy_demo.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/integrations/mcp_proxy_demo.py) |
| One alert, many causes | add replicas | causes that fit the telemetry | — | conformal plausible set | [`benchmarks/sre`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/sre) |
| When the plant can't tell | change an operator setpoint | faults that fit the sensors | backup sensor | core `Gate`, reactor twin, STL | [`cstr_reactor.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/cstr_reactor.py), [`benchmarks/tep`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/tep) |
| Idle, until it isn't | delete or deallocate an idle VM | dormant, scheduled, live | — | ranked irreversible + reversible actions, `unsafe_mass` | [`idle_vm_cleanup.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/idle_vm_cleanup.py) |
| Thousands of cleanups, no tickets | the error budget as a dial | as above | — | `sweep` | [`idle_vm_cleanup.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/idle_vm_cleanup.py) |
| The file that gave orders | pay an account named in a file | nothing planted, output *k* planted | — | `riskgate.provenance`, `nearest_violation` | [`prompt_injection.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/use_cases/prompt_injection.py), [`benchmarks/agentdojo`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/agentdojo) |
| The upgrade trap | keep the calibration after a model swap | — | — | `doer_id` in the fingerprint (`VOID` after a swap), `recalibrate` | [`benchmarks/tau2`](https://github.com/reallysaurabh/riskgate/tree/main/benchmarks/tau2) |

## Discrete worlds

```python
import riskgate as rg

worlds = rg.DiscreteWorlds(
    prior={"paid_late": 0.35, "failed": 0.40, "pending": 0.13, "paid_twice": 0.12},
    evidence={
        "app_status": {"paid_late": {"pending": 0.65, "no order": 0.35}, "failed": {"failed": 0.5, "no order": 0.5},
                       "pending": {"pending": 0.7, "no order": 0.3}, "paid_twice": {"confirmed": 0.6, "pending": 0.4}},
        "two_debit_sms": {"paid_late": 0.02, "failed": 0.02, "pending": 0.02, "paid_twice": 0.75},   # boolean field
        "switch": {"paid_late": {"SUCCESS": 0.92, "PENDING": 0.08}, "failed": {"FAILED": 0.88, "PENDING": 0.12},
                   "pending": {"PENDING": 1.0}, "paid_twice": {"DUPLICATE": 1.0}},                   # set by the lookup
    })
table = rg.OutcomeTable(
    {"confirm_order": ["paid_late"], "reply_auto_reversal": ["failed"], "issue_refund": ["paid_twice"]},
    reveals={"check_txn_status": {"paid_late": "SUCCESS", "failed": "FAILED", "pending": "PENDING",
                                  "paid_twice": "DUPLICATE"}})
gate = rg.Gate(table, table.spec, worlds, probes=table.probes(), noop=table.noop,
               min_separation=table.separation / 2, doer_id="support-bot:v1")
```

- **Scores** are `-log P(world | evidence)`. Fields that are missing from the context, or `None`, are skipped. One
  provider therefore serves both stages: after the lookup, put its reading into the context and call
  `check_ranked(..., stage=2)`.
- **Probabilities are clipped** at `floor` (1e-6), so one surprising value cannot make a world impossible.
- **`OutcomeTable`**: actions it does not name are unsafe in every world (fail closed), and read-only actions are
  safe. For decisions that depend on arguments (refund amounts), pass a callable `safe(action, world, context)`.
- **Lookups**: two worlds that give the same reading (`PENDING`) are *not* told apart, so after such a reading the
  gate asks a person. Pass `noop=table.noop`: a lookup is admissible when it is no worse than doing nothing.
- **Fingerprints**: the tables are digested into `version`, so editing a probability or a safe set voids old
  calibrations.
- **Score functions**: the default (minimum margin) is all or nothing per world. `unsafe_mass` (−log posterior mass
  of the worlds where the action is unsafe) and `nearest_violation` are graded, so calibration can trade a little risk
  for a lot of autonomy. That is what makes the error-budget dial of the idle-VM story work.

> [!TIP]
> If `Calibration.probing` is `False` after calibration, there were too few episodes, or too many stage-1 errors, for
> a stage-2 threshold. Lowering `delta` (a larger plausible set) or `probe_share` usually fixes it; `card_dispute.py`
> uses `delta=0.02`.

## Provenance

```python
from riskgate.provenance import AgentContext, ProvenanceModel, ProvenanceWorlds, ToolOutput

gate = rg.Gate(ProvenanceModel(targets={"send_money": ["recipient"]}), ProvenanceModel.spec, ProvenanceWorlds(),
               score_fn=rg.nearest_violation, doer_id="banking-agent:v3")
gate.calibrate(labelled_calls)
ctx = AgentContext(prompt="Pay my rent as the landlord's notice says.",
                   outputs=[ToolOutput("read_file:notice.txt", text)], trusted_values=["<the user's own IBAN>"])
gate.check(rg.Action("send_money", {"recipient": iban, "amount": 50}), ctx)
```

- **Hypotheses**: "nothing planted", plus one per untrusted output, "output *k* carries planted orders", scored by
  how much the output reads like orders addressed to the agent (`instruction_score`).
- **Violation**: under "output *k* was planted", a state-changing call is unsafe if a target value came from output
  *k*. Values found in no output and not in the prompt count against every output: unknown provenance cannot be ruled
  out.
- **The scorer** is deliberately simple and LLM-free; replace it with `ProvenanceWorlds(instruction_score=...)`. A
  weaker scorer costs autonomy, not validity.

## Replay

```python
report = rg.replay(gate, held_out)           # gate calibrated on OTHER episodes
print(report)       # replayed 2000 episodes at alpha=0.05 [valid]: unsafe 1.5% (95% CI 1.0%-2.1%; agent alone 88.8%) ...
study = rg.resplit(make_gate, episodes, alpha=0.05, n_splits=50, groups=task_ids)
rows = rg.sweep(make_gate, calibration, test, alphas=(0.10, 0.05, 0.02))   # where the work goes at each budget
```

- **Counting**: a decision is unsafe when the gate certified an action labelled unsafe, or a probe caused harm.
  PROBE verdicts are followed through `Episode.after_probe` exactly as in deployment, and a probe without post-probe
  data counts as ASK.
- **Hold out**: replaying the episodes you calibrated on is optimistic. `resplit` holds out for you, by group if you
  pass `groups`.

## Rules

```python
from riskgate.rules import amount_above, needs_person, never

hard_rules = [needs_person("block_account", reason="labels on people are decided by a person"),
              amount_above(10_000, tools=("issue_refund",)),         # ASK above the supervisor limit
              never("drop_database")]
```

- **Fail closed**: an amount that is not a number fires `amount_above`.
- **Pinned versions**: every rule's `version` is built from its parameters, so changing a limit voids old
  calibrations, and rebuilding the same rule does not.
