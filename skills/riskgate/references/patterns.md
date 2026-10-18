# riskgate patterns

All snippets assume `import riskgate as rg`. API reference: https://github.com/reallysaurabh/riskgate/blob/main/docs/api.md

## A. One hidden fact with a few values (most support, payments, ops decisions)

```python
worlds = rg.DiscreteWorlds(
    prior={"paid_late": 0.35, "failed": 0.40, "pending": 0.13, "paid_twice": 0.12},
    evidence={                      # field -> world -> {value: P(value | world)}, or world -> P(true) for booleans
        "app_status": {"paid_late": {"pending": 0.65, "no order": 0.35}, "failed": {"failed": 0.5, "no order": 0.5},
                       "pending": {"pending": 0.7, "no order": 0.3}, "paid_twice": {"confirmed": 0.6, "pending": 0.4}},
        "two_debit_sms": {"paid_late": 0.02, "failed": 0.02, "pending": 0.02, "paid_twice": 0.75},
        "switch": {"paid_late": {"SUCCESS": 0.92, "PENDING": 0.08}, "failed": {"FAILED": 0.88, "PENDING": 0.12},
                   "pending": {"PENDING": 1.0}, "paid_twice": {"DUPLICATE": 1.0}},   # set after the lookup
    })
table = rg.OutcomeTable(
    {"confirm_order": ["paid_late"], "reply_auto_reversal": ["failed"], "issue_refund": ["paid_twice"]},
    reveals={"check_txn_status": {"paid_late": "SUCCESS", "failed": "FAILED", "pending": "PENDING",
                                  "paid_twice": "DUPLICATE"}})
gate = rg.Gate(table, table.spec, worlds, probes=table.probes(), noop=table.noop,
               min_separation=table.separation / 2, doer_id="support-bot:v1")
```

- **Contexts** are dicts (or objects) with the evidence fields. A field that is missing or `None` has not been observed yet.
- **Two stages.** At stage 1 the gate may return PROBE. Run the lookup, put its reading into the context, then call
  `gate.check_ranked(candidates, ctx_with_reading, stage=2)`.
- **Unnamed actions** in `OutcomeTable` are unsafe in every world (fail closed). Read-only actions are safe.
- **Learning the tables from data:** `rg.DiscreteWorlds.fit(contexts, worlds)`, on data disjoint from calibration.
- **Graded scores:** `score_fn=rg.unsafe_mass` (−log posterior mass of the worlds where the action is unsafe) lets
  calibration trade a little risk for a lot of autonomy when the worlds have graded plausibility.
- **No probe chosen?** If `Calibration.probing` is `False` after calibration, add episodes or lower `delta` (e.g. 0.02).

## B. Amounts (refund, credit, transfer size)

```python
def safe(action, world, ticket):            # dry-run: never pay more than is owed in this world
    owed = ticket["price"] if world == "missing" else 0.0
    return float(action.args.get("amount", 0.0)) <= owed
table = rg.OutcomeTable(safe, reveals={"read_pick_log_and_ledger": {w: w for w in ("missing", "refunded", "second_bag")}})
```

## C. Prompt injection (instructions hidden in files, e-mails, web pages)

```python
from riskgate.provenance import AgentContext, ProvenanceModel, ProvenanceWorlds, ToolOutput
gate = rg.Gate(ProvenanceModel(targets={"send_money": ["recipient"]}), ProvenanceModel.spec, ProvenanceWorlds(),
               score_fn=rg.nearest_violation, doer_id="bank-agent:v3")
ctx = AgentContext(prompt=user_request, outputs=[ToolOutput("read_file:notice.txt", text)],
                   trusted_values=[users_own_iban])
```

A payment whose recipient came only from an instruction-like output is not certified. A bill that names an account is.

## D. A simulator, digital twin or dry-run (continuous outcomes)

```python
class Twin:
    version = "twin-v3"                       # bump on every change: old calibrations become VOID
    def predict(self, action, hypothesis, ctx):
        return {"p95_ms": ..., "error_rate": ..., "observation": [...]}   # observation: what a probe would read
spec = rg.Spec(rg.Upper("p95_ms", 800, scale=800), rg.Upper("error_rate", 0.02, scale=0.02))
```

Full guide: https://github.com/reallysaurabh/riskgate/blob/main/docs/byo_domain.md

## Rules (deterministic policy, checked before anything else)

```python
from riskgate.rules import amount_above, needs_person, never
hard_rules = [needs_person("block_account"), amount_above(10_000, tools=("issue_refund",)), never("drop_database")]
```

## Labelled episodes, calibration, replay

```python
ep = rg.Episode(context, candidates=[a1, a2], unsafe={a1.key: True, a2.key: False}, true_family="failed",
                after_probe={"check_txn_status": episode_after_the_lookup})
gate.calibrate(cal_episodes, alpha=0.05, groups=[task_id_of(e) for e in cal_episodes])
print(rg.replay(gate, held_out))                          # unsafe vs alpha, agent alone, autonomous, asked, probed
study = rg.resplit(make_gate, episodes, alpha=0.05, n_splits=20, groups=task_ids)   # validity over re-splits
rows = rg.sweep(make_gate, cal_episodes, held_out, alphas=(0.10, 0.05, 0.02))       # the error budget as a dial
```

Calibration needs at least ceil(1/alpha) independent groups after the 30% plausible-set split. With fewer, the
threshold is infinite and every check returns ASK. That is correct behaviour, not a bug.
