# Use-case examples

Small, self-contained scripts, one per family of the use cases on the RiskGate site (see [docs/use_cases.md](../../docs/use_cases.md)).
They need only `riskgate` (numpy). Every case, world mix and probability in them is **simulated**: the scripts show how
to build a gate with the building blocks, not evidence about any real platform, bank or cloud.

| Script | Story | Building blocks |
|---|---|---|
| `payments_upi.py` | "Money deducted, order not placed" | `DiscreteWorlds`, `OutcomeTable` (probe: payment switch), `rules.amount_above`, `replay`, `resplit` |
| `missing_item_refund.py` | "The butter never came" | `OutcomeTable(safe=callable)` with amounts, ledger probe, `rules.amount_above` |
| `card_dispute.py` | "I don't recognise this payment" | `DiscreteWorlds` (message cues + lookup fields), `rules.needs_person` |
| `idle_vm_cleanup.py` | "Idle, until it isn't" / "Thousands of cleanups" | ranked irreversible + reversible actions, `sweep` (the error-budget dial) |
| `from_traces.py` | onboarding from traces | `riskgate.traces`, `DiscreteWorlds.fit`, grouped calibration |
| `prompt_injection.py` | "The file that gave orders" | `riskgate.provenance`, `nearest_violation` |

```bash
cd examples/use_cases && python payments_upi.py
```
