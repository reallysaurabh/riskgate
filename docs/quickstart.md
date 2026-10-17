# Quickstart

```bash
pip install riskgate            # once published; until then: pip install -e .   (from a clone)
```

The complete runnable file is [`examples/quickstart.py`](https://github.com/reallysaurabh/riskgate/blob/main/examples/quickstart.py). It walks through five steps on a toy problem: an agent tops up a tank that may have a hidden leak.

## 1. A consequence model: what would this action do?

```python
import riskgate as rg

class TankModel:
    version = "tank-model-v1"            # bump on every change: old calibrations become VOID
    def predict(self, action, hyp, ctx):
        return {"level": ctx["level"] + action.args["v"] - hyp.params["leak"]}
```

In a real deployment this is your digital twin, simulator, `--dry-run`, database savepoint or learned model.

## 2. Hypotheses: what could be going on?

```python
class LeakDiagnoser:
    version = "leak-diag-v1"
    def hypotheses(self, ctx):          # lower score = explains the evidence better
        return [rg.Hypothesis(f"leak={l}", family=f"leak{l}", params={"leak": l},
                              score=abs(ctx["flow"] - l) / 8) for l in (0, 15, 30)]
```

## 3. The safety spec and the gate

```python
spec = rg.Spec(rg.Envelope("level", lo=10, hi=100, scale=10))    # margin > 0 means safe
gate = rg.Gate(TankModel(), spec, LeakDiagnoser(), alpha=0.05)
```

## 4. Calibrate on labelled episodes

```python
episodes = [rg.Episode(ctx, ranked_actions, unsafe_labels, true_family="leak15"), ...]
cal = gate.calibrate(episodes)               # sets tau, lambda-hat; status -> VALID
cal.save("tank.cal.json", key=SECRET)        # signed, fingerprinted artefact
```

## 5. Use it

Check proposals directly:

```python
d = gate.check(rg.Action("add", {"v": 40}), {"level": 80, "flow": 15.0})
d.verdict, d.reason, d.plausible, d.score, d.status
```

Real output from the example:

```
calibrated: lambda=-1.76, tau=1.89, n=420, status=valid
CERTIFY  add(5) at level=30, flow=0.0: plausible=['leak=0', 'leak=15'], robust margin=1.00
         reason: robust margin above calibrated threshold
REJECT   add(40) at level=80, flow=0.0: plausible=['leak=0', 'leak=15'], robust margin=-2.00
         reason: unsafe under every plausible hypothesis (binding constraint: 10<=level<=100)
ASK      add(40) at level=80, flow=15.0: plausible=['leak=0', 'leak=15', 'leak=30'], robust margin=-2.00
         reason: cannot certify: best robust margin -2.000 < threshold -1.763; worst case under hypothesis 'leak=0'
```

Or guard the function your agent framework calls as a tool:

```python
@rg.guard(gate, context=lambda: current_state(), on_ask=page_human)   # on_ask returns True to approve
def add(v: float) -> str:
    ...

add(20)   # certified -> runs and returns "added 20 L"
add(80)   # -> {'status': 'rejected', 'reason': 'unsafe under every plausible hypothesis ...', 'guarantee': 'valid'}
```

The dict is returned to the agent as the tool result, so it can re-plan. With the OpenAI Agents SDK, LangGraph, Google ADK or MCP, see [Integrations](integrations.md).

## Next steps
- [End-to-end tutorial](end_to_end.md): a production-like lifecycle with probes, outcomes, drift and recalibration.
- [Bring your own domain](byo_domain.md): templates for your own simulator and diagnoser.
- [Calibration & guarantees](calibration.md): what exactly is promised.
