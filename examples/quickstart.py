"""riskgate quickstart: put a calibrated gate in front of a tool in ~60 lines.

    cd examples && PYTHONPATH=.. python quickstart.py

The toy world: an agent tops up a tank. The tank may have a hidden leak (0, 15 or 30 L per step) that we can
only infer from a noisy flow reading. The level must stay between 10 L and 100 L.
"""
import numpy as np

import riskgate as rg


# 1) Consequence model: what happens if we add v litres, assuming a given leak?   (your simulator / dry-run)
class TankModel:
    version = "tank-model-v1"            # bump whenever the model changes: it voids old calibrations

    def predict(self, action, hyp, ctx):
        return {"level": ctx["level"] + action.args["v"] - hyp.params["leak"]}


# 2) Hypotheses: which leak explains the evidence? Lower score = better fit.     (your diagnoser)
class LeakDiagnoser:
    version = "leak-diag-v1"

    def hypotheses(self, ctx):
        return [rg.Hypothesis(f"leak={l}", family=f"leak{l}", params={"leak": l}, score=abs(ctx["flow"] - l) / 8)
                for l in (0, 15, 30)]


# 3) Safety spec: signed margin, positive = safe (here in units of 10 L).
spec = rg.Spec(rg.Envelope("level", lo=10, hi=100, scale=10))
gate = rg.Gate(TankModel(), spec, LeakDiagnoser(), alpha=0.05,    # at most 5 % unsafe certified actions
               doer_id="random-ranker-v1")                          # the agent whose proposals we calibrate on

# 4) Calibration data: past decision points with what actually happened (here simulated).
CANDIDATES = [rg.Action("add", {"v": v}) for v in (40, 20, 5, 0)]


def past_incident(rng):
    leak, level = rng.choice([0, 15, 30]), rng.uniform(25, 90)
    ctx = {"level": level, "flow": leak + rng.normal(0, 8)}
    ranked = list(rng.permutation(CANDIDATES))                       # whatever the agent proposed, in its order
    unsafe = {a.key: not 10 <= level + a.args["v"] - leak <= 100 for a in ranked}
    return rg.Episode(ctx, ranked, unsafe, true_family=f"leak{leak}")


rng = np.random.default_rng(0)
cal = gate.calibrate([past_incident(rng) for _ in range(600)])
print(f"calibrated: lambda={cal.lam:.2f}, tau={cal.tau:.2f}, n={cal.n}, status={gate.status().value}")

# 5a) Check proposals directly. Scan a few situations and show one example of each verdict.
seen = {}
for level in (30, 60, 80, 95):
    for flow in (0.0, 7.0, 15.0, 30.0):
        for v in (5, 20, 40):
            d = gate.check(rg.Action("add", {"v": v}), {"level": level, "flow": flow})
            seen.setdefault(d.verdict.value, (level, flow, v, d))
for verdict, (level, flow, v, d) in seen.items():
    print(f"{verdict.upper():8s} add({v}) at level={level}, flow={flow}: plausible={d.plausible}, "
          f"robust margin={d.score:.2f}\n         reason: {d.reason}")

# 5b) ... or guard the tool function your agent framework calls.
state = {"level": 60, "flow": 1.0}


@rg.guard(gate, context=lambda: state, on_ask=lambda d: False)    # on_ask: your human-approval hook
def add(v: float) -> str:
    state["level"] += v
    return f"added {v} L"


print(add(20))     # certified -> runs
print(add(80))     # not certified -> structured feedback for the agent instead of running
