"""'Idle, until it isn't': delete only what is dormant in every plausible world, deallocate the rest (simulated VMs).

A VM idle for three weeks is dormant (nobody needs it), scheduled (a job is due later) or live (needed within a day).
The cleanup agent ranks an irreversible delete first and a reversible deallocate second. Delete is safe only if the VM
is dormant; deallocate is safe unless it is live (assumption: the notified owner restarts it within a day). The
error budget is a dial: tightening it moves work from delete to deallocate, not to people (``rg.sweep``).

The evidence fields mirror the ones computed from the public Azure VM trace in the RiskGate study (deployment
siblings, the subscription's wake-up history); the probabilities here are invented for the example.

    python idle_vm_cleanup.py
"""
from __future__ import annotations

import numpy as np

import riskgate as rg

PRIOR = {"dormant": 0.855, "scheduled": 0.12, "live": 0.025}                               # assumed, not data
EVIDENCE = {
    "blips": {"dormant": {"0": 0.80, "1-4": 0.17, "5+": 0.03}, "scheduled": {"0": 0.45, "1-4": 0.40, "5+": 0.15},
              "live": {"0": 0.30, "1-4": 0.45, "5+": 0.25}},
    "siblings_busy": {"dormant": {"single": 0.55, "none": 0.40, "some": 0.05},
                      "scheduled": {"single": 0.35, "none": 0.25, "some": 0.40},
                      "live": {"single": 0.30, "none": 0.10, "some": 0.60}},
    "subscription_wakes": {"dormant": {"never": 0.75, "sometimes": 0.22, "often": 0.03},
                           "scheduled": {"never": 0.25, "sometimes": 0.45, "often": 0.30},
                           "live": {"never": 0.20, "sometimes": 0.40, "often": 0.40}},
}
WORLDS = rg.DiscreteWorlds(PRIOR, EVIDENCE, name="idle-vm-worlds")
TABLE = rg.OutcomeTable({"delete_vm": ["dormant"], "deallocate_vm": ["dormant", "scheduled"]})
CANDIDATES = [rg.Action("delete_vm"), rg.Action("deallocate_vm")]


def make_gate(**kw) -> rg.Gate:
    # unsafe_mass scores an action by -log P(world where it is unsafe | evidence): a graded score, so the
    # calibrated threshold can trade a little risk for a lot of autonomy (with min-margin it is all or nothing)
    # delta=0.01: the plausible set keeps the rare "live" world 99% of the time. At the default 0.05 the misses alone
    # cost more than a 2% budget, and a 2% calibration would certify nothing
    kw.setdefault("delta", 0.01)
    return rg.Gate(TABLE, TABLE.spec, WORLDS, score_fn=rg.unsafe_mass, doer_id="cleanup-agent:idle21", **kw)


def sample(seed: int) -> rg.Episode:
    rng = np.random.default_rng(seed)
    world = str(rng.choice(list(PRIOR), p=list(PRIOR.values())))
    ctx = {f: str(rng.choice(list(t[world]), p=list(t[world].values()))) for f, t in EVIDENCE.items()}
    unsafe = {a.key: not TABLE.is_safe(a, world) for a in CANDIDATES}
    return rg.Episode(ctx, CANDIDATES, unsafe, true_family=world)


def main() -> None:
    eps = [sample(i) for i in range(12_000)]
    gate = make_gate()
    gate.calibrate(eps[:8000], alpha=0.05)
    print(rg.replay(gate, eps[8000:]))
    print("\nerror budget  delete  deallocate  asked  unsafe")
    for row in rg.sweep(make_gate, eps[:8000], eps[8000:], alphas=(0.10, 0.05, 0.02)):
        ex = row["executed"]
        print(f"   {100 * row['alpha']:4.0f}%    {100 * ex.get('delete_vm', 0):5.1f}%    {100 * ex.get('deallocate_vm', 0):5.1f}%"
              f"   {100 * row['asked']:4.1f}%  {100 * row['unsafe']:4.2f}%")


if __name__ == "__main__":
    main()
