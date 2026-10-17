"""'The butter never came': check the pick log and refund ledger before paying for a missing item (simulated tickets).

Three worlds fit "item missing": never packed (a refund is owed), out of stock and already auto-refunded (a refund
pays twice), or packed in a second bag still on its way (a refund pays for a delivered item). The consequence model is
a dry-run of the refund amount against what is owed in each world, so a partial or excessive refund is caught too.

    python missing_item_refund.py
"""
from __future__ import annotations

import numpy as np

import riskgate as rg
from riskgate.rules import amount_above

PRIOR = {"missing": 0.5, "oos_refunded": 0.3, "second_bag": 0.2}                          # assumed mix, not data
WORLDS = rg.DiscreteWorlds(PRIOR, {
    "mentions_unavailable": {"missing": 0.05, "oos_refunded": 0.35, "second_bag": 0.05},
    "two_bags": {"missing": 0.35, "oos_refunded": 0.35, "second_bag": 1.0},
    "ledger": {w: {w: 1.0} for w in PRIOR},                                                  # filled in by the probe
}, name="missing-item-worlds")


def refund_is_safe(action: rg.Action, world: str, ticket: dict) -> bool:
    """Dry-run: a refund is safe iff it does not pay more than is owed in this world."""
    owed = ticket["price"] if world == "missing" else 0.0
    return float(action.args.get("amount", 0.0)) <= owed + 1e-9


TABLE = rg.OutcomeTable(refund_is_safe, reveals={"read_pick_log_and_ledger": {w: w for w in PRIOR}})


def make_gate(**kw) -> rg.Gate:
    kw.setdefault("probes", TABLE.probes())
    return rg.Gate(TABLE, TABLE.spec, WORLDS, noop=TABLE.noop, min_separation=TABLE.separation / 2, doer_id="support-bot:v1",
                   hard_rules=[amount_above(500, reason="refunds above Rs 500 need a supervisor")], **kw)


def sample(seed: int) -> rg.Episode:
    rng = np.random.default_rng(seed)
    world = str(rng.choice(list(PRIOR), p=list(PRIOR.values())))
    ticket = {"price": float(rng.choice([49, 62, 99, 145, 230])),
              "mentions_unavailable": bool(rng.random() < WORLDS.evidence["mentions_unavailable"][world]),
              "two_bags": bool(rng.random() < WORLDS.evidence["two_bags"][world])}
    cands = [rg.Action("issue_refund", {"amount": ticket["price"]})]
    unsafe = {a.key: not refund_is_safe(a, world, ticket) for a in cands}
    after = rg.Episode({**ticket, "ledger": world}, cands, unsafe, true_family=world)
    return rg.Episode(ticket, cands, unsafe, true_family=world, after_probe={"read_pick_log_and_ledger": after})


def main() -> None:
    eps = [sample(i) for i in range(4000)]
    gate = make_gate()
    gate.calibrate(eps[:2500], alpha=0.05)
    print(rg.replay(gate, eps[2500:]))
    butter = {"price": 49.0, "mentions_unavailable": False, "two_bags": False}
    refund = [rg.Action("issue_refund", {"amount": 49.0})]
    print("\nAmul Butter 100 g, Rs 49:", gate.check_ranked(refund, butter).verdict.value.upper())
    for seen in ("oos_refunded", "missing"):
        d = gate.check_ranked(refund, {**butter, "ledger": seen}, stage=2)
        print(f"  ledger says {seen:13s}: {d.verdict.value.upper():7s} {d.reason}")


if __name__ == "__main__":
    main()
