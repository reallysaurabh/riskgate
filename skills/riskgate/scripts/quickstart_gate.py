"""riskgate in one file: a payments gate that probes, certifies, rejects and asks (simulated cases).

    pip install --pre riskgate
    python quickstart_gate.py

"Money deducted, order not placed": refunding now is wrong unless the customer was charged twice. The gate reads the
payment switch first (PROBE), then certifies the remedy for what it found. Every number printed is computed on
simulated tickets with an assumed mix of outcomes; it shows the mechanism, not results on real data.
"""
from __future__ import annotations

import numpy as np

import riskgate as rg
from riskgate.rules import amount_above

PRIOR = {"paid_late": 0.35, "failed": 0.40, "pending": 0.13, "paid_twice": 0.12}
EVIDENCE = {
    "app_status": {"paid_late": {"pending": 0.65, "no order": 0.35}, "failed": {"failed": 0.5, "no order": 0.5},
                   "pending": {"pending": 0.7, "no order": 0.3},
                   "paid_twice": {"confirmed": 0.55, "pending": 0.30, "no order": 0.15}},
    "switch": {"paid_late": {"SUCCESS": 0.92, "PENDING": 0.08}, "failed": {"FAILED": 0.88, "PENDING": 0.12},
               "pending": {"PENDING": 1.0}, "paid_twice": {"DUPLICATE": 1.0}},          # filled in by the lookup
}
WORLDS = rg.DiscreteWorlds(PRIOR, EVIDENCE, name="quickstart-payments")
TABLE = rg.OutcomeTable({"confirm_order": ["paid_late"], "reply_auto_reversal": ["failed"], "issue_refund": ["paid_twice"]},
                        reveals={"check_txn_status": {"paid_late": "SUCCESS", "failed": "FAILED", "pending": "PENDING",
                                                      "paid_twice": "DUPLICATE"}})


def make_gate() -> rg.Gate:
    return rg.Gate(TABLE, TABLE.spec, WORLDS, probes=TABLE.probes(), noop=TABLE.noop,
                   min_separation=TABLE.separation / 2, doer_id="support-bot:v1",
                   hard_rules=[amount_above(10_000, tools=("issue_refund",))])


def proposals(amount: float) -> list[rg.Action]:
    return [rg.Action("issue_refund", {"amount": amount}), rg.Action("confirm_order"), rg.Action("reply_auto_reversal")]


def _pick(rng: np.random.Generator, table: dict) -> str:
    keys = list(table)
    return keys[int(rng.choice(len(keys), p=np.array(list(table.values())) / sum(table.values())))]


def sample(seed: int) -> rg.Episode:
    rng = np.random.default_rng(seed)
    world = _pick(rng, PRIOR)
    ctx = {"app_status": _pick(rng, EVIDENCE["app_status"][world])}
    cands = proposals(float(rng.integers(150, 9_000)))
    unsafe = {a.key: not TABLE.is_safe(a, world) for a in cands}
    after = rg.Episode({**ctx, "switch": _pick(rng, EVIDENCE["switch"][world])}, cands, unsafe, true_family=world)
    return rg.Episode(ctx, cands, unsafe, true_family=world, after_probe={"check_txn_status": after})


def main() -> None:
    eps = [sample(i) for i in range(8000)]
    gate = make_gate()
    gate.calibrate(eps[:5000], alpha=0.05)
    print("held-out replay (simulated):", rg.replay(gate, eps[5000:]))
    ticket = {"app_status": "no order"}
    show = lambda tag, d: print(f"  {tag:37s} {d.verdict.value.upper():8s} {d.action.name if d.action else ''}  ({d.reason})")  # noqa: E731
    show("ticket, before any lookup:", gate.check_ranked(proposals(2100), ticket))
    for reading in ("FAILED", "DUPLICATE", "PENDING"):
        show(f"after lookup, switch says {reading}:", gate.check_ranked(proposals(2100), {**ticket, "switch": reading}, stage=2))
    show("refund above the Rs 10,000 rule:", gate.check_ranked(proposals(12_000), {"app_status": "confirmed", "switch": "DUPLICATE"}, stage=2))


if __name__ == "__main__":
    main()
