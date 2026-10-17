"""'I don't recognise this payment': look the transaction up before blocking a card (simulated cases).

Four situations fit the message: an authorisation hold, a subscription under an unfamiliar name, a purchase on the
add-on card, or fraud. Blocking is right only for fraud; explaining is right in the other three. Neither is safe in
every world before the lookup, so the gate reads the transaction first. Amounts above GBP 1,000 go to the fraud desk.

    python card_dispute.py
"""
from __future__ import annotations

import numpy as np

import riskgate as rg
from riskgate.rules import amount_above

PRIOR = {"auth_hold": 0.25, "subscription": 0.30, "addon_card": 0.15, "fraud": 0.30}       # assumed mix, not data
CUES = {                                                         # P(cue in the message or account | world), assumed
    "says_pending": {"auth_hold": 0.35, "subscription": 0.03, "addon_card": 0.04, "fraud": 0.05},
    "says_sure_fraud": {"auth_hold": 0.04, "subscription": 0.06, "addon_card": 0.08, "fraud": 0.40},
    "has_addon_card": {"auth_hold": 0.25, "subscription": 0.25, "addon_card": 1.00, "fraud": 0.25},
}
LOOKUP = {                                                       # P(field value | world) returned by the lookup
    "state": {"auth_hold": {"hold": 0.95, "settled": 0.05}, "subscription": {"hold": 0.03, "settled": 0.97},
              "addon_card": {"hold": 0.15, "settled": 0.85}, "fraud": {"hold": 0.20, "settled": 0.80}},
    "history": {"auth_hold": {"none": 0.70, "seen": 0.30}, "subscription": {"none": 0.10, "recurring": 0.90},
                "addon_card": {"none": 0.55, "seen": 0.45}, "fraud": {"none": 0.97, "seen": 0.03}},
    "card": {"auth_hold": {"primary": 0.92, "addon": 0.08}, "subscription": {"primary": 0.92, "addon": 0.08},
             "addon_card": {"addon": 1.0}, "fraud": {"primary": 0.95, "addon": 0.05}},
    "burst": {"auth_hold": 0.02, "subscription": 0.03, "addon_card": 0.05, "fraud": 0.55},
}
WORLDS = rg.DiscreteWorlds(PRIOR, {**CUES, **LOOKUP}, name="card-dispute-worlds")
TABLE = rg.OutcomeTable({"block_card_and_dispute": ["fraud"], "explain_and_watch": ["auth_hold", "subscription", "addon_card"]},
                        reveals={"lookup_transaction": {w: w for w in PRIOR}})


def make_gate(**kw) -> rg.Gate:
    kw.setdefault("probes", TABLE.probes())
    # delta=0.02 keeps the true situation in the plausible set more often; at the default 0.05 too many episodes
    # certify the wrong remedy at stage 1 for a stage-2 threshold to exist (check Calibration.probing)
    kw.setdefault("delta", 0.02)
    return rg.Gate(TABLE, TABLE.spec, WORLDS, noop=TABLE.noop, min_separation=TABLE.separation / 2,
                   doer_id="dispute-bot:v1", hard_rules=[amount_above(1000, reason="fraud desk above GBP 1,000")], **kw)


def _draw(rng: np.random.Generator, row) -> object:
    if isinstance(row, dict):
        keys = list(row)
        return keys[int(rng.choice(len(keys), p=np.array(list(row.values())) / sum(row.values())))]
    return bool(rng.random() < row)


def sample(seed: int) -> rg.Episode:
    rng = np.random.default_rng(seed)
    world = str(_draw(rng, PRIOR))
    amount = float(round(np.exp(rng.uniform(np.log(3), np.log(1500))), 2))
    msg = {cue: _draw(rng, t[world]) for cue, t in CUES.items()}
    cands = [rg.Action("block_card_and_dispute", {"amount": amount}), rg.Action("explain_and_watch", {"amount": amount})]
    unsafe = {a.key: not TABLE.is_safe(a, world) for a in cands}
    looked = {**msg, **{f: _draw(rng, t[world]) for f, t in LOOKUP.items()}}
    after = rg.Episode(looked, cands, unsafe, true_family=world)
    return rg.Episode(msg, cands, unsafe, true_family=world, after_probe={"lookup_transaction": after})


def main() -> None:
    eps = [sample(i) for i in range(6000)]
    gate = make_gate()
    gate.calibrate(eps[:4000], alpha=0.05)
    print(rg.replay(gate, eps[4000:]))
    msg = {"says_pending": False, "says_sure_fraud": False, "has_addon_card": False}
    cands = [rg.Action("block_card_and_dispute", {"amount": 180.0}), rg.Action("explain_and_watch", {"amount": 180.0})]
    print("\n'I don't recognise this payment' (GBP 180):", gate.check_ranked(cands, msg).verdict.value.upper())
    # a hold the customer calls "pending" is told apart from fraud; without that cue fraud stays plausible
    # (posterior ~13% here) and the gate asks a person instead of guessing
    hotel = {**msg, "says_pending": True, "state": "hold", "history": "none", "card": "primary", "burst": False}
    fraud = {**msg, "state": "settled", "history": "none", "card": "primary", "burst": True}
    for name, ctx in (("hotel pre-authorisation", hotel), ("burst of new merchants", fraud)):
        d = gate.check_ranked(cands, ctx, stage=2)
        print(f"  lookup shows {name:24s}: {d.verdict.value.upper():7s} {d.action.name if d.verdict is rg.Verdict.CERTIFY else ''}")


if __name__ == "__main__":
    main()
