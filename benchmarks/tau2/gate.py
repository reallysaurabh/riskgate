"""RiskGate for tau2-bench database writes (retail and airline), LLM-free.

For every write the gate builds a set of HYPOTHESES about what the customer actually wants and about facts only the
customer knows, scores each by how well it explains the conversation (nonconformity r(h) = -log posterior of a
transparent lexical evidence model, weights fixed a priori and frozen on the 2025 development runs), predicts the
CONSEQUENCE of the proposed write under each hypothesis (a symbolic effect model of the tau2 tools, checked against
the real tau2 environment on a sample, plus the domain policy evaluated on the database snapshot) and checks the SPEC
"the write has the intended effect and complies with the policy". The score is

    s(a) = min { r(h) : h plausible, margin(a, h) < 0 }      (+inf if the write is safe under every hypothesis)

i.e. the nonconformity of the most plausible world in which the write is wrong. Thresholding s at lambda is exactly
RiskGate's robust rule with plausible-set level tau = lambda, so lambda-hat is calibrated directly by CRC.

Hypothesis families
  retail   readings of the request: which order, which items (sub/supersets), which new variant, which payment
           method, which cancellation reason, which action type (the type the customer first asked for vs the one
           executed: catches agents that steer customers into a different action)
  airline  facts and readings: cancellation reason (insurance covers health/weather only), which reservation,
           whether compensation was requested and for which reservation, checked-bag counts, cabin, insurance;
           policy rules the tau2 API explicitly does NOT enforce ("the agent must make sure the rules apply"):
           cancellation eligibility, no flight changes on basic economy, unchanged origin/destination, no bag
           removal, free-bag allowance, compensation eligibility and amount, booking payment limits.

Inputs are only what a deployed gate sees before the write: the conversation so far and the database (snapshot).
Gold actions and the hidden user instructions are never used.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import datetime

import numpy as np

import riskgate as rg

BIG = 50.0                       # score when no plausible hypothesis makes the write wrong (clipped "+inf")
VERSION = "tau2-gate-v1"

# --------------------------------------------------------------------------------------------- text utilities
_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(l|liter|litre|liters|litres)\b")
_INCH = re.compile(r"(\d+(?:\.\d+)?)\s*(\"|”|inches|inch|in\b)")


def norm(t: str) -> str:
    t = (t or "").lower().replace("’", "'")
    t = _UNIT.sub(lambda m: f"{int(round(float(m.group(1)) * 1000))}ml", t)
    t = _INCH.sub(lambda m: f"{m.group(1)}-inch", t)
    t = re.sub(r"(\d)\s+(ml|gb|tb|mp|mm|cm|w|mah|hz|oz|kg|lb|lbs|inch|-inch)\b", r"\1\2", t)
    return " " + re.sub(r"[^a-z0-9#.\-+]+", " ", t) + " "


def has(text: str, phrase: str) -> bool:
    p = norm(phrase).strip()
    return bool(p) and f" {p} " in text


SIZE = {"xs": ["extra small", "size xs"], "s": ["small", "size s"], "m": ["medium", "size m"], "l": ["large", "size l"],
        "xl": ["extra large", "size xl"], "xxl": ["size xxl", "2xl"]}
GENERIC = {"the", "and", "with", "for", "set", "kit", "smart", "electric", "portable", "wireless", "digital", "pro"}


def value_mentioned(text: str, key: str, value) -> bool:
    """Is option `key=value` mentioned in (normalized) customer text?"""
    v = str(value).lower().strip()
    k = key.lower()
    if v in ("yes", "true"):
        return has(text, k) and not (has(text, f"non {k}") or has(text, f"not {k}") or has(text, f"no {k}"))
    if v in ("no", "false", "none"):
        return has(text, f"non {k}") or has(text, f"not {k}") or has(text, f"no {k}") or has(text, f"without {k}")
    if v in SIZE and k in ("size", "sizes"):
        return any(has(text, s) for s in SIZE[v])
    if len(v) <= 2 and not v.isdigit():
        return has(text, f"{k} {v}") or has(text, f"{v} {k}")
    if v.isdigit() and len(v) <= 2:                     # shoe sizes etc.: need the key nearby
        return has(text, f"{k} {v}") or has(text, f"size {v}")
    return has(text, v)


def name_tokens(name: str) -> list[str]:
    toks = [w for w in norm(name).split() if len(w) > 2 and w not in GENERIC]
    return toks or norm(name).split()


def product_mentioned(text: str, name: str) -> bool:
    return any(has(text, w) or has(text, w.rstrip("s")) for w in name_tokens(name))


# --------------------------------------------------------------------------------------------- context
class Ctx:
    """What the gate sees before a write: the conversation so far and the database records it may consult."""

    __slots__ = ("rec", "user_all", "user_first", "user_last", "agent_last", "snap", "domain")

    def __init__(self, rec: dict):
        self.rec = rec
        self.domain = rec["domain"]
        users = [p["text"] for p in rec["prefix"] if p["role"] == "user" and p.get("text")]
        self.user_all = norm(" ".join(users))
        self.user_first = norm(" ".join(users[:2]))
        self.user_last = norm(users[-1]) if users else " "
        self.agent_last = norm(next((p["text"] for p in reversed(rec["prefix"]) if p["role"] == "assistant" and p.get("text")), ""))
        self.snap = rec["snapshot"]


YES = re.compile(r"\b(yes|yeah|yep|yup|correct|confirm(ed)?|go ahead|proceed|please do|sounds good|that'?s right|"
                 r"that is right|sure|ok(ay)?|absolutely|definitely|please go)\b")


def confirmed(ctx: Ctx) -> bool:
    return bool(YES.search(ctx.user_last))


# --------------------------------------------------------------------------------------------- hypotheses = readings
class Reading:
    """One hypothesis: an intended write (tool + effect key) and/or facts, with log-likelihood relative to the agent's."""

    __slots__ = ("name", "family", "ll", "effect", "facts")

    def __init__(self, name: str, family: str, ll: float, effect: str | None, facts: dict | None = None):
        self.name, self.family, self.ll, self.effect, self.facts = name, family, ll, effect, facts or {}


INTENT_WORDS = {
    "change_items": ["exchange", "swap", "switch", "replace", "trade", "change the", "modify the", "different size",
                     "different color", "instead of", "upgrade"],
    "return": ["return", "send back", "send it back", "refund"],
    "cancel": ["cancel", "cancellation"],
    "address": ["address", "ship to", "shipping to", "deliver to", "moved"],
    "payment": ["payment method", "pay with", "pay using", "charge my", "switch payment", "change payment"],
}
TOOL_INTENT = {"exchange_delivered_order_items": "change_items", "modify_pending_order_items": "change_items",
               "return_delivered_order_items": "return", "cancel_pending_order": "cancel",
               "modify_pending_order_address": "address", "modify_user_address": "address",
               "modify_pending_order_payment": "payment"}
REASON_WORDS = {"ordered by mistake": ["mistake", "accident", "accidentally", "by error", "wrong order", "ordered wrong",
                                       "didn't mean", "did not mean"],
                "no longer needed": ["no longer need", "don't need", "do not need", "not need", "no longer want",
                                     "changed my mind", "don't want", "do not want", "no need"]}


def _count(text: str, words: Sequence[str]) -> int:
    return sum(text.count(f" {norm(w).strip()} ") for w in words)


def intent_readings(ctx: Ctx, name: str) -> list[Reading]:
    """Alternative action types: the customer's FIRST-stated intent vs the executed one (agent steering)."""
    mine = TOOL_INTENT.get(name)
    if mine is None:
        return []
    c_first = {k: _count(ctx.user_first, w) for k, w in INTENT_WORDS.items()}
    c_all = {k: _count(ctx.user_all, w) for k, w in INTENT_WORDS.items()}
    out = []
    for k in INTENT_WORDS:
        if k == mine:
            continue
        ev = 1.0 * min(c_first[k], 2) + 0.5 * min(c_all[k], 3) - 1.0 * min(c_first[mine], 2) - 0.5 * min(c_all[mine], 3)
        out.append(Reading(f"intent:{k}", "intent", ev - 1.0, f"intent:{k}"))     # -1: prior for the executed type
    return out


# ----- retail slot readings
def _orders(ctx: Ctx, status: str | Sequence[str]) -> dict:
    st = {status} if isinstance(status, str) else set(status)
    return {oid: o for oid, o in (ctx.snap.get("orders") or {}).items() if o["status"] in st}


def _order_ev(ctx: Ctx, oid: str, order: dict, all_ids: Sequence[str]) -> float:
    digits = oid.lstrip("#").lower()
    ment = [o for o in all_ids if f" {o.lstrip('#').lower()} " in ctx.user_all or f" #{o.lstrip('#').lower()} " in ctx.user_all]
    ev = 0.0
    if oid in ment:
        ev += 2.0
    elif ment:
        ev -= 1.0
    if f" {digits} " in ctx.user_all:
        ev += 0.0
    names = {it["name"] for it in order["items"]}
    ev += 0.7 * sum(product_mentioned(ctx.user_all, n) for n in names)
    return ev


def _variant_ll(ctx: Ctx, product: dict, old: dict, cand: dict) -> float:
    """Evidence for new variant `cand` of `product` replacing `old` (dicts of options)."""
    keys = set(old) | set(cand)
    vals: dict[str, set] = {k: set() for k in keys}
    for v in product["variants"].values():
        for k, x in v["options"].items():
            if k in vals:
                vals[k].add(str(x))
    same_all = any(has(ctx.user_all, p) for p in ("everything else the same", "all else the same", "same everything else",
                                                  "keep everything else", "other options the same", "everything else same"))
    hits = conf = 0.0
    for k in keys:
        ment = {x for x in vals[k] if value_mentioned(ctx.user_all, k, x)}
        if len(ment) > 1 and str(old.get(k)) in ment:
            ment.discard(str(old.get(k)))                  # the customer also described the current item
        keep_same = has(ctx.user_all, f"same {k}") or has(ctx.user_all, f"{k} the same")
        if ment:
            if str(cand.get(k)) in ment:
                hits += 1
            else:
                conf += 1
        elif keep_same or same_all:
            if str(cand.get(k)) != str(old.get(k)):
                conf += 0.5 if same_all and not keep_same else 1.0
    return 1.0 * hits - 2.0 * conf


def _pm_ll(ctx: Ctx, pm_id: str, pms: dict, original: str | None) -> float:
    pm = pms.get(pm_id, {})
    src = str(pm.get("source", pm_id.split("_")[0] if "_" in pm_id else ""))
    t = ctx.user_all
    ev = 0.0
    kw = {"paypal": ["paypal"], "gift_card": ["gift card", "giftcard", "gift"], "credit_card": ["credit card", "card"]}
    mine = _count(t, kw.get(src, []))
    others = sum(_count(t, kw[s]) for s in kw if s != src)
    ev += 1.5 * min(mine, 2) - 0.5 * min(others, 2)
    for f in ("brand", "last_four"):
        if pm.get(f) and has(t, str(pm[f])):
            ev += 2.0
    if original is not None:
        orig_words = any(has(t, p) for p in ("original payment", "original method", "same card", "same payment", "original card"))
        other_words = any(has(t, p) for p in ("other card", "different card", "another card", "not the original", "different payment"))
        if pm_id == original:
            ev += 1.5 * orig_words - 1.5 * other_words
        else:
            ev += 1.0 * other_words - 1.0 * orig_words
    return ev


def retail_readings(ctx: Ctx, name: str, args: dict) -> list[Reading]:
    """Agent's reading (ll 0, effect = its own) + single-slot alternatives + intent alternatives."""
    from extract import canon_args  # noqa: PLC0415  (same package)
    snap = ctx.snap
    user = snap.get("user") or {}
    pms = user.get("payment_methods") or {}
    products = snap.get("products") or {}
    orders_all = snap.get("orders") or {}
    me = canon_args(name, args)
    out = [Reading("agent", "agent", 0.0, me)]
    conf_bonus = 0.5 if confirmed(ctx) else 0.0
    oid = args.get("order_id")
    order = orders_all.get(oid)

    def alt(nm: str, fam: str, ll: float, new_args: dict, tool: str = name):
        out.append(Reading(nm, fam, ll - conf_bonus, canon_args(tool, new_args), {"tool": tool, "args": new_args}))

    if name in ("exchange_delivered_order_items", "modify_pending_order_items", "return_delivered_order_items") and order:
        status = "delivered" if name != "modify_pending_order_items" else "pending"
        items = list(args.get("item_ids") or [])
        news = list(args.get("new_item_ids") or [])
        by_id = {it["item_id"]: it for it in order["items"]}
        # (1) which order: another order with the same products
        pids = {by_id[i]["product_id"] for i in items if i in by_id}
        base_ev = _order_ev(ctx, oid, order, list(orders_all))
        for oid2, o2 in _orders(ctx, status).items():
            if oid2 == oid:
                continue
            m2 = {}
            for it in o2["items"]:
                if it["product_id"] in pids and it["product_id"] not in m2:
                    m2[it["product_id"]] = it
            if not m2 or len(m2) < len(pids):
                continue
            ev2 = _order_ev(ctx, oid2, o2, list(orders_all))
            # how well do the items in o2 match the customer's description, vs the agent's items?
            d = 0.0
            for i in items:
                if i in by_id and by_id[i]["product_id"] in m2:
                    p = products.get(by_id[i]["product_id"])
                    if p:
                        d += _variant_ll(ctx, p, {}, m2[by_id[i]["product_id"]]["options"]) - _variant_ll(ctx, p, {}, by_id[i]["options"])
            new_items = [m2[by_id[i]["product_id"]]["item_id"] if i in by_id and by_id[i]["product_id"] in m2 else i for i in items]
            alt(f"order:{oid2}", "order", ev2 - base_ev + 0.5 * d, {**args, "order_id": oid2, "item_ids": new_items})
        # (2) which items: superset / subset of the order's items
        if len(items) > 1:
            for i in items:
                nm = by_id.get(i, {}).get("name", "")
                ev = 0.5 - 1.5 * product_mentioned(ctx.user_all, nm)
                keep = [j for j in items if j != i]
                new_args = {**args, "item_ids": keep}
                if news:
                    new_args["new_item_ids"] = [n for j, n in zip(items, news) if j != i]
                alt(f"drop:{i}", "items", ev - 0.5, new_args)
        if name == "return_delivered_order_items":
            for it in order["items"]:
                if it["item_id"] in items:
                    continue
                ev = 1.5 * product_mentioned(ctx.user_all, it["name"]) - 1.0
                alt(f"add:{it['item_id']}", "items", ev - 0.5, {**args, "item_ids": items + [it["item_id"]]})
        # (3) which new variant (per exchanged item)
        for pos, (i, n) in enumerate(zip(items, news)):
            it = by_id.get(i)
            p = products.get(it["product_id"]) if it else None
            if not p:
                continue
            ll_me = _variant_ll(ctx, p, it["options"], p["variants"].get(n, {}).get("options", {}))
            for vid, v in p["variants"].items():
                if vid in (n, i) or not v.get("available", False):
                    continue
                ll_v = _variant_ll(ctx, p, it["options"], v["options"])
                nn = list(news)
                nn[pos] = vid
                alt(f"variant:{pos}:{vid}", "variant", ll_v - ll_me, {**args, "new_item_ids": nn})
        # (4) which payment method (tool constraints: returns go to the original method or a gift card)
        cur = args.get("payment_method_id")
        original = (order.get("payment_history") or [{}])[0].get("payment_method_id")
        ll_me = _pm_ll(ctx, cur, pms, original) if cur else 0.0
        for pid in pms:
            if pid == cur:
                continue
            if name == "return_delivered_order_items" and not (pid == original or pid.startswith("gift_card")):
                continue
            alt(f"pm:{pid}", "payment", _pm_ll(ctx, pid, pms, original) - ll_me, {**args, "payment_method_id": pid})
        # (5) exchange <-> return of the same items
        if name == "exchange_delivered_order_items":
            ev_ret = 1.0 * _count(ctx.user_all, INTENT_WORDS["return"]) - 1.0 * _count(ctx.user_all, INTENT_WORDS["change_items"]) - 1.0
            alt("type:return", "type", ev_ret, {"order_id": oid, "item_ids": items, "payment_method_id": cur}, "return_delivered_order_items")
    elif name == "cancel_pending_order" and order:
        base_ev = _order_ev(ctx, oid, order, list(orders_all))
        for oid2, o2 in _orders(ctx, "pending").items():
            if oid2 != oid:
                alt(f"order:{oid2}", "order", _order_ev(ctx, oid2, o2, list(orders_all)) - base_ev, {**args, "order_id": oid2})
        r = args.get("reason")
        ll = {k: 2.0 * min(_count(ctx.user_all, w), 2) + 1.0 * min(_count(ctx.user_last, w), 1) for k, w in REASON_WORDS.items()}
        for k in REASON_WORDS:
            if k != r:
                alt(f"reason:{k}", "reason", ll[k] - ll.get(r, 0.0), {**args, "reason": k})
    elif name in ("modify_pending_order_address", "modify_pending_order_payment") and order:
        base_ev = _order_ev(ctx, oid, order, list(orders_all))
        for oid2, o2 in _orders(ctx, "pending").items():
            if oid2 != oid:
                alt(f"order:{oid2}", "order", _order_ev(ctx, oid2, o2, list(orders_all)) - base_ev, {**args, "order_id": oid2})
        if name == "modify_pending_order_payment":
            cur = args.get("payment_method_id")
            original = (order.get("payment_history") or [{}])[0].get("payment_method_id")
            ll_me = _pm_ll(ctx, cur, pms, None)
            for pid in pms:
                if pid not in (cur, original):
                    alt(f"pm:{pid}", "payment", _pm_ll(ctx, pid, pms, None) - ll_me, {**args, "payment_method_id": pid})
    if name in ("modify_pending_order_address", "modify_user_address"):
        # address fields the customer never wrote, and that are not an address already on file (default address or
        # the address of another order), are a reading the customer did not ask for
        miss = sum(1 for k in ("address1", "city", "zip") if args.get(k) and not has(ctx.user_all, str(args[k])))
        known = [user.get("address") or {}] + [o.get("address") or {} for o in orders_all.values()]
        on_file = any(str(a.get("address1", "")).lower() == str(args.get("address1", "")).lower()
                      and str(a.get("zip", "")) == str(args.get("zip", "")) for a in known)
        if miss and not on_file:
            out.append(Reading("address:as_typed", "address", 1.0 * miss - 1.0, "address:other"))
        if name == "modify_user_address" and _count(ctx.user_all, ["order", "this order", "the order"]) and not has(ctx.user_all, "default address"):
            out.append(Reading("address:order_only", "address", -1.0, "address:order_only"))
    out += intent_readings(ctx, name)
    return out


# ----- airline: facts + readings + policy
NOW = datetime(2024, 5, 15, 15, 0, 0)
FREE_BAGS = {"regular": {"basic_economy": 0, "economy": 1, "business": 2},
             "silver": {"basic_economy": 1, "economy": 2, "business": 3},
             "gold": {"basic_economy": 2, "economy": 3, "business": 4}}
REASONS = {"health": ["sick", "ill", "illness", "hospital", "medical", "surgery", "health", "doctor", "injur", "covid",
                      "flu", "unwell", "family emergency", "emergency"],
           "weather": ["weather", "storm", "hurricane", "snow", "flood", "tornado", "blizzard"],
           "change_of_plan": ["change of plan", "plans changed", "change in plans", "change in my plans", "no longer",
                              "can't make", "cannot make", "conflict", "rescheduled", "changed my mind", "don't need",
                              "not going", "something came up", "change of plans", "personal reasons"],
           "airline_cancelled": ["cancelled by the airline", "airline cancelled", "flight was cancelled", "flight got cancelled",
                                 "you cancelled", "canceled by the airline", "flight was canceled"]}


def _cabin(res: dict) -> str:
    return str(res.get("cabin", "")).replace(" ", "_")


def _statuses(ctx: Ctx, res: dict) -> list[str]:
    fl = ctx.snap.get("flights") or {}
    out = []
    for f in res.get("flights", []):
        st = (fl.get(f"{f['flight_number']}|{f['date']}") or {}).get("status") or {}
        out.append(str(st.get("status", "")))
    return out


def _created(res: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(str(res.get("created_at")).replace("Z", ""))
    except ValueError:
        return None


def _res_ev(ctx: Ctx, rid: str, res: dict, all_ids: Sequence[str]) -> float:
    t = ctx.user_all
    ment = [r for r in all_ids if f" {r.lower()} " in t]
    ev = 2.0 if rid in ment else (-1.0 if ment else 0.0)
    for code in {res.get("origin"), res.get("destination")}:
        if code and has(t, str(code)):
            ev += 0.7
    for f in res.get("flights", []):
        d = str(f.get("date", ""))
        if d and (has(t, d) or has(t, d[5:])):
            ev += 0.5
    return ev


def _reason_ll(ctx: Ctx) -> dict:
    t = ctx.user_all
    return {k: 1.5 * min(_count(t, w), 2) for k, w in REASONS.items()} | {"other": 0.0}


def cancel_allowed(ctx: Ctx, res: dict, reason: str) -> bool:
    st = _statuses(ctx, res)
    if any(s in ("landed", "flying") for s in st):
        return False                                           # a flown segment: only a human can help
    c = _created(res)
    within24 = c is not None and 0 <= (NOW - c).total_seconds() <= 24 * 3600
    airline_cancelled = any(s == "cancelled" for s in st)
    business = _cabin(res) == "business"
    insured = str(res.get("insurance", "no")) == "yes" and reason in ("health", "weather")
    return within24 or airline_cancelled or business or insured


def flight_violations(ctx: Ctx, res: dict, args: dict) -> list[str]:
    """Policy rules for update_reservation_flights that the tau2 API does not check."""
    viol = []
    new = {(f.get("flight_number"), f.get("date")) for f in args.get("flights", [])}
    old = {(f["flight_number"], f["date"]) for f in res.get("flights", [])}
    cabin_changed = str(args.get("cabin", "")).replace(" ", "_") != _cabin(res)
    flown = any(s in ("landed", "flying") for s in _statuses(ctx, res))
    if new != old and _cabin(res) == "basic_economy":
        viol.append("basic economy flights cannot be modified")
    if cabin_changed and flown:
        viol.append("cabin change after a flown segment")
    if new != old and flown:
        viol.append("flight change after a flown segment")
    fl = ctx.snap.get("flights") or {}
    legs = []
    for f in args.get("flights", []):
        info = fl.get(f"{f.get('flight_number')}|{f.get('date')}") or {}
        legs.append((str(f.get("date")), str(info.get("dep", "")), info.get("origin"), info.get("destination")))
    legs.sort()
    if legs and res.get("flight_type") == "one_way":
        o, d = legs[0][2], legs[-1][3]
        if o and d and (o != res.get("origin") or d != res.get("destination")):
            viol.append("origin/destination changed")
    if legs and res.get("flight_type") == "round_trip":
        if legs[0][2] and legs[0][2] != res.get("origin"):
            viol.append("origin changed")
    return viol


def airline_readings(ctx: Ctx, name: str, args: dict) -> list[Reading]:
    """Joint hypotheses (reading x hidden facts). A reading is CONSISTENT with the write iff it intends the same
    effect AND the write complies with the policy under that reading's facts (effect == the write's canonical key)."""
    from extract import canon_args  # noqa: PLC0415
    snap = ctx.snap
    user = snap.get("user") or {}
    resv = snap.get("reservations") or {}
    me = canon_args(name, args)
    out: list[Reading] = []
    rid = args.get("reservation_id")
    res = resv.get(rid)
    conf_bonus = 0.5 if confirmed(ctx) else 0.0
    membership = str(user.get("membership", "regular"))

    def add(nm, fam, ll, effect, facts=None):
        out.append(Reading(nm, fam, ll, effect, facts))

    # alternative reservations (inconsistent whatever the facts)
    if res is not None:
        base = _res_ev(ctx, rid, res, list(resv))
        for r2, x in resv.items():
            if r2 != rid and str(x.get("status", "")) != "cancelled":
                add(f"res:{r2}", "reservation", _res_ev(ctx, r2, x, list(resv)) - base - conf_bonus, f"res:{r2}")
    if name == "cancel_reservation" and res is not None:
        # hidden fact: the reason for cancelling (insurance covers health or weather only)
        for reason, ll in _reason_ll(ctx).items():
            ok = cancel_allowed(ctx, res, reason)
            add(f"reason:{reason}", "reason", ll - (0.5 if reason == "other" else 0.0), me if ok else "policy:cancel", {"reason": reason})
    elif name == "send_certificate":
        amount = float(args.get("amount", 0) or 0)
        asked = _count(ctx.user_all, ["compensation", "compensate", "voucher", "certificate", "make up for", "make it up",
                                      "reimburse", "reimbursement", "gesture", "credit for"])
        add("not requested", "asked", 0.5 - 1.5 * min(asked, 2), "policy:not_requested")
        for r2, x in resv.items():
            st = _statuses(ctx, x)
            n = len(x.get("passengers", []))
            eligible = membership in ("silver", "gold") or str(x.get("insurance")) == "yes" or _cabin(x) == "business"
            exp = 100 * n if "cancelled" in st else (50 * n if "delayed" in st else None)
            ok = eligible and exp is not None and abs(amount - exp) < 1e-6
            add(f"complaint:{r2}", "complaint", _res_ev(ctx, r2, x, list(resv)), me if ok else "policy:certificate")
    else:
        viol: list[str] = []
        if name == "update_reservation_flights" and res is not None:
            viol = flight_violations(ctx, res, args)
            for cab in ("basic economy", "economy", "business"):
                if cab.replace(" ", "_") != str(args.get("cabin", "")).replace(" ", "_"):
                    ev = 1.5 * has(ctx.user_all, cab) - 1.5 * has(ctx.user_all, str(args.get("cabin", "")))
                    add(f"cabin:{cab}", "cabin", ev - 1.0 - conf_bonus, f"cabin:{cab}")
        elif name == "update_reservation_baggages" and res is not None:
            tot, nonfree = int(args.get("total_baggages", 0) or 0), int(args.get("nonfree_baggages", 0) or 0)
            if tot < int(res.get("total_baggages", 0) or 0):
                viol.append("bags cannot be removed")
            free = FREE_BAGS.get(membership, FREE_BAGS["regular"]).get(_cabin(res), 0) * len(res.get("passengers", []))
            if nonfree != max(0, tot - free):
                viol.append(f"non-free bags should be {max(0, tot - free)}")
            nums = {int(x) for x in re.findall(r" (\d) ", ctx.user_all)}
            words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
            nums |= {v for k, v in words.items() if f" {k} bag" in ctx.user_all or f" {k} checked" in ctx.user_all}
            cands = set()
            for k in nums:
                cands |= {k, int(res.get("total_baggages", 0) or 0) + k}
            for cand in sorted(cands - {tot}):
                add(f"bags:{cand}", "bags", -0.5 - conf_bonus, f"bags:{cand}")
        elif name == "update_reservation_passengers" and res is not None:
            known = norm(" ".join(f"{p.get('first_name')} {p.get('last_name')} {p.get('dob')}" for p in
                                  (res.get("passengers", []) + list(user.get("saved_passengers", []) or []))))
            miss = sum(1 for p in args.get("passengers", []) for k in ("first_name", "last_name", "dob")
                       if p.get(k) and not has(ctx.user_all, str(p[k])) and not has(known, str(p[k])))
            if miss:
                add("passengers:as_typed", "passengers", 1.0 * miss - 1.0, "passengers:other")
        elif name == "book_reservation":
            pms = user.get("payment_methods") or {}
            kinds = [str((pms.get(p.get("payment_id")) or {}).get("source", "")) for p in args.get("payment_methods", [])]
            if kinds.count("certificate") > 1 or kinds.count("credit_card") > 1 or kinds.count("gift_card") > 3:
                viol.append("payment method limits")
            n = len(args.get("passengers", []))
            if n > 5:
                viol.append("more than five passengers")
            cab = str(args.get("cabin", "")).replace(" ", "_")
            free = FREE_BAGS.get(membership, FREE_BAGS["regular"]).get(cab, 0) * n
            tot, nonfree = int(args.get("total_baggages", 0) or 0), int(args.get("nonfree_baggages", 0) or 0)
            if nonfree != max(0, tot - free):
                viol.append(f"non-free bags should be {max(0, tot - free)}")
            ins = str(args.get("insurance", "no"))
            no_ins = any(has(ctx.user_all, p) for p in ("no insurance", "without insurance", "don't want insurance",
                                                        "do not want insurance", "no travel insurance", "skip the insurance",
                                                        "don't need insurance", "do not need insurance", "no to insurance"))
            yes_ins = has(ctx.user_all, "insurance") and not no_ins
            if (ins == "yes") != yes_ins:
                add("insurance:other", "insurance", -0.5 - conf_bonus, "insurance:other")
            known = norm(" ".join(f"{p.get('first_name')} {p.get('last_name')} {p.get('dob')}" for p in (user.get("saved_passengers", []) or [])))
            miss = sum(1 for p in args.get("passengers", []) for k in ("first_name", "last_name", "dob")
                       if p.get(k) and not has(ctx.user_all, str(p[k])) and not has(known, str(p[k])))
            if miss:
                add("passengers:as_typed", "passengers", 1.0 * miss - 1.0, "passengers:other")
        # the agent's own reading: consistent unless the write breaks a rule the database can check
        out.insert(0, Reading("agent", "agent", 0.0, f"policy:{'; '.join(viol)}" if viol else me, {"violations": viol}))
    return out


# --------------------------------------------------------------------------------------------- riskgate components
class IntentHypotheses:
    """riskgate HypothesisProvider: readings scored by negative log posterior (lower = explains the conversation better)."""

    version = VERSION

    def hypotheses(self, ctx: Ctx) -> list[rg.Hypothesis]:
        rec = ctx.rec
        rs = retail_readings(ctx, rec["name"], rec["args"]) if ctx.domain == "retail" else airline_readings(ctx, rec["name"], rec["args"])
        # policy/fact readings are mutually exclusive alternatives to "agent"; combine all into one posterior
        ll = np.array([r.ll for r in rs], float)
        logp = ll - (ll.max() + math.log(np.exp(ll - ll.max()).sum()))
        return [rg.Hypothesis(f"{r.name}", family=r.family, params={"effect": r.effect, "facts": r.facts}, score=float(-lp))
                for r, lp in zip(rs, logp)]


class EffectModel:
    """riskgate ConsequenceModel: the write's canonical effect vs the effect the hypothesis intends (+ policy verdict).
    The canonical effect reproduces the tau2 tools' database change (validated against the real environment)."""

    version = VERSION

    def predict(self, action: rg.Action, h: rg.Hypothesis, ctx: Ctx) -> dict:
        from extract import canon_args  # noqa: PLC0415
        intended = h.params["effect"]
        mine = canon_args(action.name, dict(action.args))
        return {"intended": intended is not None and intended == mine}


SPEC = rg.Spec(rg.Predicate("write has the intended, policy-compliant effect", lambda o: o["intended"]))


def nearest_unsafe(margins: np.ndarray, plaus: Sequence[rg.Hypothesis]) -> float:
    """Distance to the most plausible world in which the write is wrong (riskgate score_fn): absolute plausible set
    {h: r(h) <= tau}."""
    bad = [h.score for m, h in zip(margins, plaus) if m < 0]
    return float(min(bad)) if bad else BIG


nearest_unsafe.riskgate_version = "nearest-unsafe-v1"  # type: ignore[attr-defined]


def lr_gap(margins: np.ndarray, plaus: Sequence[rg.Hypothesis]) -> float:
    """Likelihood-ratio plausible set {h: r(h) <= r_min + tau}: the evidence gap between the best explanation and the
    best explanation under which the write is wrong (0 when the best explanation already makes it wrong)."""
    bad = [h.score for m, h in zip(margins, plaus) if m < 0]
    if not bad:
        return BIG
    return float(min(bad) - min(h.score for h in plaus))


lr_gap.riskgate_version = "lr-gap-v1"  # type: ignore[attr-defined]


def unsafe_mass(margins: np.ndarray, plaus: Sequence[rg.Hypothesis]) -> float:
    """-log posterior mass of the worlds in which the write is wrong (a Bayesian aggregation, for comparison)."""
    bad = np.array([h.score for m, h in zip(margins, plaus) if m < 0])
    if not bad.size:
        return BIG
    return float(-np.log(np.exp(-bad).sum()))


unsafe_mass.riskgate_version = "unsafe-mass-v1"  # type: ignore[attr-defined]
SCORE_FNS = {"nearest": nearest_unsafe, "lr": lr_gap, "mass": unsafe_mass}


class TopOne:
    """Baseline: only the single most plausible reading (what a top-1 'twin gate' would check)."""

    version = "top1-" + VERSION

    def hypotheses(self, ctx: Ctx) -> list[rg.Hypothesis]:
        hs = IntentHypotheses().hypotheses(ctx)
        return [min(hs, key=lambda h: h.score)]


def make_gate(top1: bool = False, alpha: float = 0.05, score: str = "mass") -> rg.Gate:
    """Primary configuration (frozen on the 2025 development runs): posterior-mass aggregation ("mass").
    "nearest" (worst case over an absolute plausible set) and "lr" (likelihood-ratio plausible set) are ablations."""
    return rg.Gate(EffectModel(), SPEC, TopOne() if top1 else IntentHypotheses(), alpha=alpha,
                   score_fn=SCORE_FNS[score], doer_id=None)
