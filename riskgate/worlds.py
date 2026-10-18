"""Discrete worlds: a hypothesis provider and a consequence model for "which of a few situations is this?" domains.

Many agent decisions hinge on one hidden fact with a handful of values (the use cases in docs/use_cases.md):

  * "money deducted, order not placed"  paid but confirmation late | failed, bank auto-reverses | pending | paid twice
  * "item missing"                      never packed | out of stock and already refunded | in a second bag
  * "I don't recognise this payment"    authorisation hold | subscription | add-on card | fraud
  * an idle VM                          dormant | scheduled | live

For each world one or more actions are right and the others are wrong. ``DiscreteWorlds`` scores the worlds by
negative log posterior from the evidence in the context (a naive-Bayes model whose tables you write down or fit);
``OutcomeTable`` says which action is safe in which world and what each read-only lookup would reveal, so the gate
can choose the lookup that settles the question (PROBE) before anything irreversible runs.

    worlds = DiscreteWorlds(
        prior={"paid_late": 0.35, "failed": 0.40, "pending": 0.13, "paid_twice": 0.12},
        evidence={"app_status": {"paid_late": {"pending": 0.65, "no order": 0.35}, ...},
                  "two_debit_sms": {"paid_late": 0.02, "failed": 0.02, "pending": 0.02, "paid_twice": 0.75},
                  "switch_status": {"paid_late": {"SUCCESS": 0.92, "PENDING": 0.08}, ...}})   # filled in by the probe
    table = OutcomeTable(
        safe={"confirm_order": ["paid_late"], "reply_auto_reversal": ["failed"], "issue_refund": ["paid_twice"]},
        reveals={"check_txn_status": {"paid_late": "SUCCESS", "failed": "FAILED", "pending": "PENDING",
                                      "paid_twice": "DUPLICATE"}})
    gate = Gate(table, table.spec, worlds, probes=table.probes(), noop=table.noop,
                min_separation=table.separation / 2, doer_id=...)

Evidence fields that are missing from the context (or None) are skipped: they have not been observed yet. That is
how one provider serves both stages of a probe: after the lookup, put its reading into the context and re-check.

The tables are part of the fingerprint (``version``), so editing a probability or a safe-set voids old calibrations.
Nothing here changes the guarantee: it comes from calibration on labelled episodes, whatever the tables say. Wrong
tables cost efficiency (more ASK), not validity.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from .gate import CallableModel
from .spec import Predicate, Spec
from .types import Action, Hypothesis, callable_digest, digest

__all__ = ["BIG", "DiscreteWorlds", "OutcomeTable", "nearest_violation", "read_field", "unsafe_mass"]

_MISSING = object()


def read_field(context: Any, name: str) -> Any:
    """``context[name]`` for mappings, ``context.name`` otherwise; None when absent (= not observed)."""
    if isinstance(context, Mapping):
        return context.get(name)
    return getattr(context, name, None)


def _check_prob(p: Any, where: str) -> float:
    p = float(p)
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"{where}: probabilities must be in [0, 1], got {p}")
    return p


class DiscreteWorlds:
    """Hypothesis provider over a finite set of worlds; score = -log P(world | evidence) (lower = fits better).

    prior     world -> prior probability (normalised; zero is allowed and makes a world implausible)
    evidence  field -> {world -> {value -> P(value | world)}}, or field -> {world -> P(field is true | world)} for a
              boolean field. Values not listed for a world have probability 0 (clipped to ``floor``). A world absent
              from a field's table gets no information from that field (likelihood 1).
    read      how to read a field from the context (default: mapping key or attribute; None = not observed)
    floor     probabilities are clipped to at least this, so one surprising value cannot make a world impossible
    max_score scores are capped here (the gate needs finite scores to rank worlds)
    name      prefix of ``version``; the rest is a digest of the tables
    """

    def __init__(self, prior: Mapping[str, float], evidence: Mapping[str, Mapping[str, Any]] | None = None, *,
                 read: Callable[[Any, str], Any] = read_field, floor: float = 1e-6, max_score: float = 60.0,
                 name: str = "worlds"):
        if not prior:
            raise ValueError("prior needs at least one world")
        p = {str(w): _check_prob(v, f"prior[{w!r}]") for w, v in prior.items()}
        total = sum(p.values())
        if total <= 0.0:
            raise ValueError("prior probabilities must not all be zero")
        if not (0.0 < float(floor) < 1.0):
            raise ValueError(f"floor must be in (0, 1), got {floor}")
        if not (math.isfinite(max_score) and max_score > 0.0):
            raise ValueError(f"max_score must be positive and finite, got {max_score}")
        self.worlds: tuple[str, ...] = tuple(p)
        self.prior = {w: v / total for w, v in p.items()}
        self.evidence: dict[str, dict[str, Any]] = {}
        for field_name, table in (evidence or {}).items():
            unknown = set(table) - set(self.worlds)
            if unknown:
                raise ValueError(f"evidence[{field_name!r}] names unknown worlds {sorted(unknown)}")
            clean: dict[str, Any] = {}
            for w, row in table.items():
                if isinstance(row, Mapping):
                    clean[w] = {v: _check_prob(q, f"evidence[{field_name!r}][{w!r}][{v!r}]") for v, q in row.items()}
                else:
                    clean[w] = _check_prob(row, f"evidence[{field_name!r}][{w!r}]")
            self.evidence[str(field_name)] = clean
        self.read, self.floor, self.max_score = read, float(floor), float(max_score)
        self.version = f"{name}-" + digest({"prior": self.prior, "evidence": {f: {w: (r if not isinstance(r, Mapping) else
                                            {str(k): q for k, q in r.items()}) for w, r in t.items()} for f, t in
                                            self.evidence.items()}, "floor": self.floor, "max": self.max_score,
                                            "read": callable_digest(read) if read is not read_field else None}, 12)

    @classmethod
    def fit(cls, contexts: Sequence[Mapping[str, Any]], worlds: Sequence[str], *, fields: Sequence[str] | None = None,
            smoothing: float = 1.0, **kw: Any) -> DiscreteWorlds:
        """Learn the prior and the evidence tables from labelled examples (counts with additive smoothing).

        contexts  one mapping of observed (categorical or boolean) fields per example; None = not observed
        worlds    the world each example turned out to be in (from the outcome signal, not from a judge)
        fields    which fields to use (default: every field seen)

        Fit on data that is disjoint from calibration (e.g. ``riskgate.traces.SealedSplit.fit``) and freeze it: the
        fitted tables are part of the fingerprint, and fitting on the calibration episodes breaks exchangeability.
        """
        if len(contexts) != len(worlds) or not contexts:
            raise ValueError("fit needs one world label per context, and at least one example")
        if not (math.isfinite(smoothing) and smoothing > 0.0):
            raise ValueError(f"smoothing must be positive and finite, got {smoothing}")
        names = sorted(dict.fromkeys(str(w) for w in worlds))
        counts = {w: sum(1 for x in worlds if str(x) == w) for w in names}
        prior = {w: (counts[w] + smoothing) / (len(worlds) + smoothing * len(names)) for w in names}
        use = list(fields) if fields is not None else sorted({k for c in contexts for k in c})
        evidence: dict[str, dict[str, Any]] = {}
        for f in use:
            seen = [(c.get(f), str(w)) for c, w in zip(contexts, worlds) if c.get(f) is not None]
            if not seen:
                continue
            values = sorted({v for v, _ in seen}, key=repr)
            if all(isinstance(v, bool) for v in values):
                evidence[f] = {w: (sum(1 for v, x in seen if x == w and v) + smoothing) /
                               (sum(1 for _, x in seen if x == w) + 2 * smoothing) for w in names}
                continue
            table: dict[str, dict[Any, float]] = {}
            for w in names:
                n_w = sum(1 for _, x in seen if x == w)
                table[w] = {v: (sum(1 for vv, x in seen if x == w and vv == v) + smoothing) /
                            (n_w + smoothing * len(values)) for v in values}
            evidence[f] = table
        kw.setdefault("name", "fitted-worlds")
        return cls(prior, evidence, **kw)

    def _lp(self, p: float) -> float:
        return math.log(max(p, self.floor))

    def log_posterior(self, context: Any) -> dict[str, float]:
        """log P(world | observed evidence) for every world (fields that are absent or None are skipped)."""
        ll = {w: (math.log(self.prior[w]) if self.prior[w] > 0 else -math.inf) for w in self.worlds}
        for field_name, table in self.evidence.items():
            value = self.read(context, field_name)
            if value is None:
                continue
            for w, row in table.items():
                if isinstance(row, Mapping):
                    ll[w] += self._lp(row.get(value, 0.0))
                else:                                         # boolean field: row = P(true | world)
                    ll[w] += self._lp(row if bool(value) else 1.0 - row)
        m = max(ll.values())
        if m == -math.inf:
            return dict(ll)
        z = m + math.log(sum(math.exp(v - m) for v in ll.values()))
        return {w: v - z for w, v in ll.items()}

    def posterior(self, context: Any) -> dict[str, float]:
        return {w: math.exp(v) for w, v in self.log_posterior(context).items()}

    def hypotheses(self, context: Any) -> list[Hypothesis]:
        lp = self.log_posterior(context)
        return [Hypothesis(w, family=w, params={"world": w}, score=min(-lp[w], self.max_score)) for w in self.worlds]


def _world(h: Hypothesis) -> str:
    return str(h.params.get("world", h.fam))


class OutcomeTable(CallableModel):
    """Consequence model for discrete worlds: is this action safe if this is the world?

    safe        action name -> worlds in which the action is safe (``"*"`` = every world); or a callable
                ``safe(action, world, context) -> bool`` for decisions that depend on arguments or context.
                Actions it does not name are unsafe in every world (fail closed), except read-only actions.
    reveals     probe name -> {world -> reading}: what each read-only lookup returns in each world. Worlds that give
                different readings are told apart by the probe; equal readings (e.g. "PENDING" in two worlds) are
                not. A read-only action without an entry resolves nothing.
    separation  distance between different readings in the probe observation (use ``min_separation=separation/2``)
    version     model version for the fingerprint (default: a digest of the tables)

    Outcomes are ``{"safe": bool, "world": str, "observation": [...]}``; ``spec`` reads ``safe``.

    Pass ``noop=table.noop`` to the gate as well. A lookup is admissible as a probe only if it is certifiable at the
    stage-1 threshold or no worse than doing nothing; when no write can be certified before the lookup (the usual
    case: the worlds disagree), the second condition is what lets the gate look first instead of asking.
    """

    noop = Action("do_nothing", read_only=True)        # the "do nothing" reference action: safe in every world

    def __init__(self, safe: Mapping[str, Collection[str] | str] | Callable[[Action, str, Any], bool], *,
                 reveals: Mapping[str, Mapping[str, Any]] | None = None, separation: float = 10.0,
                 version: str | None = None):
        if not (math.isfinite(separation) and separation > 0.0):
            raise ValueError(f"separation must be positive and finite, got {separation}")
        if callable(safe):
            self._safe_fn: Callable[[Action, str, Any], bool] | None = safe
            self.safe: dict[str, frozenset[str] | str] = {}
            safe_desc: Any = f"callable:{callable_digest(safe)}"
        else:
            self._safe_fn = None
            self.safe = {str(a): ("*" if ws == "*" else frozenset([ws] if isinstance(ws, str) else ws))
                         for a, ws in safe.items()}
            safe_desc = {a: (ws if ws == "*" else sorted(ws)) for a, ws in self.safe.items()}
        self.reveals = {str(p): dict(r) for p, r in (reveals or {}).items()}
        self._labels = {p: sorted({repr(v) for v in r.values()}) for p, r in self.reveals.items()}
        self.separation = float(separation)
        auto = "outcome-table-" + digest({"safe": safe_desc, "reveals": {p: {w: repr(v) for w, v in r.items()}
                                                                        for p, r in self.reveals.items()},
                                          "sep": self.separation}, 12)
        super().__init__(self._outcome, version=version or auto)
        self.spec = Spec(Predicate("safe in this world", lambda o: bool(o["safe"])))

    def probes(self, **args: Mapping[str, Any]) -> list[Action]:
        """The lookups as read-only Actions, for ``Gate(probes=...)``; pass ``name={...}`` to give one arguments."""
        return [Action(p, dict(args.get(p, {})), read_only=True) for p in self.reveals]

    def is_safe(self, action: Action, world: str, context: Any = None) -> bool:
        if action.read_only:
            return True
        if self._safe_fn is not None:
            return bool(self._safe_fn(action, world, context))
        ws = self.safe.get(action.name)
        return ws is not None and (ws == "*" or world in ws)

    def _outcome(self, action: Action, h: Hypothesis, context: Any) -> dict:
        world = _world(h)
        obs = [0.0]
        if action.read_only and action.name in self.reveals:
            labels = self._labels[action.name]
            reading = self.reveals[action.name].get(world, _MISSING)
            # one-hot reading scaled by `separation`: different readings are `separation` apart, equal ones 0;
            # a world the probe has no reading for is NaN (cannot be told apart from anything)
            obs = ([float("nan")] * len(labels) if reading is _MISSING else
                   [self.separation if lab == repr(reading) else 0.0 for lab in labels])
        return {"safe": self.is_safe(action, world, context), "world": world, "observation": obs}


# ------------------------------------------------------------------------------------------------ score functions
# Aggregations for Gate(score_fn=...). The default (minimum margin) treats every plausible world alike; these use the
# worlds' scores instead, which helps when hypotheses carry graded plausibility (provenance, intent readings).
BIG = 1e6       # score of an action that is safe in every plausible world (finite: lambda = +inf never certifies)


def nearest_violation(margins: Any, plausible: Any) -> float:
    """Score of the most plausible world in which the action is unsafe (BIG if none): an action is certified only
    if every world where it goes wrong is implausible enough."""
    bad = [h.score for h, m in zip(plausible, margins) if m < 0]
    return float(min(bad)) if bad else BIG


nearest_violation.riskgate_version = "nearest-violation-v1"   # type: ignore[attr-defined]


def unsafe_mass(margins: Any, plausible: Any) -> float:
    """-log of the total posterior mass (exp(-score)) of the plausible worlds in which the action is unsafe (BIG if
    none). With DiscreteWorlds, exp(-score) is the posterior, so this is -log P(unsafe | evidence)."""
    bad = [h.score for h, m in zip(plausible, margins) if m < 0]
    if not bad:
        return BIG
    lo = min(bad)
    return float(lo - math.log(sum(math.exp(lo - s) for s in bad)))


unsafe_mass.riskgate_version = "unsafe-mass-v1"   # type: ignore[attr-defined]
