"""From agent traces to calibration episodes: the "send us your traces and what happened afterwards" onboarding path.

A customer provides two things:

  1. **traces**: what the agent did, one record per run (conversation, incident, task) with its tool calls in order,
     their arguments and, for read tools, their outputs (exported from OpenTelemetry, LangSmith, MCP logs, ...);
  2. **an outcome signal**: what happened afterwards, joined by trace id: reversals, chargebacks, incident tickets,
     QA audits, SLO breaches, or a validator run on the post-action state.

    traces = load_traces("traces.jsonl")
    outcomes = load_outcomes("outcomes.csv")                       # trace_id[, call | tool], unsafe[, source]
    kinds = infer_tool_kinds(traces)                               # read (probe candidates) vs write (decisions)
    points = decision_points(traces, outcomes, kinds)              # one per labelled write
    split = seal_split(points, ratios=(0.2, 0.5, 0.3), seed=7)     # group-disjoint fit / cal / eval, content digest
    worlds = DiscreteWorlds.fit([features(p) for p in split.fit], [world_of(p) for p in split.fit])  # fit split only
    gate = Gate(..., doer_id="support-agent:v12")
    gate.calibrate(split.episodes("cal", context=features), groups=split.groups("cal"))
    print(replay(gate, split.episodes("eval", context=features)))

Three rules keep the guarantee (they are why this module exists, rather than a loader):

* **Outcomes, not opinions.** The label must come from what happened (or a deterministic validator), never from an
  LLM judging the trace and never from a human approving the action: an approval authorises execution, it does not
  show the outcome was safe. ``load_outcomes`` records each label's ``source``; ``decision_points`` drops unlabelled
  writes (and counts them) instead of guessing.
* **Discover on one split, calibrate on another.** Anything learnt from the traces (worlds, their tables, feature
  weights, probe choices, rule limits) is fitted on ``split.fit`` only and frozen before calibration touches
  ``split.cal``; ``split.eval`` is for the report. Fitting on the calibration split voids exchangeability.
* **Count runs, not tool calls.** Calls inside one run are dependent: the split is by group and calibration takes
  ``groups=`` (the guarantee is then about the share of a new run's decisions that are certified and unsafe).

Traces label only the action the agent actually took, so each episode has one candidate and the gate runs in
**policy mode**: the bound covers this agent (set ``doer_id``), and a model or prompt change needs new traces.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .gate import Episode
from .types import Action, canonical

__all__ = ["DecisionPoint", "SealedSplit", "ToolCall", "Trace", "TraceContext", "decision_points", "infer_tool_kinds",
           "load_outcomes", "load_traces", "seal_split"]

READ_PREFIXES = ("get_", "list_", "describe_", "search_", "read_", "lookup_", "check_", "fetch_", "query_", "find_",
                 "show_", "view_", "inspect_", "count_", "explain_")


@dataclass(frozen=True)
class ToolCall:
    """One tool call in a trace. `output` is what the tool returned (read tools: a partial snapshot of the world)."""

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    output: Any = None

    @property
    def action(self) -> Action:
        return Action(self.name, dict(self.args))


@dataclass
class Trace:
    """One agent run. `group` is the exchangeable unit (default: the trace itself); several traces of the same task
    or customer case should share a group."""

    trace_id: str
    calls: list[ToolCall]
    group: str | None = None
    prompt: str = ""
    doer_id: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def unit(self) -> str:
        return self.group if self.group is not None else self.trace_id


def _calls(raw: Iterable[Mapping[str, Any]]) -> list[ToolCall]:
    out = []
    for c in raw:
        if "name" not in c:
            raise ValueError(f"tool call without a name: {dict(c)!r}")
        args = c.get("args", c.get("arguments", {})) or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        out.append(ToolCall(str(c["name"]), dict(args), c.get("output", c.get("result"))))
    return out


def load_traces(path: str | Path) -> list[Trace]:
    """Read traces from JSONL, one run per line::

        {"trace_id": "t-1", "group": "ticket-88", "doer_id": "agent:v12", "prompt": "...",
         "calls": [{"name": "get_order", "args": {"id": "A1"}, "output": {...}},
                   {"name": "issue_refund", "args": {"order": "A1", "amount": 49.0}}]}

    ``args`` may be a JSON string (as many tracing tools store it); ``arguments`` / ``result`` are accepted as
    aliases. Other exports (OpenTelemetry spans, LangSmith runs) convert to this shape in a few lines: one record per
    root span or run, its tool-call child spans in start-time order.
    """
    traces = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if "trace_id" not in r or "calls" not in r:
            raise ValueError(f"{path}:{n}: each line needs 'trace_id' and 'calls'")
        traces.append(Trace(str(r["trace_id"]), _calls(r["calls"]), group=None if r.get("group") is None else str(r["group"]),
                            prompt=str(r.get("prompt", "")), doer_id=r.get("doer_id"), meta=dict(r.get("meta", {}))))
    ids = [t.trace_id for t in traces]
    if len(set(ids)) != len(ids):
        raise ValueError("trace_id values must be unique")
    return traces


def _as_bool(v: Any, where: str) -> bool | None:
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "unsafe", "y"):
        return True
    if s in ("0", "false", "no", "safe", "n"):
        return False
    if s in ("", "null", "none", "unknown", "nan"):
        return None
    raise ValueError(f"{where}: cannot read {v!r} as an unsafe flag (use true/false, 1/0, or empty for unknown)")


_FORBIDDEN_SOURCES = ("llm", "judge", "approval", "approved", "human_approval")


def load_outcomes(path: str | Path, *, allow_sources: Collection[str] | None = None) -> list[dict]:
    """Read outcome labels from CSV or JSONL. Columns: ``trace_id``, ``unsafe`` (true/false/empty = unknown), and
    optionally ``call`` (0-based index of the tool call in the trace), ``tool`` (applies to every call of that tool in
    the trace), ``source`` (``chargeback``, ``reversal``, ``incident``, ``qa_audit``, ``slo``, ``validator``, ...).
    Without ``call`` and ``tool``, the label applies to every write in the trace.

    Labels whose ``source`` names an LLM judge or an approval are refused: they would make the bound about agreement
    with a judge, or about rubber stamps, instead of about outcomes. Pass ``allow_sources`` to restrict sources
    explicitly.
    """
    p = Path(path)
    if p.suffix == ".csv":
        with p.open(newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    else:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    out = []
    for n, r in enumerate(rows, 1):
        where = f"{path}:{n}"
        if "trace_id" not in r or "unsafe" not in r:
            raise ValueError(f"{where}: each outcome needs 'trace_id' and 'unsafe'")
        src = str(r.get("source") or "").strip()
        low = src.lower()
        if any(k in low for k in _FORBIDDEN_SOURCES):
            raise ValueError(f"{where}: label source {src!r} is a judgement, not an outcome (LLM judges and approvals "
                             "cannot calibrate the gate; use what happened afterwards or a deterministic validator)")
        if allow_sources is not None and src not in allow_sources:
            raise ValueError(f"{where}: label source {src!r} is not in allow_sources")
        call = r.get("call")
        out.append({"trace_id": str(r["trace_id"]), "unsafe": _as_bool(r["unsafe"], where), "source": src or None,
                    "call": None if call in (None, "") else int(call), "tool": (r.get("tool") or None)})
    return out


def infer_tool_kinds(traces: Sequence[Trace], *, read: Collection[str] = (), write: Collection[str] = (),
                     read_prefixes: Sequence[str] = READ_PREFIXES) -> dict[str, str]:
    """Classify every tool seen in the traces as ``"read"`` (side-effect free: probe candidates and evidence) or
    ``"write"`` (a consequential decision the gate checks). Explicit ``read`` / ``write`` lists win; otherwise a
    name starting with a read prefix (``get_``, ``list_``, ``describe_``, ...) is read and everything else is a
    write (fail closed: an unknown tool is treated as consequential). Review the result: names can mislead."""
    both = set(read) & set(write)
    if both:
        raise ValueError(f"tools listed as both read and write: {sorted(both)}")
    kinds: dict[str, str] = {}
    for t in traces:
        for c in t.calls:
            if c.name in kinds:
                continue
            if c.name in write:
                kinds[c.name] = "write"
            elif c.name in read or c.name.lower().startswith(tuple(read_prefixes)):
                kinds[c.name] = "read"
            else:
                kinds[c.name] = "write"
    return dict(sorted(kinds.items()))


@dataclass
class TraceContext:
    """What the agent had seen when it proposed a write: the prompt and every earlier call with its output.
    Nothing after the write is included (no look-ahead)."""

    trace_id: str
    prompt: str
    history: list[ToolCall]
    kinds: Mapping[str, str]

    def called(self, tool: str) -> bool:
        """Did the agent call `tool` before this write? (missing-evidence features: refund without a dispute check)"""
        return any(c.name == tool for c in self.history)

    def last_output(self, tool: str) -> Any:
        """Output of the most recent earlier call of `tool`, or None."""
        for c in reversed(self.history):
            if c.name == tool:
                return c.output
        return None

    @property
    def reads(self) -> list[ToolCall]:
        return [c for c in self.history if self.kinds.get(c.name) == "read"]

    def to_agent_context(self, trusted_values: Collection[str] = ()) -> Any:
        """The same history as a ``riskgate.provenance.AgentContext`` (read outputs as untrusted tool outputs)."""
        from .provenance import AgentContext, ToolOutput
        outs = [ToolOutput(f"{c.name}#{i}", c.output if isinstance(c.output, str) else canonical(c.output))
                for i, c in enumerate(self.history) if self.kinds.get(c.name) == "read" and c.output is not None]
        return AgentContext(self.prompt, outs, trusted_values=trusted_values)


@dataclass
class DecisionPoint:
    """One labelled write: the action the agent took, what it had seen, and what happened afterwards."""

    trace_id: str
    group: str
    index: int                        # position of the call in the trace
    action: Action
    context: TraceContext
    unsafe: bool
    source: str | None = None
    doer_id: str | None = None

    def episode(self, context: Callable[[DecisionPoint], Any] | None = None) -> Episode:
        ctx = context(self) if context is not None else self.context
        return Episode(ctx, [self.action], {self.action.key: self.unsafe},
                       meta={"trace_id": self.trace_id, "group": self.group, "call": self.index,
                             "label_source": self.source, "doer_id": self.doer_id})


def decision_points(traces: Sequence[Trace], outcomes: Sequence[Mapping[str, Any]],
                    kinds: Mapping[str, str] | None = None) -> list[DecisionPoint]:
    """One DecisionPoint per write that has an outcome label. Writes without a label, or with an unknown (null) label,
    are left out, and so are writes with conflicting labels; the counts are in ``decision_points.last_report``.
    Leaving them out is the honest choice only if labels are missing at random: if unlabelled writes are the risky
    ones (nobody checked), get labels for them instead."""
    kinds = dict(kinds) if kinds is not None else infer_tool_kinds(traces)
    by_trace: dict[str, list[Mapping[str, Any]]] = {}
    for o in outcomes:
        by_trace.setdefault(str(o["trace_id"]), []).append(o)
    known = {t.trace_id for t in traces}
    points: list[DecisionPoint] = []
    rep = {"traces": len(traces), "writes": 0, "labelled": 0, "unlabelled": 0, "unknown_label": 0, "conflicting": 0,
           "outcomes_for_unknown_traces": sum(1 for tid in by_trace if tid not in known)}
    for t in traces:
        labels = by_trace.get(t.trace_id, [])
        for i, c in enumerate(t.calls):
            if kinds.get(c.name, "write") != "write":
                continue
            rep["writes"] += 1
            hits = [o for o in labels if (o.get("call") is not None and o["call"] == i) or
                    (o.get("call") is None and o.get("tool") is not None and o["tool"] == c.name)]
            if not hits:
                hits = [o for o in labels if o.get("call") is None and o.get("tool") is None]
            vals = {o["unsafe"] for o in hits}
            if not hits:
                rep["unlabelled"] += 1
                continue
            if len(vals) > 1:
                rep["conflicting"] += 1
                continue
            v = next(iter(vals))
            if v is None:
                rep["unknown_label"] += 1
                continue
            rep["labelled"] += 1
            src = next((o.get("source") for o in hits if o.get("source")), None)
            points.append(DecisionPoint(t.trace_id, t.unit, i, c.action,
                                        TraceContext(t.trace_id, t.prompt, list(t.calls[:i]), kinds), bool(v), src,
                                        t.doer_id))
    decision_points.last_report = rep          # type: ignore[attr-defined]
    return points


decision_points.last_report = {}               # type: ignore[attr-defined]


@dataclass
class SealedSplit:
    """Group-disjoint fit / cal / eval split of decision points, with a content digest.

    fit   discover and fit everything learnt from data (worlds, tables, weights, probe choices, rule limits)
    cal   calibrate the frozen gate once (re-calibrating on the same split and keeping the best is p-hacking)
    eval  report what the gate would have done
    """

    fit: list[DecisionPoint]
    cal: list[DecisionPoint]
    eval: list[DecisionPoint]
    seed: int
    ratios: tuple[float, float, float]
    digest: str

    def part(self, name: str) -> list[DecisionPoint]:
        if name not in ("fit", "cal", "eval"):
            raise ValueError("part must be 'fit', 'cal' or 'eval'")
        return list(getattr(self, name))

    def episodes(self, name: str, context: Callable[[DecisionPoint], Any] | None = None) -> list[Episode]:
        return [p.episode(context) for p in self.part(name)]

    def groups(self, name: str) -> list[str]:
        """Group labels aligned with ``episodes(name)``, for ``Gate.calibrate(groups=...)``."""
        return [p.group for p in self.part(name)]

    def summary(self) -> dict:
        def s(ps: list[DecisionPoint]) -> dict:
            return {"decisions": len(ps), "groups": len({p.group for p in ps}),
                    "unsafe_rate": float(np.mean([p.unsafe for p in ps])) if ps else math.nan}
        return {"digest": self.digest, "seed": self.seed, "ratios": list(self.ratios),
                "fit": s(self.fit), "cal": s(self.cal), "eval": s(self.eval)}


def seal_split(points: Sequence[DecisionPoint], ratios: tuple[float, float, float] = (0.2, 0.5, 0.3),
               seed: int = 0, group: Callable[[DecisionPoint], Hashable] | None = None) -> SealedSplit:
    """Split decision points by group into fit / cal / eval (default 20 / 50 / 30 % of the groups).

    The digest covers the seed, the ratios and every point's (group, trace, call, action, label), so a report can
    name exactly which data the threshold was calibrated on and anyone can check it was not re-drawn.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must be three non-negative shares summing to 1, got {ratios}")
    if ratios[1] <= 0:
        raise ValueError("the calibration share must be positive")
    key = group or (lambda p: p.group)
    units = sorted({str(key(p)) for p in points})
    perm = np.random.default_rng(seed).permutation(len(units))
    n_fit = int(round(ratios[0] * len(units)))
    n_cal = int(round(ratios[1] * len(units)))
    fit_u = {units[i] for i in perm[:n_fit]}
    cal_u = {units[i] for i in perm[n_fit:n_fit + n_cal]}
    parts: dict[str, list[DecisionPoint]] = {"fit": [], "cal": [], "eval": []}
    for p in points:
        u = str(key(p))
        parts["fit" if u in fit_u else "cal" if u in cal_u else "eval"].append(p)
    content = sorted([str(key(p)), p.trace_id, p.index, p.action.key, p.unsafe] for p in points)
    h = hashlib.sha256(canonical({"seed": seed, "ratios": list(ratios), "points": content}).encode()).hexdigest()[:16]
    return SealedSplit(parts["fit"], parts["cal"], parts["eval"], seed, (float(ratios[0]), float(ratios[1]),
                       float(ratios[2])), h)
