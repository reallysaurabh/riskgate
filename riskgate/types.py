"""Core value types shared across riskgate."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def _json_default(o: Any) -> Any:
    """JSON fallback that never embeds memory addresses (they would make fingerprints differ across runs)."""
    if callable(o) and not isinstance(o, type):
        return f"callable:{callable_digest(o)}"
    s = str(o)
    if " at 0x" not in s:
        return s
    describe = getattr(o, "describe", None)
    if callable(describe):
        return describe()
    if hasattr(o, "__dict__"):
        return {"__type__": f"{type(o).__module__}.{type(o).__qualname__}",
                **{k: v for k, v in vars(o).items() if not k.startswith("_")}}
    return f"{type(o).__module__}.{type(o).__qualname__}"


def canonical(obj: Any) -> str:
    """Deterministic JSON used for keys and fingerprints."""
    return json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))


def digest(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:n]


# ----------------------------------------------------------------------------------------------
# Content hashes of user callables (so that editing a rule's or constraint's logic voids calibration)
# ----------------------------------------------------------------------------------------------
_CALLABLE_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _simple(x: Any, depth: int = 0) -> Any:
    """A deterministic, JSON-able summary of a closure / default value."""
    if depth > 4:
        return "..."
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple, set, frozenset)):
        items = [_simple(v, depth + 1) for v in x]
        return sorted(items, key=repr) if isinstance(x, (set, frozenset)) else items
    if isinstance(x, Mapping):
        return {str(k): _simple(v, depth + 1) for k, v in x.items()}
    if callable(x) and not isinstance(x, type):
        return f"callable:{callable_digest(x, depth + 1)}"
    s = str(x)
    return s if " at 0x" not in s else f"{type(x).__module__}.{type(x).__qualname__}"


def _code_sig(code: Any) -> list:
    consts = [_code_sig(c) if inspect.iscode(c) else repr(c) for c in code.co_consts]
    return [code.co_code.hex(), consts, list(code.co_names)]


def callable_digest(fn: Any, _depth: int = 0) -> str:
    """Best-effort content hash of a Python callable, used in gate fingerprints.

    Uses an explicit ``fn.riskgate_version`` attribute if present; otherwise the source code (falling back to
    bytecode), default values and simple closure values. Conservative by design: editing the function (even a
    comment on the same line) changes the hash and therefore voids calibrations that depended on it.
    Computed once per function object (closure values mutated later in the same process are not re-read).
    """
    version = getattr(fn, "riskgate_version", None)
    if version is not None:
        return f"v:{version}"
    if isinstance(fn, functools.partial):
        return digest(["partial", callable_digest(fn.func, _depth + 1), _simple(fn.args, _depth + 1),
                       _simple(fn.keywords, _depth + 1)])
    plain = inspect.isfunction(fn)
    if plain:
        try:
            return str(_CALLABLE_MEMO[fn])
        except (KeyError, TypeError):
            pass
    extra: Any = None
    target = fn
    if not (plain or inspect.ismethod(fn) or inspect.isbuiltin(fn)):
        target = getattr(type(fn), "__call__", fn)          # noqa: B004 - callable instance: hash its class's __call__
        extra = _simple(vars(fn), _depth + 1) if hasattr(fn, "__dict__") else None
    f = getattr(target, "__func__", target)
    parts: list[Any] = [getattr(f, "__qualname__", type(f).__qualname__), extra]
    try:
        parts.append(inspect.getsource(f))
    except (OSError, TypeError):
        code = getattr(f, "__code__", None)
        parts.append(_code_sig(code) if code is not None else type(f).__qualname__)
    if _depth < 4:
        parts.append(_simple(getattr(f, "__defaults__", None), _depth + 1))
        parts.append(_simple(getattr(f, "__kwdefaults__", None), _depth + 1))
        cells = []
        for c in getattr(f, "__closure__", None) or ():
            try:
                cells.append(_simple(c.cell_contents, _depth + 1))
            except ValueError:                               # empty cell
                cells.append(None)
        parts.append(cells)
    out = digest(parts)
    if plain:
        try:
            _CALLABLE_MEMO[fn] = out
        except TypeError:
            pass
    return out


class Verdict(str, Enum):
    """The gate's answer for a proposed action."""

    CERTIFY = "certify"   # execute: the calibrated risk bound covers this decision
    REJECT = "reject"     # do not execute: unsafe under every plausible hypothesis (agent should re-plan)
    ASK = "ask"           # escalate to a human / fallback: the gate cannot certify
    PROBE = "probe"       # first execute a read-only, information-gathering action, then re-check


class RiskGateWarning(UserWarning):
    """A configuration that is valid but weakens what the guarantee covers (e.g. policy mode without a doer_id).

    Silence it deliberately with ``warnings.filterwarnings("ignore", category=riskgate.RiskGateWarning)``.
    """


class GuaranteeStatus(str, Enum):
    VALID = "valid"        # calibrated, component fingerprints match, no drift detected
    DEGRADED = "degraded"  # drift detected: online adaptation active, the formal bound is not guaranteed
    VOID = "void"          # not calibrated, or a component changed since calibration


@dataclass(frozen=True)
class Action:
    """A proposed tool call. `read_only` marks information-gathering actions usable as probes."""

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    read_only: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}({canonical(dict(self.args))})" if self.args else self.name

    def to_dict(self) -> dict:
        return {"name": self.name, "args": dict(self.args), "read_only": self.read_only}


@dataclass(frozen=True)
class Hypothesis:
    """One explanation of the current (partially observed) world state.

    `score` is a nonconformity score: lower means the hypothesis explains the evidence better.
    `family` groups hypotheses (e.g. all magnitudes of one fault type) for calibration labels.
    """

    name: str
    family: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def fam(self) -> str:
        return self.family or self.name


@dataclass
class Decision:
    verdict: Verdict
    action: Action | None
    score: float = float("nan")              # robust margin of `action` (min over plausible hypotheses)
    threshold: float = float("nan")          # lambda used
    margins: dict[str, float] = field(default_factory=dict)    # hypothesis -> margin (for `action`)
    binding: dict[str, str] = field(default_factory=dict)      # hypothesis -> binding constraint
    plausible: list[str] = field(default_factory=list)
    probe: Action | None = None
    reason: str = ""
    status: GuaranteeStatus = GuaranteeStatus.VOID
    calibration_id: str | None = None
    considered: list[dict] = field(default_factory=list)       # every candidate looked at, in rank order
    stage: int = 1                                             # 1 = first decision, 2 = after a probe
    latency_ms: float = 0.0
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    top_score: float = float("nan")          # best robust score among viable candidates (drift-monitor statistic)

    @property
    def approved(self) -> bool:
        return self.verdict is Verdict.CERTIFY

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["status"] = self.status.value
        d["action"] = self.action.to_dict() if self.action else None
        d["probe"] = self.probe.to_dict() if self.probe else None
        return d
