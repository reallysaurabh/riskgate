"""Ready-made hard rules: deterministic, action-level policy checked before any prediction.

    from riskgate.rules import needs_person, never, amount_above, irreversible

    hard_rules = [
        needs_person("block_account", "label_fraud", reason="labels on people are decided by a person"),
        amount_above(10_000, tools=("issue_refund",), arg="amount"),     # ASK above the supervisor limit
        never("drop_database"),
    ]

Each helper returns a ``HardRule`` with a pinned ``version`` built from its parameters, so changing a limit or a
tool list voids old calibrations, and editing nothing does not.
"""
from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

from .spec import HardRule
from .types import Action, Verdict, digest

__all__ = ["amount_above", "irreversible", "needs_person", "never"]


def _names(tools: Collection[str] | str) -> frozenset[str]:
    return frozenset([tools] if isinstance(tools, str) else tools)


def needs_person(*tools: str, reason: str = "this action is decided by a person", name: str | None = None) -> HardRule:
    """ASK for every call to one of `tools`, whatever the calibration says (e.g. blocking an account)."""
    if not tools:
        raise ValueError("needs_person needs at least one tool name")
    names = _names(tools)

    def fires(action: Action, context: Any) -> bool:
        return action.name in names

    return HardRule(name or f"person decides: {', '.join(sorted(names))}", fires, Verdict.ASK, reason,
                    version="needs_person:" + digest(sorted(names)))


def never(*tools: str, reason: str = "this action is never run by the agent", name: str | None = None) -> HardRule:
    """REJECT every call to one of `tools`."""
    if not tools:
        raise ValueError("never needs at least one tool name")
    names = _names(tools)

    def fires(action: Action, context: Any) -> bool:
        return action.name in names

    return HardRule(name or f"never: {', '.join(sorted(names))}", fires, Verdict.REJECT, reason,
                    version="never:" + digest(sorted(names)))


def amount_above(limit: float, *, arg: str = "amount", tools: Collection[str] | None = None,
                 verdict: Verdict = Verdict.ASK, name: str | None = None, reason: str | None = None) -> HardRule:
    """`verdict` (default ASK) when ``action.args[arg] > limit``, for `tools` (default: every tool with that
    argument). A value that is not a number fires the rule (fail closed)."""
    limit = float(limit)
    if not math.isfinite(limit):
        raise ValueError(f"limit must be finite, got {limit}")
    names = _names(tools) if tools is not None else None

    def fires(action: Action, context: Any) -> bool:
        if names is not None and action.name not in names:
            return False
        if arg not in action.args:
            return False
        try:
            value = float(action.args[arg])
        except (TypeError, ValueError):
            return True
        return not math.isfinite(value) or value > limit

    return HardRule(name or f"{arg} above {limit:g}", fires, Verdict(verdict), reason or f"{arg} above {limit:g}",
                    version="amount_above:" + digest([limit, arg, sorted(names) if names is not None else None,
                                                      Verdict(verdict).value]))


def irreversible(*tools: str, name: str | None = None) -> HardRule:
    """ASK for every call to one of `tools` because it cannot be undone (`needs_person` with a fixed message).
    Often better: have the agent rank a reversible alternative next (deallocate after delete) and let the gate
    certify whichever is safe in every plausible world; keep this rule for actions with no way back."""
    return needs_person(*tools, reason="irreversible: a person approves", name=name or
                        f"irreversible: {', '.join(sorted(_names(tools)))}")
