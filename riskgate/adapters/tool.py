"""Framework-agnostic adapter: guard any Python tool function with a gate.

Works with every agent framework, because tools are ultimately Python callables:

    @guard(gate, context=lambda: plant_snapshot())
    def set_setpoint(tag: str, value: float) -> str: ...

    # OpenAI Agents SDK:  function_tool(guard(gate, context=...)(set_setpoint))
    # LangChain/LangGraph: @tool on top of the guarded function
    # CrewAI / LlamaIndex / smolagents / Pydantic AI: register the guarded function as the tool

Verdict mapping (returned to the agent as the tool result unless configured otherwise):
  CERTIFY -> the tool runs
  REJECT  -> {"status": "rejected", "reason": ...}  (agent should re-plan)   or raise ActionRejected
  PROBE   -> {"status": "probe_requested", "probe": {...}}  (agent should call the probe tool first)
  ASK     -> on_ask(decision) decides (human approval hook); default raises ApprovalRequired
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from ..types import Action, Decision, Verdict


class ActionRejected(RuntimeError):
    def __init__(self, decision: Decision):
        super().__init__(decision.reason)
        self.decision = decision


class ApprovalRequired(RuntimeError):
    def __init__(self, decision: Decision):
        super().__init__(decision.reason)
        self.decision = decision


def _feedback(d: Decision) -> dict:
    out: dict[str, Any] = {"status": {"reject": "rejected", "ask": "needs_approval", "probe": "probe_requested"}.get(d.verdict.value,
                                                                                                    d.verdict.value),
           "reason": d.reason, "guarantee": d.status.value}
    if d.probe is not None:
        out["probe"] = d.probe.to_dict()
    return out


def guard(gate: Any, *, context: Callable[[], Any] = lambda: None, name: str | None = None,
          on_ask: Callable[[Decision], bool] | None = None, raise_on_reject: bool = False,
          stage: Callable[[], int] = lambda: 1, feedback: Callable[[dict], Any] | None = None) -> Callable:
    """Decorator factory. `context()` must return the gate's current context (state snapshot); if it raises,
    the call is treated as ASK (fail closed). `feedback` formats the dict returned to the agent for blocked
    calls (e.g. `feedback=json.dumps` for frameworks that stringify tool results)."""

    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        tool_name = name or fn.__name__

        def _decide(args, kwargs) -> Decision:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            action = Action(tool_name, dict(bound.arguments))
            try:
                ctx = context()
            except Exception as exc:  # noqa: BLE001 - no snapshot, no certification
                return Decision(Verdict.ASK, action, reason=f"fail-closed: context unavailable ({type(exc).__name__}: {exc})")
            return gate.check(action, ctx, stage=stage())

        def _route(d: Decision):
            if d.verdict is Verdict.CERTIFY:
                return None
            if d.verdict is Verdict.ASK and on_ask is not None and on_ask(d):
                return None
            if d.verdict is Verdict.ASK and on_ask is None:
                raise ApprovalRequired(d)
            if d.verdict is Verdict.REJECT and raise_on_reject:
                raise ActionRejected(d)
            fb = _feedback(d)
            return feedback(fb) if feedback is not None else fb

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                blocked = _route(_decide(args, kwargs))
                return blocked if blocked is not None else await fn(*args, **kwargs)
            awrapper.__riskgate__ = gate  # type: ignore[attr-defined]
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            blocked = _route(_decide(args, kwargs))
            return blocked if blocked is not None else fn(*args, **kwargs)
        wrapper.__riskgate__ = gate  # type: ignore[attr-defined]
        return wrapper

    return deco
