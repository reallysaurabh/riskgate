"""OpenAI Agents SDK integration (``pip install openai-agents``; imported lazily).

Verified against openai-agents 0.23.1 by ``tests/contract/test_openai_agents.py``. Three building blocks:

1. ``guarded_function_tool(gate, fn, context=...)``: ``agents.function_tool`` over ``riskgate.guard(fn)``.
   The gate runs inside the tool function, so it holds however the tool is invoked. CERTIFY runs ``fn``;
   REJECT/PROBE (and ASK unless ``on_ask`` approves) return the feedback dict as the tool output (the SDK
   stringifies non-string outputs with ``str()``).

2. ``riskgate_tool_input_guardrail(gate, context=...)``: a native ``agents.ToolInputGuardrail`` for
   ``function_tool(..., tool_input_guardrails=[...])``. The Runner runs it before every invocation of the
   tool; non-certified calls are answered with ``ToolGuardrailFunctionOutput.reject_content`` (JSON feedback,
   the run continues and the model re-plans) and the full decision is kept in ``output_info``.

3. ``riskgate_needs_approval(gate, tool_name, context=...)``: a ``needs_approval`` callable that returns True
   exactly when the gate says ASK, mapping ASK onto the SDK's native human-in-the-loop flow (the run stops
   with ``result.interruptions``; ``state = result.to_state(); state.approve(item)``; ``Runner.run(agent,
   state)``). Combine it with the guardrail, which lets an ASK call through only once the SDK reports it
   approved (``ToolContext.is_tool_approved``). Do not combine it with ``guarded_function_tool``: the guard
   inside the function cannot see SDK approvals and would answer the approved call with ``needs_approval``.
"""
from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Callable, Collection
from typing import Any

from ..types import Action, Decision, Verdict
from .frameworks import _check, _safe
from .tool import _feedback, guard

__all__ = ["guarded_function_tool", "riskgate_needs_approval", "riskgate_tool_input_guardrail"]


def _takes_run_context(fn: Callable[..., Any]) -> bool:
    """True if ``fn``'s first parameter is the SDK's ``RunContextWrapper``/``ToolContext`` (as the SDK checks)."""
    from agents import RunContextWrapper
    from agents.tool_context import ToolContext

    params = list(inspect.signature(fn).parameters.values())
    if not params:
        return False
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable annotations: fall back to the raw annotation
        hints = {}
    ann = hints.get(params[0].name, params[0].annotation)
    origin = typing.get_origin(ann) or ann
    return origin is RunContextWrapper or origin is ToolContext


class _ContextError:
    """Marker returned by a failing context function (see ``_FailClosedGate``)."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc


class _FailClosedGate:
    """Gate proxy for ``guard``: a context that could not be built yields ASK without calling the gate."""

    def __init__(self, gate: Any) -> None:
        self.gate = gate

    def check(self, action: Action, context: Any, *, stage: int = 1) -> Decision:
        if isinstance(context, _ContextError):
            return Decision(Verdict.ASK, action, reason=f"fail-closed: could not build the gate context: "
                                                        f"{type(context.exc).__name__}: {context.exc}",
                            agent_reason="fail-closed: the gate could not evaluate this action")
        return self.gate.check(action, context, stage=stage)


def guarded_function_tool(gate: Any, fn: Callable[..., Any], *, context: Callable[[], Any] = lambda: None,
                          on_ask: Callable[[Decision], bool] | None = None, name: str | None = None,
                          **function_tool_kwargs: Any) -> Any:
    """Wrap ``fn`` with ``riskgate.guard`` and return an Agents SDK ``FunctionTool``.

    Args:
        gate: a calibrated ``riskgate.Gate``.
        fn: the tool function (sync or async). Its signature, type hints and docstring define the tool schema
            exactly as with ``agents.function_tool``. Functions whose first parameter is the SDK run context
            are not supported here (the guard would treat the context as an action argument); gate those with
            ``riskgate_tool_input_guardrail`` instead.
        context: zero-argument callable returning the gate context (state snapshot) at call time. If it
            raises, the call fails closed (ASK feedback, ``fn`` does not run).
        on_ask: optional synchronous approval hook for ASK (True runs ``fn``; False returns
            ``{"status": "declined", ...}``). Without it, ASK returns ``{"status": "needs_approval", ...}`` to the
            model instead of raising.
        name: tool name (default ``fn.__name__``); also the ``Action`` name seen by the gate.
        **function_tool_kwargs: forwarded to ``agents.function_tool`` (e.g. ``description_override``,
            ``tool_input_guardrails``, ``timeout``).
    """
    from agents import function_tool

    if _takes_run_context(fn):
        raise TypeError(f"{getattr(fn, '__name__', fn)!r} takes the run context as its first parameter; use "
                        "function_tool(fn, tool_input_guardrails=[riskgate_tool_input_guardrail(gate, ...)]) instead")
    tool_name = name or fn.__name__

    def safe_context() -> Any:
        try:
            return context()
        except Exception as exc:  # noqa: BLE001 - rule R3: fail closed
            return _ContextError(exc)

    guarded = guard(_FailClosedGate(gate), context=safe_context, name=tool_name, on_ask=on_ask,
                    raise_on_ask=False)(fn)
    return function_tool(guarded, name_override=tool_name, **function_tool_kwargs)


def _sdk_approved(tool_context: Any) -> bool:
    """True if the SDK reports that a human approved this exact tool call."""
    is_approved = getattr(tool_context, "is_tool_approved", None)
    if is_approved is None:
        return False
    try:
        return is_approved(tool_context.tool_name, tool_context.tool_call_id) is True
    except Exception:  # noqa: BLE001 - unknown approval state counts as not approved
        return False


def riskgate_tool_input_guardrail(gate: Any, *, context: Callable[..., Any] = lambda tool_context: None,
                                  name: str = "riskgate", halt_on: Collection[Verdict] = ()) -> Any:
    """Return an ``agents.ToolInputGuardrail`` that gates every invocation of the tools it is attached to.

    Args:
        context: ``context(tool_context)`` (or ``context()``) returning the gate context. The SDK
            ``ToolContext`` exposes ``.context`` (your run context object), ``.tool_name``, ``.tool_call_id``
            and ``.tool_arguments`` (JSON string).
        name: guardrail name (appears in ``result.tool_input_guardrail_results``).
        halt_on: verdicts that stop the whole run (``raise_exception`` -> the SDK raises
            ``ToolInputGuardrailTripwireTriggered``) instead of answering the model with feedback.

    CERTIFY -> ``allow``; ASK approved through the SDK (``needs_approval`` flow) -> ``allow``; otherwise
    ``reject_content(message=<JSON feedback>)``. ``output_info`` always holds the JSON-safe decision dict.
    """
    from agents import ToolGuardrailFunctionOutput, ToolInputGuardrail

    halt = frozenset(halt_on)

    def riskgate_guardrail(data: Any) -> Any:
        tc = data.context
        try:
            parsed = json.loads(tc.tool_arguments or "{}")
        except (TypeError, ValueError) as exc:     # malformed arguments: never forward (R3)
            fb = {"status": "rejected", "reason": f"tool arguments are not valid JSON: {exc}", "guarantee": "void"}
            return ToolGuardrailFunctionOutput.reject_content(message=json.dumps(fb), output_info=fb)
        args = parsed if isinstance(parsed, dict) else {"input": parsed}
        d = _check(gate, Action(tc.tool_name, args), context, tc)
        info = _safe(d.to_dict())
        if d.verdict is Verdict.CERTIFY:
            return ToolGuardrailFunctionOutput.allow(output_info=info)
        if d.verdict is Verdict.ASK and _sdk_approved(tc):
            return ToolGuardrailFunctionOutput.allow(output_info={**info, "approved_by": "human"})
        if d.verdict in halt:
            return ToolGuardrailFunctionOutput.raise_exception(output_info=info)
        return ToolGuardrailFunctionOutput.reject_content(message=json.dumps(_safe(_feedback(d))), output_info=info)

    return ToolInputGuardrail(guardrail_function=riskgate_guardrail, name=name)


def riskgate_needs_approval(gate: Any, tool_name: str, *,
                            context: Callable[..., Any] = lambda run_context: None) -> Callable[..., Any]:
    """Return an async ``needs_approval`` callable for ``agents.function_tool``: True iff the gate says ASK.

    The SDK calls it as ``needs_approval(run_context, arguments: dict, call_id)`` before running the tool; the
    tool name is not passed, so bind it here. ``context(run_context)`` (or ``context()``) builds the gate
    context. Pair with ``riskgate_tool_input_guardrail`` so REJECT/PROBE are answered with feedback and an
    approved ASK is let through.
    """
    async def needs_approval(run_context: Any, arguments: dict[str, Any], call_id: str) -> bool:
        d = _check(gate, Action(tool_name, dict(arguments or {})), context, run_context)
        return d.verdict is Verdict.ASK

    return needs_approval
