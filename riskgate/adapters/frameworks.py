"""Adapters for Google ADK and LangGraph (duck-typed; no framework import at module load).

Verified against google-adk 2.11.0, langgraph 1.2.12 (langgraph-prebuilt 1.1.0) and langchain-core 1.6.6
by the contract tests in ``tests/contract`` (see ``docs/integrations.md``). Framework imports happen lazily
inside the functions that need them, so importing this module never imports a framework.

Google ADK
    ``LlmAgent(..., before_tool_callback=adk_before_tool_callback(gate, context=...))``. ADK calls the
    callback as ``callback(tool=..., args=..., tool_context=...)``; returning ``None`` runs the tool and
    returning a dict skips it and uses the dict as the tool's response. For ASK the callback uses ADK's
    native human-in-the-loop confirmation (``tool_context.request_confirmation``) unless ``on_ask`` is given.
    ``adk_plugin(gate, ...)`` applies the same logic to every agent of a Runner (``plugins=[...]``).

LangGraph
    Pattern A, gate node: ``model -> gate -> (tools | model)`` with ``langgraph_gate_node`` and
    ``route_after_gate`` as the conditional edge. Non-certified calls are answered with ``ToolMessage``s
    carrying the gate's feedback; certified calls are left for the ``ToolNode`` (a subset is dispatched with
    ``Send`` when only some calls of a batch are certified). Pass ``interrupt=True`` (or
    ``langgraph.types.interrupt``) to escalate ASK to a human with ``interrupt()`` / ``Command(resume=...)``.

    Pattern B, tool wrapper: ``gated_tool_node(gate, tools, context=...)`` returns a ``ToolNode`` whose
    ``wrap_tool_call`` / ``awrap_tool_call`` hooks gate every call. Works anywhere a ``ToolNode`` is accepted
    (custom graphs, ``create_react_agent``).
"""
from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Literal

from ..types import Action, Decision, Verdict
from .tool import _feedback

__all__ = [
    "adk_before_tool_callback", "adk_plugin", "gated_tool_node", "langgraph_async_tool_wrapper",
    "langgraph_gate_node", "langgraph_tool_wrapper", "route_after_gate",
]


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------
def _call_context(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a user context function with ``args`` if it accepts them, otherwise with no arguments.

    Lets users pass either ``lambda tool_context: ...`` or ``lambda: ...`` to every adapter.
    """
    try:
        inspect.signature(fn).bind(*args)
    except TypeError:
        return fn()
    except ValueError:  # builtins without a signature
        return fn(*args)
    return fn(*args)


def _check(gate: Any, action: Action, context: Callable[..., Any], *ctx_args: Any) -> Decision:
    """``gate.check`` with the context built from ``context(*ctx_args)``; a failing context fails closed (ASK)."""
    try:
        ctx = _call_context(context, *ctx_args)
    except Exception as exc:  # noqa: BLE001 - rule R3: fail closed
        return Decision(Verdict.ASK, action, reason=f"fail-closed: could not build the gate context: "
                                                    f"{type(exc).__name__}: {exc}")
    return gate.check(action, ctx)


def _safe(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats become None, unknown objects become strings."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _feedback_json(d: Decision) -> str:
    return json.dumps(_safe(_feedback(d)))


def _approved(answer: Any) -> bool:
    """Interpret a human reviewer's answer (``Command(resume=...)`` value or ``on_ask`` result)."""
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, Mapping):
        return bool(answer.get("approve", answer.get("approved", False)))
    if isinstance(answer, str):
        return answer.strip().lower() in ("approve", "approved", "yes", "y", "ok")
    return False


def _resolve_interrupt(interrupt: Callable[[Any], Any] | bool | None) -> Callable[[Any], Any] | None:
    if interrupt is True:
        from langgraph.types import interrupt as lg_interrupt  # lazy: only when human review is requested
        return lg_interrupt
    if interrupt is False or interrupt is None:
        return None
    return interrupt


# --------------------------------------------------------------------------------------------
# Google ADK
# --------------------------------------------------------------------------------------------
def _adk_decide(gate: Any, tool: Any, args: Mapping[str, Any] | None, tool_context: Any,
                context: Callable[..., Any], on_ask: Callable[[Decision], bool] | None, confirm: bool,
                passthrough: Collection[str]) -> dict | None:
    name = getattr(tool, "name", None) or str(tool)
    if name in passthrough:
        return None
    d = _check(gate, Action(name, dict(args or {})), context, tool_context)
    if d.verdict is Verdict.CERTIFY:
        return None
    if d.verdict is Verdict.ASK:
        if on_ask is not None:
            return None if on_ask(d) else _feedback(d)
        if confirm and tool_context is not None and hasattr(tool_context, "request_confirmation"):
            confirmation = getattr(tool_context, "tool_confirmation", None)
            if confirmation is None:          # first pass: pause the run and ask the user (ADK HITL)
                try:
                    tool_context.request_confirmation(
                        hint=f"riskgate cannot certify {name}({json.dumps(_safe(dict(args or {})))}): {d.reason}. "
                             "Approve or reject this tool call.",
                        payload={"riskgate": _safe(_feedback(d))})
                except Exception:  # noqa: BLE001 - e.g. no function_call_id: fall back to feedback
                    return _feedback(d)
                actions = getattr(tool_context, "actions", None)
                if actions is not None:       # mirror ADK's own confirmation gate: do not re-invoke the model
                    actions.skip_summarization = True
                return _feedback(d)
            if getattr(confirmation, "confirmed", False):
                return None                   # the user approved this call
            return {**_feedback(d), "status": "rejected", "reason": f"declined by the user: {d.reason}"}
    return _feedback(d)


def adk_before_tool_callback(gate: Any, *, context: Callable[..., Any] = lambda tool_context: None,
                             on_ask: Callable[[Decision], bool] | None = None, confirm: bool = True,
                             passthrough: Collection[str] = ()) -> Callable[..., dict | None]:
    """Build an ADK ``before_tool_callback``: ``callback(tool, args, tool_context) -> None | dict``.

    Args:
        gate: a calibrated ``riskgate.Gate`` (anything with ``check(action, context)``).
        context: ``context(tool_context)`` (or ``context()``) returning the gate context.
        on_ask: optional synchronous approval hook for ASK; True runs the tool.
        confirm: when ``on_ask`` is None, use ADK's native tool confirmation for ASK: the run pauses with an
            ``adk_request_confirmation`` function call; after the user replies ``{"confirmed": true}`` ADK re-runs
            the call, the gate re-checks it and the tool runs. Without a confirming client the model receives
            ``{"status": "needs_approval", ...}``.
        passthrough: tool names that are never gated (e.g. ``{"transfer_to_agent"}``). Use sparingly.

    Returns ``None`` (run the tool) for CERTIFY or an approved ASK; otherwise a dict (status, reason,
    guarantee[, probe]) that ADK uses as the tool's response, so the model can re-plan.
    """
    frozen = frozenset(passthrough)

    def callback(tool: Any, args: dict, tool_context: Any = None) -> dict | None:
        return _adk_decide(gate, tool, args, tool_context, context, on_ask, confirm, frozen)

    return callback


def adk_plugin(gate: Any, *, context: Callable[..., Any] = lambda tool_context: None,
               on_ask: Callable[[Decision], bool] | None = None, confirm: bool = True,
               passthrough: Collection[str] = (), name: str = "riskgate") -> Any:
    """Return an ADK ``BasePlugin`` that gates every tool call of every agent run by a Runner.

    ``InMemoryRunner(agent=root, plugins=[adk_plugin(gate, context=...)])``. Plugin callbacks run before agent
    callbacks; returning a dict skips the tool. Same semantics as ``adk_before_tool_callback``. Requires
    ``google-adk`` (imported lazily).
    """
    from google.adk.plugins.base_plugin import BasePlugin

    frozen = frozenset(passthrough)

    class RiskgatePlugin(BasePlugin):
        """Gates every tool call with a riskgate Gate."""

        async def before_tool_callback(self, *, tool: Any, tool_args: dict[str, Any],
                                       tool_context: Any) -> dict | None:
            return _adk_decide(gate, tool, tool_args, tool_context, context, on_ask, confirm, frozen)

    return RiskgatePlugin(name=name)


# --------------------------------------------------------------------------------------------
# LangGraph / LangChain messages
# --------------------------------------------------------------------------------------------
def _messages(state: Any, key: str = "messages") -> list:
    if isinstance(state, list):
        return state
    if isinstance(state, Mapping):
        return list(state.get(key) or [])
    return list(getattr(state, key, None) or [])


def _calls(message: Any) -> list[dict]:
    """Tool calls of an AI message as ``[{"id", "name", "args"}]``.

    Accepts ``AIMessage`` objects, LangChain-style dicts and OpenAI-style dicts (``{"function": {...}}``).
    """
    calls = message.get("tool_calls") if isinstance(message, Mapping) else getattr(message, "tool_calls", None)
    out = []
    for c in calls or []:
        if not isinstance(c, Mapping):
            c = {"id": getattr(c, "id", None), "name": getattr(c, "name", None), "args": getattr(c, "args", {})}
        if "function" in c and "name" not in c:          # OpenAI chat-completions format
            fn = c.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError):               # malformed: let the gate see (and fail closed on) it
                args = {"_unparsed_arguments": raw}
            out.append({"id": c.get("id"), "name": fn.get("name"),
                        "args": args if isinstance(args, dict) else {"input": args}})
        else:
            out.append({"id": c.get("id"), "name": c.get("name"), "args": dict(c.get("args") or {})})
    return out


def _tool_call_id(message: Any) -> str | None:
    if isinstance(message, Mapping):
        return message.get("tool_call_id")
    return getattr(message, "tool_call_id", None)


def _reply(call: Mapping[str, Any], d: Decision, like: Any, fmt: str) -> Any:
    """Tool reply carrying the gate's feedback: a ``ToolMessage`` (status="error") or an OpenAI-style dict."""
    content = _feedback_json(d)
    as_dict = {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": content}
    if fmt == "dict" or (fmt == "auto" and isinstance(like, Mapping)):
        return as_dict
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        if fmt == "langchain":
            raise
        return as_dict
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"], status="error",
                       artifact={"riskgate": _safe(d.to_dict())})


def _resolve_ask(d: Decision, call: Mapping[str, Any], interrupt_fn: Callable[[Any], Any] | None,
                 on_ask: Callable[[Decision], bool] | None) -> bool:
    """True if a human (interrupt) or the ``on_ask`` hook approves an ASK decision."""
    if interrupt_fn is not None:
        answer = interrupt_fn({"type": "riskgate_approval",
                               "tool_call": {"id": call["id"], "name": call["name"], "args": _safe(call["args"])},
                               "reason": d.reason, "guarantee": d.status.value, "feedback": _safe(_feedback(d))})
        return _approved(answer)
    if on_ask is not None:
        return bool(on_ask(d))
    return False


def langgraph_gate_node(gate: Any, *, context: Callable[..., Any] = lambda state: None,
                        interrupt: Callable[[Any], Any] | bool | None = None,
                        on_ask: Callable[[Decision], bool] | None = None, messages_key: str = "messages",
                        decisions_key: str | None = "riskgate",
                        message_format: Literal["auto", "langchain", "dict"] = "auto") -> Callable[[Any], dict]:
    """Build a LangGraph node ``node(state) -> update`` that gates the last AI message's tool calls.

    Place it between the model node and the ``ToolNode`` and use ``route_after_gate`` as its conditional
    edge. Every non-certified call is answered immediately with a tool message whose content is the JSON
    feedback (status, reason, guarantee[, probe]), so the model can re-plan. Certified calls get no reply
    and are executed by the ``ToolNode``.

    Args:
        context: ``context(state)`` (or ``context()``) returning the gate context; evaluated once per step.
        interrupt: ``True`` (uses ``langgraph.types.interrupt``) or an interrupt-like callable to escalate ASK
            to a human. The payload is ``{"type": "riskgate_approval", "tool_call", "reason", ...}``; resume
            with ``Command(resume={"approve": True})`` (or ``True``/``"approve"``) to run the call. Needs a
            checkpointer. Takes precedence over ``on_ask``.
        on_ask: synchronous approval hook for ASK (True runs the call).
        messages_key: state key holding the messages.
        decisions_key: state key for ``{tool_call_id: decision dict}``; dropped silently by LangGraph if the
            state schema has no such key (routing does not depend on it). ``None`` disables it.
        message_format: ``"auto"`` replies with ``ToolMessage`` objects when the AI message is a LangChain
            message (always the case inside a graph using ``add_messages``) and with dicts when it is a dict.
    """
    def node(state: Any) -> dict:
        msgs = _messages(state, messages_key)
        if not msgs:
            return {}
        last = msgs[-1]
        calls = _calls(last)
        update: dict = {}
        if not calls:
            if decisions_key:
                update[decisions_key] = {}
            return update
        interrupt_fn = _resolve_interrupt(interrupt)
        try:
            ctx = _call_context(context, state)
            ctx_error: Exception | None = None
        except Exception as exc:  # noqa: BLE001 - fail closed below
            ctx, ctx_error = None, exc
        decisions, replies = {}, []
        for c in calls:
            action = Action(c["name"], dict(c["args"] or {}))
            if ctx_error is not None:
                d = Decision(Verdict.ASK, action, reason=f"fail-closed: could not build the gate context: "
                                                         f"{type(ctx_error).__name__}: {ctx_error}")
            else:
                d = gate.check(action, ctx)
            rec = _safe(d.to_dict())
            if d.verdict is Verdict.ASK and _resolve_ask(d, c, interrupt_fn, on_ask):
                rec.update(verdict="certify", gate_verdict="ask", approved_by="human")
            decisions[c["id"]] = rec
            if rec["verdict"] != "certify":
                replies.append(_reply(c, d, last, message_format))
        if decisions_key:
            update[decisions_key] = decisions
        if replies:
            update[messages_key] = replies
        return update

    return node


def route_after_gate(state: Any, tools_node: str = "tools", model_node: str = "agent",
                     messages_key: str = "messages", decisions_key: str = "riskgate") -> Any:
    """Conditional edge after ``langgraph_gate_node``.

    Derived from the messages, so it works with plain ``MessagesState``: if every tool call of the last AI
    message is still unanswered (all certified) -> ``tools_node``; if all were answered by the gate ->
    ``model_node``; if only some were certified -> ``[Send(tools_node, [certified tool calls])]`` so the
    ``ToolNode`` executes exactly those. Falls back to the ``decisions_key`` entry when the state has no AI
    message (e.g. when called on the node's own update in unit tests).
    Use ``functools.partial`` (or a lambda) to rename the nodes.
    """
    msgs = _messages(state, messages_key)
    idx = next((i for i in range(len(msgs) - 1, -1, -1) if _calls(msgs[i])), None)
    if idx is not None:
        calls = _calls(msgs[idx])
        answered = {_tool_call_id(m) for m in msgs[idx + 1:]}
        pending = [c for c in calls if c["id"] not in answered]
        if not pending:
            return model_node
        if len(pending) == len(calls):
            return tools_node
        try:
            from langgraph.types import Send
        except ImportError:  # cannot dispatch a subset outside LangGraph: fail closed
            return model_node
        return [Send(tools_node, [{"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"}
                                  for c in pending])]
    decisions = state.get(decisions_key, {}) if isinstance(state, Mapping) else getattr(state, decisions_key, {})
    decisions = decisions or {}
    return tools_node if decisions and all(d["verdict"] == "certify" for d in decisions.values()) else model_node


def _gate_tool_request(gate: Any, request: Any, context: Callable[..., Any],
                       interrupt: Callable[[Any], Any] | bool | None,
                       on_ask: Callable[[Decision], bool] | None) -> Any:
    """Return None if the ToolNode request may execute, else a ``ToolMessage`` with the gate's feedback."""
    call = request.tool_call
    c = {"id": call.get("id"), "name": call.get("name"), "args": dict(call.get("args") or {})}
    d = _check(gate, Action(c["name"], c["args"]), context, getattr(request, "state", None))
    if d.verdict is Verdict.CERTIFY:
        return None
    if d.verdict is Verdict.ASK and _resolve_ask(d, c, _resolve_interrupt(interrupt), on_ask):
        return None
    return _reply(c, d, None, "langchain")


def langgraph_tool_wrapper(gate: Any, *, context: Callable[..., Any] = lambda state: None,
                           interrupt: Callable[[Any], Any] | bool | None = None,
                           on_ask: Callable[[Decision], bool] | None = None) -> Callable[[Any, Callable], Any]:
    """Sync ``wrap_tool_call`` hook for ``langgraph.prebuilt.ToolNode`` (langgraph-prebuilt >= 1.0).

    ``ToolNode(tools, wrap_tool_call=langgraph_tool_wrapper(gate, context=...))``: ``context(state)`` gets
    the graph state; certified (or approved) calls run via ``execute(request)``, all others return a
    ``ToolMessage(status="error")`` with the JSON feedback. ``interrupt`` works as in ``langgraph_gate_node``
    but note that ToolNode runs a step's calls concurrently: with several ASK calls in one step prefer the
    gate node (deterministic interrupt order) or ``create_react_agent(version="v2")`` (one task per call).
    Interrupts propagate only with ToolNode's default ``handle_tool_errors``.
    """
    def wrap_tool_call(request: Any, execute: Callable[[Any], Any]) -> Any:
        blocked = _gate_tool_request(gate, request, context, interrupt, on_ask)
        return blocked if blocked is not None else execute(request)

    return wrap_tool_call


def langgraph_async_tool_wrapper(gate: Any, *, context: Callable[..., Any] = lambda state: None,
                                 interrupt: Callable[[Any], Any] | bool | None = None,
                                 on_ask: Callable[[Decision], bool] | None = None) -> Callable[[Any, Callable], Any]:
    """Async ``awrap_tool_call`` counterpart of ``langgraph_tool_wrapper`` (used by ``ainvoke``/``astream``)."""
    async def awrap_tool_call(request: Any, execute: Callable[[Any], Any]) -> Any:
        blocked = _gate_tool_request(gate, request, context, interrupt, on_ask)
        return blocked if blocked is not None else await execute(request)

    return awrap_tool_call


def gated_tool_node(gate: Any, tools: Sequence[Any], *, context: Callable[..., Any] = lambda state: None,
                    interrupt: Callable[[Any], Any] | bool | None = None,
                    on_ask: Callable[[Decision], bool] | None = None, **tool_node_kwargs: Any) -> Any:
    """A ``langgraph.prebuilt.ToolNode`` whose every call is gated (sync and async paths).

    ``create_react_agent(model, tools=gated_tool_node(gate, [add], context=lambda state: snapshot()))`` or
    ``graph.add_node("tools", gated_tool_node(...))``. Extra keyword arguments go to ``ToolNode``.
    """
    from langgraph.prebuilt import ToolNode

    return ToolNode(list(tools),
                    wrap_tool_call=langgraph_tool_wrapper(gate, context=context, interrupt=interrupt, on_ask=on_ask),
                    awrap_tool_call=langgraph_async_tool_wrapper(gate, context=context, interrupt=interrupt,
                                                                 on_ask=on_ask),
                    **tool_node_kwargs)
