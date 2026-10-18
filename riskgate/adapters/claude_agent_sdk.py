"""Adapter for the Claude Agent SDK (``claude-agent-sdk``; imported lazily).

Verified against claude-agent-sdk 0.2.168 (``tests/contract/test_claude_agent_sdk.py`` checks the types). Two
integration points, use one:

PreToolUse hook (recommended): the gate runs before every matching tool call; the verdict becomes the hook's
``permissionDecision``::

    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from riskgate.adapters.claude_agent_sdk import pretooluse_hook

    options = ClaudeAgentOptions(hooks={"PreToolUse": [
        HookMatcher(matcher="mcp__payments__.*", hooks=[pretooluse_hook(gate, context=snapshot)])]})

    CERTIFY -> no decision: the call proceeds under your normal permission settings
               (``allow_on_certify=True`` returns "allow", which also skips Claude Code's own prompt)
    REJECT  -> "deny", reason = the gate's JSON feedback (Claude reads it and re-plans)
    PROBE   -> "deny", reason includes the read-only probe to call first
    ASK     -> "ask": Claude Code asks the user to approve this call, with the gate's reason

``can_use_tool`` callback: for apps that decide permissions in code; ASK goes to ``on_ask(decision)``
(deny when absent, fail closed).

Tool names: MCP tools arrive as ``mcp__<server>__<tool>``; ``name`` maps them to the gate's action names (default:
strip the ``mcp__<server>__`` prefix). Tools outside ``tools`` (when given) are left alone.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Any

from ..types import Action, Decision, Verdict
from .frameworks import _call_context, _safe
from .tool import _feedback

__all__ = ["can_use_tool", "pretooluse_hook", "strip_mcp_prefix"]


def strip_mcp_prefix(tool_name: str) -> str:
    """``mcp__payments__issue_refund`` -> ``issue_refund``; other names unchanged."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        if len(parts) == 3 and parts[2]:
            return parts[2]
    return tool_name


async def _maybe_await(v: Any) -> Any:
    return await v if inspect.isawaitable(v) else v


async def _decide(gate: Any, tool_name: str, tool_input: Mapping[str, Any] | None,
                  context: Callable[..., Any], name: Callable[[str], str], ctx_arg: Any) -> Decision:
    action = Action(name(tool_name), dict(tool_input or {}))
    try:
        ctx = await _maybe_await(_call_context(context, ctx_arg))
    except Exception as exc:  # noqa: BLE001 - rule R3: fail closed
        return Decision(Verdict.ASK, action, reason=f"fail-closed: could not build the gate context: "
                                                    f"{type(exc).__name__}: {exc}")
    return gate.check(action, ctx)


def _reason(d: Decision) -> str:
    return "riskgate: " + json.dumps(_safe(_feedback(d)))


def pretooluse_hook(gate: Any, *, context: Callable[..., Any] = lambda: None,
                    tools: Collection[str] | None = None, name: Callable[[str], str] = strip_mcp_prefix,
                    allow_on_certify: bool = False) -> Callable[[Any, str | None, Any], Awaitable[dict]]:
    """A ``HookCallback`` for ``ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[...])]})``.

    context           zero-argument (or one-argument: the hook input) sync or async function returning the gate's
                      context; if it raises, the call goes to the user (ASK, fail closed)
    tools             gate only these tool names (after ``name``); default: every tool the matcher sends
    name              maps the SDK tool name to the gate's action name
    allow_on_certify  return "allow" for CERTIFY (skips the host's permission prompt) instead of no decision
    """
    only = frozenset(tools) if tools is not None else None

    async def hook(input_data: Any, tool_use_id: str | None, hook_context: Any) -> dict:
        if input_data.get("hook_event_name") != "PreToolUse":
            return {}
        tool_name = str(input_data.get("tool_name", ""))
        if only is not None and name(tool_name) not in only:
            return {}
        d = await _decide(gate, tool_name, input_data.get("tool_input"), context, name, input_data)
        if d.verdict is Verdict.CERTIFY:
            if not allow_on_certify:
                return {}
            decision = "allow"
        else:
            decision = "ask" if d.verdict is Verdict.ASK else "deny"
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                       "permissionDecisionReason": _reason(d)}}

    hook.__riskgate__ = gate  # type: ignore[attr-defined]
    return hook


def can_use_tool(gate: Any, *, context: Callable[..., Any] = lambda: None, tools: Collection[str] | None = None,
                 name: Callable[[str], str] = strip_mcp_prefix,
                 on_ask: Callable[[Decision], bool | Awaitable[bool]] | None = None) -> Callable[..., Awaitable[Any]]:
    """A ``can_use_tool`` callback for ``ClaudeAgentOptions(can_use_tool=...)``: CERTIFY allows, REJECT and PROBE
    deny with the gate's feedback, ASK asks ``on_ask(decision)`` (deny when absent). Tools outside ``tools`` are
    allowed, so combine it with the SDK's own permission settings for those."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    only = frozenset(tools) if tools is not None else None

    async def callback(tool_name: str, tool_input: dict[str, Any], permission_context: Any) -> Any:
        if only is not None and name(tool_name) not in only:
            return PermissionResultAllow()
        d = await _decide(gate, tool_name, tool_input, context, name, permission_context)
        if d.verdict is Verdict.CERTIFY:
            return PermissionResultAllow()
        if d.verdict is Verdict.ASK and on_ask is not None and await _maybe_await(on_ask(d)):
            return PermissionResultAllow()
        return PermissionResultDeny(message=_reason(d))

    callback.__riskgate__ = gate  # type: ignore[attr-defined]
    return callback
