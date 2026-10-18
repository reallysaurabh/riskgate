"""Claude Agent SDK adapter: PreToolUse hook and can_use_tool callback (real SDK types, no model, no network)."""
from __future__ import annotations

import asyncio
import json
import typing

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

from toy_domain import ASKED, CERTIFIED, REJECTED  # noqa: E402

import riskgate as rg  # noqa: E402
from riskgate.adapters.claude_agent_sdk import can_use_tool, pretooluse_hook, strip_mcp_prefix  # noqa: E402

PRE_KEYS = set(typing.get_type_hints(claude_agent_sdk.types.PreToolUseHookSpecificOutput))
DECISIONS = set(typing.get_args(typing.get_type_hints(claude_agent_sdk.types.PreToolUseHookSpecificOutput)
                                ["permissionDecision"]))                     # NotRequired is stripped by get_type_hints


def _input(v, tool="mcp__plant__add"):
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": {"v": v}, "tool_use_id": "tu1",
            "session_id": "s", "transcript_path": "/tmp/t", "cwd": "/tmp"}


def _run(coro):
    return asyncio.run(coro)


def test_hook_output_matches_the_sdk_schema(toy_gate, state):
    hook = pretooluse_hook(toy_gate, context=lambda: state)
    out = {v: _run(hook(_input(v), "tu1", None)) for v in (CERTIFIED, ASKED, REJECTED)}
    assert out[CERTIFIED] == {}                                              # certified: normal permission flow
    for v, decision in ((ASKED, "ask"), (REJECTED, "deny")):
        spec = out[v]["hookSpecificOutput"]
        assert set(spec) <= PRE_KEYS and spec["hookEventName"] == "PreToolUse"
        assert spec["permissionDecision"] == decision and decision in DECISIONS
        fb = json.loads(spec["permissionDecisionReason"].removeprefix("riskgate: "))
        assert fb["status"] in ("needs_approval", "rejected") and fb["guarantee"] == "valid"
    allow = pretooluse_hook(toy_gate, context=lambda: state, allow_on_certify=True)
    assert _run(allow(_input(CERTIFIED), "tu1", None))["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_hook_probe_scope_names_and_fail_closed(probe_gate, toy_gate, state):
    spec = _run(pretooluse_hook(probe_gate)(_input(1.0), "tu1", None))["hookSpecificOutput"]
    assert spec["permissionDecision"] == "deny" and "read_flowmeter" in spec["permissionDecisionReason"]
    scoped = pretooluse_hook(toy_gate, context=lambda: state, tools={"other"})
    assert _run(scoped(_input(REJECTED), "tu1", None)) == {}                 # not gated
    assert _run(pretooluse_hook(toy_gate)({"hook_event_name": "PostToolUse"}, None, None)) == {}

    def broken():
        raise RuntimeError("snapshot down")
    spec = _run(pretooluse_hook(toy_gate, context=broken)(_input(CERTIFIED), "tu1", None))["hookSpecificOutput"]
    assert spec["permissionDecision"] == "ask" and "fail-closed" in spec["permissionDecisionReason"]

    async def actx(inp):                                                       # async, receives the hook input
        assert inp["tool_name"] == "mcp__plant__add"
        return state
    assert _run(pretooluse_hook(toy_gate, context=actx)(_input(CERTIFIED), "tu1", None)) == {}
    assert strip_mcp_prefix("mcp__a__b__c") == "b__c" and strip_mcp_prefix("Bash") == "Bash"


def test_hook_registers_in_options(toy_gate, state):
    opts = claude_agent_sdk.ClaudeAgentOptions(hooks={"PreToolUse": [
        claude_agent_sdk.HookMatcher(matcher="mcp__plant__.*", hooks=[pretooluse_hook(toy_gate, context=lambda: state)])]})
    assert opts.hooks["PreToolUse"][0].hooks[0].__riskgate__ is toy_gate


def test_can_use_tool(toy_gate, state):
    cb = can_use_tool(toy_gate, context=lambda: state)
    ok = _run(cb("mcp__plant__add", {"v": CERTIFIED}, None))
    assert isinstance(ok, claude_agent_sdk.PermissionResultAllow)
    for v in (ASKED, REJECTED):
        r = _run(cb("mcp__plant__add", {"v": v}, None))
        assert isinstance(r, claude_agent_sdk.PermissionResultDeny) and r.message.startswith("riskgate: ")
    approve = can_use_tool(toy_gate, context=lambda: state, on_ask=lambda d: d.verdict is rg.Verdict.ASK)
    assert isinstance(_run(approve("mcp__plant__add", {"v": ASKED}, None)), claude_agent_sdk.PermissionResultAllow)
    assert isinstance(_run(approve("mcp__plant__add", {"v": REJECTED}, None)), claude_agent_sdk.PermissionResultDeny)
    claude_agent_sdk.ClaudeAgentOptions(can_use_tool=cb)
