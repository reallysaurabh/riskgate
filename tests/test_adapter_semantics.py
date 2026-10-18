"""Outcome semantics shared by every adapter (framework-free, runs in the main CI job).

needs_approval  an ASK nobody could be asked about in-band
declined        a human (or on_ask) said no
rejected        unsafe under every plausible world: re-plan
probe_requested run the read-only probe first
"""
import pytest

import riskgate as rg


class StubGate:
    def __init__(self, verdict: rg.Verdict):
        self.verdict = verdict

    def check(self, action, context, *, stage=1):
        return rg.Decision(self.verdict, action, reason="r", status=rg.GuaranteeStatus.VALID)


def tool(v: float) -> str:
    return "ran"


@pytest.mark.parametrize("verdict, on_ask, raise_on_ask, expected", [
    (rg.Verdict.CERTIFY, None, True, "ran"),
    (rg.Verdict.ASK, lambda d: True, True, "ran"),
    (rg.Verdict.ASK, lambda d: False, True, "declined"),
    (rg.Verdict.ASK, None, False, "needs_approval"),
    (rg.Verdict.REJECT, None, True, "rejected"),
    (rg.Verdict.PROBE, None, True, "probe_requested"),
])
def test_guard_outcome_semantics(verdict, on_ask, raise_on_ask, expected):
    out = rg.guard(StubGate(verdict), name="tool", on_ask=on_ask, raise_on_ask=raise_on_ask)(tool)(1.0)
    if expected == "ran":
        assert out == "ran"
    else:
        assert out["status"] == expected and out["executed"] is False


def test_guard_raises_without_approver_by_default():
    with pytest.raises(rg.ApprovalRequired):
        rg.guard(StubGate(rg.Verdict.ASK), name="tool")(tool)(1.0)


def test_claude_agent_sdk_can_use_tool_semantics():
    sdk = pytest.importorskip("claude_agent_sdk")
    import asyncio

    from riskgate.adapters.claude_agent_sdk import can_use_tool
    no = asyncio.run(can_use_tool(StubGate(rg.Verdict.ASK), on_ask=lambda d: False)("t", {}, None))
    unasked = asyncio.run(can_use_tool(StubGate(rg.Verdict.ASK))("t", {}, None))
    assert isinstance(no, sdk.PermissionResultDeny) and '"declined"' in no.message
    assert '"needs_approval"' in unasked.message
