"""Contract tests: riskgate x OpenAI Agents SDK (real openai-agents Runner, agents.testing.ScriptedModel).

No API key and no network: the model is the SDK's own deterministic ``ScriptedModel`` and tracing is disabled.
"""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("agents")

from agents import Agent, FunctionTool, RunConfig, RunContextWrapper, Runner, ToolInputGuardrailTripwireTriggered, function_tool  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402
from toy_domain import ASKED, CERTIFIED, REJECTED  # noqa: E402

import riskgate as rg  # noqa: E402
from riskgate.adapters.openai_agents import guarded_function_tool, riskgate_needs_approval, riskgate_tool_input_guardrail  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
NO_TRACING = RunConfig(tracing_disabled=True)


class Tank:
    def __init__(self) -> None:
        self.executed: list[float] = []

    def add(self, v: float) -> str:
        """Add v litres to the tank.

        Args:
            v: litres to add.
        """
        self.executed.append(v)
        return f"added {v}"


def tank_fn(tank: Tank):
    def add(v: float) -> str:
        """Add v litres to the tank."""
        return tank.add(v)

    return add


def calls(*vs: float, prefix: str = "c") -> list[list]:
    return [[function_call("add", {"v": v}, call_id=f"{prefix}{i}")] for i, v in enumerate(vs)]


def outputs(result) -> dict[str, str]:
    """call_id -> tool output string, as sent back to the model."""
    out = {}
    for item in result.new_items:
        raw = item.raw_item
        if isinstance(raw, dict) and raw.get("type") == "function_call_output":
            out[raw["call_id"]] = raw["output"]
    return out


def invoke(tool: FunctionTool, args: dict, call_id: str = "t1"):
    payload = json.dumps(args)
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id=call_id, tool_arguments=payload)
    return asyncio.run(tool.on_invoke_tool(ctx, payload))


def test_guarded_function_tool_schema_and_direct_invocation(toy_gate, state):
    tank = Tank()
    tool = guarded_function_tool(toy_gate, tank_fn(tank), context=lambda: state)
    assert isinstance(tool, FunctionTool) and tool.name == "add"
    assert tool.params_json_schema["properties"]["v"]["type"] == "number"
    assert "Add v litres" in tool.description
    assert invoke(tool, {"v": CERTIFIED}) == f"added {CERTIFIED}"
    rejected = invoke(tool, {"v": REJECTED})
    assert rejected["status"] == "rejected" and rejected["guarantee"] == "valid"
    asked = invoke(tool, {"v": ASKED})
    assert asked["status"] == "needs_approval"                       # ASK without a hook -> feedback, no raise
    assert tank.executed == [CERTIFIED]
    approving = guarded_function_tool(toy_gate, tank_fn(tank), context=lambda: state, on_ask=lambda d: True,
                                      name="add_approved")
    assert approving.name == "add_approved" and invoke(approving, {"v": ASKED}) == f"added {ASKED}"


def test_guarded_function_tool_end_to_end_with_runner(toy_gate, state):
    tank = Tank()
    model = ScriptedModel(calls(CERTIFIED, REJECTED, ASKED) + [[assistant_message("done")]])
    agent = Agent(name="tank", model=model, tools=[guarded_function_tool(toy_gate, tank_fn(tank),
                                                                         context=lambda: state)])
    result = asyncio.run(Runner.run(agent, "fill the tank", run_config=NO_TRACING))
    out = outputs(result)
    assert tank.executed == [CERTIFIED] and result.final_output == "done"
    assert out["c0"] == f"added {CERTIFIED}"
    assert "'status': 'rejected'" in out["c1"] and "'status': 'needs_approval'" in out["c2"]   # str(dict) by the SDK
    model.assert_complete()


def test_tool_input_guardrail_answers_with_json_feedback(toy_gate, state):
    tank = Tank()
    tool = function_tool(tank_fn(tank), tool_input_guardrails=[
        riskgate_tool_input_guardrail(toy_gate, context=lambda tool_context: state)])
    model = ScriptedModel(calls(CERTIFIED, REJECTED, ASKED) + [[assistant_message("done")]])
    result = asyncio.run(Runner.run(Agent(name="tank", model=model, tools=[tool]), "go", run_config=NO_TRACING))
    out = outputs(result)
    assert tank.executed == [CERTIFIED] and result.final_output == "done"
    assert json.loads(out["c1"])["status"] == "rejected"
    assert json.loads(out["c2"])["status"] == "needs_approval"
    behaviours = [(r.guardrail.get_name(), r.output.behavior["type"], r.output.output_info["verdict"])
                  for r in result.tool_input_guardrail_results]
    assert behaviours == [("riskgate", "allow", "certify"), ("riskgate", "reject_content", "reject"),
                          ("riskgate", "reject_content", "ask")]


@pytest.mark.parametrize("approve", [True, False])
def test_needs_approval_maps_ask_onto_sdk_interruptions(toy_gate, state, approve):
    tank = Tank()
    tool = function_tool(tank_fn(tank),
                         needs_approval=riskgate_needs_approval(toy_gate, "add", context=lambda run_context: state),
                         tool_input_guardrails=[riskgate_tool_input_guardrail(toy_gate, context=lambda: state)])
    model = ScriptedModel(calls(CERTIFIED, REJECTED, ASKED) + [[assistant_message("done")]])
    agent = Agent(name="tank", model=model, tools=[tool])

    first = asyncio.run(Runner.run(agent, "go", run_config=NO_TRACING))
    assert [(i.tool_name, json.loads(i.raw_item.arguments)) for i in first.interruptions] == [("add", {"v": ASKED})]
    assert tank.executed == [CERTIFIED]                               # certified ran, rejected answered, ASK paused
    assert json.loads(outputs(first)["c1"])["status"] == "rejected"
    state_ = first.to_state()
    for item in first.interruptions:
        state_.approve(item) if approve else state_.reject(item)
    second = asyncio.run(Runner.run(agent, state_, run_config=NO_TRACING))
    assert second.interruptions == [] and second.final_output == "done"
    if approve:
        assert tank.executed == [CERTIFIED, ASKED] and outputs(second)["c2"] == f"added {ASKED}"
    else:
        assert tank.executed == [CERTIFIED] and "added" not in outputs(second)["c2"]


def test_sdk_requires_approval_when_validation_changes_arguments(toy_gate, state):
    """Pins an SDK 0.23 behaviour: a callable needs_approval is bypassed (approval required) when argument
    validation changes the input, e.g. JSON int 10 coerced to float 10.0 for a ``float`` parameter."""
    tank = Tank()
    tool = function_tool(tank_fn(tank), needs_approval=riskgate_needs_approval(toy_gate, "add",
                                                                               context=lambda: state))
    model = ScriptedModel([[function_call("add", {"v": 10}, call_id="i0")], [assistant_message("done")]])
    result = asyncio.run(Runner.run(Agent(name="tank", model=model, tools=[tool]), "go", run_config=NO_TRACING))
    assert len(result.interruptions) == 1 and tank.executed == []    # although the gate certifies add(v=10)


def test_halt_on_reject_raises_tripwire(toy_gate, state):
    tank = Tank()
    tool = function_tool(tank_fn(tank), tool_input_guardrails=[
        riskgate_tool_input_guardrail(toy_gate, context=lambda: state, halt_on={rg.Verdict.REJECT})])
    model = ScriptedModel(calls(REJECTED) + [[assistant_message("never reached")]])
    with pytest.raises(ToolInputGuardrailTripwireTriggered):
        asyncio.run(Runner.run(Agent(name="tank", model=model, tools=[tool]), "go", run_config=NO_TRACING))
    assert tank.executed == []


def test_probe_feedback_and_malformed_arguments(probe_gate, state):
    guardrail = riskgate_tool_input_guardrail(probe_gate, context=lambda: state)

    class Data:   # minimal ToolInputGuardrailData stand-in: the guardrail only reads .context
        def __init__(self, arguments: str) -> None:
            self.context = ToolContext(context=None, tool_name="add", tool_call_id="x", tool_arguments=arguments)

    out = asyncio.run(guardrail.run(Data(json.dumps({"v": 1.0}))))
    fb = json.loads(out.behavior["message"])
    assert out.behavior["type"] == "reject_content" and fb["status"] == "probe_requested"
    assert fb["probe"]["name"] == "read_flowmeter"
    bad = asyncio.run(guardrail.run(Data("{not json")))
    assert bad.behavior["type"] == "reject_content" and "not valid JSON" in bad.behavior["message"]


def test_functions_taking_run_context_are_refused(toy_gate):
    def add(ctx: RunContextWrapper, v: float) -> str:
        """Add v litres."""
        return "added"

    with pytest.raises(TypeError, match="run context"):
        guarded_function_tool(toy_gate, add)


def test_context_failure_fails_closed_in_tool_and_guardrail(toy_gate):
    def broken() -> dict:
        raise RuntimeError("no snapshot")

    tank = Tank()
    out = invoke(guarded_function_tool(toy_gate, tank_fn(tank), context=broken), {"v": CERTIFIED})
    assert out["status"] == "needs_approval" and "fail-closed" in out["reason"] and tank.executed == []

    model = ScriptedModel(calls(CERTIFIED) + [[assistant_message("done")]])
    tool = function_tool(tank_fn(tank), tool_input_guardrails=[riskgate_tool_input_guardrail(toy_gate,
                                                                                             context=broken)])
    result = asyncio.run(Runner.run(Agent(name="tank", model=model, tools=[tool]), "go", run_config=NO_TRACING))
    fb = json.loads(outputs(result)["c0"])
    assert fb["status"] == "needs_approval" and "fail-closed" in fb["reason"] and tank.executed == []
