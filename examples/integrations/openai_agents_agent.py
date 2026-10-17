"""riskgate x OpenAI Agents SDK, offline: the SDK's own ScriptedModel drives a real Runner.

    pip install -e . openai-agents
    python examples/integrations/openai_agents_agent.py

Part A: ``guarded_function_tool`` (the gate runs inside the tool function).
Part B: native SDK hooks: ``riskgate_tool_input_guardrail`` + ``riskgate_needs_approval``, so ASK becomes an SDK
approval interruption that the operator approves before the run resumes.
For a real model drop ``model=ScriptedModel(...)`` (the SDK default model is used) or pass a model name, and
set ``OPENAI_API_KEY`` (see README.md).
"""
from __future__ import annotations

import asyncio
import json

from _toy_gate import make_gate, snapshot
from agents import Agent, RunConfig, Runner, function_tool
from agents.testing import ScriptedModel, assistant_message, function_call

from riskgate.adapters.openai_agents import guarded_function_tool, riskgate_needs_approval, riskgate_tool_input_guardrail

OFFLINE = RunConfig(tracing_disabled=True)   # no trace export: the example never touches the network


def add(v: float) -> str:
    """Add v litres to the tank."""
    return f"added {v} L"


def script() -> ScriptedModel:
    # JSON floats (10.0, not 10) so the SDK's argument validation leaves them unchanged (see docs/integrations.md)
    return ScriptedModel([[function_call("add", {"v": 10.0}, call_id="c1")],      # certified -> runs
                          [function_call("add", {"v": 90.0}, call_id="c2")],      # rejected  -> feedback
                          [function_call("add", {"v": 50.0}, call_id="c3")],      # ask
                          [assistant_message("Done: 10 L added; 90 L rejected; 50 L handled per approval.")]])


def show(result) -> None:
    for item in result.new_items:
        raw = item.raw_item
        if getattr(raw, "type", None) == "function_call" and type(item).__name__ == "ToolCallItem":
            print(f"  model  -> {raw.name}({raw.arguments})")
        elif isinstance(raw, dict) and raw.get("type") == "function_call_output":
            print(f"  tool   <- {raw['output']}")
    if result.final_output:
        print(f"  model  : {result.final_output}")


async def part_a() -> None:
    print("\n=== A. guarded_function_tool: the gate runs inside the tool ===")
    gate = make_gate()
    tool = guarded_function_tool(gate, add, context=snapshot)   # ASK -> {"status": "needs_approval", ...}
    show(await Runner.run(Agent(name="tank", model=script(), tools=[tool]), "Bring the tank up safely.",
                          run_config=OFFLINE))


async def part_b() -> None:
    print("\n=== B. native SDK hooks: tool input guardrail + needs_approval (human in the loop) ===")
    gate = make_gate()
    tool = function_tool(add,
                         needs_approval=riskgate_needs_approval(gate, "add", context=snapshot),
                         tool_input_guardrails=[riskgate_tool_input_guardrail(gate, context=snapshot)])
    agent = Agent(name="tank", model=script(), tools=[tool])
    result = await Runner.run(agent, "Bring the tank up safely.", run_config=OFFLINE)
    while result.interruptions:
        state = result.to_state()
        for item in result.interruptions:
            print(f"  ** approval requested for {item.tool_name}({json.loads(item.raw_item.arguments)}): approving")
            state.approve(item)
        result = await Runner.run(agent, state, run_config=OFFLINE)
    show(result)


if __name__ == "__main__":
    asyncio.run(part_a())
    asyncio.run(part_b())
