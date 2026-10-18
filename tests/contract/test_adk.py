"""Contract tests: riskgate x Google ADK (real google-adk Runner + FunctionTool, scripted LLM, no network)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models import BaseLlm, LlmRequest, LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402
from toy_domain import ASKED, CERTIFIED, REJECTED  # noqa: E402

from riskgate.adapters.frameworks import adk_before_tool_callback, adk_plugin  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.filterwarnings("ignore::UserWarning"),
              pytest.mark.filterwarnings("ignore::FutureWarning")]

CONFIRM = "adk_request_confirmation"


class ScriptedLlm(BaseLlm):
    """A BaseLlm that replays a script: dict -> one function call, str -> final text."""

    script: list = []

    async def generate_content_async(self, llm_request: LlmRequest,
                                     stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        step = self.script.pop(0)
        part = (types.Part.from_function_call(name=step["name"], args=step["args"]) if isinstance(step, dict)
                else types.Part.from_text(text=step))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class Tank:
    def __init__(self) -> None:
        self.executed: list[float] = []

    def tool(self) -> FunctionTool:
        def add(v: float) -> dict:
            """Add v litres to the tank."""
            self.executed.append(v)
            return {"status": "ok", "added": v}

        return FunctionTool(add)


def add_call(v: float) -> dict:
    return {"name": "add", "args": {"v": v}}


async def run_turn(runner: InMemoryRunner, session_id: str, message: types.Content) -> list[tuple[str, Any]]:
    """Run one turn; return ("call", FunctionCall) / ("response", FunctionResponse) / ("text", str) items."""
    out: list[tuple[str, Any]] = []
    async for event in runner.run_async(user_id="u", session_id=session_id, new_message=message):
        for p in (event.content.parts if event.content else []) or []:
            if p.function_call:
                out.append(("call", p.function_call))
            if p.function_response:
                out.append(("response", p.function_response))
            if p.text:
                out.append(("text", p.text))
    return out


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


async def new_session(agent: LlmAgent, **runner_kw: Any) -> tuple[InMemoryRunner, str]:
    runner = InMemoryRunner(agent=agent, app_name="riskgate-test", **runner_kw)
    session = await runner.session_service.create_session(app_name="riskgate-test", user_id="u")
    return runner, session.id


def responses(items: list[tuple[str, Any]], name: str = "add") -> list[dict]:
    return [r.response for kind, r in items if kind == "response" and r.name == name]


def test_callback_is_accepted_by_llm_agent_and_follows_adk_convention(toy_gate, state):
    cb = adk_before_tool_callback(toy_gate, context=lambda tool_context: state)
    tank = Tank()
    tool = tank.tool()
    agent = LlmAgent(name="tank_agent", model=ScriptedLlm(model="scripted"), tools=[tool], before_tool_callback=cb)
    assert agent.canonical_before_tool_callbacks == [cb]
    # ADK invokes canonical before-tool callbacks with keyword arguments tool=, args=, tool_context=
    assert cb(tool=tool, args={"v": CERTIFIED}, tool_context=None) is None
    blocked = cb(tool=tool, args={"v": REJECTED}, tool_context=None)
    assert isinstance(blocked, dict) and blocked["status"] == "rejected" and blocked["guarantee"] == "valid"
    assert tank.executed == []                                       # the callback never runs the tool


def test_runner_runs_certified_and_answers_rejected_with_feedback(toy_gate, state):
    async def scenario() -> tuple[Tank, list]:
        tank = Tank()
        llm = ScriptedLlm(model="scripted", script=[add_call(CERTIFIED), add_call(REJECTED), "done"])
        agent = LlmAgent(name="tank_agent", model=llm, tools=[tank.tool()],
                         before_tool_callback=adk_before_tool_callback(toy_gate, context=lambda tc: state))
        runner, sid = await new_session(agent)
        return tank, await run_turn(runner, sid, user("fill the tank"))

    tank, items = asyncio.run(scenario())
    first, second = responses(items)
    assert tank.executed == [CERTIFIED]
    assert first == {"status": "ok", "added": CERTIFIED}
    assert second["status"] == "rejected" and "unsafe" in second["reason"]
    assert items[-1] == ("text", "done")


@pytest.mark.parametrize("confirmed", [True, False])
def test_ask_pauses_with_adk_tool_confirmation_then_resumes(toy_gate, state, confirmed):
    async def scenario() -> tuple[Tank, list, list]:
        tank = Tank()
        llm = ScriptedLlm(model="scripted", script=[add_call(ASKED), "done"])
        agent = LlmAgent(name="tank_agent", model=llm, tools=[tank.tool()],
                         before_tool_callback=adk_before_tool_callback(toy_gate, context=lambda tc: state))
        runner, sid = await new_session(agent)
        turn1 = await run_turn(runner, sid, user("add 50 L"))
        (request,) = [c for kind, c in turn1 if kind == "call" and c.name == CONFIRM]
        reply = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=request.id, name=CONFIRM, response={"confirmed": confirmed}))])
        turn2 = await run_turn(runner, sid, reply)
        return tank, turn1, turn2

    tank, turn1, turn2 = asyncio.run(scenario())
    (paused,) = responses(turn1)
    assert paused["status"] == "needs_approval" and "operator approval" in paused["reason"]
    (request,) = [c for kind, c in turn1 if kind == "call" and c.name == CONFIRM]
    assert request.args["originalFunctionCall"]["args"] == {"v": ASKED}
    assert request.args["toolConfirmation"]["payload"]["riskgate"]["status"] == "needs_approval"
    assert not any(kind == "text" for kind, _ in turn1)               # the model was not re-invoked
    (final,) = responses(turn2)
    if confirmed:
        assert tank.executed == [ASKED] and final == {"status": "ok", "added": ASKED}
    else:
        assert tank.executed == [] and final["status"] == "declined" and "declined" in final["reason"]
    assert turn2[-1] == ("text", "done")


def test_on_ask_decline_reports_declined(toy_gate, state):
    async def scenario() -> tuple[Tank, list]:
        tank = Tank()
        llm = ScriptedLlm(model="scripted", script=[add_call(ASKED), "done"])
        cb = adk_before_tool_callback(toy_gate, context=lambda: state, on_ask=lambda d: False)
        agent = LlmAgent(name="tank_agent", model=llm, tools=[tank.tool()], before_tool_callback=cb)
        runner, sid = await new_session(agent)
        return tank, await run_turn(runner, sid, user("add 50 L"))

    tank, items = asyncio.run(scenario())
    (fb,) = responses(items)
    assert tank.executed == [] and fb["status"] == "declined" and fb["executed"] is False


def test_on_ask_hook_takes_precedence_over_confirmation(toy_gate, state):
    async def scenario() -> tuple[Tank, list, list]:
        tank, seen = Tank(), []
        llm = ScriptedLlm(model="scripted", script=[add_call(ASKED), "done"])
        cb = adk_before_tool_callback(toy_gate, context=lambda: state, on_ask=lambda d: seen.append(d) or True)
        agent = LlmAgent(name="tank_agent", model=llm, tools=[tank.tool()], before_tool_callback=cb)
        runner, sid = await new_session(agent)
        return tank, seen, await run_turn(runner, sid, user("add 50 L"))

    tank, seen, items = asyncio.run(scenario())
    assert tank.executed == [ASKED] and len(seen) == 1 and seen[0].verdict.value == "ask"
    assert not any(kind == "call" and c.name == CONFIRM for kind, c in items)


def test_plugin_gates_tools_of_every_agent(toy_gate, state):
    async def scenario() -> tuple[Tank, list]:
        tank = Tank()
        llm = ScriptedLlm(model="scripted", script=[add_call(REJECTED), add_call(CERTIFIED), "done"])
        agent = LlmAgent(name="tank_agent", model=llm, tools=[tank.tool()])        # no agent-level callback
        runner, sid = await new_session(agent, plugins=[adk_plugin(toy_gate, context=lambda tc: state)])
        return tank, await run_turn(runner, sid, user("go"))

    tank, items = asyncio.run(scenario())
    rejected, ok = responses(items)
    assert rejected["status"] == "rejected" and ok == {"status": "ok", "added": CERTIFIED}
    assert tank.executed == [CERTIFIED]


def test_probe_feedback_and_passthrough(probe_gate, state):
    tool = Tank().tool()
    cb = adk_before_tool_callback(probe_gate, context=lambda tc: state)
    fb = cb(tool=tool, args={"v": CERTIFIED}, tool_context=None)
    assert fb["status"] == "probe_requested" and fb["probe"] == {"name": "read_flowmeter", "args": {},
                                                                 "read_only": True}
    open_cb = adk_before_tool_callback(probe_gate, context=lambda tc: state, passthrough={"add"})
    calls_before = len(probe_gate.calls)
    assert open_cb(tool=tool, args={"v": REJECTED}, tool_context=None) is None
    assert len(probe_gate.calls) == calls_before                     # passthrough tools skip the gate entirely


def test_context_failure_fails_closed(toy_gate):
    def broken(tool_context):
        raise RuntimeError("no snapshot")

    fb = adk_before_tool_callback(toy_gate, context=broken)(tool=Tank().tool(), args={"v": CERTIFIED},
                                                            tool_context=None)
    assert fb["status"] == "needs_approval" and "fail-closed" in fb["reason"]
