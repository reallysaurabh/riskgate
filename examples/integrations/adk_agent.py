"""riskgate x Google ADK, offline: a scripted LLM drives a real ADK Runner; the gate is a before_tool_callback.

    pip install -e . google-adk
    python examples/integrations/adk_agent.py

ASK uses ADK's native tool confirmation: the run pauses with an ``adk_request_confirmation`` call and resumes
when the user answers ``{"confirmed": true}``. For a real model use ``LlmAgent(model="gemini-2.5-flash", ...)``
with ``GOOGLE_API_KEY`` (or Vertex AI settings) instead of ``ScriptedLlm`` (see README.md).
"""
from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import AsyncGenerator

from _toy_gate import make_gate, snapshot
from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from riskgate.adapters.frameworks import adk_before_tool_callback

warnings.filterwarnings("ignore")   # ADK experimental-feature notices
logging.getLogger("google_adk.google.adk.telemetry").setLevel(logging.ERROR)   # scripted LLM reports no usage


class ScriptedLlm(BaseLlm):
    """Replays a script: a dict becomes a function call, a string the final answer."""

    script: list = []

    async def generate_content_async(self, llm_request: LlmRequest,
                                     stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        step = self.script.pop(0)
        part = (types.Part.from_function_call(name=step["name"], args=step["args"]) if isinstance(step, dict)
                else types.Part.from_text(text=step))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def add(v: float) -> dict:
    """Add v litres to the tank."""
    return {"status": "ok", "added_litres": v}


async def turn(runner: InMemoryRunner, session_id: str, message: types.Content) -> types.FunctionCall | None:
    """Run one turn, print it, and return a pending confirmation request (if any)."""
    pending = None
    async for event in runner.run_async(user_id="operator", session_id=session_id, new_message=message):
        for p in (event.content.parts if event.content else []) or []:
            if p.function_call and p.function_call.name == "adk_request_confirmation":
                pending = p.function_call
                print(f"  ** confirmation requested: {p.function_call.args['toolConfirmation']['hint']}")
            elif p.function_call:
                print(f"  model  -> {p.function_call.name}({p.function_call.args})")
            elif p.function_response:
                print(f"  tool   <- {p.function_response.response}")
            elif p.text:
                print(f"  model  : {p.text}")
    return pending


async def main() -> None:
    gate = make_gate()
    llm = ScriptedLlm(model="scripted", script=[
        {"name": "add", "args": {"v": 10}},      # certified -> runs
        {"name": "add", "args": {"v": 90}},      # rejected  -> structured feedback, model re-plans
        {"name": "add", "args": {"v": 50}},      # ask       -> ADK tool confirmation
        "Added 10 L and, with your approval, 50 L. 90 L would have overflowed the tank."])
    agent = LlmAgent(name="tank_operator", model=llm, instruction="Operate the tank safely.",
                     tools=[FunctionTool(add)],
                     before_tool_callback=adk_before_tool_callback(gate, context=lambda tool_context: snapshot()))
    runner = InMemoryRunner(agent=agent, app_name="riskgate-demo")
    session = await runner.session_service.create_session(app_name="riskgate-demo", user_id="operator")

    print("\n=== Google ADK: before_tool_callback gate ===")
    request = await turn(runner, session.id, types.Content(role="user", parts=[types.Part.from_text(
        text="Bring the tank up safely.")]))
    if request is not None:
        print("  ** operator confirms")
        reply = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=request.id, name="adk_request_confirmation", response={"confirmed": True}))])
        await turn(runner, session.id, reply)


if __name__ == "__main__":
    asyncio.run(main())
