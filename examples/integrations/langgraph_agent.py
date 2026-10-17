"""riskgate x LangGraph, offline: a scripted chat model proposes tool calls; the gate decides.

    pip install -e . langgraph langchain-core
    python examples/integrations/langgraph_agent.py

Part A builds the graph by hand (model -> riskgate gate node -> ToolNode) with human review of ASK via
``interrupt()``; part B drops ``gated_tool_node`` into the prebuilt ``create_react_agent``. To use a real model,
replace ``scripted(...)`` with e.g. ``init_chat_model("openai:<model>").bind_tools([add])`` (see README.md).
"""
from __future__ import annotations

import json
import warnings

from _toy_gate import make_gate, snapshot
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from riskgate.adapters.frameworks import gated_tool_node, langgraph_gate_node, route_after_gate

warnings.filterwarnings("ignore", category=DeprecationWarning)


def add(v: float) -> str:
    """Add v litres to the tank."""
    return f"added {v} L"


class ScriptedModel(GenericFakeChatModel):
    """Stands in for a tool-calling chat model; ``bind_tools`` is a no-op."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


def scripted(*steps: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(steps))


def tool_call(v: float, call_id: str) -> dict:
    return {"id": call_id, "name": "add", "args": {"v": v}}


def show(messages: list[BaseMessage]) -> None:
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            print("  model  ->", ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls))
        elif isinstance(m, ToolMessage):
            print(f"  tool   <- [{m.status}] {m.content}")
        elif isinstance(m, AIMessage):
            print("  model  :", m.content)


def part_a() -> None:
    print("\n=== A. StateGraph: model -> gate node -> ToolNode, ASK escalated with interrupt() ===")
    gate = make_gate()
    model = scripted(
        AIMessage("", tool_calls=[tool_call(10, "c1")]),                        # certified -> runs
        AIMessage("", tool_calls=[tool_call(90, "c2")]),                        # rejected  -> feedback
        AIMessage("", tool_calls=[tool_call(50, "c3")]),                        # ask       -> human review
        AIMessage("Added 10 L, then 50 L with operator approval; 90 L would overflow."))
    g = StateGraph(MessagesState)
    g.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]})
    g.add_node("gate", langgraph_gate_node(gate, context=lambda state: snapshot(), interrupt=True))
    g.add_node("tools", ToolNode([add]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: "gate" if s["messages"][-1].tool_calls else END, ["gate", END])
    g.add_conditional_edges("gate", route_after_gate, ["tools", "agent"])
    g.add_edge("tools", "agent")
    app = g.compile(checkpointer=InMemorySaver())       # interrupt() needs a checkpointer

    config = {"configurable": {"thread_id": "tank-1"}}
    out = app.invoke({"messages": [HumanMessage("Bring the tank up safely.")]}, config)
    for pending in out.get("__interrupt__", []):
        print(f"  ** paused for operator: {pending.value['tool_call']} because {pending.value['reason']}")
        print("  ** operator approves")
        out = app.invoke(Command(resume={"approve": True}), config)
    show(out["messages"])


def part_b() -> None:
    print("\n=== B. create_react_agent(tools=gated_tool_node(...)): one batch, mixed verdicts ===")
    from langgraph.prebuilt import create_react_agent  # deprecated in langgraph 1.x but still supported

    gate = make_gate()
    model = scripted(AIMessage("", tool_calls=[tool_call(10, "d1"), tool_call(90, "d2")]),
                     AIMessage("Added 10 L; the 90 L request was rejected by the gate."))
    agent = create_react_agent(model, tools=gated_tool_node(gate, [add], context=lambda state: snapshot()))
    show(agent.invoke({"messages": [HumanMessage("Add 10 L and 90 L.")]})["messages"])


if __name__ == "__main__":
    part_a()
    part_b()
