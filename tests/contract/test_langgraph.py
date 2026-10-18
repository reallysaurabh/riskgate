"""Contract tests: riskgate x LangGraph (real langgraph / langchain-core, fake chat model, no network)."""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, create_react_agent  # noqa: E402
from langgraph.types import Command, Send  # noqa: E402
from toy_domain import ASKED, CERTIFIED, REJECTED  # noqa: E402

from riskgate.adapters.frameworks import gated_tool_node, langgraph_gate_node, route_after_gate  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model that also accepts ``bind_tools`` (as ``create_react_agent`` requires)."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


def scripted(*steps: AIMessage | str) -> FakeToolModel:
    return FakeToolModel(messages=iter(steps))


def call(v: float, call_id: str) -> dict:
    return {"id": call_id, "name": "add", "args": {"v": v}}


def build_graph(gate, state, model, executed, *, interrupt=None, checkpointer=None):
    def add(v: float) -> str:
        """Add v litres to the tank."""
        executed.append(v)
        return f"added {v}"

    g = StateGraph(MessagesState)
    g.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]})
    g.add_node("gate", langgraph_gate_node(gate, context=lambda s: state, interrupt=interrupt))
    g.add_node("tools", ToolNode([add]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: "gate" if s["messages"][-1].tool_calls else END, ["gate", END])
    g.add_conditional_edges("gate", route_after_gate, ["tools", "agent"])
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def tool_messages(messages) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}


def test_gate_node_runs_certified_and_answers_rejected_with_tool_messages(toy_gate, state):
    executed: list[float] = []
    model = scripted(AIMessage("", tool_calls=[call(CERTIFIED, "c1")]),
                     AIMessage("", tool_calls=[call(REJECTED, "c2")]),
                     AIMessage("done"))
    out = build_graph(toy_gate, state, model, executed).invoke({"messages": [HumanMessage("fill the tank")]})
    replies = tool_messages(out["messages"])
    assert executed == [CERTIFIED]
    assert replies["c1"].content == f"added {CERTIFIED}" and replies["c1"].status == "success"
    rejected = replies["c2"]
    assert rejected.status == "error" and rejected.name == "add"
    feedback = json.loads(rejected.content)
    assert feedback["status"] == "rejected" and "unsafe" in feedback["reason"] and feedback["guarantee"] == "valid"
    assert rejected.artifact["riskgate"]["verdict"] == "reject"
    assert out["messages"][-1].content == "done"


def test_mixed_batch_dispatches_only_certified_calls_via_send(toy_gate, state):
    executed: list[float] = []
    model = scripted(AIMessage("", tool_calls=[call(CERTIFIED, "c1"), call(REJECTED, "c2")]), AIMessage("done"))
    out = build_graph(toy_gate, state, model, executed).invoke({"messages": [HumanMessage("go")]})
    replies = tool_messages(out["messages"])
    assert executed == [CERTIFIED]                                  # the rejected call never ran
    assert set(replies) == {"c1", "c2"}                             # every tool call answered exactly once
    assert sum(isinstance(m, ToolMessage) for m in out["messages"]) == 2
    assert json.loads(replies["c2"].content)["status"] == "rejected"


def test_route_after_gate_on_messages_state():
    ai = AIMessage("", tool_calls=[call(CERTIFIED, "a"), call(REJECTED, "b")])
    reply = ToolMessage(content="{}", tool_call_id="b", name="add", status="error")
    assert route_after_gate({"messages": [ai]}) == "tools"
    assert route_after_gate({"messages": [ai, reply,
                                          ToolMessage(content="{}", tool_call_id="a", name="add")]}) == "agent"
    sends = route_after_gate({"messages": [ai, reply]}, tools_node="act")
    assert isinstance(sends, list) and len(sends) == 1 and isinstance(sends[0], Send)
    assert sends[0].node == "act" and [c["id"] for c in sends[0].arg] == ["a"]
    assert sends[0].arg[0]["type"] == "tool_call"


@pytest.mark.parametrize("answer, runs", [({"approve": True}, True), ({"approve": False}, False)])
def test_interrupt_escalates_ask_and_resume_decides(toy_gate, state, answer, runs):
    executed: list[float] = []
    model = scripted(AIMessage("", tool_calls=[call(ASKED, "c3")]), AIMessage("done"))
    app = build_graph(toy_gate, state, model, executed, interrupt=True, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    first = app.invoke({"messages": [HumanMessage("add 50 L")]}, config)
    pending = first["__interrupt__"]
    assert len(pending) == 1 and pending[0].value["type"] == "riskgate_approval"
    assert pending[0].value["tool_call"] == {"id": "c3", "name": "add", "args": {"v": ASKED}}
    assert "operator approval" in pending[0].value["reason"] and executed == []
    out = app.invoke(Command(resume=answer), config)
    replies = tool_messages(out["messages"])
    if runs:
        assert executed == [ASKED] and replies["c3"].content == f"added {ASKED}"
    else:
        assert executed == [] and json.loads(replies["c3"].content)["status"] == "declined"
    assert out["messages"][-1].content == "done"


def test_two_asks_in_one_batch_are_reviewed_one_at_a_time(toy_gate, state):
    executed: list[float] = []
    model = scripted(AIMessage("", tool_calls=[call(45.0, "a"), call(55.0, "b")]), AIMessage("done"))
    app = build_graph(toy_gate, state, model, executed, interrupt=True, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    out = app.invoke({"messages": [HumanMessage("add 45 L and 55 L")]}, config)
    reviewed = []
    for answer in ({"approve": True}, "no"):                       # approve "a", decline "b"
        (pending,) = out["__interrupt__"]
        reviewed.append(pending.value["tool_call"]["id"])
        out = app.invoke(Command(resume=answer), config)
    assert reviewed == ["a", "b"] and "__interrupt__" not in out
    replies = tool_messages(out["messages"])
    assert executed == [45.0] and replies["a"].content == "added 45.0"
    assert json.loads(replies["b"].content)["status"] == "declined"


def test_gated_tool_node_inside_create_react_agent(toy_gate, state):
    executed: list[float] = []

    def add(v: float) -> str:
        """Add v litres to the tank."""
        executed.append(v)
        return f"added {v}"

    model = scripted(AIMessage("", tool_calls=[call(CERTIFIED, "c1"), call(REJECTED, "c2")]), AIMessage("done"))
    agent = create_react_agent(model, tools=gated_tool_node(toy_gate, [add], context=lambda s: state))
    out = agent.invoke({"messages": [HumanMessage("go")]})
    replies = tool_messages(out["messages"])
    assert executed == [CERTIFIED]
    assert replies["c1"].content == f"added {CERTIFIED}"
    assert replies["c2"].status == "error" and json.loads(replies["c2"].content)["status"] == "rejected"
    assert out["messages"][-1].content == "done"


def test_gated_tool_node_interrupt_in_create_react_agent(toy_gate, state):
    """create_react_agent v2 sends each call to its own ToolNode task, so wrapper interrupts resume per call."""
    executed: list[float] = []

    def add(v: float) -> str:
        """Add v litres to the tank."""
        executed.append(v)
        return f"added {v}"

    model = scripted(AIMessage("", tool_calls=[call(CERTIFIED, "c1"), call(ASKED, "c2")]), AIMessage("done"))
    agent = create_react_agent(model, tools=gated_tool_node(toy_gate, [add], context=lambda: state, interrupt=True),
                               checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    first = agent.invoke({"messages": [HumanMessage("go")]}, config)
    (pending,) = first["__interrupt__"]
    assert pending.value["tool_call"]["id"] == "c2" and executed == [CERTIFIED]
    out = agent.invoke(Command(resume=True), config)
    replies = tool_messages(out["messages"])
    assert sorted(executed) == [CERTIFIED, ASKED] and replies["c2"].content == f"added {ASKED}"
    assert out["messages"][-1].content == "done"


def test_gated_tool_node_async_path_and_ask_hook(toy_gate, state):
    executed: list[float] = []

    async def add(v: float) -> str:
        """Add v litres to the tank."""
        executed.append(v)
        return f"added {v}"

    approvals = []
    node = gated_tool_node(toy_gate, [add], context=lambda: state,
                           on_ask=lambda d: approvals.append(d.reason) or True)
    g = StateGraph(MessagesState)
    model = scripted(AIMessage("", tool_calls=[call(ASKED, "c1"), call(REJECTED, "c2")]), AIMessage("done"))

    async def agent_node(s):
        return {"messages": [await model.ainvoke(s["messages"])]}

    g.add_node("agent", agent_node)
    g.add_node("tools", node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
    g.add_edge("tools", "agent")
    out = asyncio.run(g.compile().ainvoke({"messages": [HumanMessage("go")]}))
    replies = tool_messages(out["messages"])
    assert executed == [ASKED] and len(approvals) == 1                 # ASK approved by the hook, async tool ran
    assert json.loads(replies["c2"].content)["status"] == "rejected"


def test_probe_verdict_feedback_names_the_probe(probe_gate, state):
    node = langgraph_gate_node(probe_gate, context=lambda s: state)
    upd = node({"messages": [AIMessage("", tool_calls=[call(CERTIFIED, "p1")])]})
    (reply,) = upd["messages"]
    assert isinstance(reply, ToolMessage) and reply.status == "error"
    fb = json.loads(reply.content)
    assert fb["status"] == "probe_requested" and fb["probe"]["name"] == "read_flowmeter" and fb["probe"]["read_only"]
    assert route_after_gate({"messages": [AIMessage("", tool_calls=[call(1.0, "p1")]), reply]}) == "agent"


def test_dict_messages_and_openai_style_calls(toy_gate, state):
    node = langgraph_gate_node(toy_gate, context=lambda s: state)
    openai_call = {"id": "o1", "type": "function", "function": {"name": "add", "arguments": json.dumps({"v": 90})}}
    upd = node({"messages": [{"role": "assistant", "content": "", "tool_calls": [openai_call]}]})
    assert upd["messages"][0]["role"] == "tool" and upd["messages"][0]["tool_call_id"] == "o1"
    assert json.loads(upd["messages"][0]["content"])["status"] == "rejected"
    assert route_after_gate(upd) == "agent"


def test_context_failure_fails_closed(toy_gate):
    executed: list[float] = []

    def broken(_state):
        raise RuntimeError("plant historian unreachable")

    model = scripted(AIMessage("", tool_calls=[call(CERTIFIED, "c1")]), AIMessage("done"))

    def add(v: float) -> str:
        """Add v litres to the tank."""
        executed.append(v)
        return "added"

    g = StateGraph(MessagesState)
    g.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]})
    g.add_node("gate", langgraph_gate_node(toy_gate, context=broken))
    g.add_node("tools", ToolNode([add]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: "gate" if s["messages"][-1].tool_calls else END, ["gate", END])
    g.add_conditional_edges("gate", route_after_gate, ["tools", "agent"])
    g.add_edge("tools", "agent")
    out = g.compile().invoke({"messages": [HumanMessage("go")]})
    fb = json.loads(tool_messages(out["messages"])["c1"].content)
    assert executed == [] and fb["status"] == "needs_approval" and "fail-closed" in fb["reason"]
