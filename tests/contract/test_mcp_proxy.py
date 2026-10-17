"""Contract tests: riskgate MCP proxy x official mcp SDK (in-process, JSON-RPC framed, and real stdio)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.client._memory import InMemoryTransport  # noqa: E402
from mcp_upstream_server import build_upstream  # noqa: E402
from toy_domain import ASKED, CERTIFIED, REJECTED, StubGate  # noqa: E402

import riskgate as rg  # noqa: E402
from riskgate.adapters.mcp_proxy import DECISION_META_KEY, build_proxy_server  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
HERE = Path(__file__).resolve().parent
SERVER_INFO = "io.modelcontextprotocol/serverInfo"


def text(result: Any) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", None) == "text")


async def with_proxy(gate: Any, fn, *, framed: bool = False, **proxy_kw: Any) -> tuple[Any, list[float]]:
    """Upstream MCPServer <- Client <- proxy Server <- Client (in process); returns (fn result, executed)."""
    upstream_server, executed = build_upstream()
    async with Client(upstream_server) as upstream:
        proxy = build_proxy_server(gate, upstream, **proxy_kw)
        async with Client(InMemoryTransport(proxy) if framed else proxy) as client:
            return await fn(client), executed


@pytest.mark.parametrize("framed", [False, True], ids=["in-process", "jsonrpc-framed"])
def test_certified_forwarded_rejected_and_ask_returned_as_tool_errors(toy_gate, state, framed):
    async def scenario(client: Client) -> dict:
        names = [t.name for t in (await client.list_tools()).tools]
        return {"names": names,
                "ok": await client.call_tool("add", {"v": CERTIFIED}),
                "rejected": await client.call_tool("add", {"v": REJECTED}),
                "asked": await client.call_tool("add", {"v": ASKED})}

    r, executed = anyio.run(lambda: with_proxy(toy_gate, scenario, framed=framed, context=lambda: state))
    assert r["names"] == ["add", "read_level", "history", "vent", "broken"]       # upstream tools re-exposed
    assert executed == [CERTIFIED]                                                  # only the certified call ran
    ok = r["ok"]
    assert not ok.is_error and text(ok) == f"added {CERTIFIED}"
    assert ok.structured_content == {"result": f"added {CERTIFIED}"}               # upstream result passed through
    assert ok.meta[DECISION_META_KEY]["verdict"] == "certify"
    assert ok.meta[SERVER_INFO]["name"] == "riskgate-proxy"                        # the proxy identifies itself
    for key, status in (("rejected", "rejected"), ("asked", "needs_approval")):
        res = r[key]
        fb = json.loads(text(res))
        assert res.is_error and fb["status"] == status and fb["guarantee"] == "valid"
        assert res.meta[DECISION_META_KEY]["verdict"] == ("reject" if key == "rejected" else "ask")
        assert res.structured_content is None          # 'add' declares an output schema: no conflicting object


def test_tool_without_output_schema_gets_structured_feedback(toy_gate, state):
    async def scenario(client: Client) -> Any:
        await client.list_tools()
        return await client.call_tool("vent", {"v": REJECTED})

    res, executed = anyio.run(lambda: with_proxy(toy_gate, scenario, context=lambda: state))
    assert res.is_error and executed == []
    assert res.structured_content == json.loads(text(res)) and res.structured_content["status"] == "rejected"


def test_on_ask_async_hook_can_approve(toy_gate, state):
    seen: list[rg.Decision] = []

    async def approve(decision: rg.Decision) -> bool:
        seen.append(decision)
        return True

    async def scenario(client: Client) -> Any:
        return await client.call_tool("add", {"v": ASKED})

    res, executed = anyio.run(lambda: with_proxy(toy_gate, scenario, context=lambda: state, on_ask=approve))
    assert not res.is_error and executed == [ASKED] and seen[0].verdict is rg.Verdict.ASK


def test_async_context_passthrough_and_probe(state):
    gate = StubGate(rg.Verdict.PROBE, probe=rg.Action("read_level", read_only=True))

    async def context() -> dict:
        await anyio.sleep(0)
        return state

    async def scenario(client: Client) -> tuple[Any, Any]:
        return await client.call_tool("read_level", {}), await client.call_tool("add", {"v": CERTIFIED})

    (level, probed), executed = anyio.run(lambda: with_proxy(gate, scenario, context=context,
                                                             passthrough={"read_level"}))
    assert not level.is_error and text(level) == "50.0"
    assert [a.name for a, _ in gate.calls] == ["add"]                  # passthrough tool never reached the gate
    assert gate.calls[0][1] == state                                   # async context awaited
    fb = json.loads(text(probed))
    assert probed.is_error and fb["status"] == "probe_requested" and fb["probe"]["name"] == "read_level"
    assert executed == []


def test_context_failure_and_upstream_errors(toy_gate, state):
    def broken_context() -> dict:
        raise RuntimeError("historian offline")

    async def scenario(client: Client) -> Any:
        return await client.call_tool("add", {"v": CERTIFIED})

    res, executed = anyio.run(lambda: with_proxy(toy_gate, scenario, context=broken_context))
    fb = json.loads(text(res))
    assert res.is_error and executed == [] and fb["status"] == "needs_approval" and "fail-closed" in fb["reason"]

    async def upstream_failure(client: Client) -> Any:
        return await client.call_tool("broken", {"v": 1.0})            # certified, then fails upstream

    res, _ = anyio.run(lambda: with_proxy(toy_gate, upstream_failure, context=lambda: state))
    assert res.is_error and "valve stuck" in text(res)


def test_serve_stdio_cli_end_to_end(caplog):
    """Client -> (stdio) proxy subprocess (CLI) -> (stdio) upstream subprocess; no network.

    The proxy's gate uses an audit sink that print()s every decision: the proxy must keep that off stdout.
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "riskgate.adapters.mcp_proxy", "--gate", f"{HERE / 'toy_domain.py'}:proxy_factory",
              "--passthrough", "history", "--", sys.executable, str(HERE / "mcp_upstream_server.py")])

    async def scenario() -> dict:
        with anyio.fail_after(90):
            async with Client(params) as client:
                return {"tools": [t.name for t in (await client.list_tools()).tools],
                        "ok": await client.call_tool("add", {"v": CERTIFIED}),
                        "rejected": await client.call_tool("add", {"v": REJECTED}),
                        "history": await client.call_tool("history", {})}

    r = anyio.run(scenario)
    assert "add" in r["tools"] and not r["ok"].is_error and text(r["ok"]) == f"added {CERTIFIED}"
    assert r["rejected"].is_error and json.loads(text(r["rejected"]))["status"] == "rejected"
    assert r["history"].structured_content == {"result": [CERTIFIED]}   # the rejected call never reached upstream
    assert not [rec for rec in caplog.records if "Failed to parse JSONRPC" in rec.getMessage()]
