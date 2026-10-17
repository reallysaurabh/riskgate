"""riskgate MCP proxy, offline: an upstream MCP server, the gating proxy and a client, all in one process.

    pip install -e . mcp
    python examples/integrations/mcp_proxy_demo.py

The proxy re-exposes the upstream tools unchanged; certified calls are forwarded, everything else comes back as
``isError=True`` with the gate's JSON feedback. For real use, put the proxy between any MCP client (Claude
Desktop, an IDE, an agent framework) and any stdio MCP server:

    python -m riskgate.adapters.mcp_proxy --gate examples/integrations/_toy_gate.py:make_gate \\
        --context examples/integrations/_toy_gate.py:snapshot -- python my_mcp_server.py

(``riskgate serve-mcp ...`` from the riskgate CLI wraps the same ``serve_stdio``; see README.md.)
"""
from __future__ import annotations

import json

import anyio
from _toy_gate import make_gate, snapshot
from mcp import Client
from mcp.server.mcpserver import MCPServer

from riskgate.adapters.mcp_proxy import DECISION_META_KEY, build_proxy_server

plant = MCPServer("toy-plant")          # stands in for any upstream MCP server


@plant.tool()
def add(v: float) -> str:
    """Add v litres to the tank."""
    return f"added {v} L"


@plant.tool()
def read_level() -> float:
    """Read the tank level in litres (read-only)."""
    return snapshot()["level"]


async def main() -> None:
    gate = make_gate()
    async with Client(plant) as upstream:                     # the proxy is a client of the upstream server ...
        proxy = build_proxy_server(gate, upstream, context=snapshot, passthrough={"read_level"})
        async with Client(proxy) as client:                   # ... and a server for the agent's client
            tools = await client.list_tools()
            print("\n=== MCP proxy: tools re-exposed:", [t.name for t in tools.tools], "===")
            for name, args in (("read_level", {}), ("add", {"v": 10}), ("add", {"v": 90}), ("add", {"v": 50})):
                result = await client.call_tool(name, args)
                body = "".join(c.text for c in result.content if c.type == "text")
                verdict = (result.meta or {}).get(DECISION_META_KEY, {}).get("verdict", "passthrough")
                print(f"  {name}({json.dumps(args)}): isError={result.is_error} verdict={verdict} -> {body}")


if __name__ == "__main__":
    anyio.run(main)
