"""Upstream MCP server used by the MCP proxy contract tests (in process, or as a stdio subprocess).

Run standalone: ``python mcp_upstream_server.py`` serves over stdio.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def build_upstream() -> tuple[MCPServer, list[float]]:
    """A tiny plant server: ``add`` changes the world, ``read_level``/``history`` only read."""
    executed: list[float] = []
    server = MCPServer("toy-plant")

    @server.tool()
    def add(v: float) -> str:
        """Add v litres to the tank."""
        executed.append(v)
        return f"added {v}"

    @server.tool()
    def read_level() -> float:
        """Read the current tank level (litres)."""
        return 50.0

    @server.tool()
    def history() -> list[float]:
        """Volumes added so far (lets out-of-process tests see what really executed)."""
        return list(executed)

    @server.tool(structured_output=False)
    def vent(v: float) -> str:
        """Vent v litres (declares no output schema)."""
        executed.append(-v)
        return f"vented {v}"

    @server.tool()
    def broken(v: float) -> str:
        """Always fails upstream (a reported tool error; mcp 2.x hides details of unexpected exceptions)."""
        raise ToolError("valve stuck")

    return server, executed


if __name__ == "__main__":
    build_upstream()[0].run("stdio")
