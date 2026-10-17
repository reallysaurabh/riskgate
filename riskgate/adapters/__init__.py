"""Framework adapters for riskgate (all optional: no framework is imported until its adapter is used).

==================================================  ===========================================  ===============
name                                                module                                       framework
==================================================  ===========================================  ===============
``guard``, ``ActionRejected``, ``ApprovalRequired``  ``riskgate.adapters.tool``                   any (decorator)
``adk_before_tool_callback``, ``adk_plugin``         ``riskgate.adapters.frameworks``             Google ADK
``langgraph_gate_node``, ``route_after_gate``,       ``riskgate.adapters.frameworks``             LangGraph
``gated_tool_node``, ``langgraph_tool_wrapper``,
``langgraph_async_tool_wrapper``
``guarded_function_tool``,                           ``riskgate.adapters.openai_agents``          OpenAI Agents
``riskgate_tool_input_guardrail``,
``riskgate_needs_approval``
``build_proxy_server``, ``serve_stdio``              ``riskgate.adapters.mcp_proxy``              MCP
==================================================  ===========================================  ===============

Names are resolved lazily (PEP 562), so ``from riskgate.adapters import gated_tool_node`` imports only the
module that defines it. See ``docs/integrations.md`` for verified versions and usage.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "guard": "tool", "ActionRejected": "tool", "ApprovalRequired": "tool",
    "adk_before_tool_callback": "frameworks", "adk_plugin": "frameworks",
    "langgraph_gate_node": "frameworks", "route_after_gate": "frameworks", "gated_tool_node": "frameworks",
    "langgraph_tool_wrapper": "frameworks", "langgraph_async_tool_wrapper": "frameworks",
    "guarded_function_tool": "openai_agents", "riskgate_tool_input_guardrail": "openai_agents",
    "riskgate_needs_approval": "openai_agents",
    "build_proxy_server": "mcp_proxy", "serve_stdio": "mcp_proxy",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
