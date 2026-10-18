"""MCP proxy: put a riskgate Gate in front of any MCP server (official ``mcp`` SDK; imported lazily).

Verified against mcp 2.3.0 (``tests/contract/test_mcp_proxy.py``). The proxy is a low-level
``mcp.server.lowlevel.Server`` that connects as a client to an upstream server, re-exposes its tools
unchanged, and intercepts ``tools/call``:

* CERTIFY (or an ASK approved by ``on_ask``) -> the call is forwarded upstream and the upstream
  ``CallToolResult`` is returned unchanged except for ``_meta``: the decision summary is added under
  ``"riskgate/decision"`` and the upstream's ``"io.modelcontextprotocol/serverInfo"`` stamp is dropped;
* REJECT / PROBE / ASK -> ``CallToolResult(isError=True)`` whose text content is the JSON of the gate's
  feedback (status, reason, guarantee[, probe]); ``structuredContent`` carries the same object when the tool
  declares no output schema, and ``_meta["riskgate/decision"]`` a JSON-safe summary of the decision.

Real use (stdio on both sides)::

    python -m riskgate.adapters.mcp_proxy --gate my_gate.py:make_gate -- npx -y @acme/plant-mcp-server

or from Python: ``asyncio.run(serve_stdio(gate, "npx", ["-y", "@acme/plant-mcp-server"], context=snapshot))``.
In process (tests, notebooks): ``build_proxy_server(gate, upstream)`` where ``upstream`` is an entered
``mcp.Client`` or ``mcp.ClientSession``.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Awaitable, Callable, Collection, Sequence
from pathlib import Path
from typing import Any

from ..types import Action, Decision, Verdict
from .frameworks import _safe
from .tool import _feedback

__all__ = ["build_proxy_server", "main", "serve_stdio"]

DECISION_META_KEY = "riskgate/decision"
SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"   # per-response identity stamp (2026-07-28 protocol)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _list_all_tools(upstream: Any, max_pages: int = 100) -> list:
    """All upstream tools, following pagination for both ``mcp.Client`` and ``mcp.ClientSession``."""
    tools: list = []
    page = await upstream.list_tools()
    for _ in range(max_pages):
        tools.extend(page.tools)
        cursor = getattr(page, "next_cursor", None)
        if not cursor:
            break
        params = inspect.signature(upstream.list_tools).parameters
        if "cursor" in params:                                   # mcp.Client
            page = await upstream.list_tools(cursor=cursor)
        else:                                                    # mcp.ClientSession
            import mcp.types as types

            page = await upstream.list_tools(params=types.PaginatedRequestParams(cursor=cursor))
    return tools


def _decision_summary(d: Decision, approved_by: str | None = None) -> dict:
    out = {"verdict": d.verdict.value, "reason": d.reason, "guarantee": d.status.value, "score": d.score,
           "threshold": d.threshold, "calibration_id": d.calibration_id, "stage": d.stage,
           "probe": d.probe.to_dict() if d.probe is not None else None}
    if approved_by is not None:
        out["approved_by"] = approved_by
    return _safe(out)


def build_proxy_server(gate: Any, upstream_session: Any, *,
                       context: Callable[[], Any | Awaitable[Any]] = lambda: None, name: str = "riskgate-proxy",
                       on_ask: Callable[[Decision], bool | Awaitable[bool]] | None = None,
                       passthrough: Collection[str] = (), version: str = "",
                       instructions: str | None = None) -> Any:
    """Return a low-level ``mcp.server.lowlevel.Server`` that gates ``tools/call`` before forwarding upstream.

    Args:
        gate: a calibrated ``riskgate.Gate`` (anything with ``check(action, context)``).
        upstream_session: an entered ``mcp.Client`` or ``mcp.ClientSession`` connected to the upstream server
            (anything with async ``list_tools()`` and ``call_tool(name, arguments)``).
        context: zero-argument callable (sync or async) returning the gate context for each call; e.g. read a
            state snapshot, possibly by calling a read-only upstream tool.
        name: server name advertised to downstream clients.
        on_ask: optional approval hook (sync or async) for ASK; True forwards the call.
        passthrough: tool names forwarded WITHOUT a gate check (explicit allow-list for harmless read-only
            tools). Everything else is gated; failures to build the context fail closed.
        version, instructions: advertised server metadata.

    Each call is checked as ``gate.check(Action(tool_name, arguments), context())`` at stage 1.
    """
    import mcp.types as types
    from mcp.server.lowlevel import Server

    output_schemas: dict[str, Any] = {}
    allow = frozenset(passthrough)

    def blocked(d: Decision, tool_name: str) -> Any:
        fb = _safe(_feedback(d))
        has_schema = output_schemas.get(tool_name) is not None
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(fb))], is_error=True,
                                    structured_content=None if has_schema else fb,
                                    meta={DECISION_META_KEY: _decision_summary(d)})  # type: ignore[call-arg]  # pydantic alias of _meta

    async def forward(tool_name: str, arguments: dict, d: Decision | None, approved_by: str | None = None) -> Any:
        try:
            result = await upstream_session.call_tool(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001 - report upstream failures as tool errors
            return types.CallToolResult(content=[types.TextContent(
                type="text", text=f"upstream error: {type(exc).__name__}: {exc}")], is_error=True)
        if not isinstance(result, types.CallToolResult):
            return types.CallToolResult(content=[types.TextContent(
                type="text", text=f"unsupported upstream result: {type(result).__name__}")], is_error=True)
        meta = dict(result.meta or {})
        meta.pop(SERVER_INFO_META_KEY, None)       # the proxy stamps its own identity
        if d is not None:
            meta[DECISION_META_KEY] = _decision_summary(d, approved_by=approved_by)
        return result.model_copy(update={"meta": meta or None})

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        tools = await _list_all_tools(upstream_session)
        output_schemas.clear()
        output_schemas.update({t.name: t.output_schema for t in tools})
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        tool_name, arguments = params.name, dict(params.arguments or {})
        if tool_name in allow:
            return await forward(tool_name, arguments, None)
        action = Action(tool_name, arguments)
        try:
            gate_context = await _maybe_await(context())
        except Exception as exc:  # noqa: BLE001 - rule R3: fail closed
            return blocked(Decision(Verdict.ASK, action, reason=f"fail-closed: could not build the gate context: "
                                                                f"{type(exc).__name__}: {exc}"), tool_name)
        d = gate.check(action, gate_context)
        if d.verdict is Verdict.CERTIFY:
            return await forward(tool_name, arguments, d)
        if d.verdict is Verdict.ASK and on_ask is not None and await _maybe_await(on_ask(d)):
            return await forward(tool_name, arguments, d, approved_by="human")
        return blocked(d, tool_name)

    return Server(name, version=version, instructions=instructions,
                  on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve_stdio(gate: Any, upstream_command: str, upstream_args: Sequence[str] = (), *,
                      context: Callable[[], Any | Awaitable[Any]] = lambda: None,
                      env: dict[str, str] | None = None, cwd: str | Path | None = None,
                      name: str = "riskgate-proxy", on_ask: Callable[[Decision], bool | Awaitable[bool]] | None = None,
                      passthrough: Collection[str] = ()) -> None:
    """Serve the proxy on this process's stdin/stdout, launching the upstream server as a stdio subprocess.

    ``env`` is passed to the upstream process (the mcp SDK otherwise gives it only a minimal safe
    environment: HOME, PATH, ...). Runs until the downstream client disconnects. While serving, Python-level
    ``print`` output (e.g. from audit sinks or context functions) is redirected to stderr so it can never
    corrupt the JSON-RPC stream on stdout.
    """
    from mcp import Client, StdioServerParameters
    from mcp.server.stdio import stdio_server

    params = StdioServerParameters(command=upstream_command, args=list(upstream_args), env=env,
                                   cwd=str(cwd) if cwd is not None else None)
    async with Client(params) as upstream:
        server = build_proxy_server(gate, upstream, context=context, name=name, on_ask=on_ask,
                                    passthrough=passthrough)
        async with stdio_server() as (read, write):      # claims fd 1 first; only then swap sys.stdout
            with contextlib.redirect_stdout(sys.stderr):
                await server.run(read, write, server.create_initialization_options())


def _load(spec: str) -> Any:
    """Load ``module:attr`` or ``path/to/file.py:attr`` (a file is executed once even if named twice)."""
    target, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected 'module:attr' or 'file.py:attr', got {spec!r}")
    if target.endswith(".py"):
        path = Path(target).resolve()
        module = _FILE_MODULES.get(path)
        if module is None:
            mod_name = f"_riskgate_user_{path.stem}"
            mod_spec = importlib.util.spec_from_file_location(mod_name, path)
            if mod_spec is None or mod_spec.loader is None:
                raise SystemExit(f"cannot load {target}")
            module = importlib.util.module_from_spec(mod_spec)
            sys.modules[mod_name] = module
            if str(path.parent) not in sys.path:        # let the file import its siblings
                sys.path.insert(0, str(path.parent))
            mod_spec.loader.exec_module(module)
            _FILE_MODULES[path] = module
    else:
        module = importlib.import_module(target)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


_FILE_MODULES: dict[Path, Any] = {}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m riskgate.adapters.mcp_proxy --gate SPEC [--context SPEC] [--passthrough a,b] -- CMD ...``.

    ``--gate`` names a ``Gate`` or a zero-argument factory returning a ``Gate`` or ``(Gate, context_fn)``;
    ``--context`` names a zero-argument (sync or async) context function. Logs go to stderr.
    """
    import anyio

    ap = argparse.ArgumentParser(prog="python -m riskgate.adapters.mcp_proxy",
                                 description="riskgate MCP proxy: gate tools/call of an upstream stdio MCP server")
    ap.add_argument("--gate", required=True, help="module:attr or file.py:attr (Gate or factory)")
    ap.add_argument("--context", default=None, help="module:attr or file.py:attr (zero-arg context function)")
    ap.add_argument("--passthrough", default="", help="comma-separated tool names forwarded without a check")
    ap.add_argument("--name", default="riskgate-proxy")
    ap.add_argument("upstream", nargs=argparse.REMAINDER, help="-- upstream command and arguments")
    args = ap.parse_args(argv)
    upstream = list(args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream)
    if not upstream:
        ap.error("missing upstream command (put it after --)")
    obj = _load(args.gate)
    if callable(obj) and not hasattr(obj, "check"):
        obj = obj()
    gate, context = (obj if isinstance(obj, tuple) else (obj, None))
    if args.context:
        context = _load(args.context)
    passthrough = [p for p in args.passthrough.split(",") if p]
    anyio.run(lambda: serve_stdio(gate, upstream[0], upstream[1:], context=context or (lambda: None),
                                  name=args.name, passthrough=passthrough))


if __name__ == "__main__":
    main()
