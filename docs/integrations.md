# riskgate integrations

*Verified on 11 October 2026 against the framework versions listed below. Every statement marked "verified"
is exercised by the contract tests in [`tests/contract/`](https://github.com/reallysaurabh/riskgate/tree/main/tests/contract), which run the real frameworks with
fake/scripted models and never call an LLM API or the network.*

## 1. Summary

| Framework | Tested version | Integration point | riskgate entry points | Status |
|---|---|---|---|---|
| **Any framework** | — | wrap the Python tool function | `riskgate.guard` | stable |
| **Google ADK** | google-adk 2.11.0 (google-genai 2.29.0) | `LlmAgent.before_tool_callback`; Runner plugins | `adk_before_tool_callback`, `adk_plugin` | verified |
| **LangGraph** | langgraph 1.2.12, langgraph-prebuilt 1.1.0, langgraph-checkpoint 4.2.0, langchain-core 1.6.6 | gate node + conditional edge; `ToolNode(wrap_tool_call=...)` | `langgraph_gate_node`, `route_after_gate`, `gated_tool_node`, `langgraph_tool_wrapper`, `langgraph_async_tool_wrapper` | verified |
| **OpenAI Agents SDK** | openai-agents 0.23.1 (openai 3.24.0) | guarded `FunctionTool`; `ToolInputGuardrail`; `needs_approval` | `guarded_function_tool`, `riskgate_tool_input_guardrail`, `riskgate_needs_approval` | verified |
| **Claude Agent SDK** | claude-agent-sdk 0.2.168 | `PreToolUse` hook; `can_use_tool` | `pretooluse_hook`, `can_use_tool` (`riskgate.adapters.claude_agent_sdk`) | verified (types; no model) |
| **MCP** | mcp 2.3.0 (mcp-types 2.3.0) | proxy server in front of any MCP server | `build_proxy_server`, `serve_stdio`, `python -m riskgate.adapters.mcp_proxy`, `riskgate serve-mcp` | verified |

Environment of the verification run: CPython 3.12.13, pydantic 2.13.5, pytest 9.1.1 (`.venv-int`, see §8).
All adapters live in `riskgate.adapters` and import their framework lazily: `import riskgate.adapters`
imports no framework, so the core keeps its numpy-only dependency footprint.

## 2. Verdict mapping

| Verdict | `guard` | ADK callback / plugin | LangGraph gate node / tool wrapper | OpenAI Agents | MCP proxy |
|---|---|---|---|---|---|
| CERTIFY | tool runs | `None` → tool runs | no reply; `ToolNode` runs the call | tool runs (guardrail `allow`) | forwarded upstream, upstream `CallToolResult` returned (+ `_meta["riskgate/decision"]`) |
| REJECT | `{"status":"rejected",...}` or `ActionRejected` | dict response (tool skipped) | `ToolMessage(status="error")`, JSON content | feedback as tool output (`guard`) or `reject_content` (guardrail); optional `raise_exception` via `halt_on` | `isError=true`, JSON text |
| PROBE | `{"status":"probe_requested","probe":{...}}` | dict incl. `probe` | `ToolMessage` incl. `probe` | as REJECT, incl. `probe` | `isError=true`, JSON incl. `probe` |
| ASK | `on_ask(decision)`; else `ApprovalRequired` | `on_ask`, else ADK tool confirmation (`request_confirmation`), else dict | `interrupt()` → `Command(resume=...)`, or `on_ask`, else `ToolMessage` | `on_ask` (`guard`), or `needs_approval` → SDK interruption → approved call allowed by the guardrail | `on_ask` (sync/async) forwards, else `isError=true` |

The feedback object is always `_feedback(decision)`: `{"status": "rejected" | "needs_approval" |
"probe_requested", "reason": str, "guarantee": "valid" | "degraded" | "void"[, "probe": {"name", "args",
"read_only"}]}`. Adapters that produce text (LangGraph, OpenAI guardrail, MCP) serialise it as JSON.

**Fail closed everywhere.** If the user's context function raises, every adapter returns an ASK decision with
reason `fail-closed: could not build the gate context: ...` instead of running the tool (verified for all four
integrations). Context functions may take the framework object (`tool_context`,
`state`, `run_context`) or no argument at all; adapters call `context(obj)` if the signature accepts it and
`context()` otherwise.

## 3. Google ADK (google-adk 2.11.0)

```python
from riskgate.adapters.frameworks import adk_before_tool_callback, adk_plugin

agent = LlmAgent(name="operator", model="gemini-2.5-flash", tools=[FunctionTool(set_setpoint)],
                 before_tool_callback=adk_before_tool_callback(gate, context=lambda tool_context: snapshot()))
# or, for every agent of an app:
runner = InMemoryRunner(agent=root_agent, plugins=[adk_plugin(gate, context=lambda tool_context: snapshot())])
```

Verified facts about the installed ADK:

- `LlmAgent.before_tool_callback: Callable[[BaseTool, dict[str, Any], Context], dict | None | Awaitable[...]]`
  (or a list). ADK invokes canonical callbacks with **keyword arguments** `tool=`, `args=`, `tool_context=`
  (`google.adk.utils._callback_pipeline`) and stops at the first **non-`None`** result (`_stop_on_non_none`),
  which becomes the tool's response; the tool and ADK's own confirmation gate are then skipped.
- Plugins (`BasePlugin.before_tool_callback(self, *, tool, tool_args, tool_context)`, note `tool_args`) run
  **before** agent callbacks, with the same non-`None` early exit. `adk_plugin` returns such a plugin.
- `ToolContext` is `google.adk.agents.context.Context`; it has `function_call_id`, `state`, `actions`,
  `tool_confirmation` and `request_confirmation(*, hint, payload)`.
- **ASK → native confirmation** (default, `confirm=True`, when `on_ask` is None): the callback calls
  `tool_context.request_confirmation(...)`, sets `actions.skip_summarization = True` (as ADK's own confirmation
  gate does) and returns the `needs_approval` feedback. The run pauses with an `adk_request_confirmation`
  function call whose args carry `originalFunctionCall` and `toolConfirmation` (our `hint` and
  `payload={"riskgate": feedback}`). When the client replies with
  `FunctionResponse(name="adk_request_confirmation", id=<call id>, response={"confirmed": true})`, ADK re-runs
  the original call with `tool_context.tool_confirmation` set; the gate re-checks and the tool runs. With
  `{"confirmed": false}` the model receives `{"status": "rejected", "reason": "declined by the user: ..."}`.
- `passthrough={"transfer_to_agent", ...}` exempts tool names from the gate (plugins see ADK-internal tools
  such as agent transfers too; the default is to gate everything).

## 4. LangGraph (langgraph 1.2.12 / langgraph-prebuilt 1.1.0 / langchain-core 1.6.6)

### Pattern A: gate node (deterministic, supports per-call human review)

```python
from riskgate.adapters.frameworks import langgraph_gate_node, route_after_gate

g = StateGraph(MessagesState)
g.add_node("agent", call_model)                                   # model bound to the tools
g.add_node("gate", langgraph_gate_node(gate, context=lambda state: snapshot(), interrupt=True))
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", lambda s: "gate" if s["messages"][-1].tool_calls else END, ["gate", END])
g.add_conditional_edges("gate", route_after_gate, ["tools", "agent"])
g.add_edge("tools", "agent")
app = g.compile(checkpointer=InMemorySaver())                     # interrupt() needs a checkpointer
```

- Reads `AIMessage.tool_calls` (`[{"name", "args", "id", "type": "tool_call"}]`); OpenAI-style dict calls
  (`{"function": {"name", "arguments"}}`) are accepted too.
- Non-certified calls are answered immediately with
  `ToolMessage(content=<JSON feedback>, tool_call_id=..., name=..., status="error", artifact={"riskgate": decision})`.
  When the incoming message is a plain dict, replies are OpenAI-style dicts (`message_format="auto"`; force
  with `"langchain"` / `"dict"`).
- **Routing is derived from the messages**, not from a state key: LangGraph silently drops node-update keys that
  are not in the state schema (verified: `{"riskgate": ...}` vanishes with `MessagesState`). `route_after_gate`
  returns `"tools"` if all calls of the last AI message are still unanswered, `"agent"` if all were answered by
  the gate, and `[Send("tools", [certified tool calls])]` for **mixed batches**, so the `ToolNode` executes
  exactly the certified calls (a list of `ToolCall` dicts is the `Send` payload `ToolNode` hydrates state for).
  Every tool call therefore gets exactly one `ToolMessage`, as chat APIs require.
- **ASK + `interrupt=True`**: payload `{"type": "riskgate_approval", "tool_call": {...}, "reason", "guarantee",
  "feedback"}` appears in `app.invoke(...)["__interrupt__"]`; resume with `Command(resume={"approve": True})`
  (also `True`/`"approve"`/`"yes"`). Several ASK calls in one batch are reviewed one interrupt at a time
  (verified). On resume LangGraph re-executes the node, so the gate re-checks with fresh context and the
  decision is audited again.
- The decisions dict is also returned under `decisions_key="riskgate"` (kept only if your schema has that key).

### Pattern B: gated `ToolNode` (drop-in, works with prebuilt agents)

```python
from riskgate.adapters.frameworks import gated_tool_node

agent = create_react_agent(model, tools=gated_tool_node(gate, tools, context=lambda state: snapshot()))
```

- Uses the `ToolNode(tools, wrap_tool_call=..., awrap_tool_call=...)` hooks that exist in langgraph-prebuilt
  1.1.0 (`ToolCallRequest(tool_call, tool, state, runtime)` + `execute(request)`); both sync (`invoke`) and async
  (`ainvoke`) paths are covered. Blocked calls return `ToolMessage(status="error")` with the JSON feedback.
- `create_react_agent` is deprecated in LangGraph 1.x (moved to `langchain.agents.create_agent`) but still
  works with a `ToolNode` instance (verified). The `langchain` package (and its middleware API) was not
  installed or tested.
- Caveats: `ToolNode` runs a step's calls concurrently, so with several ASK calls *and* `interrupt` in one step
  prefer Pattern A. An `interrupt()` raised inside the wrapper propagates only with ToolNode's default
  `handle_tool_errors`; with `handle_tool_errors=True` LangGraph converts it into an error message.

## 5. OpenAI Agents SDK (openai-agents 0.23.1)

```python
from riskgate.adapters.openai_agents import (guarded_function_tool, riskgate_needs_approval,
                                              riskgate_tool_input_guardrail)

# (a) gate inside the tool function (holds however the tool is invoked)
tool = guarded_function_tool(gate, set_setpoint, context=snapshot, on_ask=None)

# (b) native SDK hooks: guardrail for REJECT/PROBE, needs_approval for ASK (human in the loop)
tool = function_tool(set_setpoint,
                     needs_approval=riskgate_needs_approval(gate, "set_setpoint", context=snapshot),
                     tool_input_guardrails=[riskgate_tool_input_guardrail(gate, context=snapshot)])
result = await Runner.run(agent, "...")
while result.interruptions:                       # ASK
    state = result.to_state()
    for item in result.interruptions:
        state.approve(item)                       # or state.reject(item)
    result = await Runner.run(agent, state)
```

What exists in the installed SDK (verified by introspection and the contract tests):

- `agents.function_tool(..., needs_approval=bool | async (run_context, arguments: dict, call_id) -> bool,
  tool_input_guardrails=[...], tool_output_guardrails=[...], ...)`.
- Tool guardrails: `ToolInputGuardrail(guardrail_function, name)`, decorator `tool_input_guardrail`,
  `ToolInputGuardrailData(context: ToolContext, agent)`, outputs `ToolGuardrailFunctionOutput.allow(output_info)`,
  `.reject_content(message, output_info)` (the message replaces the tool output; the run continues) and
  `.raise_exception(output_info)` (the Runner raises `ToolInputGuardrailTripwireTriggered`). Results are in
  `result.tool_input_guardrail_results`. `ToolContext` exposes `tool_name`, `tool_call_id`, `tool_arguments`
  (JSON string), `context` and `is_tool_approved(tool_name, call_id)`.
- Order inside `Runner.run`: `needs_approval` is evaluated first; if it returns True the run stops with a
  `ToolApprovalItem` in `result.interruptions`. Input guardrails run right before the tool is invoked
  (after approval). `riskgate_tool_input_guardrail` allows an ASK call only when
  `ToolContext.is_tool_approved(...)` is True, so approval and gating compose; each call is therefore checked
  (and audited) twice.
- `guarded_function_tool` builds the schema from the original signature (`functools.wraps` + `__wrapped__`);
  ASK without `on_ask` returns `{"status": "needs_approval", ...}` instead of raising (an exception would reach
  the model only as the SDK's fixed `"An error occurred while running the tool. Please try again."`, without
  the reason). Functions whose first
  parameter is `RunContextWrapper`/`ToolContext` are refused (the guard would treat the context as an action
  argument); use the guardrail for those.

Caveats:

- **Argument coercion forces approval** (pinned by a test): with a callable `needs_approval`, the SDK requires
  approval whenever pydantic validation changes the arguments, e.g. JSON `{"v": 10}` for a `v: float`
  parameter (`10` → `10.0`), regardless of the gate's verdict. Use exact types (`int`, `int | float`) for values
  models emit as integers.
- Non-string tool outputs are stringified with `str()` by the SDK, so `guarded_function_tool` feedback reaches
  the model as a Python dict repr; the guardrail path sends JSON.
- Do not combine `guarded_function_tool` with `needs_approval` (the in-function guard cannot see SDK approvals
  and would answer an approved call with `needs_approval`), and do not enable
  `RunConfig(tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True))` together with
  `riskgate_needs_approval` (the guardrail would reject ASK calls before the human is asked).
- Offline testing: the SDK ships `agents.testing.ScriptedModel`, `function_call(...)`, `assistant_message(...)`;
  run with `RunConfig(tracing_disabled=True)`.

## 6. MCP proxy (mcp 2.3.0)

```bash
python -m riskgate.adapters.mcp_proxy --gate my_gate.py:make_gate --context my_gate.py:snapshot \
    --passthrough read_level -- npx -y @acme/plant-mcp-server
riskgate serve-mcp --gate pkg.module:make_gate --calibration cal.json --context pkg.module:snapshot -- <cmd>
```

```python
from riskgate.adapters.mcp_proxy import build_proxy_server, serve_stdio

async with Client(upstream_server_or_params) as upstream:          # mcp.Client or an entered ClientSession
    proxy = build_proxy_server(gate, upstream, context=snapshot, passthrough={"read_level"})
    async with Client(proxy) as client: ...                           # in process (tests, notebooks)
await serve_stdio(gate, "npx", ["-y", "@acme/plant-mcp-server"], context=snapshot)   # real use
```

- The proxy is a low-level `mcp.server.lowlevel.Server(name, on_list_tools=..., on_call_tool=...)`; mcp 2.x
  registers handlers in the constructor and calls them as `handler(ctx, params)`. `tools/list` re-exposes the
  upstream tools unchanged (all pages, followed for both `Client.list_tools(cursor=)` and
  `ClientSession.list_tools(params=)`); `tools/call` builds `Action(name, arguments)` and calls
  `gate.check(action, await context())` (sync or async context).
- CERTIFY → forwarded; the upstream `CallToolResult` (content, `structuredContent`, `isError`) is returned with
  `_meta["riskgate/decision"]` added and the upstream's `_meta["io.modelcontextprotocol/serverInfo"]` stamp
  removed so the proxy identifies itself. REJECT/PROBE/ASK → `CallToolResult(isError=True)` with the JSON
  feedback as text, the same object as `structuredContent` when the tool declares no `outputSchema` (omitted
  otherwise, to avoid contradicting the schema), and a JSON-safe decision summary in `_meta`. Upstream
  exceptions become `isError` results (`upstream error: ...`).
- `passthrough` is an explicit allow-list of tools forwarded without a check (for harmless reads that the
  consequence model does not cover); everything else is gated.
- mcp 2.x API changes that matter (verified): FastMCP is now `mcp.server.mcpserver.MCPServer`
  (`mcp.server.fastmcp` raises); `mcp.Client(server)` connects in process to a `Server`/`MCPServer`, to a URL,
  to `StdioServerParameters` or to a transport; `mcp.shared.memory.create_connected_server_and_client_session`
  no longer exists (tests use `Client(proxy)` and `Client(mcp.client._memory.InMemoryTransport(proxy))` for real
  JSON-RPC framing; `_memory` is a private module); model fields are snake_case (`is_error`,
  `structured_content`, `next_cursor`, `meta` ↔ `_meta`); `MCPServer` hides details of unexpected tool
  exceptions (raise `ToolError` for messages clients should see).
- stdio hygiene: `stdio_server()` claims fds 0/1 while serving, but block-buffered Python `print` output was
  still flushed onto the JSON-RPC stream at shutdown in our tests. `serve_stdio` therefore redirects
  `sys.stdout` to stderr while serving (regression-tested with a printing audit sink). The upstream
  subprocess receives only the SDK's minimal default environment unless you pass `env=`.
- Not implemented / not verified: forwarding upstream `InputRequiredResult` rounds (2026-07-28 protocol) or
  elicitation to the downstream client; ASK via MCP elicitation (use `on_ask`); a Streamable HTTP front end
  (`build_proxy_server(...)` returns a low-level `Server`, whose `streamable_http_app()` should work but is
  untested).

## 6b. Claude Agent SDK (claude-agent-sdk 0.2.168)

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from riskgate.adapters.claude_agent_sdk import can_use_tool, pretooluse_hook

options = ClaudeAgentOptions(hooks={"PreToolUse": [
    HookMatcher(matcher="mcp__payments__.*", hooks=[pretooluse_hook(gate, context=snapshot)])]})
# or, for apps that decide permissions in code:
options = ClaudeAgentOptions(can_use_tool=can_use_tool(gate, context=snapshot, on_ask=ask_a_human))
```

- **The hook returns** `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": ..., "permissionDecisionReason": ...}}`:
  - CERTIFY → no decision (normal permissions apply; `allow_on_certify=True` returns `"allow"`);
  - REJECT and PROBE → `"deny"`, with the gate's JSON feedback as the reason;
  - ASK → `"ask"` (the user is prompted).
- **Why the hook is preferred:** a hook deny holds even in `bypassPermissions` mode, while `can_use_tool` does not
  fire for auto-approved tools.
- **Tool names:** `mcp__<server>__<tool>` is mapped to `<tool>` for the gate's action name (`name=` overrides this).
  A failing context function gives ASK.
- **Verified:** `tests/contract/test_claude_agent_sdk.py` checks the output keys and decision values against the
  SDK's `PreToolUseHookSpecificOutput` type, and checks registration in `ClaudeAgentOptions`. It runs no model.

Agent skills for Claude, Gemini CLI and Codex: [Agent skills](agent_skills.md).

## 7. General caveats

- **Stage.** Adapters call `gate.check(..., stage=1)`. Two-stage probing (re-checking at stage 2 after the agent
  ran the suggested probe) is not automated by the adapters; track it in your context function or call
  `gate.check(action, ctx, stage=2)` yourself.
- **Batches.** Each call is checked on its own; joint effects of several world-changing calls proposed in one
  model turn are not modelled. The LangGraph gate node checks a whole batch against one snapshot before
  anything runs. The wrapper-style integrations (ADK callback/plugin, `gated_tool_node`, `guard`, the OpenAI
  guardrail, the MCP proxy) check each call right before it executes and re-read the context, but frameworks
  may execute a turn's calls concurrently. If joint effects matter, serialise execution (OpenAI:
  `RunConfig(tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1))`) or disable parallel tool
  calls in the model settings, and make the context function read live state.
- **Audit volume.** Human-review paths re-check the call on resume (LangGraph, ADK) and the OpenAI
  approval+guardrail combination checks twice; audit logs contain one record per check.
- **Blocking.** `gate.check` is synchronous; in async frameworks a slow consequence model blocks the event loop
  for its duration.
- Framework hooks change quickly: re-run `tests/contract` whenever you bump a framework (see §8).

## 8. Reproducing the verification

```bash
cd riskgate
uv venv .venv-int --python 3.12
uv pip install --python .venv-int/bin/python -e . pytest langgraph langchain-core openai-agents google-adk mcp
.venv-int/bin/python -m pytest -q tests            # core + tests/contract (skip cleanly without frameworks)
```

| Contract test file | What it exercises |
|---|---|
| `tests/contract/test_langgraph.py` | `StateGraph` + gate node + `ToolNode`: certified/rejected, mixed batch via `Send`, `interrupt` approve/decline (single and multiple ASK), `gated_tool_node` in `create_react_agent` (incl. `interrupt` inside the wrapper), async wrapper + `on_ask`, PROBE, dict/OpenAI-format messages, fail-closed context |
| `tests/contract/test_adk.py` | `LlmAgent` + `InMemoryRunner` + `FunctionTool` with a scripted `BaseLlm`: callback convention, certified/rejected, native confirmation approve/decline, `on_ask`, `adk_plugin`, PROBE, `passthrough`, fail-closed context |
| `tests/contract/test_openai_agents.py` | `Runner` with `agents.testing.ScriptedModel`: guarded tool (direct `on_invoke_tool` and Runner), guardrail feedback and results, `needs_approval` approve/reject, argument-coercion pin, `halt_on` tripwire, PROBE + malformed JSON, context-parameter refusal, fail-closed context |
| `tests/contract/test_mcp_proxy.py` | upstream `MCPServer` + proxy + `Client` in process and JSON-RPC framed: forwarding, `isError` feedback, structured content rules, async `on_ask`, async context, `passthrough`, PROBE, fail-closed context, upstream errors; CLI over real stdio across two subprocesses |

Runnable offline demos of every integration are in [`examples/integrations/`](https://github.com/reallysaurabh/riskgate/tree/main/examples/integrations)
(see its README for switching to real models).
