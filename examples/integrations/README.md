# riskgate integration examples (offline)

Each script runs a **real** agent framework with a **fake, scripted model**, so it needs no API key and no
network. Every gate decision is printed (`[riskgate] add({"v":90}) -> REJECT ...`) by the `PrintAudit` sink in
[`_toy_gate.py`](_toy_gate.py), followed by what the agent saw.

The toy domain is a tank with a hidden leak: `add(v)` is the only world-changing tool and the gate (calibrated
at α = 0.1 on 200 simulated episodes) says `add(10)` → CERTIFY, `add(90)` → REJECT (overflow under every
plausible leak), `add(50)` → ASK (an operator-approval hard rule).

| Script | Framework (tested version) | What it shows |
|---|---|---|
| [`langgraph_agent.py`](langgraph_agent.py) | langgraph 1.2.12, langchain-core 1.6.6 | gate node + `route_after_gate` with `interrupt()` approval; `gated_tool_node` inside `create_react_agent` with a mixed batch |
| [`adk_agent.py`](adk_agent.py) | google-adk 2.11.0 | `before_tool_callback` gate; ASK through ADK's native tool confirmation |
| [`openai_agents_agent.py`](openai_agents_agent.py) | openai-agents 0.23.1 | `guarded_function_tool`; native tool input guardrail + `needs_approval` interruptions |
| [`mcp_proxy_demo.py`](mcp_proxy_demo.py) | mcp 2.3.0 | the gating MCP proxy between a client and an upstream server |

## Run

```bash
cd riskgate                                   # repository root
uv venv .venv-int --python 3.12
uv pip install --python .venv-int/bin/python -e . langgraph langchain-core google-adk openai-agents mcp
.venv-int/bin/python examples/integrations/langgraph_agent.py
.venv-int/bin/python examples/integrations/adk_agent.py
.venv-int/bin/python examples/integrations/openai_agents_agent.py
.venv-int/bin/python examples/integrations/mcp_proxy_demo.py
```

## Switching to real models

Only the model changes; the gate wiring stays exactly the same. Replace the toy gate with your own
(`make_gate()` → your consequence model, hypotheses, spec and calibration) and `snapshot()` with a function
that reads the live state the gate should evaluate against.

**LangGraph.** In the hand-built graph the model node must emit tool calls, so bind the tools:

```python
from langchain_openai import ChatOpenAI            # or langchain_anthropic / langchain_google_genai ...
model = ChatOpenAI(model="<model-name>").bind_tools([add])        # replaces scripted(...)
```

`create_react_agent(model, tools=gated_tool_node(gate, [add], context=...))` takes the unbound chat model and
binds the tools itself. (In LangGraph 1.x `create_react_agent` is deprecated in favour of
`langchain.agents.create_agent`, which accepts the same `ToolNode`-level `wrap_tool_call` hooks via middleware;
that path is not covered by the contract tests.) Interrupts need a checkpointer (`InMemorySaver` here; use a
persistent one in production).

**Google ADK.** Use a model string instead of `ScriptedLlm`:

```python
agent = LlmAgent(name="tank_operator", model="gemini-2.5-flash", tools=[FunctionTool(add)],
                 before_tool_callback=adk_before_tool_callback(gate, context=lambda tool_context: snapshot()))
```

with `GOOGLE_API_KEY=...` (Gemini API) or `GOOGLE_GENAI_USE_VERTEXAI=TRUE`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION` (Vertex AI). Your UI must answer `adk_request_confirmation` function calls with a
`FunctionResponse(name="adk_request_confirmation", id=<call id>, response={"confirmed": true|false})`, exactly
as the example does. To gate every agent of a multi-agent app,
pass `plugins=[adk_plugin(gate, context=...)]` to the Runner/App instead.

**OpenAI Agents SDK.** Drop `model=script()` (the SDK default model is used) or pass a model name, set
`OPENAI_API_KEY`, and remove `run_config=OFFLINE` if you want traces. Prefer exact parameter types for tools
that use `needs_approval` (e.g. `int` or `int | float` for values the model emits as JSON integers): the SDK
requires approval whenever argument validation changes the input (JSON `10` coerced to `10.0`), whatever the
gate says.

**MCP.** Point your MCP client at the proxy instead of the server. For example, in a Claude Desktop / IDE
`mcpServers` entry:

```json
{
  "mcpServers": {
    "plant": {
      "command": "/abs/path/.venv-int/bin/python",
      "args": ["-m", "riskgate.adapters.mcp_proxy",
               "--gate", "/abs/path/my_gate.py:make_gate", "--context", "/abs/path/my_gate.py:snapshot",
               "--passthrough", "read_level",
               "--", "npx", "-y", "@acme/plant-mcp-server"]
    }
  }
}
```

`--gate` names a `Gate`, or a zero-argument factory returning a calibrated `Gate` or `(gate, context_fn)`.
The riskgate CLI offers the same proxy as `riskgate serve-mcp --gate pkg.module:make_gate --calibration
cal.json [--context pkg.module:snapshot] -- <upstream command>`. The upstream subprocess gets only the
minimal environment the mcp SDK passes by default; use `serve_stdio(..., env={...})` from Python when it needs
credentials.
