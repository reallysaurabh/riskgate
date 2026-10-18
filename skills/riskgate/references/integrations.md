# Wiring riskgate into an agent

Verdict handling is the same everywhere:
- **CERTIFY**: the tool runs.
- **REJECT** and **PROBE**: the agent gets JSON feedback, `{"status": "rejected" | "probe_requested", "reason", "guarantee"[, "probe"]}`.
- **ASK**: goes to a person.
- **Context errors**: if building the context fails, the verdict is ASK (fail closed).

## Any Python tool

```python
@rg.guard(gate, context=lambda: snapshot(), on_ask=ask_a_human)    # on_ask(decision) -> bool; without it: ApprovalRequired
def issue_refund(order: str, amount: float) -> str: ...
```

## Claude Agent SDK (`pip install --pre "riskgate[claude-agent-sdk]"`)

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
from riskgate.adapters.claude_agent_sdk import pretooluse_hook

options = ClaudeAgentOptions(
    mcp_servers={"payments": payments_server},
    hooks={"PreToolUse": [HookMatcher(matcher="mcp__payments__.*", hooks=[pretooluse_hook(gate, context=snapshot)])]})
```

- **Mapping to the hook:** CERTIFY → no decision (your normal permissions apply; `allow_on_certify=True` returns "allow"),
  REJECT and PROBE → "deny" with the feedback as the reason, ASK → "ask" (the user is prompted).
- **Tool names:** MCP tool names `mcp__<server>__<tool>` are mapped to `<tool>` (`name=` overrides this).
- **App-level permissions:** `can_use_tool(gate, context=..., on_ask=...)` is the alternative to the hook.

## LangGraph, Google ADK, OpenAI Agents SDK

```python
from riskgate.adapters.frameworks import gated_tool_node, adk_before_tool_callback, adk_plugin
from riskgate.adapters.openai_agents import guarded_function_tool, riskgate_tool_input_guardrail, riskgate_needs_approval
tools_node = gated_tool_node(gate, tools, context=lambda state: snapshot(state))                 # LangGraph
agent = LlmAgent(..., before_tool_callback=adk_before_tool_callback(gate, context=snapshot))      # ADK / Gemini
```

Details: https://github.com/reallysaurabh/riskgate/blob/main/docs/integrations.md

## Any MCP client (Claude Desktop, Claude Code, Gemini CLI, IDEs): put the gate in front of the server

```bash
pip install --pre "riskgate[mcp]"
riskgate serve-mcp --gate mygates:make_gate --calibration gate.cal.json --context mygates:snapshot \
    --passthrough get_order,list_orders -- npx -y @acme/payments-mcp-server
```

- **What the client sees:** the upstream tools, unchanged. Certified calls are forwarded, everything else returns
  `isError` with the gate's feedback.
- **Configuration:** register the `riskgate serve-mcp ...` command as the MCP server in the client's config, in place
  of the upstream command.
- **Calibration file:** `gate.cal.json` comes from `riskgate calibrate --gate ... --episodes ... --out gate.cal.json`.
  Sign it with `--key`, or set `RISKGATE_KEY`.
