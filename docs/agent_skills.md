# Agent skills

riskgate ships two [Agent Skills](https://agentskills.io): folders of instructions and scripts that coding and chat
agents load when a task matches. Install them, then ask your agent in plain words. For example: *"put a gate in
front of my refund tool"*, or *"here are my LangSmith traces and chargebacks: how often could this agent run
unsupervised?"*

| Skill | What the agent does with it |
|---|---|
| `riskgate` | Wraps one consequential tool with a gate. It names the worlds, adds read-only lookups and hard rules, calibrates on labelled episodes, replays on held-out ones, and wires the gate into your framework or MCP server. Includes a runnable quickstart (`scripts/quickstart_gate.py`). |
| `riskgate-from-traces` | Takes your agent's traces plus an outcome signal. It checks the data (`scripts/inspect_traces.py`), seals a fit / cal / eval split, learns the worlds on fit only, calibrates once, and writes an honest replay report. See [Start from your traces](from_traces.md). |

Both skills tell the agent the rules that protect the guarantee:

- never label outcomes with an LLM or with approvals;
- hold out data, and calibrate once;
- count groups, not tool calls;
- set `doer_id`;
- never weaken fail-closed behaviour.

They install the library from PyPI (`pip install --pre riskgate`). The agent needs Python 3.10+ and network access.

## Install

=== "Claude Code"

    ```text
    /plugin marketplace add reallysaurabh/riskgate
    /plugin install riskgate@riskgate
    ```

    From a shell, use `claude plugin marketplace add reallysaurabh/riskgate` and then
    `claude plugin install riskgate@riskgate`. The skills appear as `riskgate:riskgate` and
    `riskgate:riskgate-from-traces`.

    Without the plugin system, copy a skill folder to `~/.claude/skills/` (all projects) or `.claude/skills/` (one project).

=== "Claude apps (claude.ai, Desktop)"

    1. Download `riskgate.zip` and/or `riskgate-from-traces.zip` from the
       [latest release](https://github.com/reallysaurabh/riskgate/releases).
    2. Go to **Customize → Skills → + → Upload a skill** and pick the zip.

    Code execution must be enabled. Each zip contains the skill folder at its root.

=== "Claude API / Agent SDK"

    **Messages API:** upload the folder with `POST /v1/skills`. In Python:

    ```python
    import anthropic
    from anthropic.lib import files_from_dir

    client = anthropic.Anthropic()
    skill = client.skills.create(files=files_from_dir("skills/riskgate"))
    ```

    **Agent SDK:** copy the skill folders to `.claude/skills/` in the agent's working directory.

    To *enforce* the gate in an agent built with the Claude Agent SDK, use the PreToolUse hook as well:

    ```python
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from riskgate.adapters.claude_agent_sdk import pretooluse_hook

    options = ClaudeAgentOptions(hooks={"PreToolUse": [
        HookMatcher(matcher="mcp__payments__.*", hooks=[pretooluse_hook(gate, context=snapshot)])]})
    ```

=== "Gemini CLI"

    ```bash
    gemini extensions install https://github.com/reallysaurabh/riskgate
    ```

    The repository is a Gemini CLI extension, and its `skills/` folder is loaded with it. Without the extension, copy a
    skill folder to `~/.gemini/skills/` or `.gemini/skills/`, or run
    `gemini skills install https://github.com/reallysaurabh/riskgate`.

=== "Codex and other agents"

    Copy the skill folders to `.agents/skills/` in your repository, or to `~/.agents/skills/`. Codex CLI and other agents
    that follow the Agent Skills standard read them from there.

## Skills guide; the gate enforces

A skill shapes what the agent writes and suggests. It does not stop a tool call at run time. To enforce the
verdicts, put the gate in the call path:

- `@rg.guard` for any Python tool;
- the framework adapters: [Integrations](integrations.md), including `pretooluse_hook` for the Claude Agent SDK;
- `riskgate serve-mcp` in front of an MCP server, for any MCP client: Claude Desktop, Claude Code, Gemini CLI, IDEs.

## Updating

The skills are versioned with the package (`metadata.version` in each `SKILL.md`).

- **Claude Code:** `claude plugin update riskgate@riskgate`.
- **Gemini CLI:** `gemini extensions update riskgate`.
- **Copied folders:** re-copy them from the new release.
