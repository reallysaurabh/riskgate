# riskgate

**Certify, reject, probe or ask: calibrated, consequence-aware checks for the actions of LLM agents.**

riskgate sits between an AI agent and its tools. Before a consequential action runs, it predicts what the action
would do under **every explanation of the situation that the evidence cannot rule out**, and lets it through only
if the worst plausible outcome clears a threshold **calibrated on labelled outcomes**, so that unsafe actions get
through at most a chosen fraction α of the time, whatever the quality of your simulator.

| Verdict | When | What happens |
|---|---|---|
| **CERTIFY** | safe under every plausible explanation, by a calibrated margin | the tool runs |
| **REJECT** | unsafe under every plausible explanation (or blocked by a hard rule) | the agent gets the reason and re-plans |
| **PROBE** | the explanations disagree and a read-only check can tell them apart | run the check, then decide again |
| **ASK** | anything else: ambiguity, unfamiliar evidence, stale calibration, errors | a human decides, with the gate's explanation |

## For everyone: the careful night-shift assistant

A careless assistant presses the button that fits its first guess. A careful one asks *"what could be going on,
what would this button do in each case, and is it safe in all of them?"*; if it cannot tell, it first looks at a
second gauge or calls the supervisor. riskgate is the careful assistant, and it keeps score: tested on hundreds of
past or simulated situations, it can say *"when I say safe, I am wrong at most 5% of the time"*, and it says
*"I don't know"* when a situation looks unlike anything it was tested on.

## Where to start

| If you want to… | Read |
|---|---|
| understand the idea | [Concepts](concepts.md) |
| run something in five minutes | [Quickstart](quickstart.md) |
| see a realistic lifecycle with real output | [End-to-end tutorial](end_to_end.md) |
| start from your agent's traces and outcome labels (no simulator) | [Start from your traces](from_traces.md) |
| plug in your own simulator and diagnoser | [Bring your own domain](byo_domain.md) |
| see the use cases of the RiskGate site | [Use cases](use_cases.md) |
| know exactly what is guaranteed | [Calibration & guarantees](calibration.md) |
| run it in production | [Operations](operations.md) · [Configuration](configuration.md) · [CLI](cli.md) |
| use it with LangGraph, ADK, OpenAI Agents SDK or MCP | [Integrations](integrations.md) |
| look up a class or parameter | [API reference](api.md) |
| see the evidence (simulated domains) | [Results](results.md) |

## Install

```bash
pip install riskgate              # after the PyPI release; until then: pip install -e ".[dev]" from a clone
```

Python ≥ 3.10; numpy is the only runtime dependency. Optional extras: `yaml`, `otel`, `mcp`, `langgraph`, `adk`,
`openai-agents`, `docs`.

## Status

Release candidate **1.0.0rc1** (Apache-2.0). All results in these docs come from **simulated** domains with scripted
agents; real-LLM evaluations are planned. riskgate is a supervisory layer: it does not replace certified safety
systems, access control or human oversight.
