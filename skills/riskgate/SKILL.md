---
name: riskgate
description: Gate an AI agent's risky tool calls (refunds, payments, deletes, scaling, account blocks) with riskgate so they run only when calibrated as safe; otherwise check first, re-plan or ask a person.
license: Apache-2.0
compatibility: Python 3.10+ and network access to install the riskgate package from PyPI (pip install --pre riskgate).
metadata:
  version: "1.0.0rc7"
  repository: https://github.com/reallysaurabh/riskgate
---

# riskgate: certify, reject, probe or ask

riskgate sits between an agent and a consequential tool. For each proposed call it lists the situations
("worlds") the evidence cannot rule out. It checks whether the action is safe in every one of them, using a
threshold calibrated on labelled outcomes so that unsafe actions get through at most a chosen fraction alpha of
the time. Verdicts:
**CERTIFY** (run it), **PROBE** (run a read-only check first), **REJECT** (unsafe in every world: re-plan), **ASK** (a person
decides). No LLM is on the approval path, and anything unexpected fails closed to ASK.

Use this skill when the user wants to add guardrails, approval gates or "human in the loop" to agent tools that move
money, change infrastructure, delete data or label people. If they only have **traces and outcome labels**, use the
`riskgate-from-traces` skill instead.

## Workflow

1. **Install and check**: `pip install --pre riskgate` then `riskgate version`. Run `python scripts/quickstart_gate.py`
   from this skill's folder to see all four verdicts in a few seconds.
2. **Pick ONE consequential action** and write down, with the user:
   - the action and its arguments;
   - what the agent knows at decision time (the context, nothing from the future);
   - what "unsafe" means, measurable after the fact;
   - the acceptable rate alpha, e.g. 0.05.
3. **Name the worlds**: the few hidden facts that decide whether the action is right. For example: paid late / failed /
   pending / paid twice. Or: hold / subscription / fraud. If the decision hinges on one such fact, use
   `rg.DiscreteWorlds` and `rg.OutcomeTable` (see `references/patterns.md`, pattern A). Otherwise write a consequence model
   (pattern D).
4. **Add the read-only lookups** that tell the worlds apart (`OutcomeTable(reveals=...)`): the gate PROBEs with them
   before anything irreversible runs.
5. **Add hard rules** for policy that is not statistical: `riskgate.rules.needs_person(...)`, `amount_above(...)`, `never(...)`.
6. **Calibrate on labelled episodes** (`gate.calibrate(episodes, alpha=..., groups=...)`), then check on held-out ones with
   `rg.replay(gate, held_out)` or `rg.resplit(make_gate, episodes)`. Report the unsafe rate next to alpha and the agent-alone
   rate, and say plainly whether the data was simulated.
7. **Wire it into the agent** (`references/integrations.md`): `@rg.guard` for any Python tool, adapters for LangGraph,
   Google ADK, the OpenAI Agents SDK and the Claude Agent SDK (`pretooluse_hook`), or `riskgate serve-mcp` in front of
   any MCP server.

## Rules you must follow

- **Never invent labels.** "Unsafe" comes from what happened afterwards, or a deterministic check: reversals,
  chargebacks, incidents, audits, a validator.
  - Never from an LLM judging the transcript.
  - Never from a human approving the action.
  - If the user has no labels, say so and propose simulated episodes for a demo only, clearly labelled as simulated.
- **Hold out.** Never report replay numbers on the episodes the gate was calibrated on. Never recalibrate on the same data until the numbers look good.
- **Count tasks, not tool calls.** Calls from one conversation or incident are dependent: pass `groups=`.
  - The calibration split needs at least ceil(1/alpha) groups (20 at alpha = 5%).
  - Plan for about 10x that before anything gets certified.
- **Set `doer_id`** (model + prompt version) in policy mode. A different agent needs recalibration; riskgate marks the old calibration VOID.
- **Do not weaken fail-closed behaviour.**
  - Don't catch `ApprovalRequired` and run the tool anyway.
  - Don't show the agent `Decision.reason`, `score` or `threshold`: adapters send the numbers-free `agent_reason`.
  - `require_calibration=False` is for analysis, not shadow mode: it never certifies while the status is VOID.
  - Dry-runs need an isolated copy (a database clone, branch or snapshot), never a savepoint on a production database.
  - Don't turn ASK into CERTIFY.
  - Don't raise alpha just to get autonomy without telling the user.
- **Explain what is not guaranteed.**
  - The bound is on average over decisions, not per decision.
  - It assumes future decisions look like the calibration data; the drift monitor flags when they don't.
  - A weak model costs more ASKs, not safety.

## Files

- `scripts/quickstart_gate.py`: a runnable, self-contained gate (simulated payments domain) that prints each verdict.
- `references/patterns.md`: code for discrete worlds, refunds with amounts, prompt injection, and custom consequence models.
- `references/integrations.md`: framework wiring, including Claude Agent SDK hooks and the MCP proxy.
- Full docs: https://github.com/reallysaurabh/riskgate/tree/main/docs
