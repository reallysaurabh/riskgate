---
name: riskgate-from-traces
description: Turn an AI agent's exported traces plus outcome labels (reversals, chargebacks, incidents) into a calibrated riskgate gate and an honest replay report, without writing a simulator.
license: Apache-2.0
compatibility: Python 3.10+ and network access to install the riskgate package from PyPI (pip install --pre riskgate).
metadata:
  version: "1.0.0rc7"
  repository: https://github.com/reallysaurabh/riskgate
---

# riskgate from traces

Use this skill when the user has **agent traces** and wants to know how often the agent's consequential actions
could run without a human, and what a gate would have stopped. Traces are OpenTelemetry spans, LangSmith or Phoenix
runs, or MCP logs. Consequential actions are things like refunds, writes, deploys and account changes. For gating a
single tool by hand, use the `riskgate` skill.

## What you need from the user (ask for exactly this)

1. **Traces**: one record per run, with the tool calls in order, their arguments, and what the read tools returned.
2. **What happened afterwards**, joined by trace id:
   - Good sources: reversals, chargebacks, incident tickets, QA audits, SLO breaches, or a deterministic validator.
   - Not acceptable: an LLM's judgement of the transcript, or "a human approved it". An approval authorises
     execution; it does not show the outcome was safe.
   - If they have no outcome signal, stop and say what is missing. Do not label the traces yourself.
3. **One action and alpha**: which write to gate, and the acceptable rate of wrong approvals (e.g. 0.05).

## Workflow

1. **Convert** their export to riskgate's JSONL trace format and the outcomes to CSV (`references/formats.md`). Write the
   converter as a small script you show the user. Map their run or ticket id to `group` and their agent version to `doer_id`.
2. **Inspect**: `python scripts/inspect_traces.py traces.jsonl outcomes.csv` (from this skill's folder). It reports:
   - tool kinds (read vs write), and you **review the list with the user**: names can mislead;
   - label coverage and label sources;
   - groups and agent versions;
   - which alpha the data supports.

   Exit code 1 means the data cannot support a calibrated gate yet; explain why.
3. **Seal the split** with `seal_split(points, ratios=(0.2, 0.5, 0.3), seed=...)`. Record `split.digest`.
4. **Discover on `split.fit` ONLY**:
   - **Evidence features**: what the agent had seen before the write, e.g. `p.context.last_output("get_order")`.
   - **Missing reads**, e.g. `p.context.called("check_refund_ledger")`.
   - **Read-vs-write mismatches**, e.g. the amount against the order total.
   - **Worlds**: name them from the outcome reasons (already refunded, delivered late, wrong account) and fit with
     `rg.DiscreteWorlds.fit(...)`.
   - **Rules**: limits such as `amount_above(<p99 on fit>)`.
5. **Show the draft to the user** (worlds, features, rules) and let them edit it. Freeze it before touching `split.cal`.
6. **Calibrate once** with `gate.calibrate(split.episodes("cal", context=features), alpha=..., groups=split.groups("cal"))`.
7. **Report on `split.eval`** with `rg.replay(...)`:
   - the unsafe rate next to alpha and next to the agent alone;
   - the autonomous share and the ask share;
   - **where the asks come from** (e.g. writes made without a read that would settle them), and the next fix.
8. **Optional: deploy.**
   - Save the calibration: `cal.save("gate.cal.json", key=...)`.
   - Wire the gate into the agent with the `riskgate` skill if it is installed, or follow
     https://github.com/reallysaurabh/riskgate/blob/main/docs/integrations.md

## Rules you must follow

- **Never let anything learnt from traces touch `cal` or `eval`.** That covers worlds, tables, weights, probes and limits.
  Never recalibrate on `cal` to improve numbers: it is p-hacking, and `split.digest` makes the data auditable.
- **Count groups, not calls.** The calibration split needs at least ceil(1/alpha) groups, and about 10x that to certify much.
- **The bound covers the agent version in the traces** (policy mode). Traces label only the action taken, so other
  actions cannot be compared without a sandbox or simulator. Say so.
- **Report honestly.**
  - Never present numbers from simulated or example data as the user's results.
  - Show the replay line as printed.
  - If almost everything is ASK, say so and explain which missing evidence causes it.

## Files

- `scripts/inspect_traces.py`: data checks before building anything.
- `references/formats.md`: trace and outcome formats, converters from OpenTelemetry and LangSmith, and a complete end-to-end script.
- Docs: https://github.com/reallysaurabh/riskgate/blob/main/docs/from_traces.md
