# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/).
The calibration artefact schema is versioned separately (`riskgate.calibration/vN`).

## [Unreleased]

## [1.0.0rc7] — 2026-10-18 (release candidate)

Documentation-only release: no code changes since 1.0.0rc6.

### Documentation
- Documentation site: https://reallysaurabh.github.io/riskgate/ (MkDocs Material on GitHub Pages, rebuilt on every push to `main` by `.github/workflows/docs.yml`). PyPI's *Documentation* link points there.
- Fixed 17 lists across 8 pages that rendered as run-on text on the site.

## [1.0.0rc6] — 2026-10-18 (release candidate)

Four fixes before the 1.0.0 API freeze. No calibration maths and no artefact schema (`riskgate.calibration/v1`)
change, and the text of `Decision.reason` is unchanged.

### Security
- **Agent-facing feedback no longer contains robust margins or the calibrated threshold.**
  - New `Decision.agent_reason`, a numbers-free text the agent may see; adapters send `agent_reason or reason`.
  - `Decision.reason` is unchanged and stays in audit records, `on_ask` and approver payloads.
  - Fail-closed decisions get a generic agent text, so exception details (paths, connection strings) stay internal.
  - The ADK confirmation hint uses the agent text too.
- **MCP proxy `_meta["riskgate/decision"]`** no longer carries scores, the threshold or the full reason unless
  `expose_scores=True` (`riskgate serve-mcp --expose-scores`) is set for a trusted client.

### Fixed
- **A stale calibration no longer acts.** With `require_calibration=False`, a stale artefact (status `VOID`) no
  longer produces `CERTIFY` or `PROBE`:
  - such decisions are returned as `ASK`;
  - the would-be verdict goes in `Decision.would_verdict` and the audit record;
  - `REJECT` is unchanged.
  - The docs no longer call this setting "shadow mode".

### Changed
- **Human declines are reported as declined.** When a human or `on_ask` says no, the agent gets
  `{"status": "declined", ...}`. This applies in `guard`, the ADK adapter (previously `"rejected"`), the LangGraph
  adapters, the MCP proxy and the Claude Agent SDK `can_use_tool` callback. `needs_approval` now only means that no
  approver was reachable, or that an ADK confirmation is still pending.
- **Every blocked-call feedback** carries `"executed": false`.
- **New `guard(..., raise_on_ask=True)`.** With `False`, an ASK without `on_ask` returns `needs_approval` feedback
  instead of raising. `guarded_function_tool` now uses this instead of a stand-in `on_ask=lambda d: False`, which
  would have reported every ASK as declined.

### Documentation
- **New "Safe database dry-runs" section** (`docs/operations.md`): a savepoint on the live database is not a sandbox
  (MySQL, MariaDB and Oracle commit DDL implicitly; PostgreSQL rollbacks still lock and write WAL; `EXPLAIN ANALYZE`
  executes).
- **Savepoint wording corrected** in the README, byo_domain, operations, calibration, faq, quickstart and SPEC. The
  refunds example explains why its per-ticket in-memory SQLite savepoint is safe.

### Tests
- `tests/test_adapter_semantics.py`: the outcome-semantics table.
- New VOID tests (certify and probe), agent-reason tests, and decline contract tests for MCP and ADK.

## [1.0.0rc5] — 2026-10-18 (release candidate)

### Added
- **Agent skills** (`skills/`, Agent Skills standard):
  - `riskgate`: gate a risky tool, with a runnable quickstart;
  - `riskgate-from-traces`: traces + outcomes → a calibrated gate and a replay report, with `inspect_traces.py`.
- **Packaging for the skills:**
  - a Claude Code plugin and marketplace (`.claude-plugin/`; `/plugin install riskgate@riskgate`);
  - a Gemini CLI extension (`gemini-extension.json`);
  - claude.ai zips attached to every GitHub release.
- **`riskgate.adapters.claude_agent_sdk`**: `pretooluse_hook` and `can_use_tool` for the Claude Agent SDK, with
  contract tests against claude-agent-sdk 0.2.168. Extra: `riskgate[claude-agent-sdk]`.
- **Docs:** `docs/agent_skills.md`.

## [1.0.0rc4] — 2026-10-18 (release candidate)

### Added
- **`riskgate.traces`**: onboarding from agent traces plus an outcome signal, without a simulator.
  - `load_traces` (JSONL runs with tool calls and read outputs) and `load_outcomes` (CSV / JSONL joined by trace id).
    Label sources that name an LLM judge or an approval are refused.
  - `infer_tool_kinds`: read vs write; unknown tools count as writes.
  - `decision_points`: one labelled write each, with a no-look-ahead `TraceContext`; unlabelled and conflicting writes
    are counted, not guessed.
  - `seal_split`: group-disjoint fit / cal / eval with a content digest.
- **`DiscreteWorlds.fit`**: learn the prior and evidence tables from labelled examples (use the fit split).
- **Docs**: `docs/from_traces.md`, including the three rules that keep the guarantee. Example
  `examples/use_cases/from_traces.py`.

## [1.0.0rc3] — 2026-10-18 (release candidate)

Building blocks for the use cases on the RiskGate site (map of all sixteen: `docs/use_cases.md`).

### Added
- **`riskgate.worlds`** for decisions that hinge on one hidden fact with a few values:
  - `DiscreteWorlds`: naive-Bayes hypothesis provider. Scores are −log posterior; unobserved fields are skipped, so
    one provider serves both probe stages; the tables are fingerprinted.
  - `OutcomeTable`: which action is safe in which world (or a callable) and what each read-only lookup reveals;
    unnamed actions are unsafe (fail closed). `table.probes()`, `table.noop`, `table.spec`.
  - Score functions `nearest_violation` and `unsafe_mass` (exported).
- **`riskgate.provenance`**: prompt-injection gate. `AgentContext`, `ToolOutput`, `ProvenanceWorlds` (one
  hypothesis per untrusted output, scored by an LLM-free `instruction_score`), `ProvenanceModel`, `entities`.
- **`riskgate.replay`**: `replay(gate, episodes)` → `ReplayReport` (follows probes through `after_probe`);
  `resplit(...)` validity study over random (grouped) splits; `sweep(...)`, the error budget as a dial.
- **`riskgate.rules`**: `needs_person`, `never`, `amount_above` (fails closed on non-numbers), `irreversible`, with
  versions pinned to their parameters.
- **`examples/use_cases/`**: payments (UPI), missing-item refund, card dispute, idle-VM cleanup, prompt injection.
  All simulated; run in the test suite.

### Changed
- **Probe selection when nothing is certifiable before the probe** (λ₁ = +∞). The gate used to find no
  "disagreement" between hypotheses, so it never probed. It now judges disagreement by the spec itself (margin ≥ 0),
  the same rule in calibration and deployment. The two-stage bound is unchanged; gates whose λ₁ is finite behave as
  before.

## [1.0.0rc2] — 2026-10-13 (release candidate)

### Added
- **Grouped calibration** — `Gate.calibrate(groups=...)`, for decision points that are not exchangeable on their own,
  such as several tool calls from one agent run, or repeated runs of the same task:
  - the τ split is drawn by group;
  - CRC and LTT run on each group's mean loss.
  - Guarantee: E[share of a new group's decisions that are certified and unsafe] ≤ α.
  - It cannot be combined with `weights`, and two-stage probing is turned off when groups are given.
  - Tests: `tests/test_groups.py`, including validity under strong within-group dependence for both guarantees.
- **`Gate.score_episode(episode, tau=None, tau_ood=None)`** returns the robust score of every candidate. It runs exactly
  the deployed code path (hard rules, the OOD check, fail-closed errors), so you can evaluate offline without recalibrating.
- **`benchmarks/`** evaluates the library on real, logged agent decisions from public benchmarks:
  - τ²-bench, with eight 2026 frontier-model runs and four 2025 runs;
  - AgentDojo prompt-injection runs;
  - ITBench SRE trajectories;
  - RCAEval telemetry, Refund Arena, and the Tennessee Eastman process.
  - Shared protocol: `benchmarks/README.md`. Harness: `benchmarks/common.py`, tested against `Gate.calibrate(groups=...)`.
  - `benchmarks/` is not part of the PyPI distribution; it reads third-party logs in place and never redistributes them.
- **`RiskGateWarning`** (exported): a valid configuration that weakens what the guarantee covers.

### Changed
- **Policy-mode calibration without `doer_id` now warns** (`RiskGateWarning`). The policy-mode guarantee covers only
  the agent that produced the calibration episodes, and without a `doer_id` the fingerprint cannot notice an agent
  swap. The offline replays motivate this: a threshold calibrated on 2025 agents gave 14.6% unsafe on 2026 agents
  (τ²-bench airline, α = 10%), and swapping AgentDojo pipelines without recalibrating exceeded α in 35% of pairs.
  Nothing else changes (same thresholds); set `doer_id`, use agnostic mode, or filter the warning.
- **Packaging:** README links and images are absolute, so they render on PyPI. Added project URLs (repository,
  changelog, issues) and keywords. The release workflow builds once, checks that the tag matches the version, and
  publishes the same artefacts to TestPyPI (manual run) or PyPI (GitHub release).

## [1.0.0rc1] — 2026-10-11 (release candidate)

### Added
- **Calibration**
  - `guarantee="high_probability"`: Learn-then-Test, P(risk ≤ α) ≥ 1 − δ.
  - `extra_losses={name: (fn, alpha_j)}`: control several risks jointly, e.g. safety plus cost.
  - `weights=` with `Gate(weight_fn=...)`: weighted CRC under a known covariate shift.
  - Sample splitting between the plausible-set threshold τ and risk control. This fixes a validity issue in 0.1.
- **Runtime**
  - `timeout_s` (fail-closed), an LRU margin cache (`cache_size`, `context_key`), `parallel` evaluation for models without `predict_batch`, and custom score aggregation (`score_fn`).
  - `doer_id`: part of the fingerprint in policy mode.
  - `Decision.decision_id` and `Gate.record_outcome(...)` for delayed labels; `Gate.recalibrate(...)`.
  - Monitor: pending decisions and observed risk.
- **Artefacts:** HMAC-SHA256 signatures (`save(path, key=...)`, `load(..., key=..., require_signature=True)`).
- **Modules**
  - `riskgate.stl`: signal temporal logic with quantitative robustness.
  - `riskgate.multirisk`: Learn-then-Test p-values and thresholds.
  - `riskgate.weighted`: weighted conformal methods.
  - `riskgate.config`: JSON/TOML/YAML configuration with declarative hard rules (no code evaluation).
  - `riskgate.telemetry.OTelSink`: OpenTelemetry spans and metrics.
  - `riskgate.doers.LLMRanker`: real-LLM doer for any OpenAI-compatible endpoint.
- **Audit:** `MultiSink`, `summarize_audit`, and redaction hooks in `AuditLog`.
- **CLI:** `riskgate version | report | validate | calibrate | audit-summary | serve-mcp`.
- **Adapters:** OpenAI Agents SDK and an MCP proxy server. The LangGraph and Google ADK adapters are verified against the installed framework versions (see `docs/integrations.md`).
- **Examples**
  - `quickstart.py` and `end_to_end_sre.py`, a full lifecycle including drift and recalibration.
  - Three realistic domains with released calibration data, plus resplit validity analysis and ablations.
- **Infrastructure:** CI (lint, types, tests on Python 3.10–3.13, contract tests, examples), a release workflow with PyPI trusted publishing, and pre-commit.

### Changed
- CRC runs over the exact grid of observed scores. There is no artificial floor or ceiling.
- Probe selection maximises the share of hypothesis disagreements a probe resolves.
- **Two-stage probing budget (validity fix).** λ₂ is now calibrated on *all* episodes of its split at
  (1 − probe_share)·α, with zero loss for episodes the gate would not probe and probe harm counted as a constant loss.
  The total risk is ≤ α. The previous version calibrated λ₂ conditionally on probing at level α, which only bounds
  the total by probe_share·α + α·P(probe) (it exceeded α in the reactor example, where ~80% of episodes are probed).
  Probing is calibrated only for `guarantee="expectation"` without extra losses or weights.
- **Calibration mirrors deployment.** Hard-rule screening, the OOD check and fail-closed handling of evaluation errors
  are applied during calibration exactly as in `check()` (errors are counted in `diagnostics["evaluation_failures"]`).
- **Fingerprints cover logic.** Content hashes of hard-rule functions, Custom/Predicate constraints, `score_fn` and
  `weight_fn` (`riskgate.types.callable_digest`; pin with `fn.riskgate_version` or `HardRule(version=...)`).
  Fingerprints no longer depend on memory addresses (deterministic across processes).
- The drift monitor is fed the best robust score of each stage-1 decision (`Decision.top_score`), the same statistic
  as the calibration reference; previously it was fed the evaluated action's score, which caused false alarms.
- `Monitor(latch=True)`: the alarm stays on until recalibration (`bind`) or `reset()`; `bind()` clears pending
  decisions, counters and the derived step size.
- CLI: `validate` exits with 3 when the artefact is VOID (2 is reserved for usage errors); `serve-mcp` gained
  `--name` and `--passthrough`.
- OpenTelemetry: negative margins go to a new `riskgate.margin_deficit` histogram (histograms reject negative values).
- Packaging: SPDX license expression, tested minimum versions for framework extras, `docs` extra.

### Fixed
- An `ASK` hard rule on a lower-ranked candidate no longer lets that candidate be certified.
- `cache_size > 0` now requires a content-derived `context_key` (id-keyed caches could serve stale margins).
- `calibrate(mode=...)` / `load_calibration` set `gate.mode`, so a policy-mode artefact always binds the doer id.
- An exception in the audit sink downgrades a CERTIFY to ASK instead of propagating out of `check()`.
- Policy-mode `extra_losses` keep real values in [0, 1] instead of being binarised.
- Infinite margins are clipped so that λ = +∞ never certifies; NaN robust scores are −∞.
- `guard()` treats a failing context function as ASK and accepts a `feedback=` formatter.
- Config: `noop` may be a table, an empty `[monitor]` creates a default monitor, tables may be nested under `[gate]`,
  calibration artefacts are verified with a key from `calibration_key_env` (`require_signature` supported), and
  `[audit] redact = [...]` drops sensitive fields.

## [0.1.0] — 2026-10-11
- First reference implementation:
  - core gate, CRC in policy and agnostic modes, two-stage probing;
  - monitor, audit log, `guard` decorator, experimental ADK and LangGraph adapters;
  - three example domains.
