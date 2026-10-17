# FAQ

**Is this a guardrail like Llama Guard or NeMo Guardrails?**
No. Those check *content* and *permissions*: is this prompt malicious, is this tool allowed. riskgate checks *consequences*: what will this action do to the system, and how often is that judgement wrong? Use both. Your allow-lists and policy rules become riskgate `HardRule`s, evaluated before anything statistical.

**Does riskgate call an LLM?**
Not when certifying (design rule R1). Your agent can be any LLM; riskgate never trusts it. `riskgate.doers.LLMRanker` exists only to let a real model *propose* actions in the examples.

**What exactly is guaranteed?**
For exchangeable episodes, the expected rate of executed, certified actions that turn out unsafe is at most α. In agnostic mode this holds for any agent; in policy mode, for the agent you calibrated with. With `guarantee="high_probability"` it holds with probability 1 − δ for the artefact you deployed. It is not a per-situation guarantee, it does not survive arbitrary distribution shift, and it relies on correct labels. See [Calibration & guarantees](calibration.md).

**My simulator is inaccurate. Is the guarantee useless?**
No. Validity does not depend on model accuracy; calibration measures how wrong the model is and sets the threshold accordingly. An inaccurate model makes the gate ask more often. That is the honest price.

**Why so many ASKs?**
Usually one of three reasons:
1. the plausible set is wide (diagnosis is ambiguous): add probes or better evidence;
2. the consequence model is noisy: improve it, or add locally adaptive scores (`score_fn`);
3. α is very small or n is small.
The [results](results.md) show ask rates of about 23–34% at α = 5% in our simulated domains (single-stage, re-split means); probing removed 14–29% of them.

**What happens when I change my model, prompt or spec?**
The gate's fingerprint changes and the status becomes `VOID`. Every check returns `ASK` until you recalibrate. This is deliberate (rule R4). Bump the `version` string of your models whenever they change. The code of hard rules, custom constraints and score functions is hashed automatically, so editing a rule also voids the artefact; set `fn.riskgate_version = "..."` (or `HardRule(version=...)`) to pin an identity explicitly. In policy mode, pass `doer_id` (model + prompt version) so that changing the LLM voids it too.

**Is the 5% guaranteed for my deployment, or only on average?**
By default (conformal risk control) on average over calibration sets: one particular calibrated gate can be above α, by more when many candidates share almost the same score near the threshold (see the reactor example in [Results](results.md)). Use `guarantee="high_probability"` to get P(risk ≤ α) ≥ 1 − δ for the artefact you deploy, at the cost of more `ASK`s when the evidence is thin.

**How do I get labels without a simulator?**
- Sandbox replays: staging namespaces, database savepoints, payment test modes.
- Audited production outcomes, using policy mode.
- Expert labelling of proposed alternatives.

Off-policy conformal methods for logs that only contain executed actions are on the research roadmap.

**Can an attacker prompt-inject the agent into getting a dangerous action certified?**
The agent can *propose* anything. The certify decision depends only on your models, hypotheses, spec and calibration, never on text the LLM produced. If an injected action is unsafe under the plausible hypotheses, it is not certified.

**How fast is it?**
Gate overhead is about 0.1 ms per check (p95 below 0.2 ms) with a trivial model. Your consequence model dominates: in our examples a check takes about 0.8 ms with a SQLite dry-run, 10 ms with a queueing twin, and about 0.1 s for 19 hypotheses × 7 actions on a reactor twin.

**Does it work with LangGraph / OpenAI Agents SDK / Google ADK / MCP?**
Yes. The `guard` decorator works with any framework, and there are dedicated adapters plus an MCP proxy. See [Integrations](integrations.md).

**Is riskgate certified for safety-critical use?**
No. It is a research-grade supervisory layer. Keep certified safety systems (SIS, interlocks), access control and human oversight in place.
