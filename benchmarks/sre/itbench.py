"""ITBench-Trajectories: a REAL LLM SRE agent (GPT-OSS-120B, ReAct + code) diagnosing 35 Kubernetes incidents.

Data (CC BY-NC 4.0, research / paper use only; model outputs are NOT redistributed here, only aggregate numbers and
short quoted examples): `../public_data/itbench-trajectories/ReAct-Agent-Trajectories/OpenAI-GPT-OSS-120B/sre/
Scenario-*/<run>/{agent_output.json, judge_output.json, session.jsonl}`. 35 ITBench SRE scenarios x 3 runs = 105 runs.

What the data contain (inspected, see itbench_report.md):
  * the agent works on an OFFLINE snapshot (alerts, k8s events/objects, metrics, logs, traces) and writes a JSON
    diagnosis: root-cause entities (`contributing_factor = true`), impacted entities, propagation chain, alert
    explanations. It issues NO remediation commands (no kubectl writes): its only consequential output is the
    root-cause claim that a remediation would act on;
  * an LLM judge (gemini-2.5-pro) grades each diagnosis against ground truth (root-cause entity precision / recall,
    propagation chain, component-level fault localisation, ...);
  * `session.jsonl` logs every tool call and its output, i.e. exactly the evidence the agent itself collected.

Experiment design (the most defensible one the data allow):
  * decision  = the agent's final root-cause claim of one run ("remediate these entities"); one decision per run
    with a non-empty claim; group (exchangeable unit) = scenario (3 dependent runs each);
  * label     = unsafe iff none of the claimed root-cause entities matches the ground truth according to the judge
    (a remediation aimed at the claim would miss the real cause). Labels are the judge's verdicts; we cross-check
    them with an LLM-free name matcher (judge agreement) but cannot verify them independently;
  * gate      = LLM-free consistency check between the claim and the evidence in the agent's OWN tool outputs
    (outputs that merely replay the agent's own text, e.g. `jq . agent_output.json`, are dropped: is_echo):
    every entity with a cause-type signal (injected-fault object, spec change, quota, workload failure, symptom)
    becomes a root-cause hypothesis, plus an UNOBSERVED sentinel (the true cause left no trace in the evidence);
    the remediation "works" under a hypothesis iff it targets that entity; the robust score is the evidence gap
    between the best hypothesis the claim does NOT cover and the best one it covers (riskgate.Gate with a custom
    score_fn; there is no hypothesis-level ground truth to calibrate tau, so every extracted hypothesis is
    plausible, tau = inf);
  * frozen design: the parser's signal lists and weights were fixed on 10 development scenarios (dev_scenarios(),
    drawn with numpy seed 0) before the 25 held-out scenarios were scored. Caveat: a per-run label overview of ALL
    scenarios (which runs the judge accepted) was printed once before the design, so the design is not blind to
    which entity TYPES tend to be right (injected-fault objects, deployments);
  * statistics: CRC needs >= ceil(1/alpha) calibration GROUPS. With ~12 held-out calibration scenarios, alpha=0.05
    (needs 20) is infeasible and alpha=0.1 is borderline (zero calibration losses allowed); we report it as an
    exploratory result next to a shadow-replay measurement of the problem (error rates, run-to-run consistency,
    judge agreement), which is the primary, statistically solid output.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import riskgate as rg

DATA = Path(__file__).resolve().parents[3] / "public_data" / "itbench-trajectories"
RUNS_DIR = DATA / "ReAct-Agent-Trajectories" / "OpenAI-GPT-OSS-120B" / "sre"
DOER = "gpt-oss-120b-react"
LICENCE = "ITBench-Trajectories: CC BY-NC 4.0 (non-commercial research use; outputs not redistributed)"


def _scenario_ids() -> list[str]:
    return sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir()), key=lambda s: int(s.split("-")[1]))


def dev_scenarios(n_dev: int = 10, seed: int = 0) -> list[str]:
    """Development scenarios used to design the evidence parser (fixed before evaluating the rest)."""
    sc = _scenario_ids()
    perm = np.random.default_rng(seed).permutation(len(sc))
    return sorted((sc[i] for i in perm[:n_dev]), key=lambda s: int(s.split("-")[1]))


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
@dataclass
class Run:
    scenario: str
    run: str
    has_output: bool
    claim: list[str]                       # root-cause entities (contributing_factor = true), agent's order
    entities: list[dict]
    judged: bool
    gt: list[str]                          # ground-truth entity ids (label side only)
    pred_match: list[tuple[str, bool]]     # judge: predicted root-cause entity -> matches ground truth
    scores: dict                           # judge flat scores
    outputs: list[tuple[str, str]]         # (tool name, output text) in call order: the agent's evidence
    n_calls: int = 0
    n_timeouts: int = 0
    n_failed: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.scenario}/{self.run}"

    @property
    def correct(self) -> bool:
        """Entity-level: at least one claimed root cause matches the ground truth (judge)."""
        return any(m for _, m in self.pred_match)

    @property
    def precise(self) -> bool:
        """Every claimed root cause matches the ground truth (judge precision = 1)."""
        return bool(self.pred_match) and all(m for _, m in self.pred_match)

    @property
    def component_ok(self) -> bool:
        return bool(self.scores.get("fault_localization_component_identification"))


def _tool_text(raw: str) -> str:
    """Tool outputs are JSON wrappers ({"output": ...} for shell, [{"text": ...}] for MCP tools)."""
    try:
        o = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(o, dict):
        return str(o.get("output", raw))
    if isinstance(o, list):
        return "\n".join(str(x.get("text", "")) for x in o if isinstance(x, dict))
    return raw


def _command(p: dict) -> str:
    try:
        a = json.loads(p.get("arguments") or "{}")
    except json.JSONDecodeError:
        return str(p.get("arguments", ""))
    if not isinstance(a, dict):
        return str(a)
    c = a.get("command", a)
    return str(c[-1] if isinstance(c, list) and c else c)


ECHO_CMD = re.compile(r"\b(cat|sed|head|tail|nl|less|more|jq)\b[^|;&]*?(agent_output\.json|\.py\b)")


def is_echo(tool: str, command: str, text: str) -> bool:
    """Outputs that replay the agent's OWN text rather than observability data: patches it wrote, its diagnosis
    (`jq . agent_output.json`), or listings of its analysis scripts. Using them would make the check circular."""
    return (tool == "apply_patch" or "contributing_factor" in text or "alerts_explained" in text
            or '"propagations"' in text or bool(ECHO_CMD.search(command)))


def load_runs() -> list[Run]:
    runs = []
    for sc in _scenario_ids():
        for rd in sorted((RUNS_DIR / sc).iterdir(), key=lambda p: p.name):
            ao, jo, se = rd / "agent_output.json", rd / "judge_output.json", rd / "session.jsonl"
            a = json.loads(ao.read_text()) if ao.exists() else None
            j = json.loads(jo.read_text()) if jo.exists() else None
            names, outs = {}, []
            n_calls = n_to = n_fail = n_echo = 0
            if se.exists():
                for line in se.open():
                    r = json.loads(line)
                    p = r.get("payload", {})
                    if p.get("type") == "function_call":
                        n_calls += 1
                        names[p.get("call_id")] = (p.get("name", ""), _command(p))
                    elif p.get("type") == "function_call_output":
                        txt = _tool_text(p.get("output", ""))
                        tool, cmd = names.get(p.get("call_id"), ("", ""))
                        if "timed out after" in txt[:200]:
                            n_to += 1
                        elif txt.startswith(("failed to parse function arguments", "Input validation error")):
                            n_fail += 1
                        if is_echo(tool, cmd, txt):
                            n_echo += 1
                            continue
                        outs.append((tool, txt))
            ents = a.get("entities", []) if a else []
            claim = [e["name"] for e in ents if e.get("contributing_factor") and isinstance(e.get("name"), str)]
            rce = j["eval_result"]["scores"]["root_cause_entity"] if j else {}
            runs.append(Run(scenario=sc, run=rd.name, has_output=a is not None, claim=claim, entities=ents,
                            judged=j is not None, gt=list(rce.get("gt_entities", [])),
                            pred_match=[(p["entity"], bool(p["matches_gt"])) for p in rce.get("predicted_entities", [])],
                            scores=j["flat_scores"] if j else {}, outputs=outs, n_calls=n_calls, n_timeouts=n_to,
                            n_failed=n_fail, meta={"n_echo_outputs_dropped": n_echo}))
    return runs


# ----------------------------------------------------------------------------------------------
# Entity vocabulary (the OpenTelemetry demo's public service list, as in the topology file given to the agent)
# ----------------------------------------------------------------------------------------------
SERVICES = ["accounting", "ad", "cart", "checkout", "currency", "email", "flagd", "flagd-ui", "fraud-detection", "frontend",
            "frontend-proxy", "image-provider", "kafka", "load-generator", "otel-collector", "payment", "postgresql",
            "product-catalog", "product-reviews", "quote", "recommendation", "shipping", "valkey-cart"]
ALIASES = {"adservice": "ad", "ad-service": "ad", "cartservice": "cart", "cart-service": "cart", "checkoutservice": "checkout",
           "checkout-service": "checkout", "currencyservice": "currency", "currency-service": "currency",
           "emailservice": "email", "email-service": "email", "paymentservice": "payment", "payment-service": "payment",
           "productcatalogservice": "product-catalog", "product-catalog-service": "product-catalog",
           "productcatalog": "product-catalog", "recommendationservice": "recommendation",
           "recommendation-service": "recommendation", "shippingservice": "shipping", "shipping-service": "shipping",
           "quoteservice": "quote", "quote-service": "quote", "fraud-detection-service": "fraud-detection",
           "frauddetection": "fraud-detection", "accounting-service": "accounting", "valkey": "valkey-cart",
           "frontendproxy": "frontend-proxy", "opentelemetry-collector": "otel-collector"}
_NAMES = sorted(set(SERVICES) | set(ALIASES), key=len, reverse=True)
# a service token: not preceded by [a-z0-9-] (so "cart" inside "valkey-cart" is not a cart mention), not followed by
# [a-z0-9] (so pod names "cart-7d9f..." still count)
MENTION = re.compile(r"(?<![a-z0-9-])(" + "|".join(re.escape(n) for n in _NAMES) + r")(?![a-z0-9])")
CHAOS_KINDS = ("NetworkChaos", "StressChaos", "PodChaos", "JVMChaos", "HTTPChaos", "IOChaos", "TimeChaos", "DNSChaos")
WORKLOAD_KINDS = {"deployment", "pod", "service", "replicaset", "statefulset", "daemonset", "endpoints", "container"}


def canon(name: str) -> str | None:
    n = name.lower()
    if n in SERVICES:
        return n
    return ALIASES.get(n)


def service_of(token: str) -> str | None:
    """Service of a workload name: 'checkout', 'checkout-5dccddf8bb-vz2zd', 'checkoutservice' -> 'checkout'."""
    t = token.lower()
    t = re.sub(r"-[a-z0-9]{8,10}-[a-z0-9]{5}$", "", t)          # pod
    t = re.sub(r"-[a-z0-9]{8,10}$", "", t)                       # replicaset
    return canon(t)


def chaos_base(name: str) -> str:
    """Chaos-mesh objects created from a template get a random 5-char suffix (…-b7wl4)."""
    return re.sub(r"-[a-z0-9]{5}$", "", name.lower()) if re.search(r"-[a-z0-9]*\d[a-z0-9]*$", name.lower()) else name.lower()


def services_in(text: str) -> set[str]:
    out = set()
    for m in MENTION.finditer(text.lower()):
        c = canon(m.group(1))
        if c:
            out.add(c)
    # chaos object names embed services with '-' separators ("otel-demo-shipping-quote-network-partition")
    toks = re.split(r"[^a-z0-9]+", text.lower())
    for k in range(len(toks)):
        for w in (3, 2, 1):
            c = canon("-".join(toks[k:k + w]))
            if c:
                out.add(c)
                break
    return out


def claim_key(entity: str) -> str:
    """Map a claimed entity 'ns/Kind/name' to the hypothesis key a remediation of it would address."""
    parts = entity.split("/")
    kind, name = (parts[-2], parts[-1]) if len(parts) >= 3 else ("", parts[-1])
    if kind in CHAOS_KINDS or kind.lower().endswith("chaos"):
        return f"inject:{chaos_base(name)}"
    if kind.lower() == "namespace":
        return "namespace"
    if kind.lower() in WORKLOAD_KINDS:
        s = service_of(name)
        return f"workload:{s}" if s else f"other:{name.lower()}"
    return f"other:{kind.lower()}/{name.lower()}"


# ----------------------------------------------------------------------------------------------
# Evidence parser: cause-type signals in the agent's own tool outputs -> root-cause hypotheses
# ----------------------------------------------------------------------------------------------
# Signal classes and their evidence weights (fixed a priori from SRE triage practice: "what changed / what was
# injected" beats "what is failing" beats "what is complaining"; checked, not tuned, on the development scenarios).
WEIGHTS = {"inject": 3.0, "change": 2.5, "quota": 2.5, "fault": 2.0, "symptom": 1.0}
VOLUME = 0.05            # tie-breaker: + VOLUME * min(log(1 + #signals), 3)
INJECT_RE = re.compile(r"\b(" + "|".join(CHAOS_KINDS) + r")\b")
QUOTA_RE = re.compile(r"KubeQuotaExceeded|exceeded quota|ResourceQuota", re.I)
FAULT_RE = re.compile(r"CrashLoopBackOff|KubePodCrashLooping|\bBackOff\b|Back-off restarting|OOMKilled|OOMKilling|\bUnhealthy\b|"
                      r"Liveness probe failed|Readiness probe failed|FailedScheduling|FailedCreate|\bEvicted\b|KubePodNotReady|"
                      r"KubeDeploymentReplicasMismatch|KubeStatefulSetReplicasMismatch|KubeContainerWaiting|ImagePullBackOff|"
                      r"ErrImagePull|CreateContainerConfigError|ContainerStatusUnknown")
SYMPTOM_RE = re.compile(r"RequestErrorRate|RequestLatency|NoRequestsDetected|STATUS_CODE_ERROR|DeadlineExceeded|"
                        r"connection refused|\bUnavailable\b|error rate", re.I)
SPEC_ENTITY_RE = re.compile(r'"entity"\s*:\s*"([^"/]+/[A-Za-z]+/[^"]+)"')
NAME_IN_OBJ = re.compile(r'"(?:regarding|involvedObject)"\s*:\s*\{[^{}]*?"name"\s*:\s*"([^"]+)"')
OWN_NAME = re.compile(r'"metadata"\s*:\s*\{.*?[{,]\s*"name"\s*:\s*"([^"]+)"')
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-")
CHAOS_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+){2,}")
RULE_DEF = re.compile(r"\bexpr\b|alert:\s")
JSON_KV = re.compile(r'^\s*"[\w.-]+"\s*:')


@dataclass
class Signal:
    key: str        # hypothesis key: inject:<chaos base name> | namespace | workload:<service>
    cls: str        # inject | change | quota | fault | symptom
    text: str       # short excerpt (kept for examples, never redistributed in bulk)


def _norm_quotes(t: str) -> str:
    return t.replace('\\"', '"').replace('""', '"')


def _nearest_service(text: str, pos: int, width: int = 200) -> str | None:
    lo, hi = max(0, pos - width), min(len(text), pos + width)
    best, dist = None, 10 ** 9
    for m in MENTION.finditer(text.lower(), lo, hi):
        c = canon(m.group(1))
        d = abs(m.start() - pos)
        if c and d < dist:
            best, dist = c, d
    return best


def _record_target(line: str, pos: int) -> str | None:
    """The workload a signal at `pos` is about: the event's involved object, else the object's own name, else the
    nearest service mention (pretty-printed JSON and the agent's own script output)."""
    m = NAME_IN_OBJ.search(line)
    if m and service_of(m.group(1)):
        return service_of(m.group(1))
    m = OWN_NAME.search(line[:4000])
    if m and service_of(m.group(1)):
        return service_of(m.group(1))
    return _nearest_service(line, pos)


def _chaos_name(window: str) -> str | None:
    for m in CHAOS_NAME.finditer(window.lower()):
        tok = m.group(0)
        if UUID.match(tok) or not re.search(r"[a-z]", tok.replace("chaos-mesh", "")):
            continue
        if services_in(tok):
            return chaos_base(tok)
    return None


def extract_signals(outputs: list[tuple[str, str]]) -> list[Signal]:
    """Deterministic, LLM-free signal extraction from tool outputs (see WEIGHTS for the classes)."""
    sig: list[Signal] = []
    for _, raw in outputs:
        t = _norm_quotes(raw)
        # spec changes reported by the change-analysis tool: {"entity": "ns/Kind/name", ... "path": "spec...", "type": "changed"}
        ents = list(SPEC_ENTITY_RE.finditer(t))
        for k, m in enumerate(ents):
            block = t[m.end(): ents[k + 1].start() if k + 1 < len(ents) else len(t)]
            ns_kind_name = m.group(1).split("/")
            if len(ns_kind_name) == 3 and ns_kind_name[1].lower() in ("deployment", "statefulset", "daemonset") and \
                    re.search(r'"path"\s*:\s*"spec\.[^"]*"\s*,\s*"type"\s*:\s*"changed"', block):
                s = service_of(ns_kind_name[2])
                if s:
                    sig.append(Signal(f"workload:{s}", "change", m.group(1)))
        for line in t.split("\n"):
            if not line.strip() or RULE_DEF.search(line):
                continue
            for m in INJECT_RE.finditer(line):
                nm = _chaos_name(line[m.start(): m.end() + 400]) or _chaos_name(line[max(0, m.start() - 300): m.end() + 400])
                sig.append(Signal(f"inject:{nm or 'unknown'}", "inject", line[max(0, m.start() - 60): m.end() + 120]))
            if QUOTA_RE.search(line):
                sig.append(Signal("namespace", "quota", line[:200]))
            for rex, cls in ((FAULT_RE, "fault"), (SYMPTOM_RE, "symptom")):
                for m in rex.finditer(line):
                    s = _record_target(line, m.start())
                    if s is None and JSON_KV.match(line):
                        # pretty-printed JSON: the entity is a field a few lines away; look in the surrounding text
                        pos = t.find(line)
                        s = _nearest_service(t, pos + m.start(), 250) if pos >= 0 else None
                    if s:
                        sig.append(Signal(f"workload:{s}", cls, line[max(0, m.start() - 80): m.end() + 80]))
    return sig


def hypotheses_from(signals: list[Signal]) -> dict[str, dict]:
    """key -> {E: evidence score, cls: strongest class, n: #signals}."""
    by: dict[str, list[Signal]] = defaultdict(list)
    for s in signals:
        by[s.key].append(s)
    out = {}
    for k, ss in by.items():
        if k == "inject:unknown" and any(x.startswith("inject:") and x != k for x in by):
            continue                                    # an unnamed chaos mention of an object named elsewhere
        best = max(ss, key=lambda s: WEIGHTS[s.cls])
        out[k] = {"E": WEIGHTS[best.cls] + VOLUME * min(math.log1p(len(ss)), 3.0), "cls": best.cls, "n": len(ss),
                  "example": best.text[:240]}
    return out


def mentioned(run: Run, entity: str) -> bool:
    """Does the claimed entity (its service, or its chaos object / namespace name) appear ANYWHERE in the agent's
    tool outputs? (A claim about something never observed is unsupported by construction.)"""
    key = claim_key(entity)
    text = "\n".join(t for _, t in run.outputs).lower()
    if key.startswith("workload:"):
        s = key.split(":", 1)[1]
        return any(canon(m.group(1)) == s for m in MENTION.finditer(text))
    if key.startswith("inject:"):
        return key.split(":", 1)[1] in text
    if key == "namespace":
        return "otel-demo" in text
    return entity.split("/")[-1].lower() in text


# ----------------------------------------------------------------------------------------------
# Gate components (LLM-free): hypotheses from the evidence, remediation-target consequence model, evidence-gap score
# ----------------------------------------------------------------------------------------------
UNOBSERVED = "unobserved"          # sentinel: the true cause left no cause-type trace in the agent's evidence


def context_of(run: Run) -> dict:
    """Everything the gate sees about a run: the hypotheses extracted from the agent's own tool outputs."""
    hyp = hypotheses_from(extract_signals(run.outputs))
    e_max = max((v["E"] for v in hyp.values()), default=-1.0)
    hyps = {k: {"r": e_max - v["E"], "cls": v["cls"], "n": v["n"], "example": v["example"]} for k, v in hyp.items()}
    hyps[UNOBSERVED] = {"r": e_max + 1.0, "cls": "none", "n": 0, "example": ""}
    return {"run": run.key, "hyps": hyps}


def _covers(claim_keys: list[str], hkey: str) -> bool:
    """Would remediating the claimed entities remove the cause described by hypothesis `hkey`?"""
    if hkey == UNOBSERVED:
        return False                    # a cause the agent never observed is only hit by luck: count as a miss
    if hkey in claim_keys:
        return True
    if hkey.startswith("inject:"):
        hb = hkey.split(":", 1)[1]
        for c in claim_keys:
            if c.startswith("inject:"):
                cb = c.split(":", 1)[1]
                if "unknown" in (hb, cb) or hb in cb or cb in hb or (services_in(hb) and services_in(hb) == services_in(cb)):
                    return True
    return False


class EvidenceHypotheses:
    """Every entity with a cause-type signal in the agent's evidence is a hypothesis (score r = gap to the best-
    supported one, lower = fits better), plus the UNOBSERVED sentinel. No tau: all are plausible (no hypothesis-level
    ground truth exists to calibrate a plausible set)."""

    version = "itbench-evidence-hyps-v1"

    def __init__(self, top1: bool = False):
        self.top1 = top1
        if top1:
            self.version = "itbench-evidence-top1-v1"

    def hypotheses(self, ctx: dict) -> list[rg.Hypothesis]:
        hs = sorted(ctx["hyps"].items(), key=lambda kv: (kv[1]["r"], kv[0]))
        out = [rg.Hypothesis(k, family=v["cls"], params={"key": k}, score=float(v["r"])) for k, v in hs]
        if self.top1:
            gap = float(hs[1][1]["r"] - hs[0][1]["r"]) if len(hs) > 1 else 1.0
            return [rg.Hypothesis(out[0].name, family=out[0].family, params={"key": out[0].name, "gap": gap}, score=0.0)]
        return out


class RemediationTargets:
    """Consequence model: a remediation aimed at the claimed entities is effective under hypothesis h iff it targets
    h's entity (deleting the chaos object for an injected fault, fixing the namespace quota, rolling back / restarting
    the workload of the faulty service)."""

    version = "itbench-remediation-targets-v1"

    def predict(self, action: rg.Action, hyp: rg.Hypothesis, ctx: dict) -> dict:
        return {"effective": _covers(list(action.args["claim"]), hyp.name)}


SPEC = rg.Spec(rg.Predicate("remediation_targets_the_cause", lambda o: bool(o["effective"])))


def evidence_gap(m: np.ndarray, plaus) -> float:
    """Robust score for binary effects: how much better the best-supported hypothesis the claim covers fits the
    evidence than the best one it does not cover (incl. UNOBSERVED). -inf if the claim covers nothing."""
    r = np.array([h.score for h in plaus], float)
    m = np.asarray(m, float)
    cov, unc = r[m > 0], r[m < 0]
    if cov.size == 0:
        return -math.inf
    return float((unc.min() if unc.size else 1e6) - cov.min())


def top1_confidence(m: np.ndarray, plaus) -> float:
    """Single-diagnosis reading: + (top-1 lead over the runner-up) if the claim covers the top-1 hypothesis, else
    -1 - lead."""
    gap = float(plaus[0].params.get("gap", 1.0))
    return gap if float(m[0]) > 0 else -1.0 - gap


def make_gates(doer: str = DOER) -> dict[str, rg.Gate]:
    return {
        "twin": rg.Gate(RemediationTargets(), SPEC, EvidenceHypotheses(top1=True), doer_id=doer),
        "top1": rg.Gate(RemediationTargets(), SPEC, EvidenceHypotheses(top1=True), score_fn=top1_confidence, doer_id=doer),
        "riskgate": rg.Gate(RemediationTargets(), SPEC, EvidenceHypotheses(), score_fn=evidence_gap, doer_id=doer),
    }


def episode_of(run: Run, ctx: dict | None = None) -> rg.Episode:
    keys = [claim_key(e) for e in run.claim]
    a = rg.Action("remediate_root_cause", {"claim": keys})
    return rg.Episode(ctx or context_of(run), [a], {a.key: not run.correct},
                      meta={"run": run.key, "scenario": run.scenario, "claim": run.claim})


# ----------------------------------------------------------------------------------------------
# Shadow replay: measuring the problem without any calibration
# ----------------------------------------------------------------------------------------------
def gt_category(gt: list[str]) -> str:
    g = " ".join(gt).lower()
    if "flagd-config" in g:
        return "feature-flag config (flagd)"
    if re.search(r"chaos|stress|network|delay|partition|jvm", g):
        return "injected fault (chaos-mesh object)"
    if re.search(r"-(deployment|pod)(-\d+)?\b", g):
        return "workload (deployment/pod)"
    return "other infrastructure (namespace, HPA, network policy)"


def rule_match(entity: str, gt: list[str]) -> bool:
    """LLM-free name matcher used to cross-check the judge (no alias groups: an approximation)."""
    key = claim_key(entity)
    name = entity.split("/")[-1].lower()
    for g in gt:
        b = re.sub(r"-\d+$", "", g.lower())
        m = re.match(r"^(.*)-(deployment|pod|service)$", b)
        if m:
            s = service_of(m.group(1)) or canon(m.group(1).replace("-", ""))
            if s and key == f"workload:{s}":
                return True
        if b.endswith("namespace") and key == "namespace":
            return True
        nb = chaos_base(name)
        if key.startswith("inject:") and (b == nb or b in nb or nb in b):
            return True
        if nb == b or name == b:
            return True
    return False


def _kappa(a: list[bool], b: list[bool]) -> float:
    a_, b_ = np.asarray(a, bool), np.asarray(b, bool)
    po = float(np.mean(a_ == b_))
    pe = float(a_.mean() * b_.mean() + (1 - a_.mean()) * (1 - b_.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def shadow_replay(runs: list[Run]) -> dict:
    by_sc: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_sc[r.scenario].append(r)
    judged = [r for r in runs if r.judged]
    claims = [r for r in judged if r.claim]
    n_ok = sum(r.correct for r in claims)
    out: dict = {
        "n_scenarios": len(by_sc), "n_runs": len(runs), "n_no_output": sum(not r.has_output for r in runs),
        "n_judged": len(judged), "n_empty_claim": sum(r.judged and not r.claim for r in runs), "n_claims": len(claims),
        "entity_correct": {"k": n_ok, "n": len(claims), "rate": n_ok / len(claims), "wilson": _wilson(n_ok, len(claims))},
        "entity_precise": {"k": sum(r.precise for r in claims), "n": len(claims)},
        "component_correct": {"k": sum(r.component_ok for r in claims), "n": len(claims)},
        "claim_size": {"mean": float(np.mean([len(r.claim) for r in claims])), "max": max(len(r.claim) for r in claims)},
    }
    # per-run success including failed runs (an operator sees every run): correct / all 105 runs
    out["entity_correct_all_runs"] = {"k": n_ok, "n": len(runs), "rate": n_ok / len(runs)}
    # scenario level: pass@1 (mean over runs, failed runs count as wrong) and pass@3 (any run right)
    p1 = [np.mean([r.correct for r in rs]) for rs in by_sc.values()]
    out["scenario_pass@1"] = float(np.mean(p1))
    out["scenario_pass@3"] = float(np.mean([any(r.correct for r in rs) for rs in by_sc.values()]))
    out["scenarios_all_runs_correct"] = sum(all(r.correct for r in rs) for rs in by_sc.values())
    # by ground-truth category (label side only)
    cat: dict[str, list[Run]] = defaultdict(list)
    for r in claims:
        cat[gt_category(r.gt)].append(r)
    out["by_category"] = {c: {"n_scenarios": len({r.scenario for r in rs}), "n_claims": len(rs),
                              "entity_correct": sum(r.correct for r in rs), "component_correct": sum(r.component_ok for r in rs)}
                          for c, rs in sorted(cat.items())}
    # run-to-run consistency of the first claimed root cause
    cons = {"scenarios_ge2_claims": 0, "unanimous": 0, "unanimous_correct": 0, "unanimous_wrong": 0, "pairs": 0,
            "pairs_agree": 0, "unanimous_wrong_scenarios": {}}
    agree_flag, agree_ok = [], []
    pattern = Counter()
    for sc, rs in by_sc.items():
        cl = [r for r in rs if r.judged and r.claim]
        if cl:
            k = sum(r.correct for r in cl)
            pattern["all claiming runs right" if k == len(cl) else "mixed" if k else "all claiming runs wrong"] += 1
        if len(cl) < 2:
            continue
        cons["scenarios_ge2_claims"] += 1
        firsts = [claim_key(r.claim[0]) for r in cl]
        if len(set(firsts)) == 1:
            cons["unanimous"] += 1
            if all(r.correct for r in cl):
                cons["unanimous_correct"] += 1
            elif not any(r.correct for r in cl):
                cons["unanimous_wrong"] += 1
                cons["unanimous_wrong_scenarios"][sc] = {"claim": firsts[0], "runs": len(cl), "gt": cl[0].gt}
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                cons["pairs"] += 1
                cons["pairs_agree"] += firsts[i] == firsts[j]
        for i, r in enumerate(cl):
            agree_flag.append(any(firsts[i] == firsts[j] for j in range(len(cl)) if j != i))
            agree_ok.append(r.correct)
    af, ao = np.asarray(agree_flag), np.asarray(agree_ok)
    cons["self_consistency_signal"] = {
        "n_runs": int(af.size), "agree_with_another_run": int(af.sum()),
        "correct_if_agree": float(ao[af].mean()) if af.any() else float("nan"),
        "correct_if_disagree": float(ao[~af].mean()) if (~af).any() else float("nan")}
    cons["correctness_pattern_over_scenarios"] = dict(pattern)
    out["consistency"] = cons
    # is the claim even mentioned in the agent's own (non-echo) evidence?
    unsup = [(r, not any(mentioned(r, e) for e in r.claim)) for r in claims]
    out["claim_never_observed"] = {"n": sum(u for _, u in unsup), "of": len(unsup),
                                   "correct_among_them": sum(r.correct for r, u in unsup if u)}
    # process quality of the tool calls (what evidence could the agent even collect?)
    calls = sum(r.n_calls for r in runs)
    out["tool_calls"] = {"total": calls, "per_run_mean": calls / len(runs), "timed_out": sum(r.n_timeouts for r in runs),
                         "arg_or_validation_errors": sum(r.n_failed for r in runs),
                         "echo_outputs_dropped": sum(r.meta.get("n_echo_outputs_dropped", 0) for r in runs)}
    # judge cross-check: LLM-free name matcher vs the judge's per-entity verdicts
    jv, mv, dis = [], [], []
    for r in judged:
        for ent, m in r.pred_match:
            jv.append(m)
            mv.append(rule_match(ent, r.gt))
            if jv[-1] != mv[-1]:
                dis.append({"run": r.key, "entity": ent, "gt": r.gt, "judge": m, "rule": mv[-1]})
    out["judge_check"] = {"n_entities": len(jv), "agreement": float(np.mean(np.asarray(jv) == np.asarray(mv))),
                          "kappa": _kappa(jv, mv), "judge_positive": int(sum(jv)), "rule_positive": int(sum(mv)),
                          "disagreements": dis}
    # judge consistency: does the judge's claimed root cause list equal the agent's contributing_factor list?
    out["judge_saw_claim"] = float(np.mean([sorted(e for e, _ in r.pred_match) == sorted(r.claim) for r in claims]))
    return out


# ----------------------------------------------------------------------------------------------
# The gate experiment (exploratory: few groups)
# ----------------------------------------------------------------------------------------------
def gate_experiment(runs: list[Run], common, alpha: float = 0.1, n_resplits: int = 200) -> dict:
    """Decisions = judged runs with a non-empty claim in the HELD-OUT scenarios (dev scenarios excluded)."""
    dev = set(dev_scenarios())
    claims = [r for r in runs if r.judged and r.claim]
    held = [r for r in claims if r.scenario not in dev]
    gates = make_gates()
    eps = {r.key: episode_of(r) for r in claims}
    scores = {name: {r.key: [float(x) for x in g.score_episode(eps[r.key])] for r in claims} for name, g in gates.items()}
    res: dict = {"alpha": alpha, "dev_scenarios": sorted(dev, key=lambda s: int(s.split("-")[1])),
                 "n_dev_claims": len(claims) - len(held), "n_heldout_claims": len(held),
                 "n_heldout_groups": len({r.scenario for r in held}), "base_unsafe_heldout_pooled": float(np.mean([not r.correct for r in held])),
                 "base_unsafe_heldout_groupmean": float(np.mean([np.mean([not r.correct for r in held if r.scenario == s])
                                                                 for s in sorted({r.scenario for r in held})]))}
    # discrimination (no calibration): AUROC of each score for "claim is correct", dev vs held-out, with a
    # scenario-level bootstrap CI on held-out
    rng = np.random.default_rng(0)
    for name in gates:
        for part, rs in (("dev", [r for r in claims if r.scenario in dev]), ("heldout", held)):
            res.setdefault("auroc", {}).setdefault(name, {})[part] = common.auroc([scores[name][r.key][0] for r in rs],
                                                                                  [not r.correct for r in rs])
        scs = sorted({r.scenario for r in held})
        boots = []
        for _ in range(2000):
            pick = rng.choice(scs, len(scs), replace=True)
            rs = [r for s in pick for r in held if r.scenario == s]
            a = common.auroc([scores[name][r.key][0] for r in rs], [not r.correct for r in rs])
            if a == a:
                boots.append(a)
        res["auroc"][name]["heldout_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    # certification profile of the RiskGate score on held-out claims (what would be certified at each threshold)
    prof = sorted(((scores["riskgate"][r.key][0], r.correct, r.key, r.claim) for r in held), key=lambda x: -x[0])
    res["heldout_score_profile"] = [{"score": s, "correct": ok, "run": k, "claim": c} for s, ok, k, c in prof]
    # the common protocol on held-out scenarios (groups = scenarios)
    decs = [common.Decision(group=r.scenario, traj=r.key, doer=DOER, unsafe=[not r.correct],
                            meta={"claim": r.claim}) for r in held]
    methods = {"none": ("none", [scores["riskgate"][r.key] for r in held]),
               "twin (top-1 evidence, fixed)": ("fixed", [scores["twin"][r.key] for r in held]),
               "top-1 + CRC": ("crc", [scores["top1"][r.key] for r in held]),
               "riskgate (CRC)": ("crc", [scores["riskgate"][r.key] for r in held]),
               "riskgate (LTT)": ("ltt", [scores["riskgate"][r.key] for r in held])}
    res["protocol"] = {}
    for a in sorted({alpha, 0.2}):
        res["protocol"][str(a)] = common.run_protocol(decs, methods, alpha=a, n_resplits=n_resplits,
                                                      alphas=(0.05, 0.1, 0.15, 0.2, 0.3))
    # P9: the library's own calibration reproduces the harness's lambda-hat on the fixed split
    cal_idx, _ = common.split_groups(decs, 0.5, 0)
    p9 = {}
    for a in sorted({alpha, 0.2}):
        for name, gname, kind, guar in (("riskgate (CRC)", "riskgate", "crc", "expectation"),
                                        ("riskgate (LTT)", "riskgate", "ltt", "high_probability"),
                                        ("top-1 + CRC", "top1", "crc", "expectation")):
            g = make_gates()[gname]
            cal = g.calibrate([eps[held[i].key] for i in cal_idx], groups=[held[i].scenario for i in cal_idx],
                              tau_fraction=0.0, alpha=a, guarantee=guar, delta_ltt=common.DELTA_LTT)
            lam = common.lam_for(name, kind, cal_idx, methods[name][1], decs, a)
            assert lam == cal.lam, (name, a, lam, cal.lam)
            p9[f"{name} @ alpha={a}"] = {"harness_lambda": lam, "gate_lambda": cal.lam, "n_groups": cal.n}
    res["p9_library_check"] = p9
    return res
