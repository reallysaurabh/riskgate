"""Provenance: keep instructions hidden in files, e-mails and web pages from moving money (prompt injection).

The world the gate cannot rule out is "this tool output carries planted instructions". One hypothesis per untrusted
output the agent has seen, scored by how much the output reads like orders (lower score = more instruction-like =
more plausible), plus the nominal world where nothing was planted. Under the hypothesis "output k was planted", a
state-changing call is unsafe if one of its target values (an account, an e-mail address, a URL, a long number) came
from output k and not from the user. Values found in no output and not in the prompt are attributed to every output:
unknown provenance cannot be ruled out.

    ctx = AgentContext(prompt="Pay my rent as the landlord's notice says.",
                       outputs=[ToolOutput("read_file:landlord-notice.txt", text)])
    gate = Gate(ProvenanceModel(), ProvenanceModel.spec, ProvenanceWorlds(), score_fn=nearest_violation,
                doer_id="banking-agent:v3", alpha=0.05)
    gate.calibrate(episodes)                 # logged calls labelled "carried out a planted instruction" or not

With ``score_fn=nearest_violation`` the robust score of a call is the instruction score of the most instruction-like
output it depends on (negated), so a real bill that names an account is certified and a note that tells the agent
what to do is not. The threshold comes from calibration, as everywhere in riskgate. This is the design evaluated on
the public AgentDojo logs in ``benchmarks/agentdojo`` (there with richer cue weights fitted on a development subset).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .spec import Predicate, Spec
from .types import Action, Hypothesis, callable_digest, digest
from .worlds import nearest_violation

__all__ = ["AgentContext", "ProvenanceModel", "ProvenanceWorlds", "ToolOutput", "entities", "instruction_score",
           "nearest_violation"]

# ------------------------------------------------------------------------------------------------ entities
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_URL = re.compile(r"(?<![\w@.-])(?:https?://)?(?:[A-Za-z0-9-]+\.)+(?:com|org|net|io|co|ai|info|biz|edu|gov|me|us|uk|de|"
                  r"ch|fr|it|eu|in|app|dev|xyz|site|online|tech|ly|tv)\b(?::\d+)?(?:/[^\s'\"<>)\]},]*)?", re.I)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_LONGNUM = re.compile(r"(?<![\w.-])\d{9,}(?![\w.-])")
_PATTERNS = (_EMAIL, _URL, _IBAN, _LONGNUM)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip("/."))


def entities(text: str) -> set[str]:
    """Normalised e-mail addresses, URLs / domains, IBANs and long numbers (>= 9 digits) found in `text`."""
    found: set[str] = set()
    for pat in _PATTERNS:
        found.update(_norm(m.group(0)) for m in pat.finditer(text or ""))
    found.update(_norm(e.split("://", 1)[1]) for e in list(found) if "://" in e)        # url without scheme
    return found


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _strings(v)


# ------------------------------------------------------------------------------------------------ instruction cues
_SENT = re.compile(r"(?<=[.!?:;])\s+|\n+")
_IMPERATIVE = frozenset("""send transfer pay wire forward email mail reply delete remove erase update change reset share
invite add create post upload download click visit open go call book cancel schedule grant give approve buy order
ignore disregard forget follow execute run do make use write""".split())
_FILLERS = frozenset("please kindly now also then first and so just".split())
_CUES: tuple[tuple[str, re.Pattern, float], ...] = (
    ("override", re.compile(r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:previous|prior|above|earlier|all|any|other)\b.{0,20}"
                            r"\b(?:instructions?|rules?|tasks?|prompts?)\b", re.I), 2.0),
    ("address_ai", re.compile(r"\b(?:ai|assistant|chatbot|language model|llm|agent|claude|gpt|gemini)\b", re.I), 1.5),
    ("you_must", re.compile(r"\b(?:you (?:must|should|need to|have to|are required to)|make sure (?:to|you)|do not tell|"
                            r"don'?t tell)\b", re.I), 1.0),
    ("task", re.compile(r"\b(?:before you|after you|instead of|your (?:task|instructions?|goal)|the user (?:wants|asked))\b", re.I), 1.0),
    ("urgency", re.compile(r"\b(?:important|urgent(?:ly)?|immediately|asap|right now|at once)\b", re.I), 0.5),
)


def _imperative_with_entity(sentence: str) -> tuple[float, float]:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", sentence)]
    while words and words[0] in _FILLERS:
        words = words[1:]
    imp = float(len(words) >= 3 and words[0] in _IMPERATIVE)
    return imp, float(imp and bool(entities(sentence)))


def instruction_score(text: str, window: int = 6) -> float:
    """How much `text` reads like orders addressed to the agent (>= 0; higher = more instruction-like).

    Per sentence: an imperative opening, an imperative that names an account / address, "ignore previous
    instructions", addressing an AI, "you must", task talk and urgency. The score is the best sum over `window`
    consecutive sentences of weight * log(1 + count), so a long, benign document does not accumulate a high score.
    A deliberately simple, LLM-free scorer: replace it with your own (``ProvenanceWorlds(instruction_score=...)``).
    """
    sentences = [s for s in _SENT.split(text or "") if s.strip()]
    if not sentences:
        return 0.0
    rows = []
    for s in sentences:
        imp, imp_ent = _imperative_with_entity(s)
        rows.append([imp, imp_ent] + [float(len(p.findall(s))) for _, p, _ in _CUES])
    weights = [0.5, 0.5] + [w for _, _, w in _CUES]     # bills also say "please pay to ...": imperatives weigh less
    best = 0.0
    for start in range(max(1, len(rows) - window + 1)):
        counts = [sum(r[j] for r in rows[start:start + window]) for j in range(len(weights))]
        best = max(best, sum(w * math.log1p(c) for w, c in zip(weights, counts)))
    return best


instruction_score.riskgate_version = "instruction-cues-v1"   # type: ignore[attr-defined]


# ------------------------------------------------------------------------------------------------ context
@dataclass(frozen=True)
class ToolOutput:
    """One untrusted output the agent has seen: `source` names it (tool and object), `text` is what it said."""

    source: str
    text: str


@dataclass
class AgentContext:
    """What the gate sees at a tool call.

    prompt          trusted text: the user's request (and the system prompt)
    outputs         untrusted tool outputs visible to the agent so far, in order
    trusted_values  values the principal owns (their own IBAN, mailbox, saved payees): trusted like the prompt
    """

    prompt: str
    outputs: Sequence[ToolOutput] = ()
    trusted_values: Collection[str] = ()
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._trusted = {_norm(v) for v in self.trusted_values} | entities(self.prompt)
        self._prompt_norm = _norm(self.prompt)
        self._output_norm = [_norm(o.text) for o in self.outputs]

    def origins(self, value: str) -> list[int] | None:
        """None if `value` is trusted, else the indices of the outputs it occurs in ([] = found nowhere)."""
        v = _norm(value)
        if not v or v in self._trusted or v in self._prompt_norm:
            return None
        return [k for k, t in enumerate(self._output_norm) if v in t]


# ------------------------------------------------------------------------------------------------ hypotheses
class ProvenanceWorlds:
    """Hypotheses: "nothing planted" (score -nominal) and "output k was planted" (score -instruction_score(k))."""

    def __init__(self, instruction_score: Callable[[str], float] = instruction_score, nominal: float = 2.0):
        if not math.isfinite(nominal):
            raise ValueError(f"nominal must be finite, got {nominal}")
        self.instruction_score, self.nominal = instruction_score, float(nominal)
        self.version = "provenance-" + digest([callable_digest(instruction_score), self.nominal], 12)

    def hypotheses(self, ctx: AgentContext) -> list[Hypothesis]:
        hs = [Hypothesis("nothing_planted", family="none", params={"k": None}, score=-self.nominal)]
        for k, o in enumerate(ctx.outputs):
            hs.append(Hypothesis(f"planted:{k}:{o.source}", family="planted", params={"k": k, "source": o.source},
                                 score=-float(self.instruction_score(o.text))))
        return hs


# ------------------------------------------------------------------------------------------------ consequences
class ProvenanceModel:
    """Under "output k was planted", the call is unsafe if it changes state and one of its target values came from
    output k (or from nowhere). Read-only calls and tools listed in `harmless` never violate.

    targets   tool -> argument names that say who or what receives the effect (recipient, url, iban, ...).
              Tools not listed: every string argument is searched for entities (e-mail, URL, IBAN, long numbers).
    harmless  tool names that never change state (in addition to Action(read_only=True))
    """

    spec = Spec(Predicate("no state change on behalf of untrusted data", lambda o: not o["violations"]))

    def __init__(self, targets: Mapping[str, Sequence[str]] | None = None, harmless: Collection[str] = ()):
        self.targets = {t: tuple(a) for t, a in (targets or {}).items()}
        self.harmless = frozenset(harmless)
        self.version = "provenance-dryrun-" + digest([self.targets, sorted(self.harmless)], 12)

    def target_values(self, action: Action) -> set[str]:
        names = self.targets.get(action.name)
        if names is not None:
            return {s for n in names if n in action.args for s in _strings(action.args[n]) if s.strip()}
        return {e for s in _strings(dict(action.args)) for e in entities(s)}

    def predict(self, action: Action, h: Hypothesis, ctx: AgentContext) -> dict:
        k = h.params.get("k")
        if k is None or action.read_only or action.name in self.harmless:
            return {"violations": []}
        bad = []
        for v in sorted(self.target_values(action)):
            where = ctx.origins(v)
            if where is not None and (k in where or not where):
                bad.append(v)
        return {"violations": bad}
