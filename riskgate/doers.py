"""LLM doers: let a real language model propose / rank candidate actions (no SDK dependency).

The doer is deliberately *outside* the trust boundary: whatever it proposes is checked by the gate.
Use `doer_id` as Gate(doer_id=...) in policy mode, so that changing the model or prompt voids the
policy-mode calibration (agnostic mode stays valid for any doer).

    ranker = LLMRanker()                                  # env: RISKGATE_LLM_BASE_URL / _API_KEY / _MODEL
    ranked = ranker.rank("p95 latency 2.1 s, DB CPU 96 %, deploy 12 min ago", candidates, policy=RUNBOOK)
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Sequence

from .types import Action, digest

DEFAULT_SYSTEM = ("You are an operations assistant. Rank the candidate actions from most to least appropriate "
                  "for the situation, following the policy. Respond ONLY with JSON: "
                  '{"ranked": [<candidate index>, ...], "rationale": "<one sentence>"}')


class LLMRanker:
    """Ranks candidate actions with any OpenAI-compatible /chat/completions endpoint (OpenAI, Gemini's
    OpenAI-compatible API, vLLM, Ollama, LM Studio, ...). Falls back to the given order on any failure."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, temperature: float = 0.0, system_prompt: str = DEFAULT_SYSTEM):
        self.base_url = (base_url or os.environ.get("RISKGATE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("RISKGATE_LLM_API_KEY", "")
        self.model = model or os.environ.get("RISKGATE_LLM_MODEL", "gpt-4o-mini")
        self.timeout, self.temperature, self.system_prompt = timeout, temperature, system_prompt
        self.last_rationale: str = ""
        self.last_error: str | None = None

    @property
    def doer_id(self) -> str:
        return f"llm:{self.model}:t{self.temperature}:{digest(self.system_prompt, 8)}"

    def _prompt(self, situation: str, candidates: Sequence[Action], policy: str) -> str:
        lines = [f"{i}: {a.name} {json.dumps(dict(a.args))}" for i, a in enumerate(candidates)]
        return f"Policy:\n{policy}\n\nSituation:\n{situation}\n\nCandidates:\n" + "\n".join(lines)

    def rank(self, situation: str, candidates: Sequence[Action], policy: str = "") -> list[Action]:
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "system", "content": self.system_prompt},
                                        {"role": "user", "content": self._prompt(situation, candidates, policy)}]}).encode()
        parsed_url = urllib.parse.urlparse(self.base_url)
        if parsed_url.scheme not in ("http", "https"):
            self.last_error = f"ValueError: unsupported URL scheme {parsed_url.scheme!r} (expected http or https)"
            return list(candidates)
        host = parsed_url.hostname or ""
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))       # local models: never via a proxy
                  if host in ("localhost", "127.0.0.1", "::1") else urllib.request.build_opener())
        try:
            with opener.open(req, timeout=self.timeout) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            out = json.loads(text[text.find("{"): text.rfind("}") + 1])
            order = [int(i) for i in out["ranked"] if not isinstance(i, bool) and 0 <= int(i) < len(candidates)]
            order = list(dict.fromkeys(order)) + [i for i in range(len(candidates)) if i not in order]
            self.last_rationale, self.last_error = str(out.get("rationale", "")), None
            return [candidates[i] for i in order]
        except Exception as exc:  # noqa: BLE001 - the doer is untrusted anyway; keep the given order
            self.last_error = f"{type(exc).__name__}: {exc}"
            return list(candidates)
