"""The doer: an agent that proposes ranked supervisory actions.

Two implementations share one interface:

* MockLLMDoer -- a deterministic stand-in that behaves like a plausible LLM
  operator assistant: it reads the evidence, trusts its single best hypothesis
  and prefers the least disruptive move (a realistic failure mode), with some
  run-to-run inconsistency. Used for all reported experiments (reproducible,
  no API key needed).
* LLMDoer -- calls any OpenAI-compatible chat-completions endpoint (OpenAI,
  Gemini's OpenAI-compatible API, vLLM, Ollama, ...) with the plant context,
  the operating procedure and the allowed action schema, and parses a JSON
  ranking. Falls back to the mock if the call or parsing fails.

The doer is deliberately NOT trusted: whatever it proposes goes to the checker.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from .plant import T_SP_NOM, Segment
from .twin import HYPOTHESES, Observation, kind_scores


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    sp: float = T_SP_NOM
    q_scale: float = 1.0
    backup_sensor: bool = False

    def segment(self, duration: float) -> Segment:
        return Segment(duration, sp=self.sp, q_scale=self.q_scale, backup_sensor=self.backup_sensor)


ACTIONS: dict[str, Action] = {a.name: a for a in [
    Action("do_nothing", "Keep current setpoints and feed rate."),
    Action("reduce_feed_10", "Reduce feed flow by 10% to lower heat generation.", q_scale=0.90),
    Action("reduce_feed_25", "Reduce feed flow by 25%.", q_scale=0.75),
    Action("reduce_feed_40", "Reduce feed flow by 40% (large throughput loss).", q_scale=0.60),
    Action("lower_sp_5", "Lower the reactor temperature setpoint by 5 K.", sp=T_SP_NOM - 5.0),
    Action("switch_backup", "Switch the temperature controller to the backup thermocouple.", backup_sensor=True),
    Action("switch_backup_reduce_feed_25", "Switch to backup thermocouple and reduce feed by 25%.",
           q_scale=0.75, backup_sensor=True),
]}

SOP = """\
SOP-R101 Abnormal temperature behaviour (abridged)
1. Never exceed 362 K (runaway risk) and never fall below 340 K (off-spec product / precipitation).
2. If coolant demand rises (coolant temperature falling) the reactor is losing cooling margin or
   generating more heat: reduce throughput in steps; large feed cuts cost production.
3. If the temperature transmitter TT-101 is suspected, switch the loop to backup thermocouple TT-101B.
4. Prefer the least disruptive action that keeps the reactor inside the envelope.
"""


def evidence_summary(obs: Observation, residuals: np.ndarray) -> dict:
    ks = kind_scores(residuals)
    best = min(ks, key=ks.get)
    return {
        "minutes_since_alarm": round(obs.now, 1),
        "measured_T_K": round(float(obs.t_meas[-1]), 2),
        "measured_T_change_K": round(float(obs.t_meas[-1] - obs.t_meas[0]), 2),
        "coolant_T_K": round(float(obs.tc[-1]), 2),
        "coolant_T_change_K": round(float(obs.tc[-1] - obs.tc[0]), 2),
        "CA_analyser_mol_per_L": round(float(obs.ca_meas[-1]), 4),
        "hypothesis_fit_residuals_lower_is_better": {k: round(v, 2) for k, v in ks.items()},
        "best_hypothesis": best,
    }


class MockLLMDoer:
    """Plausible-LLM playbook: trusts the top hypothesis, prefers minimal moves, sometimes inconsistent."""

    def __init__(self, inconsistency: float = 0.2):
        self.inconsistency = inconsistency

    def propose(self, obs: Observation, residuals: np.ndarray, rng: np.random.Generator) -> tuple[list[str], str]:
        ev = evidence_summary(obs, residuals)
        best = ev["best_hypothesis"]
        if best in ("cooling_loss", "feed_conc"):
            ranked = ["reduce_feed_10", "lower_sp_5", "reduce_feed_25", "reduce_feed_40",
                      "switch_backup_reduce_feed_25", "do_nothing", "switch_backup"]
            why = f"Coolant demand rising ({ev['coolant_T_change_K']} K) -> {best}; trim throughput gently first."
        elif best == "sensor_bias":
            ranked = ["switch_backup", "lower_sp_5", "do_nothing", "switch_backup_reduce_feed_25",
                      "reduce_feed_10", "reduce_feed_25", "reduce_feed_40"]
            why = "Measured T inconsistent with coolant response -> suspect TT-101; move to backup sensor."
        else:
            ranked = ["do_nothing", "lower_sp_5", "reduce_feed_10", "switch_backup", "reduce_feed_25",
                      "switch_backup_reduce_feed_25", "reduce_feed_40"]
            why = "No clear fault signature; hold and monitor (lower setpoint if worried)."
        if rng.random() < self.inconsistency:
            ranked[0], ranked[1] = ranked[1], ranked[0]
        return ranked, why


class LLMDoer:
    """Any OpenAI-compatible chat endpoint. Env: CRA_LLM_BASE_URL, CRA_LLM_API_KEY, CRA_LLM_MODEL."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("CRA_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("CRA_LLM_API_KEY", "")
        self.model = model or os.environ.get("CRA_LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.fallback = MockLLMDoer(inconsistency=0.0)

    def _prompt(self, obs: Observation, residuals: np.ndarray) -> str:
        actions = {k: a.description for k, a in ACTIONS.items()}
        return (
            "You are an operator assistant for a jacket-cooled exothermic CSTR (reactor R-101).\n"
            f"{SOP}\nCurrent evidence (JSON):\n{json.dumps(evidence_summary(obs, residuals), indent=1)}\n"
            f"Allowed actions (id: description):\n{json.dumps(actions, indent=1)}\n"
            "Rank ALL allowed action ids from most to least appropriate. Respond ONLY with JSON: "
            '{"ranked_actions": ["id", ...], "rationale": "one sentence"}'
        )

    def propose(self, obs: Observation, residuals: np.ndarray, rng: np.random.Generator) -> tuple[list[str], str]:
        body = json.dumps({"model": self.model, "temperature": 0.2,
                           "messages": [{"role": "user", "content": self._prompt(obs, residuals)}]}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            text = text[text.find("{"): text.rfind("}") + 1]
            out = json.loads(text)
            ranked = [a for a in out["ranked_actions"] if a in ACTIONS]
            ranked += [a for a in ACTIONS if a not in ranked]
            return ranked, str(out.get("rationale", ""))
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the mock
            ranked, why = self.fallback.propose(obs, residuals, rng)
            return ranked, f"[LLM call failed: {exc!s:.80}; fallback] {why}"


__all__ = ["ACTIONS", "Action", "MockLLMDoer", "LLMDoer", "SOP", "evidence_summary", "HYPOTHESES"]
