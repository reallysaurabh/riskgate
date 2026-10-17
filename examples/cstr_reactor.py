"""Example domain 1: an operator-assistant agent on a jacket-cooled exothermic CSTR (reactor R-101).

Plant, faults, twin and LLM-like doer are vendored from the certify-reject-ask demo (cstr_lib/).
This module only wires them into riskgate's generic interfaces:

  hypotheses   19 fault hypotheses (none, cooling loss, stronger feed, high-reading thermocouple),
               scored by the twin's residual on TT-101, the coolant temperature and the CA analyser
  consequence  the nominal first-principles twin (it does not know each plant's true kinetics,
               heat transfer or feed temperature), rolled out 30 min under each action
  spec         340 K <= true reactor temperature <= 362 K at every point of the horizon
  probes       read the backup thermocouple / trim feed 10 % / nudge setpoint +2 K, each for 3 min
"""
from __future__ import annotations

import numpy as np
from cstr_lib import plant as P
from cstr_lib.doer import ACTIONS as CRA_ACTIONS
from cstr_lib.doer import MockLLMDoer
from cstr_lib.pipeline import HORIZON, clipT, extend_with_probe, observe, sample_episode, true_rollouts, twin_rollouts
from cstr_lib.pipeline import PROBES as CRA_PROBES
from cstr_lib.plant import Segment
from cstr_lib.twin import HYPOTHESES, SIG_CA, SIG_T, SIG_TC, diagnose

import riskgate as rg

NAME = "cstr_reactor"
META = {"system": "Jacket-cooled exothermic CSTR R-101 with PI temperature control (TIC-101)", "simulated": True,
        "note": "Textbook CSTR parameters; per-episode plant/twin mismatch; faults ramp in over 20 min. Not plant data.",
        "envelope_K": [P.T_MIN, P.T_MAX], "tags": {"TT-101": "reactor temperature (primary)", "TT-101B": "backup",
                                                  "TC-101.OUT": "coolant temperature", "AT-101": "CA analyser"}}
ACTIONS = {k: rg.Action(k, {"sp_K": a.sp, "feed_factor": a.q_scale, "backup_tc": a.backup_sensor})
           for k, a in CRA_ACTIONS.items()}
PROBES = [rg.Action(k, read_only=True) for k in CRA_PROBES]
NOOP = ACTIONS["do_nothing"]
SPEC = rg.Spec(rg.Envelope("T", P.T_MIN, P.T_MAX, name="340K<=T<=362K"), floor=-15.0)

_CACHE: dict = {}


def _cached(key, obs, fn):
    """Memoise per context object. Entries keep a strong reference to `obs` and are checked by identity, so a
    recycled id() can never serve another context's results."""
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is obs:
        return hit[1]
    if len(_CACHE) > 200_000:
        _CACHE.clear()
    val = fn()
    _CACHE[key] = (obs, val)
    return val


class Diagnoser:
    version = "cstr-diag-v1"

    def hypotheses(self, obs) -> list[rg.Hypothesis]:
        res = _cached(("diag", id(obs)), obs, lambda: diagnose(obs))
        return [rg.Hypothesis(h.label(), family=h.kind, params={"idx": i}, score=float(r))
                for i, (h, r) in enumerate(zip(HYPOTHESES, res))]


class Twin:
    """Batched twin. Rolls out ALL 19 hypotheses x ALL actions (or probes + hold) once per context, then
    serves any requested subset from the cache, so recalibrations and resplits never re-simulate."""

    version = "cstr-twin-v1"
    _ALL_H = [rg.Hypothesis(h.label(), family=h.kind, params={"idx": i}) for i, h in enumerate(HYPOTHESES)]

    def predict_batch(self, actions, hyps, obs):
        probe_mode = any(a.read_only for a in actions)
        universe = (PROBES + [NOOP]) if probe_mode else list(ACTIONS.values())
        full = _cached(("tw", id(obs), probe_mode), obs, lambda: self._predict(universe, self._ALL_H, obs, probe_mode))
        a_idx = {a.key: i for i, a in enumerate(universe)}
        return [[full[a_idx[a.key]][h.params["idx"]] for h in hyps] for a in actions]

    def _predict(self, actions, hl_h, obs, probe_mode):
        hl = [HYPOTHESES[h.params["idx"]] for h in hl_h]
        if probe_mode:                       # probes and the noop baseline share the probe's time structure
            plans = [CRA_PROBES[a.name] if a.read_only else [Segment(3.0), Segment(3.0)] for a in actions]
        else:
            plans = [[CRA_ACTIONS[a.name].segment(HORIZON)] for a in actions]
        tw = twin_rollouts(hl, obs, plans)
        n_p = len(plans)
        k = int(round(3.0 / 0.5)) + 1
        out = []
        for i, a in enumerate(actions):
            row = []
            for j in range(len(hl)):
                e = j * n_p + i
                o = {"T": clipT(tw.t_true[e])}
                if a.read_only:
                    o["observation"] = np.concatenate([tw.t_meas[e, :k] / SIG_T, tw.tc[e, :k] / SIG_TC, tw.ca[e, :k] / SIG_CA])
                row.append(o)
            out.append(row)
        return out

    def predict(self, action, hyp, obs):
        return self.predict_batch([action], [hyp], obs)[0][0]


def _labels(ep_cra) -> dict:
    plans = [[CRA_ACTIONS[k].segment(HORIZON)] for k in ACTIONS]
    bad = P.violates(true_rollouts(ep_cra, plans))
    return {ACTIONS[k].key: bool(b) for k, b in zip(ACTIONS, bad)}


def _ranked(obs, seed, tag) -> list[rg.Action]:
    ranked, _ = MockLLMDoer().propose(obs, diagnose(obs), np.random.default_rng([seed, tag]))
    return [ACTIONS[k] for k in ranked]


def sample(seed: int, shift: bool = False) -> rg.Episode:
    e = observe(sample_episode(seed + (1_000_000 if shift else 0), shift=shift))
    fam = e.kind if e.kind in ("none", "cooling_loss", "feed_conc", "sensor_bias") else "unseen"
    ep = rg.Episode(e.obs, _ranked(e.obs, seed, 9), _labels(e), true_family=fam,
                    meta={"seed": seed, "truth": {"kind": e.kind, "magnitude": round(e.magnitude, 3)},
                          "window_true_T": e.window_true})
    for name, segs in CRA_PROBES.items():
        e2 = extend_with_probe(e, segs[0])
        probe_bad = bool(P.violates(P.simulate(e.params, e.state_now, [segs[0]]))[0])
        ep.after_probe[rg.Action(name, read_only=True).key] = rg.Episode(
            e2.obs, _ranked(e2.obs, seed, 11), _labels(e2), true_family=fam, meta={"probe_unsafe": probe_bad})
    return ep


def hypotheses():
    return Diagnoser()


def make_gate(hypotheses=None, probes=None, **kw) -> rg.Gate:
    return rg.Gate(Twin(), SPEC, hypotheses or Diagnoser(), probes=PROBES if probes is None else probes,
                   noop=NOOP, noop_tol=0.5, min_separation=1.0, **kw)


def record(ep: rg.Episode) -> dict:
    o = ep.context
    r = lambda a: [round(float(x), 3) for x in a]
    return {
        "event_id": f"R101-EVT-{ep.meta['seed']:06d}", "unit": "R-101", "alarm": "TIC-101 deviation / coolant demand change",
        "minutes_since_onset": round(float(o.now), 1),
        "trend_30s": {"TT-101_K": r(o.t_meas), "TC-101.OUT_K": r(o.tc), "AT-101_molL": r(o.ca_meas)},
        "agent_ranked_actions": [a.name for a in ep.candidates],
        "envelope_violation_by_action": {k.split("(")[0]: v for k, v in ep.unsafe.items()},
        "ground_truth_fault": ep.meta["truth"],
    }
