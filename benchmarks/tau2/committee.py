"""Exploratory (added after the frozen primary evaluation): hypotheses proposed by OTHER AGENT SAMPLES.

RiskGate's hypothesis provider is pluggable, and rule R1 allows LLM-derived inputs to ADD hypotheses. Here the readings
of a customer's request are the outcomes of independent agent runs on the same request:

  * self-consistency  the same agent's other trials on the task (K = 3 in tau2-bench): deployable by sampling the
                      agent a few extra times in shadow before a consequential write;
  * committee         every other run on the task by agents of the same generation (2026: 7 other frontier models x 4
                      trials + own 3 others; 2025: 3 x 4 + 3): an expensive but deployable "second opinion" panel.

Each run c is one hypothesis ("the customer wants what run c did"), all equally plausible (r = log K). The write is
consistent with c iff c executed a write with the same canonical effect (extract.canon_args). Runs that made no write
count as readings in which the write is wrong. Score = -log(posterior mass of readings in which the write is wrong).
No gold label or hidden instruction is used; groups (tasks) stay exchangeable because the score of a decision only
uses runs of its own task, and calibration/test splits are drawn by task.

Deployment caveat: logged runs are full independent conversations (their simulated customers phrase things
differently), whereas a deployed gate would sample continuations at the same decision point.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import riskgate as rg  # noqa: E402

BIG = 50.0
VERSION = "tau2-committee-v1"


def _sims(job):
    sub, path = job
    d = json.loads(Path(path).read_text())
    return [(sub, str(s["task_id"]), s.get("id"), s.get("trial")) for s in d["simulations"]]


def sim_index(domain: str) -> list[tuple]:
    p = HERE / "cache" / f"sims_{domain}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    from extract import sources
    with ProcessPoolExecutor(12) as ex:
        rows = [x for part in ex.map(_sims, sources(domain)) for x in part]
    p.write_bytes(pickle.dumps(rows))
    return rows


class CommitteeCtx:
    __slots__ = ("rec", "runs")

    def __init__(self, rec: dict, runs: list[frozenset]):
        self.rec, self.runs = rec, runs


class CommitteeHypotheses:
    version = VERSION

    def hypotheses(self, ctx: CommitteeCtx) -> list[rg.Hypothesis]:
        k = max(1, len(ctx.runs))
        return [rg.Hypothesis(f"run{j}", family="run", params={"effects": eff}, score=math.log(k)) for j, eff in enumerate(ctx.runs)]


class CommitteeModel:
    version = VERSION

    def predict(self, action: rg.Action, h: rg.Hypothesis, ctx: CommitteeCtx) -> dict:
        from extract import canon_args
        return {"intended": canon_args(action.name, dict(action.args)) in h.params["effects"]}


SPEC = rg.Spec(rg.Predicate("another run made the same write", lambda o: o["intended"]))


def make_gate() -> rg.Gate:
    from gate import unsafe_mass
    return rg.Gate(CommitteeModel(), SPEC, CommitteeHypotheses(), score_fn=unsafe_mass)


def scores(domain: str, recs: list[dict]) -> np.ndarray:
    """(n, 2): self-consistency and committee scores, aligned with recs."""
    from extract import MODELS, canon_args
    sims = sim_index(domain)
    effects: dict = defaultdict(set)
    for r in recs:
        effects[r["sim_id"]].add(canon_args(r["name"], r["args"]))
    by_task: dict = defaultdict(list)
    for sub, task, sid, _ in sims:
        by_task[task].append((sub, sid))
    g = make_gate()
    out = np.zeros((len(recs), 2))
    for i, r in enumerate(recs):
        year = r["year"]
        own = [frozenset(effects.get(sid, ())) for sub, sid in by_task[r["task_id"]] if sub == r["sub"] and sid != r["sim_id"]]
        others = [frozenset(effects.get(sid, ())) for sub, sid in by_task[r["task_id"]]
                  if sub != r["sub"] and MODELS[sub][1] == year]
        a = rg.Action(r["name"], r["args"])
        for j, runs in enumerate((own, own + others)):
            if not runs:
                out[i, j] = -float("inf")
                continue
            out[i, j] = float(g.score_episode(rg.Episode(CommitteeCtx(r, runs), [a], {}), tau=float("inf"), tau_ood=float("inf"))[0])
    return out
