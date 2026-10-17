"""Validate the gate's consequence model against the REAL tau2 environment.

For a random sample of 2026 write decisions, the simulation is replayed in a fresh tau2 Environment up to the write;
then the agent's call, every gold write of the task, and (retail) every executable alternative reading the gate
generated are each executed on a copy of the database. Claim checked: two calls have the same canonical effect key
(extract.canon_args, what the gate compares) if and only if they leave the database in the same state (tau2's own
database hash). Calls that raise a tool error are reported separately (the gate only proposes executable readings).

    .venv-public/bin/python benchmarks/tau2/validate_effects.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def _job(job):
    domain, path, sims = job
    logging.disable(logging.CRITICAL)
    from loguru import logger
    logger.remove()
    from extract import canon_args
    from gate import Ctx, IntentHypotheses
    from tau2.data_model.message import ToolCall
    from tau2.environment.toolkit import ToolType
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment
    else:
        from tau2.domains.airline.environment import get_environment
    from run import lite
    recs = {(r["sim_id"], r["k"]): r for r in lite(domain) if r["sim_id"] in sims}
    d = json.loads(Path(path).read_text())
    hp = IntentHypotheses()
    stats = defaultdict(int)
    for sim in d["simulations"]:
        if sim.get("id") not in sims:
            continue
        env = get_environment()
        wt = {n for n in env.tools.get_tools() if env.tools.tool_type(n) == ToolType.WRITE}
        results = {m.get("id"): m for m in sim["messages"] if m.get("role") == "tool"}
        k = 0
        for m in sim["messages"]:
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                call = ToolCall(id=tc.get("id") or "x", name=tc["name"], arguments=tc.get("arguments") or {}, requestor="assistant")
                if tc["name"] in wt:
                    rec = recs.get((sim.get("id"), k))
                    k += 1
                    if rec is not None:
                        cands = [(tc["name"], tc.get("arguments") or {})] + [(g["name"], g["args"]) for g in rec["gold_writes"]]
                        if domain == "retail":
                            for h in hp.hypotheses(Ctx(rec)):
                                f = h.params.get("facts") or {}
                                if "tool" in f:
                                    cands.append((f["tool"], f["args"]))
                        base = env.tools.db.model_copy(deep=True)
                        outs = []
                        for name, args in cands:
                            env.tools.db = base.model_copy(deep=True)
                            resp = env.get_response(ToolCall(id="v", name=name, arguments=args, requestor="assistant"))
                            outs.append((canon_args(name, args), None if resp.error else env.get_db_hash()))
                        env.tools.db = base
                        stats["decisions"] += 1
                        stats["calls"] += len(outs)
                        stats["errors"] += sum(o[1] is None for o in outs)
                        ok = [o for o in outs if o[1] is not None]
                        for i in range(len(ok)):
                            for j in range(i + 1, len(ok)):
                                same_key, same_db = ok[i][0] == ok[j][0], ok[i][1] == ok[j][1]
                                stats["pairs"] += 1
                                stats["agree"] += same_key == same_db
                                stats["key_eq_db_neq"] += same_key and not same_db
                                stats["key_neq_db_eq"] += (not same_key) and same_db
                if env._has_tool(tc["name"]) and env._is_mutating_tool(tc["name"]):
                    resp = env.get_response(call)
                    assert bool(resp.error) == bool((results.get(tc.get("id")) or {}).get("error")), "replay mismatch"
    return dict(stats)


def main(n_retail: int = 160, n_airline: int = 80) -> None:
    from extract import sources
    from run import lite
    random.seed(0)
    jobs, total = [], defaultdict(int)
    for domain, n in (("retail", n_retail), ("airline", n_airline)):
        recs = [r for r in lite(domain) if r["year"] == "2026"]
        pick = random.sample(recs, n)
        by_sub = defaultdict(set)
        for r in pick:
            by_sub[r["sub"]].add(r["sim_id"])
        for sub, path in sources(domain):
            if sub in by_sub:
                jobs.append((domain, str(path), by_sub[sub]))
    with ProcessPoolExecutor(12) as ex:
        for (domain, _, _), st in zip(jobs, ex.map(_job, jobs)):
            for k, v in st.items():
                total[f"{domain}:{k}"] += v
    print(json.dumps(total, indent=1))
    (HERE.parent / "results" / "tau2_effect_validation.json").write_text(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()
