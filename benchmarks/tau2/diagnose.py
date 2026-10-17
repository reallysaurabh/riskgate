"""Dev-set diagnostics (2025 runs only): hypothesis-set coverage of the true intent, and which families matter."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
from extract import canon_args  # noqa: E402
from gate import Ctx, IntentHypotheses  # noqa: E402
from run import lite  # noqa: E402

for domain in sys.argv[1:] or ["retail", "airline"]:
    recs = [r for r in lite(domain) if r["year"] == "2025"]
    hp = IntentHypotheses()
    cov, rank, fam_true, fam_near, kinds = [], [], Counter(), Counter(), Counter()
    feats = []
    for r in recs:
        ctx = Ctx(r)
        hs = hp.hypotheses(ctx)
        me = canon_args(r["name"], r["args"])
        agent = next((h for h in hs if h.name == "agent"), None)
        bad = [h for h in hs if h.params["effect"] != me]
        near = min(bad, key=lambda h: h.score) if bad else None
        p_agent = float(np.exp(-agent.score)) if agent else float("nan")
        feats.append((r["unsafe"], p_agent, len(hs), near.score if near else 50.0, r["k"], len(r["prefix"]), r["confirmed_kw"]))
        if r["unsafe"]:
            golds = {canon_args(g["name"], g["args"]) for g in r["gold_writes"]}
            same_tool = [g for g in r["gold_writes"] if g["name"] == r["name"]]
            hit = [h for h in hs if h.params["effect"] in golds]
            cov.append(bool(hit))
            if hit:
                srt = sorted(hs, key=lambda h: h.score)
                rank.append(min(srt.index(h) for h in hit))
                fam_true[hit[0].family] += 1
            else:
                kinds["no write expected" if not r["gold_writes"] else ("other tool" if not same_tool else "same tool, not generated")] += 1
            if near:
                fam_near[near.family] += 1
    f = np.array(feats, float)
    u = f[:, 0].astype(bool)
    print(f"===== {domain} dev: {len(recs)} writes, {u.sum()} unsafe")
    print(f"  true intent in hypothesis set: {np.mean(cov):.2f} ({sum(cov)}/{len(cov)}); rank of true reading: {Counter(rank).most_common(6)}")
    print(f"  families of the true reading when covered: {fam_true.most_common()}")
    print(f"  not covered: {kinds.most_common()}")
    print(f"  nearest-unsafe family on unsafe writes: {fam_near.most_common(8)}")
    for j, nm in enumerate(["p(agent reading)", "|H|", "nearest-unsafe r", "write index k", "prefix length", "confirmed kw"], start=1):
        print(f"  AUROC {nm:18s} {C.auroc(f[:, j], u):.3f}  (mean safe {f[~u, j].mean():.2f} / unsafe {f[u, j].mean():.2f})")
