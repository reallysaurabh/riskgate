"""Domain regression: the refunds example (SQLite dry-run) keeps its risk below alpha on random re-splits.

Seeded and small (about 15 s); the full evaluation lives in examples/run_examples.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


def test_refunds_domain_risk_below_alpha_on_resplits():
    import harness
    import refunds_agent as R

    eps = [R.sample(10_000 + i) for i in range(1800)]
    rng = np.random.default_rng(0)
    unsafe, asks, first_choice = [], [], []
    for r in range(3):
        perm = rng.permutation(len(eps))
        cal, test = [eps[i] for i in perm[:1200]], [eps[i] for i in perm[1200:]]
        g = R.make_gate(probes=())
        g.calibrate(cal, alpha=0.05, seed=r)
        s = harness.summarise([harness.run_policy(g, ep, "riskgate") for ep in test], test)
        unsafe.append(s["unsafe"])
        asks.append(s["ask"])
        first_choice.append(np.mean([ep.unsafe[ep.candidates[0].key] for ep in test]))
    assert np.mean(first_choice) > 0.25            # the scripted doer is often wrong ...
    assert np.mean(unsafe) <= 0.05 + 0.02          # ... the gate keeps the risk near or below alpha
    assert np.mean(asks) < 0.6                     # ... without asking about everything


def test_use_case_examples_run_and_hold_alpha(capsys):
    """Every examples/use_cases script runs end to end; the replay line it prints stays near or under alpha."""
    import importlib
    import re

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "use_cases"))
    for name in ("payments_upi", "missing_item_refund", "card_dispute", "idle_vm_cleanup", "prompt_injection"):
        importlib.import_module(name).main()
        out = capsys.readouterr().out
        m = re.search(r"\[valid\]: unsafe ([0-9.]+)%", out)
        assert m is not None, (name, out)
        assert float(m.group(1)) <= 5.0 + 1.0, (name, out)
