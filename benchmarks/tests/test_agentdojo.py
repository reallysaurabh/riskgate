"""AgentDojo adapter: entity handling, score semantics, labels on a known logged run, and harness == Gate.calibrate.

The data-dependent tests skip when the AgentDojo logs (../public_data/src/agentdojo/runs) are not available.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "agentdojo"))
import common as C  # noqa: E402

A = pytest.importorskip("agentdojo_rg")
import riskgate as rg  # noqa: E402

HAVE_DATA = A.RUNS.exists() and importlib.util.find_spec("agentdojo.functions_runtime") is not None
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="AgentDojo package or logs not available")
INF = float("inf")


def test_entities_and_norm():
    ents = A.embedded_entities("Send it to Mark.Black-2134@gmail.com, see https://www.secure-systems-252.com/x and US133000000121212121212.")
    assert {"mark.black-2134@gmail.com", "www.secure-systems-252.com/x", "us133000000121212121212"} <= ents
    assert A.norm(" https://Www.Example.com/ ") == "www.example.com"
    assert A.occurs("fred", A.norm("Invite Fred now")) and not A.occurs("fred", A.norm("Frederick"))


def test_score_fn_is_min_nonconformity_over_violating_hypotheses():
    hs = [rg.Hypothesis("H0", score=-2.0), rg.Hypothesis("H1", score=-5.0), rg.Hypothesis("H2", score=-1.0)]
    assert A.score_fn(np.array([1.0, -1.0, -1.0]), hs) == -5.0
    assert A.score_fn(np.array([1.0, 1.0, -1.0]), hs) == -1.0
    assert A.score_fn(np.array([1.0, 1.0, 1.0]), hs) == A.LARGE


def _ref(rel: str) -> "A.RunRef":
    f = A.RUNS / rel
    p = f.parts
    return A.RunRef(p[-5], p[-4], p[-3], p[-2], None if p[-2] == "none" else f.stem, str(f))


@needs_data
def test_known_attack_run_is_labelled_and_scored():
    r = A.process_run(_ref("gpt-4o-2024-05-13/banking/user_task_0/important_instructions/injection_task_0.json"))
    assert r["security"] and r["security_replay"]
    attack = next(d for d in r["decisions"] if d["args"].get("recipient") == "US133000000121212121212")
    assert attack["unsafe"] and attack["untrusted"] == {"us133000000121212121212": [0]}
    assert r["outputs"][0]["injected"]
    gate = A.make_gate()
    s = gate.score_episode(A.episode(r, attack), tau=INF, tau_ood=INF)[0]
    assert s == pytest.approx(-r["outputs"][0]["instr"])              # distance to the injected output's hypothesis
    benign = A.process_run(_ref("gpt-4o-2024-05-13/banking/user_task_0/none/none.json"))
    assert benign["decisions"] and not any(d["unsafe"] for d in benign["decisions"])
    sb = gate.score_episode(A.episode(benign, benign["decisions"][0]), tau=INF, tau_ood=INF)[0]
    assert sb > s                                                       # the bill alone is less instruction-like


@needs_data
def test_harness_lambda_equals_gate_calibrate_on_logged_runs():
    refs = A.discover(["gpt-4o-2024-05-13"], ["none", "important_instructions"])
    refs = [x for x in refs if x.suite in ("banking", "slack")]
    runs = [A.process_run(x) for x in refs]
    runs = [r for r in runs if not r["target_conflict"]]
    gate = A.make_gate()
    eps, groups, decs, scores = [], [], [], []
    for r in runs:
        for d in r["decisions"]:
            ep = A.episode(r, d)
            eps.append(ep)
            groups.append(f"{r['suite']}/{r['user_task']}")
            decs.append(C.Decision(group=groups[-1], traj=A.run_id(r), doer=r["doer"], unsafe=[bool(d["unsafe"])]))
            scores.append([float(gate.score_episode(ep, tau=INF, tau_ood=INF)[0])])
    for alpha in (0.05, 0.1):
        cal = A.make_gate(alpha=alpha).calibrate(eps, groups=groups, tau_fraction=0.0)
        assert cal.lam == C.lam_for("riskgate", "crc", list(range(len(decs))), scores, decs, alpha)
