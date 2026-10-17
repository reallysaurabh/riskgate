import http.server
import json
import math
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import riskgate as rg
from riskgate.cli import main as cli_main
from riskgate.conformal import crc_threshold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import ACTIONS, LeakHyps, Tank, episode, episodes, make_gate  # noqa: E402


def calibrated(**kw):
    g = make_gate(alpha=0.1, **kw)
    g.calibrate(episodes(11, 300))
    return g


# --------------------------------------------------------------------------- performance & robustness
class SlowTank(Tank):
    version = "slow-tank-v1"

    def predict(self, action, h, ctx):
        time.sleep(0.2)
        return super().predict(action, h, ctx)


class CountingTank(Tank):
    version = "tank-v1"

    def __init__(self):
        self.calls = 0

    def predict(self, action, h, ctx):
        self.calls += 1
        return super().predict(action, h, ctx)


def test_timeout_fails_closed():
    g = calibrated()
    g.consequence = SlowTank()
    g.calibration.fingerprint = g.fingerprint()       # pretend calibrated for the slow model
    g.timeout_s = 0.05
    d = g.check(ACTIONS[2], {"level": 50, "evidence": 0})
    assert d.verdict is rg.Verdict.ASK and d.reason.startswith("timeout")


def test_lru_cache_and_parallel_evaluation():
    m = CountingTank()
    g = rg.Gate(m, rg.Spec(rg.Envelope("level", 10, 100, scale=10)), LeakHyps(), alpha=0.1, cache_size=128,
                context_key=lambda c: (round(c["level"], 3), round(c["evidence"], 3)))
    g.calibrate(episodes(12, 200))
    ctx = {"level": 50.0, "evidence": 1.0}
    d1 = g.check_ranked(ACTIONS, ctx)
    calls = m.calls
    d2 = g.check_ranked(ACTIONS, dict(ctx))
    assert m.calls == calls and d1.verdict == d2.verdict and d1.score == d2.score
    gp = rg.Gate(Tank(), g.spec, LeakHyps(), alpha=0.1, parallel=4)
    gp.load_calibration(g.calibration)
    assert gp.check_ranked(ACTIONS, ctx).verdict == d1.verdict


def test_doer_fingerprint_voids_policy_mode_only():
    g = calibrated(doer_id="llm:model-a")
    assert g.status() is rg.GuaranteeStatus.VALID
    g.doer_id = "llm:model-b"
    assert g.status() is rg.GuaranteeStatus.VOID
    ga = calibrated(mode="agnostic", doer_id="llm:model-a")
    ga.doer_id = "llm:model-b"
    assert ga.status() is rg.GuaranteeStatus.VALID


def test_custom_score_fn():
    def mean_margin(row, hyps):
        return float(np.mean(row))
    g = calibrated(score_fn=mean_margin)
    assert g.check(ACTIONS[2], {"level": 50, "evidence": 0}).verdict in (rg.Verdict.CERTIFY, rg.Verdict.ASK)
    assert g.fingerprint() != calibrated().fingerprint()


def test_fuzz_failures_never_certify():
    rng = np.random.default_rng(13)

    class Flaky(Tank):
        version = "tank-v1"

        def predict(self, action, h, ctx):
            if rng.random() < 0.3:
                raise RuntimeError("simulator crashed")
            out = super().predict(action, h, ctx)
            if rng.random() < 0.2:
                out["level"] = float("nan")
            return out

    g = calibrated()
    g.consequence = Flaky()
    for _ in range(200):
        ctx = {"level": rng.uniform(0, 120), "evidence": rng.uniform(-10, 40)}
        d = g.check(ACTIONS[int(rng.integers(4))], ctx)
        if d.verdict is rg.Verdict.CERTIFY:      # only possible when no failure hit this check
            lvl = ctx["level"] + d.action.args["v"]
            assert all(math.isfinite(v) for v in d.margins.values())
            assert lvl > 0


# --------------------------------------------------------------------------- artefacts, outcomes, recalibration
def test_signed_artefacts(tmp_path):
    g = calibrated()
    p = tmp_path / "cal.json"
    g.calibration.save(p, key="s3cret")
    assert rg.Calibration.load(p, key="s3cret").lam == g.calibration.lam
    with pytest.raises(rg.SignatureError):
        rg.Calibration.load(p, key="wrong")
    d = json.loads(p.read_text())
    d["lam"] = -100.0                                   # tamper: loosen the threshold
    p.write_text(json.dumps(d))
    with pytest.raises(rg.SignatureError):
        rg.Calibration.load(p, key="s3cret")
    g.calibration.save(tmp_path / "unsigned.json")
    with pytest.raises(rg.SignatureError):
        rg.Calibration.load(tmp_path / "unsigned.json", require_signature=True)


def test_delayed_outcomes_and_recalibration():
    g = make_gate(alpha=0.1, monitor=rg.Monitor(alarm_level=20))
    g.calibrate(episodes(14, 300))
    d = g.check(ACTIONS[2], {"level": 50, "evidence": 0})
    assert d.verdict is rg.Verdict.CERTIFY
    assert g.record_outcome(d, unsafe=False) is True
    assert g.record_outcome(d.decision_id, unsafe=False) is False           # already reported
    assert g.monitor.observed_risk == 0.0
    for _ in range(60):
        g.check(ACTIONS[0], {"level": 75, "evidence": 25})
    assert g.status() is rg.GuaranteeStatus.DEGRADED
    g.recalibrate(episodes(15, 300))
    assert g.status() is rg.GuaranteeStatus.VALID


def test_extra_risk_controlled_jointly():
    # second risk: "big moves" (|v| >= 20 litres) should be executed at most 20% of the time
    big = lambda ep, a: float(abs(a.args.get("v", 0)) >= 20)
    g = make_gate(alpha=0.1)
    cal = g.calibrate(episodes(16, 400), extra_losses={"big_move": (big, 0.2)})
    assert cal.extra_alphas == {"big_move": 0.2}
    assert cal.diagnostics["calibration_risk_big_move"] <= 0.2 + 1e-9
    assert cal.diagnostics["calibration_risk_at_lambda"] <= 0.1 + 1e-9


def test_high_probability_guarantee_is_more_conservative():
    pytest.importorskip("riskgate.multirisk")
    eps = episodes(17, 600)
    g1, g2 = make_gate(alpha=0.1), make_gate(alpha=0.1)
    lam_e = g1.calibrate(eps).lam
    cal = g2.calibrate(eps, guarantee="high_probability", delta_ltt=0.1)
    assert cal.guarantee == "high_probability" and cal.lam >= lam_e


def test_weighted_calibration_under_shift():
    pytest.importorskip("riskgate.weighted")
    rng = np.random.default_rng(18)
    eps = [episode(rng) for _ in range(400)]
    w = [1.0 + (ep.context["level"] < 40) * 2.0 for ep in eps]          # target over-weights low levels
    g = make_gate(alpha=0.1, weight_fn=lambda ctx: 1.0 + (ctx["level"] < 40) * 2.0)
    cal = g.calibrate(eps, weights=w)
    assert cal.weighted is not None
    lam_low, lam_high = g._threshold(1, {"level": 20}), g._threshold(1, {"level": 80})
    assert lam_low >= lam_high
    d = g.check(ACTIONS[2], {"level": 60, "evidence": 0})
    assert d.verdict in (rg.Verdict.CERTIFY, rg.Verdict.ASK, rg.Verdict.REJECT)


# --------------------------------------------------------------------------- randomized CRC validity (property test)
def test_crc_validity_on_random_monotone_losses():
    rng = np.random.default_rng(19)
    for _ in range(25):
        alpha = rng.uniform(0.05, 0.3)
        m = 60
        grid = np.concatenate([np.sort(rng.normal(size=m - 1)), [np.inf]])
        viol = []
        for _ in range(200):
            def draw(k, m=m):
                cut = rng.integers(0, m, size=k)        # each episode: loss 1 for lambda below a random cut
                cut = np.where(rng.random(k) < 0.4, cut, 0)
                return (np.arange(m)[None, :] < cut[:, None]).astype(float)
            cal, test = draw(100), draw(1)
            lam = crc_threshold(cal, grid, alpha)
            j = int(np.searchsorted(grid, lam))
            viol.append(test[0, j])
        assert np.mean(viol) <= alpha + 0.06


# --------------------------------------------------------------------------- config, CLI, telemetry, doers
def test_config_toml_builds_gate(tmp_path):
    from riskgate.config import gate_from_config, load_config
    p = tmp_path / "gate.toml"
    p.write_text(textwrap.dedent("""
        [gate]
        alpha = 0.1
        mode = "policy"
        [[spec]]
        type = "envelope"
        key = "level"
        lo = 10
        hi = 100
        scale = 10
        [[hard_rules]]
        name = "no draining"
        action = "add"
        arg = "v"
        op = "<"
        value = 0
        verdict = "reject"
        message = "never drain"
        [monitor]
        alarm_level = 50
    """))
    cfg = load_config(p)
    g = gate_from_config(cfg, Tank(), LeakHyps())
    g.calibrate(episodes(20, 200))
    assert g.check(rg.Action("add", {"v": -10}), {"level": 50, "evidence": 0}).verdict is rg.Verdict.REJECT
    assert g.monitor is not None and g.monitor.alarm_level == 50


def test_cli_report_validate_calibrate_audit(tmp_path, capsys, monkeypatch):
    mod = tmp_path / "toy_gate.py"
    mod.write_text(textwrap.dedent(f"""
        import sys, numpy as np
        sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
        from test_core import make_gate as _mk, episode
        def make_gate():
            return _mk(alpha=0.1)
        def episodes(n=300):
            rng = np.random.default_rng(21)
            return [episode(rng) for _ in range(n)]
    """))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cal.json"
    assert cli_main(["calibrate", "--gate", "toy_gate:make_gate", "--episodes", "toy_gate:episodes", "--n", "250",
                     "--out", str(out), "--key", "k"]) == 0
    assert cli_main(["report", str(out), "--key", "k"]) == 0
    assert "lambda=" in capsys.readouterr().out
    assert cli_main(["validate", str(out), "--gate", "toy_gate:make_gate", "--key", "k"]) == 0
    assert cli_main(["report", str(out), "--key", "wrong"]) == 1
    log = tmp_path / "decisions.jsonl"
    g = make_gate(alpha=0.1, audit=rg.AuditLog(log, redact=lambda r: {k: v for k, v in r.items() if k != "margins"}))
    g.load_calibration(out, key="k")
    for v in (0, 20, 80):
        g.check(rg.Action("add", {"v": v}), {"level": 50, "evidence": 0})
    assert "margins" not in json.loads(log.read_text().splitlines()[0])
    capsys.readouterr()
    assert cli_main(["audit-summary", str(log)]) == 0
    assert json.loads(capsys.readouterr().out)["decisions"] == 3
    assert cli_main(["version"]) == 0


def test_otel_sink_with_fake_tracer():
    from riskgate.telemetry import OTelSink

    class Span:
        def __init__(self, store):
            self.store = store

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_attribute(self, k, v):
            self.store[k] = v

    class Tracer:
        def __init__(self):
            self.attrs = {}

        def start_as_current_span(self, name):
            return Span(self.attrs)

    class Inst:
        def __init__(self):
            self.values = []

        def add(self, v, attrs=None):
            self.values.append((v, attrs))

        def record(self, v, attrs=None):
            self.values.append((v, attrs))

    class Meter:
        def create_counter(self, *a, **k):
            return Inst()

        def create_histogram(self, *a, **k):
            return Inst()

    tracer = Tracer()
    sink = OTelSink(tracer=tracer, meter=Meter())
    g = calibrated(audit=rg.MultiSink(rg.MemoryAudit(), sink))
    g.check(ACTIONS[2], {"level": 50, "evidence": 0})
    assert tracer.attrs["riskgate.verdict"] in ("certify", "ask", "reject") and sink.decisions.values


def test_llm_ranker_against_local_server():
    from riskgate.doers import LLMRanker

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["messages"][1]["content"].startswith("Policy:")
            reply = {"choices": [{"message": {"content": '{"ranked": [2, 0], "rationale": "smallest move first"}'}}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = LLMRanker(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="x", model="local")
        ranked = r.rank("tank at 50 L", ACTIONS[:3], policy="keep 10-100 L")
        assert [a.args["v"] for a in ranked] == [0, 40, 20] and r.last_rationale and r.last_error is None
        assert r.doer_id.startswith("llm:local:")
    finally:
        srv.shutdown()
    bad = LLMRanker(base_url="http://127.0.0.1:9/v1", api_key="x", timeout=0.5)
    assert bad.rank("x", ACTIONS[:3]) == ACTIONS[:3] and bad.last_error


# --------------------------------------------------------------------------- regressions (review findings)
def test_ask_rule_on_lower_ranked_candidate_is_never_certified():
    g = calibrated(hard_rules=[rg.HardRule("big adds need approval", lambda a, c: a.args.get("v", 0) >= 45,
                                           rg.Verdict.ASK, ">= 45 L")])
    ctx = {"level": 20, "evidence": 0}
    d = g.check_ranked([rg.Action("add", {"v": -40}), rg.Action("add", {"v": 45})], ctx)
    assert not (d.verdict is rg.Verdict.CERTIFY and d.action.args["v"] == 45)
    assert any(c.get("verdict") == "ask" for c in d.considered)
    # preferred action needs approval -> escalate immediately
    assert g.check_ranked([rg.Action("add", {"v": 45}), rg.Action("add", {"v": 20})], ctx).verdict is rg.Verdict.ASK


def test_cache_requires_content_key():
    with pytest.raises(ValueError, match="context_key"):
        make_gate(alpha=0.1, cache_size=16)


def test_calibrate_mode_override_binds_doer():
    g = make_gate(alpha=0.1, mode="agnostic", doer_id="llm:model-a")
    g.calibrate(episodes(11, 300), mode="policy")
    assert g.mode == "policy" and g.status() is rg.GuaranteeStatus.VALID
    g.doer_id = "llm:model-b"
    assert g.status() is rg.GuaranteeStatus.VOID
    g2 = make_gate(alpha=0.1, mode="policy", doer_id="llm:model-a")
    g2.load_calibration(make_gate(alpha=0.1, mode="agnostic").calibrate(
        episodes(11, 300)))
    assert g2.mode == "agnostic" and g2.status() is rg.GuaranteeStatus.VALID


def test_audit_failure_fails_closed():
    class Broken:
        def write(self, d):
            raise OSError("disk full")

    g = calibrated(audit=Broken())
    for v in (0, 20, 40, -10):
        d = g.check(rg.Action("add", {"v": v}), {"level": 50, "evidence": 0})
        assert d.verdict is not rg.Verdict.CERTIFY
    reasons = [g.check(rg.Action("add", {"v": v}), {"level": 50, "evidence": 0}).reason for v in (0, 20)]
    assert any("audit sink error" in r for r in reasons)


def test_fingerprint_tracks_rule_and_constraint_logic():
    def rules(limit):
        return [rg.HardRule("no draining", lambda a, c: a.args.get("v", 0) < limit, rg.Verdict.REJECT)]
    f0 = make_gate(hard_rules=rules(0)).fingerprint()
    assert make_gate(hard_rules=rules(0)).fingerprint() == f0            # deterministic
    assert make_gate(hard_rules=rules(-5)).fingerprint() != f0           # closure value changed
    pinned = rg.HardRule("no draining", lambda a, c: False, rg.Verdict.REJECT, version="r1")
    assert make_gate(hard_rules=[pinned]).fingerprint() != f0
    def cap(limit):
        return rg.Custom("cap", lambda o: limit - o["level"])
    assert rg.Spec(cap(100)).fingerprint() == rg.Spec(cap(100)).fingerprint()
    assert rg.Spec(cap(100)).fingerprint() != rg.Spec(cap(90)).fingerprint()


def test_fingerprint_is_identical_across_processes():
    import subprocess
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
        import riskgate as rg
        from test_core import make_gate
        g = make_gate(hard_rules=[rg.HardRule("no draining", lambda a, c: a.args.get("v", 0) < 0)],
                      probes=[rg.Action("look", read_only=True)])
        print(g.fingerprint())
    """)
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
            for _ in range(2)}
    assert len(outs) == 1 and len(next(iter(outs))) == 16


def test_infinite_margin_never_certified_at_infinite_threshold():
    g = rg.Gate(rg.CallableModel(lambda a, h, c: {"x": 0.0}), rg.Spec(rg.Custom("vacuous", lambda o: math.inf)))
    g.load_calibration(rg.Calibration(alpha=0.05, mode="policy", delta=0.05, tau=math.inf, tau_ood=math.inf,
                                      lam=math.inf, fingerprint=g.fingerprint(), n=100))
    assert g.check(rg.Action("noop"), {}).verdict is not rg.Verdict.CERTIFY


def test_policy_extra_losses_keep_real_values():
    from riskgate.conformal import lambda_grid, policy_losses
    scores = [np.array([1.0, 2.0])]
    grid = lambda_grid(scores)
    L = policy_losses(scores, [np.array([0.3, 0.0])], grid)
    assert np.isclose(L[0, 0], 0.3)


def test_monitor_bind_resets_and_alarm_latches():
    g = make_gate(alpha=0.1, monitor=rg.Monitor(alarm_level=5))
    g.calibrate(episodes(14, 300))
    mon = g.monitor
    for _ in range(40):
        mon.update_score(-1e6)                      # scores far below anything seen in calibration
    assert mon.alarm and g.status() is rg.GuaranteeStatus.DEGRADED
    for _ in range(400):
        mon.update_score(1e6)                       # martingale decays, the latched alarm stays on
    assert mon.alarm
    d = g.check(ACTIONS[2], {"level": 50, "evidence": 0})
    g.recalibrate(episodes(15, 300))
    assert not mon.alarm and mon.pending == {} and mon.outcomes["certified_with_outcome"] == 0
    assert g.record_outcome(d, unsafe=False) is False          # decision from the previous calibration


def test_config_nested_tables_noop_dict_empty_monitor_and_signed_calibration(tmp_path, monkeypatch):
    from riskgate.config import gate_from_config
    g0 = make_gate(alpha=0.1)
    cal = g0.calibrate(episodes(16, 200))
    cal.save(tmp_path / "cal.json", key="k3y")
    cfg = {"gate": {"alpha": 0.1, "noop": {"name": "add", "args": {"v": 0}},
                    "spec": [{"type": "envelope", "key": "level", "lo": 10, "hi": 100, "scale": 10}],
                    "monitor": {}, "calibration": str(tmp_path / "cal.json"),
                    "calibration_key_env": "RG_TEST_KEY", "require_signature": True}}
    monkeypatch.setenv("RG_TEST_KEY", "k3y")
    g = gate_from_config(cfg, Tank(), LeakHyps())
    assert g.noop == rg.Action("add", {"v": 0}) and isinstance(g.monitor, rg.Monitor)
    assert g.calibration is not None and g.status() is rg.GuaranteeStatus.VALID
    monkeypatch.setenv("RG_TEST_KEY", "wrong")
    with pytest.raises(rg.SignatureError):
        gate_from_config(cfg, Tank(), LeakHyps())


def test_cli_void_exit_code(tmp_path, monkeypatch):
    mod = tmp_path / "toy_gate2.py"
    mod.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
        from test_core import make_gate as _mk
        def make_gate():
            return _mk(alpha=0.1)
        def make_other():
            return _mk(alpha=0.1, probes=[__import__("riskgate").Action("look", read_only=True)])
    """))
    monkeypatch.chdir(tmp_path)
    g = make_gate(alpha=0.1)
    g.calibrate(episodes(17, 200)).save(tmp_path / "c.json")
    assert cli_main(["validate", str(tmp_path / "c.json"), "--gate", "toy_gate2:make_gate"]) == 0
    assert cli_main(["validate", str(tmp_path / "c.json"), "--gate", "toy_gate2:make_other"]) == 3


def test_monitor_does_not_alarm_without_drift():
    """The monitored statistic (best robust score at stage 1) must match the calibration reference."""
    alarms = 0
    for seed in range(5):
        g = make_gate(alpha=0.1, monitor=rg.Monitor(alarm_level=20, seed=seed))
        g.calibrate(episodes(100 + seed, 400))
        rng = np.random.default_rng(200 + seed)
        for _ in range(300):
            ep = episode(rng)
            g.check_ranked(ep.candidates, ep.context)
        alarms += int(g.monitor.alarm)
    assert alarms <= 1          # false-alarm probability <= 1/20 per run under exchangeability


def test_otel_sdk_records_negative_margins_as_deficit():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider

    from riskgate.telemetry import OTelSink

    reader = InMemoryMetricReader()
    sink = OTelSink(tracer=TracerProvider().get_tracer("t"), meter=MeterProvider(metric_readers=[reader]).get_meter("m"))
    g = calibrated(audit=sink)
    g.check(rg.Action("add", {"v": 80}), {"level": 50, "evidence": 0})        # negative robust margin
    g.check(rg.Action("add", {"v": 0}), {"level": 50, "evidence": 0})         # positive robust margin
    names = {m.name: m for rm in reader.get_metrics_data().resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
    assert "riskgate.margin_deficit" in names and "riskgate.margin" in names
    assert sum(dp.count for dp in names["riskgate.margin_deficit"].data.data_points) >= 1


def test_redteam_parameter_and_constraint_validation():
    from riskgate.conformal import conformal_quantile
    with pytest.raises(ValueError, match="scale"):
        rg.Upper("x", 10, scale=0)
    with pytest.raises(ValueError, match="scale"):
        rg.Lower("x", 10, scale=-1)
    with pytest.raises(ValueError, match="lo <= hi"):
        rg.Envelope("x", 20, 10)
    with pytest.raises(ValueError, match="safe_margin"):
        rg.Predicate("p", lambda o: True, safe_margin=-1.0, unsafe_margin=1.0)
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([1.0, 2.0], 1.0)
    with pytest.raises(ValueError, match="NaN"):
        conformal_quantile([1.0, float("nan")], 0.1)
    with pytest.raises(ValueError, match="alpha"):
        make_gate(alpha=0.0)
    with pytest.raises(ValueError, match="delta"):
        make_gate(delta=1.0)
    with pytest.raises(ValueError, match="timeout_s"):
        make_gate(timeout_s=-1.0)


def test_redteam_empty_candidates_and_ask_rule_never_rejected():
    g = calibrated(hard_rules=[rg.HardRule("needs_human", lambda a, c: a.args.get("v", 0) == 45, rg.Verdict.ASK)])
    d_empty = g.check_ranked([], {"level": 50, "evidence": 0})
    assert d_empty.verdict is rg.Verdict.ASK and "no candidate" in d_empty.reason
    # When one candidate is unsafe under all hypotheses (-40 at level 12 -> 2 < 10) and another hits an ASK rule (45),
    # the decision must remain ASK (human can still approve 45), never REJECT.
    d_mix = g.check_ranked([rg.Action("add", {"v": -40}), rg.Action("add", {"v": 45})], {"level": 12, "evidence": 0})
    assert d_mix.verdict is rg.Verdict.ASK
    # Invalid stage fails closed to ASK
    d_stage = g.check(ACTIONS[2], {"level": 50, "evidence": 0}, stage=99)
    assert d_stage.verdict is rg.Verdict.ASK and "stage" in d_stage.reason


def test_redteam_tau_ood_calibrated_without_true_family_and_unbiased_split():
    rng = np.random.default_rng(42)
    unlabelled = []
    for _ in range(250):
        ep = episode(rng)
        ep.true_family = None
        unlabelled.append(ep)
    g = make_gate(alpha=0.1)
    cal = g.calibrate(unlabelled)
    assert cal.tau == math.inf and math.isfinite(cal.tau_ood)
    # OOD context is caught even when true_family was omitted during calibration
    d_ood = g.check(ACTIONS[2], {"level": 50, "evidence": 1e6})
    assert d_ood.verdict is rg.Verdict.ASK and "out of distribution" in d_ood.reason


def test_redteam_zero_weight_and_file_url_blocked():
    from riskgate.doers import LLMRanker
    rng = np.random.default_rng(43)
    eps = [episode(rng) for _ in range(250)]
    g = make_gate(alpha=0.1, weight_fn=lambda ctx: 0.0)
    g.calibrate(eps, weights=[1.0] * len(eps))
    d = g.check(ACTIONS[2], {"level": 50, "evidence": 0})
    assert d.verdict is rg.Verdict.ASK and "positive weight" in d.reason
    ranker = LLMRanker(base_url="file:///etc/passwd")
    assert ranker.rank("s", ACTIONS[:2]) == ACTIONS[:2] and "unsupported URL scheme" in (ranker.last_error or "")
    mon = rg.Monitor(max_history=5)
    mon.bind(g.calibration)
    for _ in range(20):
        mon.update_score(0.0)
    assert len(mon.history) == 5

