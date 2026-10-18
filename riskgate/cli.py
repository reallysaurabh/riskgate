"""Command-line interface.

    riskgate version
    riskgate report CALIBRATION.json [--json] [--key KEY]
    riskgate validate CALIBRATION.json --gate pkg.module:make_gate [--key KEY]
    riskgate calibrate --gate pkg.module:make_gate --episodes pkg.module:load_episodes [--n 1000]
                       [--alpha 0.05] [--mode policy|agnostic] [--guarantee expectation|high_probability]
                       [--delta-ltt 0.1] --out CALIBRATION.json [--key KEY]
    riskgate audit-summary decisions.jsonl
    riskgate serve-mcp --gate pkg.module:make_gate --calibration CALIBRATION.json [--context pkg.module:get_context]
                       [--name NAME] [--passthrough tool_a,tool_b] -- UPSTREAM_COMMAND [ARGS...]

`--gate` names a zero-argument factory returning a riskgate.Gate; `--episodes` names a function returning a
list of riskgate.Episode (it receives `n` if it accepts an argument). The current directory is put on
sys.path, so project-local modules work.

Exit codes: 0 ok, 1 runtime error, 2 usage error (argparse), 3 guarantee VOID (validate).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
from collections.abc import Callable
from typing import Any

from . import __version__
from .audit import summarize_audit
from .gate import Calibration

EXIT_VOID = 3


def _load(ref: str) -> Callable[..., Any]:
    if ":" not in ref:
        raise SystemExit(f"expected 'module:attribute', got {ref!r}")
    mod, attr = ref.split(":", 1)
    sys.path.insert(0, os.getcwd())
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "inf" if math.isinf(x) and x > 0 else "-inf" if math.isinf(x) else f"{x:.4g}"
    return str(x)


def cmd_report(a: argparse.Namespace) -> int:
    cal = Calibration.load(a.calibration, key=a.key, require_signature=bool(a.key))
    d = cal.to_dict()
    if a.json:
        print(json.dumps({k: v for k, v in d.items() if k not in ("curve", "reference_scores", "weighted")}, indent=1))
        return 0
    print(f"riskgate calibration {cal.id}  (schema riskgate.calibration/v1, created {cal.created})")
    print(f"  guarantee   : {cal.guarantee}  alpha={cal.alpha}" + (f"  delta={cal.delta_ltt}" if cal.delta_ltt else "")
          + (f"  extra risks={cal.extra_alphas}" if cal.extra_alphas else ""))
    print(f"  mode        : {cal.mode}    episodes for risk control n={cal.n}  (tau split n={cal.diagnostics.get('n_tau', 0)})")
    print(f"  plausible   : tau={_fmt(cal.tau)} (target coverage {1 - cal.delta:.2f}), "
          f"mean |H|={cal.diagnostics.get('mean_plausible_hypotheses', float('nan')):.2f}, tau_ood={_fmt(cal.tau_ood)}")
    print(f"  threshold   : lambda={_fmt(cal.lam)}  calibration ask rate={cal.diagnostics.get('calibration_ask_rate_at_lambda', float('nan')):.3f}"
          f"  calibration risk={cal.diagnostics.get('calibration_risk_at_lambda', float('nan')):.3f}")
    if cal.probing:
        print(f"  probing     : lambda1={_fmt(cal.lam_probe1)} lambda2={_fmt(cal.lam_probe2)} share={cal.probe_share} "
              f"probed episodes={cal.n_probe}")
    else:
        print(f"  probing     : not calibrated (probed episodes={cal.n_probe})")
    print(f"  fingerprint : {cal.fingerprint}" + ("  [weighted CRC data stored]" if cal.weighted else ""))
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    gate = _load(a.gate)()
    gate.load_calibration(a.calibration, key=a.key, require_signature=bool(a.key))
    st = gate.status()
    print(f"{st.value.upper()}: gate fingerprint {gate.fingerprint()} vs artefact {gate.calibration.fingerprint}")
    return 0 if st.value != "void" else EXIT_VOID


def cmd_calibrate(a: argparse.Namespace) -> int:
    gate = _load(a.gate)()
    fn = _load(a.episodes)
    episodes = fn(a.n) if a.n is not None and len(inspect.signature(fn).parameters) else fn()
    cal = gate.calibrate(episodes, alpha=a.alpha, mode=a.mode, guarantee=a.guarantee, delta_ltt=a.delta_ltt)
    cal.save(a.out, key=a.key)
    print(f"calibrated on {len(episodes)} episodes -> {a.out} (id {cal.id}, lambda={_fmt(cal.lam)})")
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    print(json.dumps(summarize_audit(a.log), indent=1))
    return 0


def cmd_serve_mcp(a: argparse.Namespace) -> int:  # pragma: no cover - needs the mcp package and a live upstream
    try:
        from .adapters.mcp_proxy import serve_stdio
    except ImportError as exc:
        print(f"serve-mcp needs the MCP adapter and `pip install mcp`: {exc}", file=sys.stderr)
        return 1
    import anyio
    gate = _load(a.gate)()
    if a.calibration:
        gate.load_calibration(a.calibration, key=a.key, require_signature=bool(a.key))
    ctx = _load(a.context) if a.context else (lambda: None)
    if not a.upstream:
        print("missing upstream command after --", file=sys.stderr)
        return 1
    passthrough = [t for t in (a.passthrough or "").split(",") if t]
    anyio.run(lambda: serve_stdio(gate, a.upstream[0], a.upstream[1:], context=ctx, name=a.name,
                                  passthrough=passthrough, expose_scores=a.expose_scores))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="riskgate", description="Certify, reject or ask: risk control for agent actions")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    r = sub.add_parser("report", help="summarise a calibration artefact")
    r.add_argument("calibration")
    r.add_argument("--json", action="store_true")
    r.add_argument("--key", default=os.environ.get("RISKGATE_KEY"))
    v = sub.add_parser("validate", help="check an artefact against a gate (exit 3 if VOID)")
    v.add_argument("calibration")
    v.add_argument("--gate", required=True)
    v.add_argument("--key", default=os.environ.get("RISKGATE_KEY"))
    c = sub.add_parser("calibrate", help="calibrate a gate on labelled episodes")
    c.add_argument("--gate", required=True)
    c.add_argument("--episodes", required=True)
    c.add_argument("--n", type=int, default=None)
    c.add_argument("--alpha", type=float, default=None)
    c.add_argument("--mode", choices=["policy", "agnostic"], default=None)
    c.add_argument("--guarantee", choices=["expectation", "high_probability"], default="expectation")
    c.add_argument("--delta-ltt", type=float, default=0.1)
    c.add_argument("--out", required=True)
    c.add_argument("--key", default=os.environ.get("RISKGATE_KEY"))
    au = sub.add_parser("audit-summary", help="summarise a JSONL audit log")
    au.add_argument("log")
    m = sub.add_parser("serve-mcp", help="run an MCP proxy that gates an upstream MCP server's tools")
    m.add_argument("--gate", required=True)
    m.add_argument("--calibration", default=None)
    m.add_argument("--context", default=None)
    m.add_argument("--key", default=os.environ.get("RISKGATE_KEY"))
    m.add_argument("--name", default="riskgate-proxy", help="server name announced to MCP clients")
    m.add_argument("--passthrough", default="", help="comma-separated read-only tools forwarded without gating")
    m.add_argument("--expose-scores", action="store_true",
                   help="put robust scores and the threshold into _meta (trusted clients only)")
    m.add_argument("upstream", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "version":
        print(__version__)
        return 0
    if a.cmd == "serve-mcp" and a.upstream and a.upstream[0] == "--":
        a.upstream = a.upstream[1:]
    try:
        return {"report": cmd_report, "validate": cmd_validate, "calibrate": cmd_calibrate,
                "audit-summary": cmd_audit, "serve-mcp": cmd_serve_mcp}[a.cmd](a)
    except Exception as exc:  # noqa: BLE001
        print(f"riskgate: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
