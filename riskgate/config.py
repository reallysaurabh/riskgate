"""Declarative configuration (JSON, TOML, or YAML if PyYAML is installed).

No code is ever evaluated from configuration files: hard rules are expressed as simple conditions.

Example (TOML):

    [gate]
    alpha = 0.05
    mode = "policy"
    noop = "noop"
    probes = ["pg_stat_statements_top", "redis_info_stats"]

    [[spec]]
    type = "upper"
    key = "p95_ms"
    limit = 800
    scale = 800

    [[hard_rules]]
    name = "max scale factor"
    action = "kubectl_scale"          # optional: only for this tool
    arg = "factor"
    op = ">"
    value = 3
    verdict = "reject"
    message = "never scale more than 3x in one step"
"""
from __future__ import annotations

import json
import logging
import operator
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gate import Gate
from .monitor import Monitor
from .spec import Constraint, Envelope, HardRule, Lower, Spec, Upper
from .types import Action, Verdict, digest

log = logging.getLogger("riskgate")
_HOISTED = ("spec", "spec_floor", "hard_rules", "monitor", "audit", "calibration", "calibration_key_env",
            "require_signature")

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq,
        "!=": operator.ne, "in": lambda a, b: a in b, "not in": lambda a, b: a not in b}


def load_config(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        return json.loads(text)
    if p.suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(text)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("YAML configs need `pip install pyyaml` (or use .toml / .json)") from exc
        return yaml.safe_load(text)
    raise ValueError(f"unsupported config format: {p.suffix}")


def spec_from_config(items: Sequence[Mapping[str, Any]], floor: float | None = None) -> Spec:
    cons: list[Constraint] = []
    for it in items:
        t = it["type"].lower()
        name = it.get("name")
        scale = float(it.get("scale", 1.0))
        if t == "upper":
            cons.append(Upper(it["key"], it["limit"], scale, name))
        elif t == "lower":
            cons.append(Lower(it["key"], it["limit"], scale, name))
        elif t == "envelope":
            cons.append(Envelope(it["key"], it["lo"], it["hi"], scale, name))
        else:
            raise ValueError(f"unknown constraint type {t!r} (use upper, lower, envelope; Custom/STL in code)")
    return Spec(*cons, floor=floor)


def rule_from_config(it: Mapping[str, Any]) -> HardRule:
    op = _OPS[it.get("op", "==")]
    tool, arg, value = it.get("action"), it.get("arg"), it.get("value")
    verdict = Verdict(it.get("verdict", "reject").lower())

    def fires(action: Action, context: Any) -> bool:
        if tool is not None and action.name != tool:
            return False
        if arg is None:
            return True
        if arg not in action.args:
            return False
        try:
            return bool(op(action.args[arg], value))
        except TypeError:
            return True                                  # incomparable -> fire (fail closed)

    return HardRule(it["name"], fires, verdict, it.get("message", ""), version="cfg:" + digest(dict(it)))


def _action(x: Any, read_only: bool = False) -> Action:
    if isinstance(x, str):
        return Action(x, read_only=read_only)
    if isinstance(x, Mapping):
        return Action(x["name"], dict(x.get("args", {})), bool(x.get("read_only", read_only)))
    raise ValueError(f"cannot build an Action from {x!r} (use a string or a table with name/args)")


def _redactor(fields: Sequence[str]):
    """Drop dotted paths (e.g. "action.args.card_number") from audit records."""
    paths = [f.split(".") for f in fields]

    def redact(rec: dict) -> dict:
        for path in paths:
            node: Any = rec
            for part in path[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and path[-1] in node:
                node[path[-1]] = "[redacted]"
        return rec

    return redact


def gate_from_config(cfg: Mapping[str, Any], consequence: Any, hypotheses: Any = None, **overrides: Any) -> Gate:
    """Build a Gate from a parsed config. Top-level tables (spec, hard_rules, monitor, audit, calibration) may
    also be nested under [gate]. The calibration artefact is verified with the HMAC key read from the
    environment variable named by `calibration_key_env` (never put keys in config files); set
    `require_signature = true` to refuse unsigned artefacts."""
    cfg = dict(cfg)
    g = dict(cfg.get("gate", {}))
    for k in _HOISTED:
        if k in g:
            if k in cfg:
                raise ValueError(f"'{k}' is given both at the top level and under [gate]")
            cfg[k] = g.pop(k)
    if "spec" not in cfg:
        raise ValueError("config needs a [[spec]] list of constraints")
    probes = [_action(p, read_only=True) for p in g.pop("probes", [])]
    noop = g.pop("noop", None)
    mon = cfg.get("monitor")
    kw: dict[str, Any] = dict(
        alpha=g.pop("alpha", 0.05), mode=g.pop("mode", "policy"), delta=g.pop("delta", 0.05),
        hard_rules=[rule_from_config(r) for r in cfg.get("hard_rules", [])], probes=probes,
        noop=_action(noop) if noop is not None else None, **g)
    if mon is not None:                      # an empty [monitor] table means "monitor with defaults"
        mon = dict(mon)
        kw["monitor"] = Monitor(alarm_level=float(mon.pop("alarm_level", 100.0)), eta=mon.pop("eta", None),
                                latch=bool(mon.pop("latch", True)))
        if mon:
            raise ValueError(f"unknown [monitor] keys: {sorted(mon)}")
    if "audit" in cfg:
        from .audit import AuditLog
        a = dict(cfg["audit"])
        fields = list(a.pop("redact", []) or [])
        kw["audit"] = AuditLog(a.pop("path"), redact=_redactor(fields) if fields else None)
        if a:
            raise ValueError(f"unknown [audit] keys: {sorted(a)}")
    kw.update(overrides)
    gate = Gate(consequence, spec_from_config(cfg["spec"], cfg.get("spec_floor")), hypotheses, **kw)
    if cfg.get("calibration"):
        p = Path(cfg["calibration"])
        env = cfg.get("calibration_key_env")
        key = os.environ.get(env) if env else None
        require = bool(cfg.get("require_signature", False))
        if env and key is None:
            raise ValueError(f"calibration_key_env={env!r} is set but the environment variable is missing")
        if p.exists():
            gate.load_calibration(p, key=key, require_signature=require)
        else:
            log.warning("riskgate: calibration %s not found; gate is uncalibrated (every check returns ASK)", p)
    return gate
