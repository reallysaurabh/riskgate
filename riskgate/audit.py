"""Append-only JSONL audit log of every decision."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Decision


class AuditLog:
    """Append-only JSONL decision log. `redact(record) -> record` can drop or hash sensitive fields."""

    def __init__(self, path: str | Path, extra: dict | None = None,
                 redact: Callable[[dict], dict] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extra = extra or {}
        self.redact = redact
        self._lock = threading.Lock()

    def write(self, decision: Decision, **fields: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **self.extra, **fields, **decision.to_dict()}
        if self.redact is not None:
            rec = self.redact(rec)
        line = json.dumps(rec, default=str)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")


class MemoryAudit:
    """In-memory audit sink (tests, notebooks)."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def write(self, decision: Decision, **fields: Any) -> None:
        with self._lock:
            self.records.append({**fields, **decision.to_dict()})


class MultiSink:
    """Fan a decision out to several sinks (e.g. AuditLog + OTelSink). A failing sink never blocks the others."""

    def __init__(self, *sinks: Any):
        self.sinks = list(sinks)

    def write(self, decision: Decision, **fields: Any) -> None:
        for s in self.sinks:
            try:
                s.write(decision, **fields)
            except Exception:  # noqa: BLE001 - telemetry must not affect decisions
                pass


def summarize_audit(path: str | Path) -> dict:
    """Verdict/status counts and latency percentiles of a JSONL audit log."""
    import numpy as np
    verdicts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    lat: list[float] = []
    n = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        verdicts[r.get("verdict", "?")] = verdicts.get(r.get("verdict", "?"), 0) + 1
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
        if isinstance(r.get("latency_ms"), (int, float)):
            lat.append(float(r["latency_ms"]))
    pct = {f"p{q}": float(np.percentile(lat, q)) for q in (50, 95, 99)} if lat else {}
    return {"decisions": n, "verdicts": verdicts, "status": statuses, "latency_ms": pct}
