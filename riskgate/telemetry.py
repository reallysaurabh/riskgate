"""OpenTelemetry integration (optional dependency).

    from riskgate.telemetry import OTelSink
    gate = rg.Gate(..., audit=rg.MultiSink(rg.AuditLog("decisions.jsonl"), OTelSink()))

Emits one span `riskgate.check` per decision (attributes: verdict, status, action, score, threshold,
calibration id, stage, probe, decision id) and four instruments:
  riskgate.decisions       counter, attributes {verdict, status}
  riskgate.latency_ms      histogram
  riskgate.margin          histogram of non-negative robust margins (finite only), attribute {verdict}
  riskgate.margin_deficit  histogram of -margin for negative robust margins, attribute {verdict}
OpenTelemetry histograms accept only non-negative values, hence the split: a decision contributes to exactly
one of the two margin histograms.
A tracer/meter can be injected (tests, custom providers); otherwise the global OpenTelemetry API is used.
"""
from __future__ import annotations

import math
from typing import Any

from .types import Decision


class OTelSink:
    def __init__(self, tracer: Any = None, meter: Any = None, name: str = "riskgate"):
        if tracer is None or meter is None:
            try:
                from opentelemetry import metrics, trace  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError("OTelSink needs `pip install opentelemetry-api` (and an SDK/exporter)") from exc
            tracer = tracer or trace.get_tracer(name)
            meter = meter or metrics.get_meter(name)
        self.tracer = tracer
        self.decisions = meter.create_counter("riskgate.decisions", description="gate decisions by verdict/status")
        self.latency = meter.create_histogram("riskgate.latency_ms", unit="ms", description="decision latency")
        self.margin = meter.create_histogram("riskgate.margin", description="non-negative robust margin of the evaluated action")
        self.deficit = meter.create_histogram("riskgate.margin_deficit",
                                              description="how far below zero the robust margin is (negative margins only)")

    def write(self, decision: Decision, **fields: Any) -> None:
        attrs: dict[str, Any] = {
            "riskgate.verdict": decision.verdict.value,
            "riskgate.status": decision.status.value,
            "riskgate.action": decision.action.name if decision.action else "",
            "riskgate.stage": decision.stage,
            "riskgate.calibration_id": decision.calibration_id or "",
            "riskgate.decision_id": decision.decision_id,
            "riskgate.probe": decision.probe.name if decision.probe else "",
        }
        if math.isfinite(decision.score):
            attrs["riskgate.score"] = float(decision.score)
        if math.isfinite(decision.threshold):
            attrs["riskgate.threshold"] = float(decision.threshold)
        with self.tracer.start_as_current_span("riskgate.check") as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)
            for k, v in fields.items():
                span.set_attribute(f"riskgate.extra.{k}", str(v))
        self.decisions.add(1, {"verdict": decision.verdict.value, "status": decision.status.value})
        self.latency.record(float(decision.latency_ms))
        if math.isfinite(decision.score):
            v = float(decision.score)
            tag = {"verdict": decision.verdict.value}
            if v >= 0:
                self.margin.record(v, tag)
            else:
                self.deficit.record(-v, tag)
