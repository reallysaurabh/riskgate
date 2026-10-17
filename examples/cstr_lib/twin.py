"""Hybrid digital twin + evidence-grounded diagnosis (an AgentRCA-style stand-in).

The twin is the nominal first-principles CSTR: it does NOT know the
episode's true kinetics / heat-transfer / feed temperature (model error), and
it reasons over an explicit grid of fault hypotheses. Diagnosis scores every
hypothesis by how well the twin, under that hypothesis, reproduces the
measurements; the *plausible set* keeps every hypothesis whose residual is
below a conformally calibrated threshold (so the true fault type is retained
with probability >= 1 - delta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import plant as P
from .plant import PlantParams, Segment, State

# Noise levels used to normalise residuals (measurement noise + typical model error)
SIG_T, SIG_TC, SIG_CA = 0.3, 0.6, 0.01


@dataclass(frozen=True)
class Hypothesis:
    kind: str          # "none" | "cooling_loss" | "feed_conc" | "sensor_bias"
    magnitude: float   # ua_scale | caf_scale | bias [K]; 0 for none

    def label(self) -> str:
        if self.kind == "none":
            return "no fault"
        unit = {"cooling_loss": "x UA", "feed_conc": "x CAf", "sensor_bias": "K bias"}[self.kind]
        return f"{self.kind} ({self.magnitude:g} {unit})"


KINDS = ("none", "cooling_loss", "feed_conc", "sensor_bias")

HYPOTHESES: list[Hypothesis] = (
    [Hypothesis("none", 0.0)]
    + [Hypothesis("cooling_loss", m) for m in (0.92, 0.84, 0.76, 0.70, 0.64, 0.58, 0.52)]
    + [Hypothesis("feed_conc", m) for m in (1.12, 1.24, 1.36, 1.48, 1.60, 1.72)]
    + [Hypothesis("sensor_bias", m) for m in (2.0, 4.0, 6.0, 8.0, 10.0)]
)


def twin_params(hyps: list[Hypothesis]) -> PlantParams:
    """Nominal (possibly wrong) physics + hypothesised fault, one element per hypothesis."""
    n = len(hyps)
    p = PlantParams.nominal(n)
    ua = np.array([h.magnitude if h.kind == "cooling_loss" else 1.0 for h in hyps])
    caf = np.array([h.magnitude if h.kind == "feed_conc" else 1.0 for h in hyps])
    bias = np.array([h.magnitude if h.kind == "sensor_bias" else 0.0 for h in hyps])
    return PlantParams(k0=p.k0, ua=p.ua, tf=p.tf, caf_scale=caf, ua_scale=ua, sensor_bias=bias)


@dataclass
class Observation:
    """What the operator / agent can see: measured T, controller output Tc, CA analyser."""

    time: np.ndarray
    t_meas: np.ndarray
    tc: np.ndarray
    ca_meas: np.ndarray
    segments: list[Segment]      # supervisory inputs applied since fault onset (t=0)
    x0_meas: tuple[float, float, float]   # (T_meas, CA_meas, Tc) at t = 0

    @property
    def now(self) -> float:
        return float(sum(s.duration for s in self.segments))


def estimate_state(t_meas: float, ca_meas: float, tc: float, sp: float, bias: np.ndarray, clock: float,
                   ramp_fraction: float) -> State:
    """Per-hypothesis state estimate from current measurements.

    The true temperature is the measurement minus the hypothesised (partially developed) bias;
    the PI integral is back-computed so the twin reproduces the current controller output.
    """
    n = bias.shape[0]
    t_true = t_meas - ramp_fraction * bias
    e = sp - t_meas
    integ = (tc - P.TC_SS - P.KC * e) * P.TAU_I / P.KC
    return State(ca=np.full(n, ca_meas), t=t_true, integ=np.full(n, integ), clock=clock)


def _ramp(clock: float) -> float:
    return float(np.clip(clock / P.FAULT_RAMP, 0.0, 1.0))


def diagnose(obs: Observation, hyps: list[Hypothesis] = HYPOTHESES) -> np.ndarray:
    """Normalised RMS residual of each hypothesis over the observation window."""
    tp = twin_params(hyps)
    t0, ca0, tc0 = obs.x0_meas
    x0 = estimate_state(t0, ca0, tc0, obs.segments[0].sp, tp.sensor_bias, clock=0.0, ramp_fraction=0.0)
    sim = P.simulate(tp, x0, obs.segments)
    n = min(sim.time.shape[0], obs.time.shape[0])
    rt = (sim.t_meas[:, :n] - obs.t_meas[None, :n]) / SIG_T
    rc = (sim.tc[:, :n] - obs.tc[None, :n]) / SIG_TC
    ra = (sim.ca[:, :n] - obs.ca_meas[None, :n]) / SIG_CA
    return np.sqrt(np.mean(rt ** 2 + rc ** 2 + ra ** 2, axis=1) / 3.0)


def kind_scores(residuals: np.ndarray, hyps: list[Hypothesis] = HYPOTHESES) -> dict[str, float]:
    """Best (lowest) residual per fault kind."""
    out = {}
    for k in KINDS:
        idx = [i for i, h in enumerate(hyps) if h.kind == k]
        out[k] = float(residuals[idx].min())
    return out


def rollout(hyps: list[Hypothesis], obs: Observation, segments: list[Segment]) -> P.Trajectory:
    """Roll the twin forward from 'now' under each hypothesis and the given future segments."""
    tp = twin_params(hyps)
    now = obs.now
    cur_sp = obs.segments[-1].sp
    x = estimate_state(float(obs.t_meas[-1]), float(obs.ca_meas[-1]), float(obs.tc[-1]), cur_sp,
                       tp.sensor_bias, clock=now, ramp_fraction=_ramp(now))
    # if the controller is currently on the backup sensor, the measurement is unbiased
    if obs.segments[-1].backup_sensor:
        x = State(ca=x.ca, t=np.full(len(hyps), float(obs.t_meas[-1])), integ=x.integ, clock=now)
    return P.simulate(tp, x, segments)
