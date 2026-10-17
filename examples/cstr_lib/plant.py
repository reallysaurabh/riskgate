"""Closed-loop CSTR simulator with injectable faults.

Classic exothermic CSTR (A -> B), jacket-cooled, with a PI temperature
controller manipulating the coolant temperature. Everything is vectorised
over a batch dimension so that many hypotheses x candidate actions can be
rolled out at once.

States per batch element: CA [mol/L], T [K], integral of control error.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# ----------------------------------------------------------------------------
# Nominal physical parameters (Seborg-style textbook CSTR)
# ----------------------------------------------------------------------------
Q_NOM = 100.0        # feed flow [L/min]
V = 100.0            # volume [L]
CAF_NOM = 1.0        # feed concentration [mol/L]
TF_NOM = 350.0       # feed temperature [K]
RHO_CP = 1000.0 * 0.239   # rho*Cp [J/(L K)]
DH = 5.0e4           # heat of reaction (exothermic, positive here) [J/mol]
E_R = 8750.0         # activation energy / R [K]
K0_NOM = 7.2e10      # pre-exponential factor [1/min]
UA_NOM = 5.0e4       # heat-transfer coefficient * area [J/(min K)]

# Operating point and limits
T_SP_NOM = 350.0     # nominal temperature setpoint [K]
TC_MIN, TC_MAX = 285.0, 320.0   # coolant temperature range (coolant supply limits) [K]
FAULT_RAMP = 20.0    # faults develop linearly over this many minutes after onset (t=0)
T_MIN, T_MAX = 340.0, 362.0     # SAFE ENVELOPE on the TRUE reactor temperature [K]

# PI controller (reverse acting: T above SP -> lower coolant temperature)
KC = 10.0
TAU_I = 5.0          # [min]

DT = 0.05            # integration step [min]


@dataclass(frozen=True)
class PlantParams:
    """Per-element physical + fault parameters. Fields are arrays of shape (B,)."""

    k0: np.ndarray
    ua: np.ndarray
    tf: np.ndarray
    caf_scale: np.ndarray   # feed-concentration multiplier (process fault)
    ua_scale: np.ndarray    # cooling-capacity multiplier (fouling / valve fault)
    sensor_bias: np.ndarray  # additive bias on the PRIMARY temperature sensor [K]

    @property
    def batch(self) -> int:
        return int(np.asarray(self.k0).shape[0])

    @staticmethod
    def nominal(batch: int = 1) -> "PlantParams":
        o = np.ones(batch)
        return PlantParams(k0=K0_NOM * o, ua=UA_NOM * o, tf=TF_NOM * o,
                           caf_scale=o.copy(), ua_scale=o.copy(), sensor_bias=0 * o)

    def take(self, idx) -> "PlantParams":
        return PlantParams(**{f: np.asarray(getattr(self, f))[idx] for f in self.__dataclass_fields__})

    def repeat(self, n: int) -> "PlantParams":
        """Repeat each element n times (element-major)."""
        return PlantParams(**{f: np.repeat(np.asarray(getattr(self, f)), n) for f in self.__dataclass_fields__})

    def tile(self, n: int) -> "PlantParams":
        return PlantParams(**{f: np.tile(np.asarray(getattr(self, f)), n) for f in self.__dataclass_fields__})


@dataclass(frozen=True)
class Segment:
    """A piecewise-constant supervisory input applied for `duration` minutes.

    `sp`, `q_scale` and `backup_sensor` may be scalars or arrays of shape (B,)
    (one value per batch element); `duration` must be a scalar.
    """

    duration: float
    sp: float = T_SP_NOM
    q_scale: float = 1.0
    backup_sensor: bool = False   # True: controller uses the unbiased backup sensor


@dataclass
class State:
    ca: np.ndarray
    t: np.ndarray
    integ: np.ndarray
    clock: float = 0.0   # minutes since fault onset

    def copy(self) -> "State":
        return State(self.ca.copy(), self.t.copy(), self.integ.copy(), self.clock)

    def repeat(self, n: int) -> "State":
        return State(np.repeat(self.ca, n), np.repeat(self.t, n), np.repeat(self.integ, n), self.clock)

    def tile(self, n: int) -> "State":
        return State(np.tile(self.ca, n), np.tile(self.t, n), np.tile(self.integ, n), self.clock)


@dataclass
class Trajectory:
    """Time series of shape (B, N)."""

    time: np.ndarray
    t_true: np.ndarray
    t_meas: np.ndarray      # what the controller / operator sees (primary or backup)
    ca: np.ndarray
    tc: np.ndarray
    final: State = field(repr=False, default=None)


def _ramp(clock: float) -> float:
    return float(np.clip(clock / FAULT_RAMP, 0.0, 1.0))


def _rates(ca, t, tc, q, p: PlantParams, f: float):
    """f in [0,1] is the fraction of the fault that has developed."""
    t = np.clip(t, 250.0, 600.0)
    ca = np.clip(ca, 0.0, 5.0)
    k = p.k0 * np.exp(-E_R / t)
    r = k * ca
    caf = CAF_NOM * (1.0 + f * (p.caf_scale - 1.0))
    ua = p.ua * (1.0 + f * (p.ua_scale - 1.0))
    dca = q / V * (caf - ca) - r
    dt = q / V * (p.tf - t) + DH / RHO_CP * r + ua / (V * RHO_CP) * (tc - t)
    return dca, dt


def steady_state_tc(t_target: float = T_SP_NOM) -> tuple[float, float]:
    """Coolant temperature and CA giving a steady state at t_target (nominal plant)."""
    k = K0_NOM * np.exp(-E_R / t_target)
    ca = Q_NOM / V * CAF_NOM / (Q_NOM / V + k)
    r = k * ca
    # 0 = q/V (Tf-T) + DH/rhoCp r + UA/(V rhoCp)(Tc-T)
    tc = t_target - (Q_NOM / V * (TF_NOM - t_target) + DH / RHO_CP * r) * V * RHO_CP / UA_NOM
    return float(tc), float(ca)


TC_SS, CA_SS = steady_state_tc()


def initial_state(batch: int) -> State:
    return State(ca=np.full(batch, CA_SS), t=np.full(batch, T_SP_NOM), integ=np.zeros(batch))


def simulate(p: PlantParams, x0: State, segments: list[Segment], rng: np.random.Generator | None = None,
             meas_noise: float = 0.0, record_every: int = 10) -> Trajectory:
    """Simulate the closed loop for a list of supervisory segments (batched).

    The PI controller acts on the measured temperature: the primary sensor
    (true + bias + noise) unless `backup_sensor` is set in the segment.
    Integration: RK4 on (CA, T) with the controller held over each step.
    """
    b = p.batch
    ca, t, integ = x0.ca.copy(), x0.t.copy(), x0.integ.copy()
    times, tt, tm, cas, tcs = [], [], [], [], []
    clock = x0.clock
    step = 0
    for seg in segments:
        n = int(round(seg.duration / DT))
        q = Q_NOM * seg.q_scale
        for _ in range(n):
            noise = rng.normal(0.0, meas_noise, b) if (rng is not None and meas_noise > 0) else 0.0
            f = _ramp(clock)
            t_meas = t + np.where(seg.backup_sensor, 0.0, f * p.sensor_bias) + noise
            e = seg.sp - t_meas
            tc_unsat = TC_SS + KC * (e + integ / TAU_I)
            tc = np.clip(tc_unsat, TC_MIN, TC_MAX)
            # conditional integration (anti-windup)
            sat_low = (tc_unsat <= TC_MIN) & (e < 0)
            sat_high = (tc_unsat >= TC_MAX) & (e > 0)
            integ = integ + np.where(sat_low | sat_high, 0.0, e) * DT
            if step % record_every == 0:
                times.append(clock); tt.append(t.copy()); tm.append(np.broadcast_to(t_meas, (b,)).copy())
                cas.append(ca.copy()); tcs.append(np.broadcast_to(tc, (b,)).copy())
            k1 = _rates(ca, t, tc, q, p, f)
            k2 = _rates(ca + DT / 2 * k1[0], t + DT / 2 * k1[1], tc, q, p, f)
            k3 = _rates(ca + DT / 2 * k2[0], t + DT / 2 * k2[1], tc, q, p, f)
            k4 = _rates(ca + DT * k3[0], t + DT * k3[1], tc, q, p, f)
            ca = np.maximum(ca + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0.0)
            t = t + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t = np.clip(t, 250.0, 600.0)
            clock += DT
            step += 1
    return Trajectory(time=np.array(times), t_true=np.stack(tt, 1), t_meas=np.stack(tm, 1),
                      ca=np.stack(cas, 1), tc=np.stack(tcs, 1), final=State(ca, t, integ, clock))


def violates(traj: Trajectory) -> np.ndarray:
    """Boolean (B,): did the TRUE temperature leave the safe envelope?"""
    return (traj.t_true.max(1) > T_MAX) | (traj.t_true.min(1) < T_MIN)
