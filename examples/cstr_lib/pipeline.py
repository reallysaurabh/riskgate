"""Episodes, the checker, baselines, calibration and evaluation.

Pipeline for one episode
------------------------
1. A fault (or none) starts developing at t = 0 in a plant whose physics differ
   slightly from the twin's (unknown model error).
2. After a short observation window the diagnoser scores 19 fault hypotheses.
3. The doer (mock or real LLM) proposes a ranked list of supervisory actions.
4. A method decides what reaches the plant:
     no_checker     execute the doer's first choice
     twin_gate      deterministic check on the twin under the single best hypothesis
                    (the 2025-26 state of the art: binary, trusts the twin)
     conformal_top  conformal tube + CRC threshold, single best hypothesis
     cra            certify / reject / ask: conformal tubes under EVERY plausible
                    hypothesis + CRC threshold + out-of-distribution guard
     cra_probe      cra, plus a certified probing action when hypotheses are ambiguous
5. Ground truth: the real plant is simulated under the executed action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import plant as P
from .conformal import conformal_quantile, crc_threshold
from .doer import ACTIONS, MockLLMDoer
from .plant import PlantParams, Segment, State
from .twin import HYPOTHESES, SIG_CA, SIG_T, SIG_TC, Hypothesis, Observation, diagnose, estimate_state, twin_params

WINDOW = 6.0       # minutes of observation before the first decision
HORIZON = 30.0     # minutes over which an action must keep the TRUE temperature in the envelope
PRE_RUN = 15.0     # minutes of fault-free operation before onset (lets the plant settle)
NOISE_T, NOISE_CA = 0.15, 0.004
ACTION_NAMES = list(ACTIONS)

PROBES: dict[str, list[Segment]] = {
    # a probe must only be safe until the checker re-decides (probe + 3 min of slack)
    "probe_backup_sensor_3min": [Segment(3.0, backup_sensor=True), Segment(3.0)],
    "probe_feed_minus10_3min": [Segment(3.0, q_scale=0.9), Segment(3.0)],
    "probe_sp_plus2_3min": [Segment(3.0, sp=P.T_SP_NOM + 2.0), Segment(3.0)],
}


# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------
@dataclass
class Episode:
    seed: int
    kind: str
    magnitude: float
    params: PlantParams          # true plant (batch 1)
    obs: Observation = None
    state_now: State = None      # true plant state at decision time
    window_true: np.ndarray = None   # true temperature during the observation window (for display)
    probe_true: np.ndarray = None


def sample_episode(seed: int, shift: bool = False) -> Episode:
    rng = np.random.default_rng(seed)
    if shift:
        kinds, probs = ["none", "cooling_loss", "feed_conc", "sensor_bias", "sensor_bias_low"], [.1, .25, .25, .25, .15]
        sd_k0, sd_ua, sd_tf = 0.04, 0.05, 1.5
    else:
        kinds, probs = ["none", "cooling_loss", "feed_conc", "sensor_bias"], [.16, .28, .28, .28]
        sd_k0, sd_ua, sd_tf = 0.015, 0.02, 0.5
    kind = str(rng.choice(kinds, p=probs))
    ua_s, caf_s, bias = 1.0, 1.0, 0.0
    if kind == "cooling_loss":
        ua_s = rng.uniform(0.45, 0.95) if shift else rng.uniform(0.50, 0.95)
    elif kind == "feed_conc":
        caf_s = rng.uniform(1.05, 1.90) if shift else rng.uniform(1.05, 1.75)
    elif kind == "sensor_bias":
        bias = rng.uniform(1.5, 13.0) if shift else rng.uniform(1.5, 10.0)
    elif kind == "sensor_bias_low":          # unseen fault type: transmitter reads LOW
        bias = -rng.uniform(3.0, 8.0)
    mag = {"cooling_loss": ua_s, "feed_conc": caf_s}.get(kind, bias)
    params = PlantParams(k0=np.array([P.K0_NOM * (1 + rng.normal(0, sd_k0))]),
                         ua=np.array([P.UA_NOM * (1 + rng.normal(0, sd_ua))]),
                         tf=np.array([P.TF_NOM + rng.normal(0, sd_tf)]),
                         caf_scale=np.array([caf_s]), ua_scale=np.array([ua_s]), sensor_bias=np.array([bias]))
    return Episode(seed=seed, kind=kind, magnitude=float(mag), params=params)


def _noise_rng(ep: Episode, tag: int) -> np.random.Generator:
    return np.random.default_rng([ep.seed, tag])


def _measure(ep: Episode, traj: P.Trajectory, rng: np.random.Generator):
    t_meas = traj.t_meas[0]  # controller already saw noisy measurement; re-noise for the operator view
    ca_meas = traj.ca[0] + rng.normal(0, NOISE_CA, traj.ca.shape[1])
    return t_meas, traj.tc[0], ca_meas


def observe(ep: Episode) -> Episode:
    rng = _noise_rng(ep, 1)
    x = P.initial_state(1)
    x.clock = -PRE_RUN
    pre = P.simulate(ep.params, x, [Segment(PRE_RUN)], rng=rng, meas_noise=NOISE_T)
    win = P.simulate(ep.params, pre.final, [Segment(WINDOW)], rng=rng, meas_noise=NOISE_T)
    t_m, tc, ca_m = _measure(ep, win, rng)
    ep.obs = Observation(time=win.time, t_meas=t_m, tc=tc, ca_meas=ca_m, segments=[Segment(WINDOW)],
                         x0_meas=(float(t_m[0]), float(ca_m[0]), float(tc[0])))
    ep.state_now = win.final
    ep.window_true = win.t_true[0]
    return ep


def extend_with_probe(ep: Episode, probe: Segment) -> Episode:
    """Apply a probe segment to the TRUE plant and append the new measurements."""
    rng = _noise_rng(ep, 2)
    tr = P.simulate(ep.params, ep.state_now, [probe], rng=rng, meas_noise=NOISE_T)
    t_m, tc, ca_m = _measure(ep, tr, rng)
    o = ep.obs
    new = Observation(time=np.concatenate([o.time, tr.time]), t_meas=np.concatenate([o.t_meas, t_m]),
                      tc=np.concatenate([o.tc, tc]), ca_meas=np.concatenate([o.ca_meas, ca_m]),
                      segments=o.segments + [probe], x0_meas=o.x0_meas)
    return Episode(ep.seed, ep.kind, ep.magnitude, ep.params, new, tr.final, ep.window_true, tr.t_true[0])


# ----------------------------------------------------------------------------
# Batched roll-outs of plans (a plan = list of segments with shared durations)
# ----------------------------------------------------------------------------
def _stack_plans(plans: list[list[Segment]], reps: int, tile: bool) -> list[Segment]:
    """Build batched segments; element order (outer, inner) = (rep, plan) if tile else (plan, rep)."""
    out = []
    for j in range(len(plans[0])):
        sp = np.array([pl[j].sp for pl in plans], float)
        q = np.array([pl[j].q_scale for pl in plans], float)
        bk = np.array([pl[j].backup_sensor for pl in plans], bool)
        f = (lambda a: np.tile(a, reps)) if tile else (lambda a: np.repeat(a, reps))
        out.append(Segment(plans[0][j].duration, sp=f(sp), q_scale=f(q), backup_sensor=f(bk)))
    return out


def twin_rollouts(hyps: list[Hypothesis], obs: Observation, plans: list[list[Segment]]) -> P.Trajectory:
    """Twin trajectories, shape (nH * nPlans, N), element index = h * nPlans + plan."""
    nh, npl = len(hyps), len(plans)
    tp = twin_params(hyps).repeat(npl)
    now = obs.now
    on_backup = obs.segments[-1].backup_sensor
    x = estimate_state(float(obs.t_meas[-1]), float(obs.ca_meas[-1]), float(obs.tc[-1]), obs.segments[-1].sp,
                       np.zeros(nh) if on_backup else twin_params(hyps).sensor_bias, clock=now,
                       ramp_fraction=float(np.clip(now / P.FAULT_RAMP, 0, 1)))
    return P.simulate(tp, x.repeat(npl), _stack_plans(plans, nh, tile=True))


def true_rollouts(ep: Episode, plans: list[list[Segment]], tag: int = 3) -> P.Trajectory:
    npl = len(plans)
    x = State(np.repeat(ep.state_now.ca, npl), np.repeat(ep.state_now.t, npl), np.repeat(ep.state_now.integ, npl),
              ep.state_now.clock)
    return P.simulate(ep.params.repeat(npl), x, _stack_plans(plans, 1, tile=True),
                      rng=_noise_rng(ep, tag), meas_noise=NOISE_T)


CLIP_LO, CLIP_HI = P.T_MIN - 15.0, P.T_MAX + 15.0   # errors far outside the envelope are irrelevant to safety


def clipT(t: np.ndarray) -> np.ndarray:
    return np.clip(t, CLIP_LO, CLIP_HI)


def margins(traj_t: np.ndarray, q: float) -> np.ndarray:
    """Worst-case distance of the tube [T-q, T+q] from the envelope boundary (negative = violation)."""
    traj_t = clipT(traj_t)
    return np.minimum(P.T_MAX - (traj_t + q), (traj_t - q) - P.T_MIN).min(axis=-1)


# ----------------------------------------------------------------------------
# Per-episode record (everything the methods need, computed once)
# ----------------------------------------------------------------------------
@dataclass
class Record:
    ep: Episode
    residuals: np.ndarray
    plausible: np.ndarray            # bool over HYPOTHESES
    ood: bool
    top: int                         # index of best hypothesis
    ranked: list[str]
    m_hyp: np.ndarray                # (nH, nA) margins of the set-calibrated tube under each hypothesis
    m_point: np.ndarray              # (nH, nA) margins without tube (q = 0)
    m_top_tube: np.ndarray           # (nA,) margins of the top-hypothesis-calibrated tube
    unsafe: np.ndarray = None        # (nA,) ground truth (calibration / evaluation)
    extra: dict = field(default_factory=dict)

    def robust_margin(self) -> np.ndarray:
        if self.ood or not self.plausible.any():
            return np.full(self.m_hyp.shape[1], -np.inf)
        return self.m_hyp[self.plausible].min(axis=0)

    def top_margin(self) -> np.ndarray:
        return self.m_top_tube


@dataclass
class Calibration:
    tau: float          # hypothesis-set threshold (true kind retained w.p. >= 1 - delta)
    tau_ood: float      # if even the best hypothesis fits worse than this -> out of distribution
    q_tube: float       # tube half-width: truth within q of SOME plausible hypothesis' prediction w.p. >= 1-a
    q_top: float = 0.0  # tube half-width for the single top hypothesis (baseline)
    lam_cra: float = 0.0
    lam_top: float = 0.0
    lam_p1: float = 0.0     # cra_probe: stage-1 threshold (calibrated at share * alpha)
    lam_p2: float = 0.0     # cra_probe: post-probe threshold (calibrated on the probed sub-population)
    alpha: float = 0.05


def build_record(ep: Episode, cal: Calibration | None, doer, rng, with_truth: bool = True) -> Record:
    res = diagnose(ep.obs)
    tau = cal.tau if cal else np.inf
    plausible = res <= tau
    ood = bool(cal is not None and res.min() > cal.tau_ood)
    top = int(np.argmin(res))
    ranked, why = doer.propose(ep.obs, res, rng)
    plans = [[ACTIONS[a].segment(HORIZON)] for a in ACTION_NAMES]
    tw = twin_rollouts(HYPOTHESES, ep.obs, plans)
    tt = tw.t_true.reshape(len(HYPOTHESES), len(plans), -1)
    q = cal.q_tube if cal else 0.0
    q_top = cal.q_top if cal else 0.0
    rec = Record(ep, res, plausible, ood, top, ranked, margins(tt, q), margins(tt, 0.0), margins(tt[top], q_top))
    rec.extra["why"] = why
    if with_truth:
        tr = true_rollouts(ep, plans)
        rec.unsafe = P.violates(tr)
        rec.extra["true_traj"] = tr
    rec.extra["twin_traj"] = tt
    return rec


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def _true_kind_indices(kind: str) -> list[int]:
    return [i for i, h in enumerate(HYPOTHESES) if h.kind == kind]


LAMBDAS = np.round(np.arange(-60.0, 15.01, 0.1), 2)   # wide enough that no score is truncated


def policy_loss_curves(recs: list["Record"], margin_fn, lambdas: np.ndarray = LAMBDAS) -> np.ndarray:
    """Per-episode loss of the EXECUTED action as a function of lambda, made monotone.

    L_i(lam) = 1{the first doer-ranked action with margin >= lam is unsafe}. This raw loss need
    not be monotone in lam, so we use its upper envelope  L~_i(lam) = max_{lam' >= lam} L_i(lam'),
    which is non-increasing and >= L_i, so CRC on L~ still bounds the true risk.
    """
    out = np.zeros((len(recs), len(lambdas)))
    for i, r in enumerate(recs):
        m = margin_fn(r)
        order = [ACTION_NAMES.index(a) for a in r.ranked]
        for j, lam in enumerate(lambdas):
            act = next((k for k in order if m[k] >= lam), None)
            out[i, j] = float(act is not None and r.unsafe[act])
    return np.maximum.accumulate(out[:, ::-1], axis=1)[:, ::-1]


def crc_policy(recs: list["Record"], margin_fn, alpha: float, lambdas: np.ndarray = LAMBDAS) -> float:
    if not recs:
        return float("inf")
    curves = policy_loss_curves(recs, margin_fn, lambdas)
    n = len(recs)
    for j, lam in enumerate(lambdas):
        if n / (n + 1) * curves[:, j].mean() + 1 / (n + 1) <= alpha:
            return float(lam)
    return float("inf")


def calibrate(seeds_a: list[int], seeds_b: list[int], alpha: float = 0.05, delta: float = 0.05,
              alpha_tube: float = 0.05, doer=None, verbose: bool = True,
              probe_share: float = 0.6) -> tuple[Calibration, list[Record]]:
    doer = doer or MockLLMDoer()
    # --- stage A: hypothesis-set threshold, OOD threshold, tube widths -----------------
    eps, ress, r_true, r_best = [], [], [], []
    for s_ in seeds_a:
        ep = observe(sample_episode(s_))
        res = diagnose(ep.obs)
        eps.append(ep); ress.append(res)
        r_true.append(res[_true_kind_indices(ep.kind)].min())
        r_best.append(res.min())
    tau = conformal_quantile(np.array(r_true), delta)
    tube_set, tube_top = [], []
    plans = [[ACTIONS[a].segment(HORIZON)] for a in ACTION_NAMES]
    for ep, res in zip(eps, ress):
        a = int(np.random.default_rng(ep.seed).integers(len(ACTION_NAMES)))
        tw = clipT(twin_rollouts(HYPOTHESES, ep.obs, [plans[a]]).t_true)        # (nH, N)
        tr = clipT(true_rollouts(ep, [plans[a]]).t_true[0])
        # Safety depends only on the extremes of the trajectory (the margin uses max T and min T),
        # so we calibrate the error of the predicted PEAK and TROUGH, not pointwise error
        # (pointwise error explodes on runaway *timing*, which is irrelevant to the verdict).
        err = np.maximum(np.abs(tw.max(axis=1) - tr.max()), np.abs(tw.min(axis=1) - tr.min()))
        plaus = res <= tau
        # score: distance from the truth to the NEAREST plausible hypothesis' prediction
        tube_set.append(err[plaus].min() if plaus.any() else np.inf)
        tube_top.append(err[int(np.argmin(res))])
    cal = Calibration(tau=tau, tau_ood=conformal_quantile(np.array(r_best), 0.01),
                      q_tube=conformal_quantile(np.array(tube_set), alpha_tube),
                      q_top=conformal_quantile(np.array(tube_top), alpha_tube), alpha=alpha)
    if verbose:
        print(f"[cal-A] tau={cal.tau:.2f}  tau_ood={cal.tau_ood:.2f}  q_set={cal.q_tube:.2f} K  q_top={cal.q_top:.2f} K")
    # --- stage B: CRC thresholds on the accept decision --------------------------------
    recs = []
    for s_ in seeds_b:
        ep = observe(sample_episode(s_))
        recs.append(build_record(ep, cal, doer, np.random.default_rng([s_, 9])))
    cal.lam_cra = crc_policy(recs, Record.robust_margin, alpha)
    cal.lam_top = crc_policy(recs, Record.top_margin, alpha)
    # two-stage calibration for probing: stage 1 at probe_share*alpha, stage 2 at alpha on the
    # sub-population that actually gets probed -> total risk <= probe_share*alpha + alpha*P(probe)
    cal.lam_p1 = crc_policy(recs, Record.robust_margin, probe_share * alpha)
    probed = []
    for r in recs:
        if _first_certified(r.ranked, r.robust_margin(), cal.lam_p1) is None and not r.ood:
            name, _ = choose_probe(r, cal, lam=cal.lam_p1)
            if name is not None:
                ep2 = extend_with_probe(r.ep, PROBES[name][0])
                probed.append(build_record(ep2, cal, doer, np.random.default_rng([r.ep.seed, 11])))
    cal.lam_p2 = crc_policy(probed, Record.robust_margin, alpha) if len(probed) >= 20 else np.inf
    if verbose:
        print(f"[cal-B] lambda_cra={cal.lam_cra:.2f}  lambda_top={cal.lam_top:.2f}  "
              f"lambda_p1={cal.lam_p1:.2f}  lambda_p2={cal.lam_p2:.2f} (n_probed={len(probed)}, alpha={alpha})")
    return cal, recs


# ----------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------
@dataclass
class Decision:
    method: str
    verdict: str            # "execute" | "ask"
    action: str | None
    unsafe: bool            # executed AND the true trajectory left the envelope
    probed: str | None = None
    note: str = ""
    detail: dict = field(default_factory=dict)


def _first_certified(ranked: list[str], margin: np.ndarray, lam: float) -> str | None:
    for a in ranked:
        if margin[ACTION_NAMES.index(a)] >= lam:
            return a
    return None


def decide(rec: Record, cal: Calibration, method: str, doer=None) -> Decision:
    r = rec
    if method == "no_checker":
        a = r.ranked[0]
    elif method == "twin_gate":
        a = _first_certified(r.ranked, r.m_point[r.top], 0.0)
    elif method == "conformal_top":
        a = _first_certified(r.ranked, r.top_margin(), cal.lam_top)
    elif method == "cra":
        a = _first_certified(r.ranked, r.robust_margin(), cal.lam_cra)
    elif method == "cra_probe":
        a = _first_certified(r.ranked, r.robust_margin(), cal.lam_p1)
        if a is None and not r.ood:
            return _probe_then_decide(r, cal, doer or MockLLMDoer())
    else:
        raise ValueError(method)
    if a is None:
        return Decision(method, "ask", None, False, note="ood" if r.ood else "")
    return Decision(method, "execute", a, bool(r.unsafe[ACTION_NAMES.index(a)]))


PROBE_PLAIN = {
    "probe_backup_sensor_3min": "read the spare thermometer for 3 minutes",
    "probe_feed_minus10_3min": "trim the feed by 10% for 3 minutes",
    "probe_sp_plus2_3min": "nudge the target temperature up by 2 K for 3 minutes",
}
MIN_SEPARATION = 1.0


def choose_probe(rec: Record, cal: Calibration, lam: float | None = None) -> tuple[str | None, float]:
    """Pick the certified probe that best separates plausible hypotheses that DISAGREE about safety.

    Two plausible hypotheses disagree if some candidate action is certifiable under one and not
    the other. Only disagreements matter: resolving them is what can unlock a certified action.
    """
    lam = cal.lam_p2 if lam is None else lam
    if not np.isfinite(lam):
        lam = cal.lam_p1
    hyps_idx = np.flatnonzero(rec.plausible)
    if len(hyps_idx) < 2 or rec.ood:
        return None, 0.0
    ok = rec.m_hyp[hyps_idx] >= lam                              # (nP, nA)
    pairs = [(a, b) for a in range(len(hyps_idx)) for b in range(a + 1, len(hyps_idx))
             if np.any(ok[a] != ok[b])]
    if not pairs:
        return None, 0.0
    hyps = [HYPOTHESES[i] for i in hyps_idx]
    names = list(PROBES)
    hold = [Segment(sg.duration) for sg in PROBES[names[0]]]               # baseline: just hold
    tw = twin_rollouts(hyps, rec.ep.obs, [PROBES[n] for n in names] + [hold])
    nh, npb = len(hyps), len(names)
    mm = margins(tw.t_true.reshape(nh, npb + 1, -1), cal.q_tube)           # (nH, nProbes + 1)
    # A probe is admissible if, under EVERY plausible hypothesis, it is certifiably safe or no worse
    # than holding (some hypotheses are doomed in the short term whatever we do; probing then
    # costs nothing and can rule them out).
    safe = np.all((mm[:, :npb] >= lam) | (mm[:, :npb] >= mm[:, [npb]] - 0.5), axis=0)
    tw_t_meas = tw.t_meas.reshape(nh, npb + 1, -1)[:, :npb].reshape(nh * npb, -1)
    tw_tc = tw.tc.reshape(nh, npb + 1, -1)[:, :npb].reshape(nh * npb, -1)
    tw_ca = tw.ca.reshape(nh, npb + 1, -1)[:, :npb].reshape(nh * npb, -1)
    k = int(round(3.0 / 0.5)) + 1                                            # samples covering the probe
    feats = np.stack([tw_t_meas.reshape(nh, npb, -1)[..., :k] / SIG_T,
                      tw_tc.reshape(nh, npb, -1)[..., :k] / SIG_TC,
                      tw_ca.reshape(nh, npb, -1)[..., :k] / SIG_CA], -1)
    best, best_sep = None, MIN_SEPARATION
    for j, name in enumerate(names):
        if not safe[j]:
            continue
        sep = min(np.sqrt(((feats[a, j] - feats[b, j]) ** 2).sum(-1)).max() for a, b in pairs)
        if sep > best_sep:
            best, best_sep = name, sep
    return best, best_sep


def _probe_then_decide(rec: Record, cal: Calibration, doer) -> Decision:
    name, sep = choose_probe(rec, cal)
    if name is None:
        return Decision("cra_probe", "ask", None, False, note="no certifiable probe")
    probe_seg = PROBES[name][0]
    ep2 = extend_with_probe(rec.ep, probe_seg)
    rec2 = build_record(ep2, cal, doer, np.random.default_rng([rec.ep.seed, 11]))
    a = _first_certified(rec2.ranked, rec2.robust_margin(), cal.lam_p2)
    probe_unsafe = bool(P.violates(P.simulate(rec.ep.params, rec.ep.state_now, [probe_seg]))[0])
    detail = {"rec2": rec2, "ep2": ep2, "separation": sep}
    if a is None:
        return Decision("cra_probe", "ask", None, probe_unsafe, probed=name, note="ambiguous after probe",
                        detail=detail)
    return Decision("cra_probe", "execute", a, bool(rec2.unsafe[ACTION_NAMES.index(a)]) or probe_unsafe,
                    probed=name, detail=detail)


METHODS = ["no_checker", "twin_gate", "conformal_top", "cra", "cra_probe"]


def evaluate(seeds: list[int], cal: Calibration, shift: bool = False, doer=None,
             methods: list[str] = METHODS) -> tuple[list[Record], dict[str, list[Decision]]]:
    doer = doer or MockLLMDoer()
    recs, out = [], {m: [] for m in methods}
    for s in seeds:
        ep = observe(sample_episode(s, shift=shift))
        rec = build_record(ep, cal, doer, np.random.default_rng([s, 9]))
        recs.append(rec)
        for m in methods:
            out[m].append(decide(rec, cal, m, doer))
    return recs, out


def summarise(recs: list[Record], decisions: dict[str, list[Decision]]) -> dict[str, dict[str, float]]:
    n = len(recs)
    any_safe = np.array([not r.unsafe.all() for r in recs])
    table = {}
    for m, ds in decisions.items():
        unsafe = np.array([d.unsafe for d in ds])
        ask = np.array([d.verdict == "ask" for d in ds])
        ok = np.array([d.verdict == "execute" and not d.unsafe for d in ds])
        table[m] = {
            "unsafe_rate": unsafe.mean(),
            "ask_rate": ask.mean(),
            "safe_execution_rate": ok.mean(),
            "avoidable_ask_rate": (ask & any_safe).mean(),
            "probe_rate": np.mean([d.probed is not None for d in ds]),
            "n": n,
        }
    return table
