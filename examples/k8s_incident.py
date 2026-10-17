"""Example domain 2: an SRE agent responding to a latency incident on a Kubernetes service.

Simulated system (all numbers chosen to resemble a typical production checkout service):
  * `checkout-api` Deployment, 4 workers per pod, replicas sized by the HPA for ~55 % utilisation, quota 40
  * every pod opens a pool of 10 connections to a Postgres primary with max_connections = 200 (<= 20 pods)
  * a Redis cache absorbs 75 % of reads; each request then does ~3 primary queries; the primary
    sustains ~4,800 queries/s (about 50 % busy at peak)
  * diurnal traffic: 300-800 requests/s at the time of the incident
  * client SDKs retry failed requests (~0.3 retries per failure)

Latent incident causes (the agent cannot see them directly):
  traffic_surge   marketing push / flash sale: offered load x1.3-2.6
  slow_db_query   a deploy shipped an unindexed query: DB capacity x0.35-0.85
  cache_outage    Redis eviction storm / failover: 30-100 % of the cache hit ratio is lost
  bad_pod_version CPU regression in the new release: app service time x1.4-2.8
  none            alert fired on noise

Seven remediation actions, three read-only probes (each costs 2 minutes of incident time), and an
SLO-based safety spec: every minute from t+4 to t+30 must have p95 <= 800 ms and errors <= 2 %.
The truth simulator randomises plant parameters, adds metric noise and uses continuous fault
magnitudes; the twin uses nominal parameters and a grid of 13 hypotheses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

import riskgate as rg

NAME = "k8s_incident"
WORKERS, POOL, PG_MAX, QUOTA, HPA_TARGET, H0 = 4, 10, 200, 40, 0.55, 0.75
S_APP_MS, QPR, DB_CAP, DB_BASE_MS, TIMEOUT_MS = 35.0, 3.0, 4800.0, 1.5, 2000.0
T_DECIDE, HORIZON, GRACE, PROBE_MIN = 10, 30, 4, 2
P95_SLO, ERR_SLO = 800.0, 0.02
META = {"system": "checkout-api (Kubernetes Deployment) + Postgres primary", "simulated": True,
        "note": "Synthetic incidents from a queueing simulator with realistic parameters; not production data.",
        "slo": {"p95_ms": P95_SLO, "error_rate": ERR_SLO, "window_min": [GRACE, HORIZON]}}

ACTIONS = {
    "noop": rg.Action("noop"),
    "scale_out_1_5x": rg.Action("kubectl_scale", {"deployment": "checkout-api", "factor": 1.5}),
    "scale_out_2x": rg.Action("kubectl_scale", {"deployment": "checkout-api", "factor": 2.0}),
    "rollback_deploy": rg.Action("kubectl_rollout_undo", {"deployment": "checkout-api"}),
    "rate_limit": rg.Action("set_rate_limit", {"route": "/checkout", "rps_factor": 1.15}),
    "failover_reads": rg.Action("route_reads_to_replica", {"db": "orders-pg", "share": 0.4}),
    "rollback_and_scale": rg.Action("rollback_and_scale", {"deployment": "checkout-api", "factor": 1.5}),
}
NAMES = {a.key: k for k, a in ACTIONS.items()}
PROBES = [rg.Action("pg_stat_statements_top", {"db": "orders-pg"}, read_only=True),
          rg.Action("redis_info_stats", {"cache": "checkout-redis"}, read_only=True),
          rg.Action("canary_version_compare", {"deployment": "checkout-api"}, read_only=True)]
PROBE_NOISE = {"pg_stat_statements_top": 0.12, "redis_info_stats": 0.08, "canary_version_compare": 0.08}
PROBE_KEY = {"pg_stat_statements_top": "db_ms", "redis_info_stats": "cache_hit", "canary_version_compare": "canary_ratio"}
KINDS = ["none", "traffic_surge", "slow_db_query", "cache_outage", "bad_pod_version"]


# ----------------------------------------------------------------------------------------------
# Dynamics (vectorised over a batch of scenario x action elements)
# ----------------------------------------------------------------------------------------------
@dataclass
class Params:
    lam0: np.ndarray        # baseline offered load [req/s]
    surge: np.ndarray       # traffic multiplier at full development
    db_f: np.ndarray        # DB capacity factor
    retry: np.ndarray       # client retries per failed request
    cache_loss: np.ndarray  # fraction of the cache hit ratio lost at full development
    app_g: np.ndarray       # app service time factor
    deploy_fault: np.ndarray  # fault came with the last deploy (rollback fixes it)
    blip: np.ndarray        # downstream error blip at onset (triggers retry storms)
    ramp: np.ndarray        # minutes for the fault to fully develop
    s_app: np.ndarray
    qpr: np.ndarray
    db_cap: np.ndarray
    n0: np.ndarray          # replicas before the incident

    def take(self, idx):
        return Params(**{k: np.asarray(v)[idx] for k, v in vars(self).items()})

    @property
    def b(self):
        return self.lam0.shape[0]


def erlang_c(c: int, a: np.ndarray) -> np.ndarray:
    a = np.minimum(a, c - 1e-6)
    s, term = np.zeros_like(a), np.ones_like(a)
    for k in range(c):
        if k > 0:
            term = term * a / k
        s += term
    top = term * a / c * c / (c - a)
    return top / (s + top)


@dataclass
class Plan:
    """Action schedule per batch element (applied at minute `at`)."""

    at: np.ndarray
    factor: np.ndarray        # scale factor (1 = none)
    rollback: np.ndarray
    rate_limit: np.ndarray
    failover: np.ndarray


def plan_for(keys: list[str], at: float) -> Plan:
    f = {"scale_out_1_5x": 1.5, "scale_out_2x": 2.0, "rollback_and_scale": 1.5}
    return Plan(at=np.full(len(keys), float(at)), factor=np.array([f.get(k, 1.0) for k in keys]),
                rollback=np.array([k in ("rollback_deploy", "rollback_and_scale") for k in keys]),
                rate_limit=np.array([k == "rate_limit" for k in keys]),
                failover=np.array([k == "failover_reads" for k in keys]))


def simulate(p: Params, plan: Plan, t_end: int, rng: np.random.Generator | None = None, noise: float = 0.0):
    """Minute-resolution fluid/queueing simulation from incident onset (t=0) to t_end."""
    b = p.b
    e_prev = np.full(b, 0.001)          # server-side errors (5xx/timeouts) drive client retries
    keys = ("rps", "p95_ms", "error_rate", "cpu", "db_cpu", "db_conn", "pods", "db_ms", "cache_hit", "canary_ratio")
    out = {k: np.zeros((b, t_end)) for k in keys}
    for t in range(t_end):
        f = np.clip(t / p.ramp, 0.0, 1.0)
        after = t >= plan.at
        rolled = plan.rollback & (t >= plan.at + 4)
        rolling = plan.rollback & after & ~rolled
        fixed = rolled & p.deploy_fault
        surge = 1 + f * (p.surge - 1)
        db_f = np.where(fixed, 1.0, 1 + f * (p.db_f - 1))
        g = np.where(fixed, 1.0, 1 + f * (p.app_g - 1))
        retry = p.retry
        blip = np.where((t >= 1) & (t < 4), p.blip, 0.0)
        lam_user = p.lam0 * surge
        if rng is not None and noise:
            lam_user = lam_user * (1 + rng.normal(0, noise / 2, b))
        lam_off = lam_user * (1 + retry * e_prev)
        cap = np.where(plan.rate_limit & (t >= plan.at + 1), 1.15 * p.lam0, np.inf)
        lam_adm = np.minimum(lam_off, cap)
        e_rl = 1 - lam_adm / lam_off
        pods = np.where(after & (t >= plan.at + 2), np.minimum(QUOTA, np.ceil(p.n0 * plan.factor)), p.n0).astype(float)
        ready = pods * np.where(rolling, 0.75, 1.0)
        conn_pods = np.minimum(ready, PG_MAX // POOL)
        e_conn = 1 - conn_pods / ready
        hit = H0 * (1 - f * p.cache_loss)
        qpr = p.qpr * (1 - hit) / (1 - H0) * np.where(plan.failover & (t >= plan.at + 3), 0.6, 1.0)
        qps = lam_adm * (1 - e_conn) * qpr
        rho_db = qps / (p.db_cap * db_f)
        e_db = np.clip(1 - 1 / np.maximum(rho_db, 1e-9), 0, 1)
        d_ms = DB_BASE_MS / np.maximum(1 - np.minimum(rho_db, 0.985), 0.015)
        s_ms = p.s_app * g + qpr * d_ms
        mu = conn_pods * WORKERS / (s_ms / 1000.0)
        lam_srv = lam_adm * (1 - e_conn) * (1 - e_db)
        rho = lam_srv / mu
        a = rho * WORKERS
        pw = erlang_c(WORKERS, a)
        wq95 = np.where(pw > 0.05, np.log(np.maximum(pw, 1e-12) / 0.05) * s_ms / (WORKERS * np.maximum(1 - rho, 1e-3)), 0.0)
        p95 = np.where(rho < 0.995, 2.0 * s_ms + wq95, TIMEOUT_MS)
        p95 = np.minimum(p95, TIMEOUT_MS)
        e_sat = np.where(rho >= 0.995, 1 - 0.995 / rho, 0.0) + np.where(p95 >= TIMEOUT_MS, 0.05, 0.0)
        e_switch = np.where(plan.failover & (t >= plan.at) & (t < plan.at + 1), 0.01, 0.0)
        err = 1 - (1 - e_rl) * (1 - e_conn) * (1 - e_db) * (1 - np.clip(e_sat, 0, 1)) * (1 - blip) * (1 - e_switch)
        err = np.clip(err + 0.001, 0, 1)
        e_prev = np.clip(1 - (1 - err) / np.maximum(1 - e_rl, 1e-9), 0, 1)   # 429s honour Retry-After
        vals = {"rps": lam_off, "p95_ms": p95, "error_rate": err, "cpu": np.clip(rho * p.s_app * g / s_ms, 0, 1),
                "db_cpu": np.clip(rho_db, 0, 1), "db_conn": ready * POOL, "pods": ready, "db_ms": d_ms,
                "cache_hit": hit, "canary_ratio": g}
        for k, v in vals.items():
            if rng is not None and noise and k not in ("pods", "db_conn"):
                v = v * (1 + rng.normal(0, noise, b))
            out[k][:, t] = v
    return out


# ----------------------------------------------------------------------------------------------
# Hypotheses, twin and doer
# ----------------------------------------------------------------------------------------------
GRID = ([("none", 0.0)] + [("traffic_surge", m) for m in (1.4, 1.8, 2.3)] + [("slow_db_query", m) for m in (0.75, 0.55, 0.4)]
        + [("cache_outage", m) for m in (0.4, 0.7, 1.0)] + [("bad_pod_version", m) for m in (1.5, 2.0, 2.6)])


def hpa_replicas(lam0, s_ms=S_APP_MS + QPR * DB_BASE_MS):
    return np.maximum(3, np.ceil(lam0 * s_ms / 1000.0 / (WORKERS * HPA_TARGET)))


def twin_params(hyps: list[tuple[str, float]], lam0: float, deploy_seen: bool, n0: float, db_cap: float = DB_CAP) -> Params:
    n = len(hyps)
    k = [h[0] for h in hyps]
    m = np.array([h[1] for h in hyps])
    one = np.ones(n)
    return Params(lam0=lam0 * one, surge=np.where(np.array(k) == "traffic_surge", m, 1.0),
                  db_f=np.where(np.array(k) == "slow_db_query", m, 1.0),
                  retry=0.3 * one, cache_loss=np.where(np.array(k) == "cache_outage", m, 0.0),
                  app_g=np.where(np.array(k) == "bad_pod_version", m, 1.0),
                  deploy_fault=np.isin(k, ["slow_db_query", "bad_pod_version"]) & deploy_seen,
                  blip=0.0 * one, ramp=5.0 * one,
                  s_app=S_APP_MS * one, qpr=QPR * one, db_cap=db_cap * one, n0=n0 * one)


OBS_KEYS = {"rps": 0.06, "p95_ms": None, "error_rate": None, "cpu": 0.05, "db_cpu": 0.05}


@dataclass
class Context:
    t: int                                   # decision minute
    lam0: float                              # pre-incident request rate (from dashboards)
    deploy_seen: bool                        # a deploy happened in the last 30 minutes
    window: dict                             # observed metrics, minutes 0..t-1
    probes: dict = field(default_factory=dict)   # probe name -> observed value
    plan_at: int = T_DECIDE


def residuals(ctx: Context, hyps: list[tuple[str, float]], db_cap: float = DB_CAP) -> np.ndarray:
    tp = twin_params(hyps, ctx.lam0, ctx.deploy_seen, ctx.window["pods"][0], db_cap)
    sim = simulate(tp, plan_for(["noop"] * len(hyps), 1e9), ctx.t)
    r = np.zeros(len(hyps))
    for k, sd in OBS_KEYS.items():
        obs, pred = np.asarray(ctx.window[k]), sim[k]
        if k == "p95_ms":
            z = (np.log(pred) - np.log(obs)) / 0.12
        elif k == "error_rate":
            z = (np.log(pred + 0.002) - np.log(obs + 0.002)) / 0.25
        else:
            z = (pred - obs) / (sd * np.maximum(np.abs(obs), 1e-3) if k == "rps" else sd)
        r += np.mean(z ** 2, axis=1)
    for name, val in ctx.probes.items():
        key = PROBE_KEY[name]
        pred = sim[key][:, -1] if key != "db_ms" else sim[key][:, -1] / DB_BASE_MS
        r += ((pred - val) / (PROBE_NOISE[name] * max(abs(val), 0.2))) ** 2
    dep = np.array([ctx.deploy_seen == (h[0] in ("slow_db_query", "bad_pod_version")) for h in hyps])
    r += np.where(dep, 0.0, 1.5)                                 # deploy evidence (deploys are frequent)
    return np.sqrt(r / (len(OBS_KEYS) + len(ctx.probes) + 1))


class Diagnoser:
    def __init__(self, db_cap: float = DB_CAP):
        self.db_cap = db_cap
        self.version = "k8s-diag-v1" if db_cap == DB_CAP else f"k8s-diag-v1-cap{db_cap:.0f}"

    def hypotheses(self, ctx: Context) -> list[rg.Hypothesis]:
        r = residuals(ctx, GRID, self.db_cap)
        return [rg.Hypothesis(f"{k}:{m:g}" if k != "none" else "none", family=k, params={"kind": k, "mag": m}, score=float(s))
                for (k, m), s in zip(GRID, r)]


class Twin:
    def __init__(self, db_cap: float = DB_CAP):
        self.db_cap = db_cap
        self.version = "k8s-twin-v1" if db_cap == DB_CAP else f"k8s-twin-v1-cap{db_cap:.0f}"

    def predict_batch(self, actions, hyps, ctx: Context):
        hl = [(h.params["kind"], h.params["mag"]) for h in hyps]
        keys = [NAMES.get(a.key, a.name) for a in actions]
        na, nh = len(actions), len(hl)
        tp = twin_params(hl, ctx.lam0, ctx.deploy_seen, ctx.window["pods"][0], self.db_cap).take(np.tile(np.arange(nh), na))
        is_probe = [a.read_only for a in actions]
        plan_keys = ["noop" if pr else k for k, pr in zip(keys, is_probe) for _ in range(nh)]
        sim = simulate(tp, plan_for(plan_keys, ctx.plan_at), ctx.plan_at + HORIZON + 1)
        w0, w1 = ctx.plan_at + GRACE, ctx.plan_at + HORIZON
        out = []
        for i, a in enumerate(actions):
            row = []
            for j in range(nh):
                e = i * nh + j
                o = {"p95_ms": sim["p95_ms"][e, w0:w1], "error_rate": sim["error_rate"][e, w0:w1]}
                if a.read_only:
                    t_obs = min(ctx.plan_at + PROBE_MIN, sim["db_ms"].shape[1]) - 1
                    val = sim[PROBE_KEY[a.name]][e, t_obs] / (DB_BASE_MS if a.name == "pg_stat_statements_top" else 1.0)
                    o["observation"] = [math.log(max(val, 1e-3)) / PROBE_NOISE[a.name]]   # noise units (log scale)
                row.append(o)
            out.append(row)
        return out

    def predict(self, action, hyp, ctx):
        return self.predict_batch([action], [hyp], ctx)[0][0]


SPEC = rg.Spec(rg.Upper("p95_ms", P95_SLO, scale=P95_SLO, name="p95<=800ms"),
               rg.Upper("error_rate", ERR_SLO, scale=ERR_SLO, name="errors<=2%"), floor=-5.0)


def doer_rank(ctx: Context, rng: np.random.Generator) -> list[str]:
    """A plausible LLM SRE assistant: pattern-matches dashboards, loves scaling out."""
    w = ctx.window
    cpu, dbc, err = w["cpu"][-1], w["db_cpu"][-1], w["error_rate"][-1]
    rps_up = w["rps"][-1] / ctx.lam0
    if err > 0.05 and ctx.deploy_seen:
        r = ["rollback_deploy", "rollback_and_scale", "scale_out_2x", "scale_out_1_5x", "failover_reads", "rate_limit", "noop"]
    elif cpu > 0.6 or (dbc > 0.75 and rng.random() < 0.6):
        r = ["scale_out_2x", "scale_out_1_5x", "rollback_and_scale", "failover_reads", "rate_limit", "rollback_deploy", "noop"]
    elif rps_up > 1.4:
        r = ["rate_limit", "scale_out_1_5x", "scale_out_2x", "failover_reads", "rollback_and_scale", "rollback_deploy", "noop"]
    elif dbc > 0.75:
        r = ["failover_reads", "scale_out_1_5x", "rollback_deploy", "scale_out_2x", "rate_limit", "rollback_and_scale", "noop"]
    else:
        r = ["noop", "scale_out_1_5x", "failover_reads", "rollback_deploy", "rate_limit", "scale_out_2x", "rollback_and_scale"]
    if rng.random() < 0.15:
        r[0], r[1] = r[1], r[0]
    return r


# ----------------------------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------------------------
def sample_truth(rng: np.random.Generator, shift: bool, db_cap_factor: float = 1.0) -> tuple[str, float, Params]:
    probs = [0.10, 0.25, 0.22, 0.18, 0.25]
    kind = str(rng.choice(KINDS, p=probs))
    mag = {"none": 0.0, "traffic_surge": rng.uniform(1.3, 2.9 if shift else 2.6),
           "slow_db_query": rng.uniform(0.3 if shift else 0.35, 0.85), "cache_outage": rng.uniform(0.3, 1.0),
           "bad_pod_version": rng.uniform(1.4, 3.0 if shift else 2.8)}[kind]
    sd = 2.0 if shift else 1.0
    p = Params(lam0=np.array([rng.uniform(300, 800)]),
               surge=np.array([mag if kind == "traffic_surge" else 1.0]),
               db_f=np.array([mag if kind == "slow_db_query" else 1.0]),
               retry=np.array([rng.uniform(0.2, 0.4)]), cache_loss=np.array([mag if kind == "cache_outage" else 0.0]),
               app_g=np.array([mag if kind == "bad_pod_version" else 1.0]),
               deploy_fault=np.array([kind in ("slow_db_query", "bad_pod_version")]),
               blip=np.array([0.0]),
               ramp=np.array([rng.uniform(3, 8)]),
               s_app=np.array([rng.normal(S_APP_MS, 2.5 * sd)]), qpr=np.array([rng.uniform(2.8, 3.2)]),
               db_cap=np.array([rng.normal(DB_CAP, 250 * sd) * db_cap_factor]), n0=np.array([0.0]))
    p.n0 = hpa_replicas(p.lam0, p.s_app + p.qpr * DB_BASE_MS)
    return kind, mag, p


def _labels(p: Params, at: int, rng_seed, extra_noop_min: int = 0) -> dict:
    keys = list(ACTIONS)
    pp = p.take(np.zeros(len(keys), int))
    sim = simulate(pp, plan_for(keys, at), at + HORIZON, rng=np.random.default_rng(rng_seed), noise=0.03)
    w = slice(at + GRACE, at + HORIZON)
    bad = (sim["p95_ms"][:, w] > P95_SLO).any(1) | (sim["error_rate"][:, w] > ERR_SLO).any(1)
    return {ACTIONS[k].key: bool(b) for k, b in zip(keys, bad)}


def sample(seed: int, shift: bool = False, db_cap_factor: float = 1.0) -> rg.Episode:
    """One labelled incident. `db_cap_factor` changes the real database capacity (e.g. 0.6 after moving to a smaller
    instance class); the twin keeps its own assumption unless it is rebuilt with make_gate(db_cap=...)."""
    rng = np.random.default_rng(seed)
    kind, mag, p = sample_truth(rng, shift, db_cap_factor)
    deploy_seen = bool(p.deploy_fault[0]) if rng.random() < 0.9 else not bool(p.deploy_fault[0])
    if not p.deploy_fault[0]:
        deploy_seen = rng.random() < 0.35
    sim = simulate(p, plan_for(["noop"], 1e9), T_DECIDE, rng=np.random.default_rng([seed, 1]), noise=0.03)
    window = {k: sim[k][0].tolist() for k in ("rps", "p95_ms", "error_rate", "cpu", "db_cpu", "db_conn", "pods")}
    ctx = Context(t=T_DECIDE, lam0=float(p.lam0[0] * (1 + rng.normal(0, 0.03))), deploy_seen=deploy_seen, window=window)
    ranked = doer_rank(ctx, rng)
    cands = [ACTIONS[k] for k in ranked]
    unsafe = _labels(p, T_DECIDE, [seed, 2])
    ep = rg.Episode(ctx, cands, unsafe, true_family=kind,
                    meta={"truth": {"kind": kind, "magnitude": round(float(mag), 3)}, "seed": seed})
    # after-probe episodes: 2 minutes pass under noop while the probe runs, then decide
    full = simulate(p, plan_for(["noop"], 1e9), T_DECIDE + PROBE_MIN, rng=np.random.default_rng([seed, 1]), noise=0.03)
    for pr in PROBES:
        key = PROBE_KEY[pr.name]
        val = float(full[key][0, -1] / (DB_BASE_MS if key == "db_ms" else 1.0))
        val *= 1 + np.random.default_rng([seed, 3, len(pr.name)]).normal(0, PROBE_NOISE[pr.name])
        w2 = {k: full[k][0].tolist() for k in window}
        ctx2 = replace(ctx, t=T_DECIDE + PROBE_MIN, window=w2, probes={pr.name: val}, plan_at=T_DECIDE + PROBE_MIN)
        ep.after_probe[pr.key] = rg.Episode(ctx2, cands, _labels(p, T_DECIDE + PROBE_MIN, [seed, 2]), true_family=kind,
                                            meta={"probe": pr.name, "probe_value": round(val, 3)})
    return ep


def hypotheses():
    return Diagnoser()


def make_gate(hypotheses=None, probes=None, db_cap: float = DB_CAP, **kw) -> rg.Gate:
    return rg.Gate(Twin(db_cap), SPEC, hypotheses or Diagnoser(db_cap), probes=PROBES if probes is None else probes,
                   noop=ACTIONS["noop"], noop_tol=0.05, min_separation=2.0, **kw)


def record(ep: rg.Episode) -> dict:
    c: Context = ep.context
    alert = "HighErrorRate" if c.window["error_rate"][-1] > 0.02 else "HighLatencyP95" if c.window["p95_ms"][-1] > 500 else "LatencySLOBurnRate"
    return {
        "incident_id": f"INC-{ep.meta['seed']:06d}", "service": "checkout-api", "alert": alert,
        "minutes_since_onset": c.t, "baseline_rps": round(c.lam0, 1), "recent_deploy": c.deploy_seen,
        "metrics_1m": {k: [round(v, 4) for v in vals] for k, vals in c.window.items()},
        "agent_ranked_actions": [NAMES.get(a.key, a.name) for a in ep.candidates],
        "outcome_slo_breach_by_action": {NAMES[k]: v for k, v in ep.unsafe.items()},
        "ground_truth_cause": ep.meta["truth"],
    }
