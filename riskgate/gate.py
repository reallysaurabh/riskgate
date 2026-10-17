"""The gate: hypotheses -> consequences -> margins -> calibrated verdict.

Design rules (see docs/SPEC.md):
  R1 independence   the certify path never calls an LLM; LLM-derived inputs may only make the gate
                    more conservative (add hypotheses, add rules)
  R2 validity       the risk bound comes from conformal risk control, not from model quality:
                    a bad consequence model produces more ASK, not more unsafe certifications
  R3 fail closed    errors, timeouts, missing or stale calibration -> ASK, never CERTIFY
  R4 humility       the gate reports whether its guarantee is VALID, DEGRADED or VOID
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .conformal import ask_rate, conformal_quantile, crc_threshold, lambda_grid, policy_losses
from .spec import HardRule, Spec, _get
from .types import Action, Decision, GuaranteeStatus, Hypothesis, RiskGateWarning, Verdict, callable_digest, canonical, digest

log = logging.getLogger("riskgate")
INF = float("inf")
MAX_SCORE = 1e300      # +inf margins are clipped here so that lambda = +inf never certifies anything


# --------------------------------------------------------------------------------------------
# Component protocols
# --------------------------------------------------------------------------------------------
@runtime_checkable
class ConsequenceModel(Protocol):
    """Predicts the outcome of an action under a hypothesis. Optional: predict_batch(actions, hyps, ctx)."""

    version: str

    def predict(self, action: Action, hypothesis: Hypothesis, context: Any) -> Any: ...


@runtime_checkable
class HypothesisProvider(Protocol):
    version: str

    def hypotheses(self, context: Any) -> list[Hypothesis]: ...


class SingleHypothesis:
    """For domains without latent-state ambiguity."""

    version = "single-v1"

    def __init__(self, name: str = "nominal"):
        self._h = [Hypothesis(name, score=0.0)]

    def hypotheses(self, context: Any) -> list[Hypothesis]:
        return list(self._h)


class CallableModel:
    """Wrap plain functions as a ConsequenceModel."""

    def __init__(self, fn: Callable[[Action, Hypothesis, Any], Any], version: str = "callable-v1",
                 batch_fn: Callable[[list[Action], list[Hypothesis], Any], list[list[Any]]] | None = None):
        self.fn, self.version = fn, version
        if batch_fn is not None:
            self.predict_batch = batch_fn  # type: ignore[assignment]

    def predict(self, action: Action, hypothesis: Hypothesis, context: Any) -> Any:
        return self.fn(action, hypothesis, context)


# --------------------------------------------------------------------------------------------
# Calibration data and artefact
# --------------------------------------------------------------------------------------------
@dataclass
class Episode:
    """One calibration unit: a decision point with ground-truth labels.

    candidates  the doer's proposals in rank order (first = preferred)
    unsafe      action.key -> True if executing it led (or would lead) to a spec violation.
                'agnostic' mode needs a label for every candidate; unlabeled candidates are
                treated as unsafe (conservative).
    true_family optional label of the true hypothesis family (calibrates the plausible set)
    after_probe probe.key -> the Episode as it looks after executing that probe (two-stage probing);
                set meta["probe_unsafe"]=True on the post-probe episode if the probe itself caused harm
    """

    context: Any
    candidates: list[Action]
    unsafe: Mapping[str, bool]
    true_family: str | None = None
    after_probe: Mapping[str, Episode] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _enc(x: float | None) -> Any:
    if x is None:
        return None
    return "inf" if x == INF else "-inf" if x == -INF else x


def _dec(x: Any) -> float | None:
    if x is None:
        return None
    return INF if x == "inf" else -INF if x == "-inf" else float(x)


def _key_bytes(key: bytes | str) -> bytes:
    return key.encode() if isinstance(key, str) else key


class SignatureError(ValueError):
    """Raised when a calibration artefact's HMAC signature is missing or wrong."""


@dataclass
class Calibration:
    alpha: float
    mode: str
    delta: float
    tau: float
    tau_ood: float
    lam: float
    fingerprint: str
    n: int
    lam_probe1: float | None = None
    lam_probe2: float | None = None
    probe_share: float | None = None
    n_probe: int = 0
    created: str = ""
    guarantee: str = "expectation"                               # or "high_probability" (Learn-then-Test)
    delta_ltt: float | None = None
    extra_alphas: dict = field(default_factory=dict)             # additional risks controlled jointly
    curve: list[list[float]] = field(default_factory=list)       # [lambda, empirical risk, ask rate]
    reference_scores: list[float] = field(default_factory=list)  # top-candidate scores (drift monitoring)
    weighted: dict | None = None                                 # stored losses for covariate-shift weighting
    diagnostics: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return digest([self.fingerprint, self.created, _enc(self.lam), self.n], 12)

    @property
    def probing(self) -> bool:
        return self.lam_probe1 is not None and self.lam_probe2 is not None and math.isfinite(self.lam_probe2)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("tau", "tau_ood", "lam", "lam_probe1", "lam_probe2"):
            d[k] = _enc(d[k])
        d["curve"] = [[_enc(a), b, c] for a, b, c in self.curve]
        d["reference_scores"] = [_enc(s) for s in self.reference_scores]
        d["id"] = self.id
        d["schema"] = "riskgate.calibration/v1"
        return d

    def save(self, path: str | Path, key: bytes | str | None = None) -> None:
        """Write the artefact as JSON; with `key`, add an HMAC-SHA256 signature (tamper evidence)."""
        d = self.to_dict()
        if key is not None:
            d["signature"] = hmac.new(_key_bytes(key), canonical(d).encode(), hashlib.sha256).hexdigest()
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def load(path: str | Path, key: bytes | str | None = None, require_signature: bool = False) -> Calibration:
        d = json.loads(Path(path).read_text())
        if not isinstance(d, dict):
            raise ValueError("calibration artefact must be a JSON object")
        sig = d.pop("signature", None)
        if key is not None or require_signature:
            if sig is None:
                raise SignatureError("calibration artefact is not signed")
            if key is None:
                raise SignatureError("a key is required to verify the signature")
            expect = hmac.new(_key_bytes(key), canonical(d).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expect):
                raise SignatureError("calibration artefact signature does not match (tampered or wrong key)")
        d.pop("id", None)
        schema = d.pop("schema", None)
        if schema not in (None, "riskgate.calibration/v1"):
            raise ValueError(f"unsupported calibration schema {schema!r}")
        for k in ("tau", "tau_ood", "lam", "lam_probe1", "lam_probe2"):
            d[k] = _dec(d.get(k))
        d["curve"] = [[_dec(a), b, c] for a, b, c in d.get("curve", [])]
        d["reference_scores"] = [_dec(s) for s in d.get("reference_scores", [])]
        cal = Calibration(**d)
        if not (0.0 < cal.alpha < 1.0 and 0.0 < cal.delta < 1.0):
            raise ValueError(f"calibration alpha ({cal.alpha}) and delta ({cal.delta}) must be in (0, 1)")
        return cal


# --------------------------------------------------------------------------------------------
# Loss helpers for value-valued (not only binary) losses
# --------------------------------------------------------------------------------------------
def _agnostic_values(scores: Sequence[np.ndarray], values: Sequence[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """L_i(lambda) = max over candidates with score >= lambda of value (0 if none)."""
    out = np.zeros((len(scores), grid.shape[0]))
    for i, (s, v) in enumerate(zip(scores, values)):
        s, v = np.asarray(s, float), np.asarray(v, float)
        if s.size == 0:
            continue
        order = np.argsort(-s, kind="stable")
        s_desc, cmax = s[order], np.maximum.accumulate(v[order])
        cnt = np.searchsorted(-s_desc, -grid, side="right")      # number of candidates with s >= lambda
        out[i] = np.where(cnt > 0, cmax[np.maximum(cnt - 1, 0)], 0.0)
    return out


def _pack(losses: np.ndarray) -> str:
    return base64.b64encode(np.packbits(losses > 0.5).tobytes()).decode()


def _unpack(blob: str, shape: Sequence[int]) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(blob), dtype=np.uint8))
    return bits[: int(np.prod(shape))].reshape(shape).astype(float)


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------
class Gate:
    """Certify, reject, probe or ask for proposed agent actions.

    Parameters beyond the core components:
      doer_id       identifier of the proposing agent (model, prompt version); part of the fingerprint in
                    policy mode, because the policy-mode guarantee is specific to the calibrated doer
      timeout_s     wall-clock budget per check; exceeding it returns ASK (fail closed)
      cache_size    LRU cache of margin matrices, keyed by context_key(context), action keys, hypothesis names
      parallel      worker threads for models without predict_batch
      weight_fn     context -> likelihood ratio w(x) for weighted CRC under a known covariate shift
      score_fn      custom aggregation of a candidate's margins across plausible hypotheses
                    (default: minimum, i.e. the worst plausible case)
    """

    def __init__(
        self,
        consequence: ConsequenceModel,
        spec: Spec,
        hypotheses: HypothesisProvider | None = None,
        *,
        alpha: float = 0.05,
        mode: str = "policy",
        delta: float = 0.05,
        hard_rules: Sequence[HardRule] = (),
        probes: Sequence[Action] = (),
        noop: Action | None = None,
        noop_tol: float = 0.0,
        min_separation: float = 1.0,
        min_resolved: float = 0.3,
        ood_alpha: float = 0.01,
        audit: Any = None,
        monitor: Any = None,
        require_calibration: bool = True,
        doer_id: str | None = None,
        timeout_s: float | None = None,
        cache_size: int = 0,
        context_key: Callable[[Any], Hashable] | None = None,
        parallel: int = 0,
        weight_fn: Callable[[Any], float] | None = None,
        score_fn: Callable[[np.ndarray, Sequence[Hypothesis]], float] | None = None,
    ):
        if mode not in ("policy", "agnostic"):
            raise ValueError("mode must be 'policy' or 'agnostic'")
        alpha, delta, ood_alpha = float(alpha), float(delta), float(ood_alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if not 0.0 < ood_alpha < 1.0:
            raise ValueError(f"ood_alpha must be in (0, 1), got {ood_alpha}")
        if not 0.0 <= float(min_resolved) <= 1.0:
            raise ValueError(f"min_resolved must be in [0, 1], got {min_resolved}")
        if not (math.isfinite(min_separation) and float(min_separation) >= 0.0):
            raise ValueError(f"min_separation must be finite and >= 0, got {min_separation}")
        if not (math.isfinite(noop_tol) and float(noop_tol) >= 0.0):
            raise ValueError(f"noop_tol must be finite and >= 0, got {noop_tol}")
        if timeout_s is not None and not (math.isfinite(timeout_s) and float(timeout_s) > 0.0):
            raise ValueError(f"timeout_s must be positive and finite, got {timeout_s}")
        if int(cache_size) < 0 or int(parallel) < 0:
            raise ValueError("cache_size and parallel must be >= 0")
        self.consequence, self.spec = consequence, spec
        self.hypotheses = hypotheses or SingleHypothesis()
        self.alpha, self.mode, self.delta = alpha, mode, delta
        self.hard_rules, self.probes = list(hard_rules), list(probes)
        self.noop, self.noop_tol, self.min_separation, self.ood_alpha = noop, float(noop_tol), float(min_separation), ood_alpha
        self.min_resolved = float(min_resolved)
        self.audit, self.monitor, self.require_calibration = audit, monitor, require_calibration
        self.doer_id, self.timeout_s, self.parallel = doer_id, timeout_s, int(parallel)
        self.weight_fn, self.score_fn = weight_fn, score_fn
        if cache_size and context_key is None:
            raise ValueError("cache_size > 0 needs context_key: a function returning a hashable key derived from the "
                             "context's CONTENT (e.g. an incident id + revision). id() is unsafe: mutated or recycled "
                             "contexts would be served stale margins.")
        self.cache_size = int(cache_size)
        self._context_key: Callable[[Any], Hashable] = context_key or id
        self._cache: OrderedDict = OrderedDict()
        self.calibration: Calibration | None = None
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._weighted_cache: tuple | None = None
        self._cal_failures = 0

    # ------------------------------------------------------------------ identity / status
    def fingerprint(self) -> str:
        return digest({
            "spec": self.spec.fingerprint(),
            "model": getattr(self.consequence, "version", type(self.consequence).__name__),
            "hypotheses": getattr(self.hypotheses, "version", type(self.hypotheses).__name__),
            "probes": [p.key for p in self.probes],
            "rules": [r.describe() if hasattr(r, "describe") else r.name for r in self.hard_rules],
            "mode": self.mode,
            "doer": self.doer_id if self.mode == "policy" else None,
            "score_fn": callable_digest(self.score_fn) if self.score_fn else None,
            "weight_fn": callable_digest(self.weight_fn) if self.weight_fn else None,
        })

    def status(self) -> GuaranteeStatus:
        cal = self.calibration
        if cal is None or cal.fingerprint != self.fingerprint():
            return GuaranteeStatus.VOID
        if self.monitor is not None and getattr(self.monitor, "alarm", False):
            return GuaranteeStatus.DEGRADED
        return GuaranteeStatus.VALID

    def load_calibration(self, cal: Calibration | str | Path, key: bytes | str | None = None,
                         require_signature: bool = False) -> Calibration:
        self.calibration = cal if isinstance(cal, Calibration) else Calibration.load(cal, key, require_signature)
        if self.calibration.mode in ("policy", "agnostic"):
            self.mode = self.calibration.mode   # the fingerprint (doer binding) follows the calibration's mode
        self._weighted_cache = None
        if self.monitor is not None:
            self.monitor.bind(self.calibration)
        return self.calibration

    # ------------------------------------------------------------------ scoring
    def _evaluate(self, actions: Sequence[Action], hyps: Sequence[Hypothesis], context: Any) -> list[list[Any]]:
        if hasattr(self.consequence, "predict_batch"):
            return self.consequence.predict_batch(list(actions), list(hyps), context)
        if self.parallel and len(actions) * len(hyps) > 1:
            with self._lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=self.parallel, thread_name_prefix="riskgate")
            futs = [[self._pool.submit(self.consequence.predict, a, h, context) for h in hyps] for a in actions]
            return [[f.result() for f in row] for row in futs]
        return [[self.consequence.predict(a, h, context) for h in hyps] for a in actions]

    def margin_matrix(self, actions: Sequence[Action], hyps: Sequence[Hypothesis], context: Any):
        """(n_actions, n_hypotheses) margins and binding-constraint names (LRU-cached if cache_size > 0)."""
        key = None
        if self.cache_size:
            key = (self._context_key(context), tuple(a.key for a in actions), tuple(h.name for h in hyps))
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    return hit[0].copy(), [list(r) for r in hit[1]]
        outs = self._evaluate(actions, hyps, context)
        m = np.empty((len(actions), len(hyps)))
        names: list[list[str]] = []
        for i, row in enumerate(outs):
            r = []
            for j, o in enumerate(row):
                m[i, j], b = self.spec.evaluate(o)
                r.append(b)
            names.append(r)
        if key is not None:
            with self._lock:
                self._cache[key] = (m.copy(), names)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return m, names

    def _robust(self, m: np.ndarray, plaus: Sequence[Hypothesis]) -> np.ndarray:
        if m.shape[1] == 0:
            return np.full(m.shape[0], -INF)
        if self.score_fn is None:
            r = m.min(axis=1)
        else:
            r = np.array([float(self.score_fn(m[i], plaus)) for i in range(m.shape[0])])
        r = np.where(np.isnan(r), -INF, r)
        return np.minimum(r, MAX_SCORE)

    def _plausible(self, hyps: list[Hypothesis], cal: Calibration | None) -> tuple[list[Hypothesis], bool]:
        tau = cal.tau if cal is not None else INF
        tau_ood = cal.tau_ood if cal is not None else INF
        ood = bool(hyps) and min(h.score for h in hyps) > tau_ood
        return [h for h in hyps if h.score <= tau], ood

    def _weighted_lambda(self, context: Any) -> float:
        from .weighted import WeightedCRC  # lazy: optional module
        cal = self.calibration
        assert cal is not None and cal.weighted is not None
        if self._weighted_cache is None or self._weighted_cache[0] != cal.id:
            w = cal.weighted
            grid = np.asarray(w["grid"], float)
            grid = np.where(grid >= 1e300, INF, grid)          # +inf is stored as 1e300 in JSON
            wcrc = WeightedCRC(_unpack(w["losses"], w["shape"]), grid, cal.alpha, np.asarray(w["weights"], float))
            self._weighted_cache = (cal.id, wcrc)
        _, wcrc = self._weighted_cache
        assert self.weight_fn is not None
        w_test = float(self.weight_fn(context))
        if math.isnan(w_test) or w_test <= 0.0:
            raise ValueError(f"weight_fn(context) must return a positive weight (> 0), got {w_test}")
        return float(wcrc.threshold(w_test))

    def _threshold(self, stage: int, context: Any = None) -> float:
        cal = self.calibration
        if cal is None:
            return INF
        lam: float
        if self.probes and cal.probing:
            lam = float(cal.lam_probe1 if stage == 1 else cal.lam_probe2)  # type: ignore[arg-type]
        elif self.weight_fn is not None and cal.weighted is not None:
            lam = self._weighted_lambda(context)
        else:
            lam = cal.lam
        online = getattr(self.monitor, "online_lambda", None) if self.monitor is not None else None
        if online is not None and self.status() is GuaranteeStatus.DEGRADED:
            lam = max(lam, online)          # online adaptation may only tighten (rule R1/R4)
        return float(lam)

    # ------------------------------------------------------------------ probing
    def select_probe(self, actions: Sequence[Action], plaus: Sequence[Hypothesis], m: np.ndarray, context: Any,
                     lam: float) -> tuple[Action | None, float]:
        """Choose a read-only probe that resolves the largest share of DISAGREEMENTS between plausible
        hypotheses (pairs for which some candidate is certifiable under one and not the other). A pair
        counts as resolved if the predicted observations differ by >= min_separation (noise units).
        Admissible if, under every plausible hypothesis, the probe is certifiable or no worse than
        `noop`. Requires resolved share >= min_resolved. Returns (probe, resolved share).
        When lambda is not finite (nothing can be certified before a probe, e.g. a refund that is wrong in some
        world until the ledger is read), disagreement is judged by the spec itself (margin >= 0): the same rule runs
        in calibration and deployment, so the two-stage guarantee is unchanged."""
        if len(plaus) < 2 or not self.probes:
            return None, 0.0
        m = np.minimum(np.where(np.isnan(m), -INF, m), MAX_SCORE)
        ok = m >= (lam if math.isfinite(lam) else 0.0)
        pairs = [(j, k) for j in range(len(plaus)) for k in range(j + 1, len(plaus)) if np.any(ok[:, j] != ok[:, k])]
        if not pairs:
            return None, 0.0
        extra = [self.noop] if self.noop is not None else []
        outs = self._evaluate(list(self.probes) + extra, list(plaus), context)
        noop_m = [self.spec.margin(o) for o in outs[-1]] if extra else None
        best, best_key = None, (self.min_resolved, 0.0)
        for pi, p in enumerate(self.probes):
            pm = [min(self.spec.margin(o), MAX_SCORE) for o in outs[pi]]
            if noop_m is not None:
                admissible = all(a >= lam or a >= b - self.noop_tol for a, b in zip(pm, noop_m))
            else:
                admissible = min(pm) >= lam
            if not admissible:
                continue
            obs = [np.atleast_1d(np.asarray(_get(o, "observation"), float)) for o in outs[pi]]
            seps = np.array([float(np.max(np.abs(obs[j] - obs[k]))) for j, k in pairs])
            resolved = float(np.mean(seps >= self.min_separation))        # share of disagreements it settles
            key = (resolved, float(np.mean(np.minimum(seps, 10 * self.min_separation))))
            if key > best_key or (best is None and resolved >= self.min_resolved and key >= best_key):
                best, best_key = p, key
        return best, best_key[0] if best is not None else 0.0

    # ------------------------------------------------------------------ deciding
    def check(self, action: Action, context: Any, *, stage: int = 1) -> Decision:
        return self.check_ranked([action], context, stage=stage)

    def check_ranked(self, candidates: Sequence[Action], context: Any, *, stage: int = 1) -> Decision:
        t0 = time.perf_counter()
        status = self.status()
        first = candidates[0] if candidates else None
        try:
            if self.timeout_s is None:
                d = self._decide(list(candidates), context, stage, status)
            else:
                with self._lock:
                    if self._pool is None:
                        self._pool = ThreadPoolExecutor(max_workers=max(2, self.parallel), thread_name_prefix="riskgate")
                fut = self._pool.submit(self._decide, list(candidates), context, stage, status)
                try:
                    d = fut.result(timeout=self.timeout_s)
                except FutureTimeout:
                    d = Decision(Verdict.ASK, first, reason=f"timeout: no decision within {self.timeout_s:g}s")
        except Exception as exc:  # noqa: BLE001 - rule R3: fail closed
            d = Decision(Verdict.ASK, first, reason=f"fail-closed: {type(exc).__name__}: {exc}")
        return self._finish(d, t0, status, stage)

    def _screen(self, candidates: Sequence[Action], context: Any) -> tuple[list[Action], list[dict], Decision | None]:
        """Apply hard rules in rank order. Returns (viable candidates, considered log, early decision or None).
        Calibration applies exactly the same screening, so the calibrated risk is the risk of the deployed policy."""
        if not candidates:
            return [], [], Decision(Verdict.ASK, None, reason="no candidate actions proposed", considered=[])
        first = candidates[0]
        considered: list[dict] = []
        viable: list[Action] = []
        for a in candidates:
            fired = [r for r in self.hard_rules if r.fires(a, context)]
            rej = [r for r in fired if r.verdict is Verdict.REJECT]
            ask = [r for r in fired if r.verdict is Verdict.ASK]
            if rej:
                considered.append({"action": a.key, "verdict": "reject", "rule": rej[0].name})
                continue
            if ask:
                considered.append({"action": a.key, "verdict": "ask", "rule": ask[0].name})
                if not viable:       # the preferred remaining action needs approval: escalate it
                    return viable, considered, Decision(Verdict.ASK, a, considered=considered,
                                                        reason=f"hard rule '{ask[0].name}': {ask[0].message}")
                continue             # a lower-ranked action that needs approval is never certified automatically
            viable.append(a)
        if not viable:
            msgs = "; ".join(f"{c['action']}: {c.get('rule')}" for c in considered)
            return viable, considered, Decision(Verdict.REJECT, first, reason=f"blocked by hard rules ({msgs})",
                                                considered=considered)
        return viable, considered, None

    def _decide(self, candidates: list[Action], context: Any, stage: int, status: GuaranteeStatus) -> Decision:
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        cal = self.calibration
        first = candidates[0] if candidates else None
        if status is GuaranteeStatus.VOID and self.require_calibration:
            return Decision(Verdict.ASK, first, reason="guarantee void: gate not calibrated or a component changed "
                                                       "since calibration")
        viable, considered, early = self._screen(candidates, context)
        if early is not None:
            early.top_score = -INF               # same statistic as the calibration reference scores
            return early

        hyps = self.hypotheses.hypotheses(context)
        plaus, ood = self._plausible(hyps, cal)
        names = [h.name for h in plaus]
        if ood:
            return Decision(Verdict.ASK, viable[0], reason="evidence unlike anything seen in calibration (out of "
                                                           "distribution)", considered=considered, top_score=-INF)
        if not plaus:
            return Decision(Verdict.ASK, viable[0], reason="no hypothesis explains the evidence", considered=considered,
                            top_score=-INF)
        m, bind = self.margin_matrix(viable, plaus, context)
        robust = self._robust(m, plaus)
        top_score = float(np.max(robust))
        lam = self._threshold(stage, context)
        for i, a in enumerate(viable):
            considered.append({"action": a.key, "score": float(robust[i]), "certified": bool(robust[i] >= lam)})
            if robust[i] >= lam:
                return Decision(Verdict.CERTIFY, a, float(robust[i]), lam, dict(zip(names, map(float, m[i]))),
                                dict(zip(names, bind[i])), names, reason="robust margin above calibrated threshold",
                                considered=considered, top_score=top_score)
        top = 0
        base = Decision(Verdict.ASK, viable[top], float(robust[top]), lam, dict(zip(names, map(float, m[top]))),
                        dict(zip(names, bind[top])), names, considered=considered, top_score=top_score)
        if stage == 1 and self.probes and cal is not None and cal.probing:
            probe, share = self.select_probe(viable, plaus, m, context, lam)
            if probe is not None:
                base.verdict, base.probe = Verdict.PROBE, probe
                base.reason = (f"plausible hypotheses disagree about safety; probe '{probe.name}' is expected to "
                               f"resolve {100 * share:.0f}% of the disagreements")
                return base
        has_ask_rule = any(c.get("verdict") == "ask" for c in considered)
        if np.all(m.max(axis=1) < 0) and not has_ask_rule:
            worst = bind[top][int(np.argmin(m[top]))]
            base.verdict = Verdict.REJECT
            base.reason = f"unsafe under every plausible hypothesis (binding constraint: {worst})"
        else:
            worst_h = names[int(np.argmin(m[top]))]
            base.reason = (f"cannot certify: best robust margin {robust.max():.3f} < threshold {lam:.3f}; "
                           f"worst case under hypothesis '{worst_h}'")
        return base

    def _finish(self, d: Decision, t0: float, status: GuaranteeStatus, stage: int) -> Decision:
        d.status, d.stage = status, stage
        d.calibration_id = self.calibration.id if self.calibration is not None else None
        d.latency_ms = (time.perf_counter() - t0) * 1e3
        if self.audit is not None:
            try:
                self.audit.write(d)
            except Exception as exc:  # noqa: BLE001 - rule R3: an unrecorded decision must not run autonomously
                log.exception("riskgate: audit sink failed")
                if d.verdict is Verdict.CERTIFY:
                    d.verdict = Verdict.ASK
                    d.reason = f"fail-closed: audit sink error ({type(exc).__name__}: {exc}); decision not recorded"
        if self.monitor is not None and status is not GuaranteeStatus.VOID:
            try:
                # the drift statistic must match the calibration reference: best robust score at stage 1
                if stage == 1 and not math.isnan(d.top_score):
                    self.monitor.update_score(d.top_score)
                if d.verdict is Verdict.CERTIFY and hasattr(self.monitor, "register"):
                    self.monitor.register(d)
            except Exception:  # noqa: BLE001 - monitoring must never block a decision
                log.exception("riskgate: monitor update failed")
        return d

    # ------------------------------------------------------------------ outcomes and recalibration
    def record_outcome(self, decision: Decision | str, unsafe: bool) -> bool:
        """Report the realised outcome of an executed decision (labels may arrive later). Returns True if the
        decision was known to the monitor. Feeds the tighten-only online threshold."""
        if self.monitor is None:
            return False
        did = decision if isinstance(decision, str) else decision.decision_id
        return bool(self.monitor.report_outcome(did, unsafe))

    def recalibrate(self, episodes: Sequence[Episode], **kw: Any) -> Calibration:
        """Calibrate on fresh labelled episodes (e.g. after a drift alarm) and reset the monitor."""
        cal = self.calibrate(episodes, **kw)
        if self.monitor is not None:
            self.monitor.bind(cal)
        return cal

    # ------------------------------------------------------------------ calibration
    def _episode_scores(self, ep: Episode, tau: float, tau_ood: float) -> np.ndarray:
        """Robust score of every candidate exactly as the deployed gate would compute it: candidates removed or
        escalated by hard rules, out-of-distribution contexts and evaluation errors get -inf (never certified)."""
        out = np.full(len(ep.candidates), -INF)
        try:
            viable, _, early = self._screen(ep.candidates, ep.context)
            if early is not None:
                return out
            hyps = self.hypotheses.hypotheses(ep.context)
            plaus = [h for h in hyps if h.score <= tau]
            if not plaus or (hyps and min(h.score for h in hyps) > tau_ood):
                return out
            m, _ = self.margin_matrix(viable, plaus, ep.context)
            r = self._robust(m, plaus)
        except Exception:  # noqa: BLE001 - deployment fails closed (ASK) on the same error, so loss 0 is exact
            self._cal_failures += 1
            return out
        pos = {a.key: i for i, a in enumerate(ep.candidates)}
        for a, v in zip(viable, r):
            out[pos[a.key]] = v
        return out

    @staticmethod
    def _labels(ep: Episode) -> np.ndarray:
        return np.array([bool(ep.unsafe.get(a.key, True)) for a in ep.candidates])

    def score_episode(self, episode: Episode, tau: float | None = None, tau_ood: float | None = None) -> np.ndarray:
        """Robust score of every candidate of `episode`, exactly as the deployed gate computes it (hard rules,
        out-of-distribution check and evaluation errors give -inf). Uses the calibrated tau / tau_ood unless given;
        +inf (every hypothesis plausible, no OOD check) when the gate is not calibrated. For offline evaluation and
        benchmarking: the scores do not depend on lambda-hat."""
        cal = self.calibration
        t = (cal.tau if cal is not None else INF) if tau is None else float(tau)
        t_ood = (cal.tau_ood if cal is not None else INF) if tau_ood is None else float(tau_ood)
        return self._episode_scores(episode, t, t_ood)

    def _losses(self, scores, values, grid, mode):
        if mode == "agnostic":
            return _agnostic_values(scores, values, grid)
        return policy_losses(scores, values, grid)

    def calibrate(self, episodes: Sequence[Episode], *, alpha: float | None = None, mode: str | None = None,
                  delta: float | None = None, probe_share: float = 0.6, probe_split: float = 0.5,
                  tau_fraction: float = 0.3, seed: int = 0, guarantee: str = "expectation",
                  delta_ltt: float = 0.1,
                  extra_losses: Mapping[str, tuple[Callable[[Episode, Action], float], float]] | None = None,
                  weights: Sequence[float] | None = None,
                  groups: Sequence[Hashable] | None = None) -> Calibration:
        """Calibrate tau (plausible set), tau_ood, lambda-hat and, if probes are configured, lambda_1/lambda_2.

        guarantee     "expectation": E[risk] <= alpha (conformal risk control);
                      "high_probability": P(risk <= alpha) >= 1 - delta_ltt (Learn-then-Test)
        extra_losses  {name: (fn(episode, executed_action) -> [0, 1], alpha_j)} further risks controlled
                      jointly with safety (e.g. cost overruns); each holds at its own level
        weights       per-episode likelihood ratios w(x) for weighted CRC under covariate shift
                      (use together with Gate(weight_fn=...))
        groups        one label per episode (e.g. a conversation, task or incident id). Decision points that share
                      a label are dependent (several tool calls of one agent run), so exchangeability holds between
                      GROUPS, not between decision points: the tau split is drawn by group and risk control runs on
                      each group's mean loss, which is still non-increasing in lambda, bounded by 1 and 0 at +inf.
                      The guarantee then reads E[share of a new group's decisions that are certified and unsafe]
                      <= alpha (or holds with probability >= 1 - delta_ltt). Not combinable with `weights`;
                      two-stage probing is disabled when groups are given.

        Sample splitting keeps the guarantee exact: a `tau_fraction` share of the episodes fixes the
        score function (tau, tau_ood); risk control then runs on the remaining, disjoint episodes. Hard rules,
        out-of-distribution contexts and evaluation errors are applied exactly as at deployment.

        Probing (two-stage): the risk-control episodes are split (`probe_split`). lambda_1 is calibrated on split
        A at probe_share * alpha (risk of certifying before any probe); lambda_2 on ALL of split B at
        (1 - probe_share) * alpha, where an episode's loss is 0 unless the gate would probe it, and otherwise the
        harm caused by the probe plus the loss of the post-probe decision. Total risk <= alpha.
        """
        alpha = float(self.alpha if alpha is None else alpha)
        mode = self.mode if mode is None else mode
        delta = float(self.delta if delta is None else delta)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if not 0.0 < float(probe_share) < 1.0:
            raise ValueError(f"probe_share must be in (0, 1), got {probe_share}")
        if not 0.0 < float(probe_split) < 1.0:
            raise ValueError(f"probe_split must be in (0, 1), got {probe_split}")
        if not 0.0 <= float(tau_fraction) < 1.0:
            raise ValueError(f"tau_fraction must be in [0, 1), got {tau_fraction}")
        if not 0.0 < float(delta_ltt) < 1.0:
            raise ValueError(f"delta_ltt must be in (0, 1), got {delta_ltt}")
        if guarantee not in ("expectation", "high_probability"):
            raise ValueError("guarantee must be 'expectation' or 'high_probability'")
        if mode not in ("policy", "agnostic"):
            raise ValueError("mode must be 'policy' or 'agnostic'")
        if mode == "policy" and self.doer_id is None:
            # Offline replays of real agents showed this failure: a policy-mode threshold calibrated on one agent
            # exceeded alpha after an agent upgrade (tau2-bench airline) or swap (AgentDojo). Without a doer_id the
            # fingerprint cannot notice the swap, so the calibration would stay VALID.
            warnings.warn("policy-mode calibration without doer_id: the guarantee covers only the agent that "
                          "produced these episodes, but swapping that agent will not void the calibration. Set "
                          "Gate(doer_id=...) (model, prompt version, temperature), or calibrate in agnostic mode.",
                          RiskGateWarning, stacklevel=2)
        rng = np.random.default_rng(seed)
        self._cal_failures = 0
        episodes = list(episodes)
        if weights is not None and len(weights) != len(episodes):
            raise ValueError("weights must have one entry per episode")
        if groups is not None:
            if len(groups) != len(episodes):
                raise ValueError("groups must have one entry per episode")
            if weights is not None:
                raise ValueError("weights and groups cannot be combined")
            gkeys = list(dict.fromkeys(groups))                      # first-appearance order (singletons = episodes)
            gpos = {g: k for k, g in enumerate(gkeys)}
            ep_group = np.array([gpos[g] for g in groups], dtype=int)
        else:
            ep_group = np.arange(len(episodes))
        n_units = int(ep_group.max()) + 1 if len(episodes) else 0
        labelled_idx = [i for i, ep in enumerate(episodes) if ep.true_family is not None]
        k_tau = max(1, int(round(tau_fraction * n_units))) if tau_fraction > 0 and n_units else 0
        want_tau_split = bool(tau_fraction > 0 and (
            labelled_idx or (not isinstance(self.hypotheses, SingleHypothesis)
                             and n_units - k_tau >= math.ceil(1 / alpha))
        ))
        tau_idx: list[int] = []
        if want_tau_split:
            perm = rng.permutation(n_units)
            tau_units = {int(u) for u in perm[:k_tau]}
            tau_idx = [i for i in range(len(episodes)) if int(ep_group[i]) in tau_units]
        tau_set = set(tau_idx)
        crc_idx = [i for i in range(len(episodes)) if i not in tau_set]
        crc_eps = [episodes[i] for i in crc_idx]
        crc_units = list(dict.fromkeys(int(ep_group[i]) for i in crc_idx))
        upos = {u: k for k, u in enumerate(crc_units)}
        crc_gidx = np.array([upos[int(ep_group[i])] for i in crc_idx], dtype=int)
        n = len(crc_units)                                          # risk-control units (episodes or groups)
        if n < math.ceil(1 / alpha):
            unit = "groups" if groups is not None else "episodes"
            raise ValueError(f"need at least {math.ceil(1 / alpha)} risk-control {unit} for alpha={alpha} (got {n})")

        def by_unit(mat: np.ndarray) -> np.ndarray:
            """Per-unit mean loss (identity without groups): exchangeable units, non-increasing rows in [0, 1]."""
            if groups is None:
                return mat
            out = np.zeros((n, mat.shape[1]))
            np.add.at(out, crc_gidx, mat)
            return out / np.bincount(crc_gidx, minlength=n).astype(float)[:, None]

        # 1) plausible-set threshold tau and out-of-distribution threshold (on the tau split)
        if tau_idx:
            tau_eps = [episodes[i] for i in tau_idx]
            tau_hyps = [self.hypotheses.hypotheses(ep.context) for ep in tau_eps]
            r_true = [min([h.score for h in hs if h.fam == ep.true_family] or [INF])
                      for ep, hs in zip(tau_eps, tau_hyps) if ep.true_family is not None]
            tau = conformal_quantile(r_true, delta) if r_true else INF
            best_scores = [min(h.score for h in hs) for hs in tau_hyps if hs]
            eff_ood_alpha = max(self.ood_alpha, 1.0 / (len(best_scores) + 1)) if best_scores else self.ood_alpha
            tau_ood = conformal_quantile(best_scores, eff_ood_alpha) if best_scores else INF
        else:
            tau, tau_ood = INF, INF
        hyp_lists = [self.hypotheses.hypotheses(ep.context) for ep in crc_eps]

        # 2) scores, losses and the risk-controlling threshold
        scores = [self._episode_scores(ep, tau, tau_ood) for ep in crc_eps]
        labels = [self._labels(ep).astype(float) for ep in crc_eps]
        grid = lambda_grid(scores)
        losses = by_unit(self._losses(scores, labels, grid, mode))
        mats, alphas = [losses], [alpha]
        for fn, a_j in (extra_losses or {}).values():
            vals = [np.array([float(fn(ep, a)) for a in ep.candidates]) for ep in crc_eps]
            mats.append(by_unit(self._losses(scores, vals, grid, mode)))
            alphas.append(a_j)
        if guarantee == "high_probability":
            from .multirisk import ltt_threshold  # lazy: optional module
            lam = float(ltt_threshold(mats, grid, alphas, delta_ltt))
        else:
            lam = max(crc_threshold(mt, grid, a_j) for mt, a_j in zip(mats, alphas))
        asks = ask_rate(scores, grid)
        risk = losses.mean(axis=0)
        step = max(1, len(grid) // 200)
        curve = [[float(g), float(r), float(a)] for g, r, a in zip(grid[::step], risk[::step], asks[::step])]
        self.mode = mode                         # fingerprint below must reflect the mode actually calibrated
        cal = Calibration(alpha=alpha, mode=mode, delta=delta, tau=tau, tau_ood=tau_ood, lam=lam,
                          fingerprint=self.fingerprint(), n=n, created=datetime.now(timezone.utc).isoformat(),
                          guarantee=guarantee, delta_ltt=delta_ltt if guarantee == "high_probability" else None,
                          extra_alphas={k: v[1] for k, v in (extra_losses or {}).items()},
                          curve=curve, reference_scores=[float(np.max(s)) for s in scores][:5000])
        j = int(np.searchsorted(grid, lam)) if math.isfinite(lam) else None
        cal.diagnostics = {
            "tau_coverage_target": 1 - delta,
            "calibration_ask_rate_at_lambda": float(np.mean([np.max(s) < lam for s in scores])),
            "calibration_risk_at_lambda": float(losses[:, j].mean()) if j is not None else 0.0,
            "mean_plausible_hypotheses": float(np.mean([sum(h.score <= tau for h in hs) for hs in hyp_lists])),
            "n_tau": len(tau_idx),
            "evaluation_failures": self._cal_failures,
        }
        if groups is not None:
            cal.diagnostics.update({"grouped": True, "n_groups": n, "n_episodes": len(crc_eps),
                                    "n_tau_groups": len({int(ep_group[i]) for i in tau_idx})})
        if self._cal_failures:
            log.warning("riskgate: %d calibration episodes failed to evaluate (counted as never certified, "
                        "exactly as the gate behaves in deployment)", self._cal_failures)
        for name, mt in zip((extra_losses or {}).keys(), mats[1:]):
            cal.diagnostics[f"calibration_risk_{name}"] = float(mt[:, j].mean()) if j is not None else 0.0
        if weights is not None:
            w = np.asarray([weights[i] for i in crc_idx], float)
            cal.weighted = {"grid": [float(g) if math.isfinite(g) else 1e300 for g in grid], "weights": w.tolist(),
                            "losses": _pack(losses), "shape": list(losses.shape)}

        # 3) two-stage calibration for probing (sample splitting keeps the stages independent). Supported for the
        #    plain expectation guarantee; with high-probability, extra risks, weights or groups the gate does not probe.
        simple = guarantee == "expectation" and not extra_losses and weights is None and groups is None
        if self.probes and not simple:
            cal.diagnostics["probing_disabled"] = "two-stage probing supports guarantee='expectation' without " \
                                                  "extra_losses, weights or groups"
        if self.probes and simple and any(ep.after_probe for ep in crc_eps):
            idx = rng.permutation(n)
            n_a = int(round(n * probe_split))
            a_idx, b_idx = idx[:n_a], idx[n_a:]
            grid_a = lambda_grid([scores[i] for i in a_idx])
            lam1 = crc_threshold(self._losses([scores[i] for i in a_idx], [labels[i] for i in a_idx], grid_a, mode),
                                 grid_a, probe_share * alpha)
            # Stage 2 is calibrated UNCONDITIONALLY on every split-B episode: the loss is 0 when stage 1 certifies
            # or no probe is chosen, and otherwise the loss of the probe path (harm caused by the probe itself, a
            # constant in lambda_2, plus the post-probe certification loss). CRC at (1 - probe_share) * alpha then
            # bounds E[stage-2 loss] over all episodes, so stage 1 + stage 2 <= alpha in total.
            post_s, post_u, post_harm, n_b, n_missing = [], [], [], len(b_idx), 0
            for i in b_idx:
                ep, hs = crc_eps[i], hyp_lists[i]
                if np.any(scores[i] >= lam1):
                    continue                                   # stage 1 certifies: no probe
                try:
                    viable, _, early = self._screen(ep.candidates, ep.context)
                    if early is not None:
                        continue
                    plaus = [h for h in hs if h.score <= tau]
                    if not plaus or min(h.score for h in hs) > tau_ood:
                        continue
                    m, _ = self.margin_matrix(viable, plaus, ep.context)
                    probe, _ = self.select_probe(viable, plaus, m, ep.context, lam1)
                except Exception:  # noqa: BLE001 - deployment would ASK: no probe, loss 0
                    continue
                if probe is None:
                    continue
                ep2 = ep.after_probe.get(probe.key)
                if ep2 is None:                                # no post-probe data: count as a loss (conservative)
                    n_missing += 1
                    post_s.append(np.full(1, -INF))
                    post_u.append(np.zeros(1))
                    post_harm.append(True)
                    continue
                post_s.append(self._episode_scores(ep2, tau, tau_ood))
                post_u.append(self._labels(ep2).astype(float))
                post_harm.append(bool(ep2.meta.get("probe_unsafe", False)))
            cal.n_probe = len(post_s)
            cal.probe_share = probe_share
            alpha2 = (1.0 - probe_share) * alpha
            if post_s and n_b >= math.ceil(1 / alpha2):
                grid_b = lambda_grid(post_s)
                rows = self._losses(post_s, post_u, grid_b, mode)
                for k, harmed in enumerate(post_harm):
                    if harmed:
                        rows[k] = 1.0                          # harm already done, whatever lambda_2 is
                full = np.zeros((n_b, grid_b.shape[0]))
                full[: rows.shape[0]] = rows                   # non-probed split-B episodes contribute zero loss
                lam2 = crc_threshold(full, grid_b, alpha2)
                if math.isfinite(lam2):
                    cal.lam_probe1, cal.lam_probe2 = lam1, lam2
            cal.diagnostics["probe_episodes"] = len(post_s)
            cal.diagnostics["probe_episodes_missing_post_probe_data"] = n_missing
            cal.diagnostics["probe_split_b"] = n_b
        self.load_calibration(cal)
        return cal
