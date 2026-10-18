"""riskgate: certify, reject or ask -- distribution-free risk control for the actions of LLM agents."""
from . import traces
from .adapters.tool import ActionRejected, ApprovalRequired, guard
from .audit import AuditLog, MemoryAudit, MultiSink, summarize_audit
from .conformal import conformal_quantile, crc_threshold
from .gate import (
                   Calibration,
                   CallableModel,
                   ConsequenceModel,
                   Episode,
                   Gate,
                   HypothesisProvider,
                   SignatureError,
                   SingleHypothesis,
)
from .monitor import Monitor
from .replay import ReplayReport, replay, resplit, sweep
from .spec import Custom, Envelope, HardRule, Lower, Predicate, Spec, Upper
from .types import Action, Decision, GuaranteeStatus, Hypothesis, RiskGateWarning, Verdict
from .worlds import DiscreteWorlds, OutcomeTable, nearest_violation, unsafe_mass

__version__ = "1.0.0rc6"

__all__ = [
    "Action", "ActionRejected", "ApprovalRequired", "AuditLog", "Calibration", "CallableModel", "ConsequenceModel",
    "Custom", "Decision", "DiscreteWorlds", "Envelope", "Episode", "Gate", "GuaranteeStatus", "HardRule", "Hypothesis",
    "HypothesisProvider", "Lower", "MemoryAudit", "Monitor", "MultiSink", "OutcomeTable", "Predicate", "ReplayReport",
    "RiskGateWarning", "SignatureError", "SingleHypothesis", "Spec", "Upper", "Verdict", "conformal_quantile",
    "crc_threshold", "guard", "nearest_violation", "replay", "resplit", "summarize_audit", "sweep", "traces", "unsafe_mass",
]
