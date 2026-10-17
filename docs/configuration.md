# Configuration

`riskgate.config` builds a [`Gate`](api.md#gate) from a declarative file in JSON, TOML or YAML. The file holds what operators tune and review: risk levels, the safety spec, hard rules, probes, the monitor, the audit log and the calibration artefact. Your code supplies what only code can: the consequence model and the hypothesis provider.

```py
from riskgate.config import load_config, gate_from_config
gate = gate_from_config(load_config("gate.toml"), consequence=MyTwin(), hypotheses=MyDiagnoser(), **overrides)
```

> [!NOTE]
> **No code is ever evaluated from a configuration file.** Hard rules are simple declarative conditions. Custom constraints, STL formulas, `context_key`, `score_fn` and `weight_fn` are code: pass them as keyword `overrides` or build them in Python (see [Mixing configuration and code](#mixing-configuration-and-code)).

## Formats

| Extension | Loader | Requirement |
|---|---|---|
| `.toml` | `tomllib` | Python ≥ 3.11 (on Python 3.10 install `tomli`) |
| `.yaml`, `.yml` | `yaml.safe_load` | `pip install "riskgate[yaml]"` (PyYAML) |
| `.json` | `json` | none |

Any other extension raises `ValueError`. All three formats produce the same dictionary, and `gate_from_config` accepts a dictionary from any source.

## Schema

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `gate` | table | `{}` | Gate parameters ([below](#gate)). |
| `spec` | array of tables | **required** | Constraints of the safety spec ([below](#spec)). |
| `spec_floor` | number | none | Lower clip for margins (`Spec(floor=...)`). Keep it negative, for example `-5.0`. |
| `hard_rules` | array of tables | `[]` | Declarative hard rules ([below](#hard_rules)). |
| `monitor` | table | none | Drift monitor ([below](#monitor)). An **empty** table creates a monitor with default settings. |
| `audit` | table | none | JSONL audit log ([below](#audit)). |
| `calibration` | string | none | Path of the calibration artefact. It is loaded if the file exists. If it does not, a warning is logged and the gate stays uncalibrated, so every check returns `ASK`. |
| `calibration_key_env` | string | none | **Name** of the environment variable that holds the HMAC key used to verify the artefact. If the variable is missing, `ValueError`. Never put the key itself in a file. |
| `require_signature` | bool | `false` | Refuse unsigned artefacts. Needs `calibration_key_env`. |

Every key except `gate` may also be nested **under `gate`**, as in the example of [SPEC §5.9](SPEC.md). A key given in both places raises `ValueError`. So does a missing `spec`.

### `gate`

| Key | Default | Notes |
|---|---|---|
| `alpha` | `0.05` | Default risk level for `calibrate`. |
| `mode` | `"policy"` | `"policy"` or `"agnostic"`. |
| `delta` | `0.05` | Plausible-set miscoverage. |
| `probes` | `[]` | Each entry is a tool name (`"redis_info_stats"`) or a table `{name, args, read_only}`. Probes are read-only unless a table says otherwise. |
| `noop` | none | A tool name or a table `{name, args}`: the "do nothing" action for probe admissibility. |
| `noop_tol`, `min_separation`, `min_resolved`, `ood_alpha` | `0.0`, `1.0`, `0.3`, `0.01` | Probe selection and OOD guard. |
| `doer_id` | none | Doer identity, fingerprinted in policy mode. |
| `require_calibration` | `true` | `false` gives shadow mode ([API](api.md#constructor-parameters)). |
| `timeout_s`, `parallel` | none, `0` | Per-check time budget, and worker threads for models without `predict_batch`. |
| `cache_size` | `0` | Needs a `context_key` override, otherwise `ValueError`. |

Any other key is passed to `Gate(...)` as a keyword argument, so a misspelt key fails loudly: `TypeError: Gate.__init__() got an unexpected keyword argument 'alpah'. Did you mean 'alpha'?`. The full meaning of each parameter is in the [API reference](api.md#constructor-parameters).

### `spec`

Each entry is one constraint. The spec is their conjunction: its margin is the minimum over the constraints.

| `type` | Required keys | Optional | Margin |
|---|---|---|---|
| `upper` | `key`, `limit` | `scale` (1.0), `name` | (limit − max x) / scale |
| `lower` | `key`, `limit` | `scale` (1.0), `name` | (min x − limit) / scale |
| `envelope` | `key`, `lo`, `hi` | `scale` (1.0), `name` | min(hi − max x, min x − lo) / scale |

`type` is case-insensitive, and any other type raises `ValueError`. Choose `scale` so that one margin unit means the same thing for every constraint, for example the limit itself, which makes the margins relative ([Bring your own domain](byo_domain.md#3-write-the-spec)).

### `hard_rules`

| Key | Default | Meaning |
|---|---|---|
| `name` | **required** | Shown in `Decision.reason` and in the audit log. |
| `action` | any tool | Only actions with this name are checked. |
| `arg` | none | Argument to test. Without `arg`, the rule fires for **every** call of `action` (or of every tool if `action` is also missing). |
| `op` | `"=="` | One of `>`, `>=`, `<`, `<=`, `==`, `!=`, `in`, `not in`. Anything else raises `KeyError` at load time. |
| `value` | none | Right-hand side. With `in`/`not in`, give a list: `in` is a deny-list, `not in` an allow-list. |
| `verdict` | `"reject"` | `"reject"` (never execute) or `"ask"` (needs a human). Case-insensitive. Only these two have an effect. |
| `message` | `""` | Appended to the reason. |

How a rule is evaluated for an action `a`:

1. If `action` is set and `a.name` differs, the rule does not fire.
2. With no `arg`, the rule fires.
3. If `arg` is not among `a.args`, the rule does **not** fire.
4. Otherwise it fires if `op(a.args[arg], value)` is true. If the comparison raises `TypeError` (incomparable types), the rule fires: fail closed.

Each declarative rule gets `version = "cfg:<digest of its table>"`, so editing a rule in the file voids the artefact. Declarative rules see only the action, never the context. Write a [`HardRule`](api.md#hardrule) in Python for context-dependent rules.

> [!WARNING]
> Step 3 means a rule on an argument does not fire when the agent **omits** that argument. Tools wrapped with [`guard`](api.md#tool-adapter-guard) always pass every argument, defaults included. Framework adapters pass whatever the model sent. To refuse calls that lack a required argument, validate tool schemas in the framework, or add a Python `HardRule`.

### `monitor`

| Key | Default | Meaning |
|---|---|---|
| `alarm_level` | `100` | Martingale alarm threshold. The false-alarm probability is about ≤ 1/`alarm_level`. |
| `eta` | derived | Step size of the tighten-only online threshold (default: 5% of the calibration score spread). |
| `latch` | `true` | Keep the alarm, and `DEGRADED`, on until recalibration or `reset()`. |

Unknown keys raise `ValueError`. See [Operations → Monitoring](operations.md#monitoring-drift-alarm-and-tighten-only-threshold).

### `audit`

| Key | Default | Meaning |
|---|---|---|
| `path` | **required** | JSONL file. Parent directories are created when the gate is built. |
| `redact` | `[]` | Dotted paths into each record, for example `"action.args.customer_email"`, replaced by `"[redacted]"` before writing. |

Unknown keys raise `ValueError`. For anything richer, such as hashing values, adding static `extra` fields, `MultiSink` or `OTelSink`, pass `audit=...` as an override ([Operations → Audit logs](operations.md#audit-logs-and-redaction)).

## Complete example

This is the configuration of an SRE agent that remediates latency incidents on a Kubernetes service. The [k8s example](https://github.com/reallysaurabh/riskgate/blob/main/examples/k8s_incident.py) uses the same spec, probes and noop.

<!-- file: gate.toml -->
```toml
# gate.toml: riskgate configuration for the checkout-api SRE agent
calibration = "artefacts/checkout-api.cal.json"   # loaded if the file exists
calibration_key_env = "RISKGATE_KEY"               # name of the env var holding the HMAC key
require_signature = true                           # refuse unsigned or tampered artefacts
spec_floor = -5.0

[gate]
alpha = 0.05
mode = "policy"
delta = 0.05
doer_id = "sre-assistant:prompt-v7"
noop = "noop"
noop_tol = 0.05
min_separation = 2.0
timeout_s = 2.0
probes = [
  { name = "pg_stat_statements_top", args = { db = "orders-pg" } },
  { name = "redis_info_stats", args = { cache = "checkout-redis" } },
  "canary_version_compare",
]

[[spec]]
type = "upper"
key = "p95_ms"
limit = 800
scale = 800
name = "p95<=800ms"

[[spec]]
type = "upper"
key = "error_rate"
limit = 0.02
scale = 0.02
name = "errors<=2%"

[[hard_rules]]
name = "max scale factor"
action = "kubectl_scale"
arg = "factor"
op = ">"
value = 3
verdict = "reject"
message = "never scale more than 3x in one step"

[[hard_rules]]
name = "protected namespaces"
arg = "namespace"
op = "in"
value = ["kube-system", "payments"]
verdict = "reject"
message = "agents may not touch these namespaces"

[[hard_rules]]
name = "rollbacks need approval"
action = "kubectl_rollout_undo"
verdict = "ask"
message = "a human approves every rollback"

[monitor]
alarm_level = 100

[audit]
path = "logs/decisions.jsonl"
redact = ["action.args.customer_email"]
```

The checks below load the file and build the gate. The consequence model here is a stand-in that returns a fixed outcome. Use your twin or dry-run instead.

```python
import os

import riskgate as rg
from riskgate.config import gate_from_config, load_config

cfg = load_config("gate.toml")
model = rg.CallableModel(lambda a, h, ctx: {"p95_ms": 420.0, "error_rate": 0.004, "observation": [0.0]},
                         version="checkout-twin-stub-v1")      # stand-in for your consequence model
os.environ["RISKGATE_KEY"] = "demo-key"                         # in production: injected from your secret store
gate = gate_from_config(cfg, model, rg.SingleHypothesis())      # logs a warning: the artefact does not exist yet
print(gate.alpha, gate.mode, gate.doer_id, gate.timeout_s, gate.noop.key)
# -> 0.05 policy sre-assistant:prompt-v7 2.0 noop
print([p.key for p in gate.probes])
# -> ['pg_stat_statements_top({"db":"orders-pg"})', 'redis_info_stats({"cache":"checkout-redis"})', 'canary_version_compare']
print([c.name for c in gate.spec.constraints], gate.spec.floor, gate.monitor.alarm_level, gate.audit.path, gate.status())
# -> ['p95<=800ms', 'errors<=2%'] -5.0 100.0 logs/decisions.jsonl GuaranteeStatus.VOID
```

The hard rules, evaluated on their own:

```python
def fired(action):
    return [r.name for r in gate.hard_rules if r.fires(action, None)]

print(fired(rg.Action("kubectl_scale", {"deployment": "checkout-api", "factor": 4})))   # -> ['max scale factor']
print(fired(rg.Action("kubectl_scale", {"namespace": "payments", "factor": 1.5})))     # -> ['protected namespaces']
print(fired(rg.Action("kubectl_rollout_undo", {"deployment": "checkout-api"})))         # -> ['rollbacks need approval']
print(fired(rg.Action("kubectl_scale", {"deployment": "checkout-api"})))                 # -> []  (no `factor` argument)
```

The full loop: calibrate once (here on stand-in episodes), sign the artefact with the key, store it at the configured path, and build the gate from the file again. This time the artefact is verified and loaded:

```python
import numpy as np

rng = np.random.default_rng(0)
acts = [rg.Action("kubectl_scale", {"deployment": "checkout-api", "factor": f}) for f in (1.5, 2.0)] + [rg.Action("noop")]
episodes = [rg.Episode({"incident": i}, acts, {a.key: bool(rng.random() < 0.01) for a in acts}) for i in range(200)]
cal = gate.calibrate(episodes)
os.makedirs("artefacts", exist_ok=True)
cal.save(cfg["calibration"], key=os.environ["RISKGATE_KEY"])

gate = gate_from_config(cfg, model, rg.SingleHypothesis())      # verifies the HMAC signature, then loads
print(gate.status(), gate.calibration.id == cal.id)             # -> GuaranteeStatus.VALID True
d = gate.check(rg.Action("send_status_email", {"customer_email": "jane@example.com", "incident": 42}), {"incident": 1})
import json
print(d.verdict, json.loads(open("logs/decisions.jsonl").read().splitlines()[-1])["action"]["args"])
# -> Verdict.CERTIFY {'customer_email': '[redacted]', 'incident': 42}
```

### The same file in YAML and JSON

<!-- file: gate.yaml -->
```yaml
calibration: artefacts/checkout-api.cal.json
calibration_key_env: RISKGATE_KEY
require_signature: true
spec_floor: -5.0
gate:
  alpha: 0.05
  mode: policy
  delta: 0.05
  doer_id: sre-assistant:prompt-v7
  noop: noop
  noop_tol: 0.05
  min_separation: 2.0
  timeout_s: 2.0
  probes:
    - {name: pg_stat_statements_top, args: {db: orders-pg}}
    - {name: redis_info_stats, args: {cache: checkout-redis}}
    - canary_version_compare
spec:
  - {type: upper, key: p95_ms, limit: 800, scale: 800, name: "p95<=800ms"}
  - {type: upper, key: error_rate, limit: 0.02, scale: 0.02, name: "errors<=2%"}
hard_rules:
  - {name: max scale factor, action: kubectl_scale, arg: factor, op: ">", value: 3, verdict: reject,
     message: never scale more than 3x in one step}
  - {name: protected namespaces, arg: namespace, op: in, value: [kube-system, payments], verdict: reject,
     message: agents may not touch these namespaces}
  - {name: rollbacks need approval, action: kubectl_rollout_undo, verdict: ask,
     message: a human approves every rollback}
monitor: {alarm_level: 100}
audit: {path: logs/decisions.jsonl, redact: [action.args.customer_email]}
```

```python
with open("gate.json", "w") as f:
    json.dump(cfg, f, indent=1)
print(load_config("gate.yaml") == cfg, load_config("gate.json") == cfg)    # -> True True
```

## Overrides

Keyword arguments to `gate_from_config` override the file. They are passed straight to `Gate(...)` and replace anything the file set for the same parameter. Use them for objects and callables:

```python
gate = gate_from_config(
    cfg, model, rg.SingleHypothesis(),
    cache_size=256, context_key=lambda ctx: ctx["incident"],                  # content-derived cache key
    audit=rg.MultiSink(rg.AuditLog("logs/decisions.jsonl", extra={"service": "checkout-api"})),
)
print(gate.cache_size, type(gate.audit).__name__, gate.status())             # -> 256 MultiSink GuaranteeStatus.VALID
```

`hard_rules`, `probes` and `noop` can be overridden the same way. An override replaces the configured value entirely: it does not merge with it.

## Mixing configuration and code

The file can only express `upper`, `lower` and `envelope` constraints. To add [`Custom`](api.md#constraints) or [STL](api.md#riskgatestl) constraints, build the spec in code from the file's items. The constraints become part of the fingerprint, so calibrate (or validate) the gate that has the full spec:

```python
from riskgate.config import spec_from_config
from riskgate.stl import Always, Pred, STLConstraint

recovers = STLConstraint("p95 recovers within 4 steps", Always(Pred("p95_ms", "<=", 800, scale=800), 4))
spec = rg.Spec(*spec_from_config(cfg["spec"]).constraints, recovers, floor=cfg.get("spec_floor"))
code_cfg = {k: v for k, v in cfg.items() if k not in ("calibration", "calibration_key_env", "require_signature")}
gate = gate_from_config(code_cfg, model, rg.SingleHypothesis())
gate.spec = spec                         # before calibrating: the spec is part of the fingerprint
print([c.name for c in gate.spec.constraints], gate.status())               # -> ['p95<=800ms', 'errors<=2%', 'p95 recovers within 4 steps'] GuaranteeStatus.VOID  (calibrate this gate next)
```

`rule_from_config(table)` and `spec_from_config(items, floor=None)` are public too, so you can build any part of a gate from configuration and the rest in code.

## Errors

| Problem | Exception and message |
|---|---|
| No `spec` | `ValueError: config needs a [[spec]] list of constraints` |
| Key both at the top level and under `[gate]` | `ValueError: 'spec' is given both at the top level and under [gate]` |
| Unknown constraint type | `ValueError: unknown constraint type 'stl' (use upper, lower, envelope; Custom/STL in code)` |
| Unknown `[gate]` key | `TypeError: Gate.__init__() got an unexpected keyword argument 'alpah'. Did you mean 'alpha'?` |
| `cache_size` without a `context_key` override | `ValueError: cache_size > 0 needs context_key: ...` |
| Unknown `[monitor]` / `[audit]` key | `ValueError: unknown [monitor] keys: ['alarm']` |
| Unknown rule `op` / `verdict` | `KeyError: '~'` / `ValueError: 'maybe' is not a valid Verdict` |
| `calibration_key_env` names a missing variable | `ValueError: calibration_key_env='NOPE_RG_KEY' is set but the environment variable is missing` |
| Wrong key or tampered artefact | `SignatureError: calibration artefact signature does not match (tampered or wrong key)` |
| Unsupported file extension / YAML without PyYAML | `ValueError: unsupported config format: .ini` / `ImportError: YAML configs need pip install pyyaml ...` |
