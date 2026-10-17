# Command-line interface

Installing riskgate adds the `riskgate` console script. `python -m riskgate.cli` is equivalent. The CLI calibrates, inspects and validates artefacts, summarises audit logs and runs the MCP proxy.

| Command | Purpose |
|---|---|
| [`riskgate version`](#riskgate-version) | Print the installed version. |
| [`riskgate calibrate`](#riskgate-calibrate) | Calibrate a gate on labelled episodes, and save (and sign) the artefact. |
| [`riskgate report`](#riskgate-report) | Summarise a calibration artefact. |
| [`riskgate validate`](#riskgate-validate) | Check that an artefact matches a gate. Exits 3 if the guarantee would be `VOID`. |
| [`riskgate audit-summary`](#riskgate-audit-summary) | Count verdicts and statuses and report latency percentiles of a JSONL audit log. |
| [`riskgate serve-mcp`](#riskgate-serve-mcp) | Run an MCP proxy that gates the tools of an upstream MCP server. |

## Conventions

- **`--gate`, `--episodes`, `--context`** take `module:attribute` references, where `attribute` may be a dotted path. The current directory is put on `sys.path`, so project-local modules work. `--gate` must name a **zero-argument factory** that returns a `riskgate.Gate`. `--episodes` names a function that returns a list of `riskgate.Episode`. It receives `--n` if it takes a parameter.
- **`--key`** is the HMAC key used to sign and verify artefacts. It defaults to the environment variable `RISKGATE_KEY`. With a key, `report`, `validate` and `serve-mcp` **require** a signed artefact whose signature verifies. Without one, signatures are ignored. Prefer the environment variable to a command-line flag, which other users can read from the process list.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success (`validate`: the guarantee is `VALID`). |
| `1` | Runtime error, printed as `riskgate: error: <Exception>: <message>` on stderr. For example: import failure, signature mismatch, missing file, too few episodes. A malformed `module:attribute` reference also exits 1. |
| `2` | Usage error (invalid or missing arguments, from `argparse`). |
| `3` | `validate` only: the artefact does not match the gate, so its guarantee would be `VOID`. |

## Worked example

Every output on this page comes from a real run (riskgate `1.0.0rc1`). Artefact ids and timestamps differ from run to run. The example uses this toy module (the leaky tank of the [API reference](api.md#setup-used-on-this-page)), saved as `toy_gate.py` in the working directory:

<!-- file: toy_gate.py -->
```python
"""Toy gate factory and episode source for the riskgate CLI: the leaky tank."""
import numpy as np

import riskgate as rg


class Tank:
    version = "tank-v1"

    def predict(self, action, hyp, ctx):
        return {"level": ctx["level"] + action.args.get("v", 0.0) - hyp.params["leak"]}


class LeakHyps:
    version = "leak-v1"

    def hypotheses(self, ctx):
        return [rg.Hypothesis(f"leak{l}", family=f"leak{l}", params={"leak": l},
                              score=abs(ctx["evidence"] - l) / 5) for l in (0, 10, 25)]


ACTIONS = [rg.Action("add", {"v": v}) for v in (40, 20, 0, -10)]


def make_gate():
    """--gate: a zero-argument factory returning a riskgate.Gate."""
    return rg.Gate(Tank(), rg.Spec(rg.Envelope("level", lo=10, hi=100, scale=10)), LeakHyps(), alpha=0.1)


def make_gate_v2():
    """The same gate after a model update: the fingerprint changes, so old artefacts are VOID."""
    gate = make_gate()
    gate.consequence.version = "tank-v2"
    return gate


def episodes(n=400):
    """--episodes: labelled calibration episodes (receives --n because it takes an argument)."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        leak, level = int(rng.choice([0, 10, 25])), float(rng.uniform(30, 70))
        ctx = {"level": level, "evidence": leak + rng.normal(0, 6)}
        ranked = list(ACTIONS)
        rng.shuffle(ranked)
        unsafe = {a.key: not 10 <= level + a.args["v"] - leak <= 100 for a in ranked}
        out.append(rg.Episode(ctx, ranked, unsafe, true_family=f"leak{leak}"))
    return out


def snapshot():
    """--context for serve-mcp: the current state of the tank."""
    return {"level": 50.0, "evidence": 1.0}
```

## `riskgate version`

```console
$ riskgate version
1.0.0rc1
```

## `riskgate calibrate`

```text
riskgate calibrate --gate MODULE:FACTORY --episodes MODULE:FUNCTION --out FILE
                   [--n N] [--alpha A] [--mode {policy,agnostic}]
                   [--guarantee {expectation,high_probability}] [--delta-ltt D] [--key KEY]
```

| Flag | Default | Meaning |
|---|---|---|
| `--gate` | required | Factory of the gate to calibrate. |
| `--episodes` | required | Function returning the labelled episodes. |
| `--out` | required | Path of the artefact to write (JSON). |
| `--n` | none | Passed to the episodes function if it accepts a parameter. |
| `--alpha` | the gate's `alpha` | Risk level α. |
| `--mode` | the gate's `mode` | `policy` or `agnostic`. It also sets the gate's mode, so the fingerprint matches the artefact. Prefer setting the mode in the factory. |
| `--guarantee` | `expectation` | `expectation` (conformal risk control) or `high_probability` (Learn-then-Test). |
| `--delta-ltt` | `0.1` | Family-wise level δ for `high_probability`. |
| `--key` | `RISKGATE_KEY` | Sign the artefact with HMAC-SHA256. |

It calls `gate.calibrate(episodes, alpha=..., mode=..., guarantee=..., delta_ltt=...)` and saves the artefact. The reported count is the total number of episodes, including the τ split. `report` shows how many went to risk control. Options that need code, such as `extra_losses`, `weights`, `probe_share` and `tau_fraction`, are only available from Python ([`Gate.calibrate`](api.md#gatecalibrate)).

```console
$ riskgate calibrate --gate toy_gate:make_gate --episodes toy_gate:episodes --n 400 --out tank.cal.json --key demo-key
calibrated on 400 episodes -> tank.cal.json (id 5bb204785b4e, lambda=-1.499)
```

A missing required flag is a usage error:

```console
$ riskgate calibrate --gate toy_gate:make_gate --out x.json
usage: riskgate calibrate [-h] --gate GATE --episodes EPISODES [--n N]
                          [--alpha ALPHA] [--mode {policy,agnostic}]
                          [--guarantee {expectation,high_probability}]
                          [--delta-ltt DELTA_LTT] --out OUT [--key KEY]
riskgate calibrate: error: the following arguments are required: --episodes
$ echo $?
2
```

## `riskgate report`

```text
riskgate report CALIBRATION.json [--json] [--key KEY]
```

| Flag | Meaning |
|---|---|
| `--json` | Print the artefact as JSON, without the bulky `curve`, `reference_scores` and `weighted` fields. |
| `--key` | Verify the signature first (default `RISKGATE_KEY`). Exits 1 if it is missing or wrong. |

```console
$ riskgate report tank.cal.json --key demo-key
riskgate calibration 5bb204785b4e  (schema riskgate.calibration/v1, created 2026-10-11T19:08:00.289577+00:00)
  guarantee   : expectation  alpha=0.1
  mode        : policy    episodes for risk control n=280  (tau split n=120)
  plausible   : tau=2.259 (target coverage 0.95), mean |H|=1.50, tau_ood=2.42
  threshold   : lambda=-1.499  calibration ask rate=0.029  calibration risk=0.054
  probing     : not calibrated (probed episodes=0)
  fingerprint : 476aaff52ab03d1c
```

| Line | What it tells you |
|---|---|
| `guarantee` | Guarantee type and α. For `high_probability` also δ, and any jointly controlled extra risks. |
| `mode` | Loss mode, and how the episodes were split between risk control (`n`) and the plausible-set threshold (`tau split`). |
| `plausible` | τ and its target coverage 1 − δ, the mean size of the plausible set per risk-control episode, and the out-of-distribution threshold τ_ood. |
| `threshold` | λ̂. The *calibration ask rate* is the share of risk-control episodes in which no candidate reaches λ̂. The *calibration risk* is the empirical loss at λ̂. Both are in-sample diagnostics, not guarantees. |
| `probing` | λ₁, λ₂, the stage-1 share and the number of probed episodes, or "not calibrated". |
| `fingerprint` | The gate fingerprint that this artefact is valid for. It is followed by `[weighted CRC data stored]` if the artefact carries weighted-CRC data. |

The JSON form includes all diagnostics, for example `evaluation_failures`, and `probing_disabled` when probing was skipped:

```console
$ riskgate report tank.cal.json --json --key demo-key
{
 "alpha": 0.1,
 "mode": "policy",
 "delta": 0.05,
 "tau": 2.258805909872077,
 "tau_ood": 2.419992828279894,
 "lam": -1.4987972395723084,
 "fingerprint": "476aaff52ab03d1c",
 "n": 280,
 "lam_probe1": null,
 "lam_probe2": null,
 "probe_share": null,
 "n_probe": 0,
 "created": "2026-10-11T19:08:00.289577+00:00",
 "guarantee": "expectation",
 "delta_ltt": null,
 "extra_alphas": {},
 "diagnostics": {
  "tau_coverage_target": 0.95,
  "calibration_ask_rate_at_lambda": 0.02857142857142857,
  "calibration_risk_at_lambda": 0.05357142857142857,
  "mean_plausible_hypotheses": 1.4964285714285714,
  "n_tau": 120,
  "evaluation_failures": 0
 },
 "id": "5bb204785b4e",
 "schema": "riskgate.calibration/v1"
}
```

Signature handling:

```console
$ riskgate report tank.cal.json --key wrong-key
riskgate: error: SignatureError: calibration artefact signature does not match (tampered or wrong key)
$ echo $?
1
$ riskgate report tank.cal.json          # no key (and RISKGATE_KEY unset): the signature is not checked
riskgate calibration 5bb204785b4e  (schema riskgate.calibration/v1, created 2026-10-11T19:08:00.289577+00:00)
...
```

## `riskgate validate`

```text
riskgate validate CALIBRATION.json --gate MODULE:FACTORY [--key KEY]
```

It builds the gate with the factory, loads the artefact (verifying the signature if a key is given) and compares fingerprints. The fingerprint covers the spec, the `version` strings of the consequence model and the hypothesis provider, the probes, the hard rules, the mode, `doer_id` in policy mode, and `score_fn`/`weight_fn` (see [fingerprint](api.md#gatefingerprint-gatestatus-gateload_calibration)). The command prints the status, `VALID` or `VOID`, and both fingerprints. A freshly built gate has no drift history, so `validate` never reports `DEGRADED`.

```console
$ riskgate validate tank.cal.json --gate toy_gate:make_gate --key demo-key
VALID: gate fingerprint 476aaff52ab03d1c vs artefact 476aaff52ab03d1c
$ echo $?
0
$ riskgate validate tank.cal.json --gate toy_gate:make_gate_v2 --key demo-key     # the model version was bumped
VOID: gate fingerprint e67ca5ec81092636 vs artefact 476aaff52ab03d1c
$ echo $?
3
```

Use it as a deployment gate in CI/CD, so that a build whose components changed never ships with a stale artefact (such a build would answer `ASK` to everything):

```bash
riskgate validate artefacts/tank.cal.json --gate myservice.gate:make_gate    # key from RISKGATE_KEY
status=$?
if [ "$status" -eq 3 ]; then echo "artefact does not match this build: recalibrate" >&2; exit 1; fi
if [ "$status" -ne 0 ]; then exit "$status"; fi
```

## `riskgate audit-summary`

```text
riskgate audit-summary DECISIONS.jsonl
```

This prints verdict and status counts and latency percentiles of a JSONL audit log written by [`AuditLog`](api.md#auditlog). The output is the same as `riskgate.summarize_audit(path)`. To produce a small log, save this as `run_checks.py` and run it:

<!-- file: run_checks.py -->
```python
"""Run a few checks with an audit log so that `riskgate audit-summary` has something to read."""
import riskgate as rg
from toy_gate import make_gate

gate = make_gate()
gate.audit = rg.AuditLog("decisions.jsonl", extra={"service": "tank-1"})
gate.load_calibration("tank.cal.json", key="demo-key", require_signature=True)
for v, level, evidence in [(20, 50, 1.0), (0, 50, 1.0), (40, 30, 1.0), (80, 50, 1.0), (5, 50, 60.0)]:
    d = gate.check(rg.Action("add", {"v": v}), {"level": level, "evidence": evidence})
    print(f"add({v}) at level {level}, evidence {evidence}: {d.verdict.value}")
```

```console
$ python run_checks.py
add(20) at level 50, evidence 1.0: certify
add(0) at level 50, evidence 1.0: certify
add(40) at level 30, evidence 1.0: certify
add(80) at level 50, evidence 1.0: reject
add(5) at level 50, evidence 60.0: ask
$ riskgate audit-summary decisions.jsonl
{
 "decisions": 5,
 "verdicts": {
  "certify": 3,
  "reject": 1,
  "ask": 1
 },
 "status": {
  "valid": 5
 },
 "latency_ms": {
  "p50": 0.2858900697901845,
  "p95": 0.3460879670456052,
  "p99": 0.3536415519192815
 }
}
```

Each line of the log is one decision record (`ts`, the `extra` fields, then `Decision.to_dict()`):

```text
{"ts": "2026-10-11T19:08:25.903376+00:00", "service": "tank-1", "verdict": "certify", "action": {"name": "add",
 "args": {"v": 20}, "read_only": false}, "score": 3.0, "threshold": -1.4987972395723084, "margins": {"leak0": 3.0,
 "leak10": 4.0}, "binding": {...}, "plausible": ["leak0", "leak10"], "probe": null, "reason": "robust margin above
 calibrated threshold", "status": "valid", "calibration_id": "5bb204785b4e", "considered": [...], "stage": 1, ...}
```

## `riskgate serve-mcp`

```text
riskgate serve-mcp --gate MODULE:FACTORY [--calibration CALIBRATION.json] [--key KEY]
                   [--context MODULE:FUNCTION] [--name NAME] [--passthrough TOOL1,TOOL2]
                   -- UPSTREAM_COMMAND [ARGS...]
```

| Flag | Default | Meaning |
|---|---|---|
| `--gate` | required | Gate factory. |
| `--calibration` | none | Artefact to load. Without it the gate stays uncalibrated, and every gated call is answered `ASK`. |
| `--key` | `RISKGATE_KEY` | Verify the artefact's signature (required if set). |
| `--context` | `lambda: None` | Zero-argument function (sync or async) that returns the gate context for each call. If it raises, the call fails closed. |
| `--name` | `riskgate-proxy` | Server name announced to MCP clients. |
| `--passthrough` | none | Comma-separated read-only tools forwarded **without** a gate check. Use it sparingly. |
| `-- UPSTREAM...` | required | Command that starts the upstream MCP server over stdio. |

The proxy serves MCP on its own stdin/stdout, starts the upstream server as a subprocess, and re-exposes the upstream tools. It intercepts every `tools/call`:

- A certified call (or an ASK approved by an `on_ask` hook, Python API only) is forwarded upstream.
- Any other call gets `isError=true` and the gate's JSON feedback.
- The decision summary is attached under `_meta["riskgate/decision"]`.

It needs the MCP extra: `pip install "riskgate[mcp]"`. Without it the command exits 1 (here, in an environment without `mcp`/`anyio`):

```console
$ riskgate serve-mcp --gate toy_gate:make_gate --calibration tank.cal.json --key demo-key -- python plant_server.py
riskgate: error: ModuleNotFoundError: No module named 'anyio'
```

**End to end.** A toy upstream server, saved as `plant_server.py`:

<!-- file: plant_server.py -->
```python
"""A tiny upstream MCP server (stdio): `add` changes the world, `read_level` only reads."""
from mcp.server.mcpserver import MCPServer

server = MCPServer("tank-plant")


@server.tool()
def add(v: float) -> str:
    """Add v litres to the tank."""
    return f"added {v} L"


@server.tool()
def read_level() -> float:
    """Read the current tank level (litres)."""
    return 50.0


if __name__ == "__main__":
    server.run("stdio")
```

An MCP client, saved as `mcp_client.py`, that launches `riskgate serve-mcp` in front of it. Any MCP host (an IDE, a desktop assistant, an agent framework) can launch the same command line.

<!-- file: mcp_client.py -->
```python
"""An MCP client that talks to `riskgate serve-mcp`, which proxies plant_server.py."""
import json
import sys

import anyio
from mcp import Client, StdioServerParameters

proxy = StdioServerParameters(command=sys.executable, args=[
    "-m", "riskgate.cli", "serve-mcp", "--gate", "toy_gate:make_gate", "--calibration", "tank.cal.json",
    "--key", "demo-key", "--context", "toy_gate:snapshot", "--passthrough", "read_level",
    "--", sys.executable, "plant_server.py"])


async def main() -> None:
    async with Client(proxy) as client:
        print("tools:", [t.name for t in (await client.list_tools()).tools])
        for name, args in (("add", {"v": 20}), ("add", {"v": 80}), ("read_level", {})):
            res = await client.call_tool(name, args)
            text = "".join(c.text for c in res.content if getattr(c, "type", None) == "text")
            verdict = (res.meta or {}).get("riskgate/decision", {}).get("verdict", "(not gated)")
            print(f"{name}({json.dumps(args)}): is_error={res.is_error} verdict={verdict} -> {text}")


anyio.run(main)
```

Output with mcp 2.3.0:

```console
$ python mcp_client.py
tools: ['add', 'read_level']
add({"v": 20}): is_error=False verdict=certify -> added 20.0 L
add({"v": 80}): is_error=True verdict=reject -> {"status": "rejected", "reason": "unsafe under every plausible hypothesis (binding constraint: 10<=level<=100)", "guarantee": "valid"}
read_level({}): is_error=False verdict=(not gated) -> 50.0
```

An equivalent entry point is `python -m riskgate.adapters.mcp_proxy --gate SPEC [--context SPEC] [--passthrough a,b] [--name NAME] -- UPSTREAM...`. It also accepts `path/to/file.py:attr`, and a factory that returns `(gate, context_fn)`. It has no `--calibration`/`--key` flags, so the factory must load the artefact itself. In-process use (`build_proxy_server`), `on_ask` hooks and the verified SDK version are covered in [Integrations](integrations.md).

## Planned

The CLI has no `riskgate recalibrate`, no episode loader from JSONL files, and no flags for `extra_losses`, `weights` or probing parameters. Use the Python API for these ([Operations → Recalibration](operations.md#recalibration-workflow)).
