# Contributing to riskgate

Thank you for helping! riskgate is a safety-relevant library, so we value **correctness, clarity and honest guarantees** over features.

## Ground rules
1. **Never weaken the design rules** (docs/SPEC.md §2):
   - no LLM in the certify path;
   - validity comes from calibration, not from the model;
   - fail closed;
   - report the guarantee status;
   - adapters stay thin;
   - everything is auditable;
   - the core depends only on numpy.
2. Any change to calibration must keep the statistical tests passing (`tests/test_core.py::test_crc_controls_risk_in_both_modes`, `tests/test_features.py::test_crc_validity_on_random_monotone_losses`). New guarantees need a written argument in the docstring and in `docs/calibration.md`.
3. **Adapters** go in `riskgate/adapters/`:
   - import the framework lazily;
   - add contract tests under `tests/contract/` that `importorskip` the framework and never call a real LLM or the network;
   - record the tested framework version in `docs/integrations.md`.
4. **Example domains** must label all generated data as simulated (`META["simulated"] = True`).

## Development setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,yaml,otel]"
pre-commit install            # ruff + mypy on commit
pytest -q tests --ignore=tests/contract
pip install -e ".[mcp,langgraph,adk,openai-agents]" && pytest -q tests/contract   # optional
```

## Pull requests
- Keep PRs focused. Include tests and update the docs and `CHANGELOG.md`.
- Sign off your commits (DCO): `git commit -s`.
- Label issues with `good first issue`, `adapter`, `domain-example`, `calibration` or `docs`.

## Reporting security issues
Please do **not** open public issues for vulnerabilities. See `SECURITY.md`.
