# Releasing riskgate

Everything up to "publish" can be done locally and has been checked for `1.0.0rc3` (18 Oct 2026: `python -m build`,
`twine check --strict`, wheel smoke test in isolation, ruff, mypy, core and contract tests, `mkdocs build --strict`).
The steps marked 🔑 need the author's accounts.

## 0. Before the repository goes public

- [ ] `.gitignore` keeps out the benchmark caches (`benchmarks/*/cache`, `*/scratch`, `*.pkl`: about 1.4 GB derived from
      third-party logs, including CC BY-NC data), `TMLR/` (double-blind submission material) and the business notes
      (`REDTEAM_AND_COMMERCIALIZATION.md`, `COMMERCIALIZATION_CASE_STUDIES.md`). Check with `git status --ignored`
      after `git add .`, before the first push.
- [ ] `benchmarks/results/*.json` and `*_report.md` contain aggregate numbers and short quoted excerpts only.
- [ ] The README links are absolute (`https://github.com/reallysaurabh/riskgate/...`) so that they render on PyPI; if
      the repository lives elsewhere, rerun the rewrite with the new owner/repo before building.
- [ ] Confirm the name is free: https://pypi.org/project/riskgate/ should be a 404 (a web search on 13 Oct 2026 found no
      such project).

## 1. Pre-flight (local)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,yaml,otel,docs]" build twine
ruff check riskgate tests examples/*.py examples/integrations
mypy riskgate
pytest -q tests --ignore=tests/contract                 # core, features, STL, LTT, weighted
pip install -e ".[mcp,langgraph,adk,openai-agents]" && pytest -q tests/contract   # framework contracts
(cd examples && PYTHONPATH=.. python quickstart.py && PYTHONPATH=.. python end_to_end_sre.py --quick)
mkdocs build
python -m build && twine check dist/*
```

Regenerate the example results when the core changes (CSTR takes the longest):

```bash
cd examples && PYTHONPATH=.. python run_examples.py && PYTHONPATH=.. python ablations.py && cd ..
```

## 2. 🔑 GitHub

1. Create `github.com/<you>/riskgate` (the URLs in `pyproject.toml` assume `reallysaurabh/riskgate`; change them if needed).
2. `git init && git add . && git commit -m "riskgate 1.0.0rc3" && git remote add origin ... && git push -u origin main`.
   The `.gitignore` excludes virtual environments, build output, `examples/e2e_output/`, the benchmark caches and the
   private material listed in step 0.
3. Check that the `ci` workflow passes (test matrix, contract, examples, docs, package).
4. Optional: GitHub Pages for the docs: `mkdocs gh-deploy`.

## 3. 🔑 PyPI (trusted publishing, no tokens in the repo)

1. On test.pypi.org and on pypi.org: *Your projects → Publishing → Add a pending publisher*: project `riskgate`,
   owner/repo, workflow `release.yml`, environments `testpypi` and `pypi` respectively.
2. In the GitHub repo: *Settings → Environments → New environment* `testpypi` and `pypi` (require a reviewer on `pypi`).
3. Bump the version in `pyproject.toml` and `riskgate/__init__.py` if needed; update `CHANGELOG.md`.
4. Dry run: *Actions → release → Run workflow*. It builds, runs `twine check --strict`, smoke-tests the wheel and
   publishes to TestPyPI. Check the page renders, then `pip install -i https://test.pypi.org/simple/
   --extra-index-url https://pypi.org/simple/ riskgate==1.0.0rc3`.
5. Create a GitHub release with tag `v1.0.0rc3` (mark it as a pre-release). The workflow checks that the tag matches
   the package version and publishes the same build to PyPI. While only pre-releases exist, a plain
   `pip install riskgate` installs the newest one; after a final release, pre-releases need `--pre`.
6. Verify: `pip install riskgate==1.0.0rc3` in a fresh environment and run `examples/quickstart.py`.

## 4. 🔑 Real-LLM evaluation (before 1.0.0 final)

```bash
export RISKGATE_LLM_API_KEY=...                 # any OpenAI-compatible endpoint
export RISKGATE_LLM_BASE_URL=https://...        # optional (defaults to the OpenAI API)
export RISKGATE_LLM_MODEL=...                   # optional
cd examples && PYTHONPATH=.. python end_to_end_sre.py
```

Use the same pattern (`riskgate.doers.LLMRanker`) in the domain `sample()` functions to replace the scripted doers,
re-run `run_examples.py`, and report the doer id with every number. Calibrate in policy mode per model, or use
agnostic mode to compare models under one calibration.

## 5. Paper

The paper is under submission. When a public version exists, add its link to the README citation and to
`CHANGELOG.md`.

## 6. After the release

- Announce (blog post / social), open `good first issue`s for new domain examples and adapters.
- Track adoption questions in GitHub Discussions; keep the calibration artefact schema versioned
  (`riskgate.calibration/vN`) independently of the package version.
