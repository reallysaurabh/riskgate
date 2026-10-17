"""Fixtures for adapter contract tests (real framework versions, fake models, no network).

Every test module calls ``pytest.importorskip`` for its framework, so the suite skips cleanly when the
integration extras are not installed. The domain itself lives in ``toy_domain.py``.
"""
from __future__ import annotations

import pytest
from toy_domain import ASKED, CERTIFIED, REJECTED, STATE, StubGate, make_toy_gate

import riskgate as rg


@pytest.fixture(scope="session")
def toy_gate() -> rg.Gate:
    gate = make_toy_gate()
    for v, expected in ((CERTIFIED, rg.Verdict.CERTIFY), (ASKED, rg.Verdict.ASK), (REJECTED, rg.Verdict.REJECT)):
        assert gate.check(rg.Action("add", {"v": v}), STATE).verdict is expected   # guards the fixture itself
    return gate


@pytest.fixture
def state() -> dict:
    return dict(STATE)


@pytest.fixture
def probe_gate() -> StubGate:
    return StubGate(rg.Verdict.PROBE, probe=rg.Action("read_flowmeter", read_only=True))
