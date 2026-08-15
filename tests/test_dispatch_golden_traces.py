# tests/test_dispatch_golden_traces.py — self-check of the execute() golden traces.
#
# For every scenario in tests/golden_traces/scenarios.py this test re-captures
# the dispatch-pipeline trace (guard-call order + arguments + final text, all
# normalized) and compares it byte-for-byte against the committed fixture in
# tests/golden_traces/fixtures/<scenario>.json. Run it BEFORE and AFTER a
# dispatch refactor: identical fixtures prove identical local behavior.
#
# Regenerate fixtures (only on the KNOWN-GOOD tree):
#   OUROBOROS_UPDATE_GOLDEN_TRACES=1 .venv/bin/python -m pytest tests/test_dispatch_golden_traces.py -q
#
# Scenarios that spawn real processes (run_command / run_script / services /
# verify_and_record) carry @pytest.mark.serial (see tests/conftest.py: the CI
# parallel pass excludes them; they run in the serial lane). Everything else is
# parallel-safe: tmp_path-isolated contexts, monkeypatched env, stubbed
# check_safety, no network and no LLM calls.
import json
import os
import pathlib

import pytest

from tests.golden_traces import capture, scenarios

_FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "golden_traces" / "fixtures"
_UPDATE = os.environ.get("OUROBOROS_UPDATE_GOLDEN_TRACES") == "1"


def _params():
    out = []
    for scenario in scenarios.SCENARIOS:
        marks = [pytest.mark.serial] if scenario.serial else []
        out.append(pytest.param(scenario, id=scenario.name, marks=marks))
    return out


@pytest.mark.parametrize("scenario", _params())
def test_dispatch_golden_trace(scenario, tmp_path, monkeypatch):
    for key, value in scenario.env.items():
        monkeypatch.setenv(key, value)
    # tmp-dir-dependent env (a stub binary's PATH entry) must be in place BEFORE
    # build(), because the scenario body resolves the backend through PATH.
    if scenario.env_factory is not None:
        for key, value in scenario.env_factory(tmp_path).items():
            monkeypatch.setenv(key, value)

    doc = capture.scenario_document(scenario, tmp_path)
    fixture_path = _FIXTURES_DIR / f"{scenario.name}.json"

    if _UPDATE:
        capture.dump_fixture(doc, fixture_path)

    assert fixture_path.is_file(), (
        f"missing golden fixture {fixture_path}; generate with "
        "OUROBOROS_UPDATE_GOLDEN_TRACES=1 pytest tests/test_dispatch_golden_traces.py"
    )
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert doc == expected


def test_every_fixture_has_a_scenario():
    """No orphaned fixtures: each committed JSON maps to a live scenario."""
    names = {s.name for s in scenarios.SCENARIOS}
    assert len(names) == len(scenarios.SCENARIOS), "duplicate scenario names"
    on_disk = {p.stem for p in _FIXTURES_DIR.glob("*.json")}
    assert on_disk <= names, f"orphaned fixtures: {sorted(on_disk - names)}"
