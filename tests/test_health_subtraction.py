"""The health block carries obligations with an action and plain facts, never reassurance:
no ``OK:`` line, no identity/scratchpad or cost verdict (and no usage-store read for one), and
a custody read that fails is said with its reader instead of replayed by three fallbacks."""
from __future__ import annotations

import json

import pytest

from ouroboros import context_health
from ouroboros.delegate_custody import RunCustody
from tests._context_shared import _make_health_env


def test_a_quiet_install_renders_facts_without_ok_lines_or_cost_verdicts(tmp_path, monkeypatch):
    from supervisor import state as supervisor_state

    def no_usage_read(*_a, **_k):
        raise AssertionError("the health block must not open the usage store for a cost verdict")

    monkeypatch.setattr(supervisor_state, "per_task_cost_summary", no_usage_read)
    env = _make_health_env(tmp_path)
    (tmp_path / "memory" / "identity.md").write_text("I am Ouroboros.", encoding="utf-8")
    text = context_health.build_health_invariants(env)
    assert "identity.md last updated 0.0 h ago" in text
    for gone in ("OK:", "HIGH-COST", "high-cost", "STALE IDENTITY", "THIN IDENTITY", "SCRATCHPAD", "version sync",
                 "budget drift within"):
        assert gone not in text, gone


def test_counts_and_obligations_with_an_action_stay(tmp_path):
    env = _make_health_env(tmp_path, events_lines=[
        json.dumps({"type": "llm_api_error", "model": "provider/model"}),
        json.dumps({"type": "local_context_overflow", "model": "provider/model"}),
    ])
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "9.9.9"', encoding="utf-8")
    text = context_health.build_health_invariants(env)
    assert "WARNING: PROVIDER/ROUTING ERRORS — 1 recent failures (provider/model x1)" in text
    assert "WARNING: LOCAL CONTEXT OVERFLOW — 1 recent overflow event(s)" in text
    assert "CRITICAL: VERSION DESYNC" in text and "OK:" not in text


def test_an_unreadable_custody_read_is_said_with_its_reader_and_replays_nothing(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody
    import ouroboros.delegate_custody_current as current

    def unreadable(root, task_id=""):
        raise OSError("custody store locked")

    replayed: list = []  # recorded, not raised: the blocks are fail-soft and would swallow an error
    monkeypatch.setattr(current, "health_runs", unreadable)
    for name in ("replay", "settled_unread_outputs", "undisposed_patches"):
        monkeypatch.setattr(custody, name, lambda *a, _name=name, **k: replayed.append(_name) or [])
    monkeypatch.setattr(custody, "open_containment_faults",
                        lambda root: [{"run_id": "run-live", "reason": "stop unverified", "task_id": "t", "ts": "T"}])
    text = context_health.build_health_invariants(_make_health_env(tmp_path), task_id="reader")
    line = next(row for row in text.splitlines() if "CUSTODY UNAVAILABLE" in row)
    assert "OSError: custody store locked" in line and "nothing here proves their absence" in line
    assert "query_data(op='sql', db='current', query=\"SELECT run_id, task_id, state FROM custody_runs" in line
    assert "CRITICAL: DELEGATED RUN MAY STILL BE LIVE — run run-live" in text  # its own read, kept
    assert "DELEGATED WORK" not in text and "NEVER READ" not in text and "AWAITS DISPOSITION" not in text
    assert replayed == [], "no block replays every row for itself while custody is unavailable"


def test_a_readable_custody_read_still_names_the_obligations(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody

    unread = RunCustody(run_id="run-u", task_id="reader")
    unread.output_artifact = "delegated/run-u.out"
    seen = []
    monkeypatch.setattr(custody, "settled_unread_outputs", lambda root, state=None: seen.append(state) or [unread])
    monkeypatch.setattr(custody, "undisposed_patches", lambda root, state=None: [])
    text = context_health.build_health_invariants(_make_health_env(tmp_path), task_id="reader")
    assert "CUSTODY UNAVAILABLE" not in text and "DELEGATED RESULT NEVER READ — run run-u" in text
    assert seen and seen[0] is not None  # the one selected read reached the block


@pytest.mark.parametrize("name", ["_memory_health_lines", "SCRATCHPAD_BLOAT_WARN_CHARS"])
def test_the_threshold_helpers_are_gone(name):
    from ouroboros import context_budget

    assert not hasattr(context_health, name) and not hasattr(context_budget, name)
