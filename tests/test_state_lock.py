"""Regression tests for STATE_LOCK / HTTP isolation.

v6.0.0 fixed `update_budget_from_usage` so that the OpenRouter HTTP request
runs OUTSIDE the file lock (HTTP can take ~10s and would otherwise block all
state I/O). The same regression then re-appeared in `init_state()`. These
tests pin both call sites to that contract.

Run: python -m pytest tests/test_state_lock.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Point supervisor.state at a tmp_path so tests don't touch real Drive."""
    from supervisor import state as state_mod

    monkeypatch.setattr(state_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state" / "state.json")
    monkeypatch.setattr(state_mod, "STATE_LAST_GOOD_PATH",
                        tmp_path / "state" / "state.last_good.json")
    monkeypatch.setattr(state_mod, "STATE_LOCK_PATH", tmp_path / "locks" / "state.lock")
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH",
                        tmp_path / "state" / "queue_snapshot.json")
    return state_mod


def _probe_lock_free(state_mod) -> bool:
    """Try to acquire STATE_LOCK with a short timeout. Returns True if free.

    Uses the same primitive (`acquire_file_lock`) as the rest of state.py so
    the probe matches real contention semantics exactly.
    """
    probe_fd = state_mod.acquire_file_lock(state_mod.STATE_LOCK_PATH, timeout_sec=0.2)
    try:
        return probe_fd is not None
    finally:
        state_mod.release_file_lock(state_mod.STATE_LOCK_PATH, probe_fd)


def test_init_state_does_not_hold_lock_during_http(tmp_state, monkeypatch):
    """init_state() must NOT hold STATE_LOCK while making the OpenRouter
    HTTP request — same contract as the v6.0.0 fix for update_budget_from_usage.
    """
    state_mod = tmp_state
    observed = {"lock_free_during_http": None}

    def fake_check():
        observed["lock_free_during_http"] = _probe_lock_free(state_mod)
        return {"total_usd": 1.23, "daily_usd": 0.45}

    monkeypatch.setattr(state_mod, "check_openrouter_ground_truth", fake_check)

    result = state_mod.init_state()

    assert observed["lock_free_during_http"] is True, (
        "STATE_LOCK was held during check_openrouter_ground_truth() — "
        "regression of the v6.0.0 'HTTP outside STATE_LOCK' fix."
    )
    # Sanity: ground truth was applied to state under the lock.
    assert result["openrouter_total_usd"] == 1.23
    assert result["openrouter_daily_usd"] == 0.45
    assert result["session_total_snapshot"] == 1.23


def test_init_state_handles_http_failure_without_writing_stale_data(tmp_state, monkeypatch):
    """If the OpenRouter call fails, init_state() must still complete and use
    a 0.0 baseline (matches pre-fix behavior — no regression).
    """
    state_mod = tmp_state

    def fake_check():
        return None  # Simulate HTTP failure

    monkeypatch.setattr(state_mod, "check_openrouter_ground_truth", fake_check)

    result = state_mod.init_state()

    assert result["session_total_snapshot"] == 0.0
    assert result["budget_drift_pct"] is None
    assert result["budget_drift_alert"] is False


def test_update_budget_from_usage_does_not_hold_lock_during_http(tmp_state, monkeypatch):
    """Pin the original v6.0.0 contract too, so a future refactor can't silently
    re-introduce the same regression there.
    """
    state_mod = tmp_state
    observed = {"lock_free_during_http": None}

    def fake_check():
        observed["lock_free_during_http"] = _probe_lock_free(state_mod)
        return {"total_usd": 2.0, "daily_usd": 0.1}

    monkeypatch.setattr(state_mod, "check_openrouter_ground_truth", fake_check)

    # spent_calls % 50 == 0 triggers the ground-truth check; seed state so the
    # first call lands on a multiple of 50.
    state_mod.save_state({"spent_calls": 49})
    state_mod.update_budget_from_usage({"cost": 0.0, "rounds": 1})

    assert observed["lock_free_during_http"] is True, (
        "STATE_LOCK was held during check_openrouter_ground_truth() in "
        "update_budget_from_usage — v6.0.0 fix has regressed."
    )
