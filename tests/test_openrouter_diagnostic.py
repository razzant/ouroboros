"""OpenRouter sampling never blocks the caller or publishes against another basis."""
from __future__ import annotations

import contextlib
import io
import threading
from types import SimpleNamespace
import urllib.request

import pytest

from supervisor import state

pytestmark = pytest.mark.serial


@pytest.fixture
def sample(tmp_path, monkeypatch):
    import ouroboros.usage_accounting as accounting

    prior_root, prior_budget = state.DRIVE_ROOT, state.TOTAL_BUDGET_LIMIT
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-A")
    state.init(tmp_path)
    state.save_state({})
    state.update_state(lambda st: st.update(
        session_total_snapshot=1000.0, session_openrouter_settled_snapshot=0.0,
        session_spent_snapshot=0.0, session_openrouter_key_fp=state._openrouter_key_fingerprint(),
    ))
    ledger = {
        "physical_calls": 51, "accounted_usd": 140.0, "settled_usd": 140.0,
        "prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0,
        "confirmed_usd": 140.0, "estimated_usd": 0.0, "reserved_usd": 0.0,
        "unresolved_upper_bound_usd": 0.0, "unknown_unmetered": 0,
        "cost_final": True, "attempt_counts": {"settled": 51}, "integrity_degraded": False,
        "by_provider": {"openrouter": {"settled_usd": 40.0}}, "_ledger_high_water_seq": [0, 51],
    }
    monkeypatch.setattr(accounting, "usage_writer_snapshot", lambda *_a, **_k: dict(ledger))
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_k: dict(ledger))
    # Even a missing test double cannot reach a real provider.
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("unexpected HTTP"))
    yield ledger
    state.init(prior_root, prior_budget)


def _settled(diagnostic):
    assert diagnostic.latch.acquire(timeout=5), "diagnostic did not settle"
    diagnostic.latch.release()


@contextlib.contextmanager
def _held_http(monkeypatch, result=None):
    entered, release = threading.Event(), threading.Event()
    requests = []
    diagnostics = []

    def fetch(api_key):
        requests.append((api_key, threading.get_ident()))
        diagnostics.append(state._OPENROUTER_DIAGNOSTIC)
        entered.set()
        assert release.wait(5), "test did not release the HTTP observation"
        return result or {"total_usd": 1040.0, "daily_usd": 40.0}

    monkeypatch.setattr(state, "check_openrouter_ground_truth", fetch)
    try:
        yield entered, release, requests
    finally:
        release.set()
        for diagnostic in diagnostics:
            _settled(diagnostic)


def test_budget_writer_returns_and_releases_state_lock_while_http_waits(sample, monkeypatch):
    with _held_http(monkeypatch) as (entered, _release, requests):
        assert state.update_budget_from_usage({}) is True
        assert entered.wait(5)
        assert requests == [("test-key-A", requests[0][1])]
        assert requests[0][1] != threading.get_ident()
        state.update_state(lambda st: st.update(message_offset=123), lock_timeout_sec=0.1)
        pending = state.load_state()
        assert pending["message_offset"] == 123 and pending["spent_usd"] == 140.0
        assert "openrouter_last_check_at" not in pending
    result = state.load_state()
    assert result["budget_drift_pct"] == 0.0 and result["message_offset"] == 123
    assert result["openrouter_checked_ledger_settled_usd"] == 40.0


def test_initialization_returns_with_unknown_baseline_while_http_waits(sample, monkeypatch):
    with _held_http(monkeypatch) as (entered, _release, _requests):
        initialized = state.init_state()
        assert initialized.quality == "current"
        assert entered.wait(5)
        pending = state.load_state()
        assert pending["session_total_snapshot"] is None
        assert pending["budget_drift_pct"] is None
        state.update_state(lambda st: st.update(message_offset=321), lock_timeout_sec=0.1)
    result = state.load_state()
    assert result["session_total_snapshot"] == 1040.0
    assert result["session_openrouter_settled_snapshot"] == 40.0
    assert result["session_openrouter_key_fp"] == state._openrouter_key_fingerprint()
    assert result["budget_drift_pct"] is None and result["message_offset"] == 321


def test_failed_initialization_sample_rebaselines_later_instead_of_comparing_with_zero(sample, monkeypatch):
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda _key: None)
    assert state.init_state().quality == "current"
    _settled(state._OPENROUTER_DIAGNOSTIC)
    assert state.load_state()["session_total_snapshot"] is None
    monkeypatch.setattr(state, "check_openrouter_ground_truth",
                        lambda _key: {"total_usd": 5000.0, "daily_usd": 50.0})
    assert state.update_budget_from_usage({}) is True
    _settled(state._OPENROUTER_DIAGNOSTIC)
    result = state.load_state()
    assert result["session_total_snapshot"] == 5000.0
    assert result["session_openrouter_settled_snapshot"] == 40.0
    assert result["budget_drift_pct"] is None and result["budget_drift_alert"] is False


def test_no_key_starts_no_thread_at_boot_or_crossing(sample, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr(state, "threading", SimpleNamespace(
        Lock=threading.Lock, Thread=lambda **_kwargs: pytest.fail("no-key diagnostic started"),
    ))
    assert state.init_state().quality == "current"
    assert state.update_budget_from_usage({}) is True
    result = state.load_state()
    assert result["openrouter_last_check_call"] == 51
    assert result["session_total_snapshot"] is None
    assert "openrouter_last_check_at" not in result


def test_busy_crossing_is_consumed_and_late_result_uses_its_paired_ledger(sample, monkeypatch):
    with _held_http(monkeypatch) as (entered, _release, requests):
        assert state.update_budget_from_usage({}) is True
        assert entered.wait(5)
        sample.update(physical_calls=100, accounted_usd=180.0, _ledger_high_water_seq=[0, 100])
        sample["by_provider"] = {"openrouter": {"settled_usd": 80.0}}
        assert state.update_budget_from_usage({}) is True
        assert len(requests) == 1
        assert state.load_state()["openrouter_last_check_call"] == 100
    result = state.load_state()
    assert result["spent_usd"] == 180.0 and result["openrouter_ledger_settled_usd"] == 80.0
    assert result["openrouter_checked_ledger_settled_usd"] == 40.0
    assert result["budget_drift_pct"] == 0.0
    assert "openrouter tracked: $40.00 vs OpenRouter key: $40.00" in state.status_text({}, [], {})
    checks = []
    monkeypatch.setattr(state, "check_openrouter_ground_truth",
                        lambda key: checks.append(key) or {"total_usd": 1080.0, "daily_usd": 80.0})
    assert state.update_budget_from_usage({}) is True and checks == []
    sample.update(physical_calls=150, _ledger_high_water_seq=[0, 150])
    assert state.update_budget_from_usage({}) is True
    _settled(state._OPENROUTER_DIAGNOSTIC)
    assert checks == ["test-key-A"] and state.load_state()["budget_drift_pct"] == 0.0


def test_request_uses_the_captured_key_even_if_settings_change_before_http(sample, monkeypatch):
    queued, headers = [], []

    class QueuedThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target, self.args = target, args

        def start(self):
            queued.append(self)

    def fetch(request, **_kwargs):
        headers.append(request.get_header("Authorization"))
        return io.BytesIO(b'{"data": {"usage": 5000, "usage_daily": 50}}')

    monkeypatch.setattr(state, "threading", SimpleNamespace(Lock=threading.Lock, Thread=QueuedThread))
    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    assert state.update_budget_from_usage({}) is True
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-B")
    request = queued.pop(0)
    request.target(*request.args)
    assert headers == ["Bearer test-key-A"]
    rejected = state.load_state()
    assert rejected["session_total_snapshot"] == 1000.0
    assert "openrouter_last_check_at" not in rejected

    sample.update(physical_calls=100, _ledger_high_water_seq=[0, 100])
    assert state.update_budget_from_usage({}) is True
    request = queued.pop(0)
    request.target(*request.args)
    result = state.load_state()
    assert headers == ["Bearer test-key-A", "Bearer test-key-B"]
    assert result["session_total_snapshot"] == 5000.0
    assert result["session_openrouter_key_fp"] == state._openrouter_key_fingerprint()
    assert result["budget_drift_pct"] is None


@pytest.mark.parametrize("changed", [
    {"session_total_snapshot": 2000.0},
    {"session_openrouter_settled_snapshot": 30.0},
    {"session_id": "another-session"},
    {"openrouter_last_check_at": "newer-observation", "openrouter_total_usd": 3000.0},
])
def test_late_response_cannot_replace_a_new_baseline_session_or_observation(sample, monkeypatch, changed):
    with _held_http(monkeypatch) as (entered, _release, _requests):
        assert state.update_budget_from_usage({}) is True
        assert entered.wait(5)
        state.update_state(lambda st: st.update(changed))
        saved = state.STATE_PATH.read_bytes()
    assert state.STATE_PATH.read_bytes() == saved


@pytest.mark.parametrize("end", ["stop", "reinitialize"])
def test_closed_generation_discards_late_response(sample, monkeypatch, end):
    stopped = threading.Event()
    state.init(state.DRIVE_ROOT, stop_requested=stopped.is_set)
    with _held_http(monkeypatch) as (entered, _release, requests):
        assert state.update_budget_from_usage({}) is True
        assert entered.wait(5)
        if end == "stop":
            stopped.set()
        else:
            state.init(state.DRIVE_ROOT)
        saved = state.STATE_PATH.read_bytes()
    assert state.STATE_PATH.read_bytes() == saved
    assert len(requests) == 1
    if end == "stop":
        sample.update(physical_calls=100, _ledger_high_water_seq=[0, 100])
        assert state.update_budget_from_usage({}) is True
        assert len(requests) == 1


def test_reinitializing_the_same_key_discards_a_preinitialization_response(sample, monkeypatch):
    with _held_http(monkeypatch) as (entered, _release, _requests):
        assert state.update_budget_from_usage({}) is True
        assert entered.wait(5)
        monkeypatch.setattr(state, "check_openrouter_ground_truth",
                            lambda _key: {"total_usd": 2000.0, "daily_usd": 20.0})
        assert state.init_state().quality == "current"
        _settled(state._OPENROUTER_DIAGNOSTIC)
        saved = state.STATE_PATH.read_bytes()
    assert state.STATE_PATH.read_bytes() == saved
    assert state.load_state()["session_total_snapshot"] == 2000.0


def test_thread_start_failure_releases_latch_without_replaying_the_crossing(sample, monkeypatch, caplog):
    class FailedThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(state, "threading", SimpleNamespace(Lock=threading.Lock, Thread=FailedThread))
        assert state.update_budget_from_usage({}) is True
        assert "OpenRouter diagnostic could not start" in caplog.text
        assert state.load_state()["openrouter_last_check_call"] == 51
    _settled(state._OPENROUTER_DIAGNOSTIC)
    checks = []
    monkeypatch.setattr(state, "check_openrouter_ground_truth",
                        lambda key: checks.append(key) or {"total_usd": 1040.0, "daily_usd": 40.0})
    assert state.update_budget_from_usage({}) is True and checks == []
    sample.update(physical_calls=100, _ledger_high_water_seq=[0, 100])
    assert state.update_budget_from_usage({}) is True
    _settled(state._OPENROUTER_DIAGNOSTIC)
    assert checks == ["test-key-A"] and state.load_state()["budget_drift_pct"] == 0.0


def test_fetch_failure_keeps_previous_observation_and_money(sample, monkeypatch, caplog):
    state.update_state(lambda st: st.update(openrouter_last_check_at="previous", openrouter_total_usd=1020.0))

    def unavailable(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    assert state.update_budget_from_usage({}) is True
    _settled(state._OPENROUTER_DIAGNOSTIC)
    result = state.load_state()
    assert result["openrouter_last_check_at"] == "previous" and result["openrouter_total_usd"] == 1020.0
    assert result["spent_usd"] == 140.0 and result["openrouter_last_check_call"] == 51
    assert "Failed to fetch OpenRouter ground truth" in caplog.text
