"""Indexed post-task recovery remains independent from slow monetary housekeeping."""
from __future__ import annotations

import queue as queues
import threading
from types import SimpleNamespace

import pytest

from tests.test_synthesis_ownership_recovery import work as work, checkpoint
from tests.test_delegate_body_collection import _until

pytestmark = pytest.mark.serial


def test_real_maintenance_recovers_unsent_phase_while_usage_duty_is_blocked(work, monkeypatch):
    from ouroboros import server_maintenance as sm, post_task_checkpoint as phase
    from supervisor import workers

    entered, release = threading.Event(), threading.Event()
    threads, bus = [], queues.Queue()
    monkeypatch.setattr(sm, "DATA_DIR", work.root)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(sm, "_DUTIES", {})
    monkeypatch.setattr(sm, "_stop_requested", lambda token=None: bool(token and token.is_set()))
    monkeypatch.setattr(sm, "time", SimpleNamespace(time=lambda: 1000.0))
    for name in ("_CANCEL_INTENT_SWEEP_LOCK", "_DELIVERY_SWEEP_LOCK", "_PAUSE_NOTICE_SWEEP_LOCK",
                 "_USAGE_SWEEP_LOCK", "_SYNTHESIS_SWEEP_LOCK"):
        monkeypatch.setattr(sm, name, threading.Lock())
    monkeypatch.setattr(workers, "get_event_q", lambda **kw: bus)
    monkeypatch.setattr("supervisor.task_lifecycle.sweep_cancel_intents", lambda: {})
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda *a: None)
    monkeypatch.setattr("ouroboros.pause_notices.reconcile_pause_notices", lambda *a, **kw: None)
    def held(_root):
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(sm, "_reconcile_abandoned_usage", held)
    def tracked(**kw):
        thread = threading.Thread(**kw)
        threads.append(thread)
        return thread
    monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=tracked))
    try:
        sm._periodic_supervisor_maintenance([1000.0], [1000.0])
        assert entered.wait(3)
        _until(lambda: checkpoint(work)["post_task_synthesis"] == "completed")
        assert not release.is_set()
        assert work.calls == ["fallback", "promotion"]
        _until(lambda: not phase.post_task_synthesis_in_flight(work.root, work.task["id"]))
        # Another ordinary pass has no paid remainder to repeat.
        latch = threading.Lock()
        latch.acquire()
        sm._run_terminal_duty(work.root, "post_task_synthesis_recovery", latch)
        assert work.calls == ["fallback", "promotion"]
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
    assert all(not thread.is_alive() for thread in threads)
