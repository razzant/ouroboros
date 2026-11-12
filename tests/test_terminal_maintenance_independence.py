"""A blocked recovery duty cannot occupy the other domains' dispatch slot."""
from __future__ import annotations

import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from tests._delegate_capture_child import idle_maintenance_child  # noqa: F401
from tests._cases import case_folders

pytestmark = [pytest.mark.usefixtures("idle_maintenance_child")]

STEPS = ("cancel", "delivery", "questions", "notice", "usage")
THREADS = {"cancel": "cancel_intent_sweep", "delivery": "terminal_delivery_replay",
           "questions": "question_delivery_replay", "notice": "pause_notice_replay",
           "usage": "abandoned_usage_reconciliation"}


def test_each_blocked_duty_leaves_the_other_four_runnable(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for blocked in STEPS:
        tmp_path = next(case_paths)
        with subtests.test(blocked=blocked), monkeypatch.context() as monkeypatch:
            from ouroboros import server_maintenance as sm
            from supervisor import queue

            clock = [1000.0]
            entered, release = threading.Event(), threading.Event()
            finished = {name: threading.Event() for name in STEPS}
            calls, threads = Counter(), []
            guard = threading.Lock()
            monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
            monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
            monkeypatch.setattr(sm, "_DUTIES", {})
            monkeypatch.setattr(sm, "_STEP_FAILURES", {})
            monkeypatch.setattr(sm, "time", SimpleNamespace(time=lambda: clock[0]))
            monkeypatch.setattr(sm, "_stop_requested", lambda stop=None: bool(stop and stop.is_set()))
            monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
            for name in ("_CANCEL_INTENT_SWEEP_LOCK", "_DELIVERY_SWEEP_LOCK", "_QUESTION_SWEEP_LOCK", "_PAUSE_NOTICE_SWEEP_LOCK",
                         "_USAGE_SWEEP_LOCK"):
                monkeypatch.setattr(sm, name, threading.Lock())

            def run(name):
                with guard:
                    calls[name] += 1
                if name == blocked:
                    entered.set()
                    assert release.wait(10), "fixture must release its blocked duty"
                finished[name].set()
                return []

            def tracked(**kwargs):
                thread = threading.Thread(**kwargs)
                threads.append(thread)
                return thread

            monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=tracked))
            monkeypatch.setattr("supervisor.task_lifecycle.sweep_cancel_intents", lambda: run("cancel"))
            monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda _root: run("delivery"))

            async def questions(_root):
                run("questions")

            monkeypatch.setattr("ouroboros.gateway.task_decision.recover_question_deliveries", questions)
            monkeypatch.setattr("ouroboros.pause_notices.reconcile_pause_notices", lambda _root, **_kw: run("notice"))
            monkeypatch.setattr(sm, "_reconcile_abandoned_usage", lambda _root: run("usage"))
            try:
                sm._periodic_supervisor_maintenance([clock[0]], [clock[0]])
                assert entered.wait(2)
                assert all(finished[name].wait(2) for name in STEPS if name != blocked)
                assert not finished[blocked].is_set()
                # A duty ends when its latch is released (the question duty after its event loop closes).
                for thread in threads:
                    if thread.name != THREADS[blocked]:
                        thread.join(2)
                clock[0] += 25
                sm._periodic_supervisor_maintenance([clock[0]], [clock[0]])
                for thread in threads:
                    if thread.name != THREADS[blocked]:
                        thread.join(2)
                assert calls[blocked] == 1
                assert all(calls[name] == 2 for name in STEPS if name != blocked)
            finally:
                release.set()
                for thread in threads:
                    thread.join(5)
            assert all(not thread.is_alive() for thread in threads)


def test_closed_generation_runs_no_terminal_duty_and_releases_its_own_latch(tmp_path, monkeypatch):
    from ouroboros import server_maintenance as sm

    stop, latch = threading.Event(), threading.Lock()
    stop.set()
    latch.acquire()
    calls = []
    monkeypatch.setattr(sm, "_reconcile_abandoned_usage", lambda _root: calls.append("called"))
    sm._run_terminal_duty(tmp_path, "abandoned_usage_reconciliation", latch, stop)
    assert calls == []
    assert not latch.locked()


def test_failed_duty_releases_latch_without_clearing_another_failure(tmp_path, monkeypatch):
    from ouroboros import server_maintenance as sm

    latch = threading.Lock()
    latch.acquire()
    monkeypatch.setattr(sm, "_STEP_FAILURES", {"terminal_delivery_replay": 3})
    monkeypatch.setattr(sm, "_stop_requested", lambda _stop=None: False)

    def fail(_root):
        raise OSError("controlled bookkeeping failure")

    monkeypatch.setattr(sm, "_reconcile_abandoned_usage", fail)
    sm._run_terminal_duty(tmp_path, "abandoned_usage_reconciliation", latch)
    assert not latch.locked()
    assert sm._STEP_FAILURES == {"terminal_delivery_replay": 3, "abandoned_usage_reconciliation": 1}


def test_cold_transfer_observation_uses_body_cadence_without_running_each_tick(tmp_path, monkeypatch):
    from ouroboros import server_maintenance as sm
    from ouroboros.delegate_feed import _TICK_SEC
    from supervisor import active_activity, queue

    clock, started = [1000.0], []
    monkeypatch.setattr(sm, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(sm, "_stop_requested", lambda _stop=None: False)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [clock[0]])
    monkeypatch.setattr(sm, "_LAST_COLD_CUSTODY_SWEEP", [0.0])
    monkeypatch.setattr(sm, "_COLD_CUSTODY_SWEEP_LOCK", threading.Lock())
    monkeypatch.setattr(queue, "RUNNING", {"current": {}})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(actors=lambda: []))

    def start(latch, name, target, args):
        if name != "schedule-admission":  # the schedule duty starts on every tick by design
            started.append(name)
        latch.release()
        return True

    monkeypatch.setattr(sm, "_start_maintenance_thread", start)
    def tick():
        sm._periodic_supervisor_maintenance([clock[0]], [clock[0]])
    tick()
    assert started == ["cold-custody"]
    clock[0] += _TICK_SEC / 2
    tick()
    assert started == ["cold-custody"]
    clock[0] += _TICK_SEC / 2
    tick()
    assert started == ["cold-custody", "cold-custody"]
    queue.RUNNING.clear()
    clock[0] += _TICK_SEC
    tick()
    assert len(started) == 2
