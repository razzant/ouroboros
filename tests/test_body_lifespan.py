"""Server ownership and cold sleep composed with the real background collector."""
from __future__ import annotations

import copy
import json
import threading

import pytest

from tests.test_delegate_body_collection import _until, started, terminal

pytestmark = pytest.mark.serial


def test_lifespan_collects_without_cognitive_supervisor_and_stops_exact_generation(tmp_path, monkeypatch):
    from ouroboros import delegate_collection as body, delegate_custody_current as current, server_maintenance as sm
    from ouroboros import usage_accounting

    started(tmp_path)
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **k: None)
    class Gateway:
        def get_run(self, rid, **kw):
            return terminal(rid)
        def close(self):
            pass
    original = body.start_body_runtime
    monkeypatch.setattr(body, "start_body_runtime", lambda root, **kw: original(root, gateway_factory=Gateway, **kw))
    stop, ended_loop = threading.Event(), threading.Event()
    monkeypatch.setattr(sm, "_supervisor_stop", ended_loop)
    start = sm.bind_body_lifespan(tmp_path, stop)
    start()
    runtime = body.body_runtime(tmp_path)
    try:
        start()
        assert body.body_runtime(tmp_path) is runtime
        _until(lambda: current.run(tmp_path, "run-one").output_complete)
        assert not current.run(tmp_path, "run-one").output_consumed
        # The cognitive loop can leave; only the lifespan token stops its body.
        ended_loop.set()
        assert runtime.active
        stop.set()
        assert not runtime.active
        start()
        assert body.body_runtime(tmp_path) is None
    finally:
        stop.set()
        runtime.close(wait=True)


@pytest.mark.parametrize("close_successor", [False, True])
def test_lifespan_handover_waits_for_physical_return_without_blocking_start(tmp_path, monkeypatch, close_successor):
    from ouroboros import delegate_collection as body, server_maintenance as sm

    started(tmp_path)
    entered, release = threading.Event(), threading.Event()
    class Gateway:
        def get_run(self, rid, **kw):
            entered.set()
            assert release.wait(10)
            return {"summary": {"runId": rid, "state": "running"}}
        def close(self):
            pass
    original = body.start_body_runtime
    monkeypatch.setattr(body, "start_body_runtime", lambda root, **kw: original(root, gateway_factory=Gateway, **kw))
    first_stop, next_stop = threading.Event(), threading.Event()
    first = original(tmp_path, stop_event=first_stop, gateway_factory=Gateway)
    assert entered.wait(5)
    first_stop.set()
    first.close(wait=False)
    start = sm.bind_body_lifespan(tmp_path, next_stop)
    try:
        start()
        start()
        assert first.thread.is_alive() and body.body_runtime(tmp_path) is None
        if close_successor:
            next_stop.set()
        release.set()
        first.close(wait=True)
        if close_successor:
            _until(lambda: not any(t.name == "body-lifespan-handover" for t in threading.enumerate()))
            assert body.body_runtime(tmp_path) is None
        else:
            _until(lambda: body.body_runtime(tmp_path) is not None)
            assert body.body_runtime(tmp_path) is not first
    finally:
        release.set()
        next_stop.set()
        first.close(wait=True)
        body.stop_body_runtime(tmp_path, wait=True)


def test_real_body_cold_terminal_wakes_before_money_and_saved_carrier_is_not_orphaned(tmp_path, monkeypatch):
    from ouroboros import delegate_collection as body, delegate_custody_current as current
    from ouroboros import budget_pause as bp, server_maintenance as sm, sleep_custody as sc, task_status, usage_accounting
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_cold_external_custody import _setup, _park

    env = _setup(tmp_path, monkeypatch)
    finished = threading.Event()
    class Gateway:
        def get_run(self, rid, **kw):
            return terminal(rid) if finished.is_set() else {"summary": {"runId": rid, "state": "running"}}
        def close(self):
            pass
    monkeypatch.setattr(usage_accounting, "record_subscription_session",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("controlled money store busy")))
    runtime = body.start_body_runtime(tmp_path, gateway_factory=Gateway, max_workers=2)
    env.runtime = runtime
    monkeypatch.setattr(sc, "_runtime", body.body_runtime)
    try:
        row = _park(env, tmp_path, monkeypatch)
        assert sc.accepted_receipt("sleeper", row)
        task = env.workers.PENDING[0]
        saved = copy.deepcopy(task)
        env.workers.PENDING.clear()
        monkeypatch.setattr(task_status, "_load_queue_snapshot", lambda _root: {"pending": [{"task": saved}]})
        assert "sleeper" in sm._startup_live_task_ids(tmp_path, include_pending=True)
        assert sc.reconcile(env.q, direct=[saved], runtime=runtime) == {"sleeper"}
        assert bp.budget_pause_row(tmp_path, "sleeper")["external_custody"] == row["external_custody"]
        env.workers.PENDING.append(task)
        assert wake_ready_sleepers(env.q) == [], "a running leaf does not wake its sleeper"
        finished.set()
        assert runtime.ensure_runs(row["external_custody"]["runs"], task_id="sleeper")
        _until(lambda: current.run(tmp_path, "run-one").terminal_state)
        runtime.tick()
        _until(lambda: current.run(tmp_path, "run-one").output_complete)
        fresh = current.run(tmp_path, "run-one")
        assert not fresh.settled and not fresh.ledger_recorded and not fresh.output_consumed
        # A readiness read never waits on the store: one that meets the live collector's write is
        # left to the next assignment tick (sleep_wake._readiness), so tick until a pass decides.
        result = []
        _until(lambda: result.extend(wake_ready_sleepers(env.q)) or result)
        assert result[0]["ok"] and result[0]["ready"]["reason"] == "run:run-one", json.dumps(result)
        assert task["_budget_pause_resume"]["pause"]["checkpoint"]["task_attempt"] == row["task_attempt"]
    finally:
        runtime.close(wait=True)
