"""Exact paused work across a stop, a missing source, and racing owner actions."""

import errno
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _parked, _quiet_external


def technical(root, mp):
    from ouroboros import budget_pause as bp

    q, state, workers = _install_queue(root, mp)
    mp.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    from tests.test_state_authority import _prior

    state.save_state(_prior())
    task, row = _parked(root, mp, task_id="technical")
    row.update(
        reason="authority",
        scope="task",
        rail=bp.RAIL_CONTROL_AUTHORITY,
        cause="dispatch_fence_invalid",
        state="paused",
        auto_resume=True,
    )
    bp.set_budget_pause(root, task["id"], row)
    task["_budget_pause"] = bp.exact_pause_marker(row)
    assert q.publish_queue_snapshot("technical_park")
    return q, workers, task


@pytest.mark.parametrize("door", ["owner_restart", "panic_boot", "update"])
def test_stop_vs_update(tmp_path, monkeypatch, door):
    from ouroboros import budget_pause as bp
    from supervisor.events_budget import budget_hold_fact
    from supervisor.sleep_wake import wake_ready_sleepers

    q, workers, task = technical(tmp_path, monkeypatch)
    if door == "owner_restart":
        workers.kill_workers(
            force=True, terminal_status="cancelled", result_reason="Owner restart", hold_never_started=True
        )
    else:
        workers.PENDING.clear()
        q.init(tmp_path)
        flag = tmp_path / "state/panic_stop.flag"
        if door == "panic_boot":
            flag.write_text("panic", encoding="utf-8")
        assert q.restore_pending_from_snapshot() == 1
        if door == "panic_boot":
            assert not any(x["ok"] for x in wake_ready_sleepers(q))
            from supervisor.worker_chat_lane import auto_resume_after_restart

            auto_resume_after_restart()
            assert not flag.exists()
    hold = budget_hold_fact(workers.PENDING[0])
    assert bool(hold) == (door != "update")
    woke = any(x["ok"] for x in wake_ready_sleepers(q))
    assert bp.budget_pause_row(tmp_path, task["id"])["state"] == ("resume_granted" if woke else "paused")
    assert woke == (door == "update")


def test_restore_busy(tmp_path, monkeypatch):
    from ouroboros import store, store_hot_facts

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    task, _ = _parked(tmp_path, monkeypatch, task_id="already-saved")
    assert q.publish_queue_snapshot("before_restart")
    workers.PENDING.clear()
    q.init(tmp_path)

    from ouroboros import budget_pause as bp

    hidden, _ = _parked(tmp_path, monkeypatch, task_id="saved-envelope", extra={"_is_direct_chat": True})
    hidden_row = bp.budget_pause_row(tmp_path, hidden["id"])
    bp.set_budget_pause(tmp_path, hidden["id"], {**hidden_row, "state": "pausing", "park_task": dict(hidden)})
    workers.PENDING.clear()
    before = q.QUEUE_SNAPSHOT_PATH.read_bytes()

    def busy(*args):
        raise store.StoreBusy("one indexed read was busy")

    with monkeypatch.context() as m:
        m.setattr(store_hot_facts, "pausing_envelopes", busy)
        recovered = q.restore_pending_from_snapshot()
        assert [q.publish_queue_changes() for _ in range(3)] == [None] * 3
        assert q.publish_queue_snapshot("still_busy") is False
        assert q.QUEUE_SNAPSHOT_PATH.read_bytes() == before
    initial_pending = [row["id"] for row in q.PENDING]
    q.PENDING.append({"id": "new-arrival", "type": "task", "chat_id": 0})
    published = q.publish_queue_snapshot("admit")
    ids = [r["id"] for r in json.loads(q.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["pending"]]
    assert recovered == 1 and initial_pending == [task["id"]]
    assert published and set(ids) == {task["id"], hidden["id"], "new-arrival"}


def test_late_orphan_sweep_reads_the_live_tree(tmp_path, monkeypatch):
    """A boot whose envelope index was busy sweeps its orphaned latches later, in the live queue. A latch it left
    over a dead tree goes as at boot; a tree that runs again and a latch raised since keep refusing money."""
    from ouroboros import store, store_hot_facts
    from ouroboros.queue_observation import dispatch_probe
    from supervisor.events_budget import _handle_budget_root_fence, _set_root_budget_pause_locked
    from tests.test_budget_pause_monetary_selection import CHILD, FENCED, ROOT, _reserve, _tree

    q, _, workers = _tree(tmp_path, monkeypatch)
    running = dict(workers.RUNNING)
    workers.RUNNING.clear()
    with q._queue_lock:
        for root in (ROOT, "settled-root"):
            _set_root_budget_pause_locked(root, {})
    assert q.publish_queue_snapshot("latched")
    q.init(tmp_path)

    def busy(_root):
        raise store.StoreBusy("index busy at boot")

    with monkeypatch.context() as m:
        m.setattr(store_hot_facts, "pausing_envelopes", busy)
        assert q.restore_pending_from_snapshot() == 0
    workers.RUNNING.update(running)  # the tree runs again in this process
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=lambda _e: None),
                          persist_queue_snapshot=q.persist_queue_snapshot)
    _handle_budget_root_fence({"type": "budget_root_fence", "task_id": "later-child",
                               "resource_limit": {"scope": "root", "root_task_id": "later-root"}}, ctx)
    assert q.publish_queue_snapshot("after_late_pass")
    latched = {row["root_task_id"] for row in dispatch_probe(tmp_path)["budget_root_fences"]}
    assert latched == {ROOT, "later-root"} and _reserve(tmp_path, CHILD) == FENCED


@pytest.mark.parametrize("presence", [False, True])
def test_pause_during_shutdown(tmp_path, monkeypatch, presence):
    from ouroboros import budget_pause as bp
    from ouroboros.store_hot_facts import pausing_envelopes

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    assert q.publish_queue_snapshot("before_turn")
    tid = "turn"
    ctx, limit = _loop_ctx(tmp_path, tid, direct=True)
    task = {
        "id": tid,
        "type": "presence" if presence else "task",
        "_is_direct_chat": True,
        "_attempt": 1,
        "title": "Retain",
        "text": "work",
        "chat_id": 0,
        "task_contract": {"objective": "work"},
        "root_task_id": tid,
    }
    if presence:
        task["_presence_turn"] = True
    ctx.model_wait_context = SimpleNamespace(task=task, continuation_state=lambda: {})
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    with pytest.raises(bp.BudgetPauseRequested):
        bp.request_pause(
            limit,
            rail=bp.RAIL_CONTROL_AUTHORITY,
            scope="task",
            reason_text="invalid dispatch fence",
            cause="dispatch_fence_invalid",
        )
    bp.end_dispatch_fence(tid)
    assert tid in pausing_envelopes(tmp_path)
    workers.RUNNING[tid] = {"task": task, "attempt": 1}
    write = q.atomic_write_text

    def full(path, data):
        if path == q.QUEUE_SNAPSHOT_PATH:
            raise OSError(errno.ENOSPC, "disk full")
        return write(path, data)

    with monkeypatch.context() as m:
        m.setattr(q, "atomic_write_text", full)
        workers.kill_workers(
            force=True,
            terminal_status="cancelled",
            result_reason="Server shutdown",
            retain_saved_work=True,
            stop_source="server_shutdown",
        )
    old = json.loads(q.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    saved = bp.budget_pause_row(tmp_path, tid)
    workers.PENDING.clear()
    workers.RUNNING.clear()
    q.init(tmp_path)
    restored = q.restore_pending_from_snapshot()
    assert not old["pending"] and saved["state"] == "pausing"
    assert restored == 1 and bp.budget_pause_row(tmp_path, tid)["state"] == "paused"
    assert q.resume_budget_paused_task(tid)["ok"]


@pytest.mark.parametrize("late", [False, True])
def test_late_direct_park_vs_resume(tmp_path, monkeypatch, late):
    from ouroboros import budget_pause as bp
    from supervisor import events_budget as events
    from tests._budget_pause_exact_helpers import _supervisor_ctx

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id="direct", scope="root", extra={"_is_direct_chat": True})
    tid = task["id"]
    evt = bp.pause_event({k: v for k, v in task.items() if k != "_budget_pause"}, row)
    assert q.publish_queue_snapshot("before")
    sup = _supervisor_ctx(tmp_path, workers, q, [], [])
    sup.publish_queue_snapshot = q.publish_queue_snapshot
    entered, release = threading.Event(), threading.Event()
    capture = bp.exact_pause_marker

    def deliver():
        return events._handle_budget_pause(evt, sup)

    def yield_marker(*args, **kw):
        value = capture(*args, **kw)
        if threading.current_thread().name.startswith("late-direct"):
            entered.set()
            assert release.wait(12)
        return value

    if late:
        monkeypatch.setattr(bp, "exact_pause_marker", yield_marker)
        with ThreadPoolExecutor(1, thread_name_prefix="late-direct") as pool:
            future = pool.submit(deliver)
            try:
                assert entered.wait(12)
                resume = q.resume_budget_paused_task(tid)
                assert resume["ok"] and tid not in q.BUDGET_ROOT_FENCES
            finally:
                release.set()
            future.result()
    else:
        deliver()
        resume = q.resume_budget_paused_task(tid)
        assert resume["ok"]
    after = bp.budget_pause_row(tmp_path, tid)
    assert after["grant"]["grant_id"] == resume["grant_id"]
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, UsageScope, reserve_attempt, usage_scope

    with usage_scope(UsageScope(drive_root=tmp_path, root_task_id=tid, task_id=tid)):
        try:
            reserve_attempt(AttemptRequest(model="m", provider="p", task_id=tid, reservation_usd=0.01))
            admitted = True
        except BudgetExceeded:
            admitted = False
    assert tid not in q.BUDGET_ROOT_FENCES and admitted


@pytest.mark.parametrize("repaired", [False, True])
def test_same_owner_read_on_wake(tmp_path, monkeypatch, repaired):
    from ouroboros import budget_pause as bp, owner_pause, usage_admission
    from ouroboros.usage_accounting import AttemptRequest, UsageScope, reserve_attempt, usage_scope
    from ouroboros.task_results import load_task_result
    from supervisor.sleep_wake import authority_readable, wake_ready_sleepers

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    root, child = "former-root", "live-child"
    with usage_scope(UsageScope(drive_root=tmp_path, root_task_id=root, task_id=root)):
        reserve_attempt(AttemptRequest(model="m", provider="p", task_id=root, reservation_usd=0.01))
    task, row = _parked(tmp_path, monkeypatch, task_id=child, root_task_id=root)
    row.update(
        state="paused",
        reason="authority",
        rail=bp.RAIL_CONTROL_AUTHORITY,
        scope="task",
        cause="owner_pause_authority_unreadable",
        auto_resume=True,
    )
    bp.set_budget_pause(tmp_path, child, row)
    task["_budget_pause"] = bp.exact_pause_marker(row)
    assert not load_task_result(tmp_path, root, strict=True) and usage_admission.ledger_billing_binding(
        tmp_path, root
    )
    assert q.publish_queue_snapshot("before")
    actor = SimpleNamespace(
        task_id=child, root_task_id=root, budget_drive_root=tmp_path, task_lifecycle_bound=True
    )
    if repaired:
        from ouroboros.task_results import write_task_result

        write_task_result(tmp_path, root, "running")
    actual = owner_pause.member_fence(actor).get("reason")
    from ouroboros.store import bounded_wait

    observed = []
    original = owner_pause.member_fence

    def probe(source):
        observed.append(bounded_wait(5))
        return original(source)

    with monkeypatch.context() as m:
        m.setattr(owner_pause, "member_fence", probe)
        ready = authority_readable(task, row, tmp_path)
    assert observed == [0]
    grant = wake_ready_sleepers(q)
    assert bool(actual) != repaired
    assert ready == repaired and any(item.get("ok") for item in grant) == repaired


def test_consumed_notice(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp
    from supervisor.sleep_wake import wake_ready_sleepers

    q, _, task = technical(tmp_path, monkeypatch)
    assert wake_ready_sleepers(q)[0]["ok"]
    handoff = task["_budget_pause_resume"]
    ctx, limit = _loop_ctx(tmp_path, task["id"])
    ctx.budget_pause_resume = handoff
    blob = bp.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *a, **kw: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *a, **kw: None)
    monkeypatch.setattr(bp, "_hold_control_reason", lambda *_: "test_stop")
    messages = list(limit.messages)
    bp.resume_paused_loop(limit.tools, blob, messages, {}, {}, set(), budget_remaining_usd=5.0)
    notice = messages[-1]["content"].split("(paused ")[0]
    assert handoff["authority"] == "authority_readable" and not handoff["by"]
    assert "explicit Resume" not in notice and ("dispatch_fence_invalid" in notice or "stopped for" in notice)


@pytest.mark.parametrize("technical_wait", [False, True])
def test_working_recovery_keeps_same_auto_resume_policy(tmp_path, technical_wait):
    from ouroboros import working_checkpoint as wc, budget_pause as bp
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _loop_ctx

    _, ctx = _loop_ctx(tmp_path)
    write_task_result(tmp_path, "pause-task", "running")
    wc.save_round(ctx, "pre_effect")
    seed = {
        "pause_id": "p1",
        "state": bp.STATE_PAUSING,
        "reason": "authority" if technical_wait else "budget",
        "rail": bp.RAIL_CONTROL_AUTHORITY if technical_wait else "soft_land",
        "pause_generation": 1,
        "scope": "member",
        "task_attempt": 1,
        "source_ref": None,
        **({"cause": "dispatch_fence_invalid", "auto_resume": True, "resume_policy": "authority_readable"}
           if technical_wait else {}),
    }
    bp.set_budget_pause(tmp_path, "pause-task", seed)
    row = wc.complete_pause_from_working(tmp_path, "pause-task", seed, 1)
    assert row["source_ref"] and row.get("cause") == seed.get("cause")
    assert row["auto_resume"] is technical_wait
    assert row["resume_policy"] == ("authority_readable" if technical_wait else bp.RESUME_POLICY)


def test_recovered_park_confirms_the_source_that_moved_it(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp
    from supervisor import queue_snapshot, restart_retention

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    task, pause = _parked(tmp_path, monkeypatch)
    workers.PENDING.clear()
    bp.set_budget_pause(tmp_path, task["id"], {**pause, "state": "pausing", "park_task": {**task, "text": "old"}})
    task["text"] = "current"
    parked = restart_retention.park_saved_pause(task, 1, tmp_path, pause_source="restart_owner_wait_handoff")
    assert parked["text"] == "current"
    workers.PENDING.append(parked)
    queue_snapshot._publish_restored()
    assert bp.budget_pause_row(tmp_path, task["id"])["pause_source"] == "restart_owner_wait_handoff"


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("decision", ["acceptance", "cancel_intent", "terminal", "ready"])
def test_envelope_discovery_obeys_the_same_boot_decisions(tmp_path, monkeypatch, decision, late):
    from ouroboros import budget_pause as bp, store, store_hot_facts
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import write_task_result
    from ouroboros.work_stop import continuation_refusal
    from supervisor.restart_retention import publish_pending_parks
    from supervisor.sleep_wake import wake_ready_sleepers

    q, workers, _ = technical(tmp_path, monkeypatch)
    tid = "direct"
    if decision == "acceptance":
        q.ACCEPTANCE_FENCES[tid] = {"status": "active", "root_task_id": tid, "token": "review", "generation": 1}
    assert q.publish_queue_snapshot("before_restart")
    workers.PENDING.clear()
    q.init(tmp_path)
    hidden, pause = _parked(tmp_path, monkeypatch, task_id=tid, extra={"_is_direct_chat": True})
    pause.update(state="pausing", reason="authority", rail=bp.RAIL_CONTROL_AUTHORITY, scope="task",
                 cause="dispatch_fence_invalid", auto_resume=True)
    hidden["_budget_pause"] = bp.exact_pause_marker(pause)
    bp.set_budget_pause(tmp_path, tid, {**pause, "park_task": dict(hidden)})
    workers.PENDING.clear()

    def busy(root):
        raise store.StoreBusy("index busy at boot")

    if late:
        with monkeypatch.context() as m:
            m.setattr(store_hot_facts, "pausing_envelopes", busy)
            assert q.restore_pending_from_snapshot() == 1
    if decision == "cancel_intent":
        request_cancel(tmp_path, tid, reason="owner_cancel", source="task_cancel")
    elif decision == "terminal":
        write_task_result(tmp_path, tid, "cancelled", result="owner cancelled")
    if not late:
        q.restore_pending_from_snapshot()

    q.PENDING.append({"id": "other", "type": "task", "chat_id": 0})
    assert q.publish_queue_snapshot("other_work")
    ids = {row["id"] for row in json.loads(q.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["pending"]}
    assert {"other", "technical"} <= ids
    assert (tid in ids) == (decision in {"ready", "acceptance"})
    if tid in ids:
        publish_pending_parks(q, "confirm")
        paused = next(row for row in q.PENDING if row["id"] == tid)
        assert paused.get("_budget_pause")
        assert bool(continuation_refusal(paused)) == (decision == "acceptance")
        assert any(item.get("ok") for item in wake_ready_sleepers(q, only=tid)) == (decision == "ready")


@pytest.mark.parametrize("busy_read", ["every", "final"])
def test_a_busy_pause_read_leaves_the_late_decision_to_the_next_publication(tmp_path, monkeypatch, busy_read):
    """The late pass reads each saved envelope's pause row. While a writer holds the store that read decides
    nothing, whether it is the classifying read or the park's own read after it: the saved queue stays and the
    next publication restores the pause, which then resumes. A readable row whose cancel is confirmed is still
    not restored."""
    from ouroboros import budget_pause as bp, store, store_hot_facts
    from ouroboros.cancel_intents import has_active_intent, request_cancel
    from supervisor.restart_retention import publish_pending_parks

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _parked(tmp_path, monkeypatch, task_id="known")
    assert q.publish_queue_snapshot("before_restart")
    workers.PENDING.clear()
    q.init(tmp_path)
    for tid in ("direct", "stopped"):
        task, pause = _parked(tmp_path, monkeypatch, task_id=tid, extra={"_is_direct_chat": True})
        bp.set_budget_pause(tmp_path, tid, {**pause, "state": "pausing", "park_task": dict(task)})
    workers.PENDING.clear()
    request_cancel(tmp_path, "stopped", reason="owner_cancel", source="task_cancel")

    def busy(_root):
        raise store.StoreBusy("index busy at boot")

    with monkeypatch.context() as m:
        m.setattr(store_hot_facts, "pausing_envelopes", busy)
        assert q.restore_pending_from_snapshot() == 1
    read, writer = bp.budget_pause_row, {"holds": True}

    def contended(root, task_id):
        # "final": the classifying read gets through and the park's own read of the row meets the writer.
        caller = sys._getframe(1).f_code.co_name
        if task_id == "direct" and writer["holds"] and (busy_read == "every" or caller == "park_saved_pause"):
            raise store.StoreBusy("a writer holds the store for this publication")
        return read(root, task_id)

    monkeypatch.setattr(bp, "budget_pause_row", contended)
    first = q.publish_queue_snapshot("pause_row_busy")
    writer["holds"] = False
    second = q.publish_queue_snapshot("pause_row_readable")
    publish_pending_parks(q, "confirm")
    assert {row["id"] for row in q.PENDING} == {"known", "direct"} and not has_active_intent(tmp_path, "direct", strict=True)
    assert [first, second] == [False, True] and q.resume_budget_paused_task("direct")["ok"]
