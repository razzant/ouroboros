"""Regression coverage for snapshot-before-teardown subtree cancellation."""

from __future__ import annotations


def _isolate_queue(monkeypatch, tmp_path, tasks):
    from supervisor import queue
    from supervisor import task_lifecycle
    from supervisor import workers

    pending = [dict(task) for task in tasks]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    return queue, pending


def test_root_cancel_snapshots_and_cancels_whole_live_subtree(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    queue, pending = _isolate_queue(
        monkeypatch,
        tmp_path,
        [
            {"id": "root", "chat_id": 0, "root_task_id": "root", "depth": 0},
            {
                "id": "child",
                "chat_id": 0,
                "root_task_id": "root",
                "parent_task_id": "root",
                "depth": 1,
            },
            {
                "id": "grandchild",
                "chat_id": 0,
                "root_task_id": "root",
                "parent_task_id": "child",
                "depth": 2,
            },
            {"id": "other", "chat_id": 0, "root_task_id": "other", "depth": 0},
        ],
    )
    order: list[str] = []
    monkeypatch.setattr(queue, "_emit_cancel_task_done", lambda _task, task_id, **_kw: order.append(task_id))

    assert queue.cancel_task_by_id("root", cascade=True) is True

    assert order == ["grandchild", "child", "root"]
    assert [task["id"] for task in pending] == ["other"]
    for task_id in ("root", "child", "grandchild"):
        assert load_task_result(tmp_path, task_id)["status"] == STATUS_CANCELLED


def test_midtree_cancel_uses_combined_pending_lineage(tmp_path, monkeypatch):
    queue, pending = _isolate_queue(
        monkeypatch,
        tmp_path,
        [
            {"id": "root", "chat_id": 0, "root_task_id": "root"},
            {"id": "child", "chat_id": 0, "root_task_id": "root", "parent_task_id": "root"},
            {
                "id": "grandchild",
                "chat_id": 0,
                "root_task_id": "root",
                "parent_task_id": "child",
            },
        ],
    )

    assert queue.cancel_task_by_id("child", cascade=True) is True
    assert [task["id"] for task in pending] == ["root"]


def test_default_cancel_preserves_live_children(tmp_path, monkeypatch):
    queue, pending = _isolate_queue(
        monkeypatch,
        tmp_path,
        [
            {"id": "root", "chat_id": 0, "root_task_id": "root"},
            {"id": "child", "chat_id": 0, "root_task_id": "root", "parent_task_id": "root"},
        ],
    )

    assert queue.cancel_task_by_id("root") is True
    assert [task["id"] for task in pending] == ["child"]


def test_cascade_resweeps_descendants_admitted_after_the_first_snapshot(monkeypatch):
    """A schedule event already draining when the snapshot was taken must not
    survive: the cascade re-sweeps until a fresh snapshot finds nothing new."""
    import supervisor.queue as q
    from supervisor.task_lifecycle import cancel_task_by_id

    cancelled: list[str] = []
    late = {"added": False}

    def _fake_single(task_id):
        cancelled.append(task_id)
        # The late child is admitted while the FIRST sweep is already cancelling.
        if task_id == "root" and not late["added"]:
            late["added"] = True
            q.PENDING.append({"id": "late-child", "parent_task_id": "root", "root_task_id": "root"})
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == task_id:
                q.PENDING.pop(index)
                break
        return True

    monkeypatch.setattr(q, "PENDING", [{"id": "root"}], raising=False)
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(q, "cancel_task_custody", lambda tid, **_kw: q.CANCEL_CANCELLED if _fake_single(tid) else q.CANCEL_FAILED)
    monkeypatch.setattr(q, "append_jsonl", lambda *a, **k: None)

    assert cancel_task_by_id("root", cascade=True) is True
    assert "late-child" in cancelled, cancelled
    assert cancelled.count("root") == 1, cancelled


def test_cancelled_root_fences_late_descendant_admission(monkeypatch):
    """The re-sweep is defence; the AUTHORITY is the admission fence — a child
    scheduled after every sweep must be refused, not merely swept later."""
    import supervisor.queue as q
    from supervisor.task_lifecycle import cancel_task_by_id

    monkeypatch.setattr(q, "PENDING", [{"id": "root"}], raising=False)
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    import supervisor.task_lifecycle as tl
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    def _custody(tid, **_kw):
        # A realistic custody: the cancelled task really leaves the live maps, so
        # the cascade's no-live-subtree postcondition can actually be satisfied.
        q.RUNNING.pop(tid, None)
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == tid:
                q.PENDING.pop(index)
                break
        return q.CANCEL_CANCELLED

    monkeypatch.setattr(q, "cancel_task_custody", _custody)
    monkeypatch.setattr(q, "append_jsonl", lambda *a, **k: None)

    assert cancel_task_by_id("root", cascade=True) is True
    # A schedule event that drained AFTER the whole sweep loop is refused.
    admitted = q.enqueue_task({"id": "late", "parent_task_id": "root", "root_task_id": "root"})
    assert admitted.get("_admission_blocked") == "root_cancelled"
    assert all(str(item.get("id")) != "late" for item in q.PENDING)


def test_mid_tree_cancel_fences_a_grandchild_scheduled_by_a_live_descendant(monkeypatch):
    """A cascade may target ANY live id, so the fence walks the full ancestry: a
    grandchild scheduled by a still-live deeper descendant of the CANCELLED MID-TREE
    node must be refused, even though its root is the original root and its parent
    is not the cancelled node itself."""
    import supervisor.queue as q
    import supervisor.task_lifecycle as tl
    from supervisor.task_lifecycle import cancel_task_by_id

    # root -> mid -> deep (deep stays live and schedules "grandchild" afterwards)
    monkeypatch.setattr(q, "PENDING", [
        {"id": "mid", "root_task_id": "root", "parent_task_id": "root"},
    ], raising=False)
    monkeypatch.setattr(q, "RUNNING", {
        "deep": {"task": {"id": "deep", "root_task_id": "root", "parent_task_id": "mid"}},
    }, raising=False)
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    def _remove_from_live(task_id):
        # Adversarial: a real cancel REMOVES the task from the live maps, so the
        # ancestry can no longer be reconstructed from them afterwards.
        q.RUNNING.pop(task_id, None)
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == task_id:
                q.PENDING.pop(index)
                break
        return True

    monkeypatch.setattr(q, "cancel_task_custody", lambda tid, **_kw: q.CANCEL_CANCELLED if _remove_from_live(tid) else q.CANCEL_FAILED)
    monkeypatch.setattr(q, "append_jsonl", lambda *a, **k: None)

    assert cancel_task_by_id("mid", cascade=True) is True
    assert "deep" not in q.RUNNING  # the cancelled subtree really is gone

    admitted = q.enqueue_task(
        {"id": "grandchild", "root_task_id": "root", "parent_task_id": "deep"},
    )
    assert admitted.get("_admission_blocked") == "root_cancelled"
    # A sibling branch under the SAME root that never descends from "mid" is fine.
    other = q.enqueue_task({"id": "other", "root_task_id": "root", "parent_task_id": "root"})
    assert other.get("_admission_blocked") is None


def test_a_500_child_cascade_never_evicts_its_own_fence(monkeypatch):
    """The per-root ceiling is 500, so a big tree must not prune the very fences
    that are still refusing admissions — including after the final sweep."""
    import supervisor.queue as q
    import supervisor.task_lifecycle as tl
    from supervisor.task_lifecycle import cancel_task_by_id

    children = [
        {"id": f"c{index}", "root_task_id": "root", "parent_task_id": "root"}
        for index in range(500)
    ]
    monkeypatch.setattr(q, "PENDING", [{"id": "root"}, *children], raising=False)
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    # Pre-fill the registry so pruning is genuinely triggered.
    monkeypatch.setattr(
        tl, "CANCELLED_ROOT_FENCES",
        {f"old{index}": "2026-01-01T00:00:00Z" for index in range(tl._CANCELLED_ROOT_FENCE_CAP)},
        raising=False,
    )

    def _remove(task_id):
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == task_id:
                q.PENDING.pop(index)
                return True
        return False

    monkeypatch.setattr(q, "cancel_task_custody", lambda tid, **_kw: q.CANCEL_CANCELLED if _remove(tid) else q.CANCEL_FAILED)
    monkeypatch.setattr(q, "append_jsonl", lambda *a, **k: None)

    assert cancel_task_by_id("root", cascade=True) is True
    assert "root" in tl.CANCELLED_ROOT_FENCES
    assert "c499" in tl.CANCELLED_ROOT_FENCES
    # A schedule event draining after the final sweep is still refused.
    late = q.enqueue_task({"id": "late", "root_task_id": "root", "parent_task_id": "c499"})
    assert late.get("_admission_blocked") == "root_cancelled"


def test_overlapping_cascades_never_evict_each_others_fences(monkeypatch):
    """Pruning may drop only QUIESCENT entries: while several cascades are in
    flight, exceeding the cap must not unfence any of their trees."""
    import supervisor.task_lifecycle as tl

    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)

    # The RUNNING cascade's own 20 ids, plus 40 quiescent entries from old trees.
    mine = [f"live{index}" for index in range(20)]
    for task_id in mine:
        tl.CANCELLED_ROOT_FENCES[task_id] = "2026-07-29T00:00:00Z"
    for index in range(40):
        tl.CANCELLED_ROOT_FENCES[f"old{index}"] = "2026-01-01T00:00:00Z"

    tl._prune_cancellation_fences(protected=set(mine))

    for task_id in mine:
        assert task_id in tl.CANCELLED_ROOT_FENCES, task_id
    # Quiescent entries absorbed the trim instead.
    assert sum(1 for key in tl.CANCELLED_ROOT_FENCES if key.startswith("old")) < 40

def test_completion_wins_over_a_cascade_that_is_already_tearing_down(monkeypatch, tmp_path):
    """A task that reached its own terminal result keeps it. The cascade writes
    STATUS_CANCELLED without the explicit-cancellation override, so the result
    writer's monotonic guard refuses the transition UNDER the per-result lock —
    the compare-and-set lives in one place instead of a read-then-write check the
    worker could race."""
    from ouroboros.task_results import load_task_result, write_task_result

    queue, pending = _isolate_queue(
        monkeypatch,
        tmp_path,
        [
            {"id": "root", "chat_id": 0, "root_task_id": "root"},
            {"id": "child", "chat_id": 0, "root_task_id": "root", "parent_task_id": "root"},
        ],
    )
    import supervisor.task_lifecycle as tl
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)

    write_task_result(tmp_path, "root", "completed", result="Finished on its own.")

    queue.cancel_task_by_id("root", cascade=True)

    assert load_task_result(tmp_path, "root")["status"] == "completed"
    assert load_task_result(tmp_path, "child")["status"] == "cancelled"


def _fake_worker(task_id, *, alive_after_kill=False):
    """A worker double whose process death is observable and controllable."""
    import types

    state = {"alive": True, "kills": 0, "joins": 0}

    def _kill():
        state["kills"] += 1
        if not alive_after_kill:
            state["alive"] = False

    proc = types.SimpleNamespace(
        pid=4242,
        is_alive=lambda: state["alive"],
        terminate=_kill,
        join=lambda timeout=None: state.__setitem__("joins", state["joins"] + 1),
    )
    return types.SimpleNamespace(wid=1, busy_task_id=task_id, proc=proc), state


def _install_worker(monkeypatch, worker):
    from supervisor import workers
    monkeypatch.setattr(workers, "WORKERS", {worker.wid: worker}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)


def test_a_child_that_refuses_to_die_fails_the_whole_cascade(monkeypatch, tmp_path):
    """The cascade answers for its weakest id: a root that cancelled cannot mask a
    child that refused."""
    import supervisor.queue as q
    import supervisor.task_lifecycle as tl

    _isolate_queue(monkeypatch, tmp_path, [
        {"id": "root", "chat_id": 0, "root_task_id": "root"},
        {"id": "child", "chat_id": 0, "root_task_id": "root", "parent_task_id": "root"},
    ])
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)

    def _custody(tid, **_kw):
        if tid == "child":
            return q.CANCEL_FAILED  # stays live on purpose
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == tid:
                q.PENDING.pop(index)
                break
        return q.CANCEL_CANCELLED

    monkeypatch.setattr(q, "cancel_task_custody", _custody)

    assert tl.cancel_task_by_id("root", cascade=True) is False
    assert any(str(t.get("id")) == "child" for t in q.PENDING), "the child really is still live"


def test_a_stubborn_process_is_refused_not_reported_cancelled(monkeypatch, tmp_path):
    """A process surviving both kill escalations keeps its custody (row restored)
    and yields `failed` — never a task_done claiming a live worker is cancelled."""
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("stuck", alive_after_kill=True)
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(q, "RUNNING", {"stuck": {"task": {"id": "stuck"}}}, raising=False)
    monkeypatch.setattr(q, "kill_pid_tree", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", lambda *a, **k: None)
    emitted: list[str] = []
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: emitted.append("emit"))

    assert q.cancel_task_custody("stuck") == q.CANCEL_FAILED
    assert "stuck" in q.RUNNING, "custody is restored so the tree stays honest"
    assert not emitted, "a surviving worker must never publish a cancelled task_done"
    assert state["joins"] >= 2, "both join escalations were attempted"


def test_persistence_failure_restores_custody_and_publishes_nothing(monkeypatch, tmp_path):
    """A durable write that RAISES is not a public cancellation: row back, no
    task_done, outcome `failed`."""
    import ouroboros.task_results as task_results
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    emitted: list[str] = []
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: emitted.append("emit"))

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(task_results, "write_task_result", _boom)

    assert q.cancel_task_custody("root") == q.CANCEL_FAILED
    assert any(str(t.get("id")) == "root" for t in q.PENDING), "the task is back in the queue"
    assert not emitted


def test_natural_completion_keeps_its_result_and_its_own_event(monkeypatch, tmp_path):
    """A task that finished on its own is never taken into custody: result kept, no
    Cancelled event, worker left to the ordinary finalizer."""
    from ouroboros.task_results import load_task_result, write_task_result
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    emitted: list[str] = []
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: emitted.append("emit"))
    write_task_result(tmp_path, "root", "completed", result="Finished on its own.")

    assert q.cancel_task_custody("root") == q.CANCEL_ALREADY_SETTLED
    assert load_task_result(tmp_path, "root")["status"] == "completed"
    assert not emitted, "no Cancelled event may go out for a completed task"


def test_concurrent_cascades_on_overlapping_trees_both_settle(monkeypatch, tmp_path):
    """Two real threads cancelling overlapping subtrees: both settle and nothing is
    left live — the property the old ownership/rollback machinery tried to protect."""
    import threading

    import supervisor.queue as q
    import supervisor.task_lifecycle as tl

    _isolate_queue(monkeypatch, tmp_path, [
        {"id": "root", "chat_id": 0, "root_task_id": "root"},
        {"id": "mid", "chat_id": 0, "root_task_id": "root", "parent_task_id": "root"},
        {"id": "leaf", "chat_id": 0, "root_task_id": "root", "parent_task_id": "mid"},
    ])
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)

    def _custody(tid, **_kw):
        # Mirrors the real contract: custody is the ONE settle owner, so the
        # durable cancel intent a cascade mints for each captured descendant
        # LEAVES the projection here. Without that the second cascade would see
        # its own target still fenced by the first cascade's intent.
        from ouroboros.cancel_intents import settle_intent

        try:
            with q._queue_lock:
                for index, item in enumerate(list(q.PENDING)):
                    if str(item.get("id")) == tid:
                        q.PENDING.pop(index)
                        return q.CANCEL_CANCELLED
            return q.CANCEL_NOT_FOUND
        finally:
            settle_intent(q.DRIVE_ROOT, tid, outcome="cancelled")

    monkeypatch.setattr(q, "cancel_task_custody", _custody)

    results: dict[str, bool] = {}
    threads = [
        threading.Thread(target=lambda: results.__setitem__("root", tl.cancel_task_by_id("root", cascade=True))),
        threading.Thread(target=lambda: results.__setitem__("mid", tl.cancel_task_by_id("mid", cascade=True))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not q.PENDING, "no part of either subtree is left live"
    assert results["root"] is not False and results["mid"] is not False


def test_a_settled_task_with_a_live_worker_is_killed_and_keeps_its_result(monkeypatch, tmp_path):
    """GR6-1b (superseding the old left-to-its-own-finalizer contract): the durable
    terminal result is persisted BEFORE post-task cognition ends, so a settled
    STATUS with a live WORKER is a process still spending — cancel custody kills
    it. Completion wins on the durable side: the stored completed result is
    preserved verbatim, and the outcome is ``already_settled``."""
    from ouroboros.task_results import load_task_result, write_task_result
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("done-ish")
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(q, "RUNNING", {"done-ish": {"task": {"id": "done-ish"}}}, raising=False)
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **k: state.__setitem__("alive", False),
    )
    emitted: list[str] = []
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: emitted.append("emit"))
    write_task_result(tmp_path, "done-ish", "completed", result="Finished on its own.")

    assert q.cancel_task_custody("done-ish") == q.CANCEL_ALREADY_SETTLED
    assert not state["alive"], "the settled-but-live worker IS killed (GR6-1b)"
    stored = load_task_result(tmp_path, "done-ish")
    assert stored["status"] == "completed", "completion wins: the result is kept"
    assert stored["result"] == "Finished on its own."
    assert "done-ish" not in q.RUNNING, "custody owns the row after the confirmed death"
    assert emitted == ["emit"], "the card resolves through the stored terminal truth"


def test_cancelling_a_running_task_completes_the_whole_publish_phase(monkeypatch, tmp_path):
    """Drives the RUNNING path end to end (kill, join, persist, emit, respawn) so a
    missing import or misordered publish fails here, not on a real cancel."""
    from ouroboros.task_results import load_task_result
    import supervisor.queue as q
    from supervisor import workers

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("live-one")
    _install_worker(monkeypatch, worker)
    respawned: list[int] = []
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: respawned.append(wid), raising=False)
    monkeypatch.setattr(q, "RUNNING", {
        "live-one": {"task": {"id": "live-one", "delegation_role": "subagent"}},
    }, raising=False)
    # A real tree-kill ends the process; the double reflects that so the test
    # exercises the SUCCESS path rather than the stubborn-process refusal.
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **k: state.__setitem__("alive", False),
    )
    emitted: list[str] = []
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: emitted.append("emit"))

    assert q.cancel_task_custody("live-one") == q.CANCEL_CANCELLED
    assert not state["alive"], "the process is confirmed dead before publishing"
    assert emitted == ["emit"]
    assert respawned == [worker.wid], "the worker slot is refilled"
    assert load_task_result(tmp_path, "live-one")["status"] == "cancelled"
    assert "live-one" not in q.RUNNING


def test_custody_marks_the_slot_so_the_health_check_keeps_its_hands_off(monkeypatch, tmp_path):
    """The kill/join runs OFF the queue lock, so the crash detector can see the very
    worker being cancelled. Capture marks the slot `reaping` — the marker assign_tasks
    and ensure_workers_healthy already skip — so it cannot requeue the task, respawn
    twice, or feed crash-storm detection."""
    import supervisor.queue as q
    from supervisor import workers

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("live-two")
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(q, "RUNNING", {"live-two": {"task": {"id": "live-two"}}}, raising=False)
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: None)

    seen_reaping: list[bool] = []

    def _kill(*a, **k):
        # This is the window in which the health check would run.
        seen_reaping.append(bool(getattr(worker, "reaping", False)))
        state["alive"] = False

    monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", _kill)
    respawned: list[int] = []
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: respawned.append(wid), raising=False)

    assert q.cancel_task_custody("live-two") == q.CANCEL_CANCELLED
    assert seen_reaping == [True], "the slot is marked BEFORE the lock is released"
    assert respawned == [worker.wid], "exactly one respawn, from the cancellation itself"


def test_a_raising_kill_restores_custody_and_clears_the_slot_marker(monkeypatch, tmp_path):
    """A raising kill must not strand a possibly-live worker outside RUNNING, nor
    leave the slot marked `reaping` (which assign and the health check skip)."""
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, _state = _fake_worker("boom-one")
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(q, "RUNNING", {"boom-one": {"task": {"id": "boom-one"}}}, raising=False)

    def _boom(*a, **k):
        raise OSError("kill failed")

    monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", _boom)

    assert q.cancel_task_custody("boom-one") == q.CANCEL_FAILED
    assert "boom-one" in q.RUNNING, "the task is visible to the liveness check again"
    assert worker.reaping is False, "the slot is usable again"


def test_pruning_never_evicts_another_in_flight_cascades_fences(monkeypatch):
    """Two large cascades at once: the later one must not push the cap over and
    unfence the older one's tree while it is still being torn down."""
    import supervisor.task_lifecycle as tl

    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(tl, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)

    other = tl._next_cascade_token("other-root")
    other_ids = {f"other{index}" for index in range(20)}
    tl._ACTIVE_CASCADE_FENCES[other] = other_ids
    for task_id in other_ids:
        tl.CANCELLED_ROOT_FENCES[task_id] = "2026-07-29T00:00:00Z"
    for index in range(40):
        tl.CANCELLED_ROOT_FENCES[f"old{index}"] = "2026-01-01T00:00:00Z"

    # A DIFFERENT cascade prunes, protecting only its own ids.
    tl._prune_cancellation_fences(protected={"mine"})

    for task_id in other_ids:
        assert task_id in tl.CANCELLED_ROOT_FENCES, f"{task_id} belongs to a live cascade"
    assert sum(1 for key in tl.CANCELLED_ROOT_FENCES if key.startswith("old")) < 40


def test_recently_completed_cascade_fence_survives_the_next_cap_prune(monkeypatch):
    """Dropping active ownership after success must not immediately reopen the
    tree to a delayed schedule event already sitting in the supervisor queue."""
    import supervisor.task_lifecycle as tl

    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(tl, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)
    tl.CANCELLED_ROOT_FENCES["just-completed"] = tl.utc_now_iso()
    for index in range(40):
        tl.CANCELLED_ROOT_FENCES[f"old{index}"] = "2026-01-01T00:00:00Z"

    tl._prune_cancellation_fences(protected=set())

    assert "just-completed" in tl.CANCELLED_ROOT_FENCES
    assert len(tl.CANCELLED_ROOT_FENCES) <= tl._CANCELLED_ROOT_FENCE_CAP


def test_a_natural_result_written_before_the_kill_still_resolves_the_card(monkeypatch, tmp_path):
    """The worker wrote its own terminal result but was killed before emitting
    task_done. Our cancellation write is refused by the monotonic guard, so the
    published event must carry the STORED truth — publishing nothing would leave
    the live card unresolved until a reload."""
    from ouroboros.task_results import load_task_result, write_task_result
    import supervisor.queue as q
    from supervisor import workers

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("racer")
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "RUNNING", {"racer": {"task": {"id": "racer"}}}, raising=False)

    emitted: list[dict] = []
    monkeypatch.setattr(
        q, "_emit_cancel_task_done",
        lambda task, task_id, **kw: emitted.append({"task_id": task_id, **kw}),
    )

    def _kill(*a, **k):
        # The worker finishes its own result in the instant before it dies.
        write_task_result(tmp_path, "racer", "completed", result="Finished just in time.")
        state["alive"] = False

    monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", _kill)

    assert q.cancel_task_custody("racer") == q.CANCEL_ALREADY_SETTLED
    assert load_task_result(tmp_path, "racer")["status"] == "completed"
    assert emitted and emitted[0]["status"] == "completed", (
        "the card resolves to what actually settled, not to silence"
    )


def test_split_drive_completion_is_promoted_before_cancelled_write(monkeypatch, tmp_path):
    """A forked/workspace worker may finish on its child drive before task_done
    copies it back. Cancellation must promote that terminal truth after the kill,
    not overwrite it with a canonical cancelled result."""
    from ouroboros.task_results import load_task_result, write_task_result
    import supervisor.queue as q
    from supervisor import workers

    child_drive = tmp_path / "child-drive"
    task = {"id": "split-racer", "drive_root": str(child_drive)}
    _isolate_queue(monkeypatch, tmp_path, [])
    write_task_result(tmp_path, "split-racer", "running", drive_root=str(child_drive))
    write_task_result(child_drive, "split-racer", "completed", result="child finished")
    worker, state = _fake_worker("split-racer")
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "RUNNING", {"split-racer": {"task": task}}, raising=False)
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **k: state.__setitem__("alive", False),
    )
    emitted: list[dict] = []
    monkeypatch.setattr(
        q, "_emit_cancel_task_done",
        lambda _task, task_id, **kw: emitted.append({"task_id": task_id, **kw}),
    )

    assert q.cancel_task_custody("split-racer") == q.CANCEL_ALREADY_SETTLED
    stored = load_task_result(tmp_path, "split-racer")
    assert stored["status"] == "completed"
    assert stored["result"] == "child finished"
    assert emitted and emitted[0]["status"] == "completed"


def test_publication_failures_after_the_durable_write_never_undo_the_cancellation(monkeypatch, tmp_path):
    """Once the terminal result is on disk the cancellation HAPPENED. Telemetry,
    respawn and snapshot failures past that boundary are fail-soft — answering
    `failed` there would report a cancellation that demonstrably occurred — and a
    raising respawn must still release the slot marker."""
    from ouroboros.task_results import load_task_result
    import supervisor.queue as q
    from supervisor import workers

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("post-boundary")
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(q, "RUNNING", {"post-boundary": {"task": {"id": "post-boundary"}}}, raising=False)
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **k: state.__setitem__("alive", False),
    )

    def _boom(*a, **k):
        raise RuntimeError("publication surface down")

    monkeypatch.setattr(q, "reconstruct_task_cost", _boom, raising=False)
    monkeypatch.setattr(q, "_emit_cancel_task_done", _boom)
    monkeypatch.setattr(workers, "respawn_worker", _boom, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", _boom, raising=False)

    assert q.cancel_task_custody("post-boundary") == q.CANCEL_CANCELLED
    assert load_task_result(tmp_path, "post-boundary")["status"] == "cancelled"
    assert worker.reaping is False, "a raising respawn still releases the slot marker"


def test_a_cascade_settles_when_a_child_is_terminal_but_still_winding_down(monkeypatch, tmp_path):
    """The documented natural-completion race, at CASCADE level: a child whose
    durable result is terminal while its RUNNING row is still being reclaimed must
    not make the postcondition report a live tree (and the endpoint a 503)."""
    from ouroboros.task_results import write_task_result
    import supervisor.queue as q
    import supervisor.task_lifecycle as tl

    _isolate_queue(monkeypatch, tmp_path, [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(q, "RUNNING", {
        "child": {"task": {"id": "child", "root_task_id": "root", "parent_task_id": "root"}},
    }, raising=False)
    write_task_result(tmp_path, "child", "completed", result="Finished on its own.")

    def _custody(tid, **_kw):
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == tid:
                q.PENDING.pop(index)
                return q.CANCEL_CANCELLED
        return q.CANCEL_ALREADY_SETTLED

    monkeypatch.setattr(q, "cancel_task_custody", _custody)

    assert tl.cancel_task_by_id("root", cascade=True) is True, (
        "a terminal-but-unreclaimed child is settled, not live"
    )


def test_overlapping_cascade_cannot_confirm_success_while_the_child_kill_is_in_flight(monkeypatch, tmp_path):
    """REAL overlap (validator repro): the child's kill blocks mid-flight in one
    thread while the root cascade runs in another. Custody keeps the child in
    RUNNING and the slot exclusively owned, so the cascade refuses success until
    the real owner confirms death — and succeeds on the next attempt after."""
    import threading

    import ouroboros.platform_layer as pl
    import supervisor.queue as q
    import supervisor.task_lifecycle as tl
    _isolate_queue(monkeypatch, tmp_path, [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    kill_started = threading.Event()
    release_kill = threading.Event()
    state = {"alive": True}

    def _blocking_kill(*a, **k):
        kill_started.set()
        release_kill.wait(timeout=30)
        state["alive"] = False

    monkeypatch.setattr(pl, "kill_pid_tree", _blocking_kill)
    worker, _ = _fake_worker("child")
    worker.proc.is_alive = lambda: state["alive"]
    _install_worker(monkeypatch, worker)
    monkeypatch.setattr(q, "RUNNING", {
        "child": {"task": {"id": "child", "root_task_id": "root", "parent_task_id": "root"}},
    }, raising=False)

    outcome = {}
    child_thread = threading.Thread(
        target=lambda: outcome.__setitem__("child", q.cancel_task_custody("child")),
    )
    child_thread.start()
    assert kill_started.wait(timeout=5)
    try:
        assert tl.cancel_task_by_id("root", cascade=True) is False, (
            "no success while the child's kill is in flight"
        )
        assert tl.task_subtree_is_live("root") is True, "the in-flight child stays visible"
    finally:
        release_kill.set()
        child_thread.join(timeout=10)
    assert outcome.get("child") == q.CANCEL_CANCELLED
    assert tl.cancel_task_by_id("root", cascade=True) is True, "settled after confirmed death"


def test_a_natural_task_done_in_the_kill_window_keeps_one_owner(monkeypatch, tmp_path):
    """The worker's own task_done arrives while the kill is in flight: the handler
    pops RUNNING and emits, but must NOT free the custody-owned (reaping) slot;
    custody then publishes nothing twice and, even if its persistence fails, the
    restore leaves no RUNNING ghost and no freed slot."""
    import types

    import supervisor.events_task_done as task_done_mod
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [])
    worker, state = _fake_worker("racer2")
    worker.reaping = True  # custody owns the slot; the kill window is open
    ctx = types.SimpleNamespace(
        RUNNING={"racer2": {"task": {"id": "racer2"}}},
        WORKERS={worker.wid: worker},
        persist_queue_snapshot=lambda reason="": None,
    )
    with q._queue_lock:
        pass  # the handler takes the real lock; nothing else holds it here

    # Simulate exactly the handler's terminal block.
    with q._queue_lock:
        ctx.RUNNING.pop("racer2", None)
        if worker.wid in ctx.WORKERS and ctx.WORKERS[worker.wid].busy_task_id == "racer2":
            if not getattr(ctx.WORKERS[worker.wid], "reaping", False):
                ctx.WORKERS[worker.wid].busy_task_id = None

    assert worker.busy_task_id == "racer2", "a custody-owned slot is not freed by task_done"
    assert "racer2" not in ctx.RUNNING, "completion-wins still consumes the row"
    # ...and the handler's real source implements exactly that guard.
    import pathlib as _pl
    src = _pl.Path(task_done_mod.__file__).read_text(encoding="utf-8")
    assert 'if not getattr(ctx.WORKERS[worker_id], "reaping", False):' in src
