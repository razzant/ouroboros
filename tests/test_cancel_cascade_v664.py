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


def _request(drive, task_id, *, cascade=False):
    """Every cancel ingress records the durable intent first (invariant 14); a cascade
    records it on the root with the cascade scope."""
    from ouroboros.cancel_intents import SCOPE_CASCADE, request_cancel

    request_cancel(drive, task_id, scope=SCOPE_CASCADE if cascade else "", allow_settled_target=True)


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

    assert cancel_task_by_id("root") is True
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

    _request(tmp_path, "root", cascade=True)
    queue.cancel_task_by_id("root")

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


def test_persistence_failure_restores_custody_and_publishes_nothing(monkeypatch, tmp_path):
    """A durable write that RAISES is not a public cancellation: row back, no
    task_done, outcome `failed`."""
    import ouroboros.task_results as task_results
    import supervisor.queue as q

    _isolate_queue(monkeypatch, tmp_path, [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    emitted: list[str] = []
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: emitted.append("emit"))
    _request(tmp_path, "root")

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(task_results, "write_task_result", _boom)

    assert q.cancel_task_custody("root") == q.CANCEL_FAILED
    assert any(str(t.get("id")) == "root" for t in q.PENDING), "the task is back in the queue"
    assert not emitted


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
    _request(tmp_path, "post-boundary")

    assert q.cancel_task_custody("post-boundary") == q.CANCEL_CANCELLED
    assert load_task_result(tmp_path, "post-boundary")["status"] == "cancelled"
    assert worker.reaping is False, "a raising respawn still releases the slot marker"
