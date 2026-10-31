"""Pause-generation, persistence-failure and control regressions at real consumers."""

import copy
import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx, _parked, _supervisor_ctx
from tests.test_budget_pause_holds import _fenced_member, _idle_worker, _source_file

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("saved_pause,selected", [(True, True), (False, True), (True, False), (False, False)])
def test_new_saved_pause_does_not_rearm_historic_selected_panic(tmp_path, monkeypatch, saved_pause, selected):
    from supervisor.events_budget import BUDGET_HOLD_KEY, HOLD_PANIC
    from ouroboros import budget_pause
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    task, _ = _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    if not saved_pause:
        task.pop("_budget_pause")
    if selected:
        task["_resume_selection"] = {"selected_by": "owner", "fence_id": "older-fence", "root_task_id": "root"}
    else:
        task[BUDGET_HOLD_KEY] = {"reason": HOLD_PANIC, "root_task_id": "root"}
    before = copy.deepcopy(task)
    saved = budget_pause.budget_pause_row(tmp_path, "root")
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert not sent
    assert budget_pause.budget_pause_row(tmp_path, "root") == saved
    if saved_pause or not selected:
        assert task.get(BUDGET_HOLD_KEY) == before.get(BUDGET_HOLD_KEY)
        assert task.get("_resume_selection") == before.get("_resume_selection")
        assert task.get("_budget_pause") == before.get("_budget_pause")
    else:
        assert BUDGET_HOLD_KEY not in task and "_resume_selection" not in task


@pytest.mark.parametrize("late_carrier", ["none", "before_assignment", "before_root_resume"])
def test_new_root_pause_invalidates_child_selection_until_selected_again(tmp_path, monkeypatch, late_carrier):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    root, _ = _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    child, _ = _parked(tmp_path, monkeypatch, task_id="child", root_task_id="root")
    sibling = _fenced_member(workers, "sibling", "root")
    first = queue.resume_budget_paused_task("root")
    assert first["ok"] and queue.resume_budget_paused_task("child", selected_by="root")["ok"]
    assert queue.resume_budget_paused_task("sibling", selected_by="root")["ok"]
    old_child = copy.deepcopy(child)
    workers.PENDING.remove(root)  # root ran, then paused for a second time
    root2, _ = _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    assert "_budget_pause_resume" not in child and "_budget_pause" in child
    assert "_budget_pause_hold" in sibling and "_resume_selection" not in sibling
    if late_carrier != "none":
        child.clear()
        child.update(old_child)  # a stale queue carrier cannot bypass either consumer
    sent = []
    _idle_worker(workers, sent)
    if late_carrier != "before_root_resume":
        workers.assign_tasks()
        assert sent == []
    second = queue.resume_budget_paused_task("root")
    assert second["ok"] and second["grant_id"] != first["grant_id"]
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["root"]
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["root"]  # root Resume only made children eligible
    assert queue.resume_budget_paused_task("child", selected_by="root")["ok"]
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["root", "child"]
    assert sent[-1]["_budget_pause_resume"]["root_grant_id"] == second["grant_id"]
    assert "_budget_pause_hold" in sibling and "_resume_selection" not in sibling


def test_assignment_rechecks_root_grant_even_when_no_fence_is_left(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    root, _ = _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    child, _ = _parked(tmp_path, monkeypatch, task_id="child", root_task_id="root")
    assert queue.resume_budget_paused_task("root")["ok"]
    assert queue.resume_budget_paused_task("child", selected_by="root")["ok"]
    workers.PENDING.remove(root)
    row = budget_pause.budget_pause_row(tmp_path, "root")
    row["grant"]["grant_id"] = "new-root-grant"
    row["resume_generation"] += 1
    budget_pause.set_budget_pause(tmp_path, "root", row)
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == [] and "_budget_pause_resume" not in child
    assert queue.resume_budget_paused_task("child", selected_by="root")["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["child"]


@pytest.mark.parametrize("marker", [False, True])
def test_legacy_root_resume_selects_only_root_then_one_child(tmp_path, monkeypatch, marker):
    from supervisor.events_budget import _set_root_budget_pause_locked

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    root = _fenced_member(workers, "root", "root")
    root.pop("parent_task_id")
    _fenced_member(workers, "child", "root")
    sibling = _fenced_member(workers, "sibling", "root")
    fence = _set_root_budget_pause_locked("root", {})
    if marker:
        root["_budget_pause"] = {**fence, "physical_calls": 0, "replay_safe": True}
    assert queue.resume_budget_paused_task("root")["ok"]
    assert queue.BUDGET_ROOT_FENCES["root"] == fence
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root"]
    assert queue.resume_budget_paused_task("child", selected_by="root")["ok"]
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root", "child"] and sibling in workers.PENDING
    # A subsequent pause creates a new fence even though the legacy latch stayed up.
    assert _set_root_budget_pause_locked("root", {})["fence_id"] != fence["fence_id"]


def _dead_granted_task(tmp_path, monkeypatch):
    from supervisor import worker_health

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id="dead")
    assert queue.resume_budget_paused_task("dead")["ok"]
    _idle_worker(workers, [])
    workers.assign_tasks()
    monkeypatch.setattr(worker_health, "_dead_job_is_current", lambda _job: True)
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_k: None)
    job = {"worker": workers.WORKERS[0], "worker_id": 0, "task_id": "dead", "task": task,
           "meta": workers.RUNNING["dead"], "exitcode": 1, "drive_root": str(tmp_path)}
    return queue, workers, job, row


@pytest.mark.parametrize("snapshot_ok", [True, False])
def test_dead_unconsumed_grant_keeps_exact_hold_when_revocation_stores_fail(tmp_path, monkeypatch, snapshot_ok):
    from ouroboros import budget_pause
    from ouroboros.work_stop import continuation_refusal
    from supervisor import events_budget, worker_health

    queue, workers, job, row = _dead_granted_task(tmp_path, monkeypatch)
    original_grant = copy.deepcopy(job["task"]["_budget_pause_resume"])
    writer, snapshot = budget_pause.set_budget_pause, queue.persist_queue_snapshot
    monkeypatch.setattr(budget_pause, "set_budget_pause", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(events_budget, "write_task_result", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    if not snapshot_ok:
        monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_k: False)
    terminals = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *_a, **_k: terminals.append(True))
    worker_health._recover_crashed_task_without_terminal(job, queue)
    assert terminals == [] and not workers.RUNNING
    held = workers.PENDING[0]
    hold = continuation_refusal(held)
    assert hold["grant_id"] == original_grant["grant_id"] and hold["pause_id"] == row["pause_id"]
    assert hold["snapshot_persisted"] is snapshot_ok and "_budget_pause_resume" not in held
    assert held["_budget_pause"]["checkpoint"]["source_ref"] == row["source_ref"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == []
    monkeypatch.setattr(budget_pause, "set_budget_pause", writer)
    monkeypatch.setattr(queue, "persist_queue_snapshot", snapshot)
    assert queue.resume_budget_paused_task("dead")["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["dead"]


def test_consumed_grant_without_new_working_source_is_never_reexecuted(tmp_path, monkeypatch):
    from ouroboros import budget_pause, delegate_recovery
    from ouroboros.task_results import load_task_result
    from supervisor import worker_health

    queue, workers, job, _ = _dead_granted_task(tmp_path, monkeypatch)
    ctx, _ = _loop_ctx(tmp_path, "dead")
    handoff = job["task"]["_budget_pause_resume"]
    # The real consumption owner marks the grant before any subsequent effect.
    saved = budget_pause.load_budget_pause(ctx, handoff)
    from ouroboros import owner_wait

    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, [], {}, {}, set(),
                                    budget_remaining_usd=5.0)
    from ouroboros.working_checkpoint import discard
    discard(tmp_path, "dead")  # no current source: old Resume bytes are never a retry source
    monkeypatch.setattr(delegate_recovery, "reconcile_unrecoverable_task", lambda *_a, **_k: None)
    monkeypatch.setattr(queue, "enqueue_task", lambda *_a, **_k: pytest.fail("completed effects replayed"))
    terminals = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *_a, **kw: terminals.append(kw))
    worker_health._recover_crashed_task_without_terminal(job, queue)
    assert not workers.PENDING and not workers.RUNNING and len(terminals) == 1
    assert load_task_result(tmp_path, "dead", strict=True)["status"] == "failed"
    assert budget_pause.budget_pause_row(tmp_path, "dead")["state"] == budget_pause.STATE_RESUMED


@pytest.mark.parametrize("fences", ["budget", "acceptance", "both"])
@pytest.mark.parametrize("source_state", ["intact", "missing", "corrupt"])
def test_invalid_fences_retain_exact_source_locators(tmp_path, monkeypatch, fences, source_state):
    from ouroboros.work_stop import continuation_refusal

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    task, row = _parked(tmp_path, monkeypatch, task_id="saved")
    source = _source_file(tmp_path, "saved", row)
    if source_state == "missing":
        source.unlink()
    elif source_state == "corrupt":
        source.write_text("broken checkpoint")
    queue.persist_queue_snapshot(reason="test")
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    for field, name in (("budget_root_fences", "budget"), ("acceptance_fences", "acceptance")):
        if fences in {name, "both"}:
            snapshot[field] = [{"status": "active"}]
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snapshot))
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    held = workers.PENDING[0]
    assert held["_budget_pause"]["checkpoint"] == task["_budget_pause"]["checkpoint"]
    assert continuation_refusal(held) and held["_budget_pause"]["checkpoint"]["source_ref"] == row["source_ref"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == []
    if source_state != "intact":
        assert not queue.resume_budget_paused_task("saved")["ok"]


@pytest.mark.parametrize("surface,key", [("handoff", "pause_id"), ("handoff", "grant_id"),
                                          ("row", "pause_id"), ("grant", "grant_id")])
def test_empty_resume_identity_cannot_match_or_overwrite(tmp_path, monkeypatch, surface, key):
    from ouroboros import budget_pause
    from ouroboros import task_results
    from ouroboros.work_stop import continuation_refusal
    from supervisor.budget_resume import revoke_exact_budget_resume

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    task, _ = _parked(tmp_path, monkeypatch, task_id="malformed")
    assert queue.resume_budget_paused_task("malformed")["ok"]
    row = budget_pause.budget_pause_row(tmp_path, "malformed")
    if surface == "handoff":
        task["_budget_pause_resume"][key] = ""
    else:
        (row if surface == "row" else row["grant"])[key] = ""
    monkeypatch.setattr(budget_pause, "budget_pause_row", lambda *_a: copy.deepcopy(row))
    writes = []
    monkeypatch.setattr(budget_pause, "set_budget_pause", lambda *_a, **_k: writes.append(True))
    assert revoke_exact_budget_resume(task, "restart_before_dispatch") is False
    assert not writes and "_budget_pause_resume" not in task and continuation_refusal(task)
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == [] and not writes
    if surface != "handoff":
        # Resume independently reads the durable result, not the revoker's cached row.
        read = task_results.load_task_result
        monkeypatch.setattr(task_results, "load_task_result", lambda root, tid, **kw:
            {**read(root, tid, **kw), "budget_pause": row} if tid == "malformed" else read(root, tid, **kw))
        assert not queue.resume_budget_paused_task("malformed")["ok"] and not writes
    else:
        # A damaged handoff cannot permanently veto a fresh, valid durable source.
        assert queue.resume_budget_paused_task("malformed")["ok"] and len(writes) == 2


def test_resume_tool_selects_root_grandchild_and_parent_child_but_no_other_tree(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.tools import control, join_ledger
    from supervisor.events_budget import _handle_budget_resume_child

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    _parked(tmp_path, monkeypatch, task_id="grand", root_task_id="root",
            extra={"parent_task_id": "middle", "delegation_role": "subagent"})
    for tid, parent, root in (("middle", "root", "root"), ("grand", "middle", "root"),
                              ("foreign", "other", "other")):
        write_task_result(tmp_path, tid, STATUS_SCHEDULED, parent_task_id=parent,
                          root_task_id=root, delegation_role="subagent")
    events = []
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, evt: events.append(evt) or "live")
    monkeypatch.setattr(join_ledger, "_record_child_decision_beacon", lambda *_a, **_k: None)
    ctx = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={})
    assert "Resume requested" in join_ledger._resume_child_task(ctx, "grand")
    assert "not a child" in join_ledger._resume_child_task(ctx, "foreign")
    assert "not a child" in join_ledger._resume_child_task(ctx, "unknown")
    ctx.task_id = "middle"
    assert "Resume requested" in join_ledger._resume_child_task(ctx, "grand")
    assert "not a child" in join_ledger._resume_child_task(ctx, "root")
    assert queue.resume_budget_paused_task("root")["ok"]
    outcomes = []
    _handle_budget_resume_child(events[0], _supervisor_ctx(tmp_path, workers, queue, [], outcomes))
    assert outcomes[-1]["ok"] is True


@pytest.mark.parametrize("control", ["panic", "stop"])
def test_hold_reads_controls_from_canonical_budget_root(tmp_path, monkeypatch, control):
    from ouroboros import budget_pause, model_wait
    from ouroboros.cancel_intents import request_cancel

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    ctx, _ = _loop_ctx(tmp_path / "execution", "split")
    ctx.budget_drive_root = tmp_path
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: None)
    assert budget_pause._hold_control_reason(ctx) == ""
    if control == "panic":
        monkeypatch.setattr(model_wait, "current_model_wait",
                            lambda: SimpleNamespace(control_reason=lambda: "panic"))
    else:
        request_cancel(tmp_path, "split", reason="owner_stopped", source="test")
    assert budget_pause._hold_control_reason(ctx) == ("panic" if control == "panic" else "cancelled")


def test_failed_finite_lifetime_read_refuses_delegation_without_reset(tmp_path, monkeypatch):
    from ouroboros import config, model_wait
    from ouroboros.tools import delegate

    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 100.0)
    ctx = SimpleNamespace(task_id="bounded", task_started_at=1.0)
    waiter = SimpleNamespace(task_id="bounded", execution_window_remaining=lambda: (_ for _ in ()).throw(OSError("unknown")))
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    assert delegate.bounded_max_seconds(ctx, 20).refusal_code == "task_lifetime_unknown"
    waiter.execution_window_remaining = lambda: 8.5
    result = delegate.bounded_max_seconds(ctx, 20)
    assert not result.refusal_code and result.seconds == 8


def test_direct_actor_releases_local_fence_after_parking_same_id(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor import message_bus, worker_chat_lane
    from tests._budget_pause_exact_helpers import _fast_hold, _quiet_external

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    ctx, limit = _loop_ctx(tmp_path, "direct", direct=True)
    ctx.current_chat_id = 7
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="budget")
    task = {"id": "direct", "type": "task", "chat_id": 7, "text": "work",
            "project_id": "fixture", "_is_direct_chat": True}
    events, released = [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda _evt: None))
    agent = SimpleNamespace(handle_task=lambda _task: [budget_pause.pause_event(task, raised.value.pause)])
    assert budget_pause.dispatch_fenced("direct")
    try:
        assert worker_chat_lane._execute_chat_task({"task": task, "agent": agent, "chat_id": 7,
                                                    "registry": SimpleNamespace(unregister=released.append)})
        assert released == ["direct"] and not budget_pause.dispatch_fenced("direct")
        assert workers.PENDING[0]["id"] == "direct" and workers.PENDING[0]["_budget_pause"]
        assert not events  # inline park consumed the pause; no duplicate supervisor event
        assert queue.resume_budget_paused_task("direct")["ok"]
    finally:
        budget_pause.end_dispatch_fence("direct")


def test_direct_actor_releases_local_fence_even_when_the_inline_park_fails(tmp_path, monkeypatch):
    """Negative twin of the inline park: when the in-process park raises, the pause
    event takes the ordinary supervisor path (never lost) AND the turn's local
    dispatch fence is still released — the actor has unwound, and a fence left
    closed in the supervisor process would refuse the resumed turn's sends."""
    from ouroboros import budget_pause
    from supervisor import events_budget, message_bus, worker_chat_lane
    from tests._budget_pause_exact_helpers import _fast_hold, _quiet_external

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    ctx, limit = _loop_ctx(tmp_path, "direct-fb", direct=True)
    ctx.current_chat_id = 7
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="budget")
    task = {"id": "direct-fb", "type": "task", "chat_id": 7, "text": "work",
            "project_id": "fixture", "_is_direct_chat": True}
    events, released = [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda _evt: None))
    monkeypatch.setattr(events_budget, "install_exact_budget_pause",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("queue lock unavailable")))
    agent = SimpleNamespace(handle_task=lambda _task: [budget_pause.pause_event(task, raised.value.pause)])
    assert budget_pause.dispatch_fenced("direct-fb")
    try:
        assert worker_chat_lane._execute_chat_task({"task": task, "agent": agent, "chat_id": 7,
                                                    "registry": SimpleNamespace(unregister=released.append)})
        assert released == ["direct-fb"] and not budget_pause.dispatch_fenced("direct-fb")
        # The pause was not parked here, so its event reached the ordinary path intact.
        assert [e["type"] for e in events] == ["budget_pause"] and events[0]["_is_direct_chat"] is True
        assert workers.PENDING == []
    finally:
        budget_pause.end_dispatch_fence("direct-fb")


def _bare_loop(tmp_path, monkeypatch):
    """The lightest real ``run_llm_loop`` driver: a bare registry, review off, no model."""
    from ouroboros import loop as loop_mod
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(loop_mod, "call_llm_with_retry",
                        lambda *_a, **_k: pytest.fail("a control-ended hold buys no model call"))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    return loop_mod, registry


def _run(loop_mod, registry, tmp_path, task_id):
    import queue as queue_mod

    return loop_mod.run_llm_loop(
        messages=[{"role": "user", "content": "go"}], tools=registry,
        llm=SimpleNamespace(default_model=lambda: "test-model"), drive_logs=tmp_path,
        emit_progress=lambda _text, *, incident=None: None, incoming_messages=queue_mod.Queue(),
        task_id=task_id, drive_root=tmp_path)


def test_hold_ended_by_deadline_outside_the_model_call_is_a_truthful_terminal(tmp_path, monkeypatch):
    """A budget-pause HOLD ended by the task's own deadline raises ``ModelWaitInterrupted``
    from the budget tails, OUTSIDE the model-call try. The loop rejoins the same
    control rails a live wait uses — a no-call ``deadline_local`` terminal — instead
    of surfacing a generic task exception (``infra_failed``/``task_exception``)."""
    from ouroboros.model_wait import ModelWaitInterrupted

    loop_mod, registry = _bare_loop(tmp_path, monkeypatch)
    # The pre-round exit (soft landing -> request_pause -> hold) is where the hold lives.
    monkeypatch.setattr(loop_mod, "_maybe_early_finalize",
                        lambda *_a, **_k: (_ for _ in ()).throw(ModelWaitInterrupted("deadline")))
    text, usage, trace = _run(loop_mod, registry, tmp_path, "hold-deadline")
    assert usage["execution_status"] == "failed" and usage["reason_code"] == "deadline_local"
    assert trace["forced_finalization"]["control_reason"] == "deadline"
    assert isinstance(text, str) and text


def test_an_interruption_the_model_call_rail_already_routed_is_not_routed_twice(tmp_path, monkeypatch):
    """The outer rail is for holds only: a Stop the model-call handler re-raised (the
    supervisor owns its settlement) keeps propagating with its evidence, and the
    control handler runs exactly once for it."""
    from ouroboros.model_wait import ModelWaitInterrupted

    loop_mod, registry = _bare_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(loop_mod, "_call_round_model",
                        lambda _call: (_ for _ in ()).throw(ModelWaitInterrupted("cancelled")))
    real = loop_mod._handle_model_wait_control
    routed = []

    def counting(ctx, error, **kw):
        routed.append(error.control_reason)
        return real(ctx, error, **kw)

    monkeypatch.setattr(loop_mod, "_handle_model_wait_control", counting)
    with pytest.raises(ModelWaitInterrupted) as raised:
        _run(loop_mod, registry, tmp_path, "stop-once")
    assert routed == ["cancelled"] and raised.value.control_rails_seen is True
    assert isinstance(getattr(raised.value, "_ouroboros_loop_usage", None), dict)


def test_a_zero_dispatch_selection_is_rechecked_against_the_live_root_grant_at_dispatch(tmp_path, monkeypatch):
    """#1196 review finding 4: a child carrying a SELECTED ``_budget_pause_hold``
    (no exact grant handoff) returned True from ``budget_resume_dispatch_allowed``
    immediately, so after a global-scope re-pause of its root — which raises no
    new fence — it dispatched on the old selection. Both carriers are now bound
    to the root grant they were selected under; a stale one returns to an
    UNSELECTED hold and is re-bound and re-selectable under the next root Resume."""
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    root, _ = _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    sibling = _fenced_member(workers, "sibling", "root")
    first = queue.resume_budget_paused_task("root")
    assert first["ok"] and queue.resume_budget_paused_task("sibling", selected_by="root")["ok"]
    assert "_resume_selection" in sibling and "_budget_pause_hold" not in sibling
    assert sibling["_resume_selection"]["root_grant_id"] == first["grant_id"]
    workers.PENDING.remove(root)  # the root ran on, then paused AGAIN under global scope: no new fence
    _parked(tmp_path, monkeypatch, task_id="root", scope="global")
    assert "root" not in queue.BUDGET_ROOT_FENCES
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == [] and "_budget_pause_hold" in sibling and "_resume_selection" not in sibling
    assert sibling["_budget_pause_hold"]["detail"] == "root_resume_generation_stale"
    second = queue.resume_budget_paused_task("root")
    assert second["ok"] and second["grant_id"] != first["grant_id"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root"]  # the root Resume made the sibling eligible only
    assert queue.resume_budget_paused_task("sibling", selected_by="root")["ok"]
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root", "sibling"]
    assert sent[-1]["_resume_selection"]["root_grant_id"] == second["grant_id"]


def test_a_hold_ended_by_control_inside_the_refused_dispatch_rail_is_a_truthful_terminal(tmp_path, monkeypatch):
    """#1196 review finding 7: ``_handle_budget_exceeded`` runs INSIDE the loop's
    ``except BudgetExceeded`` clause; a hold it entered through ``request_pause``
    that a deadline ended raised ``ModelWaitInterrupted`` past the sibling
    ``except Exception`` and reached task_exception. It now rejoins the common
    control rails: a no-call ``deadline_local`` terminal, no model call."""
    from ouroboros import budget_pause, usage_accounting
    from ouroboros.model_wait import ModelWaitInterrupted

    loop_mod, registry = _bare_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(loop_mod, "_call_round_model", lambda _call: (_ for _ in ()).throw(
        usage_accounting.BudgetExceeded("global model budget exhausted", limit_scope="global")))
    monkeypatch.setattr(usage_accounting, "usage_breakdown", lambda *_a, **_k: {"physical_calls": 3})
    monkeypatch.setattr(budget_pause, "request_pause",
                        lambda *_a, **_k: (_ for _ in ()).throw(ModelWaitInterrupted("deadline")))
    text, usage, trace = _run(loop_mod, registry, tmp_path, "hold-budget-rail")
    assert usage["execution_status"] == "failed" and usage["reason_code"] == "deadline_local"
    assert trace["forced_finalization"]["control_reason"] == "deadline"
    assert isinstance(text, str) and text


def test_consumed_resume_crash_retries_current_working_source_without_old_grant(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_wait, working_checkpoint as wc
    from supervisor import worker_health

    queue, workers, job, _ = _dead_granted_task(tmp_path, monkeypatch)
    monkeypatch.setattr(worker_health, "_dead_job_is_current", lambda j: workers.RUNNING.get("dead") is j["meta"])
    ctx, limit = _loop_ctx(tmp_path, "dead")
    old = copy.deepcopy(job["task"]["_budget_pause_resume"])
    saved = budget_pause.load_budget_pause(ctx, old)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, [], {}, {}, set(), budget_remaining_usd=5.0)
    # A newer batch owns an unfinished effect. Recovery must keep its UNKNOWN closure.
    limit.messages.append({"role": "user", "content": "new work after Resume"})
    wc.save_round(limit, "pre_effect")
    before = budget_pause.budget_pause_row(tmp_path, "dead")
    worker_health._recover_crashed_task_without_terminal(job, queue)
    [retry] = workers.PENDING
    assert retry["id"] == "dead" and retry["_attempt"] == 2 and not workers.RUNNING
    assert "_budget_pause_resume" not in retry and "_budget_pause" not in retry
    assert budget_pause.budget_pause_row(tmp_path, "dead") == before
    with pytest.raises(ValueError, match="no live single-use grant"):
        budget_pause.load_budget_pause(ctx, old)
    restored = wc.load_recovery(ctx, retry["_working_recovery"])
    assert restored["working"]["boundary"] == "pre_effect"
    assert restored["working"]["pending_tool_call_ids"] == ["call_b"]
    assert any(row.get("content") == "new work after Resume" for row in restored["messages"])
    from tests.test_budget_pause_holds import _idle_worker
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["dead"]
    assert sent[0]["_working_recovery"] == retry["_working_recovery"]
    from ouroboros import loop
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, "dead", 2)
    registry._ctx.working_recovery = retry["_working_recovery"]
    monkeypatch.setattr(registry, "execute_result", lambda *_a, **_k: pytest.fail("UNKNOWN effect replayed"))
    seen = []

    def dispatch(call, disposition, **kwargs):
        seen.extend(call.messages)
        return {"role": "assistant", "content": "Continued new work", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    kwargs = {**_loop_kwargs(tmp_path, registry, [], llm=SimpleNamespace(default_model=lambda: "m")), "task_id": "dead"}  # the checkpoint ran on Main
    result, _usage, _trace = loop.run_llm_loop(**kwargs)
    assert result == "Continued new work"
    assert any(row.get("content") == "done a" for row in seen)
    unknown = next(row for row in seen if row.get("tool_call_id") == "call_b")
    assert "UNKNOWN" in unknown["content"] and "NOT re-executed" in unknown["content"]
