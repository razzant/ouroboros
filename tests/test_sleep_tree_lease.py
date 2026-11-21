"""A sleep is its actor's wait, never a stop of its tree.

Driven through the real consumers: the sleep's arming and both of its censuses (``model_sleep``),
the exact park (``budget_pause``) and the supervisor's install, the launch gate every member's
tool, model and run start passes (``owner_pause.launch_admission``), the assignment chooser with
its tree-scoped project lane (``project_lease.lane_holders``), the sleep-wake pass and its grant,
and the queue snapshot's restore. A cold root's children start and run; its tree keeps the lane
while they live and a foreign root of that project waits; the owner's Pause still holds them.
Each guard is shown firing beside the case it lets through.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external, _supervisor_ctx


def cold_park(root, monkeypatch, workers, queue, task_id, *, tree="", **selected):
    """Arm and take a real cold sleep of RUNNING ``task_id`` (member of ``tree``), then park it in PENDING."""
    from ouroboros import budget_pause as bp, model_sleep
    from supervisor.events_budget import install_exact_budget_pause

    ctx, limit = _loop_ctx(root, task_id)
    ctx.root_task_id = tree or task_id
    ctx.task_metadata = {"root_task_id": tree or task_id}
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    armed = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, **selected), model_sleep.MODE_COLD)
    assert armed["reason"] == "sleep_armed", armed
    try:
        with pytest.raises(bp.BudgetPauseRequested) as raised:
            bp.enter_cold_sleep(limit)
    finally:
        bp.end_dispatch_fence(task_id)
    install_exact_budget_pause(_supervisor_ctx(root, workers, queue, [], []), task_id,
                               bp.exact_pause_marker(raised.value.pause)["checkpoint"])
    return raised.value.pause


def _task(root, task_id, *, tree="", status="running", project="room", **stored):
    """A root (no ``tree``) or ``tree``'s subagent: its stored result and the task its queue row carries."""
    from ouroboros.task_results import write_task_result

    lineage = ({"root_task_id": tree, "parent_task_id": tree, "delegation_role": "subagent"} if tree
               else {"root_task_id": task_id})
    write_task_result(root, task_id, status, chat_id=0, project_id=project, **lineage, **stored)
    return {"id": task_id, "type": "task", "chat_id": 0, "_attempt": 1, "project_id": project,
            "budget_drive_root": str(root), **lineage, **({} if status == "running" else {"admitted_dispatch": "none"})}


def _run(workers, task):
    workers.RUNNING[task["id"]] = {"task": task, "worker_id": None, "attempt": 1}
    return task


def _launch(root, task_id, tree):
    """One member launch through the gate its tool calls, model sends and run starts share."""
    from ouroboros.owner_pause import launch_admission

    with launch_admission(SimpleNamespace(task_id=task_id, root_task_id=tree, drive_root=root)):
        return True


def _eligible(queue, workers, task):
    from supervisor.worker_assignment import _candidate_ineligible

    return not _candidate_ineligible(queue, task, 5.0, list(workers.RUNNING.values()))


def _writing(root, workers, task_id, tree):
    """A running native child in the middle of a write: its tool call is handed off and unreturned."""
    return _run(workers, _task(root, task_id, tree=tree, launch_handoffs={"op-1": {"tool": "write_file"}}))


def test_a_cold_roots_writing_child_runs_and_a_foreign_root_of_its_project_waits(tmp_path, monkeypatch):
    from ouroboros.owner_pause import OwnerPauseRefused
    from ouroboros.project_lease import lane_holders

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _run(workers, _task(tmp_path, "root"))
    _writing(tmp_path, workers, "kid", "root")
    queued_kid = _task(tmp_path, "kid-2", tree="root", status="scheduled")
    foreign, foreign_kid = _task(tmp_path, "foreign", status="scheduled"), _task(
        tmp_path, "foreign-kid", tree="foreign", status="scheduled")
    workers.PENDING.extend([queued_kid, foreign, foreign_kid])
    cold_park(tmp_path, monkeypatch, workers, queue, "root", tasks=["kid"])
    assert list(workers.RUNNING) == ["kid"] and workers.PENDING[-1]["_budget_pause"]["reason"] == "sleep"
    assert lane_holders(workers.RUNNING.values(), workers.PENDING) == {"room": {"root"}}, "the tree keeps its lane"
    assert _launch(tmp_path, "kid", "root"), "the sleeping root fences no launch of its child"
    with pytest.raises(OwnerPauseRefused, match="model_sleep"):
        _launch(tmp_path, "root", "root")  # its own sleep still fences the sleeper's own late launch
    assert _eligible(queue, workers, queued_kid), "a child of the sleeping tree starts under its lane"
    assert not _eligible(queue, workers, foreign), "a foreign root of the project waits for that lane"
    assert not _eligible(queue, workers, foreign_kid), "so does a child of another tree"
    workers.RUNNING.pop("kid")
    workers.PENDING.remove(queued_kid)  # the tree's members ended: a sleep alone keeps no lane
    assert lane_holders(workers.RUNNING.values(), workers.PENDING) == {}
    assert _eligible(queue, workers, foreign) and _eligible(queue, workers, foreign_kid)


def test_a_parent_sleeps_cold_beside_running_and_queued_children_and_they_start(tmp_path, monkeypatch):
    from ouroboros import model_sleep
    from tests.worker_pool_support import affirm_ready

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _run(workers, _task(tmp_path, "root"))
    _writing(tmp_path, workers, "kid-run", "root")
    workers.PENDING.append(_task(tmp_path, "kid-queued", tree="root", status="scheduled"))
    pause = cold_park(tmp_path, monkeypatch, workers, queue, "root", tasks=["kid-run", "kid-queued"])
    assert pause["reason"] == "sleep" and pause["sleep"]["tasks"] == ["kid-run", "kid-queued"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                                      in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["kid-queued"], "the queued child starts while its parent sleeps"
    assert {"kid-queued", "kid-run"} <= set(workers.RUNNING)
    # Only what no live row of the tree carries still holds a cold sleep, and only the sleep that releases the
    # tree's project lane: a top-level project sleeper's, held by an ENDED child's open handoff.
    _task(tmp_path, "other", project="other-room")
    _task(tmp_path, "other-kid", tree="other", status="failed", project="other-room",
          launch_handoffs={"op-1": {"tool": "x"}})
    other, _limit = _loop_ctx(tmp_path, "other")
    other.task_metadata = {"root_task_id": "other", "project_id": "other-room"}
    with pytest.raises(model_sleep.ColdSleepBlocked, match="member_custody other-kid"):
        model_sleep.request_sleep(other, model_sleep.selectors(other, wake_after_sec=60), model_sleep.MODE_COLD)
    # A helper of that tree, or a root with no project, releases no lane: the sibling's leftover is the sibling's.
    _task(tmp_path, "other-helper", tree="other", project="other-room")
    helper, _limit = _loop_ctx(tmp_path, "other-helper")
    helper.task_metadata = {"root_task_id": "other", "parent_task_id": "other", "project_id": "other-room"}
    assert model_sleep.request_sleep(helper, model_sleep.selectors(helper, wake_after_sec=60),
                                     model_sleep.MODE_COLD)["reason"] == "sleep_armed"
    other.task_metadata = {"root_task_id": "other"}
    assert model_sleep.request_sleep(other, model_sleep.selectors(other, wake_after_sec=60),
                                     model_sleep.MODE_COLD)["reason"] == "sleep_armed"


def test_at_one_worker_a_parents_unnamed_wait_on_its_queued_child_frees_the_worker_for_it(tmp_path, monkeypatch):
    """Owner Q2=A at C=1, through the public tool: while the parent runs, the queued child waits for the one
    worker; ``wait_task(timeout_sec=600)`` with no named mode sleeps cold, the park releases that worker and the
    child starts on it (an in-slot wait, the old default, kept the worker the child needed)."""
    import json

    from ouroboros import budget_pause as bp
    from ouroboros.tools.control_task_results import _wait_for_task
    from supervisor.events_budget import install_exact_budget_pause
    from tests.worker_pool_support import affirm_ready

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _run(workers, _task(tmp_path, "root"))
    workers.PENDING.append(_task(tmp_path, "kid", tree="root", status="scheduled"))
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id="root", reaping=False,
                                                      in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    assert list(workers.WORKERS) == [0], "one installed slot; the fixed pool never starts another"
    workers.assign_tasks()
    assert sent == [], "the running parent holds the one worker"
    ctx, limit = _loop_ctx(tmp_path, "root")
    ctx.task_metadata = {"root_task_id": "root"}
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    armed = json.loads(_wait_for_task(ctx, "kid", timeout_sec=600))
    assert armed["reason"] == "sleep_armed" and armed["mode"] == "cold" and "kept_process" not in armed, armed
    try:
        with pytest.raises(bp.BudgetPauseRequested) as raised:
            bp.enter_cold_sleep(limit)
    finally:
        bp.end_dispatch_fence("root")
    install_exact_budget_pause(_supervisor_ctx(tmp_path, workers, queue, [], []), "root",
                               bp.exact_pause_marker(raised.value.pause)["checkpoint"])
    workers.WORKERS[0].busy_task_id = None  # the parked parent's worker is free
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["kid"], "the queued child starts on the one worker its parent left"


def test_the_owners_pause_of_the_tree_still_holds_a_sleeping_roots_members(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.owner_mailbox import write_owner_message
    from supervisor.events_budget import budget_resume_dispatch_allowed
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _run(workers, _task(tmp_path, "root"))
    _writing(tmp_path, workers, "kid", "root")
    _run(workers, _task(tmp_path, "napper", tree="root"))
    workers.PENDING.append(_task(tmp_path, "kid-queued", tree="root", status="scheduled"))
    cold_park(tmp_path, monkeypatch, workers, queue, "napper", tree="root")  # beside its running root and sibling
    cold_park(tmp_path, monkeypatch, workers, queue, "root", tasks=["kid"])
    napper = next(task for task in workers.PENDING if task["id"] == "napper")
    assert write_owner_message(tmp_path, "come back", "napper", msg_id="napper-wake")
    owner_pause.install_fence(tmp_path, "root", request_id="owner")
    vetoed = {row["task_id"]: row for row in wake_ready_sleepers(queue)}
    assert vetoed["napper"].get("veto") == "owner_pause", vetoed
    for member in ("kid", "kid-queued"):
        with pytest.raises(owner_pause.OwnerPauseRefused, match="owner_pause"):
            _launch(tmp_path, member, "root")
    owner_pause.release_fence(tmp_path, "root", reason="owner_resume")
    woke = {row["task_id"]: row for row in wake_ready_sleepers(queue)}
    assert woke["napper"]["ok"] is True, woke
    assert budget_resume_dispatch_allowed(queue, napper) is True, "a sleeping root does not hold its child's wake"
    assert _eligible(queue, workers, napper) and _launch(tmp_path, "kid", "root") and _launch(tmp_path, "kid-queued", "root")


def test_a_restart_keeps_a_cold_roots_lane_and_its_live_children(tmp_path, monkeypatch):
    from ouroboros.project_lease import lane_holders
    from ouroboros.projects_registry import create_project
    from supervisor.events_budget import budget_hold_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    create_project(tmp_path, "room", name="Room")  # restore readmits each queued row to its registered room
    _run(workers, _task(tmp_path, "root"))
    queued_kid = _task(tmp_path, "kid", tree="root", status="scheduled")
    workers.PENDING.extend([queued_kid, _task(tmp_path, "foreign", status="scheduled")])
    cold_park(tmp_path, monkeypatch, workers, queue, "root", tasks=["kid"])
    assert queue.persist_queue_snapshot(reason="before_update")  # an update or crash: no owner Restart or Panic
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 3
    rows = {task["id"]: task for task in workers.PENDING}
    assert rows["root"]["_budget_pause"]["reason"] == "sleep" and budget_hold_fact(rows["root"]) is None
    # Unacknowledged queued work waits for an explicit Resume (the restart's own rule, untouched here);
    # the sleeping root's tree keeps its lane across the restart all the same.
    assert lane_holders(workers.RUNNING.values(), workers.PENDING) == {"room": {"root"}}
    for task_id in ("kid", "foreign"):
        assert queue.resume_budget_paused_task(task_id)["ok"], task_id
    assert _eligible(queue, workers, rows["kid"]) and _launch(tmp_path, "kid", "root")
    assert not _eligible(queue, workers, rows["foreign"]), "a foreign root of the project still waits"
    workers.PENDING.remove(rows["kid"])
    assert _eligible(queue, workers, rows["foreign"]), "once the tree has no live member the lane is free"
