"""A member's exact pause names its tree from ``task_metadata``, through its REAL consumers.

A worker's ``ToolContext`` carries lineage only in ``task_metadata`` (it has no
``root_task_id`` field), so the owner-Pause and cold-sleep writers, the
Resume's owner-fence consumer and the grant all resolve the root the fence
reader and the dispatch guard read. A pause the older writers saved with the
member's own id as its root — never granted, or granted and then revoked as
``root_resume_generation_stale`` — resumes under the live root grant through
actual assignment and consumption, and that grant still goes stale when the
root's Resume generation or fence moves on. The guard itself is unchanged.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _mock_pause_observation

pytestmark = pytest.mark.serial

ROOT, CHILD = "lineage-root", "lineage-child"


def _member_ctx(tmp_path, task_id=CHILD, root=ROOT):
    """The worker's ToolContext (``agent.py``) plus the loop attributes it binds."""
    from ouroboros.tools.tool_context import ToolContext

    meta = {"root_task_id": root, "budget_drive_root": str(tmp_path)}
    if task_id != root:
        meta.update(parent_task_id=root, delegation_role="subagent")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path),
                      task_metadata=meta, task_id=task_id, task_lifecycle_bound=True)
    ctx.task_attempt, ctx.task_started_at = 1, time.time() - 100.0
    ctx.owner_wait_callback = lambda *_a, **_k: "owner_input"
    ctx.model_wait_context = ctx._cost_ceiling = ctx.budget_pause_resume = None
    # What the (stubbed) cognition restore would have rebound on Resume.
    ctx.active_model, ctx.active_effort, ctx.active_use_local = "m", "high", False
    assert not hasattr(ctx, "root_task_id"), "the lineage lives in task_metadata only"
    limit_ctx = SimpleNamespace(
        tools=SimpleNamespace(_ctx=ctx), accumulated_usage={"cost": 0.5}, round_idx=3,
        messages=[{"role": "user", "content": "do it"}], llm_trace={}, owner_msg_seen=set(),
        tool_schemas=[])
    return ctx, limit_ctx


def _sup(tmp_path, queue, workers):
    return SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                           WORKERS=workers.WORKERS, sort_pending=lambda: None,
                           persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)


def _workers(workers):
    sent = []
    for wid in (0, 1):
        workers.WORKERS[wid] = SimpleNamespace(wid=wid, busy_task_id=None, reaping=False,
                                               in_q=SimpleNamespace(put=lambda task: sent.append(dict(task))))
    return sent


def _live_tree(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    write_task_result(tmp_path, CHILD, STATUS_RUNNING, chat_id=0, root_task_id=ROOT, parent_task_id=ROOT)
    workers.RUNNING[ROOT] = {"task": {"id": ROOT, "type": "task", "chat_id": 0, "root_task_id": ROOT},
                             "worker_id": 0, "attempt": 1}
    workers.RUNNING[CHILD] = {"task": {
        "id": CHILD, "type": "task", "chat_id": 0, "root_task_id": ROOT, "parent_task_id": ROOT,
        "delegation_role": "subagent"}, "worker_id": 1, "attempt": 1}
    assert request_owner_pause(ROOT, request_id="p")["ok"]
    return queue, state, workers


def _park(limit_ctx, *, old_writer_root=""):
    """The member's owner-Pause boundary; ``old_writer_root`` replays what the
    pre-fix writer passed for a ToolContext (its own id) instead."""
    from ouroboros import budget_pause
    from ouroboros.owner_pause import member_fence

    ctx = limit_ctx.tools._ctx
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        if old_writer_root:
            ctx._owner_pause_fence_id = member_fence(ctx)["fence_id"]
            budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_OWNER_PAUSE, scope="root",
                                       reason_text="The owner paused this task tree.",
                                       root_task_id=old_writer_root)
        else:
            budget_pause.enter_owner_pause(limit_ctx)
    budget_pause.end_dispatch_fence(str(ctx.task_id))
    return raised.value.pause


def _consume(monkeypatch, ctx, limit_ctx, handoff):
    """The resumed worker's real consumption of its grant (cognition stubbed)."""
    from ouroboros import budget_pause

    ctx.budget_pause_resume = handoff
    blob = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    # A regression must raise, never spin in the consumption hold.
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "test_stop")
    budget_pause.resume_paused_loop(limit_ctx.tools, blob, list(limit_ctx.messages), {}, {}, set(),
                                    budget_remaining_usd=5.0)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")


def _resumed_root_over_parked_child(tmp_path, monkeypatch, *, old_writer):
    """Owner Pause of a live tree, both members parked, the root Resumed and consumed."""
    from ouroboros import budget_pause, owner_pause
    from supervisor.events_budget import install_exact_budget_pause

    queue, state, workers = _live_tree(tmp_path, monkeypatch)
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    sup = _sup(tmp_path, queue, workers)
    child_ctx, child_limit = _member_ctx(tmp_path)
    child_row = _park(child_limit, old_writer_root=CHILD if old_writer else "")
    install_exact_budget_pause(sup, CHILD, budget_pause.exact_pause_marker(child_row)["checkpoint"])
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    root_row = _park(root_limit)
    install_exact_budget_pause(sup, ROOT, budget_pause.exact_pause_marker(root_row)["checkpoint"])
    assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, ROOT))

    sent = _workers(workers)
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
    assert owner_pause.read_fence(tmp_path, ROOT)["state"] == owner_pause.FENCE_RELEASED
    root_grant = budget_pause.budget_pause_row(tmp_path, ROOT)["grant"]
    assert root_grant["consumed_at"] and root_grant["generation"] == 1
    sent.clear()
    child = next(task for task in workers.PENDING if task["id"] == CHILD)
    return queue, workers, sent, child, (child_ctx, child_limit), root_grant


def _replay_old_member_grant(queue, workers, child):
    """The pre-fix grant over a self-rooted pause: bound to no root grant, and it
    popped the member-keyed latch the self-rooted park had installed."""
    from ouroboros import budget_pause

    row = budget_pause.budget_pause_row(queue.DRIVE_ROOT, CHILD)
    pause = child.pop("_budget_pause")
    grant = {"grant_id": "old-member-grant", "granted_at": "2026-10-05T23:40:44+00:00",
             "granted_at_ts": time.time(), "single_use": True, "paused_duration_sec": 1.0,
             "pause_id": row["pause_id"], "pause_generation": row["pause_generation"], "generation": 1,
             "selected_by": "owner", "authority": "explicit_resume", "sleep_id": "",
             "root_grant_id": "", "root_resume_generation": 0, "root_fence_id": ""}
    budget_pause.set_budget_pause(queue.DRIVE_ROOT, CHILD, {**row, "state": budget_pause.STATE_RESUME_GRANTED,
                                                            "grant": grant, "resume_generation": 1})
    child["_budget_pause_resume"] = {
        **pause["checkpoint"], "grant_id": grant["grant_id"], "granted_at": grant["granted_at"],
        "grant_generation": 1, "pause_id": row["pause_id"], "pause_generation": row["pause_generation"],
        "paused_duration_sec": 1.0, "pause": pause, "external_runs": {},
        **{key: grant[key] for key in ("selected_by", "authority", "sleep_id", "root_grant_id",
                                       "root_resume_generation", "root_fence_id")}}
    queue.BUDGET_ROOT_FENCES.pop(CHILD, None)
    return pause


@pytest.mark.parametrize("writer", ["owner", "sleep"])
def test_a_real_member_context_saves_its_tree_root_and_takes_the_tree_launch_lock(tmp_path, monkeypatch, writer):
    from ouroboros import budget_pause, model_sleep, owner_pause
    from supervisor.events_budget import install_exact_budget_pause

    queue, _state, workers = _live_tree(tmp_path, monkeypatch)
    if writer == "sleep":
        owner_pause.release_fence(tmp_path, ROOT, reason="test")
        queue.BUDGET_ROOT_FENCES.pop(ROOT, None)
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    locks: list = []
    real_lock = owner_pause.launch_lock
    monkeypatch.setattr(owner_pause, "launch_lock",
                        lambda drive, root_id, **kw: locks.append(root_id) or real_lock(drive, root_id, **kw))
    ctx, limit_ctx = _member_ctx(tmp_path)
    if writer == "owner":
        row = _park(limit_ctx)
    else:
        monkeypatch.setattr(model_sleep, "cold_blockers", lambda _ctx, **_kw: [])
        ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=3600)}
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            budget_pause.enter_cold_sleep(limit_ctx)
        budget_pause.end_dispatch_fence(CHILD)
        row = raised.value.pause
    assert row["root_task_id"] == ROOT
    assert budget_pause.budget_pause_row(tmp_path, CHILD)["root_task_id"] == ROOT
    assert ROOT in locks and CHILD not in locks, "the seed closes the TREE's launch admission"
    event = budget_pause.pause_event({"id": CHILD, "root_task_id": ROOT}, row)
    assert event["root_task_id"] == event["resource_limit"]["root_task_id"] == ROOT
    install_exact_budget_pause(_sup(tmp_path, queue, workers), CHILD,
                               budget_pause.exact_pause_marker(row)["checkpoint"])
    assert workers.PENDING[0]["_budget_pause"]["root_task_id"] == ROOT
    assert CHILD not in queue.BUDGET_ROOT_FENCES, "no latch keyed by the member itself"
    if writer == "owner":
        assert row["owner_fence_id"] == owner_pause.read_fence(tmp_path, ROOT)["fence_id"]
        assert queue.BUDGET_ROOT_FENCES[ROOT]["cause"] == "owner_pause"


@pytest.mark.parametrize("history", ["fixed_writer", "self_rooted_never_granted", "self_rooted_granted_then_revoked"])
def test_a_member_pause_resumes_under_the_live_root_grant_through_dispatch_and_consumption(
        tmp_path, monkeypatch, history):
    from ouroboros import budget_pause, owner_pause
    from supervisor.events_budget import budget_resume_dispatch_allowed

    queue, workers, sent, child, (ctx, limit_ctx), root_grant = _resumed_root_over_parked_child(
        tmp_path, monkeypatch, old_writer=history != "fixed_writer")
    saved = budget_pause.budget_pause_row(tmp_path, CHILD)
    assert saved["root_task_id"] == (ROOT if history == "fixed_writer" else CHILD)
    assert child["root_task_id"] == ROOT, "the queue row keeps the canonical lineage"
    self_latch = queue.BUDGET_ROOT_FENCES.get(CHILD)
    assert (self_latch is None) == (history == "fixed_writer")
    if history == "self_rooted_granted_then_revoked":
        # The live failure: the old grant is refused at actual assignment and revoked.
        stale_pause = _replay_old_member_grant(queue, workers, child)
        assert budget_resume_dispatch_allowed(queue, child) is False
        workers.assign_tasks()
        assert sent == []
        revoked = budget_pause.budget_pause_row(tmp_path, CHILD)
        assert revoked["state"] == budget_pause.STATE_PAUSED and revoked["resume_generation"] == 1
        assert revoked["grant"]["revoke_reason"] == "root_resume_generation_stale"
        assert child["_budget_pause"] == stale_pause
        assert stale_pause["root_task_id"] == CHILD and stale_pause["fence_id"] == self_latch["fence_id"]
        assert CHILD not in queue.BUDGET_ROOT_FENCES

    granted = queue.resume_budget_paused_task(CHILD)
    assert granted["ok"] is True and granted["root_task_id"] == ROOT, granted
    handoff = child["_budget_pause_resume"]
    assert handoff["root_grant_id"] == root_grant["grant_id"]
    assert handoff["root_resume_generation"] == root_grant["generation"]
    assert handoff["grant_generation"] == (2 if history == "self_rooted_granted_then_revoked" else 1)
    assert budget_resume_dispatch_allowed(queue, child) is True
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]

    _consume(monkeypatch, ctx, limit_ctx, sent[0]["_budget_pause_resume"])
    consumed = budget_pause.budget_pause_row(tmp_path, CHILD)
    assert consumed["state"] == budget_pause.STATE_RESUMED and consumed["grant"]["consumed_at"]
    assert owner_pause.read_fence(tmp_path, ROOT)["state"] == owner_pause.FENCE_RELEASED
    assert owner_pause.member_fence(ctx) == {}
    # Disclosed residue, no migration: a never-granted self-rooted park leaves its
    # member-keyed latch behind; nothing reads it for this member or its tree.
    assert (CHILD in queue.BUDGET_ROOT_FENCES) == (history == "self_rooted_never_granted")


@pytest.mark.parametrize("root_moves", ["new_resume_generation", "new_root_fence"])
def test_the_repaired_member_grant_still_goes_stale_when_the_root_moves_on(tmp_path, monkeypatch, root_moves):
    from ouroboros import budget_pause
    from supervisor.events_budget import _set_root_budget_pause_locked, budget_resume_dispatch_allowed

    queue, workers, sent, child, _ctx, root_grant = _resumed_root_over_parked_child(
        tmp_path, monkeypatch, old_writer=True)
    _replay_old_member_grant(queue, workers, child)
    workers.assign_tasks()
    assert sent == []
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    assert budget_resume_dispatch_allowed(queue, child) is True, "bound to the live root grant"
    if root_moves == "new_resume_generation":
        row = budget_pause.budget_pause_row(tmp_path, ROOT)
        budget_pause.set_budget_pause(tmp_path, ROOT, {
            **row, "resume_generation": 2, "grant": {**root_grant, "grant_id": "later-root-grant", "generation": 2}})
        assert budget_resume_dispatch_allowed(queue, child) is False
        reason = "root_resume_generation_stale"
    else:
        with queue._queue_lock:
            _set_root_budget_pause_locked(ROOT, {"reason": "owner"})
        reason = "new_root_budget_fence"
    workers.assign_tasks()
    assert sent == [] and "_budget_pause_resume" not in child
    revoked = budget_pause.budget_pause_row(tmp_path, CHILD)
    assert revoked["state"] == budget_pause.STATE_PAUSED and revoked["grant"]["revoke_reason"] == reason


def test_a_selected_member_of_a_terminal_root_consumes_the_closed_fence_from_its_metadata(tmp_path, monkeypatch):
    """The grant names the still-closed owner fence; consumption must select this
    member on the ROOT's fence (its own record has none) and keep siblings fenced,
    through the FIRST send: reservation, launch handoff and dispatch, unstubbed."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.llm_attempt import _PhysicalSendNotStarted, require_physical_dispatch_window
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.usage_accounting import AttemptRequest, UsageScope, mark_dispatched, reserve_attempt, usage_scope
    from supervisor.events_budget import install_exact_budget_pause

    queue, _state, workers = _live_tree(tmp_path, monkeypatch)
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    ctx, limit_ctx = _member_ctx(tmp_path)
    row = _park(limit_ctx)
    install_exact_budget_pause(_sup(tmp_path, queue, workers), CHILD,
                               budget_pause.exact_pause_marker(row)["checkpoint"])
    workers.RUNNING.pop(ROOT)
    write_task_result(tmp_path, ROOT, STATUS_COMPLETED, chat_id=0, root_task_id=ROOT)
    fence = owner_pause.read_fence(tmp_path, ROOT)
    assert owner_pause.fence_closed(fence)

    sent = _workers(workers)
    granted = queue.resume_budget_paused_task(CHILD)
    assert granted["ok"] is True, granted
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]
    handoff = sent[0]["_budget_pause_resume"]
    assert budget_pause.budget_pause_row(tmp_path, CHILD)["grant"]["owner_pause_fence_id"] == fence["fence_id"]

    _consume(monkeypatch, ctx, limit_ctx, handoff)
    after = owner_pause.read_fence(tmp_path, ROOT)
    assert owner_pause.fence_closed(after) and after["fence_id"] == fence["fence_id"]
    assert after["selected_members"][CHILD]["grant_id"] == handoff["grant_id"]
    assert owner_pause.member_fence(ctx) == {}
    sibling, _limit = _member_ctx(tmp_path, "lineage-sibling")
    assert owner_pause.fence_closed(owner_pause.member_fence(sibling)), "siblings stay fenced"

    monkeypatch.setattr("ouroboros.usage_accounting._reservation_cost", lambda _r: 0.01)  # pricing only
    with usage_scope(UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
        reservation = reserve_attempt(AttemptRequest(model="m", provider="p", task_id=CHILD))
        require_physical_dispatch_window()
        mark_dispatched(reservation)
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="lineage-sibling", root_task_id=ROOT)):
        unselected = reserve_attempt(AttemptRequest(model="m", provider="p", task_id="lineage-sibling"))
        with pytest.raises(_PhysicalSendNotStarted):
            require_physical_dispatch_window()
        with pytest.raises(_PhysicalSendNotStarted):
            mark_dispatched(unselected)


@pytest.mark.parametrize("saved_root", ["metadata", "member_id"])
def test_a_child_cold_wake_binds_the_task_row_root_and_waits_for_a_paused_root(tmp_path, monkeypatch, saved_root):
    """Characterization of the grant's lineage on the sleep rail: whatever root the
    pause saved, the wake binds the queue row's root, and a parked root refuses the
    wake typed (``root_still_paused``) instead of a grant the guard would revoke."""
    from ouroboros import budget_pause, model_sleep
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.events_budget import budget_resume_dispatch_allowed, install_exact_budget_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    write_task_result(tmp_path, CHILD, STATUS_RUNNING, chat_id=0, root_task_id=ROOT, parent_task_id=ROOT)
    workers.RUNNING[ROOT] = {"task": {"id": ROOT, "type": "task", "chat_id": 0, "root_task_id": ROOT},
                             "worker_id": 0, "attempt": 1}
    workers.RUNNING[CHILD] = {"task": {"id": CHILD, "type": "task", "chat_id": 0, "root_task_id": ROOT,
                                       "parent_task_id": ROOT}, "worker_id": 1, "attempt": 1}
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda _ctx, **_kw: [])
    ctx, limit_ctx = _member_ctx(tmp_path)
    ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=1)}
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        if saved_root == "metadata":
            budget_pause.enter_cold_sleep(limit_ctx)
        else:
            budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_MODEL_SLEEP, scope="task",
                                       reason_text="The model chose a cold sleep.", root_task_id=CHILD)
    budget_pause.end_dispatch_fence(CHILD)
    row = raised.value.pause
    assert row["root_task_id"] == (ROOT if saved_root == "metadata" else CHILD)
    install_exact_budget_pause(_sup(tmp_path, queue, workers), CHILD,
                               budget_pause.exact_pause_marker(row)["checkpoint"])
    time.sleep(1.2)
    child = next(task for task in workers.PENDING if task["id"] == CHILD)

    root_task = workers.RUNNING.pop(ROOT)["task"]
    workers.PENDING.append({**root_task, "_attempt": 1, "_budget_pause": {
        "status": budget_pause.STATUS_PAUSED_EXACT, "exact_continuation": True, "scope": "global",
        "root_task_id": ROOT}})
    refused = wake_ready_sleepers(queue)
    assert [(item["task_id"], item.get("error")) for item in refused] == [(CHILD, "root_still_paused")]
    assert "_budget_pause_resume" not in child
    assert budget_pause.budget_pause_row(tmp_path, CHILD)["sleep_ready"]["reason"] == "timeout"

    workers.PENDING.remove(next(task for task in workers.PENDING if task["id"] == ROOT))
    workers.RUNNING[ROOT] = {"task": root_task, "worker_id": 0, "attempt": 1}
    woke = wake_ready_sleepers(queue)
    assert woke[0]["ok"] is True and woke[0]["root_task_id"] == ROOT, woke
    assert child["_budget_pause_resume"]["selected_by"] == "sleep_wake"
    assert budget_resume_dispatch_allowed(queue, child) is True
