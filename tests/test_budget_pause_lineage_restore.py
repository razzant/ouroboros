"""A member's exact pause names ONE tree through RESTORE and the FIRST SEND.

Every scenario starts from a real worker ``ToolContext`` (lineage in
``task_metadata`` only), the real queue, the real park/grant/assignment/
consumption seams and the real ``reserve_attempt`` fence check; only cognition
restore and the loop-side custody observation are stubbed, as in the existing
exact-pause suites. The selected child's sends under a monetary latch are in
``test_budget_pause_monetary_selection``, which reuses these helpers.

- ``test_an_omitted_root_pause_locks_and_records_one_tree``: a budget rail that
  omits ``root_task_id`` must take its seed launch lock on the SAME tree its
  durable row names.
- ``test_an_explicit_pause_root_is_kept_verbatim``: an explicit root (the
  dispatch-refused rail passes ``BudgetExceeded.root_task_id``, which
  ``usage_admission.raise_group_refusal`` may fill from the billing group) is
  neither re-derived nor replaced.
- ``test_restore_follows_the_roots_durable_owner_fence``: after the root's
  Resume released owner fence A, a parked child's canonical marker must not
  resurrect A at restore; a newer closed fence B must not be overwritten by
  A; unreadable or missing authority keeps the latch (fail closed).
- ``test_a_child_the_restore_parks_after_the_owner_release_raises_no_latch``:
  the same released fence over a marker an earlier restore created from the
  child's durable row (its park event never ran).
- ``test_a_self_rooted_owner_marker_restores_from_its_queue_roots_fence``: a
  marker an older writer saved under the member's own id finds the owner's
  authority at its queue row's root: closed raises that fence, released none.
- ``test_restore_raises_a_monetary_latch_only_for_a_pause_the_snapshot_has_not_seen``:
  a MONETARY latch lifted by the root's own consumed Resume stays lifted; a root
  grant revoked by the restart and a child parked by the restore itself raise it.
- ``test_a_child_resumed_after_the_roots_release_returns_to_its_pause_without_the_latch``:
  the child's unused grant returns it to the same pause, never the released latch.
- ``test_restore_keeps_the_snapshots_newer_latch_over_an_older_markers_identity``:
  latch B in the map is never overwritten by a marker's older A (monetary, or owner
  authority unreadable).
- ``test_a_stale_consumed_child_carrier_keeps_its_settlement_and_the_maps_own_latch``
  and ``test_an_old_self_rooted_locator_whose_resume_was_consumed_raises_no_latch``:
  snapshots older than the durable truth re-arm nothing.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _mock_pause_observation

pytestmark = pytest.mark.serial

ROOT, CHILD, SIBLING = "rl-root", "rl-child", "rl-sibling"


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


def _tree(tmp_path, monkeypatch, *, sibling=False):
    """Running root and child (and optionally a queued, never-started sibling)."""
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    write_task_result(tmp_path, CHILD, STATUS_RUNNING, chat_id=0, root_task_id=ROOT, parent_task_id=ROOT)
    workers.RUNNING[ROOT] = {"task": {"id": ROOT, "type": "task", "chat_id": 0, "root_task_id": ROOT},
                             "worker_id": 0, "attempt": 1}
    workers.RUNNING[CHILD] = {"task": {
        "id": CHILD, "type": "task", "chat_id": 0, "root_task_id": ROOT, "parent_task_id": ROOT,
        "delegation_role": "subagent"}, "worker_id": 1, "attempt": 1}
    if sibling:
        write_task_result(tmp_path, SIBLING, STATUS_SCHEDULED, chat_id=0, root_task_id=ROOT, parent_task_id=ROOT)
        workers.PENDING.append({"id": SIBLING, "type": "task", "chat_id": 0, "root_task_id": ROOT,
                                "parent_task_id": ROOT, "delegation_role": "subagent", "_attempt": 1,
                                "admitted_dispatch": "none"})
    from ouroboros import budget_pause

    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    return queue, state, workers


def _owner_park(limit_ctx):
    from ouroboros import budget_pause

    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit_ctx)
    budget_pause.end_dispatch_fence(str(limit_ctx.tools._ctx.task_id))
    return raised.value.pause


def _planning_park(limit_ctx):
    """The soft-land (planning-margin) rail exactly as ``loop_budget`` calls it: no root argument."""
    from ouroboros import budget_pause

    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_SOFT_LAND, scope="root",
                                   reason_text="Per-task tree cap leaves no working room. Budget exhausted.")
    budget_pause.end_dispatch_fence(str(limit_ctx.tools._ctx.task_id))
    return raised.value.pause


def _install(sup, task_id, row):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause

    install_exact_budget_pause(sup, task_id, budget_pause.exact_pause_marker(row)["checkpoint"])


def _consume(monkeypatch, ctx, limit_ctx, handoff):
    """The resumed worker's real consumption of its grant (cognition stubbed)."""
    from ouroboros import budget_pause

    ctx.budget_pause_resume = handoff
    blob = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "test_stop")
    budget_pause.resume_paused_loop(limit_ctx.tools, blob, list(limit_ctx.messages), {}, {}, set(),
                                    budget_remaining_usd=5.0)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")


def _complete_root(tmp_path, workers):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    workers.RUNNING.pop(ROOT, None)
    for worker in workers.WORKERS.values():
        if worker.busy_task_id == ROOT:
            worker.busy_task_id = None
    write_task_result(tmp_path, ROOT, STATUS_COMPLETED, chat_id=0, root_task_id=ROOT, result="done")


def _acknowledged_restart(queue, workers):
    """A returning Restart (owner 2026-10-08, quiz d2f7532b): its acknowledged transaction
    returns the formerly runnable queue; a holding stop would keep it for Resume instead."""
    from ouroboros import delegate_recovery
    from supervisor.restart_retention import prepare_restart_returns

    import uuid

    transaction_id = uuid.uuid4().hex  # every Restart is its own fresh transaction
    prepare_restart_returns(queue.DRIVE_ROOT, workers.RUNNING, workers.PENDING, transaction_id=transaction_id)
    row = delegate_recovery._read_restart_transaction(queue.DRIVE_ROOT, transaction_id)
    delegate_recovery._write_restart_transaction(queue.DRIVE_ROOT, {**row, "status": "normal_exit_acknowledged"})


def _restart(queue, workers, monkeypatch):
    """A new process after a Restart: empty queue memory, empty owner-fence read cache, then restore."""
    from ouroboros import owner_pause

    _acknowledged_restart(queue, workers)
    assert queue.persist_queue_snapshot(reason="test_before_restart")
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    for worker in workers.WORKERS.values():
        worker.busy_task_id = None
    monkeypatch.setattr(owner_pause, "_CACHE", {})
    return queue.restore_pending_from_snapshot()


def _pending(workers, task_id):
    return next(task for task in workers.PENDING if task["id"] == task_id)


def _record(label, facts):
    """Print the observed facts whole (assertion reprs truncate): retained evidence."""
    import json

    print("EVIDENCE " + label + " " + json.dumps(facts, default=str, sort_keys=True))


# --- (1) one tree per pause: the seed lock and the durable row --------------------------

@pytest.mark.parametrize("rail,scope", [("soft_land", "root"), ("global_exhausted", "global")])
def test_an_omitted_root_pause_locks_and_records_one_tree(tmp_path, monkeypatch, rail, scope):
    from ouroboros import budget_pause, owner_pause

    queue, _state, workers = _tree(tmp_path, monkeypatch)
    locks: list = []
    real_lock = owner_pause.launch_lock
    monkeypatch.setattr(owner_pause, "launch_lock",
                        lambda drive, root_id, **kw: locks.append(root_id) or real_lock(drive, root_id, **kw))
    _ctx, limit_ctx = _member_ctx(tmp_path)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=rail, scope=scope, reason_text="money gone")
    budget_pause.end_dispatch_fence(CHILD)
    row = raised.value.pause
    observed = {"row_root": row["root_task_id"],
                "durable_root": budget_pause.budget_pause_row(tmp_path, CHILD)["root_task_id"],
                "lock_roots": sorted(set(locks))}
    # The queue marker falls back to the queue row's root on BOTH trees (events_budget
    # parking passes ``default_root``): only the durable row and the lock differ.
    _install(_sup(tmp_path, queue, workers), CHILD, row)
    assert _pending(workers, CHILD)["_budget_pause"]["root_task_id"] == ROOT
    if scope == "root":
        assert ROOT in queue.BUDGET_ROOT_FENCES and CHILD not in queue.BUDGET_ROOT_FENCES
    _record(f"omitted_root[{rail}]", observed)
    assert observed == {"row_root": ROOT, "durable_root": ROOT, "lock_roots": [ROOT]}, observed


def test_an_explicit_pause_root_is_kept_verbatim(tmp_path, monkeypatch):
    """Normalizing an OMITTED root must not re-derive an explicit one."""
    from ouroboros import budget_pause, owner_pause

    _tree(tmp_path, monkeypatch)
    explicit = "rl-billing-group"  # what ``raise_group_refusal`` names when the scope has no root
    locks: list = []
    real_lock = owner_pause.launch_lock
    monkeypatch.setattr(owner_pause, "launch_lock",
                        lambda drive, root_id, **kw: locks.append(root_id) or real_lock(drive, root_id, **kw))
    _ctx, limit_ctx = _member_ctx(tmp_path)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_DISPATCH_REFUSED, scope="root",
                                   reason_text="whole-work budget exhausted", root_task_id=explicit)
    budget_pause.end_dispatch_fence(CHILD)
    observed = {"row_root": raised.value.pause["root_task_id"],
                "durable_root": budget_pause.budget_pause_row(tmp_path, CHILD)["root_task_id"],
                "lock_roots": sorted(set(locks))}
    _record("explicit_root", observed)
    assert observed == {"row_root": explicit, "durable_root": explicit, "lock_roots": [explicit]}, observed


# --- (2)+(3) restore of an OWNER latch from a parked child's canonical marker -----------

def _owner_paused_then_root_resumed(tmp_path, monkeypatch):
    """Owner Pause A over root+child+queued sibling; both members park; the root's
    Resume is granted, assigned and consumed (A released); the child stays parked."""
    from ouroboros import owner_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _tree(tmp_path, monkeypatch, sibling=True)
    sup = _sup(tmp_path, queue, workers)
    assert request_owner_pause(ROOT, request_id="pause-a")["ok"]
    fence_a = owner_pause.read_fence(tmp_path, ROOT)["fence_id"]
    child_ctx, child_limit = _member_ctx(tmp_path)
    _install(sup, CHILD, _owner_park(child_limit))
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(sup, ROOT, _owner_park(root_limit))
    marker = _pending(workers, CHILD)["_budget_pause"]
    sent = _workers(workers)
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
    sent.clear()
    assert owner_pause.read_fence(tmp_path, ROOT)["state"] == owner_pause.FENCE_RELEASED
    assert ROOT not in queue.BUDGET_ROOT_FENCES, "the root's own Resume lifted its latch"
    return queue, workers, sent, marker, fence_a, (child_ctx, child_limit), (root_ctx, root_limit)


def _spy_latches(monkeypatch):
    """Record every latch the restore installs (root, fence id actually installed)."""
    from supervisor import events_budget, queue as queue_mod

    installed: list = []
    real = events_budget._set_root_budget_pause_locked

    def spy(root_task_id, pause, **kwargs):
        row = real(root_task_id, pause, **kwargs)
        installed.append((str(root_task_id), str(queue_mod.BUDGET_ROOT_FENCES[str(root_task_id)]["fence_id"])))
        return row

    monkeypatch.setattr(events_budget, "_set_root_budget_pause_locked", spy)
    return installed


@pytest.mark.parametrize("authority", ["released", "newer_closed", "unreadable", "missing"])
def test_restore_follows_the_roots_durable_owner_fence(tmp_path, monkeypatch, authority):
    from ouroboros import owner_pause
    from ouroboros.task_results import task_result_path
    from supervisor.events_budget import budget_resume_dispatch_allowed
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import budget_pause_fact

    queue, workers, sent, marker, fence_a, (child_ctx, child_limit), (root_ctx, root_limit) = \
        _owner_paused_then_root_resumed(tmp_path, monkeypatch)
    observed_marker = {"root_task_id": marker["root_task_id"], "fence_id": marker.get("fence_id"),
                       "reason": marker["reason"]}
    fence_b = ""
    if authority == "released":
        # The owner selects the never-started sibling (no live latch to bind to).
        assert queue.resume_budget_paused_task(SIBLING)["ok"]
        assert budget_pause_fact(_pending(workers, SIBLING)) is None
        _complete_root(tmp_path, workers)
    elif authority == "newer_closed":
        assert request_owner_pause(ROOT, request_id="pause-b")["ok"]
        fence_b = owner_pause.read_fence(tmp_path, ROOT)["fence_id"]
        assert fence_b != fence_a and queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"] == fence_b
        root_ctx2, root_limit2 = _member_ctx(tmp_path, ROOT, ROOT)
        _install(_sup(tmp_path, queue, workers), ROOT, _owner_park(root_limit2))
    else:
        _complete_root(tmp_path, workers)
    installed = _spy_latches(monkeypatch)
    if authority == "unreadable":
        import json

        path = task_result_path(tmp_path, ROOT)
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["owner_pause"] = {**stored["owner_pause"], "state": "not-a-fence-state"}
        assert queue.persist_queue_snapshot(reason="test_before_restart")
        path.write_text(json.dumps(stored), encoding="utf-8")
    elif authority == "missing":
        assert queue.persist_queue_snapshot(reason="test_before_restart")
        task_result_path(tmp_path, ROOT).unlink()

    _restart(queue, workers, monkeypatch)
    latch = queue.BUDGET_ROOT_FENCES.get(ROOT)
    evidence = {"child_marker": observed_marker, "fence_a": fence_a, "fence_b": fence_b,
                "installed": installed, "latch_after_restore": dict(latch or {}),
                "child_keyed_latch": queue.BUDGET_ROOT_FENCES.get(CHILD)}

    if authority == "released":
        # Every consequence is observed before any assertion, so a RED run names them all.
        sibling = _pending(workers, SIBLING)
        evidence["sibling_fenced"] = budget_pause_fact(sibling) is not None
        evidence["sibling_dispatch_allowed"] = budget_resume_dispatch_allowed(queue, sibling)
        workers.assign_tasks()
        evidence["dispatched_first"] = [task["id"] for task in sent]
        # Surviving positive path: the parked child of the now-terminal root stays
        # owner-resumable; a live root grant is not mandatory for owner selection.
        sent.clear()
        granted = queue.resume_budget_paused_task(CHILD)
        evidence["child_grant"] = {key: granted.get(key) for key in ("ok", "error", "root_task_id")}
        workers.assign_tasks()
        evidence["dispatched_child"] = [task["id"] for task in sent]
        if evidence["dispatched_child"] == [CHILD]:
            evidence["child_handoff"] = {key: sent[0]["_budget_pause_resume"].get(key) for key in (
                "selected_by", "root_grant_id", "root_fence_id")}
            _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
            evidence["child_consumed"] = True
        _record(f"owner_restore[{authority}]", evidence)
        assert evidence["latch_after_restore"] == {}, evidence
        assert not evidence["sibling_fenced"] and evidence["sibling_dispatch_allowed"], evidence
        assert evidence["dispatched_first"] == [SIBLING], evidence
        assert evidence["child_grant"]["ok"] is True and evidence.get("child_consumed"), evidence
    elif authority == "newer_closed":
        granted = queue.resume_budget_paused_task(ROOT)
        evidence["root_grant"] = {key: granted.get(key) for key in ("ok", "error")}
        evidence["latch_after_root_resume"] = queue.BUDGET_ROOT_FENCES.get(ROOT)
        _record(f"owner_restore[{authority}]", evidence)
        assert evidence["latch_after_restore"].get("fence_id") == fence_b, evidence
        assert all(fence_id == fence_b for root, fence_id in installed if root == ROOT), evidence
        assert evidence["root_grant"]["ok"] and evidence["latch_after_root_resume"] is None, \
            ("the root's Resume of B lifts its latch", evidence)
    else:
        # No positive evidence that the owner's Pause ended: the latch stays (fail closed).
        _record(f"owner_restore[{authority}]", evidence)
        assert evidence["latch_after_restore"].get("cause") == "owner_pause", evidence


def test_a_child_the_restore_parks_after_the_owner_release_raises_no_latch(tmp_path, monkeypatch):
    """The child saved its owner pause but its park event never ran (a RUNNING row,
    so the root's Resume waits on it): the first restore parks the child from its
    durable row under the still-closed A. The root's Resume then releases A, and a
    second restore must not raise A again from that restore-created marker."""
    from ouroboros import budget_pause, owner_pause
    from supervisor.events_budget import budget_resume_dispatch_allowed
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import budget_pause_fact

    queue, _state, workers = _tree(tmp_path, monkeypatch, sibling=True)
    assert request_owner_pause(ROOT, request_id="pause-a")["ok"]
    fence_a = owner_pause.read_fence(tmp_path, ROOT)["fence_id"]
    child_ctx, child_limit = _member_ctx(tmp_path)
    assert _owner_park(child_limit)["root_task_id"] == ROOT  # saved; its park event never ran
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(_sup(tmp_path, queue, workers), ROOT, _owner_park(root_limit))
    refused = queue.resume_budget_paused_task(ROOT)
    assert refused["error"] == "owner_pause_effects_unsettled", refused

    _restart(queue, workers, monkeypatch)
    child = _pending(workers, CHILD)
    assert child["_budget_pause"]["reason"] == "owner" and child["_budget_pause"]["root_task_id"] == ROOT
    assert queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"] == fence_a, "closed A: its latch is due"
    sent = _workers(workers)
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
    sent.clear()
    assert owner_pause.read_fence(tmp_path, ROOT)["state"] == owner_pause.FENCE_RELEASED
    assert queue.resume_budget_paused_task(SIBLING)["ok"]
    _complete_root(tmp_path, workers)

    _restart(queue, workers, monkeypatch)
    sibling = _pending(workers, SIBLING)
    evidence = {"latch_after_restore": dict(queue.BUDGET_ROOT_FENCES.get(ROOT) or {}),
                "sibling_fenced": budget_pause_fact(sibling) is not None,
                "sibling_dispatch_allowed": budget_resume_dispatch_allowed(queue, sibling)}
    workers.assign_tasks()
    evidence["dispatched_first"] = [task["id"] for task in sent]
    sent.clear()
    granted = queue.resume_budget_paused_task(CHILD)
    evidence["child_grant"] = {key: granted.get(key) for key in ("ok", "error", "root_task_id")}
    workers.assign_tasks()
    evidence["dispatched_child"] = [task["id"] for task in sent]
    _record("owner_restore[parked_by_restore_then_released]", evidence)
    assert evidence["latch_after_restore"] == {}, evidence
    assert not evidence["sibling_fenced"] and evidence["sibling_dispatch_allowed"], evidence
    assert evidence["dispatched_first"] == [SIBLING], evidence
    # Surviving positive path: the restored child stays exactly paused and owner-resumable.
    assert evidence["child_grant"] == {"ok": True, "error": None, "root_task_id": ROOT}, evidence
    assert evidence["dispatched_child"] == [CHILD], evidence
    _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
    assert budget_pause.budget_pause_row(tmp_path, CHILD)["state"] == budget_pause.STATE_RESUMED


@pytest.mark.parametrize("authority", ["closed", "released"])
def test_a_self_rooted_owner_marker_restores_from_its_queue_roots_fence(tmp_path, monkeypatch, authority):
    from ouroboros import budget_pause, owner_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _tree(tmp_path, monkeypatch)
    sup = _sup(tmp_path, queue, workers)
    assert request_owner_pause(ROOT, request_id="pause-a")["ok"]
    fence_a = owner_pause.read_fence(tmp_path, ROOT)["fence_id"]
    _child_ctx, child_limit = _member_ctx(tmp_path)
    child_limit.tools._ctx._owner_pause_fence_id = fence_a
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:  # the pre-fix writer's root
        budget_pause.request_pause(child_limit, rail=budget_pause.RAIL_OWNER_PAUSE, scope="root",
                                   reason_text="The owner paused this task tree.", root_task_id=CHILD)
    budget_pause.end_dispatch_fence(CHILD)
    _install(sup, CHILD, raised.value.pause)
    assert _pending(workers, CHILD)["_budget_pause"]["root_task_id"] == CHILD
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(sup, ROOT, _owner_park(root_limit))
    if authority == "released":
        sent = _workers(workers)
        assert queue.resume_budget_paused_task(ROOT)["ok"]
        workers.assign_tasks()
        _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
        _complete_root(tmp_path, workers)
        assert owner_pause.read_fence(tmp_path, ROOT)["state"] == owner_pause.FENCE_RELEASED
    installed = _spy_latches(monkeypatch)

    _restart(queue, workers, monkeypatch)
    evidence = {"installed": installed, "latch": dict(queue.BUDGET_ROOT_FENCES.get(ROOT) or {}),
                "child_marker_fence": _pending(workers, CHILD)["_budget_pause"].get("fence_id")}
    _record(f"self_rooted_owner_restore[{authority}]", evidence)
    assert all(root == ROOT for root, _fence_id in installed), evidence
    if authority == "closed":
        assert evidence["latch"]["fence_id"] == fence_a and evidence["latch"]["cause"] == "owner_pause", evidence
        assert evidence["child_marker_fence"] == fence_a, evidence
    else:
        assert installed == [] and evidence["latch"] == {}, evidence
        # The original incident had this historical self-rooted marker beneath
        # a now-terminal root. Test actual recovery, not only absence of a latch.
        sent = _workers(workers)
        granted = queue.resume_budget_paused_task(CHILD)
        assert granted["ok"] and granted["root_task_id"] == ROOT, granted
        workers.assign_tasks()
        assert [task["id"] for task in sent] == [CHILD]
        _consume(monkeypatch, _child_ctx, child_limit, sent[0]["_budget_pause_resume"])
        from ouroboros.usage_accounting import AttemptRequest, UsageScope, mark_dispatched, reserve_attempt, usage_scope
        from ouroboros.llm_attempt import require_physical_dispatch_window

        monkeypatch.setattr("ouroboros.usage_accounting._reservation_cost", lambda _r: 0.01)
        with usage_scope(UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
            reservation = reserve_attempt(AttemptRequest(model="m", provider="p", task_id=CHILD))
            require_physical_dispatch_window()
            mark_dispatched(reservation)


# --- (4) the same restore with a MONETARY root latch -----------------------------------

def _snapshot_latch_roots(tmp_path):
    import json

    snap = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    return sorted(str(row.get("root_task_id") or "") for row in snap.get("budget_root_fences") or [])


def _restart_from(queue, workers, monkeypatch, snapshot_text):
    """A new process that finds THIS older snapshot (its later publications were lost)."""
    from ouroboros import owner_pause

    _acknowledged_restart(queue, workers)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    for worker in workers.WORKERS.values():
        worker.busy_task_id = None
    monkeypatch.setattr(owner_pause, "_CACHE", {})
    queue.QUEUE_SNAPSHOT_PATH.write_text(snapshot_text, encoding="utf-8")
    return queue.restore_pending_from_snapshot()


def _monetary_root_and_child(tmp_path, monkeypatch, *, child_parks=True, root_parks=True, sibling=False):
    """A child's planning pause latches the running root (monetary F); the root parks too.
    ``child_parks=False``: the child saved its pause, but its park event never ran."""
    queue, _state, workers = _tree(tmp_path, monkeypatch, sibling=sibling)
    sup = _sup(tmp_path, queue, workers)
    child_ctx, child_limit = _member_ctx(tmp_path)
    child_row = _planning_park(child_limit)
    fence_f = ""
    if child_parks:
        _install(sup, CHILD, child_row)
        fence_f = queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"]
        assert "cause" not in queue.BUDGET_ROOT_FENCES[ROOT], "a monetary latch"
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    if root_parks:
        _install(sup, ROOT, _planning_park(root_limit))
    return queue, workers, sup, (child_ctx, child_limit), (root_ctx, root_limit), fence_f


def _resume_and_run_root(monkeypatch, queue, workers, root_ctx, root_limit):
    sent = _workers(workers)
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    assert ROOT not in queue.BUDGET_ROOT_FENCES, "the root's own Resume lifted F"
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
    sent.clear()
    return sent


@pytest.mark.parametrize("history", ["root_resume_consumed", "root_grant_undispatched", "child_parked_at_restore"])
def test_restore_raises_a_monetary_latch_only_for_a_pause_the_snapshot_has_not_seen(tmp_path, monkeypatch, history):
    """The snapshot's fence map is the latch authority; a missing latch is raised again
    only for a pause it never recorded.

    - ``root_resume_consumed``: the root's own consumed Resume lifted F and the root
      completed; the parked child's old marker must not revive F, and the child stays
      exactly paused and owner-resumable;
    - ``root_grant_undispatched`` (control): the restart comes before the root's grant
      reached a worker; the root returns to its pause, so its latch is due again;
    - ``child_parked_at_restore`` (control): the child saved its pause but its park
      event never ran (a RUNNING row in the snapshot); its pause is new to the map.
    """
    from ouroboros import budget_pause

    parked = history != "child_parked_at_restore"
    queue, workers, _sup_ctx, (child_ctx, child_limit), (root_ctx, root_limit), fence_f = \
        _monetary_root_and_child(tmp_path, monkeypatch, child_parks=parked, root_parks=parked)
    if not parked:
        assert CHILD in workers.RUNNING and not queue.BUDGET_ROOT_FENCES
    elif history == "root_resume_consumed":
        _resume_and_run_root(monkeypatch, queue, workers, root_ctx, root_limit)
        _complete_root(tmp_path, workers)
    else:
        assert queue.resume_budget_paused_task(ROOT)["ok"]
        assert ROOT not in queue.BUDGET_ROOT_FENCES
    assert queue.persist_queue_snapshot(reason="test_before_restart")
    facts = {"snapshot_latch_roots": _snapshot_latch_roots(tmp_path), "fence_f": fence_f}
    installed = _spy_latches(monkeypatch)

    _restart(queue, workers, monkeypatch)
    latch = queue.BUDGET_ROOT_FENCES.get(ROOT)
    facts.update(latch_after_restore=dict(latch or {}), installed=installed,
                 pending_markers={task["id"]: (task.get("_budget_pause") or {}).get("root_task_id")
                                  for task in workers.PENDING})
    if history == "root_resume_consumed":
        sent = _workers(workers)
        granted = queue.resume_budget_paused_task(CHILD)
        facts["child_grant"] = {key: granted.get(key) for key in ("ok", "error", "root_task_id")}
        workers.assign_tasks()
        facts["dispatched"] = [task["id"] for task in sent]
    _record(f"monetary_restore[{history}]", facts)
    assert facts["snapshot_latch_roots"] == [], facts
    assert facts["pending_markers"].get(CHILD) == ROOT, facts
    if history == "root_resume_consumed":
        assert latch is None and installed == [], ("a lifted latch stays lifted", facts)
        assert facts["child_grant"] == {"ok": True, "error": None, "root_task_id": ROOT}, facts
        assert facts["dispatched"] == [CHILD], facts
        _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
        assert budget_pause.budget_pause_row(tmp_path, CHILD)["state"] == budget_pause.STATE_RESUMED
    else:
        assert latch is not None and "cause" not in latch, facts
        if history == "root_grant_undispatched":
            assert facts["pending_markers"].get(ROOT) == ROOT and latch["fence_id"] == fence_f, facts


@pytest.mark.parametrize("child_state", ["pending_dispatch", "running_unconsumed"])
def test_a_child_resumed_after_the_roots_release_returns_to_its_pause_without_the_latch(
        tmp_path, monkeypatch, child_state):
    """The root's Resume lifted F; the owner then Resumes the parked child, and a restart
    comes before the child consumed its grant. The child returns to the SAME pause; that
    old marker is no reason to raise the released latch again."""
    from ouroboros import budget_pause

    queue, workers, _sup_ctx, (child_ctx, child_limit), (root_ctx, root_limit), fence_f = \
        _monetary_root_and_child(tmp_path, monkeypatch)
    sent = _resume_and_run_root(monkeypatch, queue, workers, root_ctx, root_limit)
    _complete_root(tmp_path, workers)
    pause_id = budget_pause.budget_pause_row(tmp_path, CHILD)["pause_id"]
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    if child_state == "running_unconsumed":
        workers.assign_tasks()
        assert [task["id"] for task in sent] == [CHILD]
        sent.clear()
    installed = _spy_latches(monkeypatch)

    _restart(queue, workers, monkeypatch)
    row = budget_pause.budget_pause_row(tmp_path, CHILD)
    child = _pending(workers, CHILD)
    facts = {"fence_f": fence_f, "installed": installed, "latch": queue.BUDGET_ROOT_FENCES.get(ROOT),
             "child_marker_pause": (child["_budget_pause"].get("checkpoint") or {}).get("pause_id"),
             "row": {"state": row["state"], "revoked": bool(row["grant"].get("revoked_at"))}}
    granted = queue.resume_budget_paused_task(CHILD)
    facts["regrant"] = {key: granted.get(key) for key in ("ok", "error")}
    _record(f"child_resumed_after_release[{child_state}]", facts)
    assert facts["latch"] is None and installed == [], facts
    assert facts["child_marker_pause"] == pause_id, facts
    assert facts["row"] == {"state": budget_pause.STATE_PAUSED, "revoked": True}, facts
    assert facts["regrant"] == {"ok": True, "error": None}, facts


@pytest.mark.parametrize("cause", ["monetary", "owner_authority_unreadable"])
def test_restore_keeps_the_snapshots_newer_latch_over_an_older_markers_identity(tmp_path, monkeypatch, cause):
    """The map holds the CURRENT latch B; a parked child's marker still names the older
    A. Restore keeps B (monetary marker, or an owner marker whose durable authority is
    unreadable) and never installs A, not even transiently."""
    import json

    from ouroboros.task_results import task_result_path
    from supervisor.owner_pause_control import request_owner_pause

    if cause == "monetary":
        queue, workers, sup, _child, (root_ctx, root_limit), fence_a = _monetary_root_and_child(tmp_path, monkeypatch)
        _resume_and_run_root(monkeypatch, queue, workers, root_ctx, root_limit)
        _install(sup, ROOT, _planning_park(root_limit))  # the root's new pause: a new latch B
    else:
        queue, workers, _sent, _marker, fence_a, _child, (root_ctx, root_limit) = \
            _owner_paused_then_root_resumed(tmp_path, monkeypatch)
        assert request_owner_pause(ROOT, request_id="pause-b")["ok"]
        _install(_sup(tmp_path, queue, workers), ROOT, _owner_park(root_limit))
    fence_b = queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"]
    assert fence_b != fence_a and _pending(workers, CHILD)["_budget_pause"]["fence_id"] == fence_a
    assert queue.persist_queue_snapshot(reason="test_before_restart")
    if cause == "owner_authority_unreadable":
        path = task_result_path(tmp_path, ROOT)
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["owner_pause"] = {**stored["owner_pause"], "state": "not-a-fence-state"}
        path.write_text(json.dumps(stored), encoding="utf-8")
    snapshot = queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8")
    installed = _spy_latches(monkeypatch)

    _restart_from(queue, workers, monkeypatch, snapshot)
    facts = {"fence_a": fence_a, "fence_b": fence_b, "installed": installed,
             "latch": dict(queue.BUDGET_ROOT_FENCES.get(ROOT) or {}),
             "markers": {task["id"]: task["_budget_pause"].get("fence_id") for task in workers.PENDING
                         if task.get("_budget_pause")}}
    _record(f"newer_latch_kept[{cause}]", facts)
    assert facts["latch"].get("fence_id") == fence_b, facts
    assert all(fence_id == fence_b for root, fence_id in installed if root == ROOT), facts
    assert facts["markers"] == {CHILD: fence_b, ROOT: fence_b}, facts


def test_a_stale_consumed_child_carrier_keeps_its_settlement_and_the_maps_own_latch(tmp_path, monkeypatch):
    """A snapshot older than the durable truth: the child's grant handoff, which its
    worker already CONSUMED, keeps the existing settlement path (fenced as the running
    work it names, never re-armed as a pause), and the map's own latch decides (a queued
    sibling keeps the tree live, so the orphan sweep leaves that latch alone)."""
    from ouroboros import budget_pause
    from ouroboros.cancel_intents import has_active_intent

    queue, workers, _sup_ctx, (child_ctx, child_limit), _root, fence_f = \
        _monetary_root_and_child(tmp_path, monkeypatch, root_parks=False, sibling=True)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    lagging = queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8")
    sent = _workers(workers)
    workers.assign_tasks()
    _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
    installed = _spy_latches(monkeypatch)
    _restart_from(queue, workers, monkeypatch, lagging)
    consumed = {"pending": sorted(task["id"] for task in workers.PENDING), "installed": list(installed),
                "latch": queue.BUDGET_ROOT_FENCES.get(ROOT, {}).get("fence_id"),
                "fenced": has_active_intent(tmp_path, CHILD, strict=True),
                "row_state": budget_pause.budget_pause_row(tmp_path, CHILD)["state"]}
    _record("stale_consumed_carrier", consumed)
    assert consumed["pending"] == [SIBLING] and consumed["fenced"], consumed
    assert consumed["row_state"] == budget_pause.STATE_RESUMED, ("never re-armed", consumed)
    assert consumed["latch"] == fence_f and consumed["installed"] == [], ("the map's own latch", consumed)


def test_an_old_self_rooted_locator_whose_resume_was_consumed_raises_no_latch(tmp_path, monkeypatch):
    """A snapshot that kept only the root's own locator while the durable row already
    records its consumed Resume: the locator stays held for visibility, and being the
    root's own marker is no reason to raise the root's latch."""
    import json

    from ouroboros import budget_pause

    queue, _state, workers = _tree(tmp_path, monkeypatch)
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(_sup(tmp_path, queue, workers), ROOT, _planning_park(root_limit))
    old = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    _resume_and_run_root(monkeypatch, queue, workers, root_ctx, root_limit)
    assert budget_pause.budget_pause_row(tmp_path, ROOT)["state"] == budget_pause.STATE_RESUMED
    old["budget_root_fences"] = []  # an older snapshot: only the root's locator survived
    installed = _spy_latches(monkeypatch)
    _restart_from(queue, workers, monkeypatch, json.dumps(old))
    root = _pending(workers, ROOT)
    facts = {"installed": installed, "latch": queue.BUDGET_ROOT_FENCES.get(ROOT),
             "hold": (root.get("_budget_pause_hold") or {}).get("reason"),
             "marker_root": root["_budget_pause"]["root_task_id"]}
    _record("old_self_rooted_locator", facts)
    assert facts["latch"] is None and installed == [], facts
    assert facts["marker_root"] == ROOT and str(facts["hold"]).startswith("restore_refused:"), facts


@pytest.mark.parametrize("history", ["child_saved_before_park", "root_grant_unused", "child_grant_after_release"])
def test_a_stop_door_records_the_latch_its_park_owes_in_the_final_snapshot(tmp_path, monkeypatch, history):
    """``kill_workers`` parks a saved RUNNING pause and returns unused grants to their
    pauses before the final snapshot. Restore trusts that snapshot's latch map, so the
    stop itself raises the latch a NEW pause owes (a child saved before its park event,
    the root's own unused grant) — and still none for a member's unused grant under a
    latch the root's own Resume released."""
    from types import SimpleNamespace

    from ouroboros import budget_pause

    child_parks = history != "child_saved_before_park"
    queue, workers, _sup_ctx, _child, (root_ctx, root_limit), fence_f = _monetary_root_and_child(
        tmp_path, monkeypatch, child_parks=child_parks, root_parks=child_parks)
    if history == "root_grant_unused":
        assert queue.resume_budget_paused_task(ROOT)["ok"]
        assert ROOT not in queue.BUDGET_ROOT_FENCES
    elif history == "child_grant_after_release":
        _resume_and_run_root(monkeypatch, queue, workers, root_ctx, root_limit)
        _complete_root(tmp_path, workers)
        assert queue.resume_budget_paused_task(CHILD)["ok"]
    workers.WORKERS.clear()
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None), raising=False)
    workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)
    final = queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8")
    facts = {"fence_f": fence_f, "final_snapshot_latches": _snapshot_latch_roots(tmp_path),
             "parked_at_stop": sorted(task["id"] for task in workers.PENDING if task.get("_budget_pause"))}
    installed = _spy_latches(monkeypatch)

    _restart_from(queue, workers, monkeypatch, final)
    latch = queue.BUDGET_ROOT_FENCES.get(ROOT)
    facts.update(latch=dict(latch or {}), installed=installed,
                 child_row_state=budget_pause.budget_pause_row(tmp_path, CHILD)["state"])
    _record(f"stop_door_latch[{history}]", facts)
    if history == "child_grant_after_release":
        assert facts["final_snapshot_latches"] == [] and latch is None, facts
        assert facts["child_row_state"] == budget_pause.STATE_PAUSED, facts
    else:
        assert facts["final_snapshot_latches"] == [ROOT], facts
        assert latch is not None and "cause" not in latch and installed == [], ("the map's own latch", facts)
        assert (ROOT if history == "root_grant_unused" else CHILD) in facts["parked_at_stop"], facts
        if history == "root_grant_unused":
            assert latch["fence_id"] == fence_f, facts


@pytest.mark.parametrize("authority", ["unreadable", "missing"])
def test_historical_self_rooted_owner_marker_keeps_canonical_snapshot_b(tmp_path, monkeypatch, authority):
    import json

    from ouroboros import budget_pause
    from ouroboros.task_results import task_result_path
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, _sent, _marker, fence_a, _child, (_root_ctx, root_limit) = \
        _owner_paused_then_root_resumed(tmp_path, monkeypatch)
    child = _pending(workers, CHILD)
    child["_budget_pause"]["root_task_id"] = CHILD
    row = budget_pause.budget_pause_row(tmp_path, CHILD)
    budget_pause.set_budget_pause(tmp_path, CHILD, {**row, "root_task_id": CHILD},
                                 expected_pause_id=row["pause_id"], expected_state=row["state"])
    assert request_owner_pause(ROOT, request_id="probe-pause-b")["ok"]
    _install(_sup(tmp_path, queue, workers), ROOT, _owner_park(root_limit))
    fence_b = queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"]
    assert fence_a != fence_b
    assert queue.persist_queue_snapshot(reason="probe_unknown_owner_b_selfrooted_marker")
    root_path = task_result_path(tmp_path, ROOT)
    record = json.loads(root_path.read_text(encoding="utf-8"))
    if authority == "unreadable":
        record["owner_pause"]["state"] = "not-a-fence-state"
    else:
        record.pop("owner_pause", None)
    root_path.write_text(json.dumps(record), encoding="utf-8")
    installed = _spy_latches(monkeypatch)
    _restart(queue, workers, monkeypatch)
    facts = {"authority": authority, "fence_a": fence_a, "fence_b": fence_b,
             "installed": installed, "root_latch": queue.BUDGET_ROOT_FENCES.get(ROOT),
             "child_latch": queue.BUDGET_ROOT_FENCES.get(CHILD),
             "child_marker": _pending(workers, CHILD)["_budget_pause"]}
    _record("self_rooted_owner_unknown_b", facts)
    assert all(root == ROOT and fence_id == fence_b for root, fence_id in installed), facts
    assert facts["child_latch"] is None, facts
    assert facts["child_marker"]["fence_id"] == fence_b, facts


@pytest.mark.parametrize("door", ["snapshot_restore", "kill_workers"])
def test_an_unused_root_grant_cannot_replace_a_newer_child_latch(tmp_path, monkeypatch, door):
    """The root's unused grant names F, but a still-running member parks under B
    before the root dispatches. Re-parking that old grant must keep the map's B."""
    queue, workers, sup, _child, _root, fence_f = _monetary_root_and_child(
        tmp_path, monkeypatch, sibling=True)
    sibling = _pending(workers, SIBLING)
    workers.PENDING.remove(sibling)
    workers.RUNNING[SIBLING] = {"task": sibling, "worker_id": 2, "attempt": 1}
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    _ctx, limit = _member_ctx(tmp_path, SIBLING, ROOT)
    _install(sup, SIBLING, _planning_park(limit))
    fence_b = queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"]
    assert fence_b != fence_f
    installed = _spy_latches(monkeypatch)
    if door == "snapshot_restore":
        _restart(queue, workers, monkeypatch)
    else:
        workers.WORKERS.clear()
        monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
        monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None), raising=False)
        workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)
    facts = {"fence_f": fence_f, "fence_b": fence_b, "installed": installed,
             "latch": queue.BUDGET_ROOT_FENCES[ROOT],
             "root_marker": _pending(workers, ROOT)["_budget_pause"]}
    _record(f"unused_root_grant_newer_latch[{door}]", facts)
    assert facts["latch"]["fence_id"] == facts["root_marker"]["fence_id"] == fence_b, facts
    assert all(fence == fence_b for _root, fence in installed), facts
