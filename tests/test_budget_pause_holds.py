"""Durable budget-pause HOLDS, generation-bound grants, restart parking and the Q10
last-fit relaxation (#1196) — the supervisor-side half beside ``test_budget_pause_exact``.

Static authoring note: these tests were WRITTEN against the candidate but NOT
RUN by their author (no runtime imports were permitted in that lane); the
parent's isolated harness is the first execution.

Every refusal to restore, grant or dispatch RETAINS the saved pause: a corrupt
source, an unwritten revocation or an acceptance fence over the root becomes a
typed hold beside the ``_budget_pause`` marker; a fence-lifted zero-dispatch
sibling is held until an explicit selection under its root's LIVE grant; a
RUNNING row whose pause completed before a shutdown is parked, never fenced.
"""

from __future__ import annotations

import pathlib
import time
from types import SimpleNamespace

import pytest
from tests.worker_pool_support import affirm_ready

from tests._budget_pause_exact_helpers import (  # noqa: F401 -- shared fixtures of the exact-pause suite
    _install_queue,
    _loop_ctx,
    _parked,
    _pause,
    _supervisor_ctx,
)


def test_explicit_resume_relaxes_the_last_fit_two_reservation_rail(monkeypatch):
    """Owner Q10: after a graceful Resume the one affordable call is admitted (disclosed),
    never re-paused on the number that paused it; a hard rail keeps both reservations."""
    from ouroboros import loop_budget, task_pacing

    calls = []
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits",
                        lambda **kw: calls.append(kw.get("reservation_count")) or False)
    relaxed = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(_budget_resume_last_fit_relaxed=True)),
                              accumulated_usage={}, round_idx=5)
    assert loop_budget._second_reservation_fits(relaxed, {"x": 1}, True, relaxed=True) is None
    assert relaxed.accumulated_usage["budget_resume_last_fit_admitted"] == {
        "round_idx": 5, "reservations_affordable": 1, "basis": "owner_resume_relaxed_last_fit"}
    assert calls == [2]
    strict = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace()), accumulated_usage={}, round_idx=5)
    assert loop_budget._second_reservation_fits(strict, {"x": 1}, True, relaxed=False) is False
    assert "budget_resume_last_fit_admitted" not in strict.accumulated_usage
    # A harder stop (one reservation does not fit) already decides: no probe at all.
    assert loop_budget._second_reservation_fits(strict, {"x": 1}, False, relaxed=True) is None
    assert calls == [2, 2]


def test_restore_refusal_is_typed_and_an_installation_flag_is_not_work_authority(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    task, _row = _parked(tmp_path, monkeypatch, task_id="typed-1")
    assert budget_pause.budget_pause_restore_refusal(tmp_path, task) == ""
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic")
    assert budget_pause.budget_pause_restore_refusal(tmp_path, task) == ""  # restorable; not dispatchable
    assert queue.resume_budget_paused_task("typed-1")["ok"]
    (tmp_path / "state" / "panic_stop.flag").unlink()
    assert budget_pause.budget_pause_restore_refusal(tmp_path, {"id": "typed-1"}) == budget_pause.RESTORE_REFUSAL_NOT_EXACT
    assert budget_pause.budget_pause_restore_refusal(tmp_path, {
        "id": "typed-1", "_budget_pause": {"exact_continuation": True, "checkpoint": {"pause_id": "other"}},
    }) == budget_pause.RESTORE_REFUSAL_IDENTITY_MISMATCH
    assert budget_pause.budget_pause_restore_refusal(tmp_path, {
        "id": "absent", "_budget_pause": {"exact_continuation": True, "checkpoint": {"pause_id": "p"}},
    }) == budget_pause.RESTORE_REFUSAL_RECORD_MISSING


def _source_file(tmp_path, task_id, row):
    from ouroboros.artifacts import task_artifact_dir_path

    return task_artifact_dir_path(tmp_path, task_id, create=False).joinpath(
        *pathlib.PurePosixPath(str(row["source_ref"]["path"])).parts)


def test_unrestorable_source_holds_the_row_with_its_marker_and_a_later_grant_releases_it(tmp_path, monkeypatch):
    """A corrupt/missing source never drops or cancels the saved pause: the row is
    HELD beside its marker, refuses Resume typed while unreadable, and the grant
    that re-validates a readable source releases the hold."""
    from ouroboros import budget_pause
    from ouroboros.work_stop import continuation_refusal

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id="corrupt-1")
    source = _source_file(tmp_path, "corrupt-1", row)
    original = source.read_bytes()
    source.unlink()
    assert budget_pause.budget_pause_restore_refusal(tmp_path, task) == budget_pause.RESTORE_REFUSAL_SOURCE_UNREADABLE
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 1
    held = workers.PENDING[0]
    assert held["id"] == "corrupt-1" and held["_budget_pause"]["exact_continuation"] is True
    hold = continuation_refusal(held)
    assert hold["reason"] == "restore_refused:pause_source_unreadable" and "release" not in hold
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t)))))
    workers.assign_tasks()
    assert sent == []  # held, not assignable
    assert queue.resume_budget_paused_task("corrupt-1")["error"] == "pause_source_unreadable"
    assert "_budget_pause" in held and bool(continuation_refusal(held))  # retained, still held
    source.write_bytes(original)
    granted = queue.resume_budget_paused_task("corrupt-1")
    assert granted["ok"] is True and "released_hold" not in granted
    assert not continuation_refusal(held) and "_budget_pause_hold" not in held and "_budget_pause_resume" in held
    from ouroboros.task_results import load_task_result
    assert "refusal" not in load_task_result(tmp_path, "corrupt-1")["resource_limit"]
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["corrupt-1"]


def test_orphaned_unwritten_revocation_is_written_by_the_next_resume_before_a_new_grant(tmp_path, monkeypatch):
    """A revocation the queue could not write HOLDS the row with its marker (the
    handoff leaves with the hold, so that grant is never dispatched); the next
    Resume writes the deferred revocation, then mints generation 2."""
    from ouroboros import budget_pause
    from supervisor.budget_resume import revoke_exact_budget_resume
    from ouroboros.work_stop import continuation_refusal

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="orphan-1")
    assert queue.resume_budget_paused_task("orphan-1")["ok"] is True
    first_grant = task["_budget_pause_resume"]["grant_id"]
    real_writer = budget_pause.set_budget_pause
    monkeypatch.setattr(budget_pause, "set_budget_pause",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only file system")))
    assert revoke_exact_budget_resume(task, "budget_exhausted_before_dispatch") is False
    monkeypatch.setattr(budget_pause, "set_budget_pause", real_writer)
    hold = continuation_refusal(task)
    assert hold["reason"] == "resume_grant_revocation_unwritten" and hold["grant_id"] == first_grant
    assert task["_budget_pause"]["exact_continuation"] is True and "_budget_pause_resume" not in task
    assert budget_pause.budget_pause_row(tmp_path, "orphan-1")["state"] == budget_pause.STATE_RESUME_GRANTED
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t)))))
    workers.assign_tasks()
    assert sent == []  # a stale grant never dispatches
    second = queue.resume_budget_paused_task("orphan-1")
    assert second["ok"] is True and second["grant_id"] != first_grant
    assert second["grant_generation"] == 2 and not continuation_refusal(task)
    row = budget_pause.budget_pause_row(tmp_path, "orphan-1")
    assert row["grant"]["grant_id"] == second["grant_id"] and row["resume_generation"] == 2
    ctx, _limit = _loop_ctx(tmp_path, "orphan-1")
    with pytest.raises(ValueError):  # the orphaned grant is dead for good
        budget_pause.load_budget_pause(ctx, {"pause_id": row["pause_id"], "grant_id": first_grant})
    workers.assign_tasks()
    assert [t["_budget_pause_resume"]["grant_id"] for t in sent] == [second["grant_id"]]


def test_root_re_resume_rebinds_fence_lifted_sibling_holds_to_the_live_grant(tmp_path, monkeypatch):
    """pauseA -> Resume -> pauseB -> Resume: a zero-dispatch sibling held from the first
    Resume is re-bound to the live grant and selectable by the model under it."""
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, budget_hold_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    root, _row = _parked(tmp_path, monkeypatch, task_id="root-3", scope="root")
    sibling = {"id": "sib-3", "type": "task", "chat_id": 0, "root_task_id": "root-3",
               "parent_task_id": "root-3", "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(sibling)
    first = queue.resume_budget_paused_task("root-3")
    assert first["ok"] is True and first["held_siblings"] == ["sib-3"]
    hold = budget_hold_fact(sibling)
    assert hold["reason"] == HOLD_ROOT_FENCE_LIFTED and hold["root_grant_id"] == first["grant_id"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t)))))
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["root-3"]  # the sibling waits for an explicit selection
    pause_a = sent[0]["_budget_pause_resume"]["pause_id"]
    # The root pauses AGAIN (pause B, a new pause id) and the owner resumes it again.
    workers.RUNNING.clear()
    workers.PENDING[:] = [sibling]
    _root_b, row_b = _parked(tmp_path, monkeypatch, task_id="root-3", scope="root")
    assert row_b["pause_id"] != pause_a
    second = queue.resume_budget_paused_task("root-3")
    assert second["ok"] is True and second["grant_id"] != first["grant_id"]
    assert second["rebound_held_siblings"] == ["sib-3"]
    assert budget_hold_fact(sibling)["root_grant_id"] == second["grant_id"]
    # The model's selection is granted under the LIVE root grant only.
    selected = queue.resume_budget_paused_task("sib-3", selected_by="root-3")
    assert selected["ok"] is True and selected["selection"] == "budget_hold_released"
    assert budget_hold_fact(sibling) is None and sibling["_resume_selection"]["selected_by"] == "root-3"


def test_model_issued_child_selection_needs_the_roots_live_grant(tmp_path, monkeypatch):
    """Owner Q9: lineage alone never revives a paused descendant; the child is granted
    only under its root's live owner-derived Resume, and the grant names the requester."""
    from ouroboros import budget_pause
    from supervisor.events_budget import _handle_budget_resume_child

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    root, _r = _parked(tmp_path, monkeypatch, task_id="root-6", scope="root")
    child, _c = _parked(tmp_path, monkeypatch, task_id="child-6", root_task_id="root-6")
    child["parent_task_id"] = "root-6"
    pushed = []
    sctx = _supervisor_ctx(tmp_path, workers, queue, [], pushed)
    request = {"type": "budget_resume_child", "task_id": "child-6", "requested_by": "root-6", "reason": "still needed"}
    _handle_budget_resume_child(dict(request), sctx)
    assert pushed[-1]["type"] == "budget_resume_child_outcome"
    assert pushed[-1]["ok"] is False and pushed[-1]["error"] == "root_resume_grant_missing"
    assert "_budget_pause" in child and "_budget_pause_resume" not in child
    assert queue.resume_budget_paused_task("root-6")["ok"] is True
    _handle_budget_resume_child(dict(request), sctx)
    assert pushed[-1]["ok"] is True and child["_budget_pause_resume"]["grant_id"]
    row = budget_pause.budget_pause_row(tmp_path, "child-6")
    root_grant = budget_pause.budget_pause_row(tmp_path, "root-6")["grant"]["grant_id"]
    assert row["grant"]["selected_by"] == "root-6" and row["grant"]["root_grant_id"] == root_grant


def test_restart_parks_a_running_row_whose_pause_was_already_complete(tmp_path, monkeypatch):
    """A RUNNING row at shutdown whose durable pause (row + source) was complete is parked
    under its exact marker, its root fence raised — never handed to the shutdown cancel fence."""
    from ouroboros import budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, _limit, _pause_row = _pause(tmp_path, monkeypatch, task_id="park-1", scope="root")
    budget_pause.end_dispatch_fence("park-1")
    task = {"id": "park-1", "type": "task", "chat_id": 0, "root_task_id": "park-1", "_attempt": 1, "_queue_seq": 3}
    workers.RUNNING["park-1"] = {"task": task, "worker_id": 0, "attempt": 1, "started_at": time.time(),
                                 "last_heartbeat_at": time.time(), "last_progress_at": time.time()}
    queue.persist_queue_snapshot(reason="test")
    workers.RUNNING.clear()
    workers.PENDING[:] = []
    terminalized = []
    assert queue.restore_pending_from_snapshot(terminalized=terminalized) == 1
    assert terminalized == []
    parked = workers.PENDING[0]
    assert parked["id"] == "park-1" and parked["_budget_pause"]["exact_continuation"] is True
    assert parked["_budget_pause"]["fence_id"] and queue.BUDGET_ROOT_FENCES["park-1"]["status"] == "paused"
    row = budget_pause.budget_pause_row(tmp_path, "park-1")
    assert row["state"] == budget_pause.STATE_PAUSED and row["pause_source"] == "restart_during_pausing"


def test_a_consumed_grant_is_never_re_armed_by_a_later_revocation(tmp_path, monkeypatch):
    """Negative: money vanishing (or a restart) AFTER the loop consumed its grant must not
    put the task back on the pause path. The queue row is a stale carrier: its handoff
    leaves, no ``_budget_pause`` marker is re-minted over a task that is not paused, the
    durable RESUMED row is not written over, and the task's own status is not rewritten."""
    from ouroboros.work_stop import continuation_refusal
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result
    from supervisor.budget_resume import revoke_exact_budget_resume
    from supervisor.events_budget import HOLD_GRANT_CONSUMED_STALE_ROW

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="loop-4")
    assert queue.resume_budget_paused_task("loop-4")["ok"] is True
    assert "_budget_pause" not in task and isinstance(task["_budget_pause_resume"], dict)
    granted = budget_pause.budget_pause_row(tmp_path, "loop-4")
    grant_id = str((granted.get("grant") or {}).get("grant_id") or "")
    # The worker consumed the grant: the durable row says RESUMED, as the loop writes it.
    consumed_grant = {**dict(granted.get("grant") or {}), "consumed_at": time.time()}
    budget_pause.set_budget_pause(
        tmp_path, "loop-4", {**granted, "state": budget_pause.STATE_RESUMED, "grant": consumed_grant},
        expected_pause_id=str(granted.get("pause_id") or ""),
        expected_state=budget_pause.STATE_RESUME_GRANTED, expected_grant_id=grant_id)

    assert revoke_exact_budget_resume(task, "money_vanished") is False
    assert "_budget_pause_resume" not in task
    assert set(task["_budget_pause"]) == {"refusal"}  # no new pause identity or executable source
    assert "_budget_pause_consumed" not in task
    assert continuation_refusal(task)["grant_id"] == grant_id
    assert continuation_refusal(task)["reason"] == HOLD_GRANT_CONSUMED_STALE_ROW
    after = budget_pause.budget_pause_row(tmp_path, "loop-4")
    assert after["state"] == budget_pause.STATE_RESUMED and after["grant"]["consumed_at"]
    assert load_task_result(tmp_path, "loop-4", strict=True)["status"] != "cancelled"


def test_malformed_acceptance_fence_evidence_retains_a_saved_pause_but_fails_closed_elsewhere(
        tmp_path, monkeypatch):
    """Corrupt acceptance-fence evidence fails CLOSED: an ordinary row that cannot prove
    its root's review state is cancelled rather than started. A saved exact budget pause
    is the one exception — retained as a typed, non-dispatchable hold carrying its
    ORIGINAL locator, never cancelled, until the owner's next Resume re-validates it."""
    from ouroboros.work_stop import continuation_refusal
    from ouroboros.task_results import load_task_result
    from supervisor.events_budget import HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _paused, row = _parked(tmp_path, monkeypatch, task_id="child-6", root_task_id="root-6")
    workers.PENDING.append({"id": "plain-6", "type": "task", "chat_id": 0,
                            "root_task_id": "root-6", "_attempt": 1})
    # An ACTIVE fence with no root id is evidence nothing can be proven from.
    queue.ACCEPTANCE_FENCES["root-6"] = {"status": "active", "token": "tok", "generation": 1}
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    try:
        assert queue.restore_pending_from_snapshot() == 1
    finally:
        queue.ACCEPTANCE_FENCES.pop("root-6", None)
    assert [task["id"] for task in workers.PENDING] == ["child-6"]
    held = workers.PENDING[0]
    assert held["_budget_pause"]["exact_continuation"] is True
    # The ORIGINAL locator survives: the checkpoint still names the exact saved pause.
    assert held["_budget_pause"]["checkpoint"]["pause_id"] == row["pause_id"]
    assert held["_budget_pause"]["checkpoint"]["source_ref"] == row["source_ref"]
    assert continuation_refusal(held)["reason"] == HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT
    assert load_task_result(tmp_path, "child-6", strict=True)["status"] != "cancelled"
    # The ordinary row failed closed: not restored, and terminalized as cancelled.
    assert load_task_result(tmp_path, "plain-6", strict=True)["status"] == "cancelled"


def test_acceptance_fence_at_restore_holds_a_saved_pause_instead_of_cancelling(tmp_path, monkeypatch):
    from ouroboros.work_stop import continuation_refusal
    from supervisor.events_budget import HOLD_ROOT_ACCEPTANCE_FENCED
    from ouroboros.task_results import load_task_result

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    child, _row = _parked(tmp_path, monkeypatch, task_id="child-5", root_task_id="root-5")
    child["parent_task_id"] = "root-5"
    queue.ACCEPTANCE_FENCES["root-5"] = {"status": "active", "root_task_id": "root-5", "token": "tok", "generation": 1}
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    try:
        assert queue.restore_pending_from_snapshot() == 1
    finally:
        queue.ACCEPTANCE_FENCES.pop("root-5", None)
    held = workers.PENDING[0]
    assert held["id"] == "child-5" and held["_budget_pause"]["exact_continuation"] is True
    assert continuation_refusal(held)["reason"] == HOLD_ROOT_ACCEPTANCE_FENCED
    assert load_task_result(tmp_path, "child-5", strict=True)["status"] != "cancelled"


def _fenced_member(workers, task_id, root_task_id):
    member = {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": root_task_id,
              "parent_task_id": root_task_id, "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(member)
    return member


def _idle_worker(workers, sent):
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t)))))


def test_selecting_one_member_of_a_fenced_root_never_lifts_the_latch(tmp_path, monkeypatch):
    """F2: the zero-dispatch branch used to clear the ROOT's admission latch when a
    CHILD was nominated — every sibling became assignable at once, and a
    model-issued request was granted without the root's live Resume grant. The
    nomination is now one explicit per-row selection: the latch stands, the
    unselected siblings stay fenced, and only the selected row dispatches."""
    from supervisor.events_budget import _set_root_budget_pause_locked, budget_hold_fact
    from supervisor.queue_transitions import budget_pause_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    fence = _set_root_budget_pause_locked("root-f2", {"scope": "root", "root_task_id": "root-f2"})
    first = _fenced_member(workers, "m1-f2", "root-f2")
    second = _fenced_member(workers, "m2-f2", "root-f2")

    # A MODEL-issued selection needs the root's live owner Resume grant (Q9).
    refused = queue.resume_budget_paused_task("m1-f2", selected_by="root-f2")
    assert refused["error"] == "root_resume_grant_missing" and refused["action"] == "resume_root_first"
    assert queue.BUDGET_ROOT_FENCES["root-f2"]["fence_id"] == fence["fence_id"]
    assert budget_hold_fact(first) is not None  # held, not released, not dropped

    # The owner's own explicit act selects exactly this row.
    granted = queue.resume_budget_paused_task("m1-f2")
    assert granted["ok"] is True and granted["selection"] == "budget_hold_released"
    assert queue.BUDGET_ROOT_FENCES["root-f2"]["fence_id"] == fence["fence_id"]  # the latch stands
    assert "_budget_pause_hold" not in first and "_resume_selection" in first
    assert first["_resume_selection"]["fence_id"] == fence["fence_id"]
    # The UI truth follows the same rule: one row released, the other still paused.
    assert budget_pause_fact(first) is None
    assert budget_pause_fact(second)["fence_id"] == fence["fence_id"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["m1-f2"]  # no fan-out of the fenced tree


@pytest.mark.parametrize("worker_busy", [False, True])
def test_a_selection_bound_to_an_older_fence_does_not_pre_release_a_new_one(tmp_path, monkeypatch, worker_busy):
    """The selection names the fence generation it was granted against: a root that
    paused AGAIN raises a new latch, and the earlier selection is not a key to it."""
    from supervisor.events_budget import _set_root_budget_pause_locked
    from supervisor.queue_transitions import budget_pause_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _set_root_budget_pause_locked("root-f2b", {"scope": "root", "root_task_id": "root-f2b"})
    member = _fenced_member(workers, "m1-f2b", "root-f2b")
    assert queue.resume_budget_paused_task("m1-f2b")["ok"] is True
    assert budget_pause_fact(member) is None
    queue.BUDGET_ROOT_FENCES.pop("root-f2b", None)
    newer = _set_root_budget_pause_locked("root-f2b", {"scope": "root", "root_task_id": "root-f2b",
                                                       "fence_id": "fence-2"})
    assert newer["fence_id"] == "fence-2"
    assert budget_pause_fact(member)["fence_id"] == "fence-2"
    sent = []
    _idle_worker(workers, sent)
    if worker_busy:
        workers.WORKERS[0].busy_task_id = "other"
    workers.assign_tasks()
    assert sent == []
    assert "_resume_selection" not in member


def test_root_resume_holds_a_fence_bound_child_marker_instead_of_stranding_it(tmp_path, monkeypatch):
    """F2: a child carrying a NON-exact marker minted from the root's own latch was
    skipped by the root's Resume and then refused forever with
    ``root_budget_fence_missing`` — the fence it named was gone. The marker is the
    fence, so it becomes the same hold a fence-only sibling takes, selectable
    under the root's live grant."""
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, budget_hold_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    root, _row = _parked(tmp_path, monkeypatch, task_id="root-f3", scope="root")
    fence = queue.BUDGET_ROOT_FENCES["root-f3"]
    child = _fenced_member(workers, "child-f3", "root-f3")
    child.pop("admitted_dispatch")  # historical positive pause survives conversion to a hold
    child["_budget_pause"] = {"status": "paused_before_dispatch", "scope": "root",
                              "root_task_id": "root-f3", "fence_id": fence["fence_id"],
                              "replay_safe": True, "physical_calls": 0, "auto_resume": False}

    granted = queue.resume_budget_paused_task("root-f3")
    assert granted["ok"] is True and granted["held_siblings"] == ["child-f3"]
    assert "_budget_pause" not in child
    hold = budget_hold_fact(child)
    assert hold["reason"] == HOLD_ROOT_FENCE_LIFTED and "replaced_fence_pause" in hold
    assert hold["root_grant_id"] == granted["grant_id"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["root-f3"]  # eligibility is not continuation

    selected = queue.resume_budget_paused_task("child-f3", selected_by="root-f3")
    assert selected["ok"] is True and selected["selection"] == "budget_hold_released"
    assert budget_hold_fact(child) is None
    workers.WORKERS[0].busy_task_id = None  # the slot the root took is free again
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["root-f3", "child-f3"]


def test_a_consumed_carrier_never_drops_a_newer_owner_wait_continuation(tmp_path, monkeypatch):
    """F3: a spent exact-resume handoff that reaches a revocation (a restore, lost
    money, a new root fence) beside a NEWER owner-wait handoff is retired by the
    revocation seam once the durable grant reads as consumed — fencing the row as
    "consumed" would drop a valid planned-restart continuation. An UNCONSUMED
    grant is never merely retired: it is revoked under its own identity and the
    SAME id returns to its exact pause."""
    from ouroboros import budget_pause, owner_wait
    from supervisor.budget_resume import revoke_exact_budget_resume
    from supervisor.events_budget import budget_hold_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="wait-3")
    assert queue.resume_budget_paused_task("wait-3")["ok"] is True
    first_grant = str(task["_budget_pause_resume"]["grant_id"])

    # An UNCONSUMED grant is revoked, not retired: the restart's revocation
    # re-parks the SAME id under its exact marker.
    assert revoke_exact_budget_resume(task, "restart_before_dispatch") is True
    assert "_budget_pause_resume" not in task and task["_budget_pause"]["exact_continuation"] is True
    revoked = budget_pause.budget_pause_row(tmp_path, "wait-3")
    assert revoked["state"] == budget_pause.STATE_PAUSED and revoked["grant"]["grant_id"] == first_grant
    assert revoked["grant"]["revoke_reason"] == "restart_before_dispatch"

    assert queue.resume_budget_paused_task("wait-3")["ok"] is True
    carrier = dict(task["_budget_pause_resume"])
    granted = budget_pause.budget_pause_row(tmp_path, "wait-3")
    grant_id = str(granted["grant"]["grant_id"])
    assert grant_id != first_grant
    consumed = {**dict(granted["grant"]), "consumed_at": time.time()}
    budget_pause.set_budget_pause(
        tmp_path, "wait-3", {**granted, "state": budget_pause.STATE_RESUMED, "grant": consumed},
        expected_pause_id=str(granted["pause_id"]),
        expected_state=budget_pause.STATE_RESUME_GRANTED, expected_grant_id=grant_id)
    task["_owner_wait_resume"] = {"wait_id": "w-3", "restart_transaction_id": "tx-3",
                                  "task_attempt": 1, "source_ref": {"path": "owner-wait.json"},
                                  "started_at": time.time() - 10.0}
    # The spent carrier is retired beside the newer owner-wait handoff: no hold,
    # no re-minted marker, the durable RESUMED row untouched.
    assert revoke_exact_budget_resume(task, "restart_before_dispatch") is False
    assert "_budget_pause_resume" not in task and task["_owner_wait_resume"]["wait_id"] == "w-3"
    assert "_budget_pause" not in task and "_budget_pause_consumed" not in task
    assert budget_hold_fact(task) is None

    # A snapshot taken BEFORE that retirement still restores the owner wait.
    task["_budget_pause_resume"] = carrier
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    monkeypatch.setattr(owner_wait, "restore_owner_wait_allowed", lambda *_a, **_k: True)
    assert queue.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert restored["id"] == "wait-3" and restored["_owner_wait_resume"]["wait_id"] == "w-3"
    assert "_budget_pause_resume" not in restored and "_budget_pause_consumed" not in restored
    assert budget_hold_fact(restored) is None
    after = budget_pause.budget_pause_row(tmp_path, "wait-3")
    assert after["state"] == budget_pause.STATE_RESUMED  # never re-armed


def test_owner_wait_keeps_the_budget_paused_carrier_across_a_planned_restart(tmp_path, monkeypatch):
    """F5: the ONE serializer carries the paused interval, so a task that was budget
    paused and later parks in an owner wait resumes on the SAME execution clock.
    Without it the restart's finite-lifetime reader charges the pause as execution."""
    from ouroboros import config, model_wait, owner_wait
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "wait-5", STATUS_RUNNING, result="running")
    ctx, _limit = _loop_ctx(tmp_path, "wait-5")
    clock = {"revision": 0, "elapsed_sec": 30.0, "observed_at": time.time(), "active": False}
    ctx.model_wait_context = SimpleNamespace(continuation_state=lambda: {
        "overrides": {}, "auto_continue": {}, "budget_paused_sec": 600.0, "quota_clock": clock})
    handoff = owner_wait.checkpoint_owner_wait(ctx, [], {}, {}, 3, [], set())
    assert handoff["budget_paused_sec"] == 600.0

    now = time.time()
    started = now - 1000.0
    meta = {"started_at": started, "model_wait_quota_clock": handoff["model_wait_quota_clock"],
            "budget_paused_sec": handoff["budget_paused_sec"]}
    # ONE shared clock: 1000s of wall time minus 30s of quota wait minus 600s paused.
    assert model_wait.execution_elapsed_seconds(meta, now) == pytest.approx(370.0, abs=5.0)

    owner_wait.set_owner_wait(tmp_path, "wait-5", {**handoff, "state": "waiting", "started_at": started})
    monkeypatch.setattr("ouroboros.delegate_recovery._read_restart_transaction",
                        lambda _root, _tid: {"status": "normal_exit_acknowledged", "task_ids": ["wait-5"]})
    monkeypatch.setattr("ouroboros.delegate_recovery._ack_direct_exec_successor", lambda _root: None)
    monkeypatch.setattr("ouroboros.cancel_intents.has_active_intent", lambda *_a, **_k: False)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 500.0)
    row = {"id": "wait-5", "_owner_wait_resume": {**handoff, "restart_transaction_id": "tx-5",
                                                  "started_at": started}}
    assert owner_wait.restore_owner_wait_allowed(tmp_path, row) is True
    # The negative control: the same wall clock WITHOUT the carrier is 970s and
    # would read as an exhausted lifetime — the pause must not be charged twice.
    owner_wait.set_owner_wait(tmp_path, "wait-5", {**handoff, "state": "waiting",
                                                   "started_at": started, "budget_paused_sec": 0.0},
                              expected_wait_id=handoff["wait_id"])
    assert owner_wait.restore_owner_wait_allowed(tmp_path, row) is False


def test_an_unwritten_grant_rollback_holds_the_row_instead_of_granting_forever(tmp_path, monkeypatch):
    """F6: when the snapshot cannot be persisted AND the durable rollback cannot be
    written, the row keeps a grant the queue no longer carries. Recorded as the
    EXISTING unwritten-revocation hold, the next Resume writes that revocation
    before minting — instead of answering ``resume_already_granted`` forever."""
    from ouroboros.work_stop import continuation_refusal
    from ouroboros import budget_pause
    from supervisor.events_budget import HOLD_REVOCATION_UNWRITTEN

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="f6-1")
    real_writer = budget_pause.set_budget_pause
    real_snapshot = queue.persist_queue_snapshot

    def _no_rollback(root, task_id, row, expected_pause_id=None, **kwargs):
        if kwargs.get("expected_state") == budget_pause.STATE_RESUME_GRANTED:
            raise OSError("read-only file system")
        return real_writer(root, task_id, row, expected_pause_id, **kwargs)

    monkeypatch.setattr(budget_pause, "set_budget_pause", _no_rollback)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": False)
    refused = queue.resume_budget_paused_task("f6-1")
    assert refused["error"] == "snapshot_not_persisted" and refused["held"] == HOLD_REVOCATION_UNWRITTEN
    hold = continuation_refusal(task)
    assert hold["reason"] == HOLD_REVOCATION_UNWRITTEN and hold["grant_id"] == refused["grant_id"]
    assert task["_budget_pause"]["exact_continuation"] is True and "_budget_pause_resume" not in task
    assert budget_pause.budget_pause_row(tmp_path, "f6-1")["state"] == budget_pause.STATE_RESUME_GRANTED
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == []  # held: the orphaned grant never dispatches

    monkeypatch.setattr(budget_pause, "set_budget_pause", real_writer)
    monkeypatch.setattr(queue, "persist_queue_snapshot", real_snapshot)
    second = queue.resume_budget_paused_task("f6-1")
    assert second["ok"] is True and second["grant_id"] != refused["grant_id"]
    assert second["grant_generation"] == 2
    assert not continuation_refusal(task)
    row = budget_pause.budget_pause_row(tmp_path, "f6-1")
    assert row["grant"]["grant_id"] == second["grant_id"] and row["resume_generation"] == 2


def test_a_root_selected_against_its_retained_fence_reserves_while_siblings_stay_refused(tmp_path, monkeypatch):
    """#1196 review finding 3: a legacy zero-dispatch root Resume records the
    selection but keeps the root latch, and ``reserve_attempt`` used to refuse
    EVERY reservation under that fence — Resume "succeeded", the worker's first
    send hit the fence, and the root re-paused without a model call. Reservation
    admission now honours the selection recorded against the exact fence; the
    unselected siblings are still refused at the same gate (owner Q9)."""
    from ouroboros import usage_accounting as accounting
    from supervisor.events_budget import _set_root_budget_pause_locked

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    root = _fenced_member(workers, "root-r", "root-r")
    root.pop("parent_task_id")
    _fenced_member(workers, "sib-r", "root-r")
    fence = _set_root_budget_pause_locked("root-r", {})
    assert queue.resume_budget_paused_task("root-r")["ok"] is True
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root-r"]
    assert queue.BUDGET_ROOT_FENCES["root-r"]["fence_id"] == fence["fence_id"]  # latch retained

    def _reserve(task_id):
        with accounting.usage_scope(accounting.UsageScope(
                drive_root=tmp_path, task_id=task_id, root_task_id="root-r", global_limit_usd=100.0)):
            return accounting.reserve_attempt(accounting.AttemptRequest(
                model="fixture", provider="openai", reservation_usd=1.0))

    assert _reserve("root-r").attempt_id  # the selected row's first send is admitted
    with pytest.raises(accounting.BudgetExceeded) as refused:
        _reserve("sib-r")
    assert refused.value.limit_scope == "root"
    # A selection recorded against an OLDER fence generation is no key to a new latch.
    queue.BUDGET_ROOT_FENCES["root-r"] = {**fence, "fence_id": "fence-next"}
    queue.persist_queue_snapshot(reason="test")
    with pytest.raises(accounting.BudgetExceeded):
        _reserve("root-r")


def test_owner_wait_restart_assignment_and_the_cold_loop_keep_the_budget_paused_carrier(tmp_path, monkeypatch):
    """#1196 review finding 5: after a budget Resume an owner-wait checkpoint
    stores ``budget_paused_sec`` but assignment read only ``paused_duration_sec``,
    so after a planned restart the RUNNING row charged the old pause as execution;
    the cold owner-wait restore also left ``ctx._budget_paused_sec`` unset. One
    shared reader (``model_wait.budget_paused_seconds``) serves either handoff."""
    from ouroboros import owner_wait

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    started = time.time() - 1000.0
    workers.PENDING.append({
        "id": "wait-6", "type": "task", "chat_id": 0, "_attempt": 1,
        "_owner_wait_resume": {"wait_id": "w6", "restart_transaction_id": "tx-6", "started_at": started,
                               "budget_paused_sec": 600.0, "model_wait_quota_clock": {}},
    })
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["wait-6"]
    meta = workers.RUNNING["wait-6"]
    assert meta["started_at"] == pytest.approx(started) and meta["budget_paused_sec"] == 600.0

    ctx, _limit = _loop_ctx(tmp_path, "wait-6")
    state_blob = {"messages": [], "trace": {}, "usage": {}, "seen": [], "owner_directives": [],
                  "route": {}, "delivery": {}, "acceptance": {}, "delivery_candidate": None,
                  "model_wait": {"budget_paused_sec": 600.0}}
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=ctx), state_blob, [], {}, {}, set())
    assert ctx._budget_paused_sec == 600.0


@pytest.mark.parametrize("race", ["consumed", "unreadable"])
def test_worker_death_reads_the_revokers_hold_before_reparking(tmp_path, monkeypatch, race):
    from ouroboros.work_stop import continuation_refusal
    from ouroboros import budget_pause
    from supervisor import budget_resume, worker_health
    from supervisor.events_budget import HOLD_GRANT_CONSUMED_STALE_ROW, HOLD_RECORD_UNREADABLE_AT_REVOKE
    from tests.test_worker_crash_retry import _make_worker

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    task, _ = _parked(tmp_path, monkeypatch, task_id="death-race")
    assert queue.resume_budget_paused_task(task["id"])["ok"]
    granted = budget_pause.budget_pause_row(tmp_path, task["id"])
    workers.PENDING.remove(task)
    worker = _make_worker(busy_task_id=task["id"], exitcode=1)
    workers.WORKERS[0] = worker
    meta = workers.RUNNING[task["id"]] = {"task": task, "attempt": 1, "worker_id": 0}
    job = {"worker": worker, "task": task, "task_id": task["id"], "meta": meta,
           "attempt": 1, "worker_id": 0, "drive_root": str(tmp_path)}
    revoke = budget_resume.revoke_exact_budget_resume

    def raced_revoke(row, reason):
        assert reason == "worker_death_before_consumption"
        if race == "consumed":
            budget_pause.set_budget_pause(tmp_path, task["id"], {
                **granted, "state": budget_pause.STATE_RESUMED,
                "grant": {**granted["grant"], "consumed_at": time.time()}})
            return revoke(row, reason)
        with monkeypatch.context() as reading:
            def unreadable(*_args, **_kwargs):
                raise OSError("pause authority unreadable during revocation")
            reading.setattr(budget_pause, "budget_pause_row", unreadable)
            return revoke(row, reason)

    monkeypatch.setattr(budget_resume, "revoke_exact_budget_resume", raced_revoke)
    parked, fenced = worker_health._complete_exact_budget_pause_after_death(
        job, tmp_path, task, task["id"], 1)
    assert fenced and parked is (race == "unreadable")
    assert "_budget_pause_consumed" not in task
    expected = HOLD_GRANT_CONSUMED_STALE_ROW if race == "consumed" else HOLD_RECORD_UNREADABLE_AT_REVOKE
    assert continuation_refusal(task)["reason"] == expected
    if race == "consumed":
        assert not workers.PENDING and workers.RUNNING[task["id"]] is meta
        assert budget_pause.budget_pause_row(tmp_path, task["id"])["state"] == budget_pause.STATE_RESUMED
    else:
        assert not workers.RUNNING and len(workers.PENDING) == 1
        assert workers.PENDING[0]["_budget_pause"]["exact_continuation"]


def test_assignment_keeps_a_paused_roots_reason_over_the_selection_it_ran_under(tmp_path, monkeypatch):
    """The owner's Resume releases a root from its stop hold (``saved_work_hold``) and it runs
    carrying that selection; the owner then Pauses it. Its exact pause and the Pause's latch decide
    the row, so assignment never repairs the selection it ran under as a stale zero-dispatch choice:
    that re-hold rewrote ``owner_paused`` to ``saved_work_hold`` and a Restart then showed the
    paused root as work held after a stop. A zero-dispatch row with the same stale choice is still
    returned to an unselected hold."""
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events_budget import BUDGET_HOLD_KEY, budget_hold_fact, budget_resume_dispatch_allowed, install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.restart_retention import hold_stopped_queue
    from tests._budget_pause_exact_helpers import _fast_hold, _mock_pause_observation
    from tests.test_budget_pause_lineage import _member_ctx, _park, _sup

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    write_task_result(tmp_path, "alpha", "scheduled", chat_id=0, root_task_id="alpha")
    workers.PENDING.append(hold_stopped_queue({"id": "alpha", "type": "task", "chat_id": 0, "_attempt": 1,
                                               "root_task_id": "alpha", "admitted_dispatch": "none"},
                                              tmp_path, "app_stop"))
    assert queue.resume_budget_paused_task("alpha")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["alpha"]
    ran_under = dict(workers.RUNNING["alpha"]["task"]["_resume_selection"])
    assert ran_under["selected_by"] == "owner" and "reason" not in ran_under

    assert request_owner_pause("alpha", request_id="pause-after-resume")["ok"]
    _ctx, limit = _member_ctx(tmp_path, "alpha", "alpha")
    install_exact_budget_pause(_sup(tmp_path, queue, workers), "alpha",
                               budget_pause.exact_pause_marker(_park(limit))["checkpoint"])
    [paused] = workers.PENDING
    assert paused["_budget_pause"]["reason"] == "owner" and paused["_resume_selection"] == ran_under
    assert budget_resume_dispatch_allowed(queue, paused) is False, "the old choice names no Pause latch"
    assert load_task_result(tmp_path, "alpha", strict=True)["reason_code"] == "owner_paused"
    member = {"id": "alpha-child", "type": "task", "chat_id": 0, "_attempt": 1, "root_task_id": "alpha",
              "parent_task_id": "alpha", "admitted_dispatch": "none", "_resume_selection": dict(ran_under)}
    workers.PENDING.append(member)
    assert budget_resume_dispatch_allowed(queue, member) is False

    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["alpha"]
    assert paused["_resume_selection"] == ran_under and "_budget_pause" in paused
    assert load_task_result(tmp_path, "alpha", strict=True)["reason_code"] == "owner_paused"
    assert "_resume_selection" not in member and budget_hold_fact(member)
    assert member[BUDGET_HOLD_KEY]["detail"] == "root_resume_generation_stale"
