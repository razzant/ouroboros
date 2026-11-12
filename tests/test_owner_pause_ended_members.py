"""What an owner Pause owes the delegated work its tree's ended members left (#2669), through the real consumers.

The accept step (``request_owner_pause``), the off-loop settle pass, the tree census, the queue latch at the real
``task_done`` and at Restart's restore, the resumed root's first facts and the room's work projection are driven
directly; only the engine is a stand-in, never a real daemon. A terminal result is not ended execution.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx
from tests._owner_pause_ended_helpers import (
    _Engine, _fence, _kept, _park_root, _run, _tail_tree, _tick,
)
from tests.test_owner_pause import _live_tree


def _start(root):
    from ouroboros import delegate_custody as dc

    assert dc.emit(root, dc.START_REQUESTED, {"invocation_id": "inv-1", "task_id": "child-dead", "root_task_id": "root-1",
                                              "route": "codex", "surface": "delegate", "request": {"prompt": "work"}})


def _snapshot(root):
    """The census rows and the works projection ``/api/state`` builds from them, through their real producers."""
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot

    raw, owners = {}, []
    census = _chat_activities_snapshot_safe(root, work_facts=raw, decision_owners=owners)
    return census, works_snapshot(root, owners, raw)


def _works(root):
    return _snapshot(root)[1]["works"]


def test_a_pause_stops_what_a_failed_nanny_left_at_once_and_holds_until_it_ends(tmp_path, monkeypatch):
    """Running root, terminal nanny, healthy leaf. The press issues nothing; the off-loop pass issues the one stop without
    waiting for the root's boundary; every later look only reads it back; the Pause settles only on a confirmed end."""
    from supervisor.owner_pause_control import request_owner_pause, settlement_look_owed
    from tests.test_owner_pause_participants import _resume_messages

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch)
    ack = request_owner_pause("root-1", request_id="press")
    assert ack["ok"] and ack["state"] == "requested" and ack["members"] == ["root-1"]
    assert engine.opened == 0, "the HTTP door and the queue lock see no network"
    assert settlement_look_owed(queue), "the press wakes the existing pass"

    assert _tick(queue) == [] and engine.cancels == ["run-leaf"], "the root's own boundary is not what the leaf waits for"
    fence = _fence(tmp_path)
    assert fence["state"] == "requested" and fence["request_id"] == "press"
    [row] = _kept(tmp_path)["external_runs"]["runs"]
    assert (row["run_id"], row["state"], row["stop_policy"], row["stop_outcome"]) == (
        "run-leaf", "stop_requested", "request_stop", "requested")
    assert _kept(tmp_path)["episode_runs"][0]["stop_requested"] is True

    # Repeated looks read the same stop back: no second control, and an unchanged fact is not rewritten.
    reads = len(engine.reads)
    assert _tick(queue) == [] and _tick(queue) == []
    assert engine.cancels == ["run-leaf"] and len(engine.reads) > reads
    assert _fence(tmp_path) == fence, "no write, so no timestamp moved"

    _park_root(tmp_path, monkeypatch, queue, workers)
    assert _fence(tmp_path)["state"] == "requested", "the actor parked; the tree's own sent work has not ended"
    refused = queue.resume_budget_paused_task("root-1")
    assert refused["ok"] is False and refused["error"] == "owner_pause_effects_unsettled", refused

    engine.state = "cancelled"
    assert _tick(queue) == ["root-1"]
    settled = _fence(tmp_path)
    assert settled["state"] == "paused" and engine.cancels == ["run-leaf"]
    assert settled["requested_at"] == fence["requested_at"], "the Pause is dated from its press"
    assert queue.BUDGET_ROOT_FENCES["root-1"]["cause"] == "owner_pause"

    # What the resumed root is told names the starter and offers only the doors this root really has.
    notice = "\n".join(_resume_messages(tmp_path, monkeypatch, queue, workers, task_id="root-1"))
    assert "run run-leaf (codex), started by ended task child-dead: this Pause requested its stop" in notice
    assert "engine terminal: cancelled" in notice
    assert "continue_from=" not in notice and 'delegate_wait(run_id="run-leaf")' not in notice
    assert 'get_task_result("child-dead")' in notice and 'integrate_delegated_patch(run_id="run-leaf")' in notice
    assert "may be continued" not in notice


@pytest.mark.parametrize("root_status", ["completed", "failed"])
def test_an_ended_root_with_a_live_tail_takes_the_pause_and_releases_only_on_a_confirmed_end(tmp_path, monkeypatch, root_status):
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status=root_status)
    ack = request_owner_pause("root-1", request_id="press")
    assert ack["ok"] and ack["state"] == "requested" and ack["members"] == [], ack
    assert engine.opened == 0 and queue.BUDGET_ROOT_FENCES["root-1"]["cause"] == "owner_pause"
    first = _fence(tmp_path)

    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    assert _fence(tmp_path)["state"] == "requested" and "root-1" in queue.BUDGET_ROOT_FENCES
    engine.state = "cancelled"
    assert _tick(queue) == []
    fence = _fence(tmp_path)
    assert fence["state"] == "released" and fence["release_reason"] == "late_work_settled"
    assert fence["requested_at"] == first["requested_at"] and "root-1" not in queue.BUDGET_ROOT_FENCES
    assert engine.cancels == ["run-leaf"]
    # Nothing is left: the door's answer for a finished tree again, and no new Pause.
    assert request_owner_pause("root-1", request_id="again") == {"ok": False, "error": "task_not_live"}
    assert _fence(tmp_path)["fence_id"] == first["fence_id"] and engine.cancels == ["run-leaf"]


@pytest.mark.parametrize("tail", ["nothing", "settled_run", "review_run"])
def test_an_ended_root_with_no_ordinary_tail_still_refuses_as_not_live(tmp_path, monkeypatch, tail):
    from ouroboros import delegate_custody as dc
    from ouroboros.gateways import claudexor as gw
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    dc._CUSTODY.clear()
    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root-1", "completed", chat_id=0, root_task_id="root-1")
    engine = _Engine()
    monkeypatch.setattr(gw, "ClaudexorGateway", engine)
    if tail == "settled_run":
        _run(tmp_path, task_id="root-1")
        assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": "root-1", "state": "succeeded"})
    elif tail == "review_run":
        _run(tmp_path, task_id="root-1", source="review_substrate")
    assert request_owner_pause("root-1", request_id="press") == {"ok": False, "error": "task_not_live"}
    assert _fence(tmp_path) == {} and engine.opened == 0


def test_a_child_with_a_tail_is_refused_and_named_its_root(tmp_path, monkeypatch):
    from supervisor.owner_pause_control import request_owner_pause

    _queue, _workers, _engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    refused = request_owner_pause("child-dead", request_id="press")
    assert refused == {"ok": False, "error": "not_a_root_task", "root_task_id": "root-1"}
    assert _fence(tmp_path) == {}


def test_a_root_ending_under_its_pause_keeps_fence_and_latch_through_the_real_task_done(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    slot = SimpleNamespace(busy_task_id="root-1", reaping=False)
    workers.WORKERS[0] = slot
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, WORKERS=workers.WORKERS, append_jsonl=append_jsonl, RUNNING=workers.RUNNING,
                          persist_queue_snapshot=queue.persist_queue_snapshot)
    write_task_result(tmp_path, "root-1", "completed", result="answer")
    events._handle_task_done({"task_id": "root-1", "status": "completed", "worker_id": 0, "_files_prepared_attempt": 1}, ctx)
    assert "root-1" not in workers.RUNNING
    assert queue.BUDGET_ROOT_FENCES["root-1"]["cause"] == "owner_pause", "a terminal result is not ended execution"
    assert _fence(tmp_path)["state"] == "requested"
    assert engine.cancels == [], "the event loop's task_done sends nothing"

    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    engine.state = "cancelled"
    assert _tick(queue) == []
    assert _fence(tmp_path)["state"] == "released" and "root-1" not in queue.BUDGET_ROOT_FENCES


def test_a_member_ending_beside_a_running_root_is_looked_at_though_the_tree_was_at_rest(tmp_path, monkeypatch):
    """A tree of running members alone rests until one moves: that member's task_done is the transition, and what it
    left has no transition of its own."""
    from ouroboros import delegate_custody as dc
    from ouroboros.gateways import claudexor as gw
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events, owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    dc._CUSTODY.clear()
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    workers.PENDING[:] = []
    engine = _Engine()
    monkeypatch.setattr(gw, "ClaudexorGateway", engine)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == [] and engine.opened == 0
    assert owner_pause_control._AT_REST.get("root-1"), "two running members: only their own transitions move this tree"

    _run(tmp_path, task_id="child-run")  # its own park would stop this; it ends instead
    assert _tick(queue) == [] and engine.cancels == [], "still a running member's own, and the tree at rest"
    write_task_result(tmp_path, "child-run", "failed", result="the provider went away")
    workers.WORKERS[1] = SimpleNamespace(busy_task_id="child-run", reaping=False)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, WORKERS=workers.WORKERS, append_jsonl=append_jsonl, RUNNING=workers.RUNNING,
                          persist_queue_snapshot=queue.persist_queue_snapshot, send_with_budget=lambda *_a, **_k: None)
    events._handle_task_done({"task_id": "child-run", "status": "failed", "worker_id": 1, "_files_prepared_attempt": 1}, ctx)
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    assert not owner_pause_control._AT_REST.get("root-1"), "an unconfirmed stop keeps the tree due"


def test_an_unbound_start_is_held_unknown_with_no_post_and_takes_its_one_stop_once_bound(tmp_path, monkeypatch):
    """No run id, so the existing API has nothing to cancel, and a re-POST could start the work: it stays named and
    UNKNOWN, keeps the Pause requested, and gets its one stop only when it binds."""
    from ouroboros import delegate_custody as dc
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue import task_has_live_ownership, task_subtree_is_live

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed", tail=_start)
    assert request_owner_pause("root-1", request_id="press")["state"] == "requested", "a tail only a start names is a tail"
    assert _tick(queue) == [] and _tick(queue) == []
    assert (engine.cancels, engine.starts, engine.opened) == ([], [], 0)
    [row] = _kept(tmp_path)["external_runs"]["runs"]
    assert (row["invocation_id"], row["state"], row["stop_outcome"]) == ("inv-1", "stop_unknown", "pending_invocation_unbound")
    assert _fence(tmp_path)["state"] == "requested"
    # The room names it, as a run nobody can confirm, beside the Pause it holds: unknown, never a settled Paused.
    [work] = _works(tmp_path)
    assert (work["activity"], work["transition"], work["live"], work["mark"]) == ("unknown", "none", False, "ring")
    assert [(tail["code"], tail["state"], tail["ids"]) for tail in work["tails"]] == [
        ("external_runs_unconfirmed", "unknown", ["inv-1"])]
    # No door takes it: Resume refuses while the start holds the tree, and Stop's has nothing to reach.
    assert work["controls"] == []
    assert task_has_live_ownership("root-1") is False and task_subtree_is_live("root-1") is False
    # The admitted sweep that would re-POST it meets the same closed fence.
    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=engine, replay_pending=lambda: True)
    assert [(item["invocation_id"], item["action"]) for item in outcomes] == [("inv-1", "invocation_retained")]
    assert engine.starts == []

    _run(tmp_path, run_id="run-bound", invocation_id="inv-1")
    assert _tick(queue) == [] and engine.cancels == ["run-bound"]
    assert _tick(queue) == [] and engine.cancels == ["run-bound"]
    [participant] = _kept(tmp_path)["episode_runs"]
    assert (participant["run_id"], participant["invocation_id"], participant["stop_requested"]) == ("run-bound", "inv-1", True)
    [work] = _works(tmp_path)
    # Bound, Stop's door reaches it now; its own task failed, so nobody can confirm it runs: still unknown.
    assert (work["activity"], work["live"]) == ("unknown", False)
    assert [control["action"] for control in work["controls"]] == ["stop"]
    engine.state = "cancelled"
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "released"


def test_an_unreachable_gateway_owes_the_first_stop_and_recovery_pays_it_once(tmp_path, monkeypatch):
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    engine.up = False
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == [] and engine.cancels == []
    [row] = _kept(tmp_path)["external_runs"]["runs"]
    assert (row["state"], row["stop_outcome"]) == ("stop_unknown", "not_issued_gateway_unavailable")
    assert _fence(tmp_path)["state"] == "requested" and "root-1" not in owner_pause_control._AT_REST
    written = _fence(tmp_path)
    assert _tick(queue) == [] and _fence(tmp_path) == written, "the same fact is not written again"

    engine.up = True
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    assert _kept(tmp_path)["external_runs"]["runs"][0]["state"] == "stop_requested"
    # An issued stop the gateway later hides stays issued: unverified, never a second cancel.
    engine.up = False
    assert _tick(queue) == []
    assert _kept(tmp_path)["external_runs"]["runs"][0]["stop_outcome"] == "requested_unverified_gateway_unavailable"
    engine.up = True
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]


def test_an_unknown_stop_stays_requested_and_named_and_is_never_repeated(tmp_path, monkeypatch):
    from ouroboros.gateway.work_projection import unconfirmed_stops
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed", owner_status="completed")
    engine.cancel_error = engine.read_error = ClaudexorUnavailable("daemon_unreachable", "no answer")
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    [row] = _kept(tmp_path)["external_runs"]["runs"]
    assert row["state"] == "stop_unknown" and row["stop_outcome"] == "containment_fault_run_may_still_be_live"
    assert _fence(tmp_path)["state"] == "requested" and "root-1" in queue.BUDGET_ROOT_FENCES
    # The owner's surfaces name it through the one reading, its task's outcome aside.
    assert unconfirmed_stops(tmp_path, "child-dead", load_task_result(tmp_path, "child-dead")) == frozenset({"run-leaf"})
    [work] = _works(tmp_path)
    # A stop nobody could confirm keeps its run unknown: the requested Pause never turns it into running.
    assert (work["activity"], work["transition"], work["live"], work["mark"]) == ("unknown", "none", False, "ring")
    assert [(tail["code"], tail["state"], tail["ids"]) for tail in work["tails"]] == [
        ("external_runs_unconfirmed", "unknown", ["run-leaf"])]
    assert [control["action"] for control in work["controls"]] == ["stop"]

    assert _tick(queue) == [] and _tick(queue) == []
    assert engine.cancels == ["run-leaf"], "an unknown effect is re-read, never sent again"
    assert _fence(tmp_path)["state"] == "requested"
    engine.cancel_error = engine.read_error = None
    assert _tick(queue) == [] and _kept(tmp_path)["external_runs"]["runs"][0]["state"] == "stop_requested"
    [work] = _works(tmp_path)
    assert (work["activity"], work["transition"], work["live"]) == ("working", "pausing", True), "seen running again"
    engine.state = "cancelled"
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "released" and engine.cancels == ["run-leaf"]


def _ended_after_parking(tmp_path, monkeypatch, queue, workers, kind):
    """A member whose own park issued the stop and which then ended: its worker failed, the row left the queue."""
    from ouroboros import budget_pause, owner_pause, owner_wait
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import install_exact_budget_pause

    workers.RUNNING["child-dead"] = {"task": {"id": "child-dead", "type": "task", "chat_id": 0, "root_task_id": "root-1",
                                              "parent_task_id": "root-1", "delegation_role": "subagent"},
                                     "worker_id": 1, "attempt": 1}
    ctx, limit = _loop_ctx(tmp_path, "child-dead")
    ctx.root_task_id = "root-1"
    if kind == "cold":
        _fast_hold(monkeypatch, budget_pause)
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            budget_pause.enter_owner_pause(limit)
        budget_pause.end_dispatch_fence("child-dead")
        sup = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
                              sort_pending=lambda: None, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
        install_exact_budget_pause(sup, "child-dead", budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
        assert budget_pause.budget_pause_row(tmp_path, "child-dead")["owner_fence_id"] == _fence(tmp_path)["fence_id"]
        workers.PENDING[:] = [row for row in workers.PENDING if row["id"] != "child-dead"]
    else:
        def retained(_ctx, checkpoint):  # the worker keeps its stack; only its blocking wait is stood in for
            workers.RUNNING["child-dead"]["owner_wait"] = owner_wait.set_owner_wait(
                tmp_path, "child-dead", {**checkpoint, "state": "waiting"})
            return "retained"

        ctx.owner_wait_callback = retained
        owner_wait.park_owner_pause_warm(limit, ctx, fence=owner_pause.read_fence(tmp_path, "root-1"),
                                         detached=[{"operation_id": "review-op"}])
    workers.RUNNING.pop("child-dead", None)
    write_task_result(tmp_path, "child-dead", "failed", result="its worker died")


@pytest.mark.parametrize("kind", ["cold", "warm"])
def test_a_member_ending_after_its_park_hands_over_its_stop_without_a_second_cancel(tmp_path, monkeypatch, kind):
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, owner_status="running")  # alive, until its park below
    assert request_owner_pause("root-1", request_id="press")["ok"]
    _ended_after_parking(tmp_path, monkeypatch, queue, workers, kind)
    assert engine.cancels == ["run-leaf"], "its own park issued the stop"

    assert _tick(queue) == [] and _tick(queue) == []
    assert engine.cancels == ["run-leaf"], "handed over, only read back"
    kept = _kept(tmp_path)
    assert kept["external_runs"]["runs"][0]["stop_policy"] == "request_stop"
    assert kept["episode_runs"][0]["stop_requested"] is True and "episode_runs_partial" not in kept
    assert _fence(tmp_path)["state"] == "requested"
    engine.state = "cancelled"
    _park_root(tmp_path, monkeypatch, queue, workers)
    assert _tick(queue) == ["root-1"] and engine.cancels == ["run-leaf"]


def test_reviewers_of_an_ended_member_finish_beside_the_stop_and_the_projection_tells_them_apart(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as dc
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, owner_status="completed")
    _run(tmp_path, run_id="run-review", source="review_substrate")
    assert request_owner_pause("root-1", request_id="press")["ok"]
    _park_root(tmp_path, monkeypatch, queue, workers)
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"], "the reviewer is observed, never stopped"
    # Requested: the actor reads Paused, what it waits for reads as work still going, never as settled.
    [work] = _works(tmp_path)
    assert _fence(tmp_path)["state"] == "requested" and (work["activity"], work["transition"]) == ("paused", "none")
    assert sorted((tail["code"], tail["state"]) for tail in work["tails"]) == [
        ("delegated_running", "live"), ("review_work_finishing", "live")]

    engine.states["run-leaf"] = "cancelled"
    assert _tick(queue) == ["root-1"], "the reviewer the Pause lets finish holds nothing"
    fence = _fence(tmp_path)
    assert fence["state"] == "paused" and fence["finishing_reviews"] == ["child-dead"] and engine.cancels == ["run-leaf"]
    paused_at = fence["paused_at"]
    [work] = _works(tmp_path)
    assert (work["activity"], [tail["code"] for tail in work["tails"]]) == ("paused", ["review_work_finishing"])
    assert work["live"] is True and work["mark"] == "pause"

    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-review", "task_id": "child-dead", "state": "succeeded"})
    assert _tick(queue) == ["root-1"]
    fence = _fence(tmp_path)
    assert fence["state"] == "paused" and fence["finishing_reviews"] == [] and fence["paused_at"] == paused_at
    [work] = _works(tmp_path)
    assert (work["activity"], work["tails"], work["live"]) == ("paused", [], False)


def test_the_room_says_pausing_for_an_ended_roots_tail_and_nothing_once_it_ended(tmp_path, monkeypatch):
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed", owner_status="completed")
    [work] = _works(tmp_path)
    assert (work["activity"], work["transition"]) == ("working", "none")
    assert [control["action"] for control in work["controls"]] == ["pause", "stop"], "the door takes it now, so it is offered"

    assert request_owner_pause("root-1", request_id="press", by={"actor": "owner", "text": "", "at": "t"})["ok"]
    for _look in range(2):  # before the first stop and while it is only requested
        [work] = _works(tmp_path)
        assert (work["activity"], work["transition"], work["live"]) == ("working", "pausing", True)
        assert [(tail["code"], tail["ids"]) for tail in work["tails"]] == [("delegated_running", ["run-leaf"])]
        assert {"pause", "stop"} <= {control["action"] for control in work["controls"]}
        assert "resume" not in {control["action"] for control in work["controls"]}
        assert _tick(queue) == []
    engine.state = "cancelled"
    assert _tick(queue) == []
    assert _works(tmp_path) == []


@pytest.mark.parametrize("case", ["failed_owner", "unbound", "live"])
def test_a_requested_pause_types_its_tail_as_the_work_does_and_never_promotes_unknown(tmp_path, monkeypatch, case):
    """The nanny's projection probe, kept as a countertest. An ended root's requested Pause shows its reason beside
    what its tree left, typed by the work's own run typing (``work_status.run_tail``). A run nobody can confirm (its own
    task failed, or its stop could not be confirmed) and an unbound start stay unknown: never running, never a settled
    Paused, and no control their doors refuse. Only a run typed live keeps the Pause settling (Pausing…)."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue import task_subtree_is_live

    owner = "failed" if case in {"failed_owner", "unbound"} else "completed"
    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed", owner_status=owner,
                                         tail=_start if case == "unbound" else _run)
    if case == "unknown_stop":
        engine.cancel_error = engine.read_error = ClaudexorUnavailable("daemon_unreachable", "no answer")
    before = _works(tmp_path)
    expected_before = {"failed_owner": [("unknown", "none", False)], "unknown_stop": [("working", "none", True)],
                       "unbound": [], "live": [("working", "none", True)]}[case]
    assert [(work["activity"], work["transition"], work["live"]) for work in before] == expected_before
    assert request_owner_pause("root-1", request_id="press", by={"actor": "owner", "text": "", "at": "t"})["ok"]
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "requested"
    census, projected = _snapshot(tmp_path)
    [work], [row] = projected["works"], [row for row in census if row["activity_id"] == "root-1"]
    [room] = projected["rooms"].values()
    tail = ("delegated_running", "live") if case == "live" else ("external_runs_unconfirmed", "unknown")
    assert [(item["code"], item["state"], item["ids"]) for item in work["tails"]] == [
        (*tail, ["inv-1" if case == "unbound" else "run-leaf"])]
    assert work["because"][0]["code"] == "paused_by_owner" and work["stop"]["state"] == "requested"
    assert work["confirmed"] is True and "resume" not in {control["action"] for control in work["controls"]}
    assert queue.resume_budget_paused_task("root-1")["error"] == "owner_pause_effects_unsettled", "Resume refuses"
    if case == "live":  # what still executes as far as its facts say: the Pause is settling
        assert (work["activity"], work["transition"], work["live"], work["mark"]) == (
            "working", "pausing", True, "moving")
        assert [control["action"] for control in work["controls"]] == ["pause", "stop"]
        assert (row["phase"], row["live"], room["activity"], room["transition"]) == (
            "budget_pausing", True, "working", "pausing")
        return
    assert (work["activity"], work["transition"], work["live"], work["mark"]) == ("unknown", "none", False, "ring")
    assert (row["live"], room["activity"], room["mark"], room["confirmed"]) == (False, "unknown", "ring", True)
    # Stop is offered exactly where its door reaches an open run; nothing is offered where no door takes it.
    stoppable = case != "unbound"
    assert task_subtree_is_live("root-1") is stoppable
    assert [control["action"] for control in work["controls"]] == (["stop"] if stoppable else [])


def test_no_pause_no_stop_a_nannys_death_alone_leaves_the_healthy_leaf_running(tmp_path, monkeypatch):
    """The inverted floor stands without a Pause; a budget latch keeps its own release, custody or not."""
    from ouroboros import delegate_custody as dc
    from ouroboros.outcomes import infra_failed_axes
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import settle_requested_owner_pauses, settlement_look_owed
    from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch)
    write_task_result(tmp_path, "child-dead", "failed", outcome_axes=infra_failed_axes("worker_crash"))
    assert not settlement_look_owed(queue) and settle_requested_owner_pauses(queue) == []
    assert engine.opened == 0
    [outcome] = dc.reconcile_orphaned_runs(tmp_path, {"root-1"}, gateway_factory=engine)
    assert outcome["action"] == "left_live" and engine.cancels == []

    queue.BUDGET_ROOT_FENCES["root-2"] = {"status": "paused", "root_task_id": "root-2", "fence_id": "budget"}
    _run(tmp_path, run_id="run-budget", task_id="root-2")
    assert clear_budget_root_fence_for_settled_tree({"id": "root-2", "root_task_id": "root-2"}) is True


def test_an_unread_tail_stays_owed_and_the_latch_stands_on_an_unknown_census(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    real = current.tree_closure

    def unreadable(*_a, **_k):
        raise OSError("custody store busy")

    monkeypatch.setattr(current, "tree_closure", unreadable)
    assert request_owner_pause("root-1", request_id="press")["state"] == "requested", "unknown is never 'nothing left'"
    assert _tick(queue) == [] and engine.cancels == [] and "root-1" not in owner_pause_control._AT_REST
    assert _fence(tmp_path)["state"] == "requested"
    monkeypatch.setattr(current, "tree_closure", real)
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]

    monkeypatch.setattr("supervisor.continuation_admission.conflicting_writers", unreadable)
    assert clear_budget_root_fence_for_settled_tree({"id": "root-1", "root_task_id": "root-1"}) is False
    assert "root-1" in queue.BUDGET_ROOT_FENCES


def test_a_stop_whose_record_failed_to_land_is_still_never_sent_twice(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    assert request_owner_pause("root-1", request_id="press")["ok"]
    real = owner_pause.set_fence_state
    monkeypatch.setattr(owner_pause, "set_fence_state", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")))
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    assert "ended_member_runs" not in _fence(tmp_path) and "root-1" not in owner_pause_control._AT_REST
    monkeypatch.setattr(owner_pause, "set_fence_state", real)
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"], "this process still knows what it sent"
    assert _kept(tmp_path)["external_runs"]["runs"][0]["stop_outcome"] == "requested"
    assert not owner_pause_control._UNWRITTEN_STOPS


def test_a_settled_tree_rests_and_a_repeated_look_reads_nothing(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros import delegate_custody_current as current
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    _park_root(tmp_path, monkeypatch, queue, workers)
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    engine.state = "cancelled"
    assert _tick(queue) == ["root-1"]
    fence, reads = _fence(tmp_path), len(engine.reads)

    def forbidden(*_a, **_k):
        raise AssertionError("a tree at rest is not read")

    monkeypatch.setattr(current, "tree_closure", forbidden)
    monkeypatch.setattr(budget_pause, "observe_task_runs", forbidden)
    for _look in range(3):
        assert _tick(queue) == []
    assert _fence(tmp_path) == fence and len(engine.reads) == reads and engine.cancels == ["run-leaf"]


def test_a_restart_keeps_the_same_pause_its_latch_and_every_stop_it_already_issued(tmp_path, monkeypatch):
    """No queue row is left to carry this tree: only its open custody keeps the latch through the restore, the next
    process's first look re-reads the stop from the fence, and the press's identity stands."""
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    ack = request_owner_pause("root-1", request_id="press")
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    before = _fence(tmp_path)
    assert queue.persist_queue_snapshot(reason="before-restart")

    queue.BUDGET_ROOT_FENCES.clear()
    workers.PENDING[:] = []
    owner_pause_control._AT_REST.clear()
    owner_pause_control._CHANGED.clear()
    queue.restore_pending_from_snapshot()
    latch = queue.BUDGET_ROOT_FENCES["root-1"]
    assert latch["cause"] == "owner_pause" and latch["fence_id"] == ack["fence_id"]

    assert _tick(queue) == [] and engine.cancels == ["run-leaf"], "a first look after a restart issues nothing twice"
    after = _fence(tmp_path)
    assert (after["fence_id"], after["request_id"], after["requested_at"], after["state"]) == (
        before["fence_id"], "press", before["requested_at"], "requested")
    replay = request_owner_pause("root-1", request_id="press")
    assert replay["ok"] and replay["duplicate"] is True and replay["fence_id"] == ack["fence_id"]
    engine.state = "cancelled"
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "released" and "root-1" not in queue.BUDGET_ROOT_FENCES


def test_a_restored_latch_with_nothing_left_is_still_swept(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as dc
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import sweep_orphaned_budget_fences

    queue, _workers, _engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert sweep_orphaned_budget_fences([], queue.BUDGET_ROOT_FENCES, tmp_path) == [], "its tail is its member"
    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": "child-dead", "state": "cancelled"})
    assert sweep_orphaned_budget_fences([], queue.BUDGET_ROOT_FENCES, tmp_path) == ["root-1"]


@pytest.mark.parametrize("terminal_source", ["absent", "long", "unreadable"])
def test_the_ended_members_lines_never_offer_a_root_the_doors_of_a_foreign_starter(tmp_path, terminal_source):
    import json

    from ouroboros import delegate_custody as dc
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.external_runs import episode_disclosure

    dc._CUSTODY.clear()
    ref = {}
    if terminal_source != "absent":
        ref = store_actor_source_bytes(tmp_path, "child-dead", category="tool_results", source_id="terminal",
                                       data=json.dumps({"summary": {"continuity": [
                                           {"tryIndex": index} for index in range(5)]}}).encode(), extension="json")
        if terminal_source == "unreadable":
            ref = {**ref, "sha256": "0" * 64}
    _run(tmp_path)
    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": "child-dead", "state": "cancelled"})
    if ref:
        assert dc.emit(tmp_path, dc.TERMINAL_OBSERVED, {
            "run_id": "run-leaf", "task_id": "child-dead", "state": "cancelled",
            "observation_seq": 1, "source_ref": ref})
    assert dc.emit(tmp_path, dc.START_REQUESTED, {
        "invocation_id": "inv-1", "task_id": "child-dead", "root_task_id": "root-1", "route": "codex",
        "surface": "delegate", "request": {"prompt": "work"}})
    rows = [{"run_id": "run-leaf", "stop_requested": True, "stop_outcome": "confirmed"}, {"invocation_id": "inv-1"},
            {"run_id": "run-review", "review_owned": True}]
    own = episode_disclosure(tmp_path, "child-dead", rows)
    assert 'it may be continued: delegate_start(subagent_id=..., continue_from="run-leaf")' in own
    assert 'delegate_wait(run_id="run-leaf")' in own and 'delegate_start(retry_of="inv-1")' in own
    foreign = episode_disclosure(tmp_path, "child-dead", rows, reader="root-1")
    assert "continue_from=" not in foreign and "retry_of=" not in foreign and "delegate_wait" not in foreign
    assert foreign.count("started by ended task child-dead") == 3
    assert "delegate_start refuses any other task" in foreign and 'get_task_result("child-dead")' in foreign
    assert "a review panel's run" in foreign
    assert 'read_file(root="artifact_store"' not in foreign, "a child's relative path is not the root's source"
    if terminal_source == "long":
        assert "2 earlier continued tries" in own and "2 earlier continued tries" in foreign
        assert "try 0 (" not in foreign and "try 4 (" in foreign
    if terminal_source == "unreadable":
        assert "failed local verification" in own and "failed local verification" in foreign
    many = [{"run_id": f"run-{index}"} for index in range(9)]
    assert 'get_task_result("child-dead") reads them' in episode_disclosure(tmp_path, "child-dead", many, reader="root-1")


def test_a_pausing_tail_keeps_resume_refused_and_the_owners_stop_still_reaches_it(tmp_path, monkeypatch):
    from ouroboros.delegate_custody_reconcile import stop_tree_runs
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == [] and engine.cancels == ["run-leaf"]
    refused = queue.resume_budget_paused_task("root-1")
    assert refused["ok"] is False and refused["error"] == "owner_pause_effects_unsettled", refused
    assert [(item["kind"], item["run_id"]) for item in refused["blockers"]] == [("delegated_run", "run-leaf")]
    assert _fence(tmp_path)["state"] == "requested", "a refusal leaves the Pause as it was"

    # A Stop is the owner's own second control, spoken for itself; the Pause then sees the run gone.
    stop_tree_runs(tmp_path, "root-1", gateway_factory=engine)
    assert engine.cancels == ["run-leaf", "run-leaf"]
    engine.state = "cancelled"
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "released"
    assert engine.cancels == ["run-leaf", "run-leaf"], "the Pause sent nothing of its own after its first"


def test_only_this_pauses_root_actor_is_told_what_its_ended_members_left(tmp_path, monkeypatch):
    """A warm park's notice takes the same lines as a cold Resume (``paused_runs_notice``); a sibling or an older
    park's record does not: those runs are neither its stop nor its doors."""
    from ouroboros.external_runs import paused_runs_notice
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, _engine = _tail_tree(tmp_path, monkeypatch)
    ack = request_owner_pause("root-1", request_id="press")
    assert _tick(queue) == []
    root_ctx, _limit = _loop_ctx(tmp_path, "root-1")
    notice = paused_runs_notice(root_ctx, {"fence_id": ack["fence_id"], "episode_runs": []})
    assert "run run-leaf (codex), started by ended task child-dead" in notice and "none named" not in notice
    assert "it may still be running; never start a second writer over it" in notice
    assert paused_runs_notice(root_ctx, {"fence_id": "older", "episode_runs": []}) == "- none named"
    sibling, _limit = _loop_ctx(tmp_path, "sibling")
    sibling.root_task_id = "root-1"
    assert paused_runs_notice(sibling, {"fence_id": ack["fence_id"], "episode_runs": []}) == "- none named"


def test_a_record_handed_over_without_a_list_stays_partial_and_another_pauses_record_hands_over_nothing():
    from ouroboros.external_runs import ended_member_record, member_stop_record

    external = {"custody_read": "ok", "runs": [{"run_id": "r", "state": "stop_requested", "stop_policy": "request_stop",
                                                "stop_outcome": "requested", "terminal_source": {"big": "x"}}]}
    assert member_stop_record({"budget_pause": {"owner_fence_id": "other", "external_runs": external}}, "this") == {}
    assert member_stop_record({"owner_wait": {"owner_pause": {"fence_id": "other", "external_runs": external}}}, "this") == {}
    assert member_stop_record({}, "this") == {}
    cold = {"owner_fence_id": "this", "external_runs": external}
    assert member_stop_record({"budget_pause": cold}, "this") == cold
    first = ended_member_record({}, external)
    assert first["episode_runs"] == [{"run_id": "r", "stop_requested": True, "stop_outcome": "requested"}]
    assert "episode_runs_partial" not in first, "its first observation begins the list"
    assert "terminal_source" not in first["external_runs"]["runs"][0], "every launch gate reads this fence"
    assert ended_member_record(cold, external)["episode_runs_partial"] is True


def test_the_owners_stop_outranks_a_pause_only_an_unbound_start_holds(tmp_path, monkeypatch):
    """Existing Stop authority, preserved (this passes on the unchanged baseline too): the root's recorded Stop ends the
    Pause as it always did, and only then may the sweep take the start, under its own rule. Without ``owner_stopped``
    the new census hold would break it (mutation control, evidence label 52)."""
    from ouroboros import cancel_intents
    from ouroboros import delegate_custody as dc
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events, task_lifecycle
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, tail=_start)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    _park_root(tmp_path, monkeypatch, queue, workers)
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "requested"
    refused = queue.resume_budget_paused_task("root-1")
    assert refused["ok"] is False and refused["error"] == "owner_pause_effects_unsettled", (
        "a parked author's Resume would start a writer over it", refused)

    cancel_intents.request_cancel(tmp_path, "root-1", reason="owner stop", source="http_cascade", requested_by="owner")
    assert task_lifecycle.cancel_task_custody("root-1", deliver=False) == task_lifecycle.CANCEL_CANCELLED
    row = load_task_result(tmp_path, "root-1")
    assert row["status"] == "cancelled" and row["cancel_origin"]["request_id"]
    # The real consumer of the task_done that terminal published: its census meets the recorded Stop.
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, WORKERS=workers.WORKERS, append_jsonl=append_jsonl, RUNNING=workers.RUNNING,
                          persist_queue_snapshot=queue.persist_queue_snapshot)
    events._handle_task_done({"task_id": "root-1", "status": "cancelled", "root_task_id": "root-1"}, ctx)
    assert _fence(tmp_path)["state"] == "released" and "root-1" not in queue.BUDGET_ROOT_FENCES

    engine.may_start = True
    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=engine, replay_pending=lambda: True)
    assert engine.starts == ["inv-1"] and engine.cancels == ["run-bound"], outcomes
    # A stopped root takes no new Pause, and its work offers none: Stop owns what remains.
    assert request_owner_pause("root-1", request_id="later") == {"ok": False, "error": "task_not_live"}
    assert all(control["action"] != "pause" for work in _works(tmp_path) for control in work["controls"])


def test_resume_stays_refused_while_an_unbound_start_holds_an_answered_tree(tmp_path, monkeypatch):
    """No relaxation: the start stays named and UNKNOWN, nothing is posted or cleared, the Pause and its latch stand."""
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import sweep_orphaned_budget_fences

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed", tail=_start)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == []
    before = _fence(tmp_path)
    refused = queue.resume_budget_paused_task("root-1")
    assert refused["ok"] is False and refused["error"] == "owner_pause_effects_unsettled", refused
    assert [(item["kind"], item["invocation_id"], item["stop_outcome"]) for item in refused["blockers"]] == [
        ("delegated_run", "inv-1", "pending_invocation_unbound")]
    assert _fence(tmp_path) == before and "root-1" in queue.BUDGET_ROOT_FENCES
    assert (engine.cancels, engine.starts) == ([], [])
    assert sweep_orphaned_budget_fences([], queue.BUDGET_ROOT_FENCES, tmp_path) == [], "a restart keeps holding it too"


def test_the_minds_roster_names_the_same_unbound_start(tmp_path, monkeypatch):
    from ouroboros import peer_roster
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, _engine = _tail_tree(tmp_path, monkeypatch, root_status="completed", tail=_start)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == []
    tails = peer_roster._tree_tails(tmp_path, {"running": [], "runs": []}, _fence(tmp_path), "root-1")
    assert [(tail["code"], tail["ids"]) for tail in tails] == [("external_runs_unconfirmed", ["inv-1"])]


def test_a_member_not_proven_ended_is_never_stopped_and_one_with_no_row_left_is(tmp_path, monkeypatch):
    """Only a stored terminal, or no row at all (a collected result's owner is gone too), makes a run this pass's. A
    non-terminal row outside the queue may be a worker mid-handoff: its park or terminal comes first, and it stays due."""
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, owner_status="running")
    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == [] and engine.cancels == []
    assert "root-1" not in owner_pause_control._AT_REST and "ended_member_runs" not in _fence(tmp_path)
    _run(tmp_path, run_id="run-collected", task_id="collected-member")
    assert _tick(queue) == [] and engine.cancels == ["run-collected"]
    assert list(_fence(tmp_path)["ended_member_runs"]) == ["collected-member"]


def test_the_http_door_accepts_an_ended_roots_tail_and_keeps_its_404_for_a_finished_tree(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import delegate_custody as dc
    from ouroboros.gateway.task_pause import api_task_pause

    queue, _workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed")
    client = TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/pause", api_task_pause, methods=["POST"])]))
    accepted = client.post("/api/tasks/root-1/pause", json={"request_id": "press", "actor": "owner"})
    assert accepted.status_code == 200 and accepted.json()["state"] == "requested", accepted.text
    assert engine.opened == 0, "the door sends nothing"
    child = client.post("/api/tasks/child-dead/pause", json={"request_id": "press"})
    assert child.status_code == 409 and child.json()["reason_code"] == "not_a_root_task"
    assert child.json()["root_task_id"] == "root-1"
    engine.state = "cancelled"
    assert _tick(queue) == [] and _fence(tmp_path)["state"] == "released"
    assert [run.run_id for run in dc.open_runs(tmp_path)] == []
    assert client.post("/api/tasks/root-1/pause", json={"request_id": "later"}).status_code == 404
