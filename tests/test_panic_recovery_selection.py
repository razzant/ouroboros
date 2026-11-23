"""Panic recovery through the real owner gateway over isolated queue state.

No supervisor lifespan, worker dispatch, process discovery or stop/kill call.
"""
import copy
import os
import threading

import pytest

from ouroboros.control_actor import mind_control, owner_door
from ouroboros.task_results import write_task_result
from ouroboros.work_stop import panic_hold_id
from supervisor import events_budget as eb
from tests._queue_state import gateway as _gateway, read_state

gateway = _gateway


def _held(world, task_id, *, parent=None, chat_id=1, legacy=False):
    root_id = parent or task_id
    row = {"id": task_id, "root_task_id": root_id, "chat_id": chat_id, "type": "task", "_attempt": 1,
           "admitted_dispatch": "none", "title": f"Work {task_id}"}
    if parent:
        row["parent_task_id"] = parent
        row["delegation_role"] = "subagent"
    write_task_result(world.root, task_id, "scheduled", root_task_id=root_id, chat_id=chat_id,
                      **({"parent_task_id": parent, "delegation_role": "subagent"} if parent else {}))
    eb.hold_budget_row(row, reason=eb.HOLD_PANIC)
    if legacy:
        row[eb.BUDGET_HOLD_KEY].update(reason=eb.HOLD_SAVED_WORK, door="panic", release="resume")
    world.pending.append(row)
    return row


def _resume(world, row, token=None, **extra):
    return world.client.post(f"/api/tasks/{row['id']}/resume",
                             json={"expected_panic_hold_id": token or panic_hold_id(row), **extra})


@pytest.fixture
def recovered_gateway(gateway, monkeypatch):
    """The fixture's own queue is published after the task-readiness latch, without booting anything."""
    from supervisor import queue

    ready = threading.Event()
    ready.set()
    gateway.client.app.state.supervisor_ready_event = ready
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", gateway.root / "state" / "queue_snapshot.json")
    return gateway


def test_owner_census_reads_other_rooms_and_members_before_filtering(recovered_gateway):
    gateway = recovered_gateway
    root = _held(gateway, "root", chat_id=91, legacy=True)
    child = _held(gateway, "child", parent="root", chat_id=91)
    other = _held(gateway, "restart")
    other[eb.BUDGET_HOLD_KEY].update(reason=eb.HOLD_OWNER_RESTART, release="resume")
    state = read_state(gateway)
    assert state["panic_recovery_complete"] is True
    assert [(r["task_id"], r["panic_hold_id"]) for r in state["panic_recovery"]] == [
        ("root", panic_hold_id(root)), ("child", panic_hold_id(child))]
    assert all(r["task_name"] and r["chat_id"] == 91 for r in state["panic_recovery"])
    assert state["panic_recovery"][1]["root_task_id"] == "root"
    assert not {r["task_id"] for r in state["main_attention"]} & {"root", "child"}
    assert {r["activity_id"] for r in state["active_chat_activities"]}.isdisjoint({"child"})


@pytest.mark.parametrize("gap", ["hold-identity", "activity-availability"])
def test_a_census_gap_preserves_known_rows_without_certifying_all(recovered_gateway, monkeypatch, gap):
    _held(recovered_gateway, "known")
    if gap == "hold-identity":
        unknown = _held(recovered_gateway, "undated", legacy=True)
        unknown[eb.BUDGET_HOLD_KEY].pop("held_at")
    else:
        from ouroboros.gateway import state as gateway_state

        def incomplete(*, availability):
            availability["complete"] = False
            return []

        monkeypatch.setattr(gateway_state, "_direct_turns_snapshot_safe", incomplete)
    state = read_state(recovered_gateway)
    assert state["boot"]["tasks_ready"] is True
    assert state["panic_recovery_complete"] is False
    assert [r["task_id"] for r in state["panic_recovery"]] == ["known"]


def test_own_panic_hold_lists_beneath_an_independent_pause_but_not_from_running(recovered_gateway):
    from ouroboros.owner_pause import install_fence

    gateway = recovered_gateway
    row = _held(gateway, "pausedwork", chat_id=91)
    install_fence(gateway.root, row["id"], request_id="later-Pause", by={"actor": "owner"})
    stale = {**row, "id": "runningwork", "root_task_id": "runningwork"}
    write_task_result(gateway.root, "runningwork", "running", chat_id=91, root_task_id="runningwork")
    gateway.running["runningwork"] = {"task": stale, "attempt": 1, "started_at": 1.0}

    state = read_state(gateway)

    activity = next(r for r in state["active_chat_activities"] if r["activity_id"] == row["id"])
    assert activity["stop"]["cause"] == "pause"
    assert state["panic_recovery_complete"] is True
    assert [(r["task_id"], r["panic_hold_id"]) for r in state["panic_recovery"]] == [
        (row["id"], panic_hold_id(row))]


@pytest.mark.parametrize("source", ["point", "parked-view"])
def test_unreadable_pending_controls_keep_known_holds_without_certifying_all(recovered_gateway, monkeypatch, source):
    from ouroboros import budget_pause, work_stop

    gateway = recovered_gateway
    rows = [_held(gateway, task_id) for task_id in ("unreadable", "known")]
    if source == "point":
        module, reader = budget_pause, "budget_pause_row"
        task_index = 1
    else:
        module, reader = work_stop, "parked_stop_view"
        task_index = 2
    real_reader = getattr(module, reader)

    def read(*args, **kwargs):
        if args[task_index] == "unreadable":
            raise RuntimeError("fixture: control facts unreadable")
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(module, reader, read)
    state = read_state(gateway)

    assert state["panic_recovery_complete"] is False
    assert [(r["task_id"], r["panic_hold_id"]) for r in state["panic_recovery"]] == [
        (row["id"], panic_hold_id(row)) for row in rows]


def test_panic_recovery_uses_verified_room_not_registered_workspace_scope(recovered_gateway, monkeypatch):
    from ouroboros.projects_registry import RoomResolver, bind_task_to_project, create_project, project_scope_admission

    gateway = recovered_gateway
    folder = gateway.root / "room-checkout"
    folder.mkdir()
    leased = create_project(gateway.root, "leased", name="Leased folder", working_dir=str(folder))
    scope = project_scope_admission(gateway.root, workspace_root=str(folder))["project_id"]
    assert scope == leased["id"]
    bound = create_project(gateway.root, "bound", name="Actual project room")
    # Neither immediate parent nor root is bound. A nearer ended ancestor was moved to a different room.
    write_task_result(gateway.root, "moved", "completed", chat_id=1, root_task_id="top", parent_task_id="top")
    write_task_result(gateway.root, "parent", "completed", chat_id=1, root_task_id="top", parent_task_id="moved")
    bind_task_to_project(gateway.root, "moved", bound["id"], origin={"absent": "system"})
    rows = [_held(gateway, task_id, **kwargs) for task_id, kwargs in (
        ("main-lease", {}), ("bound-elsewhere", {"parent": "parent"}), ("no-label", {}),
        ("own-project", {"chat_id": leased["chat_id"]}),
    )]
    for row in rows:
        row["project_id"] = scope
    rows[1]["root_task_id"] = "top"
    for row in rows:
        write_task_result(gateway.root, row["id"], "scheduled", **{key: row[key] for key in (
            "project_id", "chat_id", "root_task_id", "parent_task_id", "delegation_role") if key in row})

    resolve = RoomResolver.__call__

    def room(self, task_id, *args):
        if self.strict and task_id == "no-label":
            raise ValueError("fixture: room proof unavailable")
        return resolve(self, task_id, *args)

    monkeypatch.setattr(RoomResolver, "__call__", room)
    state = read_state(gateway)

    by_id = {r["task_id"]: r for r in state["panic_recovery"]}
    assert {tid: r["panic_hold_id"] for tid, r in by_id.items()} == {r["id"]: panic_hold_id(r) for r in rows}
    assert by_id["main-lease"]["chat_id"] == 1 and "project_id" not in by_id["main-lease"]
    assert (by_id["bound-elsewhere"]["chat_id"], by_id["bound-elsewhere"]["project_id"]) == (bound["chat_id"], bound["id"])
    assert (by_id["own-project"]["chat_id"], by_id["own-project"]["project_id"]) == (leased["chat_id"], leased["id"])
    assert not {"chat_id", "project_id"} & by_id["no-label"].keys()
    assert state["panic_recovery_complete"] is True  # coverage of holds, not coverage of display labels


@pytest.mark.parametrize("legacy", [False, True])
def test_conditional_http_resume_uses_only_this_hold_and_owner_door(gateway, legacy):
    from supervisor.queue_transitions import resume_budget_paused_task

    selected = _held(gateway, "selected", legacy=legacy)
    sibling = _held(gateway, "untouched")
    token = panic_hold_id(selected)
    original = copy.deepcopy(sibling)
    assert token and panic_hold_id(sibling) != token
    assert _resume(gateway, sibling, token).json()["error"] == "panic_selection_changed"
    for call in (lambda: resume_budget_paused_task("selected"),
                 lambda: resume_budget_paused_task("selected", expected_panic_hold_id=token)):
        assert call()["error"] == "panic_requires_owner_resume"
    with owner_door("fixture"), mind_control("mind", "choose another task"):
        assert resume_budget_paused_task("selected", expected_panic_hold_id=token)["error"] == "panic_requires_owner_resume"
    assert _resume(gateway, selected, token, accept_unknown=["unconfirmed"]).status_code == 400
    response = _resume(gateway, selected, token)
    assert response.status_code == 200 and response.json()["ok"], response.text
    assert eb.BUDGET_HOLD_KEY not in selected
    assert sibling == original and not gateway.running
    repeat = _resume(gateway, selected, token)
    assert repeat.status_code == 409 and repeat.json()["error"] == "panic_selection_changed"


@pytest.mark.parametrize("changed", ["newer-Panic", "different-hold", "owner-overlay"])
def test_revalidates_after_the_off_lock_observation(gateway, monkeypatch, changed):
    from ouroboros.owner_pause import install_fence

    row = _held(gateway, "selected", legacy=True)
    token = panic_hold_id(row)
    original = eb.observe_held_budget_selection

    def observe(*args):
        value = original(*args)
        if changed == "owner-overlay":
            install_fence(gateway.root, row["id"], request_id="later-Pause", by={"actor": "owner"})
        else:
            row[eb.BUDGET_HOLD_KEY]["held_at"] = "2026-11-22T23:45:56.000002+00:00"
            if changed == "different-hold":
                row[eb.BUDGET_HOLD_KEY]["door"] = "owner_restart"
        return value

    monkeypatch.setattr(eb, "observe_held_budget_selection", observe)
    response = _resume(gateway, row, token)
    assert response.status_code == 409, response.text
    assert response.json()["error"] == ("selection_owner_held" if changed == "owner-overlay"
                                         else "panic_selection_changed")
    assert eb.BUDGET_HOLD_KEY in row and not gateway.running


@pytest.mark.parametrize(("case", "error"), [
    ("budget-marker", "selection_owner_held"), ("deadline", "deadline_passed"),
    ("lifetime", "lifetime_exhausted"), ("money", "budget_still_exhausted"),
    ("owner-stop", "selection_owner_held"),
])
def test_group_press_keeps_independent_authority(gateway, monkeypatch, case, error):
    from ouroboros.cancel_intents import request_cancel
    from supervisor import state

    row = _held(gateway, "selected", legacy=True)
    if case == "budget-marker":
        row["_budget_pause"] = {"reason": "budget", "status": "paused_before_dispatch"}
    elif case == "deadline":
        row["deadline_at"] = "2026-01-01T00:00:00Z"
    elif case == "lifetime":
        row.update(execution_timeout_sec=5, _execution_handoff_sec=6)
    elif case == "money":
        monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 0.0)
    else:
        request_cancel(gateway.root, row["id"], reason="fixture Stop", source="http", requested_by="owner")
    prior = copy.deepcopy(row)
    response = _resume(gateway, row)
    assert response.status_code == 409 and response.json()["error"] == error, response.text
    assert row == prior and not gateway.running


@pytest.mark.parametrize("readable", [True, False])
@pytest.mark.parametrize("independent_hold", [False, True])
def test_aborted_update_keeps_checkpoint_cause_and_observed_panic_door(gateway, monkeypatch, readable, independent_hold):
    from types import SimpleNamespace

    from ouroboros.task_results import load_task_result
    from ouroboros.work_stop import record_policy
    from supervisor import active_activity, queue, restart_retention as retention

    tid = "saved-update-work"
    write_task_result(gateway.root, tid, "running", root_task_id=tid, chat_id=1)
    row = {"id": tid, "root_task_id": tid, "type": "task", "chat_id": 1}
    prior = (eb.hold_budget_row(row, reason=eb.HOLD_OWNER_RESTART, result_root=gateway.root)
             if independent_hold else None)
    meta = {"task": row, "attempt": 1, retention.RETAINED_FOR_BOOT: True}
    gateway.running[tid] = meta
    monkeypatch.setattr(retention, "_update_returns", {
        "supervisor_pid": os.getpid(), "root": gateway.root, "running": gateway.running, "quiesced": True,
        "rows": {tid: {"meta": meta, "attempt": 1, "returning": True,
                       "proc": SimpleNamespace(is_alive=lambda: False)}},
    })
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(actors=lambda: ()))
    monkeypatch.setattr(retention, "panic_recorded", lambda root: root == gateway.root)
    monkeypatch.setattr(retention, "park_saved_running_rows", lambda *_args: [])
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda *, reason: reason == "update_abort_handoff")

    def checkpoint(root, task, attempt, cause):
        assert (root, attempt, cause) == (gateway.root, 1, "update_aborted")
        if readable:
            task["_working_recovery"] = {"source_task_id": tid, "from_attempt": attempt,
                                         "cause": cause, "boundary": "saved_fixture"}
        return readable

    monkeypatch.setattr(retention, "_recovery_source", checkpoint)
    assert retention.recover_aborted_update(gateway.root, gateway.running, gateway.pending)
    [recovered] = gateway.pending
    assert recovered["_attempt"] == 2 and not gateway.running
    assert recovered["_working_recovery"]["cause"] == "update_aborted"
    assert (recovered["_working_recovery"].get("boundary") == "saved_fixture") is readable
    hold = recovered[eb.BUDGET_HOLD_KEY]
    if independent_hold:
        assert hold == prior
    else:
        assert hold["reason"] == eb.HOLD_SAVED_WORK and hold["door"] == "panic"
        assert record_policy(hold)["release"] == "owner"
    result = load_task_result(gateway.root, tid, strict=True)
    assert result["reason_code"] == record_policy(hold)["ui_code"] == (
        "owner_restart_hold" if independent_hold else "panic_hold")


def test_owner_panic_return_reads_current_money_outside_queue_and_launch_locks(gateway, monkeypatch):
    from ouroboros.owner_pause import launch_lock
    from supervisor import budget_resume, queue, state

    row = _held(gateway, "money-observed")
    sibling = _held(gateway, "not-selected")
    saved_sibling = copy.deepcopy(sibling)
    reads = []

    def observe(name):
        assert not queue._queue_lock._is_owned()
        with launch_lock(gateway.root, row["id"], wait=False):
            reads.append(name)

    def global_money(_state, *, strict, allow_stale):
        assert strict and not allow_stale
        observe("global")
        return 1.0

    def tree_money(root, task, tree):
        assert (root, task["id"], tree) == (gateway.root, row["id"], row["id"])
        observe("tree")
        return None

    monkeypatch.setattr(state, "budget_remaining", global_money)
    monkeypatch.setattr(budget_resume, "_root_money_refusal", tree_money)
    response = _resume(gateway, row)
    assert response.status_code == 200 and response.json()["ok"], response.text
    assert sorted(reads) == ["global", "tree"]
    assert eb.BUDGET_HOLD_KEY not in row and sibling == saved_sibling and not gateway.running


def test_owner_panic_return_completes_interrupted_member_choice_only(gateway, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from supervisor import task_admission

    root, child_id = "completed-tree", "saved-member"
    write_task_result(gateway.root, root, "running", root_task_id=root, chat_id=91)
    fence, _ = owner_pause.install_fence(gateway.root, root, request_id="owner-pause", by={"actor": "owner"})
    write_task_result(gateway.root, root, "completed", root_task_id=root, chat_id=91)
    child = _held(gateway, child_id, parent=root, chat_id=91, legacy=True)
    sibling = _held(gateway, "other-member", parent=root, chat_id=91)
    saved_sibling = copy.deepcopy(sibling)
    child.update(_attempt=2, _working_recovery={"source_task_id": child_id, "from_attempt": 1,
                                               "cause": "update_aborted"})
    grant = {"grant_id": "used-grant", "authority": "explicit_resume", "selection_door": "direct",
             "consumed_at": "2026-11-23T00:00:00Z", "owner_pause_fence_id": fence["fence_id"]}
    record = {"state": budget_pause.STATE_RESUMED, "task_attempt": 1, "grant": grant}
    real_reader = budget_pause.budget_pause_row
    monkeypatch.setattr(budget_pause, "budget_pause_row", lambda disk, tid:
                        copy.deepcopy(record) if (disk, tid) == (gateway.root, child_id) else real_reader(disk, tid))
    monkeypatch.setattr(task_admission, "working_resume_refusal", lambda task, disk:
                        "" if task["id"] == child_id and disk == gateway.root else "saved_work_unverified")

    response = _resume(gateway, child)
    assert response.status_code == 200 and response.json()["ok"], response.text
    selected = owner_pause.read_fence(gateway.root, root)["selected_members"]
    assert set(selected) == {child_id} and selected[child_id]["grant_id"] == grant["grant_id"]
    assert eb.BUDGET_HOLD_KEY not in child and sibling == saved_sibling and not gateway.running
