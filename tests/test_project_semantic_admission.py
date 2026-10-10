"""Real admission owners: semantic freshness, exact cleanup and restore custody.

Run through scripts/safe_test.py; every root and file below is synthetic.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_schedule_occurrence import q, _row, _rows  # noqa: F401

pytestmark = pytest.mark.serial


def task(tid="attempt", **extra):
    return {"id": tid, "type": "task", "text": "Synthetic admission", "chat_id": 1,
            "project_id": "target", "root_task_id": tid, **extra}


@pytest.fixture
def room(q):  # noqa: F811
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(q.root)
    registry.create_project(q.root, "target", name="Target")
    registry.create_project(q.root, "other", name="Other")
    return q


@pytest.mark.parametrize("activity", ["unchanged", "own_touch", "other_touch", "visible", "result"])
def test_activity_is_not_a_routing_change(room, activity):
    basis = registry.project_admission_view(room.root, "target")
    if activity.endswith("touch"):
        registry.touch_project(room.root, "target" if activity == "own_touch" else "other")
    elif activity == "visible":
        registry.increment_project_visible_revision(room.root, project_id="target")
    elif activity == "result":
        registry.update_project(room.root, "other", last_task_result_id="finished")
    admitted = room.queue.enqueue_task(task(), project_admission=basis)
    assert not admitted.get("_admission_blocked")
    assert room.pending == [admitted]


@pytest.mark.parametrize("writer", [False, True])
def test_registry_lock_occupancy_does_not_delay_nested_queue_admission(room, writer):
    held, release, finished = threading.Event(), threading.Event(), threading.Event()
    outcomes = []

    def holder():
        lock = registry._file_write_lock(registry._registry_path(room.root)) if writer else registry._LOCK
        with lock:
            held.set()
            release.wait(10)

    def admit():
        # Actual callers enter enqueue while already owning this reentrant lock.
        with room.queue._queue_lock:
            outcomes.append(room.queue.enqueue_task(task()))
        finished.set()

    thread = threading.Thread(target=holder)
    caller = threading.Thread(target=admit)
    thread.start()
    try:
        assert held.wait(3)
        caller.start()
        assert finished.wait(3), "admission waited behind a registry lock"
        assert not outcomes[0].get("_admission_blocked")
    finally:
        release.set()
        thread.join(5)
        if caller.ident:
            caller.join(5)
    assert not thread.is_alive() and not caller.is_alive()


@pytest.mark.parametrize("change", ["rebind", "aba", "delete", "missing", "reconstruct"])
def test_original_preparation_basis_rejects_semantic_changes(room, change):
    basis = registry.project_admission_view(room.root, "target")
    if change in {"rebind", "aba"}:
        registry.update_project(room.root, "target", working_dir="/synthetic/other")
        if change == "aba":
            registry.update_project(room.root, "target", working_dir="")
    elif change == "delete":
        registry.begin_project_deletion(room.root, "target")
    else:
        store = room.root / "projects" / "target"
        store.mkdir(parents=True)
        registry.reconcile_projects(room.root)
        data = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))
        data["projects"] = [row for row in data["projects"] if row["id"] != "target"]
        registry._save(room.root, data)
        if change == "reconstruct":
            registry.reconcile_projects(room.root)
            assert registry.get_reserved_project(room.root, "target")["routing_incarnation"] != basis["project"]["routing_incarnation"]
    assert room.queue.reserve_task_admission("attempt", "ours", drive_root=room.root)["status"] == "reserved"
    refused = room.queue.enqueue_task(task(_admission_token="ours"), project_admission=basis)
    assert refused["_admission_blocked"] == ("project_routing_fence" if change == "delete" else "project_routing_fence_changed")
    assert refused["_admission_detail"] and not room.pending
    assert "attempt" not in room.queue.ADMISSION_RESERVATIONS


@pytest.mark.parametrize("corruption", ["json", "lifecycle", "generation", "duplicate", "folder", "permission", "missing_file"])
def test_raw_unavailable_authority_is_typed_and_never_normalized_active(room, monkeypatch, corruption):
    basis = registry.project_admission_view(room.root, "target")
    path = registry._registry_path(room.root)
    if corruption == "permission":
        original = type(path).open
        def denied(self, *args, **kwargs):
            if self == path:
                raise PermissionError("synthetic permission failure")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(type(path), "open", denied)
    elif corruption == "missing_file":
        path.unlink()
    elif corruption == "json":
        path.write_text("{torn", encoding="utf-8")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        first = data["projects"][0]
        if corruption == "duplicate":
            data["projects"].append(dict(first))
        else:
            first[{"lifecycle": "lifecycle", "generation": "routing_generation", "folder": "working_dir"}[corruption]] = {
                "lifecycle": "banana", "generation": -1, "folder": {"bad": "path"}}[corruption]
        path.write_text(json.dumps(data), encoding="utf-8")
    refused = room.queue.enqueue_task(task(), project_admission=basis)
    assert refused["_admission_blocked"] == "project_routing_fence_lookup_failed"
    assert refused["_project_lifecycle"] == "" and refused["_admission_detail"]
    assert not room.pending


def test_frozen_explicit_resource_ignores_new_room_default_but_not_deletion(room):
    basis = registry.project_admission_view(room.root, "target", frozen=True)
    registry.update_project(room.root, "target", working_dir="/synthetic/new-default")
    admitted = room.queue.enqueue_task(task(workspace_root="/synthetic/approved"), project_admission=basis)
    assert not admitted.get("_admission_blocked") and admitted["workspace_root"] == "/synthetic/approved"
    registry.begin_project_deletion(room.root, "target")
    assert room.queue.enqueue_task(task("second"), project_admission=basis)["_admission_blocked"] == "project_routing_fence"


def test_unregistered_scope_cannot_be_claimed_by_a_new_room(room):
    basis = registry.project_scope_admission(room.root, project_id="scope-only")
    assert basis["project"] is None
    assert not room.queue.enqueue_task(task("first", project_id="scope-only"), project_admission=basis).get("_admission_blocked")
    registry.create_project(room.root, "scope-only")
    refused = room.queue.enqueue_task(task("second", project_id="scope-only"), project_admission=basis)
    assert refused["_admission_blocked"] == "project_routing_fence_changed"


def test_foreign_reservation_is_preserved(room):
    room.queue.reserve_task_admission("attempt", "foreign", drive_root=room.root)
    refused = room.queue.enqueue_task(task(_admission_token="ours"))
    assert refused["_admission_blocked"] == "admission_reservation_owned"
    assert room.queue.ADMISSION_RESERVATIONS["attempt"] == "foreign"


def test_overlapping_delete_observer_cannot_see_partial_queue_membership(room, monkeypatch):
    from ouroboros import project_admission

    original = project_admission._strict_admission_snapshot
    basis = registry.project_admission_view(room.root, "target")
    observed = []
    threads = []

    def read_then_delete(root, **kwargs):
        snapshot = original(root, **kwargs)
        if not threads:
            def writer():
                registry.begin_project_deletion(root, "target")
                with room.queue._queue_lock:
                    observed.extend(row["id"] for row in room.pending)
            thread = threading.Thread(target=writer)
            threads.append(thread)
            thread.start()
        return snapshot

    monkeypatch.setattr(project_admission, "_strict_admission_snapshot", read_then_delete)
    admitted = room.queue.enqueue_task(task(), project_admission=basis)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not admitted.get("_admission_blocked") and observed == ["attempt"]


@pytest.mark.parametrize("write_failure", [False, True])
def test_mixed_restore_conserves_closed_project_task(room, monkeypatch, write_failure):
    from supervisor import task_admission

    room.queue.enqueue_task(task("project"))
    write_task_result(room.root, "project", "scheduled")
    room.queue.enqueue_task(task("main", project_id=""))
    assert room.queue.persist_queue_snapshot(reason="synthetic restart")
    registry.begin_project_deletion(room.root, "target")
    room.queue.init_queue_refs([], {}, {"value": 0})
    if write_failure:
        monkeypatch.setattr(task_admission, "write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic write failure")))
    room.queue.restore_pending_from_snapshot()
    snapshot = json.loads(room.queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    kept = {row["task"]["id"]: row["task"] for row in snapshot["pending"]}
    assert "main" in kept
    if write_failure:
        assert kept["project"]["_terminalization_retry"]["trigger"] == "project_routing_fence"
        assert load_task_result(room.root, "project")["status"] == "scheduled"
    else:
        assert "project" not in kept
        result = load_task_result(room.root, "project")
        assert result["status"] == "failed" and result["reason_code"] == "project_routing_fence"


def test_schedule_same_occurrence_survives_benign_activity(room, monkeypatch):
    from supervisor import schedule_occurrence

    _row(room, project_id="target", intent={"kind": "explicit_none"})
    original = schedule_occurrence.prepare
    def prepare(claim):
        item = original(claim)
        registry.touch_project(room.root, "other")
        return item
    monkeypatch.setattr(schedule_occurrence, "prepare", prepare)
    room.queue.check_scheduled_tasks()
    [queued] = room.pending
    assert queued["id"] == _rows(room)["s1"]["occurrence"]["task_id"]
    assert load_task_result(room.root, queued["id"])["status"] == "scheduled"
    room.queue.check_scheduled_tasks()
    assert room.pending == [queued]


def test_schedule_room_preparation_detects_aba_during_workspace_validation(room, tmp_path, monkeypatch):
    from ouroboros import workspace_admission

    folder = tmp_path / "folder"
    folder.mkdir()
    registry.update_project(room.root, "target", working_dir=str(folder))
    _row(room, project_id="target", intent={"kind": "room_default", "project_id": "target"})
    original = workspace_admission.validate_workspace_root
    def validate(*args, **kwargs):
        value = original(*args, **kwargs)
        registry.update_project(room.root, "target", working_dir=str(tmp_path / "elsewhere"))
        registry.update_project(room.root, "target", working_dir=str(folder))
        return value
    monkeypatch.setattr(workspace_admission, "validate_workspace_root", validate)
    room.queue.check_scheduled_tasks()
    assert not room.pending
    assert _rows(room)["s1"]["hold"]["reason"] == "project_routing_fence_changed"


def test_schedule_room_folder_uses_the_carried_basis_and_repo_owner(room, tmp_path, monkeypatch):
    """The folder comes from the same read admission fences, validated against the worker repo."""
    from ouroboros import projects_registry, workspace_admission
    from supervisor import workers

    folder = tmp_path / "folder"
    folder.mkdir()
    registry.update_project(room.root, "target", working_dir=str(folder))
    _row(room, project_id="target", intent={"kind": "room_default", "project_id": "target"})
    views, repos = [], []
    original_view, original_validate = projects_registry.project_admission_view, workspace_admission.validate_workspace_root
    def view(*args, **kwargs):
        views.append(args)
        if len(views) > 1:  # a second preparation read would observe a replaced registry
            raise OSError("registry replaced after the basis was captured")
        return original_view(*args, **kwargs)
    def validate(requested, **kwargs):
        repos.append(kwargs["system_repo_dir"])
        return original_validate(requested, **kwargs)
    monkeypatch.setattr(projects_registry, "project_admission_view", view)
    monkeypatch.setattr(workspace_admission, "validate_workspace_root", validate)
    room.queue.check_scheduled_tasks()
    [queued] = room.pending
    assert "hold" not in _rows(room)["s1"] and len(views) == 1 and repos == [workers.REPO_DIR]
    assert queued["workspace_root"] == str(folder.resolve())
    assert queued["_project_admission"]["project"]["working_dir"] == str(folder)
    assert load_task_result(room.root, queued["id"])["schedule_admission"]["dispatch"] == "none"


@pytest.mark.parametrize("change", ["benign", "delete"])
def test_real_api_settles_only_its_refused_attempt(room, monkeypatch, tmp_path, change):
    from ouroboros.gateway import tasks as api
    from supervisor import workers
    from starlette.applications import Starlette
    from starlette.requests import Request

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    app = Starlette()
    app.state.drive_root, app.state.repo_dir = room.root, tmp_path / "repo"
    request = Request({"type": "http", "app": app})
    original = api.prepare_task_drive
    drives = []
    def prepare(*args, **kwargs):
        child = original(*args, **kwargs)
        drives.append(child)
        if change == "delete":
            registry.begin_project_deletion(room.root, "target")
        else:
            registry.touch_project(room.root, "other")
        return child
    monkeypatch.setattr(api, "prepare_task_drive", prepare)
    response = api._create_task_from_body(request, {
        "task_id": "api-task", "description": "Synthetic API work", "project_id": "target", "memory_mode": "empty"})
    record = load_task_result(room.root, "api-task")
    assert "api-task" not in room.queue.ADMISSION_RESERVATIONS
    if change == "benign":
        assert response.status_code == 200 and record["status"] == "scheduled"
        assert [row["id"] for row in room.pending] == ["api-task"]
    else:
        assert response.status_code == 409 and record["admission_outcome"] == "never_admitted"
        assert record["reason_code"] == "project_routing_fence"
        assert not room.pending and not drives[0].exists()
        assert room.queue.reserve_task_admission("api-task", "new", drive_root=room.root)["reason"] == "duplicate_task_id"


@pytest.mark.parametrize("failure", ["snapshot", "before_receipt", "after_receipt"])
def test_api_unknown_persistence_never_removes_possible_admission(room, monkeypatch, failure):
    from ouroboros.gateway import tasks as api
    from ouroboros.headless import prepare_task_drive

    child = prepare_task_drive(room.root, "api-unknown", "empty")
    room.queue.reserve_task_admission("api-unknown", "token", drive_root=room.root)
    if failure == "snapshot":
        monkeypatch.setattr(room.queue, "persist_queue_snapshot", lambda **kw: False)
    else:
        original = api.write_task_result
        def write(*args, **kwargs):
            if failure == "after_receipt":
                original(*args, **kwargs)
            raise OSError("synthetic receipt observer failure")
        monkeypatch.setattr(api, "write_task_result", write)
    response = api._complete_api_task_admission(
        task("api-unknown", _admission_token="token", _require_unique_task_id=True),
        drive_root=room.root, task_id="api-unknown", admission_token="token", project_id="target",
        description="Synthetic API work", allowed_resources={}, deadline_at="", workspace_root=None,
        workspace_mode="", memory_mode="empty", child_drive=child, artifacts=[], metadata={})
    assert [row["id"] for row in room.pending] == ["api-unknown"] and child.exists()
    if failure == "after_receipt":
        assert response.status_code == 200
        assert load_task_result(room.root, "api-unknown")["api_admission"] == {"token": "token", "status": "accepted"}
    else:
        assert response.status_code == 503 and json.loads(response.body)["status"] == "unconfirmed"
        assert not load_task_result(room.root, "api-unknown")


def test_own_ensure_returns_the_exact_winning_preparation_row(room, tmp_path, monkeypatch):
    from ouroboros import subagent_worktrees

    chosen = tmp_path / "own-folder"
    chosen.mkdir()
    monkeypatch.setattr(subagent_worktrees, "provision_genesis_project", lambda **kw: SimpleNamespace(path=chosen))
    row = registry.ensure_project_workspace(room.root, "target", tmp_path / "repo", return_project=True)
    basis = registry.project_admission_basis("target", row)
    assert row["working_dir"] == str(chosen)
    assert not room.queue.enqueue_task(task(workspace_root=str(chosen)), project_admission=basis).get("_admission_blocked")


def test_derived_scope_retains_realpath_matching_and_refuses_new_folder_claim(room, tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    registry.update_project(room.root, "target", working_dir=str(folder / ".." / "folder"))
    basis = registry.project_scope_admission(room.root, workspace_root=str(folder))
    assert basis["project_id"] == "target"
    assert not room.queue.enqueue_task(task(), project_admission=basis).get("_admission_blocked")
    unclaimed = tmp_path / "unclaimed"
    unclaimed.mkdir()
    basis = registry.project_scope_admission(room.root, workspace_root=str(unclaimed))
    registry.create_project(room.root, "new-room", working_dir=str(unclaimed))
    refused = room.queue.enqueue_task(task("claim", project_id=basis["project_id"]), project_admission=basis)
    assert refused["_admission_blocked"] == "project_routing_fence_changed"


def test_never_admitted_refusal_is_durable_but_not_a_late_project_completion(room, monkeypatch):
    from ouroboros.terminal_projection import reconcile_terminal_projections
    from supervisor.events_project_routing import _persist_promote_rejection

    registry.bind_task_to_project(room.root, "refused", "target", origin={"absent": "system"})
    _persist_promote_rejection(SimpleNamespace(DRIVE_ROOT=room.root),
                              {"task_id": "refused", "project_id": "target", "objective": "Never started"},
                              {"task_id": "refused", "reason": "project_routing_fence_changed", "never_admitted": True})
    assert load_task_result(room.root, "refused")["admission_outcome"] == "never_admitted"
    queued = []
    monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery", lambda *a, **kw: queued.append(a) or True)
    assert reconcile_terminal_projections(room.root) == 0
    assert not queued
    assert not (room.root / "logs" / "chat.jsonl").exists()
    # A failure lacking this POSITIVE producer fact remains ordinary terminal debt.
    write_task_result(room.root, "executed", "failed", project_id="target", chat_id=1,
                      root_task_id="executed", started_at="2026-01-01T00:00:00Z", result="Actual failure")
    assert load_task_result(room.root, "executed")["canonical_terminal_projection_origin"] == "terminal_transition"
    assert reconcile_terminal_projections(room.root) == 1


def test_subagent_rejection_carries_truthful_cause_and_prepared_drive_cleanup(room):
    from ouroboros.headless import prepare_task_drive
    from supervisor.events_schedule_task import _reject_schedule_task
    from supervisor.task_admission import enqueue_subagent_with_scheduled_result, scheduled_admission_rejection

    child = prepare_task_drive(room.root, "child", "empty")
    basis = registry.project_admission_view(room.root, "target", frozen=True)
    registry.begin_project_deletion(room.root, "target")
    ctx = SimpleNamespace(DRIVE_ROOT=room.root, PENDING=room.pending, RUNNING={}, enqueue_task=room.queue.enqueue_task)
    admitted, reason, _, _ = enqueue_subagent_with_scheduled_result(
        ctx, task("child", _project_admission=basis, parent_task_id="parent", delegation_role="subagent"),
        result_fields={}, admitted_task_contract={}, admitted_depth_provenance={}, direct_child_count=0,
        pending_ref=room.pending)
    assert not reason and admitted["_admission_blocked"] == "project_routing_fence"
    rejection = scheduled_admission_rejection(admitted, project_id="target", root_task_id="parent")
    _reject_schedule_task(ctx, tid="child", chat_id=None, delegation_role="subagent", parent_id="parent",
                          root_task_id="parent", role="researcher", result_fields={"child_drive_root": str(child)}, **rejection)
    assert not child.exists() and not room.pending
    assert load_task_result(room.root, "child")["admission_outcome"] == "never_admitted"
    unreadable = scheduled_admission_rejection(
        {"_admission_blocked": "project_routing_fence_lookup_failed"}, project_id="target", root_task_id="parent")
    assert "could not be checked" in unreadable["detail"] and "closed" not in unreadable["detail"]


from tests.test_swarm_host_admission import host  # noqa: E402, F401


@pytest.mark.parametrize("change", ["benign", "aba"])
def test_promotion_uses_original_folder_basis_and_cleans_refused_fork(host, tmp_path, monkeypatch, change):  # noqa: F811
    from ouroboros import headless, workspace_admission
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    folder = tmp_path / "room-folder"
    folder.mkdir()
    registry.create_project(host.root, "target", working_dir=str(folder))
    registry.create_project(host.root, "other")
    drives = []
    original = headless.prepare_task_drive
    def prepare(*args, **kwargs):
        child = original(*args, **kwargs)
        drives.append(child)
        return child
    def preflight(*args, **kwargs):
        if change == "aba":
            registry.update_project(host.root, "target", working_dir=str(tmp_path / "elsewhere"))
            registry.update_project(host.root, "target", working_dir=str(folder))
        else:
            registry.touch_project(host.root, "other")
        return {}
    monkeypatch.setattr(headless, "prepare_task_drive", prepare)
    monkeypatch.setattr(workspace_admission, "bounded_workspace_preflight", preflight)
    outcome = _handle_promote_chat_to_task({
        "task_id": "promoted", "routing_token": "promotion-token", "objective": "Synthetic promoted work",
        "project_id": "target", "chat_id": 1, "host_initiated": True,
    }, host.ctx)
    if change == "benign":
        assert outcome["status"] == "scheduled"
        assert len(host.pending) == 1 and host.pending[0]["workspace_root"] == str(folder)
        assert load_task_result(host.root, "promoted")["status"] == "scheduled"
    else:
        assert outcome["reason"] == "project_routing_fence_changed" and not host.pending
        assert load_task_result(host.root, "promoted")["admission_outcome"] == "never_admitted"
        assert drives and not drives[0].exists()
        assert len([notice for notice in host.notices if notice.get("system_type") == "task_not_started"]) == 1
    from supervisor import queue

    assert "promoted" not in queue.ADMISSION_RESERVATIONS
    assert not host.attempts


def test_legacy_registered_binding_is_not_laundered_into_an_unregistered_scope(room):
    registry.bind_task_to_project(room.root, "legacy", "target", origin={"absent": "system"})
    data = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))
    data["projects"] = [row for row in data["projects"] if row["id"] != "target"]
    registry._save(room.root, data)
    refused = room.queue.enqueue_task(task("legacy"), restoring_snapshot=True)
    assert refused["_admission_blocked"] == "project_routing_fence_changed"
    assert not room.pending


def test_new_unregistered_api_scope_and_legacy_routing_omissions_remain_supported(room):
    data = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))
    for row in data["projects"]:
        for key in ("lifecycle", "routing_generation", "chat_id", "routing_incarnation"):
            row.pop(key, None)
    registry._save(room.root, data)
    assert not room.queue.enqueue_task(task("legacy"), restoring_snapshot=True).get("_admission_blocked")
    view = registry.project_scope_admission(room.root, project_id="scope-only")
    assert not room.queue.enqueue_task(task("scope", project_id="scope-only"), project_admission=view).get("_admission_blocked")


def test_api_refusal_cannot_terminalize_or_clean_a_new_reservation_owner(room):
    from ouroboros.gateway.tasks import _admission_rejection_response
    from ouroboros.headless import prepare_task_drive

    child = prepare_task_drive(room.root, "attempt", "empty")
    room.queue.reserve_task_admission("attempt", "ours", drive_root=room.root)
    registry.begin_project_deletion(room.root, "target")
    refused = room.queue.enqueue_task(task(_admission_token="ours"))
    assert refused["_admission_blocked"] == "project_routing_fence"
    assert room.queue.reserve_task_admission("attempt", "foreign", drive_root=room.root)["status"] == "reserved"
    with pytest.raises(RuntimeError, match="ownership"):
        _admission_rejection_response(refused, drive_root=room.root, task_id="attempt",
                                       project_id="target", workspace_root=None, child_drive=child)
    assert room.queue.ADMISSION_RESERVATIONS["attempt"] == "foreign"
    assert child.exists() and load_task_result(room.root, "attempt") is None


@pytest.mark.parametrize("late_exception", [False, True])
def test_refusal_receipt_requires_exact_readback_before_cleanup(room, monkeypatch, late_exception):
    from supervisor import task_admission

    original = task_admission.write_task_result
    def write(*args, **kwargs):
        if late_exception:
            original(*args, **kwargs)
        raise OSError("synthetic receipt observer failure")
    monkeypatch.setattr(task_admission, "write_task_result", write)
    if late_exception:
        stored = task_admission.persist_never_admitted_refusal(room.root, "refused", result="Not started")
        assert stored["status"] == "failed" and stored["_admission_refusal_token"]
    else:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            task_admission.persist_never_admitted_refusal(room.root, "refused", result="Not started")
        assert load_task_result(room.root, "refused") is None


def test_subagent_foreign_reservation_reason_never_becomes_acceptance_refusal():
    from supervisor.task_admission import scheduled_admission_rejection

    for reason in ("admission_reservation_owned", "admission_reservation_lost"):
        result = scheduled_admission_rejection({"_admission_blocked": reason}, project_id="target", root_task_id="root")
        assert result["reason_code"] == reason and result["persist_result"] is False


def corrupt_neighbor(root, **fields):
    path = registry._registry_path(root)
    data = json.loads(path.read_text(encoding="utf-8"))
    row = next(row for row in data["projects"] if row["id"] == "other")
    row.update(fields)
    if "working_dir" not in fields:
        row.pop("working_dir", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    return row


@pytest.mark.parametrize("operation", ["create", "rename", "folder", "bind", "visible_id",
                                        "visible_chat", "delete", "delete_failure", "tombstone"])
def test_registry_writers_preserve_unrelated_raw_rows(room, operation):
    if operation in {"delete_failure", "tombstone"}:
        registry.begin_project_deletion(room.root, "target")
    raw = corrupt_neighbor(room.root, routing_generation="broken", lifecycle=["unknown"],
                           visible_revision={"opaque": 7}, delete_error=["preserve"], extra={"x": None})
    if operation == "create":
        changed = registry.create_project(room.root, "fresh", name="Fresh")
        assert changed["created"] is True
    elif operation in {"rename", "folder"}:
        changed = registry.update_project(room.root, "target", **{
            "name" if operation == "rename" else "working_dir": "updated"})
        assert changed["name" if operation == "rename" else "working_dir"] == "updated"
    elif operation == "bind":
        changed = registry.bind_task_to_project(room.root, "bound", "target", origin={"absent": "system"})
        assert changed["project_id"] == "target"
    elif operation.startswith("visible"):
        changed = registry.increment_project_visible_revision(room.root, **(
            {"project_id": "target"} if operation == "visible_id"
            else {"chat_id": registry.project_admission_view(room.root, "target")["project"]["chat_id"]}))
        assert changed["visible_revision"] == 1
    else:
        changed = {"delete": registry.begin_project_deletion,
                   "delete_failure": lambda root, pid: registry.fail_project_deletion(root, pid, "pending"),
                   "tombstone": registry.complete_project_deletion}[operation](room.root, "target")
        assert changed["lifecycle"] == ("tombstoned" if operation == "tombstone" else "deleting")
    rows = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))["projects"]
    assert next(row for row in rows if row["id"] == "other") == raw
    assert "working_dir" not in next(row for row in rows if row["id"] == "other")


@pytest.mark.parametrize("corruption", ["target", "duplicate_id", "duplicate_chat", "json"])
def test_selected_writer_refuses_invalid_authority_without_changing_bytes(room, corruption):
    path = registry._registry_path(room.root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "target":
        data["projects"][0]["routing_generation"] = "unknown"
    elif corruption == "duplicate_id":
        data["projects"].append(dict(data["projects"][0]))
    elif corruption == "duplicate_chat":
        data["projects"][1]["chat_id"] = data["projects"][0]["chat_id"]
    path.write_text("{torn" if corruption == "json" else json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        registry.update_project(room.root, "target", name="Must not land")
    assert path.read_bytes() == before


@pytest.mark.parametrize("shape", ["existing_folder", "existing_empty", "new_room"])
def test_real_promotion_provisions_healthy_room_beside_malformed_neighbor(host, tmp_path, monkeypatch, shape):  # noqa: F811
    from ouroboros import subagent_worktrees
    from supervisor import workers
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    folder = tmp_path / "prepared-room"
    folder.mkdir()
    if shape != "new_room":
        registry.create_project(host.root, "target", working_dir=str(folder) if shape == "existing_folder" else "")
    registry.create_project(host.root, "other")
    raw = corrupt_neighbor(host.root, routing_generation="unknown", visible_revision=[1], delete_error={"x": 2})
    monkeypatch.setattr(subagent_worktrees, "provision_genesis_project", lambda **kw: SimpleNamespace(path=folder))
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    outcome = _handle_promote_chat_to_task({
        "task_id": "healthy-promotion", "routing_token": "healthy-token", "objective": "Synthetic work",
        "project_id": "target", "chat_id": 1, "host_initiated": True,
    }, host.ctx)
    assert outcome["status"] == "scheduled", outcome
    [queued] = host.pending
    assert queued["id"] == "healthy-promotion" and queued["workspace_root"] == str(folder)
    assert load_task_result(host.root, queued["id"])["status"] == "scheduled"
    assert registry.project_binding_for_task(host.root, queued["id"])["project_id"] == "target"
    rows = json.loads(registry._registry_path(host.root).read_text(encoding="utf-8"))["projects"]
    assert next(row for row in rows if row["id"] == "other") == raw
    assert not host.attempts


@pytest.mark.parametrize("operation", ["create", "rename", "delete", "folder", "route"])
def test_named_project_consumers_keep_healthy_neighbor_available(room, tmp_path, monkeypatch, operation):
    import asyncio
    from ouroboros.gateway import projects
    from ouroboros.workspace_admission import room_chat_lens_dir
    from ouroboros.tools.control_routing import _route_to_project
    from tests.test_route_to_project import _ctx

    folder = tmp_path / "target-folder"
    folder.mkdir()
    registry.update_project(room.root, "target", working_dir=str(folder))
    raw = corrupt_neighbor(room.root, routing_generation="broken", visible_revision=[1])
    calls = []
    monkeypatch.setattr("supervisor.task_lifecycle.start_project_deletion", lambda *a: calls.append(a))
    monkeypatch.setattr(projects, "_broadcast_projects_changed", lambda *a: None)

    class Request:
        app = SimpleNamespace(state=SimpleNamespace(drive_root=room.root, repo_dir=tmp_path / "repo"))
        path_params = {"project_id": "target"}

        async def json(self):
            return {"id": "fresh-room", "name": "Updated"}

    if operation == "folder":
        assert room_chat_lens_dir(room.root, "target") == (str(folder), "")
    elif operation == "route":
        events = []
        answer = _route_to_project(_ctx(room.root, events), "target", "Continue", predecessor_task_id="")
        assert answer.startswith("⚠️ ROUTE_UNCONFIRMED:"), answer
        assert len(events) == 1 and events[0]["project_id"] == "target"
    else:
        handler = {"create": projects.api_projects_create, "rename": projects.api_project_update,
                   "delete": projects.api_project_delete}[operation]
        response = asyncio.run(handler(Request()))
        assert response.status_code == 200, response.body
        if operation == "delete":
            assert len(calls) == 1 and calls[0][1] == "target"
            assert registry.get_reserved_project(room.root, "target", strict=True)["lifecycle"] == "deleting"
        else:
            pid = "fresh-room" if operation == "create" else "target"
            assert registry.get_project(room.root, pid, strict=True)["name"] == "Updated"
    rows = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))["projects"]
    assert next(row for row in rows if row["id"] == "other") == raw


def test_conversion_adopts_origin_room_beside_malformed_neighbor(room, monkeypatch):
    from tests.test_project_lease_ui_conversion import _convert, _origin_ref, _seed_origin_task, _live_queue, _OWNER_TEXT

    ref = _origin_ref()
    _seed_origin_task(room.root, "turn", ref)
    registry.bind_task_to_project(room.root, "origin-root", "target", origin={"ref": ref, "text": _OWNER_TEXT})
    raw = corrupt_neighbor(room.root, lifecycle=[], working_dir={"unavailable": True})
    _live_queue(monkeypatch, room.root, {}, [])
    assert registry.project_id_for_origin(room.root, ref, strict=True) == "target"
    response = _convert(room.root, "turn")
    body = json.loads(response.body)
    assert response.status_code == 200 and body["adopted"] is True, body
    assert body["project"]["id"] == "target"
    assert registry.project_binding_for_task(room.root, "turn")["project_id"] == "target"
    rows = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))["projects"]
    assert {row["id"] for row in rows} == {"target", "other"}
    assert next(row for row in rows if row["id"] == "other") == raw


@pytest.mark.parametrize("corruption", ["target", "identity", "json"])
def test_named_strict_readers_preserve_refusal_for_relevant_unknown_authority(room, corruption):
    from tests.test_project_lease_ui_conversion import _origin_ref, _OWNER_TEXT

    ref = _origin_ref()
    registry.bind_task_to_project(room.root, "bound", "target", origin={"ref": ref, "text": _OWNER_TEXT})
    path = registry._registry_path(room.root)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "target":
        rows["projects"][0]["routing_generation"] = "broken"
    elif corruption == "identity":
        rows["projects"][1]["chat_id"] = rows["projects"][0]["chat_id"]
    path.write_text("{torn" if corruption == "json" else json.dumps(rows), encoding="utf-8")
    before = path.read_bytes()
    for read in (lambda: registry.get_project(room.root, "target", strict=True),
                 lambda: registry.get_reserved_project(room.root, "target", strict=True),
                 lambda: registry.project_id_for_origin(room.root, ref, strict=True)):
        with pytest.raises(ValueError):
            read()
        assert path.read_bytes() == before
