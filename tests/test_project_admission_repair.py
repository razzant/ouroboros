"""Discriminating producer/cleanup boundaries for semantic project admission."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_shared import unreadable_result
from tests.test_project_semantic_admission import room, task  # noqa: F401
from tests.test_schedule_occurrence import q  # noqa: F401
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def _drop_project(root):
    data = json.loads(registry._registry_path(root).read_text(encoding="utf-8"))
    data["projects"] = [row for row in data["projects"] if row["id"] != "target"]
    registry._save(root, data)


@pytest.mark.parametrize("receipt", ["ours", "foreign", "unreadable", "unconfirmed", "late_write"])
def test_promotion_attachments_wait_for_exact_refusal_receipt(host, tmp_path, monkeypatch, receipt):  # noqa: F811
    from ouroboros import artifacts
    from supervisor import task_admission
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    registry.create_project(host.root, "target")
    source = tmp_path / "input.txt"
    source.write_text("must survive until custody is confirmed", encoding="utf-8")
    captured = []
    real_stage = artifacts.stage_task_attachments
    def stage(*args, **kwargs):
        manifest = real_stage(*args, **kwargs)
        captured.extend(Path(row["abs_path"]) for row in manifest)
        return manifest
    monkeypatch.setattr(artifacts, "stage_task_attachments", stage)
    real_enqueue = host.ctx.enqueue_task
    from supervisor import queue
    def enqueue(payload):
        registry.begin_project_deletion(host.root, "target")
        rejected = real_enqueue(payload)
        if receipt == "foreign":
            assert queue.reserve_task_admission("promoted", "foreign", drive_root=host.root)["status"] == "reserved"
        return rejected
    host.ctx.enqueue_task = enqueue
    real_write = task_admission.write_task_result
    if receipt in {"unconfirmed", "late_write"}:
        def write(*args, **kwargs):
            if receipt == "late_write":
                real_write(*args, **kwargs)
            raise OSError("synthetic receipt observer failure")
        monkeypatch.setattr(task_admission, "write_task_result", write)
    if receipt == "unreadable":
        monkeypatch.setattr(task_admission, "load_task_result", lambda *a, **k: (_ for _ in ()).throw(PermissionError("synthetic read failure")))
    result = _handle_promote_chat_to_task({
        "task_id": "promoted", "routing_token": "ours", "objective": "Use the input",
        "project_id": "target", "workspace": "none", "chat_id": 1,
        "attachment_uploads": [{"path": str(source)}],
    }, host.ctx)
    assert captured and not host.pending
    assert not any(key.startswith("_admission_cleanup") for key in result)
    if receipt in {"ours", "late_write"}:
        assert load_task_result(host.root, "promoted")["admission_outcome"] == "never_admitted"
        assert all(not path.exists() for path in captured)
    else:
        assert result["status"] == "unconfirmed"
        assert all(path.read_text(encoding="utf-8") == source.read_text(encoding="utf-8") for path in captured)
        assert load_task_result(host.root, "promoted") is None
    if receipt == "foreign":
        assert queue.ADMISSION_RESERVATIONS["promoted"] == "foreign"


@pytest.mark.parametrize("occupant", ["ours", "foreign_reservation", "foreign_result", "unreadable", "pending"])
def test_api_preparation_cleanup_keeps_ownership_until_resources_settle(room, monkeypatch, occupant):  # noqa: F811
    from ouroboros import headless
    from ouroboros.gateway.tasks import _cleanup_api_admission_attempt

    tid = "preparation"
    child = headless.prepare_task_drive(room.root, tid, "empty")
    artifact = headless.task_artifacts_dir(room.root, tid) / "keep.txt"
    artifact.write_text("owned input", encoding="utf-8")
    room.queue.reserve_task_admission(tid, "ours", drive_root=room.root)
    if occupant == "foreign_reservation":
        room.queue.ADMISSION_RESERVATIONS[tid] = "foreign"
    elif occupant == "foreign_result":
        write_task_result(room.root, tid, "scheduled", result="foreign")
    elif occupant == "unreadable":
        unreadable_result(monkeypatch, tid)  # the store cannot say whose the id is: cleanup keeps ownership
    elif occupant == "pending":
        room.pending.append(task(tid))
    original = headless.remove_subagent_task_drive
    def remove(*args, **kwargs):
        assert room.queue.ADMISSION_RESERVATIONS.get(tid) == "ours", "reservation released before cleanup"
        return original(*args, **kwargs)
    monkeypatch.setattr(headless, "remove_subagent_task_drive", remove)
    _cleanup_api_admission_attempt(room.root, tid, "ours", child)
    if occupant == "ours":
        assert not child.exists() and not artifact.exists()
        assert tid not in room.queue.ADMISSION_RESERVATIONS
    else:
        assert child.exists() and artifact.read_text(encoding="utf-8") == "owned input"
        assert room.queue.ADMISSION_RESERVATIONS[tid] == ("foreign" if occupant == "foreign_reservation" else "ours")


@pytest.mark.parametrize("phase", ["partial_drive", "staging", "composition"])
def test_real_api_known_preparation_exception_cleans_only_its_attempt(room, tmp_path, monkeypatch, phase):  # noqa: F811
    from ouroboros.gateway import tasks as api
    from ouroboros.headless import task_artifacts_dir
    from starlette.requests import Request
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=room.root, repo_dir=tmp_path / "repo"))})
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    child_paths = []
    original = api.prepare_task_drive
    def prepare(*args, **kwargs):
        child = original(*args, **kwargs)
        child_paths.append(child)
        if phase == "partial_drive":
            raise OSError("synthetic partial drive failure")
        return child
    monkeypatch.setattr(api, "prepare_task_drive", prepare)
    if phase != "partial_drive":
        target = "stage_initial_task_attachments" if phase == "staging" else "_compose_task_text"
        monkeypatch.setattr(api, target, lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic preparation failure")))
    response = api._create_task_from_body(request, {
        "task_id": "prepare-failed", "description": "Use input", "project_id": "target",
        "memory_mode": "empty", "attachments": [{"path": str(source)}]})
    assert response.status_code == 503 and not room.pending
    assert "prepare-failed" not in room.queue.ADMISSION_RESERVATIONS
    assert child_paths and not child_paths[0].exists()
    assert not task_artifacts_dir(room.root, "prepare-failed", create=False).exists()
    assert load_task_result(room.root, "prepare-failed") is None


@pytest.mark.parametrize("stage", ["before_scope", "before_bind", "before_ensure", "during_ensure"])
def test_promotion_never_recreates_a_vanished_prepared_room(host, tmp_path, monkeypatch, stage):  # noqa: F811
    from ouroboros import genesis_project
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    registry.create_project(host.root, "target")
    basis = registry.project_admission_view(host.root, "target")
    if stage == "before_scope":
        _drop_project(host.root)
    else:
        name = "bind_task_to_project" if stage == "before_bind" else "ensure_project_workspace"
        original = getattr(registry, name)
        if stage != "during_ensure":
            def changed(*args, **kwargs):
                _drop_project(host.root)
                return original(*args, **kwargs)
            monkeypatch.setattr(registry, name, changed)
    folder = tmp_path / "provisioned"
    folder.mkdir()
    def provision(**kwargs):
        if stage == "during_ensure":
            _drop_project(host.root)
        return SimpleNamespace(path=folder)
    monkeypatch.setattr(genesis_project, "provision_genesis_project", provision)
    result = _handle_promote_chat_to_task({
        "task_id": "gone-room", "routing_token": "ours", "objective": "Use prepared room",
        "project_id": "target", "chat_id": 1, "_project_admission": basis,
    }, host.ctx)
    assert result["status"] != "scheduled" and not host.pending
    assert registry.get_reserved_project(host.root, "target") is None


def test_source_preparation_does_not_recreate_a_vanished_room(room, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import project_sources
    from ouroboros.promotion_source import resolve_promote_source

    folder = tmp_path / "attached"
    folder.mkdir()
    original = project_sources.validate_attach_path
    def validate(*args, **kwargs):
        result = original(*args, **kwargs)
        _drop_project(room.root)
        return result
    monkeypatch.setattr(project_sources, "validate_attach_path", validate)
    result = resolve_promote_source(SimpleNamespace(DRIVE_ROOT=room.root, REPO_DIR=tmp_path / "repo"), str(folder), "target")
    assert result[2] and not result[0]
    assert registry.get_reserved_project(room.root, "target") is None


@pytest.mark.parametrize("bad_id", [7, {"bad": "authority"}, "target/invalid", " target "])
def test_legacy_malformed_binding_is_unknown_not_unregistered(room, bad_id):  # noqa: F811
    _drop_project(room.root)
    registry._save_bindings(room.root, {"bindings": {"legacy": {"project_id": bad_id}}})
    rejected = room.queue.enqueue_task(task("legacy"), restoring_snapshot=True)
    assert rejected["_admission_blocked"] == "project_routing_fence_lookup_failed"
    assert not room.pending


@pytest.mark.parametrize("field,value", [
    ("lifecycle", None), ("routing_generation", True), ("routing_generation", "0"),
    ("routing_generation", 0.0), ("chat_id", False), ("chat_id", "4"),
    ("working_dir", None), ("routing_incarnation", ""), ("created_at", []),
])
def test_selected_writers_preserve_invalid_neighbor_without_laundering_it(room, field, value):  # noqa: F811
    path = registry._registry_path(room.root)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["projects"][0][field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):  # the malformed selected target always refuses
        registry.update_project(room.root, "target", name="rename")
    assert path.read_bytes() == before
    if field == "chat_id":  # ambiguous/invalid identity blocks every room
        with pytest.raises(ValueError):
            registry.create_project(room.root, "new")
        registry.touch_project(room.root, "other")
        registry.reconcile_projects(room.root)
        assert path.read_bytes() == before
    else:
        assert registry.create_project(room.root, "new")["created"] is True
        registry.touch_project(room.root, "other")
        registry.reconcile_projects(room.root)
        after = json.loads(path.read_text(encoding="utf-8"))
        assert next(row for row in after["projects"] if row["id"] == "target") == data["projects"][0]
        assert {row["id"] for row in after["projects"]} == {"target", "other", "new"}


@pytest.mark.parametrize("raw", [
    '{"projects":[],"projects":[]}',
    '{"projects":[{"id":"target","id":"other"}]}',
    '{"projects":[null]}',
    '{"projects":[{"id":"target/invalid"}]}',
])
def test_duplicate_or_malformed_raw_authority_never_admits_or_rewrites(room, raw):  # noqa: F811
    basis = registry.project_admission_view(room.root, "target")
    path = registry._registry_path(room.root)
    path.write_text(raw, encoding="utf-8")
    rejected = room.queue.enqueue_task(task(), project_admission=basis)
    assert rejected["_admission_blocked"] == "project_routing_fence_lookup_failed"
    with pytest.raises(ValueError):
        registry.create_project(room.root, "new")
    assert path.read_text(encoding="utf-8") == raw and not room.pending


def test_legacy_omitted_routing_fields_survive_writer_and_new_preparation(room, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import genesis_project

    path = registry._registry_path(room.root)
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["projects"]:
        for field in ("lifecycle", "routing_generation", "chat_id", "working_dir", "routing_incarnation", "created_at"):
            row.pop(field, None)
    path.write_text(json.dumps(data), encoding="utf-8")
    registry.touch_project(room.root, "other")
    basis = registry.project_admission_view(room.root, "target")
    assert basis["project"]["lifecycle"] == "active" and basis["project"]["routing_generation"] == 0
    folder = tmp_path / "legitimate-new-room"
    folder.mkdir()
    monkeypatch.setattr(genesis_project, "provision_genesis_project", lambda **kw: SimpleNamespace(path=folder))
    missing = registry.project_admission_view(room.root, "new-room", allow_unregistered=True)
    created = registry.ensure_project_workspace(room.root, "new-room", tmp_path / "repo",
                                                return_project=True, admission_basis=missing)
    assert created["working_dir"] == str(folder) and created["routing_incarnation"]
    assert not room.queue.enqueue_task(task("legacy"), project_admission=basis).get("_admission_blocked")


def test_api_exception_after_enqueue_preserves_possible_admission(room, monkeypatch):  # noqa: F811
    from ouroboros.gateway import tasks as api
    from ouroboros.headless import prepare_task_drive

    child = prepare_task_drive(room.root, "late-enqueue", "empty")
    room.queue.reserve_task_admission("late-enqueue", "ours", drive_root=room.root)
    enqueue = room.queue.enqueue_task
    def late_exception(payload):
        enqueue(payload)
        raise OSError("synthetic exception after append")
    monkeypatch.setattr(room.queue, "enqueue_task", late_exception)
    response = api._complete_api_task_admission(
        task("late-enqueue", _admission_token="ours", _require_unique_task_id=True),
        drive_root=room.root, task_id="late-enqueue", admission_token="ours", project_id="target",
        description="Synthetic API work", allowed_resources={}, deadline_at="", workspace_root=None,
        workspace_mode="", memory_mode="empty", child_drive=child, artifacts=[], metadata={})
    assert response.status_code == 503 and json.loads(response.body)["status"] == "unconfirmed"
    assert [row["id"] for row in room.pending] == ["late-enqueue"] and child.exists()
    assert load_task_result(room.root, "late-enqueue") is None
    assert room.queue.reserve_task_admission("late-enqueue", "other", drive_root=room.root)["reason"] == "duplicate_task_id"


@pytest.mark.parametrize("foreign", [False, True])
def test_api_acceptance_refusal_uses_the_same_exact_cleanup_boundary(room, monkeypatch, foreign):  # noqa: F811
    from ouroboros.gateway import tasks as api
    from ouroboros.headless import prepare_task_drive

    tid = "fenced"
    child = prepare_task_drive(room.root, tid, "empty")
    room.queue.reserve_task_admission(tid, "ours", drive_root=room.root)
    monkeypatch.setitem(room.queue.ACCEPTANCE_FENCES, tid, {"status": "active", "token": "acceptance"})
    enqueue = room.queue.enqueue_task
    def blocked(payload):
        result = enqueue(payload)
        assert result["_admission_blocked"] == "task_acceptance_fence"
        if foreign:
            assert room.queue.reserve_task_admission(tid, "foreign", drive_root=room.root)["status"] == "reserved"
        return result
    monkeypatch.setattr(room.queue, "enqueue_task", blocked)
    response = api._complete_api_task_admission(
        task(tid, _admission_token="ours", _require_unique_task_id=True), drive_root=room.root,
        task_id=tid, admission_token="ours", project_id="target", description="Do work",
        allowed_resources={}, deadline_at="", workspace_root=None, workspace_mode="",
        memory_mode="empty", child_drive=child, artifacts=[], metadata={})
    assert not room.pending
    if foreign:
        assert response.status_code == 503 and child.exists()
        assert room.queue.ADMISSION_RESERVATIONS[tid] == "foreign"
        assert load_task_result(room.root, tid) is None
    else:
        assert response.status_code == 409 and not child.exists()
        assert tid not in room.queue.ADMISSION_RESERVATIONS
        assert load_task_result(room.root, tid)["reason_code"] == "task_acceptance_fence"


def test_promote_post_stage_lookup_failure_cleans_attachment(host, tmp_path, monkeypatch):  # noqa: F811
    """The producer settles an exact refusal before freeing the staged input."""
    from ouroboros import artifacts
    from supervisor import queue
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    cleaned = []
    remove = artifacts.remove_staged_attachments
    def cleanup(manifest):
        receipt = load_task_result(host.root, "attach-lookup-fail", strict=True)
        assert receipt["admission_outcome"] == "never_admitted"
        assert receipt["_admission_refusal_token"]
        cleaned.extend(Path(row["abs_path"]) for row in manifest)
        assert cleaned and all(path.read_text(encoding="utf-8") == "input" for path in cleaned)
        remove(manifest)
    monkeypatch.setattr(artifacts, "remove_staged_attachments", cleanup)
    monkeypatch.setattr(registry, "project_admission_view", lambda *a, **k:
                        (_ for _ in ()).throw(OSError("registry unavailable")))
    outcome = _handle_promote_chat_to_task({
        "task_id": "attach-lookup-fail", "routing_token": "lookup-token", "chat_id": 1,
        "objective": "use input", "project_id": "lookup-project",
        "attachment_uploads": [{"path": str(source)}],
    }, host.ctx)
    assert outcome["reason"] == "project_routing_fence_lookup_failed"
    assert not host.pending and "attach-lookup-fail" not in queue.ADMISSION_RESERVATIONS
    assert cleaned and all(not path.exists() for path in cleaned)


@pytest.mark.parametrize("failure", ["enqueue", "snapshot"])
def test_promote_queue_failure_preserves_receipt_first_custody(host, tmp_path, monkeypatch, failure):  # noqa: F811
    """Refusal is certain only before admission; uncertain persistence retains inputs."""
    from supervisor import queue
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    registry.create_project(host.root, "target")
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    captured = []
    real_enqueue = host.ctx.enqueue_task
    def enqueue(payload):
        captured.append(payload)
        if failure == "enqueue":
            registry.begin_project_deletion(host.root, "target")
        return real_enqueue(payload)
    host.ctx.enqueue_task = enqueue
    if failure == "snapshot":
        host.ctx.persist_queue_snapshot = lambda **kwargs: False
    outcome = _handle_promote_chat_to_task({
        "task_id": "promoted", "routing_token": "ours", "chat_id": 1,
        "objective": "use input", "project_id": "target", "workspace": "none",
        "attachment_uploads": [{"path": str(source)}],
    }, host.ctx)
    assert captured
    staged = Path(captured[0]["attachments"][0]["abs_path"])
    receipt = load_task_result(host.root, "promoted") or {}
    assert "promoted" not in queue.ADMISSION_RESERVATIONS
    if failure == "enqueue":
        assert outcome["reason"] == "project_routing_fence"
        assert receipt["admission_outcome"] == "never_admitted" and receipt["_admission_refusal_token"]
        assert not host.pending and not staged.exists()
    else:
        assert outcome["status"] == "unconfirmed"
        assert outcome["reason"] == "queue_snapshot_persist_failed"
        assert [row["id"] for row in host.pending] == ["promoted"]
        assert staged.read_text(encoding="utf-8") == "input"
        assert receipt.get("admission_outcome") != "never_admitted"
        assert receipt.get("status") != "failed"
