"""#1381 remaining contracts through real queue, API, restore and schedule owners."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result
from tests.test_project_semantic_admission import room, task  # noqa: F401
from tests.test_schedule_occurrence import q, _row, _rows  # noqa: F401
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


def change_row(root, pid, **fields):
    path = registry._registry_path(root)
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["projects"]:
        if row["id"] == pid:
            row.update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")


def drop(root, pid):
    path = registry._registry_path(root)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["projects"] = [r for r in data["projects"] if r["id"] != pid]
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("change", ["create", "rebind", "delete", "remove", "reorder"])
def test_derived_admission_ignores_irrelevant_routing_edits(room, tmp_path, selected, change):  # noqa: F811
    folder = tmp_path / "chosen"
    folder.mkdir()
    if selected:
        registry.update_project(room.root, "target", working_dir=str(folder))
    basis = registry.project_scope_admission(room.root, workspace_root=str(folder))
    if change == "create":
        registry.create_project(room.root, "new", working_dir=str(tmp_path / "other-folder"))
    elif change == "rebind":
        registry.update_project(room.root, "other", working_dir=str(tmp_path / "other-folder"))
    elif change == "delete":
        registry.begin_project_deletion(room.root, "other")
    elif change == "remove":
        drop(room.root, "other")
    else:
        path = registry._registry_path(room.root)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["projects"].reverse()
        path.write_text(json.dumps(data), encoding="utf-8")
    admitted = room.queue.enqueue_task(task(project_id=basis["project_id"], workspace_root=str(folder)), project_admission=basis)
    assert room.pending == [admitted] and not admitted.get("_admission_blocked")


@pytest.mark.parametrize("change", ["new", "moved", "lost", "aba", "competing", "reorder"])
def test_relevant_claim_edits_are_not_laundered(room, tmp_path, change):  # noqa: F811
    folder = tmp_path / "chosen"
    folder.mkdir()
    registry.update_project(room.root, "target", working_dir=str(folder))
    if change == "reorder":
        registry.update_project(room.root, "other", working_dir=str(folder))
    basis = registry.project_scope_admission(room.root, workspace_root=str(folder))
    if change in {"new", "competing"}:
        registry.create_project(room.root, "new", working_dir=str(folder / ".." / "chosen"))
    elif change == "moved":
        registry.update_project(room.root, "other", working_dir=str(folder))
    elif change in {"lost", "aba"}:
        registry.update_project(room.root, "target", working_dir=str(tmp_path / "away"))
        if change == "aba":
            registry.update_project(room.root, "target", working_dir=str(folder))
    else:
        path = registry._registry_path(room.root)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["projects"].reverse()
        path.write_text(json.dumps(data), encoding="utf-8")
    refused = room.queue.enqueue_task(task(), project_admission=basis)
    assert refused["_admission_blocked"] == "project_routing_fence_changed" and not room.pending


@pytest.mark.parametrize("bad", [None, [], "basis", 4, {}, {"frozen": True},
    {"project_id": "target", "project": "bad", "frozen": True}])
def test_invalid_carried_basis_keeps_mixed_restore_and_raw_evidence(room, bad):  # noqa: F811
    room.pending.extend([task("broken", _project_admission=bad), task("main", project_id="")])
    assert room.queue.persist_queue_snapshot(reason="synthetic invalid carrier")
    room.pending.clear()
    room.queue.restore_pending_from_snapshot()
    rows = {r["id"]: r for r in room.pending}
    assert set(rows) == {"broken", "main"}
    assert rows["broken"]["_project_admission"] == bad
    assert rows["broken"]["_project_admission_restore_hold"]["reason"] == "project_routing_fence_lookup_failed"
    assert not load_task_result(room.root, "broken")
    snapshot = json.loads(room.queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    kept = {r["task"]["id"]: r["task"] for r in snapshot["pending"]}
    assert kept["broken"]["_project_admission"] == bad
    room.pending.clear()
    room.queue.restore_pending_from_snapshot()
    assert {r["id"] for r in room.pending} == {"broken", "main"}


@pytest.mark.parametrize("field,value", [("frozen", 1), ("registry_present", "yes"),
    ("project", []), ("workspace_claims", []), ("workspace_claims", {"workspace": []})])
def test_invalid_carrier_fields_are_typed_before_enqueue(room, field, value):  # noqa: F811
    basis = registry.project_admission_view(room.root, "target")
    basis[field] = value
    denied = room.queue.enqueue_task(task(_project_admission=basis))
    assert denied["_admission_blocked"] == "project_routing_fence_lookup_failed"
    assert not room.pending


def test_unreadable_restore_conserves_without_releasing_when_source_recovers(room):  # noqa: F811
    room.queue.enqueue_task(task("held"))
    room.queue.enqueue_task(task("main", project_id=""))
    room.queue.persist_queue_snapshot()
    path = registry._registry_path(room.root)
    original = path.read_bytes()
    path.write_text("{torn", encoding="utf-8")
    room.pending.clear()
    room.queue.restore_pending_from_snapshot()
    assert {r["id"] for r in room.pending} == {"held", "main"}
    path.write_bytes(original)
    room.pending.clear()
    room.queue.restore_pending_from_snapshot()
    held = next(r for r in room.pending if r["id"] == "held")
    assert held["_project_admission_restore_hold"] and not load_task_result(room.root, "held")


@pytest.mark.parametrize("bad", [{"lifecycle": []}, {"routing_generation": {}}, {"routing_generation": "0"},
                                 {"working_dir": []}])
def test_healthy_display_thread_and_source_identity_survive_malformed_neighbor(room, tmp_path, bad):  # noqa: F811
    registry.bind_task_to_project(room.root, "bound", "target", origin={"absent": "system"})
    target = registry.get_project(room.root, "target")
    change_row(room.root, "other", **bad)
    assert any(row["id"] == "target" and row["name"] == "Target" for row in registry.projects_summary(room.root))
    assert target["chat_id"] in registry.reserved_project_chat_ids(room.root)
    assert target["chat_id"] in registry.reserved_project_chat_ids(room.root, strict=True)
    frame = {"chat_id": target["chat_id"]}
    registry.stamp_project_thread(room.root, frame)
    assert frame["project_thread"] is True  # Main exclusion used by live/history consumers
    assert registry.task_presentation_snapshot(room.root, "bound")["project_name"] == "Target"
    # Healthy named operations preserve unknown neighbours; target authority and
    # the whole-registry derived-folder census remain strict.
    assert not room.queue.enqueue_task(task()).get("_admission_blocked")
    assert registry.project_scope_admission(room.root, project_id="target")["project"]["id"] == "target"
    assert room.queue.enqueue_task(task("other-work", project_id="other"))["_admission_blocked"] == (
        "project_routing_fence_lookup_failed")
    with pytest.raises(ValueError):
        registry.project_scope_admission(room.root, workspace_root=str(tmp_path))
    raw = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))
    neighbor = next(row for row in raw["projects"] if row["id"] == "other")
    assert registry.update_project(room.root, "target", name="Updated")["name"] == "Updated"
    current = json.loads(registry._registry_path(room.root).read_text(encoding="utf-8"))
    assert next(row for row in current["projects"] if row["id"] == "other") == neighbor
    assert [row["id"] for row in room.pending] == ["attempt"]


@pytest.mark.parametrize("bad", [{"chat_id": "4"}, {"id": "other/invalid"}, "duplicate"])
def test_malformed_neighbor_identity_still_refuses_every_room(room, bad):  # noqa: F811
    from ouroboros.server_routing_context import _reserved_project_for_chat

    chat = registry.get_project(room.root, "target")["chat_id"]
    if bad == "duplicate":
        change_row(room.root, "other", id="target")
    else:
        change_row(room.root, "other", **bad)
    assert room.queue.enqueue_task(task())["_admission_blocked"] == "project_routing_fence_lookup_failed"
    with pytest.raises(ValueError):
        _reserved_project_for_chat(SimpleNamespace(DRIVE_ROOT=room.root), chat)
    with pytest.raises(ValueError):
        registry.reserved_project_chat_ids(room.root, strict=True)
    assert not room.pending


@pytest.mark.parametrize("intent_kind", ["explicit_none", "explicit_resource", "room_default"])
@pytest.mark.parametrize("evidence", ["origin_task_id", "origin_root_task_id"])
def test_schedule_followup_keeps_known_membership_after_room_loss(room, tmp_path, intent_kind, evidence):  # noqa: F811
    registry.bind_task_to_project(room.root, "origin", "target", origin={"absent": "system"})
    folder = tmp_path / "frozen"
    folder.mkdir()
    intent = {"kind": intent_kind, "project_id": "target", "root": str(folder)}
    # Followup origins are host-recorded task+root pairs; root evidence names an unbound child.
    _row(room, intent=intent, metadata={evidence: "origin"} if evidence == "origin_task_id"
         else {"origin_task_id": "origin-child", "origin_root_task_id": "origin"})
    drop(room.root, "target")
    room.queue.check_scheduled_tasks()
    assert not room.pending
    assert _rows(room)["s1"]["hold"]["reason"] == "project_routing_fence_changed"


def test_schedule_intent_only_room_and_unregistered_scope_keep_distinct_authority(room):  # noqa: F811
    _row(room, "known", intent={"kind": "explicit_none", "project_id": "target"})
    _row(room, "unregistered", project_id="scope-only", intent={"kind": "explicit_none"})
    room.queue.check_scheduled_tasks()
    assert {r["project_id"] for r in room.pending} == {"target", "scope-only"}
    known = next(r for r in room.pending if r["project_id"] == "target")
    assert known["_project_admission"]["project"]["id"] == "target"


def test_strict_incoming_routing_cannot_use_display_fallback(room):  # noqa: F811
    from ouroboros.server_routing_context import _reserved_project_for_chat

    ctx = SimpleNamespace(DRIVE_ROOT=room.root)
    project_chat = registry.get_project(room.root, "target")["chat_id"]
    other_chat = registry.get_project(room.root, "other")["chat_id"]
    change_row(room.root, "other", lifecycle=[])
    assert _reserved_project_for_chat(ctx, project_chat)["id"] == "target"  # a healthy room keeps routing
    with pytest.raises(ValueError):  # the malformed room itself is unavailable, never Main
        _reserved_project_for_chat(ctx, other_chat)
    assert _reserved_project_for_chat(ctx, 1) == {}


@pytest.mark.parametrize("sender_room", ["target", "other", None])
def test_explicit_cross_room_relay_keeps_its_note_separate_from_owner_words(room, sender_room):  # noqa: F811
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor.steering import _handle_steer_task

    target = {"id": "foreign", "chat_id": 7, "drive_root": str(room.root), "_owner_hold": True}
    ctx = SimpleNamespace(DRIVE_ROOT=room.root, RUNNING={}, PENDING=[target])
    chat_id = registry.get_project(room.root, sender_room)["chat_id"] if sender_room else 1
    _handle_steer_task({"target_task_id": "foreign", "message": "Context, not an owner directive",
                       "quotes": [], "relay_id": "cross-room", "routing_token": "cross-room",
                       "chat_id": chat_id, "issuer": {"kind": "task", "task_id": "teller"}}, ctx)
    [entry] = drain_owner_entries(room.root, "foreign")
    assert entry["text"] == "Context, not an owner directive" and entry["owner_rows"] == []
    assert entry["source_task_id"] == "teller" and ctx.PENDING == [target]
    assert target["_owner_hold"] is True  # delivery is not Resume


def test_real_api_derived_scope_survives_other_room_creation(room, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros.gateway import tasks as api
    from starlette.requests import Request
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    folder = tmp_path / "api-workspace"
    folder.mkdir()
    registry.update_project(room.root, "target", working_dir=str(folder))
    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=room.root, repo_dir=tmp_path / "repo"))})
    original = api.prepare_task_drive
    def prepare(*args, **kwargs):
        registry.create_project(room.root, "new-unrelated", working_dir=str(tmp_path / "elsewhere"))
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "prepare_task_drive", prepare)
    response = api._create_task_from_body(request, {
        "task_id": "api-derived", "description": "Use chosen folder", "workspace_root": str(folder),
        "memory_mode": "empty"})
    assert response.status_code == 200
    [admitted] = room.pending
    assert admitted["id"] == "api-derived" and admitted["project_id"] == "target"
    assert load_task_result(room.root, "api-derived")["api_admission"]["status"] == "accepted"


def test_derived_guard_resolves_every_claimed_folder_again_under_q(room, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import project_facts

    folder = tmp_path / "folder"
    folder.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(folder, target_is_directory=True)
    registry.update_project(room.root, "target", working_dir=str(alias))
    basis = registry.project_scope_admission(room.root, workspace_root=str(folder))
    assert basis["project_id"] == "target"
    registry.update_project(room.root, "other", working_dir=str(tmp_path / "irrelevant-new-path"))
    calls = []
    original = project_facts._normalized_workspace
    def normalize(path):
        calls.append((path, room.queue._queue_lock._is_owned()))
        return original(path)
    monkeypatch.setattr(project_facts, "_normalized_workspace", normalize)
    admitted = room.queue.enqueue_task(task(), project_admission=basis)
    assert not admitted.get("_admission_blocked")  # an unchanged folder through an alias still admits
    assert calls == [(basis["workspace_claims"]["workspace"], True), (str(alias), True),
                     (str(tmp_path / "irrelevant-new-path"), True)]


@pytest.mark.parametrize("swap", [None, "room_folder", "task_folder"])
def test_derived_admission_refuses_a_folder_swapped_onto_a_room_after_preparation(room, tmp_path, swap):  # noqa: F811
    """R5: preparation's realpath is not reused. A registered room folder, or the task's own
    folder, replaced by a symlink before the final read cannot admit a second derived identity
    for the room's folder; the unchanged folder still admits its prepared derived scope."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    registry.update_project(room.root, "target", working_dir=str(old))
    basis = registry.project_scope_admission(room.root, workspace_root=str(new))
    derived = basis["project_id"]
    assert derived.startswith("proj_") and basis["workspace_claims"]["owners"] == []
    if swap:
        moved, destination = (old, new) if swap == "room_folder" else (new, old)
        moved.rename(tmp_path / f"{moved.name}-aside")
        moved.symlink_to(destination, target_is_directory=True)
        # A fresh preparation now selects the registered room for the task's folder.
        assert registry.project_scope_admission(room.root, workspace_root=str(new))["project_id"] == "target"
    admitted = room.queue.enqueue_task(task("derived", project_id=derived, workspace_root=str(new)),
                                       project_admission=basis)
    if swap:
        assert admitted["_admission_blocked"] == "project_routing_fence_changed" and not room.pending
    else:
        assert room.pending == [admitted] and admitted["_project_admission"]["project_id"] == derived


def test_restored_invalid_basis_never_reaches_worker_and_main_sibling_runs(tmp_path, monkeypatch):
    from tests._budget_pause_exact_helpers import _install_queue

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    registry.create_project(tmp_path, "target")
    workers.PENDING.extend([task("held", _project_admission={}), task("main", project_id="")])
    queue.persist_queue_snapshot()
    workers.PENDING.clear()
    queue.restore_pending_from_snapshot()
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t)))))
    workers.assign_tasks()
    assert [r["id"] for r in sent] == ["main"]
    assert [r["id"] for r in workers.PENDING] == ["held"]
    assert workers.PENDING[0]["_project_admission"] == {}
    assert not load_task_result(tmp_path, "held")


def test_incoming_routing_refuses_corrupt_authority_before_direct_or_mailbox(room, monkeypatch):  # noqa: F811
    import server
    from ouroboros.project_dialogue import latest_chat_annotations
    from tests.test_project_routing_v664 import _ctx

    direct, receipts = [], []
    ctx = _ctx(room.root, direct=lambda *_a, **_k: direct.append(True))
    chat_id = registry.get_project(room.root, "target")["chat_id"]
    change_row(room.root, "target", lifecycle=[])  # the addressed room's own routing is corrupt
    class Bridge:
        def get_updates(self, **kwargs):
            return [{"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": 1},
                    "text": "Continue", "source": "web", "client_message_id": "refused-origin"}}]
        def send_routing_ack(self, *args, **kwargs):
            receipts.append(kwargs)
        def broadcast(self, _payload):
            pass
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)
    assert not direct and not ctx.PENDING
    assert receipts[-1]["status"] == "project_unavailable"
    assert latest_chat_annotations(room.root)["refused-origin"]["reason"] == "project_routing_fence_lookup_failed"


def test_project_provisioning_does_not_run_before_strict_authority_check(room, tmp_path, monkeypatch):  # noqa: F811
    import asyncio
    from ouroboros.gateway import projects

    folder = tmp_path / "owner-folder"
    folder.mkdir()
    change_row(room.root, "target", routing_generation=[])
    calls = []
    monkeypatch.setattr("ouroboros.project_sources.attach_snapshot_init", lambda *a, **k: calls.append(True))
    class Request:
        app = SimpleNamespace(state=SimpleNamespace(drive_root=room.root, repo_dir=tmp_path / "repo"))
        async def json(self):
            return {"id": "target", "name": "Target", "path": str(folder), "init_git": True}
    response = asyncio.run(projects.api_projects_create(Request()))
    assert response.status_code >= 400 and not calls and not (folder / ".git").exists()
