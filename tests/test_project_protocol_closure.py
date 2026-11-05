"""#1381 remaining contracts through real queue, API, restore and schedule owners."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from tests.test_project_semantic_admission import room, task  # noqa: F401
from tests.test_schedule_occurrence import q, _row, _rows  # noqa: F401



def change_row(root, pid, **fields):
    path = registry._registry_path(root)
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["projects"]:
        if row["id"] == pid:
            row.update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("field,value", [("frozen", 1), ("registry_present", "yes"),
    ("project", []), ("workspace_claims", []), ("workspace_claims", {"workspace": []})])
def test_invalid_carrier_fields_are_typed_before_enqueue(room, field, value):  # noqa: F811
    basis = registry.project_admission_view(room.root, "target")
    basis[field] = value
    denied = room.queue.enqueue_task(task(_project_admission=basis))
    assert denied["_admission_blocked"] == "project_routing_fence_lookup_failed"
    assert not room.pending


def test_schedule_intent_only_room_and_unregistered_scope_keep_distinct_authority(room):  # noqa: F811
    _row(room, "known", intent={"kind": "explicit_none", "project_id": "target"})
    _row(room, "unregistered", project_id="scope-only", intent={"kind": "explicit_none"})
    room.queue.check_scheduled_tasks()
    assert {r["project_id"] for r in room.pending} == {"target", "scope-only"}
    known = next(r for r in room.pending if r["project_id"] == "target")
    assert known["_project_admission"]["project"]["id"] == "target"


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
