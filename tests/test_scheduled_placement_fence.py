"""RWSB2-05 — the schedule fires, the placement resolves, the fence guards the window.

A schedule template stores `project_id` only. Where that project lives is resolved when
the schedule FIRES, through the same `workspace_admission` path `/api/tasks` uses. But
"resolve" and "insert into PENDING" are not one instant, and in the gap the owner can
delete the project, rebind it, or retire the connection. So the resolved placement carries
a fence — routing generation plus connection trust identity — that `enqueue_task`
revalidates under `_queue_lock` immediately before insertion.

The refusal is a REFUSAL, never a re-resolution: re-resolving would mean an owner's rebind
retroactively moved work that was scheduled against the previous target.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest


def _init_git_repo(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "init"],
        cwd=str(path), check=True,
    )


def _remote_placement():
    return {
        "kind": "ssh",
        "connection_id": "conn-1",
        "remote_root": "/srv/work/app",
        "workspace_id": "ws-1",
    }


@pytest.fixture()
def scheduler(tmp_path, monkeypatch):
    """A live queue with a data drive OUTSIDE the workspace tree (an overlapping room
    would be refused by admission for the right reason and hide the one under test)."""
    from supervisor import queue, state as supervisor_state

    drive = tmp_path / "data"
    drive.mkdir()
    supervisor_state.init(drive)
    queue.init(drive, 600, 1800)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    monkeypatch.setattr(
        "ouroboros.workspace_admission.bounded_workspace_preflight",
        lambda root, **kw: {"schema_version": 1, "workspace_root": str(root)},
    )
    monkeypatch.setattr("supervisor.scheduled_dispatch._system_repo_dir", lambda: tmp_path / "norepo")
    return queue, drive, pending


def _schedule_for_project(queue, project_id: str, schedule_id: str = "nightly") -> None:
    queue.upsert_scheduled_task({
        "id": schedule_id,
        "name": schedule_id,
        "enabled": True,
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "next_run_at": "2000-01-01T00:00:00+00:00",
        "task": {"type": "task", "text": "scheduled work", "project_id": project_id},
    })


# --- the successful path ------------------------------------------------------

def test_local_project_schedule_resolves_and_seals_at_fire_time(tmp_path, scheduler):
    """The template carried only `project_id`; the fire resolves the CURRENT placement,
    seals it, binds the fence, and prepares the isolated child drive."""
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.workspace_admission import PLACEMENT_FENCE_KEY
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    queue, drive, pending = scheduler
    room = tmp_path / "room"
    _init_git_repo(room)
    create_project(drive, "nightly-project", name="Nightly", origin="test")
    update_project(drive, "nightly-project", working_dir=str(room))
    _schedule_for_project(queue, "nightly-project")

    queue.check_scheduled_tasks()

    assert len(pending) == 1
    task = pending[0]
    assert task["project_id"] == "nightly-project"
    assert task["metadata"][SEALED_WORKSPACE_REF_KEY] == {
        "kind": "local",
        "local_root": str(room.resolve()),
    }
    assert task["workspace_root"] == str(room.resolve())
    assert task["workspace_mode"] == "external"
    assert task["memory_mode"] == "forked"
    assert "[HEADLESS_WORKSPACE]" in task["text"]
    assert task["child_drive_root"] and task["budget_drive_root"] == str(drive)
    fence = task[PLACEMENT_FENCE_KEY]
    assert fence["project_id"] == "nightly-project"
    assert fence["routing_generation"] == 0
    assert fence["connection_id"] == ""


def test_remote_project_schedule_seals_the_target_spelling_and_binds_trust(tmp_path, scheduler, monkeypatch):
    import ouroboros.workspace_admission as admission
    from ouroboros.workspace_admission import PLACEMENT_FENCE_KEY
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    queue, drive, pending = scheduler
    from ouroboros.projects_registry import _load, _save, create_project

    create_project(drive, "remote-project", name="Remote", origin="test")
    registry = _load(drive)
    for row in registry["projects"]:
        if row.get("id") == "remote-project":
            row["placement"] = _remote_placement()
    _save(drive, registry)
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-fingerprint-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})
    _schedule_for_project(queue, "remote-project")

    queue.check_scheduled_tasks()

    assert len(pending) == 1
    task = pending[0]
    assert task["metadata"][SEALED_WORKSPACE_REF_KEY] == _remote_placement()
    assert task["workspace_root"] == "/srv/work/app"
    # The fence pins the exact connection trust identity that was true at resolve time.
    assert task[PLACEMENT_FENCE_KEY]["connection_id"] == "conn-1"
    assert task[PLACEMENT_FENCE_KEY]["connection_trust"] == "host-fingerprint-1"


# --- the races the fence exists for ------------------------------------------

def test_project_deleted_between_resolve_and_insert_is_refused(tmp_path, scheduler, monkeypatch):
    """Resolve → project deleted → insert. The task must be refused, not inserted with a
    placement resolved against a project that no longer exists."""
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    queue, drive, pending = scheduler
    room = tmp_path / "room"
    _init_git_repo(room)
    create_project(drive, "doomed", name="Doomed", origin="test")
    update_project(drive, "doomed", working_dir=str(room))
    _schedule_for_project(queue, "doomed")

    real_enqueue = queue.enqueue_task
    fired: list = []

    def _delete_then_enqueue(task, front=False, **kwargs):
        # The window: the placement is already resolved and sealed on `task`.
        fired.append(dict(task))
        from ouroboros.projects_registry import _load, _save

        registry = _load(drive)
        registry["projects"] = [row for row in registry["projects"] if row.get("id") != "doomed"]
        _save(drive, registry)
        return real_enqueue(task, front, **kwargs)

    monkeypatch.setattr(queue, "enqueue_task", _delete_then_enqueue)
    queue.check_scheduled_tasks()

    assert len(fired) == 1  # the placement WAS resolved
    assert pending == []  # and the task was still not admitted
    result = load_task_result(drive, fired[0]["id"])
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "placement_project_missing"
    schedule = queue.list_scheduled_tasks()["tasks"][0]
    assert schedule["failure_count"] == 1
    assert "placement_project_missing" in schedule["last_error"]


def test_project_rebound_between_resolve_and_insert_is_refused(tmp_path, scheduler, monkeypatch):
    """A rebind bumps `routing_generation`; the stale generation is refused rather than
    re-resolved, so the rebind never retroactively moves already-scheduled work."""
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.task_results import load_task_result

    queue, drive, pending = scheduler
    room = tmp_path / "room"
    _init_git_repo(room)
    elsewhere = tmp_path / "elsewhere"
    _init_git_repo(elsewhere)
    create_project(drive, "moving", name="Moving", origin="test")
    update_project(drive, "moving", working_dir=str(room))
    _schedule_for_project(queue, "moving")

    real_enqueue = queue.enqueue_task
    fired: list = []

    def _rebind_then_enqueue(task, front=False, **kwargs):
        fired.append(dict(task))
        from ouroboros.projects_registry import _load, _save

        registry = _load(drive)
        for row in registry["projects"]:
            if row.get("id") == "moving":
                row["working_dir"] = str(elsewhere)
                row["routing_generation"] = int(row.get("routing_generation") or 0) + 1
        _save(drive, registry)
        return real_enqueue(task, front, **kwargs)

    monkeypatch.setattr(queue, "enqueue_task", _rebind_then_enqueue)
    queue.check_scheduled_tasks()

    assert pending == []
    assert load_task_result(drive, fired[0]["id"])["reason_code"] == "placement_routing_generation_stale"
    # The refused task's own seal still names the ORIGINAL target: nothing re-resolved it.
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    assert fired[0]["metadata"][SEALED_WORKSPACE_REF_KEY]["local_root"] == str(room.resolve())


@pytest.mark.parametrize(
    ("later_trust", "reason"),
    [
        ({}, "placement_connection_retired"),
        ({"conn-1": "host-fingerprint-2"}, "placement_connection_trust_changed"),
    ],
)
def test_connection_retired_or_retrusted_between_resolve_and_insert_is_refused(
    tmp_path, scheduler, monkeypatch, later_trust, reason
):
    """The remote twin of the generation race: a connection retired — or re-trusted to a
    DIFFERENT host — after the placement resolved must not carry the task onto a host the
    owner no longer vouched for."""
    import ouroboros.workspace_admission as admission
    from ouroboros.projects_registry import _load, _save, create_project
    from ouroboros.task_results import load_task_result

    queue, drive, pending = scheduler
    create_project(drive, "remote-race", name="Remote", origin="test")
    registry = _load(drive)
    for row in registry["projects"]:
        if row.get("id") == "remote-race":
            row["placement"] = _remote_placement()
    _save(drive, registry)
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})
    trust = {"index": {"conn-1": "host-fingerprint-1"}}
    monkeypatch.setattr(admission, "known_connections", lambda: dict(trust["index"]))
    _schedule_for_project(queue, "remote-race")

    real_enqueue = queue.enqueue_task
    fired: list = []

    def _retire_then_enqueue(task, front=False, **kwargs):
        fired.append(dict(task))
        trust["index"] = later_trust
        return real_enqueue(task, front, **kwargs)

    monkeypatch.setattr(queue, "enqueue_task", _retire_then_enqueue)
    queue.check_scheduled_tasks()

    assert pending == []
    assert load_task_result(drive, fired[0]["id"])["reason_code"] == reason


def test_missing_project_fails_closed_at_fire_time_without_enqueueing(tmp_path, scheduler, monkeypatch):
    """A schedule pointing at a project that no longer exists never reaches the queue."""
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    queue, drive, pending = scheduler
    _schedule_for_project(queue, "never-existed")
    attempts: list = []
    monkeypatch.setattr(queue, "enqueue_task", lambda task, front=False, **kw: attempts.append(task) or task)

    queue.check_scheduled_tasks()

    assert attempts == []
    schedule = queue.list_scheduled_tasks()["tasks"][0]
    result = load_task_result(drive, schedule["last_task_id"])
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "project_not_found"
    # Identity is preserved for diagnosis even on the refusal path.
    assert result["project_id"] == "never-existed"
    assert result["schedule_id"] == "nightly"


def test_legacy_project_id_is_validated_exactly_at_fire_time(tmp_path, scheduler, monkeypatch):
    """`"PROD"` must not be case-normalized into the live `prod` project's work."""
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import load_task_result

    queue, drive, pending = scheduler
    create_project(drive, "prod", name="Prod", origin="test")
    queue.upsert_scheduled_task({
        "id": "legacy",
        "name": "legacy",
        "enabled": True,
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "next_run_at": "2000-01-01T00:00:00+00:00",
        "task": {"type": "task", "text": "x", "project_id": " prod "},
    })
    monkeypatch.setattr(
        queue, "enqueue_task",
        lambda task, front=False, **kw: pytest.fail("a malformed project id was normalized into a live project"),
    )

    queue.check_scheduled_tasks()

    schedule = queue.list_scheduled_tasks()["tasks"][0]
    result = load_task_result(drive, schedule["last_task_id"])
    assert result["reason_code"] == "invalid_project_id"


# --- the template may not carry a placement ----------------------------------

def test_schedule_template_carrying_a_placement_field_is_rejected(tmp_path):
    """`RESERVED_TEMPLATE_FIELDS` is the fail-closed door: a template that persists a
    placement is refused at both the top level and inside metadata."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.schedules import api_schedules_upsert
    from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    for key in ("workspace_ref", "connection_id", "executor_ref", "workspace_root", SEALED_WORKSPACE_REF_KEY):
        assert key in RESERVED_TEMPLATE_FIELDS

    app = Starlette(routes=[Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"])])
    app.state.drive_root = tmp_path
    client = TestClient(app)
    base = {"id": "s1", "name": "s1", "trigger": {"type": "cron", "expr": "* * * * *"}}

    for payload, fragment in (
        ({**base, "task": {"type": "task", "text": "x", "workspace_ref": {"kind": "ssh"}}}, "workspace/drive fields"),
        ({**base, "task": {"type": "task", "text": "x", "connection_id": "conn-1"}}, "workspace/drive fields"),
        (
            {**base, "task": {"type": "task", "text": "x", "metadata": {SEALED_WORKSPACE_REF_KEY: {"kind": "ssh"}}}},
            "reserved lineage/workspace fields",
        ),
    ):
        response = client.post("/api/schedules", json=payload)
        assert response.status_code == 400, response.text
        assert fragment in response.json()["error"]

    # A malformed project_id is refused for SHAPE at write time...
    dirty = client.post("/api/schedules", json={**base, "task": {"type": "task", "text": "x", "project_id": "Bad Name!"}})
    assert dirty.status_code == 400
    assert "filesystem-safe" in dirty.json()["error"]
    # ...while a clean one is accepted, because EXISTENCE is a fire-time fact.
    ok = client.post("/api/schedules", json={**base, "task": {"type": "task", "text": "x", "project_id": "not-yet"}})
    assert ok.status_code == 200, ok.text
    assert ok.json()["schedule"]["task"]["project_id"] == "not-yet"
