"""RWS v2 §3.1 — the placement is SEALED at admission and only READ afterwards.

Decision D "immutable task placement" is only real if three things hold together:

* both creation surfaces (`/api/tasks` and the promote path) seal the SAME ref under the
  SAME key, resolved through the ONE admission SSOT;
* the worker READS that seal instead of re-deriving a placement — so a local task keeps
  its Home `Path` and a remote task gets NO Home path at all, while still counting as a
  workspace task (a remote task that fell back to `self_modification` over the live
  Ouroboros repo is the failure this contract exists to prevent);
* editing the project afterwards cannot move a task that is already admitted.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY


@pytest.fixture(autouse=True)
def _managed_worker_pool_available(monkeypatch):
    """Rebase onto v6.82: `/api/tasks` now RESERVES admission with
    `require_worker_pool=True` and answers 503 `worker_pool_unavailable` when the pool
    is empty. These cases are about PLACEMENT, not about pool readiness, so they model a
    ready server exactly as upstream's own HTTP task tests do
    (`tests/test_headless_cli.py::_managed_worker_pool_available`)."""
    import types

    import supervisor.workers as workers

    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")



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


def _project_with_placement(data: pathlib.Path, project_id: str, placement: dict | None, working_dir: str = "") -> None:
    from ouroboros.projects_registry import _load, _save, create_project, update_project

    create_project(data, project_id, name=project_id, origin="test")
    if working_dir:
        update_project(data, project_id, working_dir=working_dir)
    if placement is not None:
        registry = _load(data)
        for row in registry["projects"]:
            if row.get("id") == project_id:
                row["placement"] = placement
        _save(data, registry)


def _task_api_client(tmp_path, monkeypatch, captured):
    from ouroboros.gateway.tasks import api_tasks_create

    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True, exist_ok=True)
    (data / "memory" / "identity.md").write_text("seed", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)

    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    # Must return True: v6.82's `_enqueue_api_task_durably` treats anything else as a
    # FAILED durable snapshot and rolls the admission back with a 503, so a `None`
    # stub would model the failure path instead of a successful admission.
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    app.state.supervisor_ready = True
    return TestClient(app), data, repo


# --- 1. both creation surfaces seal ------------------------------------------

def test_api_tasks_seals_the_local_placement(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_repo(workspace)
    captured: list = []
    client, _data, _repo = _task_api_client(tmp_path, monkeypatch, captured)

    response = client.post("/api/tasks", json={"description": "fix it", "workspace_root": str(workspace)})
    assert response.status_code == 200, response.text
    task = captured[0]
    assert task["metadata"][SEALED_WORKSPACE_REF_KEY] == {
        "kind": "local",
        "local_root": str(workspace.resolve()),
    }
    # The display spelling still says "this is a workspace task" for the string plumbing.
    assert task["workspace_root"] == str(workspace.resolve())
    assert task["workspace_mode"] == "external"


def test_api_tasks_seals_a_remote_placement_inherited_from_the_project(tmp_path, monkeypatch):
    """A remote placement comes from the PROJECT, and the sealed task carries the target
    spelling with no Home path anywhere in it."""
    import ouroboros.workspace_admission as admission

    captured: list = []
    client, data, _repo = _task_api_client(tmp_path, monkeypatch, captured)
    _project_with_placement(data, "remote", _remote_placement())
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})

    response = client.post("/api/tasks", json={"description": "fix it", "project_id": "remote"})
    assert response.status_code == 200, response.text
    task = captured[0]
    assert task["metadata"][SEALED_WORKSPACE_REF_KEY] == _remote_placement()
    assert task["workspace_root"] == "/srv/work/app"
    assert task["workspace_mode"] == "external"
    assert task["memory_mode"] == "forked"
    # No Home path was fabricated for the remote tree, and the preflight reports the
    # TARGET's own facts rather than a Home walk over a tree that is not here.
    assert str(tmp_path) not in task["workspace_root"]
    preflight = task["metadata"]["workspace_preflight"]
    assert preflight["placement"] == "ssh"
    assert preflight["connection_id"] == "conn-1"
    assert preflight["canonical_root"] == "/srv/work/app"
    assert "error" not in preflight
    # A Home directory listing would be the wrong filesystem's answer, so there is none.
    assert "entries" not in preflight and "files" not in preflight


def test_api_tasks_refuses_a_client_chosen_remote_placement(tmp_path, monkeypatch):
    """The client may not pick a remote target: the project registry is where the routing
    generation and the connection trust live, so a per-task placement would be a
    placement with no generation to fence."""
    captured: list = []
    client, _data, _repo = _task_api_client(tmp_path, monkeypatch, captured)

    for field, value in (("workspace_ref", _remote_placement()), ("connection_id", "conn-1")):
        response = client.post("/api/tasks", json={"description": "fix it", field: value})
        assert response.status_code == 400, response.text
        assert "inherited from project_id" in response.json()["error"]
    assert captured == []

    # BOTH DOORS. `metadata` used to be closed by the reserved-key STRIP rather than by a
    # refusal, and only for one of the two halves: `metadata.workspace_ref` was silently
    # dropped and `metadata.connection_id` was not in that set at all, so it was stored
    # and then ignored by everything downstream. A placement the owner named and nothing
    # honoured is worse than a refusal — the same argument `project_id`-in-metadata is
    # already refused by, and what `docs/ARCHITECTURE.md` promised while the code fell
    # short of it.
    for field, value in (("workspace_ref", _remote_placement()), ("connection_id", "conn-1")):
        response = client.post(
            "/api/tasks", json={"description": "fix it", "metadata": {field: value}}
        )
        assert response.status_code == 400, (field, response.text)
        assert "inherited from project_id" in response.json()["error"]
    assert captured == []

    # The seal's own PRIVATE key stays a strip, deliberately: `_sealed_workspace_ref` is
    # not a spelling anyone chooses in order to express intent — it is what a RETURNED
    # metadata blob contains — so a client replaying one must not turn into a 400. Refusing
    # it authority is a different job from refusing the request.
    reserved = client.post(
        "/api/tasks",
        json={"description": "fix it", "metadata": {SEALED_WORKSPACE_REF_KEY: _remote_placement()}},
    )
    assert reserved.status_code == 200, reserved.text
    assert SEALED_WORKSPACE_REF_KEY not in captured[0]["metadata"]


def test_api_tasks_refuses_a_workspace_root_against_a_remote_project(tmp_path, monkeypatch):
    import ouroboros.workspace_admission as admission

    workspace = tmp_path / "workspace"
    _init_git_repo(workspace)
    captured: list = []
    client, data, _repo = _task_api_client(tmp_path, monkeypatch, captured)
    _project_with_placement(data, "remote", _remote_placement())
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})

    response = client.post(
        "/api/tasks",
        json={"description": "fix it", "project_id": "remote", "workspace_root": str(workspace)},
    )
    assert response.status_code == 400
    assert "inherited from project_id" in response.json()["error"]
    assert captured == []


def test_api_tasks_refuses_a_remote_project_whose_target_is_unreachable(tmp_path, monkeypatch):
    """No silent fallback at the CREATION surface either: an unverifiable remote project
    yields a typed refusal, not a workspace-less task over the system repo."""
    import ouroboros.workspace_admission as admission

    captured: list = []
    client, data, _repo = _task_api_client(tmp_path, monkeypatch, captured)
    _project_with_placement(data, "remote", _remote_placement())
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})

    response = client.post("/api/tasks", json={"description": "fix it", "project_id": "remote"})
    assert response.status_code == 400
    # The store has no such connection, so the refusal is the MALFORMED-request one.
    # Either way it is a refusal and no task exists — the point of the invariant.
    assert "not an active remote connection" in response.json()["error"]
    assert captured == []


def test_api_tasks_refuses_a_per_task_executor_ref_for_a_remote_placement(tmp_path, monkeypatch):
    import ouroboros.workspace_admission as admission

    captured: list = []
    client, data, _repo = _task_api_client(tmp_path, monkeypatch, captured)
    _project_with_placement(data, "remote", _remote_placement())
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})

    response = client.post(
        "/api/tasks",
        json={
            "description": "fix it",
            "project_id": "remote",
            "executor_ref": {"type": "docker_exec", "container_name": "c1"},
        },
    )
    assert response.status_code == 400
    assert "derived from the project's remote placement" in response.json()["error"]


# --- 2. the worker READS the seal --------------------------------------------

def test_worker_reads_the_seal_local_gets_a_path_remote_gets_none(tmp_path):
    """The `_prepare_task_context` seam: the sealed ref decides whether a Home path
    exists. A remote task is still a WORKSPACE task — the tool profile must not fall back
    to self_modification over the Ouroboros repo."""
    from ouroboros.agent import _read_sealed_placement
    from ouroboros.tools.registry import ToolContext
    from ouroboros.workspace_ref import RemoteWorkspacePathError

    local_meta = {SEALED_WORKSPACE_REF_KEY: {"kind": "local", "local_root": str(tmp_path / "ws")}}
    local_root = _read_sealed_placement({"workspace_root": str(tmp_path / "ws")}, local_meta)
    assert local_root == (tmp_path / "ws").resolve()

    remote_meta = {SEALED_WORKSPACE_REF_KEY: _remote_placement()}
    assert _read_sealed_placement({"workspace_root": "/srv/work/app"}, remote_meta) is None
    # The seal survives the read unchanged (it is read, never re-derived).
    assert remote_meta[SEALED_WORKSPACE_REF_KEY] == _remote_placement()

    ctx = ToolContext(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
        branch_dev="dev",
        system_repo_dir=tmp_path / "repo",
        workspace_root=None,
        workspace_mode="external",
        task_metadata=remote_meta,
    )
    assert ctx.is_workspace_mode() is True
    with pytest.raises(RemoteWorkspacePathError):
        ctx.active_repo_dir()


def test_legacy_unsealed_task_reads_as_the_local_variant(tmp_path):
    """A task record written before this change (bare `workspace_root`, no seal) is
    placement-identical after a restart."""
    from ouroboros.agent import _read_sealed_placement

    ws = tmp_path / "legacy"
    metadata: dict = {}
    assert _read_sealed_placement({"workspace_root": str(ws)}, metadata) == ws.resolve()
    assert metadata[SEALED_WORKSPACE_REF_KEY] == {"kind": "local", "local_root": str(ws.resolve())}
    assert _read_sealed_placement({}, {}) is None


def _promote_into_project(tmp_path, monkeypatch, project_id: str, working_dir: str = "", placement=None):
    """Drive the promote path for a project room and return the enqueued task."""
    import types

    import supervisor.message_bus as mbus
    import supervisor.workers as workers

    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    _project_with_placement(data, project_id, placement, working_dir=working_dir)

    monkeypatch.setattr(workers, "DRIVE_ROOT", data)
    monkeypatch.setattr(workers, "REPO_DIR", repo)
    monkeypatch.setattr(mbus, "get_bridge", lambda: types.SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    monkeypatch.setattr(
        "ouroboros.workspace_admission.bounded_workspace_preflight",
        lambda root, **kw: {"schema_version": 1, "workspace_root": str(root)},
    )
    enqueued: list = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task) or task,
        load_state=lambda: {"owner_chat_id": 1},
    )
    outcome = workers.promote_chat_to_task(
        {
            "type": "promote_chat_to_task",
            "task_id": f"promote-{project_id}",
            "objective": "Build it",
            "project_id": project_id,
            "chat_id": 1,
        },
        ctx,
    )
    return outcome, enqueued


def test_promote_path_seals_through_the_same_authority(tmp_path, monkeypatch):
    """The promote path was the DEGRADED twin once; it now seals the same key from the
    same SSOT, so the two surfaces cannot drift apart again."""
    room = tmp_path / "room"
    _init_git_repo(room)
    _outcome, enqueued = _promote_into_project(tmp_path, monkeypatch, "room", working_dir=str(room))

    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["metadata"][SEALED_WORKSPACE_REF_KEY] == {
        "kind": "local",
        "local_root": str(room.resolve()),
    }
    assert task["workspace_root"] == str(room.resolve())
    assert task["workspace_mode"] == "external"


def test_promote_path_seals_a_remote_room_without_a_home_path(tmp_path, monkeypatch):
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})
    _outcome, enqueued = _promote_into_project(
        tmp_path, monkeypatch, "remoteroom", placement=_remote_placement()
    )

    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["metadata"][SEALED_WORKSPACE_REF_KEY] == _remote_placement()
    assert task["workspace_root"] == "/srv/work/app"
    # The Home preflight walk was NOT run against a tree that does not live here: the
    # summary carries the TARGET's identity instead.
    preflight = task["metadata"]["workspace_preflight"]
    assert preflight["placement"] == "ssh"
    assert preflight["canonical_root"] == "/srv/work/app"
    assert str(tmp_path) not in repr(preflight)


def test_promote_path_loud_fails_an_unverifiable_remote_room(tmp_path, monkeypatch):
    """A remote room whose target cannot be reached fails LOUDLY on the promote path —
    it does not become a workspace-less task over the system repo."""
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    outcome, enqueued = _promote_into_project(
        tmp_path, monkeypatch, "deadroom", placement=_remote_placement()
    )

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_unusable"
    assert enqueued == []


# --- 2b. the third creation path: subagent spawn -----------------------------

def test_subagent_child_inherits_the_parents_sealed_placement(tmp_path):
    """The THIRD task-creation path. A child runs where its parent runs, so it must
    inherit the parent's sealed REF — not just its root spelling. Given only the spelling,
    a remote parent's target path would normalize as the LOCAL variant in the child: a
    fabricated Home path pointing at a tree that does not exist here (C-2:963)."""
    from supervisor.events import _build_scheduled_task_payload
    from ouroboros.workspace_ref import workspace_ref_for

    remote_child = _build_scheduled_task_payload({
        "tid": "child-1",
        "root_task_id": "root-1",
        "delegation_role": "subagent",
        "workspace_root": "/srv/work/app",
        "workspace_mode": "external",
        "workspace_ref": _remote_placement(),
    })
    assert remote_child["metadata"][SEALED_WORKSPACE_REF_KEY] == _remote_placement()
    ref = workspace_ref_for(remote_child)
    assert ref.kind == "ssh" and ref.remote_root == "/srv/work/app"
    with pytest.raises(Exception):
        ref.home_path()

    # A LOCAL parent that predates the ref (spelling only) still reads as local — the
    # legacy normalization keeps the old spawn path placement-identical.
    local_child = _build_scheduled_task_payload({
        "tid": "child-2",
        "root_task_id": "root-1",
        "delegation_role": "subagent",
        "workspace_root": str(tmp_path / "ws"),
        "workspace_mode": "external",
    })
    assert workspace_ref_for(local_child).kind == "local"

    # A workspace-less child seals nothing at all.
    plain_child = _build_scheduled_task_payload({"tid": "child-3", "root_task_id": "root-1"})
    assert SEALED_WORKSPACE_REF_KEY not in plain_child["metadata"]
    assert workspace_ref_for(plain_child) is None


def test_spawn_event_carries_the_parents_sealed_placement(tmp_path):
    """The parent-side half: the spawn event itself carries the ref, read ONCE for the
    whole wave so two children of one parent cannot disagree about where they run."""
    from ouroboros.tools.control import _populate_subagent_event_extras
    from ouroboros.workspace_ref import SshWorkspaceRef

    evt: dict = {}
    _populate_subagent_event_extras(
        evt,
        current_chat_id=1,
        child_drive=None,
        workspace_root="/srv/work/app",
        workspace_mode="external",
        executor_ref=None,
        context="",
        parent_task_id="parent-1",
        workspace_ref=SshWorkspaceRef(connection_id="conn-1", remote_root="/srv/work/app", workspace_id="ws-1"),
    )
    assert evt["workspace_ref"] == _remote_placement()

    # A local parent with no sealed ref (legacy) adds no field at all.
    legacy: dict = {}
    _populate_subagent_event_extras(
        legacy,
        current_chat_id=1,
        child_drive=None,
        workspace_root=str(tmp_path / "ws"),
        workspace_mode="external",
        executor_ref=None,
        context="",
        parent_task_id="parent-1",
    )
    assert "workspace_ref" not in legacy


# --- 3. the seal is immutable across a project rebind ------------------------

def test_project_rebind_cannot_move_an_already_admitted_task(tmp_path, monkeypatch):
    """The seal is the placement. Re-pointing the project afterwards changes what NEW
    tasks get and nothing about the task already admitted."""
    import ouroboros.workspace_admission as admission
    from ouroboros.agent import _read_sealed_placement
    from ouroboros.projects_registry import update_project

    first = tmp_path / "first"
    _init_git_repo(first)
    second = tmp_path / "second"
    _init_git_repo(second)
    captured: list = []
    client, data, _repo = _task_api_client(tmp_path, monkeypatch, captured)
    _project_with_placement(data, "room", None, working_dir=str(first))
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})

    assert client.post(
        "/api/tasks", json={"description": "one", "project_id": "room", "workspace_root": str(first)}
    ).status_code == 200
    admitted = captured[0]
    sealed_before = dict(admitted["metadata"][SEALED_WORKSPACE_REF_KEY])

    # The project is re-pointed AFTER admission — first to another folder, then remotely.
    update_project(data, "room", working_dir=str(second))
    assert admitted["metadata"][SEALED_WORKSPACE_REF_KEY] == sealed_before
    assert _read_sealed_placement(admitted, dict(admitted["metadata"])) == first.resolve()

    _project_with_placement(data, "remote-room", _remote_placement())
    assert admitted["metadata"][SEALED_WORKSPACE_REF_KEY] == sealed_before
