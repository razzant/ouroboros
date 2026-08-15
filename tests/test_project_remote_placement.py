"""The owner's path to a remote project: create, rebind, and the fence between them.

`projects_registry` is where a placement becomes DURABLE, and this file pins the
three properties that make that safe:

1. **One authority per case.** A remote project stores a sealed
   ``SshWorkspaceRef`` and NO ``working_dir``; a local project stores a
   ``working_dir`` and NO placement. Neither can acquire the other's field, so no
   reader ever has two answers to "where does this project live".
2. **Every placement change advances ``routing_generation``.** That counter is the
   fence `supervisor/queue.py` revalidates before a task becomes PENDING, so a
   rebind cannot leave work aimed at the previous target looking current.
3. **Legacy reads as local, garbage reads as an error.** A pre-RWS row (no
   ``placement``) is a local project; a row whose placement this build cannot honor
   fails LOUDLY rather than degrading to Home.
"""
from __future__ import annotations

import pytest

from ouroboros.projects_registry import (
    create_project,
    get_project,
    project_placement,
    projects_summary,
    set_project_placement,
    update_project,
)

_REMOTE = {
    "kind": "ssh",
    "connection_id": "conn-1",
    "remote_root": "/srv/work/app",
    "workspace_id": "ws-1",
}


def _drive(tmp_path):
    drive = tmp_path / "data"
    drive.mkdir(exist_ok=True)
    return drive


def _init_git_repo(path):
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "init"],
        cwd=str(path), check=True,
    )


# --- 1. one authority per case ------------------------------------------------

def test_a_remote_project_persists_the_sealed_placement_and_no_working_dir(tmp_path):
    drive = _drive(tmp_path)
    entry = create_project(drive, "remote-app", name="Remote app", placement=_REMOTE)

    assert entry["placement"] == _REMOTE
    assert entry["working_dir"] == ""
    # Durable, not just returned: a fresh read of the registry file agrees.
    stored = get_project(drive, "remote-app")
    assert project_placement(stored).to_payload() == _REMOTE
    assert project_placement(stored).kind == "ssh"


def test_a_local_placement_is_refused_because_working_dir_already_is_one(tmp_path):
    """Storing a local ref would duplicate `working_dir`, and duplicated facts drift."""
    drive = _drive(tmp_path)
    with pytest.raises(ValueError) as err:
        create_project(
            drive, "dup", placement={"kind": "local", "local_root": str(tmp_path)}
        )
    assert "working_dir" in str(err.value)


def test_a_remote_project_cannot_be_given_a_home_working_dir(tmp_path):
    """Both directions: at creation, and afterwards through `update_project`.

    A remote project holding a Home folder would give every `working_dir` reader a
    local path to prefer over the target the owner actually chose.
    """
    drive = _drive(tmp_path)
    with pytest.raises(ValueError) as both:
        create_project(drive, "both", working_dir=str(tmp_path), placement=_REMOTE)
    assert "no Home working_dir" in str(both.value)

    create_project(drive, "remote-app", placement=_REMOTE)
    with pytest.raises(ValueError) as later:
        update_project(drive, "remote-app", working_dir=str(tmp_path))
    assert "rebind its placement" in str(later.value)
    # The refusal is not partial: the row is untouched.
    assert get_project(drive, "remote-app")["working_dir"] == ""

    # A rename still works on a remote project — only `working_dir` is refused.
    assert update_project(drive, "remote-app", name="Renamed")["name"] == "Renamed"


# --- 2. every placement change advances routing_generation --------------------

def test_rebind_advances_the_routing_generation_and_is_idempotent(tmp_path):
    drive = _drive(tmp_path)
    created = create_project(drive, "remote-app", placement=_REMOTE)
    assert created["routing_generation"] == 0

    moved = set_project_placement(
        drive, "remote-app", {**_REMOTE, "remote_root": "/srv/work/other", "workspace_id": "ws-2"}
    )
    assert moved["placement"]["remote_root"] == "/srv/work/other"
    assert moved["routing_generation"] == 1

    # Re-submitting the SAME placement is a no-op: it must not invalidate fences
    # that are still correct.
    again = set_project_placement(
        drive, "remote-app", {**_REMOTE, "remote_root": "/srv/work/other", "workspace_id": "ws-2"}
    )
    assert again["routing_generation"] == 1

    # A different CONNECTION is also a placement change, not only a different path.
    across = set_project_placement(
        drive, "remote-app",
        {**_REMOTE, "connection_id": "conn-2", "remote_root": "/srv/work/other", "workspace_id": "ws-2"},
    )
    assert across["routing_generation"] == 2


def test_rebind_is_compare_and_set_on_the_routing_generation(tmp_path):
    """Two owner rebinds racing must not interleave silently; the loser is told."""
    drive = _drive(tmp_path)
    create_project(drive, "remote-app", placement=_REMOTE)
    set_project_placement(drive, "remote-app", {**_REMOTE, "workspace_id": "ws-2"})

    with pytest.raises(ValueError) as stale:
        set_project_placement(
            drive, "remote-app", {**_REMOTE, "workspace_id": "ws-3"},
            expected_routing_generation=0,
        )
    assert str(stale.value) == "project_routing_generation_changed"
    # The winner's placement stands.
    assert get_project(drive, "remote-app")["placement"]["workspace_id"] == "ws-2"

    accepted = set_project_placement(
        drive, "remote-app", {**_REMOTE, "workspace_id": "ws-3"},
        expected_routing_generation=1,
    )
    assert accepted["placement"]["workspace_id"] == "ws-3"


def test_a_home_folder_project_is_not_silently_converted_to_a_remote_one(tmp_path):
    """Rebinding a folder-bearing project would discard the owner's attachment."""
    drive = _drive(tmp_path)
    folder = tmp_path / "local-repo"
    folder.mkdir()
    create_project(drive, "local-app", working_dir=str(folder))

    with pytest.raises(ValueError) as err:
        set_project_placement(drive, "local-app", _REMOTE)
    assert "bound to a Home folder" in str(err.value)
    assert get_project(drive, "local-app")["working_dir"] == str(folder)
    assert get_project(drive, "local-app").get("placement") is None


def test_placement_changes_cannot_take_the_update_project_door(tmp_path):
    """`update_project` has no `placement` key, so nothing can move a placement
    without advancing the generation the fence reads."""
    drive = _drive(tmp_path)
    create_project(drive, "remote-app", placement=_REMOTE)
    entry = update_project(drive, "remote-app", placement={**_REMOTE, "workspace_id": "ws-9"})
    assert entry["placement"] == _REMOTE
    assert entry["routing_generation"] == 0


def test_rebind_of_an_unknown_or_inactive_project_is_none_not_a_new_row(tmp_path):
    from ouroboros.projects_registry import begin_project_deletion

    drive = _drive(tmp_path)
    assert set_project_placement(drive, "nope", _REMOTE) is None

    create_project(drive, "doomed", placement=_REMOTE)
    begin_project_deletion(drive, "doomed")
    assert set_project_placement(drive, "doomed", {**_REMOTE, "workspace_id": "ws-2"}) is None


# --- 3. legacy reads as local, garbage reads as an error ----------------------

def test_a_legacy_row_without_a_placement_reads_as_local(tmp_path):
    drive = _drive(tmp_path)
    folder = tmp_path / "local-repo"
    folder.mkdir()
    entry = create_project(drive, "local-app", working_dir=str(folder))

    assert "placement" not in entry
    assert project_placement(entry) is None
    assert project_placement({}) is None
    assert project_placement(None) is None


def test_a_placement_this_build_cannot_honour_fails_loudly(tmp_path):
    """Never "absent": reading an unhonourable placement as absent would run a
    remote project's work on Home."""
    from ouroboros.projects_registry import _load, _save

    drive = _drive(tmp_path)
    create_project(drive, "broken", placement=_REMOTE)
    registry = _load(drive)
    for row in registry["projects"]:
        if row.get("id") == "broken":
            row["placement"] = {"kind": "webdav", "remote_root": "/srv/work/app"}
    _save(drive, registry)

    with pytest.raises(ValueError) as err:
        project_placement(get_project(drive, "broken"))
    assert "must be 'local' or 'ssh'" in str(err.value)

    # And a broken row is still REPAIRABLE: its own unparseable placement must not
    # block the rebind that fixes it.
    repaired = set_project_placement(drive, "broken", _REMOTE)
    assert repaired["placement"] == _REMOTE
    assert repaired["routing_generation"] == 1


def test_a_task_born_in_a_remote_project_inherits_the_sealed_ssh_ref(tmp_path, monkeypatch):
    """The whole point of storing a placement: `/api/tasks` reads it and seals it.

    Asserted on the connected path (`gateway/tasks._admit_task_placement`) rather
    than on a hand-written registry row, so the durable write and the durable read
    are proven to agree.
    """
    import ouroboros.workspace_admission as admission
    from ouroboros.gateway.task_placement import _admit_task_placement

    drive = _drive(tmp_path)
    create_project(drive, "remote-app", placement=_REMOTE)
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(
        admission, "remote_session_facts",
        lambda ref, **_: {"canonical_root": "/srv/work/app", "workspace_id": "ws-1"},
    )

    ref = _admit_task_placement(
        {"project_id": "remote-app"}, system_repo_dir=tmp_path / "sys", drive_root=drive
    )
    assert ref.to_payload() == _REMOTE

    # A workspace_root of its own cannot be combined with a remote project: the
    # placement is the project's, and a second one would have no generation to fence.
    local = tmp_path / "local-repo"
    _init_git_repo(local)
    with pytest.raises(ValueError) as combined:
        _admit_task_placement(
            {"project_id": "remote-app", "workspace_root": str(local)},
            system_repo_dir=tmp_path / "sys", drive_root=drive,
        )
    assert "inherited from project_id" in str(combined.value)


def test_a_rebind_makes_the_previous_generation_stale_at_the_fence(tmp_path, monkeypatch):
    """Rebind → the fence taken before it refuses, by generation, not by guesswork."""
    import ouroboros.workspace_admission as admission

    drive = _drive(tmp_path)
    create_project(drive, "remote-app", placement=_REMOTE)
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    ref = project_placement(get_project(drive, "remote-app"))

    fence = admission.placement_fence_for(drive, "remote-app", ref)
    assert fence["routing_generation"] == 0
    assert fence["connection_id"] == "conn-1"
    assert admission.placement_fence_stale_reason(drive, fence) == ""

    set_project_placement(drive, "remote-app", {**_REMOTE, "remote_root": "/srv/work/other"})
    assert (
        admission.placement_fence_stale_reason(drive, fence)
        == "placement_routing_generation_stale"
    )
    # A fence taken AFTER the rebind is current again — the fence tracks routing, it
    # does not simply decay.
    fresh = admission.placement_fence_for(
        drive, "remote-app", project_placement(get_project(drive, "remote-app"))
    )
    assert fresh["routing_generation"] == 1
    assert admission.placement_fence_stale_reason(drive, fresh) == ""


def test_retiring_the_connection_breaks_the_project_honestly(tmp_path, monkeypatch):
    """Retire is not a silent demotion to local: both doors refuse TYPED.

    The fence names `placement_connection_retired`, and admission of a new task in
    that project refuses with "not a known remote connection" — never a Home
    `working_dir` stand-in (a remote project has none) and never a workspace-less
    task, which would resolve to the self_modification profile over the system repo.
    """
    import ouroboros.workspace_admission as admission
    from ouroboros import config, connection_store

    store = tmp_path / "connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    row = connection_store.add_connection(name="Build box", ssh_alias="build", path=store)

    drive = _drive(tmp_path)
    placement = {**_REMOTE, "connection_id": row["id"]}
    create_project(drive, "remote-app", placement=placement)
    monkeypatch.setattr(
        admission, "remote_session_facts",
        lambda ref, **_: {"canonical_root": "/srv/work/app", "workspace_id": "ws-1"},
    )
    ref = project_placement(get_project(drive, "remote-app"))
    fence = admission.placement_fence_for(drive, "remote-app", ref)
    assert admission.placement_fence_stale_reason(drive, fence) == ""

    connection_store.retire_connection(row["id"], path=store)

    assert (
        admission.placement_fence_stale_reason(drive, fence)
        == "placement_connection_retired"
    )
    resolved, error = admission.resolve_room_workspace(
        drive_root=drive, system_repo_dir=tmp_path / "sys", project_id="remote-app"
    )
    assert resolved is None
    assert "not a known remote connection" in error
    # The placement itself is untouched: retiring a connection is not a rewrite of
    # every project that used it, so a re-trust restores the project as it was.
    assert get_project(drive, "remote-app")["placement"] == placement


# --- 4. the half-open admission door -----------------------------------------

def _facts(**overrides):
    facts = {"canonical_root": "/srv/work/app", "workspace_id": "target-ws-7"}
    facts.update(overrides)
    return facts


def test_admit_remote_placement_seals_the_target_allocated_workspace_id(monkeypatch):
    """The owner names host + path; the target names the workspace identity.

    A client that could name the identity could claim a workspace it never opened,
    which is why the project surface takes two fields and not a ref blob.
    """
    import ouroboros.workspace_admission as admission

    asked = {}
    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(
        admission, "_session_facts",
        lambda connection_id, remote_root, **kwargs: asked.update(
            {"connection_id": connection_id, "remote_root": remote_root, **kwargs}
        ) or _facts(),
    )

    ref = admission.admit_remote_placement(
        connection_id=" conn-1 ", remote_root="/srv/work/app/", project_id="remote-app"
    )
    assert ref.to_payload() == {
        "kind": "ssh",
        "connection_id": "conn-1",
        "remote_root": "/srv/work/app",
        "workspace_id": "target-ws-7",
    }
    # The broker is asked with NO workspace identity and the project as its scope.
    assert asked["workspace_id"] == ""
    assert asked["project_id"] == "remote-app"


@pytest.mark.parametrize(
    ("connection_id", "remote_root", "fragment"),
    [
        ("conn-1", "srv/work/app", "absolute POSIX path"),
        ("conn-1", "/srv/../etc", "traversal segments"),
        ("conn-1", "/", "git worktree root"),
        ("conn-1", "", "remote_root is required"),
        ("", "/srv/work/app", "connection_id is required"),
        ("conn\n1", "/srv/work/app", "connection_id is required"),
    ],
)
def test_admit_remote_placement_refuses_form_before_any_store_or_target_call(
    monkeypatch, connection_id, remote_root, fragment
):
    import ouroboros.workspace_admission as admission

    touched = []
    monkeypatch.setattr(
        admission, "known_connections", lambda: touched.append("store") or {"conn-1": ""}
    )
    monkeypatch.setattr(
        admission, "_session_facts",
        lambda *a, **k: touched.append("target") or _facts(),
    )

    with pytest.raises(admission.WorkspaceRootError) as err:
        admission.admit_remote_placement(
            connection_id=connection_id, remote_root=remote_root, project_id="p"
        )
    assert fragment in str(err.value)
    assert touched == []


def test_admit_remote_placement_needs_a_named_project(monkeypatch):
    """The project id is the broker's session scope AND the row that holds the
    routing generation, so an unnamed placement has nothing to fence."""
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": ""})
    with pytest.raises(admission.WorkspaceRootError) as err:
        admission.admit_remote_placement(
            connection_id="conn-1", remote_root="/srv/work/app", project_id="  "
        )
    assert "NAMED project" in str(err.value)


def test_admit_remote_placement_refuses_an_unknown_connection(monkeypatch):
    import ouroboros.workspace_admission as admission

    probed = []
    monkeypatch.setattr(admission, "known_connections", lambda: {"other": ""})
    monkeypatch.setattr(
        admission, "_session_facts", lambda *a, **k: probed.append(a) or _facts()
    )

    with pytest.raises(admission.WorkspaceRootError) as err:
        admission.admit_remote_placement(
            connection_id="conn-1", remote_root="/srv/work/app", project_id="p"
        )
    assert "not a known remote connection: conn-1" in str(err.value)
    assert probed == []
    assert not isinstance(err.value, admission.RemoteWorkspaceUnavailableError)


def test_admit_remote_placement_refuses_a_subdirectory_of_a_worktree(monkeypatch):
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": ""})
    monkeypatch.setattr(
        admission, "_session_facts",
        lambda *a, **k: _facts(canonical_root="/srv/work"),
    )
    with pytest.raises(admission.WorkspaceRootError) as err:
        admission.admit_remote_placement(
            connection_id="conn-1", remote_root="/srv/work/app", project_id="p"
        )
    assert str(err.value) == "workspace_ref.remote_root must be the git worktree root: /srv/work"


def test_admit_remote_placement_refuses_typed_without_a_workspace_identity(monkeypatch):
    """A target that admits but names no workspace cannot be sealed — and that is an
    UNAVAILABLE target, not a malformed request."""
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": ""})
    monkeypatch.setattr(
        admission, "_session_facts", lambda *a, **k: _facts(workspace_id="")
    )
    with pytest.raises(admission.RemoteWorkspaceUnavailableError) as err:
        admission.admit_remote_placement(
            connection_id="conn-1", remote_root="/srv/work/app", project_id="p"
        )
    assert "no workspace identity" in str(err.value)


def test_admit_remote_placement_never_falls_back_to_home(tmp_path, monkeypatch):
    """No broker in this process is a TYPED refusal, exactly as for a task ref."""
    import ouroboros.workspace_admission as admission
    from ouroboros import config, connection_store

    store = tmp_path / "connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    row = connection_store.add_connection(name="Build box", ssh_alias="build", path=store)

    with pytest.raises(admission.RemoteWorkspaceUnavailableError) as err:
        admission.admit_remote_placement(
            connection_id=row["id"], remote_root="/srv/work/app", project_id="p"
        )
    assert err.value.code == admission.REMOTE_TRANSPORT_UNAVAILABLE
    assert "remote_workspace_unavailable" in str(err.value)


# --- 5. the endpoints the owner actually reaches ------------------------------


class _FakeBroker:
    """The two broker calls the project surface makes, and nothing else.

    `admit_workspace` answers like `remote_session_admission.session_admission_result`
    (identities + target-native facts, no transport handle); `close_project_session`
    records the provisional closes so a leaked session is a visible test failure.
    """

    def __init__(self, *, canonical_root="/srv/work/app", workspace_id="target-ws-7", on_admit=None):
        self.canonical_root = canonical_root
        self.workspace_id = workspace_id
        self.on_admit = on_admit
        self.admissions: list[dict] = []
        self.closed: list[tuple] = []

    def admit_workspace(self, connection, *, remote_root, project_id, workspace_id="", task_id=""):
        self.admissions.append({
            "connection_id": connection.get("id"),
            "remote_root": remote_root,
            "project_id": project_id,
            "workspace_id": workspace_id,
            "task_id": task_id,
        })
        if self.on_admit is not None:
            self.on_admit()
        return {
            "ok": True,
            "canonical_root": self.canonical_root,
            "workspace_id": self.workspace_id,
            "host_id": "host-1",
            "handshake": {"platform": {"system": "Linux", "python": "3.12.3"}},
        }

    def close_project_session(self, workspace_ref, *, project_id):
        self.closed.append((dict(workspace_ref), project_id))
        return True


def _remote_env(tmp_path, monkeypatch, **broker_kwargs):
    """A drive, a trusted connection in the owner store, and a live fake broker.

    Rebase onto v6.82: `/api/tasks` now RESERVES admission with
    `require_worker_pool=True` and requires `persist_queue_snapshot` to return exactly
    `True`, otherwise it rolls the admission back with a 503. These cases are about
    PLACEMENT, so the pool is modelled ready exactly as upstream's own HTTP task tests
    do (`tests/test_headless_cli.py::_managed_worker_pool_available`) and the snapshot
    is stubbed successful — the durability of the snapshot has its own tests.
    """
    from types import SimpleNamespace

    import supervisor.workers as workers

    from ouroboros import config, connection_store, remote_workspace

    drive = _drive(tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    # The placement fence re-reads the project registry from the QUEUE's drive, so it has
    # to be the same drive the test wrote the project into — otherwise every remote
    # admission here is refused `placement_project_missing`. Pre-v6.82 this was masked:
    # the handler persisted the SCHEDULED record BEFORE enqueueing, so `/api/tasks/{id}`
    # returned a full record even when the fence had refused the task. Upstream now
    # publishes only after a successful enqueue, which is why the gap became visible.
    monkeypatch.setattr("supervisor.queue.DRIVE_ROOT", drive)

    store = tmp_path / "connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    row = connection_store.add_connection(name="Build box", ssh_alias="build", path=store)
    broker = _FakeBroker(**broker_kwargs)
    monkeypatch.setattr(remote_workspace, "_REMOTE_WORKSPACE_SERVICE", broker, raising=False)

    class _Req:
        def __init__(self, body, path_params=None):
            self._body = body
            self.path_params = path_params or {}
            self.app = SimpleNamespace(state=SimpleNamespace(
                drive_root=drive,
                repo_dir=tmp_path / "repo",
                remote_workspace_service=broker,
                remote_connections_path=store,
            ))

        async def json(self):
            return self._body

    return SimpleNamespace(drive=drive, store=store, connection=row, broker=broker, Req=_Req)


def _json(response):
    import json

    return json.loads(response.body)


def test_creating_a_remote_project_is_admitted_on_the_target_and_sealed(tmp_path, monkeypatch):
    """The owner path, end to end: two fields in, a sealed placement out.

    The request never names the workspace identity and the response carries the one
    the TARGET allocated, which is the whole reason the contract takes two halves.
    """
    import asyncio

    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    response = asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))
    payload = _json(response)
    assert response.status_code == 200, payload

    project = payload["project"]
    assert project["placement"] == {
        "kind": "ssh",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
        "workspace_id": "target-ws-7",
    }
    assert project["working_dir"] == ""
    assert project["provenance"] == "remote"
    # Choosing the folder IS the grant (notification trust model), same as attach.
    assert project["trusted_at"]

    # The broker was asked for a placement with NO identity, scoped to the project.
    assert env.broker.admissions == [{
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
        "project_id": "remote-app",
        "workspace_id": "",
        "task_id": "",
    }]
    assert env.broker.closed == []

    # Durable, and a task born in the room inherits it.
    assert project_placement(get_project(env.drive, "remote-app")).workspace_id == "target-ws-7"


def test_creating_a_remote_project_refuses_an_unknown_connection(tmp_path, monkeypatch):
    import asyncio

    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    response = asyncio.run(api_projects_create(env.Req({
        "name": "Remote app", "connection_id": "conn-nope", "remote_root": "/srv/work/app",
    })))
    payload = _json(response)
    assert response.status_code == 400
    assert payload["error_code"] == "invalid_remote_placement"
    assert "not a known remote connection" in payload["error"]
    # Refused BEFORE the target is touched, and no half-built project row remains.
    assert env.broker.admissions == []
    assert get_project(env.drive, "remote-app") is None


def test_creating_a_remote_project_refuses_a_path_that_is_not_the_worktree_root(tmp_path, monkeypatch):
    import asyncio

    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch, canonical_root="/srv/work")
    response = asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))
    payload = _json(response)
    assert response.status_code == 400
    assert payload["error_code"] == "invalid_remote_placement"
    assert "/srv/work" in payload["error"]
    assert get_project(env.drive, "remote-app") is None


def test_an_unreachable_target_is_a_503_not_a_local_project(tmp_path, monkeypatch):
    """No silent fallback at the owner surface either: the project is NOT created
    as a local one when the target cannot be consulted."""
    import asyncio

    from ouroboros import remote_workspace
    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    monkeypatch.setattr(remote_workspace, "_REMOTE_WORKSPACE_SERVICE", None, raising=False)

    response = asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))
    payload = _json(response)
    assert response.status_code == 503
    assert payload["error_code"] == "remote_transport_unavailable"
    # The action comes from the REFUSAL, not from the handler
    # (`remote_refusal_actions`): there is no broker in this process, and no amount of
    # Bootstrap installs one. This line asserted `bootstrap_connection` until the
    # handler stopped hardcoding it — which was correct for a stale executor and a
    # dead end for every other reason a target can be unreachable.
    assert payload["action"] == "restart_ouroboros"
    assert get_project(env.drive, "remote-app") is None


def test_both_admission_doors_answer_an_unreachable_target_the_same_way(tmp_path, monkeypatch):
    """ONE condition, ONE typed answer — whichever endpoint asked.

    `RemoteWorkspaceUnavailableError` subclasses `ValueError` deliberately, so every
    legacy `except ValueError` admission site keeps refusing loudly. The cost is that
    a GENERIC `except ValueError` swallows it: `/api/tasks` answered a bare 400 whose
    only clue was the UPPERCASE constant embedded in English prose, while
    `/api/projects` answered 503 + `remote_transport_unavailable` + a next step for
    the identical situation. A client cannot branch on prose, and a 400 tells it to
    fix a request that was never wrong.

    Both doors now take the action off the refusal instead of naming one, so "the
    same answer" also means the same NEXT STEP — and for this condition (no broker in
    this process) that step is a restart, not the Bootstrap both used to advise.
    """
    import asyncio

    import ouroboros.workspace_admission as admission
    from ouroboros import remote_workspace
    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.gateway.tasks import api_tasks_create

    env = _remote_env(tmp_path, monkeypatch)
    # The placement names the connection the OWNER STORE actually holds, so the
    # refusal below is about reaching the target and not about an unknown id.
    placement = {**_REMOTE, "connection_id": env.connection["id"]}
    create_project(env.drive, "remote-app", placement=placement)
    monkeypatch.setattr(
        admission, "known_connections", lambda: {env.connection["id"]: ""},
    )
    # No broker in this server: the placement is well formed and simply cannot be
    # consulted — the one condition both doors have to render identically.
    monkeypatch.setattr(remote_workspace, "_REMOTE_WORKSPACE_SERVICE", None, raising=False)

    task_response = asyncio.run(api_tasks_create(env.Req({
        "description": "build the thing", "project_id": "remote-app",
    })))
    project_response = asyncio.run(api_projects_create(env.Req({
        "name": "Second app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))

    for response in (task_response, project_response):
        payload = _json(response)
        assert response.status_code == 503, payload
        assert payload["error_code"] == "remote_transport_unavailable"
        assert payload["action"] == "restart_ouroboros"
    # The wire spelling, not the authority's constant leaking into a field the
    # browser and the CLI both compare case-sensitively.
    assert admission.REMOTE_TRANSPORT_UNAVAILABLE != _json(task_response)["error_code"]

    # A genuinely malformed request is STILL a 400: the typed arm must not have
    # widened into "any admission refusal is a 503".
    malformed = asyncio.run(api_tasks_create(env.Req({
        "description": "build", "project_id": "remote-app", "connection_id": "conn-1",
    })))
    assert malformed.status_code == 400
    assert "never chosen per task" in _json(malformed)["error"]


@pytest.mark.parametrize(
    "body",
    [
        {"name": "R", "connection_id": "c"},
        {"name": "R", "remote_root": "/srv/work/app"},
    ],
)
def test_half_a_remote_placement_is_refused_not_ignored(tmp_path, monkeypatch, body):
    import asyncio

    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    response = asyncio.run(api_projects_create(env.Req(body)))
    payload = _json(response)
    assert response.status_code == 400
    assert payload["error_code"] == "invalid_remote_placement"
    assert "two halves" in payload["error"]
    # Silently ignoring half a request would have created a FILE-LESS project.
    assert get_project(env.drive, "r") is None


def test_a_remote_source_cannot_be_combined_with_a_local_one(tmp_path, monkeypatch):
    import asyncio

    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    response = asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
        "with_workspace": True,
    })))
    assert response.status_code == 400
    assert "choose ONE source" in _json(response)["error"]
    assert env.broker.admissions == []


def test_a_failed_registry_commit_closes_the_provisional_session(tmp_path, monkeypatch):
    """Donor finding 2.4: admission OPENS a session on the target, so a commit that
    never lands must not leave it behind holding the connection."""
    import asyncio

    from ouroboros.gateway import projects as projects_gateway

    env = _remote_env(tmp_path, monkeypatch)

    def _explode(*args, **kwargs):
        raise RuntimeError("registry is on fire")

    monkeypatch.setattr("ouroboros.projects_registry.create_project", _explode)
    response = asyncio.run(projects_gateway.api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))
    assert response.status_code == 500
    assert len(env.broker.admissions) == 1
    assert env.broker.closed == [({
        "kind": "ssh",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
        "workspace_id": "target-ws-7",
    }, "remote-app")]


def test_the_local_create_path_writes_no_placement(tmp_path, monkeypatch):
    """The local sources are byte-identical: no placement key appears on their rows."""
    import asyncio

    from ouroboros.gateway.projects import api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    folder = tmp_path / "local-repo"
    _init_git_repo(folder)

    fileless = _json(asyncio.run(api_projects_create(env.Req({"name": "Chat only"}))))["project"]
    attached = _json(asyncio.run(api_projects_create(env.Req({
        "name": "Local app", "path": str(folder),
    }))))["project"]

    assert fileless.get("placement") is None and fileless["provenance"] == "none"
    assert attached.get("placement") is None and attached["provenance"] == "attached"
    assert attached["working_dir"] == str(folder.resolve())
    assert env.broker.admissions == []


def test_the_update_endpoint_rebinds_and_advances_the_generation(tmp_path, monkeypatch):
    import asyncio

    from ouroboros.gateway.projects import api_project_update, api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    created = _json(asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    }))))["project"]
    assert created["routing_generation"] == 0

    env.broker.canonical_root = "/srv/work/other"
    env.broker.workspace_id = "target-ws-9"
    response = asyncio.run(api_project_update(env.Req(
        {"connection_id": env.connection["id"], "remote_root": "/srv/work/other"},
        path_params={"project_id": "remote-app"},
    )))
    payload = _json(response)
    assert response.status_code == 200, payload
    assert payload["project"]["placement"]["remote_root"] == "/srv/work/other"
    assert payload["project"]["placement"]["workspace_id"] == "target-ws-9"
    assert payload["project"]["routing_generation"] == 1
    assert env.broker.closed == []

    # A rename still works alone, and combined with a rebind.
    renamed = _json(asyncio.run(api_project_update(env.Req(
        {"name": "Renamed"}, path_params={"project_id": "remote-app"},
    ))))["project"]
    assert renamed["name"] == "Renamed"
    assert renamed["routing_generation"] == 1

    # And an empty body is a refusal, not a silent no-op.
    empty = asyncio.run(api_project_update(env.Req({}, path_params={"project_id": "remote-app"})))
    assert empty.status_code == 400
    assert "name or connection_id+remote_root" in _json(empty)["error"]


def test_a_rebind_is_refused_while_the_project_has_live_tasks(tmp_path, monkeypatch):
    """Refused BEFORE admission: a running task's placement was sealed at its own
    admission, so a rebind could not redirect it — the owner would believe the project
    moved while work kept writing to the old host."""
    import asyncio

    from ouroboros.gateway import projects as projects_gateway
    from ouroboros.gateway.projects import api_project_update, api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))
    env.broker.admissions.clear()
    monkeypatch.setattr(projects_gateway, "_project_has_live_tasks", lambda *_: True)

    response = asyncio.run(api_project_update(env.Req(
        {"connection_id": env.connection["id"], "remote_root": "/srv/work/other"},
        path_params={"project_id": "remote-app"},
    )))
    payload = _json(response)
    assert response.status_code == 409
    assert payload["error_code"] == "project_has_live_tasks"
    # No target call at all, so there is no session to leak.
    assert env.broker.admissions == [] and env.broker.closed == []
    assert project_placement(get_project(env.drive, "remote-app")).remote_root == "/srv/work/app"


def test_a_rebind_is_refused_when_the_live_task_check_ITSELF_fails(tmp_path, monkeypatch):
    """FAIL-CLOSED: "could not tell" must refuse, not read as "no live tasks".

    `_project_has_live_tasks` wrapped its whole body in `except Exception: return False`,
    and `False` is the value that lets a rebind through — so any failure inside the lookup
    turned the guard into a green light for the exact rebind its docstring promises to
    refuse. That is worse than having no guard, because the surface claims a check that is
    not happening. The lookup is made to RAISE here, which is the condition the old code
    swallowed, and the assertions are the ones the busy case makes: same 409, same typed
    code, and no target call, so there is no session to leak either.

    `gateway/connections._connection_busy` already answered this shape of question this
    way (`None` = could not tell = busy), so this is the codebase agreeing with itself
    rather than a new idea.
    """
    import asyncio

    from ouroboros.gateway import projects as projects_gateway
    from ouroboros.gateway.projects import api_project_update, api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))
    env.broker.admissions.clear()

    def exploding(*_args, **_kwargs):
        raise RuntimeError("queue snapshot unreadable")

    # Patched at the SEAM the production function imports, so the real
    # `_project_has_live_tasks` body runs and its own `except` arm is what is under test.
    import supervisor.task_lifecycle as task_lifecycle

    monkeypatch.setattr(task_lifecycle, "_live_project_task_ids", exploding)
    assert projects_gateway._project_has_live_tasks(env.drive, "remote-app") is None, (
        "an unanswerable lookup must report None, not False"
    )

    response = asyncio.run(api_project_update(env.Req(
        {"connection_id": env.connection["id"], "remote_root": "/srv/work/other"},
        path_params={"project_id": "remote-app"},
    )))
    payload = _json(response)
    assert response.status_code == 409
    assert payload["error_code"] == "project_has_live_tasks"
    # The message distinguishes it from a genuinely busy project even though the code and
    # the owner action are the same — the next step IS the same, the cause is not.
    assert "could not determine" in payload["error"], payload
    assert env.broker.admissions == [] and env.broker.closed == []
    assert project_placement(get_project(env.drive, "remote-app")).remote_root == "/srv/work/app"


def test_a_rebind_that_loses_the_race_refuses_and_closes_its_session(tmp_path, monkeypatch):
    """A concurrent rebind landing DURING admission makes this one the loser.

    The generation is read before admission and asserted at the write, so the loser
    is told (`project_routing_generation_changed`) rather than overwriting the winner
    — and the session its own admission opened is closed instead of leaked.
    """
    import asyncio

    from ouroboros.gateway.projects import api_project_update, api_projects_create

    env = _remote_env(tmp_path, monkeypatch)
    asyncio.run(api_projects_create(env.Req({
        "name": "Remote app",
        "connection_id": env.connection["id"],
        "remote_root": "/srv/work/app",
    })))

    def _concurrent_rebind():
        set_project_placement(
            env.drive, "remote-app",
            {**_REMOTE, "connection_id": env.connection["id"], "remote_root": "/srv/work/winner"},
        )

    env.broker.on_admit = _concurrent_rebind
    env.broker.canonical_root = "/srv/work/loser"
    response = asyncio.run(api_project_update(env.Req(
        {"connection_id": env.connection["id"], "remote_root": "/srv/work/loser"},
        path_params={"project_id": "remote-app"},
    )))
    payload = _json(response)
    assert response.status_code == 409
    assert payload["error_code"] == "project_routing_generation_changed"
    assert payload["action"] == "reload_projects"
    # The winner's placement stands, and the loser's session was closed.
    assert project_placement(get_project(env.drive, "remote-app")).remote_root == "/srv/work/winner"
    assert [ref["remote_root"] for ref, _ in env.broker.closed] == ["/srv/work/loser"]


def test_the_sidebar_projection_says_where_a_project_lives(tmp_path):
    drive = _drive(tmp_path)
    folder = tmp_path / "local-repo"
    folder.mkdir()
    create_project(drive, "local-app", working_dir=str(folder), origin="owner_ui")
    create_project(drive, "remote-app", placement=_REMOTE, origin="owner_ui")

    rows = {row["id"]: row for row in projects_summary(drive)}
    assert rows["remote-app"]["placement"] == _REMOTE
    assert rows["remote-app"]["working_dir"] == ""
    assert rows["local-app"]["placement"] is None
    assert rows["local-app"]["working_dir"] == str(folder)


def test_the_admission_evidence_the_browser_reads_is_the_one_home_writes(tmp_path, monkeypatch):
    """M5 end to end: admission → task record → `/api/tasks/{id}` → the JS reducer.

    `web/modules/remote_task_state.js` derives a remote status from durable task rows
    whenever there is no live `connection_state` frame — which is every page reload.
    It used to look for `task.remote_admission` and
    `metadata._remote_admission_evidence`, and NO Python producer ever wrote either:
    both evidence branches were permanently false, every derived status collapsed to
    'unknown', `stateSource` was always 'derived', and `canReconnect` was therefore
    structurally unreachable. The owner saw "SSH Unknown" with nothing to do.

    Home does publish admission evidence — under the name it already had. The remote
    arm of the sealed preflight summary IS target identity (which host, which release,
    which canonical root answered at admission), and a remote preflight that could not
    be taken discloses `error` instead of raising. This test pins the exact key names
    on the wire; the reducer half is pinned in
    `web/tests/remote_task_state.test.js::"a reloaded page derives remote status from
    the preflight admission really seals"`, and the two must be read together.
    """
    import asyncio

    import ouroboros.workspace_admission as admission
    from ouroboros.gateway.tasks import api_task_get, api_tasks_create

    env = _remote_env(tmp_path, monkeypatch)
    # The workspace identity is the TARGET's to allocate, so the stored placement
    # carries the one this broker hands out — a mismatch is its own refusal.
    placement = {
        **_REMOTE,
        "connection_id": env.connection["id"],
        "workspace_id": env.broker.workspace_id,
    }
    create_project(env.drive, "remote-app", placement=placement)
    monkeypatch.setattr(
        admission, "known_connections", lambda: {env.connection["id"]: ""},
    )

    created = _json(asyncio.run(api_tasks_create(env.Req({
        "description": "build the thing", "project_id": "remote-app",
    }))))
    assert "task_id" in created and created.get("ok") is True, created
    task_id = created["task_id"]

    fetched = _json(asyncio.run(api_task_get(env.Req({}, {"task_id": task_id}))))
    metadata = fetched["metadata"]
    # The placement read the reducer does first (`remotePlacementFromTask`).
    assert metadata["_sealed_workspace_ref"]["kind"] == "ssh"
    assert metadata["_sealed_workspace_ref"]["connection_id"] == env.connection["id"]
    # And the admission evidence it derives from: `placement: 'ssh'` means the TARGET
    # answered, and these are the identities it answered with.
    preflight = metadata["workspace_preflight"]
    assert preflight["placement"] == "ssh"
    assert preflight["host_id"] == "host-1"
    assert preflight["canonical_root"] == "/srv/work/app"
    assert "error" not in preflight
    # The keys the browser looked for do not exist and never did.
    assert "remote_admission" not in fetched
    assert "_remote_admission_evidence" not in metadata


def test_an_admission_that_could_not_reach_the_target_discloses_it_in_the_task_row(tmp_path, monkeypatch):
    """The negative half of the same evidence, on the same wire.

    `placement_preflight_summary` DISCLOSES a failed remote preflight rather than
    raising — by then the placement is admitted, and a task must not be lost because
    its context block could not be decorated. That `error` is what lets the browser
    derive 'degraded' (and therefore offer Reconnect) for a task whose failure really
    is about the connection, without inventing SSH health for a model failure.
    """
    import asyncio

    import ouroboros.workspace_admission as admission
    from ouroboros.gateway.tasks import api_task_get, api_tasks_create

    # Admission itself succeeds and only the SECOND consultation — the preflight
    # read that decorates the task — finds the target gone. That gap is exactly the
    # window the disclosing arm exists for.
    calls = {"n": 0}

    def fail_after_admission():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("target went away")

    env = _remote_env(tmp_path, monkeypatch, on_admit=fail_after_admission)
    placement = {
        **_REMOTE,
        "connection_id": env.connection["id"],
        "workspace_id": env.broker.workspace_id,
    }
    create_project(env.drive, "remote-app", placement=placement)
    monkeypatch.setattr(
        admission, "known_connections", lambda: {env.connection["id"]: ""},
    )

    created = _json(asyncio.run(api_tasks_create(env.Req({
        "description": "build the thing", "project_id": "remote-app",
    }))))
    fetched = _json(asyncio.run(api_task_get(env.Req({}, {"task_id": created["task_id"]}))))
    preflight = fetched["metadata"]["workspace_preflight"]
    assert "placement" not in preflight
    assert "target went away" in preflight["error"]
