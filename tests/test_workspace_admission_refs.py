"""RWS v2 §3.3 — discriminated workspace admission returning a SEALED placement.

Three things are pinned here:

1. **Local parity.** The local branch's accept/reject decisions AND its exact error
   messages are the v6.58.0 ones; only the RETURN TYPE became a sealed ref. The
   parity is asserted against the messages themselves, not a paraphrase.
2. **SSH form validation is complete and LOCAL.** Every part of an ssh ref a Home
   process can judge (shape, absoluteness, traversal, identity fields, unknown
   fields, connection known to the owner store) is judged before any target fact is
   requested — a malformed ref never reaches the transport.
3. **No silent fallback.** A well-formed ssh ref whose target cannot be consulted
   refuses TYPED (`REMOTE_TRANSPORT_UNAVAILABLE`); it never degrades to a Home path,
   and the Home-path projection refuses typed too.
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


def _ssh_payload(**overrides):
    payload = {
        "kind": "ssh",
        "connection_id": "conn-1",
        "remote_root": "/srv/work/app",
        "workspace_id": "ws-1",
    }
    payload.update(overrides)
    return payload


# --- 1. local parity ----------------------------------------------------------

def test_local_branch_messages_and_decisions_are_unchanged(tmp_path):
    """Byte-level parity of the LOCAL branch: same order (overlap → existence →
    worktree-root), same message text, only the return type is now a sealed ref."""
    from ouroboros.workspace_admission import WorkspaceRootError, validate_workspace_root

    sysrepo = tmp_path / "sys"
    data = tmp_path / "data"
    ws = tmp_path / "repo"
    _init_git_repo(ws)
    _init_git_repo(sysrepo)

    ref = validate_workspace_root(str(ws), system_repo_dir=sysrepo, drive_root=data)
    assert ref.kind == "local"
    assert ref.home_path() == ws.resolve()
    assert ref.to_payload() == {"kind": "local", "local_root": str(ws.resolve())}

    # Empty input is "no workspace requested", not an error.
    assert validate_workspace_root("", system_repo_dir=sysrepo, drive_root=data) is None
    assert validate_workspace_root(None, system_repo_dir=sysrepo, drive_root=data) is None

    # Overlap is checked BEFORE existence: a non-existent path under the system repo
    # still reports the overlap message, exactly as before.
    with pytest.raises(WorkspaceRootError) as overlap:
        validate_workspace_root(str(sysrepo / "nope"), system_repo_dir=sysrepo, drive_root=data)
    assert str(overlap.value) == "workspace_root must not overlap the Ouroboros system repo"

    with pytest.raises(WorkspaceRootError) as drive_overlap:
        validate_workspace_root(str(data / "nope"), system_repo_dir=sysrepo, drive_root=data)
    assert str(drive_overlap.value) == "workspace_root must not overlap the Ouroboros data drive"

    missing = tmp_path / "missing"
    with pytest.raises(WorkspaceRootError) as absent:
        validate_workspace_root(str(missing), system_repo_dir=sysrepo, drive_root=data)
    assert str(absent.value) == f"workspace_root is not a directory: {missing}"

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorkspaceRootError) as not_git:
        validate_workspace_root(str(plain), system_repo_dir=sysrepo, drive_root=data)
    assert str(not_git.value) == "workspace_root must be a git worktree root"

    sub = ws / "src"
    sub.mkdir()
    with pytest.raises(WorkspaceRootError) as subdir:
        validate_workspace_root(str(sub), system_repo_dir=sysrepo, drive_root=data)
    assert str(subdir.value) == f"workspace_root must be the git worktree root: {ws.resolve()}"


def test_local_ref_payload_is_admitted_like_its_path_spelling(tmp_path):
    """A caller may pass the local placement as a REF instead of a bare string; the
    admitted result is identical, so a re-submitted sealed ref is stable."""
    from ouroboros.workspace_admission import validate_workspace_root

    ws = tmp_path / "repo"
    _init_git_repo(ws)
    kwargs = {"system_repo_dir": tmp_path / "sys", "drive_root": tmp_path / "data"}
    from_text = validate_workspace_root(str(ws), **kwargs)
    from_ref = validate_workspace_root(from_text.to_payload(), **kwargs)
    assert from_ref == from_text


# --- 2. ssh form validation, fully local -------------------------------------

@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        (_ssh_payload(remote_root="srv/work/app"), "absolute POSIX path"),
        (_ssh_payload(remote_root="/srv/../etc"), "traversal segments"),
        (_ssh_payload(remote_root="/"), "git worktree root"),
        (_ssh_payload(connection_id=""), "connection_id is required"),
        (_ssh_payload(workspace_id=""), "workspace_id is required"),
        (_ssh_payload(connection_id="conn\n1"), "control character"),
        ({**_ssh_payload(), "host": "10.0.8.31"}, "unknown fields"),
        ({"kind": "docker_exec", "local_root": "/x"}, "reserved"),
        ({"kind": "webdav", "remote_root": "/x"}, "must be 'local' or 'ssh'"),
    ],
)
def test_ssh_ref_form_is_refused_locally_before_any_target_call(tmp_path, monkeypatch, payload, fragment):
    """A malformed ssh ref is refused on FORM alone: the connection store and the
    target fact door are never consulted."""
    import ouroboros.workspace_admission as admission

    touched = []
    monkeypatch.setattr(admission, "known_connections", lambda: touched.append("store") or {"conn-1": "host-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: touched.append("target") or {})

    with pytest.raises(admission.WorkspaceRootError) as err:
        admission.validate_workspace_root(payload, system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data")
    assert fragment in str(err.value)
    assert touched == []


def test_unknown_connection_is_refused_before_the_target_is_consulted(tmp_path, monkeypatch):
    import ouroboros.workspace_admission as admission

    probed = []
    monkeypatch.setattr(admission, "known_connections", lambda: {"other": "host-2"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: probed.append(ref) or {})

    with pytest.raises(admission.WorkspaceRootError) as err:
        admission.validate_workspace_root(
            _ssh_payload(), system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data"
        )
    assert "not a known remote connection: conn-1" in str(err.value)
    assert probed == []
    # An UNKNOWN connection is a malformed request, not an unavailable transport.
    assert not isinstance(err.value, admission.RemoteWorkspaceUnavailableError)


def test_target_worktree_identity_is_enforced_in_the_target_spelling(tmp_path, monkeypatch):
    """The ssh twin of the local worktree-root gate: the target's own toplevel must
    equal the requested root, judged in POSIX spellings with no Home pathlib."""
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    kwargs = {"system_repo_dir": tmp_path / "sys", "drive_root": tmp_path / "data"}

    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": ""})
    with pytest.raises(admission.WorkspaceRootError) as not_git:
        admission.validate_workspace_root(_ssh_payload(), **kwargs)
    assert "must be a git worktree root on the target" in str(not_git.value)

    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work"})
    with pytest.raises(admission.WorkspaceRootError) as subdir:
        admission.validate_workspace_root(_ssh_payload(), **kwargs)
    assert str(subdir.value) == "workspace_ref.remote_root must be the git worktree root: /srv/work"

    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})
    admitted = admission.validate_workspace_root(_ssh_payload(), **kwargs)
    assert admitted.kind == "ssh"
    assert admitted.to_payload() == _ssh_payload()


# --- 3. no silent fallback ----------------------------------------------------

def test_wellformed_ssh_ref_refuses_typed_when_no_broker_can_be_reached(tmp_path, monkeypatch):
    """A KNOWN connection with no reachable broker is a TYPED refusal, never a Home path.

    This is the shape a process without a live broker takes — a CLI, a worker before
    lifespan, a build without the transport. The placement is well formed and the
    connection is real, so the only two possible answers are "admitted on target
    evidence" and "refused"; there is deliberately no third.
    """
    import ouroboros.workspace_admission as admission
    from ouroboros import config, connection_store

    store = tmp_path / "connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    row = connection_store.add_connection(name="Build box", ssh_alias="build", path=store)
    monkeypatch.setattr(admission, "known_connections", lambda: {row["id"]: ""})

    with pytest.raises(admission.RemoteWorkspaceUnavailableError) as err:
        admission.validate_workspace_root(
            _ssh_payload(connection_id=row["id"]),
            system_repo_dir=tmp_path / "sys",
            drive_root=tmp_path / "data",
        )
    assert err.value.code == admission.REMOTE_TRANSPORT_UNAVAILABLE
    assert admission.REMOTE_TRANSPORT_UNAVAILABLE in str(err.value)
    # The broker's own typed code is disclosed inside the admission refusal, so a
    # diagnosis can still name which layer said no.
    assert "remote_workspace_unavailable" in str(err.value)
    # Still a ValueError, so every existing `except ValueError` admission call site
    # keeps refusing loudly rather than admitting an unverified placement.
    assert isinstance(err.value, ValueError)


def test_a_retired_connection_leaves_the_trust_index_entirely(tmp_path, monkeypatch):
    """Retired means UNKNOWN to admission, not known-and-untrusted.

    Admission's membership check and its trust comparison read the same index; if a
    retired row stayed in it with a blank identity, a placement fenced against a
    blank identity would match.
    """
    from ouroboros import config, connection_store

    store = tmp_path / "connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    row = connection_store.add_connection(name="Old box", ssh_alias="old", path=store)
    assert connection_store.connection_trust_index(store) == {row["id"]: ""}
    connection_store.retire_connection(row["id"], path=store)
    assert connection_store.connection_trust_index(store) == {}


def test_missing_connection_store_refuses_typed_not_permissively(tmp_path, monkeypatch):
    import ouroboros.workspace_admission as admission

    monkeypatch.setattr(admission, "known_connections", lambda: None)
    with pytest.raises(admission.RemoteWorkspaceUnavailableError) as err:
        admission.validate_workspace_root(
            _ssh_payload(), system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data"
        )
    assert "no remote connection store" in str(err.value)


def test_home_path_projection_refuses_a_remote_placement(tmp_path, monkeypatch):
    """`local_admitted_path` is the ONLY Home-path door for an admitted placement and
    it refuses ssh typed — the gateway's legacy `_resolve_workspace_root` therefore
    cannot hand a remote spelling to a `pathlib` consumer."""
    import ouroboros.workspace_admission as admission
    from ouroboros.workspace_ref import RemoteWorkspacePathError, SshWorkspaceRef

    assert admission.local_admitted_path(None) is None
    with pytest.raises(RemoteWorkspacePathError):
        admission.local_admitted_path(
            SshWorkspaceRef(connection_id="conn-1", remote_root="/srv/work/app", workspace_id="ws-1")
        )

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})
    from ouroboros.gateway.task_placement import _resolve_workspace_root

    with pytest.raises(RemoteWorkspacePathError):
        _resolve_workspace_root(
            _ssh_payload(), system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data"
        )


# --- room resolution for a remote project room -------------------------------

def test_remote_project_room_resolves_through_the_ssh_branch(tmp_path, monkeypatch):
    """C-2:102 resolution — a project whose registry entry carries a PLACEMENT ref
    resolves through the ssh branch; the orchestration stays Home, the validation is
    the target's, and there is no Home `working_dir` stand-in to fall back to."""
    import ouroboros.workspace_admission as admission
    from ouroboros.projects_registry import _load, _save, create_project

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "remote-room", name="Remote", origin="test")
    registry = _load(data)
    for row in registry["projects"]:
        if row.get("id") == "remote-room":
            row["placement"] = _ssh_payload()
    _save(data, registry)

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    monkeypatch.setattr(admission, "remote_session_facts", lambda ref, **_: {"canonical_root": "/srv/work/app"})
    ref, error = admission.resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="remote-room"
    )
    assert error == ""
    assert ref.kind == "ssh" and ref.remote_root == "/srv/work/app"

    # An unverifiable remote room LOUD-FAILS; it never silently becomes workspace-less
    # (which would resolve to the self_modification profile over the system repo).
    monkeypatch.setattr(admission, "known_connections", lambda: None)
    ref_absent, error_absent = admission.resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="remote-room"
    )
    assert ref_absent is None
    assert admission.REMOTE_TRANSPORT_UNAVAILABLE in error_absent
