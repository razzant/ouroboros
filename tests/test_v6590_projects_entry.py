"""v6.59.0 (Phase 3) — project entry points: attach/clone sources, provenance,
delete/hide, and the directory-browser confinement.
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


# --- attach validation ----------------------------------------------------------

def test_validate_attach_path_boundaries(tmp_path):
    from ouroboros.project_sources import validate_attach_path

    sys_repo = tmp_path / "sysrepo"
    data = tmp_path / "data"
    sys_repo.mkdir()
    data.mkdir()
    ok_dir = tmp_path / "mycode"
    ok_dir.mkdir()

    resolved, error = validate_attach_path(str(ok_dir), system_repo_dir=sys_repo, drive_root=data)
    assert error == "" and resolved == ok_dir.resolve()

    # Missing path
    _, err_missing = validate_attach_path(str(tmp_path / "nope"), system_repo_dir=sys_repo, drive_root=data)
    assert "does not exist" in err_missing
    # A file, not a directory
    f = tmp_path / "file.txt"
    f.write_text("x")
    _, err_file = validate_attach_path(str(f), system_repo_dir=sys_repo, drive_root=data)
    assert "not a directory" in err_file
    # Home root itself is refused
    _, err_home = validate_attach_path(str(pathlib.Path.home()), system_repo_dir=sys_repo, drive_root=data)
    assert "home directory" in err_home
    # repo/data overlap refused
    _, err_repo = validate_attach_path(str(sys_repo), system_repo_dir=sys_repo, drive_root=data)
    assert "system repo" in err_repo
    _, err_data = validate_attach_path(str(data / "sub"), system_repo_dir=sys_repo, drive_root=data)
    assert err_data  # nonexistent AND under data — either error is a refusal


def test_attach_snapshot_init_is_opt_in_and_idempotent(tmp_path):
    from ouroboros.project_sources import attach_snapshot_init

    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "notes.txt").write_text("hello\n", encoding="utf-8")
    assert attach_snapshot_init(folder) == ("", [])
    log_out = subprocess.run(["git", "log", "-1", "--format=%s %an"], cwd=str(folder), capture_output=True, text=True).stdout
    assert "ouroboros: attach snapshot" in log_out and "Ouroboros" in log_out
    # Idempotent for an existing repo (no second snapshot, no error).
    assert attach_snapshot_init(folder) == ("", [])
    count = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(folder), capture_output=True, text=True).stdout.strip()
    assert count == "1"


# --- clone URL forms + typed errors ----------------------------------------------

def test_valid_git_url_forms():
    from ouroboros.project_sources import derive_repo_dir_name, valid_git_url

    assert valid_git_url("https://github.com/user/repo.git")
    assert valid_git_url("https://github.com/user/repo")
    assert valid_git_url("ssh://git@github.com/user/repo.git")
    assert valid_git_url("git@github.com:user/repo.git")
    assert not valid_git_url("/local/path")
    assert not valid_git_url("ftp://x/y")
    assert not valid_git_url("")
    assert derive_repo_dir_name("https://github.com/user/My-Repo.git") == "My-Repo"
    assert derive_repo_dir_name("git@github.com:user/tool.git") == "tool"


def test_clone_project_repo_local_source_and_atomicity(tmp_path, monkeypatch):
    from ouroboros.project_sources import clone_project_repo

    projects_root = tmp_path / "projects"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    # invalid URL is typed
    _, code, _ = clone_project_repo("not-a-url")
    assert code == "invalid_url"
    # A real clone from a local https-shaped URL isn't possible offline; validate the
    # existing-destination refusal instead (atomic tmp never left behind).
    (projects_root / "taken").mkdir(parents=True)
    _, code2, detail2 = clone_project_repo("https://github.com/user/taken.git", "taken")
    assert code2 == "exists" and "taken" in detail2
    leftovers = [p.name for p in projects_root.iterdir() if ".tmp." in p.name]
    assert leftovers == []


# --- registry: provenance + delete ------------------------------------------------

def test_update_project_provenance_fields_and_tombstone(tmp_path):
    from ouroboros.projects_registry import (
        begin_project_deletion,
        bind_task_to_project,
        complete_project_deletion,
        create_project,
        delete_project,
        get_project,
        get_reserved_project,
        project_binding_for_task,
        update_project,
    )

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "p1", name="P1", origin="owner_ui")
    update_project(data, "p1", provenance="attached", clone_url="", trusted_at="2026-07-09T00:00:00Z")
    entry = get_project(data, "p1")
    assert entry["provenance"] == "attached"
    assert entry["trusted_at"].startswith("2026-")

    bind_task_to_project(data, "t1", "p1", entry.get("chat_id"), origin={"absent": "system"})
    assert project_binding_for_task(data, "t1") is not None

    folder = tmp_path / "keepme"
    folder.mkdir()
    update_project(data, "p1", working_dir=str(folder))
    before_deletion = get_project(data, "p1")["routing_generation"]
    with pytest.raises(RuntimeError, match="cancellation/quiescence"):
        delete_project(data, "p1")
    deleting = begin_project_deletion(data, "p1")
    assert deleting["lifecycle"] == "deleting"
    # The preceding working-dir change also advances the routing generation;
    # deletion must advance from THAT source instead of resetting the count.
    assert deleting["routing_generation"] == before_deletion + 1
    complete_project_deletion(data, "p1")
    assert delete_project(data, "p1") is True  # idempotent compatibility completion
    assert get_project(data, "p1") is None
    assert get_reserved_project(data, "p1")["lifecycle"] == "tombstoned"
    assert project_binding_for_task(data, "t1") is not None
    with pytest.raises(ValueError, match="permanently reserved"):
        create_project(data, "p1", name="Resurrected")
    assert folder.is_dir()  # the folder is NEVER touched by delete


# --- fs/dirs confinement ------------------------------------------------------------

def test_api_fs_dirs_confined_to_home(tmp_path):
    import asyncio
    import json

    from ouroboros.gateway.projects import api_fs_dirs

    class _Req:
        def __init__(self, path=""):
            self.query_params = {"path": path} if path else {}

    # Home itself works, and the honesty marker is present (not silently truncated).
    resp = asyncio.run(api_fs_dirs(_Req()))
    payload = json.loads(resp.body)
    assert payload["path"] == str(pathlib.Path.home().resolve())
    assert payload["parent"] == ""
    assert "truncated" in payload and isinstance(payload["truncated"], bool)
    # Outside-home is refused — and NEVER leaks existence (triad r4: the same
    # confined 400 for an existing and a nonexistent outside path, no 404 oracle).
    resp2 = asyncio.run(api_fs_dirs(_Req("/etc")))
    payload2 = json.loads(resp2.body)
    assert resp2.status_code == 400 and "confined" in payload2["error"]
    resp3 = asyncio.run(api_fs_dirs(_Req("/definitely-not-a-real-dir-xyz")))
    payload3 = json.loads(resp3.body)
    assert resp3.status_code == 400 and "confined" in payload3["error"]
    # Inside-home nonexistent still gets an honest 404.
    resp4 = asyncio.run(api_fs_dirs(_Req(str(pathlib.Path.home() / "no-such-dir-xyz-404"))))
    assert resp4.status_code == 404


@pytest.mark.parametrize("init_git", [False, True])
def test_api_projects_create_attach_git_is_optional(tmp_path, init_git):
    """An attached ordinary folder admits room tasks without creating Git state;
    an explicit init still provisions the existing attach snapshot."""
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.projects_registry import get_project
    from ouroboros.workspace_admission import resolve_room_workspace

    data = tmp_path / "data"
    data.mkdir()
    plain = tmp_path / "plain_folder"
    plain.mkdir()
    (plain / "notes.txt").write_text("owner notes\n", encoding="utf-8")

    class _Req:
        def __init__(self, body):
            self._body = body
            self.app = SimpleNamespace(state=SimpleNamespace(drive_root=data, repo_dir=tmp_path / "repo"))

        async def json(self):
            return self._body

    body = {"name": "Plain", "path": str(plain)}
    if init_git:
        body["init_git"] = True
    resp = asyncio.run(api_projects_create(_Req(body)))
    payload = json.loads(resp.body)
    assert resp.status_code == 200 and payload["project"]["provenance"] == "attached"
    assert get_project(data, "plain")["working_dir"] == str(plain.resolve())
    assert (plain / ".git").exists() is init_git
    assert (plain / "notes.txt").read_text(encoding="utf-8") == "owner notes\n"
    assert resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "repo", project_id="plain",
    ) == (str(plain.resolve()), "")


@pytest.mark.parametrize("init_git", [False, True])
def test_attach_git_subdir_only_initializes_when_explicitly_requested(tmp_path, init_git):
    """Explicit init can make a selected subdirectory a standalone project;
    a plain attach never creates that Git boundary implicitly."""
    import asyncio
    from types import SimpleNamespace

    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.projects_registry import get_project
    from ouroboros.promotion_source import resolve_promote_source
    from ouroboros.workspace_admission import resolve_room_workspace

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "owner-repo"
    _init_git_repo(repo)
    sub = repo / "subdir"
    sub.mkdir()
    (sub / "notes.txt").write_text("owner notes\n", encoding="utf-8")
    parent_head = (repo / ".git" / "HEAD").read_bytes()
    parent_index = (repo / ".git" / "index").read_bytes()
    ctx = SimpleNamespace(DRIVE_ROOT=data, REPO_DIR=tmp_path / "system")

    class _Req:
        app = SimpleNamespace(state=SimpleNamespace(drive_root=data, repo_dir=ctx.REPO_DIR))

        async def json(self):
            return {"id": "subdir", "path": str(sub), "init_git": init_git}

    ws, _, error, _, created = resolve_promote_source(ctx, str(sub), "subdir")
    assert not ws and "git worktree root" in error and not created
    assert get_project(data, "subdir") is None
    assert not (sub / ".git").exists()
    resp = asyncio.run(api_projects_create(_Req()))
    if init_git:
        assert resp.status_code == 200, resp.body
        assert get_project(data, "subdir")["working_dir"] == str(sub.resolve())
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=sub,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert pathlib.Path(top).resolve() == sub.resolve()
        assert resolve_room_workspace(
            drive_root=data, system_repo_dir=ctx.REPO_DIR, project_id="subdir",
        ) == (str(sub.resolve()), "")
        subject = subprocess.run(
            ["git", "log", "-1", "--format=%s"], cwd=sub,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert subject == "ouroboros: attach snapshot"
    else:
        assert resp.status_code == 400
        assert get_project(data, "subdir") is None
        assert not (sub / ".git").exists()
    assert (sub / "notes.txt").read_text(encoding="utf-8") == "owner notes\n"
    assert (repo / ".git" / "HEAD").read_bytes() == parent_head
    assert (repo / ".git" / "index").read_bytes() == parent_index


# --- create: existing id + new source is a 409, checked before any clone -----------

def test_api_projects_create_conflicts_on_existing_id_with_source(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.projects_registry import create_project, get_project

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "taken", name="Taken", origin="owner_ui")

    class _Req:
        def __init__(self, body):
            self._body = body
            self.app = SimpleNamespace(state=SimpleNamespace(drive_root=data, repo_dir=tmp_path / "repo"))

        async def json(self):
            return self._body

    # Re-sourcing an existing id must 409 BEFORE any clone/attach side effect.
    called = {"clone": 0}

    def _no_clone(*a, **k):  # pragma: no cover - the guard must prevent this call
        called["clone"] += 1
        raise AssertionError("clone must not run for a conflicting id")

    monkeypatch.setattr("ouroboros.project_sources.clone_project_repo", _no_clone)
    resp = asyncio.run(api_projects_create(_Req({"id": "taken", "git_url": "https://example.com/x.git"})))
    payload = json.loads(resp.body)
    assert resp.status_code == 409
    assert payload["error_code"] == "project_exists"
    assert called["clone"] == 0
    # The registry row is untouched.
    entry = get_project(data, "taken")
    assert entry and str(entry.get("working_dir") or "") == ""
    # A source-less create for the same id stays idempotent (no conflict) AND
    # preserves the historical provenance facts (triad r1: the unconditional
    # trailing stamp used to relabel an attached project provenance=none).
    # (see also test_promote_source_registers_derived_project_and_mirrors_conflict
    # for the agent-side sibling rules)
    from ouroboros.projects_registry import update_project

    folder = tmp_path / "attached_folder"
    folder.mkdir()
    update_project(
        data, "taken", working_dir=str(folder), provenance="attached",
        clone_url="", trusted_at="2026-07-09T00:00:00Z",
    )
    resp2 = asyncio.run(api_projects_create(_Req({"id": "taken"})))
    assert resp2.status_code == 200
    entry2 = get_project(data, "taken")
    assert entry2["provenance"] == "attached"
    assert entry2["trusted_at"] == "2026-07-09T00:00:00Z"
    assert entry2["working_dir"] == str(folder)


# --- promote(source=): derived project id + conflict mirror (triad r2) --------------

def test_promote_source_registers_derived_project_and_mirrors_conflict(tmp_path, monkeypatch):
    """The one-liner `promote_chat_to_task(source=…)` must register a project even
    with NO project_id/name (derived from the source name), stay idempotent on a
    same-folder re-attach (trusted_at preserved), and refuse re-sourcing an existing
    project whose folder differs — mirroring the gateway 409 rule."""
    from types import SimpleNamespace

    import ouroboros.config as config
    from ouroboros.projects_registry import get_project as get_reg_project
    from ouroboros.promotion_source import resolve_promote_source

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    folder = tmp_path / "myrepo"
    folder.mkdir()
    (folder / "x.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=str(folder), check=True)
    ctx = SimpleNamespace(repo_dir=str(tmp_path / "repo"))

    # An ordinary folder registers and admits without an implicit git init.
    nogit = tmp_path / "plain_folder"
    nogit.mkdir()
    ws0, _, err0, pid0, created0 = resolve_promote_source(ctx, str(nogit), "")
    assert ws0 == str(nogit.resolve()) and err0 == "" and created0
    assert get_reg_project(data, pid0)["working_dir"] == ws0
    assert not (nogit / ".git").exists()

    # No pid given: derived from the folder name, registered with provenance facts.
    ws, note, err, pid, created = resolve_promote_source(ctx, str(folder), "")
    assert err == "" and pid == "myrepo" and ws
    assert created is True  # B1: fresh registration reports the creation fact
    entry = get_reg_project(data, "myrepo")
    assert entry is not None
    assert entry["provenance"] == "attached"
    assert entry["working_dir"] == ws
    trusted_first = entry["trusted_at"]
    assert trusted_first

    # Same folder again: idempotent, original trusted_at preserved.
    ws2, _, err2, pid2, created2 = resolve_promote_source(ctx, str(folder), "myrepo")
    assert err2 == "" and pid2 == "myrepo" and ws2 == ws
    assert created2 is False  # idempotent re-attach: no second creation fact
    assert get_reg_project(data, "myrepo")["trusted_at"] == trusted_first

    # A DIFFERENT folder for the same project id: refused (registry must not lie).
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(other), check=True)
    ws3, _, err3, _, _ = resolve_promote_source(ctx, str(other), "myrepo")
    assert ws3 == "" and "conflict" in err3
    assert get_reg_project(data, "myrepo")["working_dir"] == ws

    # A git-URL source for an already-bound project id: conflict BEFORE the clone
    # side effect — clone_project_repo must never run (triad r7: no dangling clone
    # behind a refusal).
    import ouroboros.tools.control as control_mod

    def _never_clone(*a, **k):  # pragma: no cover - the guard must prevent this
        raise AssertionError("clone_project_repo must not run on conflict")

    monkeypatch.setattr(control_mod, "clone_project_repo", _never_clone, raising=False)
    monkeypatch.setattr(
        "ouroboros.project_sources.clone_project_repo", _never_clone
    )
    ws4, _, err4, _, _ = resolve_promote_source(ctx, "https://example.com/myrepo.git", "myrepo")
    assert ws4 == "" and "conflict" in err4


def test_promote_with_a_source_keeps_the_display_name_the_model_gave(tmp_path, monkeypatch):
    """A sourced promotion registers the row in its off-loop half, BEFORE the named create.

    ``create_project`` returns an existing row untouched, so the name must reach the first
    call: withheld, a non-Latin name (whose id is ``proj_<digest>``) showed the id everywhere.
    """
    from types import SimpleNamespace

    import ouroboros.config as config
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.projects_registry import PROJECT_NAME_MAX, create_project, get_project
    from ouroboros.promotion_source import resolve_promote_source

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    ctx = SimpleNamespace(repo_dir=str(tmp_path / "repo"))
    name = "Переключатели субагентов"
    pid = project_id_from_display_name(name)
    assert pid.startswith("proj_")  # no Latin slug: the id is a digest, the NAME is what people read
    folder = tmp_path / "work"
    folder.mkdir()

    _, _, err, got, created = resolve_promote_source(ctx, str(folder), pid, project_name=name)
    assert err == "" and got == pid and created is True
    assert get_project(data, pid)["name"] == name
    # The worker-side named create that follows finds the row and changes nothing.
    assert create_project(data, pid, name=name, origin="promote_chat_to_task")["created"] is False
    assert get_project(data, pid)["name"] == name

    # No name given stays what it was: the row is named by its id.
    plain = tmp_path / "plain"
    plain.mkdir()
    _, _, err2, pid2, _ = resolve_promote_source(ctx, str(plain), "plain-project")
    assert err2 == "" and get_project(data, pid2)["name"] == "plain-project"

    # A title over the registry limit never turns a finished clone/attach into a refusal.
    longer = tmp_path / "longer"
    longer.mkdir()
    _, _, err3, pid3, created3 = resolve_promote_source(
        ctx, str(longer), "long-title", project_name="x" * (PROJECT_NAME_MAX + 1))
    assert err3 == "" and created3 is True and get_project(data, pid3)["name"] == "long-title"
