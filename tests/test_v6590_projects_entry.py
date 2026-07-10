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


def test_attach_snapshot_init_excludes_credential_shaped_files(tmp_path):
    """Triad r4 security critical: an attach snapshot must never bake `.env`/keys
    into git history. Credential-shaped files are unstaged (same SSOT classifier
    as workspace patch / coop checkpoint), disclosed in the returned list, and
    kept untracked via .git/info/exclude — the owner's files are never edited."""
    from ouroboros.project_sources import attach_snapshot_init

    folder = tmp_path / "with_secrets"
    folder.mkdir()
    (folder / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (folder / ".env").write_text("API_KEY=hunter2\n", encoding="utf-8")
    (folder / "deploy.pem").write_text("PRIVATE KEY\n", encoding="utf-8")
    error, skipped = attach_snapshot_init(folder)
    assert error == ""
    assert sorted(skipped) == [".env", "deploy.pem"]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(folder), capture_output=True, text=True
    ).stdout.split()
    assert "app.py" in tracked
    assert ".env" not in tracked and "deploy.pem" not in tracked
    # The secret files still EXIST on disk, untouched.
    assert (folder / ".env").read_text(encoding="utf-8") == "API_KEY=hunter2\n"
    # And stay untracked (info/exclude), so later commits don't sweep them either.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(folder), capture_output=True, text=True
    ).stdout
    assert ".env" not in status and "deploy.pem" not in status


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

def test_update_project_provenance_fields_and_delete(tmp_path):
    from ouroboros.projects_registry import (
        bind_task_to_project,
        create_project,
        delete_project,
        get_project,
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

    bind_task_to_project(data, "t1", "p1", entry.get("chat_id"))
    assert project_binding_for_task(data, "t1") is not None

    folder = tmp_path / "keepme"
    folder.mkdir()
    update_project(data, "p1", working_dir=str(folder))
    assert delete_project(data, "p1") is True
    assert get_project(data, "p1") is None
    assert project_binding_for_task(data, "t1") is None
    assert folder.is_dir()  # the folder is NEVER touched by delete


def test_projects_summary_shows_owner_ui_projects_without_activity(tmp_path):
    from ouroboros.projects_registry import create_project, projects_summary

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "fresh", name="Fresh", origin="owner_ui")
    rows = projects_summary(data)
    fresh = next(r for r in rows if r["id"] == "fresh")
    assert fresh["has_thread_activity"] is True  # owner-created is always visible
    assert "provenance" in fresh


# --- ui preferences: project_hidden ------------------------------------------------

def test_ui_preferences_project_hidden_merge_and_unhide():
    from ouroboros.gateway.ui_preferences import _normalize_preferences

    prefs = _normalize_preferences({"project_hidden": {"a": True, "b": False}}, fill_defaults=False)
    assert prefs["project_hidden"] == {"a": True, "b": False}
    with pytest.raises(ValueError):
        _normalize_preferences({"project_hidden": ["a"]}, fill_defaults=False)


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


def test_api_projects_create_attach_requires_git_unless_init(tmp_path, monkeypatch):
    """Triad r5: task admission requires a git worktree root, so attaching a
    non-git folder WITHOUT init_git must refuse (actionable 400) BEFORE any
    registry mutation — never a project whose room tasks are born dead."""
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    plain = tmp_path / "plain_folder"
    plain.mkdir()

    class _Req:
        def __init__(self, body):
            self._body = body
            self.app = SimpleNamespace(state=SimpleNamespace(drive_root=data, repo_dir=tmp_path / "repo"))

        async def json(self):
            return self._body

    resp = asyncio.run(api_projects_create(_Req({"name": "Plain", "path": str(plain)})))
    payload = json.loads(resp.body)
    assert resp.status_code == 400
    assert payload["error_code"] == "attach_requires_git"
    assert get_project(data, "plain") is None  # no registry mutation
    # With init_git the same folder attaches (snapshot-init makes it a git root).
    resp2 = asyncio.run(api_projects_create(_Req({"name": "Plain", "path": str(plain), "init_git": True})))
    payload2 = json.loads(resp2.body)
    assert resp2.status_code == 200 and payload2["project"]["provenance"] == "attached"
    assert (plain / ".git").exists()


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
    from ouroboros.tools.control import _resolve_promote_source

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    folder = tmp_path / "myrepo"
    folder.mkdir()
    (folder / "x.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=str(folder), check=True)
    ctx = SimpleNamespace(repo_dir=str(tmp_path / "repo"))

    # A NON-git folder is refused with an actionable error (triad r5: task
    # admission requires a git worktree root — no born-dead project rooms).
    nogit = tmp_path / "plain_folder"
    nogit.mkdir()
    ws0, _, err0, _ = _resolve_promote_source(ctx, str(nogit), "")
    assert ws0 == "" and "not a git repository" in err0

    # No pid given: derived from the folder name, registered with provenance facts.
    ws, note, err, pid = _resolve_promote_source(ctx, str(folder), "")
    assert err == "" and pid == "myrepo" and ws
    entry = get_reg_project(data, "myrepo")
    assert entry is not None
    assert entry["provenance"] == "attached"
    assert entry["working_dir"] == ws
    trusted_first = entry["trusted_at"]
    assert trusted_first

    # Same folder again: idempotent, original trusted_at preserved.
    ws2, _, err2, pid2 = _resolve_promote_source(ctx, str(folder), "myrepo")
    assert err2 == "" and pid2 == "myrepo" and ws2 == ws
    assert get_reg_project(data, "myrepo")["trusted_at"] == trusted_first

    # A DIFFERENT folder for the same project id: refused (registry must not lie).
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(other), check=True)
    ws3, _, err3, _ = _resolve_promote_source(ctx, str(other), "myrepo")
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
    ws4, _, err4, _ = _resolve_promote_source(ctx, "https://example.com/myrepo.git", "myrepo")
    assert ws4 == "" and "conflict" in err4
