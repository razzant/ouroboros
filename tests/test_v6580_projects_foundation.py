"""v6.58.0 (Phase 2) — projects foundation: registry-first identity, admission SSOT,
room-workspace wiring with the loud-fail invariant, the §3.4 mirror truncation fix,
and the coop no-op / checkpoint-commit pair.
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


# --- 2.1 registry-first identity ----------------------------------------------

def test_resolve_project_id_registry_first(tmp_path, monkeypatch):
    """A workspace task whose folder IS a registered project's working_dir resolves
    to the REGISTRY id (one folder = one identity/lease/store), not a proj_<hash>."""
    import ouroboros.config as cfg
    from ouroboros.project_facts import resolve_project_id
    from ouroboros.projects_registry import create_project, update_project

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data, raising=True)
    ws = tmp_path / "mysite"
    _init_git_repo(ws)

    create_project(data, "mysite", name="My Site", origin="test")
    update_project(data, "mysite", working_dir=str(ws))

    resolved = resolve_project_id({"workspace_root": str(ws)})
    assert resolved == "mysite"

    # An UNregistered folder still derives the stable hash id.
    other = tmp_path / "other"
    _init_git_repo(other)
    derived = resolve_project_id({"workspace_root": str(other)})
    assert derived.startswith("proj_") and len(derived) == len("proj_") + 12

    # Explicit project_id always wins.
    assert resolve_project_id({"project_id": "explicit", "workspace_root": str(ws)}) == "explicit"


def test_projects_registry_stamps_schema_version(tmp_path):
    import json

    from ouroboros.projects_registry import create_project

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "p1", name="P1", origin="test")
    payload = json.loads((data / "state" / "projects.json").read_text(encoding="utf-8"))
    assert payload.get("_schema_version") == 1
    assert any(p.get("id") == "p1" for p in payload.get("projects", []))


# --- 2.2 admission SSOT + loud-fail -------------------------------------------

def test_validate_workspace_root_is_shared_ssot(tmp_path):
    from ouroboros.workspace_admission import WorkspaceRootError, validate_workspace_root

    ws = tmp_path / "repo"
    _init_git_repo(ws)
    resolved = validate_workspace_root(str(ws), system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data")
    assert resolved == ws.resolve()

    with pytest.raises(WorkspaceRootError):
        validate_workspace_root(str(tmp_path / "missing"), system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data")
    # A subdir of a git tree is rejected (must be the worktree ROOT).
    sub = ws / "src"
    sub.mkdir()
    with pytest.raises(WorkspaceRootError):
        validate_workspace_root(str(sub), system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data")


def test_resolve_room_workspace_defaults_to_project_working_dir(tmp_path):
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.workspace_admission import resolve_room_workspace

    data = tmp_path / "data"
    data.mkdir()
    ws = tmp_path / "roomdir"
    _init_git_repo(ws)
    create_project(data, "room", name="Room", origin="test")
    update_project(data, "room", working_dir=str(ws))

    resolved, error = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="room"
    )
    assert error == ""
    assert resolved == str(ws.resolve())

    # workspace="none" opts out even when the room has a working_dir.
    resolved_none, error_none = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="room",
        workspace_sentinel="none",
    )
    assert (resolved_none, error_none) == ("", "")

    # A file-less project admits a workspace-less task with NO error.
    create_project(data, "fileless", name="Fileless", origin="test")
    resolved_fl, error_fl = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="fileless"
    )
    assert (resolved_fl, error_fl) == ("", "")


def test_resolve_room_workspace_loud_fails_on_broken_working_dir(tmp_path):
    """THE loud-fail invariant: a room task with a SET-but-broken working_dir must
    surface an error — never silently admit a workspace-less (self_modification-
    profile) task over the system repo."""
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.workspace_admission import resolve_room_workspace

    data = tmp_path / "data"
    data.mkdir()
    gone = tmp_path / "deleted-folder"
    _init_git_repo(gone)
    create_project(data, "broken", name="Broken", origin="test")
    update_project(data, "broken", working_dir=str(gone))
    import os
    import shutil
    import stat

    def _chmod_and_retry(func, path, _exc):
        # Windows: git object files are read-only; plain rmtree hits WinError 5.
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(gone, onerror=_chmod_and_retry)  # the folder disappears after registration

    resolved, error = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="broken"
    )
    assert resolved == ""
    assert "unusable" in error and "broken" in error


# --- §3.4 mirror truncation fix -------------------------------------------------

def test_owner_request_mirror_gets_full_hint_not_60_chars(tmp_path):
    """The frontend objective_hint reaches the chat MIRROR untruncated; only the NAME
    candidate is capped at 60 chars (the 'Сделай html сайтик … в…' incident)."""
    from ouroboros.gateway.projects import _owner_request_text

    long_ask = "Сделай html сайтик где опишешь кратко человеческим языком в чем суть проекта и как он работает " + "деталь " * 30
    # No persisted/live task -> the hint IS the owner text; it must come back whole.
    got = _owner_request_text(tmp_path, "no-such-task", " ".join(long_ask.split()))
    assert got == " ".join(long_ask.split())
    assert len(got) > 200 and "…" not in got


# --- 2.4 coop no-op + checkpoint-commit ----------------------------------------

def _coop_tree_with_child_work(projects_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A host-minted-style coop tree with a child's committed base and a patch file
    representing the child's work that is ALREADY in the tree."""
    tree = projects_root / "coop_abc"
    _init_git_repo(tree)
    (tree / "app.py").write_text("print('v1')\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(tree), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "child work"],
        cwd=str(tree), check=True,
    )
    patch = subprocess.run(
        ["git", "format-patch", "--stdout", "HEAD~1..HEAD"],
        cwd=str(tree), capture_output=True, text=True, check=True,
    ).stdout
    patch_path = projects_root / "child.patch"
    # Normalize to a plain diff (git apply accepts format-patch output too).
    patch_path.write_text(patch, encoding="utf-8")
    return tree, patch_path


def test_checkpoint_commit_coop_roots_commits_dirty_tree_and_skips_secrets(tmp_path, monkeypatch):
    from ouroboros.coop_checkpoint import checkpoint_commit_coop_roots
    from ouroboros.task_results import write_task_result

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    data = tmp_path / "data"
    data.mkdir()

    tree = projects_root / "coop_root1"
    _init_git_repo(tree)
    # Child result records the coop tree as its write_root.
    write_task_result(
        data, "child1", "completed",
        delegation_role="subagent", parent_task_id="root1", root_task_id="root1",
        task_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(tree)},
    )
    # Dirty tree: one normal file + one credential-shaped file.
    (tree / "feature.txt").write_text("new work\n", encoding="utf-8")
    (tree / ".env").write_text("SECRET=x\n", encoding="utf-8")

    receipts = checkpoint_commit_coop_roots(data, "root1", title="Site build")
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["committed"] is True and receipt.get("sha")
    assert any(s["path"] == ".env" for s in receipt["skipped_sensitive"])
    # The commit exists with the expected message; .env stayed uncommitted.
    # encoding pinned: Windows decodes subprocess text with the ANSI code page,
    # mangling the em-dash in the commit subject.
    log_out = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=str(tree),
        capture_output=True, text=True, encoding="utf-8",
    ).stdout
    assert "ouroboros: checkpoint after task root1 — Site build" in log_out
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(tree), capture_output=True, text=True).stdout
    assert ".env" in status  # still dirty/untracked — never baked into history

    # Live tree tasks -> the checkpoint is skipped entirely.
    (tree / "more.txt").write_text("y\n", encoding="utf-8")
    assert checkpoint_commit_coop_roots(data, "root1", has_live_tree_tasks=True) == []


def test_checkpoint_never_touches_attached_folders(tmp_path, monkeypatch):
    """An owner-attached folder (outside the subagent-projects root) is NEVER
    auto-committed, even when a child recorded it as write_root."""
    from ouroboros.coop_checkpoint import checkpoint_commit_coop_roots
    from ouroboros.task_results import write_task_result

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    data = tmp_path / "data"
    data.mkdir()
    attached = tmp_path / "owners-own-repo"
    _init_git_repo(attached)
    write_task_result(
        data, "child2", "completed",
        delegation_role="subagent", parent_task_id="root2", root_task_id="root2",
        workspace_root=str(attached),
    )
    (attached / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    assert checkpoint_commit_coop_roots(data, "root2") == []
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(attached), capture_output=True, text=True).stdout
    assert "dirty.txt" in status  # untouched


def test_coop_noop_verdict_for_non_workspace_parent(tmp_path, monkeypatch):
    """2.4A: a NON-workspace parent integrating a coop child gets a SUCCESSFUL no-op
    (work already in the host-minted tree), not a parent-missing error."""
    from types import SimpleNamespace

    from ouroboros.tools.subagent_integration import _maybe_coop_noop_verdict

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    tree, patch_path = _coop_tree_with_child_work(projects_root)

    drive = tmp_path / "data"
    drive.mkdir()
    ctx = SimpleNamespace(
        repo_dir=str(tmp_path / "sys"), drive_root=drive, task_id="parent1",
        workspace_mode="", workspace_root=None, task_metadata={},
    )
    result = _maybe_coop_noop_verdict(
        ctx,
        child_task_id="childX",
        reason="",
        patch_path=patch_path,
        manifest={"sha256": "abc"},
        child_result={"task_constraint": {"write_root": str(tree)}},
        touched=["app.py"],
    )
    assert result.startswith("OK: cooperative no-op")
    assert "ALREADY in" in result
    # Not the coop case (a path outside the projects root) -> empty (falls through).
    outside = tmp_path / "outside"
    _init_git_repo(outside)
    assert _maybe_coop_noop_verdict(
        ctx, child_task_id="childY", reason="", patch_path=patch_path,
        manifest={}, child_result={"task_constraint": {"write_root": str(outside)}},
        touched=["app.py"],
    ) == ""
