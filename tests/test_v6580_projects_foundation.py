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
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@local",
            "-c",
            "maintenance.auto=false",
            "-c",
            "gc.auto=0",
            "commit",
            "-qm",
            "init",
        ],
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
    assert payload.get("_schema_version") == 2
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


@pytest.mark.parametrize("kind", ["ordinary", "dirty", "unborn", "linked"])
def test_workspace_admission_preserves_existing_folder_state(tmp_path, kind):
    from ouroboros.workspace_admission import validate_workspace_root

    ws = tmp_path / "workspace"
    if kind == "linked":
        source = tmp_path / "source"
        _init_git_repo(source)
        subprocess.run(["git", "worktree", "add", "--detach", str(ws)], cwd=source, check=True)
    elif kind == "dirty":
        _init_git_repo(ws)
        (ws / "README.md").write_text("owner edits\n", encoding="utf-8")
        (ws / ".gitignore").write_text("ignored\n", encoding="utf-8")
        (ws / "ignored").write_bytes(b"owner ignored bytes")
    else:
        ws.mkdir()
        if kind == "unborn":
            subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    (ws / "notes.txt").write_text("existing notes\n", encoding="utf-8")
    before = {p.relative_to(ws): p.read_bytes() for p in ws.rglob("*") if p.is_file()}

    assert validate_workspace_root(
        ws, system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data",
    ) == ws.resolve()
    assert {p.relative_to(ws): p.read_bytes() for p in ws.rglob("*") if p.is_file()} == before


def test_workspace_admission_plain_folder_does_not_require_git_binary(tmp_path, monkeypatch):
    from ouroboros.workspace_admission import validate_workspace_root

    ws = tmp_path / "documents"
    ws.mkdir()

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("ouroboros.workspace_admission.subprocess.run", missing_git)
    assert validate_workspace_root(
        ws, system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data",
    ) == ws.resolve()


@pytest.mark.parametrize("kind", ["broken_linked", "bare"])
def test_workspace_admission_does_not_disguise_invalid_git_as_plain(tmp_path, kind):
    from ouroboros.workspace_admission import WorkspaceRootError, validate_workspace_root

    ws = tmp_path / "workspace"
    ws.mkdir()
    if kind == "bare":
        subprocess.run(["git", "init", "--bare", "-q"], cwd=ws, check=True)
    else:
        (ws / ".git").write_text("gitdir: /missing/worktree/gitdir\n", encoding="utf-8")
    with pytest.raises(WorkspaceRootError):
        validate_workspace_root(ws, system_repo_dir=tmp_path / "sys", drive_root=tmp_path / "data")


@pytest.mark.parametrize("use_git", [False, True])
def test_resolve_room_workspace_defaults_to_project_working_dir(tmp_path, use_git):
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.workspace_admission import resolve_room_workspace

    data = tmp_path / "data"
    data.mkdir()
    ws = tmp_path / "roomdir"
    if use_git:
        _init_git_repo(ws)
    else:
        ws.mkdir()
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
        # Git maintenance may remove a transient lock between rmtree's failed
        # unlink and this callback; an already-absent path is cleanup success.
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except FileNotFoundError:
            pass

    shutil.rmtree(gone, onerror=_chmod_and_retry)  # the folder disappears after registration

    resolved, error = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="broken"
    )
    assert resolved == ""
    assert "unusable" in error and "broken" in error


def test_resolve_room_workspace_loud_fails_when_registry_is_unreadable(tmp_path, monkeypatch):
    """An UNREADABLE registry is not the same fact as "no working_dir".

    Swallowing the read error returned ("", "") — indistinguishable from a file-less
    project — so admission continued and the task ran workspace-less on the
    self_modification profile over the SYSTEM repo. That is the same silent
    degradation the v6.58.0 loud-fail invariant exists to kill."""
    import ouroboros.projects_registry as projects_registry
    from ouroboros.workspace_admission import resolve_room_workspace

    data = tmp_path / "data"
    data.mkdir()

    def _boom(*_a, **_kw):
        raise OSError("registry unreadable")

    monkeypatch.setattr(projects_registry, "get_project", _boom)

    resolved, error = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="room"
    )
    assert resolved == ""
    assert error, "an unreadable registry must NOT resolve to a silent workspace-less task"
    assert "unreadable" in error and "room" in error

    # An explicit workspace_root never consults the registry, so it is unaffected.
    ws = tmp_path / "explicit"
    _init_git_repo(ws)
    resolved_x, error_x = resolve_room_workspace(
        drive_root=data, system_repo_dir=tmp_path / "sys", project_id="room",
        explicit_workspace=str(ws),
    )
    assert error_x == "" and resolved_x == str(ws.resolve())


# --- canonical source-ref truncation guard -------------------------------------

def test_owner_request_source_ref_gets_full_hint_not_60_chars(tmp_path):
    """Source-ref lookup sees the full hint; only the name candidate is capped."""
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
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@local",
            "-c",
            "maintenance.auto=false",
            "-c",
            "gc.auto=0",
            "commit",
            "-qm",
            "child work",
        ],
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


def test_checkpoint_ignores_readonly_children_and_bare_workspace_roots(tmp_path, monkeypatch):
    """Delegation-usefulness fix (owner 2026-08-30): a READ-ONLY child's
    workspace_root must never qualify a coop tree for the checkpoint commit -
    a research task auto-committed the owner's pre-existing dirty tree it never
    wrote to. Mutative authority (constraint write_root, non-readonly mode) is
    the only qualifier."""
    from ouroboros.coop_checkpoint import checkpoint_commit_coop_roots
    from ouroboros.task_results import write_task_result

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    data = tmp_path / "data"
    data.mkdir()

    tree = projects_root / "coop_ro"
    _init_git_repo(tree)
    # Pre-existing dirt the readonly child had nothing to do with.
    (tree / "owner_wip.txt").write_text("uncommitted owner work\n", encoding="utf-8")

    # 1) A readonly child that merely OBSERVED the tree (workspace_root, no
    #    write_root, local_readonly mode) never qualifies it.
    write_task_result(
        data, "child-ro", "completed",
        delegation_role="subagent", parent_task_id="root-ro", root_task_id="root-ro",
        workspace_root=str(tree),
        task_constraint={"mode": "local_readonly_subagent", "surface": "external_workspace"},
    )
    assert checkpoint_commit_coop_roots(data, "root-ro") == []
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(tree),
                            capture_output=True, text=True).stdout
    assert "owner_wip.txt" in status, "the dirty tree stays exactly as the owner left it"

    # 2) A mutative child qualifying via bare workspace_root only (no granted
    #    write_root) does not qualify either: authority, not presence.
    write_task_result(
        data, "child-ws", "completed",
        delegation_role="subagent", parent_task_id="root-ro", root_task_id="root-ro",
        workspace_root=str(tree),
        task_constraint={"mode": "acting_subagent", "surface": "external_workspace"},
    )
    assert checkpoint_commit_coop_roots(data, "root-ro") == []

    # 3) The readonly PREDICATE itself must skip: a readonly child whose
    #    constraint somehow carries a write_root still never qualifies.
    write_task_result(
        data, "child-ro-wr", "completed",
        delegation_role="subagent", parent_task_id="root-ro", root_task_id="root-ro",
        workspace_root=str(tree),
        task_constraint={"mode": "local_readonly_subagent",
                         "surface": "external_workspace", "write_root": str(tree)},
    )
    assert checkpoint_commit_coop_roots(data, "root-ro") == []

    # 4) Positive control: a MUTATIVE child granted write_root still gets its
    #    coop tree checkpointed - the fix narrows attribution, not capability.
    write_task_result(
        data, "child-mut", "completed",
        delegation_role="subagent", parent_task_id="root-ro", root_task_id="root-ro",
        workspace_root=str(tree),
        task_constraint={"mode": "acting_subagent",
                         "surface": "external_workspace", "write_root": str(tree)},
    )
    committed = checkpoint_commit_coop_roots(data, "root-ro")
    assert [row["root"] for row in committed] == [str(tree)]
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(tree),
                            capture_output=True, text=True).stdout
    assert "owner_wip.txt" not in status, "the granted tree got its checkpoint commit"


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
    """Coop success now uses the same target/read admission as ordinary B."""
    from hashlib import sha256
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    tree, patch_path = _coop_tree_with_child_work(projects_root)
    drive = tmp_path / "state"; drive.mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "sys", drive_root=drive, task_id="parent1")
    art = task_artifact_dir_path(drive, "childX", create=True)
    patch = patch_path.read_bytes()
    (art / "workspace.patch").write_bytes(patch)
    (art / "workspace_patch.json").write_text(json.dumps({
        "status": "ready_with_changes", "workspace_root": str(tree),
        "sha256": sha256(patch).hexdigest(), "tracked_changed": ["app.py"]}))
    write_task_result(drive, "childX", "completed", parent_task_id="parent1", root_task_id="parent1",
                      delegation_role="subagent", workspace_root=str(tree),
                      task_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(tree)})
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=drive); registry.set_context(ctx)
    result = registry.execute_result("integrate_subagent_patch", {"task_id": "childX"}).text
    assert result.startswith("OK: cooperative no-op"), result
    assert "ALREADY in" in result
    assert "TARGET_MISMATCH" in registry.execute_result(
        "integrate_subagent_patch", {"task_id": "childX", "target_root": str(tmp_path / "outside")}).text
