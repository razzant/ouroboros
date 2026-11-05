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


# --- 2.2 admission SSOT + loud-fail -------------------------------------------


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


# --- 2.4 coop no-op + checkpoint-commit ----------------------------------------


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
