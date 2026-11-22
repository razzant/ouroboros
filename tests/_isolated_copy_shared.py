"""A self-worktree test's admitted copy and its parent context."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolContext
from ouroboros.workspace_copies import admitted_copy_metadata
from supervisor.events_subagent_admission import _resolve_subagent_constraint
from tests._workspace_git import seeded_repository as repo


@pytest.fixture
def case(tmp_path, monkeypatch):
    system = repo(tmp_path / "system", {"system.txt": b"own body\n"})
    source = repo(tmp_path / "project", {"a.txt": b"base\n", "BIBLE.md": b"project docs\n"})
    data, copies = tmp_path / "data", tmp_path / "copies"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("OUROBOROS_BOOT_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "")
    return system, source, data, copies


def admit(case, task_id, parent_id, source=None):
    system, project, data, _ = case
    source = source or project
    constraint, folder, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=system, DRIVE_ROOT=data), tid=task_id,
        requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root=str(source), workspace_mode="external", base_sha="", parent_task_id=parent_id)
    assert not error, error
    binding = admitted_copy_metadata(folder)
    task = {"id": task_id, "parent_task_id": parent_id,
            "workspace_root": folder, "workspace_mode": mode,
            "task_constraint": constraint, "workspace_copy": binding,
            "metadata": {"workspace_copy": binding}}
    return Path(folder), task


def parent_context(case, task_id="parent", source=None, task=None):
    system, project, data, _ = case
    return ToolContext(repo_dir=system, drive_root=data, task_id=task_id,
                       workspace_root=source or project,
                       workspace_mode=task["workspace_mode"] if task else "external",
                       task_constraint=task["task_constraint"] if task else None,
                       task_metadata=task["metadata"] if task else {})
