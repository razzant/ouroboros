from __future__ import annotations

import pathlib
import subprocess

import pytest


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT,
    ).strip()


def _repo(path: pathlib.Path, marker: str) -> pathlib.Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "BIBLE.md").write_text(f"{marker} base\n", encoding="utf-8")
    (path / f"{marker}.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "base")
    return path


def _registry(tmp_path: pathlib.Path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = _repo(tmp_path / "system", "system")
    project = _repo(tmp_path / "project", "project")
    data = tmp_path / "data"
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        drive_root=data,
        workspace_root=project,
        workspace_mode="external",
        task_id="vcs-binding-test",
    )
    registry = ToolRegistry(system, data)
    registry.set_context(ctx)
    return registry, ctx, system, project


def _plain_registry(tmp_path: pathlib.Path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = _repo(tmp_path / "plain-system", "plain-system")
    data = tmp_path / "plain-data"
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        drive_root=data,
        task_id="plain-vcs-binding-test",
    )
    registry = ToolRegistry(system, data)
    registry.set_context(ctx)
    return registry, system


@pytest.fixture(autouse=True)
def _allow_safety(monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))


def test_generic_vcs_schema_has_only_active_or_system_root(tmp_path):
    registry, _ctx, _system, _project = _registry(tmp_path)
    for name in ("vcs_status", "vcs_diff", "vcs_pull_ff", "vcs_restore", "vcs_revert"):
        schema = registry.get_schema_by_name(name)
        root = schema["function"]["parameters"]["properties"]["root"]
        assert root == {
            "type": "string",
            "enum": ["active_workspace", "system_repo"],
            "default": "active_workspace",
            "description": "Omit for the active project workspace; use system_repo for Ouroboros source.",
        }


def test_delegated_child_profiles_keep_their_specialized_vcs_surface(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint

    registry, ctx, _system, _project = _registry(tmp_path)
    ctx.task_constraint = TaskConstraint(
        mode="acting_subagent",
        allow_enable=False,
        surface="external_workspace",
    )
    status_schema = registry.get_schema_by_name("vcs_status")
    assert status_schema["function"]["parameters"]["properties"]["root"]["enum"] == [
        "active_workspace"
    ]
    assert registry.get_schema_by_name("vcs_restore") is None

    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    readonly_status = registry.get_schema_by_name("vcs_status")
    assert readonly_status["function"]["parameters"]["properties"]["root"]["enum"] == [
        "active_workspace", "system_repo"
    ]
    assert registry.get_schema_by_name("vcs_restore") is None


def test_status_and_diff_default_to_project_and_explicit_system_is_distinct(tmp_path):
    registry, _ctx, system, project = _registry(tmp_path)
    (project / "project.txt").write_text("project changed\n", encoding="utf-8")
    (system / "system.txt").write_text("system changed\n", encoding="utf-8")

    project_status = registry.execute("vcs_status", {})
    system_status = registry.execute("vcs_status", {"root": "system_repo"})
    project_diff = registry.execute("vcs_diff", {})
    system_diff = registry.execute("vcs_diff", {"root": "system_repo"})

    assert "project.txt" in project_status and "system.txt" not in project_status
    assert "system.txt" in system_status and "project.txt" not in system_status
    assert "project changed" in project_diff and "system changed" not in project_diff
    assert "system changed" in system_diff and "project changed" not in system_diff
    assert f"root=active_workspace; repo={project}" in project_status
    assert f"root=system_repo; repo={system}" in system_status


def test_restore_protects_names_only_for_explicit_system_root(tmp_path):
    registry, _ctx, system, project = _registry(tmp_path)
    (project / "BIBLE.md").write_text("ordinary project change\n", encoding="utf-8")
    (system / "BIBLE.md").write_text("system change\n", encoding="utf-8")

    project_result = registry.execute(
        "vcs_restore", {"paths": ["BIBLE.md"], "confirm": True},
    )
    system_result = registry.execute(
        "vcs_restore", {"root": "system_repo", "paths": ["BIBLE.md"], "confirm": True},
    )

    assert "RESTORE_BLOCKED" not in project_result
    assert (project / "BIBLE.md").read_text(encoding="utf-8") == "project base\n"
    assert "RESTORE_BLOCKED" in system_result
    assert (system / "BIBLE.md").read_text(encoding="utf-8") == "system change\n"


def test_restore_protects_plain_default_when_active_workspace_is_system(tmp_path):
    registry, system = _plain_registry(tmp_path)
    (system / "BIBLE.md").write_text("system change\n", encoding="utf-8")

    result = registry.execute(
        "vcs_restore", {"paths": ["BIBLE.md"], "confirm": True},
    )

    assert "RESTORE_BLOCKED" in result
    assert (system / "BIBLE.md").read_text(encoding="utf-8") == "system change\n"


def test_revert_protects_names_only_for_explicit_system_root(tmp_path):
    registry, _ctx, system, project = _registry(tmp_path)
    for repo, text in ((system, "system commit\n"), (project, "project commit\n")):
        (repo / "BIBLE.md").write_text(text, encoding="utf-8")
        _git(repo, "add", "BIBLE.md")
        _git(repo, "commit", "-qm", "change Bible")

    system_sha = _git(system, "rev-parse", "HEAD")
    project_sha = _git(project, "rev-parse", "HEAD")
    system_result = registry.execute(
        "vcs_revert", {"root": "system_repo", "sha": system_sha, "confirm": True},
    )
    project_result = registry.execute(
        "vcs_revert", {"sha": project_sha, "confirm": True},
    )

    assert "REVERT_BLOCKED" in system_result
    assert "REVERT_BLOCKED" not in project_result
    assert "New revert commit created" in project_result
    assert (project / "BIBLE.md").read_text(encoding="utf-8") == "project base\n"


def test_revert_protects_plain_default_when_active_workspace_is_system(tmp_path):
    registry, system = _plain_registry(tmp_path)
    (system / "BIBLE.md").write_text("system commit\n", encoding="utf-8")
    _git(system, "add", "BIBLE.md")
    _git(system, "commit", "-qm", "change Bible")
    sha = _git(system, "rev-parse", "HEAD")

    result = registry.execute("vcs_revert", {"sha": sha, "confirm": True})

    assert "REVERT_BLOCKED" in result
    assert (system / "BIBLE.md").read_text(encoding="utf-8") == "system commit\n"


def test_pull_uses_the_same_selected_binding_as_other_generic_vcs(tmp_path, monkeypatch):
    from ouroboros.tools import git_vcs_ops as git_tools

    registry, _ctx, system, project = _registry(tmp_path)
    seen = []
    monkeypatch.setattr(
        git_tools,
        "_ff_pull",
        lambda repo: seen.append(pathlib.Path(repo)) or "pull simulated",
    )

    project_result = registry.execute("vcs_pull_ff", {})
    system_result = registry.execute("vcs_pull_ff", {"root": "system_repo"})

    assert seen == [project, system]
    assert f"root=active_workspace; repo={project}" in project_result
    assert f"root=system_repo; repo={system}" in system_result


def test_light_mode_uses_selected_vcs_target_and_commit_stays_system_intrinsic(
    tmp_path, monkeypatch,
):
    registry, _ctx, system, project = _registry(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    (project / "project.txt").write_text("project changed\n", encoding="utf-8")
    (system / "system.txt").write_text("system changed\n", encoding="utf-8")

    project_preview = registry.execute("vcs_restore", {"confirm": False})
    system_preview = registry.execute(
        "vcs_restore", {"root": "system_repo", "confirm": False},
    )
    system_commit = registry.execute(
        "commit_reviewed", {"commit_message": "must remain system governed"},
    )

    assert "LIGHT_MODE_BLOCKED" not in project_preview
    assert f"root=active_workspace; repo={project}" in project_preview
    assert "LIGHT_MODE_BLOCKED" in system_preview
    assert "LIGHT_MODE_BLOCKED" in system_commit


def test_workspace_focus_exposes_but_does_not_retarget_system_review_lifecycle(tmp_path):
    import inspect

    from ouroboros.tools import claude_advisory_review, git as git_tools

    registry, _ctx, _system, _project = _registry(tmp_path)
    assert registry.get_schema_by_name("advisory_review") is not None
    assert registry.get_schema_by_name("commit_reviewed") is not None
    assert "repo_dir = pathlib.Path(ctx.repo_dir)" in inspect.getsource(
        claude_advisory_review._handle_advisory_pre_review
    )
    assert "ctx.repo_dir" in inspect.getsource(git_tools._repo_commit_push)
