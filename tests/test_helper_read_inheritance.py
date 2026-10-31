"""Parent-equivalent reads survive different starting folders and nested drives."""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.task_results import write_task_result
from ouroboros.tool_access import build_resolved_resource_binding, decide_tool_access
from ouroboros.tool_access_reads import capture_parent_workspace
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def geometry(tmp_path, monkeypatch):
    home, repo, data, first = [tmp_path / name for name in ("home", "repo", "data", "off-home")]
    for path in (home, repo, data, first):
        path.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    return home, repo, data, first


def child(geometry, task_id, parent_id, workspace, parent_workspace):
    _home, repo, data, _first = geometry
    drive = data / "state" / "headless_tasks" / task_id / "data"
    drive.mkdir(parents=True)
    workspace.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=task_id,
                      workspace_root=workspace, workspace_mode="read_only",
                      budget_drive_root=str(data), task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    ctx.task_metadata = {"parent_task_id": parent_id, "root_task_id": "root",
                         "delegation_role": "subagent", "budget_drive_root": str(data),
                         "workspace_root": str(workspace), "workspace_mode": "read_only",
                         "parent_workspace": parent_workspace}
    write_task_result(data, task_id, "running", **ctx.task_metadata,
                      drive_root=str(drive), child_drive_root=str(drive))
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)
    return registry, ctx


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_deep_child_keeps_every_ancestor_folder_and_canonical_runtime(geometry, monkeypatch, mode):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    home, repo, data, first = geometry
    root = ToolContext(repo_dir=repo, drive_root=data, task_id="root",
                       workspace_root=first, workspace_mode="external")
    _, middle = child(geometry, "middle", "root", home / "middle", capture_parent_workspace(root))
    _, parent = child(geometry, "parent", "middle", home / "parent", capture_parent_workspace(middle))
    registry, ctx = child(geometry, "child", "parent", home / "child", capture_parent_workspace(parent))
    for workspace in (first, middle.workspace_root, parent.workspace_root, ctx.workspace_root):
        target = workspace / "source.txt"
        target.write_text(f"SOURCE {workspace.name}", encoding="utf-8")
        for args in ({"path": str(target)}, {"root": "active_workspace", "path": str(target)}):
            out = registry.execute("read_file", args)
            assert f"SOURCE {workspace.name}" in out, out
            assert ctx.last_read_view["target"] == str(target)
        binding = build_resolved_resource_binding(ctx, root="active_workspace", operation="read", path=str(target))
        assert binding.base_path == workspace
    canonical = data / "shared.txt"
    canonical.write_text("CANONICAL RUNTIME", encoding="utf-8")
    assert "CANONICAL RUNTIME" in registry.execute("read_file", {"path": str(canonical)})
    binding = build_resolved_resource_binding(ctx, root="runtime_data", operation="read", path=str(canonical))
    assert binding.base_path == data and binding.logical_base_path == ctx.drive_root
    own = ctx.drive_root / "local.txt"
    own.write_text("CHILD LOCAL", encoding="utf-8")
    assert "CHILD LOCAL" in registry.execute("read_file", {"root": "runtime_data", "path": "local.txt"})
    assert "outside selected root" in registry.execute("read_file", {"root": "artifact_store", "path": str(canonical)})
    for name, args in (
        ("write_file", {"path": str(first / "source.txt"), "content": "changed"}),
        ("edit_text", {"path": str(first / "source.txt"), "old_str": "SOURCE", "new_str": "changed"}),
        ("run_command", {"command": "true", "cwd": str(first)}),
        ("run_script", {"script": "print('changed')"}),
        ("commit_reviewed", {"message": "changed"}),
    ):
        assert registry.execute_result(name, args).status != "ok", name
    assert (first / "source.txt").read_text(encoding="utf-8") == "SOURCE off-home"


def test_start_folder_does_not_replace_the_inherited_read_path_class(geometry, monkeypatch):
    from ouroboros.tool_access import user_files_path_block_reason

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    home, repo, data, first = geometry
    target = first.parent / "another-off-home.txt"
    target.write_text("OUTSIDE", encoding="utf-8")
    external = ToolContext(repo_dir=repo, drive_root=data, workspace_root=first, workspace_mode="external")
    _, ctx = child(geometry, "child", "root", home / "selected", capture_parent_workspace(external))
    assert user_files_path_block_reason(ctx, target, operation="read") == ""
    assert user_files_path_block_reason(ctx, target, operation="write")
    ordinary = ToolContext(repo_dir=repo, drive_root=data)
    _, confined = child(geometry, "confined", "root", home / "another", capture_parent_workspace(ordinary))
    assert user_files_path_block_reason(confined, target, operation="read")


@pytest.mark.parametrize("folder", ["repo", "data"])
def test_readonly_focus_inside_source_or_data_is_not_a_mutation_workspace(geometry, monkeypatch, folder):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    home, repo, data, first = geometry
    selected = (repo if folder == "repo" else data) / "subdir"
    parent = ToolContext(repo_dir=repo, drive_root=data)
    registry, ctx = child(geometry, "child", "root", selected, capture_parent_workspace(parent))
    (selected / "readme.txt").write_text("SELECTED", encoding="utf-8")
    assert ctx.active_repo_dir() == selected
    assert "SELECTED" in registry.execute("read_file", {"path": "readme.txt"})
    assert registry.execute_result("write_file", {"path": "readme.txt", "content": "x"}).status != "ok"


@pytest.mark.parametrize("profile", ["local_readonly_subagent", "acting_subagent"])
def test_read_matrix_matches_parent_without_granting_writes(profile, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    from ouroboros.tool_access import _ALL_ROOTS
    for root in _ALL_ROOTS:
        for operation in ("read", "list", "search"):
            assert decide_tool_access(profile=profile, root=root, operation=operation).allow
        if profile == "local_readonly_subagent" or root != "active_workspace":
            for operation in ("write", "edit", "shell", "service"):
                assert not decide_tool_access(profile=profile, root=root, operation=operation).allow


def test_read_start_mode_alone_cannot_bypass_workspace_write_admission(geometry):
    _home, repo, data, _first = geometry
    selected = repo / "subdir"
    selected.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=data, workspace_root=selected, workspace_mode="read_only")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    result = registry.execute_result("write_file", {"path": "new.txt", "content": "unadmitted"})
    assert result.status == "blocked" and "WORKSPACE_MODE_BLOCKED" in result.text
    assert not (selected / "new.txt").exists()


def test_home_search_does_not_replay_ancestry_for_each_file(geometry, monkeypatch):
    from ouroboros import tool_access_reads

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    home, repo, data, first = geometry
    root = ToolContext(repo_dir=repo, drive_root=data, task_id="root", workspace_root=first, workspace_mode="external")
    registry, _ctx = child(geometry, "child", "root", home / "work", capture_parent_workspace(root))
    documents = home / "documents"
    documents.mkdir()
    for index in range(40):
        (documents / f"input-{index}.txt").write_text(f"LOCAL_INPUT_{index}\n", encoding="utf-8")
    def unexpected_ancestry(_ctx):
        pytest.fail("an in-home file asked for the outside-home ancestry policy")
    monkeypatch.setattr(tool_access_reads, "read_allows_outside_home", unexpected_ancestry)
    read = registry.execute("read_file", {"root": "user_files", "path": str(documents / "input-0.txt")})
    assert "LOCAL_INPUT_0" in read, read
    result = registry.execute("search_code", {"root": "user_files", "path": str(documents), "query": "LOCAL_INPUT_"})
    assert "Found 40 matches" in result, result
