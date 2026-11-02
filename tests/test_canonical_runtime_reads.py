"""Canonical read parity uses real dispatch, with synthetic settings and explicit restrictions."""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.tool_access_reads import canonical_runtime_reader as _canonical_runtime_reader
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import build_resolved_resource_binding, filesystem_affordance_map
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_authored_memory import _run, _speaker


@pytest.fixture(params=["light", "advanced", "pro", "cyber_pro"])
def world(tmp_path, monkeypatch, request):
    repo, data, home, work = [tmp_path / name for name in ("repo", "data", "home", "work")]
    fork = data / "state" / "headless_tasks" / "reader" / "data"
    for path in (repo, data, home, work, fork):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", request.param)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", data)
    for rel in ("settings.json", "memory/deep_review.md", "memory/knowledge_history.jsonl", "logs/task_reflections.jsonl"):
        for base, marker in ((data, "CANONICAL_SYNTHETIC"), (fork, "LOCAL_SYNTHETIC")):
            path = base / rel
            path.parent.mkdir(exist_ok=True)
            path.write_text(f'{marker} {rel}\n', encoding="utf-8")
    return SimpleNamespace(repo_dir=repo, drive_root=data, fork=fork, work=work, mode=request.param)


def actor(world, profile="root", **fields):
    meta = {"budget_drive_root": str(world.drive_root), "delegation_role": "root", "root_task_id": "reader"}
    if profile != "root":
        meta.update(delegation_role="subagent", parent_task_id="parent", root_task_id="parent")
        fields["task_constraint"] = TaskConstraint(mode=profile, surface="external_workspace", write_root=str(world.work))
    ctx = ToolContext(repo_dir=world.repo_dir, drive_root=world.fork, task_id="reader",
                      budget_drive_root=str(world.drive_root), workspace_root=world.work, workspace_mode="external",
                      project_id="p1", task_metadata=meta, **fields)
    reg = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    reg.set_context(ctx)
    return ctx, reg


@pytest.mark.parametrize("profile", ["root", "local_readonly_subagent", "acting_subagent"])
def test_nested_fork_lists_searches_and_reads_original_without_write_aliases(world, profile):
    ctx, registry = actor(world, profile)
    source = world.drive_root / "settings.json"
    reader = _canonical_runtime_reader(ctx, world.drive_root, "settings.json")
    assert "CANONICAL_SYNTHETIC settings.json" in _run(ctx, reader)
    binding = build_resolved_resource_binding(ctx, root="runtime_data", operation="read", path=str(source))
    assert binding.base_path == world.drive_root and binding.logical_base_path == world.fork
    assert {"root": "runtime_data", "path": str(world.drive_root)} in filesystem_affordance_map(ctx)["inherited_read_bindings"]
    for explicit in ({"root": "runtime_data"}, {}):
        assert "settings.json" in registry.execute("list_files", {**explicit, "path": str(world.drive_root)})
        found = registry.execute("search_code", {**explicit, "path": str(source), "query": "CANONICAL_SYNTHETIC"})
        assert "CANONICAL_SYNTHETIC settings.json" in found, found
        assert "CANONICAL_SYNTHETIC settings.json" in registry.execute("read_file", {**explicit, "path": str(source)})
    assert "LOCAL_SYNTHETIC settings.json" in registry.execute("read_file", {"root": "runtime_data", "path": "settings.json"})
    for wrong in ("active_workspace", "system_repo", "task_drive", "artifact_store"):
        result = registry.execute_result("read_file", {"root": wrong, "path": str(source)})
        assert result.status != "ok" and "CANONICAL_SYNTHETIC" not in result.text, result.text
    writable_name = world.drive_root / "logs" / "task_reflections.jsonl"
    original = writable_name.read_bytes()
    for root in ({"root": "runtime_data"}, {}):
        for name, payload in (("write_file", {"content": "changed"}),
                              ("edit_text", {"old_str": "CANONICAL_SYNTHETIC", "new_str": "changed"})):
            result = registry.execute_result(name, {**root, "path": str(writable_name), **payload})
            assert result.status != "ok", (name, result.text)
    assert writable_name.read_bytes() == original
    # READ aliases never enter shell/service resolution either.
    for operation in ("shell", "service"):
        with pytest.raises(ValueError):
            build_resolved_resource_binding(ctx, root="runtime_data", operation=operation, path=str(world.drive_root))


def test_root_never_inherits_workspace_edges_or_execution_drives(world):
    ctx, registry = actor(world)
    ancestor = world.work.parent / "ancestor"
    ancestor.mkdir()
    source = ancestor / "source.txt"
    source.write_text("ANCESTOR_ONLY", encoding="utf-8")
    ctx.task_metadata.update(parent_workspace={"root": str(ancestor), "mode": "external"},
                             workspace_root=str(ancestor), child_drive_root=str(ancestor))
    for root in ("active_workspace", "runtime_data"):
        result = registry.execute_result("read_file", {"root": root, "path": str(source)})
        assert result.status != "ok" and "ANCESTOR_ONLY" not in result.text


def test_emitted_reader_keeps_disabled_tools_resource_policy_and_network_restrictions(world):
    source = world.drive_root / "memory" / "deep_review.md"
    ctx, registry = actor(world, task_contract={"disabled_tools": ["read_file", "list_files", "search_code"],
                                              "allowed_resources": {"network": False, "web": False}})
    reader = _canonical_runtime_reader(ctx, world.drive_root, "memory/deep_review.md")
    assert "RESOURCE_CONSTRAINT_BLOCKED" in _run(ctx, reader)
    for name, extra in (("list_files", {}), ("search_code", {"query": "CANONICAL_SYNTHETIC"})):
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute(name, {"root": "runtime_data", "path": str(source), **extra})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("web_search", {"query": "never sent"})
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"id": "retained-report", "paths": [str(source)], "allow": ["execute"]}]}}
    assert "RESOURCE_POLICY_BLOCKED" in _run(ctx, reader)
    found = registry.execute("search_code", {"root": "runtime_data", "path": str(source.parent), "query": "CANONICAL_SYNTHETIC"})
    assert "CANONICAL_SYNTHETIC memory/deep_review.md" not in found
    # The restriction is specific: the unprotected historical log remains readable.
    assert "CANONICAL_SYNTHETIC logs/task_reflections.jsonl" in _run(
        ctx, _canonical_runtime_reader(ctx, world.drive_root, "logs/task_reflections.jsonl"))


def test_presence_keeps_its_selected_physical_runtime_root(world):
    from tests.frozen_presence_capabilities import PresenceResourceTarget, PresenceToolTarget

    for prefix in ("memory", "."):
        ctx = _speaker(world, "speaker", PresenceToolTarget("builtin", "read_file"),
                       PresenceResourceTarget("runtime_data", ("read",), prefix))
        ctx.drive_root = world.fork
        ctx.budget_drive_root = str(world.drive_root)
        reader = _canonical_runtime_reader(ctx, world.drive_root, "memory/deep_review.md")
        result = _run(ctx, reader)
        assert result.startswith("⚠️") and "CANONICAL_SYNTHETIC" not in result, result
        assert "LOCAL_SYNTHETIC memory/deep_review.md" in _run(
            ctx, "read_file(root='runtime_data', path='memory/deep_review.md')")


def _project_reflections_reader(_ctx, project_id: str) -> str:
    """The scoped journal reader of a project's historical reflection log."""
    return f"journal_read(project_id={project_id!r}, source='reflections')"


def test_project_store_stays_behind_the_scoped_reader(world):
    from ouroboros.project_facts import project_reflections_path

    for pid in ("p1", "p2"):
        path = project_reflections_path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"ts":"2026-10-13T00:00:00Z","reflection":"PROJECT_' + pid + '"}\n', encoding="utf-8")
    ctx, registry = actor(world)
    own = _project_reflections_reader(ctx, "p1")
    other = _project_reflections_reader(ctx, "p2")
    assert "PROJECT_p1" in _run(ctx, own) and "PROJECT_p2" in _run(ctx, other)
    assert "ACCESS_DENIED" in registry.execute("read_file", {"root": "runtime_data", "path": str(project_reflections_path("p1"))})
    for profile in ("local_readonly_subagent", "acting_subagent"):
        child, _registry = actor(world, profile)
        if profile == "acting_subagent" and world.mode == "cyber_pro":
            assert "PROJECT_p1" in _run(child, own)
            assert "TOOL_FORBIDDEN" in _run(child, other)
        else:
            # Existing child name ceilings omit journal_read. A source pointer
            # must not be mistaken for a grant or expand those ceilings.
            refusal = "LOCAL_READONLY_SUBAGENT_BLOCKED" if profile == "local_readonly_subagent" else "ACTING_SUBAGENT_BLOCKED"
            assert refusal in _run(child, own) and refusal in _run(child, other)
    # Missing metadata is not proof of root authority even though runtime reads now have parity.
    ctx.task_metadata = {"budget_drive_root": str(world.drive_root)}
    assert "TOOL_FORBIDDEN" in _run(ctx, other)
