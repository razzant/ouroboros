from __future__ import annotations

import pathlib

import pytest

from ouroboros.tool_access import _POLICY, build_resolved_resource_binding
from ouroboros.tools.registry import ToolContext, ToolRegistry


_EXPECTED_TOP_LEVEL_POLICY = {
    "active_workspace": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "system_repo": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "runtime_data": {"read", "list", "search", "write", "edit"},
    "task_drive": {"read", "list", "write", "edit", "shell", "service"},
    "skill_payload": {"read", "list", "search", "write", "edit", "review", "shell"},
    "artifact_store": {"read", "list", "write", "shell", "service"},
    "user_files": {"read", "list", "search", "write", "edit", "shell", "service"},
    "subagent_projects": {"read", "list", "search"},
    "deliverables": {"read", "list", "search"},
}


def test_ordinary_top_level_presets_share_one_exact_principal_matrix():
    ordinary = ("workspace_task", "external_workspace_task", "self_modification")

    for profile in ordinary:
        assert _POLICY[profile] == _EXPECTED_TOP_LEVEL_POLICY

    assert _POLICY[ordinary[0]] is _POLICY[ordinary[1]]
    assert _POLICY[ordinary[1]] is _POLICY[ordinary[2]]


def test_shared_top_level_principal_does_not_widen_specialized_profiles():
    assert "shell" not in _POLICY["local_readonly_subagent"]["skill_payload"]
    assert "shell" not in _POLICY["skill_repair"]["skill_payload"]
    assert "skill_payload" not in _POLICY["acting_subagent"]
    for profile in ("local_readonly_subagent", "skill_repair", "acting_subagent"):
        assert "search" not in _POLICY[profile]["runtime_data"]
    assert "delegate" in _POLICY["operator_control"]["active_workspace"]


def test_private_binding_argument_is_rejected_at_public_boundary(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    result = registry.execute(
        "read_file",
        {"path": "README.md", "_resolved_binding": "model-forged"},
    )

    assert result.startswith("⚠️ TOOL_ARG_ERROR (read_file)"), result
    assert "_resolved_binding" not in result


def _skill(root: pathlib.Path, location: str, name: str) -> pathlib.Path:
    skill_dir = root / "skills" / location / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return skill_dir


def test_binding_selects_canonical_skill_from_forked_drive(tmp_path):
    repo = tmp_path / "repo"
    parent_data = tmp_path / "parent-data"
    child_data = tmp_path / "child-data"
    repo.mkdir()
    child_data.mkdir()
    native = _skill(parent_data, "native", "alpha")
    (native / "notes.txt").write_text("canonical", encoding="utf-8")
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=child_data,
        task_metadata={"budget_drive_root": str(parent_data)},
    )

    binding = build_resolved_resource_binding(
        ctx,
        root="skill_payload",
        operation="read",
        path="notes.txt",
        bucket="native",
        skill_name="alpha",
    )

    assert binding.base_path == native.resolve()
    assert binding.target_path == (native / "notes.txt").resolve()
    assert binding.state_drive_root == parent_data.resolve()
    assert binding.source == "native"
    assert child_data not in binding.target_path.parents


def test_binding_selects_configured_user_repo_without_migration(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    checkout = tmp_path / "skills-checkout"
    repo.mkdir()
    data.mkdir()
    skill_dir = checkout / "group" / "beta"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# beta\n", encoding="utf-8")
    (skill_dir / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    ctx = ToolContext(repo_dir=repo, drive_root=data)

    binding = build_resolved_resource_binding(
        ctx,
        root="skill_payload",
        operation="edit",
        path="tool.py",
        bucket="user_repo",
        skill_name="beta",
    )

    assert binding.base_path == skill_dir.resolve()
    assert binding.target_path == (skill_dir / "tool.py").resolve()
    assert binding.source == "user_repo"
    assert not (data / "skills" / "user_repo").exists()


def test_binding_collision_blocks_mutation_but_exact_read_stays_inspectable(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    checkout = tmp_path / "checkout"
    repo.mkdir()
    data_skill = _skill(data, "external", "same")
    user_skill = checkout / "same"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("# same\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    ctx = ToolContext(repo_dir=repo, drive_root=data)

    read_binding = build_resolved_resource_binding(
        ctx,
        root="skill_payload",
        operation="read",
        bucket="external",
        skill_name="same",
    )
    assert read_binding.base_path == data_skill.resolve()

    with pytest.raises(ValueError, match="collision"):
        build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="write",
            path="SKILL.md",
            bucket="external",
            skill_name="same",
        )
    assert not (data / "state" / "skills" / "same").exists()


def test_binding_preserves_project_room_read_lens_but_not_write_target(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    room = tmp_path / "room"
    for path in (repo, data, room):
        path.mkdir()
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        is_direct_chat=True,
        task_metadata={"_project_room_dir": str(room)},
    )

    read_binding = build_resolved_resource_binding(
        ctx, root="active_workspace", operation="read", path="README.md"
    )
    write_binding = build_resolved_resource_binding(
        ctx, root="active_workspace", operation="write", path="README.md"
    )

    assert read_binding.base_path == room.resolve()
    assert write_binding.base_path == repo.resolve()


def test_binding_synthesizes_only_manifest_first_external_write_target(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=data)

    binding = build_resolved_resource_binding(
        ctx,
        root="skill_payload",
        operation="write",
        path="SKILL.md",
        bucket="external",
        skill_name="new-skill",
    )
    assert binding.base_path == (data / "skills" / "external" / "new-skill").resolve()
    assert not binding.base_path.exists()

    with pytest.raises(ValueError, match="not found"):
        build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="write",
            path="SKILL.md",
            bucket="clawhub",
            skill_name="missing",
        )


def test_registry_builds_once_and_injects_the_same_private_object(tmp_path, monkeypatch):
    import ouroboros.safety as safety
    import ouroboros.tools.registry as registry_module

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    original = registry_module.build_resolved_resource_binding
    built = []
    observed = []

    def counted(*args, **kwargs):
        value = original(*args, **kwargs)
        built.append(value)
        return value

    def handler(ctx, path, root="active_workspace", _resolved_binding=None):
        observed.append(_resolved_binding)
        return "OK"

    monkeypatch.setattr(registry_module, "build_resolved_resource_binding", counted)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    registry.override_handler("read_file", handler)

    assert registry.execute("read_file", {"path": "README.md"}) == "OK"
    assert len(built) == 1
    assert observed == [built[0]]


def test_forged_private_argument_is_rejected_before_binding(tmp_path, monkeypatch):
    import ouroboros.tools.registry as registry_module

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    calls = []
    monkeypatch.setattr(
        registry_module,
        "build_resolved_resource_binding",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = registry.execute(
        "read_file", {"path": "README.md", "_resolved_binding": "forged"}
    )

    assert result.startswith("⚠️ TOOL_ARG_ERROR (read_file)")
    assert calls == []


def test_target_sensitive_override_without_private_keyword_fails_loudly(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    registry.override_handler("read_file", lambda ctx, path: "must not run")

    result = registry.execute("read_file", {"path": "README.md"})

    assert result.startswith("⚠️ TOOL_INTERNAL_ERROR (read_file)")
    assert "_resolved_binding" in result


def test_direct_handler_fallback_builds_once(tmp_path, monkeypatch):
    import ouroboros.tools.core as core

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "README.md").write_text("direct\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=data)
    original = core.build_resolved_resource_binding
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "build_resolved_resource_binding", counted)
    result = core._read_file(ctx, "README.md")

    assert "direct" in result
    assert len(calls) == 1


def test_write_batch_carries_ordered_binding_tuple(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    observed = []

    def handler(ctx, files=None, _resolved_binding=None, **kwargs):
        observed.extend(_resolved_binding)
        return "OK"

    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    registry.override_handler("write_file", handler)
    result = registry.execute("write_file", {"files": [
        {"path": "b.txt", "content": "b"},
        {"path": "a.txt", "content": "a"},
    ]})

    assert result == "OK"
    assert [item.target_path.name for item in observed] == ["b.txt", "a.txt"]


def test_explicit_skill_root_bypasses_repo_named_directories(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "scripts").mkdir()
    (repo / "tests").mkdir()
    skill = _skill(data, "external", "alpha")
    (skill / "scripts").mkdir()
    (skill / "tests").mkdir()
    (skill / "scripts" / "tool.py").write_text("SKILL_MARKER = 1\n", encoding="utf-8")
    (skill / "tests" / "test_x.py").write_text("old = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    selector = {"root": "skill_payload", "bucket": "external", "skill_name": "alpha"}

    read = registry.execute("read_file", {**selector, "path": "scripts/tool.py"})
    listing = registry.execute("list_files", {**selector, "path": "tests"})
    search = registry.execute("search_code", {
        **selector, "path": "scripts", "query": "SKILL_MARKER",
    })
    written = registry.execute("write_file", {
        **selector, "path": "scripts/new.py", "content": "VALUE = 2\n",
    })
    edited = registry.execute("edit_text", {
        **selector, "path": "tests/test_x.py", "old_str": "old = 1", "new_str": "new = 2",
    })

    assert "SKILL_MARKER" in read
    assert "test_x.py" in listing
    assert "SKILL_MARKER" in search
    assert written.startswith("OK: wrote")
    assert "Replaced" in edited
    assert (skill / "scripts" / "new.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (skill / "tests" / "test_x.py").read_text(encoding="utf-8") == "new = 2\n"
    assert not (repo / "scripts" / "new.py").exists()


def test_query_code_explicit_skill_root_is_reachable_through_registry(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    skill = _skill(data, "external", "alpha")
    (skill / "worker.py").write_text("class SkillWorker:\n    pass\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))

    result = registry.execute("query_code", {
        "op": "symbols",
        "root": "skill_payload",
        "bucket": "external",
        "skill_name": "alpha",
        "path": ".",
    })

    assert "SkillWorker" in result
    assert "worker.py" in result


def test_selectors_without_skill_root_do_not_retarget_workspace(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "module.py").write_text("repo = 1\n", encoding="utf-8")
    skill = _skill(data, "external", "alpha")
    (skill / "module.py").write_text("skill = 1\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))

    result = registry.execute("edit_text", {
        "path": "module.py", "old_str": "repo = 1", "new_str": "repo = 2",
        "bucket": "external", "skill_name": "alpha",
    })

    assert "Replaced" in result
    assert (repo / "module.py").read_text(encoding="utf-8") == "repo = 2\n"
    assert (skill / "module.py").read_text(encoding="utf-8") == "skill = 1\n"


def test_explicit_user_repo_keeps_existing_payload_sidecar_block(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    checkout = tmp_path / "checkout"
    repo.mkdir()
    data.mkdir()
    skill = checkout / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
    (skill / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    edited = registry.execute("edit_text", {
        "root": "skill_payload", "bucket": "user_repo", "skill_name": "alpha",
        "path": "tool.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2",
    })

    result = registry.execute("write_file", {
        "root": "skill_payload", "bucket": "user_repo", "skill_name": "alpha",
        "path": ".clawhub.json", "content": "{}\n",
    })

    assert "Replaced" in edited
    assert (skill / "tool.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert result.startswith("⚠️ DATA_WRITE_BLOCKED")
    assert not (skill / ".clawhub.json").exists()


def test_skill_repair_explicit_root_infers_its_existing_selector(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    skill = _skill(data, "external", "alpha")
    (skill / "notes.txt").write_text("repair target\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_constraint = TaskConstraint(
        mode="skill_repair", skill_name="alpha", payload_root="skills/external/alpha"
    )

    result = registry.execute(
        "read_file", {"root": "skill_payload", "path": "notes.txt"}
    )

    assert "repair target" in result
