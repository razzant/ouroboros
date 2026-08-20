from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from ouroboros.skill_loader import compute_content_hash, load_enabled
from ouroboros.skill_owner_attestation import review_skill_owner_attest
from ouroboros.skill_review import SkillReviewOutcome
from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
from ouroboros.tool_access import build_resolved_resource_binding, load_bound_skill
from ouroboros.tools import core as core_tools
from ouroboros.tools import git_plumbing
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.skill_preflight import _handle_skill_preflight


def _manifest(name: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Workspace authority consumer test.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "permissions: []\n"
        "---\n"
        "Use the payload.\n"
    )


def _skill(root: pathlib.Path, location: str, name: str, body: str = "VALUE = 1\n") -> pathlib.Path:
    skill_dir = root / "skills" / location / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_manifest(name), encoding="utf-8")
    (skill_dir / "tool.py").write_text(body, encoding="utf-8")
    return skill_dir


def _fork_ctx(tmp_path: pathlib.Path) -> tuple[ToolContext, pathlib.Path, pathlib.Path, pathlib.Path]:
    repo = tmp_path / "repo"
    parent = tmp_path / "parent-data"
    child = tmp_path / "child-data"
    for item in (repo, parent, child):
        item.mkdir()
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=child,
        task_metadata={"budget_drive_root": str(parent)},
    )
    return ctx, repo, parent, child


def test_new_external_payload_and_provenance_use_canonical_root_not_child_shadow(
    tmp_path, monkeypatch,
):
    import ouroboros.safety as safety

    ctx, repo, parent, child = _fork_ctx(tmp_path)
    shadow = child / "skills" / "external" / "alpha"
    shadow.mkdir(parents=True)
    (shadow / "shadow.txt").write_text("untouched\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=child)
    registry._ctx = ctx
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "SKILL.md",
            "content": _manifest("alpha"),
        },
    )

    payload = parent / "skills" / "external" / "alpha"
    assert result.startswith("OK: wrote overwrite skill_payload:SKILL.md")
    assert (payload / "SKILL.md").is_file()
    assert (payload / ".self_authored.json").is_file()
    assert (parent / "state" / "skills" / "alpha" / "self_authored.json").is_file()
    binding = build_resolved_resource_binding(
        ctx, root="skill_payload", operation="review", path=".", skill_name="alpha",
    )
    assert load_bound_skill(binding).source == "self_authored"
    assert (shadow / "shadow.txt").read_text(encoding="utf-8") == "untouched\n"
    assert not (child / "state" / "skills" / "alpha").exists()


@pytest.mark.parametrize("layout", ["direct", "flat", "grouped"])
def test_user_repo_layouts_mutate_selected_bytes_without_data_provenance(
    tmp_path, monkeypatch, layout,
):
    import ouroboros.safety as safety

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    if layout == "direct":
        checkout = tmp_path / "beta"
        skill_dir = checkout
    elif layout == "flat":
        checkout = tmp_path / "checkout"
        skill_dir = checkout / "beta"
    else:
        checkout = tmp_path / "checkout"
        skill_dir = checkout / "group" / "beta"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_manifest("beta"), encoding="utf-8")
    (skill_dir / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    result = registry.execute(
        "edit_text",
        {
            "root": "skill_payload",
            "bucket": "user_repo",
            "skill_name": "beta",
            "path": "tool.py",
            "old_str": "VALUE = 1",
            "new_str": "VALUE = 2",
        },
    )

    assert "source=user_repo" in result
    assert (skill_dir / "tool.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert not (skill_dir / ".self_authored.json").exists()
    assert not (data / "state" / "skills" / "beta" / "self_authored.json").exists()


def test_preflight_reads_bound_canonical_payload_not_fork_shadow(tmp_path):
    ctx, _repo, parent, child = _fork_ctx(tmp_path)
    canonical = _skill(parent, "external", "alpha", body="if True print('broken')\n")
    shadow = _skill(child, "external", "alpha", body="VALUE = 1\n")
    binding = build_resolved_resource_binding(
        ctx, root="skill_payload", operation="review", path=".", skill_name="alpha",
    )

    payload = json.loads(
        _handle_skill_preflight(ctx, skill="alpha", _resolved_binding=binding)
    )

    assert binding.base_path == canonical.resolve()
    assert binding.base_path != shadow.resolve()
    assert payload["ok"] is False
    assert any(
        item.get("path") == "tool.py" and not item.get("ok")
        for item in payload["files"]
    )


def test_registry_preflight_infers_unique_canonical_skill_binding(tmp_path):
    ctx, repo, parent, child = _fork_ctx(tmp_path)
    canonical = _skill(parent, "external", "alpha", body="if True print('broken')\n")
    shadow = _skill(child, "external", "alpha", body="VALUE = 1\n")
    registry = ToolRegistry(repo_dir=repo, drive_root=child)
    registry.set_context(ctx)

    payload = json.loads(registry.execute("skill_preflight", {"skill": "alpha"}))

    assert canonical.resolve() != shadow.resolve()
    assert payload["ok"] is False
    assert any(
        item.get("path") == "tool.py" and not item.get("ok")
        for item in payload["files"]
    )
    assert not (child / "state" / "skills" / "alpha").exists()


def test_review_lifecycle_enablement_and_job_state_follow_binding_state_root(
    tmp_path, monkeypatch,
):
    import ouroboros.skill_review_runner as runner

    ctx, _repo, parent, child = _fork_ctx(tmp_path)
    canonical = _skill(parent, "external", "alpha")
    _skill(child, "external", "alpha", body="SHADOW = True\n")
    binding = build_resolved_resource_binding(
        ctx, root="skill_payload", operation="review", path=".", skill_name="alpha",
    )
    content_hash = compute_content_hash(canonical)

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="clean",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            auto_flow=True,
        )

    monkeypatch.setattr(
        runner, "_reconcile_deps_after_pass_review",
        lambda *_args, **_kwargs: ("not_required", ""),
    )
    monkeypatch.setattr(
        runner, "_reconcile_extension_payload",
        lambda *_args, **_kwargs: ("extension_inactive", "not_extension"),
    )

    result = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        _resolved_binding=binding,
    )

    assert result["status"] == "clean"
    assert load_enabled(parent, "alpha") is True
    assert (parent / "state" / "skills" / "alpha" / "review_job.json").is_file()
    assert not (child / "state" / "skills" / "alpha").exists()


def test_owner_attestation_reuses_lifecycle_binding(tmp_path):
    ctx, _repo, parent, child = _fork_ctx(tmp_path)
    canonical = _skill(parent, "external", "alpha")
    _skill(child, "external", "alpha", body="SHADOW = True\n")
    binding = build_resolved_resource_binding(
        ctx, root="skill_payload", operation="review", path=".", skill_name="alpha",
    )
    ctx._skill_review_resolved_binding = binding

    outcome = review_skill_owner_attest(ctx, "alpha")

    assert outcome.status == "clean"
    assert outcome.content_hash == compute_content_hash(canonical)
    assert (parent / "state" / "skills" / "alpha" / "owner_attestation.json").is_file()
    assert not (child / "state" / "skills" / "alpha").exists()


def test_collision_blocks_bound_lifecycle_before_state_write(tmp_path, monkeypatch):
    ctx, _repo, parent, child = _fork_ctx(tmp_path)
    checkout = tmp_path / "checkout"
    _skill(parent, "external", "same")
    user = checkout / "same"
    user.mkdir(parents=True)
    (user / "SKILL.md").write_text(_manifest("same"), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))

    result = run_skill_review_lifecycle_blocking(ctx, "same", source="test")

    assert result["status"] == "pending"
    assert "collision" in result["error"].lower()
    assert not (parent / "state" / "skills" / "same").exists()
    assert not (child / "state" / "skills" / "same").exists()


def test_native_installed_copy_is_reviewable_but_seed_mutation_stays_system_repo(
    tmp_path, monkeypatch,
):
    ctx, system_repo, parent, child = _fork_ctx(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    ctx.system_repo_dir = system_repo
    ctx.workspace_root = workspace
    ctx.workspace_mode = "external"
    installed = _skill(parent, "native", "native_alpha")
    seed = system_repo / "skills" / "native_alpha"
    seed.mkdir(parents=True)
    (seed / "tool.py").write_text("SEED = 1\n", encoding="utf-8")
    binding = build_resolved_resource_binding(
        ctx, root="skill_payload", operation="review", path=".",
        skill_name="native_alpha",
    )

    preflight = json.loads(
        _handle_skill_preflight(
            ctx, skill="native_alpha", _resolved_binding=binding,
        )
    )
    with pytest.raises(ValueError, match="read/review only"):
        build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="edit",
            path="tool.py",
            bucket="native",
            skill_name="native_alpha",
        )
    result = core_tools._edit_text(
        ctx,
        path="skills/native_alpha/tool.py",
        old_str="SEED = 1",
        new_str="SEED = 2",
        root="system_repo",
    )

    assert preflight["ok"] is True
    assert binding.base_path == installed.resolve()
    assert "Native seed boundary" in result
    assert (seed / "tool.py").read_text(encoding="utf-8") == "SEED = 2\n"
    assert (installed / "tool.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (child / "skills" / "native" / "native_alpha").exists()


def test_protected_name_is_judged_by_physical_repo_target(tmp_path, monkeypatch):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "project"
    data = tmp_path / "data"
    for item in (system_repo, workspace, data):
        item.mkdir()
    (system_repo / "BIBLE.md").write_text("system\n", encoding="utf-8")
    (workspace / "BIBLE.md").write_text("project\n", encoding="utf-8")
    ctx = ToolContext(
        repo_dir=system_repo,
        system_repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    )
    monkeypatch.setattr(git_plumbing, "get_runtime_mode", lambda: "light")

    workspace_result = core_tools._write_file(
        ctx, path="BIBLE.md", content="project changed\n",
        root="active_workspace",
    )
    system_result = core_tools._write_file(
        ctx, path="BIBLE.md", content="system changed\n", root="system_repo",
    )

    assert workspace_result.startswith("✅ Written")
    assert system_result.startswith("⚠️ CORE_PROTECTION_BLOCKED")
    assert (workspace / "BIBLE.md").read_text(encoding="utf-8") == "project changed\n"
    assert (system_repo / "BIBLE.md").read_text(encoding="utf-8") == "system\n"


def _protected_workspace_registry(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    system_repo = tmp_path / "system"
    workspace = tmp_path / "project"
    data = tmp_path / "data"
    for item in (system_repo, workspace, data):
        item.mkdir()
    for repo in (system_repo, workspace):
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    contract = {
        "resource_policy": {
            "protected_artifacts": [{
                "id": "reference",
                "paths": ["secret.py"],
                "role": "black_box_reference",
            }],
        },
    }
    ctx = ToolContext(
        repo_dir=system_repo,
        system_repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_contract=contract,
        task_metadata={"task_contract": contract},
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    return registry, ctx, system_repo, workspace, data


def test_relative_protected_artifact_uses_explicit_system_binding_across_file_tools(
    tmp_path, monkeypatch,
):
    registry, _ctx, system_repo, workspace, _data = _protected_workspace_registry(
        tmp_path, monkeypatch,
    )
    (system_repo / "secret.py").write_text("def hidden_marker():\n    return 42\n", encoding="utf-8")
    (system_repo / "public.py").write_text("PUBLIC = 1\n", encoding="utf-8")
    (workspace / "secret.py").write_text("PROJECT = 1\n", encoding="utf-8")

    read_result = registry.execute(
        "read_file", {"root": "system_repo", "path": "secret.py"},
    )
    write_result = registry.execute(
        "write_file",
        {"root": "system_repo", "path": "secret.py", "content": "CHANGED = 1\n"},
    )
    edit_result = registry.execute(
        "edit_text",
        {
            "root": "system_repo",
            "path": "secret.py",
            "old_str": "return 42",
            "new_str": "return 43",
        },
    )
    search_result = registry.execute(
        "search_code",
        {"root": "system_repo", "path": ".", "query": "hidden_marker"},
    )
    public_result = registry.execute(
        "read_file", {"root": "system_repo", "path": "public.py"},
    )

    for result in (read_result, write_result, edit_result):
        assert "RESOURCE_POLICY_BLOCKED" in result
    assert "secret.py" not in search_result
    assert "PUBLIC = 1" in public_result
    assert "return 42" in (system_repo / "secret.py").read_text(encoding="utf-8")


def test_query_and_vcs_diff_use_the_same_explicit_system_protected_binding(
    tmp_path, monkeypatch,
):
    registry, _ctx, system_repo, _workspace, _data = _protected_workspace_registry(
        tmp_path, monkeypatch,
    )
    secret = system_repo / "secret.py"
    secret.write_text("def hidden_marker():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "secret.py"], cwd=system_repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"],
        cwd=system_repo,
        check=True,
    )
    secret.write_text("def hidden_marker():\n    return 2\n", encoding="utf-8")

    query_result = registry.execute(
        "query_code",
        {"root": "system_repo", "op": "definition", "query": "hidden_marker"},
    )
    diff_result = registry.execute("vcs_diff", {"root": "system_repo"})

    assert "secret.py" not in query_result
    assert "RESOURCE_POLICY_BLOCKED" in diff_result


def test_relative_protected_artifact_uses_exact_skill_binding(tmp_path, monkeypatch):
    registry, ctx, _system_repo, _workspace, data = _protected_workspace_registry(
        tmp_path, monkeypatch,
    )
    skill = _skill(data, "external", "alpha")
    ctx.task_contract["resource_policy"]["protected_artifacts"][0]["paths"] = ["tool.py"]
    ctx.task_metadata["task_contract"] = ctx.task_contract

    result = registry.execute(
        "read_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "tool.py",
        },
    )

    assert "RESOURCE_POLICY_BLOCKED" in result
    assert (skill / "tool.py").read_text(encoding="utf-8") == "VALUE = 1\n"
