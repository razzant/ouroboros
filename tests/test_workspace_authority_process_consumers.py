from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.outcomes import verification_receipts_path
from ouroboros.protected_artifacts import shell_block_reason
from ouroboros.process_interpreters import resolve_process_python
from ouroboros.tool_access import (
    _side_effect_free_process_roots,
    build_resolved_resource_binding,
)
from ouroboros.tools.services import _start_service, _stop_service
from ouroboros.tools.shell import _resolve_declared_output, _run_script, _run_shell
from ouroboros.tools.verify import _verify_and_record


def _ctx(
    tmp_path: pathlib.Path,
    *,
    forked: bool = False,
    workspace: bool = True,
) -> tuple[ToolContext, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    system = tmp_path / "system"
    active = tmp_path / "active"
    canonical = tmp_path / "canonical-data"
    task_drive = tmp_path / "child-data" if forked else canonical
    for path in (system, active, canonical, task_drive):
        path.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        drive_root=task_drive,
        workspace_root=active if workspace else None,
        workspace_mode="project" if workspace else "",
        task_id="process-binding-test",
        task_metadata={"budget_drive_root": str(canonical)} if forked else {},
    )
    return ctx, system, active, canonical, task_drive


def _skill(data_root: pathlib.Path, name: str = "alpha") -> pathlib.Path:
    skill = data_root / "skills" / "external" / name
    (skill / "sub").mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return skill


def _executable(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_process_binding_default_and_label_subdir_precedence(tmp_path):
    ctx, system, active, _canonical, _task = _ctx(tmp_path)
    (system / "tools").mkdir()
    (active / "system_repo" / "tools").mkdir(parents=True)

    default = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd=""
    )
    explicit = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="system_repo/tools"
    )
    service = build_resolved_resource_binding(
        ctx, operation="service", process_cwd="system_repo/tools"
    )

    assert (default.root, default.target_path) == ("active_workspace", active.resolve())
    assert explicit.root == service.root == "system_repo"
    assert explicit.target_path == service.target_path == (system / "tools").resolve()
    assert explicit.target_path != (active / "system_repo" / "tools").resolve()


def test_process_candidate_inventory_is_pure_and_matches_execution(tmp_path, monkeypatch):
    ctx, system, active, _canonical, task = _ctx(tmp_path)
    user_root = tmp_path / "home"
    user_root.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(user_root))

    shell_roots = _side_effect_free_process_roots(ctx, "shell")
    service_roots = _side_effect_free_process_roots(ctx, "service")
    assert [label for label, _path in shell_roots] == [
        "active_workspace", "system_repo", "task_drive", "artifact_store", "user_files",
    ]
    assert [label for label, _path in service_roots] == [
        "active_workspace", "system_repo", "task_drive", "artifact_store", "user_files",
    ]
    assert not (task / "task_drives").exists()
    assert not (task / "task_results").exists()

    for label, expected in (("active_workspace", active), ("system_repo", system)):
        binding = build_resolved_resource_binding(ctx, operation="shell", process_cwd=label)
        assert binding.target_path == expected.resolve()
    task_binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="task_drive"
    )
    assert task_binding.target_path.is_dir()
    with pytest.raises(ValueError, match="cannot shell root=runtime_data"):
        build_resolved_resource_binding(
            ctx, operation="shell", process_cwd="runtime_data"
        )


def test_external_absolute_cwd_fallback_is_not_shared_by_plain_workspace(
    tmp_path, monkeypatch,
):
    ctx, _system, _active, _canonical, _task = _ctx(tmp_path)
    user_root = tmp_path / "home"
    outside = tmp_path / "host-scratch"
    user_root.mkdir()
    outside.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(user_root))

    with pytest.raises(ValueError, match="outside allowed roots"):
        build_resolved_resource_binding(
            ctx, operation="shell", process_cwd=str(outside)
        )
    ctx.workspace_mode = "external"
    binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd=str(outside)
    )
    assert binding.root == "user_files"
    assert binding.base_path == binding.target_path == outside.resolve()


def test_exact_non_native_skill_binding_and_service_refusal(tmp_path, monkeypatch):
    ctx, _system, active, canonical, _task = _ctx(tmp_path)
    skill = _skill(canonical)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))

    binding = build_resolved_resource_binding(
        ctx,
        operation="shell",
        process_cwd="skill_payload/sub",
        bucket="external",
        skill_name="alpha",
    )
    selectors_on_default = build_resolved_resource_binding(
        ctx,
        operation="shell",
        process_cwd="",
        bucket="external",
        skill_name="alpha",
    )

    assert binding.root == "skill_payload"
    assert binding.base_path == skill.resolve()
    assert binding.target_path == (skill / "sub").resolve()
    assert binding.source == "external"
    assert selectors_on_default.target_path == active.resolve()
    selected_roots = _side_effect_free_process_roots(
        ctx,
        "shell",
        bucket="external",
        skill_name="alpha",
        include_skill=True,
    )
    assert selected_roots[-1] == ("skill_payload", skill.resolve())
    with pytest.raises(ValueError, match="cannot service root=skill_payload"):
        build_resolved_resource_binding(
            ctx, operation="service", process_cwd="skill_payload"
        )
    with pytest.raises(ValueError, match="outside allowed roots"):
        build_resolved_resource_binding(
            ctx, operation="service", process_cwd=str(skill)
        )


def test_selected_skill_python_uses_exact_binding_and_traceable_fallback(
    tmp_path, monkeypatch,
):
    import ouroboros.marketplace.isolated_deps as isolated_deps
    import ouroboros.skill_loader as skill_loader
    import ouroboros.skill_readiness as skill_readiness

    ctx, _system, active, canonical, _task = _ctx(tmp_path)
    skill = _skill(canonical)
    binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="skill_payload",
        bucket="external", skill_name="alpha",
    )
    project_python = _executable(active / ".venv" / "bin" / "python")
    (active / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    skill_python = _executable(skill / ".ouroboros_env" / "python" / "bin" / "python")
    agent_python = _executable(tmp_path / "agent" / "bin" / "python")
    loaded = SimpleNamespace(name="alpha", skill_dir=skill)
    observed: list[tuple[pathlib.Path, pathlib.Path]] = []

    def exact_load(skill_dir, state_root):
        observed.append((pathlib.Path(skill_dir), pathlib.Path(state_root)))
        return loaded

    monkeypatch.setattr(skill_loader, "load_skill", exact_load)
    monkeypatch.setattr(
        skill_loader,
        "find_skill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ambient rediscovery")),
    )
    monkeypatch.setattr(
        skill_readiness, "skill_readiness_for_execution",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(isolated_deps, "read_deps_state", lambda *_a, **_k: {"status": "installed"})
    monkeypatch.setattr(isolated_deps, "python_runtime_binary", lambda *_a, **_k: skill_python)
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(agent_python))
    args = {
        "cmd": ["python", "-V"], "cwd": "skill_payload",
        "bucket": "external", "skill_name": "alpha",
    }

    resolved, trace = resolve_process_python(
        ctx, "run_command", args, runtime_mode="advanced", resolved_binding=binding,
    )
    assert resolved["cmd"][0] == str(skill_python)
    assert trace is not None and trace.environment == "isolated_skill"
    assert trace.target_root == "skill_payload"
    assert observed == [(skill.resolve(), canonical.resolve())]
    assert resolved["cmd"][0] != str(project_python)

    monkeypatch.setattr(
        skill_readiness, "skill_readiness_for_execution",
        lambda *_args, **_kwargs: SimpleNamespace(ready=False),
    )
    fallback_args, fallback = resolve_process_python(
        ctx, "run_command", args, runtime_mode="advanced", resolved_binding=binding,
    )
    assert fallback_args["cmd"][0] == str(agent_python)
    assert fallback is not None
    assert fallback.fallback_reason == "reviewed_skill_environment_unavailable"


def test_protected_artifact_guard_uses_binding_target(tmp_path):
    ctx, _system, _active, canonical, _task = _ctx(tmp_path)
    skill = _skill(canonical)
    (skill / "secret.bin").write_bytes(b"secret")
    ctx.task_contract = {
        "resource_policy": {
            "protected_artifacts": [{
                "id": "reference", "role": "black_box_reference", "paths": ["secret.bin"],
            }]
        }
    }
    binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="skill_payload",
        bucket="external", skill_name="alpha",
    )

    legacy = shell_block_reason(
        ctx, ["cat", "secret.bin"], cwd=str(skill), default_cwd=skill,
    )
    selected = shell_block_reason(
        ctx, ["cat", "secret.bin"], cwd=str(skill), default_cwd=skill, binding=binding,
    )

    assert legacy == ""
    assert "RESOURCE_POLICY_BLOCKED" in selected


def test_run_script_forwards_same_binding_and_stages_outside_skill(tmp_path, monkeypatch):
    import ouroboros.tools.shell as shell

    ctx, _system, _active, canonical, task = _ctx(tmp_path)
    skill = _skill(canonical)
    binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="skill_payload/sub",
        bucket="external", skill_name="alpha",
    )
    captured = {}

    def fake_run(_ctx, argv, **kwargs):
        captured.update({"binding": kwargs.get("_resolved_binding"), "script": pathlib.Path(argv[1])})
        assert captured["script"].is_file()
        return "ok"

    monkeypatch.setattr(shell, "_run_shell", fake_run)
    result = _run_script(
        ctx, "echo ok", interpreter="sh", cwd="skill_payload/sub",
        _resolved_binding=binding,
    )

    assert result.endswith("\nok")
    assert captured["binding"] is binding
    assert captured["script"].is_relative_to(ctx.task_drive_root())
    assert not captured["script"].is_relative_to(skill)
    assert not captured["script"].exists()
    assert (task / "task_drives" / ctx.task_id).is_dir()


def test_run_and_verify_disclose_binding_but_keep_task_custody(tmp_path):
    ctx, system, _active, canonical, task = _ctx(tmp_path, forked=True)
    skill = _skill(canonical)
    skill_binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="skill_payload/sub",
        bucket="external", skill_name="alpha",
    )
    system_binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="system_repo"
    )

    command = _run_shell(ctx, [sys.executable, "-c", "print('ok')"], _resolved_binding=system_binding)
    verified = _verify_and_record(
        ctx, contract_kind="explicit_command",
        check=[sys.executable, "-c", "print('verified')"],
        _resolved_binding=skill_binding,
    )

    assert f"cwd={system.resolve()}" in command
    assert "root=system_repo" in command
    assert "root=skill_payload" in verified
    receipt_path = verification_receipts_path(task, ctx.task_id)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["resource_binding"]["target_path"] == str(skill / "sub")
    assert receipt["resource_binding"]["source"] == "external"
    assert not verification_receipts_path(canonical, ctx.task_id).exists()


def test_exact_skill_outputs_use_binding_and_remain_task_custodied(tmp_path):
    ctx, _system, _active, canonical, task = _ctx(tmp_path, forked=True)
    skill = _skill(canonical)
    binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="skill_payload/sub",
        bucket="external", skill_name="alpha",
    )

    result = _run_shell(
        ctx,
        [sys.executable, "-c", "from pathlib import Path; Path('report.txt').write_text('ok')"],
        outputs=["report.txt"],
        _resolved_binding=binding,
    )

    assert "ARTIFACT_OUTPUTS" in result
    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert (skill / "sub" / "report.txt").read_text(encoding="utf-8") == "ok"
    assert any(path.name == "report.txt" for path in task.rglob("report.txt"))
    assert not any(path.name == "report.txt" for path in canonical.glob("task_results/**/*"))


def test_registry_injects_same_binding_for_exact_skill_process(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    ctx, system, _active, canonical, task = _ctx(tmp_path, forked=True)
    skill = _skill(canonical)
    registry = ToolRegistry(repo_dir=system, drive_root=task)
    registry.set_context(ctx)
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('registry.txt').write_text('bound')",
            ],
            "cwd": "skill_payload/sub",
            "bucket": "external",
            "skill_name": "alpha",
            "outputs": ["registry.txt"],
        },
    )

    assert "ARTIFACT_OUTPUTS" in result
    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert "root=skill_payload" in result
    assert (skill / "sub" / "registry.txt").read_text(encoding="utf-8") == "bound"
    assert any(path.name == "registry.txt" for path in task.rglob("registry.txt"))


def test_registry_light_mode_follows_selected_project_skill_or_system_target(
    tmp_path, monkeypatch,
):
    import ouroboros.safety as safety

    ctx, system, active, canonical, task = _ctx(tmp_path)
    skill = _skill(canonical)
    registry = ToolRegistry(repo_dir=system, drive_root=task)
    registry.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))

    project_result = registry.execute(
        "run_command", {"cmd": ["touch", "project.txt"]},
    )
    skill_result = registry.execute(
        "run_command",
        {
            "cmd": ["touch", "skill.txt"],
            "cwd": "skill_payload/sub",
            "bucket": "external",
            "skill_name": "alpha",
        },
    )
    system_result = registry.execute(
        "run_command",
        {"cmd": ["touch", "system.txt"], "cwd": "system_repo"},
    )

    assert "exit_code=0" in project_result
    assert "exit_code=0" in skill_result
    assert (active / "project.txt").is_file()
    assert (skill / "sub" / "skill.txt").is_file()
    assert "LIGHT_MODE_BLOCKED" in system_result
    assert not (system / "system.txt").exists()


def test_explicit_system_output_keeps_binding_relative_protected_policy(tmp_path):
    ctx, system, _active, _canonical, _task = _ctx(tmp_path)
    protected = system / "BIBLE.md"
    protected.write_text("protected", encoding="utf-8")
    binding = build_resolved_resource_binding(
        ctx, operation="shell", process_cwd="system_repo",
    )

    source, reason = _resolve_declared_output(
        ctx,
        "BIBLE.md",
        system,
        cwd_root="system_repo",
        changed_paths={"BIBLE.md"},
        binding=binding,
    )

    assert source is None
    assert "protected" in reason.lower()


def test_system_service_record_discloses_target_and_stays_task_scoped(
    tmp_path, monkeypatch,
):
    ctx, system, _active, canonical, task = _ctx(tmp_path, forked=True)
    binding = build_resolved_resource_binding(
        ctx, operation="service", process_cwd="system_repo"
    )
    invalidations = []
    monkeypatch.setattr(
        "ouroboros.tools.commit_gate._invalidate_advisory",
        lambda _ctx, **kwargs: invalidations.append(kwargs),
    )
    try:
        started = json.loads(_start_service(
            ctx,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            name="binding-service",
            cwd="system_repo",
            readiness={"timeout_sec": 0},
            _resolved_binding=binding,
        ))
        assert started["cwd_root"] == "system_repo"
        assert started["cwd"] == str(system.resolve())
        assert started["cwd_base"] == str(system.resolve())
        assert started["cwd_source"] == "system_repo"
        assert pathlib.Path(started["log_path"]).is_relative_to(task / "services")
        assert not pathlib.Path(started["log_path"]).is_relative_to(canonical)
        assert pathlib.Path(invalidations[-1]["mutation_root"]) == system.resolve()
    finally:
        _stop_service(ctx, name="binding-service")
