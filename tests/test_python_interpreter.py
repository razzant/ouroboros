from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.python_interpreter import resolve_process_python
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _executable(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _venv_python(env_root: pathlib.Path) -> pathlib.Path:
    """Platform-correct fake venv interpreter (Scripts\\python.exe on Windows)."""
    from ouroboros.platform_layer import IS_WINDOWS

    if IS_WINDOWS:
        return _executable(env_root / "Scripts" / "python.exe")
    return _executable(env_root / "bin" / "python")


def _context(
    tmp_path: pathlib.Path,
    *,
    workspace: pathlib.Path | None = None,
    workspace_mode: str = "",
) -> ToolContext:
    repo = tmp_path / "system_repo"
    data = tmp_path / "data"
    repo.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        repo_dir=repo,
        system_repo_dir=repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode=workspace_mode,
        task_id="python-resolver-test",
    )


def _tool_args(tool_name: str, *, cwd: str = "", token: str = "python") -> dict:
    if tool_name == "run_command":
        return {"cmd": [token, "-V"], "cwd": cwd}
    if tool_name == "run_script":
        return {"script": "print('ok')", "interpreter": token, "cwd": cwd}
    if tool_name == "start_service":
        return {"name": "svc", "cmd": [token, "-V"], "cwd": cwd}
    if tool_name == "verify_and_record":
        return {
            "contract_kind": "explicit_command",
            "check": [token, "-V"],
            "cwd": cwd,
        }
    raise AssertionError(tool_name)


def _interpreter(tool_name: str, args: dict) -> str:
    if tool_name in {"run_command", "start_service"}:
        return str(args["cmd"][0])
    if tool_name == "run_script":
        return str(args["interpreter"])
    return str(args["check"][0])


@pytest.mark.parametrize(
    "tool_name",
    ["run_command", "run_script", "start_service", "verify_and_record"],
)
def test_system_surfaces_use_validated_agent_python(tmp_path, monkeypatch, tool_name):
    ctx = _context(tmp_path)
    agent_python = _executable(tmp_path / "agent" / "bin" / "python")
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(agent_python))

    resolved, trace = resolve_process_python(
        ctx,
        tool_name,
        _tool_args(tool_name),
        runtime_mode="advanced",
    )

    assert _interpreter(tool_name, resolved) == str(agent_python)
    assert trace is not None
    assert trace.surface == "system_repo"
    assert trace.environment == "ouroboros_agent"


@pytest.mark.parametrize(
    "tool_name",
    ["run_command", "run_script", "start_service", "verify_and_record"],
)
def test_external_workspace_prefers_project_venv(tmp_path, monkeypatch, tool_name):
    workspace = tmp_path / "workspace"
    ctx = _context(tmp_path, workspace=workspace, workspace_mode="external")
    project_python = _venv_python(workspace / ".venv")
    (workspace / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    monkeypatch.setenv(
        "OUROBOROS_AGENT_PYTHON",
        str(_executable(tmp_path / "agent" / "bin" / "python")),
    )

    resolved, trace = resolve_process_python(
        ctx,
        tool_name,
        _tool_args(tool_name),
        runtime_mode="advanced",
    )

    assert _interpreter(tool_name, resolved) == str(project_python)
    assert trace is not None
    assert trace.surface == "external_workspace"
    assert trace.environment == "project_venv"


def test_project_venv_platform_layouts(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    env_root = tmp_path / ".venv"
    env_root.mkdir()
    (env_root / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")

    posix_python = env_root / "bin" / "python"
    posix_python.parent.mkdir()
    posix_python.write_text("", encoding="utf-8")
    posix_python.chmod(0o755)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    assert platform_layer.project_venv_python(tmp_path) == str(posix_python)

    windows_python = env_root / "Scripts" / "python.exe"
    windows_python.parent.mkdir()
    windows_python.write_text("", encoding="utf-8")
    windows_python.chmod(0o755)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    assert platform_layer.project_venv_python(tmp_path) == str(windows_python)


def test_user_files_project_venv_and_missing_venv_path_fallback(tmp_path, monkeypatch):
    user_root = tmp_path / "user_files"
    project = user_root / "project"
    project.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(user_root))
    ctx = _context(tmp_path)
    project_python = _venv_python(project / ".venv")
    (project / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    resolved, trace = resolve_process_python(
        ctx,
        "run_command",
        _tool_args("run_command", cwd=str(project)),
        runtime_mode="advanced",
    )
    assert resolved["cmd"][0] == str(project_python)
    assert trace is not None and trace.surface == "user_files"

    (project / ".venv" / "pyvenv.cfg").unlink()
    unresolved, fallback = resolve_process_python(
        ctx,
        "run_command",
        _tool_args("run_command", cwd=str(project)),
        runtime_mode="advanced",
    )
    assert unresolved["cmd"][0] == "python"
    assert fallback is not None
    assert fallback.environment == "target_path"
    assert fallback.fallback_reason == "project_venv_unavailable"


def test_executor_uses_backend_python_but_unmapped_task_drive_uses_agent(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    ctx = _context(tmp_path, workspace=workspace, workspace_mode="external")
    ctx.executor_ref = {
        "type": "docker_exec",
        "id": "programbench",
        "container_name": "programbench",
        "network": "none",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }
    _venv_python(workspace / ".venv")
    (workspace / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    agent_python = _executable(tmp_path / "agent" / "bin" / "python")
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(agent_python))

    resolved, trace = resolve_process_python(
        ctx,
        "run_command",
        _tool_args("run_command"),
        runtime_mode="advanced",
    )
    assert resolved["cmd"][0] == "python3"
    assert trace is not None and trace.environment == "backend_path"

    local, local_trace = resolve_process_python(
        ctx,
        "run_command",
        _tool_args("run_command", cwd="task_drive"),
        runtime_mode="advanced",
    )
    assert local["cmd"][0] == str(agent_python)
    assert local_trace is not None and local_trace.surface == "task_drive"


def test_reviewed_skill_environment_precedes_executor(tmp_path, monkeypatch):
    import ouroboros.marketplace.isolated_deps as isolated_deps
    import ouroboros.skill_loader as skill_loader
    import ouroboros.skill_readiness as skill_readiness

    workspace = tmp_path / "workspace"
    ctx = _context(tmp_path, workspace=workspace, workspace_mode="external")
    ctx.task_metadata = {"source": "skill_scheduled_task", "skill": "demo"}
    ctx.executor_ref = {
        "type": "docker_exec",
        "id": "executor",
        "container_name": "executor",
        "network": "none",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    skill_python = _executable(skill_dir / ".ouroboros_env" / "python" / "bin" / "python")
    loaded = SimpleNamespace(name="demo", skill_dir=skill_dir)
    monkeypatch.setattr(skill_loader, "find_skill", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        skill_readiness,
        "skill_readiness_for_execution",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(isolated_deps, "read_deps_state", lambda *args, **kwargs: {"status": "installed"})
    monkeypatch.setattr(isolated_deps, "python_runtime_binary", lambda *args, **kwargs: skill_python)

    resolved, trace = resolve_process_python(
        ctx,
        "run_command",
        _tool_args("run_command"),
        runtime_mode="advanced",
    )

    assert resolved["cmd"][0] == str(skill_python)
    assert trace is not None
    assert trace.environment == "isolated_skill"


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("run_command", {"cmd": ["/usr/bin/python", "-V"]}),
        ("run_command", {"cmd": ["python3.12", "-V"]}),
        ("run_command", {"cmd": ["sh", "-c", "python -V"]}),
        ("run_command", {"cmd": ["env", "python", "-V"]}),
        ("run_script", {"script": "pass", "interpreter": "/usr/bin/python"}),
        ("run_script", {"script": "pass", "interpreter": "python3.12"}),
        ("verify_and_record", {"contract_kind": "explicit_command", "check": "python -V"}),
        ("verify_and_record", {"contract_kind": "artifact_observation", "check": ["python", "-V"]}),
        ("remote_exec", {"cmd": ["python", "-V"]}),
    ],
)
def test_noneligible_invocations_are_byte_for_byte_unchanged(tmp_path, tool_name, args):
    ctx = _context(tmp_path)

    resolved, trace = resolve_process_python(
        ctx,
        tool_name,
        args,
        runtime_mode="advanced",
    )

    assert resolved == args
    assert trace is None


def test_light_run_script_default_cwd_uses_active_workspace_agent_python(tmp_path, monkeypatch):
    ctx = _context(tmp_path)
    agent_python = _executable(tmp_path / "agent" / "bin" / "python")
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(agent_python))

    resolved, trace = resolve_process_python(
        ctx,
        "run_script",
        {"script": "print('ok')"},
        runtime_mode="light",
    )

    assert resolved["interpreter"] == str(agent_python)
    assert trace is not None and trace.surface == "system_repo"
    assert trace.target_root == "active_workspace"


def test_registry_guard_and_handler_receive_same_resolved_verify_argv(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    ctx = _context(tmp_path)
    agent_python = _executable(tmp_path / "agent" / "bin" / "python")
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(agent_python))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(
        safety,
        "_run_llm_check",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected safety LLM call")),
    )

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    captured: dict[str, list[str]] = {}
    # (RWS v2 §3.1) the guard projection is built by dispatch_args, not inline
    # in the registry: patch the seam that actually produces guard_args.
    from ouroboros.tools import dispatch_args as dispatch_args_module

    original_guard = dispatch_args_module.process_shell_guard_args

    def capture_guard(name, args, **kwargs):
        guarded = original_guard(name, args, **kwargs)
        captured["guard"] = list(guarded["cmd"])
        return guarded

    def capture_handler(_ctx, contract_kind, check, _resolved_binding=None, **kwargs):
        assert contract_kind == "explicit_command"
        assert _resolved_binding is not None
        captured["handler"] = list(check)
        return "ok"

    monkeypatch.setattr(dispatch_args_module, "process_shell_guard_args", capture_guard)
    monkeypatch.setattr(registry, "_run_shell_safety_check", lambda *args, **kwargs: "")
    registry._entries["verify_and_record"].handler = capture_handler

    result = registry.execute(
        "verify_and_record",
        {"contract_kind": "explicit_command", "check": ["python", "-m", "pytest", "--version"]},
    )

    expected = [str(agent_python), "-m", "pytest", "--version"]
    assert result == "ok"
    assert captured == {"guard": expected, "handler": expected}
    events_path = ctx.drive_logs() / "events.jsonl"
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["type"] == "python_interpreter_resolution"
    assert event["requested_interpreter"] == "python"
    assert event["resolved_interpreter"] == str(agent_python)
    assert "cmd" not in event
    assert "check" not in event


def test_registry_uses_current_process_python_before_server_bootstrap(
    tmp_path, monkeypatch,
):
    import sys

    ctx = _context(tmp_path)
    monkeypatch.delenv("OUROBOROS_AGENT_PYTHON", raising=False)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *args, **kwargs: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    observed = {}

    def handler(_ctx, cmd, _resolved_binding=None, **_kwargs):
        assert _resolved_binding is not None
        observed["cmd"] = cmd
        return "ok"

    registry._entries["run_command"].handler = handler

    result = registry.execute("run_command", {"cmd": ["python", "-V"]})

    assert result == "ok"
    assert observed["cmd"][0] == str(pathlib.Path(sys.executable).absolute())


def test_run_script_accepts_registry_attested_versioned_agent_python(
    tmp_path, monkeypatch,
):
    import ouroboros.tools.shell as shell

    ctx = _context(tmp_path)
    versioned_python = _executable(tmp_path / "agent" / "bin" / "python3.12")
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(versioned_python))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(shell, "_run_shell", lambda *_args, **_kwargs: "ok")

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr(registry, "_run_shell_safety_check", lambda *args, **kwargs: "")

    result = registry.execute(
        "run_script",
        {"script": "print('ok')", "interpreter": "python3"},
    )

    assert result.endswith("\nok")
    assert "RUN_SCRIPT_BLOCKED" not in result
    assert not hasattr(ctx, "_active_python_resolution")


def test_run_script_still_blocks_unattested_versioned_interpreter(tmp_path):
    from ouroboros.tools.shell import _run_script

    result = _run_script(
        _context(tmp_path),
        "print('unsafe')",
        interpreter=str(tmp_path / "untrusted" / "python3.12"),
    )

    assert result.startswith("⚠️ RUN_SCRIPT_BLOCKED:")


def test_safety_fast_path_requires_matching_verified_resolver_provenance(tmp_path, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.python_interpreter import PythonResolutionTrace

    agent_python = str(_executable(tmp_path / "agent" / "bin" / "python"))
    verified = PythonResolutionTrace(
        tool="run_command",
        requested_interpreter="python",
        resolved_interpreter=agent_python,
        surface="system_repo",
        environment="ouroboros_agent",
        reason="agent_python",
    )
    calls: list[list[str]] = []

    def llm_check(_tool, arguments, *_args, **_kwargs):
        calls.append(list(arguments["cmd"]))
        return False, "llm-called"

    monkeypatch.setattr(safety, "_run_llm_check", llm_check)
    safe = safety.check_safety(
        "run_command",
        {"cmd": [agent_python, "-m", "pytest", "-q"]},
        python_resolution=verified,
    )
    arbitrary = safety.check_safety(
        "run_command",
        {"cmd": [agent_python, "-m", "pytest", "-q"]},
    )
    mismatch = safety.check_safety(
        "run_command",
        {"cmd": [str(tmp_path / "other" / "python"), "-m", "pytest", "-q"]},
        python_resolution=verified,
    )

    assert safe == (True, "")
    assert arbitrary == (False, "llm-called")
    assert mismatch == (False, "llm-called")
    assert len(calls) == 2


def test_verified_python_c_body_still_requires_safety_review(tmp_path, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.python_interpreter import PythonResolutionTrace

    agent_python = str(_executable(tmp_path / "agent" / "bin" / "python"))
    verified = PythonResolutionTrace(
        tool="run_command",
        requested_interpreter="python3",
        resolved_interpreter=agent_python,
        surface="system_repo",
        environment="ouroboros_agent",
        reason="agent_python",
    )
    monkeypatch.setattr(safety, "_run_llm_check", lambda *args, **kwargs: (False, "reviewed"))

    result = safety.check_safety(
        "run_command",
        {"cmd": [agent_python, "-c", "print('not allowlisted by module')"]},
        python_resolution=verified,
    )

    assert result == (False, "reviewed")


def test_resolution_trace_failure_is_fail_soft(tmp_path, monkeypatch):
    import ouroboros.python_interpreter as resolver

    trace = resolver.PythonResolutionTrace(
        tool="run_command",
        requested_interpreter="python",
        resolved_interpreter="python3",
        surface="executor",
        environment="backend_path",
        reason="executor_backend_python3",
    )
    monkeypatch.setattr(resolver, "append_jsonl", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("full")))

    resolver.record_python_resolution(_context(tmp_path), trace)
