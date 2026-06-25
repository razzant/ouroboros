from __future__ import annotations

import json
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.shell_parse import is_absolute_path_text, shell_argv_with_path_tokens
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.workspace_executor import execute, map_backend_path, normalize_executor_ref


def test_is_absolute_path_text_is_cross_platform():
    """Deterministic, OS-independent guard for the predicate behind the Windows
    protected-artifact / backend-output path fix. pathlib.Path('/x').is_absolute()
    is False on Windows (no drive letter), which is exactly the POSIX bias that
    caused backend paths to bypass map_backend_path. is_absolute_path_text must
    treat POSIX roots, drive-letter paths, and UNC paths as absolute on every OS,
    and relative tokens / tilde / flags as not-absolute."""
    for text in ("/workspace/x", "/", r"C:\\x", "C:/x", r"\\\\unc\\share"):
        assert is_absolute_path_text(text) is True, text
    for text in ("", "rel/path", "x", "-flag", "~/x", "~"):
        assert is_absolute_path_text(text) is False, text


def test_shell_path_token_extractor_ignores_html_closing_tags():
    tokens = shell_argv_with_path_tokens(["python3", "-c", "Path('site/index.html').write_text('<h1>ok</h1>')"])

    assert "/h1>" not in tokens
    assert "/etc/passwd" in shell_argv_with_path_tokens("tool:/etc/passwd")
    assert "/etc/passwd" in shell_argv_with_path_tokens("cat</etc/passwd")
    assert "/secret>" in shell_argv_with_path_tokens("cat /secret>")


def test_changed_path_covers_directory_entries():
    from ouroboros.tools.shell import _changed_path_covers

    assert _changed_path_covers("site", {"site/index.html"})
    assert _changed_path_covers("site/index.html", {"site/"})
    assert not _changed_path_covers("site/index.html", {"other/"})


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_normalize_executor_ref_rejects_malformed_backend_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base = {
        "type": "local",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }
    for bad in ("workspace", "/", "/workspace/..", "/workspace/./x"):
        payload = dict(base, workspace_backend_path=bad)
        with pytest.raises(ValueError):
            normalize_executor_ref(payload)

    with pytest.raises(ValueError):
        normalize_executor_ref(
            {
                **base,
                "path_mappings": [{"host_path": str(workspace / "x"), "backend_path": "relative"}],
            }
        )


def test_map_backend_path_prefers_longest_backend_prefix(tmp_path):
    broad_host = tmp_path / "longer-host-name"
    nested_host = tmp_path / "n"
    broad_host.mkdir()
    nested_host.mkdir()
    executor = normalize_executor_ref(
        {
            "type": "local",
            "workspace_host_path": str(broad_host),
            "workspace_backend_path": "/workspace",
            "path_mappings": [
                {"host_path": str(nested_host), "backend_path": "/workspace/nested"},
            ],
        }
    )
    assert executor is not None

    mapped = map_backend_path(executor, "/workspace/nested/file.txt")

    assert mapped == (nested_host / "file.txt").resolve(strict=False)


def test_run_command_local_executor_routes_through_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()

    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "local",
            "id": "local-test",
            "network": "host",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("run_command", {"cmd": [sys.executable, "-c", "print('executor-ok')"]})

    assert "executor-ok" in result
    assert "EXECUTOR_TRACE" in result
    assert '"executor_id": "local-test"' in result


def test_run_command_executor_trace_redacts_secret_like_args(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    secret = "OPENAI_API_KEY=sk-secrettraceabcdefghijk123456"
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "local",
            "id": "local-redact",
            "network": "host",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("run_command", {"cmd": [sys.executable, "-c", "print('ok')", secret]})

    assert "EXECUTOR_TRACE" in result
    assert "ok" in result
    assert secret not in result
    assert "***REDACTED***" in result


def test_run_command_with_executor_ref_uses_local_for_unmapped_task_drive_cwd(tmp_path, monkeypatch):
    from ouroboros.tool_access import resource_root_path

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="task-drive-cwd",
        executor_ref={
            "type": "local",
            "id": "local-executor",
            "network": "host",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    task_drive = resource_root_path(ctx, "task_drive")
    task_drive.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "from pathlib import Path; Path('local.txt').write_text('ok'); print('local-cwd')"],
            "cwd": str(task_drive),
        },
    )

    assert "local-cwd" in result
    assert "EXECUTOR_TRACE" not in result
    assert (task_drive / "local.txt").read_text(encoding="utf-8") == "ok"


def test_run_script_with_docker_executor_ref_uses_local_for_unmapped_task_drive_cwd(tmp_path, monkeypatch):
    from ouroboros.tool_access import resource_root_path

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="script-task-drive",
        executor_ref={
            "type": "docker_exec",
            "id": "docker-executor",
            "container_name": "benchmark-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    task_drive = resource_root_path(ctx, "task_drive")
    task_drive.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_script",
        {
            "interpreter": "python3",
            "script": "from pathlib import Path; Path('script-local.txt').write_text('ok'); print('script-local')",
            "cwd": str(task_drive),
        },
    )

    assert "script-local" in result
    assert "EXECUTOR_TRACE" not in result
    assert (task_drive / "script-local.txt").read_text(encoding="utf-8") == "ok"


def test_run_script_external_workspace_uses_workspace_temp_script_path(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="workspace-script",
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_script",
        {
            "interpreter": "python3",
            "script": "from pathlib import Path; Path('script-workspace.txt').write_text('ok'); print('workspace-script')",
            "cwd": str(workspace),
        },
    )

    assert "workspace-script" in result
    assert f"# script_path={workspace / '.ouroboros' / 'tmp_scripts'}" in result
    assert (workspace / "script-workspace.txt").read_text(encoding="utf-8") == "ok"
    assert not (workspace / ".ouroboros").exists()


def test_run_script_external_workspace_registers_changed_directory_output(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="workspace-dir-output",
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_script",
        {
            "interpreter": "python3",
            "script": "from pathlib import Path; Path('site').mkdir(); Path('site/index.html').write_text('<h1>ok</h1>')",
            "cwd": str(workspace),
            "outputs": ["site"],
        },
    )

    assert "ARTIFACT_OUTPUTS" in result
    assert "registered directory output" in result
    artifact_dir = data / "task_results" / "artifacts" / "workspace-dir-output"
    assert list(artifact_dir.glob("site.*.manifest.json"))
    assert list(artifact_dir.glob("site.*.zip"))


def test_run_script_directory_output_blocks_sensitive_members(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="workspace-sensitive-dir-output",
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_script",
        {
            "interpreter": "python3",
            "script": "from pathlib import Path; Path('site/.ssh').mkdir(parents=True); Path('site/.ssh/config').write_text('x')",
            "cwd": str(workspace),
            "outputs": ["site"],
        },
    )

    assert "ARTIFACT_OUTPUT_ERROR" in result
    assert "hidden/control output path component .ssh" in result
    artifact_dir = data / "task_results" / "artifacts" / "workspace-sensitive-dir-output"
    assert not list(artifact_dir.glob("site.*.zip"))


def test_run_command_external_workspace_unchanged_directory_output_is_cosmetic(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    (workspace / "site").mkdir()
    (workspace / "site" / "index.html").write_text("<h1>old</h1>", encoding="utf-8")
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="workspace-unchanged-dir-output",
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "print('noop')"],
            "cwd": str(workspace),
            "outputs": ["site"],
        },
    )

    # C2 (v6.36.0): present-but-unchanged is a cosmetic note, NOT a blocking error
    # (a deterministic re-run / re-verify is not a failure). Missing outputs still block.
    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert "unchanged output (cosmetic): site" in result


def test_run_command_executor_failure_keeps_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "local",
            "id": "local-fail",
            "network": "host",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("run_command", {"cmd": [sys.executable, "-c", "import sys; sys.exit(7)"]})

    assert "SHELL_EXIT_ERROR" in result
    assert "EXECUTOR_TRACE" in result
    assert '"executor_id": "local-fail"' in result


def test_executor_workspace_still_enforces_protected_artifact_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    if os.name == "nt":
        executable = workspace / "executable.cmd"
        execute_cmd = ["cmd.exe", "/c", str(executable)]
        executable.write_text("@echo reference-ok\r\n", encoding="utf-8")
    else:
        executable = workspace / "executable"
        execute_cmd = ["./executable"]
        executable.write_text("#!/bin/sh\necho reference-ok\n", encoding="utf-8")
    executable.chmod(0o700)

    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_contract={
            "resource_policy": {
                "protected_artifacts": [
                    {
                        "id": "reference",
                        "role": "black_box_reference",
                        "paths": [executable.name],
                        "allow": ["execute"],
                        "deny": ["read_bytes", "hash", "static_introspection", "dynamic_trace", "debug"],
                    }
                ]
            }
        },
        executor_ref={
            "type": "local",
            "id": "local-protected",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    read_block = registry.execute("run_command", {"cmd": ["cat", executable.name]})
    execute_allowed = registry.execute("run_command", {"cmd": execute_cmd})

    assert "RESOURCE_POLICY_BLOCKED" in read_block
    assert "EXECUTOR_TRACE" not in read_block
    assert "reference-ok" in execute_allowed
    assert "EXECUTOR_TRACE" in execute_allowed


def test_docker_executor_protected_artifact_policy_matches_host_and_backend_spellings(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    (workspace / "executable").write_text("black-box bytes\n", encoding="utf-8")

    def registry_for_policy(path_text: str) -> ToolRegistry:
        ctx = ToolContext(
            repo_dir=system_repo,
            drive_root=data,
            workspace_root=workspace,
            workspace_mode="external",
            task_contract={
                "resource_policy": {
                    "protected_artifacts": [
                        {
                            "id": "reference",
                            "role": "black_box_reference",
                            "paths": [path_text],
                            "allow": ["execute"],
                            "deny": ["read_bytes", "hash", "static_introspection", "dynamic_trace", "debug"],
                        }
                    ]
                }
            },
            executor_ref={
                "type": "docker_exec",
                "id": "pb-container",
                "container_name": "pb-container",
                "network": "none",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
            },
        )
        registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
        registry.set_context(ctx)
        return registry

    backend_policy_host_arg = registry_for_policy("/workspace/executable").execute("run_command", {"cmd": ["cat", "executable"]})
    relative_policy_backend_arg = registry_for_policy("executable").execute("run_command", {"cmd": ["cat", "/workspace/executable"]})
    backend_policy_interpreter_arg = registry_for_policy("/workspace/executable").execute(
        "run_command",
        {"cmd": ["python3", "-c", "open('executable','rb').read()"]},
    )

    assert "RESOURCE_POLICY_BLOCKED" in backend_policy_host_arg
    assert "RESOURCE_POLICY_BLOCKED" in relative_policy_backend_arg
    assert "RESOURCE_POLICY_BLOCKED" in backend_policy_interpreter_arg


def test_executor_workspace_blocks_claude_code_edit_host_path_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-testexecutorblockedabcdefghijk")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "docker_exec",
            "id": "docker-claude-block",
            "container_name": "benchmark-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("claude_code_edit", {"prompt": "edit something"})

    assert "CLAUDE_CODE_EDIT_BLOCKED" in result
    assert "ANTHROPIC_API_KEY" not in result


def test_executor_workspace_allows_claude_code_edit_unmapped_task_drive_cwd_to_reach_normal_auth_gate(tmp_path, monkeypatch):
    from ouroboros.tool_access import resource_root_path

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="claude-task-drive",
        executor_ref={
            "type": "docker_exec",
            "id": "docker-claude-task-drive",
            "container_name": "benchmark-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    task_drive = resource_root_path(ctx, "task_drive")
    task_drive.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("claude_code_edit", {"prompt": "edit something", "cwd": str(task_drive)})

    assert "CLAUDE_CODE_EDIT_BLOCKED" not in result
    assert "CLAUDE_CODE_UNAVAILABLE" in result or "CAPABILITY_UNAVAILABLE" in result


def test_executor_workspace_allows_claude_code_edit_unmapped_user_files_cwd_to_reach_normal_auth_gate(tmp_path, monkeypatch):
    from ouroboros.tool_access import resource_root_path

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="claude-user-files",
        executor_ref={
            "type": "docker_exec",
            "id": "docker-claude-user-files",
            "container_name": "benchmark-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    user_files = resource_root_path(ctx, "user_files")
    user_files.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("claude_code_edit", {"prompt": "edit something", "cwd": str(user_files)})

    assert "CLAUDE_CODE_EDIT_BLOCKED" not in result
    assert "CLAUDE_CODE_UNAVAILABLE" in result or "CAPABILITY_UNAVAILABLE" in result


def test_claude_code_edit_schema_documents_docker_executor_workspace_block():
    from ouroboros.tools.shell import get_tools

    tool = next(entry for entry in get_tools() if entry.name == "claude_code_edit")
    cwd_description = tool.schema["parameters"]["properties"]["cwd"]["description"]

    assert "docker executor-backed external workspaces" in cwd_description
    assert "mapped active_workspace cwd is blocked" in cwd_description
    assert "task_drive" in cwd_description
    assert "artifact_store" in cwd_description


def test_executor_local_service_lifecycle_hides_private_snapshot(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    import ouroboros.workspace_executor as workspace_executor

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    bootstrap_calls: list[str] = []
    monkeypatch.setattr(workspace_executor, "bootstrap_process_path", lambda: bootstrap_calls.append("bootstrap"))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-test",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    started = json.loads(
        registry.execute(
            "start_service",
            {
                "name": "svc",
                "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(30)"],
                "readiness": {"log_contains": "READY", "timeout_sec": 5},
            },
        )
    )
    status = json.loads(registry.execute("service_status", {"name": "svc"}))
    logs = json.loads(registry.execute("service_logs", {"name": "svc", "tail": 1000}))
    stopped_raw = registry.execute("stop_service", {"name": "svc"})
    stopped = json.loads(stopped_raw)

    assert started["ready"] is True
    assert status["state"] == "running"
    assert "READY" in logs["tail"]
    assert stopped["state"] == "stopped"
    assert "_before_outputs" not in stopped_raw
    assert bootstrap_calls


def test_start_service_with_executor_ref_uses_local_for_unmapped_task_drive_cwd(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    from ouroboros.tool_access import resource_root_path

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-task-drive",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    task_drive = resource_root_path(ctx, "task_drive")
    task_drive.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    started = json.loads(
        registry.execute(
            "start_service",
            {
                "name": "svc",
                "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(30)"],
                "cwd": str(task_drive),
                "readiness": {"log_contains": "READY", "timeout_sec": 5},
            },
        )
    )
    status = json.loads(registry.execute("service_status", {"name": "svc"}))
    logs = json.loads(registry.execute("service_logs", {"name": "svc", "tail": 1000}))
    stopped = json.loads(registry.execute("stop_service", {"name": "svc"}))

    assert "executor" not in started
    assert started["cwd_root"] == "task_drive"
    assert status["state"] == "running"
    assert "READY" in logs["tail"]
    assert stopped["state"] == "exited"


def test_executor_local_service_can_restart_after_exit(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-restart",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    first = json.loads(registry.execute("start_service", {"name": "short", "cmd": [sys.executable, "-c", "print('one')"]}))
    import time

    time.sleep(0.5)
    second = json.loads(registry.execute("start_service", {"name": "short", "cmd": [sys.executable, "-c", "print('two')"]}))

    assert first["backend_pid"] != second["backend_pid"]
    assert second.get("note") != "already_running"
    records = list((data / "state" / "workspace_executor_processes").glob("*.json"))
    assert len(records) == 1
    durable = json.loads(records[0].read_text(encoding="utf-8"))
    assert str(durable["host_pid"]) == str(second["backend_pid"])
    assert str(durable["host_pid"]) != str(first["backend_pid"])


def test_executor_local_service_sanitizes_env_and_redacts_logs(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-executor-service")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-env",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    registry.execute(
        "start_service",
        {
            "name": "svc",
            "cmd": [
                sys.executable,
                "-c",
                "import os, time; print(os.environ.get('OPENROUTER_API_KEY','missing'), flush=True); time.sleep(30)",
            ],
            "readiness": {"log_contains": "missing", "timeout_sec": 5},
        },
    )
    logs = json.loads(registry.execute("service_logs", {"name": "svc", "tail": 1000}))
    registry.execute("stop_service", {"name": "svc"})

    assert "missing" in logs["tail"]
    assert "sk-secret-executor-service" not in logs["tail"]


def test_executor_service_status_and_durable_record_redact_secret_like_args(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    secret = "OPENAI_API_KEY=sk-secretservicetraceabcdefghijk123456"
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-redact",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    registry.execute(
        "start_service",
        {
            "name": "svc",
            "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(30)", secret],
            "readiness": {"log_contains": "READY", "timeout_sec": 5},
        },
    )
    try:
        status_raw = registry.execute("service_status", {"name": "svc"})
        records = list((data / "state" / "workspace_executor_processes").glob("*.json"))
        durable_text = "\n".join(path.read_text(encoding="utf-8") for path in records)
    finally:
        registry.execute("stop_service", {"name": "svc"})

    assert secret not in status_raw
    assert secret not in durable_text
    assert "***REDACTED***" in status_raw
    assert "***REDACTED***" in durable_text


def test_executor_services_participate_in_task_and_global_cleanup(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    from ouroboros.tools.services import kill_all_services, stop_task_services
    import ouroboros.workspace_executor as workspace_executor

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    workspace_executor._SERVICES.clear()
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-cleanup",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    registry.execute("start_service", {"name": "tasksvc", "cmd": [sys.executable, "-c", "import time; time.sleep(30)"]})
    stopped = stop_task_services(ctx)
    assert any(item.get("name") == "tasksvc" for item in stopped)
    assert workspace_executor.service_status(ctx, "tasksvc") is None

    registry.execute("start_service", {"name": "globalsvc", "cmd": [sys.executable, "-c", "import time; time.sleep(30)"]})
    killed = kill_all_services(data)
    assert any(item.get("name") == "globalsvc" for item in killed)
    assert workspace_executor.service_status(ctx, "globalsvc") is None


def test_executor_keep_alive_service_survives_task_teardown(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    import ouroboros.workspace_executor as workspace_executor
    from ouroboros.tools.services import kill_all_services, stop_task_services

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    workspace_executor._SERVICES.clear()
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="svc-keep",
        executor_ref={
            "type": "local",
            "id": "local-service",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    registry.execute("start_service", {
        "name": "keptsvc",
        "cmd": [sys.executable, "-c", "import time; time.sleep(30)"],
        "keep_alive": True,
    })
    finalized = stop_task_services(ctx)
    assert finalized[0]["name"] == "keptsvc"
    assert finalized[0]["lifecycle"] == "kept"
    assert workspace_executor.service_status(ctx, "keptsvc") is not None

    killed = kill_all_services(data)
    assert any(item.get("name") == "keptsvc" for item in killed)
    assert workspace_executor.service_status(ctx, "keptsvc") is None


def test_executor_panic_cleanup_kills_durable_foreground_and_service_processes(tmp_path):
    import time
    import ouroboros.workspace_executor as workspace_executor
    from ouroboros.platform_layer import subprocess_new_group_kwargs

    data = tmp_path / "data"
    data.mkdir()
    foreground = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **subprocess_new_group_kwargs(),
    )
    service = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **subprocess_new_group_kwargs(),
    )
    try:
        workspace_executor._register_process(
            data,
            {
                "record_type": "foreground",
                "executor_type": "local",
                "executor_id": "local-foreground",
                "host_pid": foreground.pid,
            },
        )
        workspace_executor._register_process(
            data,
            {
                "record_type": "service",
                "service_id": "task:svc",
                "task_id": "task",
                "name": "svc",
                "executor_type": "local",
                "executor_id": "local-service",
                "host_pid": service.pid,
            },
        )

        killed_foreground = workspace_executor.kill_all_foreground(data, wait=False)
        killed_services = workspace_executor.kill_all_services(data, wait=False)

        deadline = time.time() + 15
        while time.time() < deadline and (foreground.poll() is None or service.poll() is None):
            time.sleep(0.05)
        assert foreground.poll() is not None
        assert service.poll() is not None
        assert any(item.get("executor_type") == "local" for item in killed_foreground)
        assert any(item.get("service_id") == "task:svc" for item in killed_services)
        assert not list((data / "state" / "workspace_executor_processes").glob("*.json"))
    finally:
        for proc in (foreground, service):
            if proc.poll() is None:
                proc.kill()


def test_executor_cleanup_scans_child_drive_records_from_parent_data_root(tmp_path):
    import time
    import ouroboros.workspace_executor as workspace_executor
    from ouroboros.platform_layer import subprocess_new_group_kwargs

    data = tmp_path / "data"
    child_data = data / "state" / "headless_tasks" / "task-1" / "data"
    child_data.mkdir(parents=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **subprocess_new_group_kwargs(),
    )
    # kill_all_foreground's PID-reuse safety check compares the process command-sha recorded at
    # registration against the one it recomputes at kill time. Right after fork+exec the child's
    # command line may not yet be readable, so registering too eagerly makes the two shas diverge
    # and the kill is silently skipped (flaky). Wait until the command-sha is readable & stable.
    for _ in range(200):
        if workspace_executor._process_command_sha256(proc.pid):
            break
        time.sleep(0.02)
    try:
        workspace_executor._register_process(
            child_data,
            {
                "record_type": "foreground",
                "executor_type": "local",
                "executor_id": "child-local",
                "host_pid": proc.pid,
            },
        )
        killed = workspace_executor.kill_all_foreground(data, wait=False)
        deadline = time.time() + 15
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None
        assert any(item.get("executor_type") == "local" for item in killed)
        assert not list((child_data / "state" / "workspace_executor_processes").glob("*.json"))
    finally:
        if proc.poll() is None:
            proc.kill()


def test_docker_executor_stop_failure_preserves_service_handle(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    ctx = ToolContext(
        repo_dir=tmp_path / "repo",
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="docker-stop",
        executor_ref={
            "type": "docker_exec",
            "id": "pb-container",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    workspace_executor._SERVICES.clear()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "nohup" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="permission denied")
        if cmd[:2] == ["docker", "exec"] and "kill -0" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    workspace_executor.start_service(
        ctx,
        name="svc",
        cmd=["sleep", "30"],
        host_cwd=workspace,
        cwd_root="active_workspace",
        readiness={},
        outputs=[],
        before_outputs={},
    )
    failed = workspace_executor.stop_service(ctx, "svc")

    assert failed and failed["stop_failed"] is True
    assert "permission denied" in failed["stop_error"]
    assert workspace_executor.service_status(ctx, "svc") is not None


def test_docker_executor_service_shell_uses_process_group_stop():
    from ouroboros.workspace_executor import _docker_service_start_shell, _docker_service_stop_shell

    record = SimpleNamespace(cmd=["python3", "-c", "import time; time.sleep(30)"], backend_cwd="/workspace")

    start_shell = _docker_service_start_shell(record, "/tmp/ouroboros-service-test.log")
    stop_shell = _docker_service_stop_shell("12345")

    assert "setsid" in start_shell
    assert "sh -c 'exec python3" in start_shell
    assert "& echo $!" in start_shell
    assert "kill -TERM -$pid" in stop_shell
    assert "kill -KILL -$pid" in stop_shell


def test_docker_executor_run_script_uses_backend_script_path(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    import ouroboros.tools.shell as shell_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    captured: dict[str, object] = {}

    def fake_execute(ctx, cmd, cwd, timeout_sec):
        captured["cmd"] = list(cmd)
        captured["cwd"] = str(cwd)
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="", backend_trace={"executor_id": "pb-container"}, args=list(cmd))

    monkeypatch.setattr(shell_mod, "executor_execute", fake_execute)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "docker_exec",
            "id": "pb-container",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("run_script", {"script": "print('ok')", "interpreter": "python3"})

    assert "ok" in result
    assert captured["cmd"][1].startswith("/workspace/.ouroboros/tmp_scripts/script_")
    assert not str(captured["cmd"][1]).startswith(str(workspace))


def test_docker_executor_accepts_backend_absolute_write_targets_and_outputs(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    import ouroboros.tools.shell as shell_mod

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(system_repo)
    _init_repo(workspace)
    data.mkdir()
    captured: dict[str, object] = {}

    def fake_execute(ctx, cmd, cwd, timeout_sec):
        captured["cmd"] = list(cmd)
        (workspace / "backend-output.txt").write_text("ok\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="wrote\n", stderr="", backend_trace={"executor_id": "pb-container"}, args=list(cmd))

    monkeypatch.setattr(shell_mod, "executor_execute", fake_execute)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="backend-output",
        executor_ref={
            "type": "docker_exec",
            "id": "pb-container",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute(
        "run_command",
        {
            "cmd": ["sh", "-c", "printf ok > /workspace/backend-output.txt"],
            "outputs": ["/workspace/backend-output.txt"],
        },
    )

    assert "WORKSPACE_SHELL_BLOCKED" not in result
    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert "backend-output.txt" in result
    assert captured["cmd"] == ["sh", "-c", "printf ok > /workspace/backend-output.txt"]


def test_docker_executor_enforces_network_none_before_exec(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolContext(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "docker_exec",
            "id": "pb-container",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        raise AssertionError(cmd)

    class FakePopen:
        pid = 999991
        returncode = 0

        def __init__(self, cmd, **kwargs):
            calls.append([str(part) for part in cmd])
            self.args = cmd

        def communicate(self, timeout=None):
            return "ok\n", ""

    import ouroboros.workspace_executor as workspace_executor

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    monkeypatch.setattr(workspace_executor.subprocess, "Popen", FakePopen)
    result = execute(ctx, ["echo", "ok"], workspace, 30)

    assert result.returncode == 0
    assert result.stdout == "ok\n"
    assert calls[0][:4] == ["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}"]
    assert calls[1][:4] == ["docker", "exec", "--workdir", "/workspace"]


def test_docker_executor_timeout_cleans_backend_process(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolContext(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "docker_exec",
            "id": "pb-container",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -TERM -$pid" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    class FakePopen:
        pid = 999992
        returncode = None

        def __init__(self, cmd, **kwargs):
            calls.append([str(part) for part in cmd])
            self.args = cmd

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(self.args, timeout=timeout)

        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode

    import ouroboros.workspace_executor as workspace_executor

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    monkeypatch.setattr(workspace_executor.subprocess, "Popen", FakePopen)
    with pytest.raises(subprocess.TimeoutExpired):
        execute(ctx, ["sleep", "30"], workspace, 1)

    assert any("cat /tmp/ouroboros-exec-" in call[-1] for call in calls if call[:2] == ["docker", "exec"])
    assert any("kill -TERM -$pid" in call[-1] for call in calls if call[:2] == ["docker", "exec"])


def test_docker_executor_rejects_network_none_when_container_has_network(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ref = normalize_executor_ref(
        {
            "type": "docker_exec",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        }
    )
    assert ref is not None
    ctx = ToolContext(
        repo_dir=tmp_path / "repo",
        drive_root=tmp_path / "data",
        workspace_root=workspace,
        workspace_mode="external",
        executor_ref={
            "type": "docker_exec",
            "container_name": "pb-container",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="bridge\n", stderr="")

    import ouroboros.workspace_executor as workspace_executor

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    try:
        execute(ctx, ["echo", "ok"], workspace, 30)
    except RuntimeError as exc:
        assert "NetworkMode=none" in str(exc)
    else:  # pragma: no cover - kept explicit for failure readability
        raise AssertionError("docker network mismatch was not rejected")


def test_api_task_metadata_accepts_normalized_executor_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks
    import supervisor.queue as queue

    captured: dict[str, object] = {}

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(tmp_path / "workspace"),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {
                "type": "local",
                "id": "local-api",
                "workspace_host_path": str(tmp_path / "workspace"),
                "workspace_backend_path": "/workspace",
            },
        }

    def fake_enqueue(task):
        captured.update(task)

    _init_repo(tmp_path / "workspace")
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: tmp_path / "data")
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: tmp_path / "repo")
    monkeypatch.setattr(queue, "enqueue_task", fake_enqueue)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda *a, **k: None)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert body["ok"] is True
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["executor_ref"]["type"] == "local"
    assert metadata["executor_ref"]["id"] == "local-api"
    assert metadata["executor_ref"]["workspace_backend_path"] == "/workspace"
    assert metadata["executor_ref"]["path_mappings"][0]["host_path"] == str((tmp_path / "workspace").resolve(strict=False))


def test_api_task_rejects_executor_ref_without_external_workspace(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "executor_ref": {"type": "local", "workspace_host_path": str(tmp_path), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: tmp_path / "data")
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: tmp_path / "repo")
    (tmp_path / "data").mkdir()
    (tmp_path / "repo").mkdir()

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "executor_ref requires an external workspace_root" in body["error"]


def test_api_task_rejects_empty_executor_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    _init_repo(workspace)
    repo.mkdir()
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "executor_ref must be a JSON object" in body["error"]


def test_api_task_rejects_executor_ref_mapping_to_system_repo(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"type": "local", "workspace_host_path": str(repo), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "must not overlap the Ouroboros system repo" in body["error"]


def test_api_task_rejects_executor_ref_mapping_to_data_drive(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"type": "local", "workspace_host_path": str(data), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "must not overlap the Ouroboros data drive" in body["error"]


def test_api_task_rejects_executor_ref_not_covering_workspace(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    other = tmp_path / "other"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    other.mkdir()
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"type": "local", "workspace_host_path": str(other), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "mappings must cover workspace_root" in body["error"]


def test_api_task_rejects_reserved_executor_metadata_aliases(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "metadata": {"workspace_executor": {"type": "local"}},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "metadata.executor_ref/workspace_executor is reserved" in body["error"]


def test_api_task_rejects_reserved_executor_metadata_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "metadata": {"executor_ref": {"type": "local"}},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "metadata.executor_ref/workspace_executor is reserved" in body["error"]


def test_api_task_rejects_local_network_none(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {
                "type": "local",
                "network": "none",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
            },
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "local executor_ref cannot enforce network=none" in body["error"]


def test_api_task_rejects_malformed_executor_mapping_entry(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {
                "type": "local",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
                "path_mappings": [{"host_path": str(tmp_path / "missing_backend")}],
            },
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "path_mappings entries require host_path and backend_path" in body["error"]


def test_api_task_rejects_malformed_executor_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    _init_repo(workspace)
    repo.mkdir()
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "executor_ref.type is required" in body["error"]
