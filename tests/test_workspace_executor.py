"""Routing a workspace command or script through an executor backend.

This module owns the cross-platform absolute-path predicate and the shell path-token
extractor the routing depends on, the executor-reference normalization and backend-path
mapping, the run_command/run_script paths — including the local fallback for an
unmapped task drive, the temp script path and directory outputs of an external
workspace, the redacted trace and the trace a failure keeps — and the protected-artifact
policy that holds on host and backend alike.

Services, the docker backend and the ``POST /api/tasks`` admission of an executor
reference were split verbatim into ``tests/test_workspace_executor_services.py``,
``tests/test_workspace_executor_docker.py`` and
``tests/test_workspace_executor_admission.py``; the repository builder they share lives
in ``tests/_workspace_executor_shared.py``.

Whole-file serial suite: it spawns real processes, so ``tests/conftest.py`` tags this
module and its three siblings ``serial`` and the parallel pass excludes them.
"""
from __future__ import annotations

import os
import sys

import pytest

from ouroboros.shell_parse import is_absolute_path_text, shell_argv_with_path_tokens
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.workspace_executor import map_backend_path, normalize_executor_ref

from tests._workspace_executor_shared import _init_repo


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
            "script": (
                "from pathlib import Path; Path('site').mkdir(); "
                "Path('site/index.html').write_text('<h1>ok</h1>'); "
                "Path('site/id_rsa').write_text('SECRETKEY')"
            ),
            "cwd": str(workspace),
            "outputs": ["site"],
        },
    )

    # D4 (capinv-447): a credential-shaped MEMBER is skipped with a receipt;
    # the rest of the declared directory still exports (per-member, not atomic).
    assert "ARTIFACT_OUTPUTS" in result
    assert "registered directory output" in result
    assert "skipped 1 member(s)" in result
    assert "id_rsa" in result
    assert "credential filename" in result
    artifact_dir = data / "task_results" / "artifacts" / "workspace-sensitive-dir-output"
    zips = list(artifact_dir.glob("site.*.zip"))
    assert zips
    import zipfile as _zipfile

    names = _zipfile.ZipFile(zips[0]).namelist()
    assert any(name.endswith("index.html") for name in names)
    assert not any("id_rsa" in name for name in names)


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


def test_overlay_env_is_case_aware():
    """T17 pin: the shared local-env merge replaces a Windows case-variant key
    instead of duplicating it, and stays exact-match on POSIX."""
    from ouroboros import workspace_executor as wx

    base = {"Path": "C:/old", "HOME": "/h"}
    overlay = {"PATH": "/bundle:C:/old"}
    real = wx.IS_WINDOWS
    try:
        wx.IS_WINDOWS = True
        merged = wx.overlay_env(base, overlay)
        assert merged["PATH"] == "/bundle:C:/old" and "Path" not in merged
        wx.IS_WINDOWS = False
        merged = wx.overlay_env(base, overlay)
        assert merged["Path"] == "C:/old" and merged["PATH"] == "/bundle:C:/old"
    finally:
        wx.IS_WINDOWS = real
    assert wx.overlay_env(base, None) == base


def test_start_service_local_branch_uses_case_aware_overlay(tmp_path, monkeypatch):
    """T19 pin: the LOCAL start_service branch merges env_overlay through the
    shared case-aware helper — a stale case-variant Path never survives next to
    the attested PATH prepend (grok A-F finding)."""
    from types import SimpleNamespace

    from ouroboros import workspace_executor as wx

    captured = {}

    def fake_spawn(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(pid=4242, poll=lambda: None)

    monkeypatch.setattr("ouroboros.process_custody.spawn_supervised", fake_spawn)
    monkeypatch.setattr(wx, "IS_WINDOWS", True)
    monkeypatch.setenv("Path", "C:/stale")
    monkeypatch.delenv("PATH", raising=False)
    ctx = SimpleNamespace(
        task_id="svc-overlay",
        drive_root=tmp_path,
        executor_ref={
            "type": "local",
            "id": "local-overlay",
            "workspace_host_path": str(tmp_path),
            "workspace_backend_path": "/workspace",
        },
    )
    payload = wx.start_service(
        ctx,
        name="svc",
        cmd=["node", "server.js"],
        host_cwd=tmp_path,
        cwd_root="task_drive",
        readiness={},
        outputs=[],
        before_outputs={},
        env_overlay={"PATH": "/bundle/bin:C:/stale"},
    )
    env = captured["env"]
    assert env["PATH"] == "/bundle/bin:C:/stale"
    assert "Path" not in env
    assert payload.get("state") in ("running", "ready", "started", None) or payload
