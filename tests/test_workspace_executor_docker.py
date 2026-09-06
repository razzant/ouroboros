"""The docker executor backend: paths, network fence, timeouts and stop failures.

Split verbatim out of ``tests/test_workspace_executor.py`` by theme. This module owns
the service handle a failed stop must preserve, the process-group stop the service shell
uses, the backend script path and backend-absolute write targets, the ``network=none``
enforced before exec and refused when the container already has a network, and the
backend process a timeout cleans up.

Whole-file serial suite: it spawns real processes, so ``tests/conftest.py`` tags it
``serial`` and the parallel pass excludes it.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.workspace_executor import execute, normalize_executor_ref

from tests._workspace_executor_shared import _init_repo


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

    def fake_execute(ctx, cmd, cwd, timeout_sec, env_overlay=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = str(cwd)
        captured["env_overlay"] = env_overlay
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
    assert captured["env_overlay"] is None  # no node emergency → env untouched


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

    def fake_execute(ctx, cmd, cwd, timeout_sec, env_overlay=None):
        captured["cmd"] = list(cmd)
        captured["env_overlay"] = env_overlay
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
    assert captured["env_overlay"] is None  # no node emergency → env untouched


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


def test_docker_executor_stop_success_without_terminal_kill_preserves_handle(tmp_path, monkeypatch):
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
        task_id="docker-stop-race",
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

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "nohup" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            # The stop shell itself returned success, but the subsequent
            # kill-0 confirmation still observes a live backend.
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
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
    assert "kill-0" in failed["stop_error"]
    assert workspace_executor.service_status(ctx, "svc") is not None


def test_docker_executor_stop_unknown_probe_preserves_handle(tmp_path, monkeypatch):
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
        task_id="docker-stop-unknown",
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

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "nohup" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -0" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="daemon unavailable")
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
    assert "unknown" in failed["stop_error"]
    assert failed["state"] == "unknown"
    assert workspace_executor.service_status(ctx, "svc") is not None


def test_docker_executor_stop_state_exception_preserves_handle(tmp_path, monkeypatch):
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
        task_id="docker-stop-exception",
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

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "nohup" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    monkeypatch.setattr(workspace_executor, "_service_state", lambda _record: (_ for _ in ()).throw(RuntimeError("probe boom")))
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
    assert failed["state"] == "unknown"
    assert workspace_executor.service_status(ctx, "svc") is not None


def test_docker_executor_global_cleanup_unknown_state_keeps_handle(tmp_path, monkeypatch):
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
        task_id="docker-cleanup-unknown",
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

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="none\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "nohup" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -0" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="daemon unavailable")
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

    result = workspace_executor.kill_all_services(data)

    current = next(item for item in result if item.get("name") == "svc")
    assert current["cleanup_dispatched"] is False
    assert current["stop_failed"] is True
    assert current["state"] == "unknown"
    assert workspace_executor.service_status(ctx, "svc") is not None


def test_docker_durable_cleanup_keeps_record_until_kill_zero_terminal(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    data.mkdir()
    workspace_executor._SERVICES.clear()
    path = workspace_executor._register_process(
        data,
        {
            "record_type": "service",
            "executor_type": "docker_exec",
            "executor_id": "pb-container",
            "container_name": "pb-container",
            "backend_pid": "12345",
            "service_id": "task:durable",
            "task_id": "task",
            "name": "durable",
        },
    )
    assert path is not None

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -0" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    result = workspace_executor._kill_durable_service_records(data)

    assert result[0]["state"] == "cleanup_pending"
    assert result[0]["cleanup_dispatched"] is False
    assert path.exists()


def test_docker_durable_cleanup_keeps_record_on_unknown_kill_zero(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    data.mkdir()
    workspace_executor._SERVICES.clear()
    path = workspace_executor._register_process(
        data,
        {
            "record_type": "service",
            "executor_type": "docker_exec",
            "executor_id": "pb-container",
            "container_name": "pb-container",
            "backend_pid": "12345",
            "service_id": "task:durable-unknown",
            "task_id": "task",
            "name": "durable-unknown",
        },
    )
    assert path is not None

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "exec"] and "kill -TERM" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["docker", "exec"] and "kill -0" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="daemon unavailable")
        raise AssertionError(cmd)

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    result = workspace_executor._kill_durable_service_records(data)

    assert result[0]["state"] == "cleanup_pending"
    assert result[0]["cleanup_dispatched"] is False
    assert path.exists()
