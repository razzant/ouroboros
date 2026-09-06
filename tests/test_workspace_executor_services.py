"""Executor-backed services: their lifecycle, their records and their teardown.

Split verbatim out of ``tests/test_workspace_executor.py`` by theme. This module owns
the private snapshot a local service hides, the restart after exit, the sanitized env
and redacted logs, the status and durable record that redact secret-like arguments, the
task and global cleanup they participate in, the keep-alive that survives task
teardown, the panic cleanup that kills durable foreground and service processes, and
the child-drive records a parent data root must scan.

Whole-file serial suite: it spawns real processes, so ``tests/conftest.py`` tags it
``serial`` and the parallel pass excludes it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry

from tests._workspace_executor_shared import _init_repo


@pytest.mark.parametrize("executor_kind", [None, "local"])
def test_real_service_observes_exact_selected_env_and_cwd(tmp_path, monkeypatch, executor_kind):
    import gzip
    from ouroboros.tools import services

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("UNSELECTED_660_SECRET", "synthetic-host-only")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    workspace = tmp_path / "project with spaces"
    workspace.mkdir()
    cwd = workspace / "chosen cwd"
    cwd.mkdir()
    script = workspace / "observe.py"
    script.write_text('''import json, os, pathlib, time
observed = {"cwd": os.getcwd(), "token": os.environ["TOKEN"],
    "empty": os.environ["EMPTY"], "proxy": os.environ["HTTPS_PROXY"],
    "unselected": os.environ.get("UNSELECTED_660_SECRET")}
pathlib.Path("observed.json").write_text(json.dumps(observed))
print("TOKEN=" + os.environ["TOKEN"], flush=True)
print("READY", flush=True)
time.sleep(30)
''', encoding="utf-8")
    env = {"TOKEN": 'synthetic-660-quote"\\tail\nprivate-fragment-660', "EMPTY": "",
           "HTTPS_PROXY": "http://synthetic-user:synthetic-pass@127.0.0.1:7777",
           "DEPLOYMENT_MODE": "running", "FIELD_NAME": "state"}
    monkeypatch.setattr(services, "load_settings", lambda: {"TEST_SERVICE_KEY": env["TOKEN"]})
    registry = ToolRegistry(repo_dir=tmp_path / "system", drive_root=tmp_path / "data")
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=tmp_path / "data",
                      workspace_root=workspace, workspace_mode="external", task_id="selected-env")
    if executor_kind:
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                            "workspace_backend_path": str(workspace)}
    registry.set_context(ctx)
    try:
        started = registry.execute("start_service", {"name": "selected", "cmd": [sys.executable, str(script)],
            "cwd": str(cwd), "env": {key: value for key, value in env.items() if key != "TOKEN"},
            "env_from_settings": {"TOKEN": "TEST_SERVICE_KEY"},
            "readiness": {"log_contains": "READY", "timeout_sec": 3}})
        assert json.loads(started)["ready"]
        assert json.loads(started)["state"] == "running"
        observed = json.loads((cwd / "observed.json").read_text())
        assert observed == {"cwd": str(cwd), "token": env["TOKEN"], "empty": "",
                            "proxy": env["HTTPS_PROXY"], "unselected": None}
        logs = registry.execute("service_logs", {"name": "selected"})
        status = registry.execute("service_status", {"name": "selected"})
        assert json.loads(status)["state"] == "running"
        assert env["TOKEN"] not in started + logs + status
        assert json.dumps(env["TOKEN"])[1:-1] not in logs
        assert "private-fragment-660" not in logs
        assert "***" in json.loads(logs)["tail"]
        if not executor_kind:
            ref = json.loads(logs)["full_log_ref"]
            assert env["TOKEN"] not in gzip.decompress(Path(ref["path"]).read_bytes()).decode()
    finally:
        stopped = json.loads(services._stop_service(ctx, "selected"))
    finalization = stopped["log_finalization"]
    assert finalization["deleted_live_log"] and not finalization["errors"]
    final = gzip.decompress(Path(finalization["full_log_ref"]["path"]).read_bytes()).decode()
    assert "private-fragment-660" not in final and "READY" in final
    assert services.prune_service_logs(ctx.drive_root, retention_days=0)["archived_files"] == 0


def test_service_baseline_preserves_allowed_values_without_inheriting_credentials(monkeypatch):
    from ouroboros.tools.services import _service_env
    from ouroboros.workspace_executor import _executor_service_env

    value = "https://synthetic-user:synthetic-password@127.0.0.1/modules"
    monkeypatch.setenv("NODE_PATH", value)
    monkeypatch.setenv("UNSELECTED_660_SECRET", "synthetic-host-only")
    for build in (_service_env, _executor_service_env):
        env = build()
        assert env["NODE_PATH"] == value
        assert "UNSELECTED_660_SECRET" not in env


@pytest.mark.parametrize("invalid", [[], {"BAD=NAME": "x"}, {"": "x"}, {"KEY": None}, {"KEY": "x\x00y"}])
def test_service_rejects_invalid_env_without_spawning(tmp_path, monkeypatch, invalid):
    from ouroboros.tools.services import _start_service
    monkeypatch.setattr("ouroboros.process_custody.spawn_supervised", lambda *a, **k: pytest.fail("must not spawn"))
    result = _start_service(ToolContext(repo_dir=tmp_path, drive_root=tmp_path), [sys.executable], env=invalid)
    assert "TOOL_ARG_ERROR" in result


def test_docker_service_only_forwards_selected_env_names(tmp_path, monkeypatch):
    import os
    from ouroboros import workspace_executor as executor

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="docker-env",
        executor_ref={"type": "docker_exec", "container_name": "controlled", "network": "host",
                      "workspace_host_path": str(workspace), "workspace_backend_path": "/project"})
    selected = {"TOKEN": "synthetic-660-docker", "PATH": "/backend/bin", "EMPTY": "",
                "DOCKER_HOST": "tcp://backend-only:1234"}
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="123\n" if "nohup" in cmd[-1] else "running\n", stderr="")

    monkeypatch.setattr(executor.subprocess, "run", run)
    payload = executor.start_service(ctx, name="selected", cmd=["server"], host_cwd=workspace,
        cwd_root="active_workspace", readiness={"timeout_sec": 0}, outputs=[], before_outputs={},
        env=selected, env_overlay={"PATH": "/host-only/node", "HOST_ONLY": "never-forward"})
    cmd, kwargs = next(call for call in calls if "nohup" in call[0][-1])
    aliases = cmd[3:2 + 2 * len(selected):2]
    assert all(alias.startswith("OUROBOROS_SERVICE_ENV_") for alias in aliases)
    assert [kwargs["env"][alias] for alias in aliases] == list(selected.values())
    assert kwargs["env"]["PATH"] == os.environ["PATH"]
    assert kwargs["env"].get("DOCKER_HOST") == os.environ.get("DOCKER_HOST")
    assert "/host-only/node" not in str(kwargs) and "HOST_ONLY" not in cmd
    assert selected["TOKEN"] not in str(cmd) + json.dumps(payload)
    assert payload["backend_cwd"] == "/project"


def test_docker_environment_wrapper_preserves_values_in_real_shell(tmp_path):
    import os
    import shlex
    from ouroboros import workspace_executor as executor

    if os.name == "nt":
        pytest.skip("Docker backend shell is POSIX")
    aliases = {"TOKEN": "OUROBOROS_SERVICE_ENV_A", "odd.name": "OUROBOROS_SERVICE_ENV_B"}
    expected = {"TOKEN": 'synthetic-660-quote"\n$(not-a-command)', "odd.name": ""}
    record = SimpleNamespace(cmd=[sys.executable, "-c",
        "import os,json; print(json.dumps({key:os.environ.get(key) for key in ['TOKEN','odd.name','OUROBOROS_SERVICE_ENV_A','OUROBOROS_SERVICE_ENV_B']}))"],
        backend_cwd=str(tmp_path))
    shell = executor._docker_service_start_shell(record, str(tmp_path / "log"), aliases)
    # Exercise the exact payload executed by both setsid/non-setsid branches,
    # in a foreground owned child so this probe cannot leave a service behind.
    tokens = shlex.split(shell)
    command = tokens[tokens.index("-c") + 1]
    result = subprocess.run(["sh", "-c", command], cwd=tmp_path,
        env={**os.environ, **{aliases[key]: value for key, value in expected.items()}},
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert {key: observed[key] for key in expected} == expected
    assert observed["OUROBOROS_SERVICE_ENV_A"] is None and observed["OUROBOROS_SERVICE_ENV_B"] is None


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
                "cmd": [
                    sys.executable,
                    "-c",
                    "import os,time; os.write(1, b'READY\\n' + b'x' * 25000); time.sleep(30)",
                ],
                "readiness": {"log_contains": "READY", "timeout_sec": 5},
            },
        )
    )
    status = json.loads(registry.execute("service_status", {"name": "svc"}))
    logs = json.loads(registry.execute("service_logs", {"name": "svc", "tail": 1000}))
    stopped_raw = registry.execute("stop_service", {"name": "svc"})
    stopped = json.loads(stopped_raw)

    assert started["ready"] is True
    assert started["ready_observed_at"]
    assert status["state"] == "running"
    assert "READY" not in logs["tail"]
    assert "x" in logs["tail"]
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
    assert '"readiness"' not in durable_text
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


def test_executor_readiness_scans_before_large_log_suffix(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    log_path = tmp_path / "executor-service.log"
    log_path.write_bytes(b"READY\n" + (b"x" * 25_000))
    record = SimpleNamespace(
        executor=SimpleNamespace(kind="local"),
        backend_log_path=str(log_path),
        local_proc=SimpleNamespace(poll=lambda: None),
        ready=False,
    )
    monkeypatch.setattr(workspace_executor.time, "sleep", lambda _seconds: None)

    workspace_executor._wait_readiness(
        record,
        {"log_contains": "READY", "timeout_sec": 0.05},
    )

    assert record.ready is True


def test_executor_terminal_payload_clears_readiness(tmp_path):
    import ouroboros.workspace_executor as workspace_executor

    record = SimpleNamespace(
        service_id="task:svc",
        name="svc",
        task_id="task",
        executor=SimpleNamespace(
            executor_id="local-service",
            kind="local",
            network="host",
        ),
        backend_pid="4321",
        backend_cwd="/workspace",
        host_cwd=tmp_path,
        cwd_root="active_workspace",
        cwd_base=str(tmp_path),
        cwd_source="active_workspace",
        skill_name="",
        cmd=["service"],
        outputs=[],
        keep_alive=False,
        backend_log_path=str(tmp_path / "service.log"),
        started_at=workspace_executor.time.time(),
        ready=True,
    )

    payload = workspace_executor._service_payload(record, state="exited")

    assert payload["state"] == "exited"
    assert payload["ready"] is False
    assert record.ready is False
