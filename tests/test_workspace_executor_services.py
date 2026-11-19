"""Executor-backed services: their lifecycle, their records and their teardown.

Split verbatim out of ``tests/test_workspace_executor.py`` by theme. This module owns
the private snapshot a local service hides, the restart after exit, the sanitized env
and redacted logs, the status and durable record that redact secret-like arguments, the
task and global cleanup they participate in, the keep-alive that survives task
teardown, the panic cleanup that kills durable foreground and service processes, and
the child-drive records a parent data root must scan.

Runs in the xdist pass beside other files: its processes are owned children killed within the
test's own drive root and the per-worker registries are restored by
``_isolate_workspace_executor_globals`` (``tests/test_workspace_executor.py``), so it names no host resource.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests import test_declared_output_early_answer as output_fixtures
from tests._workspace_executor_shared import _init_repo, fake_docker_cli

project = output_fixtures.project


@pytest.mark.parametrize(("other_worker", "executor_kind"), [(False, "local"), (True, None)])
def test_quiet_service_log_stays_while_owned_and_archives_after_exit(tmp_path, monkeypatch, executor_kind, other_worker):
    from ouroboros import workspace_executor as executor
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import services

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    workspace = tmp_path / "project"
    workspace.mkdir()
    release = workspace / "release"
    script = workspace / "quiet.py"
    script.write_text("import pathlib, sys, time\nprint('READY', flush=True)\n"
                      "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.02)\n"
                      "print('AFTER_RETENTION', flush=True)\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=tmp_path / "data", task_id="quiet-task",
                      workspace_root=workspace, workspace_mode="external")
    if executor_kind:
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                            "workspace_backend_path": "/workspace"}
    records, lock = (executor._SERVICES, executor._STATE_LOCK) if executor_kind else (services._SERVICES, services._LOCK)
    key, record = "quiet-task:quiet", None
    try:
        started = json.loads(services._start_service(ctx, name="quiet", cmd=[sys.executable, str(script), str(release)],
                             cwd=str(workspace), keep_alive=True, readiness={"log_contains": "READY", "timeout_sec": 15}))
        assert started["ready"] and started["state"] == "running"
        with lock:
            record = records[key]
            if other_worker:
                records.pop(key)  # maintenance sees durable custody, not this worker's local map
        proc = record.local_proc if executor_kind else record.proc
        path = Path(record.backend_log_path) if executor_kind else record.log_path
        write_task_result(ctx.drive_root, ctx.task_id, "completed")  # a kept service outlives its task
        old = time.time() - 30 * 86400
        os.utime(path, (old, old))
        report = services.prune_service_logs(ctx.drive_root, retention_days=7)
        assert path.is_file() and proc.poll() is None
        assert report["archived_files"] == report["deleted_files"] == 0
        release.touch()
        assert proc.wait(timeout=15) == 0
        assert "AFTER_RETENTION" in path.read_text()
        os.utime(path, (old, old))
        report = services.prune_service_logs(ctx.drive_root, retention_days=7)
        assert report["archived_files"] == report["deleted_files"] == 1 and not path.exists()
    finally:
        if record is not None:
            with lock:
                records[key] = record
        services._stop_service(ctx, "quiet")


@pytest.mark.parametrize(("secret_newline", "output_newline", "executor_kind"), [
    ("\n", None, None),
    ("\r\n", "\r\n", "local"),
])
def test_real_service_observes_exact_selected_env_and_cwd(tmp_path, monkeypatch, executor_kind, output_newline, secret_newline):
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
    script.write_text('''import json, os, pathlib, sys, time
sys.stdout.reconfigure(newline=OUTPUT_NEWLINE)
observed = {"cwd": os.getcwd(), "token": os.environ["TOKEN"],
    "empty": os.environ["EMPTY"], "proxy": os.environ["HTTPS_PROXY"],
    "unselected": os.environ.get("UNSELECTED_660_SECRET")}
pathlib.Path("observed.json").write_text(json.dumps(observed))
print("TOKEN=" + os.environ["TOKEN"], flush=True)
print("READY", flush=True)
time.sleep(30)
'''.replace("OUTPUT_NEWLINE", repr(output_newline)), encoding="utf-8")
    env = {"TOKEN": 'synthetic-660-quote"\\tail' + secret_newline + 'private-fragment-660', "EMPTY": "",
           "HTTPS_PROXY": "http://synthetic-user:synthetic-pass@127.0.0.1:7777",
           "DEPLOYMENT_MODE": "running", "FIELD_NAME": "state"}
    monkeypatch.setattr(services, "load_settings", lambda: {"TEST_SERVICE_KEY": env["TOKEN"]})
    registry = ToolRegistry(repo_dir=tmp_path / "system", drive_root=tmp_path / "data")
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=tmp_path / "data",
                      workspace_root=workspace, workspace_mode="external", task_id="selected-env")
    if executor_kind:
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                            "workspace_backend_path": "/workspace"}
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
    assert finalization["deleted_live_log"] and finalization["source_scope"] == "full" and not finalization.get("error_count")
    from ouroboros.artifacts import read_actor_source_bytes

    final = read_actor_source_bytes(ctx.drive_root, ctx.task_id, finalization["source_ref"]).decode()
    assert "private-fragment-660" not in final and "READY" in final
    assert services.prune_service_logs(ctx.drive_root, retention_days=0)["archived_files"] == 0


def test_service_baseline_preserves_allowed_values_without_inheriting_credentials(monkeypatch):
    from ouroboros.tools.services import _service_env
    from ouroboros.workspace_executor import _executor_service_env

    value = "https://synthetic-user:synthetic-password@127.0.0.1/modules"
    monkeypatch.setenv("NODE_PATH", value)
    monkeypatch.setenv("UNSELECTED_660_SECRET", "synthetic-host-only")
    # The login name is passed through (keychain/identity lookups key on it);
    # a secret-shaped name that merely starts the same way still is not.
    monkeypatch.setenv("USER", "synthetic-login")
    monkeypatch.setenv("LOGNAME", "synthetic-login")
    monkeypatch.setenv("USERNAME", "synthetic-login")
    monkeypatch.setenv("USER_API_TOKEN", "synthetic-host-only")
    for build in (_service_env, _executor_service_env):
        env = build()
        assert env["NODE_PATH"] == value
        assert env["USER"] == "synthetic-login"
        assert env["LOGNAME"] == "synthetic-login"
        assert env["USERNAME"] == "synthetic-login"
        assert "UNSELECTED_660_SECRET" not in env
        assert "USER_API_TOKEN" not in env


@pytest.mark.parametrize("invalid", [[], {"BAD=NAME": "x"}, {"KEY": "x\x00y"}])
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

    fake_docker_cli(monkeypatch, run)
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


@pytest.mark.parametrize(("first_backend", "next_backend", "explicit_stop"), [
    ("host", "host", False), ("local", "local", False),
    ("host", "host", True), ("local", "local", True),
    ("host", "local", False), ("local", "host", False),
])
def test_same_name_restarts_preserve_each_log_and_its_declared_output(
        project, monkeypatch, first_backend, next_backend, explicit_stop):
    from ouroboros.artifacts import read_actor_source_bytes, registered_task_artifact
    from ouroboros.tools import services
    from tests._launch_stop_shared import _refuse_metadata_of, _wait_exited, _wait_for_service_log

    registry, workspace, _ = project
    ctx = registry._ctx
    local = {"type": "local", "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}
    first_bytes = b"FIRST_READY\xff\n"

    def start(backend, script):
        ctx.executor_ref = local if backend == "local" else {}
        result = registry.execute_result("start_service", {
            "name": "reporter", "cwd": str(workspace), "outputs": ["report.txt"],
            "cmd": [sys.executable, "-c", script], "readiness": {"log_contains": "FIRST_READY", "timeout_sec": 0.1}})
        assert result.status == "ok", result.text
        payload = json.loads(result.text)
        return payload, Path(payload["backend_log_path" if backend == "local" else "log_path"])

    first, old_log = start(first_backend,
        "import sys; from pathlib import Path; Path('report.txt').write_bytes(b'first report'); "
        f"sys.stdout.buffer.write({first_bytes!r})")
    try:
        _wait_exited(registry, "reporter")
        assert old_log.read_bytes() == first_bytes
        with monkeypatch.context() as patch:
            if explicit_stop:
                patch.setattr(services, "_MAX_SERVICE_LOG_BLOB_BYTES", 3)
                stopped = json.loads(registry.execute("stop_service", {"name": "reporter"}))
                assert stopped["log_finalization"]["live_log"] == "retained" and str(old_log) not in json.dumps(stopped)
            else:
                _refuse_metadata_of(patch, old_log)
            second, new_log = start(next_backend,
                "import time; from pathlib import Path; Path('report.txt').write_bytes(b'second report'); "
                "print('SECOND_ONLY', flush=True); time.sleep(60)")
            assert second["service_id"] == first["service_id"] and new_log != old_log
            if not explicit_stop:
                assert second["previous_service"] == {"state": "stopped", "log_not_saved": True}
            _wait_for_service_log(new_log, lambda data: b"SECOND_ONLY" in data)
            status = json.loads(registry.execute("service_status", {"name": "reporter"}))
            assert status["state"] == "running" and not status["ready"] and not status["ready_observed_at"]
            assert json.loads(registry.execute("service_logs", {"name": "reporter"}))["tail"].splitlines() == ["SECOND_ONLY"]
            assert old_log.read_bytes() == first_bytes
            artifact = registered_task_artifact(ctx.drive_root, ctx.task_id, "report.txt")
            assert Path(artifact["path"]).read_bytes() == b"first report"
            assert (workspace / "report.txt").read_bytes() == b"second report"
            if next_backend == "local":
                [record] = (ctx.drive_root / "state" / "workspace_executor_processes").glob("*.json")
                durable = json.loads(record.read_text(encoding="utf-8"))
                assert str(durable["host_pid"]) == second["backend_pid"] and durable["backend_log_path"] == str(new_log)
        stopped = json.loads(registry.execute("stop_service", {"name": "reporter"}))
        assert not stopped["artifact_output_failed"] and not new_log.exists() and old_log.read_bytes() == first_bytes
        final = stopped["log_finalization"]
        assert final["source_scope"] == "full"
        assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, final["source_ref"]).splitlines() == [b"SECOND_ONLY"]
        artifact = registered_task_artifact(ctx.drive_root, ctx.task_id, "report.txt")
        assert Path(artifact["path"]).read_bytes() == b"second report"
        versions = ctx.drive_root / "task_results" / "artifact_versions" / ctx.task_id / "report.txt"
        assert [version.read_bytes() for version in versions.iterdir()] == [b"first report"]
    finally:
        registry.execute("stop_service", {"name": "reporter"})


@pytest.mark.parametrize("backend", ["host", "local"])
@pytest.mark.parametrize("gap", ["log", "output", "stop"])
def test_failed_replacement_carries_previous_launch_to_completion(project, monkeypatch, backend, gap):
    import queue

    from ouroboros import loop_forced_finalization, service_finalization
    from ouroboros.outcomes import derive_loop_outcome
    from tests._launch_stop_shared import _wait_exited
    from tests._launch_stop_shared import _author_evidence, _exit_context, _refuse_metadata_of

    registry, work, _outside = project
    tool_ctx = registry._ctx
    if backend == "local":
        tool_ctx.executor_ref = {"type": "local", "workspace_host_path": str(work), "workspace_backend_path": "/workspace"}
    completion = _exit_context(tool_ctx.drive_root, tool_ctx, event_queue=queue.Queue())
    tool_ctx._execution_trace = completion.llm_trace  # run_llm_loop binds this before dispatching tools
    args = {"name": "paper", "cwd": str(work), "cmd": [sys.executable, "-c", "print('PRIVATE-DIAGNOSTIC')"],
            "readiness": {"timeout_sec": 0}}
    first = registry.execute_result("start_service", {**args, "outputs": ["missing.txt"] if gap == "output" else []})
    assert first.status == "ok", first.text
    launch = json.loads(first.text)
    assert launch["launch_id"]
    _wait_exited(registry, "paper")
    spawn_attempts = []

    def refuse_spawn(*args, **kwargs):
        spawn_attempts.append(True)
        raise OSError("replacement spawn refused")

    with monkeypatch.context() as patch:
        patch.setattr("ouroboros.process_custody.spawn_supervised", refuse_spawn)
        if gap == "log":
            _refuse_metadata_of(patch, Path(launch.get("log_path") or launch["backend_log_path"]))
        elif gap == "stop":
            patch.setattr("ouroboros.process_scope.end_held_processes", lambda items, **kw: {
                item["key"]: {"facts": {"exit_confirmed": False, "census_failed": True},
                              "birth": "", "known": []} for item in items})
        replacement = registry.execute_result("start_service", args)
    assert replacement.status != "ok", replacement.text
    assert len(spawn_attempts) == (0 if gap == "stop" else 1)

    trace = completion.llm_trace
    assert service_finalization.finalize(completion, trace)
    assert not service_finalization.finalize(completion, trace), "the next completion observes no new fact"
    completion.messages, completion.forced_service_evidence_fingerprint = [], ""
    loop_forced_finalization._disclose_service_finalization(completion, trace)
    shown, _ = _author_evidence(completion.messages)
    rows = [row for event in shown for row in event["services"]]
    assert all(row["launch_id"] == launch["launch_id"] for row in rows)
    frames = [completion.event_queue.get_nowait()["data"] for _ in shown]
    assert completion.event_queue.empty()
    execution = derive_loop_outcome("The preview was checked.", {}, trace)["outcome_axes"]["execution"]
    failures = (execution.get("failure") or {}).get("verification_failures", [])
    assert [item["status"] for item in failures] == (["artifact_output_error"] if gap == "output" else [])
    label = launch["launch_id"].removeprefix("launch-").removeprefix("service-")[:10]
    if gap == "stop":
        assert [row["lifecycle"] for row in rows] == ["cleanup_pending", "stopped"]
        assert rows[0]["stop_failed"] and rows[0]["process_cleanup"]["census_failed"]
        assert [frame["text"] for frame in frames] == [
            f"Background service cleanup unconfirmed: paper ({label}) · for replacement · log capture not attempted",
            f"Background service stopped: paper ({label}) · at task end · log saved"]
    else:
        [row], [frame] = rows, frames
        assert row["lifecycle"] == "stopped" and row["process_cleanup"]["exit_confirmed"]
        assert row["artifact_output_failed"] == (gap == "output") and not row.get("artifact_output_unknown")
        tail = "log capture unconfirmed" if gap == "log" else "log saved · output 1 absent"
        assert frame["text"] == f"Background service stopped: paper ({label}) · for replacement · {tail}"
    assert all("PRIVATE-DIAGNOSTIC" not in frame["text"] and str(work) not in frame["text"] for frame in frames)


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


@pytest.mark.parametrize("missing", [False, True])
def test_automatic_local_executor_stop_finalizes_declared_outputs(project, missing):
    from ouroboros import artifacts, service_finalization, workspace_executor
    from ouroboros.tools import services
    registry, workspace, _ = project
    ctx = registry._ctx
    ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                        "workspace_backend_path": "/workspace"}
    started = registry.execute_result("start_service", {
        "name": "automatic-output", "outputs": ["answer.txt"],
        "cmd": [sys.executable, "-c", "from pathlib import Path; import time; "
                "Path('answer.txt').write_text('service-bytes'); print('READY', flush=True); time.sleep(60)"],
        "readiness": {"log_contains": "READY", "timeout_sec": 10},
    })
    assert started.status == "ok", started.text
    try:
        if missing:
            (workspace / "answer.txt").unlink()
        completion = SimpleNamespace(drive_root=ctx.drive_root, task_id=ctx.task_id, tools=registry,
                                     drive_logs=ctx.drive_logs(), event_queue=None, llm_trace={})
        assert service_finalization.finalize(completion, completion.llm_trace)
        [event] = completion.llm_trace["verification_events"]
        assert event["kind"] == "services_stopped"
        stopped = event["services"]
        print(json.dumps(stopped))
        assert len(stopped) == 1 and stopped[0]["lifecycle"] == "stopped"
        assert stopped[0]["artifact_output_failed"] is missing
        assert not any(key.startswith("_") for key in stopped[0])
        assert workspace_executor.service_status(ctx, "automatic-output") is None
        assert not service_finalization.finalize(completion, completion.llm_trace)
        records = artifacts.collect_task_artifact_records(ctx.drive_root, ctx.task_id, require_registered=True)
        if missing:
            assert "missing output" in stopped[0]["artifact_outputs"] and not records
        else:
            assert len(records) == 2 and Path(records[0]["path"]).read_text() == "service-bytes"
            assert sum(row.get("immutable") is True for row in records) == 1
    finally:
        services._stop_service(ctx, "automatic-output")


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


@pytest.mark.parametrize("unknown_from", ["service", "custody"])
def test_service_log_waits_for_unknown_writer_custody(tmp_path, monkeypatch, unknown_from):
    from ouroboros import body_candidate
    from ouroboros.tools import services
    from tests._launch_stop_shared import _launch_log

    path = _launch_log(tmp_path / "services" / "unknown-task", "quiet", b"kept until proven stopped\n")
    os.utime(path, (1, 1))
    monkeypatch.setattr(services, "service_execution_facts",
                        lambda _key: {"state": "unknown"} if unknown_from == "service" else None)
    monkeypatch.setattr(body_candidate, "live_lineage_processes",
                        lambda *_args: [-1] if unknown_from == "custody" else [])
    assert services.prune_service_logs(tmp_path, retention_days=0)["retained_files"] == 1 and path.exists()
    monkeypatch.setattr(services, "service_execution_facts", lambda _key: {"state": "exited"})
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_args: [])
    assert services.prune_service_logs(tmp_path, retention_days=0)["archived_files"] == 1 and not path.exists()
