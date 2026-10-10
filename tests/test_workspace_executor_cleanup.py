from __future__ import annotations

import json
import os
import subprocess

import pytest


def _write_docker_records(state_dir, *, foreground_pidfile="/tmp/ouroboros-exec-test.pid", service_pid="12345"):
    state_dir.mkdir(parents=True)
    (state_dir / "foreground-docker.json").write_text(
        json.dumps(
            {
                "id": "foreground-docker",
                "schema_version": 1,
                "owner": "ouroboros_workspace_executor",
                "record_type": "foreground",
                "executor_type": "docker_exec",
                "executor_id": "docker",
                "host_pid": 0,
                "container_name": "bench",
                "backend_pidfile": foreground_pidfile,
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "service-docker.json").write_text(
        json.dumps(
            {
                "id": "service-docker",
                "schema_version": 1,
                "owner": "ouroboros_workspace_executor",
                "record_type": "service",
                "service_id": "task:svc",
                "task_id": "task",
                "name": "svc",
                "executor_type": "docker_exec",
                "executor_id": "docker",
                "container_name": "bench",
                "backend_pid": service_pid,
            }
        ),
        encoding="utf-8",
    )
    from ouroboros.owned_shutdown import import_inherited_records

    import_inherited_records(state_dir.parents[1])


def _index_raw(path):
    """Name a hand-written record in the ownership set, bypassing the import's validation."""
    from ouroboros.owned_shutdown import record_executor_process

    assert record_executor_process(path, json.loads(path.read_text(encoding="utf-8")))


def _install_live_docker_service(workspace_executor, tmp_path):
    executor = workspace_executor.ExecutorRef(
        kind="docker_exec",
        executor_id="docker",
        network="none",
        mappings=(),
        container_name="bench",
    )
    with workspace_executor._STATE_LOCK:
        workspace_executor._SERVICES.clear()
        workspace_executor._SERVICES["task:live"] = workspace_executor._ExecutorService(
            service_id="task:live",
            task_id="task",
            name="live",
            executor=executor,
            cmd=["sleep", "30"],
            host_cwd=tmp_path,
            backend_cwd="/workspace",
            cwd_root="active_workspace",
            outputs=[],
            before_outputs={},
            backend_pid="67890",
        )


def test_executor_panic_cleanup_wait_false_uses_bounded_docker_stop(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    state_dir = data / "state" / "workspace_executor_processes"
    _write_docker_records(state_dir)
    _install_live_docker_service(workspace_executor, tmp_path)

    docker_run_calls: list[list[str]] = []

    def fake_docker_wait(cmd, **kwargs):
        docker_run_calls.append([str(part) for part in cmd])
        # The live-service observer uses its normal ten-second bound; stop,
        # durable-record proof and marker retirement use five seconds.
        live_probe = str(cmd[-1]) == "kill -0 67890 2>/dev/null && echo running || echo exited"
        assert kwargs["timeout"] == (10 if live_probe else 5)
        if str(cmd[-1]).startswith("rm -f -- "):
            retained = json.loads((state_dir / "foreground-docker.json").read_text())
            assert retained["backend_completed"] is True
        if "printf completed" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="completed", stderr="")
        if "kill -0" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="exited\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            raise AssertionError("panic Docker cleanup must not spawn untracked helpers")

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_docker_wait)
    monkeypatch.setattr(workspace_executor.subprocess, "Popen", FakePopen)

    killed_foreground = workspace_executor.kill_all_foreground(data, wait=False)
    killed_services = workspace_executor.kill_all_services(data, wait=False)

    # One stop/proof pair per service; foreground wait proof is persisted
    # before the one bounded marker retirement. No workload is replayed.
    scripts = [call[-1] for call in docker_run_calls]
    assert scripts == [
        workspace_executor._docker_exec_pidfile_stop_shell("/tmp/ouroboros-exec-test.pid"),
        "rm -f -- /tmp/ouroboros-exec-test.pid",
        workspace_executor._docker_service_stop_shell("67890"),
        "kill -0 67890 2>/dev/null && echo running || echo exited",
        workspace_executor._docker_service_stop_shell("12345"),
        "kill -0 12345 2>/dev/null && echo running || echo exited",
    ]
    assert all(call[:2] == ["docker", "exec"] for call in docker_run_calls)
    assert any(item.get("executor_type") == "docker_exec" for item in killed_foreground)
    assert any(item.get("state") == "stopped" for item in killed_services)
    assert all(item.get("cleanup_dispatched") is True for item in killed_foreground + killed_services)
    assert not list(state_dir.glob("*.json"))
    with workspace_executor._STATE_LOCK:
        assert "task:live" not in workspace_executor._SERVICES


def test_docker_executor_confirmed_cleanup_failure_preserves_records(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    state_dir = data / "state" / "workspace_executor_processes"
    _write_docker_records(state_dir)
    _install_live_docker_service(workspace_executor, tmp_path)

    def fake_failed_docker_wait(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="permission denied")

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_failed_docker_wait)

    killed_foreground = workspace_executor.kill_all_foreground(data, wait=True)
    killed_services = workspace_executor.kill_all_services(data, wait=True)

    assert any(item.get("cleanup_dispatched") is False for item in killed_foreground)
    assert any(item.get("state") == "cleanup_pending" for item in killed_foreground + killed_services)
    assert {path.name for path in state_dir.glob("*.json")} == {"foreground-docker.json", "service-docker.json"}
    with workspace_executor._STATE_LOCK:
        assert "task:live" in workspace_executor._SERVICES
        workspace_executor._SERVICES.clear()


def test_executor_cleanup_ignores_unowned_forged_process_records(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    state_dir = data / "state" / "workspace_executor_processes"
    state_dir.mkdir(parents=True)
    (state_dir / "foreground-forged.json").write_text(
        json.dumps(
            {
                "id": "foreground-forged",
                "record_type": "foreground",
                "executor_type": "local",
                "host_pid": 1,
            }
        ),
        encoding="utf-8",
    )
    _index_raw(state_dir / "foreground-forged.json")

    monkeypatch.setattr(
        workspace_executor,
        "_kill_host_pid",
        lambda _pid: (_ for _ in ()).throw(AssertionError("forged record should be ignored")),
    )

    assert workspace_executor.kill_all_foreground(data, wait=False) == []


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="under root every live pid is signalable; the forged-record rule (pid 1 refuses signal 0) is a non-root property")
def test_executor_cleanup_ignores_owner_shaped_forged_host_pid_records(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    state_dir = data / "state" / "workspace_executor_processes"
    state_dir.mkdir(parents=True)
    (state_dir / "foreground-forged.json").write_text(
        json.dumps(
            {
                "id": "foreground-forged",
                "schema_version": 1,
                "owner": "ouroboros_workspace_executor",
                "record_type": "foreground",
                "executor_type": "local",
                "host_pid": 1,
            }
        ),
        encoding="utf-8",
    )
    _index_raw(state_dir / "foreground-forged.json")

    monkeypatch.setattr(
        workspace_executor,
        "_kill_host_pid",
        lambda _pid: (_ for _ in ()).throw(AssertionError("owner-shaped forged record should be ignored")),
    )

    assert workspace_executor.kill_all_foreground(data, wait=False) == []


def test_executor_cleanup_kills_a_hash_less_record_of_our_own_live_child(tmp_path, monkeypatch):
    """A record whose command line could not be captured at register time (macOS
    `ps` right after the spawn, Windows always) still names OUR child: it is alive
    and answers signal 0, so the panic cleanup must kill it — the forged-record
    rule above refuses only pids we cannot signal (macOS full-test 33658408570)."""
    import subprocess
    import sys

    import ouroboros.workspace_executor as workspace_executor
    from ouroboros.platform_layer import subprocess_new_group_kwargs

    data = tmp_path / "data"
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        **subprocess_new_group_kwargs(),
    )
    killed: list = []
    try:
        monkeypatch.setattr(workspace_executor, "_process_command_sha256", lambda _pid: "")
        workspace_executor._register_process(
            data, {"record_type": "foreground", "executor_type": "local", "host_pid": child.pid},
        )
        monkeypatch.setattr(workspace_executor, "_kill_host_pid", lambda pid: killed.append(int(pid)))
        assert [r["id"] for r in workspace_executor.kill_all_foreground(data, wait=False)]
        assert killed == [child.pid]
    finally:
        child.kill()
        child.wait(timeout=10)


def test_executor_cleanup_ignores_pidless_docker_service_records(tmp_path, monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    data = tmp_path / "data"
    state_dir = data / "state" / "workspace_executor_processes"
    state_dir.mkdir(parents=True)
    (state_dir / "service-docker.json").write_text(
        json.dumps(
            {
                "id": "service-docker",
                "schema_version": 1,
                "owner": "ouroboros_workspace_executor",
                "record_type": "service",
                "service_id": "task:svc",
                "task_id": "task",
                "name": "svc",
                "executor_type": "docker_exec",
                "executor_id": "docker",
                "container_name": "bench",
            }
        ),
        encoding="utf-8",
    )
    _index_raw(state_dir / "service-docker.json")

    monkeypatch.setattr(
        workspace_executor.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("pidless docker service record should be ignored")),
    )

    assert workspace_executor.kill_all_services(data, wait=False) == []
