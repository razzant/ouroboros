"""Declared-output admission through public process consumers, without a daemon.

Only the Docker CLI boundary is simulated. Real Docker filesystem/consumer proof
is opt-in in test_docker_declared_outputs_live.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests import test_declared_output_early_answer as output_fixtures

project = output_fixtures.project


@pytest.fixture
def shell_docker(project, monkeypatch):
    """Real POSIX path probe/command; Docker transport and mount facts are simulated.

    Matching host/backend roots avoid emulating a namespace. Only /proc/self/cwd's
    readlink capability check is supplied on macOS; path tests use the real OS.
    """
    if os.name == "nt":
        pytest.skip("Docker's path probe needs a POSIX shell")
    from ouroboros import workspace_executor as executor
    from ouroboros import workspace_executor_paths as paths

    registry, active, _ = project
    active = active.resolve()
    registry._ctx.executor_ref = {
        "type": "docker_exec", "container_name": "output-lab", "network": "none",
        "workspace_host_path": str(active), "workspace_backend_path": str(active),
    }

    def probe(argv):
        if argv[0] == "inspect":
            return json.dumps({"id": "pinned-id", "running": True, "network": "none",
                               "mounts": [{"Type": "bind", "Source": str(active), "Destination": str(active)}]})
        assert argv[:4] == ["exec", "pinned-id", "sh", "-c"]
        capability = 'readlink() { case "$1" in /proc/self/cwd) pwd -P;; *) command readlink "$1";; esac; }\n'
        result = subprocess.run(["sh", "-c", capability + argv[4], *argv[5:]], capture_output=True, timeout=10)
        if result.returncode:
            raise paths.OutputResolutionUnavailable(result.stderr.decode())
        return result.stdout.decode()

    def execute(ref, cmd, cwd, timeout, **kwargs):
        assert ref.container_name == "pinned-id"
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return executor.ExecutorResult(result.returncode, result.stdout, result.stderr, operation_outcome="completed")

    monkeypatch.setattr(paths, "_docker_output_probe", probe)
    monkeypatch.setattr(executor, "_execute_docker", execute)
    return registry, active


@pytest.mark.parametrize("output,setup,valid", [
    ("missing/../answer.txt", "", False),
    ("answer.txt/", "", False),
    ("created/../answer.txt", "mkdir created; ", True),
    ("answer-dir/", "mkdir answer-dir; cp answer.txt answer-dir/; ", True),
])
def test_export_requires_the_original_address_after_command(shell_docker, output, setup, valid):
    from ouroboros.artifacts import collect_task_artifact_records

    registry, active = shell_docker
    result = registry.execute_result("run_command", {
        "cmd": ["sh", "-c", "printf bytes > answer.txt; " + setup + "printf ran > marker; printf preserved-stdout"],
        "outputs": [output],
    })
    print(result.text)
    assert result.meta["exit_code"] == 0 and "preserved-stdout" in result.text
    assert (active / "marker").read_text() == "ran"
    assert result.code == ("OK" if valid else "ARTIFACT_OUTPUT_ERROR"), result.text
    records = collect_task_artifact_records(registry._ctx.drive_root, registry._ctx.task_id, require_registered=True)
    assert bool(records) == valid


@pytest.mark.parametrize("output", ["tmpfs_result/cache/answer.txt", "tmpfs_result", "tmpfs_result/sibling.txt"])
def test_hostconfig_tmpfs_refuses_hidden_bind_outputs(shell_docker, monkeypatch, output):
    from pathlib import Path

    from ouroboros.artifacts import collect_task_artifact_records
    from ouroboros import workspace_executor_paths as paths

    registry, active = shell_docker
    cache = active / "tmpfs_result" / "cache"
    cache.mkdir(parents=True)
    decoy = cache / "answer.txt"
    decoy.write_text("HIDDEN HOST DECOY")
    original = paths._docker_output_probe

    def probe(argv):
        answer = original(argv)
        if argv[0] == "inspect":
            facts = json.loads(answer)
            # Docker --tmpfs lives in HostConfig.Tmpfs, outside .Mounts.
            if ".HostConfig.Tmpfs" in argv[2] or 'index .HostConfig "Tmpfs"' in argv[2]:
                facts["tmpfs"] = {str(cache): "size=1m"}
            return json.dumps(facts)
        return answer

    monkeypatch.setattr(paths, "_docker_output_probe", probe)
    result = registry.execute_result("run_command", {
        "cmd": ["sh", "-c", "printf ran > marker; printf bind-bytes > tmpfs_result/sibling.txt"],
        "outputs": [output],
    })
    print(result.text)
    rows = collect_task_artifact_records(registry._ctx.drive_root, registry._ctx.task_id, require_registered=True)
    if output.endswith("sibling.txt"):
        assert result.code == "OK", result.text
        assert (active / "marker").read_text() == "ran"
        assert len(rows) == 1 and Path(rows[0]["path"]).read_text() == "bind-bytes"
    else:
        assert result.code == "RESOURCE_POLICY_BLOCKED", result.text
        assert "not a mapped host bind" in result.text
        assert result.meta["operation_outcome"] == "completed_no_effect"
        assert not (active / "marker").exists() and not rows
    assert decoy.read_text() == "HIDDEN HOST DECOY"


@pytest.mark.parametrize("outcome", ["export", "unavailable", "stop_failed", "keep_alive", "keep_task", "panic"])
def test_automatic_docker_service_outputs_and_custody(shell_docker, monkeypatch, outcome):
    from pathlib import Path

    from ouroboros import artifacts
    from ouroboros import workspace_executor as executor
    from ouroboros import workspace_executor_paths as paths
    from ouroboros.tools import services
    from tests._workspace_executor_shared import fake_docker_cli

    registry, active = shell_docker
    ctx = registry._ctx
    stopped = False
    real_run = subprocess.run

    def cli(cmd, **kwargs):
        nonlocal stopped
        if cmd[0] != "docker":
            return real_run(cmd, **kwargs)
        assert "pinned-id" in cmd
        if cmd[1] == "inspect":
            answer = "none\n"
        elif "nohup" in cmd[-1]:
            (active / "answer.txt").write_text("service-bytes")
            answer = "12345\n"
        elif "kill -TERM" in cmd[-1]:
            if outcome == "stop_failed":
                return subprocess.CompletedProcess(cmd, 1, "", "injected stop refused")
            stopped, answer = True, ""
        elif "_ouroboros_proc_observe" in cmd[-1]:
            answer = "exited\n" if stopped else "running\n"
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, answer, "")

    fake_docker_cli(monkeypatch, cli)
    started = registry.execute_result("start_service", {
        "name": "automatic-output", "outputs": ["answer.txt"], "cmd": ["fake-server"],
        "keep_alive": outcome == "keep_alive", "readiness": {"timeout_sec": 0},
    })
    assert started.status == "ok", started.text
    record = next(r for r in executor._services_snapshot() if r.task_id == ctx.task_id)
    assert record.executor.container_name == "pinned-id" and record.durable_record_path.is_file()
    ctx.executor_ref = None  # finalization must use the start context
    if outcome == "keep_task":
        ctx.task_metadata = {"service_teardown": "keep"}
    if outcome != "export":
        def unavailable(argv):
            if outcome != "unavailable":
                pytest.fail("kept, unconfirmed and Panic paths must not export")
            raise paths.OutputResolutionUnavailable("injected post-stop output failure")
        monkeypatch.setattr(paths, "_docker_output_probe", unavailable)
    results = (services.kill_all_services(ctx.drive_root, request_only=True) if outcome == "panic"
               else services.stop_task_services(ctx))
    print(json.dumps(results))
    assert len(results) == 1 and not any(key.startswith("_") for key in results[0])
    pending = outcome in {"stop_failed", "keep_alive", "keep_task", "panic"}
    assert record.durable_record_path.exists() is pending
    assert bool(executor._services_snapshot()) is pending
    rows = artifacts.collect_task_artifact_records(ctx.drive_root, ctx.task_id, require_registered=True)
    if pending:
        assert not stopped and not rows and "artifact_outputs" not in results[0]
        if outcome != "panic":
            assert results[0]["lifecycle"] == ("cleanup_pending" if outcome == "stop_failed" else "kept")
    else:
        assert stopped and results[0]["lifecycle"] == "stopped" and results[0]["cleanup_dispatched"]
        assert results[0]["artifact_output_failed"] is (outcome == "unavailable")
        if outcome == "export":
            assert len(rows) == 1 and Path(rows[0]["path"]).read_text() == "service-bytes"
        else:
            assert not rows and "injected post-stop output failure" in results[0]["artifact_outputs"]


def docker_context(registry, active):
    registry._ctx.executor_ref = {
        "type": "docker_exec", "container_name": "output-lab", "network": "none",
        "workspace_host_path": str(active), "workspace_backend_path": "/work",
    }


@pytest.mark.parametrize("tool", ["run_command", "run_script", "start_service"])
def test_unavailable_output_address_prevents_every_process_consumer(project, monkeypatch, tool):
    from ouroboros.workspace_executor_paths import OutputResolutionUnavailable

    registry, active, _ = project
    docker_context(registry, active)
    def unavailable(argv):
        raise OutputResolutionUnavailable("container inspection unavailable")
    monkeypatch.setattr("ouroboros.workspace_executor_paths._docker_output_probe", unavailable)
    args = {"cwd": str(active / "site"), "outputs": ["../answer.txt"]}
    if tool == "run_script":
        args.update(interpreter="sh", script="printf ran > marker")
    else:
        args["cmd"] = ["sh", "-c", "printf ran > marker"]
        if tool == "start_service":
            args["name"] = "output-test"
    result = registry.execute_result(tool, args)
    assert result.code == "CAPABILITY_UNAVAILABLE", result.text
    assert result.status == "unavailable" and result.meta["process_not_started"]
    assert not (active / "site" / "marker").exists()


def test_unmapped_cwd_selects_host_even_with_docker_metadata(project, monkeypatch):
    registry, active, outside = project
    docker_context(registry, outside)
    def unexpected(argv):
        raise AssertionError("an unselected Docker route must not be probed")
    monkeypatch.setattr("ouroboros.workspace_executor_paths._docker_output_probe", unexpected)
    result = registry.execute_result("run_command", {
        "cwd": str(active / "site"), "outputs": ["../answer.txt"],
        "cmd": [sys.executable, "-c", "from pathlib import Path; Path('../answer.txt').write_text('host')"],
    })
    assert result.code == "OK" and result.meta["artifact_registered"], result.text
    assert (active / "answer.txt").read_text() == "host"


def test_deliverables_alias_provenance_survives_physical_container_resolution(project, monkeypatch):
    registry, active, _ = project
    docker_context(registry, active)
    def probe(argv):
        if argv[0] == "inspect":
            return json.dumps({"id": "physical-id", "running": True, "network": "none",
                               "mounts": [{"Type": "bind", "Source": str(active), "Destination": "/work"}]})
        return "/work/site/answer.txt\0"
    monkeypatch.setattr("ouroboros.workspace_executor_paths._docker_output_probe", probe)
    result = registry.execute_result("run_command", {
        "cwd": str(active / "site"), "outputs": ["/work/Deliverables/alias/answer.txt"],
        "cmd": ["sh", "-c", "printf ran > marker"],
    })
    assert result.code == "RESOURCE_POLICY_BLOCKED", result.text
    assert "Deliverables output escapes" in result.text
    assert not (active / "site" / "marker").exists()


@pytest.mark.parametrize("answer", ["not JSON", '{"id":"id","mounts":[]}',
                                    '{"id":"id","running":false,"mounts":[],"network":"none"}'])
def test_incomplete_inspection_is_unavailable_and_never_an_empty_mapping(project, monkeypatch, answer):
    registry, active, _ = project
    docker_context(registry, active)
    monkeypatch.setattr("ouroboros.workspace_executor_paths._docker_output_probe", lambda argv: answer)
    result = registry.execute_result("run_command", {
        "cwd": str(active), "outputs": ["answer.txt"], "cmd": ["sh", "-c", "printf ran > marker"],
    })
    assert result.code == "CAPABILITY_UNAVAILABLE", result.text
    assert not (active / "marker").exists()
