"""Producer evidence through registry → shell/VCS → Pause, Sleep and Continue."""
import os
import subprocess
import sys

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running

pytestmark = pytest.mark.serial


def _registry(tmp_path, monkeypatch, backend=None):
    from ouroboros import workspace_executor as executor
    from ouroboros import safety
    from ouroboros.tools.registry import ToolRegistry

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    workspace = tmp_path.parent / (tmp_path.name + "-workspace")
    workspace.mkdir()
    registry = ToolRegistry(repo_dir=tmp_path.parent / (tmp_path.name + "-system"), drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = "root"
    ctx.task_attempt = 1
    ctx.workspace_mode, ctx.workspace_root = "external", workspace
    if backend:
        ctx.executor_ref = {"type": backend, "container_name": "test-container", "network": "host",
                            "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}
    monkeypatch.setattr(executor, "_panic_requested", False)
    monkeypatch.setattr(executor, "_FOREGROUND", {})
    monkeypatch.setattr(executor, "_SERVICES", {})
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    return registry, queue, workers


def _owned(root):
    """Executor records the ownership set names, as the exit reads them."""
    from ouroboros import workspace_executor as executor

    return executor._owned_process_records(root, "foreground") + executor._owned_process_records(root, "service")


def _consumers(tmp_path, registry, queue, workers, held):
    from ouroboros.model_sleep import cold_blockers, request_sleep
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree
    from supervisor.continuation_admission import admit_continuation, conflicting_writers
    from tests.test_owner_continue import _interrupted, NONCE

    claims = load_task_result(tmp_path, "root").get("launch_handoffs", {})
    # Local invocation and independent executor custody are separate facts.
    assert bool(cold_blockers(registry._ctx)) is held
    selection = {"senders": [], "tasks": [], "runs": [], "wake_at": ""}
    if held:
        with pytest.raises(ValueError, match="tool_handoff|workspace_executor|member_custody"):
            request_sleep(registry._ctx, selection, "cold")
    else:
        request_sleep(registry._ctx, selection, "cold")
    assert request_owner_pause("root", request_id="producer-check")["ok"]
    workers.RUNNING.clear()
    refresh_owner_pause_tree("root")
    assert read_fence(tmp_path, "root")["state"] == ("requested" if held else "paused")
    _interrupted(tmp_path, "root", reason_code="task_exception")
    assert bool(conflicting_writers(queue, "root")) is held
    ack = admit_continuation("root", action_nonce=NONCE)
    assert ack["ok"] and ack["held"] is held, ack
    replay = admit_continuation("root", action_nonce=NONCE)
    assert replay["successor_task_id"] == ack["successor_task_id"]
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == claims


def _docker(monkeypatch, *, code=0, backend="completed", no_start=False, timeout=False):
    """Only Docker argv are simulated; every accidental Docker call fails closed."""
    from ouroboros import workspace_executor as executor

    real_popen, real_run = subprocess.Popen, subprocess.run
    calls = []
    class Client:
        pid = 987654321
        returncode = code
        killed = False
        def communicate(self, **_kwargs):
            if timeout and not self.killed:
                raise subprocess.TimeoutExpired("docker exec", 1)
            return "command output", ""
        def kill(self):
            self.killed = True
        def wait(self, **_kwargs):
            return self.returncode
    def popen(argv, **kwargs):
        if argv[0] != "docker":
            return real_popen(argv, **kwargs)
        calls.append(("start", argv))
        if no_start:
            raise FileNotFoundError("docker executable unavailable")
        return Client()
    def run(argv, **kwargs):
        if argv[0] != "docker":
            return real_run(argv, **kwargs)
        calls.append(("probe", argv))
        if backend == "unreadable":
            raise OSError("backend unavailable")
        if backend == "probe_failed":
            return subprocess.CompletedProcess(argv, 125, "completed", "transport lost")
        return subprocess.CompletedProcess(argv, 0, backend, "")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(executor, "_process_command_sha256", lambda _pid: "")
    return calls


@pytest.mark.parametrize("code,backend", [(-9, "running"), (125, "unreadable"),
                                           (0, "running"), (7, "completed"), (0, "completed"),
                                           (-9, "completed"), (0, "probe_failed"), (0, "completed\nextra")])
def test_docker_client_exit_requires_backend_evidence(tmp_path, monkeypatch, code, backend):
    registry, queue, workers = _registry(tmp_path, monkeypatch, "docker_exec")
    calls = _docker(monkeypatch, code=code, backend=backend)
    result = registry.execute_result("run_command", {"cmd": ["backend-writer"]})
    held = backend != "completed"
    assert result.meta["exit_code"] == code
    assert result.meta.get("operation_outcome") == ("unknown" if held else "completed")
    assert [kind for kind, _ in calls] == ["start", "probe"] + ([] if held else ["probe"])
    records = list(_owned(tmp_path))
    assert len(records) == int(held)
    if held:
        assert records[0][1]["container_name"] == "test-container"
        assert records[0][1]["backend_pidfile"].endswith(".pid")
        assert records[0][1]["executor_type"] == "docker_exec"
    _consumers(tmp_path, registry, queue, workers, held)


@pytest.mark.parametrize("backend", [None, "local"])
def test_measured_local_nonzero_is_completed(tmp_path, monkeypatch, backend):
    registry, queue, workers = _registry(tmp_path, monkeypatch, backend)
    result = registry.execute_result("run_command", {"cmd": [sys.executable, "-c", "raise SystemExit(7)"]})
    assert result.code == "SHELL_EXIT_ERROR" and result.meta["exit_code"] == 7
    assert result.meta.get("operation_outcome") == "completed"
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize("service", [False, True])
def test_docker_popen_no_start_is_positive_no_effect(tmp_path, monkeypatch, service):
    registry, queue, workers = _registry(tmp_path, monkeypatch, "docker_exec")
    calls = _docker(monkeypatch, no_start=True)
    args = {"cmd": ["backend-writer"]}
    if service:
        args["name"] = "writer"
    result = registry.execute_result("start_service" if service else "run_command", args)
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert [kind for kind, _ in calls] == ["start"]
    assert not list(_owned(tmp_path))
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize("args", [{"head": "HEAD"}, {"base": "HEAD", "head": "HEAD", "staged": True}])
def test_vcs_argument_refusal_and_corrected_retry_settle_only_their_claims(tmp_path, monkeypatch, args):
    from ouroboros.tools import git
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    dispatched = []
    monkeypatch.setattr(git, "run_cmd", lambda cmd, **_kw: dispatched.append(cmd) or "")
    refused = registry.execute_result("vcs_diff", {"root": "active_workspace", **args})
    assert refused.code == "TOOL_ARG_ERROR"
    assert dispatched == []
    assert refused.meta.get("operation_outcome") == "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    retry = registry.execute_result("vcs_diff", {"root": "active_workspace"})
    assert retry.code == "OK" and dispatched == [["git", "diff"]]
    _consumers(tmp_path, registry, queue, workers, False)


def test_joined_git_error_does_not_survive_corrected_retry(tmp_path, monkeypatch):
    from ouroboros.tools import git
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    def unknown(*_a, **_kw):
        raise subprocess.TimeoutExpired("git diff", 1)
    monkeypatch.setattr(git, "run_cmd", unknown)
    result = registry.execute_result("vcs_diff", {"root": "active_workspace"})
    assert result.code == "GIT_ERROR" and not result.meta.get("operation_outcome")
    claim = load_task_result(tmp_path, "root")["launch_handoffs"]
    assert claim == {}
    monkeypatch.setattr(git, "run_cmd", lambda *_a, **_kw: "")
    registry.execute_result("vcs_diff", {"root": "active_workspace", "head": "HEAD"})
    registry.execute_result("vcs_diff", {"root": "active_workspace"})
    assert load_task_result(tmp_path, "root")["launch_handoffs"] == claim
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize("name,args", [
    ("vcs_restore", {"paths": [":(glob)*"]}),
    ("vcs_restore", {"paths": ["../outside"]}),
    ("vcs_restore", {"paths": [" "]}),
    ("vcs_revert", {"sha": " "}),
])
def test_vcs_validation_family_never_dispatches_git(tmp_path, monkeypatch, name, args):
    from ouroboros.tools import git, review_helpers
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    dispatched = []
    monkeypatch.setattr(review_helpers, "list_changed_paths_from_git_status",
                        lambda *_a, **_kw: dispatched.append("status") or ["ordinary.txt"])
    monkeypatch.setattr(git, "run_cmd", lambda cmd, **_kw: dispatched.append(cmd) or "")
    refused = registry.execute_result(name, {"root": "active_workspace", **args})
    assert refused.meta.get("operation_outcome") == "completed_no_effect", refused
    assert dispatched == []
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    assert registry.execute_result("vcs_diff", {"root": "active_workspace"}).code == "OK"
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize("service", [False, True])
def test_docker_timeout_with_unreadable_backend_keeps_launch_custody(tmp_path, monkeypatch, service):
    from ouroboros import workspace_executor as executor

    registry, queue, workers = _registry(tmp_path, monkeypatch, "docker_exec")
    _docker(monkeypatch, timeout=True, backend="unreadable")
    monkeypatch.setattr(executor, "kill_process_tree", lambda _proc: None)
    args = {"cmd": ["backend-writer"]}
    if service:
        args["name"] = "writer"
    result = registry.execute_result("start_service" if service else "run_command", args)
    assert result.status in {"error", "timeout"}
    assert not result.meta.get("operation_outcome")
    if not service:
        assert len(_owned(tmp_path)) == 1
    _consumers(tmp_path, registry, queue, workers, True)


@pytest.mark.parametrize("code", [0, 7])
@pytest.mark.skipif(os.name != "posix", reason="Exercises the POSIX backend wrapper through local sh")
def test_backend_wrapper_wait_fact_is_read_through_local_shell(tmp_path, monkeypatch, code):
    """Run the real wrapper/probe scripts locally; no daemon or Docker CLI."""
    registry, queue, workers = _registry(tmp_path, monkeypatch, "docker_exec")
    real_popen = subprocess.Popen
    calls = []
    def docker_as_shell(argv, **kwargs):
        if argv[0] == "docker":
            calls.append(argv[-1])
            argv = ["/bin/sh", "-c", argv[-1].replace("/tmp/ouroboros-exec-", str(tmp_path / "owned-"))]
        return real_popen(argv, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", docker_as_shell)
    result = registry.execute_result("run_command", {"cmd": ["/bin/sh", "-c", f"exit {code}"]})
    assert result.meta.get("operation_outcome") == "completed", result
    assert result.meta["exit_code"] == code and len(calls) == 3
    assert list(tmp_path.glob("owned-*.pid")) == []
    assert not _owned(tmp_path)
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize("content", [None, "4321", "completed\nextra", "", "completed"])
@pytest.mark.skipif(os.name != "posix", reason="Exercises the POSIX backend probe through local sh")
def test_backend_completion_probe_requires_exact_owned_wait_fact(tmp_path, monkeypatch, content):
    from ouroboros import workspace_executor as executor

    pidfile = tmp_path / "owned.pid"
    if content is not None:
        pidfile.write_text(content)
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: real_run(["/bin/sh", "-c", argv[-1]], **kw))
    assert executor._docker_exec_completed("simulated", str(pidfile)) is (content == "completed")
    if content != "completed" and content is not None:
        assert pidfile.read_text() == content


def test_business_exit_code_does_not_override_host_return(tmp_path, monkeypatch):
    from ouroboros.tools.tool_result import _publish_process_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    registry.override_handler("knowledge_read", lambda ctx, **_kw:
        _publish_process_result(ctx, "SHELL_EXIT_ERROR", "client ended", exit_code=7))
    result = registry.execute_result("knowledge_read", {"topic": "x"})
    assert result.meta["exit_code"] == 7 and "operation_outcome" not in result.meta
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize("receipt", ["", "completed"])
def test_backend_cleanup_requires_receipt_even_with_zero_cli_exit(monkeypatch, receipt):
    from ouroboros import workspace_executor as executor

    monkeypatch.setattr(subprocess, "run", lambda argv, **_kw:
        subprocess.CompletedProcess(argv, 0, receipt, ""))
    assert executor._cleanup_docker_exec_timeout("simulated", "/tmp/owned.pid") is bool(receipt)
    record = {"container_name": "simulated", "backend_pidfile": "/tmp/owned.pid"}
    assert executor._dispatch_docker_record_cleanup(record) is bool(receipt)


@pytest.mark.skipif(os.name != "posix", reason="Exercises backend receipt script through local sh")
@pytest.mark.parametrize("cleanup", [False, True])
def test_lost_completion_probe_reply_preserves_retry_receipt(tmp_path, monkeypatch, cleanup):
    from ouroboros import workspace_executor as executor
    pidfile = tmp_path / "owned.pid"
    pidfile.write_text("completed")
    real_run = subprocess.run
    dropped = []
    def run(argv, **kw):
        observed = real_run(["/bin/sh", "-c", argv[-1]], **kw)
        if not dropped:
            dropped.append(True)
            raise subprocess.TimeoutExpired(argv, 5)
        return observed
    monkeypatch.setattr(subprocess, "run", run)
    probe = executor._cleanup_docker_exec_timeout if cleanup else executor._docker_exec_completed
    assert not probe("simulated", str(pidfile))
    assert pidfile.read_text() == "completed"
    assert probe("simulated", str(pidfile))


def test_host_join_and_backend_cleanup_settle_timed_out_invocation(tmp_path, monkeypatch):
    from ouroboros import workspace_executor as executor
    registry, queue, workers = _registry(tmp_path, monkeypatch, "docker_exec")
    _docker(monkeypatch, timeout=True, backend="completed")
    monkeypatch.setattr(executor, "kill_process_tree", lambda _proc: None)
    result = registry.execute_result("run_command", {"cmd": ["backend-writer"]})
    assert result.status == "timeout" and not result.meta.get("operation_outcome")
    assert not _owned(tmp_path)
    _consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.skipif(os.name != "posix", reason="Runs the backend receipt cleanup in POSIX sh")
@pytest.mark.parametrize("failure", ["persist", "lost_delete_reply"])
def test_docker_completion_gc_keeps_recoverable_fact(tmp_path, monkeypatch, failure):
    from ouroboros import workspace_executor as executor
    marker = tmp_path / "owned.pid"
    marker.write_text("completed")
    path = executor._register_process(tmp_path, {"record_type": "foreground", "executor_type": "docker_exec",
        "container_name": "simulated", "backend_pidfile": "/tmp/ouroboros-exec-fixture.pid", "host_pid": 987654321})
    real_write, real_run = executor.atomic_write_json, subprocess.run
    calls = []
    def run(argv, **kwargs):
        result = real_run(["/bin/sh", "-c", argv[-1].replace("/tmp/ouroboros-exec-fixture.pid", str(marker))], **kwargs)
        calls.append(argv)
        if failure == "lost_delete_reply" and len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, 5)
        return result
    monkeypatch.setattr(subprocess, "run", run)
    if failure == "persist":
        monkeypatch.setattr(executor, "atomic_write_json", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
    assert not executor._retire_docker_completion(path)
    assert path.exists()
    if failure == "persist":
        assert marker.read_text() == "completed" and not calls
        monkeypatch.setattr(executor, "atomic_write_json", real_write)
        assert executor._retire_docker_completion(path)
    else:
        assert not marker.exists() and executor._load_process_record(path)["backend_completed"]
        monkeypatch.setattr(executor, "_kill_host_pid", lambda *_a: pytest.fail("completed CLI PID must not be signaled"))
        assert executor.kill_all_foreground(tmp_path)[0]["cleanup_dispatched"]
    assert not marker.exists() and not path.exists()
