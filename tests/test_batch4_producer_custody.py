"""Producer evidence through registry → shell/VCS → Pause, Sleep and Continue."""
import os
import subprocess

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running



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


@pytest.mark.parametrize("service", [False, True])
def test_docker_popen_no_start_is_positive_no_effect(tmp_path, monkeypatch, service):
    registry, queue, workers = _registry(tmp_path, monkeypatch, "docker_exec")
    calls = _docker(monkeypatch, no_start=True)
    args = {"cmd": ["backend-writer"]}
    if service:
        args["name"] = "writer"
    result = registry.execute_result("start_service" if service else "run_command", args)
    if service:
        assert result.meta["process_not_started"] is True
        assert "operation_outcome" not in result.meta  # a replacement may have retired its old instance
    else:
        assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert [kind for kind, _ in calls] == ["start"]
    assert not list(_owned(tmp_path))
    _consumers(tmp_path, registry, queue, workers, False)


def _fake_vcs_git(monkeypatch, run):
    """Answer every Git dispatch of the VCS tools with ``run(cmd)``: the facade's run_cmd, and the
    byte read behind diff's displayed text."""
    from ouroboros.tools import git, git_vcs_ops

    monkeypatch.setattr(git, "run_cmd", lambda cmd, **_kw: run(cmd))
    monkeypatch.setattr(git_vcs_ops, "_git_display_text", lambda cmd, _repo: (run(cmd), ""))


@pytest.mark.parametrize("name,args", [
    ("vcs_restore", {"paths": [":(glob)*"]}),
    ("vcs_restore", {"paths": ["../outside"]}),
    ("vcs_restore", {"paths": [" "]}),
    ("vcs_revert", {"sha": " "}),
])
def test_vcs_validation_family_never_dispatches_git(tmp_path, monkeypatch, name, args):
    from ouroboros.tools import review_helpers
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    dispatched = []
    monkeypatch.setattr(review_helpers, "list_changed_paths_from_git_status",
                        lambda *_a, **_kw: dispatched.append("status") or ["ordinary.txt"])
    _fake_vcs_git(monkeypatch, lambda cmd: dispatched.append(cmd) or "")
    refused = registry.execute_result(name, {"root": "active_workspace", **args})
    assert refused.meta.get("operation_outcome") == "completed_no_effect", refused
    assert dispatched == []
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    assert registry.execute_result("vcs_read", {"action": "diff", "root": "active_workspace"}).code == "OK"
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
