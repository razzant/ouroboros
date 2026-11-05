"""Execute the actual POSIX-sh observer/stop builders against synthetic /proc.

No Docker daemon or host signalling: only the CLI is doubled. These tests do not
qualify a container image's launcher, PID namespace or /proc mount permissions.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import workspace_executor as executor
from ouroboros.docker_process_observer import proc_observer_function
from tests._workspace_executor_shared import fake_docker_cli
from tests._cases import case_folders

pytestmark = [pytest.mark.skipif(not shutil.which("sh"), reason="POSIX sh required")]


def _stat(root, pid=42, state="S", pgrp=42, comm="a tricky ) process (name)"):
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stat"
    path.write_text(f"{pid} ({comm}) {state} 1 {pgrp} " + " ".join(["0"] * 47) + "\n")
    return path


def _observe(root, scope, *, prefix=""):
    shell = prefix + proc_observer_function(proc_root=str(root)) + f"; _ouroboros_proc_observe 42 {scope}"
    return subprocess.run(["sh", "-c", shell], capture_output=True, text=True, timeout=5)


def test_states_and_names_with_parentheses(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for state, expected in ([(s, "running") for s in ("R", "S", "D", "T", "t", "I")]
         + [(s, "exited") for s in ("Z", "X", "x")]):
        for scope in ["leader", "group"]:
            tmp_path = next(case_paths)
            with subtests.test(state=state, expected=expected, scope=scope):
                _stat(tmp_path, state=state)
                result = _observe(tmp_path, scope)
                assert (result.returncode, result.stdout) == (0, expected)


@pytest.mark.parametrize("leader", ["Z", "absent"])
def test_dead_leader_with_live_member_is_exited_for_status_only(tmp_path, leader):
    if leader != "absent":
        _stat(tmp_path, state=leader)
    _stat(tmp_path, pid=43, state="T")
    _stat(tmp_path, pid=99, state="S", pgrp=99)
    assert _observe(tmp_path, "leader").stdout == "exited"
    group = _observe(tmp_path, "group")
    assert group.stdout == "running"
    assert "43" in group.stderr and "99" not in group.stderr
    (tmp_path / "43" / "stat").unlink()
    (tmp_path / "43").rmdir()
    assert _observe(tmp_path, "group").stdout == "exited"


def test_live_leader_without_its_own_group_still_blocks_cleanup(tmp_path):
    _stat(tmp_path, pgrp=7)
    assert _observe(tmp_path, "group").stdout == "running"


@pytest.mark.parametrize(("scope", "line"), [
    ("leader", ""),
    ("leader", "42 (short) S 1 42"),
    ("leader", "42 (bad) S 1 nope"),
    ("leader", "99 (wrong) Z 1 42"),
    ("group", "garbage"),
    ("group", "42 (bad) Q 1 42"),
    ("group", "42 (bad) Z 1 4:2"),
])
def test_malformed_stat_is_unknown(tmp_path, line, scope):
    path = _stat(tmp_path)
    path.write_text(line + " " + " ".join(["0"] * 47) if line != "42 (short) S 1 42" else line)
    result = _observe(tmp_path, scope)
    assert (result.returncode, result.stdout) == (2, "unknown")


def test_unreadable_member_retains_group_but_vanished_entry_does_not(tmp_path):
    _stat(tmp_path, state="Z")
    (tmp_path / "43").mkdir()  # directory exists, but stat cannot be read
    assert _observe(tmp_path, "leader").stdout == "exited"
    assert _observe(tmp_path, "group").stdout == "unknown"
    (tmp_path / "43").rmdir()
    assert _observe(tmp_path, "group").stdout == "exited"
    # Simulate a read denial independently of the runner's UID/permission bits.
    assert _observe(tmp_path, "group", prefix="cat() { return 1; }; ").stdout == "unknown"
    # An entry may vanish between enumeration and read.
    assert _observe(tmp_path, "group", prefix='cat() { rm -rf "${1%/stat}"; return 1; }; ').stdout == "exited"


def test_missing_proc_is_unknown_and_runtime_environment_cannot_redirect_it(tmp_path, monkeypatch):
    assert _observe(tmp_path / "absent", "group").stdout == "unknown"
    monkeypatch.setenv("OUROBOROS_PROC_ROOT", str(tmp_path))
    assert "OUROBOROS_PROC_ROOT" not in proc_observer_function()
    assert "_oroot=/proc" in proc_observer_function()
    quoted = tmp_path / "with spaces and ' quotes"
    _stat(quoted, state="Z")
    assert _observe(quoted, "group").stdout == "exited"


@pytest.mark.parametrize("pidfile", [False, True], ids=["service", "foreground"])
@pytest.mark.parametrize("group_state", ["zombie", "survivor", "unknown"])
def test_both_stop_shells_require_group_quiescence(tmp_path, monkeypatch, pidfile, group_state):
    root = tmp_path / "proc"
    _stat(root, state="Z")
    if group_state == "survivor":
        _stat(root, pid=43)
    elif group_state == "unknown":
        (root / "43").mkdir()
    monkeypatch.setattr(executor, "_docker_proc_observer_function",
                        lambda: proc_observer_function(proc_root=str(root)))
    receipt = tmp_path / "receipt.pid"
    receipt.write_text("42")
    shell = (executor._docker_exec_pidfile_stop_shell(str(receipt)) if pidfile
             else executor._docker_service_stop_shell("42"))
    # Keep all generated observations real while making signals inert.
    result = subprocess.run(["sh", "-c", 'kill() { :; }; sleep() { :; }; ' + shell],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == (0 if group_state == "zombie" else 1)
    assert receipt.read_text() == ("completed" if pidfile and group_state == "zombie" else "42")
    if group_state == "survivor":
        assert "43" in result.stderr
    assert "kill -0" not in shell
    assert shell.count("sleep 0.5") == 1


def _service(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    _stat(root)
    ctx = SimpleNamespace(task_id="task", drive_root=tmp_path / "data", executor_ref={
        "type": "docker_exec", "container_name": "fixture",
        "workspace_host_path": str(tmp_path), "workspace_backend_path": "/workspace"})
    record = executor._ExecutorService(
        service_id="task:svc", task_id="task", name="svc", executor=executor.executor_ref_from_ctx(ctx),
        cmd=["sleep", "30"], host_cwd=tmp_path, backend_cwd="/workspace", cwd_root="active_workspace",
        outputs=[], before_outputs={}, backend_pid="42", ready=True)
    record.durable_record_path = executor._register_service_process(ctx.drive_root, record)
    monkeypatch.setattr(executor, "_SERVICES", {record.service_id: record})
    monkeypatch.setattr(executor, "_docker_proc_observer_function",
                        lambda: proc_observer_function(proc_root=str(root)))
    run = subprocess.run

    def cli(cmd, **kwargs):
        assert cmd[:2] == ["docker", "exec"], cmd
        shell = cmd[-1]
        assert "nohup" not in shell, "a replacement must not start with live/unknown old custody"
        assert kwargs.get("timeout") in (5, 10)
        return run(["sh", "-c", 'kill() { :; }; sleep() { :; }; ' + shell],
                   capture_output=True, text=True, timeout=5)

    fake_docker_cli(monkeypatch, cli)
    return ctx, record, root


@pytest.mark.parametrize("member", ["S", "unknown"])
def test_status_sleep_restart_and_failed_teardown_share_observation(tmp_path, monkeypatch, member):
    from ouroboros.model_sleep import _service_wake
    from ouroboros.tools.services import stop_task_services

    ctx, record, root = _service(tmp_path, monkeypatch)
    facts = executor.service_execution_facts(record.service_id)
    pin = {key: facts[key] for key in ("service_id", "started_at", "backend_pid")}
    pin["name"] = "svc"
    assert _service_wake(pin) == ""
    _stat(root, state="Z")
    if member == "S":
        _stat(root, pid=43)
    else:
        (root / "43").mkdir()
    assert _service_wake(pin) == "service:svc:exited:unknown"
    payload = executor.service_status(ctx, "svc")
    assert payload["state"] == "exited" and payload["ready"] is False
    with pytest.raises(RuntimeError, match="previous service termination is not confirmed"):
        executor.start_service(ctx, name="svc", cmd=["sleep", "30"], host_cwd=tmp_path,
                               cwd_root="active_workspace", readiness={}, outputs=[], before_outputs={})
    [pending] = stop_task_services(ctx)
    assert pending["lifecycle"] == "cleanup_pending"
    assert pending["state"] == "exited" and pending["stop_failed"] is True
    assert pending["cleanup_dispatched"] is False
    assert record.durable_record_path.exists()
    assert executor._SERVICES[record.service_id] is record
    shutil.rmtree(root / "43")
    [stopped] = stop_task_services(ctx)
    assert stopped["lifecycle"] == stopped["state"] == "stopped"
    assert stopped["cleanup_dispatched"] is True
    assert not record.durable_record_path.exists()


@pytest.mark.parametrize("wait", [False, True])
def test_durable_cleanup_keeps_live_group_and_releases_zombies(tmp_path, monkeypatch, wait):
    ctx, record, root = _service(tmp_path, monkeypatch)
    _stat(root, state="Z")
    _stat(root, pid=43)
    path = record.durable_record_path
    durable = json.loads(path.read_text())
    monkeypatch.setattr(executor, "_SERVICES", {})  # after worker loss
    [pending] = executor.kill_all_services(ctx.drive_root, wait=wait)
    assert pending["state"] == "cleanup_pending" and path.exists()
    _stat(root, pid=43, state="Z")
    [stopped] = executor.kill_all_services(ctx.drive_root, wait=wait)
    assert stopped["state"] == "stopped" and not path.exists()
    assert durable["backend_pid"] == "42"


def test_candidate_writer_fence_uses_group_not_exited_leader(tmp_path, monkeypatch):
    from ouroboros.body_candidate import live_lineage_processes

    ctx, record, root = _service(tmp_path, monkeypatch)
    _stat(root, state="Z")
    _stat(root, pid=43)
    assert live_lineage_processes(ctx.drive_root, ctx.task_id)
    (root / "43" / "stat").unlink()
    assert live_lineage_processes(ctx.drive_root, ctx.task_id)  # unknown is still custody
    (root / "43").rmdir()
    assert live_lineage_processes(ctx.drive_root, ctx.task_id) == []
    assert record.durable_record_path.exists()  # the reader does not retire records
