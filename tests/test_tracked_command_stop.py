"""The tracked command wait reads an in-process turn's Stop now (R4), and nothing else changes.

``shell_process._tracked_subprocess_run`` waited in one ``communicate(timeout)``, so a command of an inline
Presence or direct-chat turn ran its whole requested time after Stop now (the turn's Stop kills no process).
With a ``stop`` reader the same requested timeout is waited in ``communicate`` slices; Stop now kills the
command's process group as the timeout does and keeps what it already wrote. Without a reader (a pooled worker,
whose Stop is the worker kill) the wait is the one ``communicate(timeout)`` it was. Natural exit, the requested
timeout and Panic keep their meaning. Real child processes; no clock or Popen is faked.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros.platform_layer import pid_provably_gone
from ouroboros.tools import shell_process
from ouroboros.tools.shell_process import CommandStopped, _tracked_subprocess_run, turn_stop_reader

# The commands are `sh -c` scripts reporting their own and a background child's pid ($$, $!): POSIX pids only.
posix_sh = pytest.mark.skipif(os.name != "posix", reason="uses sh and its POSIX pids")


def _gone(*pids, within=5.0):
    deadline = time.monotonic() + within
    while not all(pid_provably_gone(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    return all(pid_provably_gone(pid) for pid in pids)


def _run(tmp_path, script, **kwargs):
    return _tracked_subprocess_run(["sh", "-c", script], cwd=str(tmp_path), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, **kwargs)


@posix_sh
def test_stop_now_kills_the_whole_group_and_keeps_the_output_so_far(tmp_path):
    started = time.monotonic()
    with pytest.raises(CommandStopped) as stopped:
        _run(tmp_path, "echo before; sleep 30 & echo $! > child.pid; echo $$ > shell.pid; wait; echo after",
             timeout=60, stop=lambda: time.monotonic() - started > 1.0)
    assert time.monotonic() - started < 5, "within a poll of the Stop, not the 30 s the command asked for"
    assert "before" in stopped.value.output and "after" not in stopped.value.output
    shell, child = (int((tmp_path / name).read_text()) for name in ("shell.pid", "child.pid"))
    assert _gone(shell, child), "the background grandchild went with its group"
    assert not any(proc.poll() is None for proc in shell_process._active_subprocesses)


@posix_sh
def test_without_a_reader_the_wait_is_the_one_communicate_it_was(tmp_path, monkeypatch):
    waits = []
    communicate = subprocess.Popen.communicate

    def recorded(self, *args, **kwargs):
        waits.append(kwargs.get("timeout"))
        return communicate(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", recorded)
    done = _run(tmp_path, "echo one; sleep 0.5; echo two", timeout=30)
    assert (done.returncode, done.stdout.split()) == (0, ["one", "two"])
    assert waits == [30], "no slices, no reader"
    with pytest.raises(subprocess.TimeoutExpired) as expired:
        _run(tmp_path, "sleep 5", timeout=1)
    assert expired.value.timeout == 1


@posix_sh
def test_a_reader_that_never_fires_keeps_natural_exit_and_the_requested_timeout(tmp_path):
    done = _run(tmp_path, "echo one; sleep 1; echo two", timeout=30, stop=lambda: False)
    assert (done.returncode, done.stdout.split()) == (0, ["one", "two"]), "all output, across slices"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as expired:
        _run(tmp_path, "sleep 30 & echo $! > child.pid; wait", timeout=2, stop=lambda: False)
    assert expired.value.timeout == 2 and 1.5 < time.monotonic() - started < 10
    assert _gone(int((tmp_path / "child.pid").read_text())), "the timeout still kills the group"


@posix_sh
def test_panic_still_ends_a_command_whose_wait_reads_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_process, "_panic_requested", False)  # restored after the test
    started = time.monotonic()

    def panic_once_running():
        if time.monotonic() - started > 1.0 and not shell_process._panic_requested:
            shell_process.kill_all_tracked_subprocesses(request_only=True)
        return False

    done = _run(tmp_path, "sleep 30", timeout=60, stop=panic_once_running)
    assert done.returncode != 0 and time.monotonic() - started < 10


def test_only_an_in_process_turn_gets_a_stop_reader_and_only_stop_now_stops(tmp_path):
    assert turn_stop_reader(SimpleNamespace()) is None
    assert turn_stop_reader(SimpleNamespace(model_wait_context=SimpleNamespace(worker_slot_held=True))) is None
    reader = turn_stop_reader(_turn(tmp_path))
    assert callable(reader) and reader() is False  # no open intent: no stop
    _intents(tmp_path, {"t1": {"stop_policy": "finalize_then_cancel", "request_id": "r1"}})
    assert reader() is False, "Wrap up is not an immediate kill"
    _intents(tmp_path, {"t1": {"stop_policy": "immediate", "request_id": "r1"}})
    assert reader() is True


def _intents(root, rows):
    (root / "state").mkdir(parents=True, exist_ok=True)
    text = rows if isinstance(rows, str) else json.dumps({"intents": rows})
    (root / "state" / "cancel_intents.json").write_text(text, encoding="utf-8")


def _turn(root, task_id="t1", **task):
    return SimpleNamespace(model_wait_context=SimpleNamespace(
        worker_slot_held=False, canonical_root=root, task_id=task_id, task={"_presence_turn": True, **task}))


def _errors_of_one_read(caplog, read):
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="ouroboros.cancel_intents"):
        answer = read()
    return answer, len([record for record in caplog.records if record.levelno >= logging.ERROR])


@pytest.mark.parametrize("damage", ["unreadable-projection", "this-row-malformed"])
def test_a_read_that_cannot_see_this_turns_row_is_unknown_never_no_stop(tmp_path, caplog, damage):
    """Parent 17:45: a read failure is unknown, never "no Stop", and no later poll is promised to clear it; the
    Presence wrapper still answers False. Each poll is one read, so the damage is disclosed once per poll."""
    from supervisor.owner_stop import in_process_stop_now, presence_stop_now

    _intents(tmp_path, "{not json" if damage == "unreadable-projection"
             else {"t1": "not a row", "t2": {"stop_policy": "immediate", "request_id": "r2"}})
    assert _errors_of_one_read(caplog, lambda: in_process_stop_now(_turn(tmp_path))) == (None, 1)
    assert presence_stop_now(_turn(tmp_path)) is False


def test_another_rows_damage_leaves_this_turns_own_state_readable(tmp_path, caplog):
    """A malformed row of another task hides neither this turn's immediate intent nor its absence."""
    from supervisor.owner_stop import in_process_stop_now, presence_stop_now

    _intents(tmp_path, {"t2": "not a row"})
    assert _errors_of_one_read(caplog, lambda: in_process_stop_now(_turn(tmp_path))) == (False, 1)
    assert presence_stop_now(_turn(tmp_path)) is False
    _intents(tmp_path, {"t1": {"stop_policy": "immediate", "request_id": "r1"}, "t2": "not a row"})
    assert _errors_of_one_read(caplog, lambda: in_process_stop_now(_turn(tmp_path))) == (True, 1)
    assert presence_stop_now(_turn(tmp_path)) is True


@pytest.mark.parametrize("task_id", ["", "   ", None])
def test_a_turn_without_an_identity_is_unknown_not_no_stop(tmp_path, task_id):
    """Astra 31Oct: with no task id there is no row to read, so the answer stays unknown even beside a readable projection."""
    from supervisor.owner_stop import in_process_stop_now

    _intents(tmp_path, {"t1": {"stop_policy": "immediate", "request_id": "r1"}})
    assert in_process_stop_now(_turn(tmp_path, task_id=task_id)) is None


@posix_sh
def test_an_unknown_stop_stops_nothing(tmp_path):
    """An unknown reader stops nothing: the command runs to its own end."""
    from supervisor.owner_stop import in_process_stop_now

    _intents(tmp_path, "{not json")
    done = _run(tmp_path, "sleep 1; echo done", timeout=30, stop=lambda: in_process_stop_now(_turn(tmp_path)))
    assert (done.returncode, done.stdout.strip()) == (0, "done")


@posix_sh
def test_a_local_executor_command_stops_through_its_own_tree_kill(tmp_path):
    from ouroboros import workspace_executor as executor

    ref = executor.normalize_executor_ref({"type": "local", "id": "local-test", "network": "host",
                                           "workspace_host_path": str(tmp_path), "workspace_backend_path": "/workspace"})
    started = time.monotonic()
    with pytest.raises(CommandStopped) as stopped:
        executor.execute(SimpleNamespace(drive_root=tmp_path),
                         ["sh", "-c", "echo before; sleep 30 & echo $! > child.pid; wait; echo after"], tmp_path, 60,
                         executor=ref, stop=lambda: time.monotonic() - started > 1.0)
    assert time.monotonic() - started < 5 and stopped.value.exit_observed is True
    assert "before" in stopped.value.output and "after" not in stopped.value.output
    assert _gone(int((tmp_path / "child.pid").read_text()))
    assert executor._owned_process_records(tmp_path, "foreground") == [], "an observed exit settles its record"


@pytest.mark.parametrize("confirmed", [True, False], ids=["backend-confirmed", "backend-unknown"])
def test_a_container_command_stops_through_its_owned_pidfile_never_the_container(tmp_path, monkeypatch, confirmed):
    """A controlled Docker CLI stand-in (no daemon here; the live backend is a disclosed gap): Stop now runs the
    timeout's own pidfile stop inside the container and then ends the host CLI. A stop the backend confirms is
    an observed exit and retires its record; an unconfirmed one stays unknown and keeps the record's custody."""
    from ouroboros import workspace_executor as executor

    calls = []
    real_popen, real_run = subprocess.Popen, subprocess.run

    class Cli:  # the host's `docker exec` process
        pid, returncode, args = 987654321, None, ["docker", "exec"]

        def communicate(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return "partial output\n", ""

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def popen(argv, **kwargs):
        if argv[0] != "docker":
            return real_popen(argv, **kwargs)
        calls.append(("start", argv))
        return Cli()

    def run(argv, **kwargs):
        if argv[0] != "docker":
            return real_run(argv, **kwargs)
        calls.append(("probe", argv))
        if not confirmed:
            raise OSError("backend unreachable")
        return subprocess.CompletedProcess(argv, 0, "completed", "")

    def kill_cli(proc):
        proc.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(executor, "kill_process_tree", kill_cli)
    monkeypatch.setattr(executor, "_process_command_sha256", lambda _pid: "")
    monkeypatch.setattr(executor, "_panic_requested", False)
    monkeypatch.setattr(executor, "_FOREGROUND", {})
    ref = executor.normalize_executor_ref({"type": "docker_exec", "container_name": "box", "network": "host",
                                           "workspace_host_path": str(tmp_path), "workspace_backend_path": "/work"})
    with pytest.raises(CommandStopped) as stopped:
        executor._execute_docker(ref, ["long-job"], "/work", 60, drive_root=tmp_path, stop=lambda: True)
    assert stopped.value.exit_observed is confirmed and stopped.value.output == "partial output\n"
    probes = [argv for kind, argv in calls if kind == "probe"]
    assert probes and "printf completed" in probes[0][-1], "the owned pidfile's group stop ran first"
    assert all(argv[:2] == ["docker", "exec"] for argv in probes), "never docker kill/stop/rm of the container"
    assert bool(executor._owned_process_records(tmp_path, "foreground")) is not confirmed


@posix_sh
def test_verify_and_record_records_a_stopped_check_as_failed_with_what_it_wrote(tmp_path, monkeypatch):
    from ouroboros.outcomes import verification_receipts_path
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system, workspace, data = (tmp_path / name for name in ("system", "workspace", "data"))
    for path in (system, workspace, data):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *args, **kwargs: (True, ""))
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external",
                      task_id="verify-stop")
    ctx.model_wait_context = _turn(data, "verify-stop").model_wait_context
    registry = ToolRegistry(system, data)
    registry.set_context(ctx)
    _intents(data, {"verify-stop": {"stop_policy": "immediate", "request_id": "r1"}})
    started = time.monotonic()
    result = registry.execute_result("verify_and_record", {
        "contract_kind": "explicit_command", "cwd": str(workspace), "expected": "finished",
        "check": ["sh", "-c", "echo started; sleep 30; echo finished"]})
    assert time.monotonic() - started < 10
    assert "FAIL: check ended by Stop now; its exit was observed" in result.text
    [receipt] = [json.loads(line) for line in verification_receipts_path(data, "verify-stop").read_text().splitlines()]
    assert (receipt["status"], receipt["returncode"], receipt["matched"]) == ("fail", None, None)
    assert receipt["summary"].startswith("check ended by Stop now") and "finished" not in receipt["summary"]


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_a_stopped_command_is_the_calling_tools_typed_failure_with_its_masked_output(tmp_path, monkeypatch, tool):
    """Stop now's host kill is never a success: the calling tool's typed ``SHELL_ERROR`` with everything the command
    wrote, its selected secret masked, and the host kill on the process-facts channel once its exit was observed. The
    command itself records the immediate intent after its first line, so the kill always lands after that output."""
    from ouroboros.tools.process_facts import consume_last_process_facts
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system, workspace, data = (tmp_path / name for name in ("system", "workspace", "data"))
    for path in (system, workspace, data / "state"):
        path.mkdir(parents=True)
    secret = "synthetic-stopped-command-secret"
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"CUSTOM_KEY": secret})
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external",
                      task_id="command-stop")
    ctx.model_wait_context = _turn(data, "command-stop").model_wait_context
    registry = ToolRegistry(system, data)
    registry.set_context(ctx)
    intent = json.dumps({"intents": {"command-stop": {"stop_policy": "immediate", "request_id": "r1"}}})
    program = (f"import os, pathlib, time; print('started ' + os.environ['TOKEN'], flush=True); "
               f"pathlib.Path({str(data / 'state' / 'cancel_intents.json')!r}).write_text({intent!r}); "
               "time.sleep(30); print('finished')")
    args = {"cwd": str(workspace), "env_from_settings": {"TOKEN": "CUSTOM_KEY"}, "timeout_sec": 60}
    args.update({"script": program, "interpreter": sys.executable} if tool == "run_script"
                else {"cmd": [sys.executable, "-c", program]})
    started = time.monotonic()
    result = registry.execute_result(tool, args)
    assert time.monotonic() - started < 10, "within a poll of the Stop, not the 30 s the command asked for"
    assert (result.status, result.code) == ("error", "SHELL_ERROR"), result.text
    assert result.text.startswith(f"⚠️ SHELL_ERROR ({tool}): Stop now ended this turn while the command ran")
    assert "its exit was observed" in result.text
    assert "started ***" in result.text and "finished" not in result.text and secret not in result.text
    facts = consume_last_process_facts()
    assert facts["killed_by_host"] is True and "timed_out" not in facts and "exit_code" not in facts
