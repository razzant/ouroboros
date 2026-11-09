"""The ``run_command`` schema states exactly what its handler does (CHECKLISTS
critical surface 2: the schema is the tool's contract).

Two handler facts the description used to misstate or omit:

* only a bare shell builtin as ``cmd[0]`` is refused (``cd``, ``export``, ...);
  inside ``["sh", "-c", "cd x && ..."]`` a ``cd`` works;
* a background child (``&``, ``nohup``) is not a service: while it holds
  stdout/stderr the call waits until ``timeout_sec``; afterwards nothing tracks it
  in memory, but it belongs to the TASK (owner 2A, issue #1772): the launch carried
  the task's scope marker and the command's own process group was recorded, so the
  task's accepted terminal transition ends it on a local POSIX run — Windows has no
  census (``unsupported``) and a Docker backend's children live in its container;
  ``start_service`` is the path for work that must outlive the task.

Issue #502 asked schemas to shrink, so the description's length is pinned to its
measured size plus a small allowance; a new fact re-measures it (owner Q4,
2026-11-09: tool descriptions stay full). A third fact rides the same schema: what
a stop by pid or by a name pattern reaches (audit 08.11 item 12). Real children here belong to this test process; they are ended by the
task end under test and, as a safety net, by exact pid.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import process_scope as ps
from ouroboros.tools.shell import _SHELL_BUILTINS, _run_shell, get_tools
from ouroboros.tools.shell_process import _active_subprocesses
from tests._cases import case_folders

# measured on ef33a0a956c2 (481); re-measured after the process-stop fact of audit 08.11 item 12,
# paid in part by dropping the string-array sentence the cmd parameter already states.
_DESCRIPTION_CHARS_BEFORE = 714
_ALLOWANCE = 80


def _schema() -> dict:
    return next(entry.schema for entry in get_tools() if entry.name == "run_command")


def _ctx(tmp_path: pathlib.Path, task_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id, root_task_id=task_id, task_metadata={},
        drive_logs=lambda: pathlib.Path(str(tmp_path)),
    )


@contextmanager
def _bound(ctx):
    """The invocation binding a real tool call runs under (``tool_custody.invocation_binding``); the
    fixture task has no durable row, so its invocation is not lifecycle-bound."""
    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros.tool_custody import invocation_outcome

    ctx.task_lifecycle_bound = False
    token = _TOOL_OPERATION.set((ctx, "run_command", invocation_outcome(ctx, "run_command", f"op-{ctx.task_id}")))
    try:
        yield
    finally:
        _TOOL_OPERATION.reset(token)


def _gone(pid: int, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not ps.measure_birth(pid):
            return True
        time.sleep(0.05)
    return not ps.measure_birth(pid)


def _group_members(pgid: int) -> list:
    rows = ps.census([]) or {}
    return [pid for pid, row in rows.items() if row.pgid == pgid]


def test_description_pays_for_its_words_inside_the_issue_502_budget():
    description = _schema()["description"]
    assert len(description) <= _DESCRIPTION_CHARS_BEFORE + _ALLOWANCE, len(description)


def test_description_states_the_bare_builtin_refusal_and_the_sh_c_escape():
    description = _schema()["description"]
    assert "cmd[0]" in description, "the refusal is about a builtin as cmd[0], not about cd anywhere"
    assert "cd is rejected" not in description, "cd inside sh -c works; the blanket sentence was false"
    assert '["sh", "-c"' in description
    # The cwd parameter no longer repeats the refusal the description already names.
    assert "rejected cd" not in _schema()["parameters"]["properties"]["cwd"]["description"]


def test_description_states_what_happens_to_a_background_child():
    description = _schema()["description"]
    # What the empirical tests below prove: the call stalls until timeout_sec while the
    # child holds stdio, and on a local POSIX run the TASK's end ends it by scope.
    for word in ("&", "nohup", "stalls the call until timeout_sec", "it ends with this task",
                 "start_service keep_alive=true outlives the task"):
        assert word in description, word
    assert "nothing tracks it" not in description, "the old untracked-forever contract is gone (owner 2A)"
    assert "cleaned up" not in description  # no promise the host does not keep


def test_description_limits_the_task_end_to_local_posix_runs(monkeypatch):
    """The task end reaches a background child only through the POSIX scope reap: on
    Windows ``end_scope`` reports ``unsupported`` and ends nothing, and a Docker backend's
    children run in its container, outside the host census. The sentence says so."""
    description = _schema()["description"]
    assert "it ends with this task (local POSIX runs only;" in description
    assert "lives until this task ends" not in description, "the unqualified lifetime promise is gone"
    monkeypatch.setattr(ps._pl, "IS_WINDOWS", True)
    outcome = ps.end_scope(marker=ps.mint_marker(ps.TASK_SCOPE_PREFIX), captured=[{"pid": 4242, "birth": "x"}])
    assert outcome["coverage"] == ps.COVERAGE_UNSUPPORTED and outcome["scans"] == 0
    assert outcome["ended"] == [] and outcome["settled"] is False, "nothing is ended where no census runs"


def test_handler_refuses_only_a_bare_builtin_as_cmd0(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for builtin in sorted(_SHELL_BUILTINS):
        tmp_path = next(case_paths)
        with subtests.test(builtin=builtin):
            result = _run_shell(_ctx(tmp_path), [builtin, "x"])
            assert result.startswith("⚠️ SHELL_CMD_ERROR"), result


@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_handler_runs_cd_inside_sh_c(tmp_path):
    target = tmp_path / "inner"
    target.mkdir()
    result = _run_shell(_ctx(tmp_path), ["sh", "-c", f"cd {target.as_posix()} && pwd"])
    assert "SHELL_CMD_ERROR" not in result
    assert target.resolve().name in result, result


@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_background_child_holding_stdout_makes_the_call_wait_until_timeout(tmp_path):
    from ouroboros.task_results import write_task_result

    ctx = _ctx(tmp_path, "t-stall")
    started = time.monotonic()
    with _bound(ctx):
        result = _run_shell(ctx, ["sh", "-c", "/bin/sleep 90 & echo started"], timeout_sec=1)
    elapsed = time.monotonic() - started
    record = ps.task_scope_record(tmp_path, "t-stall")
    pgid = record["groups"][0]["pgid"] if record and record["groups"] else 0
    try:
        assert result.startswith("⚠️ TOOL_TIMEOUT"), result
        assert elapsed >= 0.9, elapsed
        # The shell exited at once and nothing tracks the child in memory. Whether the
        # timeout's group kill still REACHED it is the platform's business; either way
        # its group was recorded for the task and the task end ends what survived.
        assert not any(proc.poll() is None for proc in list(_active_subprocesses)), "nothing tracks it in memory"
        assert pgid, "the command's own group is recorded on the task scope"
        write_task_result(tmp_path, "t-stall", "completed", result="done")
        ps.wait_for_cleanups()
        assert all(_gone(pid) for pid in _group_members(pgid)), "the task end leaves no member of the group"
    finally:
        for pid in _group_members(pgid) if pgid else []:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_detached_background_child_lives_until_the_task_ends(tmp_path):
    from ouroboros.task_results import load_task_result, write_task_result

    ctx = _ctx(tmp_path, "t-detached")
    with _bound(ctx):
        result = _run_shell(ctx, ["sh", "-c", "nohup /bin/sleep 90 >/dev/null 2>&1 & echo $!"], timeout_sec=5)
    pid = int(next(line for line in result.splitlines() if line.strip().isdigit()).strip())
    try:
        assert "TOOL_TIMEOUT" not in result, result
        assert ps.measure_birth(pid), "the detached child outlives the call"
        assert not any(proc.pid == pid for proc in list(_active_subprocesses)), "nothing tracks it in memory"
        record = ps.task_scope_record(tmp_path, "t-detached")
        assert os.getpgid(pid) in [group["pgid"] for group in record["groups"]], "its command's group is the task's"
        write_task_result(tmp_path, "t-detached", "running", result="still working")
        ps.wait_for_cleanups()
        assert ps.measure_birth(pid), "the call's return and a running row end nothing"
        write_task_result(tmp_path, "t-detached", "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(pid), "the task end ends it"
        assert pid in load_task_result(tmp_path, "t-detached")["process_scope"]["ended"]
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_description_states_what_a_process_stop_reaches():
    """Audit 08.11 item 12: the schema carries the fact a mind needs before ending a process, as a fact,
    not an authority rule. Text only: no guard, filter or safety policy changed with it. The argv shape
    stays stated once, by the cmd parameter's own schema."""
    description = _schema()["description"]
    assert "A stop by pid reaches exactly that process" in description
    assert "pkill -f" in description and "other tasks', the owner's and this installation's processes" in description
    assert "cmd MUST be a string array" not in description
    cmd = _schema()["parameters"]["properties"]["cmd"]
    assert cmd["type"] == "array" and cmd["items"] == {"type": "string"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process tools")
def test_a_stop_by_exact_pid_runs_through_run_command(tmp_path):
    """The owner's question on item 12 («are flexible tools kept?»): the shell path refuses nothing new.
    A child this test started is stopped by its exact pid through run_command (``kill`` is not a refused
    builtin); the command class ``pkill``/``killall`` goes through the same safety check as before."""
    ctx = _ctx(tmp_path, "t-stop")
    with _bound(ctx):
        started = _run_shell(ctx, ["sh", "-c", "/bin/sleep 60 >/dev/null 2>&1 & echo $!"], timeout_sec=5)
    pid = int(next(line for line in started.splitlines() if line.strip().isdigit()).strip())
    try:
        assert ps.measure_birth(pid)
        with _bound(ctx):
            stopped = _run_shell(ctx, ["kill", str(pid)], timeout_sec=5)
        assert "exit_code=0" in stopped and "BLOCKED" not in stopped, stopped
        assert _gone(pid), "the exact-pid stop reached the child"
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
