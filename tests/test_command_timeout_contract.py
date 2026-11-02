"""The loop's outer wait follows the handler's own child timeout (#1376).

run_command, run_script and verify_and_record bound their child through
``shell_process._resolve_effective_timeout``. The loop's wait used to follow a second
rule: settings before environment, a per-call override only for run_command and
run_script, none for verify_and_record, and no margin over the default. On 26 Oct a
``verify_and_record(timeout_sec=1800)`` under a saved 1000 s tool timeout returned
TOOL_TIMEOUT at 1000 s and its real FAIL arrived 2 s later. Each matrix case dispatches
through the REAL registry (its alias folding and argument checks included), captures the
timeout the real handler hands its subprocess and compares it with the REAL outer
resolution; two cases run a real child through the real dispatcher at scaled times.
"""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from subprocess import CompletedProcess

import pytest

import ouroboros.loop_tool_execution as lte
import ouroboros.tools.shell_process as shell_process
from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC
from ouroboros.tools.registry import ToolContext, ToolRegistry

MARGIN = NESTED_SETTLEMENT_MARGIN_SEC


def _settings(monkeypatch, value):
    document = {} if value is None else {"OUROBOROS_TOOL_TIMEOUT_SEC": value}
    for module in (lte, shell_process):  # one saved document, as both readers see it
        monkeypatch.setattr(module, "load_settings", lambda: dict(document))


def _context(tmp_path, deadline_sec=None):
    work, drive = tmp_path / "work", tmp_path / "drive"
    work.mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=work, drive_root=drive, task_id="timeout-contract")
    if deadline_sec is not None:
        ctx.task_metadata = {"deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=deadline_sec)).isoformat()}
    return ctx


def _dispatch_registry(tmp_path, monkeypatch, scaled_tool=None, deadline_sec=None):
    """A real registry; ``scaled_tool``'s entry window is scaled from minutes to one second."""
    from ouroboros import config, safety

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(safety, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(safety, "get_safety_mode", lambda kind="owner": "off")
    ctx = _context(tmp_path, deadline_sec)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    if scaled_tool:
        monkeypatch.setitem(registry._entries, scaled_tool, replace(registry._entries[scaled_tool], timeout_sec=1))
    return registry


REQUIRED = {"run_command": {"cmd": ["true"]}, "run_script": {"script": "print(1)"},
            "verify_and_record": {"contract_kind": "explicit_command", "check": ["true"]}}


def _dispatched(monkeypatch, registry, tool, args):
    """``(child timeouts, text)`` when the real dispatcher runs ``tool`` with ``args``."""
    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs["timeout"])
        return CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", run)
    return seen, registry.execute(tool, {**REQUIRED[tool], **args})


@pytest.mark.parametrize("tool,args,saved,env,ceiling,deadline", [
    pytest.param("run_command", {}, 600, None, None, None, id="default-run-command"),
    pytest.param("verify_and_record", {}, 1000, None, None, None, id="default-verify-under-saved-1000"),
    pytest.param("verify_and_record", {"timeout_sec": 1800}, 1000, None, None, None, id="explicit-verify-1800-history"),
    pytest.param("run_command", {}, 600, "1200", None, None, id="environment-over-saved"),
    pytest.param("run_script", {}, 600, "1200", None, None, id="environment-over-saved-script"),
    pytest.param("verify_and_record", {"timeout": 1800}, 600, None, None, None, id="alias-verify-1800-under-saved-600"),
    pytest.param("run_script", {"timeout": 900}, 600, None, None, None, id="alias-script"),
    pytest.param("run_command", {"timeout_sec": None}, 600, None, None, None, id="null-canonical"),
    pytest.param("verify_and_record", {"timeout_sec": 3000}, 600, None, "3600", None, id="raised-ceiling-verify"),
    pytest.param("run_command", {"timeout_sec": 900}, 600, None, "30", 3600, id="deadline-with-configured-ceiling-30"),
    pytest.param("verify_and_record", {"timeout_sec": 1800}, 600, None, None, 7200, id="deadline-explicit-verify"),
    pytest.param("run_command", {}, 600, None, None, 7200, id="deadline-default-run-command"),
    pytest.param("run_script", {"timeout_sec": 3000}, 600, None, "3600", None, id="control-explicit-script"),
    pytest.param("run_command", {"timeout_sec": 5}, 600, None, None, None, id="control-short-override"),
])
def test_outer_wait_outlasts_the_handlers_own_child_timeout(tmp_path, monkeypatch, tool, args, saved, env, ceiling,
                                                            deadline):
    _settings(monkeypatch, saved)
    for key, value in (("OUROBOROS_TOOL_TIMEOUT_SEC", env), ("OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC", ceiling)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    registry = _dispatch_registry(tmp_path, monkeypatch, deadline_sec=deadline)

    outer = lte._get_tool_timeout(registry, tool, args)  # resolved first, as the loop does before dispatch
    seen, text = _dispatched(monkeypatch, registry, tool, args)

    assert len(seen) == 1, text
    assert outer >= seen[0] + MARGIN, (tool, args, outer, seen[0])
    assert outer >= registry.get_timeout(tool), "the registered window is a floor, never lowered"


@pytest.mark.parametrize("tool", sorted(REQUIRED))
@pytest.mark.parametrize("args", [{"timeout_sec": None, "timeout": 900}, {"timeout_sec": 600, "timeout": 1800}],
                         ids=["alias-after-null", "alias-beside-canonical"])
def test_an_alias_beside_its_canonical_name_starts_no_child(tmp_path, monkeypatch, tool, args):
    """Dispatch folds ``timeout`` only into an absent ``timeout_sec``; beside it the call is
    refused before any child starts, so no child timeout exists for the wait to outlast."""
    _settings(monkeypatch, 600)
    seen, text = _dispatched(monkeypatch, _dispatch_registry(tmp_path, monkeypatch), tool, args)
    assert seen == [] and "TOOL_ARG_ERROR" in text, text


def _outer_floor(registry, tool):
    """The outer wait's own floor: the saved 600 s Tool Timeout or the registered window."""
    return max(600, registry.get_timeout(tool))


@pytest.mark.parametrize("tool", sorted(REQUIRED))
@pytest.mark.parametrize("requested", [10, 60, 7200])
@pytest.mark.parametrize("ceiling", [None, "30"], ids=["shipped-ceiling", "configured-ceiling-30"])
def test_a_requested_command_timeout_is_honored_as_written(tmp_path, monkeypatch, tool, requested, ceiling):
    """No separate per-call ceiling and no floor: 10 stays 10, 7200 stays 7200, whatever a
    stray ``OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC`` says (retired: nothing reads it)."""
    _settings(monkeypatch, 600)
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    if ceiling is None:
        monkeypatch.delenv("OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC", ceiling)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    args = {"timeout_sec": requested}

    outer = lte._get_tool_timeout(registry, tool, args)
    seen, text = _dispatched(monkeypatch, registry, tool, args)

    assert seen == [requested], text
    assert outer == max(_outer_floor(registry, tool), requested + MARGIN)


def _pinned_clock(monkeypatch):
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("ouroboros.deadline_utils.utc_now", lambda: now)
    monkeypatch.setattr("ouroboros.task_pacing.effective_finalization_reserve_sec", lambda ctx: 120.0)
    return now


@pytest.mark.parametrize("tool", sorted(REQUIRED))
@pytest.mark.parametrize("requested,left,child", [
    pytest.param(10, 7200, 10, id="short-request-long-deadline"),  # the former 60 s floor widened it
    pytest.param(60, 7200, 60, id="minute-request-long-deadline"),
    pytest.param(7200, 600, 480, id="long-request-short-deadline"),  # what is left less the reserve, not half
    pytest.param(600, 60, 1, id="inside-the-reserve"),  # a fast typed timeout, not a refusal
])
def test_a_live_task_deadline_only_narrows_the_child(tmp_path, monkeypatch, tool, requested, left, child):
    _settings(monkeypatch, 600)
    now = _pinned_clock(monkeypatch)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    registry._ctx.task_metadata = {"deadline_at": (now + timedelta(seconds=left)).isoformat()}
    args = {"timeout_sec": requested}

    outer = lte._get_tool_timeout(registry, tool, args)
    seen, text = _dispatched(monkeypatch, registry, tool, args)

    assert seen == [child], text
    assert outer == max(_outer_floor(registry, tool), child + MARGIN)


@pytest.mark.parametrize("tool", sorted(REQUIRED))
@pytest.mark.parametrize("spent", ["deadline", "execution-budget"])
def test_a_spent_task_bound_starts_no_command(tmp_path, monkeypatch, tool, spent):
    """A real expired bound refuses before any child starts; it never reopens a full window."""
    _settings(monkeypatch, 600)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    if spent == "deadline":
        registry._ctx.task_metadata = {"deadline_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    else:
        registry._ctx.task_metadata = {"execution_timeout_sec": 60}
        registry._ctx.task_started_at = time.time() - 120
    args = {"timeout_sec": 7200}

    outer = lte._get_tool_timeout(registry, tool, args)
    seen, text = _dispatched(monkeypatch, registry, tool, args)

    assert seen == [], text
    assert f"TASK_DEADLINE_BLOCKED ({tool})" in text and "no command was started" in text
    assert outer == _outer_floor(registry, tool)  # the resolution itself never outlives the bound


@pytest.mark.parametrize("tool", sorted(REQUIRED))
def test_a_bound_that_ends_in_the_final_preparation_starts_no_child(tmp_path, monkeypatch, tool):
    """R2: the task's deadline is still ahead when the command prepares; its last preparation step, the runtime
    selection over the launch environment, really takes past it (a slow PATH or filesystem lookup). The bound and
    its timeout are read after that step, beside the dispatch, so no child starts, not even with the 1 s a spent
    bound resolves to. ``run_script`` reaches that check through its own command (``_run_shell``)."""
    import ouroboros.tools.shell as shell
    import ouroboros.tools.verify as verify

    _settings(monkeypatch, 600)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    selected = []

    def slow(real):
        def select(*args, **kwargs):
            real(*args, **kwargs)
            selected.append(datetime.now(timezone.utc))
            time.sleep(2.5)
        return select

    for module in (shell, verify):
        monkeypatch.setattr(module, "record_runtime_selection", slow(module.record_runtime_selection))
    deadline = datetime.now(timezone.utc) + timedelta(seconds=2)
    registry._ctx.task_metadata = {"deadline_at": deadline.isoformat()}

    seen, text = _dispatched(monkeypatch, registry, tool, {"timeout_sec": 600})

    assert len(selected) == 1 and selected[0] < deadline, "every earlier check ran with the deadline ahead"
    assert seen == [], text
    assert "TASK_DEADLINE_BLOCKED" in text and "no command was started" in text


SLEEPER = "import time; time.sleep(2); print('child finished')"


@pytest.mark.parametrize("tool,args,env", [
    pytest.param("run_command", {"cmd": [sys.executable, "-c", SLEEPER]}, "4", id="environment-run-command"),
    pytest.param("verify_and_record", {"contract_kind": "explicit_command", "check": [sys.executable, "-c", SLEEPER],
                                       "timeout_sec": 4}, None, id="explicit-verify"),
])
def test_a_real_child_settles_inside_the_outer_wait(tmp_path, monkeypatch, tool, args, env):
    """Scaled history: the saved tool timeout and the entry window are 1 s, the child
    may run 4 s and finishes in 2 s. The wait must return the child's own result."""
    _settings(monkeypatch, 1)
    if env is None:
        monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_TOOL_TIMEOUT_SEC", env)
    registry = _dispatch_registry(tmp_path, monkeypatch, tool)
    drive_logs = registry._ctx.drive_root / "logs"
    call = {"id": "call-scaled", "function": {"name": tool, "arguments": json.dumps(args)}}

    result = lte._execute_with_timeout(registry, call, drive_logs, lte._get_tool_timeout(registry, tool, args),
                                       task_id="timeout-contract")

    assert result["tool_result"].code != "TOOL_TIMEOUT", result["result"]
    assert "child finished" in result["result"], result["result"]
    rows = [json.loads(line) for line in (drive_logs / "tools.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["tool_call_started", "tool_call"], rows


# Past what the platform can wait on (a parsed JSON integer may lie past any float), or not a finite number at all
# (JSON 1e400, -Infinity, NaN).
UNREPRESENTABLE = {"past-the-platform": 99999999999, "past-any-float": 10**400, "infinite": float("inf"),
                   "negative-infinite": float("-inf"), "nan": float("nan")}


def _settled_within(timeout):
    """A real pending future's wait accepts ``timeout`` (the loop's outer wait on a call)."""
    future = concurrent.futures.Future()
    threading.Timer(0.05, future.set_result, ("settled",)).start()
    return future.result(timeout=timeout) == "settled"


@pytest.mark.parametrize("tool", sorted(REQUIRED))
@pytest.mark.parametrize("requested", list(UNREPRESENTABLE))
def test_a_timeout_the_platform_cannot_wait_on_is_refused_before_any_child_starts(tmp_path, monkeypatch, tool, requested):
    """Opus R1: such a request reached the child's pipe reader and the loop's outer ``Future.result``, which raise
    OverflowError, and the call escaped the executor. Now each consumer refuses it, typed, before any child starts,
    naming the longest wait it can have here and the wait that sets that bound; the outer resolution stays finite."""
    _settings(monkeypatch, 600)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    args = {"timeout_sec": UNREPRESENTABLE[requested]}
    longest, bound = shell_process._longest_command_wait()

    outer = lte._get_tool_timeout(registry, tool, args)  # resolved before dispatch, as the loop does
    seen, text = _dispatched(monkeypatch, registry, tool, args)

    assert seen == [], text
    assert text.startswith(f"⚠️ TOOL_ARG_ERROR ({tool}): timeout_sec={UNREPRESENTABLE[requested]!r} seconds"), text
    assert f"waits at most {longest} seconds ({bound}). No command was started" in text
    assert _settled_within(outer)


@pytest.mark.parametrize("tool", sorted(REQUIRED))
def test_the_longest_representable_wait_runs_as_asked_and_a_saved_one_past_it_is_refused_by_name(tmp_path, monkeypatch,
                                                                                               tool):
    """The boundary itself: exactly the named maximum is honored as written (no narrowing); a saved Tool Timeout one
    second past it is refused under its own name, not silently shortened."""
    longest, _bound = shell_process._longest_command_wait()
    _settings(monkeypatch, 600)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    seen, text = _dispatched(monkeypatch, registry, tool, {"timeout_sec": longest})
    assert seen == [longest], text
    assert _settled_within(lte._get_tool_timeout(registry, tool, {"timeout_sec": longest}))

    _settings(monkeypatch, longest + 1)
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    outer = lte._get_tool_timeout(registry, tool, {})
    seen, text = _dispatched(monkeypatch, registry, tool, {})
    assert seen == [] and f"Tool Timeout (OUROBOROS_TOOL_TIMEOUT_SEC)={longest + 1} seconds" in text, text
    assert _settled_within(outer)


def test_the_named_maximum_is_the_real_limit_of_the_childs_pipe_reader():
    """The physical fact behind the bound, on this platform's own waits: a child's pipe reader accepts the named
    maximum; where ``poll(2)`` reads the pipes, one second more overflows its C int of milliseconds."""
    import os
    import selectors

    longest, _bound = shell_process._longest_command_wait()
    with subprocess.Popen([sys.executable, "-c", "print('ok')"], stdout=subprocess.PIPE) as proc:
        assert proc.communicate(timeout=longest)[0].strip() == b"ok"
    if os.name != "nt" and hasattr(selectors, "PollSelector"):
        proc = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)
        try:
            with pytest.raises(OverflowError):
                proc.communicate(timeout=longest + 1)
        finally:
            proc.kill()
            proc.wait(timeout=10)
            proc.stdout.close()


SLEEP_THEN_PRINT = "import sys, time; time.sleep(float(sys.argv[1])); print('child finished')"


@pytest.mark.parametrize("case,sleep,timeout_sec", [
    pytest.param("success", "0", 30, id="success"),
    pytest.param("timeout", "3", 1, id="timeout"),
    pytest.param("refused", "0", 99999999999, id="refused"),
    pytest.param("refused", "0", 10**400, id="refused-integer-past-any-float"),  # 401 digits, parsed from the call's JSON
])
def test_a_real_child_through_the_real_dispatcher_returns_its_own_typed_result(tmp_path, monkeypatch, case, sleep,
                                                                                timeout_sec):
    """The reviewed repro through the real dispatcher: every outcome is the call's own result, never an exception out of
    the executor; a representable timeout is the child's as written, an unrepresentable one starts nothing."""
    _settings(monkeypatch, 600)
    registry = _dispatch_registry(tmp_path, monkeypatch)
    args = {"cmd": [sys.executable, "-c", SLEEP_THEN_PRINT, sleep], "timeout_sec": timeout_sec}
    call = {"id": f"call-{case}", "function": {"name": "run_command", "arguments": json.dumps(args)}}

    result = lte._execute_with_timeout(registry, call, registry._ctx.drive_root / "logs",
                                       lte._get_tool_timeout(registry, "run_command", args), task_id="timeout-contract")

    text = result["result"]
    if case == "success":
        assert "child finished" in text, text
    elif case == "timeout":
        assert "TOOL_TIMEOUT (run_command): command exceeded the per-command timeout of 1s" in text, text
    else:
        assert text.startswith(f"⚠️ TOOL_ARG_ERROR (run_command): timeout_sec={timeout_sec} seconds"), text
        assert "child finished" not in text
