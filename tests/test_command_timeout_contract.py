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

import json
import sys
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
    pytest.param("run_command", {"timeout_sec": 900}, 600, None, "30", 3600, id="deadline-floor-above-ceiling"),
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
