"""Typed result contract + notes-after-payload (#447 В12=A, campaign organ).

Upstream landed this contract on a minimal ``result_envelope`` string subclass.
On this branch the typed organ already exists and is STRONGER — producers
publish a ``ToolResult`` (status/code/meta) through
``ouroboros/tools/tool_result.py``, and the loop reads that published object
instead of re-deriving an outcome from text. The upstream twin therefore does
not live; these are the same contracts pinned against the campaign organ.

Two regressions this pins:

- H1: a leading SAFETY_WARNING used to own line 1, so a failed command
  (SHELL_EXIT_ERROR) behind a warning was classified "ok".
- An extension/MCP tool's structured ``{"ok": false}`` answer behind a
  warning stopped being detected as a tool-reported failure.
"""
from __future__ import annotations

import pathlib
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from ouroboros.loop_tool_execution import (
    _extract_result_metadata,
    _is_tool_execution_failure,
    _typed_execution_failure,
    _typed_result_metadata,
)
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _compose_execute_result,
    _compose_execute_result_result,
)

_WARNING = "⚠️ SAFETY_WARNING: The Safety Supervisor flagged this action as suspicious."
_EXIT1 = "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1 (cwd=/tmp).\n\nSTDERR:\nboom"


# ---------------------------------------------------------------------------
# H1: safety warning trails the payload; status comes from facts, not line 1
# ---------------------------------------------------------------------------


def test_safety_warning_no_longer_masks_shell_exit_error_first_line():
    """Unmigrated-producer path: the first-line fallback parse must still see
    the failure marker on line 1 when a safety warning is present."""
    out = _compose_execute_result(_EXIT1, "", _WARNING)
    assert out.splitlines()[0].startswith("⚠️ SHELL_EXIT_ERROR")
    assert _WARNING in out
    assert _is_tool_execution_failure(True, out, fn_name="run_command") is True
    meta = _extract_result_metadata("run_command", out, True)
    assert meta["status"] == "non_zero_exit"


def test_safety_warning_with_typed_shell_meta_keeps_error_status():
    """Migrated-producer path: the typed status survives every appended note."""
    produced = ToolResult(
        status="error", code="SHELL_EXIT_ERROR", text=_EXIT1, meta={"exit_code": 1},
    )
    composed = _compose_execute_result_result(
        "run_command", produced,
        "⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: note", _WARNING,
    )
    assert _typed_execution_failure(True, composed) is True
    meta = _typed_result_metadata("run_command", composed.text, True, composed)
    assert meta["status"] == "non_zero_exit"
    assert meta["exit_code"] == 1
    # Both host notes are recorded as typed facts.
    assert dict(composed.meta)["route_note"] is True
    assert dict(composed.meta)["safety_warning"] is True
    # Payload owns line 1; notes trail it.
    assert composed.text.splitlines()[0].startswith("⚠️ SHELL_EXIT_ERROR")
    assert composed.text.index(_WARNING) > composed.text.index("boom")


def test_typed_ok_status_is_not_flipped_by_a_trailing_warning():
    produced = ToolResult(
        status="ok", code="OK",
        text="exit_code=0 (cwd=/tmp)\nSTDOUT:\nfine", meta={"exit_code": 0},
    )
    composed = _compose_execute_result_result("run_command", produced, "", _WARNING)
    assert _typed_execution_failure(True, composed) is False
    assert _typed_result_metadata(
        "run_command", composed.text, False, composed,
    )["status"] == "ok"


# ---------------------------------------------------------------------------
# Extension/MCP structured failure survives an appended warning
# ---------------------------------------------------------------------------


def test_structured_ext_failure_detected_behind_appended_warning():
    payload = '{"ok": false, "error": "screenshot backend died"}'
    out = _compose_execute_result(payload, "", _WARNING)
    assert out.splitlines()[0].startswith("{")
    adapted = LegacyTextResultAdapter.from_text("mcp__srv__shot", out)
    assert adapted.code == "TOOL_REPORTED_FAILURE"
    assert _typed_execution_failure(True, adapted) is True
    meta = _typed_result_metadata("mcp__srv__shot", out, True, adapted)
    assert meta["status"] == "tool_reported_failure"


def test_structured_failure_stays_narrow_for_plain_prose_with_json_prefix():
    # A plain string whose FULL text is not a JSON object is untouched.
    adapted = LegacyTextResultAdapter.from_text(
        "mcp__srv__shot", '{"ok": false} trailing prose',
    )
    assert adapted.code != "TOOL_REPORTED_FAILURE"


# ---------------------------------------------------------------------------
# result_meta extension: producer meta travels into the trace record
# ---------------------------------------------------------------------------


def test_producer_meta_travels_into_result_meta():
    refusal = ToolResult(
        status="blocked",
        code="LIGHT_MODE_BLOCKED",
        text="⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks this write.",
        meta={
            "policy_contract": "runtime_mode.light",
            "remaining_route": "root=user_files or root=task_drive",
        },
    )
    meta = _typed_result_metadata("write_file", refusal.text, True, refusal)
    assert meta["status"] == "light_mode_blocked"
    assert dict(refusal.meta)["policy_contract"] == "runtime_mode.light"
    assert dict(refusal.meta)["remaining_route"] == "root=user_files or root=task_drive"


# ---------------------------------------------------------------------------
# Worst-first producer: the real shell run publishes typed facts
# ---------------------------------------------------------------------------


def _ctx(tmp_path):
    return SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_logs=lambda: pathlib.Path(str(tmp_path)),
        # The registry installs this sidecar around every builtin dispatch; the
        # producer publishes into it. Present-and-None is the pre-dispatch state.
        _active_builtin_tool_result=None,
    )


@pytest.fixture
def fake_subprocess(monkeypatch):
    monkeypatch.setattr("ouroboros.tools.shell.load_settings", lambda: {})

    def _install(*, returncode: int = 0, stdout: str = "", stderr: str = ""):
        def fake_run(cmd, **kwargs):
            return CompletedProcess(cmd, returncode, stdout, stderr)

        monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)

    return _install


def test_run_shell_publishes_typed_failure_on_nonzero_exit(tmp_path, fake_subprocess):
    from ouroboros.tools.shell import _run_shell
    from ouroboros.tools.tool_result import _published_tool_result

    fake_subprocess(returncode=3, stderr="permission denied")
    ctx = _ctx(tmp_path)
    result = _run_shell(ctx, ["false"])
    assert result.startswith("⚠️ SHELL_EXIT_ERROR:")
    published = _published_tool_result(ctx, None)
    assert isinstance(published, ToolResult)
    assert (published.status, published.code) == ("error", "SHELL_EXIT_ERROR")
    assert dict(published.meta)["exit_code"] == 3


def test_run_shell_publishes_typed_ok_on_success(tmp_path, fake_subprocess):
    from ouroboros.tools.shell import _run_shell
    from ouroboros.tools.tool_result import _published_tool_result

    fake_subprocess(stdout="fine")
    ctx = _ctx(tmp_path)
    _run_shell(ctx, ["true"])
    published = _published_tool_result(ctx, None)
    assert isinstance(published, ToolResult)
    assert (published.status, published.code) == ("ok", "OK")
    assert dict(published.meta)["exit_code"] == 0
