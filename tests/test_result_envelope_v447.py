"""Typed result envelope + notes-after-payload contract (#447, В12=A minimal).

Producers that KNOW their outcome stamp typed facts on the result string
(a str subclass), and every host note — auto-route note, safety warning,
post-exec tripwire — TRAILS the payload. Two regressions this pins:

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
    _structured_tool_failure,
)
from ouroboros.tools.registry import _compose_execute_result
from ouroboros.tools.result_envelope import (
    annotate,
    append_note,
    result_payload_text,
    typed_result_meta,
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
    assert _is_tool_execution_failure(True, out) is True
    meta = _extract_result_metadata("run_command", out, True)
    assert meta["status"] == "non_zero_exit"
    assert meta["exit_code"] == 1


def test_safety_warning_with_typed_shell_meta_keeps_error_status():
    """Migrated-producer path: typed status survives every appended note."""
    produced = annotate(_EXIT1, status="non_zero_exit", is_failure=True)
    out = _compose_execute_result(produced, "⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: note", _WARNING)
    assert _is_tool_execution_failure(True, out) is True
    meta = _extract_result_metadata("run_command", out, True)
    assert meta["status"] == "non_zero_exit"
    # Both host notes are recorded as typed facts, in append order.
    assert meta["notes"][0].startswith("⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE")
    assert meta["notes"][1] == _WARNING
    # Payload owns line 1; notes trail it.
    assert out.splitlines()[0].startswith("⚠️ SHELL_EXIT_ERROR")
    assert out.index(_WARNING) > out.index("boom")


def test_typed_ok_status_is_not_flipped_by_a_trailing_warning():
    produced = annotate("exit_code=0 (cwd=/tmp)\nSTDOUT:\nfine", status="ok", is_failure=False)
    out = _compose_execute_result(produced, "", _WARNING)
    assert _is_tool_execution_failure(True, out) is False
    assert _extract_result_metadata("run_command", out, False)["status"] == "ok"


# ---------------------------------------------------------------------------
# Extension/MCP structured failure survives an appended warning
# ---------------------------------------------------------------------------


def test_structured_ext_failure_detected_behind_appended_warning():
    payload = '{"ok": false, "error": "screenshot backend died"}'
    out = append_note(payload, _WARNING)
    assert result_payload_text(out) == payload
    assert _structured_tool_failure(out) is True
    assert _is_tool_execution_failure(True, out) is True
    meta = _extract_result_metadata("mcp__srv__shot", out, True)
    assert meta["status"] == "tool_reported_failure"


def test_structured_failure_stays_narrow_for_plain_prose_with_json_prefix():
    # A plain string whose FULL text is not a JSON object is untouched.
    assert _structured_tool_failure('{"ok": false} trailing prose') is False


# ---------------------------------------------------------------------------
# result_meta extension: policy_contract / remaining_route passthrough
# ---------------------------------------------------------------------------


def test_policy_contract_and_remaining_route_travel_into_result_meta():
    refusal = annotate(
        "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks this write.",
        status="light_mode_blocked",
        is_failure=True,
        policy_contract="runtime_mode.light",
        remaining_route="root=user_files or root=task_drive",
    )
    meta = _extract_result_metadata("write_file", refusal, True)
    assert meta["status"] == "light_mode_blocked"
    assert meta["policy_contract"] == "runtime_mode.light"
    assert meta["remaining_route"] == "root=user_files or root=task_drive"


# ---------------------------------------------------------------------------
# Worst-first producer: the real shell run stamps typed facts
# ---------------------------------------------------------------------------


def _ctx(tmp_path):
    return SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_logs=lambda: pathlib.Path(str(tmp_path)),
    )


@pytest.fixture
def fake_subprocess(monkeypatch):
    monkeypatch.setattr("ouroboros.tools.shell.load_settings", lambda: {})

    def _install(*, returncode: int = 0, stdout: str = "", stderr: str = ""):
        def fake_run(cmd, **kwargs):
            return CompletedProcess(cmd, returncode, stdout, stderr)

        monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)

    return _install


def test_run_shell_stamps_typed_failure_meta_on_nonzero_exit(tmp_path, fake_subprocess):
    from ouroboros.tools.shell import _run_shell

    fake_subprocess(returncode=3, stderr="permission denied")
    result = _run_shell(_ctx(tmp_path), ["false"])
    assert result.startswith("⚠️ SHELL_EXIT_ERROR:")
    meta = typed_result_meta(result)
    assert meta is not None and meta["status"] == "non_zero_exit" and meta["is_failure"] is True


def test_run_shell_stamps_typed_ok_meta_on_success(tmp_path, fake_subprocess):
    from ouroboros.tools.shell import _run_shell

    fake_subprocess(stdout="fine")
    result = _run_shell(_ctx(tmp_path), ["true"])
    meta = typed_result_meta(result)
    assert meta is not None and meta["status"] == "ok" and meta["is_failure"] is False
