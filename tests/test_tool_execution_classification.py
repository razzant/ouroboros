from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure


def test_get_tool_timeout_honors_per_call_override(monkeypatch):
    """T3 (v6.35.0): the OUTER tool-execution timeout must rise for a per-call
    run_command/run_script timeout_sec, else the static 360s entry cap would cut
    off a long command before the handler's own subprocess timeout fires."""
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as lte

    monkeypatch.setattr(lte, "load_settings", lambda: {})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    tools = SimpleNamespace(get_timeout=lambda name: 360)

    from ouroboros.config import get_per_call_timeout_ceiling_sec

    ceil = get_per_call_timeout_ceiling_sec()
    margin = lte._PER_CALL_TIMEOUT_OUTER_MARGIN_SEC
    assert lte._get_tool_timeout(tools, "run_command", {}) == 360               # no override -> base
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": 900}) == min(max(360, 900), ceil) + margin
    assert lte._get_tool_timeout(tools, "run_script", {"timeout": 600}) == min(max(360, 600), ceil) + margin  # alias
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": 5000}) == min(5000, ceil) + margin  # clamped
    assert lte._get_tool_timeout(tools, "read_file", {"timeout_sec": 900}) == 360      # non-shell tool ignores it
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": "abc"}) == 360  # garbage -> base


def test_review_blocked_is_not_treated_as_tool_failure():
    assert not _is_tool_execution_failure(True, "⚠️ REVIEW_BLOCKED: reviewers unavailable")


def test_domain_errors_are_not_treated_as_tool_failures():
    assert not _is_tool_execution_failure(True, "⚠️ GIT_ERROR (commit): hook rejected commit")


def test_executor_failures_are_still_tool_failures():
    assert _is_tool_execution_failure(False, "anything")
    assert _is_tool_execution_failure(True, "⚠️ TOOL_ERROR (repo_commit): boom")
    assert _is_tool_execution_failure(True, "⚠️ TOOL_TIMEOUT (run_shell): exceeded 120s")


def test_shell_and_claude_failures_are_treated_as_tool_failures():
    assert _is_tool_execution_failure(
        True,
        "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1.\n\nSTDERR:\nboom",
    )
    assert _is_tool_execution_failure(
        True,
        "⚠️ CLAUDE_CODE_INSTALL_ERROR: unable to install Claude Code.",
    )
    assert _is_tool_execution_failure(
        True,
        "⚠️ CLAUDE_CODE_UNAVAILABLE: ANTHROPIC_API_KEY not set.",
    )
    core = "⚠️ CORE_PROTECTION_BLOCKED: claude_code_edit attempted to modify protected files."
    skill = "⚠️ SKILL_PAYLOAD_CONTROL_BLOCKED: claude_code_edit attempted to modify sidecars."

    assert _is_tool_execution_failure(True, core)
    assert _is_tool_execution_failure(True, skill)
    assert _extract_result_metadata("claude_code_edit", core, True)["status"] == "protected_blocked"
    assert _extract_result_metadata("claude_code_edit", skill, True)["status"] == "skill_payload_control_blocked"


def test_runtime_policy_blocks_are_semantic_tool_failures():
    cases = [
        ("write_file", "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks Ouroboros self-repo/control-plane mutation.", "light_mode_blocked"),
        ("run_command", "⚠️ SHELL_CWD_BLOCKED: cwd escapes allowed roots.", "cwd_blocked"),
        ("run_script", "⚠️ RUN_SCRIPT_BLOCKED: interpreter must be one of ['python3'].", "run_script_blocked"),
        ("run_command", "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths.", "workspace_blocked"),
        ("run_command", "⚠️ ELEVATION_BLOCKED: shell command pattern looks like an elevation attempt.", "elevation_blocked"),
        ("run_command", "⚠️ SKILL_STATE_WRITE_BLOCKED: skill trust state is owner controlled.", "skill_state_blocked"),
        ("run_command", "⚠️ ARTIFACT_OUTPUT_ERROR: command succeeded but declared output registration failed.", "artifact_output_error"),
        ("integrate_subagent_patch", "⚠️ INTEGRATE_CONFLICT: patch did not apply.", "integration_blocked"),
        ("integrate_subagent_patch", "⚠️ INTEGRATE_PATCH_NOT_FOUND: no workspace_patch.json.", "integration_blocked"),
        ("integrate_subagent_patch", "⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH: patch does not match.", "integration_blocked"),
        ("run_command", "⚠️ SAFETY_VIOLATION: blocked by policy.", "safety_violation"),
        ("run_command", "⚠️ GIT_VIA_SHELL_BLOCKED: use vcs tools.", "git_via_shell_blocked"),
        ("run_command", "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false blocks git ls-remote.", "resource_constraint_blocked"),
        ("run_command", "⚠️ RESOURCE_POLICY_BLOCKED: protected black-box artifact.", "resource_policy_blocked"),
        ("write_file", "⚠️ HEAL_MODE_BLOCKED: repair scope only.", "heal_mode_blocked"),
        ("read_file", "⚠️ REPO_READ_BLOCKED: protected path.", "blocked"),
        ("write_file", "⚠️ COGNITIVE_TOOL_REQUIRED: use update_identity for memory/identity.md.", "cognitive_tool_required"),
        ("write_file", "⚠️ ROOT_REQUIRED_USER_FILES: pass root='user_files'.", "root_required_user_files"),
        ("write_file", "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: pass root='active_workspace'.", "root_required_active_workspace"),
    ]
    for tool, text, status in cases:
        assert _is_tool_execution_failure(True, text)
        assert _extract_result_metadata(tool, text, True)["status"] == status


def test_artifact_registered_flag_set_from_full_result():
    # The structured flag is captured from the full result (before the 700-char
    # trace preview), so a late ARTIFACT_OUTPUTS marker is not lost.
    long_tail = "log line\n" * 500
    result = long_tail + "\nARTIFACT_OUTPUTS:\n- registered output /x -> artifact_store:x"
    meta = _extract_result_metadata("stop_service", result, False)
    assert meta.get("artifact_registered") is True
    # An artifact-output ERROR (failed registration) must not set the success flag.
    err = _extract_result_metadata("run_command", "⚠️ ARTIFACT_OUTPUT_ERROR: boom", True)
    assert not err.get("artifact_registered")


def test_shell_regex_autocorrect_success_is_not_tool_failure():
    result = "⚠️ SHELL_REGEX_AUTO_CORRECTED: converted grep backslash-escaped alternation\nexit_code=0\nSTDOUT:\nmatch"
    assert not _is_tool_execution_failure(True, result)
    assert _extract_result_metadata("run_command", result, False)["status"] == "ok_autocorrected"


def test_shell_regex_autocorrect_with_artifact_error_still_fails():
    result = (
        "⚠️ SHELL_REGEX_AUTO_CORRECTED: converted grep backslash-escaped alternation\n"
        "⚠️ ARTIFACT_OUTPUT_ERROR: command appears to write user_files outputs without declaring outputs=[...]."
    )
    assert _is_tool_execution_failure(True, result)
    assert _extract_result_metadata("run_command", result, True)["status"] == "artifact_output_error"


def test_shell_regex_autocorrect_nonzero_still_fails():
    result = (
        "⚠️ SHELL_REGEX_AUTO_CORRECTED: converted grep backslash-escaped alternation\n"
        "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=2.\n\nSTDERR:\nboom"
    )
    assert _is_tool_execution_failure(True, result)
    assert _extract_result_metadata("run_command", result, True)["status"] == "shell_error"


def test_live_tool_log_payload_includes_structured_result_metadata(tmp_path):
    import pathlib
    import time
    from types import SimpleNamespace
    from ouroboros.loop_tool_execution import _execute_with_timeout

    source = (pathlib.Path(__file__).resolve().parents[1] / "ouroboros" / "loop_tool_execution.py").read_text(encoding="utf-8")

    assert '"status": result_meta.get("status")' in source
    assert '"exit_code": result_meta.get("exit_code")' in source
    assert '"signal": result_meta.get("signal")' in source
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    live_events = []
    tools = SimpleNamespace(
        CODE_TOOLS={"claude_code_edit"},
        _ctx=SimpleNamespace(event_queue=SimpleNamespace(put_nowait=lambda envelope: live_events.append(envelope))),
        execute=lambda _name, _args: (time.sleep(0.05), "OK")[1],
    )
    result = _execute_with_timeout(
        tools,
        {"id": "call-1", "function": {"name": "claude_code_edit", "arguments": "{}"}},
        drive_logs,
        timeout_sec=0.001,
        task_id="task-1",
    )

    assert result["result"] == "OK"
    payloads = [event.get("data") or {} for event in live_events]
    assert any(payload.get("type") == "tool_call_late" for payload in payloads)
    assert any(payload.get("terminal_wait") is True for payload in payloads)
