import pytest

from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _compose_execute_result_result,
)


def test_dead_claude_code_result_branches_have_no_production_emitters():
    from pathlib import Path

    markers = (
        "CLAUDE_CODE_ERROR",
        "CLAUDE_CODE_TIMEOUT",
        "CLAUDE_CODE_INSTALL_ERROR",
        "CLAUDE_CODE_UNAVAILABLE",
        "claude_code_error",
    )
    sources = list(Path("ouroboros").rglob("*.py"))
    sources.extend(Path("supervisor").rglob("*.py"))
    sources.append(Path("server.py"))

    emitted = {
        str(path): marker
        for path in sources
        for marker in markers
        if marker in path.read_text(encoding="utf-8")
    }
    assert emitted == {}


def test_late_tool_settlement_runs_owner_cleanup_before_lease_close(monkeypatch):
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as lte

    class ImmediateFuture:
        def add_done_callback(self, callback):
            callback(self)

    calls = []
    monkeypatch.setattr(lte, "emit_cognitive_operation_event",
                        lambda *args, **kwargs: calls.append("lease"))
    tools = SimpleNamespace(_ctx=SimpleNamespace(event_queue=None, task_attempt=None))

    lte._attach_late_tool_settlement(
        tools,
        ImmediateFuture(),
        task_id="task",
        tool_call_id="call",
        correlation={},
        on_settled=lambda: calls.append("cleanup"),
    )

    # The ORDER is the contract: the owner-thread cleanup runs before the
    # cognitive lease closes, so a settled lease never precedes live handles.
    assert calls == ["cleanup", "lease"]


def test_get_tool_timeout_follows_the_handlers_own_child_timeout(monkeypatch):
    """The OUTER wait of a handler that bounds its own child (run_command, run_script,
    verify_and_record) is that child's resolved timeout plus the settlement margin, never
    below the registered window, so the inner timeout fires and settles first (#1376).
    The rule it replaces raised the wait only for an explicit run_command/run_script
    override, so the 600 s default child ran under a 360 s wait. The real entries carry
    dispatch's own alias folding (``timeout`` -> ``timeout_sec``)."""
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as lte
    import ouroboros.tools.shell_process as shell_process
    from ouroboros.tools import shell, verify

    monkeypatch.setattr(lte, "load_settings", lambda: {})
    monkeypatch.setattr(shell_process, "load_settings", lambda: {})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    tools = SimpleNamespace(get_timeout=lambda name: 360,
                            _entries={entry.name: entry for entry in [*shell.get_tools(), *verify.get_tools()]})

    from ouroboros.config import SETTINGS_DEFAULTS, get_per_call_timeout_ceiling_sec

    ceil = get_per_call_timeout_ceiling_sec()
    margin = lte._PER_CALL_TIMEOUT_OUTER_MARGIN_SEC
    default = min(int(SETTINGS_DEFAULTS["OUROBOROS_TOOL_TIMEOUT_SEC"]), ceil)
    assert lte._get_tool_timeout(tools, "run_command", {}) == default + margin  # the default child, not 360
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": 900}) == min(900, ceil) + margin
    assert lte._get_tool_timeout(tools, "run_script", {"timeout": 600}) == min(600, ceil) + margin  # alias
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": 5000}) == ceil + margin  # clamped
    assert lte._get_tool_timeout(tools, "verify_and_record", {"timeout_sec": 900}) == min(900, ceil) + margin
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": 5}) == 360  # never below the window
    assert lte._get_tool_timeout(tools, "read_file", {"timeout_sec": 900}) == 360  # non-shell tool ignores it
    assert lte._get_tool_timeout(tools, "run_command", {"timeout_sec": "abc"}) == default + margin  # garbage -> default


def test_review_blocked_is_not_treated_as_tool_failure():
    assert not _is_tool_execution_failure(True, "⚠️ REVIEW_BLOCKED: reviewers unavailable")


def test_domain_errors_are_not_treated_as_tool_failures():
    assert not _is_tool_execution_failure(True, "⚠️ GIT_ERROR (commit): hook rejected commit")


def test_native_review_and_git_codes_preserve_legacy_status_without_text_authority():
    review = ToolResult(
        status="ok",
        code="REVIEW_BLOCKED",
        text="review rejection text without a marker",
    )
    git = ToolResult(
        status="ok",
        code="GIT_ERROR",
        text="git refusal text without a marker",
    )

    assert not _is_tool_execution_failure(True, review.text, review)
    assert not _is_tool_execution_failure(True, git.text, git)
    # T1 §A.17: both refusals get their own outcome bucket; is_error stays false,
    # so a blocked commit is still not a reviewable effect.
    assert _extract_result_metadata(
        "commit_reviewed", review.text, False, review,
    )["status"] == "review_blocked"
    assert _extract_result_metadata(
        "vcs_read", git.text, False, git,
    )["status"] == "git_error"


def test_binding_result_mappings_preserve_legacy_loop_classification():
    cases = (
        # T1 §A.17: the argument error and the version-control refusal each carry
        # their own outcome bucket instead of the shared generic `error`.
        ("query_code", "⚠️ TOOL_ARG_ERROR (query_code): ValueError: bad root", "error", "TOOL_ARG_ERROR", True, "argument_error"),
        ("apply_patch", "⚠️ TOOL_ERROR: ValueError: bad root", "error", "TOOL_ERROR", True, "error"),
        ("vcs_read", "⚠️ GIT_ERROR: ValueError: bad root", "ok", "GIT_ERROR", False, "git_error"),
    )
    for tool, text, status, code, is_error, legacy_status in cases:
        typed = LegacyTextResultAdapter.from_text(tool, text)
        actual_error = _is_tool_execution_failure(True, text)
        assert (typed.status, typed.code) == (status, code)
        assert actual_error is is_error
        assert _extract_result_metadata(tool, text, actual_error)["status"] == legacy_status


def test_typed_safety_composition_no_longer_masks_the_underlying_failure():
    typed = _compose_execute_result_result(
        "apply_patch",
        "⚠️ TOOL_ERROR: failed",
        "⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: fixture",
        "⚠️ SAFETY_WARNING: inspect",
    )
    is_error = _is_tool_execution_failure(True, typed.text)

    # T1 §A.7: a safety warning glued to the FRONT of a failure used to record the
    # whole call as clean, on any tool. The wrapper now reveals what it wraps, and
    # the typed code it already carried is what the trace says.
    assert (typed.status, typed.code) == ("error", "TOOL_ERROR")
    assert typed.meta == {"route_note": True, "safety_warning": True}
    assert is_error is True
    assert _extract_result_metadata("apply_patch", typed.text, is_error)["status"] == "error"


def test_executor_failures_are_still_tool_failures():
    assert _is_tool_execution_failure(False, "anything")
    assert _is_tool_execution_failure(True, "⚠️ TOOL_ERROR (repo_commit): boom")
    assert _is_tool_execution_failure(True, "⚠️ TOOL_TIMEOUT (run_shell): exceeded 120s")


def test_shell_and_protected_failures_are_treated_as_tool_failures():
    assert _is_tool_execution_failure(
        True,
        "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1.\n\nSTDERR:\nboom",
    )
    core = "⚠️ CORE_PROTECTION_BLOCKED: edit_text attempted to modify protected files."

    assert _is_tool_execution_failure(True, core)
    assert _extract_result_metadata("edit_text", core, True)["status"] == "protected_blocked"


def test_the_skill_payload_control_branch_is_gone_because_nothing_emits_it():
    """SKILL_PAYLOAD_CONTROL_BLOCKED had a dedicated branch, two partition
    memberships and this test, and zero producers: the only occurrences in the tree
    were the classifier's own table and a test that synthesised the string. Its
    removal is asserted, not merely implied."""
    from pathlib import Path

    sources = [path for path in Path("ouroboros").rglob("*.py")]
    emitters = [
        str(path) for path in sources
        if "SKILL_PAYLOAD_CONTROL_BLOCKED" in path.read_text(encoding="utf-8")
    ]
    assert emitters == []
    from ouroboros._outcome_tool_errors import (
        _BLOCKING_TOOL_STATUSES,
        _POLICY_DENIAL_STATUSES,
    )

    assert "skill_payload_control_blocked" not in _BLOCKING_TOOL_STATUSES
    assert "skill_payload_control_blocked" not in _POLICY_DENIAL_STATUSES
    # A payload-control refusal, were one ever written again, still classifies as a
    # coarse block rather than falling through to a silent success.
    text = "⚠️ SKILL_PAYLOAD_CONTROL_BLOCKED: edit_text attempted to modify sidecars."
    assert _is_tool_execution_failure(True, text)
    assert _extract_result_metadata("edit_text", text, True)["status"] == "blocked"


def test_runtime_policy_blocks_are_semantic_tool_failures():
    cases = [
        ("write_file", "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks Ouroboros self-repo/control-plane mutation.", "light_mode_blocked"),
        ("run_command", "⚠️ SHELL_CWD_BLOCKED: cwd escapes allowed roots.", "cwd_blocked"),
        ("run_script", "⚠️ RUN_SCRIPT_BLOCKED: interpreter must be one of ['python3'].", "run_script_blocked"),
        ("run_command", "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths.", "workspace_blocked"),
        # The path/route-naming message shape production emits since the mode-aware
        # write-shape fix (guard B names the resolved offending path and the route).
        ("run_command", "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths. Blocked path: /x/data/y. Use the gated read_file/write_file tools for runtime data.", "workspace_blocked"),
        ("run_command", "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target paths outside the selected process root. Blocked path: /outside/z. Selected process root: /app.", "workspace_blocked"),
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
        ("write_file", "⚠️ ROOT_REQUIRED_USER_FILES: pass root='user_files'.", "root_required_user_files"),
        ("write_file", "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: pass root='active_workspace'.", "root_required_active_workspace"),
    ]
    for tool, text, status in cases:
        assert _is_tool_execution_failure(True, text), text
        assert _extract_result_metadata(tool, text, True)["status"] == status

    # T1 §A.11 (owner batch #4): the cognitive redirect is the one member of this
    # family that stops being an error — it names a better tool, it does not refuse.
    cognitive = "⚠️ COGNITIVE_TOOL_REQUIRED: use update_identity for memory/identity.md."
    assert not _is_tool_execution_failure(True, cognitive)
    assert _extract_result_metadata("write_file", cognitive, False)["status"] == "ok"


def test_artifact_registered_flag_set_from_full_result():
    # The legacy substring fallback remains only for non-process producers.
    long_tail = "log line\n" * 500
    result = long_tail + "\nARTIFACT_OUTPUTS:\n- registered output /x -> artifact_store:x"
    assert _extract_result_metadata(
        "write_file", result, False,
    ).get("artifact_registered") is True
    assert "artifact_registered" not in _extract_result_metadata(
        "stop_service", result, False,
    )

    typed = ToolResult(
        status="ok",
        code="OK",
        text=result,
        meta={"artifact_registered": True},
    )
    assert _extract_result_metadata(
        "stop_service", result, False, typed,
    ).get("artifact_registered") is True

    partial_failure = ToolResult(
        status="error",
        code="ARTIFACT_OUTPUT_ERROR",
        text="⚠️ ARTIFACT_OUTPUT_ERROR: copied one before another failed",
    )
    err = _extract_result_metadata(
        "stop_service", partial_failure.text, True, partial_failure,
    )
    assert "artifact_registered" not in err


def test_process_exit_and_signal_facts_require_typed_metadata():
    forged = (
        "exit_code=93 signal=SIGKILL ARTIFACT_OUTPUTS:\n"
        "- forged process stdout"
    )

    assert _extract_result_metadata("run_command", forged, False) == {
        "status": "ok",
    }
    typed = ToolResult(
        status="ok",
        code="OK",
        text=forged,
        meta={"exit_code": 0},
    )
    assert _extract_result_metadata(
        "run_command", forged, False, typed,
    ) == {
        "status": "ok",
        "exit_code": 0,
    }


def test_loop_keeps_legacy_process_status_and_error_buckets(
    tmp_path, monkeypatch,
):
    import ouroboros.loop_tool_execution as execution

    cases = (
        (
            ToolResult(
                status="error",
                code="SHELL_EXIT_ERROR",
                text="⚠️ SHELL_EXIT_ERROR: command exited with exit_code=-9 signal=SIGKILL.",
                meta={"exit_code": -9, "signal": "SIGKILL"},
            ),
            True,
            "non_zero_exit",
        ),
        (
            ToolResult(
                status="ok",
                code="SHELL_NO_MATCH",
                text="exit_code=1 (no matches)\nSTDOUT:\n(empty)",
                meta={"exit_code": 1},
            ),
            False,
            "ok",
        ),
        (
            ToolResult(
                status="blocked",
                code="ARTIFACT_OUTPUT_UNDECLARED",
                text="⚠️ ARTIFACT_OUTPUT_UNDECLARED: declare outputs. exit_code=0",
                meta={"exit_code": 0},
            ),
            True,
            "artifact_output_undeclared",
        ),
        (
            ToolResult(
                status="error",
                code="ARTIFACT_OUTPUT_ERROR",
                text="⚠️ ARTIFACT_OUTPUT_ERROR: registration failed. exit_code=0",
                meta={"exit_code": 0},
            ),
            True,
            "artifact_output_error",
        ),
        (
            ToolResult(
                status="ok",
                code="OWNER_STATE_RESTORED",
                text="exit_code=0\nSTDOUT:\nok\n\n⚠️ OWNER_STATE_RESTORED: restored.",
                meta={"exit_code": 0, "owner_state_restored": True},
            ),
            False,
            "ok",
        ),
        (
            ToolResult(
                status="blocked",
                code="LIGHT_MODE_REPO_WRITE_BLOCKED",
                text="⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED: blocked.",
                meta={"exit_code": 0, "light_repo_changed": True},
            ),
            True,
            "light_mode_blocked",
        ),
        (
            ToolResult(
                status="blocked",
                code="WORKSPACE_GIT_REF_CHANGED",
                text="⚠️ WORKSPACE_GIT_REF_CHANGED: blocked.",
                meta={"exit_code": 0, "workspace_git_refs_changed": True},
            ),
            True,
            "workspace_blocked",
        ),
    )
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    monkeypatch.setattr(execution, "persist_call", lambda *_args, **_kwargs: {})

    for index, (typed, expected_error, expected_status) in enumerate(cases):
        class FakeRegistry:
            CODE_TOOLS = frozenset()
            _ctx = None

            def execute_result(self, _name, _args):
                return typed

        row = execution._execute_single_tool(
            FakeRegistry(),
            {
                "id": f"call-{index}",
                "function": {"name": "run_command", "arguments": "{}"},
            },
            drive_logs,
            "task-process",
        )

        assert row["is_error"] is expected_error
        assert row["result_meta"]["status"] == expected_status
        for key in (
            "exit_code",
            "signal",
            "artifact_registered",
        ):
            if key in typed.meta:
                assert row["result_meta"][key] == typed.meta[key]


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
    # T1 §A.13: still a failure, and now named by the inner result's own first
    # line rather than by the wrapper's family.
    assert _is_tool_execution_failure(True, result)
    assert _extract_result_metadata("run_command", result, True)["status"] == "non_zero_exit"


@pytest.mark.parametrize("process_exit", [None, -9], ids=["no-process", "signal-exit"])
def test_live_tool_log_payload_includes_structured_result_metadata(tmp_path, monkeypatch, process_exit):
    import json
    import threading
    import time
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as loop_tool_execution
    from ouroboros.loop_tool_execution import _execute_with_timeout
    from ouroboros.tools.process_facts import publish_process_facts
    from ouroboros.tools.tool_result import ToolResult

    projected = loop_tool_execution._process_fact_fields(
        {"status": "x", "exit_code": -9, "signal": "SIGKILL", "killed_by_host": True}
    )
    assert projected == {"exit_code": -9, "signal": "SIGKILL", "killed_by_host": True}
    assert loop_tool_execution._process_fact_fields({"status": "ok"}) == {}
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    live_events = []
    late_notice = threading.Event()

    def put_nowait(envelope):
        live_events.append(envelope)
        if (envelope.get("data") or {}).get("type") == "tool_call_late":
            late_notice.set()

    def execute_result(_name, _args):
        # Outlive the soft timeout by construction: return only once the
        # terminal-wait notice was emitted.  A fixed 0.05s body raced a caller
        # descheduled between submit and its 1ms wait, which then found the
        # future already done and took the in-time branch.
        late_notice.wait(30)
        if process_exit is not None:
            publish_process_facts(returncode=process_exit, started_ts=time.monotonic(), killed_by_host=True)
            return ToolResult(status="error", code="SHELL_EXIT_ERROR", text="process failed")
        return ToolResult(status="ok", code="OK", text="OK")

    # D10 emptied FOREGROUND_MUTATIVE_TOOLS (claude_code_edit was its only
    # member); the terminal-wait plumbing stays wired for a successor, so pin
    # it with a fixture member.
    monkeypatch.setattr(
        loop_tool_execution, "FOREGROUND_MUTATIVE_TOOLS", frozenset({"fake_code_tool"})
    )
    tools = SimpleNamespace(
        CODE_TOOLS={"fake_code_tool"},
        _ctx=SimpleNamespace(event_queue=SimpleNamespace(put_nowait=put_nowait)),
        execute_result=execute_result,
    )
    result = _execute_with_timeout(
        tools,
        {"id": "call-1", "function": {"name": "fake_code_tool", "arguments": "{}"}},
        drive_logs,
        timeout_sec=0.001,
        task_id="task-1",
    )

    assert result["result"] == ("OK" if process_exit is None else "process failed")
    payloads = [event.get("data") or {} for event in live_events]
    assert any(payload.get("type") == "tool_call_late" for payload in payloads)
    assert any(payload.get("terminal_wait") is True for payload in payloads)
    [finished] = [row for row in payloads if row.get("type") == "tool_call_finished"]
    rows = [json.loads(line) for line in (drive_logs / "tools.jsonl").read_text(encoding="utf-8").splitlines()]
    [settled] = [row for row in rows if row.get("type") == "tool_call"]
    for row in (finished, settled):
        assert row["status"] == ("ok" if process_exit is None else "non_zero_exit")
        if process_exit is None:
            assert not {"exit_code", "signal", "killed_by_host", "duration_ms"}.intersection(row)
        else:
            assert (row["exit_code"], row["signal"], row["killed_by_host"]) == (-9, "SIGKILL", True)
            assert isinstance(row["duration_ms"], int) and row["duration_ms"] >= 0
            assert "pre_exec_failure" not in row and "timed_out" not in row


# Moved here from tests/test_loop_misc.py: both assert what the SINGLE
# classifier answers and who reads that answer, which is this module's subject,
# not the loop's message/round plumbing the wall module owns.

def test_a_tool_that_reports_its_own_failure_is_not_recorded_as_success():
    """Measured in the v6.81.1 OSWorld run: 329 tool calls returned `{"ok": false, ...}`
    in their JSON envelope and were recorded `is_error: false` / status "ok" — 302
    remote_exec, 20 screenshot, 5 key, 2 click. One agent killed the guest control
    server and then worked blind through 500-ing screenshots that all read as successes.
    The ⚠️-prefix convention only covers core-composed results; extension tools answer
    with JSON, so the failure has to be read from the payload."""
    from ouroboros.loop_tool_execution import (
        _extract_result_metadata,
        _is_tool_execution_failure,
    )
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    fail = '{"ok": false, "error": "/screenshot failed: HTTPError: 500"}'
    ok = '{"ok": true, "path": "/x/shot.png"}'
    assert LegacyTextResultAdapter.from_text("ext_1_r_x_screenshot", fail).code == "TOOL_REPORTED_FAILURE"
    assert _is_tool_execution_failure(True, fail) is True
    assert _extract_result_metadata("ext_1_r_x_screenshot", fail, False)["status"] == "tool_reported_failure"
    # Success and non-JSON prose are untouched.
    for benign in (ok, "plain text output", "", '["ok", false]', '{"ok": "false"}'):
        assert LegacyTextResultAdapter.from_text("ext_1_r_x_screenshot", benign).code != "TOOL_REPORTED_FAILURE", benign
        assert _is_tool_execution_failure(True, benign) is False, benign
    # A core ⚠️ result keeps its own typed status, not the new one.
    assert _extract_result_metadata("run_command", "⚠️ SHELL_EXIT_ERROR: 1", True)["status"] == "non_zero_exit"


def test_auto_attach_skips_a_result_that_declared_failure(tmp_path, monkeypatch):
    """A screenshot payload saying ok:false must not have an image lifted out of it."""
    import json as _json
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import _maybe_auto_attach_image
    # v7next adaptation, disclosed: the reference built the typed failure through
    # the extension dispatcher (_extension_completion, rows 187/188 — deferred on
    # this tree); the ONE adapter assigns the same TOOL_REPORTED_FAILURE to the
    # same body, which is exactly the single-classifier property under test.
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    attached, stamps = [], []
    import ouroboros.tools.vision as vision
    monkeypatch.setattr(vision, "attach_local_image_to_context",
                        lambda ctx, path, *, stamp=None, observation=None: stamps.append(stamp) or attached.append(path) or (True, "ok"))
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=[], drive_root=str(tmp_path)))
    body = _json.dumps({"ok": False, "error": "boom", "auto_attach_image": "/x/shot.png"})
    # The adapter types the self-reported failure; the guard reads that code.
    typed = LegacyTextResultAdapter.from_text("ext_1_r_unix_computer_use_screenshot", body)
    assert typed.code == "TOOL_REPORTED_FAILURE"
    failed = {"fn_name": "ext_1_r_unix_computer_use_screenshot", "is_error": False,
              "result": body, "tool_result": typed}
    _maybe_auto_attach_image(failed, tools)
    assert attached == [], "an image was attached from a failed result"

    ok_body = _json.dumps({"ok": True, "auto_attach_image": "/x/shot.png"})
    _maybe_auto_attach_image(
        {"fn_name": "ext_1_r_unix_computer_use_screenshot", "is_error": False,
         "result": ok_body,
         "tool_result": LegacyTextResultAdapter.from_text(
             "ext_1_r_unix_computer_use_screenshot", ok_body)},
        tools,
    )
    assert attached == ["/x/shot.png"], "a healthy payload must still attach"
    assert len(stamps) == 1 and stamps[0] is tools._ctx, "an eviction is stamped on the live slot"


def test_reviewed_mutator_soft_timeout_keeps_foreground_custody(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as execution

    events = []
    lifecycle = []
    monkeypatch.setattr(execution, "REVIEWED_MUTATIVE_TOOLS", frozenset({"fake_reviewed"}))

    def execute_result(_name, _args):
        from ouroboros.tools.tool_result import ToolResult

        lifecycle.append("running")
        time.sleep(0.05)
        lifecycle.append("settled")
        return ToolResult(status="ok", code="OK", text="review settled")

    tools = SimpleNamespace(
        CODE_TOOLS={"fake_reviewed"},
        _ctx=SimpleNamespace(
            event_queue=SimpleNamespace(put_nowait=events.append), task_metadata={},
        ),
        execute_result=execute_result,
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    started = time.perf_counter()
    result = execution._execute_with_timeout(
        tools,
        {"id": "review-call", "function": {"name": "fake_reviewed", "arguments": "{}"}},
        logs,
        timeout_sec=0.001,
        task_id="task-review",
    )

    assert time.perf_counter() - started >= 0.04
    assert lifecycle == ["running", "settled"]
    assert result["result"] == "review settled"
    payloads = [event.get("data") or {} for event in events]
    started = next(payload for payload in payloads if payload.get("type") == "tool_call_started")
    assert started.get("terminal_wait") is True and started.get("timeout_sec") is None
    assert not any(payload.get("type") == "tool_call_timeout" for payload in payloads)



def test_timed_out_stateful_tool_retires_the_generation_and_closes_on_the_worker(monkeypatch, tmp_path):
    """The #409/#440 wiring: a stateful-tool timeout RETIRES the browser
    generation immediately (the shared slot gets a fresh state; no cross-thread
    Playwright calls), queues the close on the RETIRING executor so it runs on
    the owning worker thread whenever the hung call settles (including the
    already-settled race), retires the executor WITHOUT cancelling that queued
    cleanup, and closes handles the worker created even AFTER the detach."""
    from types import SimpleNamespace
    from contextvars import Context, ContextVar

    import ouroboros.loop_tool_execution as lte
    from ouroboros.tools.registry import BrowserState

    closed = []
    page = SimpleNamespace(close=lambda: closed.append("page"))
    context = SimpleNamespace(close=lambda: closed.append("context"))
    chromium = SimpleNamespace(close=lambda: closed.append("browser"))
    pw = SimpleNamespace(stop=lambda: closed.append("playwright"))

    bs = BrowserState()
    bs.page, bs.browser, bs.pw_instance = page, chromium, pw
    setattr(bs, "_browser_context", context)
    setattr(bs, "_thread_id", 1)

    # A real unfinished Future retains the settlement callback while the wait expires.
    from concurrent.futures import Future

    class CleanupFuture:
        def __init__(self):
            self.callbacks = []

        def add_done_callback(self, callback):
            self.callbacks.append(callback)

    hung = Future()
    cleanup_future = CleanupFuture()

    class FakeExecutor:
        def __init__(self):
            self.queued = []
            self.retired = False
            self.reset_called = False

        def submit(self, fn, *args, **kwargs):
            if not self.queued:
                self.queued.append(("tool", fn, args))
                return hung
            self.queued.append(("cleanup", fn, args))
            return cleanup_future

        def retire(self):
            self.retired = True

        def reset(self):
            self.reset_called = True

    executor = FakeExecutor()
    monkeypatch.setattr(lte, "emit_cognitive_operation_event", lambda *a, **k: None)
    monkeypatch.setattr(lte, "_emit_live_log", lambda *a, **k: None)
    monkeypatch.setattr(
        lte, "_make_timeout_result",
        lambda *a, **k: {"tool_call_id": "call", "result": "timeout", "is_error": True},
    )
    tools = SimpleNamespace(
        _ctx=SimpleNamespace(event_queue=None, task_attempt=None, browser_state=bs),
        get_timeout=lambda name: 1,
        CODE_TOOLS=set(),
    )
    tc = {"id": "call", "function": {"name": "browse_page", "arguments": "{}"}}
    monkeypatch.setattr(lte, "load_settings", lambda: {})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)

    marker = ContextVar("stateful-test")
    token = marker.set("owning-execution")
    try:
        result = lte._execute_with_timeout(
            tools, tc, tmp_path, 1, task_id="task",
            stateful_executor=executor,
        )
    finally:
        marker.reset(token)
    assert result["is_error"] is True
    # The shared slot holds a FRESH generation; the retired one keeps the
    # handles for its owner thread.
    assert tools._ctx.browser_state is not bs
    assert tools._ctx.browser_state.page is None
    # The close is QUEUED on the retiring executor (owner thread), and the
    # executor was retired WITHOUT cancelling that queued work.
    (kind, fn, args) = executor.queued[-1]
    assert kind == "cleanup" and executor.retired and not executor.reset_called
    # Execute the actual submitted forwarding chain. A queued call bound to
    # the retired generation must refuse, not enter its replacement's body.
    _, submitted, submitted_args = executor.queued[0]
    assert isinstance(submitted.__self__, Context) and submitted.__name__ == "run"
    assert submitted.__self__.get(marker) == "owning-execution"
    from concurrent.futures import ThreadPoolExecutor
    import threading
    worker_ids = []
    def on_worker(function, *values):
        worker_ids.append(threading.get_ident())
        return function(*values)
    with ThreadPoolExecutor(max_workers=1) as worker:
        refused = worker.submit(on_worker, submitted, *submitted_args).result(timeout=5)
        assert "BROWSER_SESSION_RETIRED" in refused["result"]
        hung.set_result(refused)
        assert closed == []
        # Even a handle created AFTER detach belongs to the retiring worker.
        late_page = SimpleNamespace(close=lambda: closed.append("late_page"))
        bs.page = late_page
        worker.submit(on_worker, fn, *args).result(timeout=5)
        assert closed == ["late_page", "context", "browser", "playwright"]
        assert len(cleanup_future.callbacks) == 1
        for callback in cleanup_future.callbacks:
            worker.submit(on_worker, callback, cleanup_future).result(timeout=5)
        worker.submit(on_worker, fn, *args).result(timeout=5)
    assert len(set(worker_ids)) == 1 and worker_ids[0] != threading.get_ident()
    assert closed == ["late_page", "context", "browser", "playwright"]



def test_retire_keeps_the_queued_cleanup_and_reset_cancels_it():
    """REAL executor pin for the retire()/reset() split: the queued cleanup
    survives retire() and runs on the worker thread AFTER the hung call —
    reset()'s cancel_futures would cancel exactly that task."""
    import threading

    import ouroboros.loop_tool_execution as lte

    for method, expect_ran in (("retire", True), ("reset", False)):
        executor = lte.StatefulToolExecutor()
        gate = threading.Event()
        worker_threads = []

        def _hung():
            worker_threads.append(threading.get_ident())
            gate.wait(timeout=10)
            return "done"

        ran = threading.Event()
        cleanup_thread = []

        def _cleanup():
            cleanup_thread.append(threading.get_ident())
            ran.set()

        hung_future = executor.submit(_hung)
        try:
            hung_future.result(timeout=0.05)
        except Exception:
            pass
        cleanup_future = executor.submit(_cleanup)
        getattr(executor, method)()
        gate.set()
        hung_future.result(timeout=5)
        if expect_ran:
            assert ran.wait(timeout=5), "queued cleanup was cancelled by retire()"
            # The cleanup ran on the SAME worker thread that owned the hung call.
            assert cleanup_thread == worker_threads
        else:
            assert cleanup_future.cancelled()
            assert not ran.is_set()


def test_already_settled_call_still_cleans_on_the_worker_thread():
    """REAL executor pin for the already-settled race: a cleanup queued AFTER
    the call finished still executes on the worker thread (a done-callback
    would have run on the submitting main thread instead)."""
    import threading

    import ouroboros.loop_tool_execution as lte

    executor = lte.StatefulToolExecutor()
    worker = []
    executor.submit(lambda: worker.append(threading.get_ident())).result(timeout=5)
    cleanup_thread = []
    executor.submit(lambda: cleanup_thread.append(threading.get_ident())).result(timeout=5)
    executor.retire()
    assert cleanup_thread == worker
    assert cleanup_thread[0] != threading.get_ident()


def test_bound_wrapper_refuses_a_call_that_starts_after_retirement(tmp_path):
    """A browser call whose timeout fired before the worker reached the tool
    body must refuse instead of building a session in the NEXT command's
    state (sol MAJOR: the pre-capture window). The started call still gets its
    settlement row (#1316)."""
    import json
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as lte
    from ouroboros.tools.registry import BrowserState

    old, replacement = BrowserState(), BrowserState()
    tools = SimpleNamespace(_ctx=SimpleNamespace(browser_state=replacement))
    tc = {"id": "c1", "function": {"name": "browse_page", "arguments": "{}"}}
    out = lte._execute_browser_tool_bound(tools, tc, tmp_path, "task", old, {"invocation_id": "inv-1"})
    assert out["is_error"] is True
    assert "BROWSER_SESSION_RETIRED" in out["result"]
    row = json.loads((tmp_path / "tools.jsonl").read_text(encoding="utf-8"))
    assert (row["type"], row["invocation_id"], row["status"]) == ("tool_call", "inv-1", "refused")
    # Same generation → falls through to the real executor path (patched out).
    tools._ctx.browser_state = old
    called = []
    orig = lte._execute_single_tool
    lte._execute_single_tool = lambda *a, **k: called.append(1) or {"is_error": False}
    try:
        lte._execute_browser_tool_bound(tools, tc, None, "task", old)
    finally:
        lte._execute_single_tool = orig
    assert called == [1]
