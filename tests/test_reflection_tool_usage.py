"""WS6: reflection surfaces a tool-usage profile so the LLM can spot under-use."""

from __future__ import annotations

from ouroboros.reflection import _tool_usage_profile


def test_tool_usage_profile_counts_and_flags_shell_reader():
    trace = {"tool_calls": [
        {"tool": "run_command", "args": {"cmd": "grep -r foo ."}},
        {"tool": "run_command", "args": {"cmd": "cat src/main.py"}},
        {"tool": "search_code", "args": {"query": "foo"}},
        {"tool": "search_code", "args": {"query": "bar"}},
        {"tool": "read_file", "args": {"path": "x.py"}},
    ]}
    profile = _tool_usage_profile(trace)
    assert "search_code×2" in profile
    assert "run_command×2" in profile
    assert "read_file×1" in profile
    # grep + cat via run_command are flagged as shell-as-reader/search.
    assert "shell-as-reader/search" in profile
    assert "2 call(s)" in profile


def test_tool_usage_profile_no_shell_reader_note_when_clean():
    trace = {"tool_calls": [
        {"tool": "query_code", "args": {"op": "symbols"}},
        {"tool": "read_file", "args": {"path": "x.py"}},
    ]}
    profile = _tool_usage_profile(trace)
    assert "query_code×1" in profile
    assert "shell-as-reader" not in profile


def test_tool_usage_profile_empty():
    assert _tool_usage_profile({"tool_calls": []}) == "(no tool calls recorded)"
    assert _tool_usage_profile({}) == "(no tool calls recorded)"


def test_untyped_and_autocorrected_calls_do_not_trigger_an_error_reflection():
    """The reflection triggers read the ok-status SSOT, not a private tuple.

    `untyped` is what a dynamic provider body carries when nothing typed it — a
    SUCCESSFUL extension call has it — and `ok_autocorrected` is a command whose
    regex the host repaired. The private `("", "ok")` spelling counted both as
    errors, so a clean run was handed the error-reflection prompt with no error to
    reflect on, and the error counter in the prompt disagreed with the trace.
    """
    from ouroboros.reflection import (
        _collect_error_details,
        _has_error_evidence,
        _trace_call_errored,
    )

    clean = {"tool_calls": [
        {"tool": "ext_1_demo_screenshot", "status": "untyped", "is_error": False, "result": "{}"},
        {"tool": "run_command", "status": "ok_autocorrected", "is_error": False, "result": "exit_code=0"},
        {"tool": "read_file", "status": "ok", "is_error": False, "result": "file body"},
        {"tool": "write_file", "status": "", "is_error": False, "result": "written"},
    ]}
    assert [_trace_call_errored(call) for call in clean["tool_calls"]] == [False] * 4
    assert _has_error_evidence(clean) is False
    assert _collect_error_details(clean) == "(no error details captured)"

    # Every other status keeps its meaning, and is_error alone still counts.
    for status in ("blocked", "timeout", "non_zero_exit", "tool_reported_failure", "unavailable"):
        assert _trace_call_errored({"tool": "t", "status": status, "is_error": False}) is True
    assert _trace_call_errored({"tool": "t", "status": "ok", "is_error": True}) is True
