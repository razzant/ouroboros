"""The stored trace preview stays what it always was; each trace row keeps its own outcome.

The bounded preview is what the task card, parents and children read. Each row keeps
its own recorded outcome, never a namesake's; the sanitizer markers that return a cut
argument to its call blob are one shared list.
"""
import json

import pytest

from ouroboros._outcome_tool_errors import _OK_TOOL_STATUSES
from ouroboros.post_task_synthesis import build_trace_summary

REFUSAL = "⚠️ QUIZ_WAIT_BOUND_INVALID: max_wait_minutes applies only to wait_for_answer=true.\nsecond line"


def _refused(round_number: int) -> dict:
    return {"tool": "escalate", "tool_call_id": f"call-{round_number}",
            "args": {"question": "Which theme?", "options": ["Light", "Dark"], "assumption": "Light",
                     "wait_for_answer": False, "max_wait_minutes": 0},
            "result": REFUSAL, "is_error": True, "status": "argument_error",
            "round_id": f"exec_x:round:{round_number}"}


def _ok(tool: str, round_number: int, **args) -> dict:
    return {"tool": tool, "tool_call_id": f"ok-{round_number}", "args": args, "result": f"{tool} fine",
            "is_error": False, "status": "ok", "round_id": f"exec_x:round:{round_number}"}


def _streak_trace() -> dict:
    calls = [_ok("read_file", 1, path="a.md", start_line=1, end_line=9)]
    calls += [_refused(number) for number in range(2, 12)]
    calls += [_ok("send_user_message", 12, text="DONE")]
    return {"tool_calls": calls, "reasoning_notes": ["a note"]}


def test_the_stored_preview_is_byte_identical_to_what_it_always_was():
    trace = _streak_trace()
    preview = build_trace_summary(trace)
    assert preview.splitlines()[0] == "## Tool trace (12 calls, 10 errors)"
    assert preview.count("escalate(") == 10 and "×" not in preview and "←" not in preview
    assert "⚠️ OMISSION NOTE: 3 more args omitted" in preview
    assert preview.splitlines()[1] == (
        "1. read_file(path='a.md', start_line='1', ⚠️ OMISSION NOTE: 1 more args omitted) [status=ok]")
    long_trace = {"tool_calls": [_ok("read_file", number, path=f"f{number}") for number in range(1, 41)]}
    preview = build_trace_summary(long_trace)
    assert "⚠️ OMISSION NOTE: 10 middle tool calls omitted from trace summary." in preview
    assert "15. read_file(path='f15')" in preview and "16. read_file" not in preview and "26. read_file(path='f26')" in preview


def test_memory_maintenance_keeps_its_own_depth(monkeypatch):
    """Consolidation of large memory inputs keeps the helper's default depth until its own
    owner decision."""
    import inspect

    from ouroboros import consolidator

    assert inspect.signature(consolidator._call_consolidation_llm).parameters["reasoning_effort"].default == "low"
    assert not hasattr(consolidator, "CONSOLIDATION_REASONING_EFFORT")


def test_the_trace_row_carries_the_round_that_issued_the_call(tmp_path):
    from ouroboros.loop_tool_execution import _execute_single_tool, process_tool_results
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "note.md").write_text("hello", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="t-round")
    registry.set_context(ctx)
    ctx._current_llm_call_meta = {"execution_id": "exec_r", "round_id": "exec_r:round:7", "llm_call_id": "llm_1"}
    logs = tmp_path / "logs"
    logs.mkdir()
    good = {"id": "c-good", "function": {"name": "read_file", "arguments": json.dumps({"path": "note.md"})}}
    malformed = {"id": "c-bad", "function": {"name": "read_file", "arguments": "{not json"}}
    results = [_execute_single_tool(registry, call, logs, "t-round") for call in (good, malformed)]
    llm_trace = {"tool_calls": []}
    process_tool_results(results, [], llm_trace, emit_progress=lambda _m, *, incident=None: None, tools=registry)
    assert [row["round_id"] for row in llm_trace["tool_calls"]] == ["exec_r:round:7", "exec_r:round:7"]
    assert llm_trace["tool_calls"][1]["is_error"] is True




@pytest.mark.parametrize("args,cut", [
    ({"handle": {"_repr": "<socket object at 0x1>"}}, True),
    ({"payload": {"_error": "sanitization_failed"}}, True),
    ({"note": "_truncated"}, False),
])
def test_cut_detection_reads_the_one_shared_sanitizer_marker_list(tmp_path, args, cut):
    """A hand-rolled subset missed `_repr` and `_error` — the rows whose arguments survive
    ONLY in the call blob — and its colon-less tokens fired on a literal value."""
    from ouroboros.artifacts import SANITIZER_OMISSION_MARKERS, materialize_tool_args_source

    assert '"_repr":' in SANITIZER_OMISSION_MARKERS and '"_error":' in SANITIZER_OMISSION_MARKERS
    call = {"tool": "demo", "tool_call_id": "d1", "args": args, "result": "OK",
            "is_error": False, "status": "ok", "round_id": "e:round:1"}
    recovered, complete, issue = materialize_tool_args_source(tmp_path, call)
    # a cut row goes back to its call blob, and with none recorded it is a named gap
    assert (not complete and issue.get("status") == "source_unavailable") is cut
    assert recovered == args


def test_a_successful_untyped_or_autocorrected_call_is_not_a_failure_anywhere():
    """`untyped` (a successful extension/MCP body) and `ok_autocorrected` (a shell command the host
    repaired) are ok statuses in the one SSOT; a private spelling of "ok" once told a clean run that
    every round had failed."""
    calls = [{"tool": "ext_demo", "tool_call_id": "u1", "args": {"value": 1}, "result": "hello from extension",
              "is_error": False, "status": "untyped", "round_id": "e:round:1"},
             {"tool": "run_command", "tool_call_id": "u2", "args": {"cmd": "grep -E x"}, "result": "match",
              "is_error": False, "status": "ok_autocorrected", "round_id": "e:round:2"}]
    assert {call["status"] for call in calls} <= _OK_TOOL_STATUSES
    assert build_trace_summary({"tool_calls": calls}).splitlines()[0] == "## Tool trace (2 calls, 0 errors)"
