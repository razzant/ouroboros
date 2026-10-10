"""The post-task reflection must see what the actor saw; the stored preview must not change.

A task whose model sent the same refused call ten times produced a reflection that never
saw the refusal: the summary showed two arguments per call, no result text, a positional
window, and was cut twice (4000 then 2000 characters). These tests pin the reflection's
all-calls listing, the unchanged stored preview, and the settings-resolved effort.
"""
import json
import pathlib

import pytest

from ouroboros import reflection
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


def test_the_reflection_listing_shows_the_field_the_answer_and_the_repetition():
    listing = build_trace_summary(_streak_trace(), all_calls=True)
    # every argument, so the offending fifth one is visible
    assert "max_wait_minutes='0'" in listing and "wait_for_answer='False'" in listing
    # ten identical refused calls are ONE row that keeps its count and its rounds
    assert listing.count("escalate(") == 1
    assert "2–11. escalate(" in listing and "×10 identical, rounds 2–11" in listing
    # the answer's first line, never its tail
    assert "← ⚠️ QUIZ_WAIT_BOUND_INVALID: max_wait_minutes applies only" in listing and "second line" not in listing
    # one arithmetic fact, no label
    assert "10 of 12 rounds had only non-ok results" in listing.splitlines()[0]
    assert "OMISSION NOTE" not in listing and "- a note" in listing


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


def test_the_listing_has_no_window_and_no_total_cut_and_redacts_results():
    calls = [_ok("read_file", number, path=f"file-{number}.md", note="x" * 150) for number in range(1, 201)]
    secret = {**_refused(201), "result": "ERROR: OPENROUTER_API_KEY=sk-or-v1-" + "a" * 40 + " rejected"}
    listing = build_trace_summary({"tool_calls": calls + [secret]}, all_calls=True)
    assert len(listing) > 4000 and "middle tool calls omitted" not in listing
    assert "100. read_file(path='file-100.md'" in listing and "201. escalate(" in listing
    assert "sk-or-v1-" + "a" * 40 not in listing


def test_only_byte_identical_neighbours_fold():
    varied = [_refused(2), {**_refused(3), "args": {**_refused(3)["args"], "question": "Which one?"}}, _refused(4)]
    listing = build_trace_summary({"tool_calls": varied}, all_calls=True)
    assert listing.count("escalate(") == 3 and "×" not in listing
    # a bare-string refusal recorded ok still folds, and a repeated ok call shows what it answered
    bare = [{**_ok("schedule_subagent", number, objective="x"), "result": "⚠️ subagent_access_invalid: nope"}
            for number in (5, 6, 7)]
    listing = build_trace_summary({"tool_calls": bare}, all_calls=True)
    assert "×3 identical, rounds 5–7 ← ⚠️ subagent_access_invalid: nope" in listing


def test_identical_error_details_are_one_entry_with_its_count():
    trace = _streak_trace()
    trace["tool_calls"].append({"tool": "run_command", "is_error": True, "status": "error", "exit_code": 2,
                                "result": "make: *** [test] Error 2"})
    details = reflection._collect_error_details(trace)
    assert details.count("QUIZ_WAIT_BOUND_INVALID") == 1 and "(×10 identical) [escalate" in details
    assert "[run_command (status=error, exit_code=2)]: make:" in details and "identical) [run_command" not in details


@pytest.fixture
def captured_calls(monkeypatch):
    calls = []

    def fake_chat(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("call_type") == "pattern_register_update":
            return ({"content": reflection._PATTERNS_HEADER + "| refusal loop | 1 | cause | fix | open |\n"}, {})
        return ({"content": "The bound was refused ten times.\nMEMORY_ACTIONS_JSON: []\nBACKLOG_CANDIDATES_JSON: []"}, {})

    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    return calls


def test_the_reflection_prompt_carries_the_whole_listing_and_an_optional_verbatim_record(tmp_path, captured_calls):
    trace = _streak_trace()
    listing = build_trace_summary(trace, all_calls=True) + "\n" + "\n".join(f"pad line {n}" for n in range(400))
    assert len(listing) > 4000
    entry = reflection.generate_reflection(
        {"id": "task-streak", "text": "Ask the owner", "drive_root": str(tmp_path)},
        trace, listing, object(), {"rounds": 12, "cost": 0.0})
    assert entry["reflection"].startswith("The bound was refused")
    prompt = next(call for call in captured_calls if call.get("call_type") != "pattern_register_update")["messages"][0]["content"]
    assert "×10 identical, rounds 2–11" in prompt and "pad line 399" in prompt  # no second cut
    assert "truncated at 2000 chars" not in prompt
    # the verbatim record is named with the reader the reflection already holds, and is really there
    marker = "Complete stored record of every call"
    assert marker in prompt
    arguments = json.loads(prompt[prompt.index(marker):].split("read_file ", 1)[1].splitlines()[0])
    assert arguments["root"] == "runtime_data"
    record = (pathlib.Path(tmp_path) / arguments["path"]).read_text(encoding="utf-8")
    assert record.count("### ") == 12 and "second line" in record and '"max_wait_minutes": 0' in record
    assert "round_id=exec_x:round:11" in record


@pytest.mark.parametrize("level", ["medium", "high"])
def test_post_task_synthesis_thinks_at_the_owners_task_level_not_a_literal(tmp_path, monkeypatch, captured_calls, level):
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", level)
    monkeypatch.setattr("ouroboros.settings_scales.runtime_setting",
                        lambda key, default=None: level if key == "OUROBOROS_EFFORT_TASK" else default)
    entry = reflection.generate_reflection(
        {"id": "task-effort", "text": "Ask the owner", "drive_root": str(tmp_path)},
        _streak_trace(), "trace", object(), {"rounds": 12, "cost": 0.0})
    reflection.append_reflection(tmp_path, entry)
    efforts = {call.get("call_type"): call.get("reasoning_effort") for call in captured_calls}
    assert efforts.get("task_reflection") == level
    assert efforts.get("pattern_register_update") == level


def test_memory_maintenance_keeps_its_own_depth(monkeypatch):
    """Only post-task synthesis moved to the Task / Chat level; consolidation of large memory
    inputs keeps the helper's default until its own owner decision."""
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


def test_the_self_check_lists_outcomes_and_a_failed_answer():
    from ouroboros.loop_nudges import _build_recent_tool_trace

    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "escalate", "arguments": '{"question":"q","max_wait_minutes":0}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": REFUSAL},
        {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]},
    ]
    trace = {"tool_calls": [
        {"tool": "escalate", "tool_call_id": "c1", "args": {"question": "q", "max_wait_minutes": 0},
         "status": "argument_error", "is_error": True, "result": REFUSAL},
        {"tool": "read_file", "tool_call_id": "c2", "args": {"path": "a"},
         "status": "ok", "is_error": False, "result": "file body"}]}
    rendered = _build_recent_tool_trace(messages, llm_trace=trace)
    assert ('1. escalate({"max_wait_minutes": 0, "question": "q"}) [argument_error] ← '
            "⚠️ QUIZ_WAIT_BOUND_INVALID: max_wait_minutes applies only") in rendered
    assert '2. read_file({"path": "a"}) [ok]' in rendered and "file body" not in rendered and "second line" not in rendered
    # without a trace the list is exactly what it was
    assert _build_recent_tool_trace(messages) == (
        'Recent tool calls (oldest first):\n  1. escalate({"question":"q","max_wait_minutes":0})\n  2. read_file({"path":"a"})')


def test_each_trace_row_carries_its_own_outcome_under_a_reused_provider_id():
    """Providers are not required to mint unique call ids: GigaChat answers `call_0` every
    round and the local parser `call_local_<i>`, so ONE id names several calls. Keyed by id
    alone, the last call overwrote its namesakes and an early failed call was printed with a
    later call's error — the prompt that asks "are you repeating yourself?" accusing the
    wrong call. Name, arguments and outcome now come from one record."""
    from ouroboros.loop_nudges import _build_recent_tool_trace

    trace = {"tool_calls": [
        {"tool": "escalate", "tool_call_id": "call_0", "args": {"a": 1},
         "status": "argument_error", "is_error": True, "result": "FIRST refusal"},
        {"tool": "read_file", "tool_call_id": "call_0", "args": {"b": 2},
         "status": "ok", "is_error": False, "result": "second body"}]}
    rendered = _build_recent_tool_trace([], llm_trace=trace)
    assert '1. escalate({"a": 1}) [argument_error] ← FIRST refusal' in rendered
    assert '2. read_file({"b": 2}) [ok]' in rendered


def test_a_sanitizer_truncated_argument_still_names_a_source(tmp_path):
    """The log sanitizer replaces an oversized value with a SHORT marker, so a width test
    over the stored args measured the widest argument in the task as a small one and the
    record — the only place its result survives in full — was never retained at all."""
    from ouroboros.tools.registry import ToolContext
    from ouroboros.utils import sanitize_tool_args_for_log

    args = sanitize_tool_args_for_log("write_file", {"content": "x" * 5000})
    assert "<TRUNCATED:" in json.dumps(args)  # what the trace really holds is short
    call = {"tool": "write_file", "tool_call_id": "w1", "args": args, "result": "OK",
            "is_error": False, "status": "ok", "round_id": "exec_x:round:1"}
    pointer = reflection._verbatim_trace_pointer(
        ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="cut"), {"tool_calls": [call]})
    assert "Complete stored record of every call" in pointer
    arguments = json.loads(pointer.split("read_file ", 1)[1].splitlines()[0])
    record = (pathlib.Path(tmp_path) / arguments["path"]).read_text(encoding="utf-8")
    assert "<TRUNCATED:content:5000ch:sha=" in record
    # the claim names what it holds: the stored trace, not the original arguments
    assert "as the TRACE retained" in pointer and "as the actor saw it" not in pointer
    # and it does not promise the result in full: the trace stores the actor-visible cap,
    # naming the marker the record itself carries
    assert "each result in full" not in pointer and "FULL_RESULT_SOURCE_JSON" in pointer


@pytest.mark.parametrize("args,cut", [
    ({"handle": {"_repr": "<socket object at 0x1>"}}, True),
    ({"payload": {"_error": "sanitization_failed"}}, True),
    ({"note": "_truncated"}, False),
])
def test_cut_detection_reads_the_one_shared_sanitizer_marker_list(tmp_path, args, cut):
    """A hand-rolled subset missed `_repr` and `_error` — the rows whose arguments survive
    ONLY in the call blob — and its colon-less tokens fired on a literal value."""
    from ouroboros.artifacts import SANITIZER_OMISSION_MARKERS
    from ouroboros.tools.registry import ToolContext

    assert '"_repr":' in SANITIZER_OMISSION_MARKERS and '"_error":' in SANITIZER_OMISSION_MARKERS
    call = {"tool": "demo", "tool_call_id": "d1", "args": args, "result": "OK",
            "is_error": False, "status": "ok", "round_id": "e:round:1"}
    pointer = reflection._verbatim_trace_pointer(
        ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="markers"), {"tool_calls": [call]})
    assert bool(pointer) is cut


def test_the_same_tool_twice_under_one_id_keeps_its_own_outcome_after_compaction():
    """The counterexample no join survives: the SAME tool called twice under one reused id,
    with the earlier call evicted from `messages` by compaction. Matching (id, tool) picked
    the earlier row, so the surviving call printed the OLD refusal. Reading the trace row
    itself cannot mispair, and the evicted call stays visible as what the actor really did."""
    from ouroboros.loop_nudges import _build_recent_tool_trace

    trace = {"tool_calls": [
        {"tool": "read_file", "tool_call_id": "call_0", "args": {"path": "old"},
         "status": "argument_error", "is_error": True, "result": "OLD refusal"},
        {"tool": "read_file", "tool_call_id": "call_0", "args": {"path": "new"},
         "status": "ok", "is_error": False, "result": "new body"}]}
    # compaction left only the later call in the visible transcript
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "[compacted capsule]"}]},
        {"role": "assistant", "tool_calls": [
            {"id": "call_0", "function": {"name": "read_file", "arguments": '{"path":"new"}'}}]},
    ]
    rendered = _build_recent_tool_trace(messages, llm_trace=trace)
    assert '1. read_file({"path": "old"}) [argument_error] ← OLD refusal' in rendered
    assert '2. read_file({"path": "new"}) [ok]' in rendered
    # the later call is not handed the earlier one's failure
    assert '{"path": "new"}) [argument_error]' not in rendered


def test_without_a_trace_the_messages_render_with_no_borrowed_outcome():
    """A missing outcome is honest; a namesake's error on the wrong call is not."""
    from ouroboros.loop_nudges import _build_recent_tool_trace

    messages = [{"role": "assistant", "tool_calls": [
        {"id": "call_0", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "call_0", "function": {"name": "write_file", "arguments": "{}"}}]}]
    rendered = _build_recent_tool_trace(messages, llm_trace={"tool_calls": []})
    assert "1. read_file({})" in rendered and "2. write_file({})" in rendered
    assert "[" not in rendered.split("oldest first):", 1)[1]


def test_a_successful_untyped_or_autocorrected_call_is_not_a_failure_anywhere():
    """`untyped` (a successful extension/MCP body) and `ok_autocorrected` (a shell command the host
    repaired) are ok statuses in the one SSOT; a private spelling of "ok" once told a clean run that
    every round had failed."""
    from ouroboros.loop_nudges import _build_recent_tool_trace

    calls = [{"tool": "ext_demo", "tool_call_id": "u1", "args": {"value": 1}, "result": "hello from extension",
              "is_error": False, "status": "untyped", "round_id": "e:round:1"},
             {"tool": "run_command", "tool_call_id": "u2", "args": {"cmd": "grep -E x"}, "result": "match",
              "is_error": False, "status": "ok_autocorrected", "round_id": "e:round:2"}]
    listing = build_trace_summary({"tool_calls": calls}, all_calls=True)
    assert listing.splitlines()[0] == "## Tool trace (2 calls, 0 errors)" and "←" not in listing
    messages = [{"role": "assistant", "tool_calls": [
        {"id": "u1", "function": {"name": "ext_demo", "arguments": "{}"}},
        {"id": "u2", "function": {"name": "run_command", "arguments": "{}"}}]}]
    rendered = _build_recent_tool_trace(messages, llm_trace={"tool_calls": calls})
    assert "[untyped]" in rendered and "[ok_autocorrected]" in rendered and "←" not in rendered


def test_a_trace_the_listing_shows_whole_retains_no_verbatim_record(tmp_path, captured_calls):
    trace = {"tool_calls": [_ok("read_file", 1, path="a.md"),
                            {**_refused(2), "result": "⚠️ QUIZ_WAIT_BOUND_INVALID: one line only."}]}
    reflection.generate_reflection(
        {"id": "task-small", "text": "Ask the owner", "drive_root": str(tmp_path)},
        trace, build_trace_summary(trace, all_calls=True), object(), {"rounds": 2, "cost": 0.0})
    prompt = next(call for call in captured_calls if call.get("call_type") != "pattern_register_update")["messages"][0]["content"]
    assert "Complete per-call record" not in prompt



def test_different_long_errors_are_not_labelled_identical():
    calls = [{"tool": "probe", "is_error": True, "status": "error",
              "result": "x" * 1200 + ending} for ending in ("one", "two")]
    details = reflection._collect_error_details({"tool_calls": calls})
    assert "identical" not in details
    assert details.count("[probe") == 2


def test_post_task_consumer_receives_middle_calls_and_repeated_ok_source(tmp_path, captured_calls, monkeypatch):
    from types import SimpleNamespace
    from ouroboros.post_task_synthesis import _run_reflection
    from ouroboros.consolidator import KnowledgeReadContext
    from ouroboros.tools.registry import ToolContext

    calls = [_ok("read_file", n, path=f"file-{n}", start_line=1, third="KEEP_THIRD") for n in range(1, 81)]
    calls += [{**_ok("probe", n, query="same"), "result": "running\nIMPORTANT_TAIL"} for n in (81, 82)]
    trace = {"tool_calls": calls}
    preview = build_trace_summary(trace)
    monkeypatch.setattr(reflection, "append_reflection_routed", lambda *_a: None)
    entry = _run_reflection(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), object(),
        {"id": "end-to-end", "text": "Inspect outcomes", "drive_root": str(tmp_path)},
        {"rounds": 82, "cost": 0.0}, trace, {})
    assert entry is not None and "reflection generation failed" not in entry["reflection"]
    prompt = next(c for c in captured_calls if c.get("call_type") == "task_reflection")["messages"][0]["content"]
    assert "file-40" in prompt and "KEEP_THIRD" in prompt and "×2 identical" in prompt
    assert "middle tool calls omitted" not in prompt
    marker = "Complete stored record of every call"
    arguments = json.loads(prompt[prompt.index(marker):].split("read_file ", 1)[1].splitlines()[0])
    # Read through the very tool surface the reflection model holds, not Path alone.
    reader = KnowledgeReadContext(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="end-to-end"), "task_reflection")
    result = reader.read_call({"id": "read-source", "function": {"name": "read_file", "arguments": json.dumps(arguments)}})
    assert "IMPORTANT_TAIL" in str(result) and "file-40" in str(result)
    assert build_trace_summary(trace) == preview


def test_trace_source_failure_is_disclosed(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext

    def fail(*_args, **_kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr("ouroboros.chat_chain.retain_memory_source", fail)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    pointer = reflection._verbatim_trace_pointer(ctx, _streak_trace())
    assert "unavailable" in pointer and "omits" in pointer
