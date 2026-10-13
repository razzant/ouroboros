"""Independent exact arithmetic and producer/archive cases for the public reader."""
import pytest

from scripts.measure_wait_rounds import measure


def test_joined_rounds_duplicate_usage_and_child_denominator():
    rows = [
        {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80, "tool_call_count": 1},
        {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80, "tool_call_count": 1, "ts": "later duplicate"},
        {"tool": "wait_tasks", "tool_call_id": "w", "task_id": "parent", "round_id": 1},
        {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 2, "prompt_tokens": 300, "cached_tokens": 0, "tool_call_count": 1},
        {"tool": "read_file", "tool_call_id": "r", "task_id": "parent", "round_id": 2},
        {"type": "task_done", "task_id": "child", "parent_task_id": "parent", "status": "completed"},
    ]
    r = measure(rows)
    assert r["coverage_complete"]
    assert (r["usage_rounds"], r["wait_only_rounds"], r["prompt_tokens"], r["wait_prompt_tokens"]) == (2, 1, 400, 100)
    assert r["wait_round_fraction"] == 1 / 2
    assert r["wait_prompt_fraction"] == 1 / 4
    assert r["wait_rounds_per_terminal"] == 1
    assert r["targets"] == {"round_fraction_le_3pct": False, "rounds_per_terminal_le_1_5": True}


def test_missing_or_conflicting_usage_never_certifies_a_target():
    row = {"type": "llm_usage", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80}
    conflict = measure([row, {**row, "prompt_tokens": 101}, {"tool": "wait_task", "task_id": "parent", "round_id": 2}])
    assert not conflict["coverage_complete"] and conflict["prompt_tokens"] is None
    assert all(value is None for value in conflict["targets"].values())
    missing = measure([{**row, "prompt_tokens": None}, {"tool": "wait_task", "task_id": "parent", "round_id": 1}])
    assert missing["prompt_tokens"] is None and missing["wait_prompt_tokens"] is None
    assert missing["wait_prompt_fraction"] is None
    assert all(value is None for value in missing["targets"].values())


def _usage(count=1, **kw):
    return {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 1,
            "prompt_tokens": 100, "cached_tokens": 80, "tool_call_count": count, **kw}


def _tool(name="wait_tasks", call_id="a"):
    return {"tool": name, "tool_call_id": call_id, "task_id": "parent", "round_id": 1}


_DONE = {"type": "task_done", "task_id": "child", "parent_task_id": "parent", "status": "completed"}


def _unknown(rows):
    report = measure(rows)
    assert not report["coverage_complete"]
    assert all(v is None for v in report["targets"].values())
    assert report["wait_round_fraction"] is None
    return report


def test_absent_and_partial_tool_inventories_do_not_certify_zero_waits():
    row = _usage()
    row.pop("tool_call_count")
    _unknown([row, _DONE])
    _unknown([_usage(count=2), _tool(), _DONE])
    _unknown([_usage(), {"tool": "wait_tasks", "task_id": "parent", "round_id": 1}, _DONE])
    _unknown([_usage(count=2), _tool("read_file"), _DONE])


def test_explicit_zero_tool_round_and_host_finished_inventory_are_supported():
    report = measure([_usage(count=0), _DONE])
    assert report["coverage_complete"] and report["wait_only_rounds"] == 0
    row = _usage(count=0)
    row.pop("tool_call_count")
    report = measure([row, {"type": "llm_round_finished", "task_id": "parent", "round_id": 1,
                           "tool_call_count": 0, "response_kind": "message"}, _DONE])
    assert report["coverage_complete"]


def test_inventory_uses_distinct_call_ids_not_distinct_names_or_duplicates():
    report = measure([_usage(count=2), _tool(), _tool(call_id="b"), _DONE])
    assert report["coverage_complete"] and report["wait_only_rounds"] == 1
    report = measure([_usage(), _usage(), _tool(), _tool(), _DONE])
    assert report["coverage_complete"] and report["usage_rounds"] == 1
    _unknown([_usage(), _tool(), _tool("read_file"), _DONE])
    _unknown([_usage(), {**_usage(), "tool_call_count": 2}, _tool(), _DONE])


def test_original_eight_tool_family_and_no_invented_prompt_percentage_target():
    names = ["wait_task", "wait_tasks", "delegate_wait", "peek_task", "await_messages",
             "forward_to_worker", "get_task_result", "delegate_message"]
    report = measure([_usage(count=8), *[_tool(n, str(i)) for i, n in enumerate(names)], _DONE])
    assert report["coverage_complete"] and report["wait_only_rounds"] == 1
    assert "prompt_fraction_le_5pct" not in report["targets"]


def test_other_categories_and_missing_task_category_do_not_certify_task_targets():
    _unknown([_usage(category=None), _tool(), _DONE])
    report = _unknown([_usage(category="review"), _DONE])
    assert report["usage_rounds"] == 0


def test_every_observed_round_requires_usage_even_without_tool_rows():
    for count in (0, 1):
        report = _unknown([_usage(count=0), {
            "type": "llm_round_finished", "task_id": "parent", "round_id": 2,
            "tool_call_count": count}, _DONE])
        assert {"key": ["parent", "2"], "reason": "observed_round_usage_missing"} in report["coverage_gaps"]


def test_invalid_token_evidence_never_marks_global_coverage_complete():
    for key in ("prompt_tokens", "cached_tokens"):
        for value in (None, -1, True, "100"):
            report = _unknown([_usage(count=0, **{key: value}), _DONE])
            assert any(g["reason"] == "invalid_round_usage" for g in report["coverage_gaps"])
        row = _usage(count=0)
        del row[key]
        _unknown([row, _DONE])


@pytest.mark.parametrize("call_count,cached_tokens", [(0, 0), (2, 0), (2, None)])
@pytest.mark.serial
def test_durable_round_inventory_survives_actual_archive_and_measures(tmp_path, call_count, cached_tokens, monkeypatch, capsys):
    import json
    import queue
    from types import SimpleNamespace

    import ouroboros.loop_tool_execution as tool_execution
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.tools.tool_result import ToolResult
    from scripts.measure_wait_rounds import main
    from supervisor.events_budget import _handle_llm_usage
    from supervisor.state import rotate_jsonl_log_if_needed
    from tests.test_tool_call_log import _Registry

    worker, canonical = tmp_path / "worker", tmp_path / "canonical"
    logs = worker / "logs"
    logs.mkdir(parents=True)
    events = queue.Queue()

    class LLM:
        def chat(self, **_kwargs):
            return {"content": "done", "tool_calls": [
                {"id": f"wait-{i}", "type": "function", "function": {
                    "name": "wait_tasks", "arguments": "{}"}} for i in range(call_count)],
                "finish_reason": "tool_calls" if call_count else "stop"}, {
                "provider": "openai-compatible", "resolved_model": "fixture", "cost": 0.0,
                "prompt_tokens": 12, "completion_tokens": 3, "cached_tokens": cached_tokens,
            }

    answer, _ = call_llm_with_retry(LLM(), [{"role": "user", "content": "work"}],
                                    "openai-compatible::fixture", None, "medium", 1,
                                    logs, "parent", 1, events, {})
    assert answer["content"] == "done"
    usage = next(row for row in events.queue if row.get("type") == "llm_usage")
    _handle_llm_usage(usage, SimpleNamespace(DRIVE_ROOT=canonical,
                                             bridge=SimpleNamespace(push_log=lambda _event: None)))
    round_id = usage["round_id"]
    registry = _Registry(tmp_path, lambda *_: ToolResult(status="ok", code="OK", text="done"),
                         round_id=round_id)
    registry._ctx._current_llm_call_meta = {
        "execution_id": usage["execution_id"], "round_id": round_id,
        "llm_call_id": usage["llm_call_id"]}
    for call in answer["tool_calls"]:
        tool_execution._execute_with_timeout(registry, call, logs, 5, "parent")
    rotate_jsonl_log_if_needed(worker, "events.jsonl", "events", max_bytes=1)
    archive = next((worker / "archive").glob("events_*.jsonl"))
    rotate_jsonl_log_if_needed(canonical, "events.jsonl", "events", max_bytes=1)
    usage_archive = next((canonical / "archive").glob("events_*.jsonl"))
    tool_archive = None
    if call_count:
        rotate_jsonl_log_if_needed(worker, "tools.jsonl", "tools", max_bytes=1)
        tool_archive = next((worker / "archive").glob("tools_*.jsonl"))
    rows = [json.loads(line) for path in (archive, usage_archive)
            for line in path.read_text().splitlines() if line.strip()]
    durable = next(row for row in rows if row.get("type") == "llm_round")
    assert durable["tool_call_count"] == call_count
    assert durable["prompt_tokens"] == 12 and durable["cached_tokens"] == cached_tokens
    canonical_usage = next(row for row in rows if row.get("type") == "llm_usage")
    assert canonical_usage["prompt_tokens"] == 12 and canonical_usage["cached_tokens"] == cached_tokens
    if tool_archive:
        tool_rows = [json.loads(line) for line in tool_archive.read_text().splitlines() if line.strip()]
        assert [row["type"] for row in tool_rows] == ["tool_call_started", "tool_call"] * call_count
        assert len({row["tool_call_id"] for row in tool_rows}) == call_count
    monkeypatch.setattr("sys.argv", ["measure_wait_rounds.py", str(archive), str(usage_archive),
                                  *([str(tool_archive)] if tool_archive else [])])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["usage_rounds"] == 1 and report["prompt_tokens"] == 12
    assert report["wait_only_rounds"] == int(bool(call_count))
    if cached_tokens is None:
        assert not report["coverage_complete"] and report["wait_cached_tokens"] is None
        assert any(gap["reason"] == "invalid_round_usage" for gap in report["coverage_gaps"])
        assert all(value is None for value in report["targets"].values())
        assert report["wait_round_fraction"] is None
    else:
        assert report["coverage_complete"]
        assert report["wait_round_fraction"] == int(bool(call_count))
        assert report["wait_cached_tokens"] == 0


def test_durable_inventory_preserves_unknown_legacy_conflict_and_partial():
    durable = {"type": "llm_round", "task_id": "parent", "round_id": 1}
    missing = _usage(count=0)
    missing.pop("tool_call_count")
    _unknown([durable, missing, _DONE])
    _unknown([{**durable, "tool_call_count": 0}, _usage(count=1), _DONE])
    _unknown([{**durable, "tool_call_count": 2}, missing, _tool(), _DONE])
