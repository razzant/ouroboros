"""Independent exact arithmetic and missing-evidence cases for the public reader."""
from scripts.measure_wait_rounds import measure


def test_joined_rounds_duplicate_usage_and_child_denominator():
    rows = [
        {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80, "tool_call_count": 1},
        {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80, "tool_call_count": 1, "ts": "later duplicate"},
        {"tool": "wait_tasks", "tool_call_id": "w", "task_id": "parent", "round_id": 1},
        {"type": "llm_usage", "category": "task", "task_id": "parent", "round_id": 2, "prompt_tokens": 300, "cached_tokens": 0, "tool_call_count": 1},
        {"tool": "read_file", "tool_call_id": "r", "task_id": "parent", "round_id": 2},
        {"type": "task_done", "task_id": "child", "parent_task_id": "parent"},
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


_DONE = {"type": "task_done", "task_id": "child", "parent_task_id": "parent"}


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
