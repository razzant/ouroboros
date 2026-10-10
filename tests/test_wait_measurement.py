"""Independent exact arithmetic and missing-evidence cases for the public reader."""
from scripts.measure_wait_rounds import measure


def test_joined_rounds_duplicate_usage_and_child_denominator():
    rows = [
        {"type": "llm_usage", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80},
        {"type": "llm_usage", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80, "ts": "later duplicate"},
        {"tool": "wait_tasks", "task_id": "parent", "round_id": 1},
        {"type": "llm_usage", "task_id": "parent", "round_id": 2, "prompt_tokens": 300, "cached_tokens": 0},
        {"tool": "read_file", "task_id": "parent", "round_id": 2},
        {"type": "task_done", "task_id": "child", "parent_task_id": "parent"},
    ]
    r = measure(rows)
    assert r["coverage_complete"]
    assert (r["usage_rounds"], r["wait_only_rounds"], r["prompt_tokens"], r["wait_prompt_tokens"]) == (2, 1, 400, 100)
    assert r["wait_round_fraction"] == 1 / 2
    assert r["wait_prompt_fraction"] == 1 / 4
    assert r["wait_rounds_per_terminal"] == 1
    assert r["targets"] == {"round_fraction_le_3pct": False, "prompt_fraction_le_5pct": False, "rounds_per_terminal_le_1_5": True}


def test_missing_or_conflicting_usage_never_certifies_a_target():
    row = {"type": "llm_usage", "task_id": "parent", "round_id": 1, "prompt_tokens": 100, "cached_tokens": 80}
    conflict = measure([row, {**row, "prompt_tokens": 101}, {"tool": "wait_task", "task_id": "parent", "round_id": 2}])
    assert not conflict["coverage_complete"] and conflict["prompt_tokens"] is None
    assert all(value is None for value in conflict["targets"].values())
    missing = measure([{**row, "prompt_tokens": None}, {"tool": "wait_task", "task_id": "parent", "round_id": 1}])
    assert missing["prompt_tokens"] is None and missing["wait_prompt_tokens"] is None
    assert missing["targets"]["prompt_fraction_le_5pct"] is None
