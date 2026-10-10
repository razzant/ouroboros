from __future__ import annotations

import json


def test_iter_jsonl_objects_skips_bad_lines_and_non_dict_by_default(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"ok": 1}),
            "not-json",
            json.dumps(["list"]),
            json.dumps({"ok": 2}),
            "",
        ]),
        encoding="utf-8",
    )

    assert list(iter_jsonl_objects(path)) == [{"ok": 1}, {"ok": 2}]
    assert list(iter_jsonl_objects(path, dict_only=False)) == [{"ok": 1}, ["list"], {"ok": 2}]


def test_iter_jsonl_objects_supports_tail_and_max_entries(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"i": i}) + "\n" for i in range(5)]
    path.write_text("".join(lines), encoding="utf-8")

    assert [entry["i"] for entry in iter_jsonl_objects(path, max_entries=2)] == [3, 4]
    tail = len(lines[-1].encode("utf-8")) + 4
    assert [entry["i"] for entry in iter_jsonl_objects(path, tail_bytes=tail)] == [4]
    assert list(iter_jsonl_objects(path, tail_bytes=0)) == []


def test_iter_jsonl_objects_reports_raw_tail_truncation(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"i": 0}),
            "",
            json.dumps({"i": 1}),
            json.dumps({"i": 2}),
        ]) + "\n",
        encoding="utf-8",
    )
    gaps = set()

    rows = list(iter_jsonl_objects(path, max_entries=3, gap_reasons=gaps))

    assert [row["i"] for row in rows] == [1, 2]
    assert gaps == {"max_entries_truncated"}


def test_iter_jsonl_objects_tail_keeps_line_boundary_start(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"i": i}) + "\n" for i in range(3)]
    path.write_bytes("".join(lines).encode("utf-8"))

    tail = len("".join(lines[1:]).encode("utf-8"))
    assert [entry["i"] for entry in iter_jsonl_objects(path, tail_bytes=tail)] == [1, 2]


def test_llm_usage_helpers_normalize_flat_and_nested_costs(tmp_path):
    from ouroboros.utils import iter_llm_usage_events, llm_usage_cost

    path = tmp_path / "events.jsonl"
    flat = {"type": "llm_usage", "cost": "1.25", "prompt_tokens": "10", "completion_tokens": 2}
    nested = {"type": "llm_usage", "usage": {"cost": 0.5, "prompt_tokens": 3, "cached_tokens": "2"}}
    path.write_text(
        "\n".join([json.dumps(flat), json.dumps({"type": "other"}), json.dumps(nested)]),
        encoding="utf-8",
    )

    events = list(iter_llm_usage_events(path))
    assert [llm_usage_cost(event) for event in events] == [1.25, 0.5]


def test_supervisor_usage_summaries_use_shared_cost_normalizer(tmp_path, monkeypatch):
    from supervisor import state

    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "events.jsonl").write_text(
        "\n".join([
            json.dumps({
                "type": "llm_usage", "category": "task", "task_id": "a",
                "model": "m1", "cost": 1.25, "prompt_tokens": 10,
            }),
            json.dumps({
                "type": "llm_usage", "category": "review", "task_id": "b",
                "model": "m2", "usage": {"cost": 0.5}, "completion_tokens": 4,
            }),
        ]),
        encoding="utf-8",
    )

    from ouroboros import usage_store

    usage_store.migrate_from_journal(tmp_path)  # the lifecycle job the server runs before any display
    assert state.budget_breakdown({}) == {"task": 1.25, "review": 0.5}
    assert state.model_breakdown({})["m2"]["cost"] == 0.5
    assert [item["task_id"] for item in state.per_task_cost_summary(max_tasks=2)] == ["a", "b"]
