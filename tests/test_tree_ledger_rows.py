"""Blackboard rows name the task they came from, so a row is addressable
(forward_to_worker / peek_task take the id as written)."""
from __future__ import annotations

import json

from ouroboros.task_tree_ledger import _format_tree_ledger_row


def test_a_row_carries_the_role_and_the_full_task_id():
    row = {"ts": "2026-10-03T10:00:00Z", "kind": "note", "role": "critic",
           "task_id": "0123456789abcdef", "text": "objection"}
    line = _format_tree_ledger_row(row)
    assert "(critic 0123456789abcdef)" in line
    assert "(critic)" not in line


def test_a_row_without_a_role_shows_the_whole_id_not_an_eight_char_prefix():
    row = {"ts": "2026-10-03T10:00:00Z", "kind": "note", "task_id": "0123456789abcdef", "text": "objection"}
    line = _format_tree_ledger_row(row)
    assert "(0123456789abcdef)" in line
    assert "(01234567)" not in line


def test_unchanged_rows_cost_one_stat_and_an_append_is_seen(tmp_path, monkeypatch):
    from ouroboros import task_tree_ledger as ledger

    path = ledger.tree_ledger_path("root", data_root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"text":"first"}\n', encoding="utf-8")
    parsed, stats = [], []
    real_read, real_stat = ledger.iter_jsonl_objects, type(path).stat
    monkeypatch.setattr(ledger, "iter_jsonl_objects", lambda p: parsed.append(p) or real_read(p))
    monkeypatch.setattr(type(path), "stat", lambda p, **kw: stats.append(p) or real_stat(p, **kw))
    assert ledger.tree_ledger_rows("root", data_root=tmp_path) == [{"text": "first"}]
    stats.clear()
    for _ in range(3):
        assert ledger.tree_ledger_rows("root", data_root=tmp_path) == [{"text": "first"}]
    assert parsed == [path] and stats == [path] * 3
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"text":"second"}\n')
    assert ledger.tree_ledger_rows("root", data_root=tmp_path) == [{"text": "first"}, {"text": "second"}]
    assert parsed == [path, path]
    path.unlink()
    assert ledger.tree_ledger_rows("root", data_root=tmp_path) == []


def test_the_row_cache_keeps_32_entries_and_several_large_hot_trees(tmp_path, monkeypatch):
    from collections import OrderedDict
    from ouroboros import task_tree_ledger as ledger

    monkeypatch.setattr(ledger, "_ROWS_CACHE", OrderedDict())
    for index in range(35):
        path = ledger.tree_ledger_path(f"root-{index}", data_root=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"text": str(index)}) + "\n", encoding="utf-8")
        assert ledger.tree_ledger_rows(f"root-{index}", data_root=tmp_path) == [{"text": str(index)}]
    assert len(ledger._ROWS_CACHE) == 32
    assert ledger.tree_ledger_path("root-0", data_root=tmp_path) not in ledger._ROWS_CACHE
    parsed, real_read = [], ledger.iter_jsonl_objects
    monkeypatch.setattr(ledger, "iter_jsonl_objects", lambda p: parsed.append(p) or real_read(p))
    hot = {}
    for index, count in enumerate((5000, 5000, 9000)):
        path = ledger.tree_ledger_path(f"large-{index}", data_root=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text((json.dumps({"text": "x" * 128}) + "\n") * count, encoding="utf-8")
        hot[f"large-{index}"] = (path, count)
    for _ in range(3):
        for root_id, (path, count) in hot.items():
            assert len(ledger.tree_ledger_rows(root_id, data_root=tmp_path)) == count
            assert path in ledger._ROWS_CACHE and len(ledger._ROWS_CACHE) <= 32
    assert parsed == [path for path, _count in hot.values()], "interleaved hot trees parse only once each"


def test_the_row_cache_bounds_source_bytes_without_cutting_the_read(tmp_path, monkeypatch):
    from collections import OrderedDict
    from ouroboros import task_tree_ledger as ledger

    monkeypatch.setattr(ledger, "_ROWS_CACHE", OrderedDict())
    parsed, real_read = [], ledger.iter_jsonl_objects
    monkeypatch.setattr(ledger, "iter_jsonl_objects", lambda p: parsed.append(p) or real_read(p))
    line = json.dumps({"text": "x" * 4000}) + "\n"
    for index, count in enumerate((1000, 1000, 1000, 2500)):
        path = ledger.tree_ledger_path(f"root-{index}", data_root=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(line * count, encoding="utf-8")
        for _ in range(2):
            assert len(ledger.tree_ledger_rows(f"root-{index}", data_root=tmp_path)) == count
        assert sum(value[0][0] for value in ledger._ROWS_CACHE.values()) <= 8 * 1024 * 1024
        assert (path in ledger._ROWS_CACHE) is (path.stat().st_size <= 8 * 1024 * 1024)
        assert parsed.count(path) == (1 if path.stat().st_size <= 8 * 1024 * 1024 else 2)
