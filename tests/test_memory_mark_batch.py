"""Batch release has one reason and accounts for every id, including partial failures."""
import json

import pytest

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import _memory_mark, chronicle_tools
from ouroboros.tools.registry import ToolContext


def marks(root):
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="root0001", current_chat_id=1)
    ids = [json.loads(_memory_mark(ctx, task_id="task0001", text=f"mark {i}"))["mark_id"] for i in range(3)]
    return ctx, ids


def test_release_accounts_for_successes_and_refusals_and_uses_one_reason(tmp_path):
    ctx, ids = marks(tmp_path)
    result = json.loads(_memory_mark(ctx, release_ids=[ids[0], "missing", ids[1]], reason="considered together"))
    assert not result["ok"] and result["reason"] == "partial"
    assert result["released_ids"] == ids[:2]
    assert [item["id"] for item in result["refused"]] == ["missing"]
    assert all(item["reason"] and item["detail"] for item in result["refused"])
    store = ChronicleStore(tmp_path)
    assert [mark["id"] for mark in store.active_marks("1")] == ids[2:]
    releases = [r for r in store.records() if r["kind"] == "mark_release"]
    assert [r["reason"] for r in releases] == ["considered together"] * 2
    success = json.loads(_memory_mark(ctx, release_ids=ids[2:], reason="also done"))
    assert success["ok"] and success["released_ids"] == ids[2:] and success["refused"] == []


@pytest.mark.parametrize("bad", [[], "one", [""], [None], ["same", "same"]])
def test_invalid_arrays_do_not_release_anything(tmp_path, bad):
    ctx, ids = marks(tmp_path)
    assert "TOOL_ARG_ERROR" in _memory_mark(ctx, release_ids=bad, reason="done")
    assert [m["id"] for m in ChronicleStore(tmp_path).active_marks("1")] == ids


def test_release_cannot_be_mixed_with_another_operation_or_aliased(tmp_path):
    ctx, ids = marks(tmp_path)
    for other in ({"task_id": "task0001"}, {"mark_id": ids[0], "visibility": "meaning"}):
        assert "TOOL_ARG_ERROR" in _memory_mark(ctx, release_ids=ids, reason="done", **other)
    schemas = {entry.name: entry.schema["parameters"]["properties"] for entry in chronicle_tools()}
    assert schemas["memory_mark"]["release_ids"]["type"] == "array"
    assert "release_id" not in schemas["memory_mark"]
    assert "release_mark_ids" not in schemas["chronicle_write"]
    with pytest.raises(TypeError):
        _memory_mark(ctx, release_id=ids[0], reason="old argument")
    assert [m["id"] for m in ChronicleStore(tmp_path).active_marks("1")] == ids


def test_io_failure_does_not_hide_the_other_results(tmp_path, monkeypatch):
    ctx, ids = marks(tmp_path)
    original = ChronicleStore.release_mark

    def release(store, mark_id, author, reason):
        if mark_id == ids[1]:
            raise OSError("temporarily unavailable")
        return original(store, mark_id, author, reason)

    monkeypatch.setattr(ChronicleStore, "release_mark", release)
    result = json.loads(_memory_mark(ctx, release_ids=ids, reason="done"))
    assert result["released_ids"] == [ids[0], ids[2]]
    assert result["refused"] == [{"id": ids[1], "reason": "unavailable", "detail": "temporarily unavailable"}]
