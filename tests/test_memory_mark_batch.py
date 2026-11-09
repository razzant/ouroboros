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


@pytest.mark.parametrize("bad", [[], "one", ["same", "same"]])
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


def test_the_mark_tool_text_says_who_sees_a_mark_on_every_route():
    """memory_mark's own text names the readers the view really gives a mark (memory_view.ROLE_DEFAULTS) and that the
    owner's chat shows nothing of it; the custom codec of direct OpenAI routes keeps tool text and drops parameter
    prose, so the fact lives in the description."""
    from ouroboros.memory_view import ROLE_DEFAULTS
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom

    schema = {entry.name: entry.schema for entry in chronicle_tools()}["memory_mark"]
    reach = ("a room mark is in view of every integrating focus and of that room's helpers and session starts, "
             "a global one of every focus, helper and session start; a mark adds nothing to my human's chat")
    assert reach in schema["description"]
    wire = project_function_tools_to_openai_custom([{"type": "function", "function": schema}]).wire_tools()
    assert reach in wire[0]["custom"]["description"]
    integrating = {role for role, spec in ROLE_DEFAULTS.items() if spec.marks == "all"}
    helpers = {role for role, spec in ROLE_DEFAULTS.items() if spec.marks == "room_and_global"}
    assert integrating == {"integrator", "consciousness", "presence"} and helpers == {"child", "nanny"}
    anchor = schema["parameters"]["properties"]["anchor"]["description"]
    assert anchor.startswith("Optional action whose first result in a task also carries this mark's words:")
    assert "meets me at" not in anchor
