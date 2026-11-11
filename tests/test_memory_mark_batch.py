"""memory_mark: batch release has one reason and accounts for every id, including partial failures; the tool's
own text names who sees a mark."""
import inspect
import json

import pytest

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark, chronicle_tools
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
    owner's chat shows nothing of it; the fact lives in the description, which direct OpenAI routes carry whole
    (test_openai_chat_custom_contract.test_actual_profile_catalogs_carry_every_tool_lead_whole)."""
    from ouroboros.memory_view import ROLE_DEFAULTS

    schema = {entry.name: entry.schema for entry in chronicle_tools()}["memory_mark"]
    reach = ("A room mark (the default scope) is in view of every integrating focus and of that room's helpers and "
             "session starts, a global one of every focus, helper and session start: ")
    assert reach in schema["description"] and "; a mark adds nothing to my human's chat." in schema["description"]
    integrating = {role for role, spec in ROLE_DEFAULTS.items() if spec.marks == "all"}
    helpers = {role for role, spec in ROLE_DEFAULTS.items() if spec.marks == "room_and_global"}
    assert integrating == {"integrator", "consciousness", "presence"} and helpers == {"child", "nanny"}
    anchor = schema["parameters"]["properties"]["anchor"]["description"]
    assert anchor.startswith("Optional action whose first result in a task also carries this mark's words:")
    assert "meets me at" not in anchor
    # The reach differs by scope, so the description names the scope a mark gets when none is given (`reach`
    # above), and that scope is the handler's own default.
    assert inspect.signature(_memory_mark).parameters["scope"].default == "room"


def test_a_mark_lands_where_its_schema_says_when_no_room_is_named(tmp_path):
    """The mark schema's room default is the handler's on both paths: a record's mark lands in that record's room,
    any other target's in this task's room; either is a room mark when no scope is given."""
    here = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1)
    other = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0002", current_chat_id=7)
    note = json.loads(_chronicle_write(other, kind="note", text="a note of room seven"))
    by_record = json.loads(_memory_mark(here, node_id=note["node_id"], text="keep"))
    by_task = json.loads(_memory_mark(here, task_id="task0001", text="keep"))
    assert (by_record["room_id"], by_task["room_id"]) == ("7", "1")
    assert by_record["scope"] == by_task["scope"] == "room"
    schema = {entry.name: entry.schema for entry in chronicle_tools()}["memory_mark"]
    room = schema["parameters"]["properties"]["room_id"]["description"]
    assert room.endswith("Default: the record's room for a node_id target, else this task's own room.")


def test_a_marked_row_of_my_human_travels_to_the_work_readers_and_my_own_view_shows_the_mark(tmp_path):
    """What the description states, read where the code makes it. A mark on a chat row of my human's own words puts
    that row, verbatim and dated, into the set every helper, session, plan council and reviewer of a work in the
    mark's room reads (``owner_words.work_words``): a Main mark reaches each work outside the Projects and no
    Project work, a global mark every work; a mark on a row of mine carries nothing. My own view shows the mark's
    text, its quote while visibility is full, and its target's address (``memory_view._marks_text``)."""
    from ouroboros import chat_chain, dialogue_store as ds, memory_view as mv, owner_words as ow
    from ouroboros.projects_registry import bind_task_to_project, create_project

    ts = "2026-11-10T09:30:00+00:00"
    ds.append(tmp_path, "chat", {"ts": ts, "direction": "in", "chat_id": 1, "text": "Keep the old cookie name",
                                 "client_message_id": "cm-1", "task_id": ""})
    ds.append(tmp_path, "chat", {"ts": ts, "direction": "out", "chat_id": 1, "text": "I will rename the cookie",
                                 "client_message_id": "out-1", "task_id": "root0001"})
    rows = {row.get("client_message_id"): chat_chain.format_address(address)
            for address, row, _pos in chat_chain.iter_rows(tmp_path)}
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1)
    theirs = json.loads(_memory_mark(ctx, address=rows["cm-1"], text="Their standing ask", quote="old cookie"))
    assert json.loads(_memory_mark(ctx, address=rows["out-1"], text="My own promise"))["ok"]
    room = create_project(tmp_path, "quotas", name="Quotas")["chat_id"]
    bind_task_to_project(tmp_path, "work0003", "quotas", room, origin={"absent": "system"})

    def carried(task_id):
        work = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id, current_chat_id=1)
        return [(row["carrier"], row["ts"], row["text"]) for row in ow.work_words(work)[0]]

    def my_marks():  # the marking root's own rendered view, its marks section
        task = {"id": "root0001", "chat_id": 1}
        view = mv.render_room(mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path)))
        return view.split("## Marks I keep in view", 1)[1].split("\n## ", 1)[0]

    assert carried("work0002") == [("marked", ts, "Keep the old cookie name")]  # mine stays out
    assert carried("work0003") == []  # a Main mark is no Project work's
    shown = my_marks()
    line = next(line for line in shown.split("\n") if "Their standing ask" in line)
    assert f"; target {rows['cm-1']};" in line and "\n  quote: old cookie" in shown
    assert "Keep the old cookie name" not in shown  # the row's words travel to the others, not into my mark
    assert json.loads(_memory_mark(ctx, mark_id=theirs["mark_id"], visibility="meaning", reason="meaning only"))["ok"]
    assert "Their standing ask" in my_marks() and "quote: old cookie" not in my_marks()
    assert json.loads(_memory_mark(ctx, release_ids=[theirs["mark_id"]], reason="scope it wider"))["ok"]
    assert json.loads(_memory_mark(ctx, address=rows["cm-1"], text="For every work", scope="global"))["ok"]
    assert carried("work0003") == [("marked", ts, "Keep the old cookie name")]  # a global mark: every work
