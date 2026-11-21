"""Chat-first owner controls for generic presence bindings and selections."""

from __future__ import annotations

import json

import pytest

from ouroboros import places
from ouroboros.tools.presence import get_tools
from ouroboros.tools.registry import ToolContext
from tests._store_shared import seed_result


def _configure(ctx: ToolContext, action: str, **params):
    entry = next(item for item in get_tools() if item.name == "configure_presence")
    return entry.handler(ctx, action, **params)


def _ctx(tmp_path) -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir()
    return ToolContext(repo_dir=repo, drive_root=tmp_path)


def test_configure_place_connect_choices_and_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [{
        "skill": "bridge", "id": "room", "label": "Chat", "destination": "vendor:chat", "wakes": ["mention"]}])
    ctx = _ctx(tmp_path)
    created = json.loads(_configure(ctx, "connect", connector="bridge", account_id="a"))["place"]
    saved = json.loads(_configure(ctx, "set_state", place_id=created["place_id"], state="silent"))["place"]
    assert saved["state"] == "silent"
    assert json.loads(_configure(ctx, "list"))["places"][0]["place_id"] == created["place_id"]
    assert json.loads(_configure(ctx, "disconnect", place_id=created["place_id"]))["ok"]
    assert places.load_places(tmp_path) == []


def test_inspect_keeps_disconnected_place_words_reachable(tmp_path):
    ctx = _ctx(tmp_path)
    place = places.connect_place(tmp_path, connector="bridge", provider="chat", conversation_id="harbor")
    circle = places.create_circle(tmp_path, name="Harbor circle", text="Earlier circle words.", author="owner")
    place = places.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("add_note", {"text": "Keep this\n  exact note. "}),
        ("set_description", {"description": "Harbor purpose"}),
        ("set_circle", {"circle_id": circle["circle_id"]})])
    places.edit_circle(tmp_path, circle["circle_id"], author="owner", text="Current circle words.")
    live = json.loads(_configure(ctx, "inspect", place_id=place.place_id))
    assert live["place"]["notes"][0]["text"] == place.notes[0].text
    assert live["place"]["description"]["text"] == "Harbor purpose"
    assert live["circle"]["text"] == "Current circle words."
    assert [word["text"] for word in live["earlier"]] == ["Earlier circle words."]

    assert json.loads(_configure(ctx, "disconnect", place_id=place.place_id))["ok"]
    response = _configure(ctx, "inspect", place_id=place.place_id)
    assert "place_not_found" not in response
    gone = json.loads(response)
    assert gone["ok"] and gone["place"] is None and gone["circle"] is None
    assert gone["place_id"] == place.place_id and gone["status"] == "This place is not connected."
    assert [(word["action"], word["kind"], word["text"], word["author"]) for word in gone["earlier"]] == [
        ("disconnect", "note", place.notes[0].text, "owner"),
        ("disconnect", "description", "Harbor purpose", "owner"),
        ("edit_circle", "circle_words", "Earlier circle words.", "owner")]
    assert (gone["earlier"][0]["note_id"], gone["earlier"][0]["at"]) == (place.notes[0].note_id, place.notes[0].at)
    assert "place_not_found" in _configure(ctx, "inspect", place_id="never-connected")
    places.connect_place(tmp_path, connector="bridge", provider="chat", conversation_id="harbor")
    again = json.loads(_configure(ctx, "inspect", place_id=place.place_id))
    assert again["place"]["place_id"] == place.place_id and again["place"]["notes"] == []
    assert "status" not in again and again["earlier"] == gone["earlier"]


@pytest.mark.parametrize("has_place", [False, True])
def test_inspect_circle_words_survive_edit_and_deletion_with_or_without_places(tmp_path, has_place):
    ctx = _ctx(tmp_path)
    circle = places.create_circle(tmp_path, name="Orchard", text="First\n  exact words.", author="owner")
    if has_place:
        place = places.connect_place(tmp_path, connector="bridge", provider="chat", conversation_id="orchard")
        places.edit_place(tmp_path, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    # The id in another circle's text must not count as this circle's history.
    other = places.create_circle(tmp_path, name="Meadow", text=circle["circle_id"], author="ouroboros")
    places.edit_circle(tmp_path, other["circle_id"], author="ouroboros", text="Other current words.")
    places.edit_circle(tmp_path, circle["circle_id"], author="owner", text="Second words.")
    response = _configure(ctx, "inspect", circle_id=circle["circle_id"])
    assert "place_not_found" not in response
    live = json.loads(response)
    assert live["ok"] and live["circle"]["text"] == "Second words."
    assert [(word["action"], word["text"], word["at"]) for word in live["earlier"]] == [
        ("edit_circle", circle["text"], circle["text_at"])]
    removed = places.delete_circle(tmp_path, circle["circle_id"], author="owner")["circle"]
    gone = json.loads(_configure(ctx, "inspect", circle_id=circle["circle_id"]))
    assert gone["ok"] and gone["circle"] is None and gone["circle_id"] == circle["circle_id"]
    assert gone["status"] == "This circle no longer exists."
    assert [(word["action"], word["text"], word["author"], word["at"]) for word in gone["earlier"]] == [
        ("delete_circle", removed["text"], "owner", removed["text_at"]),
        ("edit_circle", circle["text"], "owner", circle["text_at"])]
    assert all(word["circle_id"] == circle["circle_id"] for word in gone["earlier"])
    if has_place:
        # A place id still selects that place even if a circle parameter was also supplied.
        member = json.loads(_configure(ctx, "inspect", place_id=place.place_id, circle_id=other["circle_id"]))
        assert member["place"]["place_id"] == place.place_id and member["earlier"] == gone["earlier"]
    else:
        assert places.load_places(tmp_path) == []
    unrelated = json.loads(_configure(ctx, "inspect", circle_id=other["circle_id"]))
    assert [word["text"] for word in unrelated["earlier"]] == [circle["circle_id"]]
    assert "circle_not_found" in _configure(ctx, "inspect", circle_id="unknown-circle")


def test_inspect_keeps_current_views_when_history_cannot_be_read(tmp_path):
    from ouroboros.place_words_history import journal_path

    ctx = _ctx(tmp_path)
    place = places.connect_place(tmp_path, connector="bridge", provider="chat", conversation_id="garden")
    place = places.edit_place(tmp_path, place.place_id, "add_note", author="owner", text="Still standing.")
    circle = places.create_circle(tmp_path, name="Garden", text="Still shared.", author="owner")
    places.edit_place(tmp_path, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    before = json.loads(_configure(ctx, "inspect", place_id=place.place_id))
    assert before["earlier"] == []
    path = journal_path(tmp_path)
    path.mkdir()
    with pytest.raises(OSError) as cause:
        path.read_bytes()
    response = _configure(ctx, "inspect", place_id=place.place_id)
    assert not response.startswith("ERROR:"), response
    shown = json.loads(response)
    assert shown["ok"] and shown["place"] == before["place"] and shown["circle"] == before["circle"]
    assert shown["earlier"] == {"error": "place_history_unreadable",
        "message": f"Earlier words could not be read: place_history_unreadable: {path}: {cause.value}"}
    shared = json.loads(_configure(ctx, "inspect", circle_id=circle["circle_id"]))
    assert shared["ok"] and shared["circle"]["text"] == circle["text"] and shared["earlier"] == shown["earlier"]
    path.rmdir()
    places.edit_place(tmp_path, place.place_id, "remove_note", author="owner", note_id=place.notes[0].note_id)
    readable = json.loads(_configure(ctx, "inspect", place_id=place.place_id))
    assert readable["place"]["notes"] == []
    assert [word["text"] for word in readable["earlier"]] == ["Still standing."]


def test_retired_presence_actions_are_not_registered():
    assert not {"presence_finish", "initiate_presence"} & {tool.name for tool in get_tools()}


def test_presence_cancel_work_accepts_own_binding_work_from_any_of_its_conversations(monkeypatch, tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {
        "presence": {
            "binding_id": "1" * 32,
            "event": {"conversation_key": "telegram:bot-1:room-1"},
        }
    }

    def row(task_id, *, binding="1" * 32, key="telegram:bot-1:room-1", **fields):
        seed_result(tmp_path, task_id, {
            "_schema_version": 1, "task_id": task_id, "status": "running",
            "metadata": {"presence": {"binding_id": binding, "event": {"conversation_key": key}}} if binding else {},
            **fields,
        })

    row("promoted-work-1", delegation_role="root", root_task_id="promoted-work-1")
    row("other-thread-work", key="telegram:bot-1:room-1:thread-9", delegation_role="root")
    row("foreign-binding-work", binding="2" * 32, delegation_role="root")
    row("presence-inline-turn")  # an inline turn is not deferred work
    row("owner-root", binding="", delegation_role="root")
    monkeypatch.setattr(
        "ouroboros.tools.join_ledger._cancel_task",
        lambda _ctx, task_id, reason="": f"cancel:{task_id}:{reason}",
    )
    entry = next(item for item in get_tools() if item.name == "presence_cancel_work")

    assert entry.handler(ctx, "promoted-work-1", "no longer needed") == (
        "cancel:promoted-work-1:no longer needed"
    )
    # Owner Q1: the same nonempty binding, a different thread or room of it.
    assert entry.handler(ctx, "other-thread-work") == "cancel:other-thread-work:"
    ctx.task_metadata["presence"]["event"]["conversation_key"] = "telegram:bot-1:other"
    assert entry.handler(ctx, "promoted-work-1") == "cancel:promoted-work-1:"
    for foreign in ("foreign-binding-work", "presence-inline-turn", "owner-root", "never-existed"):
        assert entry.handler(ctx, foreign).startswith("ERROR: PRESENCE_WORK_NOT_CORRELATED")
    ctx.task_metadata["presence"]["binding_id"] = ""  # an empty binding never compares equal
    assert entry.handler(ctx, "promoted-work-1").startswith("ERROR: PRESENCE_WORK_NOT_CORRELATED")
