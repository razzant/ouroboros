"""The occasion is separate from destinations and their standing delegation."""
import json

import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


def bundle(h, name, arguments):
    assert h.module.check_safety(name, arguments, ctx=h.ctx)[0]
    return json.loads(h.calls[-1]["user_prompt"].split(
        "Host evidence (provenance is not consent):\n", 1)[1].split("\nProposed tool call:", 1)[0])


@pytest.mark.parametrize("name", ["start_task", "remote_write"])
def test_focus_and_arbitrary_place_arguments_are_not_destinations(supervisor, monkeypatch, name):
    from ouroboros import places, place_guard
    h = supervisor
    h.ctx.task_metadata["focus"] = {"place_id": "focus-room"}
    record = places.Place("target", "connector", "chat", label="Target",
                          notes=(places.NoteEntry("For this target only", "owner"),))
    reads = []
    monkeypatch.setattr(places, "load_places", lambda root: reads.append(root) or [record])
    data = bundle(h, name, {"place_id": "target", "text": "unrelated work"})
    assert data["subject"]["place"] == {} and not data["standing_delegation"]["place_notes"]
    assert data["occasion"]["focus"]["place_id"] == "focus-room" and not reads
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a: {"places": [{"place_id": "target"}]})
    data = bundle(h, name, {"text": "resolved work"})
    assert data["subject"]["place"]["places"][0]["place_id"] == "target"
    assert data["standing_delegation"]["place_notes"][0]["standing_delegation"] is True


@pytest.mark.parametrize("canonical_source", ["metadata", "context"])
@pytest.mark.parametrize("name", ["configure_presence", "place_reply", "remote_write"])
def test_destination_notes_use_canonical_root_and_keep_each_place(supervisor, monkeypatch, tmp_path,
                                                                canonical_source, name):
    from ouroboros import places, place_guard
    h = supervisor
    root = tmp_path / "canonical"
    h.ctx.drive_root = tmp_path / "child"
    h.ctx.task_metadata["focus"] = {"place_id": "elsewhere"}
    if canonical_source == "metadata":
        h.ctx.task_metadata["budget_drive_root"] = str(root)
        h.ctx.budget_drive_root = str(tmp_path / "wrong-context")
    else:
        h.ctx.budget_drive_root = str(root)
    records = [places.Place(id, "connector", "chat", label=id, state=state, notes=(
        places.NoteEntry(f"{id} owner's note", "owner"), places.NoteEntry(f"{id} migration", "migration"),
        places.NoteEntry(f"{id} mind's note", "ouroboros"))) for id, state in (("a", "active"), ("b", "silent"))]
    reads = []
    monkeypatch.setattr(places, "load_places", lambda path: reads.append(path) or records)
    args = {"place_id": "a", "action": "inspect", "destination": {"conversation_id": "chat"}}
    if name == "remote_write":
        monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a: {
            "places": [{"place_id": p.place_id, "place_state": p.state} for p in records],
            "targets": {"to": ["a", "b"]}})
    data = bundle(h, name, args)
    notes = data["standing_delegation"]["place_notes"]
    assert reads == [root] and not h.ctx.drive_root.exists()
    assert {n["place_id"] for n in notes} == ({"a", "b"} if name == "remote_write" else {"a"})
    assert all(n["standing_delegation"] is (n["author"] in {"owner", "migration"}) for n in notes)
    assert "elsewhere" not in json.dumps(data["subject"])
    if name == "remote_write":
        assert data["subject"]["place"]["places"][1]["place_state"] == "silent"
    else:
        assert data["subject"]["place"]["place_state"] == "active"
