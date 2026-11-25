"""The receiving side of a linked Ouroboros: the owner's own list read names the work its letters started here, from
that work's acceptance record (``place_projection.peer_works``): who sent it, its title, its stored status and the room
it lives in. Before any ordinary message the place's arrival index is empty, yet the owner's view says what arrived and
where it is kept. Nothing is written for it (no index row, no reply, no feed), no date is claimed (the record keeps no
acceptance time), and no other reader of the place views (a place's own tools) receives the field."""
from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.testclient import TestClient

from tests._peer_letters_shared import introduce


def _accept(root, place_id, link_id, letter_id, *, title, ts, status="running", room=4242, sender="Laptop"):
    """A letter root as ``peer_letters_api._accept_new`` records it (its row, its room, the link's recorded label)."""
    from ouroboros.peer_letters import letter_task_id
    from ouroboros.task_results import write_task_result

    task_id = letter_task_id(place_id, letter_id)
    write_task_result(root, task_id, status, root_task_id=task_id, chat_id=room, title=title, ts=ts,
                      metadata={"peer_letter": {"place_id": place_id, "link_id": link_id, "letter_id": letter_id,
                                                "sender_label": sender}})
    return task_id


def test_one_read_names_each_linked_place_its_work_whose_record_changed_last(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import peer_works

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        assert peer_works(pair.root_b, [pair.place_b]) == {pair.place_b: None}, "an empty link: nothing was accepted"
        _accept(pair.root_b, pair.place_b, pair.link_id, "L1", title="Count the files",
                ts="2026-11-25T10:00:00+00:00", status="completed")
        newest = _accept(pair.root_b, pair.place_b, pair.link_id, "L2", title="Check the installer",
                         ts="2026-11-25T11:00:00+00:00")
        _accept(pair.root_b, "another-place", "another-link", "L3", title="Not this place's",
                ts="2026-11-25T12:00:00+00:00", sender="Someone else")
        works = peer_works(pair.root_b, [pair.place_b])
        assert works == {pair.place_b: {"task_id": newest, "title": "Check the installer", "status": "running",
                                         "from": "Laptop", "room_chat_id": 4242, "phase": "working"}}
        assert "accepted_at" not in works[pair.place_b], "the row's ts moves with the work: no acceptance date"
        both = peer_works(pair.root_b, [pair.place_b, "another-place", "unknown-place"])
        assert both["another-place"]["title"] == "Not this place's" and both["unknown-place"] is None
        assert peer_works(pair.root_b, []) == {}


def test_only_the_owners_list_read_carries_it_never_a_full_or_brief_view_nor_the_place_tool(tmp_path, monkeypatch):
    from ouroboros.gateway.places_api import places_routes
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import place_view, places_view
    from ouroboros.places import place_records
    from ouroboros.tools.presence import _configure_presence
    from ouroboros.tools.registry import ToolContext

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        _accept(pair.root_b, pair.place_b, pair.link_id, "L1", title="Check the installer", ts="2026-11-25T11:00:00+00:00")
        place = place_records(pair.root_b).places[pair.place_b]
        assert "peer_work" not in place_view(pair.root_b, place, declarations=[]), "a full view: no field by default"
        assert "peer_work" not in place_view(pair.root_b, place, brief=True)
        linked = lambda view: next(p for p in view["places"] if p["place_id"] == pair.place_b)  # noqa: E731
        assert "peer_work" not in linked(places_view(pair.root_b)), "any other list reader: no field"
        owner = linked(places_view(pair.root_b, owner_reading=True))
        assert owner["peer_work"]["title"] == "Check the installer" and owner["rooms"] == [], (
            "the owner's read names the work; its arrival index stays as it is")
        ctx = ToolContext(repo_dir=pair.root_b, drive_root=pair.root_b)
        listed = json.loads(_configure_presence(ctx, "list"))
        assert "peer_work" not in linked(listed), "the place tool's list (a focus may read it) carries none"
        inspected = json.loads(_configure_presence(ctx, "inspect", pair.place_b))["place"]
        assert "peer_work" not in inspected
        app = Starlette(routes=places_routes())
        app.state.drive_root = pair.root_b
        with TestClient(app) as client:
            answer = client.get("/api/places")
            assert answer.status_code == 200
            assert linked(answer.json())["peer_work"]["from"] == "Laptop", "the owner's GET /api/places opts in"


def test_an_unreadable_task_store_is_not_a_claim_that_nothing_arrived(tmp_path, monkeypatch):
    from ouroboros import store
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import peer_works

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        _accept(pair.root_b, pair.place_b, pair.link_id, "L1", title="Count the files", ts="2026-11-25T10:00:00+00:00")
        monkeypatch.setattr(store, "migrating", lambda root: True)
        assert peer_works(pair.root_b, [pair.place_b]) == {pair.place_b: {"unknown": True}}


def test_where_this_side_wrote_first_is_named_by_its_place_until_something_arrives(tmp_path, monkeypatch):
    """The sending side's first ordinary message, before anything arrived: the owner's list read names the place's own
    room (the letter door's derivation) and its receipt's time; nothing written there is ``null``, a listed room makes
    the field ``null`` again (its arrivals open it), and an unreadable history is unknown, never a guessed absence."""
    from ouroboros import dialogue_store, peer_replies
    from ouroboros.gateway import peer_letters_api
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import peer_rooms, places_view
    from ouroboros.places import place_records

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_a):
        place = place_records(pair.root_a).places[pair.place_a]
        linked = lambda view: next(p for p in view["places"] if p["place_id"] == pair.place_a)  # noqa: E731
        assert peer_rooms(pair.root_a, [place]) == {pair.place_a: None}, "nothing written yet"
        assert linked(places_view(pair.root_a, owner_reading=True))["peer_room"] is None
        assert peer_replies._record_receipt(pair.root_a, pair.link_id, place.account_id, "m-weather", "delivered",
                                            text="The forecast says rain after noon.", reply_to="",
                                            origin={"kind": "tool", "task_id": "t-weather"})
        named = peer_rooms(pair.root_a, [place])[pair.place_a]
        assert named["room_chat_id"] == peer_letters_api._room(place) and named["wrote_at"]
        owner = linked(places_view(pair.root_a, owner_reading=True))
        assert owner["rooms"] == [] and owner["peer_room"] == named, "its arrival index stays as it is"
        assert "peer_room" not in linked(places_view(pair.root_a)), "any other list reader: no field"
        assert peer_rooms(pair.root_a, []) == {}
        monkeypatch.setattr(dialogue_store, "transaction", lambda *a, **k: (_ for _ in ()).throw(OSError("busy")))
        assert peer_rooms(pair.root_a, [place]) == {pair.place_a: {"unknown": True}}
