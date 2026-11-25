"""Safety reads a letter's root and its helpers as work a linked Ouroboros occasioned, never my human's door,
with the link place's notes, circle words and my human's two facts of the link in every call's bundle."""
from __future__ import annotations

import json
from types import SimpleNamespace

from tests._peer_letters_shared import introduce, letter, receiver_client


def _bundle(prompt: str) -> dict:
    start = prompt.index("{")
    return json.loads(prompt[start:prompt.index("\nProposed tool call:")])


def test_a_letter_root_and_its_helper_reach_safety_as_a_linked_ouroboros_with_the_place_words(tmp_path, monkeypatch):
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from ouroboros.install_identity import acting_as
    from ouroboros.safety import _build_check_prompt

    pair = introduce(tmp_path, monkeypatch, note="Velm may only read files, never send mail.")
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        assert client.post("/api/letters", json=body, headers=pair.headers()).status_code == 200
    root_meta = pair.row(body["letter_id"])["metadata"]
    child_meta = {**consciousness_origin_metadata({**root_meta, "task_id": pair.task_id(body["letter_id"])}),
                  "parent_task_id": pair.task_id(body["letter_id"]), "delegation_role": "subagent"}
    assert child_meta["peer_letter"] == root_meta["peer_letter"], "a helper keeps its letter's link"
    for task_id, metadata in ((pair.task_id(body["letter_id"]), root_meta), ("helper-1", child_meta)):
        ctx = SimpleNamespace(task_id=task_id, task_metadata=metadata, drive_root=pair.root_b,
                              budget_drive_root=pair.root_b)
        with acting_as(pair.root_b):
            bundle = _bundle(_build_check_prompt("run_command", {"cmd": ["ls"]}, ctx=ctx))
        assert bundle["work"]["kind"] == "owner", task_id
        assert bundle["work"]["label"] == "linked Ouroboros", task_id
        assert bundle["occasion"]["cause"] == "letter from my Ouroboros Laptop", task_id
        assert bundle["occasion"]["cause"] != "owner door"
        notes = json.dumps(bundle["standing_delegation"]["place_notes"], ensure_ascii=False)
        assert "Velm may only read files, never send mail." in notes, task_id
        facts = bundle["occasion"]["link_facts"]  # the note no longer repeats them: the host states both facts
        assert "mine: yes" in facts and "may_start: yes" in facts, (task_id, facts)
    from ouroboros.places import AUTHOR_OWNER, edit_place

    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=False)
        bundle = _bundle(_build_check_prompt("run_command", {"cmd": ["ls"]}, ctx=ctx))
    assert "may_start: no" in bundle["occasion"]["link_facts"], "read at each check, never the admission's copy"


def test_an_ordinary_owner_task_keeps_the_owner_door(tmp_path):
    from ouroboros.safety import _build_check_prompt

    ctx = SimpleNamespace(task_id="t", task_metadata={"work_kind": "owner"}, drive_root=tmp_path)
    bundle = _bundle(_build_check_prompt("run_command", {"cmd": ["ls"]}, ctx=ctx))
    assert bundle["work"]["kind"] == "owner" and bundle["work"]["label"] == "your work"
    assert bundle["occasion"]["cause"] == "owner door" and "link_facts" not in bundle["occasion"]


def _real_focus_task(pair, monkeypatch, text):
    """The focus task the host builds for a standalone message of the linked Ouroboros: the letter door's own
    admission and event, taken at the carrier's door (nothing is dispatched), through ``presence_runner._build_task``,
    the builder every place turn uses. No actor is written by this test."""
    from ouroboros import presence_carrier, presence_runner
    from ouroboros.install_identity import acting_as
    from tests.test_peer_messages import _early_answer, _enable_consciousness

    _enable_consciousness(pair.root_b, monkeypatch)
    dispatched = []  # the carrier's early answer only: the turn starts, its body never runs here
    monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                        lambda execution, **kw: (dispatched.append(kw), _early_answer(execution)))
    with receiver_client(pair, monkeypatch) as client:
        sent = client.post("/api/letters", json=letter(kind="message", text=text), headers=pair.headers())
        assert sent.status_code == 200, sent.text
    assert len(dispatched) == 1, "consciousness is on: the message woke one focus"
    with acting_as(pair.root_b):
        return presence_runner._build_task(dispatched[0]["admission"], dispatched[0]["event"],
                                           drive_root=pair.root_b, staged_files=())


def test_the_real_focus_of_a_peer_message_carries_its_verified_writer_to_safety_and_to_the_frame(tmp_path, monkeypatch):
    """The host's own focus of a linked Ouroboros's message names the verified writer (its link and installation) and
    my human's facts of that link; Safety and the mind's frame read them from there. The message is never my
    human's: no owner account, no owner cause. A choice my human changes later is said as the one that stands."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place, place_records
    from ouroboros.presence_context import frame_focus_content
    from ouroboros.safety import _build_check_prompt

    text = "Status: the count is done."
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, note="Velm may send status notes only.")
    task = _real_focus_task(pair, monkeypatch, text)
    metadata = task["metadata"]
    actor = metadata["focus"]["actor"]
    with acting_as(pair.root_b):
        place = place_records(pair.root_b).places[pair.place_b]
    assert (actor["kind"], actor["link_id"], actor["install_id"]) == ("ouroboros", pair.link_id, place.account_id)
    assert actor["owner_confirmed"] is False and actor["authenticated"] is True and actor["is_bot"] is True
    assert "mine: yes; may_start: no." in actor["link_facts"] and "an agent, never my human" in actor["link_facts"]
    ctx = SimpleNamespace(task_id=task["id"], task_metadata=metadata, drive_root=pair.root_b,
                          budget_drive_root=pair.root_b)
    with acting_as(pair.root_b):
        bundle = _bundle(_build_check_prompt("place_reply", {"place_id": pair.place_b, "text": "Noted."}, ctx=ctx))
    assert bundle["work"]["kind"] == "consciousness" and bundle["work"]["label"] == "consciousness"
    assert bundle["occasion"]["cause"] == "linked Ouroboros in a place"
    assert bundle["occasion"]["focus"]["actor"] == actor, "the verified writer and its link's facts, as admitted"
    assert "link_facts" not in bundle["occasion"], "unchanged since the admission: said once"
    assert [row["place_id"] for row in bundle["subject"]["place"]["places"]] == [pair.place_b]
    assert "linked_ouroboros" not in bundle["subject"]["place"], "the occasion already states this link"
    notes = json.dumps(bundle["standing_delegation"]["place_notes"], ensure_ascii=False)
    assert "Velm may send status notes only." in notes
    frame = frame_focus_content(task, text)
    shown = frame if isinstance(frame, str) else frame[0]["text"]
    assert '"kind": "ouroboros"' in shown and pair.link_id in shown, "the frame's host facts of the writer"
    assert '"owner_confirmed": false' in shown and "an agent, never my human" in shown and "may_start: no." in shown
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=True)
        later = _bundle(_build_check_prompt("place_reply", {"place_id": pair.place_b, "text": "Noted."}, ctx=ctx))
    now = later["occasion"]["link_facts"]
    assert now.startswith("As they stand now, changed since this message was admitted: ") and "may_start: yes." in now


def test_a_writer_the_host_did_not_verify_stays_a_person_in_a_place(tmp_path, monkeypatch):
    """Only the letter door's own event for the link's own place makes a writer my human's linked Ouroboros. Another
    connector describing its actor that way, or an event on the linked place naming another link or installation, or
    one not authenticated, records no such writer; and no description of an actor ever confirms an owner."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import Place, WakeItem, connect_place, place_records
    from ouroboros.presence_context import focus_metadata
    from ouroboros.presence_events import PresenceTurnEvent
    from ouroboros.safety import _build_check_prompt

    def event(place, **actor):
        return PresenceTurnEvent(
            source_event_id="event-1", provider=place.provider, account_id=place.account_id,
            conversation_id=place.conversation_id, thread_id="", conversation_key="key-1", conversation={}, message={},
            text="A neutral line.", facts=("direct_message",), identity={},
            actor={"platform_actor_id": "writer-1", "display_name": "Writer", "authenticated": True, "is_bot": True,
                   "kind": "ouroboros", "link_id": place.conversation_id, "install_id": place.account_id, **actor})

    def focus(place, **actor):
        admission = SimpleNamespace(place=place, binding_id=place.place_id, allowance={}, circle=None)
        return focus_metadata(admission, event(place, **actor), "focus-1", 1, None)

    room = Place(place_id="place-room", connector="chat", provider="chat", account_id="account-1",
                 conversation_id="room-1", label="Room", wakes=(WakeItem("direct_message", "owner"),))
    claimed = focus(room)
    assert "kind" not in claimed["focus"]["actor"] and "link_facts" not in claimed["focus"]["actor"]
    assert claimed["focus"]["actor"]["owner_confirmed"] is False
    ctx = SimpleNamespace(task_id="focus-1", task_metadata=claimed, drive_root=tmp_path)
    occasion = _bundle(_build_check_prompt("run_command", {"cmd": ["ls"]}, ctx=ctx))["occasion"]
    assert occasion["cause"] == "person in a place" and "link_facts" not in occasion

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        linked = place_records(pair.root_b).places[pair.place_b]
        assert focus(linked)["focus"]["actor"]["kind"] == "ouroboros", "the door's own stamp for this link"
        for other in ({"link_id": "another-link"}, {"install_id": "another-install"}, {"authenticated": False},
                      {"kind": "person"}):
            assert "kind" not in focus(linked, **other)["focus"]["actor"], other
        # A focus recorded as a linked Ouroboros's reads as one only while its place is a link's own.
        stored = connect_place(pair.root_b, connector="chat", provider="chat", label="Room", account_id="account-1",
                               conversation_id="room-1", state="silent")
        forged = {"work_kind": "consciousness", "wake_cause": "focus", "focus": {
            "place_id": stored.place_id, "actor": {"kind": "ouroboros", "owner_confirmed": False}}}
        ctx = SimpleNamespace(task_id="focus-2", task_metadata=forged, drive_root=pair.root_b)
        occasion = _bundle(_build_check_prompt("run_command", {"cmd": ["ls"]}, ctx=ctx))["occasion"]
    assert occasion["cause"] == "person in a place" and "link_facts" not in occasion


def test_words_for_a_linked_ouroboros_are_read_with_its_facts_beside_its_notes(tmp_path, monkeypatch):
    """My human's own work writing to their linked Ouroboros lands in another installation: the check carries that
    place's notes and, beside them, the host's line of the link's facts with what the two choices do not grant."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place
    from ouroboros.safety import _build_check_prompt

    pair = introduce(tmp_path, monkeypatch)
    ctx = SimpleNamespace(task_id="work-1", task_metadata={"work_kind": "owner"}, drive_root=pair.root_a,
                          budget_drive_root=pair.root_a)
    with acting_as(pair.root_a):
        edit_place(pair.root_a, pair.place_a, "add_note", author=AUTHOR_OWNER, text="Velm may get the sample counts.")
        bundle = _bundle(_build_check_prompt("message_ouroboros", {"place_id": pair.place_a, "text": "The counts."},
                                             ctx=ctx))
    assert bundle["occasion"]["cause"] == "owner door" and "link_facts" not in bundle["occasion"]
    line = bundle["subject"]["place"]["linked_ouroboros"][pair.place_a]
    assert "another installation: an agent, never my human" in line and "mine: yes; may_start: no." in line
    assert "neither my human's consent to share with it what my human keeps private" in line
    notes = json.dumps(bundle["standing_delegation"]["place_notes"], ensure_ascii=False)
    assert "Velm may get the sample counts." in notes, "my human's own words for this place stay usable"


from tests._safety_fixtures import supervisor as supervisor_fixture  # noqa: E402 - the hermetic Supervisor

supervisor = supervisor_fixture


def test_a_letters_work_meets_my_own_lifes_coverage_not_the_owner_works(supervisor, tmp_path):
    """Owner work's coverage off, my own life's full: a command of a letter's root (and its helpers, which carry
    the same ``peer_letter``) is read by the Supervisor; ordinary owner work under the same settings is not."""
    from ouroboros.settings_integrity import _actor_scope

    sup = supervisor
    sup.modes.update(owner="off", consciousness="full")
    letter_work = {"work_kind": "owner", "peer_letter": {"place_id": "place-1", "link_id": "link-0001"}}
    sup.ctx.task_metadata = dict(letter_work)
    allowed, _ = sup.module.check_safety("run_command", {"cmd": ["rm", "-rf", "build"]}, ctx=sup.ctx)
    assert allowed and len(sup.calls) == 1, sup.events
    sup.ctx.task_metadata = {"work_kind": "owner"}
    allowed, _ = sup.module.check_safety("run_command", {"cmd": ["rm", "-rf", "dist"]}, ctx=sup.ctx)
    assert allowed and len(sup.calls) == 1 and {"type": "skip"} in sup.events
    assert _actor_scope({"metadata": letter_work}, tmp_path)[1]["autonomous"] is True, "a place's controls"
    assert _actor_scope({"metadata": {"work_kind": "owner"}}, tmp_path)[1]["autonomous"] is False


def test_an_unreadable_place_store_holds_the_call_that_needs_its_rules(supervisor, monkeypatch):
    """A call naming a place is read with that place's rules. While the store cannot be read there are none to read:
    the check holds that call, asks no model and lets nothing through, and reads it once the store answers. A call
    naming no place reads no store."""
    from ouroboros import places

    def unreadable(_root):
        raise places.PlaceError("places_unreadable", "places", "state/places.json needs repair")

    sup = supervisor
    named = {"action": "inspect", "place_id": "place-1"}
    monkeypatch.setattr(places, "place_records", unreadable)
    allowed, message = sup.module.check_safety("configure_presence", named, ctx=sup.ctx)
    assert not allowed and message.startswith("⚠️ SAFETY_UNAVAILABLE: not run.") and sup.calls == []
    assert [event["hold"] for event in sup.events if event["type"] == "safety_hold"] == ["UNAVAILABLE"]
    assert sup.module.check_safety("run_command", {"cmd": ["rm", "-rf", "build"]}, ctx=sup.ctx)[0]
    assert len(sup.calls) == 1, "no place named: the store is not asked"
    monkeypatch.setattr(places, "place_records", lambda _root: places.PlaceRecords(0, {}, (), {}))
    sup.later()
    assert sup.module.check_safety("configure_presence", named, ctx=sup.ctx)[0] and len(sup.calls) == 2


def test_a_focus_turning_to_another_place_reads_its_own_places_rules_only_beside_a_linked_ouroboros(tmp_path,
                                                                                                   monkeypatch):
    """A focus woken by a person in one place whose call lands in another. Between two ordinary places the check
    reads the destination's rules alone, as it always has. Words for my human's linked Ouroboros leave for another
    installation: there the rules standing now where the request came from are read beside the destination's, each
    under its own place, and the link's facts are stated for the place the words land in and no other."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, connect_place, edit_place
    from ouroboros.safety import _build_check_prompt

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_a):
        room, desk = (connect_place(pair.root_a, connector="chat", provider="chat", label=label, account_id="account-1",
                                    conversation_id=conversation, state="active")
                      for label, conversation in (("Room", "room-1"), ("Desk", "desk-1")))
        edit_place(pair.root_a, room.place_id, "add_note", author=AUTHOR_OWNER, text="People here may ask for status.")
        edit_place(pair.root_a, desk.place_id, "add_note", author=AUTHOR_OWNER, text="The desk gets short notes.")
        edit_place(pair.root_a, pair.place_a, "add_note", author=AUTHOR_OWNER, text="Velm may get the sample counts.")
        ctx = SimpleNamespace(task_id="focus-1", drive_root=pair.root_a, budget_drive_root=pair.root_a, task_metadata={
            "work_kind": "consciousness", "wake_cause": "focus",
            "focus": {"place_id": room.place_id, "actor": {"platform_actor_id": "person-1", "owner_confirmed": False}}})
        ordinary = _bundle(_build_check_prompt("place_reply", {"place_id": desk.place_id, "text": "A note."}, ctx=ctx))
        bundle = _bundle(_build_check_prompt("message_ouroboros", {"place_id": pair.place_a, "text": "The counts."},
                                             ctx=ctx))
    assert {(row["place_id"], row["text"]) for row in ordinary["standing_delegation"]["place_notes"]} == {
        (desk.place_id, "The desk gets short notes.")}, "between ordinary places: the destination's rules alone"
    assert "linked_ouroboros" not in ordinary["subject"]["place"]
    rules = {(row["place_id"], row["text"]) for row in bundle["standing_delegation"]["place_notes"]}
    assert rules == {(room.place_id, "People here may ask for status."),
                     (pair.place_a, "Velm may get the sample counts.")}, rules
    assert [row["place_id"] for row in bundle["subject"]["place"]["places"]] == [pair.place_a], "where it lands"
    assert bundle["occasion"]["cause"] == "person in a place" and bundle["occasion"]["focus"]["place_id"] == room.place_id
    assert set(bundle["subject"]["place"]["linked_ouroboros"]) == {pair.place_a}, "the destination's link facts only"
