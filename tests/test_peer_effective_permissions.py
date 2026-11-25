"""A linked Ouroboros's place says what my human stored and, apart from it, what is in effect now.

The stored choices (``mine``, ``may_start``) stay as my human set them; what the letter door accepts also depends on
the link (confirmed, revoked) and the place (Paused). The projection and the host's line state both, messages and
assignments apart, and the Off note says what the switch governs. Only a linked Ouroboros's place holds the choices.
"""
from __future__ import annotations

import pytest

from tests._peer_letters_shared import introduce, letter, receiver_client

CONFIRMED, PENDING, REVOKED = ({"code_check": "confirmed"}, {"code_check": "pending"},
                               {"code_check": "confirmed", "revoked_at": "2026-01-01T00:00:00+00:00"})
MINE, ANSWERS, UNMARKED = ({"mine": True, "may_start": True}, {"mine": True, "may_start": False},
                           {"mine": False, "may_start": True})


@pytest.mark.parametrize(("peer", "state", "link", "messages", "assignments", "reason", "switch"), [
    (MINE, "active", CONFIRMED, True, True, "", "on"),
    (MINE, "active", CONFIRMED, True, True, "", "off"),  # a letter is no wake: Off governs what wakes me
    (MINE, "silent", CONFIRMED, True, False, "silent", "on"),  # it still hears, but no answer could leave
    (MINE, "active", CONFIRMED, True, False, "panic", "panic"),  # new paid work waits for my human's return
    (ANSWERS, "active", CONFIRMED, True, False, "may_not_start", "panic"),
    (ANSWERS, "active", CONFIRMED, True, False, "may_not_start", "on"),
    (UNMARKED, "active", CONFIRMED, False, False, "not_mine", "on"),  # the stored may_start is not in effect
    (None, "active", CONFIRMED, False, False, "not_mine", "on"),
    (MINE, "paused", CONFIRMED, False, False, "paused", "on"),
    (MINE, "active", PENDING, False, False, "code_check_pending", "on"),
    (MINE, "active", REVOKED, False, False, "revoked", "on"),  # whatever the place's stored state says
    (MINE, "active", None, False, False, "unlinked", "on"),
])
def test_what_is_in_effect_follows_the_link_the_place_and_the_choices_in_the_doors_order(peer, state, link, messages,
                                                                                      assignments, reason, switch):
    from ouroboros.peer_links import effective_permissions

    assert effective_permissions(peer, state=state, link=link, switch=switch) == {
        "messages": messages, "assignments": assignments, "reason": reason}


def test_the_hosts_line_keeps_the_stored_choices_apart_from_what_is_in_effect():
    """An unmarked link keeps its stored ``may_start`` and never reads as one whose letters may still start work; each
    refusal names the fact that causes it, and absent choices are never told as my human's refusal."""
    from ouroboros.peer_links import effective_permissions, peer_facts_line

    def line(peer, state="active", link=CONFIRMED):
        return peer_facts_line(peer, effective_permissions(peer, state=state, link=link))

    unmarked = line(UNMARKED)
    assert "stored choices for it, set at their door: mine: no; may_start: yes." in unmarked
    assert ("In effect now: nothing it sends is accepted here, because my human has not marked it as their own "
            "Ouroboros; the stored choices stay as they are.") in unmarked
    assert "starts ordinary work" not in unmarked and "is saved in this place's conversation" not in unmarked
    answers = line(ANSWERS)
    assert "mine: yes; may_start: no." in answers and "is saved in this place's conversation" in answers
    assert "a letter that assigns work is refused, because may_start is no." in answers
    mine = line(MINE)
    assert "a letter that assigns work starts ordinary work here, also while consciousness is off." in mine
    assert "waking nothing while consciousness is off or its allowance is spent" in mine
    assert "because this link was revoked" in line(MINE, link=REVOKED)
    assert "because the link's six-digit code is not confirmed on this side yet" in line(MINE, link=PENDING)
    assert "because this place is Paused" in line(MINE, state="paused")
    silent = line(MINE, state="silent")
    assert "is saved in this place's conversation" in silent, "a Silent place still hears what it is told"
    assert "a letter that assigns work is refused, because this place is Silent here" in silent
    panic = peer_facts_line(MINE, effective_permissions(MINE, state="active", link=CONFIRMED, switch="panic"))
    assert "is refused, because a Panic stopped this installation, and new work waits for my human's manual return" in panic
    absent = line(None)
    assert "set at their door: none recorded." in absent and "my human recorded no choices for it" in absent
    assert "has not marked" not in absent, "an absent choice is not my human's typed refusal"
    unknown = peer_facts_line(MINE, {"messages": None, "assignments": None, "reason": "unknown"})
    assert "In effect now: unknown: the link registry could not be read" in unknown
    for said in (unmarked, answers, mine, absent, unknown):
        assert said.startswith("Host facts of this linked Ouroboros (another installation: an agent, never my human).")
        assert ("By themselves they are neither my human's consent to share with it what my human keeps private, nor "
                "their order for what only my human may ask") in said
        assert "my human's notes for this place, its circle's words and their rules say what they allow." in said


def _view(pair, **kwargs):
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import place_view
    from ouroboros.places import place_records

    with acting_as(pair.root_b):
        records = place_records(pair.root_b)
        return place_view(pair.root_b, records.places[pair.place_b], declarations=[], records=records,
                          allowance={"status": "available"}, **kwargs)


def test_unmarking_keeps_the_stored_choice_and_the_projection_and_the_door_agree(tmp_path, monkeypatch):
    """My human unmarks the link at their door: ``may_start`` stays stored as they set it, the projection says
    nothing is accepted (and no longer that it only hears), and the letter door refuses a message and a work letter.
    Marking it again restores both without a second choice."""
    pair = introduce(tmp_path, monkeypatch)
    view = _view(pair, switch="on")
    assert view["peer"] == MINE and view["peer_effective"] == {"messages": True, "assignments": True, "reason": ""}
    assert "quiet" not in view
    with receiver_client(pair, monkeypatch) as client:
        saved = client.post(f"/api/places/{pair.place_b}", json={"peer": {"mine": False, "may_start": True}})
        assert saved.status_code == 200, saved.text
        place = saved.json()["place"]
        assert place["peer"] == UNMARKED, "the stored choices are my human's, as they set them"
        assert place["peer_effective"] == {"messages": False, "assignments": False, "reason": "not_mine"}
        assert "mine: no; may_start: yes." in place["peer_facts"] and "nothing it sends is accepted here" in place["peer_facts"]
        for body in (letter(), letter(kind="message", text="A neutral note.")):
            refused = client.post("/api/letters", json=body, headers=pair.headers())
            assert refused.status_code == 403 and refused.json()["refusal_code"] == "peer_may_not_start", refused.text
        assert "quiet" not in _view(pair, switch="off"), "a link that accepts nothing is not hearing"
        again = client.post(f"/api/places/{pair.place_b}", json={"peer": {"mine": True, "may_start": True}})
        assert again.json()["place"]["peer_effective"] == {"messages": True, "assignments": True, "reason": ""}


def test_a_revoked_link_stays_revoked_in_effect_whatever_its_place_is_set_to(tmp_path, monkeypatch):
    """Revocation pauses the place and keeps my human's choices; making that place Active again changes its stored
    state only: nothing is accepted, and the letter door answers ``link_unknown``."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = introduce(tmp_path, monkeypatch)
    headers = pair.headers()
    with acting_as(pair.root_b):
        peer_links.revoke(pair.root_b, pair.link_id)
    view = _view(pair, switch="off")
    assert view["state"] == "paused" and view["peer"] == MINE and view["link"]["revoked_at"]
    assert view["peer_effective"] == {"messages": False, "assignments": False, "reason": "revoked"}
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="active")
    view = _view(pair, switch="off")
    assert view["state"] == "active" and view["peer_effective"]["reason"] == "revoked" and "quiet" not in view
    assert "because this link was revoked" in view["peer_facts"] and "mine: yes; may_start: yes." in view["peer_facts"]
    with receiver_client(pair, monkeypatch) as client:
        refused = client.post("/api/letters", json=letter(), headers=headers)
        assert refused.status_code == 401 and refused.json()["refusal_code"] == "link_unknown", refused.text


def test_a_revoke_that_overlaps_a_confirm_leaves_the_place_paused(tmp_path, monkeypatch):
    """A confirm starts the place before it marks the link. A revoke that lands in between pauses the place first, so
    the confirm's own start would leave a revoked link's place Active: the confirm pauses it again and refuses."""
    from ouroboros import peer_links, places
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch, confirm=False)
    real_edit, overlapped = places.edit_place, []

    def edit(root, place_id, action, **values):
        if action == "set_state" and values.get("state") == "active" and not overlapped:
            overlapped.append(True)
            peer_links.revoke(root, pair.link_id)  # the whole revoke lands before this confirm's start
        return real_edit(root, place_id, action, **values)

    monkeypatch.setattr(places, "edit_place", edit)
    with acting_as(pair.root_b):
        with pytest.raises(peer_links.PeerLinkError) as refused:
            peer_links.confirm(pair.root_b, pair.link_id)
        assert refused.value.code == "link_unknown" and overlapped == [True]
        records = places.place_records(pair.root_b)
        assert records.places[pair.place_b].state == "paused", "a revoked link's place never ends Active"
        assert peer_links.read_registry(pair.root_b)["links"][pair.link_id]["revoked_at"]
        effect = peer_links.place_permissions(pair.root_b, records.places[pair.place_b], records.peer(pair.place_b))
    assert effect == {"messages": False, "assignments": False, "reason": "revoked"}


def test_only_a_linked_ouroboros_place_holds_the_peer_choices(tmp_path):
    """Another connector's place refuses the choice at my human's own door too, and writes nothing of the same save."""
    from ouroboros.gateway.places_api import update_place
    from ouroboros.places import AUTHOR_OWNER, PlaceError, connect_place, edit_place, place_records

    place = connect_place(tmp_path, connector="chat", provider="chat", label="Room", account_id="account",
                          conversation_id="room", state="silent")
    with pytest.raises(PlaceError) as refused:
        edit_place(tmp_path, place.place_id, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=True)
    assert refused.value.code == "place_peer_invalid" and "only a linked Ouroboros's place" in str(refused.value)
    with pytest.raises(PlaceError) as refused:
        update_place(tmp_path, place.place_id, {"state": "paused", "peer": {"mine": True, "may_start": True}})
    assert refused.value.code == "place_peer_invalid"
    records = place_records(tmp_path)
    assert records.peer(place.place_id) is None and records.places[place.place_id].state == "silent"


@pytest.mark.parametrize(("cause", "kwargs"), [
    ("off", {"switch": "off"}),
    ("allowance", {"switch": "on", "allowance": "exhausted", "limit_usd": 3.0}),
])
def test_the_off_and_allowance_notes_say_what_they_govern_at_a_linked_place(cause, kwargs):
    """The switch and the allowance decide what wakes me on my own; a letter that assigns work is admitted or refused
    by the link's choices. A place that is no linked Ouroboros's reads as before."""
    from ouroboros import consciousness_view as V

    plain = V.place_quiet("active", **kwargs)
    runs = V.place_quiet("active", assignments=True, **kwargs)
    refused = V.place_quiet("active", assignments=False, **kwargs)
    assert plain["cause"] == runs["cause"] == refused["cause"] == cause and plain["note"] == V.QUIET[cause][1]
    assert runs["note"] == plain["note"] + (" A letter from the linked Ouroboros that assigns work is no wake: that work "
                                            "still starts here.")
    assert refused["note"] == plain["note"] + " A letter from the linked Ouroboros that assigns work is refused here."
    assert (runs["word"], runs["door"]) == (plain["word"], plain["door"])


def test_other_quiet_causes_keep_their_words_at_a_linked_place():
    from ouroboros import consciousness_view as V

    for kwargs in ({"switch": "panic"}, {"switch": "on", "autonomy": "observe"}):
        assert V.place_quiet("active", assignments=True, **kwargs) == V.place_quiet("active", **kwargs)


def test_the_projection_says_off_and_the_assignment_apart(tmp_path, monkeypatch):
    """With consciousness off the linked place only hears its messages; whether its letter still starts work is said
    beside that, from what is in effect."""
    pair = introduce(tmp_path, monkeypatch)
    off = _view(pair, switch="off")
    assert off["quiet"]["cause"] == "off" and off["quiet"]["note"].startswith("Consciousness is off: it only hears here")
    assert off["quiet"]["note"].endswith("is no wake: that work still starts here.")
    with receiver_client(pair, monkeypatch) as client:
        client.post(f"/api/places/{pair.place_b}", json={"peer": {"mine": True, "may_start": False}})
    answers = _view(pair, switch="off")
    assert answers["peer_effective"] == {"messages": True, "assignments": False, "reason": "may_not_start"}
    assert answers["quiet"]["note"].endswith("A letter from the linked Ouroboros that assigns work is refused here.")


def test_the_door_refuses_new_work_where_it_could_not_answer_or_a_panic_holds_and_keeps_the_rest(tmp_path, monkeypatch):
    """The letter door enforces the same reading the card states: a Silent place hears a message but refuses a work
    letter (``link_silent``: its answer could not leave); a Panic awaiting my human's return refuses new work
    (``panic_hold``, a mock of the switch's one reading) while a message is still only captured. Neither starts work."""
    from ouroboros import place_projection
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="silent")
    with receiver_client(pair, monkeypatch) as client:
        refused = client.post("/api/letters", json=letter(), headers=pair.headers())
        assert refused.status_code == 403 and refused.json()["refusal_code"] == "link_silent", refused.text
        heard = client.post("/api/letters", json=letter(kind="message", text="A neutral note."), headers=pair.headers())
        assert heard.status_code == 200 and heard.json()["state"] == "received", heard.text
    view = _view(pair, switch="on")
    assert view["peer_effective"] == {"messages": True, "assignments": False, "reason": "silent"}
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="active")
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "panic")
    with receiver_client(pair, monkeypatch) as client:
        held = client.post("/api/letters", json=letter(), headers=pair.headers())
        assert held.status_code == 403 and held.json()["refusal_code"] == "panic_hold", held.text
        heard = client.post("/api/letters", json=letter(kind="message", text="Another note."), headers=pair.headers())
        assert heard.status_code == 200 and heard.json()["state"] == "received", heard.text
    assert _view(pair, switch="panic")["peer_effective"] == {"messages": True, "assignments": False, "reason": "panic"}
    from tests._peer_letters_shared import events

    assert not events(pair.root_b, "peer_letter_accepted"), "no work started"
