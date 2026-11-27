"""The consciousness switch's words while a linked Ouroboros may start work here (``consciousness_view``, its facts from
``place_projection.starts_work_here`` and ``letters_can_start_work``, both the letter door's own reading). Off still lets its requests for work start;
its other words wake nothing while Off, so its place is heard like any other place."""
from __future__ import annotations

import time

from tests._peer_letters_shared import introduce


def test_starts_work_here_is_the_letter_doors_own_reading_never_a_second_one():
    """A view's ``peer_effective`` is ``peer_links.effective_permissions``: the switch's words read it, so Silent and
    Panic refuse here exactly as the letter door refuses, and Off never stops a request for work."""
    from ouroboros.peer_links import effective_permissions
    from ouroboros.place_projection import starts_work_here

    checked = {"code_check": "confirmed", "revoked_at": ""}

    def view(state="active", peer=None, link=checked, switch="on", connector="ouroboros"):
        peer = {"mine": True, "may_start": True} if peer is None else peer
        return {"connector": connector, "state": state, "peer": peer,
                "peer_effective": effective_permissions(peer, state=state, link=link, switch=switch)}

    assert starts_work_here(view()) and starts_work_here(view(switch="off")), "Off never stops a request for work"
    assert not starts_work_here(view(state="paused"))
    assert not starts_work_here(view(state="silent")), "Silent: work it would start could not answer"
    assert not starts_work_here(view(switch="panic"))
    assert not starts_work_here(view(peer={"mine": True, "may_start": False}))
    assert not starts_work_here(view(peer={"mine": False, "may_start": True}))
    assert not starts_work_here(view(link={"code_check": "pending", "revoked_at": ""}))
    assert not starts_work_here(view(link={"code_check": "confirmed", "revoked_at": "2026-11-23"}))
    assert not starts_work_here(view(connector="telegram"))
    unread = {"connector": "ouroboros", "peer_effective": {"messages": None, "assignments": None, "reason": "unknown"}}
    assert starts_work_here(unread) is None, "an unread registry is no claim that work starts"


def test_off_names_requests_from_a_linked_ouroboros_and_its_place_is_heard_like_any_other():
    from ouroboros.consciousness_view import (describe_consciousness_state, heard_sentence, off_sentence,
                                              position_words, switch_preview)

    linked = {"place_id": "p", "label": "Tarel", "state": "active", "connector": "ouroboros", "letters_start_work": True}
    chat = {"place_id": "c", "label": "Chat", "state": "active", "connector": "telegram"}
    described = describe_consciousness_state({"places_brief": [linked]}, False)
    assert (described["position"], described["position_short"], described["detail"]) == (
        "Only when you write to it or your Ouroboroses ask for work", "Only when asked",
        "Requests from your Ouroboroses still start work. 1 place is heard, not answered.")
    assert "write to it" not in described["position_short"], "a linked Ouroboros's words do not wake it while Off"
    assert describe_consciousness_state({"places_brief": [linked, chat]}, False)["detail"] == (
        "Requests from your Ouroboroses still start work. 2 places are heard, not answered.")
    plain = describe_consciousness_state({"places_brief": [chat]}, False)
    assert (plain["position"], plain["position_short"]) == ("Only when you write to it", "Only when you write to it")
    assert position_words(False, [{**linked, "letters_start_work": False}]) == "Only when you write to it"
    unread = {**linked, "letters_start_work": None}
    assert position_words(False, [unread]) == "Background wakes are off"
    assert position_words(False, [unread, linked]) == described["position"]
    assert "could not be read" in heard_sentence([unread])
    assert "could not be read" in switch_preview("off", [chat], letters=None)["lines"][0]
    assert "could not be read" in off_sentence([], letters=None)
    assert position_words(True, [linked]) == "Also on its own and in places"
    assert heard_sentence([{**linked, "state": "paused", "letters_start_work": False}]) == (
        "Nothing wakes it between your messages.")
    assert off_sentence([]) == "Consciousness off: Ouroboros acts only when you write to it."
    assert off_sentence([], letters=True) == (
        "Consciousness off: Ouroboros acts only when you write to it or your Ouroboroses ask for work.")
    assert sorted(place["label"] for place in switch_preview("off", [linked, chat])["places"]) == ["Chat", "Tarel"]
    assert switch_preview("off", [chat], letters=True)["lines"][0] == (
        "Ouroboros will act only when you write to it or your Ouroboroses ask for work.")
    assert switch_preview("off", [chat])["lines"][0] == "Ouroboros will act only when you write to it."


def test_the_switch_reads_from_the_store_whether_letters_start_work_here(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import letters_can_start_work
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, STATE_SILENT, edit_place

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        assert letters_can_start_work(pair.root_b), "B lets A start work there: its letters start work"
    with acting_as(pair.root_a):
        assert not letters_can_start_work(pair.root_a), "A does not let B start work"
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_SILENT)
        assert not letters_can_start_work(pair.root_b), "Silent: the letter door refuses work, so the switch says so too"
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        assert not letters_can_start_work(pair.root_b), "paused letters start nothing"
    unchecked = introduce(tmp_path / "unchecked", monkeypatch, confirm=False)
    with acting_as(unchecked.root_b):
        assert not letters_can_start_work(unchecked.root_b), "nothing starts before the code check"


def test_a_linked_place_view_carries_its_two_facts_and_its_link_and_no_letter_feed(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import place_view
    from ouroboros.places import place_records

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        place = place_records(pair.root_b).places[pair.place_b]
        full = place_view(pair.root_b, place, declarations=[])
        brief = place_view(pair.root_b, place, brief=True)
    for view in (full, brief):
        assert view["peer"] == {"mine": True, "may_start": True} and view["link"]["code_check"] == "confirmed"
        assert not {"letters", "refusals", "address", "moved"} & set(view), "its conversation is the place's own"
    assert "remote_unknown_at" in full["link"]


def test_a_place_or_link_write_moves_the_state_publication_that_carries_these_words(tmp_path, monkeypatch):
    """``/api/state`` serves one publication per revision: the place store and the link registry are part of that key,
    so the switch's words follow an introduction, a pause or a removal at once instead of at some later move."""
    from ouroboros import peer_links, utils
    from ouroboros.gateway.state_publication import _file_stamps, _place_files
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, edit_place

    pair = introduce(tmp_path, monkeypatch)
    files = _place_files(str(pair.root_b))
    assert [path.name for path in files] == ["places.json", "peer_links.json"] and all(path.exists() for path in files)
    monkeypatch.setattr(utils, "_RACY_NS", 0)  # no clock tick to wait out: a file written before a stamp proves itself

    def settled() -> tuple:
        time.sleep(0.1)  # a stamp is provable only once the wall clock passed the write: Windows' ticks last ~16 ms
        stamps = _file_stamps(files)
        assert stamps == _file_stamps(files), "an unchanged store keeps its key"
        return stamps

    before = settled()
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
    assert _file_stamps(files) != before, "a place write moves the key"
    before = settled()
    with acting_as(pair.root_b):
        peer_links.revoke(pair.root_b, pair.link_id)
    assert _file_stamps(files) != before, "a link write moves the key"



def test_owner_invitation_read_uses_registry_and_returns_only_the_check_state(monkeypatch):
    from ouroboros import peer_links

    row = {"place_id": "place1", "peer_label": "Tarel", "peer_install_id": "test-install", "code": "123456",
           "code_confirmed_at": "time", "revoked_at": "", "unused_field": "not shown"}
    monkeypatch.setattr(peer_links, "read_registry", lambda _root: {"links": {"link1": row}})
    assert peer_links.introduction_view(None, "unused") is None
    assert peer_links.introduction_view(None, "link1") == {
        "link_id": "link1", "place_id": "place1", "peer_label": "Tarel", "peer_install_id": "test-install",
        "code": "123456", "code_check": "confirmed", "revoked_at": ""}
