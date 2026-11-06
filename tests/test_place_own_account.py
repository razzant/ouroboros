"""A place's own account never wakes it; its own posts stay heard and kept.

The 28.10 flood: GitHub places woke the mind on its own PRs, comments, merges and edits (32 of 35 wakes in one
evening), whatever tool acted as the account. The account a place writes as is ``identity.sends_as`` on the event;
an arrival written by it is mine. Where the place writes as my human's own confirmed account, their posts there are
their words and keep waking as before; an actor the connector reports unauthenticated, or an event without an
identity, is never mine.
"""
from dataclasses import asdict, replace
import json

from ouroboros import dialogue_store, places as p
from ouroboros.place_intake import author_hears, decide_attention
from tests.test_place_focus import focus as focus
from tests.test_place_followup import client as client

ME = {"sends_as": {"id": "me", "label": "ouroboros-agent", "kind": "user"}}


def _own(event, **actor):
    return replace(event, identity=ME, actor={"platform_actor_id": "me", **actor})


def test_own_account_event_is_heard_and_another_actor_still_wakes(focus):
    root, _place, admission, event = focus
    decision, details = decide_attention(root, admission, _own(event))
    assert (decision, details) == ("silent", {"reason": "own_account"})  # the place selected `mention`: still quiet
    assert author_hears(decision, details, _own(event)) is None  # nor mailed to a live author of the conversation
    assert decide_attention(root, admission, replace(event, identity=ME))[0] == "focus"  # someone else, same place
    legacy_shape = replace(event, identity=ME, actor={"id": "me"})  # an actor named by `id`, as other readers accept
    assert decide_attention(root, admission, legacy_shape) == ("silent", {"reason": "own_account"})


def test_only_a_proven_own_account_is_mine(focus):
    root, place, admission, event = focus
    # The connector cannot vouch for the author (a mail From): the ordinary path.
    assert decide_attention(root, admission, _own(event, authenticated=False))[0] == "focus"
    # No identity on the event (an older connector): nothing is mine.
    assert decide_attention(root, admission, replace(_own(event), identity=None))[0] == "focus"
    # The place writes as my human's own confirmed account: their posts there are their words.
    place = p.edit_place(root, place.place_id, "confirm_owner_account", author="owner",
                         platform_actor_id="me", label="Anton")
    assert decide_attention(root, replace(admission, place=place), _own(event))[0] == "focus"


def test_a_paused_place_still_says_paused(focus):
    root, place, admission, event = focus
    p.edit_place(root, place.place_id, "set_state", author="owner", state="paused")
    assert decide_attention(root, admission, _own(event)) == ("silent", {"reason": "paused"})


def test_host_capture_keeps_the_own_post_and_starts_nothing(focus, client):
    root, _place, _admission, event = focus
    wire = asdict(_own(event))
    for key in ("delivery_reporting_version", "continuation_version"):
        wire.pop(key)
    body = {"event": wire, "staged_files": [], "continuation_version": 1, "delivery_reporting_version": 1}
    answer = client.post("/presence/turn", json=body)
    assert answer.status_code == 200, answer.text
    assert answer.json()["outcome"] == "silent"
    row, = [row for row in dialogue_store.iter_rows(root, "chat") if row.row.get("client_message_id") == "event1"]
    with dialogue_store.transaction(root, write=False) as conn:
        stored = conn.execute("SELECT dispatch_input FROM dialogue WHERE stream='chat' AND seq=?", (row.seq,)).fetchone()
    attention = json.loads(stored["dispatch_input"])["place_attention"]
    assert attention["decision"] == "heard" and attention["reason"] == "own_account"
