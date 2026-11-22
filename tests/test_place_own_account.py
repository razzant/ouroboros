"""A place's own account never wakes it; its own posts stay heard and kept.

The 28.10 flood: GitHub places woke the mind on its own PRs, comments, merges and edits (32 of 35 wakes in one
evening), whatever tool acted as the account. The account a place writes as is ``identity.sends_as`` on the event;
an arrival written by it is mine. Where the place writes as my human's own confirmed account, their posts there are
their words and keep waking as before; an actor the connector reports unauthenticated, or an event without an
identity, is never mine.
"""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import threading

from ouroboros import dialogue_store, places as p
from ouroboros.conversation_lease import PresenceTurnGate
from ouroboros.owner_mailbox import _mailbox_path, deliver_task_message
from ouroboros.place_intake import author_hears, decide_attention
from ouroboros.presence_bindings import conversation_key
from tests.test_place_focus import focus as focus
from tests.test_place_followup import client as client

ME = {"sends_as": {"id": "me", "label": "ouroboros-agent", "kind": "user"}}
# A connector that cannot name who made a change sends an actor without a platform id (GitHub's revisions do).
UNNAMED = {"platform": "chat", "attribution": "unknown"}
STATE_CHANGE = {"non_text": [{"kind": "state_change", "description": "Assigned to ouroboros-agent"}]}


def _own(event, **actor):
    return replace(event, identity=ME, actor={"platform_actor_id": "me", **actor})


def _change(event, actor=UNNAMED, message=STATE_CHANGE, text="", facts=("assigned",)):
    return replace(event, actor=dict(actor), message=dict(message), text=text, facts=facts)


def _wire(event):
    body = {key: value for key, value in asdict(event).items()
            if key not in {"delivery_reporting_version", "continuation_version"} and (key != "identity" or value)}
    return {"event": body, "staged_files": [], "continuation_version": 1, "delivery_reporting_version": 1}


def _attention(root):
    with dialogue_store.transaction(root, write=False) as conn:
        rows = conn.execute("SELECT dispatch_input FROM dialogue WHERE stream='chat' AND direction='in'").fetchall()
    return [json.loads(row["dispatch_input"] or "{}").get("place_attention") for row in rows]


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


# --- A change whose initiator the connector cannot name (owner: background, never a wake) ---


def test_an_unattributed_change_is_background_whatever_the_wake_list(focus):
    root, place, admission, event = focus
    for wakes in (["assigned"], ["any_message"]):  # the wildcard too: my own assignment reads the same as a person's
        p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=wakes)
        change = _change(event)
        decision, details = decide_attention(root, admission, change)
        assert (decision, details) == ("silent", {"reason": "change_actor_unknown", "background": True})
        assert author_hears(decision, details, change) is False  # a live author hears context, never a wake
    # An older connector's revision marker reads the same, words or not; the Host's route never overrides it.
    legacy = _change(event, message={"previous_source_event_id": "obs:7:1"}, text="Title\n\nBody", facts=("any_message",))
    routed = replace(admission, allowance={"route": True})
    assert decide_attention(root, routed, legacy) == ("silent", {"reason": "change_actor_unknown", "background": True})
    # An actor known only by an older connector's ``id`` is named; a whitespace id is not.
    assert decide_attention(root, admission, _change(event, actor={"id": "person"}))[0] == "focus"
    assert decide_attention(root, admission, _change(event, actor={"platform_actor_id": " "}))[1]["reason"] == "change_actor_unknown"


def test_a_named_actors_change_keeps_the_ordinary_path(focus):
    root, place, admission, event = focus
    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=["assigned"])
    assert decide_attention(root, admission, _change(event, actor={"platform_actor_id": "person"}))[0] == "focus"
    # A person's edit in its own thread wakes the conversation's live author like any follow-up there ...
    edit = replace(_change(event, actor={"platform_actor_id": "person"}, text="Revised words", facts=("any_message",)),
                   thread_id="issue.7")
    decision, details = decide_attention(root, admission, edit)
    assert details["reason"] == "no_wake_match" and author_hears(decision, details, edit) is True
    # ... and a state-only action of theirs (no words, no message fact) is background for that author.
    state = replace(edit, text="", facts=())
    assert author_hears(*decide_attention(root, admission, state), state) is False


def test_my_own_change_stays_mine_and_my_humans_stays_theirs(focus):
    root, place, admission, event = focus
    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=["assigned"])
    own = _own(_change(event))
    assert decide_attention(root, admission, own) == ("silent", {"reason": "own_account"})
    assert author_hears("silent", {"reason": "own_account"}, own) is None  # never mailed to a live author either
    place = p.edit_place(root, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="me", label="qa-owner")
    assert decide_attention(root, replace(admission, place=place), own)[0] == "focus"  # my human's own account


def test_new_words_from_an_unnamed_author_stay_new_words_even_when_edited_before_seen(focus):
    root, _place, admission, event = focus
    first_seen = replace(event, actor=dict(UNNAMED), message={"sent_at": "2026-11-16T09:00:00+00:00",
                                                              "edited_at": "2026-11-16T10:00:00+00:00"})
    assert decide_attention(root, admission, first_seen)[0] == "focus"  # one edit time proves no revision
    for message in ({"non_text": [{"kind": "sticker", "description": "a cat"}]},  # another kind is no change
                    {"non_text": [{"kind": "state_change", "description": " "}]},  # malformed: not this fact
                    {"previous_source_event_id": ""}, {"previous_source_event_id": 7}):
        assert decide_attention(root, admission, replace(first_seen, message=message))[0] == "focus", message


def test_bot_controls_stay_intact(focus):
    root, place, admission, event = focus
    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=["assigned", "any_message"])
    # A bot the connector names no id for: who changed it is unknown, and the bot gate still holds. Without a
    # ``from_bot`` item it reaches no live author, not even as background (the wildcard included) ...
    unnamed_bot = _change(event, actor={**UNNAMED, "is_bot": True})
    decision, details = decide_attention(root, admission, unnamed_bot)
    assert (decision, details) == ("silent", {"reason": "change_actor_unknown", "background": False})
    assert author_hears(decision, details, unnamed_bot) is None
    # ... and with one selected it is background there, still never a wake or a new focus.
    p.edit_place(root, place.place_id, "add_wake", author="owner", fact="from_bot")
    decision, details = decide_attention(root, admission, unnamed_bot)
    assert (decision, details) == ("silent", {"reason": "change_actor_unknown", "background": True})
    assert author_hears(decision, details, unnamed_bot) is False
    # A named bot's change meets the existing bot gate: no ``from_bot`` item, no wake and no author mail.
    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=["assigned"])
    named_bot = _change(event, actor={"platform_actor_id": "ci", "is_bot": True})
    decision, details = decide_attention(root, admission, named_bot)
    assert details["reason"] == "no_wake_match" and author_hears(decision, details, named_bot) is None
    # New words of an unnamed person keep the ordinary path beside all of this.
    assert decide_attention(root, admission, replace(event, actor=dict(UNNAMED), facts=("assigned",)))[0] == "focus"


def test_host_capture_keeps_an_unattributed_change_and_its_live_author_hears_it_labelled(focus, client):
    root, place, _admission, event = focus
    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=["assigned", "any_message"])
    answer = client.post("/presence/turn", json=_wire(_change(event)))
    assert answer.status_code == 200, answer.text
    assert answer.json()["outcome"] == "silent"
    assert _attention(root)[-1]["reason"] == "change_actor_unknown", "no reader: heard with its own reason"
    row, = [row.row for row in dialogue_store.iter_rows(root, "chat") if row.row.get("client_message_id") == "event1"]
    assert row["transport"]["message"] == STATE_CHANGE and row["transport"]["actor"] == UNNAMED  # kept whole
    # With a live author in the conversation: delivered as labelled background, never a wake.
    lease = PresenceTurnGate(state_root=root / "state").acquire(conversation_key("chat", "a", "r", ""))
    lease.bind_author("live-author")
    lease.intake_agent = SimpleNamespace(_owner_message_admission_lock=threading.RLock(), _accepting_owner_messages=True,
                                         _busy=True, _current_task_id="live-author")
    try:
        legacy = _change(replace(event, source_event_id="event2"), message={"previous_source_event_id": "event1"},
                         text="Title\n\nBody as it stands", facts=("any_message",))
        assert client.post("/presence/turn", json=_wire(legacy)).json()["outcome"] == "silent"
    finally:
        lease.release()
    path = _mailbox_path(root, "live-author")
    entry, = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert entry["place_arrival"]["wake"] is False and entry["provenance"] == "place_event"
    assert json.loads(entry["text"])["non_speech"] == "A change, not new words; who made it is unknown"
    rendered = []
    deliver_task_message(entry, "live-author", None, rendered.append)
    assert "never an owner directive" in rendered[0]
    assert _attention(root)[-1] == {**_attention(root)[-1], "reason": "current_author", "recipient": "live-author"}


def test_several_reading_works_never_turn_an_unattributed_change_into_a_focus(focus, monkeypatch):
    from ouroboros.place_arrivals import arrival_recipient
    from ouroboros.place_intake import ROUTE_TO_MIND, offer_arrival
    from ouroboros.presence_runner import capture_presence_event
    root, _place, admission, event = focus
    change = _change(event)
    captured = capture_presence_event(admission, change, drive_root=root)
    works = [{"task_id": "work-a", "task": {}}, {"task_id": "work-b", "task": {}}]
    monkeypatch.setattr("ouroboros.room_reader.conversation_works", lambda *_args, **_kw: works)
    monkeypatch.setattr("ouroboros.room_reader.work_reads", lambda *_args, **_kw: True)
    decision, details = decide_attention(root, admission, change)
    assert offer_arrival(root, captured, change, wake=author_hears(decision, details, change)) == ("", "")
    assert arrival_recipient(root, captured["id"]) == {}, "background pins no holder and asks no mind"
    assert offer_arrival(root, captured, change, wake=True) == (ROUTE_TO_MIND, "")  # what a wake would have done
