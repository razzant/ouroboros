"""A reaction is a retained non-speech fact: real ingress, attention, mail, history and previous-turn readers."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json
import threading

import pytest
from starlette.testclient import TestClient

from ouroboros import dialogue_store, places as p
from ouroboros.conversation_lease import PresenceTurnGate
from ouroboros.gateway import host_service
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.memory import Memory
from ouroboros.owner_mailbox import _mailbox_path, deliver_task_message
from ouroboros.place_arrivals import account_candidates
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_delivery import PresenceDeliveryRecorder
from ouroboros.presence_runner import (PresenceTurnEvent, _build_task, _read_previous_turn, capture_presence_event,
                                       current_previous_turn, run_presence_turn)
from ouroboros.task_results import write_task_result

THUMB = {"type": "name", "value": "thumbsup"}
HEART = {"type": "emoji", "value": "❤"}
USER = {"kind": "user", "platform_actor_id": "person", "display_name": "Ann"}
CHAT = {"kind": "chat", "platform_actor_id": "-100", "display_name": "News", "authenticated": False}
ANON = {"kind": "anonymous", "authenticated": False}


def fact(change="add", resolution="resolved", **extra):
    values = {"add": {"new": [THUMB]}, "remove": {"old": [THUMB]}, "replace": {"old": [THUMB], "new": [HEART]},
              "count": {"counts": [{**THUMB, "count": 3}]}}[change]
    return {"schema_version": 1, "change": change, **values,
            "target": {"message_id": "1712.0001", "resolution": resolution}, **extra}


def wire(event, actor=USER, reaction=None, *, facts=None, message=None, text="", thread_id=""):
    body = {k: v for k, v in asdict(event).items() if k not in {"delivery_reporting_version", "continuation_version", "identity"}}
    if facts is not None:
        body["facts"] = facts
    return {"event": {**body, "text": text, "actor": dict(actor), "thread_id": thread_id,
                      "message": message if message is not None else {"provider_raw": {"kept": True},
                                                                      "reaction": reaction or fact()}}}


@pytest.fixture
def room(tmp_path, monkeypatch):
    def connect(wakes=("any_message",)):
        place = p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r",
                                wakes=list(wakes))
        monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
        monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                            lambda _root: {"status": "available", "remaining_usd": 2, "settled_usd": 1})
        monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")
        async def authenticated(*_args):
            return "bridge", {}
        monkeypatch.setattr(host_service, "_authenticated", authenticated)
        calls = []
        def run(**kwargs):
            calls.append(kwargs["event"])
            return SimpleNamespace(outcome="silent", text="", task_id="focus", work_ref="")
        client = TestClient(host_service.create_host_service_app(tmp_path, presence_runner=run))
        event = PresenceTurnEvent("event1", "chat", "a", "r", "", "chat:a:r:0", dict(USER), {}, {}, "", facts=())
        return SimpleNamespace(root=tmp_path, place=place, client=client, calls=calls, event=event)
    return connect


def stored(root):
    return [entry.row for entry in dialogue_store.iter_rows(root, "chat")]


def attention(root):
    with dialogue_store.transaction(root, write=False) as conn:
        rows = conn.execute("SELECT dispatch_input FROM dialogue WHERE stream='chat' AND direction='in'").fetchall()
    return [json.loads(row["dispatch_input"] or "{}").get("place_attention") for row in rows]


def author_in(world, conversation_key="chat:a:r:0"):
    lease = PresenceTurnGate(state_root=world.root / "state").acquire(conversation_key)
    lease.bind_author("live-author")
    lease.intake_agent = SimpleNamespace(_owner_message_admission_lock=threading.RLock(), _accepting_owner_messages=True,
                                         _busy=True, _current_task_id="live-author")
    return lease


def mail(root):
    path = _mailbox_path(root, "live-author")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# --- Ingress and attention ---


@pytest.mark.parametrize("wakes,actor,facts", [
    (["any_message"], USER, []),
    (["any_message"], CHAT, []),
    (["any_message"], ANON, []),
    (["from_bot"], {**USER, "is_bot": True}, ["from_bot"]),
    (["any_message", "from_bot"], {**USER, "is_bot": True}, ["from_bot"]),
    (["mention"], USER, ["mention"]),  # a connector's claimed fact does not make a reaction a wake
    (["direct_message"], USER, ["direct_message"]),
])
def test_reaction_alone_starts_no_focus_whatever_the_wake_list(room, wakes, actor, facts):
    world = room(wakes)
    reaction = fact("count") if actor is ANON else fact()
    response = world.client.post("/presence/turn", json=wire(world.event, actor, reaction, facts=facts))
    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "silent" and not world.calls
    row, = stored(world.root)
    assert row["text"] == "" and row["transport"]["message"]["provider_raw"] == {"kept": True}
    assert attention(world.root)[0]["reason"] == "reaction_context_only"


def test_owner_reaction_is_context_too(room):
    world = room(["any_message"])
    p.edit_place(world.root, world.place.place_id, "confirm_owner_account", author="owner", platform_actor_id="person", account_id="a")
    assert world.client.post("/presence/turn", json=wire(world.event)).json()["outcome"] == "silent" and not world.calls


def test_exhausted_allowance_gives_a_reaction_no_retry(room, monkeypatch):
    world = room(["any_message"])
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "exhausted"})
    response = world.client.post("/presence/turn", json=wire(world.event))
    assert response.status_code == 200 and response.json()["outcome"] == "silent"


def test_paused_and_off_keep_their_own_reasons(room, monkeypatch):
    world = room(["any_message"])
    p.edit_place(world.root, world.place.place_id, "set_state", author="owner", state="paused")
    assert world.client.post("/presence/turn", json=wire(world.event)).json()["outcome"] == "silent"
    p.edit_place(world.root, world.place.place_id, "set_state", author="owner", state="active")
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: False)
    world.event = replace(world.event, source_event_id="event2")
    assert world.client.post("/presence/turn", json=wire(world.event)).json()["outcome"] == "silent"
    assert [row["reason"] for row in attention(world.root)] == ["paused", "off"]


@pytest.mark.parametrize("message", [
    {},  # a textless message
    {"message_id": "9", "file": "attached"},  # a textless attachment's message facts
    {"reaction": "thumbsup"},  # an old skill's word for its own marker
    {"reaction": ["👍"]},
    {"reaction": {"kind": "added", "name": "thumbsup"}},  # no schema_version: not this fact
    {"event_kind": "message_reaction", "old_reaction": [], "new_reaction": [{"type": "emoji", "emoji": "👍"}]},
])
def test_an_untyped_event_keeps_todays_wake_behaviour(room, message):
    world = room(["any_message"])
    response = world.client.post("/presence/turn", json=wire(world.event, {"platform_actor_id": "person"}, message=message))
    assert response.status_code == 200, response.text
    assert len(world.calls) == 1, "an old connector's event still wakes as it always did"


@pytest.mark.parametrize("change,actor,mutate", [
    ("add", USER, lambda f: f.update(schema_version=2)),
    ("add", USER, lambda f: f.update(schema_version="1")),
    ("add", USER, lambda f: f.update(schema_version=True)),
    ("add", USER, lambda f: f.update(change="noted")),
    ("add", USER, lambda f: f.update(extra=1)),
    ("add", USER, lambda f: f.update(new=[])),
    ("add", USER, lambda f: f.update(old=[THUMB])),
    ("add", USER, lambda f: f.update(new=[{"type": "name"}])),
    ("add", USER, lambda f: f.update(new=[{"type": "", "value": "x"}])),
    ("add", USER, lambda f: f.update(new=[{"type": "t", "value": 1}])),
    ("add", USER, lambda f: f.update(new=[{"type": "t", "value": "x", "extra": 1}])),
    ("add", USER, lambda f: f.update(new="thumbsup")),
    ("add", USER, lambda f: f.update(counts=[])),
    ("remove", USER, lambda f: f.update(old=[])),
    ("remove", USER, lambda f: f.update(new=[THUMB])),
    ("replace", USER, lambda f: f.update(new=[THUMB])),
    ("replace", USER, lambda f: f.update(old=[])),
    ("count", ANON, lambda f: f.update(old=[])),
    ("count", ANON, lambda f: f.update(counts=None)),
    ("count", ANON, lambda f: f.update(counts=[{**THUMB, "count": -1}])),
    ("count", ANON, lambda f: f.update(counts=[{**THUMB, "count": True}])),
    ("count", USER, lambda f: None),  # an aggregate has no individual
    ("add", ANON, lambda f: None),  # an individual change has one
    ("add", {"kind": "person", "platform_actor_id": "p"}, lambda f: None),
    ("add", {"platform_actor_id": "p"}, lambda f: None),
    ("add", {"kind": "user"}, lambda f: None),
    ("add", {"kind": "chat", "platform_actor_id": "-1"}, lambda f: None),  # authenticated must be false
    ("add", {**CHAT, "authenticated": True}, lambda f: None),
    ("add", {**CHAT, "is_bot": False}, lambda f: None),
    ("count", {"kind": "anonymous"}, lambda f: None),
    ("count", {**ANON, "platform_actor_id": "fabricated"}, lambda f: None),
    ("count", {**ANON, "id": "fabricated"}, lambda f: None),
    ("add", USER, lambda f: f.update(target=None)),
    ("add", USER, lambda f: f.update(target={"resolution": "resolved"})),
    ("add", USER, lambda f: f.update(target={"message_id": "", "resolution": "resolved"})),
    ("add", USER, lambda f: f.update(target={"message_id": 12, "resolution": "resolved"})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "guessed"})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "resolved", "thread_id": "t"})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "resolved", "author": "Ann"})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "resolved", "author": {"is_self": 1}})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "resolved", "excerpt": "words"})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "resolved", "excerpt": {"text": "w"}})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": "resolved",
                                             "excerpt": {"text": "x" * 501, "truncated": True}})),
])
def test_invalid_typed_fact_is_refused_before_capture(room, change, actor, mutate):
    world = room(["any_message"])
    reaction = fact(change)
    mutate(reaction)
    response = world.client.post("/presence/turn", json=wire(world.event, actor, reaction))
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "presence_reaction_invalid" and response.json()["disposition"] == "rejected"
    assert not stored(world.root) and not world.calls


def test_text_the_store_cannot_write_is_refused_as_an_invalid_fact(room):
    world = room(["any_message"])
    reaction = fact()
    reaction["new"] = [{"type": "name", "value": "\ud800"}]
    response = world.client.post("/presence/turn", content=json.dumps(wire(world.event, reaction=reaction)).encode(),
                                 headers={"content-type": "application/json"})
    assert response.json()["code"] == "presence_reaction_invalid" and not stored(world.root)


def test_a_reaction_carries_no_words_no_files_and_no_phantom_root(room):
    world = room(["any_message"])
    assert world.client.post("/presence/turn", json=wire(world.event, text="thumbs up!")).json()["code"] == "presence_reaction_invalid"
    assert world.client.post("/presence/turn", json={**wire(world.event), "staged_files": ["a.txt"]}).json()[
        "code"] == "presence_reaction_invalid"
    unresolved = wire(world.event, reaction=fact(resolution="unresolved"), thread_id="1712.0001")
    assert world.client.post("/presence/turn", json=unresolved).json()["code"] == "presence_reaction_invalid"
    assert not stored(world.root) and not world.calls


@pytest.mark.parametrize("change,actor", [("add", USER), ("remove", USER), ("replace", USER), ("add", CHAT), ("count", ANON)])
def test_every_valid_shape_is_accepted(room, change, actor):
    world = room(["any_message"])
    response = world.client.post("/presence/turn", json=wire(world.event, actor, fact(change)))
    assert response.status_code == 200, response.text
    assert not world.calls and stored(world.root)[0]["transport"]["message"]["reaction"]["change"] == change


def test_empty_aggregate_and_full_target_facts_are_valid(room):
    world = room(["any_message"])
    empty = {"schema_version": 1, "change": "count", "counts": [], "target": {"message_id": "1", "resolution": "unresolved"}}
    assert world.client.post("/presence/turn", json=wire(world.event, ANON, empty)).status_code == 200
    full = fact()
    full["target"] |= {"author": {"platform_actor_id": "U2", "display_name": "Bo", "is_self": True},
                       "excerpt": {"text": "x" * 500, "truncated": True}}
    world.event = replace(world.event, source_event_id="event2")
    assert world.client.post("/presence/turn", json=wire(world.event, reaction=full)).status_code == 200


def test_identity_does_not_claim_the_capability_before_shared_sleep_composition(room):
    world = room()
    assert "presence_reaction_context" not in world.client.get("/identity").json()


# --- A live author hears context, never a wake ---


def test_live_author_hears_a_resolved_reaction_as_labelled_background(room):
    world = room(["any_message"])
    lease = author_in(world)
    try:
        response = world.client.post("/presence/turn", json=wire(world.event))
    finally:
        lease.release()
    assert response.json()["outcome"] == "silent" and not world.calls
    entry, = mail(world.root)
    assert entry["place_arrival"]["wake"] is False and entry["provenance"] == "place_event"
    rendered = []
    deliver_task_message(entry, "live-author", None, rendered.append)
    assert "never an owner directive" in rendered[0]
    assert "Reaction, not speech: added thumbsup" in rendered[0]
    assert json.loads(entry["text"])["event"] == ""
    assert attention(world.root)[0]["reason"] == "current_author"


@pytest.mark.parametrize("actor,reaction,wakes", [
    (USER, fact(resolution="unresolved"), ["any_message"]),  # General's author is not the target's
    (ANON, fact("count", resolution="unresolved"), ["any_message"]),
    ({**USER, "is_bot": True}, fact(), ["any_message"]),  # the existing bot gate: no from_bot item
])
def test_unknown_conversation_or_unselected_bot_stays_room_only(room, actor, reaction, wakes):
    world = room(wakes)
    lease = author_in(world)
    try:
        response = world.client.post("/presence/turn", json=wire(world.event, actor, reaction))
    finally:
        lease.release()
    assert response.json()["outcome"] == "silent" and not mail(world.root) and not world.calls
    assert attention(world.root)[0]["reason"] == "reaction_context_only"


@pytest.mark.parametrize("actor", [{**USER, "is_bot": True}, CHAT])
def test_selected_bot_and_chat_actor_reactions_are_background_never_wake(room, actor):
    world = room(["from_bot"])
    lease = author_in(world)
    try:
        world.client.post("/presence/turn", json=wire(world.event, actor))
    finally:
        lease.release()
    entry, = mail(world.root)
    assert entry["place_arrival"]["wake"] is False and not world.calls


def test_only_the_target_conversations_author_hears_it(room):
    world = room(["any_message"])
    lease = author_in(world, "chat:a:r:other-thread")
    try:
        world.client.post("/presence/turn", json=wire(world.event))
    finally:
        lease.release()
    assert not mail(world.root)


def test_paused_reaction_reaches_no_author(room):
    world = room(["any_message"])
    p.edit_place(world.root, world.place.place_id, "set_state", author="owner", state="paused")
    lease = author_in(world)
    try:
        world.client.post("/presence/turn", json=wire(world.event))
    finally:
        lease.release()
    assert not mail(world.root) and attention(world.root)[0]["reason"] == "paused"


# --- History and model input ---


def history(world):
    from starlette.applications import Starlette
    from starlette.routing import Route
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(world.root))])
    chat_id = stored(world.root)[0]["chat_id"]
    with TestClient(app) as client:
        return client.get("/api/chat/history", params={"chat_id": chat_id}).json()["messages"]


def send_all(world):
    target = fact()
    target["target"] |= {"author": {"platform_actor_id": "U2", "display_name": "Bo"},
                         "excerpt": {"text": "the plan", "truncated": True}}
    rows = [("e1", USER, target), ("e2", USER, fact("remove")), ("e3", USER, fact("replace")),
            ("e4", CHAT, fact()), ("e5", ANON, fact("count")), ("e6", USER, fact(resolution="unresolved"))]
    for source, actor, reaction in rows:
        response = world.client.post("/presence/turn", json=wire(replace(world.event, source_event_id=source), actor, reaction))
        assert response.status_code == 200, response.text


def test_history_reader_shows_who_did_what_without_words(room):
    world = room(["any_message"])
    send_all(world)
    shown = history(world)
    assert len(shown) == 6
    assert all(row["text"] == "" and row["role"] == "user" for row in shown)
    summaries = [row["place_history"]["reaction"]["summary"] for row in shown]
    assert summaries[0] == 'added thumbsup on message 1712.0001 from Bo: "the plan" (excerpt, truncated)'
    assert summaries[1] == "removed thumbsup on message 1712.0001"
    assert summaries[2] == "changed thumbsup to ❤ on message 1712.0001"
    assert summaries[3] == "added thumbsup on message 1712.0001"
    assert summaries[4] == "counts observed: thumbsup 3 on message 1712.0001"
    assert summaries[5] == "added thumbsup on message 1712.0001; its conversation is not resolved"
    assert shown[3]["sender_label"] == "News"
    assert shown[4]["sender_label"] == "Anonymous reactions" and shown[4]["sender_session_id"] == ""
    assert all(row["place_history"]["attention"] == {"decision": "heard", "reason": "reaction_context_only"} for row in shown)


def test_model_readers_see_the_labelled_fact_and_attribution(room):
    from ouroboros.dialogue_provenance import dialogue_text, render_row_text, row_author, row_class
    from ouroboros.presence_continuation import _row_fact
    world = room(["any_message"])
    send_all(world)
    rows = stored(world.root)
    for row in rows:
        assert row["text"] == ""  # the store never holds invented words
        assert "[Reaction, not speech:" in dialogue_text(row) and "[Reaction, not speech:" in render_row_text(row)
    assert "added thumbsup on message 1712.0001 from Bo" in dialogue_text(rows[0])
    for row, kind, label in ((rows[0], "user", "Ann"), (rows[3], "chat", "News"), (rows[4], "anonymous", "Anonymous reactions")):
        author = row_author(row)
        assert author["kind"] == "place_fact" and author["actor_kind"] == kind and label in author["label"]
        assert row_class(row)["lane"] == 1  # shown as its fact wherever a room's rows are read, not folded into a task line
    assert "reaction" in _row_fact(SimpleNamespace(task_id="t"), rows[0]) and "not speech" in _row_fact(SimpleNamespace(task_id="t"), rows[0])
    text = Memory(drive_root=world.root).chat_history(count=20)
    assert "Reaction, not speech: counts observed: thumbsup 3" in text and "[Anonymous reactions" in text
    found = Memory(drive_root=world.root).chat_history(count=20, search="thumbsup")
    assert "removed thumbsup" in found


def test_reaction_actors_stay_out_of_owner_surfaces(room):
    world = room(["any_message"])
    # The owner's id on a chat actor must never confirm the owner, and only the individual is a candidate.
    p.edit_place(world.root, world.place.place_id, "confirm_owner_account", author="owner", platform_actor_id="-100", account_id="a")
    world.client.post("/presence/turn", json=wire(world.event, CHAT))
    world.client.post("/presence/turn", json=wire(replace(world.event, source_event_id="e2"), ANON, fact("count")))
    world.client.post("/presence/turn", json=wire(replace(world.event, source_event_id="e3"), USER))
    assert [person["platform_actor_id"] for person in account_candidates(world.root, world.place.place_id)] == ["person"]
    shown = history(world)
    assert [row["place_history"].get("owner_confirmed") for row in shown[:2]] == [False, False]


# --- Previous-turn action evidence ---


def turn(world, reports, *, source="turn1", outcome="silent", text="", one_part=False):
    event = PresenceTurnEvent(source, "chat", "a", "r", "", "chat:a:r:0", {"platform_actor_id": "person"}, {}, {}, "hello",
                              delivery_reporting_version=1, facts=("mention",))
    admission = admit_presence_turn(drive_root=world.root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    recorder = PresenceDeliveryRecorder(world.root)
    def handle(task):
        write_task_result(world.root, task["id"], "running", metadata=task["metadata"], chat_id=task["chat_id"])
        for index, (state, message, words) in enumerate(reports):
            recorder.record("bridge", {"schema_version": 1, "delivery_id": source if one_part else f"{source}-{index}", "part_id": "status" if not words else "0",
                                       "state": state, "provider": "chat", "account_id": "a", "conversation_id": "r",
                                       "thread_id": "", "text": words, "format": "", "message": message,
                                       "origin": {"kind": "tool", "task_id": task["id"]}})
        write_task_result(world.root, task["id"], "completed", result=text, terminal_origin="model_final", chat_id=task["chat_id"],
                          metadata={**task["metadata"], "presence_outcome": outcome, "presence_result_text": text})
        return [{"type": "presence_result", "task_id": task["id"], "outcome": outcome, "text": text}]
    run_presence_turn(admission=admission, event=event, repo_dir=world.root, drive_root=world.root,
                      agent_factory=lambda **_: SimpleNamespace(handle_task=handle))
    return event


def previous_line(world):
    from ouroboros.presence_context import _previous_turn_line
    previous, _ = current_previous_turn(world.root, "chat:a:r:0")
    return _previous_turn_line(previous), previous


def receipt(operation="set", new=(THUMB,), effect="applied", **extra):
    return {"kind": "operation", "reaction": {"schema_version": 1, "operation": operation, "new": list(new),
            "target": {"message_id": "77", "resolution": "resolved"}, **({"effect": effect} if effect else {}), **extra}}


@pytest.mark.parametrize("state,message,expected", [
    ("delivered", receipt(), "set reaction thumbsup, requested on message 77"),
    ("accepted", receipt(), "set reaction thumbsup, requested on message 77 (accepted by the provider)"),
    ("delivered", receipt(new=()), "cleared reactions, requested on message 77"),
    ("delivered", receipt("add"), "added reaction thumbsup, requested on message 77"),
    ("delivered", {"reaction": {"schema_version": 1, "operation": "remove", "old": [THUMB], "effect": "applied",
                                "target": {"message_id": "77", "resolution": "resolved"}}},
     "removed reaction thumbsup, requested on message 77"),
    ("delivered", receipt("add", effect="already_present"), "no change: reaction thumbsup was already present, requested on message 77"),
    ("delivered", {"reaction": {"schema_version": 1, "operation": "remove", "old": [THUMB], "effect": "already_absent",
                                "target": {"message_id": "77", "resolution": "resolved"}}},
     "no change: reaction thumbsup was already absent, requested on message 77"),
    ("failed", receipt(effect=None), "failed: set reaction thumbsup, requested on message 77"),
    ("uncertain", receipt(effect=None), "outcome unknown, may have landed, do not repeat: set reaction thumbsup, requested on message 77"),
])
def test_previous_turn_reports_real_reaction_receipts_without_speech(room, state, message, expected):
    world = room(["mention"])
    turn(world, [(state, message, "")])
    line, pointer = previous_line(world)
    assert expected in line and "nothing sent" not in line
    assert pointer["transport_sends"] == [] and pointer["message"] == ""


def test_a_later_receipt_settles_an_earlier_reaction_state(room):
    world = room(["mention"])
    turn(world, [("uncertain", receipt(effect=None), ""), ("delivered", receipt(), "")], one_part=True)
    line, pointer = previous_line(world)
    assert len(pointer["transport_actions"]) == 1 and pointer["transport_actions"][0]["state"] == "delivered"
    assert "set reaction thumbsup" in line and "outcome unknown" not in line


def test_reaction_evidence_keeps_real_speech_separate(room):
    world = room(["mention"])
    turn(world, [("delivered", receipt(), ""), ("delivered", {"message_id": "78"}, "Understood.")])
    line, pointer = previous_line(world)
    assert '"Understood."' in line and "set reaction thumbsup" in line
    assert pointer["transport_sends"] == ["Understood."]
    assert "thumbsup" not in json.dumps(pointer["transport_sends"])


def test_old_operation_receipts_and_silence_read_as_before(room):
    world = room(["mention"])
    turn(world, [("delivered", {"kind": "operation", "method": "setMessageReaction", "reaction": [{"type": "emoji", "emoji": "👍"}]}, "")])
    line, pointer = previous_line(world)
    assert "nothing sent" in line and "transport_actions" not in pointer
    turn(world, [], source="turn2")
    assert "transport_actions" not in _read_previous_turn(world.root, "chat:a:r:0")


def test_next_focus_input_carries_the_evidence(room):
    from ouroboros.presence_context import frame_focus_content
    world = room(["mention"])
    turn(world, [("delivered", receipt(), "")])
    admission = admit_presence_turn(drive_root=world.root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    event = PresenceTurnEvent("turn2", "chat", "a", "r", "", "chat:a:r:0", {"platform_actor_id": "person"}, {}, {}, "next", facts=("mention",))
    task = _build_task(admission, event, drive_root=world.root, staged_files=())
    framed = frame_focus_content(task, "next")
    assert "set reaction thumbsup, requested on message 77" in framed and "nothing sent" not in framed


def test_same_turn_send_facts_name_the_reaction(room):
    from ouroboros.presence_context import presence_send_facts
    world = room(["mention"])
    seen = {}
    event = PresenceTurnEvent("turn1", "chat", "a", "r", "", "chat:a:r:0", {"platform_actor_id": "person"}, {}, {}, "hello",
                              delivery_reporting_version=1, facts=("mention",))
    admission = admit_presence_turn(drive_root=world.root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    def handle(task):
        write_task_result(world.root, task["id"], "running", metadata=task["metadata"], chat_id=task["chat_id"])
        PresenceDeliveryRecorder(world.root).record("bridge", {"schema_version": 1, "delivery_id": "d", "part_id": "status",
            "state": "delivered", "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": "", "text": "",
            "format": "", "message": receipt(), "origin": {"kind": "tool", "task_id": task["id"]}})
        seen["facts"] = presence_send_facts(world.root, task["id"], task["metadata"]["presence"])
        write_task_result(world.root, task["id"], "completed", result="", terminal_origin="model_final", chat_id=task["chat_id"],
                          metadata={**task["metadata"], "presence_outcome": "silent", "presence_result_text": ""})
        return [{"type": "presence_result", "task_id": task["id"], "outcome": "silent", "text": ""}]
    run_presence_turn(admission=admission, event=event, repo_dir=world.root, drive_root=world.root,
                      agent_factory=lambda **_: SimpleNamespace(handle_task=handle))
    assert seen["facts"].startswith("none") and "set reaction thumbsup, requested on message 77" in seen["facts"]


def test_ordinary_capture_stays_free_of_reaction_fields(room):
    world = room(["mention"])
    admission = admit_presence_turn(drive_root=world.root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    event = replace(world.event, actor={"platform_actor_id": "p"}, text="hi")
    capture_presence_event(admission, event, drive_root=world.root)
    row, = stored(world.root)
    assert row["sender_label"] == "" and "reaction" not in row["transport"]["message"]


# --- The room's own reader of receipts, retries and an actual attachment ---


def test_receipts_reach_the_room_reader_labelled_and_wordless(room):
    world = room(["mention"])
    turn(world, [("delivered", receipt(), ""), ("failed", receipt(effect=None), ""),
                 ("delivered", {"kind": "operation", "method": "setMessageReaction", "reaction": [{"emoji": "👍"}]}, "")])
    rows = [row for row in history(world) if row["place_history"]["kind"] == "delivery"]
    assert [row["text"] for row in rows] == ["", "", ""]
    assert rows[0]["place_history"]["reaction"] == {"summary": "set reaction thumbsup, requested on message 77"}
    assert rows[0]["place_history"]["delivery_state"] == "delivered"
    assert rows[1]["place_history"]["delivery_state"] == "failed" and rows[1]["role"] == "system"
    assert "reaction" not in rows[2]["place_history"]  # the older shape stays as it was


def test_a_retried_reaction_is_one_row_and_one_silent_answer(room):
    world = room(["any_message"])
    for _ in range(2):
        assert world.client.post("/presence/turn", json=wire(world.event)).json()["outcome"] == "silent"
    assert len(stored(world.root)) == 1 and not world.calls


def test_a_textless_attachment_still_wakes_and_a_reaction_with_one_is_refused(room):
    world = room(["any_message"])
    state = world.root / "state" / "skills" / "bridge"
    state.mkdir(parents=True)
    (state / "note.txt").write_text("body", encoding="utf-8")
    body = wire(world.event, {"platform_actor_id": "person"}, message={"message_id": "9"})
    body["staged_files"] = [str(state / "note.txt")]
    assert world.client.post("/presence/turn", json=body).status_code == 200
    assert len(world.calls) == 1 and world.calls[0].text == ""
    refused = world.client.post("/presence/turn", json={**wire(replace(world.event, source_event_id="e2")),
                                                        "staged_files": [str(state / "note.txt")]})
    assert refused.json()["code"] == "presence_reaction_invalid"
    assert len(stored(world.root)) == 1, "a refused reaction is not captured"


def test_a_receipt_while_silent_names_the_gesture_in_its_existing_notice(room):
    world = room(["mention"])
    p.edit_place(world.root, world.place.place_id, "set_state", author="owner", state="silent")
    PresenceDeliveryRecorder(world.root).record("bridge", {"schema_version": 1, "delivery_id": "d", "part_id": "status",
        "state": "delivered", "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": "", "text": "",
        "format": "", "message": receipt(), "origin": {"kind": "tool"}})
    notice = next(row for row in stored(world.root) if row.get("type") == "place_notice")
    assert notice["text"].endswith("is Silent: set reaction thumbsup, requested on message 77")


def test_repair_and_a_yield_keep_the_evidence(room):
    from ouroboros import presence_runner as runner
    world = room(["mention"])
    turn(world, [("delivered", receipt(), "")])
    pointer = _read_previous_turn(world.root, "chat:a:r:0")
    task_id = pointer["task_id"]
    runner._previous_turn_path(world.root, "chat:a:r:0").unlink()
    runner._repair_pointer(world.root, "chat:a:r:0", task_id)
    assert _read_previous_turn(world.root, "chat:a:r:0")["transport_actions"] == pointer["transport_actions"]
    runner.record_continuing_turn(world.root, "chat:a:r:0", task_id, outcome="silent", message="", work_ref="",
                                  lent_at="2099-01-01T00:00:00+00:00")
    line, _ = previous_line(world)
    assert "set reaction thumbsup, requested on message 77" in line and "nothing sent" not in line


def test_memory_readers_show_a_chat_or_aggregate_reaction_as_its_fact(room):
    from ouroboros.dialogue_provenance import render_memory_row, row_class
    world = room(["any_message"])
    world.client.post("/presence/turn", json=wire(world.event, ANON, fact("count")))
    row, = stored(world.root)
    cls = row_class(row)
    text = render_memory_row({"chat_id": row["chat_id"], "ts": row["ts"], "row_sha256": "0" * 64}, row, author=cls["author"])
    assert "Anonymous reactions" in text and "[Reaction, not speech: counts observed: thumbsup 3" in text


def test_a_withheld_reaction_cannot_enter_as_an_imported_earlier_message(room):
    from ouroboros.presence_history import PresenceHistoryInvalid, parse_history_request
    world = room(["any_message"])
    event = wire(world.event, reaction=fact())["event"]
    event.pop("facts", None)
    record = {"source_timestamp": "2026-10-30T00:00:00+00:00", "authorship": "participant", "event": event}
    with pytest.raises(PresenceHistoryInvalid, match="not an earlier message"):
        parse_history_request({"import_id": "imp", "records": [record]})
    assert not stored(world.root)


def test_a_failed_noop_is_not_described_as_an_observation(room):
    from ouroboros.dialogue_provenance import reaction_action_words
    action = {"operation": "add", "new": [THUMB], "effect": "already_present", "target": {"message_id": "7"}}
    assert reaction_action_words(action, "delivered") == "no change: reaction thumbsup was already present, requested on message 7"
    assert reaction_action_words(action, "failed") == "failed: added reaction thumbsup, requested on message 7"


def test_receipt_evidence_keeps_only_its_declared_facts(room):
    world = room(["mention"])
    message = receipt()
    message["reaction"]["target"]["excerpt"] = {"text": "a receipt carries no target words", "truncated": False}
    message["reaction"]["junk"] = "x" * 100
    turn(world, [("delivered", message, "")])
    action, = _read_previous_turn(world.root, "chat:a:r:0")["transport_actions"]
    assert action == {"operation": "set", "new": [THUMB], "effect": "applied", "state": "delivered",
                      "target": {"message_id": "77", "resolution": "resolved"}}


def test_a_nonempty_row_with_a_typed_key_keeps_its_words(room):
    from ouroboros.dialogue_provenance import dialogue_text, row_author
    row = {"direction": "in", "text": "real words", "transport": {"message": {"reaction": fact()}}}
    assert dialogue_text(row) == "real words\n[Reaction, not speech: added thumbsup on message 1712.0001]"
    assert row_author(row)["kind"] == "human"


# --- Corrections: typed garbage is a refusal, and a receipt claims only what it states ---


@pytest.mark.parametrize("change,actor,mutate", [
    ("add", USER, lambda f: f.update(change=[])),
    ("add", USER, lambda f: f.update(change={})),
    ("add", {"kind": {}, "platform_actor_id": "p"}, lambda f: None),
    ("add", {"kind": [], "platform_actor_id": "p"}, lambda f: None),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": []})),
    ("add", USER, lambda f: f.update(target={"message_id": "1", "resolution": {}})),
    ("add", USER, lambda f: f.update(schema_version=[])),
])
def test_unhashable_json_values_are_a_400_not_a_500(room, change, actor, mutate):
    world = room(["any_message"])
    reaction = fact(change)
    mutate(reaction)
    response = world.client.post("/presence/turn", json=wire(world.event, actor, reaction))
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "presence_reaction_invalid" and not stored(world.root)


CLAIM = "set reaction thumbsup, requested on message 7"
TARGET = {"message_id": "7", "resolution": "resolved"}


@pytest.mark.parametrize("state,extra,expected", [
    ("delivered", {"effect": "applied"}, CLAIM),
    ("accepted", {"effect": "applied"}, CLAIM + " (accepted by the provider)"),
    ("delivered", {}, "effect not reported: " + CLAIM),
    ("accepted", {}, "effect not reported: " + CLAIM),
    ("delivered", {"effect": "maybe"}, "effect not reported: " + CLAIM),
    ("delivered", {"effect": []}, "effect not reported: " + CLAIM),
    ("delivered", {"effect": {"a": 1}}, "effect not reported: " + CLAIM),
    ("", {"effect": "applied"}, "state not recorded: " + CLAIM),
    ("mystery", {"effect": "applied"}, "state not recorded: " + CLAIM),
    ("failed", {"effect": "applied"}, "failed: " + CLAIM),
    ("failed", {"effect": "already_present"}, "failed: " + CLAIM),
    ("uncertain", {"effect": "applied"}, "outcome unknown, may have landed, do not repeat: " + CLAIM),
    ("uncertain", {}, "outcome unknown, may have landed, do not repeat: " + CLAIM),
])
def test_a_receipt_is_a_confirmed_gesture_only_when_it_says_so(state, extra, expected):
    from ouroboros.dialogue_provenance import reaction_action_words
    action = {"operation": "set", "new": [THUMB], "target": TARGET, **extra}
    assert reaction_action_words(action, state) == expected


@pytest.mark.parametrize("action", [
    {}, {"operation": []}, {"operation": "set", "new": "x", "target": "y"}, {"operation": {}, "old": [1], "new": [None]},
])
def test_malformed_receipt_facts_render_without_crashing_or_claiming(action):
    from ouroboros.dialogue_provenance import reaction_action_words
    words = reaction_action_words(action, "delivered")
    assert words.startswith("effect not reported:") and "added" not in words and "removed" not in words


@pytest.mark.parametrize("state,extra,expected", [
    ("failed", {}, "failed: set reaction thumbsup, requested on message 77"),
    ("uncertain", {}, "outcome unknown, may have landed, do not repeat: set reaction thumbsup, requested on message 77"),
    ("delivered", {}, "effect not reported: set reaction thumbsup, requested on message 77"),
    ("delivered", {"effect": "already_present"}, "no change: reaction thumbsup was already present, requested on message 77"),
])
def test_room_reader_and_model_readers_keep_the_receipts_state(room, state, extra, expected):
    from ouroboros.dialogue_provenance import dialogue_text, render_row_text
    world = room(["mention"])
    turn(world, [(state, {"reaction": {"schema_version": 1, "operation": "set", "new": [THUMB],
                                       "target": {"message_id": "77", "resolution": "resolved"}, **extra}}, "")])
    shown = next(row for row in history(world) if row["place_history"]["kind"] == "delivery")
    assert shown["place_history"]["reaction"]["summary"] == expected
    row = next(row for row in stored(world.root) if row.get("type") == "presence_delivery")
    for text in (dialogue_text(row), render_row_text(row)):
        assert text.startswith(f"[Reaction action, not speech: {expected}]")


@pytest.mark.parametrize("state,extra,expected", [
    ("accepted", {"effect": "applied"}, "set reaction thumbsup, requested on message 77 (accepted by the provider)"),
    ("delivered", {}, "effect not reported: set reaction thumbsup, requested on message 77"),
])
def test_the_late_notice_keeps_the_receipts_state(room, state, extra, expected):
    world = room(["mention"])
    p.edit_place(world.root, world.place.place_id, "set_state", author="owner", state="silent")
    PresenceDeliveryRecorder(world.root).record("bridge", {"schema_version": 1, "delivery_id": "d", "part_id": "status",
        "state": state, "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": "", "text": "", "format": "",
        "message": {"reaction": {"schema_version": 1, "operation": "set", "new": [THUMB],
                                 "target": {"message_id": "77", "resolution": "resolved"}, **extra}}, "origin": {"kind": "tool"}})
    notice = next(row for row in stored(world.root) if row.get("type") == "place_notice")
    assert notice["text"].endswith("is Silent: " + expected)


def test_reaction_receipts_are_not_counted_as_sent_messages(room):
    from ouroboros.presence_runner import _delivery_state, _live_task_rows, _turn_sends
    world = room(["mention"])
    turn(world, [("delivered", receipt(), ""), ("uncertain", receipt("add", effect=None), ""),
                 ("delivered", {"message_id": "8"}, "Words.")])
    task_id = _read_previous_turn(world.root, "chat:a:r:0")["task_id"]
    sent, uncertain = _turn_sends(_live_task_rows(world.root, task_id, "chat:a:r:0"))
    assert (sent, uncertain) == (["Words."], 0)
    assert _delivery_state(1, [], "", "silent") == "unknown"  # a wordless gesture confirms no speech delivery


def test_a_reaction_only_turn_reads_as_no_speech_delivery(room):
    world = room(["mention"])
    turn(world, [("delivered", receipt(), "")])
    line, pointer = previous_line(world)
    assert pointer["delivery"] == "unknown" and "set reaction thumbsup" in line
    assert "delivery confirmed" not in line


@pytest.mark.parametrize("state,expected", [
    ("delivered", "- reaction action for this turn, not speech: set reaction thumbsup, requested on message 77"),
    ("failed", "- reaction action for this turn, not speech: failed: set reaction thumbsup, requested on message 77"),
    ("uncertain", "- reaction action for this turn, not speech: outcome unknown, may have landed, do not repeat: set reaction thumbsup, requested on message 77"),
])
def test_review_wait_note_shows_a_reaction_receipt_as_an_action(room, state, expected):
    from ouroboros.presence_continuation import _row_fact
    world = room(["mention"])
    turn(world, [(state, receipt(effect="applied" if state == "delivered" else None), "")])
    row = next(row for row in stored(world.root) if row.get("type") == "presence_delivery")
    assert _row_fact(SimpleNamespace(task_id=row["task_id"]), row) == expected


def test_review_wait_note_shows_an_unreported_effect_and_a_noop(room):
    from ouroboros.presence_continuation import _row_fact
    world = room(["mention"])
    turn(world, [("delivered", receipt(effect=None), ""), ("delivered", receipt("add", effect="already_present"), "")])
    rows = [row for row in stored(world.root) if row.get("type") == "presence_delivery"]
    notes = [_row_fact(SimpleNamespace(task_id=row["task_id"]), row) for row in rows]
    assert notes[0].endswith("effect not reported: set reaction thumbsup, requested on message 77")
    assert notes[1].endswith("no change: reaction thumbsup was already present, requested on message 77")


# --- ``place_fact``: every author-kind reader, through the text-free projection ---


def reacted(world):
    send_all(world)
    turn(world, [("delivered", receipt(), "")], source="mine")
    return stored(world.root)


def test_the_text_free_projection_keeps_the_classification(room):
    from ouroboros.dialogue_provenance import row_author, row_class
    from ouroboros.memory_inventory import row_meta
    world = room(["any_message"])
    send_all(world)
    for row in stored(world.root):
        meta = row_meta(row)
        assert meta["text_chars"] == 0
        assert row_author(meta)["kind"] == "place_fact" and row_class(meta)["lane"] == 1
        assert row_author(meta)["actor_kind"] == row["transport"]["actor"]["kind"]


def entries_of(rows):
    from ouroboros.memory_inventory import row_meta
    return [({"chat_id": row["chat_id"], "ts": row["ts"], "row_sha256": f"{index:064d}"}, row_meta(row), index)
            for index, row in enumerate(rows)]


def test_reactions_are_not_people_replies_arrivals_or_answers(room):
    from ouroboros.dialogue_provenance import row_author
    from ouroboros.memory_inventory import place_activity, room_facts
    from ouroboros.memory_room_activity import unconsidered_people
    world = room(["any_message"])
    rows = reacted(world)
    entries = entries_of(rows)
    store = SimpleNamespace(acting_pages=lambda _room: [])
    facts = room_facts(store, "r", entries, {}, 0)
    words = sum(1 for row in rows if row.get("text"))
    assert facts["people"] == words  # only the turn's own inbound "hello"
    assert facts["mine"] == 0 and facts["reactions"] == 7
    assert unconsidered_people(entries, {}, -1) == words
    activity = place_activity(entries)
    assert activity["arrivals"] == 1 and activity["unanswered_events"] == 1 and activity["answered_events"] == 0
    assert row_author(next(row for row in rows if row.get("type") == "presence_delivery"))["actor_kind"] == "ouroboros"


def test_a_real_reply_still_answers_its_event_beside_a_reaction(room):
    from ouroboros.memory_inventory import place_activity
    world = room(["mention"])
    turn(world, [("delivered", receipt(), ""), ("delivered", {"message_id": "8"}, "Words.")])
    activity = place_activity(entries_of(stored(world.root)))
    assert activity["answered_events"] == 1 and activity["unanswered_events"] == 0


def test_a_reaction_cannot_be_quoted_as_anyones_words(room):
    from ouroboros.chronicle_quotes import SPEAKERS, quote_refusal
    from ouroboros.dialogue_provenance import render_row_text, row_author
    world = room(["any_message"])
    send_all(world)
    row = stored(world.root)[0]
    rendered, kind = render_row_text(row), row_author(row)["kind"]
    assert kind not in SPEAKERS
    for speaker in sorted(SPEAKERS):
        reason = quote_refusal([{"address": "a", "text": "added thumbsup", "speaker": speaker}], lambda _a: (rendered, kind))
        assert reason and reason[0] == "quote_mismatch" and f"spoken by {kind}" in reason[1]


def test_memory_view_shows_reactions_as_facts_and_never_counts_them_as_words(room):
    from ouroboros import memory_view as mv
    world = room(["any_message"])
    rows = reacted(world)
    chat_id = rows[0]["chat_id"]
    task = {"id": "focus1", "chat_id": 1, "metadata": {"usage_category": "consciousness", "wake_cause": "focus",
            "focus": {"room_chat_id": chat_id, "place_id": world.place.place_id, "conversation_key": "chat:a:r:0"}}}
    snapshot = mv.capture_memory_view(world.root, task, mv.view_spec_for_task(task, world.root))
    text = mv.render_room(snapshot)
    assert "[Reaction, not speech: added thumbsup on message 1712.0001 from Bo" in text
    assert "[Reaction action, not speech: set reaction thumbsup, requested on message 77]" in text
    lane1 = snapshot.room["lane1"]
    assert {item["kind"] for item in lane1 if not item["line"].count("hello")} == {"place_fact"}
    assert not any(item["owner_side"] for item in lane1 if item["kind"] == "place_fact")
    facts_line = mv._facts_text(snapshot.room["facts"], "this room")
    assert "reactions 7" in facts_line and "mine 0" in facts_line
    reaction = next(item for item in lane1 if item["kind"] == "place_fact")
    assert mv._row_pointer(reaction, "words").endswith("(reaction — read by address)")
    aggregated = mv._rows_line({"room_id": str(chat_id)}, lane1)
    assert "1 people's words" in aggregated and "7 reactions" in aggregated and "0 my replies" in aggregated


def test_memory_fallback_keeps_a_reaction_as_a_reaction(room):
    from ouroboros import memory_fallback as mf
    world = room(["any_message"])
    rows = stored_rows = reacted(world)
    reads = [({"chat_id": row["chat_id"], "ts": row["ts"], "row_sha256": f"{i:064d}"}, row) for i, row in enumerate(stored_rows)]
    items = mf._rows(world.root, reads, {f"{i:064d}": i for i in range(len(rows))})
    reaction = next(item for item in items if "Reaction, not speech" in item.line)
    assert "place_fact" in reaction.line and reaction.pointer.endswith("(reaction — read by address)")


# --- Words a receipt carries beside its reaction stay speech; a council snapshot keeps every reaction fact ---


def test_words_sent_beside_a_reaction_stay_speech_for_every_reader(room):
    from ouroboros.chronicle_quotes import quote_refusal
    from ouroboros.dialogue_provenance import dialogue_text, render_row_text, row_author
    from ouroboros.memory_inventory import place_activity, row_meta
    from ouroboros.presence_continuation import _row_fact
    world = room(["mention"])
    turn(world, [("delivered", receipt(), "On it.")])
    line, pointer = previous_line(world)
    assert pointer["transport_sends"] == ["On it."] and pointer["delivery"] == "confirmed"
    assert [action["operation"] for action in pointer["transport_actions"]] == ["set"]
    assert '"On it."' in line and "action, not speech: set reaction thumbsup" in line
    row = next(row for row in stored(world.root) if row.get("type") == "presence_delivery")
    assert row["text"] == "On it."
    assert row_author(row)["kind"] == row_author(row_meta(row))["kind"] == "ouroboros"
    for text in (dialogue_text(row), render_row_text(row)):
        assert text.startswith("On it.\n[Reaction action, not speech: set reaction thumbsup")
    assert quote_refusal([{"address": "a", "text": "On it.", "speaker": "ouroboros"}],
                         lambda _a: (render_row_text(row), row_author(row)["kind"])) is None
    assert _row_fact(SimpleNamespace(task_id=row["task_id"]), row).startswith(
        '- delivery delivered for this turn: "On it."; reaction action, not speech: set reaction thumbsup')
    assert place_activity(entries_of(stored(world.root)))["answered_events"] == 1
    shown = next(row for row in history(world) if row["place_history"]["kind"] == "delivery")
    assert shown["text"] == "On it." and "set reaction thumbsup" in shown["place_history"]["reaction"]["summary"]


def test_same_turn_send_facts_and_the_silent_notice_keep_words_beside_the_reaction(room):
    from ouroboros.presence_context import presence_send_facts
    from ouroboros.presence_runner import _live_task_rows, _turn_actions, _turn_sends
    world = room(["mention"])
    turn(world, [("delivered", receipt(), "On it."), ("uncertain", receipt("add", effect=None), "Maybe.")])
    task_id = _read_previous_turn(world.root, "chat:a:r:0")["task_id"]
    rows = _live_task_rows(world.root, task_id, "chat:a:r:0")
    assert _turn_sends(rows) == (["On it."], 1)  # the uncertain part's words may have landed
    assert [action["state"] for action in _turn_actions(rows)] == ["delivered", "uncertain"]
    facts = presence_send_facts(world.root, task_id, {"event": {"conversation_key": "chat:a:r:0"},
                                                      "delivery_reporting_version": 1})
    assert facts.startswith('"On it."; 1 more part(s)') and "action, not speech: set reaction thumbsup" in facts
    p.edit_place(world.root, world.place.place_id, "set_state", author="owner", state="silent")
    PresenceDeliveryRecorder(world.root).record("bridge", {"schema_version": 1, "delivery_id": "late", "part_id": "0",
        "state": "delivered", "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": "",
        "text": "Seen.", "format": "", "message": receipt(), "origin": {"kind": "tool"}})
    notice = next(row for row in stored(world.root) if row.get("type") == "place_notice")
    assert notice["text"].endswith("is Silent: Seen.; set reaction thumbsup, requested on message 77")


def test_a_council_room_snapshot_keeps_each_reactions_fact_and_invents_no_words(room):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.tools.plan_dialogue import dialogue_view
    world = room(["any_message"])
    send_all(world)
    turn(world, [("delivered", receipt(), ""), ("failed", receipt("add", effect=None), ""),
                 ("delivered", receipt("add"), "On it.")], source="mine")
    chat_id = stored(world.root)[0]["chat_id"]
    source = read_room_source(world.root, chat_id)
    projected = [row for row in source["rows"] if row.get("reaction")]
    retained = [row for row in stored(world.root) if (row.get("transport") or {}).get("message", {}).get("reaction")]
    assert len(projected) == len(retained) == 9
    assert [row["reaction"]["fact"] for row in projected] == [row["transport"]["message"]["reaction"] for row in retained]
    assert [row["text"] for row in projected] == [row["text"] for row in retained]  # nothing invented, nothing dropped
    assert [row["reaction"]["actor_kind"] for row in projected[:6]] == ["user", "user", "user", "chat", "anonymous", "user"]
    assert [(row["reaction"]["actor_kind"], row["reaction"]["state"]) for row in projected[6:]] == [
        ("ouroboros", "delivered"), ("ouroboros", "failed"), ("ouroboros", "delivered")]
    assert projected[0]["reaction"]["words"].startswith("[Reaction, not speech: added thumbsup on message 1712.0001 from Bo")
    assert projected[7]["reaction"]["words"].startswith("[Reaction action, not speech: failed: added reaction thumbsup")
    conversation, _snapshot, _facts = dialogue_view({**source, "lines": source["text"].count("\n")})
    assert "Anonymous reactions [provider=chat; account=a; conversation=r] · [Reaction, not speech: counts observed" in conversation
    assert "· On it.\n  [Reaction action, not speech: added reaction thumbsup, requested on message 77]" in conversation


# --- Exact quotes: a worded receipt's words are mine; the reaction and delivery labels beside them are the host's ---


LABEL = "set reaction thumbsup, requested on message 77"


def worded_receipt(world, source="turn1", reports=None):
    from ouroboros import chat_chain
    turn(world, reports or [("delivered", receipt(), "Noted, **thanks**.")], source=source)
    rows = [row for row in stored(world.root) if row.get("type") == "presence_delivery"]
    return rows[-1], chat_chain.format_address(chat_chain.row_address(rows[-1]))


def test_a_worded_receipts_words_are_quotable_as_mine_and_the_labels_beside_them_are_not(room):
    from ouroboros.dialogue_provenance import render_row_text
    from ouroboros.tools.chronicle import check_quotes
    world = room(["mention"])
    row, address = worded_receipt(world)
    shown = render_row_text(row)
    assert LABEL in shown and "[Delivery details:" in shown  # every reader still shows the act beside the words
    mine = {"address": address, "speaker": "ouroboros"}
    for words in ("Noted, **thanks**.", "Noted, thanks.", "thanks"):  # the whole message and any part of it
        assert check_quotes(world.root, [{**mine, "text": words}]) == (True, None), words
    for label in (LABEL, "[Reaction action, not speech:", "Delivery details", "kind operation; reaction effect applied",
                  "Noted, **thanks**.\n[Reaction action"):
        for speaker in ("ouroboros", "host", "unattributed"):
            assert check_quotes(world.root, [{**mine, "text": label, "speaker": speaker}]) == (False, 0), (label, speaker)


def test_quote_sources_of_other_rows_are_unchanged_and_a_wordless_reaction_has_none(room):
    from ouroboros.chronicle_quotes import SPEAKERS
    from ouroboros import chat_chain
    from ouroboros.tools.chronicle import check_quotes
    world = room(["any_message"])
    world.client.post("/presence/turn", json=wire(world.event))
    heard = stored(world.root)[0]
    plain, plain_address = worded_receipt(world, "turn2", [("delivered", {"message_id": "8"}, "Plain words.")])
    assert check_quotes(world.root, [{"address": plain_address, "speaker": "ouroboros", "text": "Plain words."}]) == (True, None)
    heard_address = chat_chain.format_address(chat_chain.row_address(heard))
    for speaker in sorted(SPEAKERS):
        assert check_quotes(world.root, [{"address": heard_address, "speaker": speaker, "text": "added thumbsup"}]) == (False, 0)


def marker(world):
    from ouroboros.tools.chronicle import _memory_mark
    from ouroboros.tools.registry import ToolContext
    ctx = ToolContext(repo_dir=world.root, drive_root=world.root, task_id="root0001", current_chat_id=1)
    return lambda **target: json.loads(_memory_mark(ctx, text="why it matters", **target))


def test_a_mark_quotes_a_receipts_words_and_never_its_labels(room):
    world = room(["mention"])
    row, address = worded_receipt(world)
    mark = marker(world)
    for target in ({"address": address}, {"task_id": row["task_id"]}):
        for label in (LABEL, "Delivery details"):
            assert mark(quote=label, **target)["reason"] == "quote_mismatch", (target, label)
        assert mark(quote="Noted, **thanks**.", **target)["ok"], target


def test_a_wordless_reaction_after_a_turns_words_is_not_its_final_words(room):
    world = room(["mention"])
    row, _address = worded_receipt(world, reports=[("delivered", {"message_id": "8"}, "Final words."),
                                                   ("delivered", receipt(), "")])
    mark = marker(world)
    assert mark(task_id=row["task_id"], quote="Final words.")["ok"]
    assert mark(task_id=row["task_id"], quote=LABEL)["reason"] == "quote_mismatch"
