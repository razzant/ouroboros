"""Captured facts retain their exact source and their existing room lifetime."""
from dataclasses import replace
import json

import pytest

from ouroboros import dialogue_store, place_arrivals, presence_runner as runner, store
from tests.test_place_focus import focus as focus


@pytest.mark.parametrize("identity", [None, {}, {
    "sends_as": {"id": "writer", "label": "Writer café", "kind": "user"},
    "reads_as": [{"id": "reader", "label": "Reader", "kind": "user"}],
}])
def test_capture_identity_and_unchanged_legacy_row_bytes(focus, monkeypatch, identity):
    root, place, admission, event = focus
    # Works before and after F1A adds the optional frozen-dataclass field.
    object.__setattr__(event, "identity", identity)
    monkeypatch.setattr(runner, "utc_now_iso", lambda: "2026-10-26T12:00:00+00:00")
    monkeypatch.setattr(runner, "read_json_dict", lambda _: {"session_id": "fixed-session"})
    task = runner.capture_presence_event(admission, event, drive_root=root)
    expected = {
        "ts": "2026-10-26T12:00:00+00:00", "session_id": "fixed-session",
        "direction": "in", "type": "place_arrival", "chat_id": task["chat_id"],
        "user_id": runner._stable_numeric_id("presence-actor-log", str(task["actor_id"])),
        "text": "hello", "format": "", "source": "presence:chat",
        "sender_label": "", "sender_session_id": "person", "client_message_id": "event1",
        "presence_event_identity": task["metadata"]["presence_event_identity"],
        "transport": {"provider": "chat", "account_id": "a", "conversation_id": "r",
                      "thread_id": "", "conversation_key": "chat:a:r",
                      "actor": {"platform_actor_id": "person"}, "conversation": {}, "message": {}},
        "presence_provenance": {"place_id": place.place_id, "binding_id": place.place_id,
                                "transport_skill": "bridge", "behavior_skill": "",
                                "profile_fingerprint": "", "state_fingerprint": "", "selection_fingerprint": "",
                                "source_event_id": "event1", "conversation_key": "chat:a:r",
                                "provider": "chat", "account_id": "a", "conversation_id": "r",
                                "thread_id": "", "actor_id": "person"},
        "task_id": task["id"], "place_id": place.place_id,
    }
    if identity is not None:
        expected["transport"]["identity"] = identity
    with dialogue_store.transaction(root, write=False) as conn:
        row, = conn.execute("SELECT body FROM dialogue WHERE stream='chat'").fetchall()
    assert row["body"] == json.dumps(expected, ensure_ascii=False)


@pytest.mark.parametrize("declared", [None, (), ("mention", "signal.words")])
def test_heard_retains_declared_facts_without_inference_or_source_rewrite(focus, declared):
    root, _, admission, event = focus
    event = replace(event, facts=declared, actor={**event.actor, "is_bot": True})
    task = runner.capture_presence_event(admission, event, drive_root=root)
    source, = dialogue_store.iter_rows(root, "chat")
    result = place_arrivals.record_heard(root, task, event, "no_wake_match")
    with dialogue_store.transaction(root, write=False) as conn:
        row, = conn.execute("SELECT body,dispatch_input FROM dialogue WHERE stream='chat'").fetchall()
    assert json.loads(row["dispatch_input"])["place_attention"] == {
        "decision": "heard", "reason": "no_wake_match", "delivery_reporting_version": 0,
        "facts": list(declared or ()),
    }
    assert json.loads(row["body"]) == source.row
    assert place_arrivals.heard_replay(root, task["id"]) == result


def test_old_heard_decision_replays_without_backfilling_facts(focus):
    root, _, admission, event = focus
    task = runner.capture_presence_event(admission, event, drive_root=root)
    legacy = {"place_attention": {"decision": "heard", "reason": "off", "delivery_reporting_version": 0}}
    with dialogue_store.transaction(root, write=True) as conn:
        conn.execute("UPDATE dialogue SET dispatch_input=? WHERE task_id=?", (json.dumps(legacy), task["id"]))
    before = place_arrivals.heard_replay(root, task["id"])
    assert before.outcome == "silent"
    assert place_arrivals.record_heard(root, task, event, "paused") == before
    with dialogue_store.transaction(root, write=False) as conn:
        row, = conn.execute("SELECT dispatch_input FROM dialogue WHERE stream='chat'").fetchall()
    assert json.loads(row["dispatch_input"]) == legacy


def test_focus_source_retains_no_attention_facts(focus):
    root, _, admission, event = focus
    task = runner.capture_presence_event(admission, event, drive_root=root)
    with place_arrivals.intake_lock(root, event.conversation_key):
        place_arrivals.retain_focus_source(root, task["id"])
    with dialogue_store.transaction(root, write=False) as conn:
        row, = conn.execute("SELECT dispatch_input FROM dialogue WHERE stream='chat'").fetchall()
    assert set(json.loads(row["dispatch_input"])) == {"place_context_source_ref"}


def seed_identity_arrival(root, place, identity, *, ts="2026-10-26T12:00:00Z", account_id=None, **fields):
    """Stored arrival shape; identity may deliberately model a malformed historical value; account_id = another
    account of a connector-wide place."""
    from ouroboros.places import conversation_chat_id
    from ouroboros.presence_bindings import conversation_key
    key = conversation_key(place.provider, account_id or place.account_id, place.conversation_id, "")
    transport = {"provider": place.provider, "account_id": account_id or place.account_id,
                 "conversation_id": place.conversation_id, "thread_id": "", "conversation_key": key,
                 "actor": {"platform_actor_id": "person"}, "conversation": {}, "message": {}}
    if identity is not None:
        transport["identity"] = identity
    return dialogue_store.append(root, "chat", {
        "ts": ts, "type": "place_arrival", "direction": "in", "chat_id": conversation_chat_id(key),
        "place_id": place.place_id, "source": "presence:" + place.provider, "text": "Captured message",
        "sender_session_id": "person", "sender_label": "Person", "transport": transport, **fields,
    })


def test_recent_identity_uses_newest_arrival_of_this_place(focus):
    root, place, _, _ = focus
    older = {"sends_as": {"id": "old", "label": "Old", "kind": "user"}}
    newest = {"sends_as": {"id": "new", "label": "New", "kind": "bot"}}
    seed_identity_arrival(root, place, newest, ts="2026-10-26T12:00:02Z")
    seed_identity_arrival(root, place, older, ts="2026-10-26T12:00:01Z")
    seed_identity_arrival(root, replace(place, place_id="another-place"), older, ts="2026-10-26T12:00:03Z")
    seed_identity_arrival(root, place, older, ts="2026-10-26T12:00:04Z", type="presence_delivery", direction="out")
    assert place_arrivals.recent_identity(root, place.place_id) == newest


@pytest.mark.parametrize("bad", [None, [], "broken", 1, True, 1.5, {"sends_as": {"id": "missing-fields"}},
                                  {"sends_as": {"id": "a", "label": "A", "kind": "user", "owner": True}},
                                  json.dumps({"sends_as": {"id": "string-not-object", "label": "S", "kind": "user"}})])
def test_recent_identity_skips_malformed_or_absent_stored_identity(focus, bad):
    root, place, _, _ = focus
    good = {"sends_as": {"id": "writer", "label": "Writer", "kind": "user"}}
    seed_identity_arrival(root, place, good, ts="2026-10-26T12:00:01Z")
    seed_identity_arrival(root, place, bad, ts="2026-10-26T12:00:02Z")
    assert place_arrivals.recent_identity(root, place.place_id) == good


def test_recent_identity_empty_declaration_supersedes_older_identity(focus):
    root, place, _, _ = focus
    seed_identity_arrival(root, place, {"reads_as": [{"id": "reader", "label": "Reader", "kind": "user"}]},
                          ts="2026-10-26T12:00:01Z")
    seed_identity_arrival(root, place, {}, ts="2026-10-26T12:00:02Z")
    assert place_arrivals.recent_identity(root, place.place_id) == {}


def test_recent_identity_without_arrivals_is_none(focus):
    root, place, _, _ = focus
    assert place_arrivals.recent_identity(root, place.place_id) is None
    seed_identity_arrival(root, replace(place, place_id="another-place"), {})
    assert place_arrivals.recent_identity(root, place.place_id) is None


def test_recent_identity_of_a_named_account_stays_inside_the_places_window(focus, monkeypatch):
    """A named account is picked out of the place's newest 50 arrivals, never read further back."""
    from tests.place_history_scenario import watch_transactions
    root, place, _, _ = focus
    mine = {"sends_as": {"id": "mine", "label": "Mine", "kind": "user"}}
    other = {"sends_as": {"id": "other", "label": "Other", "kind": "user"}}
    seed_identity_arrival(root, place, mine, ts="2026-10-26T12:00:00Z", account_id="account")
    seed_identity_arrival(root, place, other, ts="2026-10-26T12:00:01Z", account_id="elsewhere")
    assert place_arrivals.recent_identity(root, place.place_id, "account") == mine
    assert place_arrivals.recent_identity(root, place.place_id, "elsewhere") == other
    assert place_arrivals.recent_identity(root, place.place_id) == other
    assert place_arrivals.recent_identity(root, place.place_id, "nobody") is None
    for number in range(50):  # the other account fills the window: the quiet account's declaration is behind it
        seed_identity_arrival(root, place, other, ts=f"2026-10-26T13:00:{number:02d}Z", account_id="elsewhere")
    watch = watch_transactions(monkeypatch)
    assert place_arrivals.recent_identity(root, place.place_id, "account") is None
    selects = [sql for sql in watch.statements if "json_extract(body,'$.transport.identity')" in sql]
    assert len(selects) == 1 and "LIMIT 50" in selects[0] and "account_id')=?" not in selects[0]


@pytest.mark.parametrize("failure", [store.StoreBusy("held"), store.StoreLost("damaged"), OSError("gone")])
def test_recent_identity_is_unknown_when_the_room_cannot_be_read(focus, monkeypatch, caplog, failure):
    from contextlib import contextmanager
    root, place, _, _ = focus
    seed_identity_arrival(root, place, {"sends_as": {"id": "writer", "label": "Writer", "kind": "user"}})
    assert place_arrivals.recent_identity(root, place.place_id)["sends_as"]["id"] == "writer"

    @contextmanager
    def failing(*args, **kwargs):
        raise failure
        yield  # pragma: no cover
    monkeypatch.setattr(place_arrivals.dialogue_store, "transaction", failing)
    with caplog.at_level("WARNING"):
        assert place_arrivals.recent_identity(root, place.place_id) is None
    assert "Captured identity unavailable" in caplog.text


def test_recent_identity_bounds_indexed_read_and_validates_after_transaction(focus, monkeypatch):
    from ouroboros import places
    from tests.place_history_scenario import watch_transactions
    root, place, _, _ = focus
    seed_identity_arrival(root, place, {"reads_as": []}, ts="2026-10-25T12:00:00Z")
    for number in range(50):  # declared objects that fail validation: each row reaches the validator, none passes
        seed_identity_arrival(root, place, {"sends_as": {"id": "missing-fields"}}, ts=f"2026-10-26T12:00:{number:02d}Z")
    watch = watch_transactions(monkeypatch)
    validate, calls = places.validate_presence_identity, []
    def after_transaction(value):
        assert not watch.open
        calls.append(value)
        return validate(value)
    monkeypatch.setattr(places, "validate_presence_identity", after_transaction)
    assert place_arrivals.recent_identity(root, place.place_id) is None
    assert len(calls) == 50
    selects = [sql for sql in watch.statements if sql.startswith("SELECT json_type(body,'$.transport.identity'), json_extract(body,'$.transport.identity'), json_extract(body,'$.transport.account_id') FROM dialogue")]
    assert len(selects) == 1
    assert "INDEXED BY dialogue_place_recent" in selects[0] and "LIMIT 50" in selects[0]
