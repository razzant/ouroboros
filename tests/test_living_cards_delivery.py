"""The supervisor half of ``show_card``: ``send_card`` → chat row → live frame → bus events → replay.

The row carries only the card's identity (the body stays in the card store); the live
``card`` frame and the history row carry the same ``card`` object; transports without
card rendering get the plain fallback on ``chat.outbound``; a replaced body is a
``card_state`` signal, never a second row.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import types
import typing

import pytest

from ouroboros import event_bus
from ouroboros.card_store import load_card, write_card
from ouroboros.gateway import card_contracts
from supervisor import message_bus
from supervisor.card_delivery import INTERACTIVE_NOTE
from supervisor.events_chat_delivery import _handle_send_card
from tests._dialogue import chat_rows
from tests.test_contracts import _assert_envelope_parity
from tests.test_gateway_parity import _js_typedef_fields

REPO = pathlib.Path(__file__).resolve().parents[1]
ACTION = {"type": "action", "label": "Book it", "submit": {"id": "book"}}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A real bridge over a temporary data root, with frames and bus events captured."""
    import supervisor.card_delivery as card_delivery

    bridge = message_bus.LocalChatBridge({})
    frames, events = [], []
    bridge._broadcast_fn = frames.append
    bridge._chat_transports[1] = {"kind": "telegram", "conversation_id": 777}
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(card_delivery, "publish_event", lambda topic, data: events.append((topic, data)))
    ctx = types.SimpleNamespace(bridge=bridge, DRIVE_ROOT=tmp_path, append_jsonl=lambda *a, **k: None)
    return ctx, frames, events


def _card(tmp_path, *, chat_id=1, components=(ACTION,)):
    return write_card(tmp_path, chat_id=chat_id, task_id="task-1", title="Flights", fallback_text="A 9:00 or B 12:00",
                      kind="components", body={"components": list(components)})


def _rows(tmp_path):
    return chat_rows(tmp_path)


def _event(card, **extra):
    return {"type": "send_card", "chat_id": card["chat_id"], "card_id": card["card_id"], "title": card["title"],
            "fallback_text": card["fallback_text"], "kind": card["kind"], "revision": card["revision"],
            "replace": False, "interactive": True, "task_id": "task-1", "parent_task_id": "", "root_task_id": "",
            **extra}


def test_a_card_becomes_one_row_one_frame_and_two_bus_events(tmp_path, wired):
    ctx, frames, events = wired
    card = _card(tmp_path)
    _handle_send_card(_event(card), ctx)

    [row] = _rows(tmp_path)
    identity = {"card_id": card["card_id"], "kind": "components", "title": "Flights", "revision": 1}
    assert row["type"] == "card" and row["direction"] == "out" and row["task_id"] == "task-1"
    assert row["text"] == "A 9:00 or B 12:00" and row["card"] == identity
    assert "Book it" not in json.dumps(row) and "submit" not in json.dumps(row)  # the body stays in the store
    [frame] = [f for f in frames if f.get("type") == "card"]
    assert frame["card"] == identity and frame["fallback_text"] == row["text"] and frame["ts"] == row["ts"]
    assert frame["role"] == "assistant" and frame["chat_id"] == 1 and frame["row_address"]
    topics = [topic for topic, _ in events]
    assert topics == [event_bus.CHAT_CARD, event_bus.CHAT_OUTBOUND]
    card_event, outbound = events[0][1], events[1][1]
    assert card_event["card_id"] == card["card_id"] and card_event["interactive"] is True
    assert outbound["text"] == f"A 9:00 or B 12:00\n\n{INTERACTIVE_NOTE}" and outbound["card_id"] == card["card_id"]
    assert outbound["transport"] == {"kind": "telegram", "conversation_id": 777}
    assert load_card(tmp_path, card["card_id"])["publication_chat_id"] == 1


def test_telegram_addresses_the_fallback_through_its_existing_outbound_subscription(tmp_path, wired):
    from skills.telegram import plugin as telegram_plugin

    ctx, _frames, events = wired
    card = _card(tmp_path, components=[{"type": "markdown", "text": "fyi"}])
    _handle_send_card(_event(card, interactive=False), ctx)
    outbound = dict(events)[event_bus.CHAT_OUTBOUND]
    assert outbound["text"] == "A 9:00 or B 12:00"  # display-only: no "open the app" note
    assert telegram_plugin._target_chat({"TELEGRAM_CHAT_ID": ""}, outbound) == 777
    manifest = (REPO / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8")
    assert "chat.outbound" in re.search(r"subscribe_events: \[(.*)\]", manifest).group(1)


def test_history_replays_the_card_row_with_its_identity(tmp_path, wired):
    from ouroboros.gateway.history import make_chat_history_endpoint

    ctx, _frames, _events = wired
    card = _card(tmp_path)
    _handle_send_card(_event(card), ctx)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(types.SimpleNamespace(query_params={"limit": "10"})))
    [rec] = [m for m in json.loads(response.body)["messages"] if m.get("msg_type") == "card"]
    assert rec["card"] == {"card_id": card["card_id"], "kind": "components", "title": "Flights", "revision": 1}
    assert rec["text"] == "A 9:00 or B 12:00" and rec["system_type"] == "card" and rec["role"] == "assistant"


def test_the_delivery_projection_keeps_its_earlier_kinds(tmp_path):
    """The typed delivery branches moved out of gateway/history.py unchanged."""
    from ouroboros.gateway.history_deliveries import delivery_fields

    assert delivery_fields({"type": "document", "filename": "r.pdf", "size_bytes": "12"}) == {
        "msg_type": "document", "filename": "r.pdf", "mime": "application/octet-stream", "download_url": "",
        "download_url_compat": "", "caption": "", "size_bytes": 12}
    assert delivery_fields({"type": "photo"}) == {}  # a photo without a stored URL stays a text row
    assert delivery_fields({"type": "links", "actions": [{"label": "a", "url": "https://a"}]})["msg_type"] == "links"
    assert delivery_fields({"type": "card"}) == {} and delivery_fields({"type": "quiz", "quiz": {}}) == {}


def test_a_replaced_card_is_a_state_signal_not_a_second_row(tmp_path, wired):
    ctx, frames, events = wired
    card = _card(tmp_path)
    _handle_send_card(_event(card), ctx)
    newer = write_card(tmp_path, chat_id=1, task_id="task-1", title="Flights (booked)", fallback_text="Booked A",
                       kind="components", body={"components": [{"type": "callout", "text": "Booked"}]},
                       replace=card["card_id"])
    frames.clear()
    events.clear()
    _handle_send_card(_event(newer, replace=True), ctx)
    assert len(_rows(tmp_path)) == 1 and events == []
    [state] = frames
    assert state["type"] == "card_state" and state["card"]["revision"] == 2 and state["card"]["title"] == "Flights (booked)"
    assert state["submissions_summary"] == [] and state["submissions_dropped"] == 0 and state["chat_id"] == 1


def test_machine_chats_and_unknown_cards_publish_nothing(tmp_path, wired):
    ctx, frames, events = wired
    machine = _card(tmp_path, chat_id=-5)
    _handle_send_card(_event(machine), ctx)
    _handle_send_card(_event(machine, card_id="card_0000000000000000"), ctx)
    assert _rows(tmp_path) == [] and frames == [] and events == []


def test_card_frames_match_their_published_envelopes():
    for discriminator, cls, required in (
        ("card", card_contracts.CardOutbound, {"type", "role", "card", "fallback_text", "ts"}),
        ("card_state", card_contracts.CardStateOutbound,
         {"type", "card", "submissions_summary", "submissions_dropped", "ts"}),
    ):
        _assert_envelope_parity(REPO / "supervisor" / "card_delivery.py", discriminator=discriminator,
                                declared_keys=set(typing.get_type_hints(cls)), required_keys=frozenset(required),
                                envelope_name=cls.__name__)


def test_every_card_envelope_has_a_field_identical_browser_twin():
    text = (REPO / "web" / "modules" / "client_contract_types.js").read_text(encoding="utf-8")
    assert {"CardOutbound", "CardStateOutbound", "CardResponse", "CardPinResponse"} <= set(card_contracts.__all__)
    for name in card_contracts.__all__:
        cls = getattr(card_contracts, name)
        assert re.search(rf"@typedef \{{Object\}} {name}\b", text), f"client_contract_types.js lacks {name}"
        assert _js_typedef_fields(text, name) == set(typing.get_type_hints(cls, include_extras=True)), name


def test_the_card_frames_are_indexed_and_the_event_has_a_handler():
    from ouroboros.gateway import endpoint_index
    from ouroboros.gateway.contracts import WS_MESSAGE_TYPES
    from supervisor.event_taxonomy import EVENT_DISPOSITIONS
    from supervisor.events_chat_delivery import EVENT_HANDLERS

    assert {"card", "card_state"} <= set(WS_MESSAGE_TYPES)
    assert endpoint_index.WS_SHAPES["card"] == ("", "CardOutbound")
    assert EVENT_HANDLERS["send_card"] is _handle_send_card and "send_card" in EVENT_DISPOSITIONS
    assert event_bus.CHAT_CARD in event_bus.VALID_TOPICS


def test_the_model_reads_a_card_row_as_its_fallback_with_the_card_id():
    from ouroboros.dialogue_provenance import render_row_text, row_class

    row = {"type": "card", "direction": "out", "text": "A 9:00 or B 12:00",
           "card": {"card_id": "card_0123456789abcdef", "kind": "components", "title": "Flights", "revision": 1}}
    assert render_row_text(row) == "[card card_0123456789abcdef] A 9:00 or B 12:00"
    assert render_row_text({**row, "type": "links"}) == "A 9:00 or B 12:00"
    assert row_class(row)["lane"] == 1  # Ouroboros's own words to the owner, not a host fact


def test_a_card_is_a_standalone_message_for_unread_and_a_replace_is_not(tmp_path, wired):
    """The row and the room's unread move in one store transaction (``log_chat(feed=True)``)."""
    from ouroboros.projects_registry import create_project, get_project

    ctx, _frames, _events = wired
    project = create_project(tmp_path, "trips")
    chat_id = int(project["chat_id"])

    def revision() -> int:
        return int(get_project(tmp_path, "trips")["visible_revision"])

    before = revision()
    card = _card(tmp_path, chat_id=chat_id)
    _handle_send_card(_event(card), ctx)
    assert revision() - before == 1 and [row["type"] for row in _rows(tmp_path)] == ["card"]
    newer = write_card(tmp_path, chat_id=chat_id, task_id="task-1", title="Flights (booked)", fallback_text="Booked A",
                       kind="components", body={"components": [{"type": "callout", "text": "Booked"}]},
                       replace=card["card_id"])
    _handle_send_card(_event(newer, replace=True), ctx)
    assert revision() - before == 1, "a replaced body is a state signal, not a new unread message"
