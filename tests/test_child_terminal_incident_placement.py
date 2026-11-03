"""A subagent's host-salvage receipt is a row of that child's card.

The producer names the placement (``card_row`` with a stable ``card_row_id``) on
the ``terminal_incident`` row of a task whose lineage says ``subagent``; a root's
receipt stays the ordinary System row. The placement rides ``progress_meta``, so
it reaches the stored chat row, history replay, the live frame and the owed
outbox row through the plumbing every other card row already uses.
"""

from tests import _dialogue as dialogue_test

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as delivery, message_bus
from supervisor.terminal_delivery import (
    delivery_id_for,
    pending_deliveries,
    project_terminal_result_event,
    register_pending_delivery,
    replay_pending_deliveries,
)
from tests._store_shared import seed_result


RAW = "RAW INTERMEDIATE OUTPUT " * 40
LINEAGE = {"delegation_role": "subagent", "parent_task_id": "root-1", "root_task_id": "root-1",
           "subagent_role": "researcher"}
CHILD_TASK = {"id": "kid-1", "chat_id": 1, **LINEAGE}
ROW_ID = delivery_id_for("kid-1", RAW) + ":terminal_incident"


def _project(task, *, origin="host_salvage", meta=None, task_id="kid-1"):
    base = {"type": "send_message", "chat_id": 1, "task_id": task_id, "text": RAW,
            "log_text": RAW, "format": "markdown"}
    if meta is not None:
        base["progress_meta"] = dict(meta)
    return project_terminal_result_event(
        "unused", task, task_id, result_text=RAW, terminal_origin=origin, base_event=base,
    )


def test_subagent_salvage_row_names_its_card_placement():
    event = _project(CHILD_TASK)
    assert event["system_type"] == "terminal_incident" and event["role"] == "system"
    assert event["progress_meta"] == {"card_row": "timeline", "card_row_id": ROW_ID}
    assert len(ROW_ID) <= 200, "the stored row keeps an id only up to 200 characters"


def test_lineage_carried_by_the_send_event_is_enough_and_is_kept():
    """The worker's final frame already carries the child lineage in its meta;
    the stamp joins those facts instead of replacing them."""
    meta = {"subagent_task_id": "kid-1", **LINEAGE}
    event = _project({"id": "kid-1", "chat_id": 1}, meta=meta)
    assert event["progress_meta"] == {**meta, "card_row": "timeline", "card_row_id": ROW_ID}


@pytest.mark.parametrize("task", [
    {"id": "root-1", "chat_id": 1},
    {"id": "root-1", "chat_id": 1, "delegation_role": "", "root_task_id": "root-1"},
    None,
])
def test_root_salvage_row_carries_no_placement(task):
    event = _project(task, task_id="root-1")
    assert event["system_type"] == "terminal_incident"
    assert "progress_meta" not in event


@pytest.mark.parametrize("origin", ["host_notice", "model_final", None])
def test_only_the_salvage_receipt_of_a_subagent_is_placed(origin):
    event = _project(CHILD_TASK, origin=origin)
    assert "card_row" not in (event.get("progress_meta") or {})
    assert "system_type" not in event


def _wire(tmp_path, monkeypatch):
    bridge = message_bus.LocalChatBridge({})
    frames, published = [], []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda _topic, event: published.append(dict(event)))
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                           send_with_budget=message_bus.send_with_budget)
    return host, frames, published


def _stored_rows(tmp_path):
    lines = dialogue_test.log_text(tmp_path / "logs" / "chat.jsonl").splitlines()
    return [json.loads(line) for line in lines]


def _replayed_rows(tmp_path):
    response = asyncio.run(
        make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"}))
    )
    return json.loads(response.body)["messages"]


def _worker_final(tmp_path, task, **usage):
    """The worker's own terminal frame, as ``emit_task_results`` builds it."""
    from ouroboros.subagent_messages import subagent_message_meta
    from ouroboros.task_finalization import prepare_terminal_send_event

    send_event = {"type": "send_message", "chat_id": task["chat_id"], "text": RAW, "log_text": RAW,
                  "format": "markdown", "task_id": task["id"], "ts": "2026-09-20T00:00:00Z"}
    meta = subagent_message_meta(task, task_id=task["id"])
    if meta:
        send_event["progress_meta"] = dict(meta)
    return prepare_terminal_send_event(
        tmp_path, task, RAW, {"terminal_origin": "host_salvage", **usage}, send_event, presence=False,
    )


def test_placement_reaches_the_stored_row_history_the_live_frame_and_the_skill_event(tmp_path, monkeypatch):
    host, frames, published = _wire(tmp_path, monkeypatch)
    event = _worker_final(tmp_path, CHILD_TASK)

    delivery._handle_send_message(event, host)

    (stored,) = _stored_rows(tmp_path)
    assert stored["direction"] == "system" and stored["type"] == "terminal_incident"
    assert stored["card_row"] == "timeline" and stored["card_row_id"] == ROW_ID
    assert stored["delegation_role"] == "subagent" and stored["parent_task_id"] == "root-1"
    assert RAW not in stored["text"], "the receipt, never the raw salvage"
    (replayed,) = _replayed_rows(tmp_path)
    assert replayed["system_type"] == "terminal_incident" and replayed["task_id"] == "kid-1"
    assert replayed["card_row"] == "timeline" and replayed["card_row_id"] == ROW_ID
    (live,) = [frame for frame in frames if frame.get("type") == "chat"]
    assert live["role"] == "system" and live["system_type"] == "terminal_incident"
    assert live["card_row"] == "timeline" and live["card_row_id"] == ROW_ID
    assert live["delegation_role"] == "subagent" and live["task_id"] == "kid-1"
    (outbound,) = published
    assert outbound["card_row"] == "timeline" and outbound["delegation_role"] == "subagent"
    assert outbound["root_task_id"] == "root-1"


def test_root_salvage_row_stays_an_ordinary_system_row_end_to_end(tmp_path, monkeypatch):
    host, frames, _published = _wire(tmp_path, monkeypatch)

    delivery._handle_send_message(_worker_final(tmp_path, {"id": "root-1", "chat_id": 1}), host)

    (stored,) = _stored_rows(tmp_path)
    assert stored["type"] == "terminal_incident"
    assert "card_row" not in stored and "card_row_id" not in stored
    (live,) = [frame for frame in frames if frame.get("type") == "chat"]
    assert live["system_type"] == "terminal_incident" and "card_row" not in live


def test_owed_outbox_replay_keeps_the_same_row_identity(tmp_path):
    """A crash between the settle and the send replays the owed row: the
    replayed event names the same card row, so the card never grows a twin."""
    from tests._cancel_intents_shared import DeliveryBus

    event = _worker_final(tmp_path, CHILD_TASK)
    assert register_pending_delivery(tmp_path, event)
    (owed,) = pending_deliveries(tmp_path)
    assert owed["progress_meta"]["card_row_id"] == ROW_ID

    events = DeliveryBus()
    registry = tmp_path / "state" / "terminal_deliveries.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["pending"][event["delivery_id"]]["registered_at"] = "2026-01-01T00:00:00+00:00"
    registry.write_text(json.dumps(data), encoding="utf-8")
    assert replay_pending_deliveries(tmp_path, event_queue=events) == [event["delivery_id"]]
    replayed = events.get_nowait()
    assert replayed["system_type"] == "terminal_incident"
    assert replayed["progress_meta"]["card_row"] == "timeline"
    assert replayed["progress_meta"]["card_row_id"] == ROW_ID


def test_a_host_notice_beside_a_salvaged_child_answer_yields_no_second_row(tmp_path, monkeypatch):
    """A salvaged child answer that also carries a host notice: the notice stays a
    field of the RESULT (chat-voice C3) and never rides the send event, so the
    receipt is the only row and keeps its own placement."""
    host, _frames, _published = _wire(tmp_path, monkeypatch)
    notice = "Plan review stayed open; inspect the task details."
    event = _worker_final(tmp_path, CHILD_TASK, terminal_host_notice=notice)
    assert "terminal_host_notice" not in event

    delivery._handle_send_message(event, host)

    (receipt,) = _stored_rows(tmp_path)
    assert receipt["type"] == "terminal_incident" and receipt["card_row_id"] == ROW_ID
    assert notice not in receipt["text"]


def test_telegram_holds_a_child_row_only_while_its_root_is_unfinished(tmp_path):
    """The bot has no cards: a child's card row waits for its root, a root's does not.

    Both directions matter. The hold must fire for a child under a live root and
    stay quiet for the root's own placed row, which is an ordinary progress note
    there and follows the owner's progress toggle.
    """
    from tests.test_telegram_health_tasks import _api_with_data, _load_plugin

    held = _load_plugin()._child_row_held_for_root
    api, data = _api_with_data(tmp_path)
    child = {"delegation_role": "subagent", "root_task_id": "root-one", "card_row": "timeline"}
    assert held(api, child), "an unreadable root counts as unfinished"
    assert held(api, {**child, "card_row": "", "system_type": "terminal_incident"})
    seed_result(data, "root-one", {"status": "running"})
    assert held(api, child)
    seed_result(data, "root-one", {"status": "completed"})
    assert not held(api, child)
    assert not held(api, {"card_row": "timeline", "is_progress": True}), "a root's own row is never held"
    assert not held(api, {**child, "card_row": ""}), "an ordinary child message is not a card row"
