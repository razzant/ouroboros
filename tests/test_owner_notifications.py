"""The owner-notification fact: one durable row, one live log frame, one topic.

``emit_owner_notification`` is the single seam every producer uses (the Host
Service ``POST /notify`` route today; the scheduler and the agent's follow-up
tool when they gain a model-free notification). These tests pin what makes it
honest: the row lands in ``logs/events.jsonl`` and nowhere near ``chat.jsonl``,
the production server log sink turns that append into exactly ONE ``log``
frame addressed to the owner's chat, the ``owner.notification`` topic reaches
subscribers after the row, and a failed append answers ``None`` instead of a
publish nobody can find later.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import event_bus
from ouroboros import utils


def _rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def fresh_bus(monkeypatch):
    bus = event_bus.init_global_event_bus()
    yield bus
    event_bus.init_global_event_bus()


def test_emit_writes_one_events_row_and_publishes_after_it(tmp_path, fresh_bus):
    seen: list[dict] = []
    fresh_bus.subscribe("telegram", event_bus.OWNER_NOTIFICATION, seen.append)

    row = event_bus.emit_owner_notification(
        tmp_path, chat_id=1, category="notice", text="  Meeting with Ivan in 15 min  ",
        source="skill:calendar", key="cal:evt-1",
    )

    assert row is not None
    rows = _rows(tmp_path / "logs" / "events.jsonl")
    assert [r["type"] for r in rows] == ["owner_notification"]
    stored = rows[0]
    assert stored["text"] == "Meeting with Ivan in 15 min"
    assert stored["category"] == "notice"
    assert stored["source"] == "skill:calendar"
    assert stored["key"] == "cal:evt-1"
    assert stored["chat_id"] == 1
    assert "task_id" not in stored, "a skill notice never joins a task's model context"
    assert not (tmp_path / "logs" / "chat.jsonl").exists(), "never a chat row"
    assert len(seen) == 1 and seen[0]["topic"] == event_bus.OWNER_NOTIFICATION
    assert seen[0]["text"] == stored["text"] and seen[0]["ts"] == stored["ts"]


def test_emit_becomes_exactly_one_live_log_frame_through_the_production_sink(tmp_path, fresh_bus, monkeypatch):
    from supervisor import message_bus
    from supervisor.log_addressing import make_server_log_sink

    frames: list[dict] = []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    utils.set_log_sink(make_server_log_sink(bridge, tmp_path, running={}))
    try:
        event_bus.emit_owner_notification(
            tmp_path, chat_id=1, category="notice", text="hello", source="skill:calendar",
        )
    finally:
        utils.set_log_sink(None)

    assert len(frames) == 1
    frame = frames[0]
    assert frame["type"] == "log" and frame["chat_id"] == 1
    assert frame["data"]["type"] == "owner_notification" and frame["data"]["text"] == "hello"


def test_emit_refuses_bad_input_before_writing_anything(tmp_path, fresh_bus):
    for kwargs in (
        dict(chat_id=1, category="notice", text="", source="skill:x"),
        dict(chat_id=1, category="notice", text="x" * 1001, source="skill:x"),
        dict(chat_id=1, category="", text="hi", source="skill:x"),
        dict(chat_id=1, category="notice", text="hi", source=""),
        dict(chat_id=0, category="notice", text="hi", source="skill:x"),
        dict(chat_id=-5, category="notice", text="hi", source="skill:x"),
        dict(chat_id=1, category="notice", text="hi", source="skill:x", key="k" * 129),
    ):
        with pytest.raises(ValueError):
            event_bus.emit_owner_notification(tmp_path, **kwargs)
    assert not (tmp_path / "logs").exists()


def test_failed_append_answers_none_and_publishes_nothing(tmp_path, fresh_bus, monkeypatch):
    seen: list[dict] = []
    fresh_bus.subscribe("telegram", event_bus.OWNER_NOTIFICATION, seen.append)
    monkeypatch.setattr(utils, "append_jsonl", lambda *a, **k: False)

    assert event_bus.emit_owner_notification(
        tmp_path, chat_id=1, category="notice", text="hi", source="skill:x",
    ) is None
    assert seen == []


def test_owner_notification_is_a_valid_topic_for_plugins_and_companions():
    assert event_bus.OWNER_NOTIFICATION in event_bus.VALID_TOPICS
    from ouroboros.contracts.skill_manifest import _EVENT_TOPIC_RE

    assert _EVENT_TOPIC_RE.match(event_bus.OWNER_NOTIFICATION)


def test_emit_repairs_a_torn_predecessor_so_the_receipt_always_parses(tmp_path, fresh_bus):
    """A crash mid-append leaves a partial line; the next notification must not
    be glued onto it (the scheduler consumes its row on the append's success)."""
    log_path = tmp_path / "logs" / "events.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text('{"ts": "2026-09-26T00:1', encoding="utf-8")
    row = event_bus.emit_owner_notification(
        tmp_path, chat_id=1, category="notice", text="after the tear", source="skill:x",
    )
    assert row is not None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == json.dumps(row, ensure_ascii=False)
    assert json.loads(lines[-1])["text"] == "after the tear"


def test_emit_answers_none_when_the_log_cannot_be_opened(tmp_path, fresh_bus):
    (tmp_path / "logs").write_text("a file where the directory should be", encoding="utf-8")
    assert event_bus.emit_owner_notification(
        tmp_path, chat_id=1, category="notice", text="hi", source="skill:x",
    ) is None
