"""The VOICE of a progress note survives the producer, delivery and history seams.

Host-authored notes (checkpoints, fallback, plan, acceptance, nudge, transport,
density) and the model's own round narration share one frame type, so the card
cannot tell them apart by text without string matching (BIBLE P5). The worker
stamps ``progress_meta.narration`` on every note it emits instead, and this
module pins that the fact is explicit at the producer, rides the live frame at
the TOP level, and replays through the progress-meta whitelist.
"""

import asyncio
import json
import queue
from functools import partial
from types import SimpleNamespace

import pytest

from ouroboros.agent import OuroborosAgent
from ouroboros.gateway.history import make_chat_history_endpoint
from supervisor import events_chat_delivery, message_bus


def _agent():
    events = queue.Queue()
    agent = SimpleNamespace(
        _last_progress_ts=None, _event_queue=events, _current_chat_id=1,
        _current_task_id="task-1", tools=SimpleNamespace(_ctx=SimpleNamespace(task_attempt=0)),
        _subagent_progress_meta=lambda event: {},
    )
    return agent, events


def _emit(agent, text, **kwargs):
    OuroborosAgent._emit_progress(agent, text, **kwargs)


def test_every_emitted_note_declares_its_voice_explicitly():
    """Absence must never be how a host note is recognised: the default is an
    explicit False, so a reader can tell "host note" from "older worker"."""
    agent, events = _agent()
    _emit(agent, "Checkpoint 3 at round 12")
    _emit(agent, "Thinking about the next step", narration=True)
    _emit(agent, "⚡ Fallback: switching model lane",
          incident={"task_incident": "model_lane_switch", "toast_once": "lane"})

    host, narration, fallback = (events.get_nowait() for _ in range(3))
    assert host["progress_meta"]["narration"] is False
    assert narration["progress_meta"]["narration"] is True
    # An incident note is still the host talking; the typed toast pair is untouched.
    assert fallback["progress_meta"]["narration"] is False
    assert fallback["progress_meta"]["task_incident"] == "model_lane_switch"
    assert fallback["progress_meta"]["toast_once"] == "lane"


def test_the_tool_context_abi_stays_a_host_voice():
    """``ctx.emit_progress_fn`` takes a single positional argument, so every tool
    note keeps the default without the ABI having to know the fact exists."""
    agent, events = _agent()
    ctx = SimpleNamespace(emit_progress_fn=partial(OuroborosAgent._emit_progress, agent))
    ctx.emit_progress_fn("📐 plan_task: wave 1 dispatched")
    assert events.get_nowait()["progress_meta"]["narration"] is False


@pytest.mark.parametrize("content, msg, expected", [
    ("The answer is 42.", {}, "The answer is 42."),
    ([{"type": "thinking", "thinking": "x"}], {"reasoning": "weighing the options"},
     "weighing the options"),
])
def test_round_progress_is_the_only_narration_producer(content, msg, expected):
    """Both of its emissions — visible round text and display reasoning — are the
    turn's own speech."""
    from ouroboros.loop import _emit_round_progress

    seen = []

    def emit(text, **meta):
        seen.append((text, meta))

    _emit_round_progress(content, msg, emit, {"reasoning_notes": []})
    # Display reasoning additionally rides a ``meta={"reasoning": True}`` stamp; the
    # voice fact pinned here is the ``narration`` flag alone.
    assert [(text, meta.get("narration")) for text, meta in seen] == [(expected, True)]


def test_voice_rides_the_live_frame_the_stored_row_and_the_replay(tmp_path, monkeypatch):
    """Producer -> supervisor delivery -> live WS frame / progress.jsonl / history.

    The browser reads the key at the TOP level of the frame (the delivery seam
    spreads progress_meta there, exactly as it does for cancelable), and a reload
    must not hand the title back to a note live rendering refused it.
    """
    agent, events = _agent()
    _emit(agent, "Checkpoint 3 at round 12")
    _emit(agent, "Reading the failing test first.", narration=True)
    frames = [events.get_nowait() for _ in range(2)]

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").touch()
    live = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    delivery = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={"task-1": {"task": {"id": "task-1", "_attempt": 0}}},
        send_with_budget=message_bus.send_with_budget,
        append_jsonl=lambda *_: pytest.fail("delivery raised"),
    )
    for frame in frames:
        events_chat_delivery._handle_send_message(frame, delivery)

    stored = [json.loads(line) for line
              in (tmp_path / "logs" / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={"limit": "10"})))
    replay = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]

    for rows in (live, stored, replay):
        assert [row["narration"] for row in rows] == [False, True], rows
    # The voice is presentation only: the host note keeps its liveness semantics
    # (that marker is supervisor-authored HOST_NARRATION, a different key).
    assert all(events_chat_delivery.HOST_NARRATION not in row for row in live)


def test_a_stored_row_without_the_key_replays_as_a_legacy_frame(tmp_path):
    """An older worker's row carries no voice; history must not invent one, so the
    browser can keep promoting it exactly as it did before the fact existed."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").touch()
    (logs / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-09-16T00:00:00Z", "task_id": "task-1", "content": "Working on it.",
        "is_progress": True, "direction": "out", "chat_id": 1,
    }) + "\n", encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={"limit": "10"})))
    row, = json.loads(response.body)["messages"]
    assert row["text"] == "Working on it."
    assert "narration" not in row
