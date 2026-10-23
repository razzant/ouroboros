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
from ouroboros.dialogue_import import export_for_older_body
from tests._dialogue import progress_rows

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
    ctx.emit_progress_fn("📐 Plan review: wave 1 dispatched")
    assert events.get_nowait()["progress_meta"]["narration"] is False


@pytest.mark.parametrize("chat_id", [0, 5])
def test_task_bound_progress_keeps_identity_after_worker_moves_on(chat_id):
    """Production callback wiring survives idle, reused worker and hidden chat."""
    agent, events = _agent()
    agent._emit_progress = partial(OuroborosAgent._emit_progress, agent)
    agent._bind_task_progress = partial(OuroborosAgent._bind_task_progress, agent)
    agent._current_chat_id, agent._current_task_metadata = chat_id, {}
    emit_task = OuroborosAgent._bind_task_progress_for_task(agent, {"id": "task-a", "_attempt": 2})
    for current in (None, "task-b"):
        agent._current_task_id, agent._current_chat_id = current, 1
        emit_task("late review result")
        event = events.get_nowait()
        assert (event["task_id"], event["chat_id"]) == ("task-a", chat_id)
    # Surviving normal path: the new task's callback addresses the new task.
    emit_b = OuroborosAgent._bind_task_progress_for_task(agent, {"id": "task-b"})
    emit_b("current narration", narration=True)
    event = events.get_nowait()
    assert (event["task_id"], event["chat_id"], event["progress_meta"]["narration"]) == ("task-b", 1, True)
    emit_ownerless = OuroborosAgent._bind_task_progress(agent, "task-no-room", None)
    emit_ownerless("ownerless late note")
    assert events.empty()


def test_task_bound_progress_keeps_lineage_meta_after_worker_moves_to_child():
    agent, events = _agent()
    agent._emit_progress = partial(OuroborosAgent._emit_progress, agent)
    agent._bind_task_progress = partial(OuroborosAgent._bind_task_progress, agent)
    agent._current_chat_id = 5
    agent._current_task_metadata = {
        "delegation_role": "subagent", "parent_task_id": "parent-a",
        "root_task_id": "root-a", "subagent_role": "reviewer", "initiator": "consciousness",
    }
    task = {"id": "task-a", "_attempt": 2, "effective_executor": "blocked", "executor_route": ""}
    OuroborosAgent._record_executor_facts(agent, task, {"executor_blocked_route": "codex"})
    bound = OuroborosAgent._bind_task_progress_for_task(agent, task)
    agent._current_task_id, agent._current_chat_id = "child-b", 1
    agent._current_task_metadata.update(parent_task_id="parent-b", root_task_id="root-b",
                                        subagent_role="writer", executor_route="claude")
    agent.tools._ctx.task_attempt = 3
    observation = {"task_id": "task-a", "task_attempt": "2", "run_id": "run-a",
                   "attempt_id": "a01", "harness_id": "codex", "phase": "finished", "revision": 1}
    bound("late child review", executor_observation=observation)
    event = events.get_nowait()
    assert (event["task_id"], event["chat_id"]) == ("task-a", 5)
    meta = event["progress_meta"]
    assert (meta["subagent_task_id"], meta["parent_task_id"], meta["root_task_id"]) == (
        "task-a", "parent-a", "root-a")
    assert (meta["executor_route"], meta["initiator"]) == ("codex", "consciousness")
    assert meta["executor_observation"] == observation


def test_bound_spoken_row_keeps_task_provenance():
    agent, events = _agent()
    agent._emit_progress = partial(OuroborosAgent._emit_progress, agent)
    bound = OuroborosAgent._bind_task_progress(agent, "call-task", 5,
        {"root_task_id": "bound-root", "initiator": "consciousness"}, 2)
    agent._current_task_id, agent._current_chat_id = "next-task", 9
    spoken = {"call_id": "synthetic-call", "item_id": "item-1", "played_ms": 120, "status": "interrupted"}
    bound("Words generated before interruption.", narration=True,
          _progress_meta_override={"spoken": spoken, "root_task_id": "wrong-root", "source": "live_call"})
    event = events.get_nowait()
    assert (event["task_id"], event["chat_id"]) == ("call-task", 5)
    assert event["progress_meta"] == {"spoken": spoken, "root_task_id": "bound-root",
        "initiator": "consciousness", "source": "live_call", "narration": True}


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
    assert seen == [(expected, {"narration": True})]


def test_voice_rides_the_live_frame_the_stored_row_and_the_replay(tmp_path, monkeypatch):
    """Producer -> supervisor delivery -> live WS frame / progress.jsonl / history.

    The browser reads the key at the TOP level of the frame (the delivery seam
    spreads progress_meta there, exactly as it does for cancelable), and a reload
    must not hand the title back to a note live rendering refused it.
    """
    agent, events = _agent()
    _emit(agent, "Checkpoint 3 at round 12")
    spoken = {"call_id": "synthetic", "item_id": "one", "played_ms": 60, "status": "interrupted"}
    _emit(agent, "Reading the failing test first.", narration=True, _progress_meta_override={"spoken": spoken})
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

    stored = progress_rows(tmp_path)
    export_for_older_body(tmp_path)  # transition: the history reader (unit Q1b) still reads the chain files
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={"limit": "10"})))
    replay = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]

    for rows in (live, stored, replay):
        assert [row["narration"] for row in rows] == [False, True], rows
        assert [row["role"] for row in rows] == ["system", "assistant"], rows
        assert [row["system_type"] for row in rows] == ["host_progress", "model_narration"], rows
        assert rows[1]["spoken"] == spoken, rows
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


def test_a_tool_may_speak_in_the_models_voice_only_with_the_narration_fact():
    """Through the real task binder, a tool that relays the model's own words passes the
    typed ``narration`` fact and the frame is the assistant's; the same text without the
    keyword stays a host note."""
    agent, events = _agent()
    agent._emit_progress = partial(OuroborosAgent._emit_progress, agent)
    agent._bind_task_progress = partial(OuroborosAgent._bind_task_progress, agent)
    agent._current_task_metadata = {}
    ctx = SimpleNamespace(emit_progress_fn=OuroborosAgent._bind_task_progress_for_task(agent, {"id": "task-a"}))
    words = "The reviewers' note assumes a second chart; the brief fixes one, so I go on."
    ctx.emit_progress_fn(words, narration=True)
    ctx.emit_progress_fn(words)
    spoken, host = events.get_nowait(), events.get_nowait()
    assert (spoken["role"], spoken["system_type"], spoken["progress_meta"]["narration"]) == (
        "assistant", "model_narration", True)
    assert spoken["text"] == f"💬 {words}" and spoken["task_id"] == "task-a"
    assert (host["role"], host["system_type"], host["progress_meta"]["narration"]) == (
        "system", "host_progress", False)
    assert events.empty()


def test_an_unbound_tool_context_swallows_the_narration_fact(tmp_path):
    """A ToolContext nobody bound to an agent (the dataclass default) accepts the same
    keyword facts the real binder does, so a tool relaying the model's words never raises on it."""
    from ouroboros.tools.tool_context import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    assert ctx.emit_progress_fn("x", narration=True) is None
    assert ctx.emit_progress_fn("x") is None
