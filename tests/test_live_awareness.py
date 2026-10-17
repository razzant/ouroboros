"""The room reaches the call and the call reaches the room; the call turn's admission and routing.

Rows the room accepts become context items for the call (without echoing the
call's own rows), finalized utterances reach the room's running text turns
through their mailbox, a live call is admitted as a census turn of kind
``call`` and Main's text routing does not steer it.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from ouroboros import live_awareness
from ouroboros.call_protocol import CallSpec
from ouroboros.gateways.realtime_engine import ENGINE_GOOGLE
from ouroboros.utils import utc_now_iso
from supervisor import workers
from supervisor.active_activity import get_direct_activity_registry


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def _log(root, direction, chat_id, text, **fields):
    from supervisor.message_bus import log_chat

    return log_chat(direction, chat_id, 1, text, drive_root=root, **fields)


def _spec(**overrides) -> CallSpec:
    fields = dict(call_id="call-1", chat_id=1, engine=ENGINE_GOOGLE, model="gemini-3.8-live", voice="Kore",
                  opened_by="owner", presence_binding_id="", max_minutes=0, label="Test call", client_message_id="cm-1")
    fields.update(overrides)
    return CallSpec(**fields)


# --- the room reaches the call -----------------------------------------------------


def test_log_chat_notifies_room_listeners_only_after_the_row_landed(tmp_path):
    heard: list = []
    live_awareness.register_room_listener(1, heard.append)
    try:
        row = _log(tmp_path, "in", 1, "typed")
        assert row is not None and heard and heard[0]["text"] == "typed" and heard[0]["direction"] == "in"
        _log(tmp_path, "in", 2, "another room")
        assert len(heard) == 1
    finally:
        live_awareness.unregister_room_listener(1, heard.append)
    _log(tmp_path, "in", 1, "after")
    assert len(heard) == 1


def test_room_rows_become_context_items_without_self_echo(tmp_path):
    started_at = utc_now_iso()
    time.sleep(0.01)
    _log(tmp_path, "in", 1, "typed before the listener existed")       # caught up from the chain
    rooms = live_awareness.RoomRows(tmp_path, 1, "t-call", "call-1", started_at)
    rooms.start()
    try:
        _log(tmp_path, "in", 1, "typed while on the call")
        _log(tmp_path, "in", 1, "spoken on the call", source="call", task_id="t-call")      # the call's own owner row
        _log(tmp_path, "out", 1, "another turn answered", task_id="t-text")
        _log(tmp_path, "out", 1, "the call's own progress", task_id="t-call")
        _log(tmp_path, "system", 1, "mirror of a project row", record_type="main_notice")
        live_awareness.notify_chat_row(tmp_path, {"chat_id": 1, "direction": "out", "text": "a progress line",
                                                   "is_progress": True, "ts": utc_now_iso()})
        _log(tmp_path, "in", 7, "typed in a different room")
        items = rooms.drain()
        texts = [item.text for item in items]
        assert len(items) == 3, texts
        assert texts[0].startswith("The owner typed in this room [row:1@") and texts[0].endswith("typed before the listener existed")
        assert texts[1].endswith("typed while on the call") and items[1].source == "room_row:in"
        assert texts[2].startswith("Ouroboros wrote in this room (turn t-text) [row:1@") and items[2].source == "room_row:out"
        assert rooms.drain() == []
    finally:
        rooms.stop()
    _log(tmp_path, "in", 1, "after the call")
    assert rooms.drain() == []


def test_the_hidden_partition_is_a_room_and_a_row_without_a_chat_id_is_not_it(tmp_path):
    """Chat 0 (the hidden partition) is a real room; a row without an id is the chain's legacy
    main-chat row, live and in catch-up alike, never chat 0."""
    hidden, main = [], []
    live_awareness.register_room_listener(0, hidden.append)
    live_awareness.register_room_listener(1, main.append)
    try:
        for record in ({"chat_id": 0, "text": "a review panel row"}, {"text": "a row without an id"},
                       {"chat_id": None, "text": "a row with a null id"}, {"chat_id": "", "text": "a blank id"}):
            live_awareness.notify_chat_row(tmp_path, {"direction": "in", "ts": utc_now_iso(), **record})
    finally:
        live_awareness.unregister_room_listener(0, hidden.append)
        live_awareness.unregister_room_listener(1, main.append)
    assert [row["text"] for row in hidden] == ["a review panel row"]
    assert [row["text"] for row in main] == ["a row without an id", "a row with a null id", "a blank id"]
    hidden_room = live_awareness.RoomRows(tmp_path, 0, "t-call", "call-0", utc_now_iso())
    main_room = live_awareness.RoomRows(tmp_path, 1, "t-call", "call-1", utc_now_iso())
    zero, absent = {"chat_id": 0, "direction": "in", "text": "x"}, {"direction": "in", "text": "x"}
    assert hidden_room.accepts(zero) and not hidden_room.accepts(absent)
    assert main_room.accepts(absent) and not main_room.accepts(zero)


def test_two_identical_lines_are_two_facts_but_one_row_is_one(tmp_path):
    rooms = live_awareness.RoomRows(tmp_path, 1, "t-call", "call-1", utc_now_iso())
    rooms.start()
    try:
        first = _log(tmp_path, "in", 1, "yes")
        time.sleep(0.002)
        _log(tmp_path, "in", 1, "yes")
        live_awareness.notify_chat_row(tmp_path, dict(first))   # the same row delivered twice
        assert len(rooms.drain()) == 2
    finally:
        rooms.stop()


def test_a_main_call_does_not_capture_typed_project_room_text(tmp_path):
    from ouroboros.contracts.chat_id_policy import is_project_chat_id

    project_chat = next(cid for cid in range(10 ** 9, 10 ** 9 + 10 ** 6) if is_project_chat_id(cid))
    rooms = live_awareness.RoomRows(tmp_path, 1, "t-call", "call-1", utc_now_iso())
    rooms.start()
    try:
        _log(tmp_path, "in", project_chat, "typed into a project room")
        _log(tmp_path, "in", 1, "typed into Main")
        items = rooms.drain()
        assert [item.text.rsplit("\n", 1)[-1] for item in items] == ["typed into Main"]
    finally:
        rooms.stop()


def test_render_bounds_a_long_row_with_a_disclosed_omission():
    from ouroboros.owner_mailbox import TASK_MESSAGE_MAX_CHARS

    item = live_awareness.render_room_row({"direction": "in", "text": "x" * (TASK_MESSAGE_MAX_CHARS + 500)}, "row:1@t#abc")
    assert len(item.text) < TASK_MESSAGE_MAX_CHARS + 100 and "OMISSION NOTE" in item.text


# --- the call reaches the room -------------------------------------------------------


def test_utterances_reach_running_text_turns_as_independent_task_messages(tmp_path):
    from ouroboros.owner_mailbox import PROVENANCE_INDEPENDENT_TASK, drain_owner_entries

    fanout = live_awareness.UtteranceFanout(tmp_path, 1, "t-call", "call-1", recipients=lambda chat, me: ["t-text"])
    assert fanout.flush() == 0
    fanout.note("owner", "what did the review say")
    fanout.note("ouroboros", "two findings, one blocking")
    fanout.note("owner", "   ")
    assert fanout.pending == 2
    assert fanout.flush() == 1 and fanout.pending == 0
    entries = list(drain_owner_entries(tmp_path, "t-text", set(), 1))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["kind"] == "task_message" and entry["provenance"] == PROVENANCE_INDEPENDENT_TASK
    assert entry["source_task_id"] == "t-call" and entry["msg_id"] == "call:call-1:1:1"
    assert entry["text"] == ("[Live call call-1 in this room, spoken and transcribed]\n"
                             "Owner: what did the review say\nOuroboros: two findings, one blocking")
    fanout.note("owner", "thanks")
    assert fanout.flush() == 1 and fanout.delivered[-1]["msg_id"] == "call:call-1:2:1"


def test_long_batches_split_at_utterance_boundaries():
    from ouroboros.owner_mailbox import TASK_MESSAGE_MAX_CHARS

    lines = ["Owner: " + "a" * 3000, "Ouroboros: " + "b" * 3000, "Owner: " + "c" * 3000]
    parts = live_awareness.split_at_utterances(lines, TASK_MESSAGE_MAX_CHARS - 60)
    assert len(parts) == 2 and parts[0].count("\n") == 1 and parts[1] == lines[2]
    assert all(len(part) <= TASK_MESSAGE_MAX_CHARS - 60 for part in parts)
    huge = live_awareness.split_at_utterances(["Owner: " + "z" * 20000], 1000)
    assert len(huge) == 1 and len(huge[0]) <= 1000 and "OMISSION NOTE" in huge[0]


def test_running_room_turns_lists_the_rooms_running_roots_except_the_call(monkeypatch, tmp_path):
    from supervisor import queue as queue_mod

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "RUNNING", {
        "t-text": {"task": {"id": "t-text", "chat_id": 1, "text": "write the summary"}, "started_at": 1.0},
        "t-call": {"task": {"id": "t-call", "chat_id": 1, "text": "[Live call]"}, "started_at": 2.0},
        "t-other": {"task": {"id": "t-other", "chat_id": 2, "text": "elsewhere"}, "started_at": 3.0},
    })
    monkeypatch.setattr(workers, "PENDING", [{"id": "t-queued", "chat_id": 1, "text": "later"}])
    assert live_awareness.running_room_turns(1, "t-call") == ["t-text"]


# --- Main's text routing and the call turn --------------------------------------------------


class _Actor:
    def __init__(self, task_id, chat_id, text):
        self._owner_message_admission_lock = threading.RLock()
        self._busy = True
        self._current_task_id = task_id
        self._accepting_owner_messages = True
        self._current_task_metadata = {}
        self._current_task_text = text
        self._current_chat_id = chat_id
        self._current_task_started_at = time.time()


def test_main_text_routing_does_not_steer_a_call_but_still_sees_a_text_turn(monkeypatch, tmp_path):
    from ouroboros.server_routing_context import _active_direct_roots

    registry = get_direct_activity_registry()
    registry.register("t-call", 1, kind="call", actor=_Actor("t-call", 1, "[Live call]"))
    registry.register("t-text", 1, kind="direct_chat", actor=_Actor("t-text", 1, "a text turn"))
    try:
        roots = _active_direct_roots(SimpleNamespace(DRIVE_ROOT=tmp_path))  # a text turn's reader reads its drive
        assert [root["task_id"] for root in roots] == ["t-text"]
        assert {entry.activity_id for entry in registry.actors()} >= {"t-call", "t-text"}   # Stop/Panic still see it
    finally:
        registry.unregister("t-call")
        registry.unregister("t-text")


def _lane(monkeypatch, tmp_path):
    from supervisor import message_bus, state

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **kw: None))
    workers.open_repo_writer_admission()


def test_handle_call_direct_admits_a_census_turn_of_kind_call(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module

    _lane(monkeypatch, tmp_path)
    seen: list = []
    finished: list = []
    entered, release = threading.Event(), threading.Event()

    class Actor:
        def handle_task(self, task):
            seen.append(task)
            entered.set()
            assert release.wait(10)
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kwargs: Actor())
    spec = _spec(max_minutes=30)
    receipt = workers.handle_call_direct(1, spec, on_finished=lambda tid, ok: finished.append((tid, ok)))
    assert receipt["admitted"] is True and receipt["reason"] == ""
    entry = get_direct_activity_registry().get(receipt["task_id"])
    assert entry is not None and entry.kind == "call" and entry.chat_id == 1 and entry.client_message_id == "cm-1"
    assert entered.wait(10)
    task = seen[0]
    assert task["_is_direct_chat"] is True and task["chat_id"] == 1 and task["text"].startswith("[Live call]")
    assert CallSpec.from_metadata(task["metadata"]["live_call"]) == spec
    release.set()
    assert _wait_for(lambda: finished == [(receipt["task_id"], True)])
    assert get_direct_activity_registry().get(receipt["task_id"]) is None


def test_handle_call_direct_refuses_while_the_repo_writer_gate_is_closed(monkeypatch, tmp_path):
    _lane(monkeypatch, tmp_path)
    workers.close_repo_writer_admission("update")
    try:
        receipt = workers.handle_call_direct(1, _spec())
    finally:
        workers.open_repo_writer_admission("update")
    assert receipt == {"admitted": False, "task_id": "", "reason": "repo_writer_gate_closed"}
