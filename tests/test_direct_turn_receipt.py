"""An owner turn leaves one ``direct_turn`` receipt: its wait from the canonical acceptance, each
start phase's seconds, the dispatch basis and its own store counts.

The receipt rides the turn's first dispatch frame by value; the supervisor's events consumer writes
the one journal row off the turn thread and keeps it in the serving process's bounded ring. A turn
that never dispatches writes ``dispatch: not_observed`` from its unregister. Two concurrent turns
keep their own meters."""
from __future__ import annotations

import datetime
import json
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from supervisor import active_activity, events
from supervisor.active_activity import DirectActivityRegistry, TurnExecution, direct_turns, record_direct_turn
from supervisor.log_addressing import TurnEventQueue


@pytest.fixture(autouse=True)
def fresh_ring(monkeypatch):
    from ouroboros import update_generation_facts as facts

    monkeypatch.setattr(active_activity, "_DIRECT_TURNS", deque(maxlen=active_activity.DIRECT_TURNS_KEPT))
    monkeypatch.setattr(facts, "publish_pending_dispatch", lambda: None)  # the after-update flush has its own test
    monkeypatch.setattr(facts, "_direct", None)


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat()


def _frame(phase: str, **extra) -> dict:
    return {"type": "main_llm_call_state", "task_id": "owner", "phase": phase, "task_attempt": 1,
            "llm_call_id": "call-1", "execution_id": "execution-1", "round_id": "round-1", "call_attempt": 1, **extra}


def _journal(root) -> list:
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_the_receipt_rides_the_first_dispatch_frame_and_the_consumer_records_it_once(tmp_path):
    execution = TurnExecution(chat_id=1, task_id="owner", owner_turn=True, client_message_id="m-1")
    execution.accepted_at = _iso(execution.entered_at - 2.0)  # the canonical acceptance, two seconds before this thread
    execution.meter = {"current.read": 3, "current.read.sec": 0.012, "chronicle.locked": 1}
    execution.census = {"direct_turns_at_entry": 1}
    for phase in ("admitted", "started", "context_building", "context_built"):
        execution.advance(phase)
    frames: queue.Queue = queue.Queue()
    turn = TurnEventQueue(frames, "owner", 1, on_frame=execution.observe)
    turn.put(_frame("started"))
    assert "direct_turn" not in frames.get_nowait()  # a call's start is no dispatch
    turn.put(_frame("observed", model_activity={"phase": "dispatching"}))
    sent = frames.get_nowait()
    receipt = sent["direct_turn"]
    assert receipt["type"] == "direct_turn" and receipt["dispatch"] == "observed_send"
    assert list(receipt["phases"]) == ["accepted", "admitted", "started", "context_building", "context_built", "requested"]
    assert receipt["slowest_phase"] in receipt["phases"] and receipt["slowest_phase_sec"] == max(receipt["phases"].values())
    assert receipt["accepted_at"] == execution.accepted_at and 1.9 <= receipt["wait_before_entry_sec"] <= 2.5
    assert receipt["owner_wait_sec"] == pytest.approx(receipt["wait_before_entry_sec"] + receipt["handler_wait_sec"], abs=1e-3)
    assert receipt["handler_wait_sec"] == pytest.approx(execution.marks[-1][1] - execution.marks[0][1], abs=1e-3)
    assert receipt["meter"] == {"current.read": 3, "current.read.sec": 0.012, "chronicle.locked": 1}
    # Order, not a tick: two reads of a coarse clock (Windows) can be equal.
    assert receipt["census"] == {"direct_turns_at_entry": 1} and receipt["dispatched_at"] >= receipt["entered_at"]
    assert (receipt["chat_id"], receipt["task_id"], receipt["execution_id"]) == (1, "owner", execution.execution_id)

    ctx = SimpleNamespace(RUNNING={"a": {}}, PENDING=[1, 2], DRIVE_ROOT=tmp_path)
    events._handle_main_llm_call_state(sent, ctx)
    rows = [row for row in _journal(tmp_path) if row["type"] == "direct_turn"]
    assert len(rows) == 1 and rows[0]["execution_id"] == execution.execution_id and rows[0]["ts"]
    assert rows[0]["census"] == {"direct_turns_at_entry": 1, "running_when_handled": 1, "pending_when_handled": 2}
    kept = direct_turns()
    assert len(kept) == 1 and kept[0]["journal"] == "written" and kept[0]["dispatch"] == "observed_send"
    # The frame handled again, a repeated send and a later round record nothing more.
    events._handle_main_llm_call_state(sent, ctx)
    turn.put(_frame("observed", model_activity={"phase": "awaiting_response"}))
    later = frames.get_nowait()
    assert "direct_turn" not in later
    events._handle_main_llm_call_state(later, ctx)
    turn.put(_frame("started", llm_call_id="call-2", round_id="round-2"))
    assert "direct_turn" not in frames.get_nowait()
    assert len([row for row in _journal(tmp_path) if row["type"] == "direct_turn"]) == 1 and len(direct_turns()) == 1


def test_a_route_that_observes_nothing_names_the_calls_start_as_its_basis(tmp_path):
    execution = TurnExecution(chat_id=1, task_id="owner", owner_turn=True)
    frames: queue.Queue = queue.Queue()
    turn = TurnEventQueue(frames, "owner", 1, on_frame=execution.observe)
    turn.put(_frame("started"))
    frames.get_nowait()
    turn.put(_frame("finished"))
    receipt = frames.get_nowait()["direct_turn"]
    assert receipt["dispatch"] == "unobserved_call_start" and receipt["phases"]["requested"] == 0.0
    assert "accepted_at" not in receipt and "owner_wait_sec" not in receipt  # no canonical stamp: no owner wait invented
    assert receipt["handler_wait_sec"] >= 0.0


def test_a_wake_leaves_no_receipt_on_its_frame():
    execution = TurnExecution(chat_id=1, task_id="wake", owner_turn=False)
    frames: queue.Queue = queue.Queue()
    turn = TurnEventQueue(frames, "wake", 1, on_frame=execution.observe)
    turn.put(_frame("started", task_id="wake"))
    frames.get_nowait()
    turn.put(_frame("observed", task_id="wake", model_activity={"phase": "dispatching"}))
    assert execution.phase == "dispatched" and "direct_turn" not in frames.get_nowait()


def test_a_turn_that_never_dispatches_records_not_observed_from_its_unregister(monkeypatch):
    rows: list = []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: rows.append(dict(row)) or True)
    registry = DirectActivityRegistry()
    stamp = _iso(time.time() - 1.0)
    begun = registry.begin_execution(1, client_message_id="m-1")
    begun.advance("admitted")  # the mark the end measures up to
    registry.register("t-owner", 1, client_message_id="m-1", reply_expected=True,
                      origin_message_ref={"chat_id": 1, "client_message_id": "m-1", "ts": stamp, "text_sha256": "0" * 64})
    assert begun.accepted_at == stamp and begun.census == {"direct_turns_at_entry": 0}
    assert rows == [] and direct_turns() == []
    registry.unregister("t-owner")
    assert [row["type"] for row in rows] == ["direct_turn"]
    receipt = rows[0]
    assert receipt["dispatch"] == "not_observed" and receipt["execution_id"] == begun.execution_id
    assert list(receipt["phases"]) == ["accepted", "admitted"] and receipt["accepted_at"] == stamp
    assert receipt["phases"]["admitted"] == pytest.approx(begun.ended_at - begun.marks[-1][1], abs=1e-3)
    assert direct_turns()[0]["journal"] == "written" and "dispatched_at" not in receipt
    # A turn nobody awaits (a wake) and a registry reset leave none.
    registry.register("t-wake", 1, reply_expected=False)
    registry.unregister("t-wake")
    registry.register("t-cleared", 1, reply_expected=True, origin_message_ref={"ts": stamp})
    registry.clear()
    assert len(rows) == 1 and len(direct_turns()) == 1


def test_a_journal_that_cannot_be_written_leaves_the_ring_row_with_its_gap(monkeypatch):
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: False)
    kept = record_direct_turn({"type": "direct_turn", "execution_id": "e-1", "dispatch": "observed_send"})
    assert kept["journal"] == "not_written" and direct_turns() == [kept]

    def refuses(path, row, *a, **k):
        raise OSError("journal unavailable")

    monkeypatch.setattr("supervisor.state.append_jsonl", refuses)
    assert record_direct_turn({"type": "direct_turn", "execution_id": "e-2"})["journal"] == "not_written"
    assert record_direct_turn({"type": "direct_turn", "execution_id": "e-2"}) is None  # the same execution, once


def test_the_ring_keeps_the_newest_hundred_by_value(monkeypatch):
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: True)
    for index in range(active_activity.DIRECT_TURNS_KEPT + 1):
        record_direct_turn({"type": "direct_turn", "execution_id": f"e-{index}", "phases": {"accepted": 0.1}})
    kept = direct_turns()
    assert len(kept) == active_activity.DIRECT_TURNS_KEPT and kept[0]["execution_id"] == "e-1"
    newest = direct_turns(limit=5)
    assert [row["execution_id"] for row in newest] == [f"e-{index}" for index in range(96, 101)]
    newest[-1]["phases"]["accepted"] = 9.9
    assert direct_turns(limit=1)[0]["phases"]["accepted"] == 0.1 and direct_turns(limit=0) == []


def test_two_concurrent_owner_turns_keep_their_own_meters(monkeypatch, tmp_path):
    """Through the real direct lane: each turn's thread meters its own store reads inside the
    scope the lane opens; the receipts on their dispatch frames carry different counts."""
    from ouroboros import agent as agent_module
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import workers
    from tests.test_direct_turn_phase_stall import _Actor, _lane

    out = _lane(monkeypatch, tmp_path)
    write_task_result(tmp_path, "seed", "running")
    both_inside = threading.Barrier(2, timeout=10)

    def reading():
        def behave(actor, task):
            both_inside.wait()  # both turns are inside their own metered scope now
            for _ in range(3 * int(task["chat_id"]) - 2):  # chat 1 reads once, chat 2 four times
                load_task_result(tmp_path, "seed")
            both_inside.wait()
            identity = {"task_attempt": 1, "llm_call_id": f"call-{task['id']}", "execution_id": f"x-{task['id']}",
                        "round_id": "round-1", "call_attempt": 1}
            actor._event_queue.put({"type": "main_llm_call_state", "task_id": task["id"], "phase": "started", **identity})
            actor._event_queue.put({"type": "main_llm_call_state", "task_id": task["id"], "phase": "observed",
                                    "model_activity": {"phase": "dispatching"}, **identity})
            return [{"type": "send_message", "chat_id": task["chat_id"], "text": "the answer", "task_id": task["id"]}]
        return behave

    actors = [_Actor(reading()), _Actor(reading())]
    monkeypatch.setattr(agent_module, "make_agent", lambda **kwargs: actors.pop(0))
    stamp = _iso(time.time())

    def origin(chat_id: int) -> dict:
        return {"client_message_id": f"m-{chat_id}", "origin_message_ref": {
            "chat_id": chat_id, "client_message_id": f"m-{chat_id}", "ts": stamp, "text_sha256": "0" * 64}}

    threads = [threading.Thread(target=workers.handle_chat_direct, args=(chat_id, "hello"),
                                kwargs={"task_metadata": origin(chat_id)}, daemon=True) for chat_id in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not any(thread.is_alive() for thread in threads)
    frames = []
    while not out.bus.empty():
        frames.append(out.bus.get_nowait())
    receipts = {frame["chat_id"]: frame["direct_turn"] for frame in frames if "direct_turn" in frame}
    assert set(receipts) == {1, 2}
    assert receipts[1]["meter"]["current.read"] == 1 and receipts[2]["meter"]["current.read"] == 4
    assert all(receipt["accepted_at"] == stamp and receipt["dispatch"] == "observed_send" for receipt in receipts.values())
    ctx = SimpleNamespace(RUNNING={}, PENDING=[], DRIVE_ROOT=tmp_path)
    for frame in frames:
        if frame.get("type") == "main_llm_call_state":
            events._handle_main_llm_call_state(frame, ctx)
    assert sorted(row["chat_id"] for row in out.rows if row["type"] == "direct_turn") == [1, 2]
    assert len(direct_turns()) == 2
