"""The single consumer measures itself: one compact ``supervisor_load`` row a minute.

Before this row, an episode where answers waited minutes, child admission stalled and the
telemetry lane starved for 43 minutes could only be reconstructed from out-of-order event
stamps. The row carries, per event type, the handlers' count, total and longest service
time; per lane, the bus depth and oldest age at the moment of the row and the worst lag the
drain saw; and Q's outermost waits and holds with the longest hold's site. Pinned both ways:
a slow handler shows in the next row and an open window writes nothing; per-event work stays
in memory (no write, no Q); a failing source or journal never stops the loop or a delivery.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import server_liveness as sl
from supervisor import loop_load
from supervisor.event_observation import ObservedEventQueue
from supervisor.loop_load import MeasuredRLock


@pytest.fixture
def instruments(monkeypatch):
    """A fresh consumer window and a test-local Q window for the row, so it sees only this
    test's work. Q and the boot document themselves are never rebound: modules bind them by
    name at import, and a rebound object would outlive the test inside them."""
    lock = MeasuredRLock()
    monkeypatch.setattr(loop_load, "CONSUMER", loop_load.ConsumerWindow())
    monkeypatch.setattr(sl, "_load_sections", lambda: {"queue_lock": lock.drain})
    monkeypatch.setattr(sl.BOOT, "_load", None)
    return SimpleNamespace(_queue_lock=lock)


def _ago(seconds: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)).isoformat()


def _rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_a_slow_handler_appears_with_its_count_and_max_in_the_next_row(monkeypatch, tmp_path, instruments):
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod, "dispatch_event",
                        lambda evt, ctx: time.sleep(0.07) if evt["type"] == "send_message" else None)
    bus = ObservedEventQueue()
    for _ in range(2):
        bus.put({"type": "send_message", "task_id": "t", "chat_id": 1, "text": "answer", "ts": _ago(5)})
    for _ in range(3):
        bus.put({"type": "llm_usage", "task_id": "t", "ts": _ago(5)})
    for n in range(40):
        bus.put({"type": f"kind_nobody_declared_{n}", "task_id": "t"})
    assert sl.publish_supervisor_load(bus, tmp_path) is None, "an open window writes nothing"
    liveness = [time.monotonic(), {}, time.thread_time(), None]
    assert sl.drain_worker_events(bus, SimpleNamespace(DRIVE_ROOT=tmp_path), liveness,
                                  on_restart=lambda *_a: None) is False
    assert _rows(tmp_path) == [], "per-event work stays in memory: no write per event"
    bus.put({"type": "llm_usage", "task_id": "t"})
    bus.put({"type": "task_heartbeat", "task_id": "t"})  # a telemetry backlog at the moment of the row
    with instruments._queue_lock:
        time.sleep(0.04)  # a Windows sleep may end a ~15.6 ms tick early

    monkeypatch.setattr(sl, "SUPERVISOR_LOAD_WINDOW_SEC", 0.0)
    row = sl.publish_supervisor_load(bus, tmp_path)
    assert row["type"] == "supervisor_load" and row["window_sec"] >= 0
    send, usage = row["handlers"]["send_message"], row["handlers"]["llm_usage"]
    assert send["count"] == 2 and send["max_sec"] >= 0.05 and send["total_sec"] >= 0.1
    assert usage["count"] == 3 and usage["max_sec"] < 0.05
    assert row["handlers"]["undeclared"]["count"] == 40, "undeclared kinds fold into one key"
    assert set(row["handlers"]) == {"send_message", "llm_usage", "undeclared"}
    control, telemetry = row["lanes"]["control"], row["lanes"]["telemetry"]
    assert control["depth"] == 0 and control["oldest_age_sec"] is None
    assert telemetry["depth"] == 2 and telemetry["oldest_age_sec"] >= 0
    assert control["max_lag_sec"] == pytest.approx(5, abs=3) and telemetry["max_lag_sec"] == pytest.approx(5, abs=3)
    assert row["max_event_lag_sec"] == max(control["max_lag_sec"], telemetry["max_lag_sec"])
    q = row["queue_lock"]
    assert q["count"] == 1 and q["hold_max_sec"] >= 0.02
    assert q["hold_max_site"].startswith("tests/test_supervisor_load_row.py:")
    assert _rows(tmp_path) == [row]
    assert sl.BOOT.snapshot()["supervisor_load"] == {"ts": row["ts"], "max_event_lag_sec": row["max_event_lag_sec"]}

    with instruments._queue_lock:  # Q work during a stretch in which the consumer served nothing
        pass
    assert sl.publish_supervisor_load(bus, tmp_path) is None, "a window that served nothing folds into the next"
    assert len(_rows(tmp_path)) == 1
    bus.put({"type": "send_message", "task_id": "t", "chat_id": 1, "text": "later"})  # control: served first
    assert sl.drain_worker_events(bus, SimpleNamespace(DRIVE_ROOT=tmp_path), liveness, on_restart=lambda *_a: None) is False
    following = sl.publish_supervisor_load(bus, tmp_path)
    assert following["handlers"]["send_message"]["count"] == 1 and following["handlers"]["llm_usage"]["count"] == 1
    assert following["queue_lock"]["count"] == 1, "the folded stretch keeps its Q facts"
    assert following["max_event_lag_sec"] is None, "no event of the window carried a producer stamp"
    assert following["lanes"]["telemetry"]["depth"] == 0
    assert len(_rows(tmp_path)) == 2


def test_the_row_reads_the_real_q_and_neither_it_nor_the_drain_waits_for_q(monkeypatch, tmp_path):
    """The shipped section reads Q's own window; the instrumented drain and the row both
    finish while another thread holds Q (an instrument that took Q would measure itself)."""
    import threading

    from supervisor import events as events_mod, queue

    sections = sl._load_sections()
    assert set(sections) == {"queue_lock", "store_locks", "dialogue_retry"}
    from ouroboros import dialogue_retry, store_timing

    assert sections["store_locks"] is store_timing.drain and sections["dialogue_retry"] is dialogue_retry.snapshot
    assert {"rows", "bytes", "evicted", "max_rows"} <= set(sections["dialogue_retry"]())
    queue._queue_lock.drain()
    with queue._queue_lock:
        time.sleep(0.07)
    facts = sections["queue_lock"]()
    assert facts["count"] >= 1 and facts["hold_max_sec"] >= 0.05

    monkeypatch.setattr(loop_load, "CONSUMER", loop_load.ConsumerWindow())
    monkeypatch.setattr(sl.BOOT, "_load", None)
    monkeypatch.setattr(sl, "SUPERVISOR_LOAD_WINDOW_SEC", 0.0)
    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, ctx: None)
    bus = ObservedEventQueue()
    for _ in range(3):
        bus.put({"type": "llm_usage", "task_id": "t", "ts": _ago(1)})
    held, release, written = threading.Event(), threading.Event(), []

    def hold_q():
        with queue._queue_lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold_q)
    holder.start()
    try:
        assert held.wait(5)
        liveness = [time.monotonic(), {}, time.thread_time(), None]
        writer = threading.Thread(target=lambda: written.extend([
            sl.drain_worker_events(bus, SimpleNamespace(DRIVE_ROOT=tmp_path), liveness, on_restart=lambda *_a: None),
            sl.publish_supervisor_load(bus, tmp_path)]))
        writer.start()
        writer.join(5)
        assert not writer.is_alive(), "the drain or the row waited for Q"
        drained, row = written
        assert drained is False and row["handlers"]["llm_usage"]["count"] == 3 and "queue_lock" in row
    finally:
        release.set()
        holder.join(5)


def test_a_source_that_cannot_answer_is_a_typed_gap_and_the_rest_is_kept(monkeypatch, tmp_path, instruments):
    def broken():
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(sl, "SUPERVISOR_LOAD_WINDOW_SEC", 0.0)
    monkeypatch.setattr(sl, "_load_sections", lambda: {"queue_lock": instruments._queue_lock.drain, "broken": broken})
    loop_load.CONSUMER.note("task_heartbeat", "telemetry", 0.001, None)
    row = sl.publish_supervisor_load(SimpleNamespace(), tmp_path)  # a bus without lane facts
    assert row["broken"] == {"error": "RuntimeError"} and row["queue_lock"]["count"] == 0
    assert row["lanes"] == {lane: {"depth": None, "oldest_age_sec": None, "max_lag_sec": None}
                            for lane in ("control", "telemetry")}, "unknown depth is None, never 0"
    assert _rows(tmp_path) == [row]


def test_the_boot_document_carries_the_latest_headline_per_generation():
    doc = sl.BootDocument()
    assert doc.snapshot()["supervisor_load"] is None
    doc.note_supervisor_load({"ts": "2026-10-20T00:00:00+00:00", "max_event_lag_sec": 12.5})
    assert doc.snapshot()["supervisor_load"] == {"ts": "2026-10-20T00:00:00+00:00", "max_event_lag_sec": 12.5}
    doc.reset()
    assert doc.snapshot()["supervisor_load"] is None


def test_a_failing_row_write_never_stops_the_loop_or_a_delivery(monkeypatch, tmp_path, instruments):
    """The REAL loop over a real two-lane bus: the journal refuses every row, yet both ticks
    run to the end, the answer is dispatched once and its delivery membership released."""
    import server
    from supervisor import events as events_mod, workers as workers_mod
    from tests.test_server_shutdown import _run, _supervisor_harness

    rec = _supervisor_harness(monkeypatch, tmp_path, ["ok", "stop"])
    bus, dispatched, attempts = ObservedEventQueue(), [], []
    did = "final:t:0123456789abcdef"
    bus.put({"type": "send_message", "task_id": "t", "chat_id": 1, "text": "answer", "delivery_id": did})
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: bus)
    monkeypatch.setattr(workers_mod, "ensure_workers_healthy",  # every tick finds one heartbeat to serve
                        lambda: bus.put({"type": "task_heartbeat", "task_id": "t", "text": "heartbeat"}))
    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, ctx: dispatched.append(evt["text"]))
    monkeypatch.setattr(sl, "SUPERVISOR_LOAD_WINDOW_SEC", 0.0)

    def refused(path, row, **_kwargs):
        attempts.append(row["type"])
        raise OSError("journal unavailable")

    monkeypatch.setattr(sl, "append_jsonl", refused)
    _run(rec, server)
    assert dispatched == ["answer", "heartbeat", "heartbeat"] and bus.delivery_outstanding(did) is False
    assert attempts == ["supervisor_load", "supervisor_load"], "one row attempt per tick, both ticks finished"
    assert rec.stop.waits == [] and rec.alerts == [], "no tick crashed"
    assert sl.BOOT.snapshot()["supervisor_load"]["ts"], "the health headline does not depend on the journal"
