"""Real rollback-journal contention must not delay live dialogue or multiply rows."""
import contextlib
import sqlite3
import threading
import time
from collections import OrderedDict

import pytest

from ouroboros import dialogue_retry as retry, dialogue_store as ds, store, utils
from ouroboros.chat_references import projection_fields
from supervisor import message_bus as bus
from tests._dialogue import chat_rows, progress_rows


@pytest.fixture(autouse=True)
def isolated_retry(monkeypatch):
    monkeypatch.setattr(retry, "_rows", OrderedDict())
    monkeypatch.setattr(retry, "_bytes", 0)
    monkeypatch.setattr(retry, "_counts", dict(appended=0, retried_ok=0, evicted=0, lost_at_exit=0))
    monkeypatch.setattr(retry, "_eviction_burst", False)
    monkeypatch.setattr(retry, "_wake", None)
    monkeypatch.setattr(retry, "_requested", threading.Event())
    monkeypatch.setattr(utils, "_log_sink", None)


def seed(root, monkeypatch, mode="PERSIST"):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    ds.append(root, "chat", {"ts": "seed", "chat_id": 1, "text": "seed"})
    ds._plan_first(root)
    with ds.transaction(root, write=False) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == mode.lower()
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2


@contextlib.contextmanager
def pinned(root, *, writer=False):
    conn = sqlite3.connect(ds.path(root), isolation_level=None)
    conn.execute("BEGIN IMMEDIATE" if writer else "BEGIN")
    conn.execute("SELECT * FROM dialogue").fetchall()
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def bridge(root, monkeypatch):
    frames = []
    transport = bus.LocalChatBridge({})
    transport._broadcast_fn = frames.append
    monkeypatch.setattr(bus, "DATA_DIR", root)
    monkeypatch.setattr(bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(bus, "get_bridge", lambda: transport)
    monkeypatch.setattr(bus, "publish_event", lambda *_a, **_k: None)
    return frames


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_progress_precedes_a_two_second_reader_and_retries_without_live_copy(tmp_path, monkeypatch, mode):
    seed(tmp_path, monkeypatch, mode)
    frames = bridge(tmp_path, monkeypatch)
    ready = threading.Event()

    def reader():
        with pinned(tmp_path):
            ready.set()
            threading.Event().wait(2)

    thread = threading.Thread(target=reader)
    thread.start()
    assert ready.wait(2)
    try:
        started = time.monotonic()
        bus.send_with_budget(1, "Working", is_progress=True, ts="2026-10-20T00:00:01Z")
        assert time.monotonic() - started < .2
        assert len(frames) == 1 and thread.is_alive()
        assert retry.snapshot()["rows"] == (1 if mode == "PERSIST" else 0)
        if mode == "PERSIST":
            assert progress_rows(tmp_path) == []
            assert "row_address" not in frames[0]
        else:
            assert frames[0]["row_address"] == projection_fields(progress_rows(tmp_path)[0])["row_address"]
    finally:
        thread.join(3)
    retry.drain()
    assert len(progress_rows(tmp_path)) == 1 and len(frames) == 1
    assert progress_rows(tmp_path)[0]["ts"] == frames[0]["ts"]
    assert retry.snapshot()["rows"] == 0


def test_uncertain_commit_is_deduped_in_the_write_transaction(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch)
    row = {"chat_id": 1, "ts": "same", "type": "send_message", "text": "landed"}
    frames = []
    monkeypatch.setattr(utils, "_log_sink", frames.append)
    real_commit = store.commit

    def uncertain(conn, *args, **kwargs):
        real_commit(conn, *args, **kwargs)
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(store, "commit", uncertain)
        assert ds.append_quietly(tmp_path, "progress", row) is None
    assert len(progress_rows(tmp_path)) == 1 and len(frames) == 1
    assert retry.drain() == 1
    assert progress_rows(tmp_path) == [row] and frames == [row]
    # Content, stream and chat are part of the identity, not just the timestamp.
    for changed in ({**row, "text": "different"}, {**row, "chat_id": 2}):
        ds.append(tmp_path, "progress", changed, idempotent=True)
    assert len(progress_rows(tmp_path)) == 3


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
@pytest.mark.parametrize("bound", ["rows", "bytes"])
def test_overflow_evicts_oldest_counts_every_row_and_keeps_live_producers(tmp_path, monkeypatch, caplog, mode, bound):
    seed(tmp_path, monkeypatch, mode)
    monkeypatch.setattr(retry, "MAX_ROWS", 3 if bound == "rows" else 100)
    monkeypatch.setattr(retry, "MAX_BYTES", 900 if bound == "bytes" else 100_000)
    frames = []
    monkeypatch.setattr(utils, "_log_sink", frames.append)
    with pinned(tmp_path, writer=True), caplog.at_level("INFO", logger=retry.__name__):
        started = time.monotonic()
        for n in range(20):
            ds.append_quietly(tmp_path, "progress", {"chat_id": 1, "ts": str(n), "text": "x" * 80})
            fact = retry.snapshot()
            assert fact["rows"] <= fact["max_rows"] and fact["bytes"] <= fact["max_bytes"]
        assert time.monotonic() - started < .5 and len(frames) == 20
        fact = retry.snapshot()
        assert fact["appended"] == 20 and fact["evicted"] == 20 - fact["rows"] > 0
    assert len([r for r in caplog.records if "capacity exceeded" in r.message]) == 1
    retained = fact["rows"]
    assert retry.drain() == retained
    assert sorted(int(r["ts"]) for r in progress_rows(tmp_path)) == list(range(20 - retained, 20))
    assert len(frames) == 20 and retry.snapshot()["retried_ok"] == retained


def test_worker_next_append_drains_and_exit_discloses_remaining_rows(tmp_path, monkeypatch, caplog):
    seed(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    first = {"chat_id": 1, "ts": "first", "text": "worker"}
    with pinned(tmp_path):
        assert ds.append_quietly(tmp_path, "chat", first) is None
    assert ds.append_quietly(tmp_path, "chat", {**first, "ts": "next"}) is not None
    assert retry.snapshot()["retried_ok"] == 1 and retry.snapshot()["rows"] == 0
    assert len(chat_rows(tmp_path)) == 3
    with pinned(tmp_path), caplog.at_level("INFO", logger=retry.__name__):
        ds.append_quietly(tmp_path, "chat", {**first, "ts": "exit"})
        retry.finish()
    assert retry.snapshot()["lost_at_exit"] == 1 and retry.snapshot()["rows"] == 0
    assert "process exit lost 1" in caplog.text


def test_ordinary_outbound_retries_but_required_and_inbound_fail_loudly(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch)
    frames = bridge(tmp_path, monkeypatch)
    with pinned(tmp_path):
        bus.send_with_budget(1, "ordinary", ts="ordinary")
        assert len(frames) == 1 and retry.snapshot()["rows"] == 1
        started = time.monotonic()
        # A required notice keeps its producer's wait (here a short one) and fails loudly to that producer.
        with store.acquisition_budget(0.25), pytest.raises(RuntimeError, match="could not be persisted"):
            bus.send_with_budget(1, "required", ts="required", require_write=True)
        assert .2 <= time.monotonic() - started < 2.0  # the producer's short wait, not the store's own (CI runners are slow)
        assert len(frames) == 1 and retry.snapshot()["rows"] == 1
        with store.acquisition_budget(0), pytest.raises(RuntimeError, match="could not be persisted"):
            bus.accept_chat(1, 1, "inbound", client_message_id="inbound")
    assert retry.drain() == 1
    assert len(chat_rows(tmp_path)) == 2
    bus.send_with_budget(1, "required", ts="required", require_write=True)
    bus.send_with_budget(1, "required", ts="required", require_write=True)
    assert len(chat_rows(tmp_path)) == 3


def test_buffer_copies_original_row_bytes_and_wakes_existing_owner(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch)
    wake = []
    retry.bind_wake(lambda: wake.append(True))
    row = {"chat_id": 1, "ts": "original", "nested": {"text": "original"}}
    with pinned(tmp_path):
        ds.append_quietly(tmp_path, "chat", row)
    row["nested"]["text"] = "mutated"
    assert wake == [True]
    assert retry.drain() == 1
    assert any(r.get("nested") == {"text": "original"} for r in chat_rows(tmp_path))
    assert retry.snapshot()["bytes"] == 0


def test_retry_probe_uses_room_time_index(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch)
    statements = []
    with ds.transaction(tmp_path, write=False) as conn:
        conn.set_trace_callback(statements.append)
    ds.append(tmp_path, "chat", {"chat_id": 1, "ts": "indexed"}, idempotent=True)
    assert any("INDEXED BY dialogue_room_ts" in sql for sql in statements)
    assert statements.index("BEGIN IMMEDIATE") < next(i for i, sql in enumerate(statements) if "INDEXED BY dialogue_room_ts" in sql)


def test_worker_checkpoint_publishes_once_during_contention_then_drains(tmp_path, monkeypatch):
    import queue
    from ouroboros.loop_messages import _emit_checkpoint_event
    from tests.test_producer_checkpoints import SYSTEM_ROW

    seed(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    bus = queue.Queue()
    with pinned(tmp_path):
        started = time.monotonic()
        _emit_checkpoint_event(bus, "worker", logs, SYSTEM_ROW)
        assert time.monotonic() - started < .25
    frame = bus.get_nowait()
    assert bus.empty() and not progress_rows(tmp_path) and retry.snapshot()["rows"] == 1
    retry.drain()
    assert progress_rows(tmp_path) == [frame["data"]] and bus.empty()


def test_a_summary_row_without_a_projection_owner_waits_in_the_buffer(tmp_path, monkeypatch):
    """Host task facts have no retry owner but the buffer; a projection-owned row is retried by its owner."""
    from ouroboros.project_dialogue import append_canonical_task_summary

    seed(tmp_path, monkeypatch)
    facts = {"ts": "facts", "chat_id": 1, "direction": "system", "type": "task_summary", "text": "",
             "summary_kind": "host_task_facts", "summary_id": "task-facts:t1", "task_id": "t1"}
    owned = {**facts, "ts": "owned", "summary_kind": "task_terminal", "summary_id": "task-terminal:t1"}
    with pinned(tmp_path):
        assert append_canonical_task_summary(tmp_path, facts, retained=True) is False
        assert append_canonical_task_summary(tmp_path, owned) is False  # the projection appends it again
        assert retry.snapshot()["rows"] == 1
    assert retry.drain() == 1
    assert [row["summary_id"] for row in chat_rows(tmp_path) if row.get("summary_id")] == ["task-facts:t1"]
    assert append_canonical_task_summary(tmp_path, owned) is True


def test_a_childs_terminal_row_waits_in_the_buffer_and_lands_once(tmp_path, monkeypatch):
    """A child mints no readiness, so nothing settles it again: its terminal row is retained, not dropped."""
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    seed(tmp_path, monkeypatch)
    stored = write_task_result(tmp_path, "child-1", STATUS_COMPLETED, parent_task_id="root-1",
                               root_task_id="root-1", delegation_role="subagent")
    child = {"id": "child-1", "parent_task_id": "root-1", "root_task_id": "root-1",
             "delegation_role": "subagent", "chat_id": 41}
    done = {"chat_id": 41, "status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}
    with pinned(tmp_path):
        assert not append_terminal_task_projection(tmp_path, "child-1", child, stored, done)
        assert retry.snapshot()["rows"] == 1
    assert retry.drain() == 1
    rows = [row for row in chat_rows(tmp_path) if row.get("task_id") == "child-1" and row.get("type") == "task_summary"]
    assert len(rows) == 1


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_an_idle_probe_lands_what_it_can_and_reports_the_rest_without_dropping_it(tmp_path, monkeypatch, mode):
    """The idle answer carries what still cannot land; the worker keeps it (a kept spare retries it at its
    next append), and only the server's retirement of that worker counts it lost (``worker_retired``)."""
    from supervisor.worker_process import _retained_rows_after_drain

    seed(tmp_path, monkeypatch, mode)
    held = {"chat_id": 1, "ts": "2026-10-20T00:00:01Z", "text": "cannot land now"}
    retry.retain(tmp_path, "progress", held)
    with pinned(tmp_path, writer=mode == "WAL"):
        assert _retained_rows_after_drain() == 1
    assert retry.snapshot()["rows"] == 1 and retry.snapshot()["lost_at_exit"] == 0 and progress_rows(tmp_path) == []
    assert _retained_rows_after_drain() == 0
    assert retry.snapshot()["retried_ok"] == 1 and progress_rows(tmp_path) == [held]
