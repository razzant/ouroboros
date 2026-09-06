"""The worker readiness contract: a spawned or respawned slot is unassignable until its child confirms ready.

What is pinned, on fake process objects (no child is ever forked here):

* spawn installs every slot ``reaping`` and hands the whole set to the ONE readiness seam;
* respawn installs its fresh slot the same way, through the same seam, carrying its attempt count;
* the seam opens a slot only on the child's OWN ``worker_ready`` row (matched by pid), verifying the
  booted SHA in the same step; a foreign pid's row does not open it;
* no ``worker_ready`` inside the window -> the child is torn down (process tree), the slot is
  replaced through ``respawn_worker`` and a typed ``worker_ready_timeout`` row names slot, pid,
  wait and reason;
* the replacement loop is bounded: at ``WORKER_READY_MAX_ATTEMPTS`` the slot is parked (kept
  ``reaping``, no further respawn) and the owner is told;
* a child that DIED during boot is released to the crash detector, which already owns death;
* the event reader treats a missing ``events.jsonl`` as an EMPTY read (not written yet at spawn,
  the rotator's rename->touch instant, removed by hand): the watcher keeps polling and the row is
  found on whichever side of a rotation it landed;
* the watcher's own failure is never a parked wave: an exception inside it (the reader, ``load_state``,
  a teardown) releases every slot of the wave still booting with a typed ``worker_ready_released``
  row (``reason=watcher_error``), degrading to the crash detector's ownership;
* the assignment path is unchanged for a ready slot: ``assign_tasks`` skips a booting slot and
  dispatches to the open one, and a slot the seam opened is dispatched to like any other.

Readiness is deliberately NOT process liveness (``proc.is_alive``) and NOT the task idle rail: a
child deadlocked on a lock inherited across fork is alive and holds no task.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ouroboros.utils import append_jsonl


class _FakeProc:
    def __init__(self, pid: int, *, alive: bool = True, exitcode=None):
        self.pid = pid
        self._alive = alive
        self.exitcode = exitcode
        self.joined = False
        self.daemon = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.joined = True


def _rows(path, event_type: str) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == event_type:
            out.append(row)
    return out


@pytest.fixture
def pool(monkeypatch, tmp_path):
    """The pool facade rebound to a throwaway root with no live workers."""
    from supervisor import worker_pool_lifecycle as lifecycle
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "load_state", lambda: {"current_sha": "abc123", "owner_chat_id": 0})
    monkeypatch.setattr(workers, "_record_worker_pids", lambda: None)
    # A prior TestClient lifespan in the same xdist worker leaves the process-lifetime
    # event bus latched (server shutdown calls shutdown_event_q()); these tests pin the
    # readiness seam, never the bus, so the latch is isolated here — the precedent of
    # tests/test_promote_event_transport.py::_isolate_event_bus_shutdown_latch and
    # tests/test_inflight_indicator_seams.py (seen red once in the rc.15 battery).
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(workers=workers, lifecycle=lifecycle, root=tmp_path)


def _wait_for(predicate, timeout: float = 5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# (a) spawn and (b) respawn both install booting slots through the one seam
# ---------------------------------------------------------------------------

def test_spawn_installs_every_slot_booting_and_hands_the_set_to_the_readiness_seam(pool, monkeypatch):
    workers = pool.workers
    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = object()
    created = []

    def make_process(*_args, **_kwargs):
        proc = _FakeProc(1000 + len(created))
        created.append(proc)
        return proc

    fake_ctx.Process.side_effect = make_process
    monkeypatch.setattr(workers, "_CTX", fake_ctx)
    monkeypatch.setattr(workers.mp, "get_context", lambda _method: fake_ctx)
    monkeypatch.setattr(workers, "get_event_q", lambda: object())   # the seam, not the bus
    monkeypatch.setattr(workers, "_EVENT_Q_GENERATION", "test-generation")
    monkeypatch.setattr(workers, "reap_orphaned_workers", lambda: 0)
    handed = []
    monkeypatch.setattr(
        workers, "_verify_worker_sha_after_spawn",
        lambda slots, cursor, *rest: handed.append((dict(slots), cursor, rest)),
    )

    workers.spawn_workers(2)

    assert sorted(workers.WORKERS) == [0, 1]
    assert all(slot.reaping for slot in workers.WORKERS.values()), "a fresh slot must not be assignable"
    assert _wait_for(lambda: bool(handed))
    slots, cursor, rest = handed[0]
    assert slots == {0: workers.WORKERS[0], 1: workers.WORKERS[1]}
    assert isinstance(cursor, tuple) and len(cursor) == 3
    attempt, spawned_at = rest
    assert attempt == 1 and 0 < spawned_at <= time.time()  # the window counts from the child's birth


def test_respawn_installs_the_fresh_slot_booting_through_the_same_seam_with_its_attempt(pool, monkeypatch):
    workers = pool.workers
    old = workers.Worker(wid=3, proc=_FakeProc(111, alive=False), in_q=MagicMock(), busy_task_id=None, reaping=True)
    workers.WORKERS[3] = old
    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = MagicMock()
    fake_ctx.Process.side_effect = lambda *_a, **_k: _FakeProc(222)
    monkeypatch.setattr(workers, "_get_ctx", lambda: fake_ctx)
    monkeypatch.setattr(workers, "get_event_q", lambda: object())
    handed = []
    monkeypatch.setattr(
        workers, "_verify_worker_sha_after_spawn",
        lambda slots, cursor, *rest: handed.append((dict(slots), cursor, rest)),
    )

    assert workers.respawn_worker(3) is True
    fresh = workers.WORKERS[3]
    assert fresh is not old and fresh.reaping is True and fresh.busy_task_id is None
    assert _wait_for(lambda: len(handed) == 1)
    assert handed[0][0] == {3: fresh} and handed[0][2][0] == 1 and 0 < handed[0][2][1] <= time.time()

    assert workers.respawn_worker(3, ready_attempt=2) is True
    assert _wait_for(lambda: len(handed) == 2)
    assert handed[1][0] == {3: workers.WORKERS[3]} and handed[1][2][0] == 2


# ---------------------------------------------------------------------------
# The seam itself, run synchronously on fake slots
# ---------------------------------------------------------------------------

@pytest.fixture
def seam(pool, monkeypatch, request):
    """A short window, a recorded teardown and a recorded respawn around the real seam.

    The cursor is taken over a seeded ``events.jsonl`` (one noise row) unless the test asks,
    through the indirect parameter ``"missing"``, for the log not to exist at spawn.
    """
    lifecycle = pool.lifecycle
    import ouroboros.platform_layer as platform_layer

    killed, respawned, sent = [], [], []
    monkeypatch.setattr(lifecycle, "WORKER_READY_WINDOW_SEC", 0.3)
    monkeypatch.setattr(platform_layer, "kill_pid_tree", lambda pid, **_k: killed.append(pid))
    monkeypatch.setattr(lifecycle, "respawn_worker", lambda wid, **kw: respawned.append((wid, kw)))
    monkeypatch.setattr(pool.workers, "send_with_budget", lambda chat_id, text, **_k: sent.append((chat_id, text)))
    events = pool.root / "logs" / "events.jsonl"
    if getattr(request, "param", "seeded") != "missing":
        append_jsonl(events, {"type": "noise"})
    cursor = lifecycle.events_log_cursor()
    return SimpleNamespace(
        run=lifecycle._verify_worker_sha_after_spawn, cursor=cursor, events=events,
        supervisor=pool.root / "logs" / "supervisor.jsonl",
        killed=killed, respawned=respawned, sent=sent,
    )


def _booting_slot(pool, wid: int, pid: int, **proc_kwargs):
    slot = pool.workers.Worker(wid=wid, proc=_FakeProc(pid, **proc_kwargs), in_q=MagicMock(), busy_task_id=None, reaping=True)
    pool.workers.WORKERS[wid] = slot
    return slot


def test_the_slot_opens_only_on_its_own_worker_ready_row_and_verifies_the_sha(pool, seam):
    slot = _booting_slot(pool, 0, 5001)
    append_jsonl(seam.events, {"type": "worker_ready", "worker_id": 0, "pid": 5001, "git_sha": "abc123"})

    seam.run({0: slot}, seam.cursor, 1)

    assert slot.reaping is False, "the child confirmed ready: the slot is assignable"
    assert seam.killed == [] and seam.respawned == []
    verify = _rows(seam.supervisor, "worker_sha_verify")
    assert len(verify) == 1
    assert verify[0]["ok"] is True and verify[0]["worker_id"] == 0 and verify[0]["worker_pid"] == 5001
    assert verify[0]["slot_opened"] is True and verify[0]["attempt"] == 1
    assert _rows(seam.supervisor, "worker_ready_timeout") == []


def test_a_foreign_pid_row_does_not_open_the_slot(pool, seam):
    slot = _booting_slot(pool, 0, 5001)
    append_jsonl(seam.events, {"type": "worker_ready", "worker_id": 0, "pid": 4999, "git_sha": "abc123"})

    seam.run({0: slot}, seam.cursor, 1)

    assert slot.reaping is True
    assert seam.killed == [5001] and seam.respawned == [(0, {"ready_attempt": 2})]


def test_sha_mismatch_on_the_ready_row_opens_the_slot_and_tells_the_owner(pool, seam, monkeypatch):
    monkeypatch.setattr(pool.workers, "load_state", lambda: {"current_sha": "abc123", "owner_chat_id": 7})
    slot = _booting_slot(pool, 0, 5001)
    append_jsonl(seam.events, {"type": "worker_ready", "worker_id": 0, "pid": 5001, "git_sha": "other"})

    seam.run({0: slot}, seam.cursor, 1)

    assert slot.reaping is False
    assert _rows(seam.supervisor, "worker_sha_verify")[0]["ok"] is False
    assert seam.sent and "SHA mismatch" in seam.sent[0][1]


def test_no_worker_ready_inside_the_window_tears_down_replaces_and_types_the_row(pool, seam):
    slot = _booting_slot(pool, 2, 5002)

    started = time.time()
    seam.run({2: slot}, seam.cursor, 1)

    assert time.time() - started >= 0.3
    assert seam.killed == [5002], "the process tree is torn down the way the pool already does"
    assert slot.proc.joined is True
    assert seam.respawned == [(2, {"ready_attempt": 2})], "replaced through the same respawn path"
    assert slot.reaping is True, "the torn-down slot stays owned until respawn swaps it"
    rows = _rows(seam.supervisor, "worker_ready_timeout")
    assert len(rows) == 1
    row = rows[0]
    assert row["worker_id"] == 2 and row["pid"] == 5002 and row["reason"] == "no_worker_ready"
    assert row["attempt"] == 1 and row["action"] == "respawn" and row["window_sec"] == 0.3
    assert row["waited_sec"] >= 0.3 and row["max_attempts"] >= 2
    assert _rows(seam.supervisor, "worker_sha_verify") == []


def test_the_window_and_the_reported_wait_count_from_the_spawn_instant(pool, seam):
    slot = _booting_slot(pool, 4, 5040)

    started = time.time()
    seam.run({4: slot}, seam.cursor, 1, started - 10.0)

    assert time.time() - started < 0.25, "a window already spent at hand-off does not wait again"
    row = _rows(seam.supervisor, "worker_ready_timeout")[0]
    assert row["waited_sec"] >= 10.0 and seam.killed == [5040]


def test_the_replacement_loop_is_bounded_then_parked_and_reported(pool, seam, monkeypatch):
    monkeypatch.setattr(pool.lifecycle, "WORKER_READY_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(pool.workers, "load_state", lambda: {"current_sha": "abc123", "owner_chat_id": 7})
    first = _booting_slot(pool, 1, 5011)
    seam.run({1: first}, seam.cursor, 1)
    assert seam.respawned == [(1, {"ready_attempt": 2})]

    last = _booting_slot(pool, 1, 5012)
    seam.run({1: last}, seam.cursor, 2)

    assert seam.respawned == [(1, {"ready_attempt": 2})], "no respawn at the bound"
    assert seam.killed == [5011, 5012]
    assert last.reaping is True, "parked: never assignable, never respawned again"
    rows = _rows(seam.supervisor, "worker_ready_timeout")
    assert [row["action"] for row in rows] == ["respawn", "parked"]
    assert rows[1]["attempt"] == 2 and rows[1]["max_attempts"] == 2
    assert seam.sent and seam.sent[0][0] == 7 and "slot 1" in seam.sent[0][1] and "parked" in seam.sent[0][1]


def test_a_child_that_died_during_boot_is_released_to_the_crash_detector(pool, seam):
    slot = _booting_slot(pool, 0, 5003, alive=False, exitcode=1)

    seam.run({0: slot}, seam.cursor, 1)

    assert slot.reaping is False, "released: the crash detector owns process death"
    assert seam.killed == [] and seam.respawned == []
    rows = _rows(seam.supervisor, "worker_ready_released")
    assert len(rows) == 1 and rows[0]["reason"] == "died_during_boot"
    assert rows[0]["worker_id"] == 0 and rows[0]["pid"] == 5003 and rows[0]["exitcode"] == 1


def test_a_slot_the_pool_already_replaced_is_left_alone(pool, seam):
    stale = pool.workers.Worker(wid=0, proc=_FakeProc(5004), in_q=MagicMock(), busy_task_id=None, reaping=True)
    _booting_slot(pool, 0, 5005)  # the pool restarted under the seam: slot 0 is a different object now

    seam.run({0: stale}, seam.cursor, 1)

    assert seam.killed == [] and seam.respawned == []
    assert _rows(seam.supervisor, "worker_ready_timeout") == []


def test_a_generation_change_cannot_interleave_with_the_watcher_s_teardown_and_respawn(pool, monkeypatch):
    """rc.15 review MAJOR 2: ``_replace_unready_slot`` checked the slot's identity,
    dropped the queue lock, tore the child down and only then called
    ``respawn_worker(wid)``. A pool generation change in that gap
    (``kill_workers`` -> ``spawn_workers``: the supervisor's in-process restart)
    installed a FRESH live slot at ``wid``; the stale watcher's respawn then
    found that slot, swapped it out and closed its queue without ever
    terminating its process — a worker gone from WORKERS, its live child an
    orphan (the parent-sentinel lifeline does not fire: the server is alive).

    The teardown and the respawn now run under the ONE lifecycle lock every
    generation change takes (lifecycle -> queue order kept; the RLock lets the
    nested respawn re-enter), so the change WAITS for the watcher and the
    watcher never sees the fresh slot: the real ``respawn_worker`` on a fake
    context, the watcher held inside its teardown while the generation change
    is attempted through the same serializer."""
    import threading

    import ouroboros.platform_layer as platform_layer

    lifecycle, workers = pool.lifecycle, pool.workers
    stale = _booting_slot(pool, 2, 5020)
    fresh = workers.Worker(wid=2, proc=_FakeProc(5021), in_q=MagicMock(), busy_task_id=None, reaping=True)
    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = MagicMock()
    fake_ctx.Process.side_effect = lambda *_a, **_k: _FakeProc(5022)
    monkeypatch.setattr(workers, "_get_ctx", lambda: fake_ctx)
    monkeypatch.setattr(workers, "get_event_q", lambda: object())
    monkeypatch.setattr(workers, "_verify_worker_sha_after_spawn", lambda *_a, **_k: None)
    killed, in_teardown, release = [], threading.Event(), threading.Event()

    def _kill(pid, **_kwargs):
        killed.append(pid)
        in_teardown.set()
        assert release.wait(10), "the test never released the watcher"

    monkeypatch.setattr(platform_layer, "kill_pid_tree", _kill)
    watcher = threading.Thread(
        target=lifecycle._replace_unready_slot, args=(2, stale, 0, time.time(), 1), daemon=True,
    )
    watcher.start()
    assert in_teardown.wait(5), "the watcher never reached its teardown"

    generation_done = threading.Event()

    @lifecycle._serialized_worker_lifecycle
    def _new_generation():
        with lifecycle._queue_lock:
            workers.WORKERS.clear()
            workers.WORKERS[2] = fresh
        generation_done.set()

    changer = threading.Thread(target=_new_generation, daemon=True)
    changer.start()
    interleaved = generation_done.wait(0.5)
    release.set()
    watcher.join(10)
    changer.join(10)
    assert not watcher.is_alive() and generation_done.is_set()

    assert workers.WORKERS[2] is fresh, "the fresh generation's live slot was evicted by the stale watcher"
    fresh.in_q.close.assert_not_called()
    assert killed == [5020], "only the stale child is torn down; the fresh one is never touched"
    assert not interleaved, "the generation change ran inside the watcher's check/teardown/respawn window"


def test_the_first_event_reader_keeps_its_contract_over_the_list_reader(pool, seam):
    lifecycle = pool.lifecycle
    append_jsonl(seam.events, {"type": "worker_ready", "pid": 1})
    append_jsonl(seam.events, {"type": "worker_boot", "pid": 2})
    append_jsonl(seam.events, {"type": "worker_ready", "pid": 3})
    assert [row["pid"] for row in lifecycle._worker_events_since(seam.cursor, "worker_ready")] == [1, 3]
    assert lifecycle._first_worker_event_since(seam.cursor, "worker_ready")["pid"] == 1
    assert lifecycle._first_worker_event_since(seam.cursor)["pid"] == 2
    assert lifecycle._first_worker_event_since(seam.cursor, "absent") is None


# ---------------------------------------------------------------------------
# (c) a missing log is an empty read; a rotation gap loses no row; the watcher never wedges
# ---------------------------------------------------------------------------

_READY_ROW = {"type": "worker_ready", "worker_id": 0, "pid": 5001, "git_sha": "abc123"}


def _counting_reader(lifecycle, monkeypatch, before_poll):
    """Wrap the real reader: ``before_poll(n, real)`` runs ahead of poll ``n`` (1-based)."""
    real = lifecycle._worker_events_since
    polls = []

    def reader(cursor, event_type):
        polls.append(cursor)
        before_poll(len(polls), lambda: real(cursor, event_type))
        return real(cursor, event_type)

    monkeypatch.setattr(lifecycle, "_worker_events_since", reader)
    return polls


@pytest.mark.parametrize("seam", ["missing"], indirect=True)
def test_a_missing_events_log_at_spawn_is_an_empty_read_until_the_child_writes_its_row(pool, seam, monkeypatch):
    lifecycle = pool.lifecycle
    assert not seam.events.exists() and seam.cursor == (0, 0, 0), "missing log = a zeroed cursor"
    assert lifecycle._worker_events_since(seam.cursor, "worker_ready") == [], "an empty read, never None"
    monkeypatch.setattr(lifecycle, "WORKER_READY_WINDOW_SEC", 5.0)
    slot = _booting_slot(pool, 0, 5001)

    def before_poll(n, _read):
        if n == 3:  # the child creates the log with its row after two empty polls
            append_jsonl(seam.events, _READY_ROW)

    polls = _counting_reader(lifecycle, monkeypatch, before_poll)
    seam.run({0: slot}, seam.cursor, 1)

    assert len(polls) == 3 and slot.reaping is False, "the watcher kept polling and opened the slot"
    assert seam.killed == [] and seam.respawned == []
    assert _rows(seam.supervisor, "worker_ready_timeout") == []
    assert _rows(seam.supervisor, "worker_ready_released") == []
    verify = _rows(seam.supervisor, "worker_sha_verify")
    assert len(verify) == 1 and verify[0]["slot_opened"] is True and verify[0]["ok"] is True


@pytest.mark.parametrize("row_lands", ["in_the_rotated_segment", "in_the_new_live_file"])
def test_a_rotation_gap_during_polling_is_an_empty_read_and_the_row_is_found_on_either_side(
    pool, seam, monkeypatch, row_lands,
):
    """``rotate_jsonl_log_if_needed`` renames the live file, then touches a fresh one; a poll
    between the two sees no live file. The child's row is durable on one side of the rename."""
    lifecycle = pool.lifecycle
    monkeypatch.setattr(lifecycle, "WORKER_READY_WINDOW_SEC", 5.0)
    slot = _booting_slot(pool, 0, 5001)
    archive = pool.root / "archive" / "events_20260905T000000.jsonl"
    archive.parent.mkdir()
    gap_reads = []

    def before_poll(n, read):
        if n != 2:
            return
        if row_lands == "in_the_rotated_segment":
            append_jsonl(seam.events, _READY_ROW)  # appended under the lock, just before the rename
        os.replace(seam.events, archive)  # the rotator's rename ...
        gap_reads.append(read())  # ... and a poll landing before its touch
        seam.events.touch()
        if row_lands == "in_the_new_live_file":
            append_jsonl(seam.events, _READY_ROW)  # appended just after the touch

    polls = _counting_reader(lifecycle, monkeypatch, before_poll)
    seam.run({0: slot}, seam.cursor, 1)

    assert gap_reads == [[]], "the gap is an empty read, not None and not an error"
    assert len(polls) == 2 and slot.reaping is False, "the row was found on the next read"
    assert seam.killed == [] and seam.respawned == []
    assert _rows(seam.supervisor, "worker_ready_timeout") == []
    assert _rows(seam.supervisor, "worker_sha_verify")[0]["worker_pid"] == 5001
    assert _rows(archive, "noise") and lifecycle._first_worker_event_since(seam.cursor, "noise") is None, \
        "the pre-cursor noise row is in the rotated segment and stays excluded: the offset is honoured there"


@pytest.mark.parametrize("failing", ["_worker_events_since", "load_state", "kill_pid_tree"])
def test_a_watcher_failure_releases_the_whole_wave_to_the_crash_detector_with_a_typed_row(
    pool, seam, monkeypatch, caplog, failing,
):
    lifecycle = pool.lifecycle

    def boom(*_args, **_kwargs):
        raise RuntimeError("events log unreadable")

    if failing == "load_state":
        monkeypatch.setattr(pool.workers, "load_state", boom)
    elif failing == "kill_pid_tree":  # the end-of-window teardown itself raises
        import ouroboros.platform_layer as platform_layer

        monkeypatch.setattr(platform_layer, "kill_pid_tree", boom)
    else:
        monkeypatch.setattr(lifecycle, failing, boom)
    first, second = _booting_slot(pool, 0, 5001), _booting_slot(pool, 1, 5002)

    with caplog.at_level(logging.ERROR, logger="supervisor.worker_pool_lifecycle"):
        seam.run({0: first, 1: second}, seam.cursor, 1)  # the seam itself never raises

    assert first.reaping is False and second.reaping is False, "released: the crash detector owns them now"
    assert seam.respawned == [], "no respawn: nothing about the children is known"
    rows = _rows(seam.supervisor, "worker_ready_released")
    assert [(r["worker_id"], r["pid"], r["reason"], r["error_type"], r["attempt"], r["slot_released"]) for r in rows] == [
        (0, 5001, "watcher_error", "RuntimeError", 1, True),
        (1, 5002, "watcher_error", "RuntimeError", 1, True),
    ]
    assert all("events log unreadable" in r["error"] and r["waited_sec"] >= 0 for r in rows)
    assert any(rec.levelno == logging.ERROR and rec.exc_info for rec in caplog.records), "logged with the traceback"
    if failing == "kill_pid_tree":
        assert [r["worker_id"] for r in _rows(seam.supervisor, "worker_ready_timeout")] == [0], "the timeout row precedes the raise"
    else:
        assert _rows(seam.supervisor, "worker_ready_timeout") == []


def test_a_watcher_failure_after_one_slot_opened_releases_only_the_slots_still_booting(pool, seam, monkeypatch):
    lifecycle = pool.lifecycle
    opened, booting = _booting_slot(pool, 0, 5001), _booting_slot(pool, 1, 5002)
    replaced = pool.workers.Worker(wid=2, proc=_FakeProc(5003), in_q=MagicMock(), busy_task_id=None, reaping=True)
    _booting_slot(pool, 2, 5004)  # the pool replaced slot 2 under the watcher
    append_jsonl(seam.events, _READY_ROW)

    def before_poll(n, _read):
        if n == 2:
            raise OSError("poll failed")

    _counting_reader(lifecycle, monkeypatch, before_poll)
    seam.run({0: opened, 1: booting, 2: replaced}, seam.cursor, 1)

    assert opened.reaping is False and booting.reaping is False
    assert replaced.reaping is True and pool.workers.WORKERS[2].reaping is True, "a slot no longer ours is left alone"
    assert [r["worker_id"] for r in _rows(seam.supervisor, "worker_sha_verify")] == [0]
    rows = _rows(seam.supervisor, "worker_ready_released")
    assert [(r["worker_id"], r["error_type"], r["slot_released"]) for r in rows] == [(1, "OSError", True), (2, "OSError", False)]


# ---------------------------------------------------------------------------
# (e) the assignment path: a booting slot is skipped, a ready one is unchanged
# ---------------------------------------------------------------------------

def test_assignment_skips_a_booting_slot_and_dispatches_to_an_open_one_unchanged(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kwargs: 10.0)

    sent = {0: [], 1: []}
    booting = SimpleNamespace(wid=0, busy_task_id=None, reaping=True, in_q=SimpleNamespace(put=lambda t: sent[0].append(dict(t))))
    ready = SimpleNamespace(wid=1, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=lambda t: sent[1].append(dict(t))))
    workers.WORKERS[0] = booting
    workers.WORKERS[1] = ready
    workers.PENDING.append({"id": "first", "type": "task", "chat_id": 0, "priority": 1})

    workers.assign_tasks()

    assert [t["id"] for t in sent[1]] == ["first"] and sent[0] == []
    assert workers.RUNNING["first"]["worker_id"] == 1 and ready.busy_task_id == "first"
    assert booting.busy_task_id is None

    # The seam opened the slot: it is dispatched to like any other.
    booting.reaping = False
    workers.PENDING.append({"id": "second", "type": "task", "chat_id": 0, "priority": 1})
    workers.assign_tasks()
    assert [t["id"] for t in sent[0]] == ["second"]
    assert workers.RUNNING["second"]["worker_id"] == 0
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
