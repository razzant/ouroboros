"""The tick's own write leaves the supervisor thread (S4 step 4; ARCHITECTURE §5 "Tick and locks", §10.24).

``publish_queue_changes`` captures under Q as before; with the writer started (``start_queue_writer``, the server's
loop right after ``tick_publisher_started``) the capture is handed to one daemon thread that encodes and writes it,
so event dispatch never waits on ``rename`` and ``open``. The tick answers False once for each failed write the writer
completed, one tick later, and that tick's capture is the complete rewrite (FU1/R2 unchanged); a retry still on the
disk is not another failure (the third consecutive ends the generation). The writer holds
the newest capture only (a slow disk never queues a backlog), an older capture a barrier overtook is skipped under
the publish lock, a capture of an ended generation is never written, the writer's or a synchronous barrier's (#3149),
nor does its refresh extend the next generation's certificate or its tick answer that generation's failures; the
switch itself (``queue.init``) waits for a write with Q free, and a witnessed barrier holding no work reads the saved
queue at its last check, off Q. The writer ends with the loop that fed it (``stop_queue_writer``): the doors after
the loop publish synchronously where they stand. Without the writer (startup, shutdown, tests) every path is the
synchronous one it was.
"""

from __future__ import annotations

import inspect
import json
import threading

import pytest

from supervisor import queue, queue_publication, queue_snapshot
from tests.test_queue_publication_followup4 import _read, _seed, _task, published as published  # noqa: F401 -- fixture

WRITER = "queue-publisher"


class _Gate:
    """The writer thread stops at the gate (``reached`` tells the test it is there) until the test releases it."""

    def __init__(self) -> None:
        self.reached, self.release = threading.Event(), threading.Event()

    def hold(self) -> None:
        if threading.current_thread().name == WRITER and not self.reached.is_set():
            self.reached.set()
            assert self.release.wait(5), "the test released the gate"

    def passed(self) -> None:
        assert self.reached.wait(5), "the writer thread reached the gate"


def _gated_encoder(monkeypatch) -> _Gate:
    """The writer thread holds at the gate while encoding its capture (before the publish lock)."""
    gate, documents = _Gate(), queue_publication._documents

    def encode(capture):
        gate.hold()
        return documents(capture)

    monkeypatch.setattr(queue_publication, "_documents", encode)
    return gate


def _recording_writes(monkeypatch, *, gate=None, fail_once=False):
    """Every document write: (name, thread). ``gate``: the writer thread holds at it before its first write.
    ``fail_once``: the first write answers False (a failed publication)."""
    writes, real = [], queue.atomic_write_text
    state = {"failed": False}

    def write(target, content, *args, **kwargs):
        if gate is not None:
            gate.hold()
        writes.append((target.name, threading.current_thread().name))
        if fail_once and not state["failed"]:
            state["failed"] = True
            return False
        return real(target, content, *args, **kwargs)

    monkeypatch.setattr(queue, "atomic_write_text", write)
    return writes


def _snapshot_writes(writes):
    return [thread for name, thread in writes if name == "queue_snapshot.json"]


def test_with_the_writer_the_tick_hands_its_capture_over_and_the_document_lands_off_the_loop(published, monkeypatch):
    root, _now = _seed(published)
    writes = _recording_writes(monkeypatch)
    queue_publication.start_queue_writer()
    assert queue_publication.writer_alive()
    queue.PENDING.append(_task("p1"))
    assert queue.persist_queue_snapshot("admitted") is None, "a mark with a live tick publisher"
    assert queue_publication.publish_queue_changes() is True
    assert queue_publication.drain_writer()
    assert _snapshot_writes(writes) == [WRITER], "written by the publisher thread, not the loop"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["p1"]
    assert _read(root, "queue.alive")["generation"] == _read(root, "queue_snapshot.json")["generation"]


def test_without_the_writer_the_tick_writes_where_it_stands(published, monkeypatch):
    root, _now = _seed(published)
    writes = _recording_writes(monkeypatch)
    assert not queue_publication.writer_alive()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    assert queue_publication.publish_queue_changes() is True
    assert _snapshot_writes(writes) == [threading.current_thread().name]
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["p1"]


def test_a_slow_disk_never_queues_a_backlog_the_newest_capture_lands(published, monkeypatch):
    """Three ticks while the first write is still on the disk: the writer holds one capture (the newest), writes
    it once the first lands, and the loop never waited."""
    root, _now = _seed(published)
    gate = _Gate()
    writes = _recording_writes(monkeypatch, gate=gate)
    queue_publication.start_queue_writer()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("one")
    assert queue_publication.publish_queue_changes() is True  # the writer takes it and holds at the gate
    gate.passed()
    for task_id in ("p2", "p3"):
        queue.PENDING.append(_task(task_id))
        queue.persist_queue_snapshot(task_id)
        assert queue_publication.publish_queue_changes() is True  # answered at once: no wait on the disk
    assert queue_publication._WRITER.capture is not None and queue_publication._WRITER.capture["reason"] == "p3"
    gate.release.set()
    assert queue_publication.drain_writer()
    assert _snapshot_writes(writes) == [WRITER, WRITER], "the first capture, then the newest; p2's was superseded"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["p1", "p2", "p3"]


def test_a_failed_write_shows_one_tick_later_and_that_ticks_capture_is_the_complete_rewrite(published, monkeypatch):
    root, _now = _seed(published)
    writes = _recording_writes(monkeypatch, fail_once=True)
    queue_publication.start_queue_writer()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    assert queue_publication.publish_queue_changes() is True, "the capture is the writer's; nothing landed yet"
    assert queue_publication.drain_writer()
    assert queue_publication._STATE["write_failed"] and not (root / "state/queue.alive").exists(), "revoked, unknown"
    assert queue_publication.publish_queue_changes() is False, "the failure shows on the next tick, whose capture retries"
    assert queue_publication.drain_writer()
    assert not queue_publication._STATE["write_failed"]
    rewritten = [name for name, thread in writes[1:] if thread == WRITER]
    assert set(rewritten) >= {"queue_snapshot.json", "queue_dispatch.json"}, "every document, under a new generation"
    assert _read(root, "queue.alive")["generation"] == _read(root, "queue_snapshot.json")["generation"]
    assert queue_publication.publish_queue_changes() is True


def test_a_barrier_that_overtakes_a_tick_capture_leaves_the_older_capture_unwritten(published, monkeypatch):
    """The writer is encoding capture N when a barrier publishes N+1 where its caller stands: N is skipped under the
    publish lock (``published_seq``), the disk shows N+1, and nothing older ever overwrites it."""
    root, _now = _seed(published)
    writes = _recording_writes(monkeypatch)
    gate = _gated_encoder(monkeypatch)
    queue_publication.start_queue_writer()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("tick")
    assert queue_publication.publish_queue_changes() is True  # the writer takes N and holds in the encoder
    gate.passed()
    queue.PENDING.append(_task("p2"))
    assert queue.publish_queue_snapshot(reason="barrier") is True  # N+1, synchronous, on this thread
    gate.release.set()
    assert queue_publication.drain_writer()
    assert _snapshot_writes(writes) == [threading.current_thread().name], "N was never written"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["p1", "p2"]


def test_startup_cannot_overwrite_a_published_presence_admission(published):
    root, _now = published
    with queue._queue_lock:
        startup = queue_publication._capture("startup")
        queue.RUNNING["turn"] = {"task": _task("turn", type="presence"), "lane": "conversation"}
    assert queue.publish_queue_snapshot(reason="conversation_turn_admitted") is True
    admitted = _read(root, "queue_snapshot.json")
    assert [row["id"] for row in admitted["running"]] == ["turn"]
    assert queue_publication._publish(startup) is False, "an overtaken forced capture is refused; its door recaptures"
    assert _read(root, "queue_snapshot.json") == admitted


def test_a_capture_of_an_ended_generation_is_never_written(published, monkeypatch):
    """``reset`` (a new queue generation) while the writer still holds the old generation's capture: the write is
    refused under the publish lock and the documents stay the ones the next generation owns."""
    root, _now = _seed(published)
    writes = _recording_writes(monkeypatch)
    before = (root / "state/queue_snapshot.json").read_bytes()
    gate = _gated_encoder(monkeypatch)
    queue_publication.start_queue_writer()
    old_writer = queue_publication._WRITER
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("tick")
    assert queue_publication.publish_queue_changes() is True
    gate.passed()  # the writer holds the ended generation's capture, encoding it
    switched, stop_writer = threading.Event(), old_writer.stop

    def stop_after_switch():
        switched.set()  # reset replaced this writer before asking it to stop
        stop_writer()

    monkeypatch.setattr(old_writer, "stop", stop_after_switch)
    ending = threading.Thread(target=queue_publication.reset)
    ending.start()
    try:
        assert switched.wait(5)
    finally:
        gate.release.set()
        ending.join(5)
    assert not ending.is_alive() and not old_writer.alive() and queue_publication._WRITER is not old_writer
    assert _snapshot_writes(writes) == [] and (root / "state/queue_snapshot.json").read_bytes() == before


def _next_generation(root, monkeypatch):
    """The same process's next queue generation: the real ``queue.init`` (the globals it rebinds restored after)."""
    for name in ("OWNER_PID", "FINALIZATION_GRACE_SEC"):
        monkeypatch.setattr(queue, name, getattr(queue, name))
    monkeypatch.setattr(queue, "PRIOR_DIRECT_ROOTS", {})

    def begin(*rows, landed=True):
        queue.init(root)  # one switch under Q and the publish lock: no tick publisher, a new writer, nothing published
        queue_snapshot._RESTORED.add(str(queue.QUEUE_SNAPSHOT_PATH.absolute()))  # its restore took custody
        queue.PENDING[:] = [_task(task_id) for task_id in rows]
        assert queue.persist_queue_snapshot("startup") is landed

    return begin


def test_a_delayed_synchronous_barrier_never_lands_over_the_next_generation(published, monkeypatch):
    """#3149: a barrier captured under Q is still encoding when the same process's ``queue.init`` publishes its next
    generation: refused at its last check, it writes nothing, its caller is told so, and the next generation's later
    captures, numbered below the stale one, still land."""
    root, _now = _seed(published)
    begin = _next_generation(root, monkeypatch)
    for task_id in ("o1", "o2", "o3"):  # the ending generation numbered more captures than the next one will
        queue.PENDING.append(_task(task_id))
        assert queue.publish_queue_snapshot(reason=task_id) is True
    documents = queue_publication._documents

    def encode(capture):
        if capture["reason"] == "late":  # captured; the publish lock not taken yet
            begin("new")
        return documents(capture)

    monkeypatch.setattr(queue_publication, "_documents", encode)
    queue.PENDING.append(_task("departed"))
    assert queue.publish_queue_snapshot(reason="late") is False, "the ended generation's capture is no durability"
    document = _read(root, "queue_snapshot.json")
    assert [row["id"] for row in document["pending"]] == ["new"]
    assert _read(root, "queue.alive")["generation"] == document["generation"]
    queue.PENDING.append(_task("after"))
    assert queue.publish_queue_snapshot(reason="after") is True
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["new", "after"]


def test_a_refresh_measured_before_queue_init_extends_no_certificate_of_the_next_generation(published, monkeypatch):
    """The ending generation's quiet tick measured the pool; after ``queue.init`` and the next generation's own
    publication that measurement certifies nothing, though its capture number is not below the next one's; the
    next generation's own measurement refreshes as before."""
    root, now = _seed(published)
    begin = _next_generation(root, monkeypatch)

    def measured():
        with queue._queue_lock:
            return (queue_publication._pool_counts(), queue_publication._STATE["capture_seq"],
                    queue_publication.publication_identity())

    ended = measured()
    begin("new")
    certificate = (root / "state/queue.alive").read_bytes()
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC  # due: any extension would rewrite it
    assert queue_publication._refresh_alive(*ended) is None
    assert (root / "state/queue.alive").read_bytes() == certificate
    assert queue_publication._refresh_alive(*measured()) is True
    assert (root / "state/queue.alive").read_bytes() != certificate


def test_a_barrier_witness_names_the_generation_and_destination_its_caller_admitted_under(published, monkeypatch):
    """``publication_identity`` taken under Q with an admission: while both hold the barrier publishes; after
    ``queue.init``, or with the destination moved, the same witness marks, captures and writes nothing (False)."""
    root, _now = _seed(published)
    begin = _next_generation(root, monkeypatch)
    with queue._queue_lock:
        queue.PENDING.append(_task("admitted"))
        witness = queue_publication.publication_identity()
    assert queue.publish_queue_snapshot(reason="admitted", expected_identity=witness) is True
    begin("new")
    document, revision = (root / "state/queue_snapshot.json").read_bytes(), queue_publication.queue_revision()
    assert queue.publish_queue_snapshot(reason="withdrawn", expected_identity=witness) is False
    assert queue_publication.queue_revision() == revision, "nothing marked for the next generation"
    assert (root / "state/queue_snapshot.json").read_bytes() == document
    with queue._queue_lock:
        queue.PENDING.append(_task("now"))
        current = queue_publication.publication_identity()
    assert queue.publish_queue_snapshot(reason="admitted", expected_identity=current) is True
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["new", "now"]
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", root / "elsewhere" / "queue_snapshot.json")
    assert queue.publish_queue_snapshot(reason="moved", expected_identity=current) is False
    assert not (root / "elsewhere").exists()


class _WatchedLock:
    """The publish lock, telling the test when a thread starts to wait for it."""

    def __init__(self) -> None:
        self.lock, self.waiting = threading.Lock(), threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking:
            self.waiting.set()
        return self.lock.acquire(blocking, timeout)

    def release(self) -> None:
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def test_queue_init_waits_for_a_write_outside_q_and_switches_in_one_step(published, monkeypatch):
    """While a write holds the publish lock, ``queue.init`` waits for it with Q free: work under Q goes on and a
    witness taken there names the old generation whole; then drive, owner, path, writer and state switch together."""
    root, _now = _seed(published)
    begin = _next_generation(root, monkeypatch)
    watched = _WatchedLock()
    monkeypatch.setattr(queue_publication, "_PUBLISH_LOCK", watched)
    with queue._queue_lock:
        before = queue_publication.publication_identity()
    watched.lock.acquire()  # a write in flight
    starting = threading.Thread(target=begin, args=("new",), name="next-generation")
    starting.start()
    try:
        assert watched.waiting.wait(5), "init reached the publish lock"
        assert queue._queue_lock.acquire(timeout=5), "Q is not held across that wait"
        try:
            assert queue_publication.publication_identity() == before and queue.INITIALIZED
        finally:
            queue._queue_lock.release()
    finally:
        watched.lock.release()
        starting.join(5)
    assert not starting.is_alive()
    with queue._queue_lock:
        after = queue_publication.publication_identity()
    assert after != before and after[1] == root / "state" / "queue_snapshot.json"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["new"]


@pytest.mark.parametrize("threaded", [True, False], ids=["writer", "synchronous"])
def test_a_tick_across_queue_init_leaves_the_next_generations_failure_to_its_own_tick(published, monkeypatch,
                                                                                         threaded):
    """The ending generation's tick answers failures in the generation it read: ``queue.init`` once it left Q (as it
    hands its capture over, or while it encodes it) and the next generation's failed first write leave that failure
    to the next generation's loop, which counts it once on its own tick and then publishes."""
    root, _now = _seed(published)
    begin = _next_generation(root, monkeypatch)

    def next_generation_fails() -> None:
        with monkeypatch.context() as failing:
            failing.setattr(queue, "atomic_write_text", lambda *_a: False)
            begin("new", landed=False)

    if threaded:
        queue_publication.start_queue_writer()
        monkeypatch.setattr(queue_publication._WRITER, "offer", lambda _capture: next_generation_fails())
    else:
        documents = queue_publication._documents

        def encode(capture):
            if capture["reason"] == "admitted":
                next_generation_fails()
            return documents(capture)

        monkeypatch.setattr(queue_publication, "_documents", encode)
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    # The ending loop: its writer failed nothing; its own synchronous capture is refused, not landed.
    assert queue_publication.publish_queue_changes() is threaded
    queue_publication.tick_publisher_started()  # the next generation's loop
    queue_publication.start_queue_writer()
    assert queue_publication.publish_queue_changes() is False, "its own failed write, answered once"
    assert queue_publication.drain_writer()
    assert queue_publication.publish_queue_changes() is True
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["new"]


def test_a_witnessed_barrier_without_work_reads_the_saved_queue_at_its_last_check_off_q(published, monkeypatch):
    """A witnessed withdrawal holding no work leaves custody to the saved file, read under the publish lock and not
    under Q: a foreign saved queue is kept (None, its bytes stay), an absent file is a plain write, and this
    generation's own snapshot that landed without its certificate is recognized and replaced."""
    root, _now = published
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    reads, keeps = [], queue_publication._keeps_saved_queue

    def watched(path, *held):
        if not any(held):
            reads.append(queue._queue_lock._is_owned())
        return keeps(path, *held)

    monkeypatch.setattr(queue_publication, "_keeps_saved_queue", watched)

    def witness():
        with queue._queue_lock:
            return queue_publication.publication_identity()

    foreign = json.dumps({"pending": [{"id": "saved", "task": {"id": "saved"}}], "running": [], "generation": "old"})
    (state / "queue_snapshot.json").write_text(foreign, encoding="utf-8")
    assert queue.publish_queue_snapshot(reason="withdrawn", expected_identity=witness()) is None
    assert (state / "queue_snapshot.json").read_text(encoding="utf-8") == foreign, "a foreign saved queue stays"
    (state / "queue_snapshot.json").unlink()
    assert queue.publish_queue_snapshot(reason="withdrawn", expected_identity=witness()) is True
    assert _read(root, "queue_snapshot.json")["pending"] == [], "absent: written, not assumed"
    queue_publication.reset()  # a generation that has published nothing of its own yet
    queue.PENDING.append(_task("admitted"))
    with monkeypatch.context() as failing:
        failing.setattr(queue_publication, "_write_alive", lambda *_a: False)  # the snapshot lands, its certificate not
        assert queue.publish_queue_snapshot(reason="admitted", expected_identity=witness()) is False
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["admitted"]
    queue.PENDING.clear()
    assert queue.publish_queue_snapshot(reason="withdrawn", expected_identity=witness()) is True
    assert _read(root, "queue_snapshot.json")["pending"] == [] and _read(root, "queue.alive")["generation"]
    assert reads == [False, False, False], "every saved-file custody read ran off Q"


def test_a_quiet_tick_refreshes_the_certificate_on_the_writer(published, monkeypatch):
    root, now = _seed(published)
    refreshed, real = [], queue_publication._write_alive
    monkeypatch.setattr(queue_publication, "_write_alive",
                        lambda path, content: refreshed.append(threading.current_thread().name) or real(path, content))
    queue_publication.start_queue_writer()
    assert queue_publication.publish_queue_changes() is True and queue_publication.drain_writer()
    assert refreshed == [], "nothing moved and the certificate is fresh: nothing to write"
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC + 1
    assert queue_publication.publish_queue_changes() is True and queue_publication.drain_writer()
    assert refreshed == [WRITER]
    assert _read(root, "queue.alive")["generation"] == _read(root, "queue_snapshot.json")["generation"]


def test_stopping_the_writer_returns_the_tick_to_its_synchronous_write(published, monkeypatch):
    root, _now = _seed(published)
    writes = _recording_writes(monkeypatch)
    queue_publication.start_queue_writer()
    thread = queue_publication._WRITER.thread
    queue_publication.stop_queue_writer()
    assert not queue_publication.writer_alive() and not thread.is_alive(), "joined: nothing leaks past the loop"
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    assert queue_publication.publish_queue_changes() is True
    assert _snapshot_writes(writes) == [threading.current_thread().name]
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["p1"]


def test_the_supervisor_loop_starts_the_writer_with_its_tick_publisher_and_stops_it_at_every_exit():
    """Source-order pin (the shutdown tests' style): the writer starts right after the tick publisher is named and
    stops on the shared exit line every loop exit reaches (restart, shutdown, crash death), beside the watchdog."""
    import server

    source = inspect.getsource(server._run_supervisor)
    loop = source.index("while not _restart_requested.is_set()")
    exit_line = source.index("_watchdog_stop.set()  # every ordinary exit")
    assert source.index("tick_publisher_started()") < source.index("start_queue_writer()") < loop
    assert loop < exit_line < source.index("stop_queue_writer()") < source.rindex("_supervisor_thread = None")


# ---- a failed write is answered once, not on every tick while its retry is on the disk (review F2) ----------------

def _writer_writes(monkeypatch, *, fail_first: int = 0, hold: threading.Event | None = None):
    """The writer thread's document writes: the first ``fail_first`` answer False; the ones after that wait at
    ``hold`` before they land (a retry that sits on a slow disk). Other threads write as they stand."""
    real, state = queue.atomic_write_text, {"calls": 0}

    def write(target, content, *args, **kwargs):
        if threading.current_thread().name == WRITER:
            state["calls"] += 1
            if state["calls"] <= fail_first:
                return False
            if hold is not None:
                assert hold.wait(10), "the test released the slow disk"
        return real(target, content, *args, **kwargs)

    monkeypatch.setattr(queue, "atomic_write_text", write)
    return state


def test_one_failed_write_and_a_slow_retry_answer_false_once(published, monkeypatch):
    """One failed write, then its retry sits on the disk across three ticks: the loop counts one crash, not three
    (``server._run_supervisor`` ends the generation at three consecutive), and the retry that lands is answered True."""
    _root, _now = _seed(published)
    release = threading.Event()
    _writer_writes(monkeypatch, fail_first=1, hold=release)
    queue_publication.start_queue_writer()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    answers = [queue_publication.publish_queue_changes()]
    assert queue_publication.drain_writer()  # the one failed write
    for _ in range(3):  # three ticks while the single retry is still on the disk
        answers.append(queue_publication.publish_queue_changes())
    release.set()
    assert queue_publication.drain_writer()
    answers.append(queue_publication.publish_queue_changes())
    assert answers == [True, False, True, True, True]
    assert not queue_publication._STATE["write_failed"]
    assert [row["id"] for row in _read(_root, "queue_snapshot.json")["pending"]] == ["p1"]


def test_three_consecutive_failed_writes_still_answer_false_three_times(published, monkeypatch):
    """Each failed write the writer completed is one False, on the tick after it: three in a row still end the
    loop, and the write that lands after them is answered True."""
    root, _now = _seed(published)
    _writer_writes(monkeypatch, fail_first=3)
    queue_publication.start_queue_writer()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    answers = [queue_publication.publish_queue_changes()]
    for _ in range(4):
        assert queue_publication.drain_writer()
        answers.append(queue_publication.publish_queue_changes())
    assert answers == [True, False, False, False, True]
    assert _read(root, "queue.alive")["generation"] == _read(root, "queue_snapshot.json")["generation"]


# ---- a disk that keeps failing slowly still ends the generation (round-2 review N1) ----------------------------------------

def _loop_reaches_its_cutoff(answers) -> bool:
    """``server._run_supervisor``'s exact rule over the tick's answers: a False is a crash, anything else resets the
    count, three consecutive crashes end the generation (``loop_failed``)."""
    streak, reached = 0, False
    for answer in answers:
        streak = streak + 1 if answer is False else 0
        reached |= streak >= 3
    return reached


def _slow_failing_disk(monkeypatch, *, failures: int):
    """Every writer write waits until the test releases it, then the first ``failures`` fail; ``reached[i]`` tells the
    test write ``i`` is on the disk. Writes after the failures land."""
    real = queue.atomic_write_text
    reached, releases = [threading.Event() for _ in range(failures + 1)], [threading.Event() for _ in range(failures + 1)]
    state = {"calls": 0}

    def write(target, content, *args, **kwargs):
        if threading.current_thread().name != WRITER:
            return real(target, content, *args, **kwargs)
        index, state["calls"] = state["calls"], state["calls"] + 1
        if index < len(reached):
            reached[index].set()
            assert releases[index].wait(10), "the test released the slow disk"
        if index < failures:
            return False
        return real(target, content, *args, **kwargs)

    monkeypatch.setattr(queue, "atomic_write_text", write)
    return reached, releases, state


def _ticks_through_slow_failures(published, monkeypatch, *, failures: int, ticks_after_last: int):
    """Astra's probe shape: one tick while each write sits on the disk, one after its failure; then
    ``ticks_after_last`` more while the write after the last failure is still on the disk."""
    _seed(published)
    reached, releases, state = _slow_failing_disk(monkeypatch, failures=failures)
    queue_publication.start_queue_writer()
    queue.PENDING.append(_task("p1"))
    queue.persist_queue_snapshot("admitted")
    answers = [queue_publication.publish_queue_changes()]
    try:
        for index in range(failures):
            assert reached[index].wait(10)
            answers.append(queue_publication.publish_queue_changes())  # a tick while this write is on the disk
            releases[index].set()  # ...then it fails
            assert reached[index + 1].wait(10)
            answers.append(queue_publication.publish_queue_changes())
        for _ in range(ticks_after_last):
            answers.append(queue_publication.publish_queue_changes())
    finally:
        for release in releases:
            release.set()
        assert queue_publication.drain_writer()
    return answers, state


def test_three_slow_consecutive_failed_writes_end_the_loop_through_the_servers_own_count(published, monkeypatch):
    """A disk that fails every write, slowly: after three consecutive failures the publisher answers False on every
    tick until a write lands, so the loop's three-consecutive rule is reached while the fourth attempt is still on the
    disk (no fourth failure needed); a disk failing fast still ends it after three ticks."""
    answers, _state = _ticks_through_slow_failures(published, monkeypatch, failures=3, ticks_after_last=2)
    assert answers[:7] == [True, True, False, True, False, True, False], "each failure is answered once, on its tick"
    assert answers[7:] == [False, False], "three in a row: every tick is a crash until a write lands"
    assert _loop_reaches_its_cutoff(answers)
    assert queue_publication._STATE["failed_writes"] == 3, "no fourth failure was needed"
    assert queue_publication.publish_queue_changes() is True, "the fourth attempt landed: the streak is over"


def test_two_slow_consecutive_failed_writes_and_a_slow_success_never_end_the_loop(published, monkeypatch):
    """Below three, a retry on the disk is not a crash: two failures, one slow attempt that lands, no shutdown."""
    answers, _state = _ticks_through_slow_failures(published, monkeypatch, failures=2, ticks_after_last=2)
    assert answers == [True, True, False, True, False, True, True]
    assert not _loop_reaches_its_cutoff(answers) and queue_publication._STATE["failed_writes"] == 2
    assert queue_publication.publish_queue_changes() is True and not queue_publication._STATE["write_failed"]
