"""Queue publication by revision (cost law L5, acceptance law E4).

Q-held work is independent of the queue bytes; owner text is published once; ``queue.alive`` is
written without fsync and only on change; publications follow changes, not ticks; the
revoke -> documents -> certify order and the uncustodied-saved-queue rule hold under a kill after
each step; the reader accepts embedded (format 1) and shared-value (format 2) rows alike; the
rollback export embeds every row for an older reader.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import queue, queue_publication, queue_snapshot
from supervisor.work_stop_migration import FORMAT_KEY, FORMAT_VERSION
from tests.test_owner_wait_pool import pool as pool  # noqa: F401 -- fixture: a parked stack on a real store

SENTINEL = "<<the owner's words, published once>>"
OWNER_TEXT = SENTINEL + " and kept whole " * 2500  # ~40 KB, like a live Main message


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    queue_publication.reset()
    yield tmp_path, now
    queue_publication.reset()


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def _task(task_id, **extra):
    return {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": "root", "origin_message_text": OWNER_TEXT,
            "metadata": {"owner_corpus": OWNER_TEXT, "small": "fact"}, "text": f"task {task_id}", **extra}


def _writes(monkeypatch):
    """Every document and certificate write, with whether Q was held and whether it fsync'd."""
    writes, fsyncs = [], []
    writer, alive_writer, fsync = queue.atomic_write_text, queue_publication._write_alive, os.fsync

    def record(path, content):
        writes.append((path.name, queue._queue_lock._is_owned()))
        return (alive_writer if path.name == "queue.alive" else writer)(path, content)

    def count(fd):
        fsyncs.append(fd)
        return fsync(fd)

    monkeypatch.setattr(queue, "atomic_write_text", record)
    monkeypatch.setattr(queue_publication, "_write_alive", record)
    monkeypatch.setattr(os, "fsync", count)
    return writes, fsyncs


def test_marks_under_q_write_no_bytes_and_the_tick_publishes_the_last_state(published, monkeypatch):
    root, _now = published
    writes, _fsyncs = _writes(monkeypatch)
    encodings = []
    encode = queue_publication.machine_json
    def no_payload_bytes_under_q(payload):
        encoded = encode(payload)
        if queue._queue_lock._is_owned():
            # The tick's census of volatile row facts is the one encoding under Q: never a task payload.
            assert "origin_message_text" not in encoded and OWNER_TEXT[:64] not in encoded and '"text"' not in encoded, \
                "task payload serialized under Q"
        else:
            encodings.append(1)
        return encoded
    monkeypatch.setattr(queue_publication, "machine_json", no_payload_bytes_under_q)
    queue_publication.tick_publisher_started()  # this thread is the loop
    revision = queue_publication.queue_revision()
    with queue._queue_lock:
        for step in range(5):
            queue.PENDING.append(_task(f"t{step}"))
            assert queue.persist_queue_snapshot(f"admit-{step}") is None, "a mark is not a disk fact"
    assert writes == [] and encodings == [] and queue_publication.queue_revision() == revision + 5
    assert not (root / "state/queue_snapshot.json").exists()
    assert queue_publication.publish_queue_changes() is True
    assert [name for name, _ in writes] == ["queue_snapshot.json", "queue_dispatch.json", "queue.alive"]
    assert not any(held for _, held in writes), "the tick writes outside Q"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == [f"t{i}" for i in range(5)]
    writes.clear()
    assert queue_publication.publish_queue_changes() is True  # nothing moved: no bytes
    assert writes == []


def test_the_capture_shares_the_owner_text_instead_of_copying_it():
    task = _task("t")
    frozen = queue_publication._frozen(task)
    assert frozen == task and frozen is not task and frozen["metadata"] is not task["metadata"]
    assert frozen["origin_message_text"] is task["origin_message_text"]
    assert frozen["metadata"]["owner_corpus"] is task["metadata"]["owner_corpus"]


def test_owner_text_is_published_once_and_every_reader_sees_it_whole(published):
    root, _now = published
    for index in range(20):
        queue.PENDING.append(_task(f"p{index}", task_contract={"claims": ["x" * 2000]}))
    queue.RUNNING["r0"] = {"task": _task("r0"), "attempt": 1, "started_at": 1.0}
    assert queue.persist_queue_snapshot("burst") is True
    raw = (root / "state/queue_snapshot.json").read_text(encoding="utf-8")
    assert raw.count(SENTINEL) == 1, "twenty-one rows name one owner message; the document carries it once"
    assert len(raw) < 3 * len(OWNER_TEXT)
    document = _read(root, "queue_snapshot.json")
    assert document[FORMAT_KEY] == FORMAT_VERSION and queue_observation.SHARED_KEY in document
    reference = document["pending"][0]["task"]["origin_message_text"]
    assert set(reference) == {queue_observation.SHARED_KEY} and reference[queue_observation.SHARED_KEY] in document[queue_observation.SHARED_KEY]
    assert document["pending"][0]["task"]["metadata"]["small"] == "fact"
    view = queue_observation.load_queue_document(root)
    assert queue_observation.SHARED_KEY not in view
    rows = view["pending"] + view["running"]
    assert len(rows) == 21
    for row in rows:
        assert row["task"]["origin_message_text"] == OWNER_TEXT
        assert row["task"]["metadata"]["owner_corpus"] == OWNER_TEXT
    assert view["pending"][0]["task"]["task_contract"] == {"claims": ["x" * 2000]}
    assert rows[0]["task"]["origin_message_text"] is rows[1]["task"]["origin_message_text"], "one object in memory"
    restore_view = queue_observation.load_queue_document(root, fresh=True)
    assert restore_view["pending"][5]["task"]["origin_message_text"] == OWNER_TEXT
    assert task_status.queue_snapshot_observation(view)["fresh"]


def test_alive_is_written_without_fsync_and_only_on_change(published, monkeypatch):
    root, now = published
    writes, fsyncs = _writes(monkeypatch)
    queue.RUNNING["r0"] = {"task": _task("r0"), "attempt": 1, "started_at": now[0] - 10, "last_heartbeat_at": now[0] - 1}
    assert queue.persist_queue_snapshot("start") is True
    assert fsyncs, "the documents are fsync'd"
    alive_before = (root / "state/queue.alive").read_bytes()
    queue_publication.tick_publisher_started()
    writes.clear(); fsyncs.clear()
    assert queue_publication.publish_queue_changes() is True
    assert writes == [] and fsyncs == [], "an unchanged live set inside the refresh cadence writes nothing"
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC
    assert queue_publication.publish_queue_changes() is True
    assert [name for name, _ in writes] == ["queue.alive"] and fsyncs == [], "the certificate refreshes without fsync"
    assert (root / "state/queue.alive").read_bytes() != alive_before
    assert _read(root, "queue.alive")["generation"] == _read(root, "queue_snapshot.json")["generation"]
    writes.clear()
    queue.RUNNING["r0"]["last_heartbeat_at"] = now[0]
    assert queue_publication.publish_queue_changes() is True
    assert [name for name, _ in writes] == ["queue.alive"] and fsyncs == [], "a heartbeat is a change of the certificate"
    assert _read(root, "queue.alive")["heartbeats"] == {"r0": now[0]}
    observed = task_status.queue_snapshot_observation(queue_observation.load_queue_document(root))
    assert observed["fresh"] and observed["age_sec"] == 0


def test_publications_follow_changes_not_ticks(published, monkeypatch):
    root, _now = published
    writes, _fsyncs = _writes(monkeypatch)
    queue_publication.tick_publisher_started()
    generations = []
    for step in range(10):  # ten changes, each with its tick: ten publications
        queue.PENDING.append(_task(f"a{step}"))
        queue.persist_queue_snapshot("admit")
        queue_publication.publish_queue_changes()
        generations.append(_read(root, "queue.alive")["generation"])
    assert len(set(generations)) == 10
    documents = [name for name, _ in writes if name == "queue_snapshot.json"]
    assert len(documents) == 10
    writes.clear()
    for step in range(10):  # ten changes, one tick: one publication
        queue.PENDING.append(_task(f"b{step}"))
        queue.persist_queue_snapshot("admit")
    queue_publication.publish_queue_changes()
    assert [name for name, _ in writes if name == "queue_snapshot.json"] == ["queue_snapshot.json"]
    assert len(_read(root, "queue_snapshot.json")["pending"]) == 20
    writes.clear()
    for _ in range(5):  # marks without a change of bytes: no publication, same generation
        queue.persist_queue_snapshot("noise")
    generation = _read(root, "queue.alive")["generation"]
    queue_publication.publish_queue_changes()
    assert [name for name, _ in writes if name != "queue.alive"] == []
    assert _read(root, "queue.alive")["generation"] == generation


def test_a_barrier_publishes_at_once_while_the_tick_owns_ordinary_marks(published, monkeypatch):
    root, _now = published
    writes, _fsyncs = _writes(monkeypatch)
    queue_publication.tick_publisher_started()
    queue.PENDING.append(_task("marked"))
    assert queue.persist_queue_snapshot("mark") is None and writes == []
    queue.PENDING.append(_task("granted"))
    with queue._queue_lock:  # a Resume grant publishes before it is sent, even from under Q
        assert queue.publish_queue_snapshot("owner_wait_resumed") is True
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["marked", "granted"]
    assert task_status.queue_snapshot_observation(queue_observation.load_queue_document(root))["fresh"]
    writes.clear()
    assert queue_publication.publish_queue_changes() is True
    assert [name for name, _ in writes if name != "queue.alive"] == [], "the barrier left nothing unpublished"


def test_a_dead_loop_hands_publication_back_to_the_caller(published, monkeypatch):
    root, _now = published
    writes, _fsyncs = _writes(monkeypatch)
    gone = threading.Thread(target=lambda: None)
    gone.start(); gone.join()
    monkeypatch.setattr(queue_publication, "_PUBLISHER", gone)
    assert not queue_publication.tick_publisher_alive()
    queue.PENDING.append(_task("t"))
    assert queue.persist_queue_snapshot("admit") is True
    assert [name for name, _ in writes] == ["queue_snapshot.json", "queue_dispatch.json", "queue.alive"]


def test_a_failed_tick_publication_answers_false_for_the_loop_to_count(published, monkeypatch):
    """The loop raises on False (``tests/test_server_shutdown.py``: three crashes clear readiness);
    the publisher only reports it. A kept saved queue is None, never a failure."""
    _root, _now = published
    queue.PENDING.append(_task("t"))
    queue_publication.tick_publisher_started()
    queue.persist_queue_snapshot("admit")
    monkeypatch.setattr(queue, "atomic_write_text", lambda *_a: False)
    assert queue_publication.publish_queue_changes() is False
    assert queue_publication.publish_queue_changes() is False, "still unpublished: the next tick fails again"
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    queue_publication.reset()
    queue_publication.tick_publisher_started()
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "PENDING", [])
    queue.QUEUE_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)  # the stubbed writer above landed nothing
    queue.QUEUE_SNAPSHOT_PATH.write_text('{"pending": [{"id": "saved", "task": {"id": "saved"}}], "running": []}', encoding="utf-8")
    queue.persist_queue_snapshot("startup")
    assert queue_publication.publish_queue_changes() is None, "a kept saved queue is a quiet tick, not a crash"


@pytest.mark.parametrize("killed_after", ["revoke", "queue_snapshot.json", "queue_dispatch.json"])
def test_kill_after_each_step_keeps_order_and_custody(published, monkeypatch, killed_after):
    root, _now = published
    queue.PENDING.append(_task("saved"))
    assert queue.persist_queue_snapshot("seed") is True
    before = {name: (root / "state" / name).read_bytes() for name in ("queue_snapshot.json", "queue_dispatch.json")}
    generation = _read(root, "queue.alive")["generation"]
    queue.PENDING.append(_task("later"))
    queue.BUDGET_ROOT_FENCES["root"] = {"root_task_id": "root", "fence_id": "fence", "status": "active"}  # both documents change
    writer, unlink = queue.atomic_write_text, Path.unlink

    class Killed(BaseException):
        pass

    def dying_writer(path, content):
        written = writer(path, content)
        if path.name == killed_after:
            raise Killed()
        return written

    def dying_unlink(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path.name == "queue.alive" and killed_after == "revoke":
            raise Killed()

    with monkeypatch.context() as patch:
        patch.setattr(queue, "atomic_write_text", dying_writer)
        patch.setattr(Path, "unlink", dying_unlink)
        with pytest.raises(Killed):
            queue.publish_queue_snapshot("change")
    # The next generation: a reader finds no certificate, so every document is unknown, never a
    # torn mix; the dispatch probe refuses; the saved queue is not in the new process's custody
    # until a restore read it, so its own empty publication keeps the saved documents as they are.
    assert not (root / "state/queue.alive").exists()
    for name in ("queue_snapshot.json", "queue_dispatch.json"):
        seen = task_status.queue_snapshot_observation(queue_observation.load_queue_document(root, name))
        assert seen["freshness"] == "unknown"
    with pytest.raises(queue_observation.DispatchPublicationPending):
        queue_observation.dispatch_probe(root)
    written = {name: (root / "state" / name).read_bytes() for name in before}
    order = ["queue_snapshot.json", "queue_dispatch.json"]
    landed = order[:order.index(killed_after) + 1] if killed_after != "revoke" else []
    assert all((written[name] != before[name]) == (name in landed) for name in before), "documents land in order"
    queue_publication.reset()
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})  # the next generation holds nothing of its own yet
    assert queue.persist_queue_snapshot("startup") is None, "an unrestored generation keeps the saved queue"
    assert {name: (root / "state" / name).read_bytes() for name in before} == written
    assert not (root / "state/queue.alive").exists()
    saved = queue_observation.load_queue_document(root, fresh=True)
    assert saved["pending"][0]["task"]["origin_message_text"] == OWNER_TEXT  # whichever generation landed is whole
    assert saved["generation"] != generation or killed_after == "revoke"


def test_an_unchanged_dispatch_document_keeps_its_generation_and_stays_certified(published, monkeypatch):
    """The money paths' fence authority is rewritten only when its bytes change: a row admitted without a fence
    changes the snapshot alone, so the dispatch document keeps the generation it was written under, the
    certificate names that generation for it (``documents``), and a send's probe certifies it as before. A
    fence changes both documents: both are rewritten under one generation and the certificate names one."""
    root, _now = published
    writes, _fsyncs = _writes(monkeypatch)
    queue.PENDING.append(_task("a"))
    assert queue.persist_queue_snapshot("admit") is True
    first = _read(root, "queue_dispatch.json")["generation"]
    assert _read(root, "queue.alive")["generation"] == first and "documents" not in _read(root, "queue.alive")
    writes.clear()
    queue.PENDING.append(_task("b"))
    assert queue.persist_queue_snapshot("admit") is True
    assert [name for name, _ in writes] == ["queue_snapshot.json", "queue.alive"], "the unchanged dispatch document is not rewritten"
    alive, snapshot = _read(root, "queue.alive"), _read(root, "queue_snapshot.json")
    assert alive["generation"] == snapshot["generation"] != first
    assert alive["documents"] == {"queue_dispatch.json": first} and _read(root, "queue_dispatch.json")["generation"] == first
    dispatch = queue_observation.load_queue_document(root, "queue_dispatch.json")
    assert queue_observation.queue_stamp(dispatch)["generation"] == first, "certified by its own generation"
    assert queue_observation.dispatch_probe(root)["generation"] == first
    assert task_status.queue_snapshot_observation(dispatch)["fresh"]
    assert task_status.queue_snapshot_observation(queue_observation.load_queue_document(root))["fresh"]
    # An older certificate shape (no ``documents``) certifies a document only by the snapshot's generation.
    (root / "state/queue.alive").write_text(json.dumps({"generation": alive["generation"], "ts": alive["ts"], "heartbeats": {}}))
    assert not queue_observation.queue_stamp(queue_observation.load_queue_document(root, "queue_dispatch.json"))
    (root / "state/queue.alive").write_text(json.dumps(alive))
    writes.clear()
    queue.BUDGET_ROOT_FENCES["root"] = {"root_task_id": "root", "fence_id": "fence", "status": "active"}
    assert queue.persist_queue_snapshot("fence") is True
    assert [name for name, _ in writes] == ["queue_snapshot.json", "queue_dispatch.json", "queue.alive"]
    alive = _read(root, "queue.alive")
    assert "documents" not in alive and _read(root, "queue_dispatch.json")["generation"] == alive["generation"] != first
    assert queue_observation.dispatch_probe(root)["budget_root_fences"][0]["fence_id"] == "fence"


def test_rollback_export_embeds_every_row_for_an_older_reader(published):
    root, _now = published
    queue.PENDING.append(_task("a"))
    queue.PENDING.append(_task("b"))
    assert queue.persist_queue_snapshot("admit") is True
    assert queue_observation.SHARED_KEY in _read(root, "queue_snapshot.json")
    assert queue.publish_queue_snapshot("pre_restart_exit") is True
    exported = _read(root, "queue_snapshot.json")
    assert queue_observation.SHARED_KEY not in exported and exported[FORMAT_KEY] == 1, "the shape an older body calls current"
    for row in exported["pending"]:  # the shape an older body restores: ``dict(row["task"])`` with the text inline
        assert isinstance(row["task"]["origin_message_text"], str) and row["task"]["origin_message_text"] == OWNER_TEXT
        assert row["task"]["metadata"]["owner_corpus"] == OWNER_TEXT
    assert queue_observation.load_queue_document(root)["pending"][1]["task"]["origin_message_text"] == OWNER_TEXT
    queue.PENDING.append(_task("c"))
    assert queue.persist_queue_snapshot("admit") is True  # the next ordinary publication interns again
    assert queue_observation.SHARED_KEY in _read(root, "queue_snapshot.json")


def test_the_reader_parses_a_publication_once_and_hands_out_its_own_rows(published, monkeypatch):
    root, _now = published
    queue.PENDING.append(_task("a"))
    assert queue.persist_queue_snapshot("admit") is True
    parsed, reader = [], queue_observation.read_json_dict

    def counting(path):
        if path.name == "queue_snapshot.json":
            parsed.append(path)
        return reader(path)

    monkeypatch.setattr(queue_observation, "read_json_dict", counting)
    first = queue_observation.load_queue_document(root)
    for _ in range(4):
        queue_observation.load_queue_document(root)
    assert len(parsed) == 1, "one parse per publication"
    first["pending"][0]["task"]["annotation"] = "mine"
    first["pending"][0]["mine"] = True
    again = queue_observation.load_queue_document(root)
    assert "annotation" not in again["pending"][0]["task"] and "mine" not in again["pending"][0]
    queue.PENDING.append(_task("b"))
    assert queue.persist_queue_snapshot("admit") is True
    assert len(queue_observation.load_queue_document(root)["pending"]) == 2 and len(parsed) == 2


def test_a_document_that_does_not_answer_its_own_reference_is_invalid(published):
    root, _now = published
    queue.PENDING.append(_task("a"))
    assert queue.persist_queue_snapshot("admit") is True
    document = _read(root, "queue_snapshot.json")
    document[queue_observation.SHARED_KEY] = {}
    (root / "state/queue_snapshot.json").write_text(json.dumps(document), encoding="utf-8")
    assert queue_observation.load_queue_document(root) == {"_snapshot_invalid": True}
    assert queue_observation.load_queue_document(root, fresh=True) == {"_snapshot_invalid": True}


# ---- Q spans memory only: a park and a grant are marks; the store is the restore authority ----------------

def _restart_and_restore(root, monkeypatch):
    """The next generation: nothing in memory, the saved snapshot and the store on disk."""
    from supervisor import workers

    workers.RUNNING.clear()
    workers.PENDING.clear()
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    queue.init(root)
    fenced: list = []
    assert queue.restore_pending_from_snapshot(terminalized=fenced) >= 0
    return list(workers.PENDING), fenced


def _restore_facts(task):
    """What a restore decided for a row: its attempt, its recovery source and its hold; stamps aside."""
    recovery = task.get("_working_recovery") or {}
    hold = task.get("_budget_pause_hold") or {}
    return {"id": task["id"], "attempt": task.get("_attempt"), "source_kind": recovery.get("source_kind"),
            "source_ref": recovery.get("source_ref"), "wait_id": recovery.get("wait_id"),
            "from_attempt": recovery.get("from_attempt"), "budget_paused_sec": recovery.get("budget_paused_sec"),
            "started_at": recovery.get("started_at"), "hold": hold.get("reason"),
            "owner_wait_resume": task.get("_owner_wait_resume")}


@pytest.mark.parametrize("published", [True, False], ids=["published", "unpublished"])
def test_a_park_restores_alike_whether_or_not_its_mark_was_published(pool, monkeypatch, published):
    """k4's objection to a mark here (server-saturation k4_astra.r2.md:31: "a crash then restores the wrong
    state") answered by the kill after the step. The park is a store commit; the queue is marked and the tick
    publishes the row. Kill the process right after the store commit, before the tick (``unpublished``: the
    snapshot on disk is the one before the park), or right after the tick (``published``): the next generation
    restores the same row under the same attempt with the same recovery source, the park's own (the store's
    wait, its source ref and wait id), held for Resume; the snapshot row's ``owner_wait`` decides nothing."""
    from supervisor import worker_owner_wait, workers
    from supervisor.events_budget import HOLD_SAVED_WORK

    root = pool.root
    assert queue.persist_queue_snapshot("before_park") is True  # the row is on disk without its park
    before = (root / "state/queue_snapshot.json").read_bytes()
    worker_owner_wait.handle_owner_wait(pool.event, workers)  # the store commit, the mark (published at once here)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    stored = task_status.load_task_result(root, "owner")["owner_wait"]
    assert stored["state"] == "waiting" and stored["wait_id"] == "wait"
    if not published:
        (root / "state/queue_snapshot.json").write_bytes(before)  # the kill came before the tick's publication
        assert _read(root, "queue_snapshot.json")["running"][0].get("owner_wait") is None
    else:
        assert _read(root, "queue_snapshot.json")["running"][0]["owner_wait"]["state"] == "waiting"
    pending, fenced = _restart_and_restore(root, monkeypatch)
    assert fenced == [] and [row["id"] for row in pending] == ["owner"]
    assert _restore_facts(pending[0]) == {
        "id": "owner", "attempt": 4, "source_kind": "owner_wait", "source_ref": pool.wait["source_ref"],
        "wait_id": "wait", "from_attempt": 3, "budget_paused_sec": 0.0, "started_at": pool.wait["started_at"],
        "hold": HOLD_SAVED_WORK, "owner_wait_resume": None}


@pytest.mark.parametrize("published", [True, False], ids=["published", "unpublished"])
def test_a_grant_restores_the_newer_work_whether_or_not_its_mark_was_published(pool, monkeypatch, published):
    """The grant's half of the same objection: the stack resumed (the store says ``resumed``, the worker saved
    newer work) and the process died before the tick published the row, so the snapshot still shows the wait
    (``unpublished``). The next generation restores the newer working state, never the old grant: the store,
    not the snapshot row, says whether the wait stands."""
    from ouroboros import working_checkpoint as wc
    from supervisor import worker_owner_wait, workers
    from supervisor.events_budget import HOLD_SAVED_WORK
    from tests._budget_pause_exact_helpers import _loop_ctx

    root = pool.root
    worker_owner_wait.handle_owner_wait(pool.event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    assert queue.persist_queue_snapshot("parked") is True
    parked = (root / "state/queue_snapshot.json").read_bytes()
    assert _read(root, "queue_snapshot.json")["running"][0]["owner_wait"]["state"] == "waiting"
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    worker_owner_wait.grant_warm_returns()  # the grant: store commit, mark, resume_granted
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert task_status.load_task_result(root, "owner")["owner_wait"]["state"] == "resumed"
    _ctx, limit = _loop_ctx(root, "owner", attempt=3)
    assert wc.save_round(limit, "pre_effect")  # the resumed stack's newer work
    if not published:
        (root / "state/queue_snapshot.json").write_bytes(parked)  # killed before the tick: the row still says waiting
    else:
        assert queue.persist_queue_snapshot("resumed") is True
        assert _read(root, "queue_snapshot.json")["running"][0]["owner_wait"]["state"] == "resumed"
    pending, fenced = _restart_and_restore(root, monkeypatch)
    assert fenced == [] and [row["id"] for row in pending] == ["owner"]
    facts = _restore_facts(pending[0])
    assert facts["source_kind"] == wc.SOURCE_WORKING and facts["from_attempt"] == 3 and facts["attempt"] == 4
    assert facts["hold"] == HOLD_SAVED_WORK and facts["owner_wait_resume"] is None
    assert facts["wait_id"] in (None, ""), "the newer work, never the old grant"


@pytest.mark.parametrize("lands", [True, False], ids=["lands", "fails"])
def test_an_admission_while_a_restart_is_pending_publishes_its_barrier_outside_q(published, monkeypatch, lands):
    """``enqueue_task``'s own barrier (durable before the reply while a restart is pending, t2 §1.2 queue.py:519)
    runs after the function's own hold, outside Q; a failed one withdraws the row under Q again and the reply is
    the typed refusal, as before."""
    from ouroboros import server_restart

    root, _now = published
    queue_publication.tick_publisher_started()
    monkeypatch.setattr(server_restart, "agent_restart_pending", lambda: True)
    seen, real = [], queue.publish_queue_snapshot
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda reason="": seen.append((reason, queue._queue_lock._is_owned()))
                        or (real(reason=reason) if lands or reason != "agent_restart_admission" else False))
    admitted = queue.enqueue_task(_task("restart-row"))
    if lands:
        assert not admitted.get("_admission_blocked") and seen == [("agent_restart_admission", False)]
        assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["restart-row"]
        assert [row["id"] for row in queue.PENDING] == ["restart-row"]
    else:
        assert admitted["_admission_blocked"] == "restart_admission_unconfirmed"
        assert seen == [("agent_restart_admission", False), ("agent_restart_admission_unconfirmed", False)]
        assert queue.PENDING == [] and _read(root, "queue_snapshot.json")["pending"] == []


def test_embedded_and_shared_rows_restore_alike(tmp_path, monkeypatch):
    """Format 1 (every row embedded) and format 2 (shared values) restore the same parked task."""
    from tests._budget_pause_exact_helpers import _install_queue, _parked

    queue_mod, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    task, _row = _parked(tmp_path, monkeypatch, task_id="saved-root",
                         extra={"origin_message_text": OWNER_TEXT, "metadata": {"owner_corpus": OWNER_TEXT}})
    assert queue_mod.persist_queue_snapshot(reason="seed") is True
    path = tmp_path / "state" / "queue_snapshot.json"
    shared = json.loads(path.read_text())
    assert queue_observation.SHARED_KEY in shared
    embedded = json.loads(json.dumps(queue_observation.expand_shared(json.loads(path.read_text()))))
    embedded[FORMAT_KEY] = 1
    for variant in (embedded, shared):
        path.write_text(json.dumps(variant), encoding="utf-8")
        workers.PENDING[:] = []
        queue_snapshot._RESTORED.clear()
        queue_mod.init(tmp_path)
        assert queue_mod.restore_pending_from_snapshot() == 1
        [restored] = [row for row in workers.PENDING if row["id"] == "saved-root"]
        assert restored["origin_message_text"] == OWNER_TEXT
        assert restored["metadata"]["owner_corpus"] == OWNER_TEXT
        assert restored["_budget_pause"]["checkpoint"] == task["_budget_pause"]["checkpoint"]
        assert json.loads(path.read_text())[FORMAT_KEY] == FORMAT_VERSION
