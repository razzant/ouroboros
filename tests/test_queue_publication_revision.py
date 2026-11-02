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
from supervisor import direct_roots, queue, queue_publication, queue_snapshot
from supervisor.work_stop_migration import FORMAT_KEY, FORMAT_VERSION

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
    monkeypatch.setattr(direct_roots, "collect_direct_roots", lambda: {"roots": [], "incomplete": False})
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
    def no_bytes_under_q(payload):
        assert not queue._queue_lock._is_owned(), "serialization never runs under Q"
        encodings.append(1)
        return encode(payload)
    monkeypatch.setattr(queue_publication, "machine_json", no_bytes_under_q)
    queue_publication.tick_publisher_started()  # this thread is the loop
    revision = queue_publication.queue_revision()
    with queue._queue_lock:
        for step in range(5):
            queue.PENDING.append(_task(f"t{step}"))
            assert queue.persist_queue_snapshot(f"admit-{step}") is True
    assert writes == [] and encodings == [] and queue_publication.queue_revision() == revision + 5
    assert not (root / "state/queue_snapshot.json").exists()
    assert queue_publication.publish_queue_changes() is True
    assert [name for name, _ in writes] == ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json", "queue.alive"]
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
    assert queue.persist_queue_snapshot("mark") is True and writes == []
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
    assert [name for name, _ in writes] == ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json", "queue.alive"]


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


@pytest.mark.parametrize("killed_after", ["revoke", "queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"])
def test_kill_after_each_step_keeps_order_and_custody(published, monkeypatch, killed_after):
    root, _now = published
    queue.PENDING.append(_task("saved"))
    assert queue.persist_queue_snapshot("seed") is True
    before = {name: (root / "state" / name).read_bytes()
              for name in ("queue_snapshot.json", "queue_dispatch.json", "direct_roots.json")}
    generation = _read(root, "queue.alive")["generation"]
    queue.PENDING.append(_task("later"))
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
    for name in ("queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"):
        seen = task_status.queue_snapshot_observation(queue_observation.load_queue_document(root, name))
        assert seen["freshness"] == "unknown"
    with pytest.raises(queue_observation.DispatchPublicationPending):
        queue_observation.dispatch_probe(root)
    written = {name: (root / "state" / name).read_bytes() for name in before}
    order = ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"]
    landed = order[:order.index(killed_after) + 1] if killed_after != "revoke" else []
    assert all((written[name] != before[name]) == (name in landed) for name in before), "documents land in order"
    queue_publication.reset()
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "PENDING", [])
    assert queue.persist_queue_snapshot("startup") is None, "an unrestored generation keeps the saved queue"
    assert {name: (root / "state" / name).read_bytes() for name in before} == written
    assert not (root / "state/queue.alive").exists()
    saved = queue_observation.load_queue_document(root, fresh=True)
    assert saved["pending"][0]["task"]["origin_message_text"] == OWNER_TEXT  # whichever generation landed is whole
    assert saved["generation"] != generation or killed_after == "revoke"


def test_rollback_export_embeds_every_row_for_an_older_reader(published):
    root, _now = published
    queue.PENDING.append(_task("a"))
    queue.PENDING.append(_task("b"))
    assert queue.persist_queue_snapshot("admit") is True
    assert queue_observation.SHARED_KEY in _read(root, "queue_snapshot.json")
    assert queue.publish_queue_snapshot("pre_restart_exit") is True
    exported = _read(root, "queue_snapshot.json")
    assert queue_observation.SHARED_KEY not in exported and exported[FORMAT_KEY] == FORMAT_VERSION
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
