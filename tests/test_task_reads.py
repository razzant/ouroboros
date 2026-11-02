"""A history page reads its tasks' truth in batches of display views (``task_reads``).

The display view drops only the bulky evidence no display reader uses, and the page it
builds must be the page full documents build; the reads are a few store transactions per
data root, not one per task and document.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import time
from types import SimpleNamespace

import pytest

from ouroboros import task_store
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.task_reads import TaskReads
from tests._store_shared import read_result, seed_result

TS = "2026-10-16T10:00:00Z"
BULK = {  # every dropped path, filled so a reader that used one would change the page
    "loop_outcome": {"usage": {"total_rounds": 3, "cost_usd": 9.0, "extra": "U" * 100_000},
                     "trace_refs": ["t"] * 50, "status": "failed", "extra": "L" * 100_000},
    "trace_refs": ["trace"] * 200, "review_evidence": {"rounds": ["x" * 500]},
    "artifacts": [{"path": "a.txt", "status": "ok"}], "result": "R" * 2000, "trace_summary": "S" * 500,
    "governing_owner_words": "W" * 500, "verification_ledger": {"checks": ["c"] * 20},
    "plan_review_state": {"state": "open"}, "acceptance_debt": {"x": 1}, "mutation_evidence": {"m": 1},
    "configured_subagent": {"c": 1}, "context": "C" * 300, "continuation_narrative": "N" * 300,
    "origin_message_text": "O" * 300, "owner_mailbox": {"rows": []}, "subagent_availability": {"a": 1},
    "plan_record": {"steps": ["p"] * 30}, "review_status": "failed", "schedule_admission": {"s": 1},
    "task_acceptance_review_accounting": {"t": 1},
    "artifact_bundle": {"status": "complete", "artifacts": [{"path": "b.txt"}] * 20, "errors": []},
    "task_contract": {"deadline_at": "2026-10-17T00:00:00Z", "objective": "O" * 300},
}
METADATA_BULK = {key.split(".", 1)[1]: {"bulk": "M" * 200} for key in task_store.DISPLAY_DROPPED
                 if key.startswith("metadata.")}


def _seed(root, task_id, status, **extra):
    seed_result(root, task_id, {"task_id": task_id, "status": status, "chat_id": 1, "ts": TS,
                                "title": f"Task {task_id}", "metadata": {"source": "owner", **METADATA_BULK},
                                "outcome_axes": {"execution": {"status": "succeeded"}}, **BULK, **extra})


def _page(root):
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": "1"})))
    assert response.status_code == 200
    return json.loads(response.body)


def _room(root, tasks):
    """Each task's answer, its terminal summary and one narration row, as their producers write them."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    chat, progress = [], []
    for index, task_id in enumerate(tasks):
        ts = f"2026-10-16T10:00:{index:02d}Z"
        chat.append({"ts": ts, "direction": "out", "chat_id": 1, "task_id": task_id, "text": f"answer {task_id}"})
        chat.append({"ts": ts, "direction": "system", "type": "task_summary", "chat_id": 1, "task_id": task_id,
                     "summary_id": f"task-terminal:{task_id}", "text": f"done {task_id}", "tool_calls": 1,
                     "rounds": 99})  # legacy recovery counted physical attempts
        progress.append({"ts": ts, "content": f"step {task_id}", "task_id": task_id, "chat_id": 1})
    for name, rows in (("chat", chat), ("progress", progress)):
        (root / "logs" / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_display_view_drops_only_the_bulky_evidence(tmp_path):
    _seed(tmp_path, "task-a", "completed", flag=True, nothing=None, newer_fact={"kept": [1, 2]})
    seed_result(tmp_path, "task-b", {"task_id": "task-b", "status": "running", "artifact_bundle": "legacy",
                                     "task_contract": {"objective": "no deadline"}})
    views = {task_id: json.loads(text) for task_id, text in task_store.display_documents(
        tmp_path, ["task-a", "task-b", "missing"]).items()}
    full = read_result(tmp_path, "task-a")
    assert set(views) == {"task-a", "task-b"}
    view = views["task-a"]
    for path in task_store.DISPLAY_DROPPED:
        key, _, sub = path.partition(".")
        assert (sub not in view["metadata"]) if sub else (key not in view), path
    assert view["artifact_bundle"] == {"status": "complete"}
    assert view["task_contract"] == {"deadline_at": "2026-10-17T00:00:00Z"}
    assert view["loop_outcome"] == {"usage": {"total_rounds": 3}}
    assert len(json.dumps(view)) < 2000
    assert read_result(tmp_path, "task-a") == full, "display reads never rewrite the authoritative body"
    # Every other key arrives exactly as stored: booleans, nulls and an unknown new fact included.
    kept = {key: value for key, value in full.items()
            if key not in task_store.DISPLAY_DROPPED and key not in task_store.DISPLAY_KEPT and key != "metadata"}
    assert {key: view[key] for key in kept} == kept
    assert view["flag"] is True and view["nothing"] is None and view["newer_fact"] == {"kept": [1, 2]}
    assert view["metadata"] == {"source": "owner"}
    # A kept key that is not an object stays as stored; an absent kept sub-key stays absent.
    assert views["task-b"]["artifact_bundle"] == "legacy" and views["task-b"]["task_contract"] == {}


@pytest.mark.parametrize("snapshot, expected", [
    ({}, {}),
    ({"loop_outcome": None}, {"loop_outcome": None}),
    ({"loop_outcome": {}}, {"loop_outcome": {}}),
    ({"loop_outcome": {"usage": None}}, {"loop_outcome": {"usage": None}}),
    ({"loop_outcome": {"usage": {}}}, {"loop_outcome": {"usage": {}}}),
    *[({"loop_outcome": {"usage": {"total_rounds": rounds, "extra": "U" * 100_000}, "extra": "L" * 100_000}},
       {"loop_outcome": {"usage": {"total_rounds": rounds}}}) for rounds in (None, 0, 3)],
], ids=["missing", "null", "empty", "usage-null", "usage-empty", "rounds-null", "zero", "accepted"])
def test_display_rounds_keep_presence_without_bulk_or_full_reads(tmp_path, monkeypatch, snapshot, expected):
    seed_result(tmp_path, "snapshot", {"task_id": "snapshot", "status": "completed", **snapshot})
    full = read_result(tmp_path, "snapshot")
    monkeypatch.setattr(task_store, "documents_by_ids", lambda *_: pytest.fail("display read fetched full bodies"))
    view = TaskReads(tmp_path).load(tmp_path, "snapshot")
    assert {key: view[key] for key in ("loop_outcome",) if key in view} == expected
    assert read_result(tmp_path, "snapshot") == full


def test_a_page_built_from_display_views_is_the_page_full_documents_build(tmp_path, monkeypatch):
    tasks = [f"task-{index}" for index in range(6)]
    for index, task_id in enumerate(tasks):
        _seed(tmp_path, task_id, ("completed", "failed", "running", "cancelled")[index % 4],
              root_phase_checkpoint={"post_task_synthesis": "open" if index == 4 else "done"},
              completion_observations={"delivery_counts": {"send_message": {"calls": index + 1}}},
              cost_usd=0.25 * index)
    _room(tmp_path, tasks)
    with_views = _page(tmp_path)
    annotated = [row for row in with_views["messages"] if row.get("outcome_axes")]
    assert len(annotated) >= len(tasks), "the page carries each task's current truth"
    assert {row["rounds"] for row in annotated if row.get("outcome_final")} == {3}
    monkeypatch.setattr(task_store, "DISPLAY_DROPPED", ())
    monkeypatch.setattr(task_store, "DISPLAY_KEPT", {})
    assert _page(tmp_path)["messages"] == with_views["messages"]


def test_a_page_reads_its_tasks_in_a_few_store_transactions(tmp_path, monkeypatch):
    from ouroboros import store

    tasks = [f"task-{index}" for index in range(30)]
    for task_id in tasks:
        _seed(tmp_path, task_id, "completed", retry_task_id="" if task_id != "task-3" else "task-3b")
    _seed(tmp_path, "task-3b", "completed")
    _room(tmp_path, tasks)
    opened = []
    real = store.open_store

    def counted(root, *args, **kwargs):
        opened.append(str(root))
        return real(root, *args, **kwargs)

    monkeypatch.setattr(store, "open_store", counted)
    monkeypatch.setattr(task_store, "open_store", counted)
    page = _page(tmp_path)
    assert {row["task_id"] for row in page["messages"] if row.get("task_id")} == set(tasks)
    reads_of_task_bodies = len(opened)
    # The canonical page and its one successor: a wave each, plus the room's own lookups.
    assert reads_of_task_bodies < 15, reads_of_task_bodies


def test_every_load_is_a_private_copy_and_absence_is_the_stores_answer(tmp_path):
    _seed(tmp_path, "task-a", "completed")
    reads = TaskReads(tmp_path)
    reads.prefetch(["task-a", "missing"])
    first = reads.load(tmp_path, "task-a")
    first["status"] = "mutated"
    first["metadata"]["source"] = "mutated"
    assert reads.load(tmp_path, "task-a")["status"] == "completed"
    assert reads.load(tmp_path, "task-a")["metadata"]["source"] == "owner"
    assert reads.absent("missing") and not reads.absent("task-a")
    assert reads.store_reads == 1


def _age(root, seconds=60):
    """The store as a commit that many seconds ago left it: its file and its WAL log, if any."""
    from ouroboros.store import STORE_REL

    then = time.time() - seconds
    for path in (root / STORE_REL, (root / STORE_REL).with_name(STORE_REL.name + "-wal")):
        if path.exists():
            os.utime(path, (then, then))


def test_an_unchanged_child_store_is_not_reopened(tmp_path, monkeypatch):
    from ouroboros import task_reads

    child = tmp_path / "child"
    _seed(child, "task-a", "completed")
    _age(child)
    opened = []
    real = task_store.display_documents
    monkeypatch.setattr(task_store, "display_documents",
                        lambda root, ids, versions=None: opened.append(str(root)) or real(root, ids, versions))
    monkeypatch.setattr(task_reads, "_REPLICAS", task_reads.OrderedDict())
    for _ in range(3):
        assert TaskReads(tmp_path).load(child, "task-a")["status"] == "completed"
    assert opened == [str(child)]
    _seed(child, "task-a", "failed")  # a write moves the store's stamp: the next read sees it
    assert TaskReads(tmp_path).load(child, "task-a")["status"] == "failed"
    assert opened == [str(child), str(child)]


def test_a_wal_commit_older_than_the_grain_still_moves_the_stamp(tmp_path, monkeypatch):
    """Under WAL a commit lands in the log and leaves the store file as it was until a checkpoint
    moves it there: the stamp holds the log's identity too, so a child's commit is seen however
    long ago it happened, never served from the views cached before it."""
    from ouroboros import store, task_reads
    from ouroboros.store import STORE_REL
    from tests._store_journal_shared import use_journal

    use_journal(monkeypatch, "WAL")  # whatever this SQLite would choose
    monkeypatch.setattr(task_reads, "_REPLICAS", task_reads.OrderedDict())
    child = tmp_path / "child"
    _seed(child, "task-a", "completed")
    _age(child)
    assert TaskReads(tmp_path).load(child, "task-a")["status"] == "completed"  # cached under the aged stamp
    database = os.stat(child / STORE_REL)
    _seed(child, "task-a", "failed")
    log = store.log_identity(child / STORE_REL)
    assert log is not None and os.stat(child / STORE_REL).st_mtime_ns == database.st_mtime_ns  # in the log only
    then = time.time() - 60
    os.utime((child / STORE_REL).with_name(STORE_REL.name + "-wal"), (then, then))  # the commit past the grain
    assert TaskReads(tmp_path).load(child, "task-a")["status"] == "failed"


def test_a_log_reused_in_place_moves_the_stamp_while_its_size_and_times_stay(tmp_path, monkeypatch):
    """A checkpointed log is reused in place: the next commit rewrites it from its first frame, at
    the same inode and size, and Windows may keep the file's write time while a cached connection
    holds it open. The stamp reads the log's generation and frames beside its size and time."""
    from ouroboros import task_reads
    from ouroboros.store import STORE_REL
    from tests._store_journal_shared import use_journal

    use_journal(monkeypatch, "WAL")
    monkeypatch.setattr(task_reads, "_REPLICAS", task_reads.OrderedDict())
    child = tmp_path / "child"
    for index in range(4):  # a first generation longer than the commit that reuses the log
        _seed(child, f"task-{index}", "completed")
    database = child / STORE_REL
    log = database.with_name(STORE_REL.name + "-wal")
    with contextlib.closing(sqlite3.connect(database)) as conn:
        busy, frames, moved = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
    assert busy == 0 and frames == moved > 0  # every frame is in the database: the next writer restarts the log
    _age(child)
    assert TaskReads(tmp_path).load(child, "task-0")["status"] == "completed"  # cached under the aged stamp
    held, kept = os.stat(database), os.stat(log)
    _seed(child, "task-0", "failed")
    reused = os.stat(log)
    assert (reused.st_ino, reused.st_size) == (kept.st_ino, kept.st_size), "the commit rewrote the log in place"
    assert (os.stat(database).st_size, os.stat(database).st_mtime_ns) == (held.st_size, held.st_mtime_ns)
    os.utime(log, ns=(kept.st_atime_ns, kept.st_mtime_ns))  # its write time as Windows may still report it
    assert TaskReads(tmp_path).load(child, "task-0")["status"] == "failed"


def test_a_child_store_written_within_the_file_time_grain_is_read_again(tmp_path, monkeypatch):
    """A commit inside a coarse file time (FAT, HFS+, some FUSE mounts) can keep the store's size
    and time: a store that recent is never served from the cache, an older one is."""
    from ouroboros import task_reads
    from ouroboros.store import STORE_REL

    child = tmp_path / "child"
    _seed(child, "task-a", "completed")
    opened = []
    real = task_store.display_documents
    monkeypatch.setattr(task_store, "display_documents",
                        lambda root, ids, versions=None: opened.append(str(root)) or real(root, ids, versions))
    monkeypatch.setattr(task_reads, "_REPLICAS", task_reads.OrderedDict())
    stamp = os.stat(child / STORE_REL)
    for _ in range(2):
        assert TaskReads(tmp_path).load(child, "task-a")["status"] == "completed"
    assert opened == [str(child)] * 2  # fresh: read each time
    _seed(child, "task-a", "failed")
    os.utime(child / STORE_REL, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))  # same time a coarse clock keeps
    assert TaskReads(tmp_path).load(child, "task-a")["status"] == "failed"
    _age(child)
    for _ in range(2):
        assert TaskReads(tmp_path).load(child, "task-a")["status"] == "failed"
    assert opened == [str(child)] * 4  # aged: the cache answers the second read


def test_a_page_read_records_the_version_of_each_document_it_showed(tmp_path):
    for task_id in ("task-a", "task-b"):
        _seed(tmp_path, task_id, "completed")
    reads = TaskReads(tmp_path)
    reads.prefetch(["task-a", "task-b", "absent"])
    assert set(reads.versions) == {"task-a", "task-b"} and all(seq > 0 for seq in reads.versions.values())
    before = dict(reads.versions)
    _seed(tmp_path, "task-a", "failed")
    now = task_store.task_versions(tmp_path, ["task-a", "task-b", "absent"])
    assert now["task-a"][0] > before["task-a"] and now["task-b"][0] == before["task-b"]
    assert now["task-a"][1] == "failed" and "absent" not in now


@pytest.mark.parametrize("bad", ["unreadable"])
def test_an_unreadable_store_is_never_proven_absent(tmp_path, monkeypatch, bad):
    reads = TaskReads(tmp_path)

    def broken(root, ids, versions=None):
        raise OSError(bad)

    monkeypatch.setattr(task_store, "display_documents", broken)
    with pytest.raises(OSError):
        reads.absent("task-a")


@pytest.mark.parametrize("children", [10, 200])
def test_settled_children_whose_drives_are_gone_cost_no_schema_builds(tmp_path, monkeypatch, children):
    """A room's settled children name drives removed by design: each is read as an absent store, and
    the number of schema builds that costs must not grow with the number of children."""
    import threading

    from ouroboros import store_empty

    monkeypatch.setattr(store_empty, "_IDLE", threading.local())
    builds, real = [], sqlite3.connect

    def counted(database, *args, **kwargs):
        builds.append(database) if database == ":memory:" else None
        return real(database, *args, **kwargs)

    tasks = [f"child-{n}" for n in range(children)]
    for task_id in tasks:
        _seed(tmp_path, task_id, "completed", child_drive_root=str(tmp_path / "drives" / task_id))
    monkeypatch.setattr(store_empty.sqlite3, "connect", counted)
    reads = TaskReads(tmp_path)
    reads.prefetch(tasks)
    assert len(builds) == 1, len(builds)
    assert all(reads.load(tmp_path / "drives" / task_id, task_id) is None for task_id in tasks)
    assert reads.store_reads == 1 + children and not reads.failed


def test_the_prefetch_decodes_each_view_once_and_loads_stay_private(tmp_path, monkeypatch):
    from ouroboros import task_reads

    tasks = [f"task-{n}" for n in range(12)]
    for task_id in tasks:
        _seed(tmp_path, task_id, "completed", retry_task_id="task-0" if task_id == "task-11" else "")
    decoded = []

    def counted(text, *args, **kwargs):
        decoded.append(text)
        return json.loads(text, *args, **kwargs)

    monkeypatch.setattr(task_reads, "json", SimpleNamespace(loads=counted, dumps=json.dumps))
    reads = TaskReads(tmp_path)
    for _ in range(3):  # a page prefetches its candidates, then its window, then its overlays
        reads.prefetch(tasks)
    assert len(decoded) == len(tasks), "one decode per view per read, however often the read looks"
    first, second = reads.load(tmp_path, "task-1"), reads.load(tmp_path, "task-1")
    assert first == second and first is not second and first["metadata"] is not second["metadata"]
