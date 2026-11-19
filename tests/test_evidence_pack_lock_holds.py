"""A model call's manifest takes no install-wide lock and never waits for the evidence packer, and the
packer holds the store's write lock only around its candidate's check and the unlink of one checkpoint handle.

On a live install the first pass after the store migration held the store's write lock, and the
publication lock every manifest then took, for 8-24 s per task while it unlinked hundreds of loose
members on a busy disk: every model call's manifest gave up after 4 s and every store writer after
10 s, and two dozen tasks failed (issue #1754). Deletion in sections with gaps for waiters followed,
and the manifests then queued on that lock behind one another and behind the packer (issue #3745).
A manifest is now one atomic write. After one check of the candidate its blobs go under no store
lock, and each handle, which a reader may hold by its path, under a barrier of its own that reads
the store's phase and the task's row again; between two handles the pass leaves the lock free as
long as it held it.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import sqlite3
import threading
import time

import pytest

from ouroboros import evidence_packer as packer
from ouroboros import evidence_packs as packs
from ouroboros import observability, store
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
from ouroboros.utils import utc_now_iso
from tests._store_journal_shared import use_journal
from tests.test_evidence_packs import _handle_path, _result, _task
from tests.test_evidence_packs import root as root  # noqa: F401  (the shared fixture)


class _HookedUnlinks:
    """``evidence_packs.os`` with ``on_unlink(path)`` run before each unlink; the rest is ``os``."""

    def __init__(self, on_unlink):
        self.on_unlink = on_unlink

    def __getattr__(self, name):
        return getattr(os, name)

    def unlink(self, path, *args, **kwargs):
        self.on_unlink(path)
        os.unlink(path, *args, **kwargs)


class _ScriptedTime:
    """``evidence_packs.time`` whose ``monotonic()`` reads a clock only the test moves (``now``) and whose
    ``sleep(seconds)`` runs ``on_sleep(seconds)`` instead; the rest is ``time``."""

    def __init__(self, on_sleep):
        self.now, self.on_sleep = 100.0, on_sleep

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.on_sleep(seconds)


def _zero_wait_write(root):
    """A store write that will not wait, made on another thread and joined: ``committed`` or ``busy``."""
    outcome = []

    def write():
        try:
            with store.open_store(root, write=True, wait_sec=0) as txn:
                txn.store_task({"task_id": "unrelated", "status": "running", "_schema_version": 1,
                                "updated_at": utc_now_iso()})
            outcome.append("committed")
        except store.StoreBusy:
            outcome.append("busy")

    writer = threading.Thread(target=write)
    writer.start()
    writer.join(60)
    return outcome[0] if outcome else "hung"


def _before_barrier(monkeypatch, hook):
    """Run ``hook(n)`` before a deletion enters its n-th store barrier (``_unlink_phase``): n = 1 is the
    candidate's check, n >= 2 the barrier of its (n - 1)-th checkpoint handle."""
    phase, calls = packs._unlink_phase, []

    @contextlib.contextmanager
    def observed(root):
        calls.append(None)
        hook(len(calls))
        with phase(root) as value:
            yield value

    monkeypatch.setattr(packs, "_unlink_phase", observed)


def _handles(root, task_id, count):
    """``count`` more checkpoint handles of the task, beside the one ``_task`` stores."""
    return [store_actor_source_bytes(root, task_id, category="context_checkpoints", extension="md",
                                     source_id=f"focus_source_{task_id}_{n}", data=b"# focus %d\n" % n + b"z" * 3000)
            for n in range(count)]


def _loose(refs):
    return [ref for ref in refs if pathlib.Path(ref["path"]).exists()]


def _present(root, task_id, handles):
    return [handle for handle in handles if _handle_path(root, task_id, handle).exists()]


def test_a_call_manifest_takes_no_install_wide_lock_and_waits_for_no_holder(root):
    """A model call's records, a request view and a plain record alike, publish their manifests without taking
    any lock under ``state/``: the meter that counts every lock file's acquisition sees none of theirs, while it
    sees the one this test takes itself. A holder of the lock older bodies took around each manifest
    (``state/packs_publication.lock``) holds no manifest back."""
    from tests._cost_meter import CostMeter, tagged

    assert "llm_request" in observability.SHARED_ELEMENT_CALL_TYPES  # a request view: its parts are blobs of their own

    def records():
        for call_type in ("llm_request", "tool_call"):
            observability.persist_call(root, task_id="t1", call_id=f"{call_type}_1", call_type=call_type,
                                       payload={"messages": [{"role": "user", "content": f"{call_type} turn " * 40}]})

    def probe():
        with store.name_lock(root, "probe.lock", timeout_sec=5):
            pass

    with CostMeter(root) as meter:
        tagged("records", records)
        tagged("probe", probe)
    assert meter.snapshot("records")["counts"].get("lock_acquired", 0) == 0, meter.snapshot("records")["callers"]
    assert meter.snapshot("probe")["counts"].get("lock_acquired") == 1  # the meter sees a lock taken under state/
    with store.name_lock(root, "packs_publication.lock", timeout_sec=5):
        ref = observability.write_call_manifest(root, task_id="t1", call_id="held", manifest={"call_type": "tool_call"})
    assert pathlib.Path(ref["path"]).exists()


def test_a_long_deletion_holds_no_store_lock_over_its_blobs(root, monkeypatch):
    """A store writer that will not wait commits midway through the unlink of a candidate's blobs, and the one
    pass still removes every member."""
    refs, handle = _task(root, "big", blobs=12)
    blobs, loose_at_write = [], []

    def midway(path):
        if pathlib.Path(path).name.endswith(".gz"):
            blobs.append(path)
            if len(blobs) == 6:  # five blobs gone, seven still loose
                with store.open_store(root, write=True, wait_sec=0) as txn:
                    txn.store_task({"task_id": "unrelated", "status": "running", "_schema_version": 1,
                                    "updated_at": utc_now_iso()})
                loose_at_write.append(len(_loose(refs)))

    monkeypatch.setattr(packs, "os", _HookedUnlinks(midway))
    report = packer.run_pass(root)
    assert report["faults"] == 0 and loose_at_write == [7]
    assert report["unlinked"] == 13 and not _loose(refs) and not _handle_path(root, "big", handle).exists()


def test_each_handle_unlinks_inside_its_barrier_and_the_pass_then_yields_as_long_as_it_held_it(root, monkeypatch):
    """A writer that will not wait is refused while a handle unlinks (its barrier holds the store's write lock) and
    commits in the yield after it. Before the next handle's barrier the pass leaves the lock free exactly as long as
    the last barrier held it, at any disk speed: read on a clock only the handles' unlinks move, a different hold for
    each handle. No yield follows the last handle."""
    _, first = _task(root, "big", blobs=2)
    handles = [first, *_handles(root, "big", 3)]
    holds, inside, yields = iter([3.0, 5.0, 7.0, 11.0]), [], []
    clock = _ScriptedTime(lambda seconds: yields.append((seconds, _zero_wait_write(root))))

    def unlink(path):
        if "context_checkpoints" in str(path):
            inside.append(_zero_wait_write(root))
            clock.now += next(holds)  # this handle's unlink holds its barrier that long

    monkeypatch.setattr(packs, "os", _HookedUnlinks(unlink))
    monkeypatch.setattr(packs, "time", clock)
    report = packer.run_pass(root)
    assert report["faults"] == 0 and report["unlinked"] == 2 + 4 and not _present(root, "big", handles)
    assert inside == ["busy"] * 4 and yields == [(3.0, "committed"), (5.0, "committed"), (7.0, "committed")]


@pytest.mark.parametrize("reader", ["review", "resumed"])
def test_a_reader_that_begins_after_the_check_keeps_the_handles_not_yet_unlinked(root, monkeypatch, reader):
    """A review that opens, or a resumption, between two handle unlinks keeps the rest loose: each handle's barrier
    reads the row again, since a reader may hold a handle by its path. The candidate stays owed, and the pass after
    the reader ends removes the rest, every handle in one pass. One that begins before the candidate's check removes
    nothing, neither blobs nor handles."""
    refs, first = _task(root, "big", blobs=2)
    handles = [first, *_handles(root, "big", 2)]

    def begins():
        if reader == "review":
            _result(root, "big", review_operations={"op": {"surface": "task_acceptance", "state": "dispatched"}})
        else:
            _result(root, "big", "running")

    def ends():
        _result(root, "big", review_operations={"op": {"surface": "task_acceptance", "state": "closed"}})

    def run(begin_at):
        with monkeypatch.context() as patch:
            _before_barrier(patch, lambda n: begins() if n == begin_at else None)
            return packer.run_pass(root)

    report = run(1)  # before the candidate's check
    assert report["faults"] == 0 and (report["deferred"], report["unlinked"]) == (1, 0)
    assert len(_loose(refs)) == 2 and len(_present(root, "big", handles)) == 3
    ends()
    report = run(3)  # the first handle unlinked, the second's barrier next
    assert report["faults"] == 0 and report["deferred"] == 1 and report["unfinished"] == ["big"]
    assert not _loose(refs) and len(_present(root, "big", handles)) == 2
    ends()
    report = packer.run_pass(root)
    assert report["faults"] == 0 and (report["deferred"], report["unlinked"]) == (0, 2)
    assert not _present(root, "big", handles)


def test_an_export_that_begins_during_a_pass_keeps_what_is_still_loose_until_published_again(root, monkeypatch):
    """An export that begins after a candidate's check and its blobs' unlink stops its handles' unlinks (each
    handle's barrier reads the phase), and a candidate checked after the export began loses nothing, neither blobs
    nor handles (its check reads the phase). While it lasts a pass unlinks nothing, and the first pass after the
    store is published again removes the rest. Packed bytes read throughout."""
    refs, first = _task(root, "big", blobs=3)
    handles = [first, *_handles(root, "big", 2)]
    later, later_handle = _task(root, "later", blobs=1, payloads=[b"x"])  # smaller: the pass reaches it after big

    def export(n):
        if n == 2:  # big checked and its blobs unlinked: its first handle's barrier next, then later's check
            with store.open_store(root, write=True, service=True) as txn:
                txn.set_meta("phase", store.EXPORTING)

    with monkeypatch.context() as patch:
        _before_barrier(patch, export)
        report = packer.run_pass(root)
    assert report["faults"] == 0 and report["deferred"] == 2 and report["unfinished"] == ["big", "later"]
    assert not _loose(refs) and len(_present(root, "big", handles)) == 3
    assert _loose(later) == later and _present(root, "later", [later_handle])
    assert all(packs.blob_bytes(root, ref) for ref in refs)
    assert packer.run_pass(root)["unlinked"] == 0 and len(_present(root, "big", handles)) == 3
    assert _loose(later) == later and _present(root, "later", [later_handle])
    with store.open_store(root, write=True, service=True) as txn:
        txn.set_meta("phase", store.PUBLISHED)
    assert packer.run_pass(root)["faults"] == 0 and not _present(root, "big", handles)
    assert not _loose(later) and not _present(root, "later", [later_handle])
    assert all(read_actor_source_bytes(root, "big", handle) for handle in handles)
    assert all(packs.blob_bytes(root, ref) for ref in later)


def _held_reader(root):
    conn = sqlite3.connect(store.store_path(root), isolation_level=None, timeout=0)
    conn.execute("BEGIN")
    conn.execute("SELECT 1 FROM meta LIMIT 1").fetchone()  # SHARED until ROLLBACK
    return conn


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
def test_the_phase_barrier_releases_without_waiting_for_readers(root, monkeypatch, journal):
    use_journal(monkeypatch, journal)
    _task(root, "t1", blobs=1)  # a published store
    reader = _held_reader(root)
    try:
        with store.open_store(root, write=True, wait_sec=0, barrier=True):
            pass  # ROLLBACK: no reader drain, so no busy wait at all is needed
        if journal == "PERSIST":
            with pytest.raises(store.StoreBusy):
                with store.open_store(root, write=True, wait_sec=0):
                    pass  # a rowless COMMIT still has to wait for the reader
        else:
            with store.open_store(root, write=True, wait_sec=0):
                pass  # a WAL COMMIT waits for no reader: the barrier is no exception there
    finally:
        reader.execute("ROLLBACK")
        reader.close()


def test_a_barrier_refuses_to_discard_a_write(root):
    _task(root, "t1", blobs=1)
    with store.open_store(root, write=False) as txn:
        before = txn.revision()
    with pytest.raises(store.StoreError, match="barrier"):
        with store.open_store(root, write=True, barrier=True) as txn:
            txn.set_meta("commit_seq", before + 7)
    with store.open_store(root, write=False) as txn:
        assert txn.revision() == before
