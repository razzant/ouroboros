"""The evidence packer leaves its locks to waiters while it deletes, and a model call waits for
the publication lock instead of failing its task (issue #1754).

On a live install the first pass after the store migration held ``packs_publication.lock`` and
the store's write lock for 8-24 s per task while it unlinked hundreds of loose members on a busy
disk; every model call's manifest gave up after 4 s and every store writer after 10 s, and two
dozen tasks failed. Deletion now runs in sections of one ``UNLINK_SECTION_SEC`` budget under a
rollback barrier, a waiter gets a poll-length gap between sections, a stale check defers the
candidate, and a manifest waits up to ``PUBLICATION_WAIT_SEC``.
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
from tests._store_journal_shared import use_journal
from tests.test_evidence_packs import _handle_path, _task
from tests.test_evidence_packs import root as root  # noqa: F401  (the shared fixture)


class _SlowUnlinks:
    """``evidence_packs.os`` with each unlink slowed like a saturated disk; the rest is ``os``."""

    def __init__(self, delay: float, on_unlink=None):
        self.delay, self.on_unlink = delay, on_unlink

    def __getattr__(self, name):
        return getattr(os, name)

    def unlink(self, path, *args, **kwargs):
        if self.on_unlink is not None:
            self.on_unlink(path)
        time.sleep(self.delay)
        os.unlink(path, *args, **kwargs)


def _between_sections(monkeypatch, hook):
    """Run ``hook(n)`` before the n-th section of a deletion enters its phase gate (n >= 2: after section n-1)."""
    phase, calls = packs._unlink_phase, []

    @contextlib.contextmanager
    def observed(root):
        calls.append(None)
        if len(calls) >= 2:
            hook(len(calls))
        with phase(root) as value:
            yield value

    monkeypatch.setattr(packs, "_unlink_phase", observed)
    return calls


def _loose(refs):
    return [ref for ref in refs if pathlib.Path(ref["path"]).exists()]


def test_a_long_deletion_leaves_the_locks_to_waiters_between_sections(root, monkeypatch):
    refs, handle = _task(root, "big", blobs=60)
    started = threading.Event()
    monkeypatch.setattr(packs, "os", _SlowUnlinks(0.05, on_unlink=lambda _path: started.set()))
    report: dict = {}
    runner = threading.Thread(target=lambda: report.update(packer.run_pass(root)), daemon=True)
    runner.start()
    assert started.wait(60), "the pass never reached its deletion"
    # Both waiters get in while members are still loose; the base held both locks to the end.
    with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=30):
        assert _loose(refs), "the publication lock came free only after the deletion"
    with store.open_store(root, write=True, wait_sec=30):
        assert _loose(refs), "the store write lock came free only after the deletion"
    runner.join(60)
    assert not runner.is_alive() and report["faults"] == 0
    # 60 blobs and the checkpoint handle, whose own unlink in section one moved the task's stamps
    assert report["unlinked"] == 61 and not _loose(refs) and not _handle_path(root, "big", handle).exists()


def test_a_reference_published_between_sections_keeps_its_blob(root, monkeypatch):
    refs, _ = _task(root, "big", blobs=12)
    _task(root, "live", status="running", blobs=1)
    late = []

    def publish(n):
        if n == 2:  # after section one: reference every blob still loose, through the real writer
            late.extend(_loose(refs))
            observability.write_call_manifest(root, task_id="live", call_id="late", manifest={"refs": late})

    _between_sections(monkeypatch, publish)
    monkeypatch.setattr(packs, "os", _SlowUnlinks(0.1))
    report = packer.run_pass(root)
    assert report["faults"] == 0 and late, "the deletion never reached a second section"
    assert {ref["path"] for ref in _loose(refs)} == {ref["path"] for ref in late}
    assert report["kept"] == len(late) and report["unlinked"] == 13 - len(late)
    assert all(packs.blob_bytes(root, ref) for ref in refs)  # packed or loose, every byte reads


def test_a_section_that_cannot_get_the_publication_lock_defers_the_rest(root, monkeypatch):
    refs, _ = _task(root, "big", blobs=12)

    def hold(n):
        if n == 2:  # a holder outlasting the next section's whole budget, gone before the pass ends
            taken = threading.Event()

            def holder():
                with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=5):
                    taken.set()
                    time.sleep(4 * packs.UNLINK_SECTION_SEC)  # the pass's journal close waits 5 s

            threading.Thread(target=holder, daemon=True).start()
            assert taken.wait(10)

    with monkeypatch.context() as patch:
        calls = _between_sections(patch, hold)
        swept = []
        patch.setattr(packer, "record_swept", lambda *args: swept.append(args))
        patch.setattr(packs, "os", _SlowUnlinks(0.1))
        report = packer.run_pass(root)
        assert report["faults"] == 0 and len(calls) == 2  # no section after the refused one
        assert 0 < report["unlinked"] < 13 and _loose(refs) and not swept  # partial, deferred, retried later
    assert packer.run_pass(root)["faults"] == 0 and not _loose(refs)


def test_an_export_between_sections_stops_the_deletion_until_published_again(root, monkeypatch):
    refs, _ = _task(root, "big", blobs=12)

    def export(n):
        if n == 2:
            with store.open_store(root, write=True, service=True) as txn:
                txn.set_meta("phase", store.EXPORTING)

    with monkeypatch.context() as patch:
        _between_sections(patch, export)
        patch.setattr(packs, "os", _SlowUnlinks(0.1))
        report = packer.run_pass(root)
        assert report["faults"] == 0 and 0 < report["unlinked"] < 13 and _loose(refs)
        loose_after_export = len(_loose(refs))
    assert packer.run_pass(root)["unlinked"] == 0 and len(_loose(refs)) == loose_after_export
    with store.open_store(root, write=True, service=True) as txn:
        txn.set_meta("phase", store.PUBLISHED)
    assert packer.run_pass(root)["faults"] == 0 and not _loose(refs)


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


def test_a_model_call_manifest_waits_for_a_slow_holder(root):
    assert packs.PUBLICATION_WAIT_SEC == 90.0
    taken, release = threading.Event(), threading.Event()

    def hold():
        with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=5):
            taken.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert taken.wait(10)
    threading.Timer(4.5, release.set).start()  # longer than the former 4 s give-up
    started = time.monotonic()
    ref = observability.write_call_manifest(root, task_id="t1", call_id="c1", manifest={})
    assert time.monotonic() - started >= 4.0 and pathlib.Path(ref["path"]).exists()
    holder.join(10)


def test_a_manifest_still_refuses_typed_when_the_holder_never_lets_go(root, monkeypatch):
    monkeypatch.setattr(packs, "PUBLICATION_WAIT_SEC", 0.2)
    with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=5):
        with pytest.raises(store.StoreBusy):
            observability.write_call_manifest(root, task_id="t1", call_id="c1", manifest={})
