"""A model call's manifest takes no install-wide lock and never waits for the evidence packer, and the
packer leaves the store's write lock to waiters while it deletes.

On a live install the first pass after the store migration held the store's write lock, and the
publication lock every manifest then took, for 8-24 s per task while it unlinked hundreds of loose
members on a busy disk: every model call's manifest gave up after 4 s and every store writer after
10 s, and two dozen tasks failed (issue #1754). The manifests then queued on that lock behind one
another and behind the packer's deletion (issue #3745). A manifest is now one atomic write; deletion
runs in sections of one ``UNLINK_SECTION_SEC`` budget under a rollback barrier, a waiter gets a
poll-length gap between sections, and a stale check defers the candidate.
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


def test_a_long_deletion_leaves_the_locks_to_waiters_between_sections(root, monkeypatch):
    refs, handle = _task(root, "big", blobs=60)
    started = threading.Event()
    monkeypatch.setattr(packs, "os", _SlowUnlinks(0.05, on_unlink=lambda _path: started.set()))
    report: dict = {}
    runner = threading.Thread(target=lambda: report.update(packer.run_pass(root)), daemon=True)
    runner.start()
    assert started.wait(60), "the pass never reached its deletion"
    # A store writer gets in while members are still loose; the base held the lock to the end.
    with store.open_store(root, write=True, wait_sec=30):
        assert _loose(refs), "the store write lock came free only after the deletion"
    runner.join(60)
    assert not runner.is_alive() and report["faults"] == 0
    # 60 blobs and the checkpoint handle, whose own unlink in section one moved the task's stamps
    assert report["unlinked"] == 61 and not _loose(refs) and not _handle_path(root, "big", handle).exists()


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
