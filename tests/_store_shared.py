"""Store-era seams for what used to be ``task_results/<id>.json`` file mechanics in tests.

A test that wrote a result file to plant a row, read one back to inspect what production
wrote, or checked its presence, does the same thing against the current-facts store here.
Plain callables (many callers need them at module import time), never fixtures.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import time
from typing import Any, Dict, Optional, Tuple

from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION
from ouroboros.task_store import delete_document, exists, patch_document, read_document, signature


def seed_result(root: Any, task_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Plant DOCUMENT as task TASK_ID's current facts, bypassing the lifecycle writer's reducer
    exactly as writing its file used to (the row is whatever the test says it is). The store
    holds admitted rows only, so an unstamped document is stamped; a test about LEGACY
    admission (quarantine, the pre-7 latch) writes the file and lets the import judge it."""
    body = dict(document)
    body.setdefault("task_id", str(task_id))
    body.setdefault(SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION)
    return patch_document(root, task_id, lambda _current: body)


@contextlib.contextmanager
def as_install(root: Any):
    """Act as the install ROOT names, as its own processes do: a root another process booted
    carries its own identity, which any other install refuses to touch (copied-folder rule)."""
    prior = os.environ.get("OUROBOROS_DATA_DIR")
    os.environ["OUROBOROS_DATA_DIR"] = str(root)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("OUROBOROS_DATA_DIR", None)
        else:
            os.environ["OUROBOROS_DATA_DIR"] = prior


def read_result(root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """The stored document as written (no reader normalisation), ``None`` when absent."""
    return read_document(root, task_id)


def result_exists(root: Any, task_id: str) -> bool:
    return exists(root, task_id)


def result_signature(root: Any, task_id: str) -> Optional[Tuple[int, str]]:
    """``(write_seq, updated_at)``: what a test compares before and after an operation that
    must not write the row (the file era compared the file's bytes)."""
    return signature(root, task_id)


def delete_result(root: Any, task_id: str) -> bool:
    """Remove one task's row; ``True`` when it was there."""
    return delete_document(root, task_id)


def unreadable_result(monkeypatch, task_id: str, *, root=None) -> dict:
    """Make every body read of TASK_ID fail as the store does when it cannot answer: the
    store-era shape of the torn result file tests used to plant (``StoreError`` out of the
    row read, inside readers and inside the write transaction alike). ``root`` limits the
    failure to one drive's store; the returned switch (``{"on": True}``) lets a test make the
    row readable again ("after access recovers") without undoing its other patches."""
    from ouroboros import store

    real_body, real_row = store.Txn.body_text, store.Txn.row
    switch = {"on": True}
    wanted_root = None if root is None else pathlib.Path(root).resolve()

    def broken(self, tid):
        if not switch["on"] or str(tid) != str(task_id):
            return False
        return wanted_root is None or (self.root is not None and self.root.resolve() == wanted_root)

    def body_text(self, tid):
        if broken(self, tid):
            raise store.StoreError(f"simulated unreadable row: {tid}")
        return real_body(self, tid)

    def row(self, tid, *columns):
        if broken(self, tid):
            raise store.StoreError(f"simulated unreadable row: {tid}")
        return real_row(self, tid, *columns)

    from ouroboros import task_store

    real_views = task_store.display_documents

    def display_documents(root, task_ids, versions=None):  # a batch holding the row fails like a single read of it
        ids = [str(tid) for tid in task_ids]
        if switch["on"] and str(task_id) in ids and (
                wanted_root is None or pathlib.Path(root).resolve() == wanted_root):
            raise store.StoreError(f"simulated unreadable row: {task_id}")
        return real_views(root, ids, versions)

    from ouroboros import store_hot_facts

    real_hot_rows = store_hot_facts._read_rows

    def hot_rows(txn, ids):  # the typed control columns of the row are unreadable too
        rows = real_hot_rows(txn, ids)
        if str(task_id) in rows and broken(txn, task_id):
            rows[str(task_id)] = store.StoreError(f"simulated unreadable row: {task_id}")
        return rows

    monkeypatch.setattr(store.Txn, "body_text", body_text)
    monkeypatch.setattr(store.Txn, "row", row)
    monkeypatch.setattr(task_store, "display_documents", display_documents)
    monkeypatch.setattr(store_hot_facts, "_read_rows", hot_rows)
    return switch


def unavailable_store(monkeypatch) -> None:
    """Every store transaction fails with ``StoreBusy``: the whole store cannot answer."""
    from ouroboros import store, task_store

    def busy(*_args, **_kwargs):
        raise store.StoreBusy("simulated: current store unavailable")

    monkeypatch.setattr(store, "open_store", busy)
    monkeypatch.setattr(task_store, "open_store", busy)


def release_store_handles(tree: Any, *, wait_sec: float = 30.0) -> None:
    """Close this process's cached connections to every store database at or below ``tree``.

    The store keeps each thread's idle connection open (#1982). Windows refuses to delete or
    rename a file an open handle holds (WinError 32) and a folder with an open file below it
    (WinError 5), so a test that removes, renames or replaces store files it used releases them
    first, as the product does before it moves a drive (``store.close_connections_under``, one
    file: ``store.close_connections``); the next use opens a fresh connection. Any active lease
    refuses with ``StoreBusy``: another thread's lease is waited out, up to ``wait_sec``, then
    the refusal is raised.
    """
    from ouroboros import store

    tree = pathlib.Path(tree)
    if tree.is_dir():
        releases = [lambda: store.close_connections_under(tree)]
    else:
        releases = [lambda spelling=spelling: store.close_connections(spelling)
                    for spelling in dict.fromkeys((tree.absolute(), tree.resolve()))]
    attempts = max(1, int(wait_sec / 0.05))  # counted, not clocked: a test may freeze time.monotonic
    for release in releases:
        for attempt in range(attempts):
            try:
                release()
                break
            except store.StoreBusy:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.05)


def remove_store_tree(tree: Any) -> None:
    """``shutil.rmtree`` of a drive or data root this process used, its store handles released first
    (``release_store_handles``) so the removal works on Windows too."""
    import shutil

    release_store_handles(tree)
    shutil.rmtree(tree)
