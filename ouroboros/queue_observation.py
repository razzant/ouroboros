"""Read queue projections against the last successfully published generation.

The publisher (``supervisor/queue_publication.py``) writes large task values once per
document, by content address under ``$shared``; ``load_queue_document`` expands them, so
every reader sees whole rows, and keeps the parsed document of one file identity
(mtime, size, inode) so repeated readers do not parse the same bytes again.
"""

import os
import pathlib
import threading
import time
from typing import Any, Dict, Optional

from ouroboros.shared_values import SHARED_KEY, SharedValueMissing, token_digest
from ouroboros.utils import read_json_dict

_sleep = time.sleep  # the in-flight re-read's pause, kept apart from frozen test clocks
_monotonic = time.monotonic

#: ``SHARED_KEY`` (``shared_values``) names the publisher's table of shared task values and a row's reference into it.
#: A ``queue.alive`` stamp older than this proves no live supervisor to its readers (``task_status``);
#: the publisher refreshes it well inside this bound.
QUEUE_ALIVE_FRESH_SEC = 10.0

_CACHE: Dict[pathlib.Path, tuple] = {}  # path -> (identity, expanded document)
_CACHE_LOCK = threading.Lock()


def queue_stamp(document: dict, *, boot: bool = False) -> dict:
    # Boot can recover an older release's snapshot; live readers need its publisher.
    if boot and "generation" not in document:
        return document
    stamp = document.get("_queue_alive") or {}
    return stamp if document.get("generation") and document["generation"] == stamp.get("generation") else {}


def pool_observation_status(document: dict) -> str:
    """``unknown`` when the publisher could not measure its pool (``queue_publication._pool_counts``), else
    ``observed``. A document from before the status existed is unknown only when its disablement says
    ``unknown``: that publisher's failed measurement wrote zeros, which are no count."""
    status = document.get("pool_observation_status")
    if status is None:
        return "unknown" if document.get("worker_pool_disabled_reason") == "unknown" else "observed"
    return "observed" if status == "observed" else "unknown"


def expand_shared(document: dict) -> dict:
    """Resolve every ``{"$shared": address}`` in the rows from the document's own table, in place.

    A reference the table does not answer is a malformed document (``SharedValueMissing``, a
    ``ValueError``): a row with a hole is never handed to a reader as the task it names.
    """
    shared = document.pop(SHARED_KEY, None)
    if not isinstance(shared, dict):
        return document

    def value_of(item: Any) -> Any:
        digest = token_digest(item)
        if not digest:
            return item
        try:
            return shared[digest]
        except KeyError:
            raise SharedValueMissing(digest, "queue document") from None

    for lane in ("pending", "running"):
        for row in document.get(lane) or []:
            task = row.get("task") if isinstance(row, dict) else None
            if not isinstance(task, dict):
                continue
            for key, item in list(task.items()):
                if key == "metadata" and isinstance(item, dict):
                    task[key] = {name: value_of(nested) for name, nested in item.items()}
                else:
                    task[key] = value_of(item)
    return document


def _parsed(path: pathlib.Path, *, fresh: bool) -> Optional[dict]:
    """The expanded document of ``path``: the cached one while the file identity holds, else re-read.

    ``fresh`` returns an unshared copy (a restore mutates what it reads) and caches nothing.
    """
    try:
        stat = os.stat(path)
        identity = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except OSError:
        identity = None
    if not fresh and identity is not None:
        with _CACHE_LOCK:
            cached = _CACHE.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
    document = read_json_dict(path)
    if document is None:
        return None
    try:
        expand_shared(document)
    except ValueError:
        return None
    if fresh:
        return document
    if identity is not None:
        with _CACHE_LOCK:
            _CACHE[path] = (identity, document)
    return document


def _view(document: dict) -> dict:
    """A reader's own copy of the row structure; the task values below stay shared with the cache."""
    view = dict(document)
    for lane in ("pending", "running"):
        rows = document.get(lane)
        if isinstance(rows, list):
            view[lane] = [{**row, "task": dict(row["task"])} if isinstance(row, dict) and isinstance(row.get("task"), dict)
                          else (dict(row) if isinstance(row, dict) else row) for row in rows]
    for key in ("acceptance_fences", "budget_root_fences", "roots"):
        if isinstance(document.get(key), list):
            view[key] = [dict(row) if isinstance(row, dict) else row for row in document[key]]
    return view


def load_queue_document(drive_root, name="queue_snapshot.json", *, fresh: bool = False) -> dict:
    """The published document with its matching liveness stamp and derived run-time facts.

    Shared task values are expanded; the returned structure is the caller's to annotate
    (rows and task dicts are copies; the values below them are shared with the reader
    cache unless ``fresh``, which parses an unshared copy for a reader that mutates deeply).
    """
    path = pathlib.Path(drive_root) / "state" / name
    document = _parsed(path, fresh=fresh)
    if document is None:
        return {"_snapshot_invalid" if path.exists() else "_snapshot_missing": True}
    document = document if fresh else _view(document)
    alive = read_json_dict(path.parent / "queue.alive") or {}
    # The certificate names each document's own generation: the snapshot's as ``generation``, another
    # document's under ``documents`` when it was not rewritten with the snapshot (its bytes had not changed).
    others = alive.get("documents")
    if name != "queue_snapshot.json" and isinstance(others, dict) and name in others:
        alive = {**alive, "generation": others[name]}
    document["_queue_alive"] = alive
    if name == "queue_snapshot.json":
        from ouroboros.model_wait import execution_elapsed_seconds, quota_waited_seconds

        heartbeats = queue_stamp(document).get("heartbeats") or {}
        now = time.time()
        for row in document.get("running") or []:
            if not isinstance(row, dict):
                continue
            started = row.get("started_at") or 0
            started = started if isinstance(started, (int, float)) else 0
            heartbeat = heartbeats.get(str(row.get("id") or ""))
            row.update(runtime_sec=round(max(0.0, now - started), 2) if started > 0 else 0.0,
                       quota_wait_sec=quota_waited_seconds(row, now), execution_sec=execution_elapsed_seconds(row, now),
                       heartbeat_lag_sec=round(max(0.0, now - heartbeat), 2) if isinstance(heartbeat, (int, float)) and heartbeat > 0 else None)
    return document


DISPATCH_PUBLICATION_WAIT_SEC = 30.0  # the wait budget for an in-flight queue publication on a loaded host
DISPATCH_PUBLICATION_POLL_SEC = 0.05


class DispatchPublicationPending(RuntimeError):
    """A generated dispatch document is not certified yet; never permissive authority."""

    def __init__(self, message: str, *, probe_elapsed_sec: float = 0.0):
        super().__init__(message)
        self.probe_elapsed_sec = probe_elapsed_sec


def dispatch_probe(drive_root):
    """One read, with no sleep: a publication gap is distinct from a malformed document."""
    started = _monotonic()
    document = load_queue_document(drive_root, "queue_dispatch.json")
    if not document.get("generation"):
        return None
    if not queue_stamp(document):
        raise DispatchPublicationPending(
            f"root budget fence authority unavailable: {drive_root}/state/queue_dispatch.json",
            probe_elapsed_sec=max(0.0, _monotonic() - started))
    return document


def published_dispatch(drive_root, *, attempts=None, pause_sec=DISPATCH_PUBLICATION_POLL_SEC):
    """The dispatch document of the last successful publication: its generation certified by
    queue.alive, whatever its age (a stalled tick never stops a send). A publication in flight
    (the stamp is unlinked first) is re-read for up to DISPATCH_PUBLICATION_WAIT_SEC; None when
    it stays uncertified (issue #1754: a one-second window failed sends under load)."""
    if attempts is None:
        attempts = max(1, int(DISPATCH_PUBLICATION_WAIT_SEC / pause_sec))
    for attempt in range(attempts):
        document = load_queue_document(drive_root, "queue_dispatch.json")
        if not document.get("generation"):
            return None  # Only an uncertified publication can be in flight.
        if queue_stamp(document):
            return document
        if attempt + 1 < attempts:
            _sleep(pause_sec)
    return None
