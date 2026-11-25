"""Read queue documents by file identity. Readers expand shared task values and take heartbeats
and pool counts from the generation-matched snapshot stamp. The dispatch document certifies itself."""
import os
import pathlib
import threading
import time
from typing import Any, Dict, Optional

from ouroboros.shared_values import SHARED_KEY, SharedValueMissing, token_digest
from ouroboros.utils import file_identity, read_json_dict

_sleep = time.sleep  # the admission contention sleeper, independent of the snapshot observation clock

#: ``SHARED_KEY`` (``shared_values``) names the publisher's table of shared task values and a row's reference into it.
#: A ``queue.alive`` stamp older than this proves no live supervisor to its readers (``task_status``);
#: the publisher refreshes it well inside this bound.
QUEUE_ALIVE_FRESH_SEC = 10.0
#: The pool's counts (``queue_publication._pool_counts``). They move with every boot, retirement and reap,
#: with no row changing, so the certificate carries them (``queue.alive`` ``pool``) and a document keeps them as
#: of its own write; ``load_queue_document`` gives its readers the certified ones.
POOL_COUNT_KEYS = ("max_workers", "worker_total", "parked_worker_count", "busy_worker_count",
                   "acting_worker_count", "booting_count", "reaping_count", "retiring_count",
                   "assignable_idle_workers", "lane_carrier_total", "memory_admission")

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

    ``fresh`` returns an unshared copy (a restore mutates what it reads) and caches nothing. The identity
    carries the change time, so a withdrawn read permission is re-read; a failed read drops the cached one.
    """
    try:
        identity = file_identity(os.stat(path))
    except OSError:
        identity = None
    if not fresh and identity is not None:
        with _CACHE_LOCK:
            cached = _CACHE.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
    document = read_json_dict(path)
    try:
        if document is None:
            raise ValueError(path)
        expand_shared(document)
    except ValueError:
        with _CACHE_LOCK:  # an unreadable document is unknown, never an older reading
            _CACHE.pop(path, None)
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
    if name != "queue_snapshot.json":
        return document
    document["_queue_alive"] = read_json_dict(path.parent / "queue.alive") or {}
    from ouroboros.model_wait import execution_elapsed_seconds, quota_waited_seconds

    stamp = queue_stamp(document)
    pool = stamp.get("pool")
    if isinstance(pool, dict) and pool_observation_status(document) == "observed":
        # The certificate's counts are as of its stamp, the document's as of the document's own write.
        document.update((key, pool[key]) for key in POOL_COUNT_KEYS if key in pool)
    heartbeats = stamp.get("heartbeats") or {}
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


def valid_dispatch_facts(document):
    rows = document.get("budget_root_fences")
    members = document.get("members", {})
    return (isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
            and isinstance(members, dict) and all(isinstance(member, dict) for member in members.values()))


def dispatch_probe(drive_root):
    """A generated dispatch document stands from its atomic replacement."""
    document = load_queue_document(drive_root, "queue_dispatch.json")
    return document if isinstance(document.get("generation"), str) and document["generation"].strip() else None
