"""One read of many tasks' current truth (a history page): their documents in display views
(``task_store.display_documents``), fetched per data root in a few batches, the queue snapshot
and the cancel intents read once, one events tail shared.

``task_status.effective_task_result(..., reads=...)`` loads every document it overlays through
this object: the canonical one, its retry/supersession successors and its child-drive
replicas. A page therefore pays one short store read per data root and wave instead of one
store open per task and per document it overlays (P13: facts relevant to the current
operation). Within the read every loader receives its own copy, as from ``load_task_result``.
An unchanged child store is not reopened by later reads of this process: its views are kept
under the stamp of the store file and its WAL log, and a stamp that moved reads again. The
canonical documents' versions (``write_seq``, read with their views) say which truth a page
showed (``after`` cursors).
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Tuple

log = logging.getLogger(__name__)
_REPLICAS: "OrderedDict[str, Tuple[tuple, Dict[str, Any]]]" = OrderedDict()
_REPLICAS_MAX = 512  # child stores whose views a process keeps (one per delegated child drive)
_REPLICAS_LOCK = threading.Lock()
# The coarsest file time a supported file system keeps (FAT, HFS+ and some FUSE mounts: 1-2 s): a
# store or log written this recently may change again without its time moving, so it is not cached.
_STAMP_GRAIN_SEC = 2.0


def _store_stamp(root: pathlib.Path) -> Optional[tuple]:
    """The store's identity, or None when it is absent or too fresh to stand for its content.
    A rollback-journal commit moves the change counter in the file's header; a WAL commit adds a
    commit frame to the log, which keeps it until a checkpoint moves it into the file
    (``store.log_identity``: the frame and its checksum, the log's size and time), so neither journal
    trusts a file time alone, which Windows may keep while a cached connection holds the file. A
    store or log written within the time grain is not cached."""
    from ouroboros.store import STORE_REL, log_identity, read_file_bytes

    database = root / STORE_REL
    try:
        info = database.stat()
        header = read_file_bytes(database, limit=100)  # the borrowed descriptor: no lock of ours closes
    except OSError:
        return None
    log = log_identity(database)
    if time.time() - max(info.st_mtime_ns, log[4] if log else 0) / 1e9 < _STAMP_GRAIN_SEC:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, header[24:28], tuple(log) if log else None)


class TaskReads:
    """The documents, queue snapshot and cancel intents one read of many tasks uses."""

    def __init__(self, drive_root: Any) -> None:
        self.root = pathlib.Path(drive_root)
        self._documents: Dict[Tuple[str, str], Optional[str]] = {}  # (root, task id) -> view text
        self._decoded: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}  # this read's own lookups, never handed out
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._intents: Dict[str, Dict[str, Any]] = {}
        self._events: Any = None
        self._stamps: Dict[str, Optional[tuple]] = {}  # child store stamps, one stat per read
        self.versions: Dict[str, int] = {}  # canonical task id -> write_seq of the view read
        self._successors: Dict[str, str] = {}  # canonical task id -> the successor its projection overlays
        self.store_reads = 0  # store transactions this read opened (cost evidence for tests)
        self.failed = False  # a read the store could not answer: what this page shows is not known to be current

    # -- documents ------------------------------------------------------------------------
    def _fetch(self, root: pathlib.Path, task_ids: Iterable[str]) -> None:
        from ouroboros.task_store import display_documents

        key = str(root)
        wanted = sorted({str(task_id) for task_id in task_ids if (key, str(task_id)) not in self._documents})
        if not wanted:
            return
        if root != self.root and key not in self._stamps:
            self._stamps[key] = _store_stamp(root)
        stamp = self._stamps.get(key) if root != self.root else None
        with _REPLICAS_LOCK:
            cached = _REPLICAS.get(key) if stamp is not None else None
            if cached is not None and cached[0] == stamp and all(tid in cached[1] for tid in wanted):
                _REPLICAS.move_to_end(key)
                for tid in wanted:
                    self._documents[(key, tid)] = cached[1][tid]
                return
        self.store_reads += 1
        try:
            found = display_documents(root, wanted, self.versions if root == self.root else None)
        except Exception:
            self.failed = True
            raise
        for tid in wanted:
            self._documents[(key, tid)] = found.get(tid)
        if stamp is not None:
            with _REPLICAS_LOCK:
                views = dict(_REPLICAS[key][1]) if key in _REPLICAS and _REPLICAS[key][0] == stamp else {}
                views.update({tid: self._documents[(key, tid)] for tid in wanted})
                _REPLICAS[key] = (stamp, views)
                _REPLICAS.move_to_end(key)
                while len(_REPLICAS) > _REPLICAS_MAX:
                    _REPLICAS.popitem(last=False)

    def prefetch(self, task_ids: Iterable[str]) -> None:
        """Read ``task_ids`` from the canonical root, then, a wave each, the successors and
        child replicas their effective projection overlays. A batch the store cannot answer now
        is left to each ``load``, which then reads (and fails, migration included) on its own."""
        try:
            self._prefetch(task_ids)
        except Exception:
            log.debug("Batched task facts unavailable; reading per task", exc_info=True)

    def _prefetch(self, task_ids: Iterable[str]) -> None:
        from ouroboros.task_results import cancellation_blocks_child_result
        from ouroboros.task_status import _child_drive_candidates

        frontier = {str(task_id) for task_id in task_ids if str(task_id or "").strip()}
        seen = set(frontier)
        while frontier:
            self._fetch(self.root, frontier)
            successors = set()
            for task_id in frontier:
                document = self._peek(self.root, task_id) or {}
                successor = str(document.get("superseded_by") or document.get("retry_task_id") or "").strip()
                if successor:
                    self._successors[task_id] = successor
                if successor and successor not in seen:
                    successors.add(successor)
            seen |= successors
            frontier = successors
        replicas: Dict[pathlib.Path, set] = {}
        for task_id in seen:
            document = self._peek(self.root, task_id)
            if not document or cancellation_blocks_child_result(document):
                continue
            for child in _child_drive_candidates(document):
                if child != self.root:
                    replicas.setdefault(child, set()).add(task_id)
        for child, ids in replicas.items():
            self._fetch(child, ids)

    def overlaid(self, task_ids: Iterable[str]) -> set:
        """``task_ids`` and the canonical successors their effective projections overlay, as far as
        the batches read them: the documents whose versions say what the projection showed."""
        found = frontier = {str(task_id) for task_id in task_ids}
        while frontier:
            frontier = {self._successors[tid] for tid in frontier if tid in self._successors} - found
            found = found | frontier
        return found

    def _peek(self, root: pathlib.Path, task_id: str) -> Optional[Dict[str, Any]]:
        """A view decoded once per read for the prefetch's own lookups (a fetched text never changes)."""
        key = (str(root), str(task_id))
        if key not in self._decoded:
            text = self._documents.get(key)
            self._decoded[key] = json.loads(text) if text is not None else None
        return self._decoded[key]

    def load(self, root: Any, task_id: str) -> Optional[Dict[str, Any]]:
        """One document's display view (a private copy, as ``load_task_result`` returns), read
        now when no batch held it."""
        from ouroboros.task_results import _normalize_swarm_efficiency

        root = pathlib.Path(root)
        if (str(root), str(task_id)) not in self._documents:
            self._fetch(root, [str(task_id)])
        text = self._documents.get((str(root), str(task_id)))
        return _normalize_swarm_efficiency(json.loads(text)) if text is not None else None

    def absent(self, task_id: str) -> bool:
        """Whether the canonical store answers that ``task_id`` has no row (asked now when no
        batch did); a store that cannot answer raises, so uncertainty is never absence."""
        key = (str(self.root), str(task_id))
        if key not in self._documents:
            self._fetch(self.root, [str(task_id)])
        return key in self._documents and self._documents[key] is None

    def effective(self, task_id: str) -> Dict[str, Any]:
        """``effective_task_result`` of one task without artifact materialization, every
        document it overlays read through this object; ``{}`` when the store has no row."""
        from ouroboros import task_status

        return task_status.load_effective_task_result(self.root, task_id, materialize_artifacts=False, reads=self)

    # -- once-per-read facts ----------------------------------------------------------------
    def queue_snapshot(self, root: Any) -> Dict[str, Any]:
        from ouroboros.task_status import _load_queue_snapshot

        key = str(pathlib.Path(root))
        if key not in self._queue:
            self._queue[key] = _load_queue_snapshot(pathlib.Path(root))
        return self._queue[key]

    def cancel_intents(self, root: Any) -> Dict[str, Any]:
        from ouroboros.cancel_intents import active_intents

        key = str(pathlib.Path(root))
        if key not in self._intents:
            self._intents[key] = active_intents(pathlib.Path(root))
        return self._intents[key]

    @property
    def events_index(self) -> Any:
        if self._events is None:
            from ouroboros.task_status import _EventsTailIndex

            self._events = _EventsTailIndex(self.root)
        return self._events
