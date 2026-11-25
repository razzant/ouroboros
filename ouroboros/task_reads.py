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
# A child store's display views, with the revision each was read at: an ending comparison needs it.
_REPLICAS: "OrderedDict[str, Tuple[tuple, Dict[str, Tuple[Optional[str], Optional[int]]]]]" = OrderedDict()
_REPLICAS_MAX = 512  # child stores whose views a process keeps (one per delegated child drive)
_REPLICAS_LOCK = threading.Lock()
# The coarsest file time a supported file system keeps (FAT, HFS+ and some FUSE mounts: 1-2 s): a
# store or log written this recently may change again without its time moving, so it is not cached.
_STAMP_GRAIN_SEC = 2.0
# Two distinct non-empty answers, for the rare pair whose words this page cannot compare: unknown is
# never agreement, so the selector is put the question it answers by keeping the row's own ending.
_UNREADABLE_ANSWER = ("canonical answer unreadable", "replica answer unreadable")


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


def queue_task_rows(snapshot: Dict[str, Any]):
    """Running takes precedence over pending, and the first row in each lane wins."""
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED

    for lane, status in (("running", STATUS_RUNNING), ("pending", STATUS_SCHEDULED)):
        for row in snapshot.get(lane) or ():
            if isinstance(row, dict):
                task = row.get("task")
                yield str(row.get("id") or row.get("task_id") or ""), (status, task if isinstance(task, dict) else {})


class TaskReads:
    """The documents, queue snapshot and cancel intents one read of many tasks uses."""

    def __init__(self, drive_root: Any) -> None:
        self.root = pathlib.Path(drive_root)
        self._documents: Dict[Tuple[str, str], Optional[str]] = {}  # (root, task id) -> view text
        self._decoded: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}  # this read's own lookups, never handed out
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._queue_tasks: Dict[str, Dict[str, tuple]] = {}
        self._intents: Dict[str, Dict[str, Any]] = {}
        self._events: Any = None
        self._stamps: Dict[str, Optional[tuple]] = {}  # child store stamps, one stat per read
        self.versions: Dict[str, int] = {}  # canonical task id -> write_seq of the view read
        self.replica_results: set[str] = set()  # effective reads that loaded a noncanonical task document
        self._view_seqs: Dict[Tuple[str, str], Optional[int]] = {}  # (root, task id) -> the revision its view was read at
        self._ending_words: Dict[Tuple[str, str], Optional[str]] = {}  # one dropped field, read back by address
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
                    self._documents[(key, tid)], self._view_seqs[(key, tid)] = cached[1][tid]
                return
        self.store_reads += 1
        seqs: Dict[str, int] = {}  # each view's own revision, from the read that took it
        try:
            found = display_documents(root, wanted, seqs)
        except Exception:
            self.failed = True
            raise
        for tid in wanted:
            self._documents[(key, tid)] = found.get(tid)
            self._view_seqs[(key, tid)] = seqs.get(tid)
        if root == self.root:
            self.versions.update(seqs)
        if stamp is not None:
            with _REPLICAS_LOCK:
                views = dict(_REPLICAS[key][1]) if key in _REPLICAS and _REPLICAS[key][0] == stamp else {}
                views.update({tid: (self._documents[(key, tid)], self._view_seqs[(key, tid)]) for tid in wanted})
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
        if root != self.root and text is not None:
            self.replica_results.add(str(task_id))
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

    # -- the ending a display view may lend --------------------------------------------------
    def replica_overlay(self, canonical: Dict[str, Any], replica: Dict[str, Any],
                        canonical_root: Any, replica_root: Any) -> Dict[str, Any]:
        """The replica overlay the WHOLE documents would give, from the views one page read.

        ``post_task_checkpoint.canonical_keeps_replica_ending`` settles a same-attempt pair of contrary
        non-success endings partly on the answer's words, and ``task_store.DISPLAY_DROPPED`` keeps only
        ``_result_text_present`` for them: handed the views as they stand no word can contradict, so a
        reloaded card lends a losing copy's counters and closing sentence where the whole-document read
        and the copy-back refuse them. Where the views still allow that, this asks the store for just
        that one field of each row, each pinned to the revision its own view was taken at
        (``task_store.result_text_at``), and puts the pair to the SAME selector: one place still decides
        an ending. Words that are not this page's to compare (no captured revision, a row that moved
        since, a store that cannot answer) are not agreement, so the row keeps its own ending instead of
        lending one the read cannot confirm. A whole-document reader, a pair the views already settle, a
        foreign attempt, a ``completed`` or unsettled canonical row and a side holding no answer read
        nothing here, and the compared words never enter the page.
        """
        from ouroboros.post_task_checkpoint import (
            canonical_keeps_replica_ending, project_replica_task_result_fields,
        )

        pair = (canonical, replica)
        if self._views_may_contradict(canonical, replica):
            held = self._answer_words(canonical_root, canonical)
            seen = self._answer_words(replica_root, replica)
            if held is None or seen is None:
                held, seen = _UNREADABLE_ANSWER
            pair = ({**canonical, "result": held}, {**replica, "result": seen})
        overlay = project_replica_task_result_fields(*pair)
        if pair[0] is not canonical:
            overlay.pop("result", None)  # a field injected to settle an ending is no page content
        if "_result_text_present" in overlay and canonical_keeps_replica_ending(*pair):
            overlay.pop("_result_text_present", None)  # the answer's existence follows the ending that won
        return overlay

    def _views_may_contradict(self, canonical: Dict[str, Any], replica: Dict[str, Any]) -> bool:
        """Whether the answer's words could still change this pair's ending: a display pair the views
        have not settled, of one attempt, whose canonical row is a settled non-completed ending, and
        where BOTH sides say they hold an answer. Anything else is ordinary enrichment or already
        refused, so a page of those reads no answer back per id."""
        from ouroboros.post_task_checkpoint import canonical_keeps_replica_ending, replica_attempt_matches
        from ouroboros.task_results import STATUS_COMPLETED, _TRULY_TERMINAL_STATUSES

        if "result" in canonical and "result" in replica:
            return False  # a full reader holds the words already
        status = canonical.get("status")
        return (status in _TRULY_TERMINAL_STATUSES and status != STATUS_COMPLETED
                and replica_attempt_matches(canonical, replica)
                and not canonical_keeps_replica_ending(canonical, replica)
                and self._claims_answer(canonical) and self._claims_answer(replica))

    @staticmethod
    def _claims_answer(view: Dict[str, Any]) -> bool:
        """Whether this document says it holds a non-empty answer: the words, or the bit a display view
        keeps in their place. A side that holds none contradicts nothing and is worth no read."""
        return bool(view["result"]) if "result" in view else view.get("_result_text_present") is True

    def _answer_words(self, root: Any, view: Dict[str, Any]) -> Optional[str]:
        """One row's stored answer, read by address at most once per page and pinned to the revision
        this read took that view at; ``None`` when there is no revision to pin to, the row moved, or
        the store cannot answer (unknown, never an invented equality)."""
        from ouroboros.task_store import result_text_at

        key = (str(pathlib.Path(root)), str(view.get("task_id") or ""))
        if key not in self._ending_words:
            seq = self._view_seqs.get(key)
            words = None
            if seq is not None and key[1]:
                try:
                    words = result_text_at(root, key[1], seq)
                except Exception:
                    log.debug("Answer words unavailable for %s", key[1], exc_info=True)
            self._ending_words[key] = words
        return self._ending_words[key]

    # -- once-per-read facts ----------------------------------------------------------------
    def queue_snapshot(self, root: Any) -> Dict[str, Any]:
        from ouroboros.task_status import _load_queue_snapshot

        key = str(pathlib.Path(root))
        if key not in self._queue:
            self._queue[key] = _load_queue_snapshot(pathlib.Path(root))
        return self._queue[key]

    def queue_task_status(self, root: Any, task_id: str) -> tuple:
        """The same first running/pending row as task_status, indexed once for this read only."""
        key = str(pathlib.Path(root))
        snapshot = self.queue_snapshot(root)
        if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
            return "unknown", {}
        if key not in self._queue_tasks:
            index = {}
            for name, fact in queue_task_rows(snapshot):
                index.setdefault(name, fact)
            self._queue_tasks[key] = index
        return self._queue_tasks[key].get(task_id, ("", {}))

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
