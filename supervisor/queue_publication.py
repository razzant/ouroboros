"""Queue publication by revision: transitions under Q, documents outside it.

Every queue transition marks the publication dirty under Q (``persist_queue_snapshot``: one
revision bump, no bytes). ONE publisher, the supervisor tick (``publish_queue_changes``),
captures the rows under Q when the revision moved and serializes and writes them outside
Q. Publication follows revision or pool-census changes, one writer at a time,
only from a process that initialized the queue; boot without a provider writes
no snapshot, dispatch or alive document. The capture copies containers only (``_frozen``): strings are shared, so the owner's
text is never copied under the lock. A large task value that appears in more than one place (the
owner's message in twenty rows, the task contract twice per row) is published once, by
content address (``$shared``, the reference ``ouroboros/shared_values.py`` owns), so the document carries the owner's
words once; ``queue_observation.load_queue_document`` expands them in memory, so every reader still sees the whole row.

The publication order a reader relies on is kept: the old ``queue.alive`` certificate is
revoked, the documents land (fsync), the new certificate names them (no fsync). Two
documents: the snapshot and the dispatch projection (``queue_dispatch.json``, the money
paths' fence authority). Only a document whose bytes changed is rewritten, under the new
generation; one whose bytes did not keeps the generation it was written under, and the
certificate names each document's own generation (``generation`` is the snapshot's,
``documents`` the others'), so a reader certifies a document by its own generation
(``queue_observation.load_queue_document``) and a torn bundle is still unknown as a whole.
``queue.alive`` is rewritten only when a document's generation or the heartbeat map changed
or its stamp aged past ``QUEUE_ALIVE_REFRESH_SEC``; a process that never published writes
none (the uncustodied saved queue rule, ``queue_snapshot._keeps_saved_queue``).

Durability before an effect is an explicit barrier (``publish_queue_snapshot``), used only
where a caller acts on the publication; it publishes where its caller stands, outside Q (Q
spans memory only). The tick's own write leaves the supervisor thread: the server starts one
publisher thread (``start_queue_writer``, ended with the loop by ``stop_queue_writer``) that
writes the tick's captures and certificate refreshes, so event dispatch never waits on the
disk; it holds the newest capture only (a slow disk never queues a backlog), the tick answers
False once for each failed write the writer completed (on the tick after it, whose capture is
the complete rewrite, FU1's rule unchanged), True while a retry is still on the disk until
``LOOP_CRASH_LIMIT`` writes failed in a row and False on every tick from then until one lands,
an older capture a barrier overtook is skipped, and a capture of an ended generation
(``reset``) is never written.
A process with no live tick publisher (startup before the loop, shutdown after it, tests)
publishes synchronously from every mark, so the boot and exit doors keep their semantics
without a second protocol. A process-ending door (``pre_restart_exit``; the shutdown kill as
``server_shutdown``) publishes embedded rows without the shared table, so an older body
restores after a rollback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.project_admission import carried_basis
from ouroboros.queue_observation import QUEUE_ALIVE_FRESH_SEC, SHARED_KEY
from ouroboros.shared_values import token
from ouroboros.subagents import subagent_intent_fields
from ouroboros.utils import machine_json, utc_now_iso, write_text_atomic
from supervisor.queue_snapshot import QUEUE_SNAPSHOT_SCHEMA_VERSION, _PENDING_FIELDS, _keeps_saved_queue
from supervisor.work_stop_migration import FORMAT_KEY, FORMAT_VERSION


def _queue():
    """The parent module, read at call time (the D18/D33 handle: ``init`` rebinds its names)."""
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)

#: ``queue.alive`` is refreshed inside the readers' freshness bound (``task_status``), without fsync.
QUEUE_ALIVE_REFRESH_SEC = QUEUE_ALIVE_FRESH_SEC / 4
#: A task value whose JSON is at least this many characters long may be shared by content address.
SHARED_MIN_LENGTH = 1024
#: Publications that always open a new generation, whatever the bytes.
_FORCED_REASONS = frozenset({"startup", "pre_restart_exit", "server_shutdown"})
#: The process-ending doors: their publication embeds every row whole, without the shared table,
#: so an older body restores it after a rollback (the restart exit; the shutdown kill of every door).
_EXPORT_REASONS = frozenset({"pre_restart_exit", "server_shutdown"})
_ALIVE_NAME = "queue.alive"
_SNAPSHOT_NAME = "queue_snapshot.json"
#: The RUNNING-row facts a publication carries besides the task. The capture writes exactly these and
#: the tick's census watches exactly these, so a fact a worker report changes under Q without a mark
#: (a silence episode, the quota clock, an owner wait) is published by the next tick, and a fact added
#: here later is watched by construction (RV_MAIN M6).
_ROW_FACTS = ("attempt", "worker_id", "owner_wait", "started_at", "model_wait_quota_clock",
              "sleep_parked_at", "budget_paused_sec", "soft_sent")
#: Task facts a worker reports without a queue transition; every other task fact changes with a mark.
_REPORTED_TASK_FACTS = ("model_waits", "focus")

# Revision: bumped under Q by every mark and capture. Published: the revision of the last
# capture written. Kept: the (revision, custody) at which the saved queue was last kept.
# Revision: bumped under Q by every mark and capture. Published: the revision of the last capture
# written. Captures are numbered under Q too (``capture_seq``): the number decides which of two
# captures is newer, whatever their revisions (a tick captures without a bump). Census: the last
# census WRITTEN, never a census merely captured. A failed write leaves the bundle on disk unknown
# (torn or old) with its certificate revoked: from then on the publisher owes a fresh capture and a
# complete rewrite of every document under a new generation, whatever the bytes say against the last
# cache, and ``write_failed`` clears only when that write lands; no certificate is written meanwhile.
# Failed writes are counted (``failed_writes``) so the tick answers each one once (``failures_answered``:
# the count it has answered), never a retry still on the disk; ``failed_streak`` is the run of failed writes
# with no landed write between them, and once it reaches the loop's own limit every tick is a crash.
_STATE: Dict[str, Any] = {"revision": 0, "published": 0, "reason": "", "kept": None, "census": None,
                          "capture_seq": 0, "published_seq": 0, "write_failed": False,
                          "failed_writes": 0, "failures_answered": 0, "failed_streak": 0,
                          "alive_heartbeats": None, "alive_written_at": 0.0}
#: Consecutive failed writes after which the tick answers False on every tick until a write lands: the supervisor
#: loop's own ending rule (``server._run_supervisor`` dies after three consecutive crashes), so a disk that fails
#: every write slowly ends the loop as one that fails fast does, instead of resetting the loop's count on each tick
#: a retry spends on the disk.
LOOP_CRASH_LIMIT = 3
_PUBLISHER: Optional[threading.Thread] = None
_PUBLISH_LOCK = threading.Lock()


class _TickWriter:
    """The publisher thread of one queue generation: it writes the tick's captures and certificate refreshes.

    The tick offers its capture (``offer``) and goes on; a capture the thread has not taken yet is replaced by a
    newer one, so a slow disk never queues a backlog and the write that lands is the newest state (an older
    capture a barrier overtook is skipped by ``_publish`` all the same). A quiet tick asks for a certificate
    refresh (``refresh``). ``drain`` waits for everything offered before it (tests, a door that wants the
    tick's last capture down before its own export). ``stop`` ends the thread with the loop that fed it:
    the doors after the loop publish synchronously where they stand.
    """

    def __init__(self) -> None:
        self.thread: Optional[threading.Thread] = None
        self.wake = threading.Condition()
        self.capture: Optional[Dict[str, Any]] = None  # the newest capture not taken yet
        self.refresh = False
        self.syncs: list = []
        self.closed = False

    def start(self) -> None:
        with self.wake:
            if self.closed or (self.thread is not None and self.thread.is_alive()):
                return
            self.thread = threading.Thread(target=self._run, name="queue-publisher", daemon=True)
            self.thread.start()

    def alive(self) -> bool:
        thread = self.thread
        return thread is not None and thread.is_alive() and not self.closed

    def offer(self, capture: Dict[str, Any]) -> None:
        with self.wake:
            self.capture = capture  # a capture not taken yet is superseded: the newest state is what lands
            self.wake.notify()

    def ask_refresh(self) -> None:
        with self.wake:
            self.refresh = True
            self.wake.notify()

    def drain(self, timeout: float) -> bool:
        if not self.alive():
            return True
        done = threading.Event()
        with self.wake:
            self.syncs.append(done)
            self.wake.notify()
        return done.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """End the thread once it wrote what it holds (``reset`` rebinds the module's writer first, so an ended
        generation's capture is refused by ``_publish`` instead)."""
        with self.wake:
            self.closed = True
            self.wake.notify_all()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self.wake:
                while not (self.closed or self.capture is not None or self.refresh or self.syncs):
                    self.wake.wait()
                capture, refresh, syncs = self.capture, self.refresh, self.syncs
                self.capture, self.refresh, self.syncs = None, False, []
                if capture is None and not refresh and self.closed:  # stopped: what it still held was written first
                    for done in syncs:
                        done.set()
                    return
            try:
                if capture is not None:
                    _publish(capture, writer=self)  # a publication certifies too: the refresh is subsumed
                elif refresh:
                    _refresh_alive()
            except Exception:
                log.warning("queue publisher thread: the write failed", exc_info=True)
            for done in syncs:
                done.set()


_WRITER = _TickWriter()


def reset() -> None:
    """A new queue generation (``queue.init``): nothing marked, nothing published, no tick publisher, no writer."""
    global _PUBLISHER, _WRITER
    _STATE.update(revision=0, published=0, reason="", kept=None, census=None, capture_seq=0, published_seq=0,
                  write_failed=False, failed_writes=0, failures_answered=0, failed_streak=0, alive_heartbeats=None,
                  alive_written_at=0.0)
    _PUBLISHER = None
    ended, _WRITER = _WRITER, _TickWriter()  # rebound first: a capture of the ended generation is refused by identity
    ended.stop()


def start_queue_writer() -> None:
    """The server's loop, after ``tick_publisher_started``: the tick's captures are written on this daemon thread."""
    _WRITER.start()


def stop_queue_writer(timeout: float = 5.0) -> None:
    """The loop's every exit: the writer ends with the tick that fed it (the doors after it publish synchronously)."""
    _WRITER.stop(timeout)


def writer_alive() -> bool:
    return _WRITER.alive()


def drain_writer(timeout: float = 5.0) -> bool:
    """Wait until everything offered to the writer before this call was written (or failed); True when it drained
    in time, True at once without a writer."""
    return _WRITER.drain(timeout)


def tick_publisher_started() -> None:
    """The calling thread (the supervisor loop) publishes marks on its tick while it lives."""
    global _PUBLISHER
    _PUBLISHER = threading.current_thread()


def tick_publisher_alive() -> bool:
    return _PUBLISHER is not None and _PUBLISHER.is_alive()


def queue_revision() -> int:
    return _STATE["revision"]


def persist_queue_snapshot(reason: str = "") -> Optional[bool]:
    """Mark the queue changed: one revision bump under Q, published by the tick.

    True means ON DISK, nothing less. Without a live tick publisher the caller publishes
    synchronously (startup, shutdown, tests): True when published, False when the write
    failed, None when the documents on disk are kept (no queue in this process, or the saved
    queue is not in its custody). With a live tick publisher the mark answers None: the change
    is recorded and the next tick publishes it, but nothing is on disk yet, so no caller can
    read a mark as a persisted fact. A caller whose effect or reply needs the publication
    first uses ``publish_queue_snapshot`` (or ``barrier_of`` through its context).
    """
    if not _queue().INITIALIZED:  # a boot without a provider: state initialization reads a snapshot as history (#2412)
        log.info("Queue snapshot not published (reason=%s): no supervisor queue in this process", reason)
        return None
    with _queue()._queue_lock:
        _STATE["revision"] += 1
        _STATE["reason"] = reason
    if tick_publisher_alive():
        return None
    return _publish_now(reason)


def barrier_of(ctx: Any):
    """The publication barrier a handler reaches through its context: the context's own
    ``publish_queue_snapshot`` (the server's event context), else whatever stands in for its mark
    there (a test context's stub stands in for the barrier too), else None: the caller decides
    what an absent persister means for its effect."""
    return getattr(ctx, "publish_queue_snapshot", None) or getattr(ctx, "persist_queue_snapshot", None)


def publish_queue_snapshot(reason: str = "") -> Optional[bool]:
    """The barrier: capture under Q, write outside it, return only when the documents landed.

    True when published, False when the publication failed (the documents on disk are the
    previous generation's), None when they are kept. Without a live tick publisher the
    barrier and the mark are one operation, so it goes through the queue's own
    ``persist_queue_snapshot`` name: whatever stands in for the mark there (a test's stub)
    stands in for the barrier too. A reason in ``_EXPORT_REASONS`` embeds every row.
    """
    if not tick_publisher_alive():
        return _queue().persist_queue_snapshot(reason=reason)
    return _publish_now(reason)


def _publish_now(reason: str) -> Optional[bool]:
    if not _queue().INITIALIZED:
        log.info("Queue snapshot not published (reason=%s): no supervisor queue in this process", reason)
        return None
    with _queue()._queue_lock:
        _STATE["revision"] += 1
        capture = _capture(reason)
    if capture is None:
        return None
    return _publish(capture)


def publish_queue_changes() -> Optional[bool]:
    """The tick: publish when the revision or the census moved, else refresh the certificate.

    True when published or refreshed, None when the documents on disk are kept (an
    uncustodied saved queue, or nothing of this process certified yet), False when a write
    failed: the loop treats False as its own crash (``server._run_supervisor``), None as a
    quiet tick. With the writer thread running (``start_queue_writer``) the capture is
    handed to it and the tick answers False once for each failed write the writer completed
    since the tick last asked (one tick later; that tick's capture, the bundle being unknown
    until a complete rewrite lands, R2, is the retry), and not for a retry still on the disk
    while fewer than ``LOOP_CRASH_LIMIT`` writes failed in a row: one failed write is one
    crash, a slow retry that lands is none. Once that many failed in a row with nothing landed
    between them, every tick answers False until a write lands, so a disk that fails every
    write ends the loop whether its writes fail fast (three ticks) or slowly (the ticks the
    next attempt spends on the disk). True otherwise means the writer holds the newest capture.
    """
    if not _queue().INITIALIZED:
        return None
    with _queue()._queue_lock:
        if _kept_by_memo():
            return None  # the saved queue is still kept: decided once, not copied or counted again
        # The census: what the rows publish without a transition (worker reports mutate RUNNING in place
        # with no mark). One small identity per tick, so a changed fact is published and an unchanged
        # live set still captures, serializes and writes nothing. A failed write is itself a reason to
        # publish: the bundle on disk is unknown until a complete rewrite lands (R2).
        census = _census()
        moved = (_STATE["revision"] != _STATE["published"] or census != _STATE["census"] or _STATE["write_failed"])
        capture = _capture(_STATE["reason"] or "main_loop", census=census) if moved else None
    if not _WRITER.alive():
        # Written where the tick stands: the outcome answers for this tick's own write, so every failure so
        # far is answered with it.
        outcome = _refresh_alive() if not moved else (None if capture is None else _publish(capture))
        _STATE["failures_answered"] = _STATE["failed_writes"]
        return outcome
    # The writer thread lands it (the newest capture replaces one it has not taken yet); a quiet tick asks it for
    # the certificate refresh; the tick answers the writer's failures since it last asked, and a kept saved queue
    # stays None.
    if capture is not None:
        _WRITER.offer(capture)
    elif not moved and _queue().QUEUE_PUBLICATION is not None:
        _WRITER.ask_refresh()
    else:
        return None  # kept, or nothing of this process certified yet
    failed, answered = _STATE["failed_writes"], _STATE["failures_answered"]
    _STATE["failures_answered"] = failed
    return failed == answered and _STATE["failed_streak"] < LOOP_CRASH_LIMIT


def _frozen(value: Any) -> Any:
    """A copy of the containers; strings, numbers and None are shared (immutable)."""
    if isinstance(value, dict):
        return {key: _frozen(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_frozen(item) for item in value]
    return value


def _row_facts(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The published facts of one RUNNING row besides its task, as the capture writes them.

    A dated silence episode rides the row as a fact for its readers; ordinary progress
    rotates nothing.
    """
    facts = {key: meta.get(key) for key in _ROW_FACTS}
    facts.update(started_at=float(meta.get("started_at") or 0.0),
                 budget_paused_sec=float(meta.get("budget_paused_sec") or 0.0), soft_sent=bool(meta.get("soft_sent")))
    silence = meta.get("silence_attention")
    if silence:
        facts["silence_attention"] = silence
    return facts


def _census() -> str:
    """Under Q: one small identity of what the rows publish without a transition, and the pool counts."""
    rows = []
    for task_id, meta in _queue().RUNNING.items():
        if isinstance(meta, dict):
            task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
            reported = {key: task.get(key) for key in _REPORTED_TASK_FACTS}
            # A resolved wait is history the row keeps for its readers; only a wait that waits is a live fact.
            waits = reported.get("model_waits")
            if isinstance(waits, dict):
                reported["model_waits"] = {wid: row for wid, row in waits.items()
                                           if isinstance(row, dict) and row.get("state") == "waiting"}
            rows.append((task_id, _row_facts(meta), reported))
    return machine_json([rows, _pool_counts(census=True)])


def _custody_key() -> tuple:
    """Under Q: what the kept decision depends on (the revision, and whether a restore owns the saved queue)."""
    from supervisor.queue_snapshot import _RESTORED, _RESTORING

    key = str(_queue().QUEUE_SNAPSHOT_PATH.absolute())
    return (_STATE["revision"], key in _RESTORED, _RESTORING.get(key))


def _kept_by_memo() -> bool:
    """Under Q: the saved queue was decided kept at this same custody, so nothing is copied or counted."""
    return _STATE["kept"] is not None and _STATE["kept"] == _custody_key()


def _capture(reason: str, *, census: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Under Q: the thin rows and queue facts of this revision; None when the saved queue is kept."""
    if _kept_by_memo():
        return None
    census = _census() if census is None else census
    _STATE["capture_seq"] += 1
    pending_items = list(_queue().PENDING)
    running_items = [(task_id, meta if isinstance(meta, dict) else {}) for task_id, meta in _queue().RUNNING.items()]
    acceptance_fences = [_frozen(row) for row in _queue().ACCEPTANCE_FENCES.values()]
    budget_root_fences = [_frozen(row) for row in _queue().BUDGET_ROOT_FENCES.values()]
    if _keeps_saved_queue(_queue().QUEUE_SNAPSHOT_PATH, pending_items, running_items,
                          acceptance_fences, budget_root_fences):
        log.info("Queue snapshot kept (reason=%s): this process has not taken the saved queue into custody", reason)
        _STATE["kept"] = _custody_key()
        return None
    pending_rows = []
    for t in pending_items:
        pending_rows.append({
            "id": t.get("id"), "type": t.get("type"), "priority": t.get("priority"),
            "attempt": t.get("_attempt"), "queued_at": t.get("queued_at"),
            "queue_seq": t.get("_queue_seq"),
            "task": _frozen({**{key: t.get(key) for key in _PENDING_FIELDS}, **subagent_intent_fields(t), **carried_basis(t),
                             **{key: t[key] for key in ('admitted_dispatch', '_project_scope_none', '_working_recovery', '_seed_return',
                                                        # #939: the execution budget and its carried clock offset travel with the row.
                                                        'execution_timeout_sec', '_execution_consumed_sec', '_execution_handoff_sec',
                                                        '_execution_owner',  # its carrier: a returned turn keeps its lane
                                                        ) if key in t}}),
        })
    running_rows = []
    for task_id, meta in running_items:
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        running_rows.append({"id": task_id, "type": task.get("type"), "priority": task.get("priority"),
                             **_frozen(_row_facts(meta)), "task": _frozen(task)})
    from ouroboros.server_restart import _AGENT_RESTART_REPLY, agent_restart_snapshot

    return {
        "revision": _STATE["revision"], "seq": _STATE["capture_seq"], "census": census,
        "reason": reason, "rollback_readable": reason in _EXPORT_REASONS,
        "forced": reason in _FORCED_REASONS,
        "pending": pending_rows, "running": running_rows,
        "acceptance_fences": acceptance_fences, "budget_root_fences": budget_root_fences,
        "pool": _pool_counts(),
        "restart_intent": agent_restart_snapshot(), "restart_reply": dict(_AGENT_RESTART_REPLY),
        "heartbeats": {tid: meta.get("last_heartbeat_at") for tid, meta in running_items},
    }


def _pool_counts(*, census: bool = False) -> Dict[str, Any]:
    """The pool's published facts, measured by ``worker_pool_lifecycle.pool_slot_counts`` (its docstring names
    each count) from the ACTUAL pool, plus the last reading of the machine's memory (``memory_admission``:
    ``pool_admission.facts``, dated by its own ``read_at``). A slot mid-reap is not
    assignable, so no reader is told of a free slot that is not. A failed measurement publishes
    ``pool_observation_status: "unknown"`` with null counts: never zeros, which would read as an empty pool. The
    census hashes only whether starts are admitted and their reason in place of the reading's bytes, so a
    change of verdict or reason publishes and byte noise does not."""
    try:
        from supervisor import pool_admission, workers as _workers_mod
        from supervisor.worker_pool_lifecycle import pool_slot_counts

        counts = pool_slot_counts(_workers_mod.WORKERS)
        disabled = _workers_mod._worker_pool_execution_state()["disabled_reason"]
        # Absence is counted above, not an explicit disablement.
        counts["worker_pool_disabled_reason"] = "" if disabled == "no_workers" else disabled
        counts["pool_observation_status"] = "observed"
        memory = pool_admission.facts()
        counts["memory_admission"] = (bool(memory.get("starts_admitted")), memory.get("reason", "")) if census else memory
        return counts
    except Exception:
        log.debug("Worker pool facts unavailable; published as unknown", exc_info=True)  # every tick: no warning flood
        return {**dict.fromkeys(_POOL_COUNT_KEYS), "worker_pool_disabled_reason": "unknown",
                "pool_observation_status": "unknown", "pool_observation_error": "introspection_failed"}


#: The counts ``pool_slot_counts`` measures; an unknown observation publishes each as null.
_POOL_COUNT_KEYS = ("max_workers", "worker_total", "parked_worker_count", "busy_worker_count",
                    "acting_worker_count", "booting_count", "reaping_count", "retiring_count",
                    "assignable_idle_workers", "lane_carrier_total", "memory_admission")


def _documents(capture: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Outside Q: the two documents of this capture; task values interned unless rollback-readable."""
    from ouroboros.budget_pause import BUDGET_HOLD_KEY

    payload = {
        SCHEMA_VERSION_KEY: QUEUE_SNAPSHOT_SCHEMA_VERSION,
        # Embedded rows are the shape format 1 named; an older body treats only 1 as current and would
        # otherwise run its legacy conversion over this export (RV_MAIN, queue publication minor).
        FORMAT_KEY: 1 if capture["rollback_readable"] else FORMAT_VERSION,
        "pending_count": len(capture["pending"]), "running_count": len(capture["running"]),
        **capture["pool"],
        "acceptance_fences": capture["acceptance_fences"],
        "budget_root_fences": capture["budget_root_fences"],
        "pending": capture["pending"], "running": capture["running"],
        "restart_intent": capture["restart_intent"], "restart_reply": capture["restart_reply"],
    }
    fenced = {str(row.get("root_task_id") or "") for row in capture["budget_root_fences"]}
    members = {}
    for task in [row["task"] for row in capture["pending"]] + [row["task"] for row in capture["running"]]:
        if str(task.get("root_task_id") or task.get("id") or "") in fenced:
            # Rows carry every _PENDING_FIELDS key (None when unset); a member names only what is set, as the
            # live tasks did: readers ask ``key in member`` (``queue_transitions`` reads ``_working_recovery`` so).
            member = {key: task[key] for key in ("id", "_budget_pause", "_continuation_prepared")
                      if task.get(key) is not None}
            for key, fields in ((BUDGET_HOLD_KEY, ("reason", "release")),
                                ("_owner_wait_resume", ("refusal",)), ("_working_recovery", ("refusal",)),
                                ("_resume_selection", ("fence_id", "refusal")),
                                ("_budget_pause_resume", ("pause_id", "grant_id", "authority", "root_fence_id", "selected_by", "root_grant_id", "root_resume_generation"))):
                if task.get(key) is not None:
                    value = task[key]
                    member[key] = {k: value[k] for k in fields if k in value} if isinstance(value, dict) else value
            members[str(task.get("id") or "")] = member
    if not capture["rollback_readable"]:
        payload[SHARED_KEY] = _share_replicated(payload)
    return {_SNAPSHOT_NAME: payload,
            "queue_dispatch.json": {"budget_root_fences": capture["budget_root_fences"], "members": members}}


def _share_replicated(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace every large task value that appears more than once (top level and one level inside
    ``metadata``) by its address; the table of those values is returned. A value published once
    stays where it is: only replication is waste, and a raw reader still sees singletons inline.
    The rows are this publication's own frozen capture, so they are rewritten in place."""
    seen: Dict[str, list] = {}  # digest -> [value, count]
    places = []  # (container, key, digest) of every large value, in document order
    for lane in ("pending", "running"):
        for row in payload[lane]:
            task = row["task"]
            for key, value in task.items():
                if key == "metadata" and isinstance(value, dict):
                    for name, item in value.items():
                        if digest := _digest(item):
                            seen.setdefault(digest, [item, 0])[1] += 1
                            places.append((value, name, digest))
                elif digest := _digest(value):
                    seen.setdefault(digest, [value, 0])[1] += 1
                    places.append((task, key, digest))
    shared: Dict[str, Any] = {}
    for container, key, digest in places:
        value, count = seen[digest]
        if count > 1:
            shared.setdefault(digest, value)
            container[key] = token(digest)
    return shared


def _digest(value: Any) -> str:
    """The content address of a large value, or "" for one too small to be worth sharing."""
    if isinstance(value, str):
        if len(value) < SHARED_MIN_LENGTH:
            return ""
        canonical = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, (dict, list)) and value:
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if len(canonical) < SHARED_MIN_LENGTH:
            return ""
    else:
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _publish(capture: Dict[str, Any], *, writer: Optional[_TickWriter] = None) -> bool:
    """Outside Q, one writer at a time: encode, compare, then revoke -> changed documents -> certify.

    ``writer``: the tick's thread names itself, so a capture of a generation that ended while it was encoding
    (``reset`` replaced the writer) is never written over the next generation's documents.
    """
    documents = _documents(capture)
    encoded = {name: machine_json(document) for name, document in documents.items()}
    with _PUBLISH_LOCK:
        if writer is not None and writer is not _WRITER:
            return False  # an ended generation's capture: the next generation owns the documents now
        previous = _queue().QUEUE_PUBLICATION
        if previous is not None and _STATE["published_seq"] > capture["seq"] and not capture["forced"]:
            # A later capture already published newer documents; while a later write failed, the bundle on
            # disk is unknown, so that publication is no durability for this caller either.
            return not _STATE["write_failed"]
        # After a failed write the bundle on disk is unknown: byte equality with the last cache proves nothing,
        # so every document is rewritten under a new generation and certified (R2/R3-A).
        if (previous is None or previous[0] != _queue().QUEUE_SNAPSHOT_PATH or capture["forced"]
                or _STATE["write_failed"]):
            stale = list(documents)
        else:
            stale = [name for name in documents if previous[1].get(name) != encoded[name]]
        changed = bool(stale)
        generations = dict(previous[3]) if previous is not None else {}
        if changed:
            generations.update(dict.fromkeys(stale, uuid.uuid4().hex))
        state = _queue().QUEUE_SNAPSHOT_PATH.parent
        try:
            if changed:
                _revoke_alive(state)  # A partial publish must not certify the old no-fence document.
                for name in stale:
                    published = {**documents[name], "generation": generations[name], "ts": utc_now_iso(),
                                 "reason": capture["reason"]}
                    if _queue().atomic_write_text(state / name, machine_json(published)) is False:
                        raise OSError(f"queue publication failed: {name}")
            if changed or _alive_due(capture["heartbeats"]):
                _certify(state, generations, capture["heartbeats"])
            if changed:
                _queue().QUEUE_PUBLICATION = (_queue().QUEUE_SNAPSHOT_PATH, encoded, generations[_SNAPSHOT_NAME],
                                              generations)
            _STATE["published"] = max(_STATE["published"], capture["revision"])
            _STATE["published_seq"] = max(_STATE["published_seq"], capture["seq"])
            _STATE["census"] = capture["census"]  # written, so the tick's next census compares against it
            _STATE["write_failed"] = False
            _STATE["failed_streak"] = 0  # a landed write ends the run of failures
            return True
        except Exception:
            # The census stays the last WRITTEN one: the next tick captures and retries this change; and
            # the stale document on disk is not certified again until a publication lands (N9).
            _STATE["write_failed"] = True
            _STATE["failed_writes"] += 1  # one completed failed write: the tick answers it once
            _STATE["failed_streak"] += 1  # ...and at the loop's limit, on every tick until a write lands
            log.warning("Failed to persist queue snapshot (reason=%s)", capture["reason"], exc_info=True)
            return False


def _refresh_alive() -> Optional[bool]:
    """A tick with nothing to publish: keep the certificate fresh for its readers, never a new generation."""
    published = _queue().QUEUE_PUBLICATION
    if published is None:
        return None  # nothing of this process is certified; a kept saved queue stays uncertified
    if _STATE["write_failed"]:
        return False  # the document on disk is older than the facts: no certificate until a write lands
    with _queue()._queue_lock:
        heartbeats = {tid: (meta.get("last_heartbeat_at") if isinstance(meta, dict) else None)
                      for tid, meta in _queue().RUNNING.items()}
    if not _alive_due(heartbeats):
        return True
    with _PUBLISH_LOCK:
        if _STATE["write_failed"]:
            return False  # a barrier failed while this refresh was in flight: its revoked certificate stays revoked (R3-B)
        if _queue().QUEUE_PUBLICATION is not published:
            return True  # a publication landed meanwhile and certified itself
        try:
            _certify(_queue().QUEUE_SNAPSHOT_PATH.parent, published[3], heartbeats)
        except Exception:
            log.warning("Failed to refresh queue liveness", exc_info=True)
            return False
    return True


def _alive_due(heartbeats: Dict[str, Any]) -> bool:
    return (heartbeats != _STATE["alive_heartbeats"]
            or time.monotonic() - _STATE["alive_written_at"] >= QUEUE_ALIVE_REFRESH_SEC)


def _revoke_alive(state) -> None:
    for attempt in range(3):
        try:
            (state / _ALIVE_NAME).unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.01)  # Windows readers briefly deny delete sharing.


def _certify(state, generations: Dict[str, str], heartbeats: Dict[str, Any]) -> None:
    """``queue.alive`` names each document's generation on disk (``generation``: the snapshot's; ``documents``: the
    others', absent when they share it); written without fsync (a crash loses only freshness)."""
    others = {name: generation for name, generation in generations.items()
              if name != _SNAPSHOT_NAME and generation != generations[_SNAPSHOT_NAME]}
    alive = {"generation": generations[_SNAPSHOT_NAME], "ts": utc_now_iso(), "heartbeats": heartbeats,
             **({"documents": others} if others else {})}
    if _write_alive(state / _ALIVE_NAME, json.dumps(alive, separators=(",", ":"))) is False:
        raise OSError("queue liveness publication failed")
    _STATE["alive_heartbeats"] = dict(heartbeats)
    _STATE["alive_written_at"] = time.monotonic()


def _write_alive(path, content: str) -> None:
    write_text_atomic(path, content, fsync=False)
