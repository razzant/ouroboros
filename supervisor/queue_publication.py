"""Queue publication by revision: transitions under Q, documents outside it.

Every queue transition marks the publication dirty under Q (``persist_queue_snapshot``: one revision bump,
no bytes). ONE publisher, the supervisor tick (``publish_queue_changes``), captures the rows under Q when the
revision moved and serializes and writes them outside Q, one writer at a time, only from a process that
initialized the queue. The capture copies containers only (``_frozen``), so the owner's text is never copied
under the lock. A large value that appears in more than one place is published once, by content address
(``$shared``); ``queue_observation.load_queue_document`` expands it, so every reader still sees whole rows.

Two documents. The snapshot is certified by ``queue.alive``: the old certificate is revoked, the snapshot
lands (fsync, rename), the new certificate names its generation (no fsync). ``queue.alive`` also carries the
heartbeats and the pool's counts and is rewritten only when one of them or the snapshot's generation changed,
or its stamp aged past ``QUEUE_ALIVE_REFRESH_SEC``. The dispatch projection (``queue_dispatch.json``, the
money paths' fence) certifies itself: an atomic replacement under its own generation, so admission to a paid
call never waits for the snapshot. A tightening may land before the snapshot; a relaxation lands only after
the durable snapshot and its stamp; a mixed change first publishes the intersection of old and new
permissions. A failed replacement leaves the preceding fence readable and forces the next capture to rewrite.

Durability before an effect is an explicit barrier (``publish_queue_snapshot``), used only where a caller acts
on the publication, outside Q. The tick's own writes leave the supervisor thread: one publisher thread
(``start_queue_writer`` / ``stop_queue_writer``) holds the newest capture only, so a slow disk never queues a
backlog; the tick reports each completed write failure once, and after ``LOOP_CRASH_LIMIT`` consecutive
failures answers False until a write lands. While the boot still owes its late decision on saved pause
envelopes (their index was unreadable), the tick answers None and a barrier False: no publication replaces
the saved queue before that decision. A process with no live tick publisher (startup, shutdown, tests)
publishes synchronously from every mark. A process-ending door (``pre_restart_exit``; the shutdown kill as
``server_shutdown``) publishes embedded rows without the shared table, so an older body restores after a
rollback; an export in another form never counts as that door's publication.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.project_admission import carried_basis
from ouroboros.queue_observation import POOL_COUNT_KEYS as _POOL_COUNT_KEYS, QUEUE_ALIVE_FRESH_SEC, SHARED_KEY
from ouroboros.shared_values import TOKEN_PATTERN, token
from ouroboros.subagents import subagent_intent_fields
from ouroboros.utils import machine_json, read_json_dict, utc_now_iso, write_text_atomic
from supervisor.queue_snapshot import QUEUE_SNAPSHOT_SCHEMA_VERSION, _PENDING_FIELDS, _keeps_saved_queue, retry_pausing_envelopes
from supervisor.work_stop_migration import FORMAT_KEY, FORMAT_VERSION


def _queue():
    """The parent module, read at call time (the D18/D33 handle: ``init`` rebinds its names)."""
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)

#: ``queue.alive`` is refreshed inside the readers' freshness bound (``task_status``), without fsync.
QUEUE_ALIVE_REFRESH_SEC = QUEUE_ALIVE_FRESH_SEC / 4
#: A task value at least this long (a string's characters, a container's JSON text) may be shared by its address.
SHARED_MIN_LENGTH = 1024
#: ``shared_values.TOKEN_PATTERN`` as text: a reference as ``machine_json`` spells it in a skeleton's text.
_TOKEN_TEXT = re.compile(TOKEN_PATTERN.pattern.decode("ascii"))
#: Publications that always open a new generation, whatever the bytes.
_FORCED_REASONS = frozenset({"startup", "pre_restart_exit", "server_shutdown"})
#: The process-ending doors: their publication embeds every row whole, without the shared table,
#: so an older body restores it after a rollback (the restart exit; the shutdown kill of every door).
_EXPORT_REASONS = frozenset({"pre_restart_exit", "server_shutdown"})
_ALIVE_NAME = "queue.alive"
_SNAPSHOT_NAME = "queue_snapshot.json"
_DISPATCH_NAME = "queue_dispatch.json"
#: The RUNNING-row facts a publication carries besides the task. The capture writes exactly these and
#: the tick's census watches exactly these, so a fact a worker report changes under Q without a mark
#: (a silence episode, the quota clock, an owner wait) is published by the next tick, and a fact added
#: here later is watched by construction (RV_MAIN M6).
_ROW_FACTS = ("attempt", "worker_id", "owner_wait", "started_at", "model_wait_quota_clock",
              "sleep_parked_at", "budget_paused_sec", "soft_sent")
#: Task facts a worker reports without a queue transition; every other task fact changes with a mark.
_REPORTED_TASK_FACTS = ("model_waits", "focus")

# Captures take sequence numbers under Q. published_seq includes failed barriers: an older capture
# must never resurrect their permissions. published and census describe successful writes only.
# Failed writes require a complete rewrite. The tick accounts for each failure once.
_STATE: Dict[str, Any] = {"revision": 0, "published": 0, "reason": "", "kept": None, "census": None,
                          "capture_seq": 0, "published_seq": 0, "write_failed": False,
                          "failed_writes": 0, "failures_answered": 0, "failed_streak": 0,
                          "alive_heartbeats": None, "alive_pool": None, "alive_written_at": 0.0, "waits": {}}
#: Consecutive failed writes after which the tick answers False on every tick until a write lands: the supervisor
#: loop's own ending rule (``server._run_supervisor`` dies after three consecutive crashes), so a disk that fails
#: every write slowly ends the loop as one that fails fast does, instead of resetting the loop's count on each tick
#: a retry spends on the disk.
LOOP_CRASH_LIMIT = 3
_PUBLISHER: Optional[threading.Thread] = None
_PUBLISH_LOCK = threading.Lock()


class _TickWriter:
    """The tick's newest capture or refresh. Drain waits for offered work; reset rejects the old writer."""

    def __init__(self) -> None:
        self.thread: Optional[threading.Thread] = None
        self.wake = threading.Condition()
        self.capture: Optional[Dict[str, Any]] = None  # the newest capture not taken yet
        self.refresh: Optional[tuple] = None  # (pool, measured_after) of the newest quiet tick, to certify
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

    def ask_refresh(self, pool: Dict[str, Any], measured_after: int) -> None:
        with self.wake:
            self.refresh = (pool, measured_after)
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
                while not (self.closed or self.capture is not None or self.refresh is not None or self.syncs):
                    self.wake.wait()
                capture, refresh, syncs = self.capture, self.refresh, self.syncs
                self.capture, self.refresh, self.syncs = None, None, []
                if capture is None and refresh is None and self.closed:  # stopped: what it still held was written first
                    for done in syncs:
                        done.set()
                    return
            try:
                if capture is not None:
                    _publish(capture, writer=self)  # a publication certifies too: the refresh is subsumed
                elif refresh is not None:
                    _refresh_alive(*refresh)
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
                  alive_pool=None, alive_written_at=0.0, waits={})
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


def mark_dirty_locked(reason: str) -> None:
    """Record a memory transition while Q is held, including at a synchronous publication door."""
    _STATE["revision"] += 1
    _STATE["reason"] = reason


def persist_queue_snapshot(reason: str = "") -> Optional[bool]:
    """Mark a transition. A live tick returns None and publishes later. Without it, publish now.
    True means durable, False means the write failed, None also covers a queue outside our custody.
    Callers needing durability use publish_queue_snapshot."""
    if not _queue().INITIALIZED:  # a boot without a provider: state initialization reads a snapshot as history (#2412)
        log.info("Queue snapshot not published (reason=%s): no supervisor queue in this process", reason)
        return None
    with _queue()._queue_lock:
        mark_dirty_locked(reason)
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
    """Return True after publication, False on failure, None when the saved queue must be kept.
    Without a tick, the queue's synchronous mark implements this barrier."""
    if not tick_publisher_alive():
        return _queue().persist_queue_snapshot(reason=reason)
    return _publish_now(reason)


def _publish_now(reason: str) -> Optional[bool]:
    if not _queue().INITIALIZED:
        log.info("Queue snapshot not published (reason=%s): no supervisor queue in this process", reason)
        return None
    if not retry_pausing_envelopes(_queue()):
        return False
    while True:
        with _queue()._queue_lock:
            _STATE["revision"] += 1
            capture = _capture(reason)
        if capture is None:
            return None
        result = _publish(capture)
        if result or not capture["forced"] or _STATE["published_seq"] <= capture["seq"]:
            return result


def publish_queue_changes() -> Optional[bool]:
    """Publish a new revision or census, otherwise refresh liveness. A kept queue returns None.
    The async tick reports each completed write failure once. After LOOP_CRASH_LIMIT consecutive
    failures every tick returns False until a write succeeds, even while a retry is on disk."""
    if not _queue().INITIALIZED:
        return None
    if not retry_pausing_envelopes(_queue()):
        return None
    with _queue()._queue_lock:
        if _kept_by_memo():
            return None  # the saved queue is still kept: decided once, not copied or counted again
        waits = note_lane_waits()
        # One pool reading for capture and refresh. Worker-reported facts can move without a revision.
        pool, measured_after = _pool_counts(), _STATE["capture_seq"]
        census = _census(pool)
        moved = (_STATE["revision"] != _STATE["published"] or census != _STATE["census"] or _STATE["write_failed"])
        capture = _capture(_STATE["reason"] or "main_loop", census=census, pool=pool, waits=waits) if moved else None
    if not _WRITER.alive():
        # Written where the tick stands: the outcome answers for this tick's own write, so every failure so
        # far is answered with it.
        outcome = (_refresh_alive(pool, measured_after) if not moved
                   else None if capture is None else _publish(capture))
        _STATE["failures_answered"] = _STATE["failed_writes"]
        return outcome
    if capture is not None:
        _WRITER.offer(capture)
    elif not moved and _queue().QUEUE_PUBLICATION is not None:
        _WRITER.ask_refresh(pool, measured_after)
    else:
        return None  # kept, or nothing of this process certified yet
    failed, answered = _STATE["failed_writes"], _STATE["failures_answered"]
    _STATE["failures_answered"] = failed
    return failed == answered and _STATE["failed_streak"] < LOOP_CRASH_LIMIT


def note_lane_waits() -> Dict[str, Dict[str, Any]]:
    """Under Q: who keeps each queued row's lane now (``dispatch_choice.lane_waits``), for the tick and the status
    census alike. Its holders change without a queue transition (a claim, a moved row's folder read, the chooser's
    own refresh), so when the waits differ from the ones last captured or read the revision moves and both
    publications follow; a pending transition keeps its reason (a forced or export capture stays one),
    ``lane_wait`` names it only when nothing else is pending."""
    from supervisor.dispatch_choice import lane_waits

    waits = lane_waits()
    if waits != _STATE["waits"]:
        if _STATE["revision"] == _STATE["published"]:
            _STATE["reason"] = "lane_wait"
        _STATE["revision"] += 1
        _STATE["waits"] = waits
    return waits


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


def _census(pool: Optional[Dict[str, Any]] = None) -> str:
    """Under Q: one small identity of what the rows publish without a transition, and of the pool facts a reader
    decides on from the document (``_pool_census`` of the tick's measurement, else of one taken here; the counts
    ride the certificate)."""
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
    return machine_json([rows, _pool_census(_pool_counts() if pool is None else pool)])


def _custody_key() -> tuple:
    """Under Q: what the kept decision depends on (the revision, and whether a restore owns the saved queue)."""
    from supervisor.queue_snapshot import _RESTORED, _RESTORING

    key = str(_queue().QUEUE_SNAPSHOT_PATH.absolute())
    return (_STATE["revision"], key in _RESTORED, _RESTORING.get(key))


def _kept_by_memo() -> bool:
    """Under Q: the saved queue was decided kept at this same custody, so nothing is copied or counted."""
    return _STATE["kept"] is not None and _STATE["kept"] == _custody_key()


def _capture(reason: str, *, census: Optional[str] = None, pool: Optional[Dict[str, Any]] = None,
             waits: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Under Q: the thin rows and queue facts of this revision; None when the saved queue is kept. ``pool``,
    ``census`` and ``waits`` are the tick's (one measurement); a barrier's capture measures them here, once, and
    records the waits it captured so the next tick does not mark them again."""
    if _kept_by_memo():
        return None
    pool = _pool_counts() if pool is None else pool
    census = _census(pool) if census is None else census
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
    from supervisor.dispatch_choice import lane_waits

    waits = lane_waits() if waits is None else waits  # beside the row (who keeps its lane now), never in its task
    _STATE["waits"] = waits
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
            **({"lane_wait": waits[str(t.get("id") or "")]} if str(t.get("id") or "") in waits else {}),
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
        "pool": pool,
        "restart_intent": agent_restart_snapshot(), "restart_reply": dict(_AGENT_RESTART_REPLY),
        "heartbeats": {tid: meta.get("last_heartbeat_at") for tid, meta in running_items},
    }


def _pool_counts() -> Dict[str, Any]:
    """The pool's published facts, measured by ``worker_pool_lifecycle.pool_slot_counts`` (its docstring names
    each count) from the ACTUAL pool, plus the last reading of the machine's memory (``memory_admission``:
    ``pool_admission.facts``, dated by its own ``read_at``). A slot mid-reap is not
    assignable, so no reader is told of a free slot that is not. A failed measurement publishes
    ``pool_observation_status: "unknown"`` with null counts: never zeros, which would read as an empty pool."""
    try:
        from supervisor import pool_admission, workers as _workers_mod
        from supervisor.worker_pool_lifecycle import pool_slot_counts

        counts = pool_slot_counts(_workers_mod.WORKERS)
        disabled = _workers_mod._worker_pool_execution_state()["disabled_reason"]
        # Absence is counted above, not an explicit disablement.
        counts["worker_pool_disabled_reason"] = "" if disabled == "no_workers" else disabled
        counts["pool_observation_status"] = "observed"
        counts["memory_admission"] = pool_admission.facts()
        return counts
    except Exception:
        log.debug("Worker pool facts unavailable; published as unknown", exc_info=True)  # every tick: no warning flood
        return {**dict.fromkeys(_POOL_COUNT_KEYS), "worker_pool_disabled_reason": "unknown",
                "pool_observation_status": "unknown", "pool_observation_error": "introspection_failed"}


def _pool_census(counts: Dict[str, Any]) -> Dict[str, Any]:
    """The pool facts a reader decides on from the document itself: the disablement and its reason, whether any
    process is started (``model_detach.pool_available``), and whether starts are admitted and why (the verdict, not
    the reading's bytes); a failed measurement is its own shape. The counts change with every boot, retirement and
    reap while no row changes; they reach readers on the certificate (``_certify``,
    ``queue_observation.load_queue_document``), so such a change rewrites ``queue.alive`` and not the document,
    which keeps the counts of its own write."""
    if counts.get("pool_observation_status") != "observed":
        return counts
    memory = counts.get("memory_admission") or {}
    return {"worker_pool_disabled_reason": counts["worker_pool_disabled_reason"],
            "started": counts["worker_total"] + counts["lane_carrier_total"] > 0,
            "memory_admission": (bool(memory.get("starts_admitted")), memory.get("reason", ""))}


def _certified_pool(counts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The counts ``queue.alive`` carries (``_POOL_COUNT_KEYS``, the reader's ``POOL_COUNT_KEYS``); None for a
    failed measurement, whose readers keep the document's own counts."""
    if counts.get("pool_observation_status") != "observed":
        return None
    return {key: counts.get(key) for key in _POOL_COUNT_KEYS}


def _documents(capture: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Outside Q: the two documents of this capture and their texts; task values interned unless rollback-readable."""
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
                                ("_budget_pause_resume", ("pause_id", "grant_id", "authority", "root_fence_id", "selected_by", "selection_door", "root_grant_id", "root_resume_generation"))):
                if task.get(key) is not None:
                    value = task[key]
                    member[key] = {k: value[k] for k in fields if k in value} if isinstance(value, dict) else value
            members[str(task.get("id") or "")] = member
    dispatch = {"budget_root_fences": capture["budget_root_fences"], "members": members}
    snapshot = machine_json(payload) if capture["rollback_readable"] else _share_replicated(payload)
    return ({_SNAPSHOT_NAME: payload, "queue_dispatch.json": dispatch},
            {_SNAPSHOT_NAME: snapshot, "queue_dispatch.json": machine_json(dispatch)})


def _share_replicated(payload: Dict[str, Any]) -> str:
    """Replace every large task value that appears more than once (top level and one level inside
    ``metadata``) by its address, set the table of those values, and return the document's text. A
    value published once stays where it is: only replication is waste, and a raw reader still sees
    singletons inline. The rows are this publication's own frozen capture (``_frozen``: no container reached
    twice), so they are rewritten in place.

    Each place's value is encoded once: that text says whether it is large and whether it repeats (one
    text is kept per distinct value), its address is the sha256 of that text, and the document's text is
    the skeleton's (a placeholder for each value published once and for each table entry; a repeated
    value's places hold its token) with the texts spliced in, so neither a row nor the table is encoded
    again (``machine_json`` is compositional, as ``shared_values`` relies on). A splice that does not
    place every placeholder exactly once falls back to encoding the document whole.
    """
    kept: Dict[str, str] = {}  # one text per distinct value: a repeat's own copy is dropped at once
    counts: Dict[str, int] = {}  # text -> its places
    places = []  # (container, key, value, text) of every large value, in document order

    def count(container: Dict[str, Any], key: str, value: Any) -> None:
        if (text := _large_text(value)) is not None:
            text = kept.setdefault(text, text)
            counts[text] = counts.get(text, 0) + 1
            places.append((container, key, value, text))

    for lane in ("pending", "running"):
        for row in payload[lane]:
            task = row["task"]
            for key, value in task.items():
                if key == "metadata" and isinstance(value, dict):
                    for name, item in value.items():
                        count(value, name, item)
                else:
                    count(task, key, value)
    prefix, holes = uuid.uuid4().hex, {}  # placeholder address (this publication's own) -> the text it stands for

    def hole(text: str) -> Dict[str, str]:
        address = f"{prefix}{len(holes):032x}"
        holes[address] = text
        return token(address)

    addresses: Dict[str, str] = {}  # text -> its address
    shared: Dict[str, Any] = {}  # address -> the first value with that text
    table: Dict[str, Any] = {}  # address -> the placeholder of that text
    for container, key, value, text in places:
        if counts[text] == 1:
            container[key] = hole(text)
            continue
        if text not in addresses:
            digest = addresses[text] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            shared[digest], table[digest] = value, hole(text)
        container[key] = token(addresses[text])
    payload[SHARED_KEY] = table
    skeleton = machine_json(payload)
    payload[SHARED_KEY] = shared
    for container, key, value, text in places:
        if counts[text] == 1:
            container[key] = value
    pieces, position, placed = [], 0, set()
    for match in _TOKEN_TEXT.finditer(skeleton):
        text = holes.get(match.group(1))
        if text is not None:
            if match.group(1) in placed:  # a placeholder seen twice: stray data or a shared container
                return machine_json(payload)
            pieces += (skeleton[position:match.start()], text)
            position = match.end()
            placed.add(match.group(1))
    pieces.append(skeleton[position:])
    return "".join(pieces) if len(placed) == len(holes) else machine_json(payload)


def _large_text(value: Any) -> Optional[str]:
    """The text of a value large enough to share (a string of at least ``SHARED_MIN_LENGTH`` characters, a
    non-empty dict or list whose text is at least that long), else None."""
    if isinstance(value, str):
        return machine_json(value) if len(value) >= SHARED_MIN_LENGTH else None
    if isinstance(value, (dict, list)) and value:
        text = machine_json(value)
        return text if len(text) >= SHARED_MIN_LENGTH else None
    return None


def _stamped(document: Dict[str, Any], text: str, generation: str, reason: str) -> str:
    """The written text: ``document`` (whose text is ``text``) with its generation, time and reason after its
    own keys, appended to that text instead of encoding the document again."""
    stamp = {"generation": generation, "ts": utc_now_iso(), "reason": reason}
    if not document or stamp.keys() & document.keys():
        return machine_json({**document, **stamp})
    return f"{text[:-1]},{machine_json(stamp)[1:]}"


def _dispatch_restrictions(doc):
    """Monetary latches and the members selected under each. An absent latch admits the tree."""
    from ouroboros.budget_pause import budget_fence_selected

    members = doc["members"]
    return {row["root_task_id"]: (row, {tid: member for tid, member in members.items()
                                        if budget_fence_selected(member, row)})
            for row in doc["budget_root_fences"] if row.get("cause") != "owner_pause"
            and row.get("status") in {"active", "paused"}}


def _before_snapshot(old, new):
    """Only write early if permission actually shrinks. During a mixed change both sets of limits hold."""
    from ouroboros.queue_observation import valid_dispatch_facts
    if not old or not isinstance(old.get("generation"), str) or not old["generation"].strip() or not valid_dispatch_facts(old):
        return None
    try:
        before, after = _dispatch_restrictions(old), _dispatch_restrictions(new)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None  # an unreadable old permission is only replaced after durability
    if not any(root not in before or before[root][1].keys() - selected.keys()
               for root, (_, selected) in after.items()):
        return None
    if not any(root not in after or after[root][1].keys() - selected.keys()
               for root, (_, selected) in before.items()):
        return new
    fences, members = [], {}
    for root, (fence, selected) in (before | after).items():
        fences.append(fence)
        allowed = selected if root not in before else before[root][1]
        members.update((tid, member) for tid, member in selected.items() if tid in allowed)
    return {"budget_root_fences": fences, "members": members}


def _publish(capture: Dict[str, Any], *, writer: Optional[_TickWriter] = None) -> bool:
    """Outside Q, one writer at a time: encode, compare, then revoke -> changed documents -> certify.

    ``writer``: the tick's thread names itself, so a capture of a generation that ended while it was encoding
    (``reset`` replaced the writer) is never written over the next generation's documents.
    """
    documents, encoded = _documents(capture)
    with _PUBLISH_LOCK:
        if _queue().PRIOR_DIRECT_ROOTS.get("retry_pausing_envelopes"):
            return False
        if writer is not None and writer is not _WRITER:
            return False  # an ended generation's capture: the next generation owns the documents now
        previous = _queue().QUEUE_PUBLICATION
        if _STATE["published_seq"] > capture["seq"]:
            # A later capture already published newer documents; while a later write failed, the bundle on
            # disk is unknown, so that publication is no durability for this caller either.
            return not (_STATE["write_failed"] or capture["forced"])
        # After a failed write the bundle on disk is unknown: byte equality with the last cache proves nothing,
        # so every document is rewritten under a new generation and certified (R2/R3-A).
        if (previous is None or previous[0] != _queue().QUEUE_SNAPSHOT_PATH or capture["forced"]
                or _STATE["write_failed"]):
            stale = list(documents)
        else:
            stale = [name for name in documents if previous[1].get(name) != encoded[name]]
        changed = bool(stale)
        generation = uuid.uuid4().hex if changed else previous[2]
        snapshot_generation = generation if _SNAPSHOT_NAME in stale else previous[2]
        state = _queue().QUEUE_SNAPSHOT_PATH.parent

        def write(name, document, content):
            stamp = generation if document is documents[name] else uuid.uuid4().hex
            stamped = _stamped(document, content, stamp, capture["reason"])
            if _queue().atomic_write_text(state / name, stamped) is False:
                raise OSError(f"queue publication failed: {name}")

        try:
            early = None
            dispatch = documents[_DISPATCH_NAME]
            if changed:
                _revoke_alive(state)
                if _DISPATCH_NAME in stale:
                    early = _before_snapshot(read_json_dict(state / _DISPATCH_NAME), dispatch)
                    if early is not None:
                        write(_DISPATCH_NAME, early, machine_json(early))
                if _SNAPSHOT_NAME in stale:
                    write(_SNAPSHOT_NAME, documents[_SNAPSHOT_NAME], encoded[_SNAPSHOT_NAME])
            pool = _certified_pool(capture["pool"])
            if changed or _alive_due(capture["heartbeats"], pool):
                _certify(state, snapshot_generation, capture["heartbeats"], pool)
            if _DISPATCH_NAME in stale and early is not dispatch:
                write(_DISPATCH_NAME, dispatch, encoded[_DISPATCH_NAME])
            if changed:
                _queue().QUEUE_PUBLICATION = (_queue().QUEUE_SNAPSHOT_PATH, encoded, snapshot_generation)
            _STATE["published"] = max(_STATE["published"], capture["revision"])
            _STATE["published_seq"] = max(_STATE["published_seq"], capture["seq"])
            _STATE["census"] = capture["census"]  # written, so the tick's next census compares against it
            _STATE["write_failed"] = False
            _STATE["failed_streak"] = 0  # a landed write ends the run of failures
            return True
        except Exception:
            # The census stays the last WRITTEN one: the next tick captures and retries this change; and
            # the stale document on disk is not certified again until a publication lands (N9).
            # A queued older capture may carry permission this failed barrier has just refused.
            _STATE["published_seq"] = capture["seq"]
            _STATE["write_failed"] = True
            _STATE["failed_writes"] += 1  # one completed failed write: the tick answers it once
            _STATE["failed_streak"] += 1  # ...and at the loop's limit, on every tick until a write lands
            log.warning("Failed to persist queue snapshot (reason=%s)", capture["reason"], exc_info=True)
            return False


def _refresh_alive(pool: Dict[str, Any], measured_after: int) -> Optional[bool]:
    """A tick with nothing to publish: keep the certificate fresh for its readers (the heartbeats, and the counts of
    the pool that tick measured after capture ``measured_after``), never a new generation. A capture taken after that
    measurement, once written, certified newer counts of its own: the refresh leaves them."""
    published = _queue().QUEUE_PUBLICATION
    if published is None:
        return None  # nothing of this process is certified; a kept saved queue stays uncertified
    if _STATE["write_failed"]:
        return False  # the document on disk is older than the facts: no certificate until a write lands
    with _queue()._queue_lock:
        heartbeats = {tid: (meta.get("last_heartbeat_at") if isinstance(meta, dict) else None)
                      for tid, meta in _queue().RUNNING.items()}
    pool = _certified_pool(pool)
    if not _alive_due(heartbeats, pool):
        return True
    with _PUBLISH_LOCK:
        if _STATE["write_failed"]:
            return False  # a barrier failed while this refresh was in flight: its revoked certificate stays revoked (R3-B)
        if _queue().QUEUE_PUBLICATION is not published or _STATE["published_seq"] > measured_after:
            return True  # a publication landed meanwhile and certified itself, or a later capture's counts stand
        try:
            _certify(_queue().QUEUE_SNAPSHOT_PATH.parent, published[2], heartbeats, pool)
        except Exception:
            log.warning("Failed to refresh queue liveness", exc_info=True)
            return False
    return True


def _alive_due(heartbeats: Dict[str, Any], pool: Optional[Dict[str, Any]]) -> bool:
    return (heartbeats != _STATE["alive_heartbeats"] or pool != _STATE["alive_pool"]
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


def _certify(state, generation: str, heartbeats: Dict[str, Any], pool: Optional[Dict[str, Any]]) -> None:
    """Only the snapshot's pulse. A crash can lose freshness."""
    alive = {"generation": generation, "ts": utc_now_iso(), "heartbeats": heartbeats,
             **({"pool": pool} if pool is not None else {})}
    if _write_alive(state / _ALIVE_NAME, json.dumps(alive, separators=(",", ":"))) is False:
        raise OSError("queue liveness publication failed")
    _STATE["alive_heartbeats"] = dict(heartbeats)
    _STATE["alive_pool"] = pool
    _STATE["alive_written_at"] = time.monotonic()


def _write_alive(path, content: str) -> None:
    write_text_atomic(path, content, fsync=False)
