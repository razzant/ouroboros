"""Queue publication by revision: transitions under Q, documents outside it.

Every queue transition marks the publication dirty under Q (``persist_queue_snapshot``: one
revision bump, no bytes). ONE publisher, the supervisor tick (``publish_queue_changes``),
captures the rows under Q when the revision moved and serializes and writes them outside
Q. The capture copies containers only (``_frozen``): strings are shared, so the owner's
text is never copied under the lock. A large task value that appears in more than one place (the
owner's message in twenty rows, the task contract twice per row) is published once, by
content address (``$shared``), so the document carries the owner's words once; ``queue_observation.load_queue_document``
expands them in memory, so every reader still sees the whole row.

The publication order a reader relies on is kept: the old ``queue.alive`` certificate is
revoked, the documents land (fsync), the new certificate names them (no fsync).
``queue.alive`` is rewritten only when its generation or heartbeat map changed or its stamp
aged past ``QUEUE_ALIVE_REFRESH_SEC``; a process that never published writes none (the
uncustodied saved queue rule, ``queue_snapshot._keeps_saved_queue``).

Durability before an effect is an explicit barrier (``publish_queue_snapshot``), used only
where a caller acts on the publication. A process with no live tick publisher (startup
before the loop, shutdown after it, tests) publishes synchronously from every mark, so the
boot and exit doors keep their semantics without a second protocol. A process-ending door
(``pre_restart_exit``; the shutdown kill as ``server_shutdown``) publishes embedded rows
without the shared table, so an older body restores after a rollback.
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
_STATE: Dict[str, Any] = {"revision": 0, "published": 0, "reason": "", "kept": None, "census": None,
                          "capture_seq": 0, "published_seq": 0, "write_failed": False,
                          "alive_heartbeats": None, "alive_written_at": 0.0}
_PUBLISHER: Optional[threading.Thread] = None
_PUBLISH_LOCK = threading.Lock()


def reset() -> None:
    """A new queue generation (``queue.init``): nothing marked, nothing published, no tick publisher."""
    global _PUBLISHER
    _STATE.update(revision=0, published=0, reason="", kept=None, census=None, capture_seq=0, published_seq=0,
                  write_failed=False, alive_heartbeats=None, alive_written_at=0.0)
    _PUBLISHER = None


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
    from supervisor.direct_roots import collect_direct_roots

    direct = collect_direct_roots()  # Never wait for an actor while holding Q.
    with _queue()._queue_lock:
        _STATE["revision"] += 1
        capture = _capture(reason, direct)
    if capture is None:
        return None
    return _publish(capture)


def publish_queue_changes() -> Optional[bool]:
    """The tick: publish when the revision or the direct roster moved, else refresh the certificate.

    True when published or refreshed, None when the documents on disk are kept (an
    uncustodied saved queue, or nothing of this process certified yet), False when a write
    failed: the loop treats False as its own crash (``server._run_supervisor``), None as a
    quiet tick.
    """
    if not _queue().INITIALIZED:
        return None
    from supervisor.direct_roots import collect_direct_roots

    direct = collect_direct_roots()
    published = _queue().QUEUE_PUBLICATION
    direct_moved = published is not None and published[1].get("direct_roots.json") != machine_json(direct)
    with _queue()._queue_lock:
        # The census: what the rows publish without a transition (worker reports mutate RUNNING in place
        # with no mark). One small identity per tick, so a changed fact is published and an unchanged
        # live set still captures, serializes and writes nothing. A failed write is itself a reason to
        # publish: the bundle on disk is unknown until a complete rewrite lands (R2).
        census = _census()
        moved = (_STATE["revision"] != _STATE["published"] or direct_moved or census != _STATE["census"]
                 or _STATE["write_failed"])
        capture = _capture(_STATE["reason"] or "main_loop", direct, census=census) if moved else None
    if not moved:
        return _refresh_alive()
    return None if capture is None else _publish(capture)


def _frozen(value: Any) -> Any:
    """A copy of the containers; strings, numbers and None are shared (immutable)."""
    if isinstance(value, dict):
        return {key: _frozen(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_frozen(item) for item in value]
    return value


def _row_facts(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The published facts of one RUNNING row besides its task, as the capture writes them.

    Only a dated silence episode is a diagnostic's source authority (#1376): its fact, and while it
    stays open the progress that revokes it before the recovery lane writes the result. Ordinary
    progress rotates nothing.
    """
    facts = {key: meta.get(key) for key in _ROW_FACTS}
    facts.update(started_at=float(meta.get("started_at") or 0.0),
                 budget_paused_sec=float(meta.get("budget_paused_sec") or 0.0), soft_sent=bool(meta.get("soft_sent")))
    silence = meta.get("silence_attention")
    if silence:
        facts["silence_attention"] = silence
        if isinstance(silence, dict) and silence.get("state") == "pending_attention":
            facts["last_progress_at"] = meta.get("last_progress_at")
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
    return machine_json([rows, _pool_counts()])


def _capture(reason: str, direct: Dict[str, Any], *, census: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Under Q: the thin rows and queue facts of this revision; None when the saved queue is kept."""
    census = _census() if census is None else census
    _STATE["capture_seq"] += 1
    pending_items = list(_queue().PENDING)
    running_items = [(task_id, meta if isinstance(meta, dict) else {}) for task_id, meta in _queue().RUNNING.items()]
    acceptance_fences = [_frozen(row) for row in _queue().ACCEPTANCE_FENCES.values()]
    budget_root_fences = [_frozen(row) for row in _queue().BUDGET_ROOT_FENCES.values()]
    key = str(_queue().QUEUE_SNAPSHOT_PATH.absolute())
    from supervisor.queue_snapshot import _RESTORED, _RESTORING

    custody = (_STATE["revision"], key in _RESTORED, _RESTORING.get(key))
    if _STATE["kept"] == custody or _keeps_saved_queue(_queue().QUEUE_SNAPSHOT_PATH, pending_items, running_items,
                                                       acceptance_fences, budget_root_fences, (direct or {}).get("roots")):
        if _STATE["kept"] != custody:
            log.info("Queue snapshot kept (reason=%s): this process has not taken the saved queue into custody", reason)
        _STATE["kept"] = custody
        return None
    pending_rows = []
    for t in pending_items:
        pending_rows.append({
            "id": t.get("id"), "type": t.get("type"), "priority": t.get("priority"),
            "attempt": t.get("_attempt"), "queued_at": t.get("queued_at"),
            "queue_seq": t.get("_queue_seq"),
            "task": _frozen({**{key: t.get(key) for key in _PENDING_FIELDS}, **subagent_intent_fields(t), **carried_basis(t),
                             **{key: t[key] for key in ('admitted_dispatch', '_project_scope_none', '_working_recovery',
                                                        # #939: the execution budget and its carried clock offset travel with the row.
                                                        'execution_timeout_sec', '_execution_consumed_sec', '_execution_handoff_sec') if key in t}}),
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
        "pool": _pool_counts(), "direct": direct,
        "restart_intent": agent_restart_snapshot(), "restart_reply": dict(_AGENT_RESTART_REPLY),
        "heartbeats": {tid: meta.get("last_heartbeat_at") for tid, meta in running_items},
    }


def _pool_counts() -> Dict[str, Any]:
    """Honest worker-pool counts from the ACTUAL pool (not the configured max): the live pool can be
    smaller (a crash-storm/direct-chat fallback clears WORKERS) and a slot mid-reap is popped from
    RUNNING but NOT assignable, so the context queue digest never falsely advertises a free slot."""
    try:
        from supervisor import workers as _workers_mod
        from supervisor.worker_pool_lifecycle import is_project_slot, worker_ready

        def idle(slot: Any) -> bool:
            return (getattr(slot, "busy_task_id", None) is None and worker_ready(slot)
                    and getattr(slot, "active_capacity", True) and not getattr(slot, "resume_hold", ""))

        # Project slots alone: they are what MAX_WORKERS bounds and the queue assigns to. The
        # conversation lane's carriers are outside that count (owner decision 4A) and reported apart (W9).
        _ws = [w for w in _workers_mod.WORKERS.values() if is_project_slot(w)]
        lane = [w for w in _workers_mod.WORKERS.values() if not is_project_slot(w)]
        active = sum(1 for w in _ws if getattr(w, "active_capacity", True))
        disabled = _workers_mod._worker_pool_execution_state()["disabled_reason"]
        return {
            "reaping_count": sum(1 for _w in _ws if getattr(_w, "reaping", False)),
            "worker_total": len(_ws), "active_worker_count": active, "parked_worker_count": len(_ws) - active,
            # Absence is counted above, not an explicit disablement.
            "worker_pool_disabled_reason": "" if disabled == "no_workers" else disabled,
            "assignable_idle_workers": sum(1 for _w in _ws if idle(_w)),
            "lane_carrier_total": len(lane), "lane_carrier_idle": sum(1 for _w in lane if idle(_w)),
        }
    except Exception:
        return {"reaping_count": 0, "worker_total": 0, "active_worker_count": 0, "parked_worker_count": 0,
                "worker_pool_disabled_reason": "unknown", "assignable_idle_workers": 0,
                "lane_carrier_total": 0, "lane_carrier_idle": 0}


def _documents(capture: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Outside Q: the three documents of this capture; task values interned unless rollback-readable."""
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
    return {"queue_snapshot.json": payload,
            "queue_dispatch.json": {"budget_root_fences": capture["budget_root_fences"], "members": members},
            "direct_roots.json": capture["direct"]}


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
            container[key] = {SHARED_KEY: digest}
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


def _publish(capture: Dict[str, Any]) -> bool:
    """Outside Q, one writer at a time: encode, compare, then revoke -> documents -> certify."""
    documents = _documents(capture)
    encoded = {name: machine_json(document) for name, document in documents.items()}
    with _PUBLISH_LOCK:
        previous = _queue().QUEUE_PUBLICATION
        if previous is not None and _STATE["published_seq"] > capture["seq"] and not capture["forced"]:
            return True  # a later capture of this process already published newer documents
        # After a failed write the bundle on disk is unknown: byte equality with the last cache proves nothing,
        # so every document is rewritten under a new generation and certified (R2/R3-A).
        changed = (previous is None or previous[:2] != (_queue().QUEUE_SNAPSHOT_PATH, encoded) or capture["forced"]
                   or _STATE["write_failed"])
        generation = uuid.uuid4().hex if changed else previous[2]
        state = _queue().QUEUE_SNAPSHOT_PATH.parent
        try:
            if changed:
                _revoke_alive(state)  # A partial publish must not certify the old no-fence document.
                for name, document in documents.items():
                    published = {**document, "generation": generation, "ts": utc_now_iso(), "reason": capture["reason"]}
                    if _queue().atomic_write_text(state / name, machine_json(published)) is False:
                        raise OSError(f"queue publication failed: {name}")
            if changed or _alive_due(capture["heartbeats"]):
                _certify(state, generation, capture["heartbeats"])
            if changed:
                _queue().QUEUE_PUBLICATION = (_queue().QUEUE_SNAPSHOT_PATH, encoded, generation)
            _STATE["published"] = max(_STATE["published"], capture["revision"])
            _STATE["published_seq"] = max(_STATE["published_seq"], capture["seq"])
            _STATE["census"] = capture["census"]  # written, so the tick's next census compares against it
            _STATE["write_failed"] = False
            return True
        except Exception:
            # The census stays the last WRITTEN one: the next tick captures and retries this change; and
            # the stale document on disk is not certified again until a publication lands (N9).
            _STATE["write_failed"] = True
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
            _certify(_queue().QUEUE_SNAPSHOT_PATH.parent, published[2], heartbeats)
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


def _certify(state, generation: str, heartbeats: Dict[str, Any]) -> None:
    """``queue.alive`` names the generation on disk; written without fsync (a crash loses only freshness)."""
    alive = {"generation": generation, "ts": utc_now_iso(), "heartbeats": heartbeats}
    if _write_alive(state / _ALIVE_NAME, json.dumps(alive, separators=(",", ":"))) is False:
        raise OSError("queue liveness publication failed")
    _STATE["alive_heartbeats"] = dict(heartbeats)
    _STATE["alive_written_at"] = time.monotonic()


def _write_alive(path, content: str) -> None:
    write_text_atomic(path, content, fsync=False)
