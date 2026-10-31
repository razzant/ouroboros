"""The durable queue snapshot: what a restart finds and what it may restore.

The snapshot is written under the queue lock from the live PENDING/RUNNING rows and
the acceptance fences beside them, and restored only while PENDING is empty.
Accepted work survives at any age; only a fresh acknowledged Restart transaction
returns it to ordinary admission. Other application stops require explicit Resume.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import math
import pathlib
import threading
import time
import uuid
from typing import Any, Optional

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.utils import utc_now_iso
from ouroboros.subagents import subagent_intent_fields
from supervisor.work_stop_migration import FORMAT_KEY, FORMAT_VERSION, normalize_for_restore
from supervisor.restart_retention import exact_pause_row as _exact_pause_row
from supervisor.task_admission import (
    restore_terminalization_retry,
    restore_terminalization_retry_rows,
    restored_handoff_unproven,
)
from supervisor.task_lifecycle import (
    _cancel_result_fields,
    restore_queue_fences,
)


def _queue():
    """The parent module, read at call time.

    The queue owns PENDING/RUNNING, the drive root, the liveness settings and
    the lock that guards them, and ``init``/``init_queue_refs`` REBIND those
    names. Reading them through the module is what keeps one binding: a
    from-import here would freeze the value this module saw at import time
    (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)

# Snapshot paths whose saved queue this process took into custody: a ``restore_pending_from_snapshot``
# that read it and decided every row. Until then, or until it published one itself, its maps prove
# nothing about the saved queue on disk (#1952), whichever door publishes. ``_RESTORING`` names the
# thread reading one right now.
_RESTORED: set = set()
_RESTORING: dict = {}


def _keeps_saved_queue(path: Any, *held: Any) -> bool:
    """Whether a publication from this process must leave the snapshot on disk as it is: another
    thread is restoring it now, or the process holds no work at all and neither restored nor
    published that snapshot, which still names pending or running rows. That is the previous
    generation's saved queue (any publication during the boot: a stop, Update or Rollback, a boot
    without a provider); an empty or partial write would erase saved Pauses it carries, and Resume
    would then find nothing to resume."""
    key, publication = str(pathlib.Path(path).absolute()), _queue().QUEUE_PUBLICATION
    if key in _RESTORED or (publication is not None and str(pathlib.Path(publication[0]).absolute()) == key):
        return False
    if _RESTORING.get(key, threading.get_ident()) != threading.get_ident():
        return True
    if any(held):
        return False
    try:
        saved = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False  # nothing readable to keep
    return isinstance(saved, dict) and bool(saved.get("pending") or saved.get("running"))

def saved_queue_unowned() -> bool:
    """Whether the saved queue snapshot still names work this process has not taken into custody (no
    finished restore of it, no publication of its own): work admitted now would be published over it."""
    return _keeps_saved_queue(_queue().QUEUE_SNAPSHOT_PATH)


# ABI 7.0 (Q8=B): the durable queue snapshot names its schema on every write.
# Stamp-on-write ONLY — the restore path does not require the stamp (no
# compat branching), so an N−1 snapshot restores unchanged.
QUEUE_SNAPSHOT_SCHEMA_VERSION = 1


def _kept_service_pids() -> "set[int]":
    """PIDs of deliberately-kept (session-scope) services to spare from a worker
    tree-kill on cancel/hard-timeout. Best-effort; never raises."""
    try:
        from ouroboros.process_custody import live_kept_service_pids
        return live_kept_service_pids(pathlib.Path(_queue().DRIVE_ROOT))
    except Exception:
        return set()


def _retained_daemon_pids() -> "set[int]":
    """PIDs of the installation's live daemon roots (``daemon``-scope custody rows,
    the shared Claudexor daemon among them) that EVERY worker tree-kill spares:
    their lifetime is the installation's, not the worker's. Best-effort; never raises."""
    try:
        from ouroboros.process_custody import live_daemon_root_pids
        from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
        return live_daemon_root_pids(
            pathlib.Path(_queue().DRIVE_ROOT), retained_purposes={CUSTODY_PURPOSE},
        )
    except Exception:
        return set()


# Preserve routing, model intent, holds and exact continuation sources on restart.
_PENDING_FIELDS = (
    'id', 'type', 'chat_id', 'text',
    'priority', 'depth', 'description', 'objective',
    'title', 'expected_output', 'constraints', 'role',
    'context', 'parent_task_id', 'root_task_id', 'session_id',
    'actor_id', 'delegation_role', 'workspace_root', 'workspace_mode',
    'project_id', 'placement_applied_at', '_project_admission_restore_hold', 'focus', 'allowed_resources',
    'deadline_at', 'task_contract', '_owner_hold', '_consciousness_continuation',
    'model_lane', 'effective_model_lane', 'model',
    'use_local_model', 'effective_executor', 'tool_profile', 'executor_route',
    'reasoning_effort', 'effort_level', 'effort_requested', 'effort_source',
    'capability_delta', 'task_group_id', 'task_group', 'subagent_envelope',
    'configured_subagent', 'memory_mode', 'drive_root', 'parent_cognitive_route',
    'subagent_availability', 'child_drive_root', 'budget_drive_root', 'task_constraint',
    'predecessor_authority_source', 'metadata', 'origin_message_ref', 'origin_message_text',
    'image_source_ref', 'image_mime', 'image_caption',
    'attachment_manifest', 'attachment_images', 'attachments',
    '_attempt', 'review_reason', 'review_source_task_id', '_budget_pause',
    'budget_resumed_at', '_terminalization_retry', '_cancel_intent_authority_hold', '_owner_wait_resume',
    '_budget_pause_resume', '_budget_pause_hold', '_resume_selection', '_continuation_prepared', '_admission_owner_token',
    '_is_direct_chat', 'detached_at', 'detached_by',
)

def persist_queue_snapshot(reason: str = "") -> bool:
    """Persist queue snapshot for restart/recovery diagnostics.

    Snapshots PENDING/RUNNING under the queue lock: iterating the live dicts
    while HTTP handlers mutate them raised "dictionary changed size during
    iteration" in the supervisor loop (counted toward its crash limit).
    True when published, False when the publication failed, None when the
    documents on disk are kept: no queue was initialized in this process, or
    it has not taken the saved queue into custody (``_keeps_saved_queue``).
    """
    if not _queue().INITIALIZED:  # a boot without a provider: state initialization reads a snapshot as history (#2412)
        log.info("Queue snapshot not published (reason=%s): no supervisor queue in this process", reason)
        return None
    from supervisor.direct_roots import collect_direct_roots

    direct = collect_direct_roots()  # Never wait for an actor while holding Q.
    with _queue()._queue_lock:
        pending_items = [dict(t) for t in _queue().PENDING]
        running_items = [
            (task_id, dict(meta) if isinstance(meta, dict) else {})
            for task_id, meta in _queue().RUNNING.items()
        ]
        acceptance_fences = [dict(row) for row in _queue().ACCEPTANCE_FENCES.values()]
        budget_root_fences = [dict(row) for row in _queue().BUDGET_ROOT_FENCES.values()]
        if _keeps_saved_queue(_queue().QUEUE_SNAPSHOT_PATH, pending_items, running_items, acceptance_fences,
                              budget_root_fences, (direct or {}).get("roots")):
            log.info("Queue snapshot kept (reason=%s): this process has not taken the saved queue into custody", reason)
            return None
        # Honest worker-pool counts from the ACTUAL pool (not the configured max): the live
        # pool can be smaller (a crash-storm/direct-chat fallback clears WORKERS) and a slot
        # mid-reap is popped from RUNNING but NOT assignable. Surface the real assignable-idle
        # count so the context queue digest never falsely advertises a free worker slot.
        try:
            from supervisor import workers as _workers_mod
            from supervisor.worker_pool_lifecycle import worker_ready

            _ws = list(_workers_mod.WORKERS.values())
            worker_total = len(_ws)
            active_worker_count = sum(1 for w in _ws if getattr(w, "active_capacity", True))
            parked_worker_count = worker_total - active_worker_count
            worker_pool_disabled_reason = _workers_mod._worker_pool_execution_state()["disabled_reason"]
            if worker_pool_disabled_reason == "no_workers":
                worker_pool_disabled_reason = ""  # Absence is counted above, not an explicit disablement.
            reaping_count = sum(1 for _w in _ws if getattr(_w, "reaping", False))
            assignable_idle_workers = sum(
                1 for _w in _ws
                if (getattr(_w, "busy_task_id", None) is None and worker_ready(_w)
                    and getattr(_w, "active_capacity", True) and not getattr(_w, "resume_hold", ""))
            )
        except Exception:
            worker_total = 0
            active_worker_count = parked_worker_count = 0
            worker_pool_disabled_reason = "unknown"
            reaping_count = 0
            assignable_idle_workers = 0
        pending_rows = []
        for t in pending_items:
            pending_rows.append({
                "id": t.get("id"), "type": t.get("type"), "priority": t.get("priority"),
                "attempt": t.get("_attempt"), "queued_at": t.get("queued_at"),
                "queue_seq": t.get("_queue_seq"),
                "task": {**{key: t.get(key) for key in _PENDING_FIELDS}, **subagent_intent_fields(t),
                         **{key: t[key] for key in ('_project_admission', 'admitted_dispatch', '_project_scope_none', '_working_recovery',
                                                    # #939: the execution budget and its carried clock offset travel with the row.
                                                    'execution_timeout_sec', '_execution_consumed_sec', '_execution_handoff_sec') if key in t}},
            })
        running_rows = []
        for task_id, meta in running_items:
            task = meta.get("task") if isinstance(meta, dict) else {}
            started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
            paused_sec = float(meta.get("budget_paused_sec") or 0.0) if isinstance(meta, dict) else 0.0
            # Only a dated silence episode is a diagnostic's source authority (#1376):
            # its fact, and while it stays open the progress that revokes it before
            # the recovery lane writes the result. Ordinary progress rotates nothing.
            silence = copy.deepcopy(meta.get("silence_attention")) if isinstance(meta, dict) else None
            running_rows.append({
                "id": task_id, "type": task.get("type"), "priority": task.get("priority"),
                "attempt": meta.get("attempt"), "worker_id": meta.get("worker_id"),
                "owner_wait": meta.get("owner_wait"), "started_at": started,
                **({"silence_attention": silence} if silence else {}),
                **({"last_progress_at": meta.get("last_progress_at")}
                   if silence and silence.get("state") == "pending_attention" else {}),
                "model_wait_quota_clock": meta.get("model_wait_quota_clock"),
                "sleep_parked_at": meta.get("sleep_parked_at"),
                "budget_paused_sec": paused_sec,
                "soft_sent": bool(meta.get("soft_sent")), "task": task,
            })
        payload = {
            SCHEMA_VERSION_KEY: QUEUE_SNAPSHOT_SCHEMA_VERSION,
            FORMAT_KEY: FORMAT_VERSION,
            "pending_count": len(pending_items), "running_count": len(running_items),
            "reaping_count": reaping_count,
            "worker_total": worker_total,
            "active_worker_count": active_worker_count, "parked_worker_count": parked_worker_count,
            "worker_pool_disabled_reason": worker_pool_disabled_reason,
            "assignable_idle_workers": assignable_idle_workers,
            "acceptance_fences": acceptance_fences,
            "budget_root_fences": budget_root_fences,
            "pending": pending_rows, "running": running_rows,
        }
        from ouroboros.server_restart import _AGENT_RESTART_REPLY, agent_restart_snapshot

        payload.update(restart_intent=agent_restart_snapshot(), restart_reply=dict(_AGENT_RESTART_REPLY))
        from ouroboros.budget_pause import BUDGET_HOLD_KEY

        fenced = {str(row.get("root_task_id") or "") for row in budget_root_fences}
        members = {}
        for task in [*pending_items, *(meta.get("task") or {} for _, meta in running_items)]:
            if str(task.get("root_task_id") or task.get("id") or "") in fenced:
                member = {key: task[key] for key in ("id", "_budget_pause", "_continuation_prepared") if key in task}
                for key, fields in ((BUDGET_HOLD_KEY, ("reason", "release")),
                                    ("_owner_wait_resume", ("refusal",)), ("_working_recovery", ("refusal",)),
                                    ("_resume_selection", ("fence_id", "refusal")),
                                    ("_budget_pause_resume", ("pause_id", "grant_id", "authority", "root_fence_id", "selected_by", "root_grant_id", "root_resume_generation"))):
                    if key in task:
                        value = task[key]
                        member[key] = {k: value[k] for k in fields if k in value} if isinstance(value, dict) else value
                members[str(task.get("id") or "")] = member
        dispatch = {"budget_root_fences": budget_root_fences, "members": members}
        documents = {"queue_snapshot.json": payload, "queue_dispatch.json": dispatch, "direct_roots.json": direct}
        previous = _queue().QUEUE_PUBLICATION
        changed = previous is None or previous[:2] != (_queue().QUEUE_SNAPSHOT_PATH, documents) or reason in {"startup", "pre_restart_exit"}
        generation = uuid.uuid4().hex if changed else previous[2]
        state = _queue().QUEUE_SNAPSHOT_PATH.parent
        try:
            if changed:
                # A partial publish must not certify the old no-fence document.
                for attempt in range(3):
                    try:
                        (state / "queue.alive").unlink(missing_ok=True)
                        break
                    except PermissionError:
                        if attempt == 2:
                            raise
                        time.sleep(0.01)  # Windows readers briefly deny delete sharing.
                for name, document in documents.items():
                    published = {**document, "generation": generation, "ts": utc_now_iso(), "reason": reason}
                    if _queue().atomic_write_text(state / name, json.dumps(published, ensure_ascii=False, separators=(",", ":"))) is False:
                        raise OSError(f"queue publication failed: {name}")
            alive = {"generation": generation, "ts": utc_now_iso(),
                     "heartbeats": {tid: meta.get("last_heartbeat_at") for tid, meta in running_items}}
            if _queue().atomic_write_text(state / "queue.alive", json.dumps(alive, separators=(",", ":"))) is False:
                raise OSError("queue liveness publication failed")
            if changed:
                _queue().QUEUE_PUBLICATION = (_queue().QUEUE_SNAPSHOT_PATH, copy.deepcopy(documents), generation)
            return True
        except Exception:
            log.warning("Failed to persist queue snapshot (reason=%s)", reason, exc_info=True)
            return False


def _saved_pending_task(row: dict) -> dict:
    """A saved PENDING row's task with the queue time its outer row keeps (#1805).

    The outer value decides, an explicit null included; only a row without one
    falls back to a nested copy. Text and a finite epoch (``direct_pause.retain_direct_hold``
    stamps one) are known; a time no row kept stays unknown (``None``), never this restore's clock.
    """
    task = dict(row["task"])
    value = row["queued_at"] if "queued_at" in row else task.get("queued_at")
    known = (isinstance(value, str) and value.strip()) or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0)
    task["queued_at"] = value if known else None
    return task


def parse_iso_to_ts(iso_ts: str) -> Optional[float]:
    """Parse ISO timestamp to Unix time."""
    txt = str(iso_ts or "").strip()
    if not txt:
        return None
    try:
        return datetime.datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except Exception:
        log.debug("Failed to parse ISO timestamp: %s", txt, exc_info=True)
        return None


def _fence_snapshot_running_rows(rows: Any, *, restored_ids: "set[str]") -> "list[str]":
    """Fence every RUNNING row that survived the shutdown with a durable cancel intent.

    Restore is the last holder of the pre-restart running list, but it is NOT a
    terminal writer: minting the intent hands each row to the one settle owner,
    which claims it, kills a worker that outlived SIGTERM, reconciles, and only
    then writes the terminal with its own text — expiring an open quiz and
    closing the paired owner wait through the task-done seam. A second writer
    here would race that surviving worker; an intent cannot. An UNREADABLE
    cancel authority mints nothing: the unknown is disclosed, never fenced, and
    neither is a row with no durable result — custody would settle that intent as
    not_found and write nothing, leaving the boot notice naming a cancellation
    that never happens. Returns the fenced task ids.
    """
    from ouroboros.cancel_intents import has_active_intent, request_cancel
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, load_task_result,
    )

    fenced: list[str] = []
    unreadable: list[str] = []
    unrecorded: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        task_id = str(row.get("id") or "") if isinstance(row, dict) else ""
        if not task_id or task_id in restored_ids:
            continue
        try:
            stored = load_task_result(_queue().DRIVE_ROOT, task_id, strict=True) or {}
            if not stored:
                # Admission writes the durable scheduled row (task_admission.py
                # :536-543), so even a failed RUNNING mirror leaves a readable
                # row this loop fences; reaching here means the admission record
                # is missing or was deleted, and custody cannot settle an id it
                # has no record of, so the row is logged and never fenced.
                unrecorded.append(task_id)
                continue
            status = str(stored.get("status") or "")
            if status in _TRULY_TERMINAL_STATUSES or status == STATUS_CANCEL_REQUESTED:
                continue
            if has_active_intent(_queue().DRIVE_ROOT, task_id, strict=True):
                continue  # cancellation custody already owns this row
            intent = request_cancel(
                _queue().DRIVE_ROOT, task_id,
                reason="server_shutdown", source="snapshot_restore",
            )
        except Exception:
            unreadable.append(task_id)
            log.warning("Snapshot restore left running row %s unfenced: its cancel "
                        "authority is unreadable", task_id, exc_info=True)
            continue
        if not intent.get("already_settled"):
            fenced.append(task_id)
    for kind, task_ids in (("queue_restore_running_fence_unreadable", unreadable),
                           ("queue_restore_running_row_without_result", unrecorded)):
        if task_ids:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": kind, "task_ids": task_ids},
            )
    return fenced


def _retain_snapshot_pending(snapshot_pending: list, running_rows: list, *, stale: bool,
                             direct_caught: "list[str]" = (), handoffs: Optional[dict] = None) -> tuple:
    """Which snapshot rows survive the restore, and the RUNNING rows parked beside them.

    A grant that never reached a worker before this restart is not carried into
    the new generation: it returns to its exact pause and the owner re-issues
    Resume (a revocation that cannot be written leaves the row HELD). A grant
    the durable row says was CONSUMED is never re-armed: that row is stale
    carrier of work that ran on, so it is not retained at all and is fenced as
    the running work it names (its id is returned third). A RUNNING row whose
    durable pause was already complete is parked, never fenced; so is a direct
    root the stop caught (``direct_caught``) in that state: queue init's direct
    handover names these non-pooled turns, and handing such an id to the
    shutdown cancel fence would cancel a saved pause
    (#1196). Its queue record is rebuilt from the durable result the pause
    wrote (chat, lane fact, origin, metadata, attempt) and parked through the
    same path a RUNNING row takes; a turn with no complete pause (no row, no
    source, another attempt) stays on the fence path. An exact budget pause is
    retained WITHOUT waking whatever the snapshot's age — a corrupt, missing or
    refused source becomes a typed, visible HOLD beside its marker, never a
    dropped or cancelled task. A row the owner's Restart held is retained the
    same way, at any age; and while that Restart's boot marker is still
    unconsumed, every other never-started row found here
    takes the same hold BEFORE the marker can be consumed — a final snapshot
    the stop could not write never turns the older queue into dispatchable
    work. An owner-wait handoff needs its acknowledged restart transaction.
    Project holds retain custody through age/receipt uncertainty without
    dispatch permission, beside (never instead of) Restart holds, and refuse a
    possible handoff's replay themselves; a fresh host-unscoped possible handoff
    is held, never replayed. Accepted work without a returning transaction waits
    for explicit Resume regardless of age, beside any Project verification hold.
    Returns retained rows, unseen pause ids, consumed task ids, and the roots'
    pre-park rows. Unseen pauses use those transient rows to apply the ordinary
    latch generation rule before parking erased their prior selection.
    ``handoffs`` (when given) receives ``{task_id: from_attempt}`` for each saved RUNNING row of
    the stopped generation that recovery turned into its successor (#1183).
    """
    from ouroboros.budget_pause import budget_pause_restore_refusal, budget_pause_row
    from ouroboros.owner_wait import restore_owner_wait_allowed, saved_sleep_checkpoint
    from ouroboros.project_admission import host_unscoped
    from ouroboros.task_results import load_task_result
    from supervisor.budget_resume import resume_hold_requires_fence, revoke_exact_budget_resume
    from supervisor.events_budget import (HOLD_OWNER_RESTART, HOLD_PANIC, HOLD_RESTORE_REFUSED_PREFIX,
                                          budget_hold_fact, hold_budget_row, hold_restored_budget_pause)
    from supervisor.restart_retention import (fresh_return_transaction, hold_after_stop, hold_for_owner_restart,
                                              hold_stopped_queue, never_started, park_saved_pause, pause_ids,
                                              recover_saved_running,
                                              retained_pending, HOLD_SAVED_SLEEP_RECOVERY, unseen_pause)
    from supervisor.schedule_occurrence import first_attempt, first_dispatch_unsent, restore_verdict

    from ouroboros.work_stop import clear_refusal, continuation_refusal, current_stop

    from ouroboros.delegate_recovery import boot_door
    door = boot_door(_queue().DRIVE_ROOT)
    owner_restart, panicked = door == "owner_restart", door == "panic"
    panic = panicked
    sleep_hold_reason = HOLD_PANIC if panicked else HOLD_OWNER_RESTART if owner_restart else HOLD_SAVED_SLEEP_RECOVERY
    transaction = {} if panicked else fresh_return_transaction(_queue().DRIVE_ROOT)
    returning_queue = set(transaction.get("queue_ids") or []) | set(transaction.get("return_ids") or [])
    stop_door = door or "app_stop"
    prior_roots = {str(task["id"]): copy.deepcopy(task) for task in
                   snapshot_pending + [row.get("task") for row in running_rows if isinstance(row, dict)]
                   if isinstance(task, dict) and task.get("id")
                   and str(task.get("root_task_id") or task["id"]) == str(task["id"])}
    unseen: set = set()
    for task in snapshot_pending:
        if isinstance(task.get("_budget_pause_resume"), dict):
            before = pause_ids(task)
            revoke_exact_budget_resume(task, "restart_before_dispatch")
            if unseen_pause(task, before):
                unseen.add(str(task.get("id") or ""))
    direct_rows: list = []
    direct_saved: list = []
    direct_wait_handoffs: list = []
    tx_id = str(transaction.get("transaction_id") or "")
    tx_task_ids = {str(tid) for tid in (transaction.get("task_ids") or [])}
    for task_id in direct_caught:
        task_id = str(task_id or "")
        if not task_id:
            continue
        try:
            stored = load_task_result(_queue().DRIVE_ROOT, task_id, strict=True) or {}
            pause = budget_pause_row(_queue().DRIVE_ROOT, task_id)
            wait = stored.get("owner_wait") if isinstance(stored.get("owner_wait"), dict) else {}
            if not pause:
                if (wait.get("state") == "waiting" and wait.get("reason") == "owner_pause"
                        and wait.get("source_ref")):
                    pause = {**wait, "is_direct_chat": True}
                elif not (wait.get("state") == "waiting" and wait.get("reason") in {"owner", "review", "sleep"}
                          and wait.get("source_ref") and tx_id and task_id in tx_task_ids
                          and not (owner_restart and wait.get("reason") == "sleep")):
                    sleep_wait, _ = saved_sleep_checkpoint(_queue().DRIVE_ROOT, task_id)
                    if sleep_wait:
                        # The prior direct roster plus exact source is the locator;
                        # park_saved_pause performs the same conversion as a worker.
                        pause = {**sleep_wait, "is_direct_chat": True}
        except Exception:
            continue
        saved_pause = bool(stored and pause and pause.get("source_ref") and pause.get("is_direct_chat"))
        saved_wait = bool(stored and not pause and wait.get("state") == "waiting"
                          and wait.get("reason") in {"owner", "review", "sleep"} and wait.get("source_ref"))
        if not stored or not (saved_pause or saved_wait or stored.get("_is_direct_chat")):
            continue
        attempt = int(stored.get("task_attempt") or (pause or {}).get("task_attempt")
                      or wait.get("task_attempt") or 1)
        record: dict = {
            "id": task_id, "type": "task", "chat_id": stored.get("chat_id"), "_is_direct_chat": True,
            "_attempt": attempt, "root_task_id": task_id, "depth": 0,
        }
        for key in ("origin_message_ref", "origin_message_text", "metadata", "project_id",
                    "tool_profile", "detached_at", "detached_by",
                    "task_contract", "title", "text", "budget_drive_root"):
            if stored.get(key) not in (None, ""):
                record[key] = stored[key]
        if (saved_wait and int(wait.get("task_attempt") or 0) == attempt and tx_id
                and task_id in tx_task_ids and not (owner_restart and wait.get("reason") == "sleep")):
            record["_owner_wait_resume"] = {**wait, "restart_transaction_id": tx_id}
            direct_wait_handoffs.append(record)
        else:
            (direct_rows if saved_pause else direct_saved).append({"task": record, "attempt": attempt})
    before_park = {str(row["task"].get("id") or ""): pause_ids(row["task"]) for row in running_rows
                   if isinstance(row, dict) and isinstance(row.get("task"), dict)}
    parked = _park_pausing_running_rows(list(running_rows) + direct_rows + direct_saved, snapshot_pending,
        sleep_hold_reason=sleep_hold_reason)
    taken = {str(task.get("id") or "") for task in list(snapshot_pending) + direct_wait_handoffs + parked}
    # Interrupted work whose state was saved returns (or waits) under its own id
    # instead of the shutdown fence's cancellation (#1563).
    recovered = recover_saved_running(
        [row for row in list(running_rows) + direct_saved if isinstance(row, dict)
         and str((row.get("task") or {}).get("id") or "") not in taken],
        _queue().DRIVE_ROOT, transaction=transaction, cause=stop_door)
    if handoffs is not None:
        handoffs.update({str(task["id"]): source["from_attempt"] for task in recovered
                         if isinstance(source := task.get("_working_recovery"), dict)
                         and source.get("source_task_id") == task["id"] and source.get("from_attempt")})
    parked += recovered
    unseen.update(str(task.get("id") or "") for task in parked
                  if unseen_pause(task, before_park.get(str(task.get("id") or ""), ("", ""))))
    retained = []
    consumed: list = []
    for task in list(snapshot_pending) + direct_wait_handoffs + parked:
        try:
            if resume_hold_requires_fence(task, _queue().DRIVE_ROOT):
                consumed.append(str(task.get("id") or ""))
                continue
        except (OSError, ValueError, TypeError, RuntimeError):
            retained.append(task)  # unreadable continuation authority is not interrupted work
            continue
        wait_allowed = False
        if task.get("_owner_wait_resume"):
            try:
                wait_allowed = not panicked and restore_owner_wait_allowed(_queue().DRIVE_ROOT, task)
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                retained.append(hold_restored_budget_pause(task, _queue().DRIVE_ROOT,
                    reason=HOLD_RESTORE_REFUSED_PREFIX + "owner_wait_source_unavailable", detail=str(exc)))
                continue  # unknown authority holds this row, not unrelated accepted work
        if task.get("_owner_wait_resume") and (
                task["_owner_wait_resume"].get("reason") == "owner_pause" or not wait_allowed):
            # Older stop doors stamped warm pauses as question handoffs. Their
            # own pause custody wins even if that transaction was acknowledged.
            before = pause_ids(task)
            paused = park_saved_pause(task, int(task.get("_attempt") or 1),
                pathlib.Path(task.get("budget_drive_root") or _queue().DRIVE_ROOT),
                pause_source="restart_owner_wait_handoff", sleep_hold_reason=sleep_hold_reason)
            if paused is not None:
                task = paused
                if unseen_pause(task, before):
                    unseen.add(str(task.get("id") or ""))
        retry = task.get("_terminalization_retry")
        if (isinstance(retry, dict) and retry.get("trigger") == "deadline"
                and isinstance(retry.get("unsubmitted_launch"), dict)):
            # Already non-dispatchable: its reaper validates the caller's exact
            # non-put claim. A generic unknown-dispatch hold would strand it.
            retained.append(task)
            continue
        # Import opaque owner/cap carriers even for an exact saved Pause. Its
        # single-use Resume is a separate authority from ordinary queue replay.
        verdict = restore_verdict(task)
        schedule_replay = verdict == "replay"
        if schedule_replay and task.get("admitted_dispatch") == "possible" and first_dispatch_unsent(task):
            # Its receipt proves no first handoff: never started, under this stop's own rule.
            task["admitted_dispatch"] = "none"
            if continuation_refusal(task).get("dispatch_unknown"):
                clear_refusal(task)
        # Busy at boot proves neither way: this stop's rule for an unsent row, under Project
        # custody that re-reads the receipt each tick (``revalidate_project_holds``).
        recheck = (verdict == "busy" and task.get("admitted_dispatch") == "possible" and first_attempt(task)
                   and budget_hold_fact(task) is None)
        if _exact_pause_row(task):
            refusal = budget_pause_restore_refusal(_queue().DRIVE_ROOT, task)
            retained.append(task if not refusal else hold_restored_budget_pause(
                task, _queue().DRIVE_ROOT, reason=HOLD_RESTORE_REFUSED_PREFIX + refusal))
            if (owner_restart or panic) and (task.get("_budget_pause") or {}).get("reason") == "sleep":
                # A cold model sleep never wakes itself after the owner's Restart or a Panic.
                hold_after_stop(retained[-1], _queue().DRIVE_ROOT, HOLD_OWNER_RESTART if owner_restart else HOLD_PANIC)
        elif task.get("_owner_wait_resume"):
            # An existing Project hold is custody; its revalidation reads the owner-wait authority.
            if budget_hold_fact(task) or continuation_refusal(task) or wait_allowed:
                retained.append(task)
            else:  # a refused cold handoff keeps its exact wait source, held for Resume
                handoff = dict(task)
                handoff.pop("_owner_wait_resume", None)
                recovered_wait = recover_saved_running(
                    [{"task": handoff, "attempt": int(task.get("_attempt") or 1)}], _queue().DRIVE_ROOT,
                    transaction={}, cause=stop_door)
                if recovered_wait:
                    retained.extend(recovered_wait)
                else:
                    try:
                        current = (load_task_result(_queue().DRIVE_ROOT, str(task.get("id") or ""),
                                                    strict=True) or {}).get("owner_wait") or {}
                    except (OSError, ValueError, TypeError, RuntimeError):
                        current = None  # unreadable is not a closed or missing wait
                    if current is None or (current.get("state") == "waiting"
                            and current.get("wait_id") == task["_owner_wait_resume"].get("wait_id")):
                        retained.append(hold_restored_budget_pause(
                            task, _queue().DRIVE_ROOT, reason=HOLD_RESTORE_REFUSED_PREFIX + "owner_wait_source_unavailable",
                            detail="the saved wait could not be restored; its original handoff is retained, not replayed"))
        else:
            # Age or an unavailable receipt cannot erase accepted Project work;
            # assignment still requires positive original no-dispatch authority.
            project_hold, dispatch = task.get("_project_admission_restore_hold"), task.get("admitted_dispatch")
            if not project_hold and (task.get("project_id") or "_project_admission" in task) and (
                    stale or dispatch != "none" or not schedule_replay):
                project_hold = {"reason": "project_routing_fence_lookup_failed", "detail": (
                    "The saved task needs original Project and no-dispatch verification."
                    if isinstance(task.get("_project_admission"), dict) else
                    "The original Project basis has unknown historical identity; automatic recovery is not authorized.")}
            elif not project_hold and host_unscoped(task) and not stale and dispatch != "none" and schedule_replay:
                project_hold = {"reason": "project_routing_fence_lookup_failed",
                                "detail": "The task may already have reached a worker; automatic recovery is not authorized."}
            elif not project_hold and recheck:
                project_hold = {"reason": "project_routing_fence_lookup_failed", "detail": (
                    "The schedule receipt was busy at restore; recovery is checked automatically when it is readable.")}
            saved_or_accepted = never_started(task) or isinstance(task.get("_working_recovery"), dict)
            kept = retained_pending(task, sleep_hold_reason=sleep_hold_reason)
            if panicked and not current_stop(task):
                # This also covers a planned delegate return under Project custody:
                # its current stop must survive consumption of the installation input.
                row = hold_stopped_queue(dict(task), _queue().DRIVE_ROOT, stop_door)
            elif kept:
                row = task
            elif str(task.get("id") or "") in returning_queue and (saved_or_accepted and schedule_replay or recheck):
                row = task  # the fresh Restart returns it: ordinary admission, at any age
            elif owner_restart and (never_started(task) and schedule_replay or recheck):
                row = hold_for_owner_restart(dict(task), _queue().DRIVE_ROOT)
            elif (owner_restart and (not schedule_replay or not never_started(task))
                  or (dispatch == "possible" or verdict == "possible") and not project_hold):
                row = dict(task)  # a committed mark under a stale snapshot is held as visibly as its own
                hold_budget_row(row, reason="dispatch_outcome_unknown",
                                detail="an assignment may have reached a worker; reconcile its receipt before resuming",
                                result_root=_queue().DRIVE_ROOT)
            elif not project_hold and not schedule_replay:
                continue  # settled, contended or unprovable: the occurrence reconcile owns it
            elif saved_or_accepted or recheck:
                # Quit/crash retain ALL accepted work for explicit Resume (#1563).
                # Project verification is independent: healing it cannot release
                # the owner's application-stop hold, even for a fresh snapshot.
                row = hold_stopped_queue(dict(task), _queue().DRIVE_ROOT, stop_door)
            elif not project_hold and stale:
                continue  # no provable never-started fact: the old snapshot-age rule still decides
            else:
                row = task  # Project custody is retained, not replayed: terminal/cancel checks below still run
            retained.append({**row, "_project_admission_restore_hold": project_hold} if project_hold else row)
    if consumed:
        _queue().append_jsonl(
            _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "queue_restore_stale_consumed_grant_rows",
             "task_ids": consumed, "action": "fenced_as_running_work"},
        )
    from supervisor.restart_retention import _record_refused_returns

    kept_ids = {str(task.get("id") or "") for task in retained}
    _record_refused_returns(_queue().DRIVE_ROOT, {
        str(task["id"]): "planned_handoff_not_retained" for task in snapshot_pending
        if task.get("id") in set(transaction.get("task_ids") or []) and task["id"] not in kept_ids}, stop_door)
    return retained, unseen, consumed, prior_roots


def _raise_parked_root_fences(retained: list, unseen: "set[str]" = frozenset(),
                              prior_roots: Optional[dict] = None) -> None:
    """Raise the root admission latches restore owes — AFTER the snapshot's fence
    map is restored, or the restore would erase it. That map stays the authority:
    an old marker cannot replace its latch. A pause it never recorded (``unseen``)
    follows the live generation rule using the root's pre-park selection. An
    owner marker raises its queue root's durable owner fence, never the identity it
    saved: released raises nothing (only the root's own Resume lifts it), closed
    raises the current fence; unreadable or absent authority keeps the map's latch,
    else the marker's."""
    if not retained:
        return
    from supervisor.events_budget import _set_root_budget_pause_locked
    from ouroboros.owner_pause import REASON_OWNER, fence_closed, read_fence

    with _queue()._queue_lock:
        for parked in retained:
            marker = parked.get("_budget_pause") if isinstance(parked.get("_budget_pause"), dict) else {}
            task_id = str(parked.get("id") or "")
            root_id = str(parked.get("root_task_id") or task_id)
            try:
                owner = read_fence(pathlib.Path(parked.get("budget_drive_root") or _queue().DRIVE_ROOT), root_id)
                latch = {"cause": "owner_pause", "fence_id": owner["fence_id"],
                         "paused_at": owner.get("requested_at")} if fence_closed(owner) else None
            except Exception:
                owner = latch = None
            root_scoped = str(marker.get("scope") or "") == "root"
            latch_root = root_id if marker.get("reason") == REASON_OWNER else str(marker.get("root_task_id") or "")
            if root_scoped and marker.get("reason") == REASON_OWNER and owner:
                if latch:
                    marker["fence_id"] = _set_root_budget_pause_locked(root_id, latch)["fence_id"]
            elif root_scoped and latch_root:
                current = _queue().BUDGET_ROOT_FENCES.get(latch_root)
                if task_id in unseen and marker.get("reason") != REASON_OWNER:
                    pause = {**marker, "fence_id": None} if current else marker
                    marker["fence_id"] = _set_root_budget_pause_locked(
                        latch_root, pause, prior_root=(prior_roots or {}).get(latch_root, {}))["fence_id"]
                elif current:
                    marker["fence_id"] = current["fence_id"]
                elif marker.get("reason") == REASON_OWNER or task_id in unseen:
                    if marker.get("reason") == REASON_OWNER:
                        log.warning("Owner fence of %s unknown at restore; %s keeps its saved latch", root_id, task_id)
                    marker["fence_id"] = _set_root_budget_pause_locked(latch_root, marker)["fence_id"]
            if task_id and root_id == task_id:
                # A crash after the eligible snapshot but before the final
                # durable Resume commit leaves this fence closed. Rebuild its
                # projection so explicit Resume can finish the same action.
                if owner is None:
                    log.warning("Owner fence unreadable at restore; launch stays refused for %s", task_id)
                elif latch and task_id not in _queue().BUDGET_ROOT_FENCES:
                    _set_root_budget_pause_locked(task_id, latch)


def _refuse_restore_invalid_fences(snapshot_pending: list, *, budget: bool = False) -> int:
    """Malformed fences never restore ordinary work. Invalid acceptance evidence
    cancels ordinary rows; invalid budget evidence leaves them unrestored.

    A saved EXACT budget pause is the exception (#1196): it is retained as a
    typed, non-dispatchable HOLD with its ORIGINAL ``_budget_pause`` locator
    (an earlier restore hold on the same row is kept), appended to PENDING
    directly, never cancelled — the owner's next Resume re-validates it. A
    pause whose durable result is already terminal is left alone. Returns the
    number of retained rows.
    """
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, STATUS_CANCELLED, load_task_result, write_task_result,
    )
    from supervisor.events_budget import (
        HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT, HOLD_INVALID_BUDGET_FENCE_SNAPSHOT,
        hold_restored_budget_pause,
    )
    from ouroboros.work_stop import continuation_refusal
    from supervisor.restart_retention import retained_pending

    cancelled: list[str] = []
    retained: list[str] = []
    skipped_terminal: list[str] = []
    for task in snapshot_pending:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        if task.get("_project_admission_restore_hold") or task.get("project_id"):
            # Broken restore evidence belongs to the continuation, not to the owner.
            held = {**task, "_project_admission_restore_hold": task.get("_project_admission_restore_hold") or {
                "reason": "project_routing_fence_lookup_failed", "detail": "Independent restore authority is invalid.",
            }}
            hold_restored_budget_pause(held, _queue().DRIVE_ROOT,
                reason=HOLD_INVALID_BUDGET_FENCE_SNAPSHOT if budget else HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT)
            with _queue()._queue_lock:
                _append_held_pending_row(held)
            retained.append(task_id)
            continue
        if not (_exact_pause_row(task) or retained_pending(task)):
            cancelled.append(task_id)
            continue
        try:
            stored = load_task_result(_queue().DRIVE_ROOT, task_id, strict=True) or {}
            if str(stored.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                skipped_terminal.append(task_id)
                continue
        except Exception:
            log.debug("Result authority unreadable for paused row %s; retained held", task_id, exc_info=True)
        held = dict(task)
        if not continuation_refusal(held):
            held = hold_restored_budget_pause(
                held, _queue().DRIVE_ROOT, reason=(HOLD_INVALID_BUDGET_FENCE_SNAPSHOT if budget
                                                  else HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT),
                detail="the snapshot's fence evidence was malformed; the saved pause is "
                       "retained, not cancelled, until an explicit Resume re-validates it")
        with _queue()._queue_lock:
            _append_held_pending_row(held)
        retained.append(task_id)
    _queue().append_jsonl(
        _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": ("queue_restore_invalid_budget_root_fences" if budget
                                      else "queue_restore_invalid_acceptance_fences"),
         "affected_task_ids": cancelled, "action": "fail_closed_no_restore",
         "retained_budget_paused_task_ids": retained, "skipped_terminal_task_ids": skipped_terminal},
    )
    try:
        for task in ([] if budget else snapshot_pending):
            task_id = str(task.get("id") or "")
            if task_id and task_id in cancelled:
                existing = load_task_result(_queue().DRIVE_ROOT, task_id) or {}
                write_task_result(
                    _queue().DRIVE_ROOT, task_id, STATUS_CANCELLED,
                    **_cancel_result_fields(
                        task, existing=existing,
                        result="Task was not restored because its acceptance-fence snapshot was invalid."),
                )
    except Exception:
        log.warning("Failed to terminalize tasks from invalid acceptance-fence snapshot", exc_info=True)
    return len(retained)


def _append_held_pending_row(task: dict) -> None:
    """Append one proven-revivable held row to PENDING with queue-order facts (lock held).
    Its queue time is the saved one; a row that kept none stays unknown (#1805)."""
    if "_queue_seq" not in task:
        _queue().QUEUE_SEQ_COUNTER_REF["value"] += 1
        task["_queue_seq"] = _queue().QUEUE_SEQ_COUNTER_REF["value"]
    task.setdefault("queued_at", None)
    _queue().PENDING.append(task)
    _queue().sort_pending()


def _park_pausing_running_rows(running_rows: list, snapshot_pending: list, *,
                              sleep_hold_reason: str = "saved_sleep_recovery") -> list:
    """RUNNING rows whose durable budget pause is LIVE become parked PENDING rows (#1196).

    The worker wrote the ``budget_pause`` row and stored its source BEFORE the
    park event left it; a shutdown in that window leaves a RUNNING row in the
    snapshot whose exact continuation is complete on disk. Handing that row to
    the shutdown cancel fence would cancel a saved pause, so it is parked under
    its exact marker through the ONE retention predicate every stop shares
    (``restart_retention.park_saved_pause``): ``pausing``/``paused`` rows are
    confirmed ``paused`` with ``pause_source=restart_during_pausing``, and an
    unconsumed ``resume_granted`` row is revoked first — a revocation that
    cannot be written keeps the row parked under a typed hold. A consumed
    grant is ordinary running work and keeps the ordinary fence; an unreadable
    pause authority proves nothing and keeps it too; so does an explicit Stop.
    Rows already in the pending snapshot are never duplicated.
    """
    from supervisor.restart_retention import PAUSE_REQUEST_ONLY, park_saved_pause, pause_retention

    pending_ids = {str(row.get("id") or "") for row in snapshot_pending if isinstance(row, dict)}
    parked: list = []
    for row in running_rows if isinstance(running_rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("task"), dict):
            continue
        task = dict(row["task"])
        task_id = str(task.get("id") or "")
        if not task_id or task_id in pending_ids:
            continue
        attempt = int(row.get("attempt") or task.get("_attempt") or 1)
        result_root = pathlib.Path(task.get("budget_drive_root") or _queue().DRIVE_ROOT)
        try:
            if pause_retention(result_root, task_id, attempt) == PAUSE_REQUEST_ONLY:
                from ouroboros.budget_pause import budget_pause_row
                from ouroboros.cancel_intents import has_active_intent
                from ouroboros.working_checkpoint import complete_pause_from_working

                # This process is dead. Preserve its saved cognition under the same
                # pause; a working retry alone would leave the sleep launch fence shut.
                if not has_active_intent(result_root, task_id, strict=True):
                    complete_pause_from_working(result_root, task_id,
                                                budget_pause_row(result_root, task_id), attempt)
        except Exception:
            log.warning("Could not complete interrupted pause of %s from saved work", task_id, exc_info=True)
        parked_task = park_saved_pause(task, attempt, result_root, pause_source="restart_during_pausing",
                                       sleep_hold_reason=sleep_hold_reason)
        if parked_task is not None:
            parked.append(parked_task)
    if parked:
        _queue().append_jsonl(
            _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "queue_restore_parked_budget_pauses",
             "task_ids": [str(task.get("id") or "") for task in parked]},
        )
    return parked


def _descends_from(task: Any, roots: "set[str]", pending_by_id: dict) -> bool:
    """Whether this row's lineage reaches ANY of ``roots``.

    The whole ancestry is walked, not just the immediate parent: a snapshot holds
    a tree, and a grandchild of an interrupted root is as unstartable as its
    child. ``root_task_id`` answers first because a deep row names its root
    directly; the parent chain is then followed through the snapshot's own rows,
    which is every ancestor a restore can resolve without reading disk.
    """
    if not roots or not isinstance(task, dict):
        return False
    if str(task.get("root_task_id") or "") in roots:
        return True
    current = task
    seen: set[str] = set()
    while isinstance(current, dict):
        parent_id = str(current.get("parent_task_id") or "")
        if not parent_id or parent_id in seen:
            return False
        if parent_id in roots:
            return True
        seen.add(parent_id)
        current = pending_by_id.get(parent_id)
    return False


def _interrupted_ancestors(fenced_running: "list[str]", snapshot_pending: list, *, unknown: set) -> "set[str]":
    """The ids whose interruption a restored PENDING child cannot survive.

    The rows this boot just fenced, plus the ancestors an EARLIER boot already
    handed to cancellation custody: a multi-boot stop settles the root first, so
    by the time the child is read again its parent is no longer a RUNNING row —
    only an active intent or a stored ``cancelled`` result with the shutdown
    cause proves what happened to it. Queued ancestors are read alike. ``unknown``
    collects unreadable, missing or never-admitted ones: a parent wrote its
    running result before spawning a child, so absence is never permission.
    """
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    interrupted = set(fenced_running)
    candidates: set[str] = set()
    for task in snapshot_pending:
        if not isinstance(task, dict) or not str(task.get("parent_task_id") or ""):
            continue
        for key in ("parent_task_id", "root_task_id"):
            ancestor = str(task.get(key) or "")
            if ancestor and ancestor not in interrupted:
                candidates.add(ancestor)
    for ancestor in candidates:
        try:
            if has_active_intent(_queue().DRIVE_ROOT, ancestor, strict=True):
                interrupted.add(ancestor)
                continue
            stored = load_task_result(_queue().DRIVE_ROOT, ancestor, strict=True)
        except Exception:
            log.warning("Snapshot restore could not read ancestor %s", ancestor, exc_info=True)
            stored = None
        if not stored or stored.get("admission_outcome") == "never_admitted":
            unknown.add(ancestor)  # never a proven interruption: a scope-verifiable child waits
            continue
        origin = stored.get("cancel_origin")
        if (
            str(stored.get("status") or "") == STATUS_CANCELLED
            and isinstance(origin, dict)
            and str(origin.get("reason") or "") == "server_shutdown"
        ):
            interrupted.add(ancestor)
    return interrupted


def _record_queue_restore(
    *, restored: int = 0, skipped_terminal: int = 0,
    cancel_authority_holds: Optional[list] = None, blocked_admission: Optional[list] = None,
    invalid_task_depth: Optional[list] = None, terminalized_running: Optional[list] = None,
    pending_parent_interrupted: Optional[list] = None, direct_roots_incomplete: bool = False,
) -> None:
    """The one durable row a restore leaves: what it revived, what it left to
    cancellation custody, which surviving RUNNING rows it fenced, and which
    PENDING children it refused to start behind an interrupted parent. A stale
    snapshot with nothing to revive still records the fences it minted, and a
    direct-root roster that could not name its live turns is disclosed as the
    gap it is."""
    if not (restored or skipped_terminal or blocked_admission or terminalized_running
            or pending_parent_interrupted or direct_roots_incomplete):
        return
    _queue().append_jsonl(
        _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "queue_restored_from_snapshot",
            "restored_pending": restored,
            "skipped_terminal": skipped_terminal,
            "cancel_authority_holds": list(cancel_authority_holds or []),
            "blocked_admission": list(blocked_admission or []),
            "invalid_task_depth": list(invalid_task_depth or []),
            "terminalized_running": list(terminalized_running or []),
            "pending_parent_interrupted": list(pending_parent_interrupted or []),
            "direct_roots_incomplete": bool(direct_roots_incomplete),
        },
    )


def _release_successor_fences(raw_fences: Any, raw_budget: Any, pending: list, handoffs: dict) -> Any:
    """#1183: only a validated stopped-generation handoff releases a fence. ``handoffs``
    names the RUNNING rows of the previous generation that recovery turned into their
    successors (``recover_saved_running``: exact source task and attempt); that ended
    attempt's own active fence must not cancel the retained successor continuing its
    work. An attempt number alone proves nothing; malformed lists stay for the typed refusal."""
    if not isinstance(raw_fences, list) or not all(isinstance(row, dict) for row in raw_fences):
        return raw_fences
    from supervisor.queue_transitions import release_ended_attempt_fences_locked

    paused = ({str(row.get("root_task_id") or "") for row in raw_budget if isinstance(row, dict)}
              if isinstance(raw_budget, list) else set())
    fences = dict(enumerate(raw_fences))
    for task in pending:
        source = task.get("_working_recovery") if isinstance(task, dict) else None
        task_id = str(task.get("id") or "") if isinstance(task, dict) else ""
        if isinstance(source, dict) and task_id in handoffs and source.get("from_attempt") == handoffs[task_id]:
            release_ended_attempt_fences_locked(fences, task_id, task_attempt=int(handoffs[task_id]),
                                                paused_roots=paused)
    return list(fences.values())


def restore_pending_from_snapshot(
    max_age_sec: int = 900, *, terminalized: Optional[list] = None,
) -> int:
    """Restore retained pending tasks from the queue snapshot.

    Returns the PENDING count revived; ``terminalized`` separately collects
    surviving RUNNING ids fenced with cancel intents for the caller to name.
    While it reads, another thread's publication keeps the saved queue; an
    unreadable snapshot or an interrupted restore never grants the process
    custody of it (``_keeps_saved_queue``).
    """
    if _queue().PENDING:
        return 0
    key = str(pathlib.Path(_queue().QUEUE_SNAPSHOT_PATH).absolute())
    _RESTORING[key] = threading.get_ident()
    try:
        return _restore_pending(key, max_age_sec, terminalized)
    finally:
        _RESTORING.pop(key, None)


def _restore_pending(key: str, max_age_sec: int, terminalized: Optional[list]) -> int:
    try:
        from ouroboros.queue_observation import load_queue_document, queue_stamp
        snap = load_queue_document(_queue().DRIVE_ROOT)
        normalized = normalize_for_restore(snap, _queue().DRIVE_ROOT)
        # An unreadable snapshot (its marker reads as empty) names no queue row; a direct root queue init's
        # handover names as caught still restores, over an empty, stale queue (the gate below).
        unread = bool(snap.get("_snapshot_invalid") or snap.get("_snapshot_missing"))
        ts = str(queue_stamp(snap, boot=True).get("ts") or "")
        ts_unix = _queue().parse_iso_to_ts(ts)
        # Timestamp validity and freshness gate ORDINARY rows only (#1196): a readable snapshot whose stamp is
        # missing or malformed is stale, so an exact pause or an acknowledged owner-wait handoff is still
        # retained under its own authority, never vanishing for Resume to answer task_not_pending.
        if ts_unix is None and not unread:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "queue_restore_snapshot_timestamp_invalid",
                 "snapshot_ts": ts[:64], "action": "treated_as_stale"},
            )
        stale = ts_unix is None or (time.time() - ts_unix) > max_age_sec
        from supervisor.restart_retention import conserve_unreadable_pending
        from ouroboros.task_results import (
            _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, STATUS_CANCELLED,
            load_task_result, write_task_result,
        )
        raw_fences = snap.get("acceptance_fences", [])
        raw_budget_fences = snap.get("budget_root_fences", [])
        snapshot_pending = [
            _saved_pending_task(row)
            for row in (snap.get("pending") or [])
            if isinstance(row, dict) and isinstance(row.get("task"), dict)
        ]
        running_rows = snap.get("running")
        running_rows = running_rows if isinstance(running_rows, list) else []
        # The pre-restart RUNNING rows are read HERE, before the stale gate: this
        # is the last moment the list exists, and a stale snapshot is exactly the
        # case where nothing else will ever settle them. Direct-chat roots never reach the queue: only
        # the handover `queue.init` took over names them; both lists describe work the stop caught, and
        # one fence call gives them the one cancel-intent path custody settles — except a direct root
        # whose exact budget pause was complete on disk, parked under its own id like a paused RUNNING row (#1196).
        direct_roots = dict(_queue().PRIOR_DIRECT_ROOTS)
        # An in-process supervisor revival re-runs queue init while direct turns of THIS process are
        # alive: the roster then names live work, not what a stop caught.
        from supervisor.active_activity import get_direct_activity_registry
        live_direct = {str(row.get("activity_id") or "") for row in get_direct_activity_registry().snapshot()}
        pending_ids = {str(task.get("id") or "") for task in snapshot_pending}
        direct_caught = [task_id for task_id in direct_roots.get("task_ids") or []
                         if task_id not in live_direct and task_id not in pending_ids]
        if unread and not direct_caught:
            return 0
        snapshot_pending, unseen_pauses, consumed_rows, prior_roots = _retain_snapshot_pending(  # #1183 handoffs
            snapshot_pending, running_rows, stale=stale, direct_caught=direct_caught, handoffs=(handoffs := {}))
        from supervisor.restart_retention import consume_return_transaction, fresh_return_transaction

        # This boot decided the returns; a later stop can never reuse the transaction.
        consume_return_transaction(_queue().DRIVE_ROOT, fresh_return_transaction(_queue().DRIVE_ROOT))
        fenced_running = _fence_snapshot_running_rows(
            running_rows
            + [{"id": task_id} for task_id in direct_caught]
            # A stale row whose grant was consumed names work that ran on: the
            # ordinary fence, never a re-armed pause (revoke_exact_budget_resume).
            + [{"id": task_id} for task_id in consumed_rows],
            restored_ids={str(task.get("id") or "") for task in snapshot_pending},
        )
        if terminalized is not None:
            terminalized.extend(fenced_running)
        snapshot_pending, pending_by_id, restored = restore_terminalization_retry_rows(
            snapshot_pending, pending=_queue().PENDING, running=_queue().RUNNING,
            queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF, sort_pending=_queue().sort_pending,
        )
        raw_fences = _release_successor_fences(raw_fences, raw_budget_fences, snapshot_pending, handoffs)
        fenced_roots, malformed_fences, malformed_budget_fences = restore_queue_fences(raw_fences, raw_budget_fences)
        if not malformed_budget_fences:
            _raise_parked_root_fences(snapshot_pending, unseen_pauses, prior_roots)
        if malformed_budget_fences or malformed_fences:
            restored += _refuse_restore_invalid_fences(snapshot_pending, budget=malformed_budget_fences)
            _record_queue_restore(restored=restored, terminalized_running=fenced_running,
                                  direct_roots_incomplete=direct_roots.get("incomplete", False))
            if not unread:  # every row of a READ snapshot decided: this process's publications may replace it
                _RESTORED.add(key)
            if restored > 0 or normalized:
                _queue().persist_queue_snapshot(reason="queue_restored")
            return restored
        if stale and not snapshot_pending and not restored:
            _record_queue_restore(terminalized_running=fenced_running,
                                  direct_roots_incomplete=direct_roots.get("incomplete", False))
            if not unread:  # settling caught direct roots grants no custody of an unread snapshot
                _RESTORED.add(key)
                if normalized:
                    _queue().persist_queue_snapshot(reason="work_stop_format_normalized")
            return 0

        skipped_terminal, invalid_depth_restore = 0, []
        cancel_authority_holds: list[str] = []
        skipped_fenced, blocked_restore, orphan_children, lineage_unknown = [], [], [], set()
        acceptance_held: list[str] = []
        interrupted = _interrupted_ancestors(fenced_running, snapshot_pending, unknown=lineage_unknown)
        for task in snapshot_pending:
            chat_id = task.get("chat_id")
            if not task.get("id") or chat_id is None or chat_id == "":
                continue
            if _descends_from(task, fenced_roots, pending_by_id) and _exact_pause_row(task):
                # #1196: an acceptance fence over the root cancels NOTHING that was
                # paused mid-run: retained under a typed hold, the ordinary revival
                # checks below still apply, and the row is appended directly.
                from supervisor.events_budget import HOLD_ROOT_ACCEPTANCE_FENCED, hold_restored_budget_pause

                task = hold_restored_budget_pause(
                    dict(task), _queue().DRIVE_ROOT, reason=HOLD_ROOT_ACCEPTANCE_FENCED,
                    detail="the root entered acceptance review; the saved pause is retained, not cancelled")
                acceptance_held.append(str(task.get("id") or ""))
            elif _descends_from(task, fenced_roots, pending_by_id):
                task_id = str(task.get("id") or "")
                skipped_fenced.append(task_id)
                try:
                    existing = load_task_result(_queue().DRIVE_ROOT, task_id) or {}
                    stored = write_task_result(
                        _queue().DRIVE_ROOT, task_id, STATUS_CANCELLED,
                        **_cancel_result_fields(task, existing=existing,
                            result="Task was not restored after restart because its root had entered acceptance review.",
                        ),
                    )
                    if not isinstance(stored, dict) or stored.get("status") != STATUS_CANCELLED:
                        raise RuntimeError("Acceptance cancellation was not confirmed")
                except Exception:
                    log.warning("Failed to terminalize fenced snapshot task %s", task_id, exc_info=True)
                    if task.get("project_id") or task.get("_project_admission_restore_hold"):
                        task["_terminalization_retry"] = {
                            "status": STATUS_CANCELLED, "trigger": "acceptance_fence",
                            "reason": "The root entered acceptance review; cancellation is pending.",
                            "reconcile_delegate_custody": False,
                        }
                        with _queue()._queue_lock:
                            _append_held_pending_row(task)
                        restored += 1
                continue
            # AR2-10 (§8-A1): restore and the intent check share the queue lock.
            with _queue()._queue_lock:
                skip_revival = False
                cancel_authority_unreadable = False
                try:
                    existing = load_task_result(_queue().DRIVE_ROOT, str(task.get("id")), strict=True)
                    existing_status = str(existing.get("status") or "") if existing else ""
                except Exception:
                    if _exact_pause_row(task) or task.get("_owner_wait_resume"):
                        # Unreadable authority never terminalizes a saved pause or wait.
                        # The row stays PENDING under its ORIGINAL
                        # locator and a typed hold (the earlier restore hold, if
                        # any, is kept); the next Resume re-reads the authority.
                        from ouroboros.budget_pause import RESTORE_REFUSAL_RECORD_UNREADABLE
                        from supervisor.events_budget import (
                            HOLD_RESTORE_REFUSED_PREFIX, hold_restored_budget_pause,
                        )
                        from ouroboros.work_stop import continuation_refusal

                        if not continuation_refusal(task):
                            task = hold_restored_budget_pause(
                                dict(task), _queue().DRIVE_ROOT,
                                reason=HOLD_RESTORE_REFUSED_PREFIX + RESTORE_REFUSAL_RECORD_UNREADABLE)
                        acceptance_held.append(str(task.get("id") or ""))
                        log.warning("Snapshot restore retained paused row %s under a hold: its "
                                    "result authority is unreadable", task.get("id"), exc_info=True)
                    if conserve_unreadable_pending(task):
                        # A receipt read failure cannot select a terminal policy for conserved
                        # work: its pause, its Project hold, or accepted Project work held here.
                        _append_held_pending_row(task)
                        restored += 1
                        continue
                    # Result-authority loss already has terminal custody: once
                    # its retry can prove a writable result, the task is failed
                    # rather than replayed over an unknown exact-id lifecycle.
                    task["_terminalization_retry"] = {
                        "reason": "Pending task result authority is unreadable; dispatch is blocked.",
                        "status": "failed",
                        "trigger": "pending_result_authority",
                        "reconcile_delegate_custody": False,
                    }
                    restore_terminalization_retry(
                        task, pending=_queue().PENDING, running=_queue().RUNNING,
                        queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF,
                        sort_pending=_queue().sort_pending,
                    )
                    skipped_terminal += 1
                    log.debug("Snapshot restore result-authority check failed for %s", task.get("id"), exc_info=True)
                    continue
                else:
                    if (not _exact_pause_row(task) and not task.get("_owner_wait_resume")
                            and not task.get("_project_admission_restore_hold")
                            and restored_handoff_unproven(task, existing, ancestor_unknown=_descends_from(task, lineage_unknown, pending_by_id))):
                        task["_project_admission_restore_hold"] = {
                            "reason": "project_routing_fence_lookup_failed",
                            "detail": "Original no-dispatch or parent evidence is unavailable; automatic recovery is not authorized.",
                        }
                    # Terminal OR cancel-intent — both must not be resurrected as
                    # pending. Intent lives in the durable projection (phase A);
                    # the status check covers legacy latch files.
                    if not isinstance(task.get("_terminalization_retry"), dict) and (existing_status in _TRULY_TERMINAL_STATUSES or existing_status == STATUS_CANCEL_REQUESTED):
                        skip_revival = True
                    elif not isinstance(task.get("_terminalization_retry"), dict):
                        try:
                            from ouroboros.cancel_intents import has_active_intent

                            if has_active_intent(
                                _queue().DRIVE_ROOT, str(task.get("id")), strict=True,
                            ):
                                # Cancellation custody owns it; never revive a pending row.
                                skip_revival = True
                        except Exception:
                            # This is an UNKNOWN cancel fact, not a terminal
                            # outcome. Restore the ordinary row under a durable,
                            # non-dispatchable hold; the pre-dispatch SSOT later
                            # resolves it after both authorities are readable.
                            task = dict(task)
                            task["_cancel_intent_authority_hold"] = {
                                "reason": "Cancel-intent authority is unreadable; dispatch is blocked.",
                                "held_at": utc_now_iso(),
                            }
                            cancel_authority_unreadable = True
                            log.debug("Snapshot restore cancel-intent authority check failed for %s",
                                      task.get("id"), exc_info=True)
                if skip_revival:
                    skipped_terminal += 1
                    continue
                if not cancel_authority_unreadable and str(task.get("parent_task_id") or "") and not _exact_pause_row(task) and _descends_from(
                    task, interrupted, pending_by_id
                ):
                    # #1104: a planned shutdown already refuses to start these
                    # (`kill_workers(preserve_pending=True)`); an unplanned one
                    # left them in the snapshot, where reviving one starts a child
                    # whose parent no longer exists. Everything above has just
                    # proved this row is otherwise revivable and unowned, so it
                    # takes the SAME shutdown-custody marker the planned path
                    # writes: the boot's own kill step settles it with a
                    # ledger-reconstructed cost and publishes its task_done. Rows
                    # without a parent — roots, schedules, evolution — never match;
                    # nor does an exact mid-run pause (#1196: saved work, not unstarted).
                    task = dict(task)
                    task["_terminalization_retry"] = {
                        "reason": "Parent task was interrupted before this child started.",
                        "status": STATUS_CANCELLED,
                        "trigger": "pending_parent_interrupted",
                        "reconcile_delegate_custody": True,
                    }
                    restore_terminalization_retry(
                        task, pending=_queue().PENDING, running=_queue().RUNNING,
                        queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF,
                        sort_pending=_queue().sort_pending,
                    )
                    orphan_children.append(str(task.get("id") or ""))
                    skipped_terminal += 1
                    continue
                if not cancel_authority_unreadable and (str(task.get("id") or "") in acceptance_held or task.get("_project_admission_restore_hold")):
                    # Proven revivable and unowned above. Both existing holds
                    # retain custody; assignment revalidates Project evidence.
                    _append_held_pending_row(task)
                    restored += 1
                    continue
                admitted = _queue().enqueue_task(task, restoring_snapshot=True)
                if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
                    _queue().restore_invalid_depth_admission(task, admitted, drive_root=_queue().DRIVE_ROOT, pending=_queue().PENDING, blocked=blocked_restore, terminalized=invalid_depth_restore, queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF)
                    try:
                        _queue().sort_pending()
                    except (TypeError, ValueError, OverflowError):
                        log.warning("Deferred snapshot sort failed; custody retained", exc_info=True)
                    continue
                if cancel_authority_unreadable:
                    cancel_authority_holds.append(str(task.get("id") or ""))
            restored += 1
        if skipped_fenced or acceptance_held:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restore_skipped_acceptance_fence",
                    "task_ids": skipped_fenced,
                    "held_budget_paused_task_ids": acceptance_held,
                    "root_task_ids": sorted(fenced_roots),
                },
            )
        _record_queue_restore(
            restored=restored, skipped_terminal=skipped_terminal,
            cancel_authority_holds=cancel_authority_holds,
            blocked_admission=blocked_restore, invalid_task_depth=invalid_depth_restore,
            terminalized_running=fenced_running, pending_parent_interrupted=orphan_children,
            direct_roots_incomplete=direct_roots.get("incomplete", False),
        )
        from supervisor.queue_transitions import sweep_orphaned_budget_fences

        sweep_orphaned_budget_fences(
            _queue().PENDING, _queue().BUDGET_ROOT_FENCES, _queue().DRIVE_ROOT,
        )
        if not unread:
            _RESTORED.add(key)
        if restored > 0 or skipped_terminal > 0 or invalid_depth_restore or blocked_restore or normalized:
            _queue().persist_queue_snapshot(reason="queue_restored")
        return restored
    except Exception:
        log.warning("Failed to restore pending queue from snapshot", exc_info=True)
        return 0
