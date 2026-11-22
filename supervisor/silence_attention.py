"""Unknown silence: a dated fact for the silent work and one waiting ancestor, never process custody.

The timeout pass owns observations under Q. The existing admission lane persists
and delivers them outside Q; the task document and ordinary queue own recovery.
An episode is answered only by the source's own progress. No host role is minted
for it (owner, 09.11.2026): the fact rides the task document and the published row,
whose work line says since when; without a waiting ancestor that reads its mail no
mind is told, on any install (ARCHITECTURE §5).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from functools import partial

from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC

log = logging.getLogger(__name__)
_publishing: set[tuple[str, str]] = set()


def observe(queue, task_id: str, meta: dict, now: float, threshold: float) -> None:
    """Q caller. A clock can request attention, never finalization."""
    progress = float(meta.get("last_progress_at") or meta["started_at"])
    attempt = int(meta.get("attempt") or (meta.get("task") or {}).get("_attempt") or 1)
    identity = hashlib.sha256(json.dumps([task_id, attempt, meta["started_at"], progress]).encode()).hexdigest()[:24]
    fact = meta.get("silence_attention") or {}
    if fact.get("episode_id") != identity:
        fact = {"episode_id": identity, "task_id": task_id, "task_attempt": attempt,
                "last_progress_at": progress, "observed_at": now, "threshold_sec": threshold,
                "last_heartbeat_at": meta.get("last_heartbeat_at"), "state": "pending_attention"}
        meta["silence_attention"] = fact
    _submit(queue, task_id, meta)


def _submit(queue, task_id, meta):
    from supervisor.workspace_preparation import ADMISSION_LANE, current_generation

    fact = meta["silence_attention"]
    now = time.time()
    parent = _waiting_parent(queue, meta["task"], now)
    new_parent = bool(parent and not fact.get("parent_delivered")
                      and fact.get("recipient_task_id") in (None, "", parent[0]))
    if meta.get("_silence_published") == fact and fact.get("self_delivered") and not new_parent:
        return
    # Failed/held publication retries on the existing maintenance cadence. Source
    # progress or a new waiter is checked in memory.
    retry = meta.get("_silence_retry")
    if retry and retry[0] == fact and not new_parent and now < retry[1]:
        return
    key = (str(queue.DRIVE_ROOT), fact["episode_id"])
    generation = current_generation()
    if key in _publishing or generation is None or generation.launches_closed():
        return
    _publishing.add(key)
    meta["_silence_retry"] = (copy.deepcopy(fact), now + BODY_MAINTENANCE_INTERVAL_SEC)
    if not generation.submit(partial(_publish, queue, task_id, meta, key), ADMISSION_LANE):
        _publishing.discard(key)
        meta.pop("_silence_retry", None)


def recovered(queue, task_id: str, meta: dict, now: float) -> None:
    """Only real source progress ends an episode; borrowed activity does not."""
    fact = meta.get("silence_attention") or {}
    if fact and fact.get("state") != "progress_observed" and float(meta.get("last_progress_at") or 0) > fact["last_progress_at"]:
        fact.update(state="progress_observed", recovered_at=now)
    if fact and fact.get("state") == "progress_observed" and meta.get("_silence_published") != fact:
        _submit(queue, task_id, meta)


def _waiting_parent(queue, task: dict, now: float) -> tuple[str, dict, str] | None:
    parent = str(task.get("parent_task_id") or "")
    seen = set()
    responsibility = {str(task.get("id") or "")}
    while parent and parent not in seen:
        seen.add(parent)
        meta = _parent_meta(queue, parent)
        if not meta:
            return None
        selected = next((tid for row in (meta.get("active_operation_leases") or {}).values()
                      if isinstance(row, dict) and row.get("kind") == "in_slot_wait"
                      and float(row.get("until_ts") or 0) > now
                      for tid in row.get("wait_for_tasks") or () if tid in responsibility), "")
        selected = next((tid for tid in _parent_sleep(meta).get("tasks") or () if tid in responsibility), selected)
        if selected and _available_parent(meta):
            return parent, meta, selected
        responsibility.add(parent)
        parent = str((meta.get("task") or {}).get("parent_task_id") or "")
    return None


def _parent_meta(queue, task_id):
    live = queue.RUNNING.get(task_id)
    if isinstance(live, dict):
        return live
    parked = next((row for row in queue.PENDING if row.get("id") == task_id), None)
    return {"task": parked, "attempt": parked.get("_attempt", 1)} if parked else {}


def _parent_sleep(meta):
    """Only this attempt's model-selected sleep, never an owner-answer park."""
    from supervisor.worker_owner_wait import waiting_on_owner

    if waiting_on_owner(meta, int(meta.get("attempt") or 1)):
        pending = meta.get("owner_wait_park_pending") or {}
        wait = ((pending.get("event") or {}).get("checkpoint") if pending else meta.get("owner_wait")) or {}
    else:
        wait = (meta.get("task") or {}).get("_budget_pause") or {}
    chosen = wait.get("sleep")
    return chosen if wait.get("reason") == "sleep" and isinstance(chosen, dict) else {}


def _available_parent(meta):
    from supervisor.worker_owner_wait import waiting_on_owner
    from ouroboros.work_stop import current_stop

    return bool(meta and (not waiting_on_owner(meta, int(meta.get("attempt") or 1)) or _parent_sleep(meta))
                and not current_stop(meta) and not current_stop(meta.get("task")))


def _mail_text(fact) -> str:
    """The dated fact in words first, so the 200-character row a reader sees already says since when."""
    from datetime import datetime, timezone

    since = datetime.fromtimestamp(float(fact["last_progress_at"]), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"No recognized progress or active wait since {since} (idle window {int(fact['threshold_sec'])} s; "
            f"silence episode {fact['episode_id']}). This is not a stop request: a heartbeat is not progress, and "
            "silence does not prove a dead worker. Inspect the work and choose waiting or an existing control; "
            "nothing was killed, retried or replaced.")


def _save(queue, task_id, fact):
    from ouroboros.task_store import patch_document

    def update(current):
        if not current or current.get("silence_attention") == fact:
            return None
        return {**current, "silence_attention": copy.deepcopy(fact)}

    patch_document(queue.DRIVE_ROOT, task_id, update)


def _publish(queue, task_id, meta, key) -> None:
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_task_result

    try:
        with queue._queue_lock:
            if queue.RUNNING.get(task_id) is not meta or meta["silence_attention"]["episode_id"] != key[1]:
                return
            fact = copy.deepcopy(meta["silence_attention"])
            task = copy.deepcopy(meta["task"])
        stored = (load_task_result(queue.DRIVE_ROOT, task_id, strict=True) or {}).get("silence_attention") or {}
        if stored.get("episode_id") == fact["episode_id"]:
            fact = {**stored, **fact, **{k: stored[k] for k in
                    ("self_delivered", "recipient_task_id", "parent_delivered") if k in stored}}
        if fact != stored:
            _save(queue, task_id, fact)  # the observation precedes every delivery
        saved = copy.deepcopy(fact)
        text = _mail_text(fact)
        if not fact.get("self_delivered"):
            fact["self_delivered"] = write_task_message(queue._task_drive_for_task(task, task_id), text,
                task_id, source_task_id="", provenance="system", msg_id="silence:" + fact["episode_id"])
        with queue._queue_lock:
            current = meta.get("silence_attention") or {}
            if queue.RUNNING.get(task_id) is not meta or current.get("episode_id") != fact["episode_id"]:
                return
            if current.get("state") == "progress_observed":
                fact.update(state="progress_observed", recovered_at=current["recovered_at"])
            parent = _waiting_parent(queue, task, time.time())
        if fact["state"] != "progress_observed":
            if parent and not fact.get("recipient_task_id"):
                fact["recipient_task_id"] = parent[0]
                _save(queue, task_id, fact)  # fix the recipient before a possibly ambiguous delivery
            if parent and parent[0] == fact.get("recipient_task_id") and not fact.get("parent_delivered"):
                parent_id, parent_meta, waited_task = parent
                if write_task_message(queue._task_drive_for_task(parent_meta["task"], parent_id), text,
                        parent_id, source_task_id=task_id, provenance="system",
                        work_attention={"kind": "silence", "task_id": task_id,
                                        "waiting_for_task_id": waited_task, "episode_id": fact["episode_id"]},
                        msg_id="silence-parent:" + fact["episode_id"]):
                    fact["parent_delivered"] = True
        if fact != saved:
            _save(queue, task_id, fact)
        published = copy.deepcopy(fact)
        with queue._queue_lock:
            if queue.RUNNING.get(task_id) is meta and (meta.get("silence_attention") or {}).get("episode_id") == fact["episode_id"]:
                # Progress that raced persistence wins in memory and is published next pass.
                fact.update({k: v for k, v in meta["silence_attention"].items()
                             if k in {"state", "recovered_at"}})
                meta["silence_attention"] = fact
                meta["_silence_published"] = published
                meta["_silence_retry"] = (published, time.time() + BODY_MAINTENANCE_INTERVAL_SEC)
    except (OSError, ValueError, StoreError) as exc:
        log.warning("Silence attention remains pending for %s: %s", task_id, exc)
    finally:
        with queue._queue_lock:
            _publishing.discard(key)
