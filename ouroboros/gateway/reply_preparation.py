"""Which chats are owed a reply that is being actively prepared now (DESIGN §9, Telegram typing).

An in-process current-state projection for chat transports, read through the
activity census and the live owners it already consults; it stores, schedules
and publishes nothing. A ROOT counts when the owner's door expects its reply
(``active_activity.reply_expected``) and its work is active. A task's work is
fenced by Pause (the owner's or the budget's), an immediate Stop, being queued
or finalizing, and its own answer having entered the delivery path
(``reply_handed_off``, a stored answer or result). Otherwise it is active when
its own model rounds and tools run, or, while it waits (an owner question, model
access, a sleep, a slot held on mail, task terminals, a delegated run or a
review panel), when what the wait names works: a held run the host currently
observes running (``delegate_supervision.observed_run_activity``), a run a warm
or cold sleep selected that the body observer currently sees running
(``delegate_collection.current_activity``), a panel seat not waiting for model
access (``review_pause.panel_activity``), or a live descendant active by the
same rules. Children contribute only through their
root and never name a chat. Unknown facts make no positive and never cancel
another root's known one.

The destination is the one the root's answer takes: its project binding, else
its own chat, with that chat's current reply transport; the transport applies
its own mirror policy to it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Census phases in which a task is running its own work (direct turns and RUNNING rows).
_RUNNING_PHASES = frozenset({"preparing", "working"})
_SLOT_WAIT = "in_slot_wait:"


def _merge(results: List[Optional[bool]]) -> Optional[bool]:
    """Any known positive wins; otherwise an unknown stays unknown."""
    return True if True in results else None if None in results else False


def _sleep_runs(sleep: Any, receipt: Any = None, transfer: Any = None) -> Dict[str, Any]:
    """The delegated runs a model sleep SELECTED, each with the exact identity it is held by
    now: ``(run_id, "")`` while the actor holds it (warm, ``transfer`` None), the body's
    ``(run_id, invocation_id)`` pair once a cold park transferred it. A selected run outside the
    transfer is settled (``unheld``); a recorded transfer without a valid pair list is unknown."""
    selected = [str(run_id) for run_id in (sleep.get("runs") or []) if run_id] if isinstance(sleep, dict) else []
    if transfer is None:
        return {"runs": [(run_id, "") for run_id in selected], "unheld": 0, "unknown": False}
    pairs = {str(pair[0]): str(pair[1]) for pair in (receipt or {}).get("runs") or []
             if isinstance(pair, (list, tuple)) and len(pair) == 2}
    return {"runs": [(run_id, pairs[run_id]) for run_id in selected if run_id in pairs],
            "unheld": sum(1 for run_id in selected if run_id not in pairs),
            "unknown": bool(transfer) and not receipt}


def _owner_facts(drive_root: Any, task_ids: set, now: float) -> Dict[str, Dict[str, Any]]:
    """Per live task: root expectation, answer handed off, attempt, the in-slot waits it
    holds (``kind:target`` keys), an owner-wait park's reason, and a pooled task's live
    review seats. ``unknown`` marks a direct turn whose own facts could not be read."""
    from ouroboros.gateway.state import _task_activity_facts
    from ouroboros.sleep_custody import carrier_receipt
    from supervisor import queue as queue_mod
    from supervisor.active_activity import get_direct_activity_registry, reply_expected
    from supervisor.worker_owner_wait import waiting_on_owner

    facts: Dict[str, Dict[str, Any]] = {}
    with queue_mod._queue_lock:
        for task_id, meta in queue_mod.RUNNING.items():
            if str(task_id) not in task_ids or not isinstance(meta, dict):
                continue
            task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
            attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
            leases = meta.get("active_operation_leases")
            rows = [(str(key), row) for key, row in (list(leases.items()) if isinstance(leases, dict) else ())
                    if isinstance(row, dict) and float(row.get("until_ts") or 0) > now
                    and row.get("task_attempt") in (None, attempt)]
            pending = (meta.get("owner_wait_park_pending") or {}).get("event") or {}
            wait = (pending.get("checkpoint") if pending else meta.get("owner_wait")) or {}
            park = str(wait.get("reason") or "owner") if waiting_on_owner(meta, attempt) else ""
            facts[str(task_id)] = {
                "expected": reply_expected(task), "handed_off": bool(meta.get("reply_handed_off")), "attempt": attempt,
                "logical_root": str(task.get("root_task_id") or task_id),
                "root": str(task.get("budget_drive_root") or drive_root),
                "waits": [key[len(_SLOT_WAIT):] for key, row in rows if row.get("kind") == "in_slot_wait"
                          and key.startswith(_SLOT_WAIT)],
                "park": park, "sleep": _sleep_runs(wait.get("sleep")) if park == "sleep" else None,
                "seats": {(str(row.get("owner_id") or ""), str(row.get("slot_id") or ""))
                          for _key, row in rows if row.get("kind") == "review"}}
        for task in queue_mod.PENDING:
            task_id = str(task.get("id") or "") if isinstance(task, dict) else ""
            if task_id in task_ids:
                # A cold sleeper's runs are the body's by its exact accepted receipt.
                marker = task.get("_budget_pause") if isinstance(task.get("_budget_pause"), dict) else {}
                sleep = (_sleep_runs(marker["sleep"], carrier_receipt(task), marker.get("external_custody") or {})
                         if marker.get("reason") == "sleep" and isinstance(marker.get("sleep"), dict) else None)
                facts.setdefault(task_id, {"expected": reply_expected(task), "handed_off": False, "waits": [],
                                           "logical_root": str(task.get("root_task_id") or task_id),
                                           "park": "", "attempt": int(task.get("_attempt") or 1), "seats": set(),
                                           "sleep": sleep})
    for entry in get_direct_activity_registry().actors():
        if entry.activity_id not in task_ids:
            continue
        fact: Dict[str, Any] = {"expected": entry.reply_expected, "handed_off": entry.reply_handed_off,
                                "local": True, "waits": [], "park": "", "sleep": None}
        sleep = None
        try:
            context = getattr(getattr(entry.actor, "tools", None), "_ctx", None)
            marks = getattr(context, "_in_slot_waits", None)
            fact["waits"] = [key for key, until in list(marks.items()) if until > now] if isinstance(marks, dict) else []
            # A question or sleep requested in this tool batch parks at its end: it already waits.
            fact["park"] = "requested" if getattr(context, "_owner_wait_requested", "") else ""
            fact["attempt"] = int(getattr(context, "task_attempt", None) or 1)
            fact["logical_root"] = str(getattr(context, "root_task_id", None) or entry.activity_id)
            sleep = getattr(context, "_model_sleep", None)  # the armed sleep its live actor parks warm on
        except Exception:
            fact["unknown"] = True
        stored = _task_activity_facts(drive_root, entry.activity_id)
        wait = stored.get("owner_wait") or {}
        if stored.get("unavailable"):
            fact["unknown"] = True
        elif wait.get("state") == "waiting":
            fact["park"] = str(wait.get("reason") or "owner")
        if fact["park"] in {"requested", "sleep"} and isinstance(sleep, dict) and sleep.get("mode") == "warm":
            fact["sleep"] = _sleep_runs(sleep)
        facts[entry.activity_id] = fact
    return facts


def _stopped(drive_root: Any) -> Optional[set]:
    """Tasks an IMMEDIATE Stop is ending (a cascade covers its root's tree): nothing they
    hold prepares a reply. A graceful Stop's one summary answer still is. None: unreadable."""
    from ouroboros.cancel_intents import SCOPE_CASCADE, STOP_POLICY_FINALIZE, active_intents, stop_policy

    try:
        intents = active_intents(drive_root, strict=True)
    except Exception:
        log.debug("cancel intents unreadable for reply preparation", exc_info=True)
        return None
    return {key for task_id, row in intents.items() if stop_policy(row) != STOP_POLICY_FINALIZE
            for key in ((task_id, f"tree:{task_id}") if row.get("scope") == SCOPE_CASCADE else (task_id,))}


def _awaited_reply_live() -> bool:
    """Cheap gate before the census: some live task or turn may carry an awaited, undelivered reply."""
    from supervisor import queue as queue_mod
    from supervisor.active_activity import get_direct_activity_registry, reply_expected

    if any(entry.reply_expected and not entry.reply_handed_off for entry in get_direct_activity_registry().actors()):
        return True
    with queue_mod._queue_lock:
        running = [meta for meta in queue_mod.RUNNING.values() if isinstance(meta, dict)]
        return (any(not meta.get("reply_handed_off") and reply_expected(meta.get("task")) for meta in running)
                or any(reply_expected(task) for task in queue_mod.PENDING))


def reply_preparation(drive_root: Any) -> Dict[str, Any]:
    """``{"complete": bool, "chats": [{"chat_id", "transport"}], "roots": [task_id]}`` for this instant.

    ``complete`` is False when a fact could not be read; the listed chats are the
    known positives either way. Without any awaited reply alive the census is not read.
    """
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.delegate_collection import current_activity
    from ouroboros.delegate_supervision import observed_run_activity
    from ouroboros.sleep_custody import accepted_receipt
    from ouroboros.gateway.state import _answer_stored, _chat_activities_snapshot_safe, _direct_turns_snapshot_safe
    from ouroboros.gateway.state import _task_activity_facts
    from ouroboros.review_pause import local_panel_seats, panel_activity
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.work_stop import owner_fence_state
    from supervisor.log_addressing import resolve_project_chat
    from supervisor.message_bus import notification_chat_route, try_get_bridge

    try:
        if not _awaited_reply_live():
            return {"complete": True, "chats": [], "roots": []}
    except Exception:
        log.debug("reply preparation gate unavailable; reading the census", exc_info=True)
    availability = {"complete": True}
    owners: List[Dict[str, Any]] = []
    direct_rows = _direct_turns_snapshot_safe(availability=availability)
    _chat_activities_snapshot_safe(drive_root, None, direct_turns=direct_rows, availability=availability,
                                   decision_owners=owners, question_details=False)
    rows = {str(row.get("activity_id") or ""): row for row in owners if row.get("activity_id")}
    children: Dict[str, List[str]] = {}
    for task_id, row in rows.items():
        parent = str(row.get("parent_task_id") or "")
        if parent and parent != task_id:
            children.setdefault(parent, []).append(task_id)
    try:
        facts = _owner_facts(drive_root, set(rows), time.time())
    except Exception:
        log.debug("reply preparation owner facts unavailable", exc_info=True)
        return {"complete": False, "chats": [], "roots": []}
    stopped = _stopped(drive_root)
    # The public root census intentionally omits lineage; its control/delivery readers still need it.
    for task_id, fact in facts.items():
        if task_id in rows and fact.get("logical_root"):
            rows[task_id] = {**rows[task_id], "root_task_id": fact["logical_root"]}

    def sleep_of(task_id: str, row: Dict[str, Any], fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The task's model sleep, if any; a pooled cold sleep still releasing its process reads its
        own durable pause row (the census read that row for this phase), never a queue guess."""
        if fact.get("sleep") is not None or fact.get("local") or row.get("phase") != "budget_pausing":
            return fact.get("sleep")
        pause = budget_pause_row(fact.get("root") or drive_root, task_id)
        transfer = pause.get("external_custody") or {}
        if (pause.get("reason") != "sleep" or not isinstance(pause.get("sleep"), dict)
                or int(pause.get("task_attempt") or 0) != int(fact.get("attempt") or 0)):
            return None
        proposed = transfer if transfer.get("state") == "requested" and all(
            (transfer.get("binding") or {}).get(key) == pause.get(key) for key in ("pause_id", "task_attempt")) else {}
        return _sleep_runs(pause["sleep"], accepted_receipt(task_id, pause) or proposed, transfer)

    def fenced(task_id: str, row: Dict[str, Any], fact: Dict[str, Any]) -> Optional[bool]:
        phase = str(row.get("phase") or "")
        if (phase in {"queued", "finalizing"} or fact.get("handed_off") or str(row.get("status") or "") in SETTLED_STATUSES
                or _answer_stored((row.get("root_phase_checkpoint") or {}).get("post_task_synthesis"))):
            return True
        # A retained sleep describes dependencies, never permission to ignore a later control.
        owner_fence = owner_fence_state(drive_root, row, task_id)
        if owner_fence == "unknown":
            return None
        if owner_fence in {"requested", "paused"}:
            return True
        if phase in {"budget_paused", "budget_pausing"} and row.get("pause_cause") != "sleep":
            # The releasing window can lack a census cause; its current sleep supplies it.
            if row.get("pause_cause") not in {None, "unknown"} or sleep_of(task_id, row, fact) is None:
                return True
        tree = str(row.get("root_task_id") or task_id)
        return None if stopped is None else task_id in stopped or f"tree:{tree}" in stopped

    def own(row: Dict[str, Any], fact: Dict[str, Any]) -> Optional[bool]:
        """The task's OWN work runs now: True, False (waiting or not running), None (unknown)."""
        phase = str(row.get("phase") or "")
        if phase == "unknown" or fact.get("unknown"):
            return None
        if phase not in _RUNNING_PHASES or fact.get("waits") or fact.get("park"):
            return False
        attempt = int(row.get("task_attempt") or 1)
        # The author's own model-access wait; a reviewer's row (``review_operation``) is its panel's.
        return not any(isinstance(wait, dict) and wait.get("state") == "waiting" and wait.get("task_attempt") == attempt
                       and not wait.get("review_operation") for wait in (row.get("model_waits") or {}).values())

    def seats(task_id: str, row: Dict[str, Any], fact: Dict[str, Any]) -> tuple:
        if fact.get("local"):
            return local_panel_seats(task_id)
        attempt = int(row.get("task_attempt") or 1)
        waiting = {(str(block.get("owner_id") or ""), str(block.get("slot_id") or ""))
                   for wait in (row.get("model_waits") or {}).values() if isinstance(wait, dict)
                   and isinstance(block := wait.get("review_operation"), dict) and wait.get("state") == "waiting"
                   and wait.get("task_attempt") == attempt and wait.get("model_wait_owner_id") == block.get("owner_id")}
        return fact.get("seats") or set(), waiting

    def named(task_id: str, row: Dict[str, Any], fact: Dict[str, Any]) -> Optional[bool]:
        """What the task's waits name: its held delegated run, its drained or left review panel."""
        results: List[Optional[bool]] = []
        for key in fact.get("waits") or ():
            kind, _, target = str(key).partition(":")
            if kind == "delegate":
                results.append(observed_run_activity(drive_root, task_id, target, fact.get("attempt")))
            elif kind == "review":
                results.append(panel_activity(*seats(task_id, row, fact), owner_id=target.split(":", 1)[0]))
        if fact.get("park") == "review":
            results.append(panel_activity(*seats(task_id, row, fact)))
        sleep = sleep_of(task_id, row, fact)
        if sleep is not None:
            results.extend([None] if sleep["unknown"] else [False] * sleep["unheld"])
            results.extend(current_activity(drive_root, task_id, run_id, invocation_id)
                           for run_id, invocation_id in sleep["runs"])
        return _merge(results)

    def active(task_id: str, seen: set) -> Optional[bool]:
        row, fact = rows[task_id], facts.get(task_id) or {}
        fence = fenced(task_id, row, fact)
        if fence is not False:
            return None if fence is None else False
        result = own(row, fact)
        if result is False:
            results = [named(task_id, row, fact)]
            for child_id in children.get(task_id, ()):
                if True in results:
                    break
                if child_id not in seen:
                    seen.add(child_id)
                    results.append(active(child_id, seen))
            result = _merge(results)
        # A positive needs the task's own answer facts readable: unreadable is never "not answered yet".
        return None if result is True and _task_activity_facts(drive_root, task_id).get("unavailable") else result

    bridge = try_get_bridge()
    chats: Dict[int, Dict[str, Any]] = {}
    roots: List[str] = []
    for task_id, row in rows.items():
        if row.get("parent_task_id") or not (facts.get(task_id) or {}).get("expected"):
            continue
        try:
            state = active(task_id, {task_id})
        except Exception:
            log.debug("reply preparation unavailable for %s", task_id, exc_info=True)
            state = None
        if state is None:
            availability["complete"] = False
        if state is not True:
            continue
        bound = resolve_project_chat(drive_root, task_id, row.get("parent_task_id"), row.get("root_task_id"))
        chat = notification_chat_route(bound or None, row.get("chat_id"))
        if chat is None:
            continue
        roots.append(task_id)
        chats.setdefault(chat, {"chat_id": chat, "transport": bridge.chat_transport(chat) if bridge else {}})
    return {"complete": availability["complete"], "chats": list(chats.values()), "roots": sorted(roots)}


__all__ = ["reply_preparation"]
