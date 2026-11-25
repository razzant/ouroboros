"""Identity of a positively parked continuation; a hard clock adds no control authority."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
import time
from typing import Any

from ouroboros.budget_pause import STATE_PAUSED, budget_pause_row
from ouroboros.deadline_utils import task_deadline
from ouroboros.work_stop import control_identity, hold_fact, pause_marker

log = logging.getLogger(__name__)


def parked_binding(drive_root: Any, row: dict, view: dict, *, late_phase: bool = False) -> dict:
    """Bind a settled park to its physical attempt, not to prose, a paused label, or a Resume refusal.

    The caller observes PENDING (or a retained answered root) and checks ownership before acting.
    Keep this binding independent of time: a root's Close binds its parked descendants even if
    they have no deadline of their own. A released or accepted handoff is not this park.
    """
    task_id, attempt = str(row.get("id") or row.get("task_id") or ""), row.get("_attempt", row.get("task_attempt", 1))
    stop = view.get("stop") or {}
    if (not task_id or type(attempt) is not int or attempt < 1 or not stop or view.get("waiting") or view.get("refusal")
            or accepted_handoff(row) or any(row.get(key) for key in (
                "_owner_hold", "_continuation_prepared", "_terminalization_retry", "_cancel_intent_authority_hold"))):
        return {}
    identity = control_identity(view)
    if late_phase:
        if stop.get("scope") != "tree" or stop.get("cause") != "pause" or not _settled(stop):
            return {}
        point = {}
    else:
        point = budget_pause_row(Path(row.get("budget_drive_root") or drive_root), task_id)
        marker, hold = pause_marker(row), hold_fact(row) or {}
        if point.get("state") == "resume_granted":
            return {}
        if marker.get("exact_continuation"):
            checkpoint = marker.get("checkpoint") or {}
            if (point.get("state") != STATE_PAUSED or point.get("task_attempt") != attempt
                    or not point.get("source_ref") or not point.get("pause_id")
                    or point["pause_id"] != checkpoint.get("pause_id")
                    or point["source_ref"] != checkpoint.get("source_ref")):
                return {}
        elif not (hold.get("held_at") or (marker.get("status") == "paused_before_dispatch"
                  and marker.get("replay_safe") is True and type(marker.get("physical_calls")) is int
                  and marker["physical_calls"] == 0)
                  or (stop.get("scope") == "tree" and _settled(stop) and stop.get("fence_id")
                      and row.get("admitted_dispatch") == "none")):
            return {}
    if not identity.keys() & {"fence_id", "pause_id", "held_at", "paused_at"}:
        return {}
    return {"task_id": task_id, "attempt": attempt, "parent_task_id": str(row.get("parent_task_id") or ""),
            "root_task_id": str(row.get("root_task_id") or task_id), "control": identity,
            "member_control": control_identity({"stop": stop.get("member_stop")}),
            "pause_id": str(point.get("pause_id") or ""), "late_phase": late_phase}


def _settled(stop: dict) -> bool:
    """An owner fence says ``state``, a root spending fence ``status`` (``work_stop.current_stop``)."""
    return stop.get("state") == "paused" or stop.get("status") == "paused"


def accepted_handoff(row: dict) -> bool:
    """An accepted Resume or selection carries the row on; under a standing pause marker a newer Pause spent it."""
    from ouroboros.work_stop import SELECTION_KEY

    return not pause_marker(row) and (isinstance(row.get("_budget_pause_resume"), dict)
                                      or isinstance(row.get(SELECTION_KEY), dict))


def park_limit(row: dict, binding: dict) -> float | None:
    """The hard calendar limit this bound park observes (an answered root's late phase reads its own carrier)."""
    deadline = task_deadline(row, late_phase=binding["late_phase"]) if binding else None
    return deadline.timestamp() if deadline is not None else None


def deadline_offer(row: dict, binding: dict, now: float, members: list | tuple = ()) -> tuple[float | None, dict]:
    """The observed limit and the exact Close offer, if it is now due. The token binds the park, its limit and the
    frozen bindings of the members a root's Close takes, so a retry never reaches a member that parked later."""
    limit = park_limit(row, binding)
    if limit is None or now < limit:
        return limit, {}
    token = hashlib.sha256(json.dumps([binding, limit, list(members)], sort_keys=True).encode()).hexdigest()
    return limit, {"action": "close", "target": binding["task_id"], "close_expired": token,
                   "scope": "single" if binding["parent_task_id"] else "cascade"}


def expired_park(row: dict) -> bool:
    """A held row past its own hard deadline: every Resume refuses ``deadline_passed`` (``task_deadline``, bound or
    held alike), so the mind's roster hides that dead Resume; only a bound reminder supplies a Close token."""
    try:
        deadline = task_deadline(row)
        return deadline is not None and time.time() >= deadline.timestamp()
    except (TypeError, ValueError):
        return False


def close_withheld(binding: dict, live: list, custody: dict | None = None, *, owners: Any = None,
                   parent_of: Any = None) -> bool:
    """Whether work its Close would not take still stands under an expired park, so no Close token is offered.

    ``live``: ``(row, running)`` for every queue row; a helper with a live descendant is withheld (Close never alters
    it). The chain from a live row climbs its queue parents and, past an ended intermediate, ``parent_of`` (the
    durable parent id); an unreadable link raises, which every caller reads as withheld. ``custody``
    (``delegate_custody_current.tree_closure`` of the tree) adds pending invocations (a helper: its own) and
    unsettled runs: one passes only when its owner is the helper itself or in ``owners`` (a root and its frozen
    members) and its custody observed a terminal state, so only settlement is owed (``delegate_continuation``'s
    same test). A run that may still be live, or is owned elsewhere, withholds. Ordinary Stop stays the way out."""
    from ouroboros.delegate_custody import TERMINAL_STATES

    target = binding["task_id"]
    helper = bool(binding["parent_task_id"])
    if helper and live:
        parents = {str(row.get("id")): str(row.get("parent_task_id") or "") for row, _ in live if row.get("id")}
        tree = binding["root_task_id"]

        def under(task_id: str) -> bool:
            seen = set()
            while task_id and task_id not in seen and task_id != tree and len(seen) < 64:
                seen.add(task_id)
                task_id = parents[task_id] if task_id in parents else (parent_of(task_id) if parent_of else "")
                if task_id == target:
                    return True
            return len(seen) >= 64  # a chain this deep is not proven outside the target
        if any(str(row.get("id")) != target and str(row.get("root_task_id") or "") == tree and under(str(row.get("id")))
               for row, _ in live):
            return True
    if custody is None:
        return False
    owned = {target} if helper else set(owners or ())  # a late root passes none: any unsettled run holds
    if any(not helper or str(row.get("task_id") or "") == target for row in custody.get("pending") or ()):
        return True
    return any(not run.settled and (not helper or run.task_id == target)
               and (run.task_id not in owned or run.terminal_state not in TERMINAL_STATES)
               for run in (custody.get("unfinished") or {}).values())


def valid_token(token: Any) -> bool:
    return isinstance(token, str) and re.fullmatch(r"[0-9a-f]{64}", token) is not None


def record_parent_close(drive_root: Any, task_id: str, token: Any, *, requester: str, stop_action_id: str = "") -> tuple:
    """A direct parent's ``cancel_task(close_expired=...)``: durable before acknowledged, as ``(words, recorded)``.

    ``requester`` is the caller only when the target is its own direct child; the request carries that typed actor
    (``request_origin``) as every agent Stop does. While Close owns the request the supervisor binds the exact park
    or withdraws it, and mails that parent the one outcome before the request retires (closed, already ended with
    its own result kept, or changed), once per request whatever replays (``expired_park_close._owed``). A separate
    ordinary Stop that replaces it never waits for that mail: the child's task record is the outcome then."""
    from ouroboros.cancel_intents import PARKED_CLOSE_SOURCE, request_cancel
    from ouroboros.utils import utc_now_iso

    if token == "":
        return ("Close refused: close_expired was empty. Omit it for an ordinary Stop, or pass the exact offer "
                "from the deadline reminder."), False
    if not requester or not valid_token(token) or stop_action_id:
        return "Close refused: use the exact expired offer for your own direct child, without a Stop action id.", False
    try:
        intent = request_cancel(drive_root, task_id, reason="the deadline expired", source=PARKED_CLOSE_SOURCE,
                                requested_by=requester, parked_close={"requested_by": requester, "token": token},
                                observation={"observed_at": utc_now_iso(), "requested_task_id": task_id,
                                             "observed_task_id": task_id,
                                             "request_origin": {"kind": "agent_task", "task_id": requester}})
    except Exception:
        log.debug("Close of %s was not recorded", task_id, exc_info=True)
        return f"Close of {task_id} was not recorded; nothing changed. Retry the same call.", False
    if intent.get("already_settled"):
        return f"Nothing to close: {task_id} had already ended. Its result is kept.", False
    if intent.get("close_conflict"):
        return f"Close of {task_id} was not recorded: another Stop or Close of it is still open. Wait for its outcome.", False
    if intent.get("already_requested") and (intent.get("parked_close") or {}).get("requested_by") != requester:
        # The owner's open Close of this exact park already stands: nothing is recorded for you, no outcome is mailed.
        return (f"Close of {task_id} is already requested by the owner for this exact saved park; your call added "
                f"nothing and no separate outcome message will follow. Check {task_id}'s status for the result."), False
    again = " (the same request was already recorded)" if intent.get("already_requested") else ""
    return (f"Close of {task_id} recorded{again}. The supervisor rechecks that exact saved park and sends you one "
            "outcome: closed, already ended by itself with its own result kept, or not closed because the park "
            f"changed. If a separate Stop of {task_id} takes over first, no Close outcome follows; its task "
            "record is the result."), True
