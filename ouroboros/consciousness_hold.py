"""What holds my own held wake, proven from its records, and how it is said: the cause (``hold_cause`` over its pause
row and ``held_wake_cause`` over its stores), the room a Resume would add (``_resume_room``) and the host's words for
it (``held_sentence``). Read-only; the clock's snapshot carries the result as ``live_wake_hold`` and
``consciousness_view.describe_consciousness_state`` says it."""

from datetime import datetime, timezone
import logging
from typing import Any

log = logging.getLogger(__name__)


# A held wake the allowance it started with stopped: the owner's move (DESIGN "Status marks"), said in Costs too.
ALLOWANCE_HOLDS = frozenset({"allowance", "allowance_and_cap"})
_STAMPED_CAUSES = {"allowance": "allowance", "both": "allowance_and_cap", "per_task_cap": "per_task_cap"}
# What holds my own held wake and what follows, by the cause its records prove (``hold_cause``), for the held wake's
# block (DESIGN "Dashboard › Consciousness") and every /bg reply. The wake goes without its id and without a clock: the
# block names it with its start in the viewer's zone. It holds back only my next OWN wake (``tick`` answers
# ``wake_live``); wakes in places are separate, so no sentence says nothing wakes. ``{limit}``: the proven number.
_FREES = "Stop now ends it and frees the clock for its next check."
_NEXT_WAITS = "The next own wake waits until it ends."
HELD_WORDS = {
    "owner": ("Its own wake-up is paused; its card shows who paused it.",
              "The next own wake waits until it is resumed or stopped."),
    # The setting reaches only work that starts after it: this work's own cap is raised conversationally (pause_notices).
    "per_task_cap": ("Its own wake-up is paused at the per-task cost cap it started with{limit}.",
                     f"Changing the setting in Costs does not raise it: raise this work's cap in chat, then Resume. {_FREES}"),
    "launch_ceiling": ("Its own wake-up is paused at the spending limit it started with{limit}.",
                       f"Raising a limit alone resumes nothing; Resume checks what is left. {_FREES}"),
    "total_budget": ("Its own wake-up is paused: the total budget was spent.",
                     f"Raising Total budget in Costs alone resumes nothing: raise it, then Resume. {_FREES}"),
    "per_task_fence": ("Its own wake-up is paused at a spending limit of its own work; its card shows which.",
                       f"Resume once that limit has room. {_FREES}"),
    "sleep": ("Its own wake-up sleeps until what it waits for happens, then goes on by itself; its card shows what.",
              _NEXT_WAITS),
    "technical": ("Its own wake-up is paused: a record it needs could not be read; its card shows which.",
                  f"It goes on by itself once that record can be read. {_NEXT_WAITS}"),
    "queued": ("Its own wake-up waits in the queue for a worker and starts by itself.", ""),
    "": ("Its own wake-up is paused or waiting in the queue; its card shows why.", _NEXT_WAITS),
}


def held_sentence(hold: Any) -> tuple:
    """``(detail, note)`` of my held own wake: what holds it, its proven cause (``hold_cause``) in ``HELD_WORDS``, and
    what follows from it. No clock: when room returns is Costs' forecast, in one zone."""
    hold = hold if isinstance(hold, dict) else {}
    cause, ceiling = hold.get("cause"), hold.get("ceiling_usd")
    from ouroboros.cost_projection import usd_text

    # The ceiling it stopped at, which an earlier Resume may have moved: said as the number it is, never "at start".
    facts = [usd_text(float(ceiling))] if type(ceiling) in (int, float) else []
    facts += ["the per-task cap too"] if cause == "allowance_and_cap" else []
    limit = f" ({', '.join(facts)})" if facts else ""
    if cause in ALLOWANCE_HOLDS:
        room = ("Resume checks what is left now; if there is room, it continues this wake without resetting its spend."
                if "resume_additional_usd" in hold
                else "No spending estimate is available for Resume. Raising the limit alone resumes nothing.")
        return f"Its own wake-up stopped at its allowance{limit}.", f"{room} {_FREES}"
    detail, note = HELD_WORDS.get(cause) or HELD_WORDS[""]
    return detail.format(limit=limit), note


def hold_cause(metadata: dict, own: dict, members=()) -> dict:
    """``{"cause", "ceiling_usd"}``: what holds my own held wake, only as far as its records prove it (pure).

    ``own`` is the wake's pause row, ``members`` the pause rows of its paused tree members. The wake paused at the
    carried ceiling (initial or renewed), or, child first, its tree's latch refused it while a member
    holding that inherited number paused at it: the launch stamp (``consciousness_wake.wake_task_metadata``) names
    ``allowance``, ``allowance_and_cap`` or ``per_task_cap``, and a record from before the stamp is only
    ``launch_ceiling``. Otherwise the pause's own reason or rail: ``owner``, ``sleep``, ``technical``,
    ``total_budget``, ``per_task_fence``. No live pause (none recorded, a Resume granted or consumed) is ``queued``: the
    held wake waits for a worker. ``""`` proves nothing: a money stop no limit explains (a lifted allowance)."""
    from ouroboros.budget_pause import (RAIL_DISPATCH_REFUSED, RAIL_GLOBAL_EXHAUSTED, REASON_AUTHORITY, REASON_OWNER,
                                        REASON_SLEEP, STATE_PAUSED, STATE_PAUSING)
    from ouroboros.consciousness_allowance import at_wake_ceiling
    from ouroboros.task_pacing import recorded_producer_ceiling

    metadata = metadata if isinstance(metadata, dict) else {}
    own = own if isinstance(own, dict) else {}
    if own.get("state") not in (STATE_PAUSED, STATE_PAUSING):
        return {"cause": "queued", "ceiling_usd": None}
    reason, rail = own.get("reason"), own.get("rail")
    other = {REASON_OWNER: "owner", REASON_SLEEP: "sleep", REASON_AUTHORITY: "technical"}.get(reason)
    if other or reason != "budget":
        return {"cause": other or "", "ceiling_usd": None}
    lifted = isinstance(metadata.get("producer_allowance_lifted"), dict)
    ceiling = None if lifted else recorded_producer_ceiling({"metadata": metadata})
    refused_root = rail == RAIL_DISPATCH_REFUSED and own.get("scope") == "root"
    tree = refused_root and any(at_wake_ceiling(member, ceiling, "root_resolved_ceiling", legacy=True)
                                for member in members if isinstance(member, dict))
    if at_wake_ceiling(own, ceiling, "producer_allowance", legacy=True) or tree:
        return {"cause": _STAMPED_CAUSES.get(metadata.get("root_cost_ceiling_source"), "launch_ceiling"),
                "ceiling_usd": ceiling}
    if rail == RAIL_GLOBAL_EXHAUSTED or rail == RAIL_DISPATCH_REFUSED and not refused_root:
        return {"cause": "total_budget", "ceiling_usd": None}
    return {"cause": "per_task_fence" if refused_root else "", "ceiling_usd": None}


def _resume_room(drive_root, task_id: str, metadata: dict, own: dict) -> dict:
    """What an owner Resume of this stopped wake would add now, for display: the grant's own decision
    (``allowance_resume_decision``) read with ``allow_stale`` inside a short wait, never from a window another read took;
    an estimate that reserves and decides nothing. ``resume_additional_usd`` is the allowance's room under the tree's
    caps (0.0 when either has none, null when it cannot be read); ``resume_available_usd`` what the total budget leaves
    of it, named only when the decision names it. An empty total budget is a refusal of its own, not the allowance spent."""
    from ouroboros.consciousness_allowance import allowance_resume_decision
    from ouroboros.store import acquisition_budget

    try:
        with acquisition_budget(0.05):
            quote = allowance_resume_decision(drive_root, {"id": task_id, "metadata": metadata}, own, allow_stale=True)
    except Exception:
        log.debug("The room a Resume would add to wake %s could not be read", task_id, exc_info=True)
        quote = {}
    error = quote.get("error")
    if error in {"consciousness_allowance_exhausted", "root_hard_cap_exhausted",
                 "independent_cap_exhausted", "work_cap_export_unverified",
                 "work_cap_reconcile_pending"}:
        return {"resume_additional_usd": 0.0}
    if error == "budget_still_exhausted":
        return {"resume_additional_usd": None, "resume_available_usd": 0.0}
    additional, available = quote.get("additional_usd"), quote.get("available_now_usd")
    if error or type(additional) not in (int, float):
        return {"resume_additional_usd": None}
    return {"resume_additional_usd": float(additional),
            **({"resume_available_usd": float(available)} if type(available) in (int, float) else {})}


def held_wake_cause(drive_root, task_id: str, *, window=None) -> dict:
    """``hold_cause`` of my held own wake from its records: its pool row's metadata (memory), its pause row and, only
    when its tree's latch refused it, its paused members' rows (one compact hot-facts read each, no task body). An
    unreadable record proves nothing, and only what it would have proven is lost: a pause the hot facts name as the
    owner's, a sleep or a technical one stays that, and a money stop whose task record is unreadable falls back to the
    launch stamp. ``window`` (the display's own read, given only while consciousness is on) asks for the room a Resume
    would add now; the room is read afresh (``_resume_room``), never from that window. ``started_at`` (ISO) is the
    wake's own start as its pause row keeps it (a pause never moves it): its block names it in the viewer's clock."""
    from ouroboros.budget_pause import RAIL_DISPATCH_REFUSED, budget_pause_row
    from ouroboros.consciousness_allowance import allowance_stopped, fenced_allowance_members, wake_ceiling_metadata
    from ouroboros.owner_pause import fence_closed
    from ouroboros.store import acquisition_budget
    from ouroboros.store_hot_facts import controls
    from ouroboros.task_results import load_task_result
    from supervisor import queue as q
    from supervisor.consciousness_carriers import live_task_rows, paused_tree_members

    try:
        metadata = (live_task_rows([task_id]).get(task_id) or {}).get("metadata") or {}
        facts = controls(drive_root, task_id)
        own = facts.get("budget_pause") or {}
        started = own.get("started_at")
        started = ({"started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat()}
                   if type(started) in (int, float) and started > 0 else {})
        fence = facts.get("owner_pause") or {}
        if fence_closed(fence) and task_id not in (fence.get("selected_members") or {}):
            return {"cause": "owner", "ceiling_usd": None, **started}
        witnesses = []
        members = []
        if own.get("rail") == RAIL_DISPATCH_REFUSED and own.get("scope") == "root":
            with q._queue_lock:
                latch = (q.BUDGET_ROOT_FENCES.get(task_id) or {}).copy()
            if latch and latch.get("cause") != "owner_pause":
                witnesses = list(fenced_allowance_members(drive_root, task_id, latch.get("fence_id") or ""))
            members = witnesses or [budget_pause_row(root or drive_root, member)
                                    for member, root, _ in paused_tree_members(task_id)]
    except Exception:
        log.debug("The pause of held wake %s is unreadable; its cause stays unknown", task_id, exc_info=True)
        return {"cause": "", "ceiling_usd": None}
    if own.get("reason") == "budget":  # only a money stop needs the record: the ceiling an earlier Resume may have moved
        try:
            with acquisition_budget(0.05):
                metadata = wake_ceiling_metadata(load_task_result(drive_root, task_id, strict=True) or {}) or metadata
        except Exception:
            log.debug("The task record of held wake %s is unreadable; its launch stamp stands", task_id, exc_info=True)
    result = {**hold_cause(metadata, own, members), **started}
    if window is not None and allowance_stopped(metadata, own, witnesses):
        result.update(_resume_room(drive_root, task_id, metadata, own))
    return result
