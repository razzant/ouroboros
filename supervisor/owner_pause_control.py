"""The owner's Pause of a whole tree, supervisor side (owner Batch4 5A).

``request_owner_pause`` is the one accept step behind ``POST
/api/tasks/{id}/pause``. Its order is the protocol (``ouroboros/owner_pause.py``):

1. Admission under the queue lock: a LIVE ROOT (pooled RUNNING or PENDING, or
   the in-process direct turn) with no pending Stop. A child is refused — the
   Pause is the tree's, never one member's. An answered root whose late work
   is still open (D10: post-task synthesis or a retained review operation)
   is the ``late`` lane: its delivered answer stays, the remainder pauses.
2. The durable fence on the root's task result (atomic replace). Nothing is
   acknowledged before it lands; a failed write changes nothing else.
3. Under the queue lock again, the root's existing admission latch
   (``BUDGET_ROOT_FENCES`` typed ``cause=owner_pause``): no new descendant is
   admitted or assigned, including one a delegate event is draining right now,
   and a Resume grant minted but not yet dispatched returns to its pause. The
   census of RUNNING members is taken inside that same lock, so no member can
   be assigned between the census and the latch. The snapshot is persisted.
   The queue lock never waits for the root's launch lock: while that one is
   busy the answer acknowledges the durable fence as it stands, marked
   ``latch_pending``; the same ``request_id`` completes the latch, never a
   second Pause.
4. One ``owner_pause`` mailbox control per live member: a WAKE signal for a
   warm wait; every member's own launch gate and safe boundary read the fence.

The tree's truthful state is kept on the same fence: ``requested`` while any
member still runs or any parked member's own sent work is unsettled, ``paused``
once none does — reviewers that already launched finish separately, recorded as
``finishing_reviews`` and announced to the owner once each way (``refresh_owner_pause_tree``, called at every owner-reason
park, after a member's terminal and from the assignment tick's observe-only
re-check ``settle_requested_owner_pauses``). Only an explicit owner Resume of the root
reopens it — except an answered root whose late work ended with nothing saved
(D10): no member, no open phase, no live review; that Pause has nothing left
and is released. Budget-pause policy is untouched; the only stop this module's
settle tick names is the member park's own (``stop_task_owned``), re-read
from its recorded outcome, never re-issued blindly.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


def _locate_root_locked(q: Any, task_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(task_row, lane)`` for a live task (queue lock held); lane ``running``/``pending``/``direct``."""
    meta = q.RUNNING.get(task_id) if isinstance(q.RUNNING, dict) else None
    if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
        return dict(meta["task"]), "running"
    row = next((dict(item) for item in q.PENDING
                if isinstance(item, dict) and str(item.get("id") or "") == task_id), None)
    if row is not None:
        return row, "pending"
    from supervisor.workers import direct_chat_turn

    turn = direct_chat_turn(task_id)
    return (dict(turn), "direct") if isinstance(turn, dict) else (None, "")


def _late_work(q: Any, task_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """``(result_row, live_review)`` of an answered root still owning late work, else ``(None, False)``."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight, post_task_synthesis_is_open
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.task_results import load_task_result

    try:
        row = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
        live_review = bool(row) and task_has_live_review_operation(q.DRIVE_ROOT, task_id)
    except Exception:
        return None, False
    checkpoint = row.get("root_phase_checkpoint") if isinstance(row.get("root_phase_checkpoint"), dict) else {}
    if live_review or post_task_synthesis_in_flight(q.DRIVE_ROOT, task_id) or post_task_synthesis_is_open(
            checkpoint.get("post_task_synthesis")):
        return {**row, "id": task_id}, live_review
    return None, False


def _late_work_settled(root_drive: Any, root_task_id: str) -> bool:
    """An answered root with no open late phase, synthesis in flight or live review; unknown is False."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight, post_task_synthesis_is_open
    from ouroboros.review_operation import task_has_live_review_operation, paused_acceptance_preparations
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    try:
        row = load_task_result(root_drive, root_task_id, strict=True) or {}
        checkpoint = row.get("root_phase_checkpoint") if isinstance(row.get("root_phase_checkpoint"), dict) else {}
        fence = row.get('owner_pause') or {}
        return bool(row.get("status") in _TRULY_TERMINAL_STATUSES
                    and not post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))
                    and not post_task_synthesis_in_flight(root_drive, root_task_id)
                    and not task_has_live_review_operation(root_drive, root_task_id)
                    and not paused_acceptance_preparations(root_drive, root_task_id, fence.get('fence_id', '')))
    except Exception:
        return False


def _running_members_locked(q: Any, root_task_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    members = []
    for task_id, meta in q.RUNNING.items():
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
        if root_task_id in (str(task.get("root_task_id") or ""), str(task_id)):
            members.append((str(task_id), dict(task)))
    return members


def warm_paused_member(meta: Any) -> bool:
    """A RUNNING member parked WARM under its root's owner Pause (full variant).

    Its ``owner_wait`` row (``reason=owner_pause``) is durably waiting: the
    worker keeps the stack and the assignment while a reviewer it launched
    finishes; the member is saved, not running. Its stack re-parks under a
    newer Pause of the same root rather than run, so any closed generation
    counts. Any other owner wait (a question, a sleep, a review park) is live
    work the Pause still has to reach.
    """
    wait = meta.get("owner_wait") if isinstance(meta, dict) and isinstance(meta.get("owner_wait"), dict) else {}
    return bool(wait.get("state") == "waiting" and wait.get("reason") == "owner_pause")


def _warm_paused_direct_turn(root_drive: Any, task_id: str) -> bool:
    """A direct actor holds its stack while parked warm: its durable row says so."""
    from ouroboros.task_results import load_task_result

    try:
        row = load_task_result(root_drive, task_id, strict=True) or {}
    except Exception:
        return False
    return warm_paused_member({"owner_wait": row.get("owner_wait")})


def _warm_direct_root(q: Any, root_task_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    """The registered direct root parked WARM, shaped as a RUNNING warm member: its
    task from the direct registry, its park (and the stops it issued) from its row."""
    from ouroboros.task_results import load_task_result
    from supervisor.workers import direct_chat_turn

    turn = direct_chat_turn(root_task_id)
    if turn is None:
        return []
    row = load_task_result(pathlib.Path(turn.get("budget_drive_root") or q.DRIVE_ROOT), root_task_id,
                           strict=True) or {}
    meta = {"task": dict(turn), "owner_wait": row.get("owner_wait")}
    return [(root_task_id, meta)] if warm_paused_member(meta) else []


def review_finishing(blocker: Any) -> bool:
    """A reviewer's own custody the owner Pause lets finish, never a Pause blocker."""
    return (isinstance(blocker, dict) and bool(blocker.get("review_owned"))
            and blocker.get("kind") in {"delegated_run", "model_handoff"})


def request_owner_pause(task_id: str, *, request_id: str) -> Dict[str, Any]:
    """Accept the owner's Pause of ``task_id``'s tree (see module docstring)."""
    from ouroboros.owner_pause import (
        OwnerPauseRefused, fence_closed, install_fence, launch_lock, read_fence,
    )
    from ouroboros.task_results import resolve_task_lineage

    from supervisor import queue as q
    task_id = str(task_id or "").strip()
    late_row, late_review = _late_work(q, task_id)  # durable reads stay outside the queue lock
    with q._queue_lock:
        task, lane = _locate_root_locked(q, task_id)
        if task is None and late_row is not None:
            task, lane = late_row, "late"
        if task is None:
            return {"ok": False, "error": "task_not_live"}
        lineage = resolve_task_lineage(
            task_id, metadata=task.get("metadata"), root_task_id=task.get("root_task_id"),
            parent_task_id=task.get("parent_task_id"), delegation_role=task.get("delegation_role"),
            original_task_id=task.get("original_task_id"), timeout_retry_from=task.get("timeout_retry_from"))
        if not lineage["is_root_task"]:
            return {"ok": False, "error": "not_a_root_task", "root_task_id": str(task.get("root_task_id") or "")}
        from ouroboros.cancel_intents import cancel_pending

        try:
            if cancel_pending(q.DRIVE_ROOT, task_id, strict=True):
                return {"ok": False, "error": "cancel_pending"}
        except Exception:
            return {"ok": False, "error": "cancellation_authority_unavailable"}
        root_drive = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
        if lane not in {"direct", "late"}:
            try:
                q.ensure_control_task_result(task_id)
            except Exception:
                log.warning("Owner pause lifecycle authority unavailable for %s", task_id, exc_info=True)
                return {"ok": False, "error": "pause_record_unwritable"}
    from supervisor.events_budget import _set_root_budget_pause_locked

    try:
        fence, created = install_fence(root_drive, task_id, request_id=request_id, late_work=late_review)
        if fence_closed(fence):
            from ouroboros.review_operation import retain_preparing_owner_pause
            retain_preparing_owner_pause(root_drive, task_id, fence)
    except OwnerPauseRefused as exc:
        return {"ok": False, "error": str(exc) or "pause_refused"}
    except Exception as exc:
        log.warning("Owner pause fence for %s was not written", task_id, exc_info=True)
        return {"ok": False, "error": "pause_record_unwritable", "detail": str(exc)[:200]}
    answer = {"ok": True, "task_id": task_id, "root_task_id": task_id, "fence_id": fence["fence_id"],
              "duplicate": not created, "state": str(fence.get("state") or ""), "members": []}
    released = {**answer, "duplicate": True, "state": "released"}
    if not fence_closed(fence):
        return released  # a replay of an action already resumed: no queue effect
    with q._queue_lock:
        try:
            # Resume consumption can run outside the queue lock. Revalidate
            # the exact accepted generation before publishing its queue latch.
            # One root's held launch lock must not stall unrelated controls.
            with launch_lock(root_drive, task_id, wait=False):
                current = read_fence(root_drive, task_id)
                if not fence_closed(current) or fence.get("fence_id") != current.get("fence_id"):
                    return released
                latch = _set_root_budget_pause_locked(task_id, {
                    "fence_id": str(fence["fence_id"]), "cause": "owner_pause",
                    "paused_at": str(fence.get("requested_at") or utc_now_iso())})
                members = _running_members_locked(q, task_id)
                persisted = q.persist_queue_snapshot(reason="owner_pause_requested")
        except OwnerPauseRefused:
            # Only the busy launch lock refuses here, after the fence landed:
            # every launch gate already reads it. Not a refusal of the Pause.
            if (q.BUDGET_ROOT_FENCES.get(task_id) or {}).get("fence_id") == fence["fence_id"]:
                return answer  # this generation's latch already stands
            log.info("Owner pause of %s is durable; its queue latch waits for the launch lock", task_id)
            return {**answer, "latch_pending": True}
        except Exception as exc:
            log.warning("Owner pause fence for %s was not written", task_id, exc_info=True)
            return {"ok": False, "error": "pause_record_unwritable", "detail": str(exc)[:200]}
    if not persisted:
        # The durable fence already holds every launch; the queue latch lives
        # in memory until the next persisted snapshot. Disclosed, not hidden.
        log.error("Owner pause of %s: queue latch installed but its snapshot was not persisted", task_id)
    woken = _wake_members(q, task_id, fence, members, direct=(lane == "direct"))
    event = {"ts": utc_now_iso(), "type": "owner_pause_requested", "task_id": task_id,
             "root_task_id": task_id, "fence_id": fence["fence_id"], "duplicate": not created,
             "queue_latch_fence_id": latch.get("fence_id"), "members_woken": woken,
             "owner_visible": True, "toast_once": f"{task_id}:owner-pause:{fence['fence_id']}"}
    append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl", event)
    state = refresh_owner_pause_tree(task_id)
    return {**answer, "state": state or answer["state"], "members": woken, "snapshot_persisted": bool(persisted)}


def _wake_members(q: Any, root_task_id: str, fence: Dict[str, Any], members: list, *, direct: bool) -> List[str]:
    """One wake control per live member; the fence, not this row, is the authority."""
    from ouroboros.owner_mailbox import KIND_OWNER_PAUSE, write_owner_message

    targets = [(member_id, row) for member_id, row in members]
    if direct:
        targets.append((root_task_id, {"id": root_task_id}))
    woken: List[str] = []
    for member_id, row in targets:
        try:
            drive = q._task_drive_for_task(row, member_id)
            if write_owner_message(drive, "owner_pause", member_id,
                                   msg_id=f"owner_pause:{fence['fence_id']}", kind=KIND_OWNER_PAUSE):
                woken.append(member_id)
        except Exception:
            # A member that misses its wake still meets the fence at its next
            # launch handoff or boundary; only a warm wait waits longer.
            log.warning("Owner pause wake for %s was not written", member_id, exc_info=True)
    return woken


def refresh_owner_pause_tree(root_task_id: str) -> str:
    """Turn the root's fence ``paused`` once the whole tree is saved and its own work stopped.

    Saved means: every member of the tree is parked — cold under its exact
    pause, or warm (its stack retained while a reviewer it launched finishes)
    — and no direct turn is live unparked. Stopped means: no member's own
    sent work is open — a task-owned run whose stop is only requested or
    unknown keeps the tree ``requested`` exactly as it keeps that member's
    Resume refused (#1562; the settle tick re-reads it). Only the owner's one
    exception is disclosed instead of waited for: reviewers already launched,
    their delegated runs and their own model sends (``finishing_reviews``).
    Returns the fence state after the check (``""`` when no fence).
    """
    from ouroboros.owner_pause import (
        FENCE_PAUSED, FENCE_RELEASED, FENCE_REQUESTED, fence_closed, read_fence, set_fence_state,
    )
    from supervisor.workers import direct_chat_turn

    from supervisor import queue as q
    root_task_id = str(root_task_id or "").strip()
    with q._queue_lock:
        root_row = next((item for item in q.PENDING if str(item.get("id") or "") == root_task_id), None)
        root_meta = q.RUNNING.get(root_task_id)
        root_task = (root_meta.get("task") if isinstance(root_meta, dict) else None) or root_row or {}
        root_drive = pathlib.Path((root_task or {}).get("budget_drive_root") or q.DRIVE_ROOT)
        running_all = _running_members_locked(q, root_task_id)
        running_meta = {task_id: dict(q.RUNNING.get(task_id) or {}) for task_id, _task in running_all}
        queued = [dict(item) for item in q.PENDING if isinstance(item, dict)
                  and root_task_id in (str(item.get("root_task_id") or ""), str(item.get("id") or ""))]
        parked = [item for item in queued if isinstance(item.get("_budget_pause"), dict)]
    try:
        fence = read_fence(root_drive, root_task_id)
    except Exception:
        log.debug("Owner pause fence unreadable for %s", root_task_id, exc_info=True)
        return ""
    fence_id = str(fence.get("fence_id") or "")
    warm = [task_id for task_id, _task in running_all if warm_paused_member(running_meta.get(task_id))]
    running = [(task_id, task) for task_id, task in running_all if task_id not in warm]
    direct = direct_chat_turn(root_task_id)
    if direct is not None and _warm_paused_direct_turn(root_drive, root_task_id):
        direct = None
        warm.append(root_task_id)
    if (fence_closed(fence) and not running_all and not queued and direct is None
            and _late_work_settled(root_drive, root_task_id)):
        try:
            # CAS on the read state: a concurrent Resume or newer Pause wins.
            set_fence_state(root_drive, root_task_id, fence_id=str(fence.get("fence_id") or ""),
                            state=FENCE_RELEASED, expected_state=str(fence.get("state") or ""),
                            release_reason="late_work_settled")
        except Exception:
            log.debug("Settled late-work Pause of %s was not released", root_task_id, exc_info=True)
            return str(fence.get("state") or "")
        with q._queue_lock:
            latch = q.BUDGET_ROOT_FENCES.get(root_task_id) or {}
            if latch.get("cause") == "owner_pause" and latch.get("fence_id") == fence.get("fence_id"):
                q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
                q.persist_queue_snapshot(reason="owner_pause_late_work_settled")
        append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                     {"ts": utc_now_iso(), "type": "owner_pause_released", "task_id": root_task_id,
                      "root_task_id": root_task_id, "fence_id": fence.get("fence_id"),
                      "reason": "late_work_settled", "owner_visible": True})
        return FENCE_RELEASED
    if not fence or str(fence.get("state") or "") not in {FENCE_REQUESTED, FENCE_PAUSED}:
        return str(fence.get("state") or "")
    if running or direct is not None:
        return str(fence.get("state") or "")
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.review_operation import task_has_live_review_operation

    try:
        # The answered root's unsaved late phase still runs (D10): its saved pause
        # is not yet written. Its review, like every member's, finishes separately.
        if post_task_synthesis_in_flight(root_drive, root_task_id):
            return str(fence.get("state") or "")
        for item in parked:
            row = budget_pause_row(pathlib.Path(item.get("budget_drive_root") or q.DRIVE_ROOT),
                                   str(item.get("id") or ""))
            if str(row.get("settlement") or "") == SETTLEMENT_EXTERNAL_RUNNING:
                return str(fence.get("state") or "")
    except Exception:
        return str(fence.get("state") or "")
    from supervisor.continuation_admission import conflicting_writers

    blockers = conflicting_writers(q, root_task_id, drive_root=root_drive, owner_pause_fence_id=fence_id)
    if [b for b in blockers if not review_finishing(b)]:
        return str(fence.get("state") or "")
    reviews = {str(b.get("task_id") or b.get("attempt_id") or "") for b in blockers if review_finishing(b)}
    for task_id in [root_task_id, *warm, *[str(item.get("id") or "") for item in parked]]:
        try:
            if task_id and task_has_live_review_operation(root_drive, task_id):
                reviews.add(task_id)
        except Exception:
            log.debug("Live review operations of %s unreadable at the Pause census", task_id, exc_info=True)
    facts = {"parked_members": [str(item.get("id") or "") for item in parked] + warm,
             "warm_members": list(warm), "finishing_reviews": sorted(reviews - {""})}
    if str(fence.get("state") or "") == FENCE_PAUSED:
        if all(fence.get(key) == value for key, value in facts.items()):
            return FENCE_PAUSED
        try:  # the "still finishing" line changes on the same paused fence
            set_fence_state(root_drive, root_task_id, fence_id=fence_id, state=FENCE_PAUSED,
                            expected_state=FENCE_PAUSED, **facts)
            _announce_finishing_reviews(root_drive, root_task_id, fence.get("finishing_reviews"),
                                        facts["finishing_reviews"])
        except Exception:
            log.debug("Paused fence facts of %s were not refreshed", root_task_id, exc_info=True)
        return FENCE_PAUSED
    try:
        # CAS on the state this census read: a Resume that released the same
        # fence meanwhile wins; a stale census never revives released authority.
        set_fence_state(root_drive, root_task_id, fence_id=fence_id,
                        state=FENCE_PAUSED, expected_state=FENCE_REQUESTED, **facts)
    except Exception:
        try:
            moved = read_fence(root_drive, root_task_id)
        except Exception:
            moved = fence
        if moved.get("fence_id") != fence.get("fence_id") or moved.get("state") != FENCE_REQUESTED:
            return str(moved.get("state") or "")
        log.warning("Owner pause of %s is settled but its paused state was not recorded", root_task_id,
                    exc_info=True)
        return FENCE_REQUESTED
    append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                 {"ts": utc_now_iso(), "type": "owner_pause_settled", "task_id": root_task_id,
                  "root_task_id": root_task_id, "fence_id": fence.get("fence_id"),
                  "parked_members": [str(item.get("id") or "") for item in parked],
                  "finishing_reviews": facts["finishing_reviews"],
                  "owner_visible": True, "toast_once": f"{root_task_id}:owner-paused:{fence.get('fence_id')}"})
    _announce_finishing_reviews(root_drive, root_task_id, [], facts["finishing_reviews"])
    return FENCE_PAUSED


def _announce_finishing_reviews(root_drive: Any, root_task_id: str, before: Any, after: List[str]) -> None:
    """The owner SEES the reviewers its Pause lets finish (owner 2026-10-08): one host
    line in the task's chat when the set appears and one when it empties, through the
    existing progress channel. Display only; never a phase, a gate or a label."""
    before, after = sorted(before or []), sorted(after or [])
    if before == after:
        return
    try:
        from ouroboros.task_results import load_task_result
        from supervisor.message_bus import send_with_budget

        chat_id = (load_task_result(pathlib.Path(root_drive), root_task_id) or {}).get("chat_id")
        if chat_id is None:
            return
        text = ("Paused. Review work is finishing separately; Resume does not wait. Received results "
                "can be collected after Resume without buying the same review again. Nothing is committed, "
                "applied or published meanwhile." if after else
                "The review that was finishing under the Pause has ended; the task stays paused until Resume.")
        send_with_budget(int(chat_id), text, is_progress=True, task_id=root_task_id, role="system",
                         system_type="owner_pause_notice")
    except Exception:
        log.debug("Finishing-review notice of %s was not sent", root_task_id, exc_info=True)


def late_phase_settled(drive_root: Any, root_task_id: str) -> None:
    """A late phase off the pooled lane just parked, ended or was stopped (D10).

    Settle the root's Pause (``paused``, or released when nothing remains) and
    re-check held Continues. Pooled workers reach the same checks through
    their ``task_done``; outside this supervisor's queue it observes nothing.
    """
    from supervisor import queue as q

    try:
        if not getattr(q, "INITIALIZED", False) or (
                pathlib.Path(q.DRIVE_ROOT).resolve() != pathlib.Path(drive_root).resolve()):
            return
        from supervisor.continuation_admission import release_settled_continuations
        from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

        if clear_budget_root_fence_for_settled_tree({"id": root_task_id, "root_task_id": root_task_id}):
            q.persist_queue_snapshot(reason="late_phase_settled")
        refresh_owner_pause_tree(root_task_id)
        release_settled_continuations(root_task_id)
    except Exception:
        log.warning("Late-phase settlement check failed for %s", root_task_id, exc_info=True)


def retain_late_phase_latch(drive_root: Any, root_task_id: str) -> None:
    """Startup: a saved late phase keeps its owner Pause latch (and so its census row
    and Resume) even when the restored snapshot was too old to carry it. Never a grant."""
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor import queue as q
    from supervisor.events_budget import _set_root_budget_pause_locked

    try:
        if not getattr(q, "INITIALIZED", False) or (
                pathlib.Path(q.DRIVE_ROOT).resolve() != pathlib.Path(drive_root).resolve()):
            return
        fence = read_fence(drive_root, root_task_id)
        if not fence_closed(fence):
            return
        with q._queue_lock:
            if root_task_id in q.BUDGET_ROOT_FENCES:
                return
            _set_root_budget_pause_locked(root_task_id, {
                "fence_id": str(fence["fence_id"]), "cause": "owner_pause",
                "paused_at": str(fence.get("requested_at") or utc_now_iso())})
            q.persist_queue_snapshot(reason="late_phase_pause_retained")
    except Exception:
        log.warning("Saved late-phase Pause latch of %s was not retained", root_task_id, exc_info=True)


_SETTLE_RECHECK_SEC = 5.0
_LAST_SETTLE_CHECK: Dict[str, float] = {}


def _record_settled_member(q: Any, item: Dict[str, Any]) -> None:
    """A parked member whose own sent work was still open at its park: a fresh
    custody read re-requests the stop of every task-owned run still open
    (``stop_task_owned``: the same verb the park used, pushing ``requested`` to
    ``confirmed``) and records the member settled once none remains. Its
    started critics are observed, never cancelled.

    The custody read runs off-lock, so the write compares the pause, state
    and grant it read: a Resume, revocation or newer pause landing meanwhile
    wins, and this stale observation is dropped for the next pass to redo.
    """
    from ouroboros.budget_pause import (
        STOP_POLICY_TASK_OWNED, BudgetPauseSuperseded, budget_pause_row, observe_task_runs, set_budget_pause,
        task_owned_runs_open,
    )
    from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING, SETTLEMENT_SETTLED

    task_id = str(item.get("id") or "")
    result_root = pathlib.Path(item.get("budget_drive_root") or q.DRIVE_ROOT)
    row = budget_pause_row(result_root, task_id)
    if str(row.get("settlement") or "") != SETTLEMENT_EXTERNAL_RUNNING:
        return
    # The stops this pause already issued are re-READ, never issued again.
    observed = observe_task_runs(result_root, task_id, reason="owner_pause_settlement_check",
                                 stop_policy=STOP_POLICY_TASK_OWNED, prior=row.get("external_runs"))
    if task_owned_runs_open(observed):
        def facts(external: Any) -> list:
            return sorted((str(run.get("run_id") or run.get("invocation_id") or ""), str(run.get("state") or ""),
                           str(run.get("stop_outcome") or "")) for run in (external or {}).get("runs") or []
                          if isinstance(run, dict))
        if observed.get("custody_read") == "ok" and facts(observed) == facts(row.get("external_runs")):
            return  # nothing moved: no rewrite of the same facts every tick
        try:  # the fresh outcomes (requested -> confirmed, unknown) are the owner's facts
            set_budget_pause(result_root, task_id, {**row, "external_runs": observed},
                             expected_pause_id=str(row.get("pause_id") or ""),
                             expected_state=str(row.get("state") or ""),
                             expected_grant_id=str((row.get("grant") or {}).get("grant_id") or ""))
        except BudgetPauseSuperseded:
            log.debug("Owner pause observation of %s superseded by a newer pause row", task_id, exc_info=True)
        return
    try:
        set_budget_pause(result_root, task_id, {**row, "settlement": SETTLEMENT_SETTLED,
                                                "settlement_observed_at": utc_now_iso()},
                         expected_pause_id=str(row.get("pause_id") or ""),
                         expected_state=str(row.get("state") or ""),
                         expected_grant_id=str((row.get("grant") or {}).get("grant_id") or ""))
    except BudgetPauseSuperseded:
        log.debug("Owner pause settlement of %s superseded by a newer pause row", task_id, exc_info=True)


def _reread_warm_member(q: Any, task_id: str, meta: Dict[str, Any]) -> None:
    """A warm-parked member's own runs: re-read the stops its park issued (never a
    second cancel) and keep the park's facts current for its Resume notice.

    The custody read runs off-lock, so the write compares the wait identity and
    state it read: a Resume or newer park landing meanwhile wins (Restart must
    keep the resumed stack's newer cognition), and this stale observation is
    dropped for the next pass to redo. The RUNNING mirror takes only the facts.
    """
    from ouroboros.budget_pause import STOP_POLICY_TASK_OWNED, observe_task_runs
    from ouroboros.owner_wait import OwnerWaitSuperseded, set_owner_wait

    wait = dict(meta.get("owner_wait") or {})
    pause = dict(wait.get("owner_pause") or {})
    task = meta.get("task") or {}
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    observed = observe_task_runs(result_root, task_id, reason="owner_pause_settlement_check",
                                 stop_policy=STOP_POLICY_TASK_OWNED, prior=pause.get("external_runs"))
    facts = lambda external: sorted((str(r.get("run_id") or ""), str(r.get("state") or ""))  # noqa: E731
                                    for r in (external or {}).get("runs") or [] if isinstance(r, dict))
    if observed.get("custody_read") != "ok" or facts(observed) == facts(pause.get("external_runs")):
        return
    try:
        set_owner_wait(result_root, task_id, {**wait, "owner_pause": {**pause, "external_runs": observed}},
                       expected_wait_id=str(wait.get("wait_id") or ""), expected_state=str(wait.get("state") or ""))
    except OwnerWaitSuperseded:
        log.debug("Warm Pause observation of %s superseded by a Resume or newer park", task_id, exc_info=True)
        return
    with q._queue_lock:
        live = q.RUNNING.get(task_id)
        current = live.get("owner_wait") if isinstance(live, dict) else None
        if (isinstance(current, dict) and current.get("wait_id") == wait.get("wait_id")
                and current.get("state") == wait.get("state")):
            live["owner_wait"] = {**current, "owner_pause": {**dict(current.get("owner_pause") or {}),
                                                             "external_runs": observed}}


def settle_requested_owner_pauses(q: Any = None, *, now: Optional[float] = None) -> List[str]:
    """The assignment tick's re-check of owner Pauses still ``requested``.

    A tree stays ``pausing`` while a parked member's own sent work is not yet
    proven stopped; nothing else would notice that while every member is
    parked. This pass (throttled per root, no second scheduler) re-reads each
    parked — cold or warm, a direct root's warm park included — member's runs:
    a stop the Pause already issued is only read back, a run whose stop was
    never issued gets its one request, and a member whose custody is now clean
    is recorded settled; then ``refresh_owner_pause_tree`` decides. It never
    grants anything. Returns the roots that turned ``paused``.
    """
    import time

    from ouroboros.owner_pause import FENCE_PAUSED

    if q is None:
        from supervisor import queue as q
    now = time.monotonic() if now is None else now
    with q._queue_lock:
        roots = [str(root_id) for root_id, fence in q.BUDGET_ROOT_FENCES.items()
                 if isinstance(fence, dict) and fence.get("cause") == "owner_pause"]
    for stale in set(_LAST_SETTLE_CHECK) - set(roots):
        _LAST_SETTLE_CHECK.pop(stale, None)
    settled: List[str] = []
    for root_id in roots:
        if now - _LAST_SETTLE_CHECK.get(root_id, float("-inf")) < _SETTLE_RECHECK_SEC:
            continue
        _LAST_SETTLE_CHECK[root_id] = now
        with q._queue_lock:
            parked = [dict(item) for item in q.PENDING if isinstance(item, dict)
                      and root_id in (str(item.get("root_task_id") or ""), str(item.get("id") or ""))
                      and isinstance(item.get("_budget_pause"), dict)]
            warm = [(str(task_id), dict(meta)) for task_id, meta in q.RUNNING.items()
                    if warm_paused_member(meta) and root_id in (
                        str((meta.get("task") or {}).get("root_task_id") or ""), str(task_id))]
        try:
            warm.extend(_warm_direct_root(q, root_id))
        except Exception:
            log.warning("Warm Pause of direct root %s unreadable; it stays pausing", root_id, exc_info=True)
        # One member's failure never skips its siblings or the tree's own census.
        for item in parked:
            try:
                _record_settled_member(q, item)
            except Exception:
                log.warning("Owner pause settlement re-check of %s (tree %s) failed; it stays pausing",
                            item.get("id"), root_id, exc_info=True)
        for task_id, meta in warm:
            try:
                _reread_warm_member(q, task_id, meta)
            except Exception:
                log.warning("Warm Pause re-check of %s (tree %s) failed; it stays pausing", task_id, root_id,
                            exc_info=True)
        try:
            if refresh_owner_pause_tree(root_id) == FENCE_PAUSED:
                settled.append(root_id)
        except Exception:
            log.warning("Owner pause settlement re-check of %s failed; it stays pausing", root_id, exc_info=True)
    return settled
