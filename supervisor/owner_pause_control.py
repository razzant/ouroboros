"""The owner's Pause of a whole tree, supervisor side (owner Batch4 5A).

``request_owner_pause`` is the one accept step behind ``POST
/api/tasks/{id}/pause``. Its order is the protocol (``ouroboros/owner_pause.py``):

1. Admission under the queue lock: a LIVE ROOT (pooled RUNNING or PENDING, or
   the in-process direct turn) with no pending Stop. A child is refused — the
   Pause is the tree's, never one member's. A timeout retry's current attempt
   pauses its LOGICAL root's tree (#1730, ``owner_pause.root_actor``): every
   launch gate reads that root's fence and latch. An answered root whose late work
   is still open (post-task upkeep, a retained review operation, or delegated
   work its tree's ended members left: ``_ended_tail``) is the ``late`` lane: its
   delivered answer stays, the Pause ends the upkeep at once, requests that
   delegated work's stop and defers a review's unsent preparation until Resume.
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
member still runs or any member's own sent work is unsettled, an ended member's
included, ``paused`` once none does — reviewers that already launched finish separately, recorded as
``finishing_reviews`` and announced to the owner once each way (``refresh_owner_pause_tree``, called at every owner-reason
park, after a member's terminal and from the observe-only re-check
``settle_requested_owner_pauses``). Only an explicit owner Resume of the root
reopens it — except an answered root whose late work ended: no member, no
open phase, no live review, no deferred preparation and nothing its census still
holds; that Pause has nothing left and is released. A terminal result is not ended
execution: that one census decides this release and whether the queue latch may go
(``owner_latch_holds``, at ``task_done`` and Restart), except a root's recorded Stop, which
outranks the Pause (``owner_pause.owner_stopped``). Budget-pause policy is untouched;
the stops this module's settle pass names are the member park's own (``stop_task_owned``),
re-read from its recorded outcome, never re-issued blindly, and the ones no member is left
to take (``_stop_ended_members``, #2669), under the same verb and rule. A nanny's death
alone stops nothing: only an accepted Pause reaches this pass.

A settled tree costs nothing to keep paused. Its census records that the tree is at
rest with the member signature it saw (``_AT_REST``); a pass reads nothing for it while
that fence and signature stand and no transition marked it (``mark_tree_changed``). A
state string alone never proves rest: the late-work release and the member facts Resume
reads live on the paused path. A tree with something still open (a stop not yet confirmed,
a review finishing, a member with no queue row left to end) stays due on the body's
cadence, because a remote run's end or a process's exit has no transition to wake on.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ouroboros.utils import append_jsonl, utc_now_iso
from ouroboros.work_stop import pause_marker

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


def _presence_pause_refusal(drive_root: Any, task_id: str, row: Optional[Dict[str, Any]]) -> str:
    """Why Pause cannot take a turn a Presence runner holds, else "". Only a turn the owner moved to a Project, carried
    as text, saves its stack (``presence_carrier.moved_text_turn``, #2566); for any other the Pause changes nothing and
    says what it knows, witnessed live or liveness unknown, never "not live". ``row`` is the stored row, {} while a
    starting turn has none, None when unreadable. A settled or other row reads no census."""
    from ouroboros.presence_carrier import moved_text_turn
    from ouroboros.presence_runner import presence_execution_witness
    from supervisor.cancel_publication import live_presence_row

    if row and (not live_presence_row(row) or moved_text_turn(drive_root, task_id, row.get("metadata"))):
        return ""
    witness = presence_execution_witness(task_id, drive_root=drive_root)
    if witness:
        return "presence_pause_unsupported"
    return "presence_liveness_unknown" if witness is None and row != {} else ""


def _late_work(q: Any, task_id: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """``(result_row, late_work, "")`` of an answered root still owning late work, ``(None, False, "")`` when there is
    none, or ``(None, False, refusal)`` of a turn its inline author may still run (``_presence_pause_refusal``): not an
    answered root, whatever its tree holds.

    Its open or in-flight upkeep is reachable so the Pause can end it at once; a live
    review operation is reachable so its unsent preparation can wait for Resume.
    Live descendants also belong to the answered root, even after all its own phases ended;
    so does the delegated work its tree's ended members left (``_ended_tail``).
    """
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight, post_task_synthesis_is_open
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.task_results import load_task_result

    try:
        row = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
    except Exception:
        return None, False, _presence_pause_refusal(q.DRIVE_ROOT, task_id, None)
    try:
        if refusal := _presence_pause_refusal(q.DRIVE_ROOT, task_id, row):
            return None, False, refusal
        live_review = bool(row) and task_has_live_review_operation(q.DRIVE_ROOT, task_id)
    except Exception:
        return None, False, ""
    checkpoint = row.get("root_phase_checkpoint") if isinstance(row.get("root_phase_checkpoint"), dict) else {}
    with q._queue_lock:
        descendants = any(str(task.get("root_task_id") or "") == task_id and task.get("id") != task_id
                          for task in [*q.PENDING, *(m.get("task") or {} for m in q.RUNNING.values())])
    late = descendants or live_review or _ended_tail(q, task_id, row)
    if late or post_task_synthesis_in_flight(q.DRIVE_ROOT, task_id) or post_task_synthesis_is_open(
            checkpoint.get("post_task_synthesis")):
        return {**row, "id": task_id}, late, ""
    return None, False, ""


def _ended_tail(q: Any, task_id: str, row: Dict[str, Any]) -> bool:
    """A task that ended while its tree still holds ordinary delegated work: a terminal result is not ended
    execution. Unreadable custody is never "nothing left": the fence's own census decides on the next look."""
    from ouroboros.external_runs import tree_owned_custody
    from ouroboros.owner_pause import owner_stopped
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    if row.get("status") not in _TRULY_TERMINAL_STATUSES or owner_stopped(row):
        return False
    try:
        return bool(tree_owned_custody(pathlib.Path(row.get("budget_drive_root") or q.DRIVE_ROOT),
                                       str(row.get("root_task_id") or task_id)))
    except Exception:
        log.warning("Delegated custody of ended task %s unreadable at Pause admission", task_id, exc_info=True)
        return True


def _late_work_settled(root_drive: Any, root_task_id: str) -> bool:
    """An answered root with no open late phase, synthesis in flight or live review; unknown is False.
    Its own phases only: its tree's custody is the census's (``refresh_owner_pause_tree``)."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight, post_task_synthesis_is_open
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    try:
        row = load_task_result(root_drive, root_task_id, strict=True) or {}
        checkpoint = row.get("root_phase_checkpoint") if isinstance(row.get("root_phase_checkpoint"), dict) else {}
        return bool(row.get("status") in _TRULY_TERMINAL_STATUSES
                    and not post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))
                    and not post_task_synthesis_in_flight(root_drive, root_task_id)
                    and not task_has_live_review_operation(root_drive, root_task_id))
    except Exception:
        return False


def _running_members_locked(q: Any, root_task_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    members = []
    for task_id, meta in q.RUNNING.items():
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
        if root_task_id in (str(task.get("root_task_id") or ""), str(task_id)):
            members.append((str(task_id), dict(task)))
    return members


def _root_drive(q: Any, root_task_id: str) -> pathlib.Path:
    """A tree's fence and custody: its root's queue row's drive, else the queue's (an ended root has no row)."""
    with q._queue_lock:
        meta = q.RUNNING.get(root_task_id)
        task = ((meta.get("task") if isinstance(meta, dict) else None)
                or next((item for item in q.PENDING if str(item.get("id") or "") == root_task_id), None) or {})
    return pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)


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


def request_owner_pause(task_id: str, *, request_id: str, by: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Accept the owner's Pause of ``task_id``'s tree (see module docstring); ``by``: the press's declared author."""
    from ouroboros.owner_pause import (
        OwnerPauseRefused, fence_closed, install_fence, launch_lock, read_fence,
    )
    from ouroboros.task_carrier import execution_kind
    from ouroboros.task_results import resolve_task_lineage
    from ouroboros.store import acquisition_budget, StoreBusy
    from ouroboros.control_actor import control_actor, observed_control_stop
    from ouroboros.work_stop import release_refusal

    from supervisor import queue as q
    task_id = str(task_id or "").strip()
    late_row, late_review, presence_refusal = _late_work(q, task_id)  # durable reads stay outside the queue lock
    from ouroboros.presence_carrier import moved_text_turn
    from ouroboros.task_results import load_task_result

    try:  # a moved text turn takes its Project's ordinary Pause (#2566); every other Presence turn keeps its refusal
        moved = moved_text_turn(q.DRIVE_ROOT, task_id, (load_task_result(q.DRIVE_ROOT, task_id) or {}).get("metadata"))
    except Exception:
        moved = False
    with acquisition_budget(0), q._queue_lock:
        from supervisor.workers import get_direct_chat_agent

        # A live call has no resumable checkpoint. Its registered
        # actor still owns work after message admission closes; Stop can end it.
        actor = get_direct_chat_agent(task_id)
        if getattr(actor, "_realtime_turn", None) is not None:
            return {"ok": False, "error": "realtime_pause_unsupported"}
        task, lane = _locate_root_locked(q, task_id)
        if task is not None and execution_kind(task) == "presence" and not moved:  # an unmoved turn's stack is unsaved
            return {"ok": False, "error": "presence_pause_unsupported"}
        if task is None and late_row is not None:
            task, lane = late_row, "late"
        if task is None:
            return {"ok": False, "error": presence_refusal or "task_not_live"}
        lineage = resolve_task_lineage(
            task_id, metadata=task.get("metadata"), root_task_id=task.get("root_task_id"),
            parent_task_id=task.get("parent_task_id"), delegation_role=task.get("delegation_role"),
            original_task_id=task.get("original_task_id"), timeout_retry_from=task.get("timeout_retry_from"))
        if not lineage["is_root_task"]:
            return {"ok": False, "error": "not_a_root_task", "root_task_id": str(task.get("root_task_id") or "")}
        root_drive = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
        if refusal := release_refusal(observed_control_stop(
                root_drive, task, budget=q.BUDGET_ROOT_FENCES), control_actor()):
            return refusal
        root_id = lineage["root_task_id"] if lineage["is_retry_root_attempt"] else task_id
        from ouroboros.owner_pause import root_actor
        if not root_actor(root_drive, root_id, task_id):
            return {"ok": False, "error": "retry_lineage_unverified", "root_task_id": root_id}
        from ouroboros.cancel_intents import cancel_pending

        try:
            # A graceful cascade Stop stays keyed by the logical root; a SINGLE one by the attempt.
            if any(cancel_pending(q.DRIVE_ROOT, member, strict=True) for member in {root_id, task_id}):
                return {"ok": False, "error": "cancel_pending"}
        except Exception:
            return {"ok": False, "error": "cancellation_authority_unavailable"}
        if lane not in {"direct", "late"}:
            try:
                q.ensure_control_task_result(task_id)
            except Exception:
                log.warning("Owner pause lifecycle authority unavailable for %s", task_id, exc_info=True)
                return {"ok": False, "error": "pause_record_unwritable"}
    from supervisor.events_budget import _set_root_budget_pause_locked

    try:
        fence, created = install_fence(root_drive, root_id, request_id=request_id,
                                      late_work=late_review, actor_id=task_id, by=by)
    except OwnerPauseRefused as exc:
        return {"ok": False, "error": str(exc) or "pause_refused"}
    except Exception as exc:
        log.warning("Owner pause fence for %s was not written", task_id, exc_info=True)
        return {"ok": False, "error": "pause_record_unwritable", "detail": str(exc)[:200]}
    answer = {"ok": True, "task_id": task_id, "root_task_id": root_id, "fence_id": fence["fence_id"],
              "duplicate": not created, "state": str(fence.get("state") or ""), "members": []}
    released = {**answer, "duplicate": True, "state": "released"}
    if not fence_closed(fence):
        return released  # a replay of an action already resumed: no queue effect
    with acquisition_budget(0), q._queue_lock:
        try:
            # Resume consumption can run outside the queue lock. Revalidate
            # the exact accepted generation before publishing its queue latch.
            # One root's held launch lock must not stall unrelated controls.
            with launch_lock(root_drive, root_id, wait=False):
                current = read_fence(root_drive, root_id)
                if not fence_closed(current) or fence.get("fence_id") != current.get("fence_id"):
                    return released
                latest, _lane = _locate_root_locked(q, task_id)
                if (latest or {}).get("_budget_pause_hold") != task.get("_budget_pause_hold"):
                    return {"ok": False, "error": "selection_authority_changed"}
                if refusal := release_refusal(observed_control_stop(
                        root_drive, latest, budget=q.BUDGET_ROOT_FENCES, owner=current), control_actor()):
                    return refusal
                # The accepted Pause replaces only the decision this actor may release.
                # Independent exact continuation custody stays on its own marker.
                for member in [*q.PENDING, *(m.get("task") or {} for m in q.RUNNING.values())]:
                    if member.get("id") == task_id:
                        member.pop("_budget_pause_hold", None)
                latch = _set_root_budget_pause_locked(root_id, {
                    "fence_id": str(fence["fence_id"]), "cause": "owner_pause",
                    "paused_at": str(fence.get("requested_at") or utc_now_iso())})
                members = _running_members_locked(q, root_id)
        except StoreBusy as exc:
            if exc.reason != "contention":
                return {"ok": False, "error": "pause_record_unwritable", "detail": str(exc)[:200]}
            # The fence was committed before Q. A busy projection probe cannot
            # erase that acceptance or wait while holding the global queue.
            return {**answer, "latch_pending": True}
        except OwnerPauseRefused:
            # Only the busy launch lock refuses here, after the fence landed:
            # every launch gate already reads it. Not a refusal of the Pause.
            if (q.BUDGET_ROOT_FENCES.get(root_id) or {}).get("fence_id") == fence["fence_id"]:
                return answer  # this generation's latch already stands
            log.info("Owner pause of %s is durable; its queue latch waits for the launch lock", root_id)
            return {**answer, "latch_pending": True}
        except Exception as exc:
            log.warning("Owner pause fence for %s was not written", task_id, exc_info=True)
            return {"ok": False, "error": "pause_record_unwritable", "detail": str(exc)[:200]}
    # The latch is published outside Q and the launch lock (the barrier captures the live maps under Q
    # itself), before the members are woken and before the reply names it: the reply's
    # snapshot_persisted is a disk fact, and no wake runs ahead of the publication. A raise keeps the
    # typed refusal this door always had (the fence is the authority; the latch lives in memory).
    try:
        persisted = q.publish_queue_snapshot(reason="owner_pause_requested")
    except Exception as exc:
        log.warning("Owner pause latch of %s was not published", task_id, exc_info=True)
        return {"ok": False, "error": "pause_record_unwritable", "detail": str(exc)[:200]}
    if not persisted:
        # The durable fence already holds every launch; the queue latch lives
        # in memory until the next persisted snapshot. Disclosed, not hidden.
        log.error("Owner pause of %s: queue latch installed but its snapshot was not persisted", task_id)
    woken = _wake_members(q, root_id, fence, members, direct=(lane == "direct"))
    event = {"ts": utc_now_iso(), "type": "owner_pause_requested", "task_id": task_id,
             "root_task_id": root_id, "fence_id": fence["fence_id"], "duplicate": not created,
             "queue_latch_fence_id": latch.get("fence_id"), "members_woken": woken,
             "owner_visible": True, "toast_once": f"{task_id}:owner-pause:{fence['fence_id']}"}
    append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl", event)
    mark_tree_changed(root_id)  # what ended members left running: the off-loop pass's to stop, never this door's
    state = refresh_owner_pause_tree(root_id)
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


def refresh_owner_pause_tree(root_task_id: str, *, gaps: Optional[List[Dict[str, str]]] = None) -> str:
    """Turn the root's fence ``paused`` once the whole tree is saved and its own work stopped.

    Saved means: every member of the tree is parked — cold under its exact
    pause, or warm (its stack retained while a reviewer it launched finishes)
    — and no direct turn is live unparked. Stopped means: no member's own
    sent work is open, an ended member's included — a task-owned run whose stop is
    only requested or unknown keeps the tree ``requested`` exactly as it keeps that
    member's Resume refused (#1562; the settle tick re-reads it). Only the owner's one
    exception is disclosed instead of waited for: reviewers already launched,
    their delegated runs and their own model sends (``finishing_reviews``). A
    damaged old row no surviving link ties to this tree is named on the fence
    (``membership_gaps``), never waited for; a tied one holds it. A Resume's own
    read passes ``gaps``: such rows go there, and an already paused fence keeps
    the ones its Pause recorded. An answered root's Pause with no member and no late
    phase left is released, but only once this same census holds nothing either.
    Returns the fence state after the check (``""`` when no fence).
    """
    from ouroboros.external_runs import ENDED_MEMBER_RUNS
    from ouroboros.owner_pause import FENCE_PAUSED, FENCE_REQUESTED, read_fence, root_stopped, set_fence_state
    from supervisor.workers import direct_chat_turn

    from supervisor import queue as q
    root_task_id = str(root_task_id or "").strip()
    _note_rest(root_task_id, None)  # whatever moved this tree, its earlier rest proves nothing now
    with q._queue_lock:
        root_drive = _root_drive(q, root_task_id)
        running_all = _running_members_locked(q, root_task_id)
        running_meta = {task_id: dict(q.RUNNING.get(task_id) or {}) for task_id, _task in running_all}
        queued = [dict(item) for item in q.PENDING if isinstance(item, dict)
                  and root_task_id in (str(item.get("root_task_id") or ""), str(item.get("id") or ""))]
        parked = [item for item in queued if pause_marker(item)]
    try:
        fence = read_fence(root_drive, root_task_id)
    except Exception:
        log.debug("Owner pause fence unreadable for %s", root_task_id, exc_info=True)
        return ""
    fence_id = str(fence.get("fence_id") or "")
    warm = [task_id for task_id, _task in running_all if warm_paused_member(running_meta.get(task_id))]
    running = [(task_id, task) for task_id, task in running_all if task_id not in warm]
    direct = direct_chat_turn(root_task_id)
    signature = _tree_signature(running_meta.items(), queued, direct is not None)
    if direct is not None and _warm_paused_direct_turn(root_drive, root_task_id):
        direct = None
        warm.append(root_task_id)
    if not fence or str(fence.get("state") or "") not in {FENCE_REQUESTED, FENCE_PAUSED}:
        return str(fence.get("state") or "")
    if running or direct is not None:
        # Only a running member can move this tree now, and its park or terminal says so. A parked
        # sibling's open stop, a stop an ended member cannot take, or a direct actor (no event names
        # its warm park), keeps it due.
        if direct is None and not fence.get("finishing_reviews") and not _ended_stops_open(
                fence.get(ENDED_MEMBER_RUNS)) and (not parked and not warm or fence.get("state") == FENCE_PAUSED):
            _note_rest(root_task_id, (fence_id, signature), looked=False)  # no census read: a transition's mark stands
        return str(fence.get("state") or "")
    answered = not running_all and not queued and _late_work_settled(root_drive, root_task_id)
    if answered and root_stopped(root_drive, root_task_id):
        return _release_settled_late_work(q, root_drive, root_task_id, fence)
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.store_hot_facts import snapshot

    try:
        # The answered root's late phase still runs: the Pause ends it at its next
        # paid boundary. Its review, like every member's, finishes separately.
        if post_task_synthesis_in_flight(root_drive, root_task_id):
            return str(fence.get("state") or "")
        addressed = [(pathlib.Path(item.get("budget_drive_root") or q.DRIVE_ROOT), str(item.get("id") or ""))
                     for item in parked]
        with snapshot(addressed):  # one read for the tree; closed before anything below can write
            rows = [budget_pause_row(drive, task_id) for drive, task_id in addressed]
        if any(str(row.get("settlement") or "") == SETTLEMENT_EXTERNAL_RUNNING for row in rows):
            return str(fence.get("state") or "")
    except Exception:
        return str(fence.get("state") or "")
    from supervisor.continuation_admission import conflicting_writers

    found: List[Dict[str, str]] = []
    blockers = conflicting_writers(q, root_task_id, drive_root=root_drive, owner_pause_fence_id=fence_id, gaps=found)
    if gaps is not None:
        gaps.extend(found)
    if [b for b in blockers if not review_finishing(b)]:
        return str(fence.get("state") or "")
    if answered:
        return _release_settled_late_work(q, root_drive, root_task_id, fence)
    reviews = {str(b.get("task_id") or b.get("attempt_id") or "") for b in blockers if review_finishing(b)}
    from supervisor.tree_census import member_result_loader

    member_result = member_result_loader(root_drive, root_task_id)  # the rows this census just read
    for task_id in [root_task_id, *warm, *[str(item.get("id") or "") for item in parked]]:
        try:
            if task_id and task_has_live_review_operation(root_drive, task_id, result_loader=member_result):
                reviews.add(task_id)
        except Exception:
            log.debug("Live review operations of %s unreadable at the Pause census", task_id, exc_info=True)
    facts = {"parked_members": [str(item.get("id") or "") for item in parked] + warm,
             "warm_members": list(warm), "finishing_reviews": sorted(reviews - {""})}
    # Nothing left to observe: only a transition can move this tree. A tree with no queue row left has
    # no transition to come: its late-work release is observed, never announced.
    rest = (fence_id, signature) if (running_all or queued) and not facts["finishing_reviews"] else None
    paused = str(fence.get("state") or "") == FENCE_PAUSED
    if (found or fence.get("membership_gaps")) and not (paused and gaps is not None):
        # Rows no surviving link ties to this tree: named, never a member, never what holds it.
        from ouroboros.pause_notices import gap_fact
        facts["membership_gaps"] = gap_fact(found)
    if paused:
        if all(fence.get(key) == value for key, value in facts.items()):
            _note_rest(root_task_id, rest)
            return FENCE_PAUSED
        try:  # the "still finishing" line changes on the same paused fence
            set_fence_state(root_drive, root_task_id, fence_id=fence_id, state=FENCE_PAUSED,
                            expected_state=FENCE_PAUSED, **facts)
            _announce_finishing_reviews(root_drive, root_task_id, fence.get("finishing_reviews"),
                                        facts["finishing_reviews"])
            _note_rest(root_task_id, rest)
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
    _note_rest(root_task_id, rest)
    return FENCE_PAUSED


def _release_settled_late_work(q: Any, root_drive: Any, root_task_id: str, fence: Dict[str, Any]) -> str:
    """An answered root's Pause with nothing left: reopen its fence and drop its latch. Returns the fence state."""
    from ouroboros.owner_pause import FENCE_RELEASED, set_fence_state

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


def owner_latch_holds(q: Any, root_task_id: str, drive_root: Any = None) -> bool:
    """Whether the root's owner latch must stand though the queue holds no member: its closed fence's census (the
    one ``refresh_owner_pause_tree`` settles on) still names something other than a finishing review. A latch is what
    keeps the tree on the settle pass, so dropping it early would strand the Pause. Unknown holds; no write."""
    from ouroboros.owner_pause import fence_closed, read_fence, root_stopped
    from supervisor.continuation_admission import conflicting_writers

    root = pathlib.Path(drive_root or q.DRIVE_ROOT)
    try:
        fence = read_fence(root, root_task_id)
        return (fence_closed(fence) and not root_stopped(root, root_task_id)
                and any(not review_finishing(blocker) for blocker in conflicting_writers(
                    q, root_task_id, drive_root=root, owner_pause_fence_id=str(fence.get("fence_id") or ""))))
    except Exception:
        log.warning("Owner Pause census of %s unreadable; its latch stays", root_task_id, exc_info=True)
        return True


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
    """A late phase off the pooled lane just ended, was cut by a Pause or was stopped.

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


def complete_owner_pause_latch(root_task_id: str, fence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Queue lock held: project the queue latch of ``fence`` (the root's exact closed
    fence, just read) when none stands, as a replay of its accepted Pause would. A
    ``latch_pending`` Pause leaves only that durable fence, which every launch gate
    reads but no queue census or Resume lift sees. Returns the projected latch, or
    ``None`` when one already stood."""
    from supervisor import queue as q
    from supervisor.events_budget import _set_root_budget_pause_locked

    if root_task_id in q.BUDGET_ROOT_FENCES:
        return None
    return _set_root_budget_pause_locked(root_task_id, {
        "fence_id": str(fence["fence_id"]), "cause": "owner_pause",
        "paused_at": str(fence.get("requested_at") or utc_now_iso())})


def _run_facts(external: Any) -> list:
    """What one observation says of its runs, its date aside: only a change of these is worth a write."""
    return sorted((str(run.get("run_id") or run.get("invocation_id") or ""), str(run.get("state") or ""),
                   str(run.get("stop_outcome") or "")) for run in (external or {}).get("runs") or []
                  if isinstance(run, dict))


def _record_settled_member(q: Any, item: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> None:
    """A parked member whose own sent work was still open at its park: a fresh
    custody read re-requests the stop of every task-owned run still open
    (``stop_task_owned``: the same verb the park used, pushing ``requested`` to
    ``confirmed``) and records the member settled once none remains. Its
    started critics are observed, never cancelled.

    The custody read runs off-lock, so the write compares the pause, state
    and grant it read: a Resume, revocation or newer pause landing meanwhile
    wins, and this stale observation is dropped for the next pass to redo. ``row``:
    the pause row the pass's one read already holds.
    """
    from ouroboros.budget_pause import (
        STOP_POLICY_TASK_OWNED, BudgetPauseSuperseded, budget_pause_row, episode_fields, observe_task_runs,
        set_budget_pause, task_owned_runs_open,
    )
    from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING, SETTLEMENT_SETTLED

    task_id = str(item.get("id") or "")
    result_root = pathlib.Path(item.get("budget_drive_root") or q.DRIVE_ROOT)
    row = budget_pause_row(result_root, task_id) if row is None else row
    from ouroboros.sleep_custody import accepted_receipt
    from ouroboros.owner_pause import member_fence, fence_closed
    from types import SimpleNamespace
    cold_receipt = accepted_receipt(task_id, row)
    cold_fence = member_fence(SimpleNamespace(task_id=task_id,
        root_task_id=item.get("root_task_id") or task_id, budget_drive_root=result_root)) if cold_receipt else {}
    if str(row.get("settlement") or "") != SETTLEMENT_EXTERNAL_RUNNING and not fence_closed(cold_fence):
        return
    # The stops this pause already issued are re-READ, never issued again.
    observed = observe_task_runs(result_root, task_id, reason="owner_pause_settlement_check",
                                 stop_policy=STOP_POLICY_TASK_OWNED, prior=row.get("external_runs"))
    # This Pause's participants, never fewer. Only its first look at a cold receipt it never observed
    # begins a whole list; an owner tick an older body ran there left these marks, so that list is partial.
    seen = row.get("settlement_observed_at") or any(isinstance(run, dict) and run.get("stop_policy") == "request_stop"
                                                    for run in (row.get("external_runs") or {}).get("runs") or [])
    participants = episode_fields(row, observed, starts=bool(cold_receipt) and not seen)
    if task_owned_runs_open(observed):
        if observed.get("custody_read") == "ok" and _run_facts(observed) == _run_facts(row.get("external_runs")):
            return  # nothing moved: no rewrite of the same facts every tick
        try:  # the fresh outcomes (requested -> confirmed, unknown) are the owner's facts
            set_budget_pause(result_root, task_id, {**row, "external_runs": observed, **participants},
                             expected_pause_id=str(row.get("pause_id") or ""),
                             expected_state=str(row.get("state") or ""),
                             expected_grant_id=str((row.get("grant") or {}).get("grant_id") or ""))
        except BudgetPauseSuperseded:
            log.debug("Owner pause observation of %s superseded by a newer pause row", task_id, exc_info=True)
        return
    try:
        set_budget_pause(result_root, task_id, {**row, "external_runs": observed, **participants,
                                                **({} if cold_receipt else {"settlement": SETTLEMENT_SETTLED}),
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
    from ouroboros.budget_pause import STOP_POLICY_TASK_OWNED, episode_fields, observe_task_runs
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
    participants = episode_fields(pause, observed)  # this park's, never fewer; a park without a list stays partial
    try:
        set_owner_wait(result_root, task_id, {**wait, "owner_pause": {**pause, "external_runs": observed,
                                                                      **participants}},
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
                                                             "external_runs": observed, **participants}}


def _ended_stops_open(records: Any) -> bool:
    """Whether a stop the fence owes an ended member is not confirmed yet (``external_runs.ENDED_MEMBER_RUNS``)."""
    from ouroboros.external_runs import task_owned_runs_open

    return any(task_owned_runs_open(record.get("external_runs")) for record in (records or {}).values()
               if isinstance(record, dict))


def _stop_ended_members(q: Any, root_id: str, live: Iterable[str], stop_requested: Any = None) -> bool:
    """What the accepted Pause owes the delegated work its tree's ended members left running (#2669).

    Nothing of theirs will park, so nothing else issues this stop. The root's fence keeps each member's observation,
    and this pass goes through the one observer under ``stop_task_owned``: a first stop, then only read back, also
    what the member's own park issued before it ended (``member_stop_record``). A stop the gateway could not take yet
    (``not_issued_*``) stays owed and goes when it can; an unbound start stays named; reviewers are observed, never
    stopped. Off every lock. ``live``: the tree's queue members and direct turn, whose stacks answer for their own
    runs. A write follows a change only; an issued stop whose write failed stays this process's fact, never a second
    cancel. Returns whether anything stays owed or unread, which keeps the tree due. Unreadable custody raises.
    """
    from ouroboros import external_runs as runs
    from ouroboros.budget_pause import STOP_POLICY_TASK_OWNED, observe_task_runs
    from ouroboros.owner_pause import fence_closed, read_fence, set_fence_state
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    root_drive = _root_drive(q, root_id)
    fence = read_fence(root_drive, root_id)
    if not fence_closed(fence):
        return False
    fence_id = str(fence.get("fence_id") or "")
    with _BOOK_LOCK:
        changed = dict(_UNWRITTEN_STOPS.get((root_id, fence_id)) or {})
    records = {**(fence.get(runs.ENDED_MEMBER_RUNS) or {}), **changed}
    unsettled = {member for member, record in records.items() if runs.task_owned_runs_open(record.get("external_runs"))}
    unread = False
    for member in sorted((runs.tree_owned_custody(root_drive, root_id) | unsettled) - set(live)):
        if stop_requested is not None and stop_requested():
            unread = True
            break
        try:
            record = records.get(member)
            if record is None:
                row = load_task_result(root_drive, member, strict=True) or {}  # no row left: its owner is gone too
                if row and row.get("status") not in _TRULY_TERMINAL_STATUSES:
                    unread = True  # not ended: its own park or terminal comes first
                    continue
                record = runs.member_stop_record(row, fence_id)
            observed = observe_task_runs(root_drive, member, reason="owner_pause_ended_member",
                                         stop_policy=STOP_POLICY_TASK_OWNED, prior=record.get("external_runs"))
        except Exception:
            log.warning("Stop of what ended member %s left (tree %s) stays owed", member, root_id, exc_info=True)
            unread = True
            continue
        if observed.get("custody_read") != "ok":
            unread = True  # the kept observation still knows every stop already issued; the census names this one
        elif member not in records or _run_facts(observed) != _run_facts(records[member].get("external_runs")):
            changed[member] = records[member] = runs.ended_member_record(record, observed)
    if changed:
        try:
            current = read_fence(root_drive, root_id)
            if current.get("fence_id") == fence_id and fence_closed(current):
                set_fence_state(root_drive, root_id, fence_id=fence_id, state=current["state"],
                                expected_state=current["state"],
                                **{runs.ENDED_MEMBER_RUNS: {**(current.get(runs.ENDED_MEMBER_RUNS) or {}), **changed}})
            with _BOOK_LOCK:
                _UNWRITTEN_STOPS.pop((root_id, fence_id), None)
        except Exception:
            log.warning("Ended members' stops of %s were not recorded; the next look writes them", root_id,
                        exc_info=True)
            with _BOOK_LOCK:
                _UNWRITTEN_STOPS[(root_id, fence_id)] = changed
            unread = True
    return unread or _ended_stops_open(records)


# Memory only: when each fenced tree must be looked at. A process's first look reads every fenced tree
# once (nothing here survives it), so a tree that moved while no process watched is read by the next one.
_BOOK_LOCK = threading.Lock()
_LOOK_OWED = [True]
_CHANGED: set = set()  # trees a transition named since their last look
_AT_REST: Dict[str, tuple] = {}  # root -> (fence id, member signature) of the census that found nothing to observe
_UNWRITTEN_STOPS: Dict[tuple, dict] = {}  # (root, fence id) -> ended members' records whose fence write failed


def _tree_signature(running: Iterable[Tuple[str, Dict[str, Any]]], pending: Iterable[Dict[str, Any]],
                    direct: bool) -> tuple:
    """Which rows a tree holds and how they are parked: what a member's transition changes."""
    return (tuple(sorted((str(task_id), warm_paused_member(meta), str(meta.get("attempt")))
                         for task_id, meta in running)),
            tuple(sorted((str(item.get("id") or ""), bool(pause_marker(item))) for item in pending)), direct)


def _note_rest(root_task_id: str, rest: Optional[tuple], *, looked: bool = True) -> None:
    """``looked``: the census that found this rest is the look a transition's mark asked for. A transition landing
    during it moves the member signature kept here, so the tree comes due all the same."""
    with _BOOK_LOCK:
        if rest is None:
            _AT_REST.pop(root_task_id, None)
        else:
            _AT_REST[root_task_id] = rest
            if looked:
                _CHANGED.discard(root_task_id)


def mark_tree_changed(root_task_id: str) -> None:
    """A transition no direct census follows (a warm park): the tree is looked at on the next tick."""
    with _BOOK_LOCK:
        _CHANGED.add(str(root_task_id or ""))


def request_settlement_look() -> None:
    """The body's cadence: the next tick looks at every fenced tree, and reads only those not at rest."""
    with _BOOK_LOCK:
        _LOOK_OWED[0] = True


def settlement_look_owed(q: Any) -> bool:
    """The tick's scheduling hint, no store read: is a look owed and is there any owner Pause to look at?"""
    with _BOOK_LOCK:
        owed = _LOOK_OWED[0] or bool(_CHANGED)
    if not owed:
        return False
    if any(isinstance(fence, dict) and fence.get("cause") == "owner_pause"
           for fence in list(q.BUDGET_ROOT_FENCES.values())):
        return True
    with _BOOK_LOCK:  # no Pause stands: a later Pause's own census starts its own account
        _LOOK_OWED[0] = False
        _CHANGED.clear()
    return False


def settle_requested_owner_pauses(q: Any = None, *, stop_requested: Any = None) -> List[str]:
    """The off-loop re-check of owner Pauses a census did not leave at rest.

    A tree stays ``pausing`` while a member's own sent work is not yet
    proven stopped; nothing else would notice that while every member is
    parked. This pass (the look the tick owes, no second scheduler) re-reads each
    parked — cold or warm, a direct root's warm park included — member's runs and
    what ended members left (``_stop_ended_members``):
    a stop the Pause already issued is only read back, a run whose stop was
    never issued gets its one request, and a member whose custody is now clean
    is recorded settled; then ``refresh_owner_pause_tree`` decides. A tree at rest
    costs one comparison in memory. It never grants anything. Returns the roots
    its census found ``paused``.
    """
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_pause import FENCE_PAUSED
    from ouroboros.store_hot_facts import snapshot
    from supervisor.workers import direct_chat_turn

    if q is None:
        from supervisor import queue as q
    with _BOOK_LOCK:
        changed = set(_CHANGED)
        _CHANGED.clear()
        _LOOK_OWED[0] = False
    with q._queue_lock:
        latches = {str(root_id): str(fence.get("fence_id") or "") for root_id, fence in q.BUDGET_ROOT_FENCES.items()
                   if isinstance(fence, dict) and fence.get("cause") == "owner_pause"}
        running: Dict[str, list] = {root_id: [] for root_id in latches}
        pending: Dict[str, list] = {root_id: [] for root_id in latches}
        for task_id, meta in q.RUNNING.items() if latches else ():
            task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
            for root_id in {str(task.get("root_task_id") or ""), str(task_id)} & latches.keys():
                running[root_id].append((str(task_id), dict(meta)))
        for item in q.PENDING if latches else ():
            if isinstance(item, dict):
                for root_id in {str(item.get("root_task_id") or ""), str(item.get("id") or "")} & latches.keys():
                    pending[root_id].append(dict(item))
    due = []
    with _BOOK_LOCK:
        for stale in set(_AT_REST) - latches.keys():
            _AT_REST.pop(stale, None)
        for stale in [key for key in _UNWRITTEN_STOPS if latches.get(key[0]) != key[1]]:
            _UNWRITTEN_STOPS.pop(stale, None)
        rest = dict(_AT_REST)
    for root_id, fence_id in latches.items():
        signature = _tree_signature(running[root_id], pending[root_id], direct_chat_turn(root_id) is not None)
        if root_id in changed or rest.get(root_id) != (fence_id, signature):
            due.append(root_id)
    parked = {root_id: [item for item in pending[root_id] if pause_marker(item)] for root_id in due}
    addressed = {(str(item.get("id") or "")): pathlib.Path(item.get("budget_drive_root") or q.DRIVE_ROOT)
                 for items in parked.values() for item in items}
    rows: Dict[str, Any] = {}
    if addressed:
        with snapshot([(drive, task_id) for task_id, drive in addressed.items()]):
            for task_id, drive in addressed.items():  # one read for the pass; closed before anything writes
                try:
                    rows[task_id] = budget_pause_row(drive, task_id)
                except Exception as exc:
                    rows[task_id] = exc
    settled: List[str] = []
    looked: set = set()
    try:
        for root_id in due:
            if stop_requested is not None and stop_requested():
                break
            looked.add(root_id)
            warm = [(task_id, meta) for task_id, meta in running[root_id] if warm_paused_member(meta)]
            try:
                warm.extend(_warm_direct_root(q, root_id))
            except Exception:
                log.warning("Warm Pause of direct root %s unreadable; it stays pausing", root_id, exc_info=True)
            # One member's failure never skips its siblings or the tree's own census.
            for item in parked[root_id]:
                if stop_requested is not None and stop_requested():
                    return settled
                try:
                    row = rows[str(item.get("id") or "")]
                    if isinstance(row, Exception):
                        raise row
                    _record_settled_member(q, item, row)
                except Exception:
                    log.warning("Owner pause settlement re-check of %s (tree %s) failed; it stays pausing",
                                item.get("id"), root_id, exc_info=True)
            for task_id, meta in warm:
                if stop_requested is not None and stop_requested():
                    return settled
                try:
                    _reread_warm_member(q, task_id, meta)
                except Exception:
                    log.warning("Warm Pause re-check of %s (tree %s) failed; it stays pausing", task_id, root_id,
                                exc_info=True)
            live = {task_id for task_id, _meta in running[root_id]}
            live |= {str(item.get("id") or "") for item in pending[root_id]}
            owed = True  # an unread tail is never "nothing owed"
            try:
                owed = _stop_ended_members(q, root_id, live | ({root_id} if direct_chat_turn(root_id) else set()),
                                           stop_requested)
            except Exception:
                log.warning("Ended members of %s unreadable; their stops stay owed", root_id, exc_info=True)
            try:
                if refresh_owner_pause_tree(root_id) == FENCE_PAUSED:
                    settled.append(root_id)
            except Exception:
                log.warning("Owner pause settlement re-check of %s failed; it stays pausing", root_id, exc_info=True)
            if owed:  # no transition will name a remote run's end: this tree stays due on the body's cadence
                _note_rest(root_id, None)
    finally:
        with _BOOK_LOCK:  # a stopped pass owes the trees it never reached
            _CHANGED.update(changed & (set(due) - looked))
    return settled
