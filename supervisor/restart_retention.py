"""What a stop may retain: the ONE predicate every shutdown consumer shares.

Manual Restart, Panic, graceful shutdown and a crash-storm kill all end this
server generation's processes. None of them may turn a SAVED pause into a
terminal. An acknowledged Restart returns eligible saved work and its previously
runnable queue; existing pauses and holds remain (owner S1, 2026-10-08).
The Restart's cancel census, ``kill_workers``'s running
and pending handling, the snapshot restore and the unused-grant revocation all
read the facts below, so no consumer can disagree about which task is a saved
pause — the census that skipped a paused id and the kill that then terminalized
it was exactly the failure this module exists to prevent.

The durable pause row on the task result stays the authority (``budget_pause``
row, whatever its ``reason``). A hash-validated warm ``owner_wait`` sleep can
transfer to that owner atomically, marking its old wait ``retained``. This
adds no store or scheduler; quiz/review waits never transfer by analogy.

- ``saved``: a stored continuation source for the live attempt (``pausing``
  after quiescence, or ``paused``) and no active cancel intent — Stop wins.
- ``unused_grant``: a Resume grant the loop never consumed over such a source;
  the stop revokes it back to the pause (the next Resume mints a fresh one).
- ``request_only``: ``pausing`` without a source — pause INTENT that never
  reached its checkpoint. The stop interrupts it; it is never an exact Resume.
- ``""``: anything else, including a consumed grant (running work) and an
  unreadable row (no pause is invented from an unknown).

The acknowledged restart transaction selects eligible work and runnable queue
rows for return. Other accepted rows are retained under ``owner_restart_hold``
on the existing non-dispatch carrier (``events_budget.BUDGET_HOLD_KEY``).
Quit and application crashes retain work for explicit Resume. Prior holds stay
held across every door; a restart does not grant new execution authority.
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

RETAIN_SAVED = "saved"
RETAIN_UNUSED_GRANT = "unused_grant"
PAUSE_REQUEST_ONLY = "request_only"
RETAIN_SLEEP = "saved_sleep"
RETAIN_OWNER_PAUSE = "saved_owner_pause_park"
HOLD_SAVED_SLEEP_RECOVERY = "saved_sleep_recovery"


def saved_sleep_hold_reason(result_root: Any, *, owner_restart: bool = False) -> str:
    """Read the existing stop carriers; an ordinary recovery asserts no owner action.

    Kill callers read this only after signalling workers. The explicit Restart
    door also supplies its known intent, independent of a flag read.
    """
    from supervisor.events_budget import HOLD_OWNER_RESTART, HOLD_PANIC

    state = pathlib.Path(result_root) / "state"
    if owner_restart or (state / "owner_restart_no_resume.flag").exists():
        return HOLD_OWNER_RESTART
    try:
        if (state / "panic_stop.flag").read_text(encoding="utf-8").strip() == "panic":
            return HOLD_PANIC
    except OSError:
        pass
    return HOLD_SAVED_SLEEP_RECOVERY


def panic_recorded(result_root: Any) -> bool:
    """The owner's Panic stop flag (``server_control.execute_panic_stop``): nothing returns by itself."""
    try:
        return (pathlib.Path(result_root) / "state" / "panic_stop.flag").read_text(encoding="utf-8").strip() == "panic"
    except OSError:
        return False


def pause_retention(result_root: Any, task_id: str, attempt: Optional[int] = None) -> str:
    """Classify ``task_id``'s durable pause row for a stop (see module docstring)."""
    from ouroboros.budget_pause import (
        STATE_PAUSED, STATE_PAUSING, STATE_RESUME_GRANTED, budget_pause_row,
    )

    try:
        from ouroboros.owner_wait import saved_sleep_checkpoint

        wait, _state = saved_sleep_checkpoint(pathlib.Path(result_root), str(task_id), attempt)
        if wait and not _cancel_intent_active(result_root, task_id):
            return RETAIN_SLEEP
        if _owner_pause_park(result_root, task_id, attempt) and not _cancel_intent_active(result_root, task_id):
            return RETAIN_OWNER_PAUSE
    except Exception:
        log.debug("Warm sleep source unreadable for %s at a stop", task_id, exc_info=True)
    try:
        row = budget_pause_row(pathlib.Path(result_root), str(task_id))
    except Exception:
        log.debug("Pause row unreadable for %s at a stop", task_id, exc_info=True)
        return ""
    if not row:
        return ""
    state = str(row.get("state") or "")
    if attempt is not None and int(row.get("task_attempt") or 0) != int(attempt):
        return ""
    if not row.get("source_ref"):
        return PAUSE_REQUEST_ONLY if state == STATE_PAUSING else ""
    if _cancel_intent_active(result_root, task_id):
        return ""
    if state in {STATE_PAUSING, STATE_PAUSED}:
        return RETAIN_SAVED
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    if state == STATE_RESUME_GRANTED and not grant.get("consumed_at"):
        return RETAIN_UNUSED_GRANT
    return ""


def _owner_pause_park(result_root: Any, task_id: str, attempt: Optional[int]) -> Dict[str, Any]:
    """A WARM owner-Pause park (its stack died with this stop): its exact wait row, else ``{}``."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    row = load_task_result(pathlib.Path(result_root), str(task_id), strict=True) or {}
    wait = row.get("owner_wait") if isinstance(row.get("owner_wait"), dict) else {}
    if (row.get("status") in _TRULY_TERMINAL_STATUSES or wait.get("state") != "waiting"
            or wait.get("reason") != "owner_pause" or not wait.get("source_ref")
            or (attempt is not None and int(wait.get("task_attempt") or 0) != int(attempt))):
        return {}
    return wait


def retain_owner_pause_park(root: pathlib.Path, task_id: str, attempt: int) -> Dict[str, Any]:
    """Move a warm owner-Pause park to the exact cold owner pause its tree's Resume reads.

    Its reviewers died with the process (their custody keeps what they had);
    the saved cognition is copied unchanged under the new pause identity and the
    Pause's fence still stands, so nothing returns until the owner's Resume.
    """
    import json
    import uuid

    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
    from ouroboros.budget_pause import (
        RAIL_OWNER_PAUSE, REASON_OWNER, STATE_PAUSED, budget_pause_row, set_budget_pause, task_owned_runs_open,
    )
    from ouroboros.deadline_utils import parse_deadline_ts

    wait = _owner_pause_park(root, task_id, attempt)
    if not wait:
        raise ValueError("owner_pause_park_unavailable")
    state = json.loads(read_actor_source_bytes(root, task_id, wait["source_ref"]))
    pause_meta = wait.get("owner_pause") if isinstance(wait.get("owner_pause"), dict) else {}
    parked_at = parse_deadline_ts(pause_meta.get("parked_at") or wait.get("parked_at"))
    external = pause_meta.get("external_runs") if isinstance(pause_meta.get("external_runs"), dict) else {
        "runs": [], "custody_read": "failed", "error": "not_observed_at_warm_park"}
    old = budget_pause_row(root, task_id)
    pause_id = "owner-pause-" + uuid.uuid4().hex
    state = {**state, "pause_id": pause_id, "reason": REASON_OWNER, "rail": RAIL_OWNER_PAUSE, "scope": "root",
             "external_runs": external}
    source = store_actor_source_bytes(root, task_id, category="context_checkpoints",
        source_id="budget-pause-" + pause_id, data=json.dumps(state, ensure_ascii=False).encode(), extension="json")
    pause = {"pause_id": pause_id, "pause_generation": int(old.get("pause_generation") or 0) + 1,
             "task_attempt": attempt, "state": STATE_PAUSED, "reason": REASON_OWNER, "rail": RAIL_OWNER_PAUSE,
             "scope": "root", "root_task_id": str(pause_meta.get("root_task_id") or task_id), "source_ref": source,
             "retained_owner_wait_id": wait.get("wait_id"), "owner_fence_id": str(pause_meta.get("fence_id") or ""),
             "settlement": "external_writers_running" if task_owned_runs_open(external) else "settled",
             "external_runs": external, "resume_point": {"round_idx": int(state.get("round_idx") or 0),
                                                         "phase": "boundary", "unanswered_tool_call_ids": [],
                                                         "budget_tail": "tool", "abandoned_model_attempts": []},
             "execution_drive_root": wait.get("execution_drive_root"), "started_at": wait.get("started_at"),
             # Keep the warm interval open through cold retention; Resume folds it once.
             "paused_at": parked_at.timestamp() if parked_at else time.time(),
             "paused_duration_sec": float(wait.get("budget_paused_sec") or 0),
             "model_wait_quota_clock": wait.get("model_wait_quota_clock") or {},
             "exact_continuation": True, "auto_resume": False, "replay_safe": False}
    return set_budget_pause(root, task_id, pause, expected_owner_wait=wait,
        expected_pause_id=str(old.get("pause_id") or ""), expected_state=str(old.get("state") or ""),
        expected_grant_id=str((old.get("grant") or {}).get("grant_id") or ""))


def _cancel_intent_active(result_root: Any, task_id: str) -> bool:
    """An explicit Stop outranks a saved pause; an unreadable authority is NOT a Stop."""
    try:
        from ouroboros.cancel_intents import has_active_intent

        return bool(has_active_intent(pathlib.Path(result_root), str(task_id), strict=True))
    except Exception:
        log.warning("Cancel authority unreadable for %s at a stop; its saved pause is retained",
                    task_id, exc_info=True)
        return False


def census_without_saved_pauses(task_ids: Iterable[str], result_root: Any) -> List[str]:
    """The Restart cancel census minus every id whose pause is already saved.

    A pooled task whose checkpoint is stored but whose park event has not been
    processed yet, and a direct actor that has not released its registry entry,
    are both still "live" to the census; a cancel intent minted for them would
    outrank their saved pause at restore and cancel it.
    """
    return [task_id for task_id in task_ids
            if pause_retention(result_root, task_id) not in {RETAIN_SAVED, RETAIN_UNUSED_GRANT, RETAIN_SLEEP,
                                                             RETAIN_OWNER_PAUSE}]


def pause_ids(task: Dict[str, Any]) -> tuple:
    """The pause a queue row is parked under, and the one its grant handoff names."""
    marker, handoff = (task.get(key) if isinstance(task.get(key), dict) else {}
                       for key in ("_budget_pause", "_budget_pause_resume"))
    return str((marker.get("checkpoint") or {}).get("pause_id") or ""), str(handoff.get("pause_id") or "")


def unseen_pause(task: Dict[str, Any], before: tuple) -> bool:
    """Whether a stop or restore just parked this row under a pause the queue's fence
    map has not recorded (``before``: its ``pause_ids`` before its carriers changed):
    not the pause it was already parked under, and newer than the pause its grant
    named — or the ROOT's own unused grant back at its pause (that Resume had lifted
    the latch). A member's unused grant returns to the SAME pause: nothing new."""
    task_id = str(task.get("id") or "")
    pause_id, (parked_under, granted) = pause_ids(task)[0], before
    return bool(pause_id and pause_id != parked_under
                and (pause_id != granted or str(task.get("root_task_id") or task_id) == task_id))


def _latch_unseen_pause(task: Dict[str, Any]) -> None:
    """A stop's half of the park event it pre-empted: the tree's monetary latch is
    raised now under the normal generation rule, so the final snapshot carries what
    restore trusts. An owner marker is left to its durable owner fence. Queue lock held."""
    from ouroboros.owner_pause import REASON_OWNER
    from supervisor import queue as q
    from supervisor.events_budget import _set_root_budget_pause_locked

    marker = task.get("_budget_pause") if isinstance(task.get("_budget_pause"), dict) else {}
    root_id = str(marker.get("root_task_id") or "")
    if marker.get("scope") == "root" and root_id and marker.get("reason") != REASON_OWNER:
        # An unused grant's old marker never replaces a newer current latch.
        pause = {**marker, "fence_id": None} if q.BUDGET_ROOT_FENCES.get(root_id) else marker
        marker["fence_id"] = _set_root_budget_pause_locked(root_id, pause)["fence_id"]


def park_saved_pause(task: Dict[str, Any], attempt: int, result_root: Any, *,
                     pause_source: str, sleep_hold_reason: str = HOLD_SAVED_SLEEP_RECOVERY) -> Optional[Dict[str, Any]]:
    """Return ``task`` as a PENDING row under its exact pause marker, or ``None``.

    Shared by ``kill_workers`` (a RUNNING row whose checkpoint landed before its
    park event) and the snapshot restore (such a row found in the snapshot's
    running list). A ``pausing`` row is confirmed ``paused`` (its source already
    exists; a failed confirmation leaves it ``pausing``, still retained); an
    unused grant is revoked first, and a revocation that cannot be written
    keeps the row parked under a typed hold. Nothing here is a terminal.
    """
    from ouroboros.budget_pause import (
        STATE_PAUSED, STATE_PAUSING, STATE_RESUME_GRANTED, budget_pause_row, exact_pause_marker,
        set_budget_pause,
    )
    from supervisor.events_budget import HOLD_RESTART_REVOCATION_UNWRITTEN, hold_budget_row

    task_id = str(task.get("id") or "")
    result_root = pathlib.Path(result_root)
    retention = pause_retention(result_root, task_id, attempt)
    if retention not in {RETAIN_SAVED, RETAIN_UNUSED_GRANT, RETAIN_SLEEP, RETAIN_OWNER_PAUSE}:
        return None
    try:
        if retention == RETAIN_SLEEP:
            pause = retain_sleep_checkpoint(result_root, task_id, attempt,
                                            direct=bool(task.get("_is_direct_chat")))
        elif retention == RETAIN_OWNER_PAUSE:
            pause = retain_owner_pause_park(result_root, task_id, attempt)
        else:
            pause = budget_pause_row(result_root, task_id)
    except Exception:
        log.warning("Saved sleep retention could not be published for %s", task_id, exc_info=True)
        # Keep the validated locator for a later stop/restore retry. Never
        # terminalize a checkpoint merely because its conversion write failed.
        parked = dict(task)
        hold_after_stop(parked, result_root, sleep_hold_reason)
        return parked
    pause_id = str(pause.get("pause_id") or "")
    grant = pause.get("grant") if isinstance(pause.get("grant"), dict) else {}
    held_reason = ""
    if pause.get("state") == STATE_RESUME_GRANTED:
        if grant.get("revoked_at"):
            return None
        revoked = {**grant, "revoked_at": utc_now_iso(), "revoke_reason": "restart_before_consumption"}
        try:
            set_budget_pause(result_root, task_id, {**pause, "state": STATE_PAUSED, "grant": revoked},
                             expected_pause_id=pause_id, expected_state=STATE_RESUME_GRANTED,
                             expected_grant_id=str(grant.get("grant_id") or ""))
            pause = {**pause, "state": STATE_PAUSED, "grant": revoked}
        except Exception:
            log.warning("Stop could not revoke the unconsumed grant of %s; parked under a hold",
                        task_id, exc_info=True)
            held_reason = HOLD_RESTART_REVOCATION_UNWRITTEN
    elif pause.get("state") == STATE_PAUSING and pause.get("settlement") != "external_writers_running":
        # (An owner Pause over sent work still running stays ``pausing``: saved,
        # retained, but never confirmed as a clean Paused by a stop.)
        try:
            set_budget_pause(result_root, task_id, {**pause, "state": STATE_PAUSED,
                                                    "paused_confirmed_at": time.time(),
                                                    "pause_source": pause_source},
                             expected_pause_id=pause_id, expected_state=STATE_PAUSING)
        except Exception:
            log.warning("Parked pausing row %s stays 'pausing' (row unwritable at the stop)",
                        task_id, exc_info=True)
    parked = dict(task)
    parked["_attempt"] = int(attempt)
    parked.pop("_budget_pause_resume", None)
    parked.pop("_owner_wait_resume", None)
    parked["_budget_pause"] = exact_pause_marker(pause, default_root=str(task.get("root_task_id") or task_id))
    if retention == RETAIN_SLEEP:
        hold_after_stop(parked, result_root, sleep_hold_reason)
    if held_reason:
        hold_budget_row(
            parked, reason=held_reason,
            detail="a stop found an unconsumed Resume grant whose revocation could not be written",
            extra={"pause_id": pause_id, "grant_id": str(grant.get("grant_id") or ""),
                   "root_task_id": str(task.get("root_task_id") or task_id)},
            result_root=result_root)
    return parked


def retain_sleep_checkpoint(root: pathlib.Path, task_id: str, attempt: int, *, direct: bool = False) -> Dict[str, Any]:
    """Move a validated warm sleep to the existing single-use exact pause owner.

    Conversion is CAS-bound to the full owner-wait row. The original source is
    retained; its cognition is copied unchanged with the new pause identity.
    """
    import json
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.owner_wait import saved_sleep_checkpoint
    from ouroboros.budget_pause import budget_pause_row, set_budget_pause, STATE_PAUSED, RAIL_MODEL_SLEEP

    wait, state = saved_sleep_checkpoint(root, task_id, attempt)
    if not wait:
        raise ValueError("saved_sleep_checkpoint_unavailable")
    old = budget_pause_row(root, task_id)
    pause_id = "sleep-" + wait["wait_id"]
    state = {**state, "pause_id": pause_id, "reason": "sleep", "rail": RAIL_MODEL_SLEEP, "scope": "task"}
    source = store_actor_source_bytes(root, task_id, category="context_checkpoints",
        source_id="budget-pause-" + pause_id, data=json.dumps(state, ensure_ascii=False).encode(), extension="json")
    pause = {"pause_id": pause_id, "pause_generation": int(old.get("pause_generation") or 0) + 1,
             "task_attempt": attempt, "state": STATE_PAUSED, "reason": "sleep", "rail": RAIL_MODEL_SLEEP,
             "scope": "task", "source_ref": source, "retained_owner_wait_id": wait["wait_id"],
             "sleep": wait["sleep"], "sleep_seen": state.get("seen") or [],
             "execution_drive_root": wait.get("execution_drive_root"), "started_at": wait.get("started_at"),
             "paused_at": float(wait.get("sleep_started_at") or time.time()),
             "paused_duration_sec": float(wait.get("budget_paused_sec") or 0),
             "model_wait_quota_clock": wait.get("model_wait_quota_clock") or {},
             "is_direct_chat": direct,
             "exact_continuation": True, "auto_resume": False, "replay_safe": False}
    return set_budget_pause(root, task_id, pause, expected_owner_wait=wait,
        expected_pause_id=str(old.get("pause_id") or ""), expected_state=str(old.get("state") or ""),
        expected_grant_id=str((old.get("grant") or {}).get("grant_id") or ""))


def restart_held(task: Any) -> bool:
    """Whether this queued row carries an UNRELEASED owner-Restart hold."""
    from supervisor.events_budget import BUDGET_HOLD_KEY, HOLD_OWNER_RESTART

    hold = task.get(BUDGET_HOLD_KEY) if isinstance(task, dict) else None
    return isinstance(hold, dict) and str(hold.get("reason") or "") == HOLD_OWNER_RESTART \
        and not hold.get("selected")


def never_started(task: Any) -> bool:
    """A queued row that never dispatched: first attempt, no continuation carrier.

    Retry rows (a later attempt), owner-wait handoffs, exact pauses and grant
    carriers all describe work that already ran; they keep their own rules.
    """
    if not isinstance(task, dict) or not str(task.get("id") or ""):
        return False
    return (task.get("admitted_dispatch") == "none"
            and not task.get("_owner_wait_resume")
            and not task.get("_budget_pause_resume")
            and not task.get("_working_recovery")
            and not task.get("_terminalization_retry"))



def hold_for_owner_restart(task: Dict[str, Any], drive_root: Any) -> Dict[str, Any]:
    """Hold one never-started row for the owner's Restart (same id, no dispatch).

    A row already under another typed hold keeps that hold: it is already
    non-dispatchable and its own release rule stays the one that applies.
    """
    from supervisor.events_budget import HOLD_OWNER_RESTART

    return hold_after_stop(task, drive_root, HOLD_OWNER_RESTART)


def hold_after_stop(task: Dict[str, Any], drive_root: Any, reason: str) -> Dict[str, Any]:
    """Hold recovered sleep or an owner's stop, preserving stronger prior holds."""
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER, budget_hold_fact, hold_budget_row

    prior = budget_hold_fact(task)
    if prior is not None:
        if reason == HOLD_SAVED_SLEEP_RECOVERY or prior.get("reason") not in {
                HOLD_CONTINUATION_WRITER, HOLD_SAVED_SLEEP_RECOVERY}:
            return task
    # A Continue waiting on its predecessor's writers is held by the newer
    # Restart instead: a later reconciliation must not release what the owner's
    # Restart now holds (its own release still re-checks those writers).
    hold_budget_row(
        task, reason=reason,
        detail=("saved sleep recovered; held until an explicit Resume" if reason == HOLD_SAVED_SLEEP_RECOVERY
                else "held by the owner's stop until an explicit Resume"),
        extra={"root_task_id": str(task.get("root_task_id") or task.get("id") or ""),
               **({"prior_hold": dict(prior)} if prior else {})},
        result_root=pathlib.Path(task.get("budget_drive_root") or drive_root))
    return task


# --- the kill_workers consumers ------------------------------------------------------------
#
# ``kill_workers`` calls these under the queue lock AFTER every worker was
# signalled, so retention adds no prerequisite before a Panic's kill.

def park_saved_running_rows(running: Dict[str, Any], pending: List[Dict[str, Any]],
                            preserve: Iterable[str], drive_root: Any, *,
                            sleep_hold_reason: str = HOLD_SAVED_SLEEP_RECOVERY) -> List[str]:
    """Move every RUNNING row whose pause is saved into PENDING under its marker.

    A checkpoint stored before its park event was handled leaves the row
    RUNNING; any door would otherwise terminalize a saved pause. Returns the
    parked ids; the rest of ``RUNNING`` is ordinary interrupted work.
    """
    preserved = set(preserve or ())
    parked_ids: List[str] = []
    for task_id, meta in list(running.items()):
        if task_id in preserved or not isinstance(meta, dict) or not isinstance(meta.get("task"), dict):
            continue
        task = meta["task"]
        before = pause_ids(task)
        parked = park_saved_pause(task, int(meta.get("attempt") or task.get("_attempt") or 1),
                                  pathlib.Path(task.get("budget_drive_root") or drive_root),
                                  pause_source="stopped_during_pausing", sleep_hold_reason=sleep_hold_reason)
        if parked is None:
            continue
        if unseen_pause(parked, before):
            _latch_unseen_pause(parked)
        running.pop(task_id, None)
        if not any(isinstance(row, dict) and str(row.get("id") or "") == str(task_id) for row in pending):
            pending.append(parked)
        parked_ids.append(str(task_id))
    return parked_ids


def retained_pending(task: Dict[str, Any], *, sleep_hold_reason: str = "") -> bool:
    """A queued saved pause (an unused grant is first returned to it) or an
    earlier owner-Restart hold: kept as the same queued task on every door.
    A stop cause also holds cold sleeps and marker-less failed conversions;
    their own readiness must not wake what the owner just stopped."""
    if isinstance(task.get("_budget_pause_resume"), dict):
        from supervisor.budget_resume import revoke_exact_budget_resume

        before = pause_ids(task)
        revoke_exact_budget_resume(task, "restart_before_dispatch")
        if unseen_pause(task, before):
            _latch_unseen_pause(task)
    pause = task.get("_budget_pause")
    saved = isinstance(pause, dict) and pause.get("exact_continuation") is True
    from supervisor.events_budget import budget_hold_fact

    hold = budget_hold_fact(task)
    if sleep_hold_reason and ((saved and pause.get("reason") == "sleep"
                               and sleep_hold_reason != HOLD_SAVED_SLEEP_RECOVERY)
                              or (hold or {}).get("reason") == HOLD_SAVED_SLEEP_RECOVERY):
        from supervisor import queue as q

        hold_after_stop(task, q.DRIVE_ROOT, sleep_hold_reason)
    return saved or isinstance(pause, dict) or hold is not None



def child_of_interrupted(task: Dict[str, Any], running_ids: Iterable[str], interrupted_roots: Iterable[str]) -> bool:
    """A never-started child whose parent (or root) this stop interrupts."""
    parent_id = str(task.get("parent_task_id") or "")
    return bool(parent_id and (parent_id in set(running_ids)
                               or str(task.get("root_task_id") or "") in set(interrupted_roots)))


def hold_never_started(task: Dict[str, Any], running_ids: Iterable[str], interrupted_roots: Iterable[str]) -> bool:
    """Retain queued work on owner Restart; unknown dispatch stays unresolved.

    The legacy entry-point name is shared with kill_workers. Missing evidence
    is held without cancellation or a fabricated ``admitted_dispatch=none``;
    only positively never-started rows get the selectable Restart hold.
    """
    from supervisor import queue as q
    from supervisor.events_budget import hold_budget_row

    if any(task.get(key) for key in ("_owner_wait_resume", "_budget_pause_resume", "_terminalization_retry")):
        return False
    if never_started(task):
        hold_for_owner_restart(task, q.DRIVE_ROOT)
    elif "admitted_dispatch" in task:
        return False  # recorded possible dispatch keeps the ordinary interruption path
    else:
        hold_budget_row(task, reason="dispatch_outcome_unknown",
                        detail="an assignment may have reached a worker; reconcile its receipt before resuming",
                        result_root=pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT))
    return True


def held_ids(pending: Iterable[Any]) -> List[str]:
    return [str(task.get("id") or "") for task in pending if restart_held(task)]


# --- saved work across a stop (owner 2026-10-08: quizzes d2f7532b, a524d73f, 62a41a04) ------
#
# A UI Restart, a planned restart and a managed update RETURN the work that was
# active or runnable — but only under the fresh restart transaction the door
# prepared and the launcher (or a direct exec successor) acknowledged: that
# transaction names every returning id. Every other stop — Quit, an external
# signal, an application crash, Panic, a restart whose exit was never
# acknowledged — keeps the same saved work HELD under ``saved_work_hold`` for an
# explicit Resume, at any age. Neither lifts an earlier Pause, Stop, budget,
# sleep or deadline hold; a row without saved work keeps its old path.

def working_saved(root: Any, task_id: str, attempt: int, *, task: Optional[Dict[str, Any]] = None) -> bool:
    """Whether the attempt left a working checkpoint (a stat; validity is checked at freeze),
    or is parked on a question/review wait whose own source is this attempt's saved state."""
    from ouroboros.task_results import load_task_result
    from ouroboros.working_checkpoint import _carry_recovery, checkpoint_path

    try:
        if checkpoint_path(pathlib.Path(root), str(task_id), int(attempt)).is_file():
            return True
        if task and _carry_recovery(root, {**task, "_attempt": attempt}, task_id, task_id, attempt, "app_stop"):
            return True
        wait = (load_task_result(pathlib.Path(root), str(task_id), strict=True) or {}).get("owner_wait") or {}
        return bool(wait.get("state") == "waiting" and wait.get("reason") in {"owner", "review"}
                    and wait.get("source_ref") and int(wait.get("task_attempt") or 0) == int(attempt))
    except Exception:
        return False


def _txn():
    from ouroboros import delegate_recovery as dr

    return dr


# A kept RUNNING row's attempt died with the stop; the next boot owns it. A managed
# update keeps this process ticking until its restart, so no timeout rail may reap,
# retry or ask that dead attempt to finalize in the meantime.
RETAINED_FOR_BOOT = "retained_for_boot"

# Process-local custody for an update that may abort without a boot. Keep the
# original process handles: a retained marker (or a later pool's exit census)
# does not prove these attempts dead. The ordinary snapshot remains durable owner.
_update_returns: Dict[str, Any] = {}


def capture_update_returns(drive_root: Any, running: Dict[str, Any], workers: Dict[int, Any], *,
                           return_ids: Iterable[str]) -> Dict[str, Any]:
    """Capture this update's attempts and pre-stop eligibility, before pool teardown."""
    global _update_returns
    if any(meta.get(RETAINED_FOR_BOOT) for meta in running.values() if isinstance(meta, dict)):
        raise RuntimeError("previous update return handoff is still unresolved")
    returning = set(return_ids)
    _update_returns = {"root": pathlib.Path(drive_root), "running": running, "quiesced": False,
        "supervisor_pid": os.getpid(),
        "rows": {task_id: {"meta": meta, "attempt": int(meta.get("attempt") or
                    (meta.get("task") or {}).get("_attempt") or 1),
                    "proc": getattr(workers.get(meta.get("worker_id")), "proc", None),
                    "returning": task_id in returning}
                 for task_id, meta in running.items() if isinstance(meta, dict)}}
    return _update_returns


def recover_aborted_update(drive_root: Any, running: Dict[str, Any], pending: List[Dict[str, Any]]) -> bool:
    """Transfer only this process's quiesced update cohort to normal queue custody.

    Called with the queue lock, before opening writers; never by pool startup.
    This is explicit abort authority, not a fabricated launcher acknowledgement.
    A failed snapshot keeps the gate closed and the same successors for retry.
    """
    global _update_returns
    from supervisor import queue
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact, hold_budget_row

    batch = _update_returns
    if batch and batch["supervisor_pid"] != os.getpid():
        # A forked resolver can clear its local latch, never its parent's queue.
        # The supervisor's task_done release performs the actual handoff.
        return True
    if batch.get("running") is not running or batch.get("root") != pathlib.Path(drive_root):
        # No captured authority may adopt a row retained by a different stop.
        return not any(meta.get(RETAINED_FOR_BOOT) for meta in running.values() if isinstance(meta, dict))
    if not batch["quiesced"]:
        return False
    for task_id, meta in running.items():
        if isinstance(meta, dict) and meta.get(RETAINED_FOR_BOOT) and (
                task_id not in batch["rows"] or meta is not batch["rows"][task_id]["meta"]):
            return False
    selected = {}
    for task_id, captured in batch["rows"].items():
        meta = running.get(task_id)
        if meta is not captured["meta"] or not meta.get(RETAINED_FOR_BOOT):
            continue
        if int(meta.get("attempt") or (meta.get("task") or {}).get("_attempt") or 1) != captured["attempt"]:
            return False
        try:
            if captured["proc"] is None or captured["proc"].is_alive():
                return False
        except Exception:
            return False
        if any(task.get("id") == task_id for task in pending):
            return False  # ambiguous custody must not replace either row
        selected[task_id] = meta
    # Pause can land after the stop's first retention pass. Its exact authority
    # wins over returning working state, using the same park seam as shutdown.
    parked = park_saved_running_rows(selected, pending, (), drive_root)
    for task_id in parked:
        running.pop(task_id, None)
    returning = [] if panic_recorded(drive_root) else [task_id for task_id in selected
        if batch["rows"][task_id]["returning"]]
    recovered = {task["id"]: task for task in recover_saved_running(
        selected.values(), drive_root, transaction={"return_ids": returning}, cause="update_aborted")}
    for task_id, meta in selected.items():
        task = recovered.get(task_id)
        if task is None:
            # Never turn an unreadable source, Stop, or failed lineage into a
            # fresh prompt. Normal assignment settles Stop/terminal authority;
            # all other failures remain explicitly held with their old locator.
            task = dict(meta["task"])
            task["_attempt"] = batch["rows"][task_id]["attempt"] + 1
            task.setdefault("_working_recovery", {"source_task_id": task_id,
                "from_attempt": batch["rows"][task_id]["attempt"], "cause": "update_aborted"})
            if budget_hold_fact(task) is None:
                hold_budget_row(task, reason=HOLD_SAVED_WORK,
                    detail="update aborted; saved work recovery could not be verified",
                    result_root=pathlib.Path(task.get("budget_drive_root") or drive_root), existing_result_only=True)
        # Reuse restore admission for the original scope, lineage, deadline and
        # exact next-attempt source; recovery itself grants none of those gates.
        task.setdefault("_project_admission_restore_hold", {
            "reason": "project_routing_fence_lookup_failed",
            "detail": "Saved work from the aborted update awaits original admission verification."})
        pending.append(task)
        running.pop(task_id, None)
    if queue.persist_queue_snapshot(reason="update_abort_handoff") is not True:
        return False
    _update_returns = {}
    return True


def retain_saved_running(running: Dict[str, Any], preserve: Iterable[str], drive_root: Any) -> List[str]:
    """``kill_workers`` at an application stop: RUNNING rows whose attempt saved work and
    that no Stop claims stay RUNNING, marked ``RETAINED_FOR_BOOT``, in the final snapshot;
    the next boot returns or holds them."""
    kept, lineage = [], {}
    preserved = set(preserve or ())
    for task_id, meta in dict(running or {}).items():
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
        root = pathlib.Path(task.get("budget_drive_root") or drive_root)
        if (task and task_id not in preserved and not _cancel_intent_active(root, task_id)
                and working_saved(root, task_id, int(meta.get("attempt") or task.get("_attempt") or 1), task=task)):
            kept.append(str(task_id))
            lineage[str(task_id)] = {str(task.get("parent_task_id") or ""), str(task.get("root_task_id") or "")}
    # Invariant 19: a saved child of a RUNNING parent that is not itself kept is
    # interrupted with that parent (the ordinary path), never kept alone.
    # A separately preserved question/review handoff survives this stop too;
    # its saved children must not be classified as orphans before it is queued.
    interrupted = {str(task_id) for task_id in dict(running or {})} - preserved
    while True:
        orphaned = [task_id for task_id in kept
                    if (lineage[task_id] - {"", task_id}) & (interrupted - set(kept))]
        if not orphaned:
            for task_id in kept:
                running[task_id][RETAINED_FOR_BOOT] = True
            return kept
        kept = [task_id for task_id in kept if task_id not in orphaned]


def prepare_restart_returns(drive_root: Any, running: Dict[str, Any], pending: List[Dict[str, Any]], *,
                            transaction_id: str, direct: Iterable[Dict[str, Any]] = (),
                            owner_wait_ids: Iterable[str] = ()) -> set:
    """A returning door names what it returns before it stops anything.

    Running rows (and direct turns) with a working checkpoint and no saved
    pause, the owner-wait handoffs already prepared, and the formerly runnable
    never-started queue, including unstarted successors with a validated saved
    source and dispatch authority. Merged into the door's restart transaction (planned
    handoffs may share it); the launcher's exit acknowledgement makes it fresh.
    """
    import os as _os

    from supervisor.events_budget import budget_hold_fact
    from supervisor.task_admission import _working_resume_granted

    dr = _txn()
    returns = set()
    for task_id, meta in dict(running or {}).items():
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
        attempt = int(meta.get("attempt") or task.get("_attempt") or 1) if task else 0
        root = pathlib.Path(task.get("budget_drive_root") or drive_root)
        if (task and task_id not in set(owner_wait_ids) and working_saved(root, task_id, attempt, task=task)
                and pause_retention(root, task_id, attempt) not in {RETAIN_SAVED, RETAIN_UNUSED_GRANT, RETAIN_SLEEP,
                                                                    RETAIN_OWNER_PAUSE}):
            returns.add(str(task_id))
    for record in direct or ():
        task_id = str((record or {}).get("id") or "")
        if task_id and working_saved(pathlib.Path(record.get("budget_drive_root") or drive_root), task_id,
                                     int(record.get("_attempt") or 1), task=record):
            returns.add(task_id)
    queue_ids = []
    for task in pending or ():
        if (not isinstance(task, dict) or budget_hold_fact(task) is not None
                or isinstance(task.get("_budget_pause"), dict) or task.get("_owner_hold")):
            continue
        task_id = str(task.get("id") or "")
        root = pathlib.Path(task.get("budget_drive_root") or drive_root)
        if not task_id or _cancel_intent_active(root, task_id):
            continue
        try:
            if never_started(task) or ("_working_recovery" in task and _working_resume_granted(task, root)):
                queue_ids.append(task_id)
        except Exception:
            log.warning("Saved queued work of %s could not authorize a Restart return", task_id, exc_info=True)
    queue_ids.sort()
    row = dr._read_restart_transaction(drive_root, transaction_id) or {
        "schema": 1, "transaction_id": transaction_id, "status": "prepared", "supervisor_pid": _os.getpid(),
        "task_ids": [], "prepared_at": utc_now_iso(), "expected_exit_code": 42}
    row.update(task_ids=sorted(set(row.get("task_ids") or []) | returns | set(owner_wait_ids)),
               return_ids=sorted(returns), queue_ids=queue_ids)
    dr._write_restart_transaction(drive_root, row)
    from ouroboros.utils import atomic_write_json

    active = dr._active_restart_transaction_path(drive_root)
    active.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(active, {"transaction_id": transaction_id, "supervisor_pid": row["supervisor_pid"],
                               "prepared_at": row["prepared_at"]})
    return returns


def fresh_return_transaction(drive_root: Any) -> Dict[str, Any]:
    """The acknowledged, not yet restored restart transaction, else ``{}`` (a holding stop)."""
    import json as _json

    dr = _txn()
    try:
        dr._ack_direct_exec_successor(drive_root)
        active = _json.loads(dr._active_restart_transaction_path(drive_root).read_text(encoding="utf-8"))
        row = dr._read_restart_transaction(drive_root, str(active.get("transaction_id") or ""))
        if row.get("ack_source") == "direct_spawn_successor" and (
            not dr.direct_spawn_successor_matches(row)
            or any(active.get(key) != row.get(key) for key in ("transaction_id", "supervisor_pid", "prepared_at"))
        ):
            return {}
    except Exception:
        return {}
    return row if row.get("status") == "normal_exit_acknowledged" and not row.get("returns_restored_at") else {}


def consume_return_transaction(drive_root: Any, row: Dict[str, Any]) -> None:
    """One boot restores a transaction's returns; a later stop can never reuse it."""
    if row:
        try:
            _txn()._write_restart_transaction(drive_root, {**row, "returns_restored_at": utc_now_iso()})
        except Exception:
            log.warning("Restart transaction %s not marked restored", row.get("transaction_id"), exc_info=True)


def _recovery_source(root: pathlib.Path, task: Dict[str, Any], attempt: int, cause: str) -> bool:
    """Attach the frozen working checkpoint, or a parked question/review wait's exact source."""
    from ouroboros.task_results import load_task_result
    from ouroboros.working_checkpoint import SOURCE_OWNER_WAIT, attach_recovery, live_park

    task_id = str(task.get("id") or "")
    if live_park(root, task_id, attempt) == "owner_wait":
        wait = (load_task_result(root, task_id, strict=True) or {}).get("owner_wait") or {}
        if wait.get("state") != "waiting" or wait.get("reason") not in {"owner", "review"} or \
                int(wait.get("task_attempt") or 0) != int(attempt):
            return False
        # The wait's OWN continuation source, named as such (never posing as a rolling state).
        task["_working_recovery"] = {"source_kind": SOURCE_OWNER_WAIT, "source_ref": wait["source_ref"],
                                     "wait_id": str(wait.get("wait_id") or ""), "source_task_id": task_id,
                                     "from_attempt": int(attempt), "boundary": f"{wait['reason']}_wait",
                                     "seq": 0, "saved_at": str(wait.get("parked_at") or ""), "cause": cause,
                                     "started_at": wait.get("started_at"),
                                     "model_wait_quota_clock": wait.get("model_wait_quota_clock") or {},
                                     "budget_paused_sec": float(wait.get("budget_paused_sec") or 0)}
        return bool(task["_working_recovery"]["wait_id"])
    return attach_recovery(root, task, source_task_id=task_id, from_attempt=attempt, cause=cause,
                           prior_task={**task, "_attempt": attempt})


def recover_saved_running(rows: Iterable[Dict[str, Any]], drive_root: Any, *,
                          transaction: Dict[str, Any], cause: str) -> List[Dict[str, Any]]:
    """Interrupted RUNNING rows (and caught direct turns) whose work was saved.

    Each becomes its same-id successor carrying the frozen source: dispatchable
    when the fresh transaction returns it, otherwise held for an explicit Resume.
    A Stop (active cancel intent), a terminal or unreadable result, a final the
    host already owes, or no usable source keeps the ordinary interrupted path.
    """
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor.events_budget import HOLD_SAVED_WORK, hold_budget_row

    returning = set((transaction or {}).get("return_ids") or [])
    rows = [row for row in rows or () if isinstance(row, dict)]
    candidates: List[tuple] = []
    for row in rows:
        task = dict(row.get("task") or {})
        task_id = str(task.get("id") or "")
        attempt = int(row.get("attempt") or task.get("_attempt") or 1)
        root = pathlib.Path(task.get("budget_drive_root") or drive_root)
        try:
            stored = load_task_result(root, task_id, strict=True) or {}
            if (not task_id or not stored or stored.get("status") in _TRULY_TERMINAL_STATUSES
                    or _cancel_intent_active(root, task_id)):
                continue
            auto = task_id in returning
            source_cause = "restart" if auto and cause != "update_aborted" else cause
            if not _recovery_source(root, task, attempt, source_cause):
                continue
        except Exception:
            log.warning("Saved work of %s unreadable at restore; ordinary interrupted path", task_id, exc_info=True)
            continue
        candidates.append((task, attempt, root, auto))
    # Invariant 19: a child whose interrupted parent (or root) is NOT itself saved
    # keeps the ordinary interrupted-parent path; saved work never orphans it.
    interrupted = {str((row.get("task") or {}).get("id") or "") for row in rows}
    while True:
        kept = {str(task["id"]) for task, *_rest in candidates}
        orphaned = [item for item in candidates
                    if ({str(item[0].get("parent_task_id") or ""), str(item[0].get("root_task_id") or "")}
                        - {"", str(item[0]["id"])}) & (interrupted - kept)]
        if not orphaned:
            break
        candidates = [item for item in candidates if item not in orphaned]
    recovered: List[Dict[str, Any]] = []
    for task, attempt, root, auto in candidates:
        task_id = str(task["id"])
        task["_attempt"] = attempt + 1
        task.pop("_owner_wait_resume", None)
        task.pop("_budget_pause_resume", None)
        if not auto:
            hold_budget_row(task, reason=HOLD_SAVED_WORK, result_root=root,
                        detail=f"work saved before the application stopped ({cause}); continues after an explicit Resume",
                        extra={"root_task_id": str(task.get("root_task_id") or task_id), "stop_cause": cause})
        recovered.append(task)
    return recovered


def hold_stopped_queue(task: Dict[str, Any], drive_root: Any, cause: str) -> Dict[str, Any]:
    """Boot after a holding stop: accepted, never-started work waits for Resume at any age."""
    from supervisor.events_budget import HOLD_SAVED_WORK, hold_budget_row

    hold_budget_row(task, reason=HOLD_SAVED_WORK, result_root=pathlib.Path(task.get("budget_drive_root") or drive_root),
                    existing_result_only=True,
                    detail=f"accepted before the application stopped ({cause}); starts after an explicit Resume",
                    extra={"root_task_id": str(task.get("root_task_id") or task.get("id") or ""), "stop_cause": cause})
    return task
