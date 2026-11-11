"""A paused helper closes with the work it helped (#3047; the owner's choice of 10 Nov 2026).

When a root has truly ended, a child still parked for an explicit Resume has nobody left to resume it, yet its
Resume button would hold the work's card, room and Project at the owner's turn. ``close_abandoned_helpers`` finds
exactly those rows and asks for their closure through the one existing door: a durable cancel intent
(``source="work_finished"``) that custody settles. It runs inside the off-loop single-flight cancel-intent duty and
keeps no state between passes.

Retained on any doubt, and for these on purpose: a live or unreadable root, a live retry of it, late work of the
root still open, a live (also cold-sleeping) member of its tree, a delegated run or unknown start in the work that is no
abandoned helper's own (the root's, a finished or kept member's), an owner's closed Pause or hold, Panic, a dispatch of
unknown outcome, or an accepted Resume, grant or selection, including the consumed Resume or selection an application
stop's hold interrupted.
A technical restart or shutdown is not an ending: its helpers wait for the owner until that root's own Continue is
admitted. Continue cannot steer a predecessor's helpers, so nothing waits for any successor, whatever its state.

Only that watchdog-fed custody consumes the intent. The assignment pass leaves it, since rechecks and capture cannot run
under the queue lock. A live worker a Resume dispatched first withdraws it and keeps running; a row in no map goes to
the ordinary miss lane, so nothing is stranded. Every withdrawal reads the CURRENT intent under its lock: a Stop or
cascade that joined it made it an ordinary cancel, owed to a dispatched row as well. A restart is no custody either:
restore revives the parked row and fences a dispatched one like any other (``restore_custody_owns``).
``prepare_closure`` readies the settlement of the captured queue row for ``_finish_captured_pending``, the one writer of
a queued cancel. The row never saw a kill, so the saved output and workspace patch are taken here, before the cancelled
write; a failure gives the row and the intent back to the watchdog. The pause it ended stays on the result
(``pause_at_close``). Discovery's verdict is rechecked after that work, and the records it rests on inside the cancelled
write's own transaction, so a Pause, grant or retry landing first withdraws the closure. The helper's delegated runs are
stopped only after that write committed (``stop_closed_runs``) and before the intent settles: a restart in between
replays the open intent through the settled lane, which reconciles them behind the same durable verdict.
"""

from __future__ import annotations

import copy
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

from ouroboros.cancel_intents import SCOPE_CASCADE

log = logging.getLogger(__name__)

WORK_FINISHED = "work_finished"
# The ``cancel_origin`` reason; its fixed label beside the phrase (``cancel_publication.CANCEL_SOURCE_LABELS``).
CLOSED_REASON = "the work it helped has ended"
_RESTART_REASON_CODES = frozenset({"orphaned_running_after_worker_restart", "interrupted_retry_lost"})
_LINEAGE_FIELDS = ("delegation_role", "parent_task_id", "root_task_id", "role", "chat_id")
_PAUSE_FIELDS = ("_budget_pause", "_budget_pause_hold", "_working_recovery", "_budget_pause_resume")
# A refusal offers Resume, but these say the row's dispatch or publication is unknown, or that the owner holds it.
_KEPT_REFUSALS = frozenset({"dispatch_outcome_unknown", "continuation_publication_unconfirmed", "_owner_hold"})


def closes_only_parked_row(intent: Optional[Dict[str, Any]]) -> bool:
    """Whether this intent is a closure: only custody's settlement of the still-queued row may consume it. A Stop
    (``followup_stop``) or a cascade that joined it made it an ordinary cancel, owed to a dispatched row as well."""
    intent = intent or {}
    return (str(intent.get("source") or "") == WORK_FINISHED and not intent.get("followup_stop")
            and str(intent.get("scope") or "") != SCOPE_CASCADE)


def withdraw_closure(root: Any, task_id: str, claim: Optional[Dict[str, Any]], detail: str) -> Optional[Dict[str, Any]]:
    """Withdraw a closure against the CURRENT intent row (``cancel_intents.withdraw_intent``): ``None`` once it is
    withdrawn or already gone; else the row a Stop or cascade joined, an ordinary cancel now. A row that is no longer
    ``claim`` (taken over) raises, and the caller gives its custody back."""
    from ouroboros.cancel_intents import withdraw_intent

    claim = claim or {}
    return withdraw_intent(root, task_id, withdrawable=closes_only_parked_row, detail=detail,
                           expected_generation=claim.get("generation"), request_id=str(claim.get("request_id") or ""))


def yield_to_dispatch(q: Any, task_id: str, intent: Dict[str, Any], worker: Any,
                      was_reaping: bool) -> Union[str, Dict[str, Any]]:
    """Custody holds the live worker a Resume dispatched before the closure: the closure is withdrawn and the worker
    keeps running. A Stop or cascade that joined it comes back as the current intent, for custody to carry out."""
    from supervisor.cancel_publication import CANCEL_FAILED, CANCEL_NOT_FOUND
    from supervisor.task_lifecycle import _release_intent_claim, _restore_custody

    try:
        current = withdraw_closure(q.DRIVE_ROOT, task_id, intent, "no longer parked")
    except Exception:
        log.warning("The closure of dispatched %s was not withdrawn; the watchdog retries", task_id, exc_info=True)
        _restore_custody(task_id, worker=worker, worker_reaping=was_reaping)
        _release_intent_claim(q, task_id, error="closure withdrawal unresolved", intent=intent)
        return CANCEL_FAILED
    if current is None:
        _restore_custody(task_id, worker=worker, worker_reaping=was_reaping)
        return CANCEL_NOT_FOUND
    return current


def restore_custody_owns(drive_root: Any, task_id: str, *, running: bool = False) -> bool:
    """Restore's "cancellation custody owns this row": every active intent but a closure, which owns only the settlement
    of a still-parked row. A saved parked row returns (the next pass rechecks and closes it); a running one, dispatched
    by a Resume, has its closure withdrawn and is fenced like any other. An unreadable authority raises."""
    from ouroboros.cancel_intents import active_intent, has_active_intent

    if not has_active_intent(drive_root, task_id, strict=True):  # the ordinary restore read, unchanged
        return False
    intent = active_intent(drive_root, task_id, strict=True) or {}  # only a row with an intent reads it whole
    if not closes_only_parked_row(intent):
        return True
    return running and withdraw_closure(drive_root, task_id, intent, "dispatched before a restart") is not None


def _queue():
    from supervisor import queue

    return queue


def stored_helper_lineage(result: Dict[str, Any]) -> Dict[str, Any]:
    """The lineage a closed helper's ``task_done`` reads from its own result: its queue row is already gone."""
    return {key: result[key] for key in _LINEAGE_FIELDS if result.get(key) not in (None, "")}


def _is_child(row: Dict[str, Any]) -> bool:
    from ouroboros.task_results import resolve_task_lineage

    lineage = resolve_task_lineage(str(row.get("id") or ""), metadata=row.get("metadata"), **{key: row.get(key) for key in (
        "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
    return bool(lineage["parent_task_id"]) and not lineage["is_root_task"]


def _parked_for_resume(row: Dict[str, Any], fences: Dict[str, Any]) -> bool:
    """A queued row whose only way on is an explicit Resume nobody has accepted; every unknown fact answers False."""
    from ouroboros.budget_pause import STATE_RESUME_GRANTED, budget_pause_row
    from ouroboros.work_stop import SELECTION_KEY, pause_marker, queue_control

    q, task_id = _queue(), str(row.get("id") or "")
    # An accepted Resume's carrier lifts the stop it answered; under a standing pause marker a newer Pause spent it.
    if not pause_marker(row) and (isinstance(row.get("_budget_pause_resume"), dict)
                                  or isinstance(row.get(SELECTION_KEY), dict)):
        return False
    try:
        control = queue_control(q.DRIVE_ROOT, row, fences)
        view = control["view"]
        pause = budget_pause_row(pathlib.Path(row.get("budget_drive_root") or q.DRIVE_ROOT), task_id)
    except Exception:
        log.debug("The control over %s was not read; it stays", task_id, exc_info=True)
        return False
    return (control["phase"] == "budget_paused" and not control["owner_resumed"] and view.get("release") == "resume"
            and bool(view.get("resumable")) and not view.get("waiting")
            and str((view.get("refusal") or {}).get("reason") or "") not in _KEPT_REFUSALS
            and pause.get("state") != STATE_RESUME_GRANTED and not _resumed_when_stopped(row, pause))


def _resumed_when_stopped(row: Dict[str, Any], pause: Dict[str, Any]) -> bool:
    """An application stop interrupted explicitly resumed work, also after later recovery of that same task: the
    selection it replaced, kept in its recovery custody (``work_stop.interrupted_selection``), or the latest durable
    pause's consumed grant, which stop/recovery keeps while replacing the in-flight handoff. That accepted work is not
    abandoned on the second stop either; a new pause or stop supersedes both and ends this case."""
    from ouroboros.budget_pause import STATE_RESUMED
    from ouroboros.work_stop import hold_fact, interrupted_selection, record_policy

    source = row.get("_working_recovery") if isinstance(row.get("_working_recovery"), dict) else {}
    grant = pause.get("grant") if isinstance(pause.get("grant"), dict) else {}
    hold = hold_fact(row)
    if not (hold and record_policy(hold)["cause"] == "app_stop" and source.get("source_task_id") == row.get("id")):
        return False
    return bool(interrupted_selection(row) or (
        0 < int(pause.get("task_attempt") or 0) <= int(source.get("from_attempt") or 0)
        and pause.get("state") == STATE_RESUMED and grant.get("authority") == "explicit_resume"
        and grant.get("consumed_at") and not grant.get("revoked_at")))


def _fence_open(row: Dict[str, Any]) -> bool:
    from ouroboros.owner_pause import FENCE_RELEASED

    fence = row.get("owner_pause") or {}
    return isinstance(fence, dict) and (not fence or fence.get("state") == FENCE_RELEASED)


def _only_interrupted(row: Dict[str, Any]) -> bool:
    """A technical stop of the process, which says nothing about whether the work is over."""
    from ouroboros.outcomes import recorded_cause
    from ouroboros.owner_continue import TECHNICAL_CANCEL_SOURCES

    origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
    if origin:
        return str(origin.get("source") or "") in TECHNICAL_CANCEL_SOURCES or str(origin.get("reason") or "") == "server_shutdown"
    return recorded_cause(row) in _RESTART_REASON_CODES


def _records_say_ended(work: Dict[str, Any], leaf_id: str, leaf: Dict[str, Any]) -> bool:
    """What the root's and its retry leaf's own records say: settled with no newer attempt, both open to closure, no
    upkeep left, and after a technical stop its own Continue admitted (a bound claim may still fail, leaving these
    helpers the way on). Discovery reads them; the cancelled write reads them again in its own transaction."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_status import SETTLED_STATUSES

    checkpoint = leaf.get("root_phase_checkpoint") if isinstance(leaf.get("root_phase_checkpoint"), dict) else {}
    if (str(leaf.get("status") or "") not in SETTLED_STATUSES or str(leaf.get("superseded_by") or "").strip()
            or str(leaf.get("retry_task_id") or leaf_id).strip() != leaf_id
            or not _fence_open(work) or not _fence_open(leaf)
            or post_task_synthesis_is_open(str(checkpoint.get("post_task_synthesis") or ""))):
        return False
    claim = leaf.get("continued_by") if isinstance(leaf.get("continued_by"), dict) else {}
    return not _only_interrupted(leaf) or claim.get("state") == "admitted"


def _work_ended(q: Any, root: pathlib.Path, root_id: str, pending_ids: set, reads: Any) -> bool:
    """Whether the work under ``root_id`` is over: its records say so and nothing of its own lives or is queued."""
    from ouroboros.cancel_intents import _validated_single_cancel_target
    from ouroboros.review_operation import task_has_live_review_operation

    leaf = _validated_single_cancel_target(root, root_id)
    return (_records_say_ended(reads.load(root_id) or {}, leaf, reads.load(leaf) or {}) and not {root_id, leaf} & pending_ids
            and not any(q.task_has_live_ownership(tid, ownership=reads) for tid in {root_id, leaf})
            and not task_has_live_review_operation(root, leaf, result_loader=reads.load))


def _external_work(root: pathlib.Path) -> List[tuple]:
    """``(tree, owner)`` of every delegated run that may still be running (open, no terminal word from its engine)
    and every start of unknown outcome, from the live custody sets alone, whatever the history. The open starts are read
    first, so one that binds its run between the two reads is in the second; an unreadable set raises."""
    from ouroboros.delegate_custody_current import pending
    from ouroboros.delegate_custody_reconcile import open_runs, run_tree_id

    starts = [(str(record.get("root_task_id") or record.get("task_id") or ""), str(record.get("task_id") or ""))
              for record in pending(root)]
    return starts + [(run_tree_id(entry), str(entry.task_id or "")) for entry in open_runs(root) if not entry.terminal_state]


def _abandoned(rows: Dict[str, Dict[str, Any]], fences: Dict[str, Any], reads: Any,
               parked: Optional[List[str]] = None) -> Any:
    """Whether a task is one of this work's abandoned helpers: its queued child parked for an explicit Resume (``parked``
    when already read), or a helper already closed with the work, whose runs that closure stopped. Each answer is read once."""
    seen = {task_id: task_id in parked for task_id in rows} if parked is not None else {}

    def abandoned(task_id: str) -> bool:
        if task_id not in seen:
            row = rows.get(task_id)
            if row is not None:
                seen[task_id] = _parked_for_resume(row, fences)
            else:
                stored = reads.load(task_id) or {}
                origin = stored.get("cancel_origin") if isinstance(stored.get("cancel_origin"), dict) else {}
                seen[task_id] = stored.get("status") == "cancelled" and origin.get("source") == WORK_FINISHED
        return seen[task_id]

    return abandoned


def _helpers_alone(root_id: str, work: List[tuple], abandoned: Any) -> bool:
    """Whether every run and unknown start of the work under ``root_id`` (the cascade Stop's reach) is an abandoned
    helper's own, which its closure stops. Anyone else's, the root's or a finished member's, keeps them all."""
    owners = {owner for tree, owner in work if root_id in (tree, owner)}
    return not owners & {"", root_id} and all(abandoned(owner) for owner in owners)


def _queue_facts(q: Any) -> tuple:
    """``(fences, live roots, pending ids, held rows)`` of the queue as one observation; rows are copies."""
    from ouroboros.work_stop import record_policy
    from supervisor.queue_transitions import budget_pause_fact

    with q._queue_lock:
        rows = [dict(row) for row in q.PENDING if isinstance(row, dict)]  # read-only here, so the lock copies only the rows
        fences = copy.deepcopy(dict(q.BUDGET_ROOT_FENCES))
        live_roots = {str((meta.get("task") or {}).get("root_task_id") or tid) for tid, meta in q.RUNNING.items()
                      if isinstance(meta, dict)}
    held = []
    for row in rows:
        pause = budget_pause_fact(row, fences)
        if pause is None or record_policy(pause)["cause"] == "wait":
            live_roots.add(str(row.get("root_task_id") or row.get("id") or ""))  # queued or sleeping work is still live
        else:
            held.append(row)
    return fences, live_roots, {str(row.get("id") or "") for row in rows}, held


def close_abandoned_helpers() -> List[str]:
    """One discovery pass: mint the closure's intent for each abandoned helper; the watchdog settles it next tick."""
    from ouroboros.cancel_intents import SCOPE_SINGLE, request_cancel
    from supervisor.queue_snapshot import saved_queue_unowned
    from supervisor.restart_retention import panic_recorded
    from supervisor.task_ownership import TaskOwnershipRead

    q = _queue()
    root = pathlib.Path(q.DRIVE_ROOT)
    fences, live_roots, pending_ids, held = _queue_facts(q)
    # An idle tick reads nothing from disk: this duty runs every tick, whatever the history's size.
    if not any(_is_child(row) for row in held) or saved_queue_unowned() or panic_recorded(root):
        return []
    minted: List[str] = []
    reads, ended, rows, parked = TaskOwnershipRead(root), {}, {}, {}  # one verdict per work, its results read once
    for row in held:
        task_id, root_id = str(row.get("id") or ""), str(row.get("root_task_id") or "")
        try:
            # A root parked in this same pause rests without a read; its helpers' controls are read only once it ended.
            if not root_id or root_id == task_id or root_id in live_roots or root_id in pending_ids or not _is_child(row):
                continue
            if root_id not in ended:
                try:
                    ended[root_id] = _work_ended(q, root, root_id, pending_ids, reads)
                except Exception:
                    log.debug("The end of work %s was not established; its helpers stay", root_id, exc_info=True)
                    ended[root_id] = False
            if not ended[root_id]:
                continue
            rows.setdefault(root_id, {})[task_id] = row
            if _parked_for_resume(row, fences):
                parked.setdefault(root_id, []).append(task_id)
        except Exception:
            log.warning("Closing the abandoned helper %s was not requested", task_id, exc_info=True)
    if not parked:
        return minted
    try:
        work = _external_work(root)  # one read, shared by every ended work with a parked helper
    except Exception:
        log.debug("Delegated custody was not read; every helper stays", exc_info=True)
        return minted
    for root_id, helpers in parked.items():
        try:
            if not _helpers_alone(root_id, work, _abandoned(rows[root_id], fences, reads, parked=helpers)):
                continue
        except Exception:
            log.debug("The delegated work of %s was not established; its helpers stay", root_id, exc_info=True)
            continue
        for task_id in helpers:
            try:
                intent = request_cancel(root, task_id, reason=CLOSED_REASON, source=WORK_FINISHED, scope=SCOPE_SINGLE)
                if not intent.get("already_requested") and not intent.get("already_settled"):
                    minted.append(task_id)
            except Exception:
                log.warning("Closing the abandoned helper %s was not requested", task_id, exc_info=True)
    return minted


def _still_abandoned(q: Any, root: pathlib.Path, task: Dict[str, Any]) -> bool:
    """Discovery's whole verdict over this one captured row, from current addressed reads; any doubt is False."""
    from supervisor.queue_snapshot import saved_queue_unowned
    from supervisor.restart_retention import panic_recorded
    from supervisor.task_ownership import TaskOwnershipRead

    root_id = str(task.get("root_task_id") or "")
    try:
        fences, live_roots, pending_ids, held = _queue_facts(q)
        reads = TaskOwnershipRead(root)
        if not (root_id and root_id not in live_roots and not saved_queue_unowned() and not panic_recorded(root)
                and _parked_for_resume(task, fences) and _work_ended(q, root, root_id, pending_ids, reads)):
            return False
        rows = {str(row.get("id") or ""): row for row in held if str(row.get("root_task_id") or "") == root_id and _is_child(row)}
        return _helpers_alone(root_id, _external_work(root), _abandoned({**rows, str(task.get("id") or ""): task}, fences, reads))
    except Exception:
        log.debug("The closure of %s was not reconfirmed; it stays", task.get("id"), exc_info=True)
        return False


def _capture_workspace(q: Any, task: Dict[str, Any], task_id: str) -> str:
    """The workspace patch of a helper that ran, before its cancelled write; raises when the capture is unresolved."""
    from ouroboros.headless import _workspace_root_from_task, finalize_task_artifacts, task_is_readonly_subagent
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import ARTIFACT_TERMINAL_STATUSES
    from supervisor.restart_retention import never_started

    if never_started(task) or _workspace_root_from_task(task) is None or task_is_readonly_subagent(task):
        return "never_started"
    root = pathlib.Path(q.DRIVE_ROOT)
    if not load_task_result(root, task_id, strict=True):
        return "owed_no_result"
    finalize_task_artifacts(root, task)
    state = str((load_task_result(root, task_id, strict=True) or {}).get("artifact_status") or "").strip().lower()
    if state not in ARTIFACT_TERMINAL_STATUSES:
        raise RuntimeError(f"workspace capture left artifact status {state!r}")
    return "attempted"


def prepare_closure(q: Any, task_id: str, task: Dict[str, Any], intent: Optional[Dict[str, Any]], awaiting: Any,
                    cost_fields: Dict[str, Any]) -> Union[None, str, Dict[str, Any]]:
    """``None`` for any other cancel; this closure's verdict when it ended here; else what its one cancelled write adds,
    with the ``last_word`` that write must pass (``refused`` then says so)."""
    if not closes_only_parked_row(intent) or awaiting is not None:
        return None
    from ouroboros.budget_pause import STATE_RESUME_GRANTED
    from ouroboros.cancel_intents import _validated_single_cancel_target, claim_still_owned
    from ouroboros.headless import _child_drive_from_task, prepare_terminal_task_files, terminal_task_files_ready
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES
    from supervisor.cancel_publication import (
        CANCEL_ALREADY_SETTLED,
        CANCEL_FAILED,
        CANCEL_NOT_FOUND,
        _salvage_cancelled_output,
    )
    from supervisor.task_lifecycle import SETTLED_ALREADY, _release_intent_claim, _restore_custody, _settle_intent

    root = pathlib.Path(q.DRIVE_ROOT)

    def give_back(error: str) -> str:
        _restore_custody(task_id, pending=task)
        _release_intent_claim(q, task_id, error=error, intent=intent)
        return CANCEL_FAILED

    def withdraw() -> str:
        _restore_custody(task_id, pending=task)
        try:
            current = withdraw_closure(root, task_id, intent, "no longer abandoned")
        except Exception:
            log.debug("The closure of %s was not withdrawn; the watchdog retries", task_id, exc_info=True)
            current = {}
        if current is None:
            return CANCEL_NOT_FOUND
        return give_back("a Stop or cascade joined the closure" if current else "closure withdrawal unresolved")

    if not _still_abandoned(q, root, task):  # a Resume, the work's Pause or retry, or a stop that is not ours got here first
        return withdraw()
    try:
        stored = load_task_result(root, task_id, strict=True) or {}
        child = _child_drive_from_task(task)
        if (child is not None and not terminal_task_files_ready(root, task, stored)
                and str((load_task_result(child, task_id, strict=True) or {}).get("status") or "") in SETTLED_STATUSES):
            if prepare_terminal_task_files(root, task).get("terminal_source_present") is not True:
                raise RuntimeError("terminal file publication is unresolved")
            stored = load_task_result(root, task_id, strict=True) or {}
        if terminal_task_files_ready(root, task, stored):  # it finished by itself: that result and its event stand
            _settle_intent(q, task_id, outcome=SETTLED_ALREADY, detail=str(stored["status"]), intent=intent)
            q._emit_cancel_task_done(task, task_id, cost_fields=cost_fields, status=str(stored["status"]))
            q.persist_queue_snapshot(reason="cancel_pending")
            return CANCEL_ALREADY_SETTLED
        note, _text, _path = _salvage_cancelled_output(q, task, task_id)
        captured = _capture_workspace(q, task, task_id)
    except Exception:
        log.warning("Closing %s with its ended work did not finish; it stays parked", task_id, exc_info=True)
        return give_back("closure capture failed")
    if not _still_abandoned(q, root, task):  # those captures take time
        return withdraw()
    if not claim_still_owned(root, task_id, intent or {}):
        return give_back("claim lost before terminal write")
    root_id = str(task.get("root_task_id") or "")
    try:
        leaf_id = _validated_single_cancel_target(root, root_id)
    except Exception:
        log.debug("The retry leaf of %s was not established; %s stays", root_id, task_id, exc_info=True)
        return withdraw()
    refused: Dict[str, bool] = {}

    def still_ended(txn: Any, current: Dict[str, Any], document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Under the write lock: every Pause, grant or retry either committed before this read or comes after the closure."""
        pause = current.get("budget_pause") if isinstance(current.get("budget_pause"), dict) else {}
        if pause.get("state") != STATE_RESUME_GRANTED and _records_say_ended(txn.load(root_id) or {}, leaf_id, txn.load(leaf_id) or {}):
            return document
        refused["refused"] = True
        return None

    return {"result": f"Closed: {CLOSED_REASON}.{note}", "detail": "closed with its ended work",
            "last_word": still_ended, "refused": refused, "withdraw": withdraw,
            "lineage": {key: task[key] for key in _LINEAGE_FIELDS if task.get(key) not in (None, "")},
            "cancel_fields": {"artifact_capture": captured,
                              "pause_at_close": {key: task[key] for key in _PAUSE_FIELDS if isinstance(task.get(key), dict)}}}


def stop_closed_runs(q: Any, task_id: str) -> None:
    """The closure's cancelled write committed: only now, behind that durable verdict, are the helper's delegated runs
    stopped (irreversible), and what stayed open is disclosed on the closed result. Neither step raises; the intent is
    settled after this, so a restart before it replays through the settled lane, which reconciles again."""
    from ouroboros.delegate_terminal import record_terminal_reconciliation
    from ouroboros.task_results import STATUS_CANCELLED
    from supervisor.cancel_publication import _audit_delegated_runs_on_kill

    audit = _audit_delegated_runs_on_kill(q, task_id, deliberate_terminal=STATUS_CANCELLED)
    record_terminal_reconciliation(pathlib.Path(q.DRIVE_ROOT), task_id, audit)
