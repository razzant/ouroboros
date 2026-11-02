"""One due schedule occurrence becomes at most one admitted root (#1315).

The occurrence lives on the EXISTING carriers — the schedule row, the task
result and the queue snapshot — never a second ledger or scheduler:

- row ``occurrence``: ``{token, task_id, due_at, fingerprint, phase: claimed|admitted}``;
- row ``hold``: ``{reason, detail, since, retry_after}`` — a typed WAIT (capacity,
  a broken or protected folder, an unknown fact; a known folder that is only absent
  starts instead, with ``workspace_availability``). A wait mints no failed root, no
  ``failure_count`` and no room message; ``manage_schedules`` and the Activity
  schedule list show it, and the mind decides whether to say anything;
- task result ``schedule_admission``: ``{schedule_id, token, due_at, status,
  dispatch: none|possible, task}`` — the frozen task an accepted occurrence runs;
- queued task ``metadata.schedule_occurrence``: identity/timing plus the matching previous-run outcome.

A pass reads signed receipts and prepares resources outside the queue+table
locks. Under the locks it rechecks signatures, occurrence identity, edits and
current controls before enqueue, a committed receipt, the advanced row and a
snapshot whose False return counts as failure. Conditional receipt writes and
final policy checks remain locked; this is no general nonblocking-I/O promise.
A failed durable step
takes the pending task back out; the next pass reconciles from durable facts
only; confirmed contention (``contention``) defers the same identity without a
hold. ``dispatch=possible`` is written BEFORE a worker receives the task, in the
launcher's one result write with the Project mark (``dispatch_projection``,
``worker_assignment._commit_launch``), so a possibly-started occurrence is never
replayed; existing orphan/custody recovery owns it. No exactly-once external
effect is promised.

A claimed occurrence with no result and no dispatch mark may prepare again once
absent from the locked live queue: enqueue alone cannot reach a worker, and
snapshot restore requires the matching receipt. The same token and task id
survive restart; no process-local witness is needed. Unreadable or foreign
results and admitted rows with missing receipts remain unknown. Dispatch
possibility lives monotonically on the row and receipt, so a first attempt whose
receipt still says none never reached a worker (``first_dispatch_unsent``); restore
holds a positive one visibly (``restore_verdict``). Retention removal reads the
receipt from the table's exact root and preserves accepted or unknown obligations.

A refusal at admission is a hold, not a stored witness. An older admission
consumes only its own firing point, never a later authored one. A custody re-run
of an already-possible occurrence is marked again even after its row is deleted
or a successor occurrence replaced it.

A recurring row that waited keeps ONE overdue occurrence (no catch-up burst) and,
once admitted, moves to the next FUTURE cron point: a delayed run may land
between two cron instants. That reading of "the row waits" is the author's.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
import json
import logging
import pathlib
import uuid
from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import utc_now_iso
from supervisor.task_ownership import TaskOwnershipRead, BUSY, CHANGED

log = logging.getLogger(__name__)

# Host-owned row fields: never authored by a template; admission mints task facts.
OCCURRENCE_FIELDS = ("occurrence", "hold", "continuation_of", "delete_requested_at",
                     "delete_requested_by", "delete_reason", "last_outcome", "previous_run_open")
_FINGERPRINT_FIELDS = ("task", "trigger", "timezone", "cron", "name", "description", "enabled", "manual_override",
                       "followup_origin", "followup_relation", "writer_kind", "writer_origin")
_SETTLED = "settled"


def _queue():
    from supervisor import queue

    return queue


def contention(error: Optional[BaseException]) -> bool:
    """A temporary acquisition gap an automatic pass defers on, never unknown authority:
    confirmed store contention (rolled back or a read) or a held table, cancellation or
    launch lock, also when a reader wrapped it. Other ``StoreBusy`` reasons, corrupt,
    foreign and policy refusals keep their own meaning."""
    from ouroboros.owner_pause import LaunchBusy
    from supervisor.workspace_preparation import store_contention

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if store_contention(error) or (isinstance(error, (LaunchBusy, TimeoutError))
                                       and getattr(error, "reason", "") == "contention"):
            return True
        error = error.__cause__
    return False


def fingerprint(record: Dict[str, Any]) -> str:
    """What the owner/agent authored for this row; an edit changes it."""
    body = {key: record.get(key) for key in _FINGERPRINT_FIELDS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def remember_claim_basis(record: Dict[str, Any]) -> None:
    """Keep a legacy claim's pre-edit basis before an existing owner changes the row."""
    if isinstance(record.get("occurrence"), dict):
        record["occurrence"] = {"fingerprint": fingerprint(record), **record["occurrence"]}


def holding(record: Dict[str, Any], now: datetime.datetime) -> bool:
    hold = record.get("hold") if isinstance(record.get("hold"), dict) else {}
    try:
        until = datetime.datetime.fromisoformat(str(hold.get("retry_after") or ""))
    except ValueError:
        return False
    return (until if until.tzinfo else until.replace(tzinfo=datetime.timezone.utc)) > now


def set_hold(record: Dict[str, Any], reason: str, detail: str = "", *, retry_sec: Optional[int] = None) -> None:
    """A typed wait on the row itself; the first ``since`` survives a repeat."""
    if retry_sec is None:
        from ouroboros.config import get_bg_wakeup_min_sec

        retry_sec = int(get_bg_wakeup_min_sec())
    now = datetime.datetime.now(datetime.timezone.utc)
    prior = record.get("hold") if isinstance(record.get("hold"), dict) else {}
    record["hold"] = {
        "reason": str(reason), "detail": str(detail)[:500],
        "since": str(prior.get("since") or now.isoformat()) if prior.get("reason") == reason else now.isoformat(),
        "retry_after": (now + datetime.timedelta(seconds=max(0, int(retry_sec)))).isoformat(),
    }


def claim(record: Dict[str, Any], due_at: str) -> Dict[str, Any]:
    """A new occurrence for a due row: one token, one task id, kept across retries."""
    record.pop("previous_run_open", None)
    record["occurrence"] = {"token": uuid.uuid4().hex[:16], "task_id": uuid.uuid4().hex[:8],
                            "due_at": str(due_at), "phase": "claimed", "claimed_at": utc_now_iso(),
                            "fingerprint": fingerprint(record)}
    return view(record)


def view(record: Dict[str, Any], stored_task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    occ = dict(record["occurrence"])
    return {"schedule_id": str(record.get("id") or ""), "record": copy.deepcopy(record),
            "fingerprint": fingerprint(record), "occurrence": occ, "stored_task": stored_task}


def _is_live(task_id: str) -> bool:
    q = _queue()
    return task_id in q.RUNNING or any(
        isinstance(row, dict) and str(row.get("id") or "") == task_id for row in q.PENDING)


def reconcile(record: Dict[str, Any], *, reads: Optional[TaskOwnershipRead] = None) -> tuple[str, Optional[Dict[str, Any]]]:
    """Decide an existing occurrence from durable facts (caller holds the locks).

    ``live`` (T pending/running), ``settled`` (occurrence done: T possibly dispatched,
    running or terminal — existing custody owns it), ``prepare`` (claimed, never
    dispatched: prepare the SAME T again), ``reconsider`` (unstarted,
    changed definition/control: evaluate the new due point), ``republish`` (accepted but not in flight
    and never dispatched: re-enqueue the frozen task), or ``hold`` (unknown or
    missing accepted evidence — never replayed on absence alone), or ``retry``
    (the prepared receipt changed; observe again next pass)."""
    occ = record["occurrence"]
    task_id, token = str(occ.get("task_id") or ""), str(occ.get("token") or "")
    if occ.get("dispatch") == "possible":
        return _SETTLED, None
    if _is_live(task_id):
        return "live", None
    try:
        result = _read_back(task_id, reads=reads)
    except ReceiptChanged:
        return "retry", None
    except Exception as exc:
        if contention(exc):
            raise  # the same occurrence waits a pass; contention is not unreadable evidence
        set_hold(record, "occurrence_result_unreadable", f"{type(exc).__name__}: {exc}")
        return "hold", None
    admission = result.get("schedule_admission") if isinstance(result.get("schedule_admission"), dict) else {}
    ours = (bool(admission) and str(admission.get("token") or "") == token
            and admission.get("schedule_id") == record.get("id")
            and (admission.get("task") or {}).get("id") == task_id)
    if result and not ours:
        set_hold(record, "occurrence_task_conflict", f"task {task_id} result belongs to another admission")
        return "hold", None
    if ours and (admission.get("dispatch") == "possible" or str(result.get("status") or "") != "scheduled"):
        return _SETTLED, None
    if ours and isinstance(admission.get("task"), dict):
        task = copy.deepcopy(admission["task"])
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        timing = metadata.get("schedule_occurrence")
        if (isinstance(timing, dict) and timing.get("token") == token
                and timing.get("schedule_id") == record.get("id")):
            # Legacy accepted tasks froze identity before timing was projected.
            # Enrich host facts only; neither today's template nor clock is evidence.
            for key, value in (("due_at", admission.get("due_at") or occ.get("due_at")),
                               ("claimed_at", occ.get("claimed_at"))):
                if not timing.get(key) and value:
                    timing[key] = value
        for key in ("_owner_hold", "_consciousness_continuation"):
            if key in result:
                task[key] = copy.deepcopy(result[key])
        return "republish", task
    if _unstarted_claim(occ, result):
        if (not record.get("enabled", True) or
                occ.get("fingerprint", fingerprint(record)) != fingerprint(record)):
            return "reconsider", None  # unstarted work: the NEW definition must be due
        return "prepare", None
    set_hold(record, "occurrence_evidence_missing",
             f"occurrence {token} (phase={occ.get('phase') or 'unknown'}) has no usable "
             f"admission receipt for task {task_id}; execution history is unknown")
    return "hold", None


def _unstarted_claim(occ: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """Under the table transaction, only a receipt-less claim can be re-prepared.

    Admission holds this transaction through receipt/row writes; dispatch must
    retain a matching receipt and mark before a worker receives the task.
    """
    return occ.get("phase") == "claimed" and not result and occ.get("dispatch") != "possible"


def owed(record: Dict[str, Any], *, drive_root: Optional[pathlib.Path] = None,
         reads: Optional[TaskOwnershipRead] = None) -> Optional[bool]:
    """Whether the row carries an ACCEPTED occurrence that is neither in flight nor
    settled — work its deletion would silently drop. ``None``: missing, unreadable
    or conflicting evidence cannot establish that obligation."""
    occ = record.get("occurrence") if isinstance(record.get("occurrence"), dict) else None
    if occ is None or occ.get("dispatch") == "possible":
        return False
    try:
        result = _read_back(str(occ.get("task_id") or ""), drive_root=drive_root, reads=reads)
    except Exception:
        return None
    if _unstarted_claim(occ, result):
        return False
    admission = result.get("schedule_admission") if isinstance(result.get("schedule_admission"), dict) else {}
    if admission.get("token") != occ.get("token"):
        return None
    if admission.get("dispatch") == "possible" or str(result.get("status") or "") in {
            "completed", "failed", "cancelled", "infra_failed"}:
        return False
    return True if (admission.get("dispatch") == "none" and result.get("status") == "scheduled"
                    and isinstance(admission.get("task"), dict)) else None


def work_settled(record: Dict[str, Any], *, drive_root: Optional[pathlib.Path] = None,
                 reads: Optional[TaskOwnershipRead] = None) -> bool:
    """Occurrence settlement is not task settlement: keep its control row until both.

    A receipt-less claim positively never started, and a row that never admitted
    anything owes nothing: neither needs a queue observation to prove settlement.
    Otherwise the admitted task must be absent from the queue and its result
    settled. Task rows have no GC, so an absent result is no proof of settlement:
    bare absence without an occurrence is only a legacy compatibility case.
    Prepared reads retain owner_pause's refusal for authority seen by this
    process's fence reads or recorded in quarantine. Quarantine remains evidence
    across restarts. This refusal and unreadable or unknown evidence keep the row.
    """
    from supervisor.queue_schedules import _schedule_running_or_queued
    from ouroboros.task_status import SETTLED_STATUSES

    root = drive_root if drive_root is not None else _queue().DRIVE_ROOT
    occ = record.get("occurrence") if isinstance(record.get("occurrence"), dict) else {}
    tid = str(occ.get("task_id") or record.get("last_task_id") or "")
    try:
        if occ and _unstarted_claim(occ, _read_back(str(occ.get("task_id") or ""), drive_root=root, reads=reads)):
            # Only an earlier occurrence's task can be unsettled.
            occ, tid = {}, str(record.get("last_task_id") or "")
            if not tid:
                return True
        elif not occ and not tid and not record.get("completed_at"):
            return True
        if _schedule_running_or_queued(str(record.get("id") or ""), root, task_id=tid) is not False:
            return False
        if not tid:
            return not occ
        result = _read_back(tid, drive_root=root, reads=reads)
        return (not occ) if not result else result.get("status") in SETTLED_STATUSES
    except Exception:
        return False


def deletion_settled(record: Dict[str, Any], *, drive_root: Optional[pathlib.Path] = None,
                     reads: Optional[TaskOwnershipRead] = None) -> bool:
    """Whether withdrawal/GC has settled work: nothing retained still needs the row.

    An independent schedule's row has no role once its accepted run starts. A
    continuation (related, or an unrecorded published relationship) is the
    task's Stop/Restart release and ``scheduled_start`` binding until it settles.
    """
    from supervisor.followup_policy import relation_kind

    return relation_kind(record) == "independent" or work_settled(record, drive_root=drive_root, reads=reads)


def repair_followup_binding(record: Dict[str, Any], *, reads: Optional[TaskOwnershipRead] = None) -> bool:
    """Supervisor-only convergence of a resolved, already fired, unrun occurrence.

    The canonical frozen receipt commits before the live row and
    snapshot. Unqueued work is rebound by prepare and its admission write instead.
    A partial publication stays non-dispatchable at bound_task; another
    tick completes the SAME identity. No missing/possible receipt buys a rebind.
    The caller holds queue/table and origin control locks.
    """
    from supervisor.followup_policy import bind_task, relation_kind

    q = _queue()
    tid = str((record.get("occurrence") or {}).get("task_id") or record.get("last_task_id") or "")
    live = next((t for t in q.PENDING if str(t.get("id") or "") == tid), None)
    if not tid or relation_kind(record) == "unknown":
        return False
    try:
        stored = _read_back(tid, reads=reads)
        if live is None:
            return False  # busy/changed evidence still defers before claiming new work
        receipt = stored.get("schedule_admission") or {}
        frozen = receipt.get("task") or {}
        token = receipt.get("token")
        occ = record.get("occurrence") or {}
        if (not token or receipt.get("schedule_id") != record.get("id")
                or frozen.get("id") != tid or stored.get("status") != "scheduled"
                or receipt.get("dispatch") != "none" or occ.get("dispatch") == "possible"
                or (occ and (occ.get("task_id") != tid or occ.get("token") != token))
                or tid in q.RUNNING):
            return False
        if live is not None and (live.get("admitted_dispatch") != "none" or
                (live.get("metadata") or {}).get("schedule_occurrence") !=
                (frozen.get("metadata") or {}).get("schedule_occurrence")):
            return False
        rebound = copy.deepcopy(frozen)
        bind_task(q.DRIVE_ROOT, rebound, record)
        if rebound != frozen or any((stored.get(key) or {}) != (rebound.get(key) or {}) for key in ("metadata", "deadline_at", "task_contract")):
            item = {"task": rebound, "occurrence": {"token": token, "task_id": tid,
                    "due_at": receipt.get("due_at")}, "schedule_id": record["id"]}
            if not _write_receipt(item, record):
                return False
        if live is not None:
            before = copy.deepcopy(live)
            bind_task(q.DRIVE_ROOT, live, record)
            if live != before and q.persist_queue_snapshot(reason="followup_binding_repaired") is not True:
                live.clear()
                live.update(before)
                return False
        return True
    except Exception as exc:
        if isinstance(exc, ReceiptChanged) or contention(exc):
            raise  # unread is not unbound: the caller defers this row, never claims past it
        log.warning("Fired follow-up %s retains its binding hold", tid, exc_info=True)
        return False


def discard_claim(record: Dict[str, Any]) -> None:
    """Drop only unaccepted work; the audited retention pass owns row removal."""
    record.pop("occurrence", None)
    record.pop("hold", None)


def settle(record: Dict[str, Any]) -> None:
    """The occurrence is over (dispatched/ran/ended): the row may claim its next one."""
    occ = record.pop("occurrence", {}) or {}
    record.pop("hold", None)
    record["last_task_id"] = str(occ.get("task_id") or record.get("last_task_id") or "")
    if occ.get("phase") == "claimed":  # a dispatched claim whose row never advanced
        _advance(record, occ)


def _advance(record: Dict[str, Any], occ: Dict[str, Any]) -> None:
    """Admission's row effect: one-shot consumed; cron to the next FUTURE point."""
    from supervisor.schedule_time import next_cron_time, parse_schedule_time, timezone_for_schedule

    trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
    once = str(trigger.get("type") or "cron") == "once"
    due = trigger.get("run_at") if once else record.get("next_run_at")
    tz = timezone_for_schedule(record)
    due = parse_schedule_time(due, tz)
    if due is None or due != parse_schedule_time(occ.get("due_at"), tz):
        # An accepted receipt can outlive a still-claimed row after a failed
        # write. Consume only its firing point, never a later authored one.
        return
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(tz)
    record["last_run_at"] = now.isoformat()
    record["last_task_id"] = str(occ.get("task_id") or "")
    record["last_error"] = ""
    record.pop("hold", None)
    if once:
        record.update(enabled=False, completed_at=now.isoformat(), next_run_at="")
        return
    try:
        record["next_run_at"] = next_cron_time(str(trigger.get("expr") or record.get("cron") or ""), now).isoformat()
    except Exception as exc:
        record["last_error"] = f"{type(exc).__name__}: {exc}"


# --- prepare (NO locks held) -------------------------------------------------

# Memory only: a fresh claim's preparation that admission deferred on contention, by
# occurrence identity. The next pass resolves everything again and borrows only its
# forked drive, for the same claim, authored fingerprint, folder and Project (``_bind``),
# so contention never forks the task drive twice; admission still reads every control,
# Project, billing and allowance fact fresh. Restart forgets it: the durable claim then
# prepares the same identity again.
_DEFERRED: Dict[tuple, Dict[str, Any]] = {}


def _occurrence_key(item: Dict[str, Any]) -> tuple:
    occ = item["occurrence"]
    return item["schedule_id"], str(occ.get("token") or ""), str(occ.get("task_id") or "")


def retain_deferred(rows: List[Dict[str, Any]]) -> None:
    """Forget kept preparations whose occurrence no row of the table still names."""
    live = {_occurrence_key({"schedule_id": str(row.get("id") or ""), "occurrence": row["occurrence"]})
            for row in rows if isinstance(row, dict) and isinstance(row.get("occurrence"), dict)}
    for key in set(_DEFERRED) - live:
        _DEFERRED.pop(key, None)


def prepare(claimed: Dict[str, Any]) -> Dict[str, Any]:
    """Build the occurrence's task and resolve its resource, off every lock.

    Always resolved fresh (room folder, explicit folder, usability, Project basis); a
    deferred preparation of the same claim and fingerprint only lends its fork, once."""
    from supervisor.queue_schedules import load_schedule_store
    restart = (load_schedule_store(_queue().DRIVE_ROOT).get("followup_restart") or {}).get("control_id", "")
    claimed = {**claimed, "prepared_restart_episode": restart}
    key = _occurrence_key(claimed)
    if claimed.get("stored_task") or (_DEFERRED.get(key) or {}).get("fingerprint") != claimed["fingerprint"]:
        _DEFERRED.pop(key, None)
    try:
        return _prepare(claimed)
    finally:
        _DEFERRED.pop(key, None)


def _prepare(claimed: Dict[str, Any]) -> Dict[str, Any]:
    if claimed.get("stored_task"):
        from supervisor.followup_policy import bind_task

        task = copy.deepcopy(claimed["stored_task"])
        # Already accepted recovery freezes its resource and original incarnation.
        # Repair only host relationship facts here; the admission writer commits
        # them before publication, without invalidating this pass's signed read.
        try:
            bind_task(_queue().DRIVE_ROOT, task, claimed["record"])
        except ValueError as exc:
            return {**claimed, "task": task, "hold": ("followup_binding_unavailable", str(exc))}
        return {**claimed, "task": task, "project_admission": task.get("_project_admission")}
    from supervisor.queue_schedules import _task_from_schedule

    record, occ = claimed["record"], claimed["occurrence"]
    try:
        task = _task_from_schedule(record, task_id=str(occ["task_id"]))
    except (ValueError, TypeError) as exc:
        return {**claimed, "task": {}, "hold": ("followup_binding_unavailable", str(exc))}
    if task.get("chat_id") is None:
        return {**claimed, "task": task, "hold": ("owner_chat_unknown", "the owner destination is not currently known")}
    task["metadata"]["schedule_basis"] = {
        "author": (record.get("followup_relation") or {}).get("declared_by") or record.get("source") or "unknown",
        "origin": copy.deepcopy(record.get("followup_origin") or {}),
    }
    task["metadata"]["schedule_occurrence"] = {
        "schedule_id": claimed["schedule_id"], "token": occ["token"],
        "due_at": str(occ.get("due_at") or ""), "claimed_at": str(occ.get("claimed_at") or ""),
    }
    previous_id = str(record.get("last_task_id") or "")
    if previous_id:
        outcome = record.get("last_outcome") or {}
        task["metadata"]["schedule_occurrence"].update(
            previous_task_id=previous_id,
            previous=(copy.deepcopy(outcome) if outcome.get("task_id") == previous_id
                      else {"task_id": previous_id, "state": "pending_observation"}),
        )
    try:
        hold, basis = resolve_resource(task, record)
    except Exception as exc:
        hold, basis = ("resource_resolution_failed", f"{type(exc).__name__}: {exc}"), None
    return {**claimed, "task": task, "hold": hold, "binding_basis": basis,
            "project_admission": task.get("_project_admission")}



def _window(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The consciousness allowance, read exactly (a ledger lock) immediately BEFORE the
    admission locks — never under the queue lock, and never the stale read of a long
    prepare: the same freshness the ordinary enqueue door gives every wake."""
    q = _queue()
    if not q._consciousness_root(task):
        return None
    from ouroboros.consciousness_allowance import allowance_window

    return allowance_window(q.DRIVE_ROOT)


def _origin_intent(template: Dict[str, Any], record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Recover a legacy follow-up only from positive producer workspace/intent facts.
    A Main chat address alone says nothing about its selected resource."""
    from ouroboros.task_results import load_task_result

    origin = str((record.get("followup_origin") or {}).get("task_id") or
                 (template.get("metadata") or {}).get("origin_task_id") or record.get("origin_task_id") or "")
    if not origin:
        return None
    try:
        result = load_task_result(_queue().DRIVE_ROOT, origin, strict=True) or {}
    except Exception:
        return None
    if not result:
        return None
    intent = result.get("resource_intent") or (result.get("metadata") or {}).get("resource_intent")
    if isinstance(intent, dict) and intent.get("kind") in {"system_repo", "explicit_none", "explicit_resource", "room_default"}:
        return {**copy.deepcopy(intent), "recovered_from": origin}
    workspace = (result.get("task_contract") or {}).get("workspace") or {}
    root = str(result.get("workspace_root") or (result.get("metadata") or {}).get("workspace_root")
               or (workspace.get("root") if workspace.get("mode") == "external" else "") or "")
    if root:
        return {"kind": "explicit_resource", "root": root, "recovered_from": origin}
    if workspace.get("mode") == "system_repo":
        return {"kind": "system_repo", "recovered_from": origin}
    return None  # Main addressing and missing workspace never establish resource intent.


def resolve_resource(task: Dict[str, Any], record: Dict[str, Any]) -> tuple[Optional[tuple], Optional[str]]:
    """Bind the occurrence's resource from its producer-stamped INTENT.

    ``room_default``: the room's CURRENT folder (a folderless room stays folderless:
    task scratch is only its default cwd). ``explicit_resource``: the recorded folder,
    revalidated. A known folder that is merely absent now is still bound: the ordinary
    occurrence starts with that address and its availability fact (#1799, owner choice
    A), so another folder is never substituted; a broken or forbidden address waits.
    ``explicit_none``: folderless even if the room has a folder now. ``system_repo``
    and owner/skill templates: unchanged. Returns ``(hold, binding_basis)``."""
    from ouroboros.dialogue_provenance import presence_metadata_binding
    from ouroboros.workspace_admission import resolve_room_workspace, unavailable_workspace_address
    from supervisor import workers

    def bind_or_hold(root: str, error: str, address: str) -> Optional[tuple]:
        missing = unavailable_workspace_address(address, system_repo_dir=workers.REPO_DIR,
                                                drive_root=_queue().DRIVE_ROOT) if error else ""
        if error and not missing:
            return ("workspace_unusable", error)
        return _bind(task, missing or root, unavailable=error if missing else "")

    template = record.get("task") if isinstance(record.get("task"), dict) else {}
    meta = task["metadata"]
    if presence_metadata_binding(meta) is not None:
        return None, None  # Presence keeps its frozen binding (queue_schedules)
    intent = template.get("metadata", {}).get("resource_intent") if isinstance(template.get("metadata"), dict) else None
    intent = dict(intent) if isinstance(intent, dict) and intent.get("kind") else _origin_intent(template, record)
    if intent is None:
        return ("resource_intent_unknown", "the follow-up carries no recorded resource intent and its "
                "origin record does not prove one; reschedule it from the work it continues"), None
    meta["resource_intent"] = intent
    kind = str(intent.get("kind") or "")
    from ouroboros.projects_registry import project_admission_view, task_project_membership

    try:
        project_id, known_room = task_project_membership(_queue().DRIVE_ROOT, task)
        if project_id:
            task["project_id"] = project_id
            task["_project_admission"] = project_admission_view(
                _queue().DRIVE_ROOT, project_id, allow_unregistered=not known_room,
                frozen=kind != "room_default")
    except Exception as exc:
        return (getattr(exc, "reason", "registry_unreadable"), str(exc)), None
    if kind == "room_default":
        if not project_id:
            return ("resource_intent_unknown", "room_default intent names no project"), None
        room = task["_project_admission"]["project"]
        basis = str(room.get("working_dir") or "").strip()
        if not basis or str(room.get("lifecycle") or "active") != "active":
            return None, ""  # folderless (task scratch as default cwd); a closed room meets the admission fence
        root, error = resolve_room_workspace(drive_root=_queue().DRIVE_ROOT, system_repo_dir=workers.REPO_DIR,
                                             project_id=project_id, project_admission=task["_project_admission"])
        return bind_or_hold(root, error, basis), basis
    if kind == "explicit_resource":
        address = str(intent.get("root") or "")
        if not address:
            return ("workspace_unusable", "the recorded explicit_resource intent names no folder"), None
        root, error = resolve_room_workspace(drive_root=_queue().DRIVE_ROOT, system_repo_dir=workers.REPO_DIR,
                                             project_id="", explicit_workspace=address)
        return bind_or_hold(root, error, address), None
    return None, None  # explicit_none, system_repo, template: nothing to bind


def _bind(task: Dict[str, Any], root: str, *, unavailable: str = "") -> None:
    """The same external-workspace admission promotion performs (folder, forked
    memory drive, bounded preflight, the workspace block). This occurrence's deferred
    preparation (``_DEFERRED``) lends its existing fork only to the same folder and Project.
    ``unavailable`` is why a kept address is absent: it rides the metadata and the block."""
    if not root:
        return None
    from ouroboros.headless import prepare_task_drive
    from ouroboros.workspace_admission import bounded_workspace_preflight, compose_workspace_block

    q = _queue()
    task.update(workspace_root=root, workspace_mode="external", memory_mode="forked")
    if not str(task.get("project_id") or "").strip():
        from ouroboros.projects_registry import project_scope_admission

        task["_project_admission"] = project_scope_admission(q.DRIVE_ROOT, workspace_root=root)
        task["project_id"] = task["_project_admission"]["project_id"]
    occ = task["metadata"].get("schedule_occurrence") or {}
    kept = (_DEFERRED.get((str(occ.get("schedule_id") or ""), str(occ.get("token") or ""), str(task["id"])))
            or {}).get("task") or {}
    if (kept.get("workspace_root") == root and kept.get("drive_root") and pathlib.Path(kept["drive_root"]).is_dir()
            and str(kept.get("project_id") or "") == str(task.get("project_id") or "")):
        child = pathlib.Path(kept["drive_root"])
    else:
        child = prepare_task_drive(q.DRIVE_ROOT, str(task["id"]), "forked", project_id=str(task.get("project_id") or ""))
    if child is not None:
        task.update(drive_root=str(child), budget_drive_root=str(q.DRIVE_ROOT))
    preflight = bounded_workspace_preflight(root)
    task["metadata"].update(workspace_root=root, workspace_preflight=preflight)
    if unavailable:
        task["metadata"]["workspace_availability"] = {"state": "missing", "observed_at": utc_now_iso(),
                                                      "detail": unavailable[:500]}
    task["text"] = (f"{task['text']}\n\n[HEADLESS_WORKSPACE]\n"
                    + compose_workspace_block(workspace_root=root, workspace_mode="external",
                                              memory_mode="forked", workspace_preflight=preflight,
                                              unavailable=unavailable)
                    + "[END_HEADLESS_WORKSPACE]")
    return None


# --- admit (queue + table locks) ---------------------------------------------

def _admission_closed(stop: Optional[Callable[[], bool]]) -> bool:
    """Under Q: the owning generation is stopping, or a returning stop (planned or owner
    Restart, a managed update's writer gate) already named the queue it returns. A later
    admission would boot as unacknowledged work held for Resume, so the claim waits."""
    from supervisor.workers import repo_writer_admission_closed

    return bool((stop is not None and stop()) or _queue().RETURNS_NAMED or repo_writer_admission_closed())


def admit(prepared: List[Dict[str, Any]], *, stop: Optional[Callable[[], bool]] = None) -> None:
    """Recheck, enqueue, receipt, row, snapshot — each failure leaves the durable facts
    the next ``reconcile`` needs, and no half-admitted task in the queue.

    Root billing and the allowance are read before Q. A closed generation or named
    return queue (``_admission_closed``) admits nothing more: the claim stays
    unsubmitted for the same identity. Confirmed contention defers that item with
    no hold, keeping a fresh claim's preparation for the next pass."""
    q = _queue()
    from supervisor.queue_schedules import (
        ScheduleStoreUnreadable, _write_scheduled_tasks, load_schedule_store, schedule_transaction,
    )

    if stop is not None and stop():
        return
    windows, billing = {}, {}
    for p in prepared:
        if not p.get("hold"):
            windows[p["schedule_id"]] = _window(p["task"])
            try:
                billing[p["schedule_id"]] = q.prepare_root_billing(p["task"])
            except Exception as exc:
                log.warning("Root billing for scheduled %s is unavailable; it waits a pass",
                            p["schedule_id"], exc_info=not contention(exc))
                if contention(exc):
                    break  # the money store is held: the rest wait a pass too, without another wait each
    reads = prepare_receipts([p["record"] for p in prepared], drive_root=q.DRIVE_ROOT)

    def defer(item: Dict[str, Any]) -> None:
        if not item.get("stored_task") and not item.get("hold"):
            _DEFERRED[_occurrence_key(item)] = item

    with contextlib.ExitStack() as locked:
        try:
            locked.enter_context(schedule_transaction(q.DRIVE_ROOT))
        except Exception as exc:
            if contention(exc):  # a held table: every fresh preparation waits a pass intact
                for item in prepared:
                    defer(item)
            raise
        try:
            data = load_schedule_store(q.DRIVE_ROOT)
        except ScheduleStoreUnreadable:
            log.error("Scheduled task store became unreadable before admission; nothing admitted")
            return
        rows = {str(row.get("id") or ""): row for row in data.get("tasks") or []}
        admitted: List[str] = []
        changed = False
        for item in prepared:
            record = rows.get(item["schedule_id"])
            occ = item["occurrence"]
            current = record.get("occurrence") if isinstance(record, dict) else None
            if (not isinstance(current, dict) or current.get("token") != occ["token"]
                    or current.get("task_id") != occ["task_id"]):
                continue  # deleted, or another pass already moved this row on
            if _admission_closed(stop):
                break
            if not item.get("hold") and item["schedule_id"] not in billing:
                defer(item)
                continue
            try:
                verdict, stored = reconcile(record, reads=reads)  # signed facts revalidated under Q
            except Exception as exc:
                if not contention(exc):
                    raise
                defer(item)
                continue
            if verdict == "retry":
                defer(item)
                continue
            changed = True
            if verdict == "reconsider":
                discard_claim(record)
                continue
            if verdict not in {"prepare", "republish"}:
                continue
            if (verdict == "republish") != bool(item.get("stored_task")) or stored != item.get("stored_task"):
                continue  # accepted/held facts changed: prepare their frozen task on the next pass
            fresh_claim = verdict == "prepare"
            if fresh_claim and (not record.get("enabled", True) or fingerprint(record) != item["fingerprint"]):
                discard_claim(record)  # disabled/edited during prepare: future only
                continue
            if item.get("hold"):
                set_hold(record, *item["hold"])
                continue
            if ("prepared_restart_episode" in item and item["prepared_restart_episode"] !=
                    (data.get("followup_restart") or {}).get("control_id", "")):
                record["followup_wait"] = "owner_restart_in_progress"
                continue  # Retain the exact owed claim; re-prepare next pass, without a future hold.
            from supervisor.followup_policy import control_guard, refresh_policy
            prepared_task, enqueued = item["task"], False
            try:
                with control_guard(q.DRIVE_ROOT, record):
                    # A resolved relationship must reach the frozen receipt before
                    # policy checks its binding. This is repair of accepted, unrun
                    # work; current controls still decide whether it may enqueue.
                    if (item.get("stored_task") and prepared_task != item["stored_task"]
                            and not _write_receipt(item, record)):
                        continue
                    policy = refresh_policy(q.DRIVE_ROOT, data, record)
                    if policy.get("busy"):
                        defer(item)
                        continue
                    if policy.get("hold") or policy["wait"] or record.get("followup_hold"):
                        continue
                    current.pop("admission", None)  # retire the legacy refusal witness
                    with q.prepared_root_billing(prepared_task, resolved=billing[item["schedule_id"]]):
                        outcome = q.enqueue_task(prepared_task, consciousness_window=windows.get(item["schedule_id"]),
                                                 project_admission=item.get("project_admission"),
                                                 continuation=bool(record.get("continuation_of") or
                                                                   (item.get("stored_task") or {}).get("_consciousness_continuation")))
                    block = str(outcome.get("_admission_blocked") or "") if isinstance(outcome, dict) else ""
                    if block and outcome.get("_admission_contention"):
                        defer(item)
                        continue
                    if block:
                        _refused(record, block, outcome)
                        continue
                    enqueued = True
                    item["task"] = outcome  # preserve host-stamped continuation and admission facts
                    if not _write_receipt(item, record):
                        _unqueue([str(occ["task_id"])])
                        set_hold(record, "receipt_failed", "the admission receipt could not be written and verified")
                        continue
            except Exception as exc:
                if not contention(exc):
                    raise
                if enqueued:  # a rolled-back or unread receipt: reconcile prepares or republishes it
                    _unqueue([str(occ["task_id"])])
                item["task"] = prepared_task
                defer(item)
                continue
            if current.get("phase") == "claimed":
                _advance(record, current)
            current["phase"] = "admitted"
            record["occurrence"] = current
            admitted.append(str(occ["task_id"]))
        if not changed:
            return
        try:
            if _write_scheduled_tasks(data) is False:
                raise OSError("schedule write returned False")
        except Exception:
            _unqueue(admitted)  # the receipts stand; the rows' claims reconcile next pass
            log.error("Scheduled task store write failed after admission; admitted tasks withdrawn", exc_info=True)
            return
        if admitted and q.persist_queue_snapshot(reason="scheduled_tasks") is not True:
            _unqueue(admitted)  # not durable in the queue: republished from their receipts next pass
            log.error("Queue snapshot failed after scheduled admission; %d task(s) withdrawn", len(admitted))


def _refused(record: Dict[str, Any], block: str, outcome: Dict[str, Any]) -> None:
    detail = str(outcome.get("_admission_detail") or outcome.get("_worker_pool_disabled_reason") or "")
    trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
    if block == "project_routing_fence" and outcome.get("_project_lifecycle") == "tombstoned":
        message = f"Scheduled task was not queued: the project {outcome.get('_project_id') or ''} was deleted."
        discard_claim(record)
        record["last_error"] = message
        if str(trigger.get("type") or "cron") == "once":
            record.update(enabled=False, completed_at=utc_now_iso(), next_run_at="")
        else:
            set_hold(record, block, message, retry_sec=24 * 3600)
        return
    set_hold(record, block, detail)


def _write_receipt(item: Dict[str, Any], record: Dict[str, Any]) -> bool:
    """The accepted occurrence's durable receipt, carrying its frozen task and the
    task's own room address (including an explicitly selected hidden chat 0).

    Monotonic: it lands only over absence or over this token's own still-undispatched
    ``scheduled`` receipt — never over a row that says ``dispatch=possible``, a later
    status or another admission. The conditional writer returns the committed
    document (or the unchanged current one on refusal); verify that result without
    another full-body read under Q. Its atomic read/modify/write still holds Q."""
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

    task, occ = item["task"], item["occurrence"]
    timing = (task.get("metadata") or {}).get("schedule_occurrence") or {}
    receipt = {"schedule_id": item["schedule_id"], "token": occ["token"],
               "due_at": timing.get("due_at") or occ.get("due_at"),
               "status": "accepted", "dispatch": "none", "task": task}

    def _accept(current: Dict[str, Any], incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not current:
            return incoming
        prior = current.get("schedule_admission") if isinstance(current.get("schedule_admission"), dict) else {}
        if (str(current.get("status") or "") != STATUS_SCHEDULED or prior.get("token") != occ["token"]
                or prior.get("schedule_id") != item["schedule_id"]
                or (prior.get("task") or {}).get("id") != task["id"] or prior.get("dispatch") != "none"):
            return None
        for key in ("_owner_hold", "_consciousness_continuation"):
            if key in current:
                task[key] = copy.deepcopy(current[key])
        return incoming

    try:
        stored = write_task_result(
            _queue().DRIVE_ROOT, str(task["id"]), STATUS_SCHEDULED, _field_projector=_accept,
            strict_existing_dict=True,
            root_task_id=str(task["id"]), actor_id="scheduler", delegation_role="root",
            chat_id=task.get("chat_id"), project_id=str(task.get("project_id") or ""),
            description=str(task.get("description") or task.get("text") or ""),
            expected_output=str(task.get("expected_output") or ""), constraints=str(task.get("constraints") or ""),
            context=str(task.get("context") or ""), deadline_at=str(task.get("deadline_at") or ""),
            allowed_resources=task.get("allowed_resources") if isinstance(task.get("allowed_resources"), dict) else {},
            task_contract=task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {},
            result="Scheduled task queued.", metadata=dict(task.get("metadata") or {}),
            schedule_id=item["schedule_id"], schedule_name=str(record.get("name") or ""),
            schedule_admission=receipt,
            **{key: task[key] for key in ("title", "suggested_name") if task.get(key)},
            **({"reasoning_effort": task["reasoning_effort"]} if task.get("reasoning_effort") else {}),
            **({"billing_group": task["metadata"]["billing_group"]}
               if (task.get("metadata") or {}).get("billing_group") else {}),
            **{key: task[key] for key in ("_owner_hold", "_consciousness_continuation") if key in task})
        if not isinstance(stored, dict):
            return False
    except Exception as exc:
        if contention(exc):
            raise  # rolled back or committed-but-unreturned: ``reconcile`` decides next pass
        log.warning("Scheduled admission receipt failed for %s", task.get("id"), exc_info=True)
        return False
    got = stored.get("schedule_admission") if isinstance(stored.get("schedule_admission"), dict) else {}
    landed = (stored.get("task_id") == task["id"] and str(stored.get("status") or "") == STATUS_SCHEDULED
              and got == receipt)
    if landed and task.get("suggested_name"):
        from supervisor.worker_chat_lane import _broadcast_task_named

        # The live card is born named (pooled progress frames carry no name); replay reads the document.
        _broadcast_task_named({"type": "task_named", "task_id": str(task["id"]), "suggested_name": str(task["suggested_name"])})
    return landed


class ReceiptChanged(ValueError):
    """A prepared receipt no longer describes this pass; defer without a durable hold."""


def prepare_receipts(records: List[Dict[str, Any]], *, drive_root: pathlib.Path) -> TaskOwnershipRead:
    """One pass's addressed receipts, outside Q; errors remain errors at consumption.

    Body and handoff writers advance the signature. Display-only task-card writes
    do not, but none of these decisions consumes their line/counter fields.
    """
    from ouroboros.store import acquisition_budget

    reads = TaskOwnershipRead(drive_root)
    task_ids = {str(tid) for row in records for tid in
                ((row.get("occurrence") or {}).get("task_id"), row.get("last_task_id")) if tid}
    for tid in sorted(task_ids):
        try:
            with acquisition_budget(0):  # an optional observation defers on contention
                reads.load(tid)
        except Exception:
            pass  # TaskOwnershipRead retains the exact failure, never an absent row.
    return reads


def _read_back(task_id: str, *, drive_root: Optional[pathlib.Path] = None,
               reads: Optional[TaskOwnershipRead] = None) -> Dict[str, Any]:
    """Strict receipt, or a prepared body after a zero-wait signature check under Q.

    A changed/unprepared identity waits a pass; unreadable is never absence. Other
    callers keep their strict direct read and its existing locking contract.
    """
    from ouroboros.task_results import load_task_result

    if reads is not None:
        if task_id in reads.errors:
            raise reads.errors[task_id]
        if task_id not in reads.rows:
            raise ReceiptChanged(task_id)
        state = reads.recheck((task_id,))
        if state == CHANGED:
            raise ReceiptChanged(task_id)
        if state == BUSY:
            from ouroboros.store import StoreBusy
            raise StoreBusy("schedule receipt signature busy", reason="contention")
        return reads.prepared(task_id)
    return load_task_result(drive_root if drive_root is not None else _queue().DRIVE_ROOT, task_id, strict=True) or {}


def _unqueue(task_ids: List[str]) -> None:
    q = _queue()
    wanted = set(task_ids)
    if wanted:
        q.PENDING[:] = [row for row in q.PENDING if not (isinstance(row, dict) and str(row.get("id") or "") in wanted)]


def _occurrence_of(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    occ = (task.get("metadata") or {}).get("schedule_occurrence") if isinstance(task.get("metadata"), dict) else None
    return occ if isinstance(occ, dict) else None


def record_dispatch_possible(task: Dict[str, Any]) -> bool:
    """Before a worker receives a schedule-born task: durably say it MAY run now (alone).

    Strict and verified by reading the canonical file back; ``False`` means do not
    dispatch (the task stays pending). The launcher writes the same receipt mark in
    one transaction with the Project mark (``worker_assignment._commit_launch``).
    Tasks without an occurrence are untouched (``True``)."""
    from supervisor.task_admission import commit_dispatch_marks

    projection = dispatch_projection(task)
    if projection is None:
        return True
    if commit_dispatch_marks(task, [projection]) is not True:
        return False
    try:
        return mark_occurrence_row(task, redispatch=projection.redispatch)
    except Exception:
        return False


def dispatch_projection(task: Dict[str, Any], *, row: str = "") -> Optional[Any]:
    """The schedule half of a launch's dispatch mark; None for a task without an occurrence.

    Only this token's receipt, not yet terminal, is marked — a terminal or foreign
    result is never re-dispatched through here. A custody re-queue of an
    already-possible occurrence (worker retry, budget resume) is marked again: that
    re-run is custody's decision, not a replay. ``row`` is ``occurrence_row``'s
    reading: a successor's row refuses a FIRST handoff before anything is written.
    """
    from types import SimpleNamespace
    from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_RUNNING, STATUS_SCHEDULED

    occ = _occurrence_of(task)
    if occ is None:
        return None
    token, mark = str(occ.get("token") or ""), uuid.uuid4().hex[:12]
    schedule_id = str(occ.get("schedule_id") or "")
    projection = SimpleNamespace(token=token, mark=mark, redispatch=False)

    def project(current: Dict[str, Any], _incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        admission = current.get("schedule_admission") if isinstance(current.get("schedule_admission"), dict) else {}
        if current.get("_owner_hold"):
            task["_owner_hold"] = copy.deepcopy(current["_owner_hold"])
            return None
        if (task.get("_owner_hold") or not token or not schedule_id
                or str(admission.get("token") or "") != token
                or not isinstance(admission.get("task"), dict)
                or admission["task"].get("id") != task.get("id")
                or str(admission.get("schedule_id") or "") != schedule_id
                or str(current.get("status") or "") not in {STATUS_SCHEDULED, STATUS_RUNNING, STATUS_INTERRUPTED}):
            return None
        projection.redispatch = admission.get("dispatch") == "possible"
        if row == "other" and not projection.redispatch:
            return None  # a first handoff needs its own row (custody may re-run an old one)
        return {"schedule_admission": {**admission, "dispatch": "possible", "dispatch_at": utc_now_iso(),
                                       "dispatch_mark": mark},
                "status": STATUS_RUNNING, "result": "Assigned to a worker."}

    def verify(stored: Dict[str, Any]) -> bool:
        got = stored.get("schedule_admission") if isinstance(stored.get("schedule_admission"), dict) else {}
        return (got.get("token") == token and got.get("dispatch_mark") == mark
                and str(stored.get("status") or "") == STATUS_RUNNING)

    projection.project, projection.verify = project, verify
    return projection


def occurrence_row(task: Dict[str, Any]) -> str:
    """Before the mark, under the caller's table hold: ``ours`` (the row still names this
    occurrence), ``absent`` (deleted: the frozen receipt alone suffices) or ``other``."""
    from supervisor.queue_schedules import load_schedule_store, schedule_transaction

    occ = _occurrence_of(task) or {}
    with schedule_transaction(_queue().DRIVE_ROOT):
        rows = load_schedule_store(_queue().DRIVE_ROOT).get("tasks") or []
    row = next((r for r in rows if r.get("id") == occ.get("schedule_id")), None)
    if row is None:
        return "absent"
    return "ours" if (row.get("occurrence") or {}).get("token") == occ.get("token") else "other"


def mark_occurrence_row(task: Dict[str, Any], *, redispatch: bool) -> bool:
    """After the receipt says ``possible``: the row's monotonic copy, read back.

    Custody may resume an old task after its occurrence retired, without editing a
    successor; a first handoff needs its row. A deleted schedule does not
    invalidate its frozen admitted task: the receipt is independently sufficient."""
    from supervisor.queue_schedules import _write_scheduled_tasks, load_schedule_store, schedule_transaction

    occ = _occurrence_of(task) or {}
    token = str(occ.get("token") or "")
    with schedule_transaction(_queue().DRIVE_ROOT):
        data = load_schedule_store(_queue().DRIVE_ROOT)
        row = next((r for r in data["tasks"] if r.get("id") == occ.get("schedule_id")), None)
        if row is not None:
            current = row.get("occurrence") or {}
            if current.get("token") != token:
                if redispatch:
                    occ["dispatch"] = "possible"
                return redispatch
            current["dispatch"] = "possible"
            if _write_scheduled_tasks(data) is False:
                return False
            saved = load_schedule_store(_queue().DRIVE_ROOT)
            if not any((r.get("occurrence") or {}).get("token") == token and
                       (r.get("occurrence") or {}).get("dispatch") == "possible" for r in saved["tasks"]):
                return False
    occ["dispatch"] = "possible"
    return True



def restore_allowed(task: Dict[str, Any]) -> bool:
    """Snapshot restore revives a schedule-born pending row only on ``replay``."""
    return restore_verdict(task) == "replay"


def restore_verdict(task: Dict[str, Any]) -> str:
    """Snapshot restore's reading of one schedule-born pending row (``_restore_reading``)."""
    return _restore_reading(task)[0]


def _restore_reading(task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """``replay``: the receipt is readable, ours, still ``scheduled`` and never marked
    dispatchable. ``possible``: the receipt or row positively says a handoff may have
    happened and the task has not settled — restore holds it visibly, never replays
    it. ``busy``: confirmed contention — read again by the occurrence reconcile, or for
    a first attempt the snapshot calls possible, by the Project custody restore gates it with.
    ``settled`` (it ran) and ``unknown`` (missing, unreadable, foreign) are not
    replayed: custody owns a started run, and the scheduler's reconcile holds an
    unprovable one. A row without an occurrence replays unless it names a schedule."""
    occ = _occurrence_of(task)
    if occ is None:
        return ("unknown" if (task.get("metadata") or {}).get("schedule_id") else "replay"), {}
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    try:
        existing = load_task_result(_queue().DRIVE_ROOT, str(task.get("id") or ""), strict=True) or {}
    except Exception as exc:
        return ("busy" if contention(exc) else "unknown"), {}
    admission = existing.get("schedule_admission") if isinstance(existing.get("schedule_admission"), dict) else {}
    if (not occ.get("token") or admission.get("token") != occ.get("token")
            or admission.get("schedule_id") != occ.get("schedule_id")):
        return "unknown", existing
    frozen = admission.get("task")
    if not isinstance(frozen, dict) or frozen.get("id") != task.get("id"):
        return "unknown", existing
    for key in ("_owner_hold", "_consciousness_continuation"):
        if key in existing or key in frozen:
            task[key] = copy.deepcopy(existing.get(key, frozen.get(key)))
    if str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES:
        return "settled", existing
    if occ.get("dispatch") == "possible" or admission.get("dispatch") == "possible":
        return "possible", existing
    try:
        from supervisor.queue_schedules import load_schedule_store

        rows = load_schedule_store(_queue().DRIVE_ROOT).get("tasks") or []
    except Exception as exc:
        return ("busy" if contention(exc) else "unknown"), existing
    if any((r.get("occurrence") or {}).get("token") == occ.get("token") and
           (r.get("occurrence") or {}).get("dispatch") == "possible" for r in rows):
        return "possible", existing
    return ("replay" if str(existing.get("status") or "") == "scheduled" and admission.get("dispatch") == "none"
            else "unknown"), existing


def first_dispatch_unsent(task: Dict[str, Any]) -> bool:
    """Whether a schedule-born row's own receipt proves its FIRST handoff never happened.

    Every launcher marks that receipt before any put, and the mark is monotonic, so a
    first attempt with no continuation carrier, no retry lineage and a ``replay``
    receipt was never sent — even when an earlier launcher's queue row (or Project
    mark of attempt 1) already said ``possible`` before a later step failed. Restore
    and hold release then treat it as never started under the existing stop/return
    authority. Rows without an occurrence, retries, continuations and unread or
    foreign receipts stay unknown."""
    if not first_attempt(task):
        return False
    verdict, existing = _restore_reading(task)
    marked = existing.get("admitted_dispatch_attempt")
    return (verdict == "replay" and not existing.get("started_at")
            and (marked is None or type(marked) is int and marked == 1))


def first_attempt(task: Dict[str, Any]) -> bool:
    """A schedule-born FIRST attempt: no continuation carrier and no retry lineage."""
    return not (_occurrence_of(task) is None or task.get("_attempt", 1) not in (None, 1)
                or "_working_recovery" in task
                or any(task.get(key) for key in ("original_task_id", "timeout_retry_from", "_owner_wait_resume",
                                                 "_budget_pause_resume", "_terminalization_retry")))
