"""One due schedule occurrence becomes at most one admitted root (#1315).

The occurrence lives on the EXISTING carriers — the schedule row, the task
result and the queue snapshot — never a second ledger or scheduler:

- row ``occurrence``: ``{token, task_id, due_at, fingerprint, phase: claimed|admitted}``;
- row ``hold``: ``{reason, detail, since, retry_after}`` — a typed WAIT (capacity,
  a missing or broken folder, an unknown fact). A wait mints no failed root, no
  ``failure_count`` and no room message; ``manage_schedules`` and the Activity
  schedule list show it, and the mind decides whether to say anything;
- task result ``schedule_admission``: ``{schedule_id, token, due_at, status,
  dispatch: none|possible, task}`` — the frozen task an accepted occurrence runs;
- queued task ``metadata.schedule_occurrence``: ``{schedule_id, token, due_at, claimed_at}``.

A pass claims under the queue+table locks (briefly), prepares everything
expensive WITHOUT them (resource intent, folder checks, preflight, memory fork,
the consciousness allowance read), then rechecks the row, its token and its edit
fingerprint and admits under the locks: enqueue, a verified receipt, the row
advanced, a snapshot whose False return counts as failure. A failed durable step
takes the pending task back out; the next pass reconciles from durable facts
only. ``dispatch=possible`` is written BEFORE a worker receives the task
(``record_dispatch_possible``), so a possibly-started occurrence is never
replayed; existing orphan/custody recovery owns it. No exactly-once external
effect is promised.

A claimed occurrence with no result and no dispatch mark may prepare again once
absent from the locked live queue: enqueue alone cannot reach a worker, and
snapshot restore requires the matching receipt. The same token and task id
survive restart; no process-local witness is needed. Unreadable or foreign
results and admitted rows with missing receipts remain unknown. Dispatch
possibility lives monotonically on the row and receipt. Deletion reads the
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

import copy
import datetime
import hashlib
import json
import logging
import pathlib
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

# Host-owned row fields: never authored by a payload, never copied into a task.
OCCURRENCE_FIELDS = ("occurrence", "hold", "continuation_of", "delete_requested_at")
_FINGERPRINT_FIELDS = ("task", "trigger", "timezone", "cron", "name", "description", "enabled", "manual_override",
                       "followup_origin", "followup_relation")
_SETTLED = "settled"


def _queue():
    from supervisor import queue

    return queue


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


def reconcile(record: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]]]:
    """Decide an existing occurrence from durable facts (caller holds the locks).

    ``live`` (T pending/running), ``settled`` (occurrence done: T possibly dispatched,
    running or terminal — existing custody owns it), ``prepare`` (claimed, never
    dispatched: prepare the SAME T again), ``reconsider`` (unstarted,
    changed definition/control: evaluate the new due point), ``republish`` (accepted but not in flight
    and never dispatched: re-enqueue the frozen task), or ``hold`` (unknown or
    missing accepted evidence — never replayed on absence alone)."""
    occ = record["occurrence"]
    task_id, token = str(occ.get("task_id") or ""), str(occ.get("token") or "")
    if occ.get("dispatch") == "possible":
        return _SETTLED, None
    if _is_live(task_id):
        return "live", None
    try:
        result = _read_back(task_id)
    except Exception as exc:
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


def owed(record: Dict[str, Any], *, drive_root: Optional[pathlib.Path] = None) -> Optional[bool]:
    """Whether the row carries an ACCEPTED occurrence that is neither in flight nor
    settled — work its deletion would silently drop. ``None``: missing, unreadable
    or conflicting evidence cannot establish that obligation."""
    occ = record.get("occurrence") if isinstance(record.get("occurrence"), dict) else None
    if occ is None or occ.get("dispatch") == "possible":
        return False
    try:
        result = _read_back(str(occ.get("task_id") or ""), drive_root=drive_root)
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


def work_settled(record: Dict[str, Any], *, drive_root: Optional[pathlib.Path] = None) -> bool:
    """Occurrence settlement is not task settlement: keep its control row until both.

    A receipt-less claim positively never started, and a row that never admitted
    anything owes nothing: neither needs the queue, so a worker without a fresh
    snapshot deletes it at once. Otherwise the admitted task must be absent from
    the queue and settled; a collected result of a task no occurrence still
    names is settled history. Unreadable or unknown evidence keeps the row.
    """
    from supervisor.queue_schedules import _schedule_running_or_queued
    from ouroboros.task_status import SETTLED_STATUSES

    root = drive_root if drive_root is not None else _queue().DRIVE_ROOT
    occ = record.get("occurrence") if isinstance(record.get("occurrence"), dict) else {}
    tid = str(occ.get("task_id") or record.get("last_task_id") or "")
    try:
        if occ and _unstarted_claim(occ, _read_back(str(occ.get("task_id") or ""), drive_root=root)):
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
        result = _read_back(tid, drive_root=root)
        return (not occ) if not result else result.get("status") in SETTLED_STATUSES
    except Exception:
        return False


def deletion_settled(record: Dict[str, Any], *, drive_root: Optional[pathlib.Path] = None) -> bool:
    """Whether a deferred delete/GC may drop the row: nothing retained still needs it.

    An independent schedule's row has no role once its accepted run starts. A
    continuation (related, or an unrecorded published relationship) is the
    task's Stop/Restart release and ``scheduled_start`` binding until it settles.
    """
    from supervisor.followup_policy import relation_kind

    return relation_kind(record) == "independent" or work_settled(record, drive_root=drive_root)


def repair_followup_binding(record: Dict[str, Any]) -> bool:
    """Supervisor-only convergence of a resolved, already fired, unrun occurrence.

    The canonical frozen receipt is written/read back before the live row and
    snapshot. A partial publication stays non-dispatchable at bound_task; another
    tick completes the SAME identity. No missing/possible receipt buys a rebind.
    The caller holds queue/table and origin control locks.
    """
    from supervisor.followup_policy import bind_task, relation_kind

    q = _queue()
    tid = str((record.get("occurrence") or {}).get("task_id") or record.get("last_task_id") or "")
    if not tid or relation_kind(record) == "unknown":
        return False
    try:
        stored = _read_back(tid)
        receipt = stored.get("schedule_admission") or {}
        frozen = receipt.get("task") or {}
        token = receipt.get("token")
        occ = record.get("occurrence") or {}
        live = next((t for t in q.PENDING if str(t.get("id") or "") == tid), None)
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
    except Exception:
        log.warning("Fired follow-up %s retains its binding hold", tid, exc_info=True)
        return False


def discard_claim(record: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Drop unaccepted work and finish any earlier deferred delete atomically."""
    record.pop("occurrence", None)
    record.pop("hold", None)
    if record.get("delete_requested_at") and deletion_settled(record):
        data["tasks"] = [row for row in data.get("tasks") or [] if row is not record]
        return True
    return False


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

def prepare(claimed: Dict[str, Any]) -> Dict[str, Any]:
    """Build the occurrence's task and resolve its resource, off every lock."""
    if claimed.get("stored_task"):
        task = copy.deepcopy(claimed["stored_task"])
        # Already accepted recovery freezes its resource and original incarnation.
        # The final queue read still checks lifecycle; no new preparation launders it.
        return {**claimed, "task": task, "project_admission": task.get("_project_admission")}
    from supervisor.queue_schedules import _task_from_schedule

    record, occ = claimed["record"], claimed["occurrence"]
    try:
        task = _task_from_schedule(record, task_id=str(occ["task_id"]))
    except (ValueError, TypeError) as exc:
        return {**claimed, "task": {}, "hold": ("followup_binding_unavailable", str(exc))}
    if task.get("chat_id") is None:
        return {**claimed, "task": task, "hold": ("owner_chat_unknown", "the owner destination is not currently known")}
    task["metadata"]["schedule_occurrence"] = {
        "schedule_id": claimed["schedule_id"], "token": occ["token"],
        "due_at": str(occ.get("due_at") or ""), "claimed_at": str(occ.get("claimed_at") or ""),
    }
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
    revalidated; if gone the row waits — another folder is never substituted.
    ``explicit_none``: folderless even if the room has a folder now. ``system_repo``
    and owner/skill templates: unchanged. Returns ``(hold, binding_basis)``."""
    from ouroboros.dialogue_provenance import presence_metadata_binding
    from ouroboros.workspace_admission import resolve_room_workspace
    from supervisor import workers

    template = record.get("task") if isinstance(record.get("task"), dict) else {}
    meta = task["metadata"]
    if presence_metadata_binding(meta) is not None:
        return None, None  # Presence keeps its frozen binding (queue_schedules)
    intent = template.get("metadata", {}).get("resource_intent") if isinstance(template.get("metadata"), dict) else None
    intent = dict(intent) if isinstance(intent, dict) and intent.get("kind") else _origin_intent(template, record)
    if intent is None:
        return ("resource_intent_unknown", "the follow-up predates recorded resource intent and its "
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
        return (("workspace_unusable", error), basis) if error else (_bind(task, root), basis)
    if kind == "explicit_resource":
        root, error = resolve_room_workspace(drive_root=_queue().DRIVE_ROOT, system_repo_dir=workers.REPO_DIR,
                                             project_id="", explicit_workspace=str(intent.get("root") or ""))
        return (("workspace_unusable", error), None) if error or not intent.get("root") else (_bind(task, root), None)
    return None, None  # explicit_none, system_repo, template: nothing to bind


def _bind(task: Dict[str, Any], root: str) -> None:
    """The same external-workspace admission promotion performs (folder, forked
    memory drive, bounded preflight, the workspace block)."""
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
    child = prepare_task_drive(q.DRIVE_ROOT, str(task["id"]), "forked", project_id=str(task.get("project_id") or ""))
    if child is not None:
        task.update(drive_root=str(child), budget_drive_root=str(q.DRIVE_ROOT))
    preflight = bounded_workspace_preflight(root)
    task["metadata"].update(workspace_root=root, workspace_preflight=preflight)
    task["text"] = (f"{task['text']}\n\n[HEADLESS_WORKSPACE]\n"
                    + compose_workspace_block(workspace_root=root, workspace_mode="external",
                                              memory_mode="forked", workspace_preflight=preflight)
                    + "[END_HEADLESS_WORKSPACE]")
    return None


# --- admit (queue + table locks) ---------------------------------------------

def admit(prepared: List[Dict[str, Any]]) -> None:
    """Recheck, enqueue, receipt, row, snapshot — each failure leaves the durable facts
    the next ``reconcile`` needs, and no half-admitted task in the queue."""
    q = _queue()
    from supervisor.queue_schedules import (
        ScheduleStoreUnreadable, _write_scheduled_tasks, load_schedule_store, schedule_transaction,
    )

    windows = {p["schedule_id"]: _window(p["task"]) for p in prepared if not p.get("hold")}
    with schedule_transaction(q.DRIVE_ROOT):
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
            changed = True
            verdict, stored = reconcile(record)  # CURRENT facts, not the off-lock prepare snapshot
            if verdict == "reconsider":
                discard_claim(record, data)
                continue
            if verdict not in {"prepare", "republish"}:
                continue
            if (verdict == "republish") != bool(item.get("stored_task")) or stored != item.get("stored_task"):
                continue  # accepted/held facts changed: prepare their frozen task on the next pass
            fresh_claim = verdict == "prepare"
            if fresh_claim and (not record.get("enabled", True) or fingerprint(record) != item["fingerprint"]):
                discard_claim(record, data)  # disabled/edited during prepare: future only
                continue
            if item.get("hold"):
                set_hold(record, *item["hold"])
                continue
            from supervisor.followup_policy import control_guard, refresh_policy
            with control_guard(q.DRIVE_ROOT, record):
                policy = refresh_policy(q.DRIVE_ROOT, data, record)
                if policy.get("hold") or policy["wait"] or record.get("followup_hold"):
                    continue
                current.pop("admission", None)  # retire the legacy refusal witness
                outcome = q.enqueue_task(item["task"], consciousness_window=windows.get(item["schedule_id"]),
                                         project_admission=item.get("project_admission"),
                                         continuation=bool(record.get("continuation_of") or
                                                           (item.get("stored_task") or {}).get("_consciousness_continuation")))
                block = str(outcome.get("_admission_blocked") or "") if isinstance(outcome, dict) else ""
                if block:
                    _refused(record, block, outcome, data)
                    continue
                item["task"] = outcome  # preserve host-stamped continuation and admission facts
                if not _write_receipt(item, record):
                    _unqueue([str(occ["task_id"])])
                    set_hold(record, "receipt_failed", "the admission receipt could not be written and verified")
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


def _refused(record: Dict[str, Any], block: str, outcome: Dict[str, Any], data: Dict[str, Any]) -> None:
    detail = str(outcome.get("_admission_detail") or outcome.get("_worker_pool_disabled_reason") or "")
    trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
    if block == "project_routing_fence" and outcome.get("_project_lifecycle") == "tombstoned":
        message = f"Scheduled task was not queued: the project {outcome.get('_project_id') or ''} was deleted."
        if discard_claim(record, data):
            return
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
    status or another admission. Verified by reading the file back, not by trusting
    the writer's return value."""
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
                or prior.get("dispatch") != "none"):
            return None
        for key in ("_owner_hold", "_consciousness_continuation"):
            if key in current:
                task[key] = copy.deepcopy(current[key])
        return incoming

    try:
        written = write_task_result(
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
            **({"reasoning_effort": task["reasoning_effort"]} if task.get("reasoning_effort") else {}),
            **({"billing_group": task["metadata"]["billing_group"]}
               if (task.get("metadata") or {}).get("billing_group") else {}),
            **{key: task[key] for key in ("_owner_hold", "_consciousness_continuation") if key in task})
        if written is False:
            return False
        stored = _read_back(str(task["id"]))
    except Exception:
        log.warning("Scheduled admission receipt failed for %s", task.get("id"), exc_info=True)
        return False
    got = stored.get("schedule_admission") if isinstance(stored.get("schedule_admission"), dict) else {}
    return (str(stored.get("status") or "") == STATUS_SCHEDULED
            and got == receipt)


def _read_back(task_id: str, *, drive_root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """The result as it now is ON DISK (strict: unreadable raises, absence is ``{}``)."""
    from ouroboros.task_results import load_task_result

    return load_task_result(drive_root if drive_root is not None else _queue().DRIVE_ROOT, task_id, strict=True) or {}


def _unqueue(task_ids: List[str]) -> None:
    q = _queue()
    wanted = set(task_ids)
    if wanted:
        q.PENDING[:] = [row for row in q.PENDING if not (isinstance(row, dict) and str(row.get("id") or "") in wanted)]


def record_dispatch_possible(task: Dict[str, Any]) -> bool:
    """Before a worker receives a schedule-born task: durably say it MAY run now.

    Strict and verified by reading the canonical file back; ``False`` means do not
    dispatch (the task stays pending). Only this token's receipt, not yet terminal,
    is marked — a terminal or foreign result is never re-dispatched through here.
    A custody re-queue of an already-possible occurrence (worker retry, budget
    resume) is marked again: that re-run is custody's decision, not a replay.
    Tasks without an occurrence are untouched (``True``)."""
    occ = (task.get("metadata") or {}).get("schedule_occurrence") if isinstance(task.get("metadata"), dict) else None
    if not isinstance(occ, dict):
        return True
    if task.get("_owner_hold"):
        return False
    from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_RUNNING, STATUS_SCHEDULED, write_task_result

    token, mark = str(occ.get("token") or ""), uuid.uuid4().hex[:12]
    schedule_id = str(occ.get("schedule_id") or "")
    if not token or not schedule_id:
        return False
    redispatch = False

    def _mark(current: Dict[str, Any], _incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal redispatch
        admission = current.get("schedule_admission") if isinstance(current.get("schedule_admission"), dict) else {}
        if current.get("_owner_hold"):
            task["_owner_hold"] = copy.deepcopy(current["_owner_hold"])
            return None
        if (str(admission.get("token") or "") != token
                or not isinstance(admission.get("task"), dict)
                or admission["task"].get("id") != task.get("id")
                or str(admission.get("schedule_id") or "") != schedule_id
                or str(current.get("status") or "") not in {STATUS_SCHEDULED, STATUS_RUNNING, STATUS_INTERRUPTED}):
            return None
        redispatch = admission.get("dispatch") == "possible"
        return {"schedule_admission": {**admission, "dispatch": "possible", "dispatch_at": utc_now_iso(),
                                       "dispatch_mark": mark},
                "status": STATUS_RUNNING, "result": "Assigned to a worker."}

    task_id = str(task.get("id") or "")
    try:
        written = write_task_result(_queue().DRIVE_ROOT, task_id, STATUS_RUNNING,
                          _field_projector=_mark, strict_existing_dict=True)
        if written is False:
            return False
        stored = _read_back(task_id)
    except Exception:
        log.warning("Scheduled dispatch mark failed for %s; not dispatched", task_id, exc_info=True)
        return False
    got = stored.get("schedule_admission") if isinstance(stored.get("schedule_admission"), dict) else {}
    if not (got.get("token") == token and got.get("dispatch_mark") == mark
            and str(stored.get("status") or "") == STATUS_RUNNING):
        return False
    from supervisor.queue_schedules import _write_scheduled_tasks, load_schedule_store, schedule_transaction

    try:
        with schedule_transaction(_queue().DRIVE_ROOT):
            data = load_schedule_store(_queue().DRIVE_ROOT)
            row = next((r for r in data["tasks"] if r.get("id") == occ.get("schedule_id")), None)
            if row is not None:
                current = row.get("occurrence") or {}
                if current.get("token") != token:
                    # Custody may resume an old task after its occurrence retired,
                    # without editing a successor. A first handoff needs its row.
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
            # A deleted schedule does not invalidate its frozen admitted task; the
            # receipt remains monotonic and independently sufficient for dispatch.
        occ["dispatch"] = "possible"
        return True
    except Exception:
        return False



def restore_allowed(task: Dict[str, Any]) -> bool:
    """Snapshot restore: a schedule-born pending row is revived ONLY when its receipt
    is readable, ours, still ``scheduled`` and never marked dispatchable. Anything
    else — possibly dispatched, ran, missing or unreadable — is not replayed: custody
    owns a started run, and the scheduler's reconcile holds an unprovable one."""
    occ = (task.get("metadata") or {}).get("schedule_occurrence") if isinstance(task.get("metadata"), dict) else None
    if not isinstance(occ, dict):
        return not (task.get("metadata") or {}).get("schedule_id")
    from ouroboros.task_results import load_task_result

    try:
        existing = load_task_result(_queue().DRIVE_ROOT, str(task.get("id") or ""), strict=True) or {}
    except Exception:
        return False
    admission = existing.get("schedule_admission") if isinstance(existing.get("schedule_admission"), dict) else {}
    if (not occ.get("token") or admission.get("token") != occ.get("token")
            or admission.get("schedule_id") != occ.get("schedule_id")):
        return False
    frozen = admission.get("task")
    if not isinstance(frozen, dict) or frozen.get("id") != task.get("id"):
        return False
    for key in ("_owner_hold", "_consciousness_continuation"):
        if key in existing or key in frozen:
            task[key] = copy.deepcopy(existing.get(key, frozen.get(key)))
    if occ.get("dispatch") == "possible":
        return False
    try:
        from supervisor.queue_schedules import load_schedule_store

        rows = load_schedule_store(_queue().DRIVE_ROOT).get("tasks") or []
        if any((r.get("occurrence") or {}).get("token") == occ.get("token") and
               (r.get("occurrence") or {}).get("dispatch") == "possible" for r in rows):
            return False
    except Exception:
        return False
    return (str(existing.get("status") or "") == "scheduled" and admission.get("dispatch") == "none"
            and str(admission.get("token") or "") == str(occ.get("token") or ""))
