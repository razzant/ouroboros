"""Owner-governed schedule lifecycle: create/edit, disable/delete/restore, remove.

The leaf beside ``supervisor/queue_schedules.py`` (the store, transaction,
audit, projections, skill sync and dispatch): every OWNER-authored mutation of
an existing or new row lives here, over the store's own transaction and audit
seams, so the durable file keeps one writer discipline while the module that
owns dispatch stays readable in one window (DEVELOPMENT "Paying down a size
cap"). Callers reach these through ``supervisor.queue``; nothing here is a
second scheduler or a second store.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Dict

from ouroboros.schedule_contract import schedule_slug
from ouroboros.utils import utc_now_iso
from supervisor import queue_schedules as _store
from supervisor.queue_schedules import (
    SCHEDULE_ACTIONS,
    ScheduleLockTimeout,
    ScheduleRefused,
    ScheduleStoreUnreadable,
    _RUNTIME_OWNED_FIELDS,
    _audit_row,
    _audit_schedule_mutation,
    _is_consumed_once,
    _is_suppressed,
    _schedule_next_run,
    _write_scheduled_tasks,
    load_schedule_store,
    schedule_transaction,
)

log = logging.getLogger(__name__)

# Restore lifts the owner's suppression only for a schedule its skill's manifest
# STILL declares (acceptance claim: "Restore clears suppression only for extant
# manifest"). These blockers say the manifest is absent or unknowable, so the
# marker stays and the action is a typed no-change refusal; the other blockers
# (permission, readiness, identity) describe an extant row that merely cannot
# dispatch yet, and restore lifts the marker while reporting them.
MANIFEST_ABSENT_BLOCKERS: frozenset[str] = frozenset({
    "skill_unknown", "skill_discovery_failed", "skill_absent", "schedule_absent_from_manifest",
})
# Typed no-change outcomes of ``mutate_scheduled_task``: the intent and its
# refusal are both audited, the row is byte-identical afterwards.
UNCHANGED_STATUSES: frozenset[str] = frozenset({"consumed_not_rearmed", "manifest_absent", "not_suppressed"})


def _skill_schedule_dispatchable(drive_root: pathlib.Path, record: Dict[str, Any]) -> tuple[bool, str]:
    """Whether a skill-manifest row may dispatch again, and the blocker if not.

    Restore lifts the OWNER's suppression; it never enables the skill. So the
    row comes back only as far as the skill lifecycle currently allows: a skill
    that was uninstalled, lost the schedule from its manifest, lost its
    ``supervised_task`` permission or is not review-ready returns disabled with
    a named reason instead of a schedule that would fail at dispatch.
    """
    skill_name = str(record.get("skill") or "").strip()
    if not skill_name:
        return False, "skill_unknown"
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import discover_skills

        skills = discover_skills(pathlib.Path(drive_root), repo_path=get_skills_repo_path())
    except Exception:
        log.debug("skill discovery failed while restoring %s", record.get("id"), exc_info=True)
        return False, "skill_discovery_failed"
    skill = next((item for item in skills if str(getattr(item, "name", "")) == skill_name), None)
    if skill is None:
        return False, "skill_absent"
    if bool(getattr(skill, "identity_collision", False)):
        return False, "skill_identity_collision"
    manifest = getattr(skill, "manifest", None)
    declared = {
        schedule_slug("skill", skill_name, str(spec.get("name") or "").strip())
        for spec in list(getattr(manifest, "scheduled_tasks", []) or [])
        if isinstance(spec, dict) and str(spec.get("name") or "").strip() and str(spec.get("cron") or "").strip()
    }
    if str(record.get("id") or "") not in declared:
        return False, "schedule_absent_from_manifest"
    if "supervised_task" not in set(getattr(manifest, "permissions", []) or []):
        return False, "skill_permission_missing"
    try:
        from ouroboros.skill_readiness import skill_readiness_for_execution

        if not skill_readiness_for_execution(pathlib.Path(drive_root), skill).ready:
            return False, "skill_not_ready"
    except Exception:
        log.debug("skill readiness probe failed while restoring %s", record.get("id"), exc_info=True)
        return False, "readiness_unavailable"
    return True, ""


def _merge_onto_current(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an owner-authored record onto the CURRENT row, under the write lock.

    Only the fields a caller actually authors win; provenance, run history, the
    consumed receipt and the owner's suppression marker are read from the row on
    disk, so a full-record PUT built from a stale GET cannot roll back a
    concurrent scheduler tick or skill resync.
    """
    trigger_changed = dict(existing.get("trigger") or {}) != dict(incoming.get("trigger") or {})
    timezone_changed = str(existing.get("timezone") or "") != str(incoming.get("timezone") or "")
    consumed = _is_consumed_once(existing)
    merged = {**existing, **incoming}
    for key in _RUNTIME_OWNED_FIELDS:
        if key in existing:
            merged[key] = existing[key]
        else:
            merged.pop(key, None)
    if trigger_changed:
        # A new firing point is a new one-shot: the consumed receipt goes with
        # the instant that earned it, and the next run is recomputed.
        merged.pop("completed_at", None)
        merged["next_run_at"] = _schedule_next_run(merged)
    else:
        if consumed and bool(incoming.get("enabled")):
            raise ScheduleRefused(
                "consumed_not_rearmed",
                "this one-shot schedule already fired; re-arming it requires a new "
                "trigger.run_at (a fresh run_at clears completed_at)")
        if "completed_at" in existing:
            # A timezone-only edit re-reads the SAME instant in another zone; it
            # is not a new firing point and must not resurrect a fired row.
            merged["completed_at"] = existing["completed_at"]
        if timezone_changed:
            merged["next_run_at"] = _schedule_next_run(merged)
        elif "next_run_at" in existing:
            merged["next_run_at"] = existing["next_run_at"]
        if consumed:
            merged["enabled"] = False
    if str(merged.get("source") or "") == "skill_manifest":
        # A skill row's ``enabled`` is RECONCILED from skill readiness, so an
        # edit that flips it would be undone by the next resync and look durable
        # meanwhile. The owner's durable statement is the suppression marker,
        # which only the lifecycle action writes — so that is where this goes.
        if bool(existing.get("enabled", True)) != bool(merged.get("enabled", True)):
            raise ScheduleRefused(
                "lifecycle_action_required",
                "a skill-manifest schedule is enabled by its skill's readiness; use the "
                "disable/restore lifecycle action, which records the owner's decision so "
                "the skill resync cannot undo it")
        if _is_suppressed(merged):
            merged["enabled"] = False
    elif _is_suppressed(merged):
        # A suppressed notify row stays off whatever its skill re-posts.
        merged["enabled"] = False
    return merged


def upsert_scheduled_task(record: Dict[str, Any], *, drive_root: pathlib.Path | None = None,
                          actor: str = "", task_id: str = "", reason: str = "") -> Dict[str, Any]:
    """Create or replace a scheduled task record.

    Returns the stored row plus an ``audit`` field: ``recorded`` when both audit
    facts landed, ``incomplete`` when the change is durable but its outcome
    record is not. The key rides the RETURNED COPY only — it is never persisted.
    """
    root = pathlib.Path(drive_root or _store._queue().DRIVE_ROOT)
    with schedule_transaction(root):
        data = load_schedule_store(root)
        tasks = list(data.get("tasks") or [])
        incoming = dict(record)
        schedule_id = str(incoming.get("id") or "").strip() or uuid.uuid4().hex[:8]
        incoming["id"] = schedule_id
        existing = next((item for item in tasks if str(item.get("id") or "") == schedule_id), None)
        operation_id = uuid.uuid4().hex[:12]
        action = "edit" if existing is not None else "create"
        audit = {
            "drive_root": root, "operation_id": operation_id,
            "actor": str(actor or "").strip() or "owner", "task_id": task_id,
            "action": action, "schedule_id": schedule_id,
            "reason": str(reason or "").strip() or f"schedule_{action}",
        }
        if not _audit_schedule_mutation(phase="intent", result="intended",
                                        before=existing, **audit):
            raise ScheduleRefused(
                "audit_unavailable",
                "the schedule audit log could not be written; nothing was changed")
        if existing is not None:
            try:
                incoming = _merge_onto_current(existing, incoming)
            except ScheduleRefused as refusal:
                # Every announced intent gets an outcome, including a refusal:
                # a dangling intent would read as a change nobody can account for.
                _audit_schedule_mutation(phase="outcome", result=refusal.status,
                                         before=existing, **audit)
                raise
        incoming.setdefault("enabled", True)
        incoming.setdefault("created_at", utc_now_iso())
        incoming["updated_at"] = utc_now_iso()
        if not incoming.get("next_run_at"):
            incoming["next_run_at"] = _schedule_next_run(incoming)
        tasks = [item for item in tasks if str(item.get("id") or "") != schedule_id]
        tasks.append(incoming)
        data["tasks"] = tasks
        _write_scheduled_tasks(data, root)
        recorded = _audit_schedule_mutation(phase="outcome", result="stored",
                                            before=existing, after=incoming, **audit)
        if not recorded:
            log.error("schedule %s stored but its audit outcome is incomplete", schedule_id)
        return {**incoming, "audit": "recorded" if recorded else "incomplete"}


def mutate_scheduled_task(action: str, schedule_id: str, *, reason: str,
                          actor: str, task_id: str = "",
                          drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Apply one owner-governed future-dispatch mutation, audited intent-first.

    Affects DISPATCH only: a task already admitted from this schedule keeps
    running, and ``running_or_queued`` says so rather than implying a stop — or
    says ``None`` when this process cannot see the live queue, because an
    unknown in-flight run is not the same claim as no run at all.
    """
    wanted, operation = str(schedule_id or "").strip(), str(action or "").strip().lower()
    if operation not in SCHEDULE_ACTIONS:
        return {"ok": False, "changed": False, "status": "invalid_action",
                "schedule_id": wanted, "allowed": sorted(SCHEDULE_ACTIONS), "audit": "not_written"}
    if not wanted:
        return {"ok": False, "changed": False, "status": "missing_schedule_id", "audit": "not_written"}
    if not str(reason or "").strip():
        return {"ok": False, "changed": False, "status": "reason_required",
                "schedule_id": wanted, "audit": "not_written"}
    root = pathlib.Path(drive_root or _store._queue().DRIVE_ROOT)
    try:
        with schedule_transaction(root):
            try:
                data = load_schedule_store(root)
            except ScheduleStoreUnreadable as exc:
                return {"ok": False, "changed": False, "status": "store_unreadable",
                        "schedule_id": wanted, "detail": str(exc), "audit": "not_written"}
            tasks = list(data.get("tasks") or [])
            current = next((item for item in tasks if str(item.get("id") or "") == wanted), None)
            if current is None:
                return {"ok": False, "changed": False, "status": "not_found",
                        "schedule_id": wanted, "audit": "not_written"}
            before = dict(current)
            operation_id = uuid.uuid4().hex[:12]
            audit = {
                "drive_root": root, "operation_id": operation_id, "actor": actor,
                "task_id": task_id, "action": operation, "schedule_id": wanted, "reason": reason,
            }
            if not _audit_schedule_mutation(phase="intent", result="intended", before=before, **audit):
                return {"ok": False, "changed": False, "status": "audit_unavailable",
                        "schedule_id": wanted, "audit": "not_written",
                        "detail": "the schedule audit log could not be written; nothing was changed"}
            running = _store._schedule_running_or_queued(wanted, root)
            skill_row = str(current.get("source") or "") == "skill_manifest"
            # A skill's notify row is re-posted by its key, so the owner's off
            # switch needs the same durable marker a skill-manifest row keeps.
            # The owner acts from Activity (`owner:gateway`) or through
            # Ouroboros's manage_schedules at their word (`agent`) — both are
            # the owner's hand here, as for a skill-manifest row; only the row's
            # own source is the skill, and its cancel of its own row really
            # removes it — nothing of the owner's is being overridden there.
            notify_row = str(current.get("kind") or "") == _store.SCHEDULE_KIND_NOTIFY
            owner_over_notify = notify_row and str(actor or "") != str(current.get("source") or "")
            detail, removed, kept = "", False, False
            if operation == "disable":
                current["enabled"] = False
                if skill_row or owner_over_notify:
                    current["manual_override"] = "disabled"
                status = "updated"
            elif operation == "delete":
                if notify_row and _is_consumed_once(current) and (owner_over_notify or not _is_suppressed(current)):
                    # A fired reminder is a receipt, not a standing row: removing
                    # it re-arms nothing of the owner's, so either hand removes it
                    # outright, as a consumed task one-shot goes — unless the owner
                    # switched even the receipt off, which the skill may not undo.
                    tasks = [item for item in tasks if str(item.get("id") or "") != wanted]
                    status, removed = "deleted", True
                elif owner_over_notify and _is_suppressed(current):
                    # The owner already switched this reminder off and now removes
                    # the record itself: an explicit second act, so the row goes
                    # (a later post of the same key starts a fresh row).
                    tasks = [item for item in tasks if str(item.get("id") or "") != wanted]
                    status, removed = "deleted", True
                elif notify_row and not owner_over_notify and _is_suppressed(current):
                    # The skill cancelling a row the OWNER switched off: the marker
                    # is the owner's, so the record stays untouched — otherwise
                    # cancel plus a repeat of the same key would lift the owner's
                    # decision. Reported as the suppression it is, and unchanged.
                    status, kept = "suppressed", True
                elif skill_row or owner_over_notify:
                    # Retained as a suppressed record: dropping the row would only
                    # have it recreated by the next lifecycle resync (or the skill's
                    # next post of the same key), and the owner would never see
                    # that their delete did not hold.
                    current["enabled"] = False
                    current["manual_override"] = "deleted"
                    status = "suppressed"
                else:
                    tasks = [item for item in tasks if str(item.get("id") or "") != wanted]
                    status, removed = "deleted", True
            elif _is_consumed_once(current):
                status = "consumed_not_rearmed"
                detail = "a one-shot that already fired is history; schedule a new run_at instead"
            elif notify_row:
                # Nothing to probe: a reminder has no skill readiness, so restore
                # is the owner lifting their own marker and re-arming the row.
                current.pop("manual_override", None)
                current["enabled"] = True
                status = "updated"
            elif skill_row:
                # Probed UNDER the transaction: readiness is skill state on disk,
                # not a row field, so a probe taken before the lock could be
                # stale by the time the row is written (a resync or an owner
                # skill disable in between) and would re-arm a schedule its skill
                # cannot dispatch. The hold is bounded by skill discovery; the
                # honest outcome is worth that contention.
                dispatchable, blocker = _skill_schedule_dispatchable(root, current)
                if not _is_suppressed(current):
                    # Nothing of the owner's to lift: a skill row without the marker
                    # is disabled by its skill's READINESS and re-arms on resync
                    # when the skill is ready. Saying "suppression lifted" here
                    # would describe a decision nobody took.
                    status = "not_suppressed"
                    detail = ("this skill schedule is not suppressed; it is "
                              + (f"disabled by skill readiness ({blocker}) and re-arms when the skill is ready"
                                 if blocker else "already enabled"))
                elif blocker in MANIFEST_ABSENT_BLOCKERS:
                    # No extant manifest schedule to restore INTO: the owner's
                    # suppression stays on the row (the next resync would otherwise
                    # drop the unsuppressed orphan and forget the decision before a
                    # reinstall could honor it).
                    status = "manifest_absent"
                    detail = (f"{blocker}: the skill no longer declares this schedule; "
                              "suppression kept until it does")
                else:
                    current.pop("manual_override", None)
                    current["enabled"] = dispatchable
                    status = "updated"
                    if not dispatchable:
                        status, detail = "restored_not_ready", blocker
            else:
                current["enabled"] = True
                status = "updated"
            changed = status not in UNCHANGED_STATUSES and not kept
            if changed:
                if not removed:
                    current["updated_at"] = utc_now_iso()
                data["tasks"] = tasks
                _write_scheduled_tasks(data, root)
            recorded = _audit_schedule_mutation(
                phase="outcome", result=status, before=before,
                after=(None if removed else current), **audit)
            achieved = status in {"updated", "deleted", "suppressed"}
            if changed and not recorded:
                # Keep the lifecycle blocker beside the audit disclosure: a lost
                # outcome record must not erase WHY the row is still not ready.
                detail = ((f"{detail}; " if detail else "")
                          + "the change is durable; its audit outcome record could not be written")
            return {
                "ok": bool(achieved and recorded), "changed": changed,
                "status": ("changed_audit_incomplete" if (changed and not recorded) else status),
                "schedule_id": wanted, "operation_id": operation_id,
                "running_or_queued": running,
                "audit": "recorded" if recorded else ("incomplete" if changed else "not_written"),
                **({"detail": detail} if detail else {}),
                "schedule": None if removed else _audit_row(current),
            }
    except ScheduleLockTimeout as exc:
        return {"ok": False, "changed": False, "status": "lock_timeout",
                "schedule_id": wanted, "detail": str(exc), "audit": "not_written"}


def remove_scheduled_task(schedule_id: str, *, drive_root: pathlib.Path | None = None,
                          actor: str = "", task_id: str = "", reason: str = "") -> bool:
    """Remove a scheduled task record by id, through the one audited delete seam.

    A skill-manifest row is SUPPRESSED rather than dropped (see
    ``mutate_scheduled_task``); either way the row stops dispatching, which is
    what this boolean has always meant to its callers.
    """
    outcome = mutate_scheduled_task(
        "delete", schedule_id, drive_root=drive_root, task_id=task_id,
        actor=str(actor or "").strip() or "host",
        reason=str(reason or "").strip() or "schedule_removed")
    return bool(outcome.get("changed"))


