"""Owner-governed schedule lifecycle: create/edit, disable/delete/restore, remove.

The leaf beside ``supervisor/queue_schedules.py`` (the store, transaction,
audit, projections, skill sync and dispatch): every OWNER-authored mutation of
an existing or new row lives here, over the store's own transaction and audit
seams, so the durable file keeps one writer discipline while the module that
owns dispatch stays readable in one window (DEVELOPMENT "Paying down a size
cap"). Callers reach these through ``supervisor.queue``; nothing here is a
second scheduler or a second store.

Restore of a suppressed skill row rechecks skill presence, manifest declaration,
``supervised_task`` permission and readiness. A blocked row comes back disabled
as ``restored_not_ready`` (marker lifted, change recorded); an absent or
unknowable skill or schedule is ``manifest_absent`` with ``changed=false`` and
the suppression kept. A consumed one-shot is ``consumed_not_rearmed`` until a
fresh ``trigger.run_at`` is authored.
"""

from __future__ import annotations

import logging
import copy
import pathlib
import uuid
from typing import Any, Dict, Optional

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
    from supervisor.schedule_occurrence import remember_claim_basis

    remember_claim_basis(existing)
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
    if _is_suppressed(merged) or merged.get("delete_requested_at"):
        # A suppressed skill row, or a deleted row waiting only for work it
        # already accepted: neither an edit nor a save re-arms future runs.
        merged["enabled"] = False
    return merged


def _require_template_effort(template: Dict[str, Any]) -> None:
    """A template's explicit starting effort for each fired root: top-level, and a tier.

    Absent keeps the fired task's configured default. Every schedule writer (the
    owner's gateway and CLI, ``schedule_followup``) reaches this one check, so an
    invalid value is refused instead of silently starting at the default.
    """
    if "reasoning_effort" in (template.get("metadata") or {}):
        raise ScheduleRefused("invalid_template", "reasoning_effort is a top-level task template field, not metadata")
    if "reasoning_effort" in template:
        from ouroboros.settings_scales import requested_effort

        try:
            template["reasoning_effort"] = requested_effort(template["reasoning_effort"])
        except ValueError as exc:
            raise ScheduleRefused("invalid_template", str(exc)) from exc


def upsert_scheduled_task(record: Dict[str, Any], *, drive_root: pathlib.Path | None = None,
                          actor: str = "", task_id: str = "", reason: str = "",
                          host_followup: dict | None = None,
                          continuation_of: Optional[Dict[str, Any]] = None,
                          new_resource_intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create or replace a scheduled task record.

    Returns the stored row plus an ``audit`` field: ``recorded`` when both audit
    facts landed, ``incomplete`` when the change is durable but its outcome
    record is not. The key rides the RETURNED COPY only — it is never persisted.
    The occurrence protocol's host facts (``occurrence``, ``hold``,
    ``continuation_of``) are never taken from a payload: ``continuation_of`` is
    set only by the host follow-up path through its own keyword, and an authored
    change clears a wait so the next pass retries at once (#1315).
    """
    root = pathlib.Path(drive_root or _store._queue().DRIVE_ROOT)
    with schedule_transaction(root):
        data = load_schedule_store(root)
        tasks = list(data.get("tasks") or [])
        from supervisor.schedule_occurrence import OCCURRENCE_FIELDS, fingerprint

        incoming = {key: value for key, value in dict(record).items() if key not in OCCURRENCE_FIELDS}
        schedule_id = str(incoming.get("id") or "").strip() or uuid.uuid4().hex[:8]
        incoming["id"] = schedule_id
        existing = next((item for item in tasks if str(item.get("id") or "") == schedule_id), None)
        from supervisor.followup_policy import HOST_FIELDS, origin_of, normalize_template, policy_view, control_guard
        # Host-only keyword is never populated from a template or public payload.
        for key in HOST_FIELDS:
            incoming.pop(key, None)
        if existing is not None:
            for key in HOST_FIELDS:
                if key in existing:
                    incoming[key] = copy.deepcopy(existing[key])
            if origin_of(existing):
                incoming["followup_origin"] = origin_of(existing)
        elif host_followup is not None:
            incoming.update(copy.deepcopy(host_followup))
        elif incoming.get("source") != "task_followup":
            incoming["source"] = "owner"
            incoming["followup_relation"] = {"kind": "independent", "revision": uuid.uuid4().hex}
        elif origin_of(incoming):
            # Normalization below drops template lineage; keep its provenance.
            incoming["followup_origin"] = origin_of(incoming)
        incoming["task"] = normalize_template(incoming)
        _require_template_effort(incoming["task"])  # refused before the audit intent: nothing changed
        if incoming.get("followup_origin"):
            incoming["task"]["metadata"]["objective_author"] = {
                "kind": "task", "task_id": incoming["followup_origin"]["task_id"]}
        if existing is None and not (root / "state/owner_restart_no_resume.flag").exists():
            incoming["followup_restart_seen"] = (data.get("followup_restart") or {}).get("control_id", "")
        if new_resource_intent is not None:
            # The producer may describe a NEW row. Editing an old followup cannot
            # turn absent intent into self-work; preserve its current evidence.
            template = dict(incoming.get("task") or {})
            metadata = dict(template.get("metadata") or {})
            intent = ((existing.get("task") or {}).get("metadata") or {}).get("resource_intent") if existing else new_resource_intent
            if intent is not None:
                metadata.setdefault("resource_intent", dict(intent))
            incoming["task"] = {**template, "metadata": metadata}
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
        if existing is None and isinstance(continuation_of, dict):
            incoming["continuation_of"] = dict(continuation_of)
        if existing is not None and fingerprint(existing) != fingerprint(incoming):
            incoming.pop("hold", None)
        incoming.setdefault("enabled", True)
        incoming.setdefault("created_at", utc_now_iso())
        incoming["updated_at"] = utc_now_iso()
        with control_guard(root, incoming):
            view = policy_view(root, data, incoming)
            if view.get("hold"):
                incoming["followup_hold"] = view["hold"]
            incoming["followup_wait"] = view["wait"]
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
                          drive_root: pathlib.Path | None = None,
                          expected_hold_id: str = "", relation: str = "") -> Dict[str, Any]:
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
    from supervisor.followup_policy import origin_of, policy_view, resolve_relation, restore_followup
    resolved = None
    if operation == "restore" and relation:
        observed = next((r for r in load_schedule_store(root)["tasks"] if r.get("id") == wanted), {})
        try:
            origin = origin_of(observed)
            resolved = (origin, resolve_relation(root, origin, relation, declared_by=task_id or actor))
        except ValueError as exc:
            return {"ok": False, "changed": False, "status": str(exc), "schedule_id": wanted,
                    "audit": "not_written"}
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
            # An exact release request never becomes generic enable on replay,
            # and a relationship decision only resolves an observed hold.
            if operation == "restore" and (expected_hold_id or resolved or current.get("followup_hold") or
                    (not _is_consumed_once(current) and policy_view(root, data, current).get("hold"))):
                return restore_followup(root, data, current, expected_hold_id=expected_hold_id,
                                        resolved=resolved, actor=actor, task_id=task_id, reason=reason)
            before = dict(current)
            from supervisor.schedule_occurrence import remember_claim_basis

            remember_claim_basis(current)
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
            detail, removed = "", False
            if operation == "disable":
                current["enabled"] = False
                if skill_row:
                    current["manual_override"] = "disabled"
                status = "updated"
            elif operation == "delete":
                if skill_row:
                    # Retained as a suppressed record: dropping the row would only
                    # have it recreated by the next lifecycle resync, and the owner
                    # would never see that their delete did not hold.
                    current["enabled"] = False
                    current["manual_override"] = "deleted"
                    status = "suppressed"
                else:
                    from supervisor.followup_policy import relation_kind
                    from supervisor.schedule_occurrence import deletion_settled, owed

                    status, owes = "deleted", owed(current, drive_root=root)
                    continuation = relation_kind(current) != "independent"
                    if owes is not False or not deletion_settled(current, drive_root=root):
                        # An accepted run waits to be re-queued: removal is deferred until it
                        # starts, because deleting a row never takes back an admission (#1315).
                        # A continuation's row also carries its task's Stop/Restart release
                        # and start binding, so it stays until that task settles.
                        current["enabled"], current["delete_requested_at"] = False, utc_now_iso()
                        status = "delete_deferred"
                        if owes is None:
                            detail = ("the occurrence receipt is missing, unreadable or conflicting; the disabled "
                                      "row is retained until its execution history can be established")
                        elif owes:
                            detail = ("an accepted run is still owed; the row will be removed once that run starts"
                                      + (" and its task settles" if continuation else ""))
                        else:
                            detail = ("no new run starts; this row continues a task that has not settled "
                                      "and is removed once it settles")
                    else:
                        tasks = [item for item in tasks if str(item.get("id") or "") != wanted]
                        removed = True
            elif _is_consumed_once(current):
                status = "consumed_not_rearmed"
                detail = "a one-shot that already fired is history; schedule a new run_at instead"
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
                # A new explicit Restore may cancel a still-pending deletion;
                # exact hold releases return above and never change this intent.
                current["enabled"] = True
                current.pop("delete_requested_at", None)
                status = "updated"
            changed = status not in UNCHANGED_STATUSES
            if changed:
                if not removed:
                    current["updated_at"] = utc_now_iso()
                data["tasks"] = tasks
                _write_scheduled_tasks(data, root)
            recorded = _audit_schedule_mutation(
                phase="outcome", result=status, before=before,
                after=(None if removed else current), **audit)
            achieved = status in {"updated", "deleted", "delete_deferred", "suppressed"}
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
