"""Owner-governed schedule lifecycle: create/edit and disable/delete/restore.

The leaf beside ``supervisor/queue_schedules.py`` (the store, transaction,
audit, projections and dispatch): every OWNER-authored mutation of
an existing or new row lives here, over the store's own transaction and audit
seams, so the durable file keeps one writer discipline while the module that
owns dispatch stays readable in one window (DEVELOPMENT "Paying down a size
cap"). Callers reach these through ``supervisor.queue``; nothing here is a
second scheduler or a second store.

A consumed one-shot stays history until a fresh firing instant is authored.
"""

from __future__ import annotations

import logging
import copy
import pathlib
import uuid
from typing import Any, Dict, Optional

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
    _schedule_next_run,
    _write_scheduled_tasks,
    load_schedule_store,
    schedule_transaction,
)

log = logging.getLogger(__name__)

UNCHANGED_STATUSES: frozenset[str] = frozenset({"consumed_not_rearmed"})


def adopt_legacy_skill_row(record: Dict[str, Any]) -> bool:
    """Keep a former manifest row as an ordinary promise, without changing its work."""
    if not {"manual_override", "skill_content_hash"}.intersection(record):
        return False
    from supervisor.schedule_occurrence import remember_claim_basis

    remember_claim_basis(record)
    override = str(record.pop("manual_override", "") or "").strip().lower()
    record.pop("skill_content_hash", None)
    if override in {"disabled", "deleted"}:
        record["enabled"] = False
    if override == "deleted":
        record.setdefault("delete_requested_at", utc_now_iso())
        record["delete_requested_by"] = "host:skill_schedule_adoption"
        record["delete_reason"] = "deleted while skills managed schedules"
    return True


def _adoption_audit(root, before):
    return dict(drive_root=root, operation_id=uuid.uuid4().hex[:12],
                actor="host:skill_schedule_adoption", task_id="", action="adopt",
                schedule_id=str(before.get("id") or ""), reason="skills no longer manage schedules")


def _adopt_stored_row(root, data, record) -> bool:
    """Adopt under the caller's transaction, before lifecycle/policy eligibility."""
    after = copy.deepcopy(record)
    if not adopt_legacy_skill_row(after):
        return False
    before, audit = copy.deepcopy(record), _adoption_audit(root, record)
    if not _audit_schedule_mutation(phase="intent", result="intended", before=before, **audit):
        raise ScheduleRefused("audit_unavailable", "schedule adoption audit unavailable; nothing was changed")
    record.clear()
    record.update(after)
    try:
        _write_scheduled_tasks(data, root)
    except Exception:
        record.clear()
        record.update(before)
        raise
    if not _audit_schedule_mutation(phase="outcome", result="adopted", before=before, after=record, **audit):
        log.error("schedule %s adopted but its audit outcome is incomplete", record.get("id"))
    return True


def _merge_onto_current(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an owner-authored record onto the CURRENT row, under the write lock.

    Only the fields a caller actually authors win; provenance, run history, the
    consumed receipt are read from the row on
    disk, so a full-record PUT built from a stale GET cannot roll back a
    concurrent scheduler tick.
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
    if merged.get("delete_requested_at"):
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


def _collapse_description_copies(record):
    """Keep the instruction once; distinct authored descriptions remain intact."""
    template = record.get("task") or {}
    text = (record.get("notification") or {}).get("text") if record.get("kind") == "notify" else template.get("text")
    if isinstance(text, str):
        if record.get("description") == text:
            record.pop("description", None)
        if template.get("description") == text:
            template.pop("description", None)


_PATCH_FIELDS = frozenset({"name", "objective", "context", "trigger", "timezone", "reasoning_effort"})


def _patch_value(record, key):
    if key == "objective":
        return ((record.get("notification") or {}).get("text") if record.get("kind") == "notify"
                else (record.get("task") or {}).get("text"))
    if key in {"context", "reasoning_effort"}:
        return (record.get("task") or {}).get(key)
    return record.get(key)


def _patched_definition(current, patch):
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.schedule_contract import cron_error, timezone_error
    from supervisor.followup_policy import normalize_template

    incoming = copy.deepcopy(current)
    note = incoming.get("kind") == "notify"
    for key, value in patch.items():
        if key != "trigger" and not isinstance(value, str):
            raise ScheduleRefused("invalid_template", f"{key} must be a string")
        if key == "objective":
            if not value.strip():
                raise ScheduleRefused("invalid_template", "objective must not be empty")
            if note:
                incoming["notification"] = {**(incoming.get("notification") or {}), "text": value}
                if value != _patch_value(current, key):
                    incoming["notification"]["set_at"] = utc_now_iso()
            else:
                incoming.setdefault("task", {})["text"] = value
        elif key in {"context", "reasoning_effort"}:
            if note:
                raise ScheduleRefused("invalid_patch", f"{key} applies only to task schedules")
            incoming.setdefault("task", {})[key] = value
        else:
            incoming[key] = copy.deepcopy(value)
    trigger = incoming.get("trigger")
    if not isinstance(trigger, dict):
        raise ScheduleRefused("invalid_template", "trigger must be an object")
    kind = trigger.get("type")
    expected = {"type", "run_at"} if kind == "once" else {"type", "expr"}
    if kind not in {"once", "cron"} or set(trigger) != expected:
        raise ScheduleRefused("invalid_template", "trigger must be a complete once/run_at or cron/expr object")
    if error := timezone_error(incoming.get("timezone", "")):
        raise ScheduleRefused("invalid_template", error)
    if kind == "once":
        instant = parse_deadline_ts(str(trigger.get("run_at") or ""))
        if instant is None:
            raise ScheduleRefused("invalid_template", "trigger.run_at must be a parseable ISO 8601 instant")
        if "trigger" in patch:
            incoming["trigger"] = {"type": "once", "run_at": instant.isoformat()}
    elif error := cron_error(trigger.get("expr")):
        raise ScheduleRefused("invalid_template", error)
    template = normalize_template(incoming)
    _require_template_effort(template)
    if not note:
        incoming["task"] = template
    changed = sorted(key for key in patch if _patch_value(current, key) != _patch_value(incoming, key))
    if "trigger" in changed or "timezone" in changed:
        old = current.get("trigger") or {}
        # A fired one-shot given a new instant or a cron is a live promise again.
        if _is_consumed_once(current) and "trigger" in patch and (
                kind == "cron" or instant != parse_deadline_ts(str(old.get("run_at") or ""))):
            incoming.pop("completed_at", None)
            incoming["enabled"] = True
        if kind == "once":
            incoming["next_run_at"] = ""  # a one-shot is due at its run_at, as on creation
        else:
            incoming["next_run_at"] = _schedule_next_run(incoming)
    return incoming, changed


def patch_scheduled_task(schedule_id: str, patch: dict, *, reason: str, actor: str,
                         task_id: str = "", drive_root=None,
                         writer_kind: str = "consciousness", writer_metadata: Any = None) -> dict:
    """Change only supplied authored fields in one audited table write.

    Refusals return a typed no-change result; in particular adoption is staged,
    so an invalid edit of a legacy row leaves its durable bytes untouched.
    """
    from supervisor.schedule_occurrence import remember_claim_basis, fingerprint, _read_back, _unstarted_claim

    wanted = str(schedule_id or "").strip()
    root = pathlib.Path(drive_root or _store._queue().DRIVE_ROOT)
    try:
        if not str(reason or "").strip():
            raise ScheduleRefused("reason_required", "a reason is required to edit a schedule")
        if not isinstance(patch, dict) or set(patch) - _PATCH_FIELDS:
            raise ScheduleRefused("invalid_patch", "patch contains unsupported fields")
        with schedule_transaction(root):
            data = load_schedule_store(root)
            before = next((r for r in data["tasks"] if str(r.get("id") or "") == wanted), None)
            if before is None:
                raise ScheduleRefused("not_found", "schedule not found")
            current = copy.deepcopy(before)
            adopted = adopt_legacy_skill_row(current)
            if current.get("delete_requested_at"):
                raise ScheduleRefused("withdrawn", "this schedule is withdrawn; restore it first")
            remember_claim_basis(current)
            _collapse_description_copies(current)
            incoming, changed_fields = _patched_definition(current, patch)
            changed = bool(changed_fields or adopted)
            reconsidered = False
            if changed:
                from ouroboros.consciousness_authority import stamp_schedule_writer
                from supervisor.followup_policy import relation_kind

                _collapse_description_copies(incoming)
                stamp_schedule_writer(incoming, writer_kind, writer_metadata,
                                      started_root=relation_kind(incoming) == "independent")
                if fingerprint(before) != fingerprint(incoming):
                    incoming.pop("hold", None)
                    occ = current.get("occurrence") or {}
                    if occ.get("phase") == "claimed":
                        try:
                            reconsidered = _unstarted_claim(occ, _read_back(str(occ.get("task_id") or ""), drive_root=root))
                        except Exception:
                            pass  # Unknown receipt: the scan still owns its reconciliation.
                incoming["updated_at"] = utc_now_iso()
            else:
                incoming = copy.deepcopy(before)
            audit = dict(drive_root=root, operation_id=uuid.uuid4().hex[:12], actor=actor,
                         task_id=task_id, action="edit", schedule_id=wanted, reason=reason,
                         changed_fields=changed_fields)
            adoption = _adoption_audit(root, before) if adopted else None
            if not _audit_schedule_mutation(phase="intent", result="intended", before=before, **audit):
                raise ScheduleRefused("audit_unavailable", "the schedule audit log could not be written; nothing was changed")
            if adoption and not _audit_schedule_mutation(phase="intent", result="intended", before=before, **adoption):
                _audit_schedule_mutation(phase="outcome", result="audit_unavailable", before=before, **audit)
                raise ScheduleRefused("audit_unavailable", "schedule adoption audit unavailable; nothing was changed")
            if changed:
                data["tasks"] = [incoming if r is before else r for r in data["tasks"]]
                _write_scheduled_tasks(data, root)
            recorded = True
            if adoption:
                recorded = _audit_schedule_mutation(phase="outcome", result="adopted", before=before, after=incoming, **adoption)
            status = "updated" if changed else "unchanged"
            recorded = _audit_schedule_mutation(phase="outcome", result=status, before=before, after=incoming, **audit) and recorded
            return {**incoming, "status": status, "changed_fields": changed_fields,
                    "audit": "recorded" if recorded else "incomplete", "reconsidered_claim": reconsidered}
    except (ScheduleRefused, ScheduleStoreUnreadable) as exc:
        status = (exc.status if isinstance(exc, ScheduleRefused) else
                  "lock_timeout" if isinstance(exc, ScheduleLockTimeout) else "store_unreadable")
        return {"ok": False, "changed": False, "status": status, "schedule_id": wanted,
                "detail": str(exc), "audit": "not_written"}


def upsert_scheduled_task(record: Dict[str, Any], *, drive_root: pathlib.Path | None = None,
                          actor: str = "", task_id: str = "", reason: str = "",
                          host_followup: dict | None = None,
                          continuation_of: Optional[Dict[str, Any]] = None,
                          new_resource_intent: Optional[Dict[str, Any]] = None,
                          writer_kind: str = "consciousness", writer_metadata: Any = None) -> Dict[str, Any]:
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

        incoming = {key: value for key, value in dict(record).items()
                    if key not in (*OCCURRENCE_FIELDS, "manual_override", "skill_content_hash")}
        schedule_id = str(incoming.get("id") or "").strip() or uuid.uuid4().hex[:8]
        incoming["id"] = schedule_id
        existing = next((item for item in tasks if str(item.get("id") or "") == schedule_id), None)
        if existing is not None:
            _adopt_stored_row(root, data, existing)
        from supervisor.followup_policy import HOST_FIELDS, origin_of, normalize_template, policy_view, control_guard, relation_kind
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
        from ouroboros.consciousness_authority import stamp_schedule_writer
        stamp_schedule_writer(incoming, writer_kind, writer_metadata,
                              started_root=relation_kind(incoming) == "independent")
        incoming["task"] = normalize_template(incoming)
        _require_template_effort(incoming["task"])  # refused before the audit intent: nothing changed
        if incoming.get("followup_origin"):
            incoming["task"]["metadata"]["objective_author"] = {
                "kind": "task", "task_id": incoming["followup_origin"]["task_id"]}
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
        _collapse_description_copies(incoming)
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
                          expected_hold_id: str = "", relation: str = "",
                          writer_kind: str = "consciousness", writer_metadata: Any = None,
                          owned_by: Any = None) -> Dict[str, Any]:
    """Apply one owner-governed future-dispatch mutation, audited intent-first.

    Affects DISPATCH only: a task already admitted from this schedule keeps
    running, and ``running_or_queued`` says so rather than implying a stop — or
    says ``None`` when this process cannot see the live queue, because an
    unknown in-flight run is not the same claim as no run at all. ``owned_by``,
    a narrowed caller's ownership predicate, is judged on the stored row under the
    same transaction; a row it does not own reads ``not_found`` with nothing written.
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
    from supervisor.followup_policy import origin_of, policy_view, resolve_relation, restore_followup, relation_kind
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
            if current is None or (owned_by is not None and not owned_by(current)):
                return {"ok": False, "changed": False, "status": "not_found",
                        "schedule_id": wanted, "audit": "not_written"}
            try:
                _adopt_stored_row(root, data, current)
            except ScheduleRefused as exc:
                return {"ok": False, "changed": False, "status": exc.status,
                        "schedule_id": wanted, "audit": "not_written", "detail": str(exc)}
            # An exact release request never becomes generic enable on replay,
            # and a relationship decision only resolves an observed hold.
            if operation == "restore" and (expected_hold_id or resolved or current.get("followup_hold") or
                    (not _is_consumed_once(current) and policy_view(root, data, current).get("hold"))):
                return restore_followup(root, data, current, expected_hold_id=expected_hold_id,
                                        resolved=resolved, actor=actor, task_id=task_id, reason=reason,
                                        writer_kind=writer_kind, writer_metadata=writer_metadata)
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
            detail = ""
            if operation == "disable":
                current["enabled"] = False
                status = "updated"
            elif operation == "delete":
                current.update(enabled=False, delete_requested_at=utc_now_iso(),
                               delete_requested_by=str(actor), delete_reason=str(reason)[:500])
                status = "withdrawn"
                detail = "removed after the retention window once its work has settled"
            elif _is_consumed_once(current) and not current.get("delete_requested_at"):
                status = "consumed_not_rearmed"
                detail = "a one-shot that already fired is history; schedule a new run_at instead"
            else:
                # A new explicit Restore may cancel a still-pending deletion;
                # exact hold releases return above and never change this intent.
                current["enabled"] = not _is_consumed_once(current)
                for key in ("delete_requested_at", "delete_requested_by", "delete_reason"):
                    current.pop(key, None)
                status = "updated"
            changed = status not in UNCHANGED_STATUSES
            if changed:
                _collapse_description_copies(current)
                from ouroboros.consciousness_authority import stamp_schedule_writer
                stamp_schedule_writer(current, writer_kind, writer_metadata,
                                      started_root=relation_kind(current) == "independent")
                current["updated_at"] = utc_now_iso()
                data["tasks"] = tasks
                _write_scheduled_tasks(data, root)
            recorded = _audit_schedule_mutation(
                phase="outcome", result=status, before=before,
                after=current, **audit)
            achieved = status in {"updated", "withdrawn"}
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
                "schedule": _audit_row(current),
            }
    except ScheduleLockTimeout as exc:
        return {"ok": False, "changed": False, "status": "lock_timeout",
                "schedule_id": wanted, "detail": str(exc), "audit": "not_written"}
