"""Selected skill revision admission for ordinary managed development tasks.

A repair is admitted AGAINST one exact payload state: ``base_content_hash``,
captured immutably at admission (the promoted managed-task seam canonicalizes
every skill_repair constraint, so every repair passes through it, whoever asked
for it). Every payload write by the ADMITTED repair task
then checks the payload against its last known revision
(``expected_content_hash``). File-tool effects retain their attribution; after
opaque process work the record advances to an OBSERVED revision, without claiming
exclusive authorship. A different hash before the next operation is known drift
and makes the repair STALE. This is observation, not an atomic transaction
against arbitrary editors or a lock held around a long shell command.

No restore is promised — ``last_known_good`` carries version/sha/ts only, never
payload bytes, so there is nothing to restore FROM; the honest fix is a fresh
repair admitted against the new state. No staged machinery (owner 11=B).

Other writers (the owner's own light-mode edit lane, another task) are NOT
blocked here — the repair verifies its known frontier, and later drift surfaces
on its next check instead of gating everyone else's authority
(AGENTS.md proportionality).

This is also the owner-edit seam (``skill_acceptance``): when the admitted task's
run was started by the owner's own door (``owner_request``, captured at admission),
each observed revision is recorded as an owner edit of those exact bytes, so the
skill keeps running as Modified. Edits from anywhere else leave it unreviewed.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional

from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

_RECORD_NAME = "repair_admission.json"
STATUS_ADMITTED = "admitted"
STATUS_STALE = "stale"


def _admission_path(drive_root: Any, skill_name: str) -> pathlib.Path:
    from ouroboros.skill_loader import skill_state_dir

    return skill_state_dir(pathlib.Path(drive_root), str(skill_name)) / _RECORD_NAME


def _payload_dir(drive_root: Any, constraint: Any) -> pathlib.Path:
    from ouroboros.contracts.skill_payload_policy import resolve_constrained_payload_path

    return resolve_constrained_payload_path(pathlib.Path(drive_root), constraint, ".")


def _payload_hash(payload_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import compute_content_hash

    return compute_content_hash(payload_dir)


def record_repair_admission(drive_root: Any, skill_name: str, *,
                            task_id: str, base_content_hash: str,
                            owner_request: bool = False) -> None:
    """Bind one repair task to the exact payload state it was admitted against.

    ``owner_request`` is the host's owner-ingress fact of the promoted run
    (``dialogue_provenance.run_origin``): only then do this task's observed
    revisions become owner edits.

    The newest admission owns the record: a fresh repair for the same skill
    replaces a stale predecessor's binding (its writes then fail the ownership
    check, which is the correct terminal answer for a superseded repair).

    Raises on failure — the caller must NOT admit a repair whose binding did not
    land. A repair running without its record CAS-checks nothing: every write
    passes, which is the exact fail-open this module exists to remove.
    """
    path = _admission_path(drive_root, skill_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    atomic_write_json(path, {
        "schema_version": 1,
        "skill_name": str(skill_name),
        "task_id": str(task_id or ""),
        "base_content_hash": str(base_content_hash or ""),
        "expected_content_hash": str(base_content_hash or ""),
        "status": STATUS_ADMITTED,
        "owner_request": bool(owner_request),
        "admitted_at": now,
        "updated_at": now,
    }, trailing_newline=True)


def load_repair_admission(drive_root: Any, skill_name: str) -> Optional[Dict[str, Any]]:
    import json

    try:
        raw = json.loads(_admission_path(drive_root, skill_name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _save(drive_root: Any, skill_name: str, record: Dict[str, Any]) -> None:
    record["updated_at"] = utc_now_iso()
    atomic_write_json(_admission_path(drive_root, skill_name), record, trailing_newline=True)


def repair_write_cas_error(drive_root: Any, constraint: Any, *, task_id: str = "",
                           repair_task: bool = False) -> str:
    """The typed refusal for a payload write the repair's hash chain forbids, or "".

    ``repair_task`` says the WRITER IS a managed skill-repair task (its own task
    constraint, not a synthesized short-form selector). For that writer the
    binding is mandatory and every unverifiable state is STALE: no admission
    record (it never landed, or it was lost), an unreadable one, a record owned
    by a DIFFERENT repair (this one was superseded), or no task id to check
    against. Without that flag the call is an ordinary payload-editing lane —
    the owner's light-mode edit, another task — which is not gated here at all;
    its effect surfaces as drift on the admitted repair's next CAS
    (AGENTS.md proportionality). An unreadable payload fails CLOSED either way.
    """
    skill_name = str(getattr(constraint, "skill_name", "") or "")
    record = load_repair_admission(drive_root, skill_name)
    if record is None:
        if not repair_task:
            return ""
        return (
            f"⚠️ SKILL_REPAIR_STALE: this repair of {skill_name!r} has NO readable "
            "admission record, so its writes cannot be bound to the payload state it "
            "was admitted against and nothing would verify them. Do not edit blind: "
            "finalize with your findings and report that the repair admission is "
            "missing — a fresh repair must be admitted against the current state."
        )
    owner = str(record.get("task_id") or "")
    if not task_id or not owner or owner != str(task_id):
        if not repair_task:
            return ""
        return (
            f"⚠️ SKILL_REPAIR_STALE: the admission record for {skill_name!r} belongs to "
            f"repair task {owner or '(none)'!r}, not to this one, so this repair was "
            "SUPERSEDED by a newer admission (or never had one). Its writes are no "
            "longer accepted: finalize now with what you know."
        )
    if str(record.get("status") or "") == STATUS_STALE:
        return (
            f"⚠️ SKILL_REPAIR_STALE: this repair of {skill_name!r} was already "
            "terminalized as stale (the payload drifted under it). Do not continue "
            "editing — finalize with what you know; a FRESH repair must be admitted "
            "against the current payload state."
        )
    try:
        current = _payload_hash(_payload_dir(drive_root, constraint))
    except Exception as exc:
        return (
            f"⚠️ SKILL_REPAIR_STALE: the payload of {skill_name!r} became unverifiable "
            f"({type(exc).__name__}: {exc}); the repair's hash chain cannot continue. "
            "Finalize — do not write blind."
        )
    expected = str(record.get("expected_content_hash") or "")
    if not expected:
        return (
            f"⚠️ SKILL_REPAIR_STALE: the admission record for {skill_name!r} has no "
            "known revision. Finalize with your findings; a fresh repair must be "
            "admitted against the current payload."
        )
    if current != expected:
        record["status"] = STATUS_STALE
        record["drift_observed_hash"] = current
        record["drift_expected_hash"] = expected
        try:
            _save(drive_root, skill_name, record)
        except Exception:
            log.warning("Failed to persist stale repair admission for %s", skill_name,
                        exc_info=True)
        return (
            f"⚠️ SKILL_REPAIR_STALE: the payload of {skill_name!r} changed since this "
            f"task's last observation (expected {expected[:12]}, found {current[:12]}). "
            "The repair is STALE: terminalize now with your "
            "findings. No restore is possible — last_known_good holds no payload bytes "
            "— and no further writes from this repair will be accepted; a fresh repair "
            "must be admitted against the current state."
        )
    return ""


def advance_repair_expected_hash(
    drive_root: Any, constraint: Any, *, task_id: str = "", attribution: str = "file_tool",
) -> None:
    """Record the observed frontier, retaining how the operation was observed."""
    skill_name = str(getattr(constraint, "skill_name", "") or "")
    record = load_repair_admission(drive_root, skill_name)
    if record is None or str(record.get("status") or "") == STATUS_STALE:
        return
    owner = str(record.get("task_id") or "")
    if not task_id or not owner or owner != str(task_id):
        return
    try:
        payload_dir = _payload_dir(drive_root, constraint)
        record["expected_content_hash"] = _payload_hash(payload_dir)
        record["revision_attribution"] = attribution
    except Exception:
        log.warning("Failed to advance repair hash chain for %s", skill_name, exc_info=True)
        return
    try:
        _save(drive_root, skill_name, record)
    except Exception:
        log.warning("Failed to persist repair hash chain for %s", skill_name, exc_info=True)
    if record.get("owner_request") is True:
        from ouroboros.skill_acceptance import record_owner_edit

        record_owner_edit(pathlib.Path(drive_root), payload_dir, task_id=str(task_id), attribution=attribution)


__all__ = [
    "STATUS_ADMITTED",
    "STATUS_STALE",
    "advance_repair_expected_hash",
    "load_repair_admission",
    "record_repair_admission",
    "repair_write_cas_error",
]
