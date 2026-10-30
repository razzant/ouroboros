"""Tolerant reader of the append-only ``review_history.jsonl`` an earlier version wrote.

Rows were appended per lifecycle terminal (one per ``job_id``) until the skill review joined the
shared review mechanism (owner, 21.10): since then a skill review's durable record is the review
ledger (``surface="skill"``) and nothing writes here. The Skills card's history and the chat card's
detail of an old row still read it, bounded as before: every reader windows the 4 MB tail, and
``review_round`` / ``snapshot_attempt`` / ``snapshot_revised`` of rows that never persisted them are
derived at read time (``normalize_history``). Rows are never rewritten.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterator, List, Optional

from ouroboros.utils import iter_jsonl_objects, sanitize_tool_result_for_log

_DETAIL_LOOKUP_MAX_BYTES = 4 * 1024 * 1024
_DETAIL_LOOKUP_MAX_RECORDS = 128


def review_history_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "skills" / skill_name / "review_history.jsonl"


def _ordinal(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _normalize_bounded_ordinals(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize stored ordinals when a bounded read cannot derive them."""
    normalized = dict(record)
    for key in ("review_round", "snapshot_attempt"):
        try:
            normalized[key] = max(0, int(record.get(key)))
        except (TypeError, ValueError, OverflowError):
            return None
    normalized["snapshot_revised"] = bool(record.get("snapshot_revised"))
    return normalized


def normalize_history(entries: List[Dict[str, Any]], skill_name: str) -> List[Dict[str, Any]]:
    """Add read-time ordinals to legacy rows without rewriting the audit log."""
    group_rounds: Dict[str, int] = {}
    snapshot_attempts: Dict[tuple[str, str], int] = {}
    last_hash: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for source in entries:
        entry = dict(source)
        group_id = str(entry.get("group_id") or f"manual:{skill_name}")
        content_hash = str(entry.get("content_hash") or "")
        review_round = max(group_rounds.get(group_id, 0) + 1, _ordinal(entry.get("review_round")))
        group_rounds[group_id] = review_round
        attempt_key = (group_id, content_hash)
        snapshot_attempt = max(snapshot_attempts.get(attempt_key, 0) + 1, _ordinal(entry.get("snapshot_attempt")))
        snapshot_attempts[attempt_key] = snapshot_attempt
        revised = bool(last_hash.get(group_id) and last_hash[group_id] != content_hash)
        if content_hash:
            last_hash[group_id] = content_hash
        entry.update(group_id=group_id, review_round=review_round, snapshot_attempt=snapshot_attempt,
                     snapshot_revised=bool(entry.get("snapshot_revised", revised)))
        out.append(entry)
    return out


def load_history(drive_root: pathlib.Path, skill_name: str, limit: int = 3, *, group_id: str = "") -> List[Dict[str, Any]]:
    """History rows inside the bounded tail window; ``limit=0`` is every row of the window."""
    try:
        entries = normalize_history(
            [row for row in iter_jsonl_objects(review_history_path(drive_root, skill_name), tail_bytes=_DETAIL_LOOKUP_MAX_BYTES)
             if isinstance(row, dict)], skill_name)
    except OSError:
        return []
    if group_id:
        entries = [entry for entry in entries if entry.get("group_id") == group_id]
    return entries[-limit:] if limit > 0 else entries


def iter_history_rows_bounded(drive_root: pathlib.Path, skill_name: str) -> Iterator[Dict[str, Any]]:
    """RAW history rows of one skill inside the same bounded tail window."""
    for row in iter_jsonl_objects(review_history_path(pathlib.Path(drive_root), skill_name), tail_bytes=_DETAIL_LOOKUP_MAX_BYTES):
        if isinstance(row, dict):
            yield row


def find_history_job_bounded(drive_root: pathlib.Path, skill_name: str, job_id: str) -> tuple[Optional[Dict[str, Any]], str]:
    """Find one immutable terminal row inside a fixed tail window: ``(row, status)`` with ``status`` one of
    ``found`` / ``not_found`` / ``absent`` / ``unavailable`` (outside the bounded window) / ``io_error``."""
    path = review_history_path(drive_root, skill_name)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None, "absent"
    except OSError:
        return None, "io_error"
    if size <= 0:
        return None, "absent"
    byte_window_truncated = size > _DETAIL_LOOKUP_MAX_BYTES
    gaps: set[str] = set()
    try:
        parsed = list(iter_jsonl_objects(path, max_entries=_DETAIL_LOOKUP_MAX_RECORDS + 1,
                                         tail_bytes=_DETAIL_LOOKUP_MAX_BYTES, gap_reasons=gaps))
    except FileNotFoundError:
        return None, "absent"
    except OSError:
        return None, "io_error"
    records_truncated = len(parsed) > _DETAIL_LOOKUP_MAX_RECORDS
    if records_truncated:
        parsed = parsed[-_DETAIL_LOOKUP_MAX_RECORDS:]
    source_record = next((row for row in reversed(parsed) if str(row.get("job_id") or "") == job_id), None)
    # max_entries counts raw tail lines while parsed contains valid objects: a malformed or partial raw
    # row can hide ordinal-bearing authority just as surely as byte or record truncation.
    projection_incomplete = byte_window_truncated or records_truncated or bool(gaps)
    if source_record is not None:
        ordinal_fields = ("review_round", "snapshot_attempt", "snapshot_revised")
        if projection_incomplete and any(key not in source_record for key in ordinal_fields):
            return None, "unavailable"
        if projection_incomplete:
            normalized = _normalize_bounded_ordinals(source_record)
            return (normalized, "found") if normalized is not None else (None, "unavailable")
        record = next((row for row in reversed(normalize_history(parsed, skill_name))
                       if str(row.get("job_id") or "") == job_id), None)
        return record, "found"
    if projection_incomplete:
        return None, "unavailable"
    return None, "not_found" if parsed else "absent"


def _task_owns_review(record: Dict[str, Any], task_id: str, skill_name: str) -> bool:
    """Existing physical/origin/presentation ownership or the recorded task group, never a path grant."""
    lifecycle = (record.get("brief") or {}).get("skill") or record
    owners = [record.get("task_id"), *(lifecycle.get(key) for key in
              ("task_id", "origin_task_id", "presentation_owner_task_id"))]
    return task_id in owners or lifecycle.get("group_id") == f"task:{task_id}:{skill_name}"


def _ledger_review_detail(drive_root: pathlib.Path, skill_name: str, job_id: str) -> Optional[Dict[str, Any]]:
    """The review ledger record a chat row references: by its record id, or by the lifecycle job id the
    record's brief kept; ``None`` when this skill has no such record."""
    from ouroboros import review_ledger as ledger
    from ouroboros.tools.review_skill import skill_record_rows

    for row in skill_record_rows(drive_root, skill_name, limit=100):
        record_id = str(row.get("record_id") or "")
        if record_id != job_id:
            try:
                record = ledger.load_record(drive_root, record_id)
            except ValueError:
                continue
            if not record or str(((record.get("brief") or {}).get("skill") or {}).get("job_id") or "") != job_id:
                continue
        else:
            try:
                record = ledger.load_record(drive_root, record_id)
            except ValueError:
                continue
        if record:
            return record
    return None


def _render_ledger_review_detail(skill_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """One skill record rendered as the chat card's detail: the findings grouped by seat, the seats that did
    not answer, the gate's status of the verdict and the wave's accounting (paid, reused, cost). This owner
    projection is redacted; the record and the tool's return keep the reviewers' text."""
    from ouroboros.skill_review import render_skill_review_block
    from ouroboros.tools.review_change import review_result
    from ouroboros.tools.review_skill import legacy_actor_records, outcome_status, reviewer_models

    result = review_result(record, reused=False)
    status, findings, reason = outcome_status(result)
    actors = legacy_actor_records(result)
    degraded = [{"model_id": a["model_id"], "status": a["status"],
                 "raw_text": "(raw reviewer output withheld from chat; retained with the review record)"}
                for a in actors if a["status"] != "responded"]
    outcome = {"skill": skill_name, "status": status, "content_hash": str((record.get("subject") or {}).get("tree_sha") or ""),
               "findings": [{**f, "model": f.get("model") or f.get("seat_id")} for f in findings],
               "reviewer_models": reviewer_models(result), "raw_actor_records": degraded, "error": reason,
               "record_id": str(record.get("record_id") or ""), "panel": dict(record.get("panel") or {})}
    markdown = sanitize_tool_result_for_log(render_skill_review_block(outcome))
    cost = dict(record.get("cost") or {})
    accounting = []
    if record.get("dispatch_refusal"):
        accounting.append("no panel was dispatched: " + str((record.get("dispatch_refusal") or {}).get("kind") or "refused"))
    else:
        accounting.append("paid panel dispatch (counts toward Max Review Cycles)")
    accounting.append(f"cost ${float(cost.get('usd') or 0.0):.4f}" + (" (some seats unknown)" if cost.get("unknown") else ""))
    if (record.get("fingerprints") or {}).get("review_contract"):
        accounting.append(f"review contract {str(record['fingerprints']['review_contract'])[:12]}…")
    markdown += "\n\n_Review accounting: " + "; ".join(accounting) + "._"
    return {"markdown": markdown, "status": status, "content_hash": outcome["content_hash"],
            "job_status": "completed" if record.get("state") == "settled" else "running"}


def review_history_detail(
    drive_root: pathlib.Path, skill_name: str, job_id: str, *, task_id: str = "",
) -> Dict[str, Any]:
    """Locate ONE review by the id a chat row carries and render its markdown: a review ledger record of
    the skill (its own id, or the lifecycle job id its brief kept), else a terminal row an earlier version
    appended to ``review_history.jsonl`` (read-only, bounded). Raw reviewer text never leaves the record or
    the history file: findings are the already-normalized items, and degraded (non-responsive) reviewers are
    disclosed by model + status with a pointer instead of the raw body."""
    from ouroboros.skill_loader import _sanitize_skill_name
    from ouroboros.skill_review import render_skill_review_block
    from ouroboros.skill_review_status import (
        STATUS_BLOCKERS,
        STATUS_CLEAN,
        STATUS_PENDING,
        STATUS_WARNINGS,
    )

    if not skill_name or _sanitize_skill_name(skill_name) != skill_name or not job_id:
        return {"error": "unknown skill or review job", "status_code": 404}
    recorded = _ledger_review_detail(drive_root, skill_name, job_id)
    if recorded is not None:
        if task_id and not _task_owns_review(recorded, task_id, skill_name):
            return {"error": "Review does not belong to this task.", "status_code": 404}
        return _render_ledger_review_detail(skill_name, recorded)
    record, lookup_status = find_history_job_bounded(
        drive_root, skill_name, job_id,
    )
    if lookup_status == "absent":
        return {"error": "no review history for skill", "status_code": 404}
    if lookup_status == "io_error":
        return {
            "error": "review history is temporarily unavailable; retry the detail",
            "status_code": 503,
        }
    if record is None:
        error = (
            "review record unavailable outside the bounded history window"
            if lookup_status == "unavailable"
            else "review record not found"
        )
        return {"error": error, "status_code": 404}
    if task_id and not _task_owns_review(record, task_id, skill_name):
        return {"error": "Review does not belong to this task.", "status_code": 404}
    raw_actors = [
        actor for actor in (record.get("raw_actor_records") or [])
        if isinstance(actor, dict)
    ]
    actor_models = [
        str(actor.get("model_id") or actor.get("model") or "reviewer")
        for actor in raw_actors
    ]
    duplicate_models = {
        model for model in actor_models if actor_models.count(model) > 1
    }
    duplicate_occurrences: Dict[str, int] = {}
    labeled_actors = []
    for actor in raw_actors:
        model = str(actor.get("model_id") or actor.get("model") or "reviewer")
        slot_id = str(actor.get("slot_id") or "")
        duplicate_occurrences[model] = duplicate_occurrences.get(model, 0) + 1
        qualifier = slot_id or f"legacy-actor-{duplicate_occurrences[model]}"
        label = f"{model} [{qualifier}]" if model in duplicate_models else model
        labeled_actors.append((actor, label))
    findings = [
        {**item, "model": label}
        for actor, label in labeled_actors
        for item in (actor.get("parsed_items") or [])
        if isinstance(item, dict)
    ]
    reviewer_models = [
        label for _actor, label in labeled_actors
    ]
    degraded_actors = [
        {
            "model_id": label,
            "status": str(actor.get("status") or "unknown"),
            "raw_text": "(raw reviewer output withheld from chat; stored in review_history.jsonl)",
        }
        for actor, label in labeled_actors
        if str(actor.get("status") or "") != "responded"
    ]
    status = str(record.get("status") or "pending")
    terminal_reason = str(record.get("terminal_reason") or "")
    lifecycle_status = str(
        record.get("job_status") or record.get("lifecycle_status") or ""
    ).strip().lower()
    lifecycle_failed = lifecycle_status in {
        "failed", "error", "timeout", "interrupted", "cancelled",
    }
    # Interrupted/timeout/failed records carry no review verdict; surface the
    # terminal reason honestly instead of pretending a review body exists.
    error_note = (
        terminal_reason or lifecycle_status or status
        if lifecycle_failed
        else ("" if status in {
            STATUS_CLEAN, STATUS_WARNINGS, STATUS_BLOCKERS, STATUS_PENDING,
        } else (terminal_reason or status))
    )
    attempt = int(record.get("snapshot_attempt") or 1)
    outcome = {
        "skill": skill_name,
        "status": status,
        "content_hash": str(record.get("content_hash") or ""),
        "findings": findings,
        "reviewer_models": reviewer_models,
        "review_round": int(record.get("review_round") or attempt),
        "snapshot_attempt": attempt,
        "snapshot_revised": bool(record.get("snapshot_revised")),
        "raw_actor_records": degraded_actors,
        "error": error_note,
    }
    markdown = sanitize_tool_result_for_log(render_skill_review_block(outcome, attempt_idx=attempt))
    if degraded_actors:
        markdown += (
            "\n\n_A terminal reviewer slot that never started or refused has no "
            "physical-attempt ledger row; incomplete attempt coverage can therefore be final._"
        )
    # The accounting facts an earlier version put on the row ride the free-form markdown detail: the
    # response contract (SkillReviewHistoryDetailResponse) is typed to exactly four fields.
    accounting = []
    replayed = bool(record.get("replayed_from_ts"))
    if replayed:
        accounting.append(
            f"free replay of the {record.get('replayed_from_ts')} verdict; "
            "no physical reviewer dispatch for this replay"
        )
    elif record.get("paid"):
        accounting.append("paid panel dispatch (counted toward Max Review Cycles by that version)")
    if record.get("review_contract_fingerprint"):
        accounting.append(
            f"panel contract {str(record.get('review_contract_fingerprint'))[:12]}…"
        )
    if record.get("rebuttal_sha256"):
        accounting.append(f"rebuttal sha256 {str(record.get('rebuttal_sha256'))[:12]}…")
    if accounting:
        markdown += "\n\n_Review accounting: " + "; ".join(accounting) + "._"
    markdown += "\n\n_Cost: $0 (free replay)._" if replayed else "\n\n_Cost unavailable (a row of an earlier version)._"
    return {
        "markdown": markdown,
        "status": status,
        "content_hash": outcome["content_hash"],
        "job_status": str(record.get("job_status") or ""),
    }


__all__ = ["find_history_job_bounded", "iter_history_rows_bounded", "load_history", "normalize_history", "review_history_path", "review_history_detail"]
