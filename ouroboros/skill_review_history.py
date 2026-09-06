"""Read/write helpers for the existing append-only Skill Review history."""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict, Iterator, List, Optional

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.tools.review_helpers import format_obligation_excerpt
from ouroboros.utils import append_jsonl, iter_jsonl_objects, jsonl_append_lock_path, utc_now_iso

log = logging.getLogger(__name__)
USAGE_ATTRIBUTION_SCHEMA = "physical_attempt_v1"
_DETAIL_LOOKUP_MAX_BYTES = 4 * 1024 * 1024
_DETAIL_LOOKUP_MAX_RECORDS = 128
_MARKER_FACT_KEYS = (
    "review_contract_fingerprint", "rebuttal_sha256", "usage_attribution_schema",
    "group_id", "content_hash", "root_task_id",
)
ROOT_TASK_PROJECTION_RELATIVE_PATH = "state/skill_review_root_tasks.jsonl"
ROOT_TASK_PROJECTION_GAPS_RELATIVE_PATH = "state/skill_review_root_tasks.gaps.jsonl"


def _redact_history_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.observability import redact_projection

    redacted = redact_projection(payload).value
    return redacted if isinstance(redacted, dict) else {}


def review_history_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    return drive_root / "state" / "skills" / skill_name / "review_history.jsonl"


def root_task_projection_path(drive_root: pathlib.Path) -> pathlib.Path:
    return drive_root / ROOT_TASK_PROJECTION_RELATIVE_PATH


def root_task_projection_gaps_path(drive_root: pathlib.Path) -> pathlib.Path:
    return drive_root / ROOT_TASK_PROJECTION_GAPS_RELATIVE_PATH


def _record_root_task_projection_gap(
    drive_root: pathlib.Path, skill_name: str, payload: Dict[str, Any],
) -> None:
    """Durably disclose a terminal row missing from the root-task projection."""
    root_task_id = str(payload.get("root_task_id") or "")
    job_id = str(payload.get("job_id") or payload.get("wave_id") or "")
    if not root_task_id or not job_id:
        return
    row = {
        "ts": str(payload.get("ts") or ""), "root_task_id": root_task_id,
        "skill": skill_name, "job_id": job_id,
        "reason": "root_task_projection_append_failed",
    }
    if not append_jsonl(root_task_projection_gaps_path(drive_root), row):
        _emit_history_event(drive_root, {
            "type": "skill_review_root_task_projection_gap_append_failed",
            **row,
        })


def legacy_dispatch_marker_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    """The retired SINGLE-file marker (pre per-wave storage). Tolerated read-only:
    ``write_dispatch_marker`` flushes it into the history and removes it."""
    return drive_root / "state" / "skills" / skill_name / "review_dispatch.json"


def dispatch_marker_dir(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    """APPEND-ONLY per-wave dispatch markers: one file per dispatched wave, so
    two concurrent waves on one skill can never overwrite each other's paid
    fact (a lost marker silently undercounts the cycle ceiling). Each wave's
    terminal-row merge clears exactly its own file."""
    return drive_root / "state" / "skills" / skill_name / "review_dispatch"


def _wave_marker_path(drive_root: pathlib.Path, skill_name: str, wave_id: str) -> pathlib.Path:
    import hashlib
    import re

    wave = str(wave_id or "")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", wave)[:48]
    digest = hashlib.sha256(wave.encode("utf-8")).hexdigest()[:10]
    return dispatch_marker_dir(drive_root, skill_name) / f"{safe}-{digest}.json"


def _emit_history_event(drive_root: pathlib.Path, event: Dict[str, Any]) -> None:
    """Loud typed event on the existing events rail (never raises)."""
    try:
        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "events.jsonl",
            {"ts": utc_now_iso(), **event},
        )
    except Exception:
        log.debug("skill review history event emission failed", exc_info=True)


def _read_marker_file(path: pathlib.Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_dispatch_markers(drive_root: pathlib.Path, skill_name: str) -> List[Dict[str, Any]]:
    """EVERY unmerged write-ahead dispatch marker for this skill (per-wave files
    plus a tolerated legacy single-file marker), sorted by wave id. Unreadable
    files are skipped (fail-open cost accounting)."""
    markers: Dict[str, Dict[str, Any]] = {}
    try:
        files = sorted(dispatch_marker_dir(drive_root, skill_name).glob("*.json"))
    except OSError:
        files = []
    for path in files:
        marker = _read_marker_file(path)
        wave = str(marker.get("wave_id") or "")
        if wave:
            markers[wave] = marker
    legacy = _read_marker_file(legacy_dispatch_marker_path(drive_root, skill_name))
    legacy_wave = str(legacy.get("wave_id") or "")
    if legacy_wave and legacy_wave not in markers:
        markers[legacy_wave] = legacy
    return [markers[wave] for wave in sorted(markers)]


def load_dispatch_marker_for_wave(
    drive_root: pathlib.Path, skill_name: str, wave_id: str
) -> Dict[str, Any]:
    """The marker recorded for exactly this wave, ``{}`` when none/unreadable."""
    if not wave_id:
        return {}
    marker = _read_marker_file(_wave_marker_path(drive_root, skill_name, wave_id))
    if str(marker.get("wave_id") or "") == str(wave_id):
        return marker
    legacy = _read_marker_file(legacy_dispatch_marker_path(drive_root, skill_name))
    if str(legacy.get("wave_id") or "") == str(wave_id):
        return legacy
    return {}


def clear_dispatch_marker(drive_root: pathlib.Path, skill_name: str, *, wave_id: str) -> None:
    """Remove ONE wave's marker once its terminal row landed (merge complete).
    Only the named wave's own file (or a legacy single-file marker recording
    that wave) is touched — concurrent waves' markers stay in place."""
    if not wave_id:
        return
    path = _wave_marker_path(drive_root, skill_name, wave_id)
    if str(_read_marker_file(path).get("wave_id") or "") == str(wave_id):
        try:
            path.unlink()
        except OSError:
            log.debug("skill review dispatch marker unlink failed", exc_info=True)
    legacy_path = legacy_dispatch_marker_path(drive_root, skill_name)
    if str(_read_marker_file(legacy_path).get("wave_id") or "") == str(wave_id):
        try:
            legacy_path.unlink()
        except OSError:
            log.debug("legacy skill review dispatch marker unlink failed", exc_info=True)


def write_dispatch_marker(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    wave_id: str,
    group_id: str,
    content_hash: str,
    root_task_id: str = "",
    review_contract_fingerprint: str = "",
    rebuttal_sha256: str = "",
) -> None:
    """Durable WRITE-AHEAD dispatch marker (Q17; same principle as the commit
    gate's paid stamp): written immediately before the first physical reviewer
    transport call of ONE skill-review wave, shared by the lifecycle runner and
    direct ``review_skill`` callers. APPEND-ONLY per-wave storage: each wave
    writes ITS OWN marker file, so two concurrent waves on one skill can never
    overwrite each other's paid fact; each terminal-row merge clears exactly
    its own marker, and every still-unmerged marker keeps counting toward the
    ceiling (a crashed or swallowed wave spent the money). A sibling per-wave
    marker is NEVER flushed here — a concurrent wave is live by design, and
    flushing it as an infra terminal would let the idempotent merge (keyed by
    its wave id) refuse its REAL verdict row later. A LEGACY single-file
    marker from a pre-upgrade wave is read + flushed into the history as an
    infra terminal and removed. A failing write surfaces as a loud typed
    event — this is fail-open cost accounting, not a safety gate."""
    from ouroboros.utils import atomic_write_json

    legacy = _read_marker_file(legacy_dispatch_marker_path(drive_root, skill_name))
    if legacy.get("wave_id") and str(legacy["wave_id"]) != str(wave_id):
        # On success the idempotent append clears the legacy file itself (its
        # merge path recognises the legacy marker); on failure the file stays
        # so the paid fact keeps counting and a later write retries the flush.
        _flush_orphan_dispatch_marker(drive_root, skill_name, legacy)
    payload = {
        "ts": utc_now_iso(),
        "wave_id": str(wave_id),
        "group_id": str(group_id or ""),
        "content_hash": str(content_hash or ""),
        "root_task_id": str(root_task_id or ""),
        "paid": True,
        "usage_attribution_schema": USAGE_ATTRIBUTION_SCHEMA,
        "review_contract_fingerprint": str(review_contract_fingerprint or ""),
        "rebuttal_sha256": str(rebuttal_sha256 or ""),
    }
    try:
        path = _wave_marker_path(drive_root, skill_name, str(wave_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Same redaction invariant as the history rows: marker fields are all
        # hashes/ids today, so this is a no-op class guard, not a data change.
        atomic_write_json(path, _redact_history_payload(payload), trailing_newline=True)
    except Exception:
        log.warning("skill review dispatch marker write failed for %s", skill_name, exc_info=True)
        _emit_history_event(drive_root, {
            "type": "skill_review_history_append_failed",
            "skill": skill_name, "wave_id": str(wave_id),
            "reason": "dispatch marker write failed",
        })


def _flush_orphan_dispatch_marker(
    drive_root: pathlib.Path, skill_name: str, marker: Dict[str, Any]
) -> None:
    """A previous wave dispatched but never finalized (crash, or a direct-call
    infra outcome that returns without a history row): append its paid facts
    as an infra terminal row — idempotently keyed by the wave id — so the
    ledger catches up instead of forgetting the spend."""
    append_history_once(drive_root, skill_name, {
        "ts": utc_now_iso(),
        "status": "interrupted",
        "terminal_reason": "dispatched_wave_never_finalized",
        "content_hash": str(marker.get("content_hash") or ""),
        "group_id": str(marker.get("group_id") or ""),
        "root_task_id": str(marker.get("root_task_id") or ""),
        "paid": True,
        "usage_attribution_schema": str(marker.get("usage_attribution_schema") or ""),
        "review_contract_fingerprint": str(marker.get("review_contract_fingerprint") or ""),
        "rebuttal_sha256": str(marker.get("rebuttal_sha256") or ""),
        "job_id": str(marker.get("wave_id") or ""),
        "wave_id": str(marker.get("wave_id") or ""),
        "failure_signature": [],
        "fail_findings": [],
    })


def _merge_marker_facts(payload: Dict[str, Any], marker: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the write-ahead marker's paid facts into ITS wave's terminal row.

    The producer can lose the facts legitimately — a lifecycle timeout
    finalizes with no result object — but the marker recorded the dispatch
    before the first transport call, so the terminal row still carries
    ``paid``/contract/rebuttal. Rows of other waves pass through untouched;
    callers supply the exact per-wave or matching legacy marker."""
    row_wave = str(payload.get("wave_id") or payload.get("job_id") or "")
    wave = str(marker.get("wave_id") or "")
    if not wave or wave != row_wave:
        return payload
    merged = dict(payload)
    if marker.get("paid") and not merged.get("paid"):
        merged["paid"] = True
    for key in _MARKER_FACT_KEYS:
        if marker.get(key) and not merged.get(key):
            merged[key] = marker[key]
    merged.setdefault("wave_id", wave)
    return merged


def _marker_facts_landed(payload: Dict[str, Any], marker: Dict[str, Any]) -> bool:
    """Whether clearing this captured marker would lose no effective facts."""
    wave = str(marker.get("wave_id") or "")
    if not wave or str(payload.get("wave_id") or payload.get("job_id") or "") != wave:
        return False
    if marker.get("paid") and not payload.get("paid"):
        return False
    return all(
        not marker.get(key) or payload.get(key) == marker.get(key)
        for key in _MARKER_FACT_KEYS
    )


def finding_signature(findings: List[Dict[str, Any]]) -> List[str]:
    return sorted({
        f"{finding.get('item')}:{finding.get('verdict')}:{finding.get('severity')}"
        for finding in findings
        if isinstance(finding, dict) and str(finding.get("verdict") or "").upper() == "FAIL"
    })


def extract_fail_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for finding in findings:
        if not isinstance(finding, dict) or str(finding.get("verdict") or "").upper() != "FAIL":
            continue
        entry = {
            "item": str(finding.get("item") or "?"),
            "severity": str(finding.get("severity") or ""),
            "reason_excerpt": format_obligation_excerpt(str(finding.get("reason") or "")),
        }
        if finding.get("model"):
            entry["model"] = str(finding["model"])
        out.append(entry)
    return out


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
        review_round = max(
            group_rounds.get(group_id, 0) + 1,
            _ordinal(entry.get("review_round")),
        )
        group_rounds[group_id] = review_round
        attempt_key = (group_id, content_hash)
        snapshot_attempt = max(
            snapshot_attempts.get(attempt_key, 0) + 1,
            _ordinal(entry.get("snapshot_attempt")),
        )
        snapshot_attempts[attempt_key] = snapshot_attempt
        revised = bool(last_hash.get(group_id) and last_hash[group_id] != content_hash)
        if content_hash:
            last_hash[group_id] = content_hash
        entry.update(
            group_id=group_id,
            review_round=review_round,
            snapshot_attempt=snapshot_attempt,
            snapshot_revised=bool(entry.get("snapshot_revised", revised)),
        )
        out.append(entry)
    return out


def load_history(
    drive_root: pathlib.Path,
    skill_name: str,
    limit: int = 3,
    *,
    group_id: str = "",
) -> List[Dict[str, Any]]:
    """History rows inside the bounded tail window (CPL4-C12).

    Every reader is byte-bounded (the ``find_history_job_bounded`` idiom) so a
    growing per-skill log never turns a context build or a wave start into a
    whole-file scan — including the cross-skill paid-cycle count, which reads
    through ``iter_history_rows_bounded`` (it walks every installed skill, so
    it is the read the bound matters most for). Counter authorities stay exact
    across the bound because
    lifecycle terminal rows PERSIST their ordinals and ``normalize_history``
    takes ``max(stored, derived)`` — only a group whose newest ordinal-bearing
    row has aged past the window restarts low (disclosed degradation: the
    review-cycle ceiling under-counts, it never over-blocks).
    """
    try:
        raw_entries = list(iter_jsonl_objects(
            review_history_path(drive_root, skill_name),
            tail_bytes=_DETAIL_LOOKUP_MAX_BYTES,
        ))
        markers = {
            str(marker.get("wave_id") or ""): marker
            for marker in load_dispatch_markers(drive_root, skill_name)
        }
        entries = normalize_history(
            [
                _merge_marker_facts(
                    row, markers.get(str(row.get("wave_id") or row.get("job_id") or ""), {}),
                )
                for row in raw_entries
            ],
            skill_name,
        )
    except OSError:
        return []
    if group_id:
        entries = [entry for entry in entries if entry.get("group_id") == group_id]
    return entries[-limit:] if limit > 0 else entries


def iter_history_rows_bounded(
    drive_root: pathlib.Path, skill_name: str,
) -> Iterator[Dict[str, Any]]:
    """RAW history rows of one skill inside the same bounded tail window.

    ``load_history`` normalizes and merges dispatch markers; the cross-skill
    paid-cycle count needs the rows as written and must NOT carry the window
    size around itself — the bound belongs here, beside the other readers
    (CPL4-C12). This is the read that walks EVERY skill's log, so it is the one
    that must never turn into a whole-tree scan (audit #14-6b).
    """
    for row in iter_jsonl_objects(
        review_history_path(pathlib.Path(drive_root), skill_name),
        tail_bytes=_DETAIL_LOOKUP_MAX_BYTES,
    ):
        if isinstance(row, dict):
            yield row


def find_history_job_bounded(
    drive_root: pathlib.Path,
    skill_name: str,
    job_id: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Find one immutable terminal row inside a fixed tail window.

    Full-history authority readers remain unchanged. This helper exists only
    for lazy presentation: an old job outside the bounded tail is unavailable
    rather than making every detail expansion scan the growing domain log.
    """
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
        parsed = list(iter_jsonl_objects(
            path,
            max_entries=_DETAIL_LOOKUP_MAX_RECORDS + 1,
            tail_bytes=_DETAIL_LOOKUP_MAX_BYTES,
            gap_reasons=gaps,
        ))
    except FileNotFoundError:
        return None, "absent"
    except OSError:
        return None, "io_error"
    records_truncated = len(parsed) > _DETAIL_LOOKUP_MAX_RECORDS
    if records_truncated:
        parsed = parsed[-_DETAIL_LOOKUP_MAX_RECORDS:]
    source_record = next(
        (row for row in reversed(parsed) if str(row.get("job_id") or "") == job_id),
        None,
    )
    # max_entries counts raw tail lines while parsed contains valid objects.
    # Any malformed/partial raw row can hide ordinal-bearing authority just as
    # surely as byte or record truncation.
    projection_incomplete = byte_window_truncated or records_truncated or bool(gaps)
    if source_record is not None:
        marker = load_dispatch_marker_for_wave(
            drive_root,
            skill_name,
            str(source_record.get("wave_id") or source_record.get("job_id") or ""),
        )
        source_record = _merge_marker_facts(source_record, marker)
        ordinal_fields = ("review_round", "snapshot_attempt", "snapshot_revised")
        if projection_incomplete and any(key not in source_record for key in ordinal_fields):
            return None, "unavailable"
        if projection_incomplete:
            normalized = _normalize_bounded_ordinals(source_record)
            if normalized is None:
                return None, "unavailable"
            return normalized, "found"
        normalized = normalize_history([
            _merge_marker_facts(
                row,
                load_dispatch_marker_for_wave(
                    drive_root,
                    skill_name,
                    str(row.get("wave_id") or row.get("job_id") or ""),
                ),
            )
            for row in parsed
        ], skill_name)
        record = next(
            (row for row in reversed(normalized) if str(row.get("job_id") or "") == job_id),
            None,
        )
        return record, "found"
    if projection_incomplete:
        return None, "unavailable"
    return None, "not_found" if parsed else "absent"


def allocate_ordinals(
    drive_root: pathlib.Path,
    skill_name: str,
    group_id: str,
    content_hash: str,
) -> tuple[int, int, bool]:
    history = load_history(drive_root, skill_name, limit=0, group_id=group_id)
    review_round = max(
        (_ordinal(row.get("review_round")) for row in history), default=0,
    ) + 1
    snapshot_attempt = max(
        (
            _ordinal(row.get("snapshot_attempt"))
            for row in history
            if str(row.get("content_hash") or "") == content_hash
        ),
        default=0,
    ) + 1
    previous_hash = str(history[-1].get("content_hash") or "") if history else ""
    return review_round, snapshot_attempt, bool(previous_hash and previous_hash != content_hash)


def count_attempts(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    *,
    group_id: str = "",
) -> int:
    history = load_history(drive_root, skill_name, limit=0, group_id=group_id)
    return sum(1 for row in history if str(row.get("content_hash") or "") == content_hash)


def append_history(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    status: str,
    content_hash: str,
    findings: List[Dict[str, Any]],
    raw_actor_records: Optional[List[Dict[str, Any]]] = None,
    single_reviewer_no_diversity: bool = False,
    paid: bool = False,
    review_contract_fingerprint: str = "",
    rebuttal_sha256: str = "",
    replayed_from_ts: str = "",
    wave_id: str = "",
) -> None:
    try:
        payload: Dict[str, Any] = {
            "ts": utc_now_iso(),
            "status": status,
            "content_hash": content_hash,
            "failure_signature": finding_signature(findings),
            "fail_findings": extract_fail_findings(findings),
        }
        if single_reviewer_no_diversity:
            payload["single_reviewer_no_diversity"] = True
        if raw_actor_records:
            payload["raw_actor_records"] = list(raw_actor_records)
        # Max-Review-Cycles facts (Q17/Q23): the paid-dispatch fact and the
        # panel contract identity ride the history row — counts and free-replay
        # decisions are DERIVED from this ledger (P7, no counter file).
        if paid:
            payload["paid"] = True
        if review_contract_fingerprint:
            payload["review_contract_fingerprint"] = str(review_contract_fingerprint)
        if rebuttal_sha256:
            payload["rebuttal_sha256"] = str(rebuttal_sha256)
        if replayed_from_ts:
            payload["replayed_from_ts"] = str(replayed_from_ts)
        if wave_id:
            payload["wave_id"] = str(wave_id)
        marker = load_dispatch_marker_for_wave(drive_root, skill_name, str(wave_id or ""))
        payload = _merge_marker_facts(payload, marker)
        if not append_jsonl(
            review_history_path(drive_root, skill_name),
            _redact_history_payload(payload),
        ):
            raise OSError("append_jsonl reported failure")
        if _marker_facts_landed(payload, marker):
            clear_dispatch_marker(drive_root, skill_name, wave_id=wave_id)
    except Exception:
        # LOUD failure (F3): a lost terminal row silently un-counts spent
        # money and hides a verdict — log at warning and emit the typed event.
        log.warning("skill review history append failed for %s", skill_name, exc_info=True)
        _emit_history_event(drive_root, {
            "type": "skill_review_history_append_failed",
            "skill": skill_name,
            "status": str(status or ""),
            "content_hash": str(content_hash or ""),
            "wave_id": str(wave_id or ""),
            "reason": "direct history append failed",
        })


def append_history_once(
    drive_root: pathlib.Path,
    skill_name: str,
    payload: Dict[str, Any],
) -> bool:
    """Append one lifecycle terminal row, idempotently keyed by ``job_id``."""
    job_id = str(payload.get("job_id") or "")
    if not job_id:
        return False
    path = review_history_path(drive_root, skill_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        return False
    try:
        try:
            # Same bounded window as every other reader (CPL4-C12): an
            # idempotent retry lands within its wave's lifetime, far inside
            # the tail — never a whole-file scan per terminal write.
            existing = next(
                (
                    row
                    for row in iter_jsonl_objects(path, tail_bytes=_DETAIL_LOOKUP_MAX_BYTES)
                    if str(row.get("job_id") or "") == job_id
                ),
                None,
            )
            marker = load_dispatch_marker_for_wave(drive_root, skill_name, job_id)
            if existing is not None:
                # Already landed (idempotent retry): finish the merge by
                # clearing only a marker whose facts are already in the row.
                if _marker_facts_landed(existing, marker):
                    clear_dispatch_marker(drive_root, skill_name, wave_id=job_id)
                if not _append_root_task_projection_once(drive_root, skill_name, existing):
                    log.warning("skill review root-task projection did not land for %s", skill_name)
                    _record_root_task_projection_gap(drive_root, skill_name, existing)
                    return False
                return True
            payload = _merge_marker_facts(payload, marker)
            safe_payload = _redact_history_payload(payload)
            data = (json.dumps(safe_payload, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            if _marker_facts_landed(payload, marker):
                clear_dispatch_marker(
                    drive_root, skill_name,
                    wave_id=str(payload.get("wave_id") or job_id),
                )
            if not _append_root_task_projection_once(drive_root, skill_name, safe_payload):
                log.warning("skill review root-task projection did not land for %s", skill_name)
                _record_root_task_projection_gap(drive_root, skill_name, safe_payload)
                return False
            return True
        except OSError:
            log.warning("skill review terminal history append failed for %s", skill_name, exc_info=True)
            return False
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def _append_root_task_projection_once(
    drive_root: pathlib.Path, skill_name: str, payload: Dict[str, Any],
) -> bool:
    """Append the derived root-task row once, checking its whole projection."""
    root_task_id = str(payload.get("root_task_id") or "")
    outcome_id = str(payload.get("job_id") or payload.get("wave_id") or "")
    if not root_task_id or not outcome_id:
        return True
    path = root_task_projection_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        return False
    try:
        rows = iter_jsonl_objects(path)
        if any(
            str(row.get("root_task_id") or "") == root_task_id
            and str(row.get("skill") or "") == skill_name
            and str(row.get("job_id") or row.get("wave_id") or "") == outcome_id
            for row in rows
        ):
            return True
        row = {
            "ts": str(payload.get("ts") or ""), "root_task_id": root_task_id,
            "skill": skill_name, "job_id": outcome_id,
        }
        data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except OSError:
        log.warning("skill review root-task projection append failed for %s", skill_name)
        return False
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
