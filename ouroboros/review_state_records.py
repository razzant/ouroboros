"""Record types of the review ledger and the pure rules that shape them.

The obligation, readiness-debt, advisory-run and commit-attempt records, the
retention and TTL bounds they are trimmed to, the repo-scope filter that keeps a
multi-repo ledger honest, obligation fingerprinting and id allocation, attempt
identity/ordering and merging, and the timestamp helpers. Every rule here is a
pure function of its inputs. Extracted from ouroboros/review_state.py (v7 D06
split, re-cut on the v7next tip); review_state.py re-exports every name.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _rs():
    """The parent review-state module, read at call time.

    The review-state members stay monkeypatch-addressable at their historical
    ``ouroboros.review_state`` bindings (tests rebind them there), so this leaf
    resolves every such cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_state

    return review_state


_STATE_SCHEMA_VERSION = 3


_MAX_RUN_HISTORY = 10


_MAX_ATTEMPT_HISTORY = 50


_MAX_COMMIT_READINESS_DEBTS = 50


_DEFAULT_TOOL_NAME = "commit_reviewed"


_DEFAULT_ADVISORY_TOOL_NAME = "advisory_review"


_LEGACY_CURRENT_REPO_KEY = "__legacy_current_repo__"


_REVIEW_ATTEMPT_TTL_SEC = 1800


_REVIEW_ATTEMPT_GRACE_SEC = 120


_OPEN_COMMIT_READINESS_DEBT_STATUSES = frozenset({"detected", "queued", "reopened"})


_CANONICAL_OBLIGATION_ITEM_RE = re.compile(r"[a-z0-9_]+")


def _normalize_fingerprint_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _normalize_obligation_item_key(item_name: Any) -> str:
    text = _normalize_fingerprint_text(item_name)
    if not text:
        return ""
    if text.startswith("bug_") or text.startswith("risk_"):
        return ""
    if not _CANONICAL_OBLIGATION_ITEM_RE.fullmatch(text):
        return ""
    return text


def _stable_digest(*parts: Any) -> str:
    key = " | ".join(_normalize_fingerprint_text(part) for part in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _make_obligation_fingerprint(item: Any, reason: Any) -> str:
    canonical_item = _normalize_obligation_item_key(item)
    if canonical_item:
        # Include reason so same checklist item with different bugs does not coalesce.
        return f"finding:{canonical_item}:{_stable_digest(canonical_item, reason)}"
    return f"finding:{_stable_digest(item, reason)}"


def _looks_like_public_obligation_id(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.fullmatch(r"obl-\d{4,}", text))


def _max_iso_ts(left: str, right: str) -> str:
    return max(str(left or ""), str(right or ""))


def _min_iso_ts(left: str, right: str) -> str:
    candidates = [str(value or "") for value in (left, right) if str(value or "")]
    if not candidates:
        return ""
    return min(candidates)


def _filter_repo_scope(records: List[Any], repo_key: str | None) -> List[Any]:
    if repo_key is None:
        return list(records)
    exact_match_exists = any(str(getattr(record, "repo_key", "") or "") == repo_key for record in records)
    return [
        record
        for record in records
        if (str(getattr(record, "repo_key", "") or "") == repo_key)
        or (
            not exact_match_exists
            and str(getattr(record, "repo_key", "") or "") in ("", _LEGACY_CURRENT_REPO_KEY)
        )
    ]


def _commit_readiness_debts_view(state: Any) -> List["CommitReadinessDebtItem"]:
    debts = getattr(state, "commit_readiness_debts", None)
    if isinstance(debts, list):
        return debts
    debts = list(debts or [])
    setattr(state, "commit_readiness_debts", debts)
    return debts


_OBLIGATION_STR_DEFAULTS = {"obligation_id": "", "item": "", "severity": "critical", "reason": "", "source_attempt_ts": "", "source_attempt_msg": "", "status": "still_open", "resolved_by": "", "repo_key": _LEGACY_CURRENT_REPO_KEY}


_DEBT_STR_DEFAULTS = {"debt_id": "", "category": "", "summary": "", "severity": "warning", "status": "detected", "repo_key": _LEGACY_CURRENT_REPO_KEY, "fingerprint": "", "title": "Commit readiness debt", "source": "review_state", "first_seen_at": "", "last_seen_at": "", "updated_at": "", "verified_at": ""}


_RUN_STR_DEFAULTS = {"snapshot_hash": "", "commit_message": "", "status": "stale", "snapshot_summary": "", "raw_result": "", "reason_kind": "", "bypass_reason": "", "bypassed_by_task": "", "repo_key": _LEGACY_CURRENT_REPO_KEY, "tool_name": _DEFAULT_ADVISORY_TOOL_NAME, "phase": "advisory", "model_used": "", "session_id": "", "review_rebuttal": ""}


_ATTEMPT_STR_DEFAULTS = {"commit_message": "", "snapshot_hash": "", "block_reason": "", "block_details": "", "task_id": "", "repo_key": _LEGACY_CURRENT_REPO_KEY, "tool_name": _DEFAULT_TOOL_NAME, "pre_review_fingerprint": "", "post_review_fingerprint": "", "fingerprint_status": "", "scope_model": "", "block_class": "", "rebuttal_sha256": "", "review_contract_fingerprint": "", "review_retry_key": "", "root_task_id": "", "review_owner_session_id": ""}


_ATTEMPT_MERGE_INCOMING_FIRST = ("ts", "commit_message", "status", "snapshot_hash", "block_reason", "block_details", "duration_sec", "task_id", "repo_key", "tool_name", "phase", "pre_review_fingerprint", "post_review_fingerprint", "fingerprint_status", "scope_model", "block_class", "rebuttal_sha256", "review_contract_fingerprint", "review_retry_key", "root_task_id", "review_owner_session_id")


_ATTEMPT_MERGE_INCOMING_LISTS = ("critical_findings", "advisory_findings", "obligation_ids", "readiness_warnings")


_RUN_STATUS_ICONS = {"fresh": "✅", "stale": "⚠️", "bypassed": "⏭️", "skipped": "⏭️", "parse_failure": "🔴"}


def _filter_lifecycle_records(
    records: List[Any],
    *,
    repo_key: str | None = None,
    tool_name: str | None = None,
    task_id: str | None = None,
    attempt: int | None = None,
) -> List[Any]:
    results = _filter_repo_scope(records, repo_key)
    return [
        record
        for record in results
        if (tool_name is None or str(getattr(record, "tool_name", "") or "") == tool_name)
        and (task_id is None or str(getattr(record, "task_id", "") or "") == task_id)
        and (attempt is None or int(getattr(record, "attempt", 0) or 0) == int(attempt))
    ]


def _allocate_prefixed_id(items: List[Any], attr: str, next_seq: int, prefix: str) -> tuple[str, int]:
    used = {str(getattr(item, attr, "") or "").strip() for item in items if str(getattr(item, attr, "") or "").strip()}
    seq = max(1, int(next_seq or 1))
    while True:
        candidate = f"{prefix}{seq:04d}"
        seq += 1
        if candidate not in used:
            return candidate, seq


def _append_finding_lines(
    lines: List[str],
    findings: List[Dict[str, Any]],
    header: str,
    *,
    limit: int | None = None,
    with_severity: bool = False,
) -> None:
    lines.append(f"   {header} ({len(findings)}):")
    for finding in findings:
        label = str(finding.get("item", "?") if with_severity else finding.get("item") or finding.get("reason") or "?")
        reason = _rs()._truncate_review_reason(finding.get("reason", ""), limit=limit or 120)
        prefix = f"[{str(finding.get('severity', 'advisory')).upper()}] " if with_severity else "- "
        lines.append(f"     {prefix}{label}: {reason}")


@dataclass
class ObligationItem:
    """Unresolved obligation from a blocking commit attempt."""

    obligation_id: str
    item: str
    severity: str
    reason: str
    source_attempt_ts: str
    source_attempt_msg: str
    status: str = "still_open"
    resolved_by: str = ""
    repo_key: str = _LEGACY_CURRENT_REPO_KEY
    fingerprint: str = ""
    created_ts: str = ""
    updated_ts: str = ""


@dataclass
class CommitReadinessDebtItem:
    """Repo-scoped readiness debt derived from review friction."""

    debt_id: str
    category: str
    summary: str
    severity: str = "warning"
    status: str = "detected"
    repo_key: str = _LEGACY_CURRENT_REPO_KEY
    fingerprint: str = ""
    title: str = "Commit readiness debt"
    source: str = "review_state"
    source_obligation_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    first_seen_at: str = ""
    last_seen_at: str = ""
    updated_at: str = ""
    verified_at: str = ""
    occurrence_count: int = 0
    consecutive_observations: int = 0


@dataclass
class AdvisoryRunRecord:
    """Completed advisory pre-review run."""

    snapshot_hash: str
    commit_message: str
    status: str
    ts: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_summary: str = ""
    raw_result: str = ""
    # Typed cause for status="preflight_blocked" rows: "syntax" (a staged .py
    # failed compile) or "release_metadata" (deterministic release preflight).
    # "" = unknown/legacy — guidance must then point at raw_result instead of
    # asserting a specific problem class (H4, capinv-447).
    reason_kind: str = ""
    review_rebuttal: str = ""  # full argument delivered to this exact preflight
    # Existing delegate invocation owns the immutable request; this row binds
    # its intent/candidate and retains unresolved execution across a restart.
    execution: Dict[str, Any] = field(default_factory=dict)
    bypass_reason: str = ""
    bypassed_by_task: str = ""
    snapshot_paths: Optional[List[str]] = field(default=None)
    repo_key: str = _LEGACY_CURRENT_REPO_KEY
    tool_name: str = _DEFAULT_ADVISORY_TOOL_NAME
    task_id: str = ""
    attempt: int = 0
    phase: str = "advisory"
    created_ts: str = ""
    updated_ts: str = ""
    readiness_warnings: List[str] = field(default_factory=list)
    prompt_chars: int = 0
    model_used: str = ""
    session_id: str = ""
    duration_sec: float = 0.0


@dataclass
class CommitAttemptRecord:
    """Reviewed mutative tool attempt lifecycle record."""

    ts: str
    commit_message: str
    status: str
    snapshot_hash: str = ""
    block_reason: str = ""
    block_details: str = ""
    duration_sec: float = 0.0
    task_id: str = ""
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    repo_key: str = _LEGACY_CURRENT_REPO_KEY
    tool_name: str = _DEFAULT_TOOL_NAME
    attempt: int = 0
    phase: str = "review"
    blocked: bool = False
    advisory_findings: List[Dict[str, Any]] = field(default_factory=list)
    obligation_ids: List[str] = field(default_factory=list)
    readiness_warnings: List[str] = field(default_factory=list)
    late_result_pending: bool = False
    pre_review_fingerprint: str = ""
    post_review_fingerprint: str = ""
    fingerprint_status: str = ""  # "pending" | "matched" | "mismatch" | "unavailable"
    degraded_reasons: List[str] = field(default_factory=list)
    started_ts: str = ""
    updated_ts: str = ""
    finished_ts: str = ""
    triad_models: List[str] = field(default_factory=list)
    scope_model: str = ""
    triad_raw_results: List[Dict[str, Any]] = field(default_factory=list)
    scope_raw_result: Dict[str, Any] = field(default_factory=dict)
    # Max-Review-Cycles semantics (owner Q12/Q16/Q22/Q23): the typed class of a
    # blocked row ("verdict" = reviewer findings, "infra" = fit/quorum/transport/
    # revalidation facts, "" = preflight/legacy), the content hash of the rebuttal
    # supplied with the attempt (absent on old rows = no rebuttal), whether a paid
    # triad/scope dispatch physically happened for this attempt (recorded at
    # dispatch time, plan-review precedent), the review-contract fingerprint the
    # attempt ran under (roster+routes+enforcement+prompt contract; a changed
    # fingerprint lapses free-replay/refusal authority), and the root task the
    # attempt belongs to (the whole task tree shares one paid-cycle ceiling).
    block_class: str = ""
    rebuttal_sha256: str = ""
    paid: bool = False
    review_contract_fingerprint: str = ""
    review_retry_key: str = ""
    root_task_id: str = ""
    # Existing process-custody identity for the process that owns every
    # process-local reviewer thread in this paid attempt. A sibling worker boot
    # shares the session but has another pid; neither fact alone proves loss.
    review_owner_session_id: str = ""
    review_owner_pid: int = 0
    # True once the row's heavy forensic payloads (raw reviewer results, full
    # free text) were compacted because the preserved accounting row fell
    # outside the newest-50 ledger window (see _strip_attempt_heavy_payload).
    raw_stripped: bool = False


def _attempt_identity_tuple(attempt: CommitAttemptRecord) -> tuple[str, str, str, str]:
    attempt_number = int(attempt.attempt or 0)
    identity_token = (
        f"attempt:{attempt_number}"
        if attempt_number > 0
        else f"ts:{attempt.started_ts or attempt.ts or ''}"
    )
    return (
        str(attempt.repo_key or _LEGACY_CURRENT_REPO_KEY),
        str(attempt.tool_name or _DEFAULT_TOOL_NAME),
        str(attempt.task_id or ""),
        identity_token,
    )


def _attempt_order_key(attempt: CommitAttemptRecord) -> tuple[float, int, str]:
    ts_value = (
        str(getattr(attempt, "finished_ts", "") or "")
        or str(getattr(attempt, "updated_ts", "") or "")
        or str(getattr(attempt, "started_ts", "") or "")
        or str(getattr(attempt, "ts", "") or "")
    )
    ts_epoch = _parse_iso_ts(ts_value)
    return (
        ts_epoch if ts_epoch is not None else 0.0,
        int(getattr(attempt, "attempt", 0) or 0),
        ts_value,
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _infer_next_prefixed_sequence(items: List[Any], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    max_seen = 0
    for item in items:
        value = str(getattr(item, "obligation_id", "") or getattr(item, "debt_id", "") or "").strip()
        match = pattern.fullmatch(value)
        if not match:
            continue
        max_seen = max(max_seen, _coerce_int(match.group(1), 0))
    return max_seen + 1 if max_seen > 0 else 1


def _normalize_findings(items: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif item:
            normalized.append({"reason": str(item), "severity": "advisory"})
    return normalized


def _merge_attempt(existing: CommitAttemptRecord, incoming: CommitAttemptRecord) -> CommitAttemptRecord:
    data = {
        name: getattr(incoming, name) or getattr(existing, name)
        for name in _ATTEMPT_MERGE_INCOMING_FIRST
    }
    data.update({name: list(getattr(incoming, name)) for name in _ATTEMPT_MERGE_INCOMING_LISTS})
    data.update(
        attempt=int(incoming.attempt or existing.attempt or 0),
        blocked=bool(incoming.blocked or incoming.status == "blocked"),
        late_result_pending=bool(incoming.late_result_pending),
        degraded_reasons=list(incoming.degraded_reasons or existing.degraded_reasons),
        started_ts=existing.started_ts or incoming.started_ts or existing.ts,
        updated_ts=incoming.updated_ts or existing.updated_ts or _utc_now(),
        finished_ts=incoming.finished_ts or existing.finished_ts,
        triad_models=list(incoming.triad_models or existing.triad_models),
        triad_raw_results=list(getattr(incoming, "triad_raw_results", None) or getattr(existing, "triad_raw_results", None) or []),
        scope_raw_result=dict(getattr(incoming, "scope_raw_result", None) or getattr(existing, "scope_raw_result", None) or {}),
        # Once an attempt physically dispatched a paid triad/scope wave the fact is
        # durable: a later terminal update on the same row must never launder it.
        paid=bool(getattr(incoming, "paid", False) or getattr(existing, "paid", False)),
        review_owner_pid=int(
            getattr(incoming, "review_owner_pid", 0)
            or getattr(existing, "review_owner_pid", 0)
            or 0
        ),
        # The compaction mark is sticky too — an unlikely late merge onto an
        # over-window row must not present a stripped row as a full one.
        raw_stripped=bool(
            getattr(incoming, "raw_stripped", False) or getattr(existing, "raw_stripped", False)
        ),
    )
    merged = CommitAttemptRecord(**data)
    if merged.raw_stripped:
        # A late terminal merging onto an already-compacted row (a stale
        # in-flight row that slid past the newest-50 window) must not
        # resurrect heavy reviewer raw payloads onto it: re-strip the merged
        # result — accounting facts (paid/root_task_id/fingerprints/
        # block_class/critical_findings) all survive the strip by design.
        _rs()._strip_attempt_heavy_payload(merged, force=True)
    return merged


def infer_review_phase(status: str, block_reason: str = "") -> str:
    """Map an attempt status/block_reason pair to its review phase (SSOT)."""
    if status == "reviewing":
        return "review"
    if status == "blocked":
        if block_reason == "no_advisory":
            return "advisory_gate"
        if block_reason == "preflight":
            return "preflight"
        return "blocking_review"
    if status == "succeeded":
        return "commit"
    if status == "failed":
        return "infra"
    return "review"


def _parse_iso_ts(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _dedupe_strings(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _utc_now() -> str:
    from ouroboros.utils import utc_now_iso
    return utc_now_iso()
