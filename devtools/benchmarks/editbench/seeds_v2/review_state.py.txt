"""Durable advisory/review ledger persisted in state/advisory_review.json."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import (
    atomic_write_json,
    truncate_review_artifact as _truncate_review_artifact,
    truncate_review_artifact as _truncate_review_reason,
)
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

log = logging.getLogger(__name__)

_STATE_RELPATH = "state/advisory_review.json"
_LOCK_RELPATH = "locks/advisory_review.lock"
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
_RUN_STR_DEFAULTS = {"snapshot_hash": "", "commit_message": "", "status": "stale", "snapshot_summary": "", "raw_result": "", "bypass_reason": "", "bypassed_by_task": "", "repo_key": _LEGACY_CURRENT_REPO_KEY, "tool_name": _DEFAULT_ADVISORY_TOOL_NAME, "phase": "advisory", "model_used": "", "session_id": ""}
_ATTEMPT_STR_DEFAULTS = {"commit_message": "", "snapshot_hash": "", "block_reason": "", "block_details": "", "task_id": "", "repo_key": _LEGACY_CURRENT_REPO_KEY, "tool_name": _DEFAULT_TOOL_NAME, "pre_review_fingerprint": "", "post_review_fingerprint": "", "fingerprint_status": "", "scope_model": ""}
_ATTEMPT_MERGE_INCOMING_FIRST = ("ts", "commit_message", "status", "snapshot_hash", "block_reason", "block_details", "duration_sec", "task_id", "repo_key", "tool_name", "phase", "pre_review_fingerprint", "post_review_fingerprint", "fingerprint_status", "scope_model")
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
        reason = _truncate_review_reason(finding.get("reason", ""), limit=limit or 120)
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


@dataclass
class AdvisoryReviewState:
    """Top-level durable review state."""

    state_version: int = _STATE_SCHEMA_VERSION
    advisory_runs: List[AdvisoryRunRecord] = field(default_factory=list)
    attempts: List[CommitAttemptRecord] = field(default_factory=list)
    open_obligations: List[ObligationItem] = field(default_factory=list)
    next_obligation_seq: int = 1
    commit_readiness_debts: List[CommitReadinessDebtItem] = field(default_factory=list)
    next_commit_readiness_debt_seq: int = 1
    last_stale_from_edit_ts: str = ""
    last_stale_reason: str = ""
    last_stale_repo_key: str = ""

    def latest(self) -> Optional[AdvisoryRunRecord]:
        return self.advisory_runs[-1] if self.advisory_runs else None

    def latest_attempt(self) -> Optional[CommitAttemptRecord]:
        return self.attempts[-1] if self.attempts else None

    def latest_attempt_for(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> Optional[CommitAttemptRecord]:
        matches = self.filter_attempts(
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
            attempt=attempt,
        )
        return matches[-1] if matches else None

    def get_active_attempts(self, *, repo_key: str | None = None) -> List[CommitAttemptRecord]:
        active = [
            item for item in self.attempts
            if item.status == "reviewing" or item.late_result_pending
        ]
        return _filter_repo_scope(active, repo_key)

    def filter_advisory_runs(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> List[AdvisoryRunRecord]:
        return _filter_lifecycle_records(
            self.advisory_runs,
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
            attempt=attempt,
        )

    def filter_attempts(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> List[CommitAttemptRecord]:
        return _filter_lifecycle_records(
            self.attempts,
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
            attempt=attempt,
        )

    def next_attempt_number(self, repo_key: str, tool_name: str, task_id: str = "") -> int:
        candidates = self.filter_attempts(repo_key=repo_key, tool_name=tool_name, task_id=task_id)
        latest = max((int(item.attempt or 0) for item in candidates), default=0)
        return latest + 1

    def next_advisory_attempt_number(
        self,
        repo_key: str,
        task_id: str = "",
        tool_name: str = _DEFAULT_ADVISORY_TOOL_NAME,
    ) -> int:
        candidates = self.filter_advisory_runs(
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
        )
        latest = max((int(run.attempt or 0) for run in candidates), default=0)
        return latest + 1

    def find_by_hash(
        self,
        snapshot_hash: str,
        repo_key: str | None = None,
    ) -> Optional[AdvisoryRunRecord]:
        for run in reversed(_filter_repo_scope(self.advisory_runs, repo_key)):
            if run.snapshot_hash != snapshot_hash:
                continue
            return run
        return None

    def is_fresh(self, snapshot_hash: str, repo_key: str | None = None) -> bool:
        run = self.find_by_hash(snapshot_hash, repo_key=repo_key)
        return run is not None and run.status in ("fresh", "bypassed", "skipped")

    def add_run(self, run: AdvisoryRunRecord) -> None:
        if not run.attempt:
            run.attempt = self.next_advisory_attempt_number(
                str(run.repo_key or _LEGACY_CURRENT_REPO_KEY),
                str(run.task_id or ""),
                str(run.tool_name or _DEFAULT_ADVISORY_TOOL_NAME),
            )
        if not run.created_ts:
            run.created_ts = run.ts or _utc_now()
        if not run.updated_ts:
            run.updated_ts = run.created_ts
        self.mark_all_stale_except(run.snapshot_hash, repo_key=run.repo_key)
        self.advisory_runs.append(run)
        if len(self.advisory_runs) > _MAX_RUN_HISTORY:
            self.advisory_runs = self.advisory_runs[-_MAX_RUN_HISTORY:]
        if run.status in ("fresh", "bypassed", "skipped", "parse_failure"):
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
        self._sync_commit_readiness_debts(repo_key=run.repo_key or None)

    def mark_stale(self, snapshot_hash: str) -> None:
        for run in self.advisory_runs:
            if run.snapshot_hash == snapshot_hash:
                run.status = "stale"
                run.updated_ts = _utc_now()

    def mark_all_stale_except(self, snapshot_hash: str, repo_key: str = "") -> None:
        for run in self.advisory_runs:
            same_repo = not repo_key or run.repo_key == repo_key
            if same_repo and run.snapshot_hash != snapshot_hash and run.status in ("fresh", "bypassed", "skipped"):
                run.status = "stale"
                run.updated_ts = _utc_now()

    def mark_repo_stale(
        self,
        *,
        repo_key: str = "",
        reason_ts: str = "",
        reason: str = "",
        stale_repo_key: str = "",
    ) -> int:
        """Invalidate advisory runs for a repo, falling back conservatively."""
        invalidatable = [
            run for run in self.advisory_runs
            if run.status in ("fresh", "bypassed", "skipped")
        ]
        if not invalidatable:
            return 0

        if not repo_key:
            target_runs = invalidatable
        else:
            exact_matches = [run for run in invalidatable if run.repo_key == repo_key]
            legacy_present = any(run.repo_key in ("", _LEGACY_CURRENT_REPO_KEY) for run in invalidatable)
            target_runs = invalidatable if legacy_present and not exact_matches else (exact_matches or invalidatable)

        for run in target_runs:
            run.status = "stale"
            run.updated_ts = reason_ts or _utc_now()
        if target_runs:
            self.last_stale_from_edit_ts = reason_ts or _utc_now()
            self.last_stale_reason = reason
            self.last_stale_repo_key = stale_repo_key or repo_key
            self._sync_commit_readiness_debts(repo_key=stale_repo_key or repo_key or None)
        return len(target_runs)

    def add_blocking_attempt(self, attempt: CommitAttemptRecord) -> None:
        """Compatibility alias for existing callers/tests."""
        attempt.status = "blocked"
        attempt.blocked = True
        self.record_attempt(attempt)

    def record_attempt(
        self, attempt: CommitAttemptRecord, *, semantic_redirects: Optional[Dict[str, str]] = None
    ) -> CommitAttemptRecord:
        """Upsert one reviewed attempt into durable state. ``semantic_redirects`` maps a
        free-text finding fingerprint to an existing open obligation id (computed OUTSIDE
        the lock by the caller, C9.3) so a reworded restatement of an open obligation
        folds into it instead of opening a duplicate."""
        now = _utc_now()
        attempt.tool_name = str(attempt.tool_name or _DEFAULT_TOOL_NAME)
        attempt.repo_key = str(attempt.repo_key or _LEGACY_CURRENT_REPO_KEY)
        attempt.blocked = bool(attempt.blocked or attempt.status == "blocked")
        if not attempt.started_ts:
            attempt.started_ts = attempt.ts or now
        if not attempt.ts:
            attempt.ts = attempt.started_ts
        attempt.updated_ts = now
        if attempt.status in ("blocked", "failed", "succeeded") and not attempt.finished_ts:
            attempt.finished_ts = now

        merged = self._upsert_attempt(attempt)

        if merged.status == "blocked" or merged.blocked:
            merged.blocked = True
            merged.obligation_ids = self._update_obligations_from_attempt(
                merged, semantic_redirects=semantic_redirects
            )
            self._upsert_attempt(merged)
        elif merged.status == "succeeded":
            self.on_successful_commit(repo_key=merged.repo_key)
        self._sync_commit_readiness_debts(repo_key=merged.repo_key or None)

        return merged

    def _upsert_attempt(self, attempt: CommitAttemptRecord) -> CommitAttemptRecord:
        key = _attempt_identity_tuple(attempt)
        for idx, existing in enumerate(self.attempts):
            if _attempt_identity_tuple(existing) == key:
                merged = _merge_attempt(existing, attempt)
                self.attempts[idx] = merged
                return merged
        self.attempts.append(attempt)
        if len(self.attempts) > _MAX_ATTEMPT_HISTORY:
            self.attempts = self.attempts[-_MAX_ATTEMPT_HISTORY:]
        return attempt

    def _allocate_obligation_id(self) -> str:
        candidate, next_seq = _allocate_prefixed_id(
            self.open_obligations,
            "obligation_id",
            self.next_obligation_seq,
            "obl-",
        )
        self.next_obligation_seq = next_seq
        return candidate

    def _hydrate_obligation(self, obligation: ObligationItem) -> None:
        obligation.repo_key = str(obligation.repo_key or _LEGACY_CURRENT_REPO_KEY)
        obligation.fingerprint = str(
            obligation.fingerprint
            or _make_obligation_fingerprint(obligation.item, obligation.reason)
        )
        base_ts = (
            str(obligation.updated_ts or "")
            or str(obligation.created_ts or "")
            or str(obligation.source_attempt_ts or "")
            or _utc_now()
        )
        if not obligation.created_ts:
            obligation.created_ts = str(obligation.source_attempt_ts or base_ts)
        if not obligation.updated_ts:
            obligation.updated_ts = str(obligation.source_attempt_ts or obligation.created_ts)

    def _coalesce_open_obligations(self) -> None:
        merged_open: Dict[tuple[str, str], ObligationItem] = {}
        ordered: List[ObligationItem] = []
        for obligation in list(self.open_obligations or []):
            self._hydrate_obligation(obligation)
            if obligation.status != "still_open":
                ordered.append(obligation)
                continue
            merge_key = (obligation.repo_key, obligation.fingerprint or obligation.obligation_id)
            existing = merged_open.get(merge_key)
            if existing is None:
                merged_open[merge_key] = obligation
                ordered.append(obligation)
                continue
            if (
                not _looks_like_public_obligation_id(existing.obligation_id)
                and _looks_like_public_obligation_id(obligation.obligation_id)
            ):
                existing.obligation_id = obligation.obligation_id
            if not existing.item and obligation.item:
                existing.item = obligation.item
            if not existing.reason and obligation.reason:
                existing.reason = obligation.reason
            if not existing.severity and obligation.severity:
                existing.severity = obligation.severity
            if obligation.source_attempt_ts and (
                obligation.source_attempt_ts >= existing.source_attempt_ts
            ):
                existing.source_attempt_ts = obligation.source_attempt_ts
                if obligation.source_attempt_msg:
                    existing.source_attempt_msg = obligation.source_attempt_msg
            existing.created_ts = _min_iso_ts(existing.created_ts, obligation.created_ts)
            existing.updated_ts = _max_iso_ts(existing.updated_ts, obligation.updated_ts)
        self.open_obligations = ordered

    def _touch_obligation(
        self,
        obligation: ObligationItem,
        attempt: CommitAttemptRecord,
        *,
        item: str,
        reason: str,
        severity: str,
    ) -> None:
        seen_ts = str(attempt.ts or _utc_now())
        obligation.item = str(obligation.item or item or "")
        obligation.severity = str(obligation.severity or severity or "critical")
        obligation.repo_key = str(obligation.repo_key or attempt.repo_key or _LEGACY_CURRENT_REPO_KEY)
        if not obligation.reason and reason:
            obligation.reason = str(reason)
        obligation.source_attempt_ts = seen_ts
        obligation.source_attempt_msg = str(attempt.commit_message or "")
        obligation.fingerprint = str(
            obligation.fingerprint
            or _make_obligation_fingerprint(obligation.item, obligation.reason or reason)
        )
        if not obligation.created_ts:
            obligation.created_ts = seen_ts
        obligation.updated_ts = seen_ts

    def _allocate_commit_readiness_debt_id(self) -> str:
        candidate, next_seq = _allocate_prefixed_id(
            _commit_readiness_debts_view(self),
            "debt_id",
            self.next_commit_readiness_debt_seq,
            "crd-",
        )
        self.next_commit_readiness_debt_seq = next_seq
        return candidate

    def _hydrate_commit_readiness_debt(self, debt: CommitReadinessDebtItem) -> None:
        debt.repo_key = str(debt.repo_key or _LEGACY_CURRENT_REPO_KEY)
        if not debt.fingerprint:
            debt.fingerprint = f"{debt.category}:{_stable_digest(debt.summary, debt.repo_key)}"
        base_ts = (
            str(debt.updated_at or "")
            or str(debt.last_seen_at or "")
            or str(debt.first_seen_at or "")
            or _utc_now()
        )
        if not debt.first_seen_at:
            debt.first_seen_at = base_ts
        if not debt.last_seen_at:
            debt.last_seen_at = base_ts
        if not debt.updated_at:
            debt.updated_at = base_ts
        debt.source_obligation_ids = _dedupe_strings(list(debt.source_obligation_ids or []))
        debt.evidence = _dedupe_strings(list(debt.evidence or []))[:5]
        debt.occurrence_count = max(1, int(debt.occurrence_count or 1))
        if debt.status in _OPEN_COMMIT_READINESS_DEBT_STATUSES:
            debt.consecutive_observations = max(1, int(debt.consecutive_observations or debt.occurrence_count or 1))
        else:
            debt.consecutive_observations = max(0, int(debt.consecutive_observations or 0))

    def _build_commit_readiness_debt_observations(
        self,
        *,
        repo_key: str | None = None,
    ) -> List[Dict[str, Any]]:
        observations: Dict[str, Dict[str, Any]] = {}

        def _remember(observation: Dict[str, Any]) -> None:
            fingerprint = str(observation.get("fingerprint", "") or "").strip()
            if not fingerprint:
                return
            existing = observations.setdefault(fingerprint, observation)
            if existing is observation:
                return
            existing["source_obligation_ids"] = _dedupe_strings(
                list(existing.get("source_obligation_ids") or [])
                + list(observation.get("source_obligation_ids") or [])
            )
            existing["evidence"] = _dedupe_strings(
                list(existing.get("evidence") or [])
                + list(observation.get("evidence") or [])
            )[:5]

        blocked_attempts = [attempt for attempt in self.filter_attempts(repo_key=repo_key) if attempt.status == "blocked" or attempt.blocked]
        open_obs = {item.obligation_id: item for item in self.get_open_obligations(repo_key=repo_key)}
        obligation_counts: Dict[str, int] = {}
        for attempt in blocked_attempts:
            for obligation_id in _dedupe_strings(list(attempt.obligation_ids or [])):
                obligation_counts[obligation_id] = obligation_counts.get(obligation_id, 0) + 1
        for obligation_id, count in sorted(obligation_counts.items()):
            if count < 2:
                continue
            obligation = open_obs.get(obligation_id)
            if obligation is None:
                continue
            item_name = str(getattr(obligation, "item", "") or obligation_id)
            summary = f"{item_name} repeated across {count} blocked reviewed attempts."
            evidence = [f"{obligation_id}: blocked_attempts={count}"]
            if getattr(obligation, "reason", ""):
                evidence.insert(0, f"{item_name}: {getattr(obligation, 'reason', '')}")
            _remember({
                "category": "obligation_repeat",
                "title": "Repeated blocked obligation",
                "summary": summary,
                "severity": "warning",
                "repo_key": str(getattr(obligation, "repo_key", "") or repo_key or ""),
                "fingerprint": f"obligation_repeat:{obligation_id}",
                "source": "review_state",
                "source_obligation_ids": [obligation_id],
                "evidence": evidence,
            })

        stale_matches_repo = repo_key is None or self.last_stale_repo_key in ("", repo_key)
        if self.last_stale_from_edit_ts and stale_matches_repo:
            _remember({
                "category": "advisory_stale",
                "title": "Advisory freshness debt",
                "summary": "Fresh advisory coverage was invalidated by a worktree mutation before the next reviewed attempt.",
                "severity": "warning",
                "repo_key": str(self.last_stale_repo_key or repo_key or ""),
                "fingerprint": "advisory_stale",
                "source": "review_state",
                "source_obligation_ids": [],
                "evidence": [str(self.last_stale_reason or "worktree mutation invalidated advisory freshness")],
            })

        scoped_attempts = self.filter_attempts(repo_key=repo_key) if repo_key is not None else list(self.attempts)
        latest_attempt = scoped_attempts[-1] if scoped_attempts else None
        latest_success_ts = ""
        for attempt in reversed(scoped_attempts):
            if str(getattr(attempt, "status", "") or "") != "succeeded":
                continue
            latest_success_ts = str(getattr(attempt, "finished_ts", "") or getattr(attempt, "updated_ts", "") or getattr(attempt, "ts", "") or "")
            break

        if (
            latest_attempt
            and latest_attempt.readiness_warnings
            and str(getattr(latest_attempt, "status", "") or "") != "succeeded"
        ):
            for warning in latest_attempt.readiness_warnings:
                warning_text = str(warning or "").strip()
                if not warning_text:
                    continue
                _remember({
                    "category": "readiness_warning",
                    "title": "Readiness warning debt",
                    "summary": warning_text,
                    "severity": "warning",
                    "repo_key": str(getattr(latest_attempt, "repo_key", "") or repo_key or ""),
                    "fingerprint": f"readiness_warning:attempt:{_stable_digest(warning_text)}",
                    "source": "review_state",
                    "source_obligation_ids": list(getattr(latest_attempt, "obligation_ids", []) or []),
                    "evidence": [warning_text],
                })

        advisory_runs = self.filter_advisory_runs(repo_key=repo_key) if repo_key is not None else list(self.advisory_runs)
        latest_run = advisory_runs[-1] if advisory_runs else None
        latest_run_ts = str(getattr(latest_run, "updated_ts", "") or getattr(latest_run, "ts", "") or "") if latest_run else ""
        advisory_warnings_resolved = bool(latest_success_ts and latest_run_ts and _max_iso_ts(latest_run_ts, latest_success_ts) == latest_success_ts)
        if latest_run and latest_run.readiness_warnings and not advisory_warnings_resolved:
            for warning in latest_run.readiness_warnings:
                warning_text = str(warning or "").strip()
                if not warning_text:
                    continue
                _remember({
                    "category": "readiness_warning",
                    "title": "Readiness warning debt",
                    "summary": warning_text,
                    "severity": "warning",
                    "repo_key": str(getattr(latest_run, "repo_key", "") or repo_key or ""),
                    "fingerprint": f"readiness_warning:advisory:{_stable_digest(warning_text)}",
                    "source": "advisory_review",
                    "source_obligation_ids": [],
                    "evidence": [warning_text],
                })

        return list(observations.values())

    def _sync_commit_readiness_debts(self, *, repo_key: str | None = None) -> None:
        now = _utc_now()
        debts = _commit_readiness_debts_view(self)
        for debt in debts:
            self._hydrate_commit_readiness_debt(debt)

        observed = {
            (
                str(item.get("repo_key", "") or _LEGACY_CURRENT_REPO_KEY),
                str(item.get("fingerprint", "") or ""),
            ): item
            for item in self._build_commit_readiness_debt_observations(repo_key=repo_key)
        }
        existing = {
            (debt.repo_key, debt.fingerprint or debt.debt_id): debt
            for debt in debts
        }

        for key, item in observed.items():
            current = existing.get(key)
            if current is None:
                current = CommitReadinessDebtItem(
                    debt_id=self._allocate_commit_readiness_debt_id(),
                    category=str(item.get("category", "") or ""),
                    summary=str(item.get("summary", "") or ""),
                    severity=str(item.get("severity", "warning") or "warning"),
                    status="detected",
                    repo_key=str(item.get("repo_key", "") or _LEGACY_CURRENT_REPO_KEY),
                    fingerprint=str(item.get("fingerprint", "") or ""),
                    title=str(item.get("title", "Commit readiness debt") or "Commit readiness debt"),
                    source=str(item.get("source", "review_state") or "review_state"),
                    source_obligation_ids=[str(x) for x in (item.get("source_obligation_ids") or [])],
                    evidence=[str(x) for x in (item.get("evidence") or [])][:5],
                    first_seen_at=now,
                    last_seen_at=now,
                    updated_at=now,
                    occurrence_count=1,
                    consecutive_observations=1,
                )
                debts.append(current)
                existing[key] = current
                continue

            previous_status = str(current.status or "detected")
            if previous_status == "detected":
                current.status = "queued"
            elif previous_status == "verified":
                current.status = "reopened"
            current.category = str(item.get("category", "") or current.category)
            current.summary = str(item.get("summary", "") or current.summary)
            current.severity = str(item.get("severity", "") or current.severity or "warning")
            current.repo_key = str(item.get("repo_key", "") or current.repo_key)
            current.fingerprint = str(item.get("fingerprint", "") or current.fingerprint)
            current.title = str(item.get("title", "") or current.title)
            current.source = str(item.get("source", "") or current.source)
            current.source_obligation_ids = _dedupe_strings(list(item.get("source_obligation_ids") or []))
            current.evidence = _dedupe_strings(list(item.get("evidence") or []))[:5]
            current.last_seen_at = now
            current.updated_at = now
            current.occurrence_count = int(current.occurrence_count or 0) + 1
            current.consecutive_observations = int(current.consecutive_observations or 0) + 1
            current.verified_at = ""

        for debt in _filter_repo_scope(debts, repo_key):
            debt_key = (debt.repo_key, debt.fingerprint or debt.debt_id)
            if debt_key in observed:
                continue
            if debt.status in _OPEN_COMMIT_READINESS_DEBT_STATUSES:
                debt.status = "verified"
                debt.verified_at = now
                debt.updated_at = now
                debt.consecutive_observations = 0

        open_items = [debt for debt in debts if str(debt.status or "") in _OPEN_COMMIT_READINESS_DEBT_STATUSES]
        closed_items = [debt for debt in debts if str(debt.status or "") not in _OPEN_COMMIT_READINESS_DEBT_STATUSES]
        open_items.sort(key=lambda debt: str(debt.updated_at or debt.last_seen_at or debt.first_seen_at or ""), reverse=True)
        closed_items.sort(key=lambda debt: str(debt.updated_at or debt.last_seen_at or debt.first_seen_at or ""), reverse=True)
        remaining = max(0, _MAX_COMMIT_READINESS_DEBTS - len(open_items))
        self.commit_readiness_debts = open_items + closed_items[:remaining]

    def get_open_commit_readiness_debts(
        self,
        repo_key: str | None = None,
    ) -> List[CommitReadinessDebtItem]:
        debts = _commit_readiness_debts_view(self)
        results: List[CommitReadinessDebtItem] = []
        for debt in _filter_repo_scope(debts, repo_key):
            self._hydrate_commit_readiness_debt(debt)
            if debt.status not in _OPEN_COMMIT_READINESS_DEBT_STATUSES:
                continue
            results.append(debt)
        return results

    def _update_obligations_from_attempt(
        self, attempt: CommitAttemptRecord, *, semantic_redirects: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Accumulate critical findings as stable obligations. ``semantic_redirects``
        (fingerprint -> obligation_id, precomputed off-lock, C9.3) lets a reworded
        free-text finding that misses the exact fingerprint fold into the open
        obligation it duplicates instead of opening a new one."""
        if not attempt.critical_findings:
            return []
        redirects = semantic_redirects or {}

        self._coalesce_open_obligations()
        existing = {
            ob.obligation_id: ob
            for ob in self.get_open_obligations(repo_key=attempt.repo_key)
        }
        by_fingerprint = {
            str(ob.fingerprint or ""): ob
            for ob in self.get_open_obligations(repo_key=attempt.repo_key)
            if str(ob.fingerprint or "")
        }
        touched_ids: List[str] = []

        for f in attempt.critical_findings:
            if not isinstance(f, dict):
                continue
            if str(f.get("verdict", "")).upper() != "FAIL":
                continue
            if str(f.get("severity", "")).lower() != "critical":
                continue
            item = str(f.get("item", "unknown"))
            reason = str(f.get("reason", ""))
            severity = str(f.get("severity", "critical"))
            raw_explicit_id = str(f.get("obligation_id", "") or "").strip()
            # Reviewer-supplied ids must match an open compatible obligation;
            # otherwise a bogus id could corrupt durable debt links.
            explicit_id = ""
            if raw_explicit_id and _looks_like_public_obligation_id(raw_explicit_id):
                candidate = existing.get(raw_explicit_id)
                if candidate is not None:
                    canon_new = _normalize_obligation_item_key(item)
                    canon_old = _normalize_obligation_item_key(candidate.item)
                    items_compatible = (
                        (canon_new and canon_old and canon_new == canon_old)
                        or not canon_new
                        or not canon_old
                    )
                    if items_compatible:
                        explicit_id = raw_explicit_id
            fingerprint = _make_obligation_fingerprint(item, reason)

            # A reworded restatement that misses the exact fingerprint folds into the
            # open obligation the off-lock detector matched it to (C9.3), but only if
            # that obligation is still open here (fail-open: a vanished target opens a
            # new obligation). Honesty about the residual risk: the fold keeps the
            # SURVIVING obligation's item/reason, so a WRONG high-confidence merge of
            # two genuinely distinct critical findings drops the redirected finding's
            # text — and if the survivor is later resolved, the dropped one's blocking
            # clears for that attempt. It is NOT permanently lost: a still-broken
            # finding re-surfaces as a fresh obligation on the next review attempt (its
            # own fingerprint, the resolved survivor no longer an open candidate), so
            # the gate self-heals. The detector is biased hard to false-DUP (high
            # confidence + same-root-cause/same-action only) precisely because a
            # false-MERGE here is the costly direction; it never blocks review.
            redirected = existing.get(redirects.get(fingerprint, "")) if redirects else None
            obligation = None
            if explicit_id and explicit_id in existing:
                obligation = existing[explicit_id]
            elif fingerprint in by_fingerprint:
                obligation = by_fingerprint[fingerprint]
            elif redirected is not None:
                obligation = redirected
            else:
                obligation = ObligationItem(
                    obligation_id=self._allocate_obligation_id(),
                    item=item,
                    severity=severity,
                    reason=reason,
                    source_attempt_ts=str(attempt.ts or ""),
                    source_attempt_msg=str(attempt.commit_message or ""),
                    status="still_open",
                    repo_key=attempt.repo_key,
                    fingerprint=fingerprint,
                )
                self.open_obligations.append(obligation)

            self._touch_obligation(
                obligation,
                attempt,
                item=item,
                reason=reason,
                severity=severity,
            )
            existing[obligation.obligation_id] = obligation
            by_fingerprint[obligation.fingerprint] = obligation
            touched_ids.append(obligation.obligation_id)

        self._coalesce_open_obligations()
        return _dedupe_strings(touched_ids)

    def resolve_obligations(
        self,
        resolved_ids: List[str],
        resolved_by: str = "",
        repo_key: str | None = None,
    ) -> int:
        count = 0
        for ob in _filter_repo_scope(self.open_obligations, repo_key):
            if ob.obligation_id not in resolved_ids or ob.status != "still_open":
                continue
            ob.status = "resolved"
            ob.resolved_by = resolved_by
            count += 1
        return count

    def get_open_obligations(self, repo_key: str | None = None) -> List[ObligationItem]:
        return [
            ob for ob in _filter_repo_scope(self.open_obligations, repo_key)
            if ob.status == "still_open"
        ]

    def on_successful_commit(self, repo_key: str | None = None) -> None:
        now = _utc_now()
        if repo_key is None:
            self.open_obligations = []
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
            for debt in _commit_readiness_debts_view(self):
                self._hydrate_commit_readiness_debt(debt)
                if debt.status in _OPEN_COMMIT_READINESS_DEBT_STATUSES:
                    debt.status = "verified"
                    debt.verified_at = now
                    debt.updated_at = now
                    debt.consecutive_observations = 0
            return

        self.open_obligations = [
            ob for ob in self.open_obligations
            if ob not in _filter_repo_scope(self.open_obligations, repo_key)
        ]
        if self.last_stale_repo_key in ("", repo_key):
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
        self._sync_commit_readiness_debts(repo_key=repo_key)

    def expire_stale_attempts(
        self,
        *,
        now_ts: str | None = None,
        ttl_sec: int = _REVIEW_ATTEMPT_TTL_SEC,
        grace_sec: int = _REVIEW_ATTEMPT_GRACE_SEC,
    ) -> List[CommitAttemptRecord]:
        """Auto-expire stale reviewing/late attempts after TTL+grace."""
        now_ts = now_ts or _utc_now()
        now_epoch = _parse_iso_ts(now_ts)
        if now_epoch is None:
            return []

        expired: List[CommitAttemptRecord] = []
        for item in self.attempts:
            if item.status != "reviewing" and not item.late_result_pending:
                continue
            started_epoch = _parse_iso_ts(item.started_ts or item.ts)
            if started_epoch is None:
                continue
            age_sec = max(0.0, now_epoch - started_epoch)
            if age_sec < float(ttl_sec + grace_sec):
                continue

            item.status = "failed"
            item.phase = "expired"
            item.blocked = False
            item.block_reason = "infra_failure"
            item.block_details = (
                f"Auto-expired stale reviewed attempt after {ttl_sec + grace_sec}s TTL+grace."
            )
            item.duration_sec = max(item.duration_sec, round(age_sec, 1))
            item.finished_ts = now_ts
            item.updated_ts = now_ts
            item.late_result_pending = False
            item.readiness_warnings = _dedupe_strings(
                list(item.readiness_warnings or [])
                + ["Previous reviewed attempt auto-expired after exceeding TTL+grace."]
            )
            expired.append(item)

        return expired


def _obligation_from_dict(d: Dict[str, Any]) -> ObligationItem:
    return ObligationItem(
        **{key: str(d.get(key, default)) for key, default in _OBLIGATION_STR_DEFAULTS.items()},
        fingerprint=str(d.get("fingerprint", "") or _make_obligation_fingerprint(d.get("item", ""), d.get("reason", ""))),
        created_ts=str(d.get("created_ts", d.get("source_attempt_ts", ""))),
        updated_ts=str(d.get("updated_ts", d.get("source_attempt_ts", ""))),
    )


def _commit_readiness_debt_from_dict(d: Dict[str, Any]) -> CommitReadinessDebtItem:
    return CommitReadinessDebtItem(
        **{key: str(d.get(key, default)) for key, default in _DEBT_STR_DEFAULTS.items()},
        source_obligation_ids=[str(x) for x in (d.get("source_obligation_ids") or [])],
        evidence=[str(x) for x in (d.get("evidence") or [])],
        occurrence_count=_coerce_int(d.get("occurrence_count", 0)),
        consecutive_observations=_coerce_int(d.get("consecutive_observations", 0)),
    )


def _record_from_dict(d: Dict[str, Any]) -> AdvisoryRunRecord:
    raw_paths = d.get("snapshot_paths")
    ts = str(d.get("ts", ""))
    return AdvisoryRunRecord(
        **{key: str(d.get(key, default)) for key, default in _RUN_STR_DEFAULTS.items()},
        ts=ts,
        items=list(d.get("items") or []),
        snapshot_paths=list(raw_paths) if isinstance(raw_paths, list) else None,
        task_id=str(d.get("task_id", d.get("bypassed_by_task", ""))),
        attempt=_coerce_int(d.get("attempt", 0)),
        created_ts=str(d.get("created_ts", ts)),
        updated_ts=str(d.get("updated_ts", ts)),
        readiness_warnings=[str(x) for x in (d.get("readiness_warnings") or [])],
        prompt_chars=int(d.get("prompt_chars", 0) or 0),
        duration_sec=float(d.get("duration_sec", 0.0) or 0.0),
    )


def _commit_attempt_from_dict(d: Dict[str, Any]) -> CommitAttemptRecord:
    ts = str(d.get("ts", ""))
    status = str(d.get("status", "failed"))
    return CommitAttemptRecord(
        **{key: str(d.get(key, default)) for key, default in _ATTEMPT_STR_DEFAULTS.items()},
        ts=ts,
        status=status,
        duration_sec=float(d.get("duration_sec", 0.0)),
        critical_findings=list(d.get("critical_findings") or []),
        attempt=_coerce_int(d.get("attempt", 0)),
        phase=str(d.get("phase", infer_review_phase(status, str(d.get("block_reason", ""))))),
        blocked=bool(d.get("blocked", status == "blocked")),
        advisory_findings=_normalize_findings(d.get("advisory_findings") or []),
        obligation_ids=[str(x) for x in (d.get("obligation_ids") or [])],
        readiness_warnings=[str(x) for x in (d.get("readiness_warnings") or [])],
        late_result_pending=bool(d.get("late_result_pending", False)),
        degraded_reasons=[str(x) for x in (d.get("degraded_reasons") or [])],
        started_ts=str(d.get("started_ts", ts)),
        updated_ts=str(d.get("updated_ts", ts)),
        finished_ts=str(d.get("finished_ts", ts if status in ("blocked", "failed", "succeeded") else "")),
        triad_models=[str(x) for x in (d.get("triad_models") or [])],
        triad_raw_results=list(d.get("triad_raw_results") or []),
        scope_raw_result=dict(d.get("scope_raw_result") or {}),
    )


def _load_state_unlocked(drive_root: pathlib.Path) -> AdvisoryReviewState:
    path = drive_root / _STATE_RELPATH
    if not path.exists():
        return AdvisoryReviewState()

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        return AdvisoryReviewState()

    advisory_runs = [_record_from_dict(item) for item in (item for item in data.get("advisory_runs", []) if isinstance(item, dict))]
    attempts = [_commit_attempt_from_dict(item) for item in (item for item in data.get("attempts", []) if isinstance(item, dict))]
    open_obligations = [_obligation_from_dict(item) for item in (item for item in data.get("open_obligations", []) if isinstance(item, dict))]
    commit_readiness_debts = [
        _commit_readiness_debt_from_dict(item)
        for item in (item for item in data.get("commit_readiness_debts", []) if isinstance(item, dict))
    ]

    state = AdvisoryReviewState(
        state_version=_coerce_int(data.get("state_version", data.get("schema_version", _STATE_SCHEMA_VERSION))),
        advisory_runs=advisory_runs,
        attempts=attempts,
        open_obligations=open_obligations,
        next_obligation_seq=_coerce_int(
            data.get("next_obligation_seq", _infer_next_prefixed_sequence(open_obligations, "obl-")),
            _infer_next_prefixed_sequence(open_obligations, "obl-"),
        ),
        commit_readiness_debts=commit_readiness_debts,
        next_commit_readiness_debt_seq=_coerce_int(
            data.get("next_commit_readiness_debt_seq", _infer_next_prefixed_sequence(commit_readiness_debts, "crd-")),
            _infer_next_prefixed_sequence(commit_readiness_debts, "crd-"),
        ),
        last_stale_from_edit_ts=str(data.get("last_stale_from_edit_ts", "")),
        last_stale_reason=str(data.get("last_stale_reason", "")),
        last_stale_repo_key=str(data.get("last_stale_repo_key", "")),
    )

    state.attempts.sort(key=_attempt_order_key)

    state._coalesce_open_obligations()
    state.next_obligation_seq = max(
        1,
        int(state.next_obligation_seq or 1),
        _infer_next_prefixed_sequence(state.open_obligations, "obl-"),
    )
    state.next_commit_readiness_debt_seq = max(
        1,
        int(state.next_commit_readiness_debt_seq or 1),
        _infer_next_prefixed_sequence(state.commit_readiness_debts, "crd-"),
    )
    return state


def load_state(drive_root: pathlib.Path) -> AdvisoryReviewState:
    """Load review state, returning empty state on error."""
    try:
        return _load_state_unlocked(drive_root)
    except Exception as e:
        path = drive_root / _STATE_RELPATH
        log.warning("Failed to load advisory review state from %s: %s", path, e)
        return AdvisoryReviewState()


def compute_obligation_semantic_redirects(
    state: AdvisoryReviewState,
    findings: List[Any],
    *,
    repo_key: str,
    drive_root: Any,
) -> Dict[str, str]:
    """Off-lock C9.3 pre-pass: for each FAIL/critical FREE-TEXT (bug_*/risk_*) finding that
    would miss the exact obligation fingerprint, ask the shared detector whether it
    duplicates an OPEN obligation of the same repo. Returns ``{fingerprint -> obligation_id}``
    for HIGH-confidence matches only.

    Must run OUTSIDE the review-state lock (it calls a light model). Side-effect-free and
    fail-open: any failure (model down, no candidates, parse error) yields no redirect — the
    finding becomes a new obligation — and it NEVER blocks review. Canonical-anchor findings
    are skipped: they already dedup structurally via the obligation fingerprint."""
    try:
        if not findings:
            return {}
        open_obs = state.get_open_obligations(repo_key=repo_key)
        if not open_obs:
            return {}
        existing_fps = {str(ob.fingerprint or "") for ob in open_obs if ob.fingerprint}
        candidates = [
            {"id": ob.obligation_id, "text": f"{ob.item}: {ob.reason}".strip(": ")}
            for ob in open_obs[:20]
            if ob.obligation_id and (str(ob.item or "").strip() or str(ob.reason or "").strip())
        ]
        if not candidates:
            return {}

        from ouroboros.semantic_dedup import find_semantic_duplicate_id

        redirects: Dict[str, str] = {}
        for f in findings:
            if not isinstance(f, dict):
                continue
            if str(f.get("verdict", "")).upper() != "FAIL":
                continue
            if str(f.get("severity", "")).lower() != "critical":
                continue
            item = str(f.get("item", "unknown"))
            reason = str(f.get("reason", ""))
            # Only FREE-TEXT findings: a canonical-anchor item already dedups structurally.
            if _normalize_obligation_item_key(item):
                continue
            fingerprint = _make_obligation_fingerprint(item, reason)
            if fingerprint in existing_fps or fingerprint in redirects:
                continue
            dup_id = find_semantic_duplicate_id(
                f"{item}: {reason}".strip(": "),
                candidates,
                subject="code-review obligation (a critical finding that blocks commit until fixed)",
                call_type="obligation_dedup",
                drive_root=drive_root,
            )
            if dup_id:
                redirects[fingerprint] = dup_id
        return redirects
    except Exception:
        return {}


def _save_state_unlocked(drive_root: pathlib.Path, state: AdvisoryReviewState) -> None:
    path = drive_root / _STATE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_state_for_persistence(state)
    data: Dict[str, Any] = {
        "state_version": _STATE_SCHEMA_VERSION,
        "schema_version": _STATE_SCHEMA_VERSION,
        "advisory_runs": [asdict(r) for r in state.advisory_runs],
        "attempts": [asdict(r) for r in state.attempts],
        "open_obligations": [asdict(o) for o in state.open_obligations],
        "next_obligation_seq": int(state.next_obligation_seq or 1),
        "commit_readiness_debts": [asdict(item) for item in state.commit_readiness_debts],
        "next_commit_readiness_debt_seq": int(state.next_commit_readiness_debt_seq or 1),
        "last_stale_from_edit_ts": state.last_stale_from_edit_ts,
        "last_stale_reason": state.last_stale_reason,
        "last_stale_repo_key": state.last_stale_repo_key,
        "saved_at": _utc_now(),
    }
    atomic_write_json(path, data)


def save_state(drive_root: pathlib.Path, state: AdvisoryReviewState) -> None:
    """Persist review state atomically under the review-state lock.

    Raises ``TimeoutError`` on lock failure (matching ``update_state``): a
    silently skipped save left the advisory ledger reporting a stale "fresh"
    pre-review, which the commit gate then trusted — an immune-system hole,
    not a tolerable degradation.
    """
    lock_path = drive_root / _LOCK_RELPATH
    lock_fd = acquire_review_state_lock(drive_root)
    if lock_fd is None:
        raise TimeoutError(f"Could not acquire review state lock for {lock_path}")
    try:
        _save_state_unlocked(drive_root, state)
    finally:
        release_review_state_lock(drive_root, lock_fd)


def update_state(
    drive_root: pathlib.Path,
    mutator: Callable[[AdvisoryReviewState], Any],
) -> Any:
    """Run read-modify-write under an explicit lock."""
    lock_fd = acquire_review_state_lock(drive_root)
    if lock_fd is None:
        raise TimeoutError(f"Could not acquire review state lock for {drive_root / _LOCK_RELPATH}")
    try:
        state = _load_state_unlocked(drive_root)
        result = mutator(state)
        _save_state_unlocked(drive_root, state)
        return state if result is None else result
    finally:
        release_review_state_lock(drive_root, lock_fd)


def acquire_review_state_lock(
    drive_root: pathlib.Path,
    timeout_sec: float = 4.0,
    stale_sec: float = 90.0,
) -> Optional[int]:
    lock_path = drive_root / _LOCK_RELPATH
    return acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=timeout_sec,
        stale_sec=stale_sec,
        metadata=f"pid={os.getpid()} ts={_utc_now()}\n",
    )


def release_review_state_lock(drive_root: pathlib.Path, lock_fd: Optional[int]) -> None:
    lock_path = drive_root / _LOCK_RELPATH
    release_exclusive_file_lock(lock_path, lock_fd)


_SNAPSHOT_EXCLUDE_PATHS = frozenset({
    "state/advisory_review.json",
    "state/queue_snapshot.json",
})


def discover_repo_root(path: pathlib.Path) -> pathlib.Path:
    """Return nearest directory containing .git, else resolved input path."""
    resolved = path.resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return resolved if resolved.is_dir() else resolved.parent
        current = current.parent


def make_repo_key(repo_dir: pathlib.Path) -> str:
    return str(discover_repo_root(repo_dir))


def compute_snapshot_hash(
    repo_dir: pathlib.Path,
    commit_message: str = "",
    paths: list[str] | None = None,
) -> str:
    """Build a deterministic hash for the current worktree snapshot."""
    if isinstance(paths, list) and len(paths) == 0:
        paths = None

    changed_digests: List[tuple[str, str]] = []

    def _record_digest(relpath: str) -> None:
        relpath = relpath.strip()
        if not relpath or relpath in _SNAPSHOT_EXCLUDE_PATHS:
            return
        file_path = repo_dir / relpath
        try:
            if file_path.is_file():
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
            else:
                digest = "deleted"
        except Exception:
            digest = "unreadable"
        changed_digests.append((relpath, digest))

    if paths is not None:
        for relpath in paths:
            _record_digest(relpath)
    else:
        try:
            from ouroboros.tools.review_helpers import list_changed_paths_from_git_status

            for relpath in list_changed_paths_from_git_status(
                repo_dir,
                include_sources_for_renames=True,
            ):
                _record_digest(relpath)
        except Exception as e:
            log.debug("compute_snapshot_hash: git status failed: %s", e)

    h = hashlib.sha256()
    for relpath, digest in sorted(changed_digests):
        h.update(f"{relpath}:{digest}\n".encode())
    return h.hexdigest()[:32]


def mark_advisory_stale_after_edit(drive_root: pathlib.Path) -> None:
    """Mark fresh advisory runs stale after a worktree edit."""
    try:
        updated = update_state(drive_root, lambda state: _mark_advisory_stale_locked(state))
        if isinstance(updated, AdvisoryReviewState):
            log.debug("Advisory state marked stale after worktree edit")
    except Exception as e:
        log.debug("mark_advisory_stale_after_edit failed (non-fatal): %s", e)


def _mark_advisory_stale_locked(state: AdvisoryReviewState) -> None:
    has_invalidatable = any(r.status in ("fresh", "bypassed", "skipped") for r in state.advisory_runs)
    if not has_invalidatable:
        return
    state.mark_repo_stale(repo_key="", reason_ts=_utc_now(), reason="Worktree edit invalidated advisory freshness.", stale_repo_key="")


def invalidate_advisory_after_mutation(
    drive_root: pathlib.Path,
    *,
    mutation_root: pathlib.Path | None = None,
    changed_paths: Optional[List[str]] = None,
    source_tool: str = "",
) -> None:
    """Invalidate advisory freshness after mutation; ambiguous repo scope stales all."""
    try:
        changed_paths = [str(p).strip() for p in (changed_paths or []) if str(p).strip()]
        resolved_repo_keys = _resolve_mutation_repo_keys(mutation_root, changed_paths)
        reason_ts = _utc_now()
        reason = _build_invalidation_reason(source_tool, mutation_root, changed_paths, resolved_repo_keys)

        def _mutate(state: AdvisoryReviewState) -> None:
            if not resolved_repo_keys or len(resolved_repo_keys) != 1:
                state.mark_repo_stale(repo_key="", reason_ts=reason_ts, reason=reason, stale_repo_key="")
                return
            state.mark_repo_stale(
                repo_key=resolved_repo_keys[0],
                reason_ts=reason_ts,
                reason=reason,
                stale_repo_key=resolved_repo_keys[0],
            )

        update_state(drive_root, _mutate)
    except Exception as e:
        log.debug("invalidate_advisory_after_mutation failed (non-fatal): %s", e)


def format_status_section(state: AdvisoryReviewState, repo_dir: Optional[pathlib.Path] = None) -> str:
    """Render historical review state for LLM context."""
    repo_key = make_repo_key(repo_dir) if repo_dir is not None else None
    advisory_runs = state.filter_advisory_runs(repo_key=repo_key) if repo_key is not None else list(state.advisory_runs)
    attempts = state.filter_attempts(repo_key=repo_key) if repo_key is not None else list(state.attempts)
    last_attempt = state.latest_attempt_for(repo_key=repo_key) if repo_key is not None else state.latest_attempt()
    open_obs = state.get_open_obligations(repo_key=repo_key)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)

    if not advisory_runs and last_attempt is None and not open_obs and not open_debts:
        return "## Advisory Pre-Review Status\n\nNo advisory runs recorded yet."

    lines = [
        "## Advisory Pre-Review Status",
        "(Historical — run `review_status` for gate-accurate live freshness)",
    ]

    # Include all runs/attempts/findings; review history must not silently truncate.
    for run in advisory_runs:
        lines.append(f"\n{_RUN_STATUS_ICONS.get(run.status, '❓')} **{run.status.upper()}** | hash={run.snapshot_hash[:12]} | {run.ts}")
        lines.append(f"   Commit: {run.commit_message}")
        if run.bypass_reason:
            lines.append(f"   Bypassed: {run.bypass_reason}")
        if run.snapshot_summary:
            lines.append(f"   Scope: {run.snapshot_summary}")

        findings = [
            item for item in (run.items or [])
            if isinstance(item, dict) and str(item.get("verdict", "")).upper() == "FAIL"
        ]
        if findings:
            _append_finding_lines(lines, findings, "Findings", with_severity=True)
        elif run.status in ("fresh", "bypassed", "skipped", "parse_failure"):
            lines.append("   No findings recorded.")

    stale_matches_repo = repo_key is None or state.last_stale_repo_key in ("", repo_key)
    if state.last_stale_from_edit_ts and stale_matches_repo:
        lines.append(f"\n⚠️ Advisory marked stale after worktree edit at {state.last_stale_from_edit_ts}.")  # full ts — no [:16]
        if state.last_stale_reason:
            lines.append(f"   Reason: {state.last_stale_reason}")
        lines.append("   Run advisory_review again before commit_reviewed.")

    if open_debts:
        lines.append(f"\n### Commit-readiness debt ({len(open_debts)})")
        for debt in open_debts:
            lines.append(
                f"- [{debt.debt_id}] [{str(debt.status or '').upper()}] {debt.title}: {debt.summary}"
            )
            if debt.source_obligation_ids:
                lines.append(f"    obligations={', '.join(debt.source_obligation_ids)}")
            for evidence in list(debt.evidence or []):
                lines.append(f"    evidence={evidence}")

    if attempts:
        lines.append("\n### Recent reviewed attempts")
        for item in attempts:
            tool = item.tool_name or _DEFAULT_TOOL_NAME
            num = int(item.attempt or 0)
            label = f"{tool}#{num}" if num else tool
            phase = item.phase or "review"
            facts = [f"status={item.status}", f"phase={phase}", f"blocked={'yes' if item.blocked else 'no'}"]
            if item.commit_message:
                facts.append(f"commit={item.commit_message}")
            if item.late_result_pending:
                facts.append("late_result_pending=yes")
            if item.readiness_warnings:
                facts.append(f"warnings={len(item.readiness_warnings)}")
            if item.degraded_reasons:
                facts.append(f"degraded={len(item.degraded_reasons)}")
            lines.append(f"- {label}: {', '.join(facts)}")
            triad_raw = getattr(item, "triad_raw_results", None) or []
            if triad_raw:
                actor_summaries = (f"{r.get('model_id', '?')}={r.get('status', '?')}" for r in triad_raw)
                lines.append(f"    triad_actors: {', '.join(actor_summaries)}")
            scope_raw = getattr(item, "scope_raw_result", None) or {}
            if scope_raw and scope_raw.get("status"):
                lines.append(f"    scope_actor: {scope_raw.get('model_id', '?')}={scope_raw.get('status', '?')}")

    ca = last_attempt
    if ca and ca.status in ("blocked", "failed"):
        icon = "🚫" if ca.status == "blocked" else "❌"
        lines.append(f"\n{icon} **Last commit {ca.status.upper()}** | {ca.ts}")
        lines.append(f"   Commit: {ca.commit_message}")
        lines.append(f"   Tool: {ca.tool_name or _DEFAULT_TOOL_NAME}")
        if ca.attempt:
            lines.append(f"   Attempt: {ca.attempt}")
        if ca.block_reason:
            lines.append(f"   Reason: {ca.block_reason}")
        if ca.block_details:
            preview = _truncate_review_artifact(ca.block_details, limit=200).replace("\n", " ")
            lines.append(f"   Details: {preview}")
        if ca.duration_sec > 0:
            lines.append(f"   Duration: {ca.duration_sec:.1f}s")
        if ca.readiness_warnings:
            lines.append(f"   Readiness warnings ({len(ca.readiness_warnings)}):")
            for warning in ca.readiness_warnings:
                lines.append(f"     - {_truncate_review_reason(warning, limit=160)}")
        critical_findings = list(ca.critical_findings or [])
        advisory_findings = list(ca.advisory_findings or [])
        if critical_findings:
            _append_finding_lines(lines, critical_findings, "Critical findings", limit=160)
        elif advisory_findings:
            _append_finding_lines(lines, advisory_findings, "Advisory findings", limit=160)

    if open_obs:
        lines.append(f"\n📋 **Open obligations from previous blocking rounds ({len(open_obs)}):**")
        for ob in open_obs:
            lines.append(f"   [{ob.obligation_id}] [{ob.severity.upper()}] {ob.item}: {_truncate_review_reason(ob.reason, limit=120)}")
            lines.append(f"      Source: {ob.source_attempt_ts} — \"{ob.source_attempt_msg}\"")
        lines.append("   Advisory MUST verify each obligation is resolved before PASS.")

    return "\n".join(lines)


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
    )
    return CommitAttemptRecord(**data)


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


def _prepare_state_for_persistence(state: AdvisoryReviewState) -> None:
    """Normalize ledgers and counters before persistence."""
    state._coalesce_open_obligations()
    debts = _commit_readiness_debts_view(state)
    for debt in debts:
        state._hydrate_commit_readiness_debt(debt)
    state.next_obligation_seq = max(
        1,
        int(state.next_obligation_seq or 1),
        _infer_next_prefixed_sequence(state.open_obligations, "obl-"),
    )
    state.next_commit_readiness_debt_seq = max(
        1,
        int(state.next_commit_readiness_debt_seq or 1),
        _infer_next_prefixed_sequence(debts, "crd-"),
    )

def _resolve_mutation_repo_keys(
    mutation_root: pathlib.Path | None,
    changed_paths: List[str],
) -> List[str]:
    base = mutation_root.resolve() if mutation_root is not None else None
    repo_keys: List[str] = []

    def _record(candidate: pathlib.Path) -> None:
        key = make_repo_key(candidate)
        if key and key not in repo_keys:
            repo_keys.append(key)

    if base is not None:
        _record(base)
    for rel_path in changed_paths:
        candidate = pathlib.Path(rel_path)
        if not candidate.is_absolute() and base is not None:
            candidate = (base / rel_path).resolve()
        elif not candidate.is_absolute():
            continue
        _record(candidate if candidate.exists() else candidate.parent)
    return repo_keys


def _build_invalidation_reason(
    source_tool: str,
    mutation_root: pathlib.Path | None,
    changed_paths: List[str],
    repo_keys: List[str],
) -> str:
    tool = source_tool or "mutation"
    repo_hint = ""
    if len(repo_keys) == 1:
        repo_hint = f" repo={repo_keys[0]}"
    elif len(repo_keys) > 1:
        repo_hint = " repo=multiple"
    path_hint = ""
    if changed_paths:
        preview = ", ".join(changed_paths[:3])
        if len(changed_paths) > 3:
            preview += f", +{len(changed_paths) - 3} more"
        path_hint = f" paths={preview}"
    elif mutation_root is not None:
        path_hint = f" root={mutation_root}"
    return f"{tool} mutated the worktree; advisory freshness invalidated.{repo_hint}{path_hint}"


def _utc_now() -> str:
    from ouroboros.utils import utc_now_iso
    return utc_now_iso()
