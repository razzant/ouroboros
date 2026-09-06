"""The in-memory review ledger and every transition it permits.

``AdvisoryReviewState`` is the whole mutable state of one drive's review history:
advisory runs, commit attempts, open obligations and commit-readiness debts,
their lifecycle transitions, freshness and expiry rules, and the projections the
review surfaces read. It owns no persistence — loading, saving, locking and
repo identity stay with ``review_state``. Extracted from
ouroboros/review_state.py (v7 D06 split, re-cut on the v7next tip);
review_state.py re-exports the class at its historical binding. The class-level
defaults and the f-string reads the call-time handle cannot carry stay
import-bound to their ``review_state_records`` owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ouroboros.review_state_records import (
    _DEFAULT_ADVISORY_TOOL_NAME,
    _REVIEW_ATTEMPT_GRACE_SEC,
    _REVIEW_ATTEMPT_TTL_SEC,
    _STATE_SCHEMA_VERSION,
    _stable_digest,
)

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.review_state_records import (
        AdvisoryRunRecord,
        CommitAttemptRecord,
        CommitReadinessDebtItem,
        ObligationItem,
    )


def _rs():
    """The parent review-state module, read at call time.

    The review-state members stay monkeypatch-addressable at their historical
    ``ouroboros.review_state`` bindings (tests rebind them there), so this leaf
    resolves every such cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_state

    return review_state


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
            if _rs()._attempt_has_active_review_custody(item)
        ]
        return _rs()._filter_repo_scope(active, repo_key)

    def filter_advisory_runs(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> List[AdvisoryRunRecord]:
        return _rs()._filter_lifecycle_records(
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
        return _rs()._filter_lifecycle_records(
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
        for run in reversed(_rs()._filter_repo_scope(self.advisory_runs, repo_key)):
            if run.snapshot_hash != snapshot_hash:
                continue
            return run
        return None

    def is_fresh(self, snapshot_hash: str, repo_key: str | None = None) -> bool:
        run = self.find_by_hash(snapshot_hash, repo_key=repo_key)
        return run is not None and run.status in ("fresh", "bypassed", "skipped")

    def add_run(self, run: AdvisoryRunRecord) -> None:
        invocation = str(run.execution.get("invocation_id") or run.execution.get("operation_id") or "")
        for index, existing in enumerate(self.advisory_runs):
            if (run.status == "bypassed" and run.bypass_reason
                    and existing.blocks_preflight and existing.repo_key == run.repo_key):
                # Both existing bypass writers converge here under the state lock.
                # Preserve the old task, exact request token, result and custody.
                existing.bypass_reason = run.bypass_reason
                existing.bypassed_by_task = run.bypassed_by_task
                existing.updated_ts = _rs()._utc_now()
            if (existing.blocks_preflight and existing.repo_key == run.repo_key
                    and (not invocation or invocation != (existing.execution.get("invocation_id") or existing.execution.get("operation_id")))):
                raise ValueError("an unresolved preflight already owns this repository")
            if (invocation and (existing.execution.get("invocation_id") or existing.execution.get("operation_id")) == invocation
                    and (existing.repo_key, existing.task_id) == (run.repo_key, run.task_id)):
                run.attempt, run.created_ts = existing.attempt, existing.created_ts
                if existing.bypass_reason:
                    # A late result updates its own historical row in place; it
                    # cannot revoke the newer bypass or reclaim its admission.
                    run.bypass_reason, run.bypassed_by_task = existing.bypass_reason, existing.bypassed_by_task
                    self.advisory_runs[index] = run
                    return
                self.advisory_runs.pop(index)
                break
        if not run.attempt:
            run.attempt = self.next_advisory_attempt_number(
                str(run.repo_key or _rs()._LEGACY_CURRENT_REPO_KEY),
                str(run.task_id or ""),
                str(run.tool_name or _DEFAULT_ADVISORY_TOOL_NAME),
            )
        if not run.created_ts:
            run.created_ts = run.ts or _rs()._utc_now()
        if not run.updated_ts:
            run.updated_ts = run.created_ts
        self.mark_all_stale_except(run.snapshot_hash, repo_key=run.repo_key)
        self.advisory_runs.append(run)
        if len(self.advisory_runs) > _rs()._MAX_RUN_HISTORY:
            cutoff = len(self.advisory_runs) - _rs()._MAX_RUN_HISTORY
            self.advisory_runs = [
                row for index, row in enumerate(self.advisory_runs)
                if index >= cutoff or row.execution_pending
            ]
        if run.status in ("fresh", "bypassed", "skipped", "parse_failure"):
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
        self._sync_commit_readiness_debts(repo_key=run.repo_key or None)

    def mark_stale(self, snapshot_hash: str) -> None:
        for run in self.advisory_runs:
            if run.snapshot_hash == snapshot_hash:
                run.status = "stale"
                run.updated_ts = _rs()._utc_now()

    def mark_all_stale_except(self, snapshot_hash: str, repo_key: str = "") -> None:
        for run in self.advisory_runs:
            same_repo = not repo_key or run.repo_key == repo_key
            if same_repo and run.snapshot_hash != snapshot_hash and run.status in ("fresh", "bypassed", "skipped"):
                run.status = "stale"
                run.updated_ts = _rs()._utc_now()

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
            legacy_present = any(run.repo_key in ("", _rs()._LEGACY_CURRENT_REPO_KEY) for run in invalidatable)
            target_runs = invalidatable if legacy_present and not exact_matches else (exact_matches or invalidatable)

        for run in target_runs:
            run.status = "stale"
            run.updated_ts = reason_ts or _rs()._utc_now()
        if target_runs:
            self.last_stale_from_edit_ts = reason_ts or _rs()._utc_now()
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
        now = _rs()._utc_now()
        attempt.tool_name = str(attempt.tool_name or _rs()._DEFAULT_TOOL_NAME)
        attempt.repo_key = str(attempt.repo_key or _rs()._LEGACY_CURRENT_REPO_KEY)
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
        key = _rs()._attempt_identity_tuple(attempt)
        for idx, existing in enumerate(self.attempts):
            if _rs()._attempt_identity_tuple(existing) == key:
                merged = _rs()._merge_attempt(existing, attempt)
                self.attempts[idx] = merged
                return merged
        self.attempts.append(attempt)
        self._trim_attempt_history()
        return attempt

    def _trim_attempt_history(self) -> None:
        """Authority-preserving eviction (Q16 fix round, F1): trimming to the
        historical cap may only evict NON-authoritative rows, oldest first.
        Paid rows are the money ledger ``count_paid_review_cycles`` derives the
        per-root-task ceiling from, and verdict-block rows anchor the
        identical-diff refusal streak (the quoted verdict) — evicting either
        would let a capped root task loop free refusals until the ceiling and
        the refusal both forget themselves. Both preserved classes are bounded
        by the money ceiling itself, so only the noise portion (free refusals,
        preflight facts, unpaid infra rows) is capped at
        ``_MAX_ATTEMPT_HISTORY``; the list may exceed the cap only by the
        authority-bearing rows. Preserved rows that fall OUTSIDE the newest-50
        window are COMPACTED (``_strip_attempt_heavy_payload``): accounting
        facts are immortal, heavy forensic payloads are not — keeping the
        serialized ledger ~O(preserved rows x small record) instead of growing
        by full reviewer raw output per reviewed commit forever."""
        overflow = len(self.attempts) - _rs()._MAX_ATTEMPT_HISTORY
        if overflow <= 0:
            return
        kept: List[CommitAttemptRecord] = []
        for item in self.attempts:
            if overflow > 0 and _rs()._attempt_history_evictable(item):
                overflow -= 1
                continue
            kept.append(item)
        self.attempts = kept
        for item in self.attempts[:-_rs()._MAX_ATTEMPT_HISTORY]:
            if _rs()._attempt_has_active_review_custody(item):
                continue
            _rs()._strip_attempt_heavy_payload(item)

    def _allocate_obligation_id(self) -> str:
        candidate, next_seq = _rs()._allocate_prefixed_id(
            self.open_obligations,
            "obligation_id",
            self.next_obligation_seq,
            "obl-",
        )
        self.next_obligation_seq = next_seq
        return candidate

    def _hydrate_obligation(self, obligation: ObligationItem) -> None:
        obligation.repo_key = str(obligation.repo_key or _rs()._LEGACY_CURRENT_REPO_KEY)
        obligation.fingerprint = str(
            obligation.fingerprint
            or _rs()._make_obligation_fingerprint(obligation.item, obligation.reason)
        )
        base_ts = (
            str(obligation.updated_ts or "")
            or str(obligation.created_ts or "")
            or str(obligation.source_attempt_ts or "")
            or _rs()._utc_now()
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
                not _rs()._looks_like_public_obligation_id(existing.obligation_id)
                and _rs()._looks_like_public_obligation_id(obligation.obligation_id)
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
            existing.created_ts = _rs()._min_iso_ts(existing.created_ts, obligation.created_ts)
            existing.updated_ts = _rs()._max_iso_ts(existing.updated_ts, obligation.updated_ts)
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
        seen_ts = str(attempt.ts or _rs()._utc_now())
        obligation.item = str(obligation.item or item or "")
        obligation.severity = str(obligation.severity or severity or "critical")
        obligation.repo_key = str(obligation.repo_key or attempt.repo_key or _rs()._LEGACY_CURRENT_REPO_KEY)
        if not obligation.reason and reason:
            obligation.reason = str(reason)
        obligation.source_attempt_ts = seen_ts
        obligation.source_attempt_msg = str(attempt.commit_message or "")
        obligation.fingerprint = str(
            obligation.fingerprint
            or _rs()._make_obligation_fingerprint(obligation.item, obligation.reason or reason)
        )
        if not obligation.created_ts:
            obligation.created_ts = seen_ts
        obligation.updated_ts = seen_ts

    def _allocate_commit_readiness_debt_id(self) -> str:
        candidate, next_seq = _rs()._allocate_prefixed_id(
            _rs()._commit_readiness_debts_view(self),
            "debt_id",
            self.next_commit_readiness_debt_seq,
            "crd-",
        )
        self.next_commit_readiness_debt_seq = next_seq
        return candidate

    def _hydrate_commit_readiness_debt(self, debt: CommitReadinessDebtItem) -> None:
        debt.repo_key = str(debt.repo_key or _rs()._LEGACY_CURRENT_REPO_KEY)
        if not debt.fingerprint:
            debt.fingerprint = f"{debt.category}:{_stable_digest(debt.summary, debt.repo_key)}"
        base_ts = (
            str(debt.updated_at or "")
            or str(debt.last_seen_at or "")
            or str(debt.first_seen_at or "")
            or _rs()._utc_now()
        )
        if not debt.first_seen_at:
            debt.first_seen_at = base_ts
        if not debt.last_seen_at:
            debt.last_seen_at = base_ts
        if not debt.updated_at:
            debt.updated_at = base_ts
        debt.source_obligation_ids = _rs()._dedupe_strings(list(debt.source_obligation_ids or []))
        debt.evidence = _rs()._dedupe_strings(list(debt.evidence or []))[:5]
        debt.occurrence_count = max(1, int(debt.occurrence_count or 1))
        if debt.status in _rs()._OPEN_COMMIT_READINESS_DEBT_STATUSES:
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
            existing["source_obligation_ids"] = _rs()._dedupe_strings(
                list(existing.get("source_obligation_ids") or [])
                + list(observation.get("source_obligation_ids") or [])
            )
            existing["evidence"] = _rs()._dedupe_strings(
                list(existing.get("evidence") or [])
                + list(observation.get("evidence") or [])
            )[:5]

        blocked_attempts = [attempt for attempt in self.filter_attempts(repo_key=repo_key) if attempt.status == "blocked" or attempt.blocked]
        open_obs = {item.obligation_id: item for item in self.get_open_obligations(repo_key=repo_key)}
        obligation_counts: Dict[str, int] = {}
        for attempt in blocked_attempts:
            for obligation_id in _rs()._dedupe_strings(list(attempt.obligation_ids or [])):
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
        advisory_warnings_resolved = bool(latest_success_ts and latest_run_ts and _rs()._max_iso_ts(latest_run_ts, latest_success_ts) == latest_success_ts)
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
        now = _rs()._utc_now()
        debts = _rs()._commit_readiness_debts_view(self)
        for debt in debts:
            self._hydrate_commit_readiness_debt(debt)

        observed = {
            (
                str(item.get("repo_key", "") or _rs()._LEGACY_CURRENT_REPO_KEY),
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
                current = _rs().CommitReadinessDebtItem(
                    debt_id=self._allocate_commit_readiness_debt_id(),
                    category=str(item.get("category", "") or ""),
                    summary=str(item.get("summary", "") or ""),
                    severity=str(item.get("severity", "warning") or "warning"),
                    status="detected",
                    repo_key=str(item.get("repo_key", "") or _rs()._LEGACY_CURRENT_REPO_KEY),
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
            current.source_obligation_ids = _rs()._dedupe_strings(list(item.get("source_obligation_ids") or []))
            current.evidence = _rs()._dedupe_strings(list(item.get("evidence") or []))[:5]
            current.last_seen_at = now
            current.updated_at = now
            current.occurrence_count = int(current.occurrence_count or 0) + 1
            current.consecutive_observations = int(current.consecutive_observations or 0) + 1
            current.verified_at = ""

        for debt in _rs()._filter_repo_scope(debts, repo_key):
            debt_key = (debt.repo_key, debt.fingerprint or debt.debt_id)
            if debt_key in observed:
                continue
            if debt.status in _rs()._OPEN_COMMIT_READINESS_DEBT_STATUSES:
                debt.status = "verified"
                debt.verified_at = now
                debt.updated_at = now
                debt.consecutive_observations = 0

        open_items = [debt for debt in debts if str(debt.status or "") in _rs()._OPEN_COMMIT_READINESS_DEBT_STATUSES]
        closed_items = [debt for debt in debts if str(debt.status or "") not in _rs()._OPEN_COMMIT_READINESS_DEBT_STATUSES]
        open_items.sort(key=lambda debt: str(debt.updated_at or debt.last_seen_at or debt.first_seen_at or ""), reverse=True)
        closed_items.sort(key=lambda debt: str(debt.updated_at or debt.last_seen_at or debt.first_seen_at or ""), reverse=True)
        remaining = max(0, _rs()._MAX_COMMIT_READINESS_DEBTS - len(open_items))
        self.commit_readiness_debts = open_items + closed_items[:remaining]

    def get_open_commit_readiness_debts(
        self,
        repo_key: str | None = None,
    ) -> List[CommitReadinessDebtItem]:
        debts = _rs()._commit_readiness_debts_view(self)
        results: List[CommitReadinessDebtItem] = []
        for debt in _rs()._filter_repo_scope(debts, repo_key):
            self._hydrate_commit_readiness_debt(debt)
            if debt.status not in _rs()._OPEN_COMMIT_READINESS_DEBT_STATUSES:
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
            if raw_explicit_id and _rs()._looks_like_public_obligation_id(raw_explicit_id):
                candidate = existing.get(raw_explicit_id)
                if candidate is not None:
                    canon_new = _rs()._normalize_obligation_item_key(item)
                    canon_old = _rs()._normalize_obligation_item_key(candidate.item)
                    items_compatible = (
                        (canon_new and canon_old and canon_new == canon_old)
                        or not canon_new
                        or not canon_old
                    )
                    if items_compatible:
                        explicit_id = raw_explicit_id
            fingerprint = _rs()._make_obligation_fingerprint(item, reason)

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
                obligation = _rs().ObligationItem(
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
        return _rs()._dedupe_strings(touched_ids)

    def resolve_obligations(
        self,
        resolved_ids: List[str],
        resolved_by: str = "",
        repo_key: str | None = None,
    ) -> int:
        count = 0
        for ob in _rs()._filter_repo_scope(self.open_obligations, repo_key):
            if ob.obligation_id not in resolved_ids or ob.status != "still_open":
                continue
            ob.status = "resolved"
            ob.resolved_by = resolved_by
            count += 1
        return count

    def get_open_obligations(self, repo_key: str | None = None) -> List[ObligationItem]:
        return [
            ob for ob in _rs()._filter_repo_scope(self.open_obligations, repo_key)
            if ob.status == "still_open"
        ]

    def on_successful_commit(self, repo_key: str | None = None) -> None:
        now = _rs()._utc_now()
        if repo_key is None:
            self.open_obligations = []
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
            for debt in _rs()._commit_readiness_debts_view(self):
                self._hydrate_commit_readiness_debt(debt)
                if debt.status in _rs()._OPEN_COMMIT_READINESS_DEBT_STATUSES:
                    debt.status = "verified"
                    debt.verified_at = now
                    debt.updated_at = now
                    debt.consecutive_observations = 0
            return

        self.open_obligations = [
            ob for ob in self.open_obligations
            if ob not in _rs()._filter_repo_scope(self.open_obligations, repo_key)
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
        now_ts = now_ts or _rs()._utc_now()
        now_epoch = _rs()._parse_iso_ts(now_ts)
        if now_epoch is None:
            return []

        expired: List[CommitAttemptRecord] = []
        for item in self.attempts:
            if item.status != "reviewing" and not item.late_result_pending:
                continue
            # TTL cleans up unpaid legacy UI rows. It must never become
            # permission to buy over paid work whose outcome remains unknown.
            if item.late_result_pending or (item.status == "reviewing" and item.paid):
                continue
            started_epoch = _rs()._parse_iso_ts(item.started_ts or item.ts)
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
            item.readiness_warnings = _rs()._dedupe_strings(
                list(item.readiness_warnings or [])
                + ["Previous reviewed attempt auto-expired after exceeding TTL+grace."]
            )
            expired.append(item)

        return expired

    def reconcile_process_local_review_custody_after_owner_loss(
        self,
        *,
        now_ts: str | None = None,
        recoverable_invocations: Optional[Dict[str, str]] = None,
        confirmed_dead_owner_pids: Optional[set[int]] = None,
    ) -> List[CommitAttemptRecord]:
        """Settle process-local rows whose exact owning process is dead.

        Callers supply only pids whose death they proved. A delegated row
        carrying a durable ``pending_invocation_id`` is different: its
        Claudexor run may still be live and remains recoverable.

        This is intentionally not TTL policy. A worker boot, task return, or
        elapsed duration proves nothing about a sibling process. If no durable
        delegated token remains after confirmed owner death, the attempt fails
        without a review verdict and an explicit later attempt may start.
        """
        stamp = now_ts or _rs()._utc_now()
        recoverable = dict(recoverable_invocations or {})
        dead_pids = {int(pid) for pid in (confirmed_dead_owner_pids or set()) if int(pid) > 0}
        if not dead_pids:
            return []
        changed: List[CommitAttemptRecord] = []
        for item in self.attempts:
            if not _rs()._attempt_has_active_review_custody(item):
                continue
            if int(getattr(item, "review_owner_pid", 0) or 0) not in dead_pids:
                continue
            item_changed = False
            roster_rows = _rs()._attempt_review_roster_rows(item)
            for row in roster_rows:
                if not _rs()._review_roster_row_is_pending(row):
                    continue
                if str(row.get("pending_invocation_id") or "").strip():
                    continue
                operation_id = str(row.get("operation_id") or "").strip()
                recovered_token = str(recoverable.get(operation_id) or "")
                if recovered_token:
                    # A START_REQUESTED/STARTED row may land immediately before
                    # the exact-slot checkpoint. Its operation id joins the two
                    # existing ledgers without guessing by task or slot.
                    row["pending_invocation_id"] = recovered_token
                    row["late_result_pending"] = True
                    item_changed = True
                    continue
                row["operation_state"] = "settled"
                row["late_result_pending"] = False
                row["status"] = "error"
                row["failure_code"] = "process_local_review_worker_lost"
                row["error"] = "process-local review owner process was confirmed dead"
                item_changed = True
            still_pending = any(
                _rs()._review_roster_row_is_pending(row)
                for row in roster_rows
            )
            # Legacy paid stamps predate durable roster reservation. A process
            # can therefore restart with an active top-level row but no slot
            # rows at all, or with only already-terminal rows whose aggregate
            # was never written. The orchestration process is gone; preserve
            # the evidence and close the attempt as infra failure rather than
            # blocking every future commit forever.
            if not item_changed and still_pending:
                continue
            if not item_changed and not (
                item.status == "reviewing" or item.late_result_pending
            ):
                continue
            item.status = "reviewing" if still_pending else "failed"
            item.phase = "late_wait" if still_pending else "infra"
            item.block_reason = (
                "review_late_result_pending" if still_pending else "infra_failure"
            )
            item.block_details = (
                "The process-local review owner was confirmed dead; "
                "delegated rows remain available for exact reconciliation."
                if still_pending
                else "Confirmed owner death proved every tokenless review worker was lost; "
                "the paid attempt failed without a review verdict."
            )
            item.late_result_pending = still_pending
            if not still_pending:
                item.finished_ts = stamp
            item.updated_ts = stamp
            changed.append(item)
        return changed
