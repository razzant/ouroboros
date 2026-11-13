"""The in-memory review ledger and every transition it permits.

``AdvisoryReviewState`` is the whole mutable state of one drive's review history:
commit attempts and the stale markers, their lifecycle transitions, freshness and
expiry rules, and the projections the review surfaces read; the retired advisory
lane's rows and what the retired obligation layer stored ride along only as a
verbatim archive, never read as freshness, guidance or open work. It owns no
persistence — loading, saving, locking and repo identity stay with ``review_state``. Extracted from
ouroboros/review_state.py (v7 D06 split, re-cut on the v7next tip);
review_state.py re-exports the class at its historical binding. The class-level
defaults and the f-string reads the call-time handle cannot carry stay
import-bound to their ``review_state_records`` owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ouroboros.review_state_records import (
    _REVIEW_ATTEMPT_GRACE_SEC,
    _REVIEW_ATTEMPT_TTL_SEC,
    _STATE_SCHEMA_VERSION,
)

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.review_state_records import CommitAttemptRecord


def _rs():
    """The parent review-state module, read at call time.

    The review-state members stay monkeypatch-addressable at their historical
    ``ouroboros.review_state`` bindings (tests rebind them there), so this leaf
    resolves every such cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_state

    return review_state


def _row_stamp(row: Dict[str, Any]) -> str:
    """When a stored obligation row was last written: a landing after it is what cleared it."""
    return str(row.get("updated_ts") or row.get("source_attempt_ts") or "")


@dataclass
class AdvisoryReviewState:
    """Top-level durable review state."""

    state_version: int = _STATE_SCHEMA_VERSION
    # The retired advisory lane's rows exactly as stored (owner decision: archive only).
    # Nothing writes, normalizes or interprets them; saving writes them back unchanged.
    advisory_runs: List[Any] = field(default_factory=list)
    attempts: List[CommitAttemptRecord] = field(default_factory=list)
    last_stale_from_edit_ts: str = ""
    last_stale_reason: str = ""
    last_stale_repo_key: str = ""
    # The task whose mutation or review wrote the marker; "" = not recorded.
    last_stale_task_id: str = ""
    # last_stale_* is the newest marker; the markers of other scopes it displaced stay
    # here, by checkout key ("" = unscoped): {"ts", "reason", "task_id"} (#2573).
    earlier_stale_markers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # What the retired obligation layer stored (``open_obligations``, ``commit_readiness_debts`` with the advisory
    # lane's debts among them, and their ``next_*_seq`` counters), under their original keys and exactly as stored:
    # history only, never open work; written back unchanged, so an earlier body reading the file after a rollback
    # finds them where it left them.
    retired_review_rows: Dict[str, Any] = field(default_factory=dict)

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

    def next_attempt_number(self, repo_key: str, tool_name: str, task_id: str = "", *, above: int = 0) -> int:
        """The next number of this task's attempts, never one at or below ``above`` (a number the caller still holds)."""
        candidates = self.filter_attempts(repo_key=repo_key, tool_name=tool_name, task_id=task_id)
        latest = max((int(item.attempt or 0) for item in candidates), default=0)
        return max(latest, int(above or 0)) + 1

    def landed_after(self, repo_key: str, stamp: str) -> bool:
        """Whether a commit landed in ``repo_key``'s checkout (any checkout when empty) after ``stamp``: a
        ``succeeded`` attempt there that finished later. An unparseable stamp proves no landing, and a landing row
        the attempt window evicted reads as none, so what waits for a landing stays longer, never shorter."""
        return self._landed_among(self.filter_attempts(repo_key=repo_key or None), stamp)

    @staticmethod
    def _landed_among(attempts: List[CommitAttemptRecord], stamp: str) -> bool:
        since = _rs()._parse_iso_ts(str(stamp or ""))
        if since is None:
            return False
        for item in attempts:
            landed = _rs()._parse_iso_ts(str(item.finished_ts or item.updated_ts or item.ts or ""))
            if item.status == "succeeded" and landed is not None and landed > since:
                return True
        return False

    def earlier_open_obligations(self, repo_key: Optional[str]) -> List[Dict[str, Any]]:
        """The open obligations the retired layer stored for this checkout (its exact key, else the legacy and empty
        keys), as stored, until a commit lands there after them: the moment that layer would have cleared them.
        ``None`` reads every checkout's, each row in its own; a row stored without a key stays until a commit lands
        in a checkout that holds no keyed row (that layer's scope rule)."""
        stored = (getattr(self, "retired_review_rows", None) or {}).get("open_obligations")
        rows = [row for row in (stored if isinstance(stored, list) else [])
                if isinstance(row, dict) and str(row.get("status", "still_open")) == "still_open"]
        unkeyed = {"", _rs()._LEGACY_CURRENT_REPO_KEY}
        if repo_key is None:
            keyed = {str(row.get("repo_key") or "") for row in rows} - unkeyed
            outside = [item for item in self.filter_attempts(repo_key=None) if str(item.repo_key or "") not in keyed]
            return [row for row in rows if not (
                self.landed_after(str(row.get("repo_key") or ""), _row_stamp(row))
                if str(row.get("repo_key") or "") in keyed else self._landed_among(outside, _row_stamp(row)))]
        scope = {repo_key} if any(str(row.get("repo_key") or "") == repo_key for row in rows) else unkeyed
        return [row for row in rows if str(row.get("repo_key") or "") in scope
                and not self.landed_after(repo_key, _row_stamp(row))]

    def _newest_stale_marker(self) -> Dict[str, str]:
        return {"ts": self.last_stale_from_edit_ts, "reason": self.last_stale_reason,
                "task_id": self.last_stale_task_id, "repo_key": self.last_stale_repo_key}

    def _stale_markers(self) -> List[Dict[str, str]]:
        """Every marker held, with its checkout key: the newest, then those it displaced
        (a ``__new__``-built state has no factory default)."""
        earlier = getattr(self, "earlier_stale_markers", None) or {}
        rows = (self._newest_stale_marker(), *({**row, "repo_key": key} for key, row in earlier.items()))
        return [row for row in rows if row.get("ts")]

    def stale_marker(self, repo_key: str = "", look_ts: str = "") -> Dict[str, str]:
        """The marker that invalidated ``repo_key``'s look recorded at ``look_ts`` (any
        look when empty): the first written at or after it, scoped to that checkout or
        unscoped (an unscoped marker applies to every checkout); ``{}`` when none was."""
        rows = [row for row in self._stale_markers() if row["repo_key"] in ("", repo_key) and row["ts"] >= look_ts]
        return min(rows, key=lambda row: row["ts"], default={})

    def marker_postdates(self, look_ts: str, repo_key: str = "") -> bool:
        """Whether a stale marker already covers a look recorded at ``look_ts`` on ``repo_key``."""
        return bool(look_ts and self.stale_marker(repo_key, look_ts))

    def mark_look_stale(
        self, look_ts: str, *, reason_ts: str = "", reason: str = "", stale_repo_key: str = "", stale_task_id: str = "",
    ) -> bool:
        """Write the stale marker for a preflight look — a ``surface=preflight``
        review-ledger record (D5-002). The FIRST mutation after the look writes the
        marker and later ones keep it, so the marker names the mutation that
        invalidated the look, never the latest editor. The newest marker of another
        scope is displaced, not erased. Returns whether this call wrote it."""
        if not look_ts or self.marker_postdates(look_ts, stale_repo_key):
            return False
        earlier = dict(getattr(self, "earlier_stale_markers", None) or {})
        if self.last_stale_from_edit_ts and self.last_stale_repo_key != stale_repo_key:
            earlier[self.last_stale_repo_key] = {"ts": self.last_stale_from_edit_ts, "reason": self.last_stale_reason,
                                                 "task_id": self.last_stale_task_id}
        earlier.pop(stale_repo_key, None)  # superseded: it does not cover this look
        self.earlier_stale_markers = earlier
        self.last_stale_from_edit_ts = reason_ts or _rs()._utc_now()
        self.last_stale_reason = reason
        self.last_stale_repo_key = stale_repo_key
        self.last_stale_task_id = stale_task_id
        return True

    def stale_marker_provenance(self, task_id: str = "", *, marker: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Whose mutation or review wrote ``marker`` (a ``stale_marker`` row; by default the
        newest marker), and for which checkout.

        Attribution, never scoping (#783, owner 6A): staleness is a fact about
        the checkout, so every task on it keeps seeing the marker; this only
        names the writer relative to the reading ``task_id``. An unrecorded
        writer (legacy marker, or a writer without task identity) or a reader
        without identity stays ``unknown`` — never re-attributed to either side.
        An empty ``stale_repo_key`` is an ambiguous or unrecorded checkout scope,
        which is why that marker applies to every checkout.
        """
        marker = self._newest_stale_marker() if marker is None else marker
        if not marker.get("ts"):
            return {"stale_task_id": "", "stale_attribution": "", "stale_repo_key": ""}
        writer, reader = str(marker.get("task_id") or ""), str(task_id or "")
        relation = ("this_task" if writer == reader else "other_task") if writer and reader else "unknown"
        return {"stale_task_id": writer, "stale_attribution": relation,
                "stale_repo_key": str(marker.get("repo_key") or "")}

    def stale_marker_attribution_note(self, task_id: str = "", *, marker: Optional[Dict[str, str]] = None) -> str:
        """Prose form of ``stale_marker_provenance`` for text surfaces ("invalidated by …")."""
        provenance = self.stale_marker_provenance(task_id, marker=marker)
        writer = provenance["stale_task_id"]
        note = {
            "this_task": f"this task ({writer})",
            "other_task": f"another task ({writer}); the shared checkout stays stale for every task on it",
        }.get(provenance["stale_attribution"]) or (
            f"task {writer}" if writer else "an unrecorded writer (legacy or unattributed marker)")
        if not provenance["stale_repo_key"]:
            note += "; checkout scope ambiguous or unrecorded, so it applies to every checkout"
        return note

    def record_attempt(self, attempt: CommitAttemptRecord) -> CommitAttemptRecord:
        """Upsert one reviewed attempt into durable state; its findings stay as given. A landing clears the stale
        markers of its checkout (``on_successful_commit``)."""
        now = _rs()._utc_now()
        attempt.tool_name = str(attempt.tool_name or _rs()._DEFAULT_TOOL_NAME)
        attempt.repo_key = str(attempt.repo_key or _rs()._LEGACY_CURRENT_REPO_KEY)
        attempt.blocked = bool(attempt.blocked or attempt.status == "blocked")
        if not attempt.started_ts:
            attempt.started_ts = attempt.ts or now
        if not attempt.ts:
            attempt.ts = attempt.started_ts
        attempt.updated_ts = now
        if attempt.status in ("blocked", "failed", "succeeded", "reviewed") and not attempt.finished_ts:
            attempt.finished_ts = now

        merged = self._upsert_attempt(attempt)
        if merged.status == "succeeded":
            self.on_successful_commit(repo_key=merged.repo_key)
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
        per-root-task ceiling from — evicting them would let a capped root task
        loop until the ceiling forgets itself — and verdict-block rows are kept
        with them. Both preserved classes are bounded
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

    def on_successful_commit(self, repo_key: str | None = None) -> None:
        """A landing clears the stale markers of its checkout and the unscoped one; ``repo_key=None`` clears every
        marker."""
        if repo_key is None:
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
            self.last_stale_task_id = ""
            self.earlier_stale_markers = {}
            return

        if self.last_stale_repo_key in ("", repo_key):
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
            self.last_stale_task_id = ""
        earlier = getattr(self, "earlier_stale_markers", None) or {}
        self.earlier_stale_markers = {key: row for key, row in earlier.items() if key not in ("", repo_key)}

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
