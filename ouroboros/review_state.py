"""Durable advisory/review ledger persisted in state/advisory_review.json."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
import logging
import os
import pathlib
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


def _malformed_roster_row(surface: str) -> Dict[str, Any]:
    from ouroboros.review_dispatch import slot_id_for_row

    prefix = "custody_lost_scope_slot" if surface == "scope_review" else "custody_lost_slot"
    return {
        "slot_id": slot_id_for_row(0, prefix=prefix),
        "status": "error",
        "error": "durable review roster container is malformed",
        "operation_state": "custody_lost",
        "late_result_pending": True,
    }


def _commit_attempt_from_dict(d: Dict[str, Any]) -> CommitAttemptRecord:
    ts = str(d.get("ts", ""))
    status = str(d.get("status", "failed"))
    raw_triad = d.get("triad_raw_results")
    raw_scope = d.get("scope_raw_result")
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
        triad_raw_results=(
            list(raw_triad) if isinstance(raw_triad, list)
            else [] if raw_triad is None
            else [_malformed_roster_row("multi_model_review")]
        ),
        scope_raw_result=(
            dict(raw_scope) if isinstance(raw_scope, dict)
            else {} if raw_scope is None
            else {"raw_results": [_malformed_roster_row("scope_review")]}
        ),
        paid=bool(d.get("paid", False)),
        review_owner_pid=_coerce_int(d.get("review_owner_pid", 0)),
        raw_stripped=bool(d.get("raw_stripped", False)),
    )


_ATTEMPT_AUTHORITY_STRING_FIELDS = frozenset({
    "ts", "status", "phase", "started_ts", "updated_ts", "finished_ts",
    "task_id", "repo_key", "tool_name", "review_retry_key",
    "review_contract_fingerprint", "block_reason", "review_owner_session_id",
})
_ATTEMPT_AUTHORITY_BOOL_FIELDS = frozenset({
    "blocked", "paid", "late_result_pending", "raw_stripped",
})


def _validate_attempt_authority_shape(data: Dict[str, Any]) -> List[Any]:
    """Validate only mutation-authoritative attempt structure.

    Ordinary ``load_state`` remains fail-soft.  A locked read-modify-write must
    instead refuse malformed lifecycle authority before normalization can turn
    it into a different, writable value.  Nested reviewer rosters deliberately
    stay out of this validator: their existing fail-soft ``custody_lost``
    hardening is the recovery contract.
    """
    raw_attempts = data.get("attempts", [])
    if not isinstance(raw_attempts, list):
        raise ValueError("advisory review attempts must be a list")
    for index, item in enumerate(raw_attempts):
        if not isinstance(item, dict):
            raise ValueError(f"advisory review attempt {index} must be an object")
        for key in _ATTEMPT_AUTHORITY_STRING_FIELDS:
            if key in item and not isinstance(item[key], str):
                raise ValueError(
                    f"advisory review attempt {index} has invalid {key}"
                )
        for key in _ATTEMPT_AUTHORITY_BOOL_FIELDS:
            if key in item and not isinstance(item[key], bool):
                raise ValueError(
                    f"advisory review attempt {index} has invalid {key}"
                )
        if "attempt" in item and (
            not isinstance(item["attempt"], int) or isinstance(item["attempt"], bool)
        ):
            raise ValueError(
                f"advisory review attempt {index} has invalid attempt"
            )
        if "review_owner_pid" in item and (
            not isinstance(item["review_owner_pid"], int)
            or isinstance(item["review_owner_pid"], bool)
            or item["review_owner_pid"] < 0
        ):
            raise ValueError(
                f"advisory review attempt {index} has invalid review_owner_pid"
            )
        if "duration_sec" in item and (
            not isinstance(item["duration_sec"], (int, float))
            or isinstance(item["duration_sec"], bool)
        ):
            raise ValueError(
                f"advisory review attempt {index} has invalid duration_sec"
            )
    return raw_attempts


def _load_state_unlocked(
    drive_root: pathlib.Path, *, strict_attempt_authority: bool = False,
) -> AdvisoryReviewState:
    path = drive_root / _STATE_RELPATH
    if not path.exists():
        return AdvisoryReviewState()

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        if strict_attempt_authority:
            raise ValueError("advisory review state root must be an object")
        return AdvisoryReviewState()

    raw_attempts = (
        _validate_attempt_authority_shape(data)
        if strict_attempt_authority else data.get("attempts", [])
    )

    advisory_runs = [_record_from_dict(item) for item in (item for item in data.get("advisory_runs", []) if isinstance(item, dict))]
    attempts = [_commit_attempt_from_dict(item) for item in (item for item in raw_attempts if isinstance(item, dict))]
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
        state = _load_state_unlocked(drive_root, strict_attempt_authority=True)
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


def advisory_commit_ready(
    effectively_fresh: bool,
    open_obligations: Any,
    open_debts: Any,
    enforcement: str | None = None,
) -> bool:
    """SSOT for every ``repo_commit_ready`` projection (H5, capinv-447).

    Mirrors the real gate (``commit_gate._check_advisory_freshness``): a fresh /
    bypassed / skipped advisory run is required under every enforcement mode,
    while open obligations/debt block ``commit_reviewed`` only under
    ``blocking`` enforcement — under ``advisory`` the debt is disclosed and
    acknowledged, not gating. Advisory-readiness projection only: triad, scope,
    custody, and the other commit requirements stay independent.
    """
    if not effectively_fresh:
        return False
    if open_obligations or open_debts:
        if enforcement is None:
            from ouroboros.config import get_review_enforcement

            enforcement = get_review_enforcement()
        return str(enforcement or "").strip().lower() != "blocking"
    return True


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

    # Cap the historical ledger with EXPLICIT omission notes (the continuation
    # pattern): the full history stays on disk behind `review_status`, but this
    # section rides into EVERY task's context for the repo_key's lifetime, so
    # unbounded rendering grew monotonically with commit activity (~24K chars
    # live at the submarine forensics). P1: disclosed omission — the note names
    # the count and the recovery path; nothing is silently dropped.
    _LEDGER_RUN_CAP = 5
    _LEDGER_ATTEMPT_CAP = 5
    _COMMIT_MSG_CAP = 300
    if len(advisory_runs) > _LEDGER_RUN_CAP:
        lines.append(
            f"\n⚠️ OMISSION NOTE: {len(advisory_runs) - _LEDGER_RUN_CAP} older advisory "
            f"run(s) omitted (showing {_LEDGER_RUN_CAP} most recent; run `review_status` "
            "for the full ledger)."
        )
        advisory_runs = advisory_runs[-_LEDGER_RUN_CAP:]
    for run in advisory_runs:
        lines.append(f"\n{_RUN_STATUS_ICONS.get(run.status, '❓')} **{run.status.upper()}** | hash={run.snapshot_hash[:12]} | {run.ts}")
        lines.append(
            "   Commit: "
            + _truncate_review_artifact(str(run.commit_message or ""), limit=_COMMIT_MSG_CAP).replace("\n", " ")
        )
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
        lines.append("   Run preflight_review again before commit_reviewed.")

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
        if len(attempts) > _LEDGER_ATTEMPT_CAP:
            lines.append(
                f"⚠️ OMISSION NOTE: {len(attempts) - _LEDGER_ATTEMPT_CAP} older attempt(s) "
                f"omitted (showing {_LEDGER_ATTEMPT_CAP} most recent; run `review_status` "
                "for the full ledger)."
            )
            attempts = attempts[-_LEDGER_ATTEMPT_CAP:]
        for item in attempts:
            tool = item.tool_name or _DEFAULT_TOOL_NAME
            num = int(item.attempt or 0)
            label = f"{tool}#{num}" if num else tool
            phase = item.phase or "review"
            facts = [f"status={item.status}", f"phase={phase}", f"blocked={'yes' if item.blocked else 'no'}"]
            if item.commit_message:
                facts.append(
                    "commit="
                    + _truncate_review_artifact(str(item.commit_message or ""), limit=_COMMIT_MSG_CAP).replace("\n", " ")
                )
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
        lines.append(
            "   Commit: "
            + _truncate_review_artifact(str(ca.commit_message or ""), limit=_COMMIT_MSG_CAP).replace("\n", " ")
        )
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


# What the refusal quote renders at most from an over-window row's
# block_details (mirrors _quote_verdict_attempt's own limit=600), and a small
# bound for the free-text commit message on compacted rows.


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


# v7next F2.3a (D06): moved spans live in their owner leaves; re-exported
# here so this facade stays the single import surface for callers and tests.
from ouroboros.review_state_records import (  # noqa: E402, F401 -- intentional public re-exports
    AdvisoryRunRecord,
    CommitAttemptRecord,
    CommitReadinessDebtItem,
    ObligationItem,
    _ATTEMPT_MERGE_INCOMING_FIRST,
    _ATTEMPT_MERGE_INCOMING_LISTS,
    _ATTEMPT_STR_DEFAULTS,
    _CANONICAL_OBLIGATION_ITEM_RE,
    _DEBT_STR_DEFAULTS,
    _DEFAULT_ADVISORY_TOOL_NAME,
    _DEFAULT_TOOL_NAME,
    _LEGACY_CURRENT_REPO_KEY,
    _MAX_ATTEMPT_HISTORY,
    _MAX_COMMIT_READINESS_DEBTS,
    _MAX_RUN_HISTORY,
    _OBLIGATION_STR_DEFAULTS,
    _OPEN_COMMIT_READINESS_DEBT_STATUSES,
    _REVIEW_ATTEMPT_GRACE_SEC,
    _REVIEW_ATTEMPT_TTL_SEC,
    _RUN_STATUS_ICONS,
    _RUN_STR_DEFAULTS,
    _STATE_SCHEMA_VERSION,
    _allocate_prefixed_id,
    _append_finding_lines,
    _attempt_identity_tuple,
    _attempt_order_key,
    _coerce_int,
    _commit_readiness_debts_view,
    _dedupe_strings,
    _filter_lifecycle_records,
    _filter_repo_scope,
    _infer_next_prefixed_sequence,
    _looks_like_public_obligation_id,
    _make_obligation_fingerprint,
    _max_iso_ts,
    _merge_attempt,
    _min_iso_ts,
    _normalize_findings,
    _normalize_fingerprint_text,
    _normalize_obligation_item_key,
    _parse_iso_ts,
    _stable_digest,
    _utc_now,
    infer_review_phase,
)

from ouroboros.review_state_model import (  # noqa: E402, F401 -- intentional public re-exports
    AdvisoryReviewState,
)

from ouroboros.review_state_custody import (  # noqa: E402, F401 -- intentional public re-exports
    _ACTIVE_REVIEW_OPERATION_STATES,
    _STRIPPED_DETAILS_LIMIT,
    _STRIPPED_MESSAGE_LIMIT,
    _attempt_has_active_review_custody,
    _attempt_history_evictable,
    _attempt_review_roster_rows,
    _review_roster_row_is_pending,
    _strip_attempt_heavy_payload,
    checkpoint_pending_review_invocation,
)
