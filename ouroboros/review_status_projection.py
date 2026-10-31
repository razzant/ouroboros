"""Commit-review status projection over the durable review-state records.

Boundary: this leaf reads ``review_state`` records (commit attempts, obligations,
commit-readiness debts; the retired advisory lane's archive only by its address) and renders the semantic read-model plus the
human-facing ``review_status`` payload and one-line message. It knows nothing
about other evidence and must never import ``ouroboros.review_evidence`` — the
dependency runs the other way, which keeps the module a leaf. It also owns the read-only join of a split-drive task's
canonically held ``review_change`` attempts that ``review_evidence`` shares.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

from ouroboros.utils import truncate_review_artifact


def build_review_projection(
    drive_root: Any,
    *,
    repo_dir: Any = None,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: int | None = None,
    reader_task_id: str = "",
    canonical_root: Any = None,
) -> Dict[str, Any]:
    """Build the semantic read-model shared by review_status-style renderers.

    ``task_id`` is a record FILTER; ``reader_task_id`` is the calling task, the
    only identity the stale-marker attribution may be relative to, and the one
    whose ``review_change`` rows ``canonical_root`` holds: they join this drive's
    rows before any filter, so every filter reads the drive's own view. The
    retired advisory lane's rows are only named where they are kept
    (``advisory_run_archive``), never read as freshness, guidance or debt.
    """
    from ouroboros.review_state import load_state, make_repo_key

    drive_root_path = pathlib.Path(drive_root)
    repo_dir_path = pathlib.Path(repo_dir) if repo_dir else None
    state = load_state(drive_root_path)
    state.attempts, canonical_unreadable = with_canonical_change_attempts(
        state.attempts, canonical_root, drive_root_path, reader_task_id)
    repo_filter = repo_key or (make_repo_key(repo_dir_path) if repo_dir_path is not None else None)
    tool_filter = tool_name or None
    task_filter = task_id or None
    attempts = state.filter_attempts(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )
    selected_attempt = attempts[-1] if attempts else (
        None if (repo_filter or tool_filter or task_filter or attempt is not None) else state.latest_attempt()
    )
    # Without a preflight look nothing is fresh or stale: a stale marker no look precedes
    # was written by the retired advisory lane and stays history in the file (owner decision).
    stale_from_edit, effective_status, stale_ts, stale_reason, marker_attributed = False, "none", None, None, False
    look = _preflight_look(state, drive_root_path, repo_dir_path, repo_filter or "")
    if look is not None:
        # The author's preflight is the checkout's look (D5-002): a
        # marker written before the look does not stale it; a worktree that moved
        # since the look does, with the editor named when a mutation was recorded.
        stale_from_edit = bool(look["stale_from_edit"])
        effective_status = "unknown" if look.get("unreadable") else "stale" if stale_from_edit else "fresh"
        marker_attributed = bool(look["marked"])
        stale_ts = (state.last_stale_from_edit_ts if look["marked"]
                    else ("now (worktree moved)" if look["tree_moved"] else None))
        stale_reason = (state.last_stale_reason if look["marked"]
                        else (_TREE_MOVED_REASON if look["tree_moved"] else None))
    open_obligations = state.get_open_obligations(repo_key=repo_filter)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_filter)
    try:
        from ouroboros.utils import read_json_dict

        advisory_overrides = read_json_dict(drive_root_path / "state" / "advisory_overrides.json") or {}
    except Exception:
        advisory_overrides = {}
    return {
        "state": state,
        "filters": {
            "repo_key": repo_filter,
            "tool_name": tool_filter,
            "task_id": task_filter,
            "attempt": attempt,
        },
        "attempts": attempts,
        "selected_attempt": selected_attempt,
        "effective_status": effective_status,
        "stale_from_edit": stale_from_edit,
        "stale_from_edit_ts": stale_ts,
        "stale_reason": stale_reason,
        # Attribution of the marker only; stale_from_edit/freshness above never read it.
        **{key: value if marker_attributed else "" for key, value in
           state.stale_marker_provenance(reader_task_id).items()},
        "preflight": look,
        "open_obligations": open_obligations,
        "open_debts": open_debts,
        "retry_anchor": "commit_readiness_debt" if open_debts else None,
        "advisory_overrides": advisory_overrides,
        "canonical_attempts_unreadable": canonical_unreadable,
        "advisory_run_archive": _advisory_run_archive(state, drive_root_path),
    }


_ARCHIVE_NOTE = (
    "The retired advisory pre-review lane, kept as recorded: `advisory_runs` and the commit_readiness_debts "
    "it minted (source advisory_review with a readiness_warning:advisory: fingerprint, and advisory_stale "
    "copies of the stale marker). History only, never current freshness, guidance or debt, as is a stale "
    "marker no preflight look precedes. Read it with read_file(path=<path>, start_char/max_chars) or, "
    "on your own drive, query_data(op='json', kind='advisory_review', select='advisory_runs')."
)


def _advisory_run_archive(state: Any, drive_root: pathlib.Path) -> Dict[str, Any]:
    """Where this store keeps the retired advisory lane's rows and the debts they minted
    (owner decision: archive only); ``{}`` when it keeps none. ``path`` is the file this
    projection read, so a split child's own drive is never mistaken for another root."""
    from ouroboros.review_state import _STATE_RELPATH

    if not state.advisory_runs and not state.archived_advisory_debts:
        return {}
    return {"path": str(drive_root.resolve(strict=False) / _STATE_RELPATH), "advisory_runs": len(state.advisory_runs),
            "advisory_debts": len(state.archived_advisory_debts), "note": _ARCHIVE_NOTE}


def with_canonical_change_attempts(attempts: List[Any], canonical_root: Any, drive_root: Any,
                                   task_id: str) -> Tuple[List[Any], str]:
    """``attempts`` (this task's drive rows) plus the ``review_change`` rows its canonical
    root holds for it (``review_change_custody.review_state_root``), and why that root
    could not be read ("" when it was). The canonical row wins an identity both roots
    hold (a move interrupted between its two saves). Reading moves nothing; an
    unreadable canonical state is named, never read as no rows."""
    if not canonical_root or not task_id or (
            pathlib.Path(canonical_root).resolve() == pathlib.Path(drive_root).resolve()):
        return attempts, ""
    from ouroboros.review_state import _STATE_RELPATH, _load_state_unlocked
    from ouroboros.review_state_records import _attempt_identity_tuple, _attempt_order_key

    try:
        state = _load_state_unlocked(pathlib.Path(canonical_root), strict_attempt_authority=True)
    except (OSError, TypeError, ValueError) as exc:
        return attempts, f"{pathlib.Path(canonical_root) / _STATE_RELPATH} unreadable ({type(exc).__name__}: {exc})"
    owned = state.filter_attempts(task_id=task_id, tool_name="review_change")
    held = {_attempt_identity_tuple(row) for row in owned}
    return (sorted([*(row for row in attempts if _attempt_identity_tuple(row) not in held), *owned],
                   key=_attempt_order_key) if owned else attempts), ""


_TREE_MOVED_REASON = "The worktree no longer matches the tree the preflight read."


def _preflight_look(state: Any, drive_root: pathlib.Path, repo_dir: pathlib.Path | None, repo_key: str) -> Dict[str, Any] | None:
    """The checkout's newest preflight look (a ``surface=preflight`` ledger record)
    and whether the worktree moved since it: by the stale marker a recorded mutation
    wrote after the look, or by the live tree against the tree the record bound.
    ``None`` when the checkout has no look; a tree that cannot be read compares as
    unmoved (the marker still speaks). A look that cannot be read is named
    (``unreadable``), never read as no look: the checkout's marker still speaks."""
    try:
        from ouroboros.review_ledger import latest_preflight_record

        record = latest_preflight_record(drive_root, repo_key=repo_key)
    except Exception as exc:
        marked = bool(state.last_stale_from_edit_ts and state.last_stale_repo_key in ("", repo_key))
        return {"unreadable": f"{type(exc).__name__}: {exc}", "marked": marked, "tree_moved": False,
                "stale_from_edit": marked}
    if not record:
        return None
    look_ts = str(record.get("ts") or "")
    recorded_tree = str((record.get("subject") or {}).get("tree_sha") or "")
    current_tree = ""
    if repo_dir is not None and recorded_tree:
        try:
            from supervisor.update_candidate import worktree_snapshot_tree

            current_tree, _error = worktree_snapshot_tree("HEAD", cwd=str(repo_dir))
        except Exception:
            current_tree = ""
    marked = bool(state.marker_postdates(look_ts, repo_key))
    tree_moved = bool(current_tree and current_tree != recorded_tree)
    rows = [row for row in (record.get("rows") or []) if isinstance(row, dict)]
    return {
        "record_id": str(record.get("record_id") or ""),
        "ts": look_ts,
        "state": str(record.get("state") or ""),
        "aggregate": str((record.get("verdict") or {}).get("aggregate") or ""),
        "reviewer": str(rows[0].get("seat_id") or "") if rows else "",
        "tree_sha": recorded_tree[:12],
        "marked": marked,
        "tree_moved": tree_moved,
        "stale_from_edit": marked or tree_moved,
    }


def build_review_status_payload(projection: Dict[str, Any], *, next_step: str, include_raw: bool = False) -> Dict[str, Any]:
    selected_attempt = projection.get("selected_attempt")
    open_obligations = list(projection.get("open_obligations") or [])
    open_debts = list(projection.get("open_debts") or [])
    payload: Dict[str, Any] = {
        "latest_advisory_status": projection["effective_status"],
        "stale_from_edit": projection["stale_from_edit"],
        "stale_from_edit_ts": projection["stale_from_edit_ts"],
        "stale_reason": projection["stale_reason"],
        **{key: projection.get(key, "") for key in ("stale_task_id", "stale_attribution", "stale_repo_key")},
        "preflight": projection.get("preflight"),
        "filters": projection["filters"],
        "attempts": [_review_status_attempt_to_dict(item) for item in reversed(projection.get("attempts") or [])],
        "selected_commit_attempt": _review_status_attempt_payload(selected_attempt),
        "open_obligations": [_review_status_obligation_to_dict(item) for item in open_obligations],
        "open_obligations_count": len(open_obligations),
        "commit_readiness_debts": [_review_status_debt_to_dict(item) for item in open_debts],
        "commit_readiness_debts_count": len(open_debts),
        "retry_anchor": projection["retry_anchor"],
        "status_summary": _review_status_message(projection),
        "next_step": next_step,
    }
    if selected_attempt is not None and selected_attempt.phase in {"review_only", "late_wait"}:
        from ouroboros.config import get_review_enforcement
        from ouroboros.review_records import review_outcome_received

        received = review_outcome_received(
            [*selected_attempt.triad_raw_results, selected_attempt.scope_raw_result],
            findings=[*selected_attempt.critical_findings, *selected_attempt.advisory_findings],
            terminal=selected_attempt.phase == "review_only" and selected_attempt.status == "reviewed",
        )
        if get_review_enforcement() == "advisory" and received:
            payload["review_reference"] = {"surface": "commit", **{key: getattr(selected_attempt, key) for key in
                ("repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint")}}
            payload["next_step"] = ("Read the returned findings. You may revise and request another permitted review, stop, or call "
                "commit_reviewed with this review_reference and an explicit author_disposition/rationale to commit the current attributed candidate without a new panel. Independent checks still apply.")
    if projection.get("canonical_attempts_unreadable"):
        payload["canonical_attempts_unreadable"] = projection["canonical_attempts_unreadable"]
    if projection.get("advisory_run_archive"):
        payload["advisory_run_archive"] = projection["advisory_run_archive"]
    payload["message"] = payload["status_summary"]
    # Persistent advisory-enforcement visibility (BIBLE P3 loud-advisory bound):
    # how many blocking-grade signals advisory enforcement waved through.
    overrides = projection.get("advisory_overrides")
    if isinstance(overrides, dict) and overrides.get("count"):
        payload["advisory_overrides_count"] = int(overrides.get("count") or 0)
        payload["advisory_overrides_recent"] = list(overrides.get("recent") or [])
    if include_raw and selected_attempt is not None:
        payload["raw_evidence"] = {
            "attempt_ts": selected_attempt.ts,
            "attempt_number": int(selected_attempt.attempt or 0) or None,
            "tool_name": selected_attempt.tool_name or None,
            "triad_raw_results": list(selected_attempt.triad_raw_results or []),
            "scope_raw_result": dict(selected_attempt.scope_raw_result or {}),
        }
    return payload


def _review_status_attempt_payload(ca: Any) -> Dict[str, Any] | None:
    if ca is None:
        return None
    data = {
        key: getattr(ca, key) or None
        for key in ("block_reason", "repo_key", "tool_name", "task_id", "phase", "fingerprint_status")
    }
    data.update({
        "status": ca.status,
        "commit_message": ca.commit_message,
        "ts": ca.ts,
        "duration_sec": round(ca.duration_sec, 1),
        "block_details_preview": truncate_review_artifact(ca.block_details, limit=300) if ca.block_details else None,
        "attempt": int(ca.attempt or 0) or None,
        "blocked": bool(ca.blocked),
        "late_result_pending": bool(ca.late_result_pending),
        "critical_findings": len(ca.critical_findings or []),
        "advisory_findings": len(ca.advisory_findings or []),
        "obligation_ids": list(ca.obligation_ids or []),
        "readiness_warnings": list(ca.readiness_warnings or []),
        "pre_review_fingerprint": ca.pre_review_fingerprint[:12] or None,
        "post_review_fingerprint": ca.post_review_fingerprint[:12] or None,
        "degraded_reasons": list(ca.degraded_reasons or []),
        # Max-Review-Cycles accounting facts (Q16 auditability): the typed
        # block class, the dispatch-paid fact, and the identities the free
        # refusal/replay decisions key on.
        "block_class": str(getattr(ca, "block_class", "") or "") or None,
        "paid": bool(getattr(ca, "paid", False)),
        "rebuttal_sha256": str(getattr(ca, "rebuttal_sha256", "") or "")[:12] or None,
        "review_contract_fingerprint": str(getattr(ca, "review_contract_fingerprint", "") or "")[:12] or None,
        "root_task_id": str(getattr(ca, "root_task_id", "") or "") or None,
        **_review_status_actor_summary(ca),
    })
    return data


def _review_status_attempt_to_dict(item: Any) -> Dict[str, Any]:
    data = _review_status_attempt_payload(item) or {}
    data.pop("commit_message", None)
    data.pop("block_details_preview", None)
    data["ts"] = item.ts
    return data


def _review_status_actor_summary(attempt: Any) -> Dict[str, Any]:
    from ouroboros.review_records import TYPED_FAILURE_FACT_KEYS

    def actor(row):
        return {key: row[key] for key in ("slot_id", "model_id", "status", *TYPED_FAILURE_FACT_KEYS,
                                          "response_ref") if key in row}

    scope_raw = getattr(attempt, "scope_raw_result", None) or {}
    return {
        "triad_actors": [
            actor(r)
            for r in (getattr(attempt, "triad_raw_results", None) or [])
        ],
        "scope_actor": (
            actor(scope_raw)
            if scope_raw.get("status") else None
        ),
    }


def _review_status_obligation_to_dict(item: Any) -> Dict[str, Any]:
    return {
        **{key: getattr(item, key, "") for key in ("obligation_id", "fingerprint", "item", "severity", "status")},
        "reason": truncate_review_artifact(item.reason, limit=200),
        "source_ts": item.source_attempt_ts,
        "source_commit": item.source_attempt_msg,
    }


def _review_status_debt_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "debt_id": item.debt_id,
        "category": item.category,
        "title": item.title,
        "summary": truncate_review_artifact(item.summary, limit=220),
        "status": item.status,
        "severity": item.severity,
        "source": item.source,
        "repo_key": item.repo_key or None,
        "source_obligation_ids": list(item.source_obligation_ids or []),
        "evidence": list(item.evidence or []),
        "updated_at": item.updated_at,
    }


def _review_status_message(projection: Dict[str, Any]) -> str:
    ca = projection.get("selected_attempt")
    current = f"Current advisory: {projection['effective_status']}"
    if ca and ca.status in ("blocked", "failed"):
        from ouroboros.review_ledger import NOT_PERFORMED_PHRASES

        counts_only = "The commit gate counts only a PASS/FAIL answer"
        reason_map = {
            "no_advisory": "Held by the retired advisory gate (history); commits no longer wait for a preflight.",
            "pool_empty": ("Review NOT_PERFORMED (history): no row was marked Reviewer; an empty review pool "
                           "now reviews on the Main model alone, so retrying the commit runs that review."),
            "coupling_not_performed": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['coupling_not_performed']}. {counts_only}; "
                "retry the commit or configure a retrieving reviewer seat (Settings → Agents, a Reviewer row that reads the work itself)."),
            "change_unanswered": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['change_unanswered']}. {counts_only}; "
                "retry the commit — each seat's error is recorded in the attempt."),
            "review_late_result_pending": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['review_late_result_pending']}; no verdict is counted yet. "
                "The pending result settles into the ledger when it lands; retry the commit afterwards."),
            "result_unanswered": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['result_unanswered']}. A result review counts only a "
                "PASS/FAIL answer; each seat's error is recorded in the attempt."),
            "skill_unanswered": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['skill_unanswered']}. A skill review counts only a "
                "PASS/FAIL answer; each seat's error is recorded in the attempt, and the skill stays pending."),
            "critical_findings": "Reviewers found critical issues. Repair or rebut (review_rebuttal) the findings listed, then re-run commit_reviewed.",
            "review_quorum": "Not enough reviewers answered; the recorded seat causes follow below.",
            "parse_failure": "Review models could not produce parseable output. Retry the commit.",
            "infra_failure": "Infrastructure failure. Check block_details.",
            "scope_blocked": "Scope reviewer blocked the commit. Address scope review findings.",
            "preflight": "A deterministic preflight check failed (release metadata, syntax or staged companions). See block_details.",
            "tests_preflight_blocked": "The tests preflight failed before review. Fix the failures in block_details.",
            "revalidation_failed": "The staged diff changed after review. Re-run the review.",
            "fingerprint_unavailable": "The staged diff could not be fingerprinted. Fix git diff and retry.",
            "overlap_guard": "Another reviewed attempt is still active. Wait or expire it before retrying.",
            "attempt_cap_reached": "The same staged diff was review-blocked repeatedly. Change the diff or rebut via review_rebuttal.",
            "identical_diff_refused": "This exact staged diff was already review-blocked. Change the diff or supply a NEW review_rebuttal (identical bytes are never re-reviewed for pay).",
            "review_cycles_exhausted": "This task tree spent its paid review cycles (OUROBOROS_REVIEW_MAX_CYCLES). Under Advisory decide whether what was reviewed is enough; under Blocking only the owner may raise the cap.",
            "review_subject_binding_mismatch": "The reviewed managed subject is not the tree this commit would write. Re-stage the intended candidate and retry so review and commit describe the same tree.",
        }
        label = "BLOCKED" if ca.status == "blocked" else "FAILED"
        what = {"review_result": "result review", "review_skill": "skill review"}.get(getattr(ca, "tool_name", ""), "commit")
        if what == "result review":  # a result is revised and delivered again, never committed
            reason_map.update(
                critical_findings="Reviewers found critical issues in the result. Revise it or answer the findings, then deliver again.",
                review_late_result_pending=(
                    f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['review_late_result_pending']}; no verdict is counted yet. "
                    "The next delivery of the same result collects it at $0."))
        current = (
            f"Last {what} {label} ({ca.block_reason or 'unclassified'}): "
            f"{reason_map.get(ca.block_reason, ca.block_reason or 'unknown')}"
            f"  |  {current}"
        )
        if ca.block_reason == "review_quorum":
            from ouroboros.review_records import review_failure_summary
            current += "\n" + "\n".join(review_failure_summary(row) for row in (ca.triad_raw_results or [])
                                         if row.get("status") != "responded")
    if projection.get("open_debts"):
        current = f"{current}  |  Commit-readiness debt: {len(projection['open_debts'])}"
    if projection.get("canonical_attempts_unreadable"):
        current = (f"{current}  |  This task's canonically held review_change attempts are not listed: "
                   f"{projection['canonical_attempts_unreadable']}")
    return current
