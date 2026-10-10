"""Commit-review status projection over the durable review-state records.

Boundary: this leaf reads ``review_state`` records (runs, commit attempts,
obligations, commit-readiness debts) and renders the semantic read-model plus the
human-facing ``review_status`` payload and one-line message. It knows nothing
about task-acceptance packets and must never import ``ouroboros.review_evidence``
— the dependency runs the other way, which keeps the module a leaf like
``review_evidence_refs``.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

from ouroboros.utils import sanitize_tool_result_for_log, truncate_review_artifact


def build_review_projection(
    drive_root: Any,
    *,
    repo_dir: Any = None,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: int | None = None,
    snapshot_hash_fn: Any = None,
    reader_task_id: str = "",
) -> Dict[str, Any]:
    """Build the semantic read-model shared by review_status-style renderers.

    ``task_id`` is a record FILTER; ``reader_task_id`` is the calling task, the
    only identity the stale-marker attribution may be relative to.
    """
    from ouroboros.review_state import (
        compute_snapshot_hash,
        load_state,
        make_repo_key,
    )

    drive_root_path = pathlib.Path(drive_root)
    repo_dir_path = pathlib.Path(repo_dir) if repo_dir else None
    state = load_state(drive_root_path)
    repo_filter = repo_key or (make_repo_key(repo_dir_path) if repo_dir_path is not None else None)
    tool_filter = tool_name or None
    task_filter = task_id or None
    runs = state.filter_advisory_runs(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )
    attempts = state.filter_attempts(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )
    latest = runs[-1] if runs else None
    selected_attempt = attempts[-1] if attempts else (
        None if (repo_filter or tool_filter or task_filter or attempt is not None) else state.latest_attempt()
    )
    try:
        if repo_dir_path is None:
            raise ValueError("repo_dir unavailable")
        hasher = snapshot_hash_fn or compute_snapshot_hash
        current_hash = hasher(repo_dir_path, "", paths=latest.snapshot_paths if latest else None)
        hash_mismatch = bool(
            latest
            and latest.status in {"fresh", "bypassed", "skipped", "parse_failure", "preflight_blocked", "tests_preflight_blocked", "error", "pending"}
            and latest.snapshot_hash != current_hash
        )
    except Exception:
        current_hash = ""
        hash_mismatch = False
    matching_run = state.find_by_hash(current_hash, repo_key=repo_filter) if current_hash else None
    effective_is_fresh = bool(state.is_fresh(current_hash, repo_key=repo_filter) if current_hash else False)
    stale_matches_repo = state.last_stale_repo_key in ("", repo_filter)
    stale_from_edit = bool(hash_mismatch or (state.last_stale_from_edit_ts and stale_matches_repo))
    effective_status = matching_run.status if matching_run else ("stale" if latest else "none")
    stale_ts = (state.last_stale_from_edit_ts if state.last_stale_from_edit_ts and stale_matches_repo
                else ("now (hash mismatch)" if hash_mismatch else None))
    stale_reason = ((state.last_stale_reason if stale_matches_repo else "")
                    or ("Current snapshot hash no longer matches the latest advisory run." if hash_mismatch else None))
    marker_attributed = stale_matches_repo
    look = _preflight_look(state, drive_root_path, repo_dir_path, repo_filter or "")
    if look is not None:
        # The author's preflight is the look the legacy runs used to be (D5-002): a
        # marker written before the look does not stale it; a worktree that moved
        # since the look does, with the editor named when a mutation was recorded.
        stale_from_edit = bool(look["stale_from_edit"])
        effective_status = "stale" if stale_from_edit else "fresh"
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
        "runs": runs,
        "attempts": attempts,
        "latest_run": latest,
        "matching_run": matching_run,
        "guidance_run": matching_run or latest,
        "selected_attempt": selected_attempt,
        "current_hash": current_hash,
        "effective_status": effective_status,
        "effective_hash": matching_run.snapshot_hash[:12] if matching_run and matching_run.snapshot_hash else None,
        "effective_is_fresh": effective_is_fresh,
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
    }


_TREE_MOVED_REASON = "The worktree no longer matches the tree the preflight read."


def _preflight_look(state: Any, drive_root: pathlib.Path, repo_dir: pathlib.Path | None, repo_key: str) -> Dict[str, Any] | None:
    """The checkout's newest preflight look (a ``surface=preflight`` ledger record)
    and whether the worktree moved since it: by the stale marker a recorded mutation
    wrote after the look, or by the live tree against the tree the record bound.
    ``None`` when the checkout has no look; a tree that cannot be read compares as
    unmoved (the marker still speaks)."""
    try:
        from ouroboros.review_ledger import latest_preflight_record

        record = latest_preflight_record(drive_root, repo_key=repo_key)
    except Exception:
        record = None
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
        "latest_advisory_hash": projection["effective_hash"],
        "stale_from_edit": projection["stale_from_edit"],
        "stale_from_edit_ts": projection["stale_from_edit_ts"],
        "stale_reason": projection["stale_reason"],
        **{key: projection.get(key, "") for key in ("stale_task_id", "stale_attribution", "stale_repo_key")},
        "preflight": projection.get("preflight"),
        "filters": projection["filters"],
        "advisory_runs": [_review_status_run_to_dict(run) for run in reversed(projection.get("runs") or [])],
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


def _run_failure_reason(run: Any) -> str | None:
    """Typed cause for a non-parseable advisory run. Diagnostics only.

    Never consumed by the commit gate, freshness, or debt: it exists so a
    repeated deterministic failure is visible after the FIRST attempt instead of
    reading as a generic ``parse_failure`` for hours.
    """
    if str(getattr(run, "status", "") or "") != "parse_failure":
        return None
    from ouroboros.triad_review import empty_array_is_verified_clean

    raw = str(getattr(run, "raw_result", "") or "").strip()
    if not raw:
        return "empty_response"
    if empty_array_is_verified_clean(raw):
        # A contract-compliant clean verdict was still rejected: that is a
        # regression of the sentinel contract, not a model failure. Asking the
        # shared predicate — not a second substring test — is what keeps this
        # diagnostic honest when the contract changes.
        return "clean_sentinel_rejected"
    if raw.startswith("[") or raw.startswith("```"):
        return "malformed_array"
    return "non_json_prose"


def _review_status_run_to_dict(run: Any) -> Dict[str, Any]:
    findings = [
        item for item in (getattr(run, "items", []) or [])
        if isinstance(item, dict) and str(item.get("verdict", "")).upper() == "FAIL"
    ]
    data = {
        "snapshot_hash": str(getattr(run, "snapshot_hash", ""))[:12],
        "critical_findings": sum(1 for item in findings if str(item.get("severity", "")).lower() == "critical"),
        "total_findings": len(findings),
        "attempt": int(getattr(run, "attempt", 0) or 0) or None,
    }
    for key in ("commit_message", "status", "ts", "snapshot_summary"):
        data[key] = str(getattr(run, key, "") or "")
    for key in ("bypass_reason", "repo_key", "tool_name", "task_id"):
        data[key] = str(getattr(run, key, "") or "") or None
    # Already persisted per run, previously dropped from the projection: without
    # these the owner sees repeated identical statuses with no usable cause.
    data["failure_reason"] = _run_failure_reason(run)
    execution = getattr(run, "execution", {}) or {}
    data["failure_phase"] = str(execution.get("failure_phase") or "") or None
    data["failure_code"] = str(execution.get("failure_code") or "") or None
    if execution:
        data["execution"] = {key: str(execution[key]) for key in (
            "invocation_id", "pending_invocation_id", "operation_id", "operation_state", "fingerprint",
        ) if isinstance(execution.get(key), str)}
        intent = execution.get("intent")
        if isinstance(intent, dict):
            # Exact host-authored rejoin inputs, never the private reviewer prompt
            # or its raw answer; use the existing public secret redactor.
            data["execution"]["intent"] = {key: sanitize_tool_result_for_log(intent[key]) for key in (
                "commit_message", "goal", "scope", "review_rebuttal",
            ) if isinstance(intent.get(key), str)}
    data["model_used"] = str(getattr(run, "model_used", "") or "") or None
    duration = getattr(run, "duration_sec", None)
    data["duration_sec"] = round(float(duration), 2) if duration else None
    prompt_chars = getattr(run, "prompt_chars", None)
    data["prompt_chars"] = int(prompt_chars) or None if prompt_chars else None
    # Deliberately NO raw excerpt here: raw_result is untrusted reviewer output
    # that can echo secret-bearing diff content, and this projection is returned
    # to the active model. The typed reason above is derived, not raw, and the
    # complete text stays in the durable advisory run record addressed by the
    # snapshot_hash/ts already on this row.
    return data


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
    scope_raw = getattr(attempt, "scope_raw_result", None) or {}
    return {
        "triad_actors": [
            {"model_id": r.get("model_id", "?"), "status": r.get("status", "?")}
            for r in (getattr(attempt, "triad_raw_results", None) or [])
        ],
        "scope_actor": (
            {"model_id": scope_raw.get("model_id", "?"), "status": scope_raw.get("status", "?")}
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
        from ouroboros.tools.review_helpers import REVIEW_POOL_EMPTY_SENTENCE

        counts_only = "The commit gate counts only a PASS/FAIL answer"
        reason_map = {
            "no_advisory": "Held by the retired advisory gate (history); commits no longer wait for a preflight.",
            "pool_empty": f"Review NOT_PERFORMED: {REVIEW_POOL_EMPTY_SENTENCE}",
            "coupling_not_performed": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['coupling_not_performed']}. {counts_only}; "
                "retry the commit or configure a retrieving reviewer seat (Settings → Agents, a Reviewer row that reads the work itself)."),
            "change_unanswered": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['change_unanswered']}. {counts_only}; "
                "retry the commit — each seat's error is recorded in the attempt."),
            "review_late_result_pending": (
                f"Review NOT_PERFORMED: {NOT_PERFORMED_PHRASES['review_late_result_pending']}; no verdict is counted yet. "
                "The pending result settles into the ledger when it lands; retry the commit afterwards."),
            "critical_findings": "Reviewers found critical issues. Repair or rebut (review_rebuttal) the findings listed, then re-run commit_reviewed.",
            "review_quorum": "Not enough review models responded. Retry — usually transient.",
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
            "review_cycles_exhausted": "This task tree spent its paid review cycles (OUROBOROS_REVIEW_MAX_CYCLES). Finalize honestly or ask the owner to raise the cap.",
            "review_subject_binding_mismatch": "The reviewed managed subject is not the tree this commit would write. Re-stage the intended candidate and retry so review and commit describe the same tree.",
        }
        label = "BLOCKED" if ca.status == "blocked" else "FAILED"
        current = (
            f"Last commit {label} ({ca.block_reason or 'unclassified'}): "
            f"{reason_map.get(ca.block_reason, ca.block_reason or 'unknown')}"
            f"  |  {current}"
        )
    if projection.get("open_debts"):
        current = f"{current}  |  Commit-readiness debt: {len(projection['open_debts'])}"
    return current
