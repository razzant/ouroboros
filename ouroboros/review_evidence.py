"""Structured review-evidence collection for summaries, reflections, and UX."""

from __future__ import annotations

import json
import hashlib
import logging
import pathlib
import subprocess  # noqa: F401
from typing import Any, Dict, List

from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT  # noqa: F401
from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)

# The acceptance packet's typed sections, their caps and its budget live in
# their own owner below this module's seam; they are re-exported here because
# this module is their historical import site (and the site the host-diff and
# evidence-ref seams stay patchable at), and that owner must never import this
# module back.
from ouroboros.review_evidence_sections import (  # noqa: F401  (compat re-exports)
    _ACCEPT_ARGS_CAP,
    _ACCEPT_ARTIFACT_PREVIEW_CAP,
    _ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES,
    _ACCEPT_DELTA_CHILD_CAP,
    _ACCEPT_NOTES_CAP,
    _ACCEPT_OBLIGATIONS_MAX,
    _ACCEPT_RESULT_CAP,
    _ACCEPT_RETRIEVAL_URLS_MAX,
    _ACCEPT_TOTAL_BUDGET,
    _ACCEPT_TRAJECTORY_MAX_CALLS,
    _accept_artifact_manifest,
    _accept_capability_deltas,
    _accept_claim_support_refs,
    _accept_effective_claims,
    _accept_enforce_budget,
    _accept_obligation_row,
    _accept_owner_directives,
    _accept_protected_set,
    _accept_receipt_exhibits,
    _accept_redact_cap,
    _accept_task_contract,
    _accept_trajectory,
    _accept_verification_summary,
    _owner_content_projection,
    collect_turn_diff,
    obligation_is_pending,
    task_acceptance_evidence_revision,
)

# D-Q5 exhibit-key vocabulary + exact-membership resolver: extracted to the
# ``review_evidence_refs`` leaf (module size gate); re-exported here so every
# historical import site keeps resolving. ``annotate_criteria_evidence_resolution``
# below reads the two functions through THESE module globals on purpose — the
# fail-closed seam stays patchable at ``review_evidence.<name>`` (D-Q5 tests).
from ouroboros.review_evidence_refs import (
    _RESOLUTION_UNAVAILABLE_ROW as _RESOLUTION_UNAVAILABLE_ROW,
)
from ouroboros.review_evidence_refs import (
    acceptance_evidence_ref_vocabulary as acceptance_evidence_ref_vocabulary,
)
from ouroboros.review_evidence_refs import (
    CLAIM_ID_UNSUPPORTED as CLAIM_ID_UNSUPPORTED,
)
from ouroboros.review_evidence_refs import (
    NON_RESOLVING_BASIS_KINDS as NON_RESOLVING_BASIS_KINDS,
)
from ouroboros.review_evidence_refs import (
    resolve_criteria_evidence_refs as resolve_criteria_evidence_refs,
)


def annotate_criteria_evidence_resolution(actors: Any, evidence: Any) -> None:
    """Stamp per-actor ``criteria_refs_unresolved`` disclosure rows in place (D-Q5).

    Runs ONCE per acceptance panel over actor dict rows; a fully-resolving actor
    gets NO annotation (the common clean path is byte-identical). The annotation
    feeds ONLY ``task_acceptance_is_clean`` and disclosure — never parse validity,
    never quorum, never a verdict.

    TOTAL and fail-CLOSED. An annotation that did not run may never be read as
    "everything resolved": the absence of a row is what authorizes the clean bit,
    so a resolver failure that left the rows off would silently certify evidence
    nobody checked. Any failure therefore stamps the typed
    ``host_resolution_unavailable`` row on EVERY actor, landing on the same
    clean-bit rail as an unresolved ref — never a verdict, never a veto."""
    unavailable = False
    try:
        vocabulary = acceptance_evidence_ref_vocabulary(evidence)
    except Exception:
        log.warning("acceptance evidence-ref vocabulary unavailable", exc_info=True)
        vocabulary, unavailable = {}, True
    for actor in actors if isinstance(actors, list) else []:
        if not isinstance(actor, dict):
            continue
        try:
            if unavailable:
                actor["criteria_refs_unresolved"] = [dict(_RESOLUTION_UNAVAILABLE_ROW)]
                continue
            parsed = actor.get("parsed") if isinstance(actor.get("parsed"), dict) else {}
            rows = resolve_criteria_evidence_refs(parsed.get("criteria_used"), vocabulary)
            if rows:
                actor["criteria_refs_unresolved"] = rows
        except Exception:
            log.warning("acceptance evidence-ref resolution failed for one actor", exc_info=True)
            actor["criteria_refs_unresolved"] = [dict(_RESOLUTION_UNAVAILABLE_ROW)]


def build_task_acceptance_evidence(
    ctx: Any,
    *,
    llm_trace: Dict[str, Any] | None = None,
    drive_root: Any = None,
    task_id: str = "",
    task_type: str = "",
    agent_evidence: Dict[str, Any] | None = None,
    include_recent_commit: bool = False,
    canonical_subject: str = "",
    subtree_statuses: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Process-aware task-acceptance evidence packet (v6.51.0 idea-2). Typed sections with
    explicit PROVENANCE tags (`host_attested`/`agent_supplied`/`tool_result`/`artifact`/
    `hidden_or_restricted`): full task contract, a first-class verification_summary (red
    receipts surfaced), the host-collected redacted repo_diff, a bounded+redacted tool-call
    trajectory (HOW it was solved), and a leak-safe artifact manifest. Bounded by a DISCLOSED
    truncation budget (P1). Shared by the agent-tool and host-forced acceptance paths so the
    reviewer can critique outcome AND process (Bible P3/P12/P2). The reviewer prompt
    (review_substrate) is the authority that applies the anti-cheat boundary — it must never
    credit success to `hidden_or_restricted` evidence."""
    from ouroboros.observability import redact_projection
    from ouroboros.outcomes import read_verification_receipts

    ev: Dict[str, Any] = {}
    prov: Dict[str, str] = {}
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    root_task_id = str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)
    ev["canonical_payload"] = {
        "source": "review_request.subject",
        "sha256": hashlib.sha256(str(canonical_subject or "").encode("utf-8")).hexdigest(),
        "chars": len(str(canonical_subject or "")),
    }
    ev["aliases"] = {
        "task_id": str(task_id or getattr(ctx, "task_id", "") or ""),
        "root_task_id": root_task_id,
        "parent_task_id": str(meta.get("parent_task_id") or ""),
        "project_id": str(meta.get("project_id") or getattr(ctx, "project_id", "") or ""),
    }
    prov["canonical_payload"] = "host_attested"
    prov["aliases"] = "host_attested"
    if isinstance(agent_evidence, dict) and agent_evidence:
        a = dict(agent_evidence)
        if "repo_diff" in a:
            # Never let an agent-supplied value masquerade as the host diff.
            a["agent_supplied_repo_diff"] = a.pop("repo_diff")
        # Redact agent-supplied evidence too (structural key-aware) — it is serialized into an
        # external reviewer prompt, so a token/password in it is an exfil surface (review round-4).
        ev["agent_supplied"] = redact_projection(a).value
        prov["agent_supplied"] = "agent_supplied"
    contract = _accept_task_contract(ctx)
    # W2: resolve the claims that bind this task through the ONE seam — ingress
    # first, plan-frozen only when ingress is empty. The packet VIEW carries them;
    # the durable/live contract is never mutated.
    claims, claims_source = _accept_effective_claims(ctx, contract, drive_root, task_id)
    if claims_source == "plan_review":
        contract = {**contract, "acceptance_claims": claims}
    receipts = read_verification_receipts(drive_root, task_id) if (drive_root is not None and task_id) else []
    owner_directives = _accept_owner_directives(ctx, drive_root, task_id)
    if owner_directives:
        # This is an immutable verbatim corpus, not a parsed decision ledger:
        # reviewers interpret explicit approvals/changes from the owner text.
        ev["owner_requirements_and_decisions"] = redact_projection(owner_directives).value
        prov["owner_requirements_and_decisions"] = "host_attested"
    if contract:
        # Structural (key-aware) redaction of the full contract before it enters the prompt.
        ev["task_contract"] = redact_projection(contract).value
        prov["task_contract"] = "host_attested"
        if claims_source:
            ev["acceptance_claims_source"] = claims_source
            prov["acceptance_claims_source"] = "host_attested"
        support_refs = _accept_claim_support_refs(contract, receipts)
        if support_refs:
            ev["acceptance_support_refs"] = redact_projection(support_refs).value
            prov["acceptance_support_refs"] = "host_attested"
    ev["verification_summary"] = _accept_verification_summary(receipts)
    prov["verification_summary"] = "host_attested"
    receipt_exhibits = _accept_receipt_exhibits(receipts)
    if receipt_exhibits:
        # D-Q5: the indexed exhibit list the receipt-ref vocabulary derives from —
        # `verification_receipts[i]` must name a row that is HERE, never a bare count.
        ev["verification_receipts"] = redact_projection(receipt_exhibits).value
        prov["verification_receipts"] = "host_attested"
    if drive_root is not None and task_id:
        from ouroboros.mutation_attribution import load_mutation_evidence_projection

        mutation_projection = load_mutation_evidence_projection(drive_root, task_id)
        if mutation_projection:
            ev["mutation_attribution"] = mutation_projection
            prov["mutation_attribution"] = "host_attested"
        delta_aggregate = _accept_capability_deltas(drive_root, task_id, root_task_id)
        if delta_aggregate:
            ev["capability_deltas"] = redact_projection(delta_aggregate).value
            prov["capability_deltas"] = "host_attested"
    ev["repo_diff"] = collect_turn_diff(ctx, include_recent_commit=include_recent_commit)
    prov["repo_diff"] = "host_attested"
    if subtree_statuses is not None:
        ev["terminal_subtree_statuses"] = [dict(row) for row in subtree_statuses if isinstance(row, dict)]
        prov["terminal_subtree_statuses"] = "host_attested"
    if isinstance(llm_trace, dict):
        traj, omitted = _accept_trajectory(llm_trace.get("tool_calls") or [])
        if traj or omitted:
            ev["tool_trajectory"] = traj
            prov["tool_trajectory"] = "tool_result"
            if omitted:
                ev["tool_trajectory_omitted_leading"] = omitted
        notes = llm_trace.get("reasoning_notes") or []
        if notes:
            ev["reasoning_notes"] = truncate_review_artifact("\n".join(str(n) for n in notes), limit=_ACCEPT_NOTES_CAP)
            prov["reasoning_notes"] = "agent_supplied"
        # v6.54.4 CANDIDATES adjudication: when the agent enumerated candidate
        # interpretations/answers (opt-in latched block), the reviewer sees them
        # and can adjudicate which one the task actually asks for.
        candidates = llm_trace.get("candidate_answers") or []
        if candidates:
            ev["candidate_answers"] = [str(c)[:300] for c in candidates][:8]
            prov["candidate_answers"] = "agent_supplied"
        # v6.78.0 (owner Q20/Q22): host-attested NATIVE retrieval made inside the answering
        # model's own request — counts plus capped URLs, no titles/snippets. Present ONLY when
        # the provider actually searched (native main-loop search is off by default, and the
        # `web_search`/browser TOOLS issue their own calls, which never land here), so its
        # ABSENCE is not a deficiency and never implies a knowledge-only answer — the reviewer
        # rules say exactly that. Never shown to the agent.
        retrieval = llm_trace.get("retrieval") if isinstance(llm_trace.get("retrieval"), dict) else {}
        if retrieval:
            from ouroboros._outcome_receipts import disclosed_list_projection

            # The rules call these "the URLs it fetched", so a bound here must SAY what
            # it left out (BIBLE P1). Same shared projection as the receipt path sets;
            # the accumulator's own omissions (`fold_retrieval_usage` caps per task) are
            # added on, and its full-set hash carried, because the complete URL set lives
            # only in the per-call observability payloads.
            _urls = disclosed_list_projection(
                retrieval.get("urls"), key="urls", limit=_ACCEPT_RETRIEVAL_URLS_MAX,
                bound=_accept_redact_cap,
            )
            _urls["urls_omitted"] += int(retrieval.get("urls_omitted") or 0)
            if str(retrieval.get("urls_identity_sha256") or ""):
                _urls["urls_identity_sha256"] = str(retrieval.get("urls_identity_sha256"))
            ev["retrieval"] = {
                "web_search_requests": int(retrieval.get("web_search_requests") or 0),
                "source_count": int(retrieval.get("source_count") or 0),
                **_urls,
            }
            prov["retrieval"] = "host_attested"
        # v6.71.1: host-attested catalog of the acceptance obligations the host
        # raised (id/item/recommendation/status) so the reviewer can adjudicate the
        # agent's per-obligation dispositions/rebuttals — those arrive separately
        # under `agent_supplied.agent_decision.obligation_dispositions`, joinable by
        # id. Without the obligation TEXT the reviewer saw "the agent rejected
        # ob-XXXX" with no way to know what ob-XXXX asked → could not accept a valid
        # rebuttal → acceptance loops. Host facts only; the disposition reason stays
        # under agent_supplied (clean provenance, BIBLE P3).
        obligations = [o for o in (llm_trace.get("acceptance_obligations") or []) if isinstance(o, dict)]
        # Count cap (v6.71.1): OPEN obligations are ACTIVE BLOCKING STATE — the panel
        # adjudicates them and a clean PASS closes them, so clipping an open row would
        # let the loop close obligations the reviewers never saw (triad r4, P1/P3).
        # Every open row therefore always ships; only HISTORICAL disposed rows are
        # capped (most-recent fill up to _ACCEPT_OBLIGATIONS_MAX total), disclosed.
        # An open set too large to fit fails closed downstream via the packet budget
        # (__immutable_core_overflow__ → DEGRADED), never via silent hiding.
        if len(obligations) > _ACCEPT_OBLIGATIONS_MAX:
            open_rows = [o for o in obligations if obligation_is_pending(o)]
            disposed_rows = [o for o in obligations if not obligation_is_pending(o)]
            fill = max(0, _ACCEPT_OBLIGATIONS_MAX - len(open_rows))
            kept = open_rows + (disposed_rows[-fill:] if fill else [])
            if len(kept) < len(obligations):
                ev.setdefault("omissions_manifest", []).append({
                    "section": "acceptance_obligations",
                    "omitted": len(obligations) - len(kept),
                    "reason": "count_cap_disposed_only",
                })
            obligations = kept
        if obligations:
            ev["acceptance_obligations"] = [
                _accept_obligation_row(o) for o in obligations
            ]
            prov["acceptance_obligations"] = "host_attested"
    if drive_root is not None and task_id:
        arts = _accept_artifact_manifest(drive_root, task_id, _accept_protected_set(ctx))
        if arts:
            ev["artifacts"] = arts
            prov["artifacts"] = "artifact"
    # Set task_type BEFORE budget enforcement so the whole packet stays deterministically
    # bounded — callers must NOT mutate the packet after the builder returns (review round-4).
    if str(task_type).strip():
        ev["task_type"] = str(task_type)
        prov["task_type"] = "host_attested"
    ev["__provenance__"] = prov
    return _accept_enforce_budget(ev)


def collect_review_evidence(
    drive_root: Any,
    *,
    task_id: str = "",
    repo_dir: Any = None,
    max_attempts: int = 3,
    max_runs: int = 3,
    max_obligations: int | None = None,
    max_continuations: int = 3,
) -> Dict[str, Any]:
    from ouroboros.review_state import (
        _LEGACY_CURRENT_REPO_KEY,
        compute_snapshot_hash,
        load_state,
        make_repo_key,
    )
    from ouroboros.task_continuation import list_review_continuations

    drive_root_path = pathlib.Path(drive_root)
    repo_dir_path = pathlib.Path(repo_dir) if repo_dir else None
    repo_key = make_repo_key(repo_dir_path) if repo_dir_path else ""
    snapshot_hash = compute_snapshot_hash(repo_dir_path) if repo_dir_path else ""

    state = load_state(drive_root_path)
    all_runs = list(state.advisory_runs or [])
    all_attempts = list(state.attempts or [])

    if repo_key:
        repo_runs = state.filter_advisory_runs(repo_key=repo_key)
    else:
        repo_runs = all_runs

    if task_id:
        scoped_attempts = state.filter_attempts(task_id=task_id)
    elif repo_key:
        scoped_attempts = state.filter_attempts(repo_key=repo_key)
    else:
        scoped_attempts = all_attempts

    current_run = None
    if snapshot_hash:
        current_run = state.find_by_hash(snapshot_hash, repo_key=repo_key or None)

    open_obligations = state.get_open_obligations(repo_key=repo_key or None)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key or None)
    continuations, corrupt = list_review_continuations(drive_root_path)
    if task_id:
        scoped_continuations = [item for item in continuations if item.task_id == task_id]
    elif repo_key:
        scoped_continuations = [
            item for item in continuations
            if item.repo_key in ("", repo_key, _LEGACY_CURRENT_REPO_KEY)
        ]
    else:
        scoped_continuations = continuations
    scoped_continuations.sort(key=lambda item: str(item.updated_ts or item.created_ts or ""), reverse=True)
    stale_matches_repo = not repo_key or state.last_stale_repo_key in ("", repo_key)

    evidence = {
        "task_id": task_id,
        "repo_key": repo_key,
        "current_repo": {
            "snapshot_hash": snapshot_hash[:12] if snapshot_hash else "",
            "advisory_status": str(getattr(current_run, "status", "") or "missing"),
            "repo_commit_ready": bool(
                current_run is not None
                and current_run.status in ("fresh", "bypassed", "skipped")
                and not open_obligations
                and not open_debts
            ),
            "bypass_reason": str(getattr(current_run, "bypass_reason", "") or ""),
            "stale_reason": str(getattr(state, "last_stale_reason", "") or "") if stale_matches_repo else "",
            "stale_ts": str(getattr(state, "last_stale_from_edit_ts", "") or "") if stale_matches_repo else "",
        },
        "recent_attempts": [_attempt_to_dict(item) for item in (scoped_attempts[-max_attempts:] if max_attempts > 0 else [])],
        "omitted_attempts": max(0, len(scoped_attempts) - max_attempts) if max_attempts > 0 else len(scoped_attempts),
        "recent_advisory_runs": [_run_to_dict(item) for item in (repo_runs[-max_runs:] if max_runs > 0 else [])],
        "omitted_advisory_runs": max(0, len(repo_runs) - max_runs) if max_runs > 0 else len(repo_runs),
        "open_obligations": [_obligation_to_dict(item) for item in (open_obligations[:max_obligations] if max_obligations is not None else open_obligations)],
        "omitted_obligations": max(0, len(open_obligations) - max_obligations) if max_obligations is not None else 0,
        "commit_readiness_debts": [_debt_to_dict(item) for item in open_debts],
        "continuations": [_continuation_to_dict(item) for item in scoped_continuations[:max_continuations]],
        "omitted_continuations": max(0, len(scoped_continuations) - max_continuations),
        "corrupt_continuations": [str(item) for item in corrupt[:3]],
        "omitted_corrupt": max(0, len(corrupt) - 3),
    }
    evidence["has_evidence"] = any([
        evidence["recent_attempts"],
        evidence["recent_advisory_runs"],
        evidence["open_obligations"],
        evidence["commit_readiness_debts"],
        evidence["continuations"],
        evidence["corrupt_continuations"],
        evidence["current_repo"]["advisory_status"] not in ("", "missing"),
        # Omission counters signal truncated evidence even when visible lists are empty
        evidence["omitted_attempts"] > 0,
        evidence["omitted_advisory_runs"] > 0,
        evidence["omitted_obligations"] > 0,
        evidence["omitted_continuations"] > 0,
        evidence["omitted_corrupt"] > 0,
    ])
    return evidence


def format_review_evidence_for_prompt(
    evidence: Dict[str, Any],
    *,
    max_chars: int = 0,
    **_kwargs,
) -> str:
    """Format review evidence as JSON for prompt injection.

    When *max_chars* is 0 (default) the full JSON is returned — no truncation.
    Callers that inject evidence into bounded prompts (summaries, reflections)
    can pass a positive *max_chars* to get an explicit omission note instead
    of silent clipping.
    """
    if not evidence or not evidence.get("has_evidence"):
        return "(no structured review evidence)"
    full = json.dumps(evidence, ensure_ascii=False, indent=2)
    if max_chars > 0 and len(full) > max_chars:
        return full[:max_chars] + f"\n⚠️ OMISSION NOTE: review evidence truncated at {max_chars} chars; original length {len(full)}"
    return full


def build_review_projection(
    drive_root: Any,
    *,
    repo_dir: Any = None,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: int | None = None,
    snapshot_hash_fn: Any = None,
) -> Dict[str, Any]:
    """Build the semantic read-model shared by review_status-style renderers."""
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
            and latest.status in {"fresh", "bypassed", "skipped", "parse_failure", "preflight_blocked", "tests_preflight_blocked"}
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
        "stale_from_edit_ts": (
            state.last_stale_from_edit_ts if state.last_stale_from_edit_ts and stale_matches_repo
            else ("now (hash mismatch)" if hash_mismatch else None)
        ),
        "stale_reason": (
            state.last_stale_reason if stale_matches_repo else ""
        ) or ("Current snapshot hash no longer matches the latest advisory run." if hash_mismatch else None),
        "open_obligations": open_obligations,
        "open_debts": open_debts,
        "repo_commit_ready": bool(effective_is_fresh and not open_obligations and not open_debts),
        "retry_anchor": "commit_readiness_debt" if open_debts else None,
        "advisory_overrides": advisory_overrides,
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
        "filters": projection["filters"],
        "advisory_runs": [_review_status_run_to_dict(run) for run in reversed(projection.get("runs") or [])],
        "attempts": [_review_status_attempt_to_dict(item) for item in reversed(projection.get("attempts") or [])],
        "selected_commit_attempt": _review_status_attempt_payload(selected_attempt),
        "open_obligations": [_review_status_obligation_to_dict(item) for item in open_obligations],
        "open_obligations_count": len(open_obligations),
        "commit_readiness_debts": [_review_status_debt_to_dict(item) for item in open_debts],
        "commit_readiness_debts_count": len(open_debts),
        "repo_commit_ready": projection["repo_commit_ready"],
        "retry_anchor": projection["retry_anchor"],
        "status_summary": _review_status_message(projection),
        "next_step": next_step,
    }
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
        reason_map = {
            "no_advisory": "No fresh advisory review found. Run advisory_review first.",
            "critical_findings": "Reviewers found critical issues. Fix all issues listed, then re-run advisory.",
            "review_quorum": "Not enough review models responded. Retry — usually transient.",
            "parse_failure": "Review models could not produce parseable output. Retry the commit.",
            "infra_failure": "Infrastructure failure. Check block_details.",
            "scope_blocked": "Scope reviewer blocked the commit. Address scope review findings.",
            "preflight": "Preflight check failed. Stage all related files.",
            "revalidation_failed": "The staged diff changed after review. Re-run advisory and review.",
            "fingerprint_unavailable": "The staged diff could not be fingerprinted. Fix git diff and retry.",
            "overlap_guard": "Another reviewed attempt is still active. Wait or expire it before retrying.",
            "attempt_cap_reached": "The same staged diff was review-blocked repeatedly. Change the diff or rebut via review_rebuttal.",
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


def _attempt_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in ("ts", "tool_name", "status", "phase", "block_reason", "scope_model")
    }
    data.update({
        "attempt": int(getattr(item, "attempt", 0) or 0),
        "late_result_pending": bool(getattr(item, "late_result_pending", False)),
        "duration_sec": float(getattr(item, "duration_sec", 0.0) or 0.0),
        "critical_findings": list(getattr(item, "critical_findings", []) or []),
        "advisory_findings": list(getattr(item, "advisory_findings", []) or []),
        "triad_raw_results": list(getattr(item, "triad_raw_results", []) or []),
        "scope_raw_result": dict(getattr(item, "scope_raw_result", {}) or {}),
    })
    for key in ("readiness_warnings", "obligation_ids", "degraded_reasons", "triad_models"):
        data[key] = [str(x) for x in (getattr(item, key, []) or [])]
    return data


_RESPONDED_STATUSES = frozenset({"fresh", "stale"})


def _run_to_dict(item: Any) -> Dict[str, Any]:
    """Serialise AdvisoryRunRecord with responded/skipped/error status summary."""
    valid_items = [entry for entry in list(getattr(item, "items", []) or []) if isinstance(entry, dict)]
    fail_items = [
        {
            "severity": str(entry.get("severity", "") or "advisory"),
            "item": str(entry.get("item", "") or ""),
            "reason": str(entry.get("reason", "") or ""),
        }
        for entry in valid_items
        if str(entry.get("verdict", "")).upper() == "FAIL"
    ]
    total_items = len(valid_items)

    status = str(getattr(item, "status", "") or "")
    bypass_reason = str(getattr(item, "bypass_reason", "") or "")
    raw_result_text = str(getattr(item, "raw_result", "") or "")

    status_summary = status if status in {"bypassed", "skipped", "parse_failure", "error"} else status or "unknown"
    if status in _RESPONDED_STATUSES:
        status_summary = (
            "responded_with_findings" if fail_items
            else "responded_clean" if total_items > 0
            else "responded_empty"
        )

    return {
        "ts": str(getattr(item, "ts", "") or ""),
        "status": status,
        "status_summary": status_summary,
        "repo_key": str(getattr(item, "repo_key", "") or ""),
        "bypass_reason": bypass_reason,
        "snapshot_summary": str(getattr(item, "snapshot_summary", "") or ""),
        "findings": fail_items,
        "total_items": total_items,
        "raw_result_present": bool(raw_result_text),
        "readiness_warnings": [str(x) for x in (getattr(item, "readiness_warnings", []) or [])],
        "prompt_chars": int(getattr(item, "prompt_chars", 0) or 0),
        "model_used": str(getattr(item, "model_used", "") or ""),
        "duration_sec": float(getattr(item, "duration_sec", 0.0) or 0.0),
    }


def _obligation_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "obligation_id": str(getattr(item, "obligation_id", "") or ""),
        "fingerprint": str(getattr(item, "fingerprint", "") or ""),
        "item": str(getattr(item, "item", "") or ""),
        "severity": str(getattr(item, "severity", "") or ""),
        "reason": str(getattr(item, "reason", "") or ""),
        "status": str(getattr(item, "status", "") or ""),
        "created_ts": str(getattr(item, "created_ts", "") or ""),
        "updated_ts": str(getattr(item, "updated_ts", "") or ""),
    }


def _continuation_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in ("task_id", "source", "stage", "tool_name", "block_reason", "updated_ts")
    }
    data.update({
        "attempt": int(getattr(item, "attempt", 0) or 0),
        "critical_findings": list(getattr(item, "critical_findings", []) or []),
        "advisory_findings": list(getattr(item, "advisory_findings", []) or []),
        "readiness_warnings": [str(x) for x in (getattr(item, "readiness_warnings", []) or [])],
    })
    return data


def _debt_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in ("debt_id", "category", "title", "summary", "status", "severity", "source", "repo_key", "updated_at")
    }
    data["source_obligation_ids"] = [str(x) for x in (getattr(item, "source_obligation_ids", []) or [])]
    data["evidence"] = [str(x) for x in (getattr(item, "evidence", []) or [])]
    return data
