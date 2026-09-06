"""Structured review-evidence collection for summaries, reflections, and UX."""

from __future__ import annotations

import json
import hashlib
import logging
import pathlib
from typing import Any, Dict, List

from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.utils import truncate_review_artifact, truncate_within_limit  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
log = logging.getLogger(__name__)


# ── Process-aware task-acceptance evidence (v6.51.0 idea-2) ───────────────────
# The acceptance reviewer audits BOTH the final outcome AND the solving PROCESS
# (wrong tool / wrong direction / finalized over a red check). Typed sections with
# explicit PROVENANCE tags; full artifacts/trace stay durable off-axis — the prompt
# gets bounded, redacted, DISCLOSED-truncated projections (Bible P1/P3/P12/P7).
# The whole-packet ceiling below is a FLOOR, not the ceiling: the real ceiling is
# resolved per task from the review quorum's calibrated input windows
# (``acceptance_packet_budget_chars``), so a wide panel reads the packet its
# models can actually hold and a narrow one is never handed a prompt that a 400
# would reject after the money is spent (P1/P8).
# Evidence-parity (v6.71.1): the acceptance reviewer's per-result cap tracks the
# ACTOR's own default tool-result window (SSOT: tool_capabilities.DEFAULT_TOOL_RESULT_LIMIT),
# so a decider never adjudicates less of a tool result than the agent saw. The old
# hidden 700-char trace cap (loop_tool_execution) starved this and produced false
# "not shown in trace" verdicts → acceptance loops (BIBLE P1 observability / P3).


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

# Commit-review status renderers: extracted to the ``review_status_projection``
# leaf (module size gate); re-exported here so every historical import site keeps
# resolving. The private ``_review_status_*`` renderers are part of that surface
# — they are imported by name from this module today.
from ouroboros.review_status_projection import (
    build_review_projection as build_review_projection,
)
from ouroboros.review_status_projection import (
    build_review_status_payload as build_review_status_payload,
)
from ouroboros.review_status_projection import (
    _run_failure_reason as _run_failure_reason,
)
from ouroboros.review_status_projection import (
    _review_status_run_to_dict as _review_status_run_to_dict,
)
from ouroboros.review_status_projection import (
    _review_status_attempt_payload as _review_status_attempt_payload,
)
from ouroboros.review_status_projection import (
    _review_status_attempt_to_dict as _review_status_attempt_to_dict,
)
from ouroboros.review_status_projection import (
    _review_status_actor_summary as _review_status_actor_summary,
)
from ouroboros.review_status_projection import (
    _review_status_obligation_to_dict as _review_status_obligation_to_dict,
)
from ouroboros.review_status_projection import (
    _review_status_debt_to_dict as _review_status_debt_to_dict,
)
from ouroboros.review_status_projection import (
    _review_status_message as _review_status_message,
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
    undispositioned_children: List[Dict[str, Any]] | None = None,
    acceptance_dialogue_history: List[Dict[str, Any]] | None = None,
    budget_chars: int = 0,
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
    from ouroboros.outcomes import read_context_verification_receipts

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
    claims, claims_source, plan_exhibit = _accept_effective_claims(
        ctx, contract, drive_root, task_id,
    )
    if claims_source == "plan_review":
        contract = {**contract, "acceptance_claims": claims}
    receipts = read_context_verification_receipts(ctx, task_id, fallback_root=drive_root) if task_id else []
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
        if plan_exhibit:
            ev["plan_claims_exhibit"] = redact_projection(plan_exhibit).value
            prov["plan_claims_exhibit"] = "host_attested"
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

        # The writer and the outcome consumer resolve the canonical results root
        # first; on a split-root install reading the execution drive made the
        # whole section silently vanish from the packet.
        mutation_projection = load_mutation_evidence_projection(
            getattr(ctx, "budget_drive_root", None) or drive_root, task_id,
        )
        if mutation_projection:
            ev["mutation_attribution"] = mutation_projection
            prov["mutation_attribution"] = "host_attested"
        from ouroboros.delegate_evidence import (
            acceptance_capability_deltas,
            acceptance_patch_dispositions,
            acceptance_substrate_facts,
        )

        if delta_aggregate := acceptance_capability_deltas(drive_root, task_id, root_task_id):
            ev["capability_deltas"] = redact_projection(delta_aggregate).value
            prov["capability_deltas"] = "host_attested"
        if substrate_facts := acceptance_substrate_facts(ctx, task_id):
            ev["substrate_execution"] = redact_projection(substrate_facts).value
            prov["substrate_execution"] = "host_attested"
        # D-trace (owner 4=A): the parent's patch apply/reject attestations —
        # visibility for the panel, never a gate on apply. Absence = no
        # disposition recorded, not "reviewed clean".
        if patch_dispositions := acceptance_patch_dispositions(drive_root, task_id):
            ev["delegated_patch_dispositions"] = redact_projection(patch_dispositions).value
            prov["delegated_patch_dispositions"] = "host_attested"
        # The lifecycle (review status, readiness, enablement) of every skill the
        # task touched — the same VISIBILITY-ONLY charter as substrate_execution.
        from ouroboros.skill_readiness import acceptance_skill_lifecycle

        lifecycle_root = getattr(ctx, "budget_drive_root", None) or drive_root
        skill_history_coverage: Dict[str, Any] = {}
        if lifecycle := acceptance_skill_lifecycle(
            lifecycle_root, llm_trace or {}, root_task_id,
            task_started_at=str(meta.get("started_at") or meta.get("created_at") or ""),
            history_coverage=skill_history_coverage,
        ):
            ev["skill_lifecycle"] = redact_projection(lifecycle).value
            prov["skill_lifecycle"] = "host_attested"
        if skill_history_coverage:
            ev["skill_lifecycle_history_coverage"] = skill_history_coverage
            prov["skill_lifecycle_history_coverage"] = "host_attested"
            ev["skill_lifecycle_complete"] = bool(skill_history_coverage.get("complete"))
    repo_diff = collect_turn_diff(ctx, include_recent_commit=include_recent_commit)
    diff_meta: Dict[str, Any] = {}
    if "OMISSION NOTE: truncated at " in str(repo_diff or "") or "... (truncated from " in str(repo_diff or ""):
        from ouroboros.artifacts import materialize_repo_diff_evidence
        repo_getter = getattr(ctx, "active_repo_dir", None)
        repo_dir = repo_getter() if callable(repo_getter) else repo_getter or getattr(ctx, "repo_dir", None)
        exact, diff_meta = materialize_repo_diff_evidence(
            repo_dir, drive_root, task_id, include_recent_commit=include_recent_commit,
        )
        if diff_meta.get("complete"):
            repo_diff = exact
    ev["repo_diff"] = repo_diff
    prov["repo_diff"] = "host_attested"
    if diff_meta.get("source_ref"):
        ev["repo_diff_source_ref"] = redact_projection(diff_meta["source_ref"]).value
        prov["repo_diff_source_ref"] = "host_attested"
    partial_sources: List[Dict[str, Any]] = [dict(diff_meta["issue"])] if diff_meta.get("issue") else []
    if subtree_statuses is not None:
        ev["terminal_subtree_statuses"] = [dict(row) for row in subtree_statuses if isinstance(row, dict)]
        from ouroboros.depth_evidence import build_depth_summary
        ev["depth_summary"] = build_depth_summary(contract, ev["terminal_subtree_statuses"])
        prov["terminal_subtree_statuses"] = prov["depth_summary"] = "host_attested"
    if isinstance(llm_trace, dict):
        traj, omitted, unresolved = _accept_trajectory(
            llm_trace.get("tool_calls") or [], drive_root=drive_root, task_id=task_id,
        )
        if traj or omitted:
            ev["tool_trajectory"] = traj
            prov["tool_trajectory"] = "tool_result"
            from ouroboros.artifacts import persist_tool_trajectory_source
            if source_ref := persist_tool_trajectory_source(drive_root, task_id, llm_trace.get("tool_calls")):
                ev["tool_trajectory_source_ref"] = source_ref
            if omitted:
                ev["tool_trajectory_omitted_leading"] = omitted
                ev["tool_trajectory_complete"] = False
        if unresolved:
            partial_sources.extend(unresolved)
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
            if any(isinstance(row, dict) and row.get("name") == "…" for row in arts):
                partial_sources.append({"tool": "artifact_manifest", "status": "source_unavailable", "reason": "artifact_manifest_truncated_without_exact_range", "source_ref": {}})
    if ev.get("skill_lifecycle_complete") is False:
        coverage = ev.get("skill_lifecycle_history_coverage") or {}
        partial_sources.append({"tool": "skill_lifecycle", "status": "not_materialized_for_reviewer",
                                "reason": "bounded_history_projection", "source_ref": coverage.get("source_ref") or {}})
    if partial_sources:
        ev["__unresolved_partial_artifacts__"] = partial_sources
    # Set task_type BEFORE budget enforcement so the whole packet stays deterministically
    # bounded — callers must NOT mutate the packet after the builder returns (review round-4).
    if str(task_type).strip():
        ev["task_type"] = str(task_type)
        prov["task_type"] = "host_attested"
    if isinstance(undispositioned_children, list) and undispositioned_children:
        ev["undispositioned_children"] = undispositioned_children
    ev["__provenance__"] = prov
    # The host facts own identity; the bounded packet is their presentation.
    # Keep exact receipt changes visible even beyond an exhibit's text cap.
    ev[ACCEPTANCE_SOURCE_REVISION_KEY] = task_acceptance_evidence_revision({
        **ev, "verification_receipts_source": redact_projection(receipts).value,
    })
    if isinstance(acceptance_dialogue_history, list) and acceptance_dialogue_history:
        ev[UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY] = acceptance_dialogue_history
    return _accept_enforce_budget(ev, budget=budget_chars)


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
        advisory_commit_ready,
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
            "repo_commit_ready": advisory_commit_ready(
                current_run is not None and current_run.status in ("fresh", "bypassed", "skipped"),
                open_obligations, open_debts,
                matching_run=current_run if getattr(current_run, "repo_key", None) == repo_key and repo_key else None,
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


_ACCEPTANCE_PANEL_ROW_KEYS = (
    "panel_id", "surface", "authority", "aggregate_signal", "transport_status",
    "parse_status", "quorum", "superseded",
)


def _acceptance_panel_prompt_row(panel: Dict[str, Any]) -> Dict[str, Any]:
    row = {key: panel.get(key) for key in _ACCEPTANCE_PANEL_ROW_KEYS if key in panel}
    reason = str(panel.get("reason") or "")
    if reason:
        limit = 300
        row["reason"] = truncate_review_artifact(reason, limit=limit)
        row["reason_omitted_chars"] = max(0, len(reason) - limit)
        refs = [
            actor.get("response_ref") for actor in (panel.get("actors") or [])
            if isinstance(actor, dict) and actor.get("response_ref")
        ]
        if refs:
            row["response_refs"] = refs
    return row


def format_review_evidence_for_prompt(
    evidence: Dict[str, Any],
    *,
    max_chars: int = 0,
    acceptance_panels: Any = None,
    **_kwargs,
) -> str:
    """Format review evidence as JSON for prompt injection.

    When *max_chars* is 0 (default) the full JSON is returned — no truncation.
    Callers that inject evidence into bounded prompts (summaries, reflections)
    can pass a positive *max_chars* to get an explicit omission note instead
    of silent clipping.

    ``acceptance_panels`` leads with the task's OWN acceptance-panel projection.
    The commit/advisory lens knows nothing about it, so its absence statement
    names the lens it describes rather than claiming the task bought no review.
    """
    rows = [
        _acceptance_panel_prompt_row(panel)
        for panel in (acceptance_panels if isinstance(acceptance_panels, list) else [])
        if isinstance(panel, dict)
    ]
    task_id = str(evidence.get("task_id") or _kwargs.get("task_id") or "")
    source_ref: Dict[str, Any] = (
        {"kind": "task_result", "reader": "get_task_result", "task_id": task_id}
        if task_id else {}
    )
    if not source_ref:
        source_ref = next((
            ref for row in rows for ref in (row.get("response_refs") or [])
            if isinstance(ref, dict) and ref
        ), {})
    sections: List[str] = []
    if rows:
        from ouroboros._outcome_receipts import disclosed_list_projection

        keep = len(rows)
        projection = disclosed_list_projection(
            rows, key="records", limit=keep, item=lambda row: row,
        )
        while source_ref and max_chars > 0 and keep > 1:
            candidate = "TASK ACCEPTANCE PANELS:\n" + json.dumps(
                projection, ensure_ascii=False, indent=2,
            )
            if len(candidate) <= max_chars:
                break
            keep -= 1
            projection = disclosed_list_projection(
                rows, key="records", limit=keep, item=lambda row: row,
            )
        if projection["records_omitted"]:
            projection["omission_note"] = "whole trailing panel records omitted"
            projection["omission_source_ref"] = source_ref
        sections.append(
            "TASK ACCEPTANCE PANELS:\n"
            + json.dumps(projection, ensure_ascii=False, indent=2)
        )
    if evidence and evidence.get("has_evidence"):
        rendered_evidence = json.dumps(evidence, ensure_ascii=False, indent=2)
        prefix_chars = len(sections[0]) + 2 if sections else 0
        limit = max_chars - prefix_chars if max_chars > 0 else 0
        if source_ref and max_chars > 0 and len(rendered_evidence) > max(1, limit):
            rendered_evidence = truncate_review_artifact(
                rendered_evidence, limit=max(1, limit),
            )
            rendered_evidence += (
                f"\n⚠️ OMISSION SOURCE: review evidence truncated at {max_chars} chars; "
                f"canonical source_ref={json.dumps(source_ref, ensure_ascii=False)}"
            )
        sections.append(rendered_evidence)
    if not sections:
        return "(no commit/advisory review evidence recorded for this task)"
    return "\n\n".join(sections)


def _attempt_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in (
            "ts", "tool_name", "status", "phase", "block_reason", "scope_model",
            # Max-Review-Cycles accounting facts (Q16 auditability).
            "block_class", "rebuttal_sha256", "review_contract_fingerprint",
            "root_task_id",
        )
    }
    data.update({
        "paid": bool(getattr(item, "paid", False)),
        "raw_stripped": bool(getattr(item, "raw_stripped", False)),
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


# v7next F2.3a (D06): moved spans live in their owner leaves; re-exported
# here so this facade stays the single import surface for callers and tests.
from ouroboros.review_evidence_sections import (  # noqa: E402, F401 -- intentional public re-exports
    ACCEPTANCE_SOURCE_REVISION_KEY,
    ACCEPTANCE_PROMPT_OVERHEAD_CHARS,
    AcceptancePacketBudget,
    _ACCEPT_ARGS_CAP,
    _ACCEPT_ARTIFACT_PREVIEW_CAP,
    _ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES,
    _ACCEPT_DENSE_CHARS_PER_TOKEN,
    _ACCEPT_NOTES_CAP,
    _ACCEPT_OBLIGATIONS_MAX,
    _ACCEPT_RESULT_CAP,
    _ACCEPT_RETRIEVAL_URLS_MAX,
    _ACCEPT_TOTAL_BUDGET,
    _ACCEPT_TRAJECTORY_MAX_CALLS,
    acceptance_packet_budget_chars,
    _accept_artifact_manifest,
    _accept_claim_support_refs,
    _accept_effective_claims,
    _accept_enforce_budget,
    UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY,
    UNHASHED_EVIDENCE_KEYS,
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
