"""The commit preflight (the author's named row, decision 3A) and the free deterministic
checks every commit runs before review, durable commit-attempt recording, and the
commit-side Max-Review-Cycles machinery (block classification, the per-root-task
paid-cycle ceiling, and the review-contract fingerprint)."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence

from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED, review_max_cycles
from ouroboros.review_owner_custody import stamp_paid_review_owner
from ouroboros.review_state import (
    _attempt_has_active_review_custody,
    infer_review_phase,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_helpers import cyber_review_authority

log = logging.getLogger(__name__)


def _current_review_tool_name(ctx: ToolContext) -> str:
    return str(getattr(ctx, "_current_review_tool_name", "") or "commit_reviewed")


def _normalize_advisory_entries(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in list(items or []):
        if isinstance(item, dict):
            normalized.append(item)
        elif item:
            normalized.append({"reason": str(item), "severity": "advisory"})
    return normalized


def _list_or_default(items: Optional[List[Any]], fallback: List[Any]) -> List[Any]:
    if items is None:
        return list(fallback)
    return list(items)


def _continuation_source(status: str, *, late_result_pending: bool, stopped: bool = False) -> str:
    if status == "blocked":
        return "owner_stopped" if stopped else "blocked_review"
    if late_result_pending:
        return "late_result_pending"
    if status == "failed":
        return "review_failure"
    return ""


def _attempt_accepts_reviewing_update(existing: Any) -> bool:
    return existing is not None and _attempt_has_active_review_custody(existing)


# Max Review Cycles semantics on the commit gate (owner Q12/Q16/Q22/Q23): the shared
# OUROBOROS_REVIEW_MAX_CYCLES knob (``review_max_cycles()``; ``None`` = unlimited) bounds PAID
# review-wave cycles per ROOT task (the whole task tree shares one ceiling; a manual session is
# its own task; a follow-up task starts a fresh one). The ceiling counts MONEY: every attempt
# that physically dispatched a wave (``paid`` recorded at dispatch) counts regardless of how it
# terminated — only UNDISPATCHED attempts (preflight refusals, assembly failures, free replays)
# are outside the count. Exhaustion is a free replay plus ``emit_review_cycles_exhausted``: the
# prior outcome returns for an informed author choice; explicit continuation buys no reviewer and
# does not manufacture approval. The free identical-diff refusal left with Blocking; its reason
# stays for the rows that carry it.
IDENTICAL_DIFF_BLOCK_REASON = "identical_diff_refused"
_LEGACY_CAP_BLOCK_REASON = "attempt_cap_reached"  # pre-Q16 refusal rows
_REFUSAL_BLOCK_REASONS = frozenset({
    IDENTICAL_DIFF_BLOCK_REASON,
    REASON_REVIEW_CYCLES_EXHAUSTED,
    _LEGACY_CAP_BLOCK_REASON,
})

BLOCK_CLASS_VERDICT = "verdict"
BLOCK_CLASS_INFRA = "infra"

# Triad block reasons that ARE reviewer verdicts; everything else recorded at
# phase=blocking_review is an infrastructure fact about the gate. Post-review
# failure phases (post_commit_tests, commit_binding, tag_binding) need no set
# of their own: their paid dispatch stays counted.
_TRIAD_VERDICT_BLOCK_REASONS = frozenset({"critical_findings"})
# Failed-status phases that are pre/around-review infrastructure facts.
_INFRA_FAILURE_PHASES = frozenset({"infra", "expired"})


def _scope_actor_rows(scope_raw_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = scope_raw_result if isinstance(scope_raw_result, dict) else {}
    rows = [row for row in (raw.get("raw_results") or []) if isinstance(row, dict)]
    return rows or ([raw] if raw else [])


def _scope_verdict_blocked(scope_blocked: bool, scope_raw_result: Optional[Dict[str, Any]]) -> bool:
    """A scope-side VERDICT block = an authoritative (``responded``) actor row
    carrying critical findings while the scope aggregate blocked. Sub-floor,
    fit-overflow, transport, parse and quorum blocks all arrive under
    non-``responded`` statuses — they are infra facts, not verdicts."""
    if not scope_blocked:
        return False
    for row in _scope_actor_rows(scope_raw_result):
        if str(row.get("status") or "") == "responded" and (row.get("critical_findings") or []):
            return True
    return False


def classify_review_block(
    *,
    triad_blocked: bool,
    triad_block_reason: str,
    scope_blocked: bool,
    scope_raw_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Type one blocked review outcome at record time: ``verdict`` when ANY
    side delivered genuine reviewer findings, ``infra`` otherwise."""
    if triad_blocked and str(triad_block_reason or "") in _TRIAD_VERDICT_BLOCK_REASONS:
        return BLOCK_CLASS_VERDICT
    if _scope_verdict_blocked(scope_blocked, scope_raw_result):
        return BLOCK_CLASS_VERDICT
    return BLOCK_CLASS_INFRA


def attempt_block_class(item: Any) -> str:
    """The typed class of one ledger row: the recorded ``block_class`` when
    present, else a conservative legacy inference from the recorded reason.
    Non-review rows (preflight facts, refusal records) stay ``""``."""
    recorded = str(getattr(item, "block_class", "") or "")
    if recorded:
        return recorded
    if str(getattr(item, "status", "") or "") != "blocked":
        return ""
    if str(getattr(item, "block_reason", "") or "") in _REFUSAL_BLOCK_REASONS:
        return ""
    if str(getattr(item, "phase", "") or "") == "revalidation":
        # Post-review revalidation blocks (fingerprint drift, fingerprint
        # unavailable, review_subject_binding_mismatch) are facts about the
        # GATE, never reviewer verdicts, while their dispatched wave
        # (paid=True on the merged row) still counts toward the ceiling.
        return BLOCK_CLASS_INFRA
    if str(getattr(item, "phase", "") or "") != "blocking_review":
        return ""  # preflight/advisory-gate rows are neither verdict nor infra
    reason = str(getattr(item, "block_reason", "") or "")
    if reason in _TRIAD_VERDICT_BLOCK_REASONS:
        return BLOCK_CLASS_VERDICT
    if reason == "scope_blocked" and _scope_verdict_blocked(
        True, getattr(item, "scope_raw_result", None)
    ):
        return BLOCK_CLASS_VERDICT
    return BLOCK_CLASS_INFRA


def compute_rebuttal_sha256(review_rebuttal: Any) -> str:
    """Content identity of a rebuttal; "" when no rebuttal was supplied."""
    text = str(review_rebuttal or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def resolve_root_task_id(ctx: ToolContext) -> str:
    """The root of the current task tree (Q23: one paid-cycle ceiling per
    tree); a task with no recorded root is its own root. "" = unknown.

    DELIBERATE TRADEOFF (adversarial wave, machine-3): ``origin_root_task_id``
    — the follow-up chain marker — is NOT honored here, so a scheduled
    follow-up task is a FRESH root with its own ceiling; the cost is that a
    follow-up can buy new paid cycles for the same goal."""
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        metadata.get("root_task_id")
        or getattr(ctx, "root_task_id", "")
        or getattr(ctx, "task_id", "")
        or ""
    )


def commit_review_contract_fingerprint(own: Sequence[str] = (), *, packet: Optional[Sequence[str]] = None,
                                       retrieving: Sequence[str] = (), layer: str = "body",
                                       surface: str = "multi_model_review", shape: str = "two_part") -> str:
    """Identity of the commit gate's live review contract (Q22): the digest of
    ``commit_review_contract_material``. A changed fingerprint lapses
    free-refusal/replay authority (a new paid review is allowed and refusals
    never quote across the change). Fail-open "" — an unknown contract never
    matches, so nothing is refused on it."""
    try:
        material = commit_review_contract_material(own, packet=packet, retrieving=retrieving, layer=layer,
                                                   surface=surface, shape=shape)
        return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    except Exception:
        log.debug("commit review contract fingerprint unavailable (fail-open)", exc_info=True)
        return ""


def commit_review_contract_material(own: Sequence[str] = (), *, packet: Optional[Sequence[str]] = None,
                                    retrieving: Sequence[str] = (), layer: str = "body",
                                    surface: str = "multi_model_review", shape: str = "two_part") -> dict:
    """What the review contract is: the one wave's seat roster + routes, the
    ``parts`` each seat is asked (one brief, two parts), the recorded authority, the shipped
    prompt contract (contract A, contract B, the required-source policy version)
    and the text every seat's request carries of Ouroboros's own
    (``review_admission.seat_brief_material``: the surface's words, the layer's
    navigation, the seated deliveries' wrappers), never a guide's body or the
    request's data (docs/development 05). The per-row efforts below are the
    RESOLVED efforts (``row_effort`` uses a compound route's encoded effort before
    the configured surface default). Changing a global effort therefore lapses
    only rows that actually inherit it (synthesis F4 — pinned by test)."""
    from ouroboros.review_records import RECORDED_REVIEW_AUTHORITY
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools.review_admission import seat_brief_material, seat_vectors
    from ouroboros.tools.review_helpers import CRITICAL_FINDING_CALIBRATION, REVIEW_PREAMBLE
    from ouroboros.tools.scope_required_sources import SCOPE_REQUIRED_SOURCES_POLICY
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT, REVIEW_TWO_PART_OBJECT_CONTRACT

    row_plan = seat_vectors(commit_triad_delivery())
    rows = [
        [
            str(model),
            str(getattr(route, "value", route) or ""),
            str(effort or ""),
            str(target or ""),
            str(profile or ""),
            str(slot_id or ""),
        ]
        for model, route, effort, target, profile, slot_id in zip(
            row_plan["models"], row_plan["routes"], row_plan["efforts"],
            row_plan["session_targets"], row_plan["session_profiles"],
            row_plan["slot_ids"],
        )
    ]
    # Actor binding is contract identity: a configured-subagent reference
    # changes the row's DELIVERY (native retrieval vs packet), so replay/
    # refusal authority must lapse when it changes. The column is added
    # only when some row carries one, so untouched legacy configs keep
    # their exact historical bytes (conservative in the paid direction
    # only where the contract actually changed).
    actor_ids = [str(a or "") for a in (row_plan.get("subagent_ids") or [])]
    if any(actor_ids):
        for row, actor in zip(rows, actor_ids):
            row.append(actor)
    # A direct api row saved as native delivery (#1334) is the same kind of
    # contract change with no actor id to carry it; the column appears only
    # when such a row exists, so every other panel keeps its exact bytes.
    native_direct = [bool(flag) and not actor and str(getattr(route, "value", route) or "") == "api_chat"
                     for flag, actor, route in zip(row_plan.get("retrieves") or [], actor_ids
                                                   or [""] * len(rows), row_plan["routes"])]
    if any(native_direct):
        for row, native in zip(rows, native_direct):
            row.append("native_retrieval" if native else "")
    # A session seat's writable technical access (#1609) changes what it can
    # check, as in the skill/plan fingerprints and the attempt key. The column
    # appears only when some row is writable, so read-only and legacy panels
    # keep their exact bytes.
    writable = ["" if str(access or "readonly") == "readonly" else str(access)
                for access in (row_plan.get("session_accesses") or [])]
    if any(writable):
        for row, access in zip(rows, writable):
            row.append(access)
    if any(row_plan.get("additional") or []):
        for row, extra in zip(rows, row_plan["additional"]):
            row.append({"additional": bool(extra)})
    # What each seat is ASKED is contract identity: a seat that gains or
    # loses the coupling question is a different review.
    parts = {str(slot_id or ""): list(seat_parts) for slot_id, seat_parts in zip(row_plan["slot_ids"], row_plan["parts"])}
    return {"rows": rows, "parts": parts, "enforcement": RECORDED_REVIEW_AUTHORITY,
            "prompt_contract": [REVIEW_PREAMBLE, CRITICAL_FINDING_CALIBRATION, REVIEW_JSON_ARRAY_CONTRACT,
                                REVIEW_TWO_PART_OBJECT_CONTRACT, SCOPE_REQUIRED_SOURCES_POLICY],
            "brief": seat_brief_material(row_plan, layer=layer, surface=surface, shape=shape, own=own, packet=packet,
                                         retrieving=retrieving)}


def count_paid_review_cycles(ctx: ToolContext, *, root_task_id: str) -> int:
    """Paid triad/scope cycles already spent by this root task on this
    (repo, tool) gate, derived from the existing attempt ledger (P7 — no new
    counter file). The ceiling counts MONEY (machine-5): every attempt that
    physically dispatched a wave (``paid`` recorded at dispatch) counts,
    whatever its terminal — a dispatched-then-crashed or quorum-failed wave
    still spent reviewer money. Only UNDISPATCHED attempts (free refusals,
    replays, preflight/assembly failures — all ``paid=False``) stay outside
    the count; that is the whole "infra retries freely" carve-out."""
    root = str(root_task_id or "")
    if not root:
        return 0
    from ouroboros.review_state import load_state, make_repo_key

    tool = _current_review_tool_name(ctx)
    if tool in ("review_change", "review_result", "review_skill"):  # review_change waves: canonical custody
        from ouroboros.tools.review_change_custody import paid_repo_key, review_state_root

        state = load_state(review_state_root(ctx))
        owner = paid_repo_key(ctx, ctx.repo_dir)
        # Filter unrelated history before filesystem provenance reads; equal
        # physical keys share one resolution within this observation only.
        owners = {str(pathlib.Path(ctx.repo_dir).resolve()): owner}
        paid = 0
        for item in state.filter_attempts(tool_name=tool):
            if not item.paid or item.root_task_id != root:
                continue
            if tool in ("review_result", "review_skill"):  # a root's allowance, wherever the material lives
                paid += 1
                continue
            key = item.review_budget_repo_key or owners.get(item.repo_key)
            if key is None:
                key = owners[item.repo_key] = paid_repo_key(ctx, item.repo_key)
            paid += key == owner
        return paid
    else:
        state = load_state(pathlib.Path(ctx.drive_root))
        attempts = state.filter_attempts(repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)), tool_name=tool)
    return sum(
        1
        for item in attempts
        if bool(getattr(item, "paid", False))
        and str(getattr(item, "root_task_id", "") or "") == root
    )


def check_review_cycles_ceiling(
    ctx: ToolContext, *, root_task_id: str
) -> Optional[Dict[str, Any]]:
    """``None`` allows a paid dispatch; otherwise typed exhaustion facts
    (message/cycles_paid/cap) for the per-root-task paid-cycle ceiling
    (``review_max_cycles()``; ``None``/unknown root = unlimited). Fail-open on
    ledger errors — a cost guard, not a safety gate. DISCLOSED RESIDUAL
    (skill-5): the check is read-at-gate-time with no reservation, so
    concurrent dispatches sharing one root can each read ``paid < cap`` and
    overshoot by the concurrency width; the write-ahead paid stamp at first
    physical dispatch narrows but does not close that window."""
    cap = review_max_cycles()
    root = str(root_task_id or "")
    if cap is None or not root:
        return None
    try:
        paid = count_paid_review_cycles(ctx, root_task_id=root)
    except Exception:
        log.debug("paid review-cycle count failed (fail-open)", exc_info=True)
        return None
    if paid < cap:
        return None
    message = (
        f"⚠️ REVIEW_CYCLES_EXHAUSTED: this task tree (root {root}) already spent "
        f"{paid} of {cap} paid review wave(s) "
        "(OUROBOROS_REVIEW_MAX_CYCLES). Refusing to buy another review.\n"
        "The paid review allowance is spent. Decide whether what was reviewed is enough: finish, "
        "or land with a plain statement of what was not reviewed."
    )
    return {"message": message, "cycles_paid": paid, "cap": cap}


def resolve_commit_review_reference(ctx: ToolContext, reference: Any, *, state: Any = None) -> Any:
    """Resolve the exact prior outcome returned by this task's (or a recorded Continue predecessor's) commit surface."""
    from ouroboros.review_state import _load_state_unlocked, make_repo_key
    from ouroboros.tools.review_collection import continued_review_author

    if not isinstance(reference, dict) or reference.get("surface") != "commit":
        raise ValueError("review_reference must name a returned commit review")
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    locked = state is not None  # under the state lock: nothing is repaired or read; lineage is the source bound outside
    task, bound = reference.get("task_id"), getattr(getattr(ctx, "_author_commit_source", None), "task_id", None)
    if reference.get("repo_key") != repo_key or not task or not (task in {str(getattr(ctx, "task_id", "") or ""), bound}
                                                                   if locked else continued_review_author(ctx, task)):
        raise ValueError("review_reference belongs to another task or repository")
    if reference.get("tool_name") not in {"commit_reviewed", "vcs_commit_reviewed"} or type(reference.get("attempt")) is not int:
        raise ValueError("review_reference has no exact commit attempt")
    from ouroboros.tools.review_change_custody import commit_reference_source, stripped_source_bound
    state = state if locked else _load_state_unlocked(pathlib.Path(ctx.drive_root), strict_attempt_authority=True)
    source = commit_reference_source(ctx, reference, state, state.latest_attempt_for(  # an evicted free row: its record
        repo_key=repo_key, task_id=reference["task_id"], tool_name=reference["tool_name"], attempt=reference["attempt"]), locked=locked)
    if (source is None or source.phase not in {"review_only", "late_wait"}
            or not source.pre_review_fingerprint or source.pre_review_fingerprint != reference.get("pre_review_fingerprint")
            or (source.raw_stripped and not stripped_source_bound(ctx, source, locked=locked))):
        raise ValueError("the returned review outcome is missing or no longer has its exact evidence")
    from ouroboros.review_records import review_outcome_received
    if not cyber_review_authority() and not review_outcome_received(  # outside Cyber Pro: feedback first
            [*source.triad_raw_results, source.scope_raw_result], findings=[*source.critical_findings, *source.advisory_findings],
            terminal=source.phase == "review_only" and source.status == "reviewed"):
        from ouroboros.tools.review_collection import ContinuationWaits
        raise ContinuationWaits("author continuation needs received feedback or a terminal unavailable outcome; reviewers are still running")
    return source


def _record_commit_attempt(
    ctx: ToolContext,
    commit_message: Any = None,
    status: Optional[str] = None,
    **legacy_kwargs: Any,
) -> None:
    """Record a commit attempt; supports positional or keyword commit_message/status."""
    strict = bool(legacy_kwargs.pop("_strict", False))
    bind_owner = bool(legacy_kwargs.pop("_bind_owner", False))  # a reserved roster this process owns, paid or not
    if commit_message is not None:
        legacy_kwargs.setdefault("commit_message", commit_message)
    if status is not None:
        legacy_kwargs.setdefault("status", status)
    if "commit_message" not in legacy_kwargs:
        raise TypeError("_record_commit_attempt: commit_message is required")
    if "status" not in legacy_kwargs:
        raise TypeError("_record_commit_attempt: status is required")

    def _req(name: str, default: Any = "") -> Any:
        return legacy_kwargs.get(name, default)

    try:
        from ouroboros.review_state import (
            CommitAttemptRecord,
            make_repo_key,
            update_state,
            _utc_now,
        )
        commit_message = _req("commit_message")
        status = _req("status")
        block_reason = _req("block_reason")
        block_details = _req("block_details")
        duration_sec = _req("duration_sec", 0.0)
        snapshot_hash = _req("snapshot_hash")
        critical_findings = _req("critical_findings", None)
        advisory_findings = _req("advisory_findings", None)
        readiness_warnings = _req("readiness_warnings", None)
        late_result_pending = _req("late_result_pending", False)
        phase = _req("phase", None)
        pre_review_fingerprint = _req("pre_review_fingerprint")
        post_review_fingerprint = _req("post_review_fingerprint")
        fingerprint_status = _req("fingerprint_status")
        degraded_reasons = _req("degraded_reasons", None)
        triad_models = _req("triad_models", None)
        scope_model = _req("scope_model")
        triad_raw_results = _req("triad_raw_results", None)
        scope_raw_result = _req("scope_raw_result", None)
        # Ordinary advisory continuation is not an author finish.  Only an
        # explicit caller-supplied record is persisted here; the review
        # findings and advisory override remain the evidence for an unmarked
        # successful commit.
        author_disposition = _req("author_disposition", None)
        block_class = _req("block_class")
        rebuttal_sha256 = _req("rebuttal_sha256")
        paid = _req("paid", False)
        review_contract_fingerprint = _req("review_contract_fingerprint")
        review_retry_key = _req("review_retry_key")
        review_record_id = str(_req("review_record_id") or getattr(ctx, "_current_review_record_id", "") or "")
        root_task_id = resolve_root_task_id(ctx)
        dr = pathlib.Path(ctx.drive_root)
        repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
        tool_name = _current_review_tool_name(ctx)
        task_id = str(getattr(ctx, "task_id", "") or "")

        _findings_for_attempt = critical_findings
        if status == "blocked" and critical_findings and block_reason != "owner_stopped":  # no model after a stop
            try:
                from ouroboros.tools.review_synthesis import synthesize_to_canonical_issues
                from ouroboros.review_state import load_state as _ls_synth
                _state_snap = _ls_synth(dr)
                _open_obs = _state_snap.get_open_obligations(repo_key=repo_key)
                _findings_for_attempt = synthesize_to_canonical_issues(
                    list(critical_findings),
                    open_obligations=_open_obs,
                    ctx=ctx,
                )
            except Exception as _synth_exc:  # the findings stay as reviewers wrote them
                log.debug("review_synthesis: pre-lock synthesis skipped: %s", _synth_exc)

        # C9.3: resolve semantic-dedup redirects for free-text (bug_*/risk_*) obligations
        # from a PRE-LOCK snapshot — the light-model call must stay OUTSIDE the review
        # state lock. Fail-open: any failure yields no redirect (a finding opens a new
        # obligation) and never blocks the gate. Only blocked attempts mint obligations.
        _obligation_redirects: Dict[str, str] = {}
        if status == "blocked" and _findings_for_attempt and block_reason != "owner_stopped":
            try:
                from ouroboros.review_state import (
                    compute_obligation_semantic_redirects,
                    load_state as _ls_dedup,
                )
                _obligation_redirects = compute_obligation_semantic_redirects(
                    _ls_dedup(dr), _findings_for_attempt, repo_key=repo_key, drive_root=dr
                )
            except Exception as _dedup_exc:  # no redirect: a finding opens its own obligation
                log.debug("obligation semantic dedup skipped: %s", _dedup_exc)

        from ouroboros.tools.review_change_custody import continued_attempt, late_write_check
        outlived = late_write_check(ctx, phase if status == "reviewing" else "", review_record_id, triad_raw_results, scope_raw_result)

        def _mutate(state):
            state.expire_stale_attempts()
            attempt_no = int(getattr(ctx, "_current_review_attempt_number", 0) or 0)
            existing = (
                state.latest_attempt_for(
                    repo_key=repo_key,
                    tool_name=tool_name,
                    task_id=task_id,
                    attempt=attempt_no,
                )
                if attempt_no > 0
                else None
            )
            if status == "reviewing":
                if outlived(existing):
                    return  # its collector settled this wave first: a stale pending write opens no second row (#2587)
                if not _attempt_accepts_reviewing_update(existing) and getattr(ctx, "_collect_commit_source", None) is None:
                    attempt_no = state.next_attempt_number(repo_key, tool_name, task_id,
                                                           above=continued_attempt(ctx, repo_key, tool_name, task_id))
                    existing = None
                ctx._current_review_attempt_number = attempt_no
            elif attempt_no <= 0:
                # Only an already-bound call updates an old attempt. A new refusal must not adopt another
                # wave's custody, paid flag or source reference, even when overlap filtering hides that wave.
                attempt_no = state.next_attempt_number(repo_key, tool_name, task_id,
                                                       above=continued_attempt(ctx, repo_key, tool_name, task_id))
                existing = None
                ctx._current_review_attempt_number = attempt_no

            from ouroboros.review_records import RECORDED_REVIEW_AUTHORITY, validate_author_disposition

            author_record = getattr(existing, "author_disposition", {}) or {}
            if author_disposition is not None:
                subject = pre_review_fingerprint or str(getattr(existing, "pre_review_fingerprint", "") or "")
                author_record = validate_author_disposition(author_disposition, subject_hash=subject) or {}
                cyber = cyber_review_authority()
                reference = author_record.get("review_reference")
                try:
                    source = resolve_commit_review_reference(ctx, reference, state=state) if reference else None
                except ValueError:
                    source = None
                same_review = bool(existing and getattr(existing, "paid", False) and subject == existing.pre_review_fingerprint)
                if (not subject or (not cyber and not (source or same_review))
                        or (post_review_fingerprint and post_review_fingerprint != subject)
                        or (not cyber and author_record.get("enforcement") != RECORDED_REVIEW_AUTHORITY)):
                    author_record = {}
            attempt = CommitAttemptRecord(
                ts=_utc_now(),
                commit_message=commit_message,  # full message; durable evidence
                status=status,
                snapshot_hash=snapshot_hash,
                block_reason=block_reason,
                block_details=block_details,
                duration_sec=duration_sec,
                task_id=task_id,
                critical_findings=_list_or_default(
                    _findings_for_attempt,
                    list(getattr(existing, "critical_findings", []) or []),
                ),
                repo_key=repo_key,
                tool_name=tool_name,
                attempt=attempt_no,
                phase=phase or infer_review_phase(status, block_reason),
                blocked=(status == "blocked"),
                advisory_findings=_normalize_advisory_entries(
                    _list_or_default(
                        advisory_findings,
                        getattr(existing, "advisory_findings", None)
                        or getattr(ctx, "_review_advisory", []),
                    )
                ),
                readiness_warnings=[
                    str(x) for x in _list_or_default(
                        readiness_warnings,
                        list(getattr(existing, "readiness_warnings", []) or []),
                    ) if str(x).strip()
                ],
                late_result_pending=late_result_pending,
                pre_review_fingerprint=pre_review_fingerprint or getattr(existing, "pre_review_fingerprint", ""),
                post_review_fingerprint=post_review_fingerprint or getattr(existing, "post_review_fingerprint", ""),
                fingerprint_status=fingerprint_status or getattr(existing, "fingerprint_status", ""),
                degraded_reasons=[
                    str(x) for x in _list_or_default(
                        degraded_reasons,
                        list(getattr(existing, "degraded_reasons", []) or []),
                    ) if str(x).strip()
                ],
                started_ts=str(getattr(existing, "started_ts", "") or ""),
                triad_models=[
                    str(x) for x in _list_or_default(
                        triad_models,
                        list(getattr(existing, "triad_models", []) or []),
                    ) if str(x).strip()
                ],
                scope_model=scope_model or str(getattr(existing, "scope_model", "") or ""),
                triad_raw_results=list(
                    triad_raw_results
                    if triad_raw_results is not None
                    else getattr(existing, "triad_raw_results", None) or []
                ),
                scope_raw_result=dict(
                    scope_raw_result
                    if scope_raw_result is not None
                    else getattr(existing, "scope_raw_result", None) or {}
                ),
                author_disposition=author_record,
                block_class=block_class or str(getattr(existing, "block_class", "") or ""),
                rebuttal_sha256=rebuttal_sha256 or str(getattr(existing, "rebuttal_sha256", "") or ""),
                paid=bool(paid or getattr(existing, "paid", False)),
                review_contract_fingerprint=(
                    review_contract_fingerprint
                    or str(getattr(existing, "review_contract_fingerprint", "") or "")
                ),
                review_retry_key=(
                    review_retry_key or str(getattr(existing, "review_retry_key", "") or "")
                ),
                root_task_id=root_task_id or str(getattr(existing, "root_task_id", "") or ""),
                review_owner_session_id=str(
                    getattr(existing, "review_owner_session_id", "") or ""
                ),
                review_owner_pid=int(
                    getattr(existing, "review_owner_pid", 0) or 0
                ),
                review_record_id=review_record_id or str(getattr(existing, "review_record_id", "") or ""),
            )
            if status != "reviewing" and "late_result_pending" not in legacy_kwargs and cyber_review_authority():
                attempt.late_result_pending = bool(getattr(existing, "late_result_pending", False)) or _attempt_has_active_review_custody(attempt)
            stamp_paid_review_owner(attempt, paid=bool(paid) or bind_owner)
            state.record_attempt(attempt, semantic_redirects=_obligation_redirects)

        update_state(dr, _mutate)

        try:
            from ouroboros.review_state import load_state
            from ouroboros.task_continuation import (
                build_review_continuation,
                clear_review_continuation,
                save_review_continuation,
            )

            if task_id:
                if status == "succeeded":
                    clear_review_continuation(dr, task_id)
                else:
                    source = _continuation_source(status, late_result_pending=late_result_pending,
                                                  stopped=block_reason == "owner_stopped")
                    if source:
                        latest_state = load_state(dr)
                        latest_attempt = latest_state.latest_attempt_for(
                            repo_key=repo_key,
                            tool_name=tool_name,
                            task_id=task_id,
                            attempt=int(getattr(ctx, "_current_review_attempt_number", 0) or 0) or None,
                        )
                        continuation = build_review_continuation(
                            {
                                "id": task_id,
                                "type": str(getattr(ctx, "current_task_type", "") or ""),
                                "parent_task_id": str(getattr(ctx, "parent_task_id", "") or ""),
                            },
                            latest_attempt,
                            latest_state.get_open_obligations(repo_key=repo_key),
                            source=source,
                        )
                        if continuation is not None:
                            save_review_continuation(dr, continuation, expect_task_id=task_id)
        except Exception as e:
            log.warning("Failed to sync review continuation: %s", e)
        if status in ("blocked", "failed", "succeeded") and not late_result_pending:
            ctx._current_review_attempt_number = None
    except Exception as e:
        log.warning("Failed to record commit attempt: %s", e)
        if strict:
            raise


def _invalidate_advisory(
    ctx: ToolContext,
    *,
    changed_paths: Optional[List[str]] = None,
    mutation_root: Optional[pathlib.Path] = None,
    source_tool: str = "",
) -> None:
    try:
        from ouroboros.review_state import invalidate_advisory_after_mutation
        invalidate_advisory_after_mutation(
            pathlib.Path(ctx.drive_root),
            mutation_root=mutation_root or pathlib.Path(ctx.repo_dir),
            changed_paths=changed_paths,
            source_tool=source_tool or _current_review_tool_name(ctx),
            mutating_task_id=str(getattr(ctx, "task_id", "") or ""),
        )
    except Exception:
        pass


def _check_overlapping_review_attempt(ctx: ToolContext) -> Optional[str]:
    from ouroboros.review_state import (
        ReviewStateMutation,
        _REVIEW_ATTEMPT_GRACE_SEC,
        _REVIEW_ATTEMPT_TTL_SEC,
        make_repo_key,
        update_state,
        _utc_now,
    )
    from ouroboros.tool_capabilities import REVIEWED_MUTATIVE_TOOLS

    from ouroboros.tools.review_change_custody import repair_held_commit_attempts
    repair_held_commit_attempts(ctx)  # outside the lock: a settled row an earlier collector left held, repaired exactly
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    expiration_window = _REVIEW_ATTEMPT_TTL_SEC + _REVIEW_ATTEMPT_GRACE_SEC
    ctx._review_resume_pending = False
    ctx._pending_review_attempt = None
    ctx._review_cyber_pending = ""
    ctx._review_open_peers = []

    def _mutate(state):
        from ouroboros.review_ledger import counted_pending
        from ouroboros.review_state_custody import _attempt_review_roster_rows

        expired = state.expire_stale_attempts(now_ts=_utc_now())
        rows = [
            item for item in state.get_active_attempts(repo_key=repo_key)
            if item.tool_name in REVIEWED_MUTATIVE_TOOLS
            and counted_pending(_attempt_review_roster_rows(item), pending=True)
        ]
        return ReviewStateMutation(rows, changed=bool(expired))

    try:
        active_attempts = update_state(pathlib.Path(ctx.drive_root), _mutate)
    except Exception as e:
        log.warning("Failed to check overlapping review attempts: %s", e)
        if cyber_review_authority():
            ctx._review_cyber_pending = f"Review custody is unreadable: {e}. No new reviewer will be dispatched."
            return None
        return (
            "⚠️ REVIEW_STATE_UNAVAILABLE: active paid-review custody could not "
            "be verified, so no reviewer dispatch was started. Retry after the "
            "review state store is readable."
        )
    source = getattr(ctx, "_author_commit_source", None)
    if source is not None:
        # Free author work gets a new invocation; same-task critics keep their custody.
        active_attempts = [item for item in active_attempts if
            (item.repo_key, item.tool_name, item.task_id) !=
            (source.repo_key, source.tool_name, source.task_id)]
    elif getattr(ctx, "_collect_commit_source", None) is None:
        from ouroboros.tools.review_collection import decided_round_attempts

        # A round whose recorded continuation a crash left unprojected is this task's open work too: the staged
        # bytes decide whether this call rejoins it (``recorded_round_choice``), so no new attempt row opens first.
        ctx._review_open_peers = decided_round_attempts(ctx, repo_key)
    if not active_attempts:
        return None
    task_id = str(getattr(ctx, "task_id", "") or "")
    tool_name = _current_review_tool_name(ctx)
    if _readers_frozen(ctx, active_attempts):
        # Their reviewers keep custody over retained frozen checkouts, so restaging the live
        # index changes nothing they read: this call stages its own candidate and either
        # rejoins its exact-identity attempt or gets its own panel (``exact_open_peer``).
        ctx._review_open_peers = [*ctx._review_open_peers, *(item for item in active_attempts if item.task_id == task_id
                                  and item.tool_name == tool_name and item.review_retry_key)]
        return None
    # The caller's own exact open attempt is rejoined in every mode before any
    # free-cycle or Cyber continuation: one request never buys a second wave.
    if len(active_attempts) == 1:
        candidate = active_attempts[0]
        if (
            (candidate.late_result_pending or candidate.paid)
            and candidate.task_id == task_id
            and candidate.tool_name == tool_name
            and candidate.review_retry_key
        ):
            ctx._review_resume_pending = True
            ctx._pending_review_attempt = candidate
            ctx._current_review_attempt_number = int(candidate.attempt or 0)
            return None
    if cyber_review_authority():
        # Cyber Pro: foreign custody is no veto, and no new reviewer is bought beside it.
        ctx._review_cyber_pending = (
            "Existing review custody remains active: "
            + ", ".join(f"{item.tool_name}#{item.attempt}" for item in active_attempts)
            + ". No new reviewer will be dispatched; the original attempts remain collectible."
        )
        return None

    active = active_attempts[-1]
    attempt_label = (
        f"{active.tool_name}#{active.attempt}"
        if int(active.attempt or 0) > 0
        else active.tool_name
    )
    return (
        f"⚠️ REVIEWED_ATTEMPT_IN_PROGRESS: {attempt_label} is still active "
        f"(status={active.status}, late_result_pending={bool(active.late_result_pending)}, "
        f"started={active.started_ts or active.ts}). "  # full ts — no [:19] truncation
        "Do not start another reviewed attempt for this repo. An exact retry may "
        "reconcile retained custody; otherwise operator recovery is required. "
        f"Only an unpaid legacy row auto-expires after {expiration_window}s."
    )


def _readers_frozen(ctx: ToolContext, attempts: Sequence[Any]) -> bool:
    """Every open attempt's reviewers read its round's retained isolated checkout (a wave
    whose author could leave it); a live-index reader, or an unverifiable one, keeps the
    index as part of its wave's identity."""
    from ouroboros.tools.review_subject import round_checkout_repo

    try:
        return bool(attempts) and all(item.review_retry_key and round_checkout_repo(
            ctx, item.review_retry_key, item.task_id).is_dir() for item in attempts)
    except Exception:
        return False


def exact_open_peer(ctx: ToolContext, commit_message: str, *, goal: str, scope: str, review_rebuttal: str,
                    fingerprint: str) -> Any:
    """The caller's own open attempt whose exact identity this call repeats — the same staged
    bytes and intent under that attempt's own paid contract — or ``None``: a distinct candidate
    then gets its own panel and the open attempt keeps its custody, never blessing these bytes."""
    from ouroboros.tools.parallel_review import _commit_review_retry_key

    current = getattr(ctx, "_current_review_contract_fingerprint", "")
    try:
        for peer in getattr(ctx, "_review_open_peers", None) or []:
            ctx._current_review_contract_fingerprint = str(peer.review_contract_fingerprint or current)
            if _commit_review_retry_key(ctx, commit_message, goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                                        binding_fingerprint=fingerprint) == peer.review_retry_key:
                return peer
    finally:
        ctx._current_review_contract_fingerprint = current
    return None


def review_failure_is_technical(facts: Dict[str, Any]) -> bool:
    """Classify delivery failures; read coverage is diagnostic, not a failure."""
    return (
        facts.get("failure_phase") in {
            "context", "delivery", "format", "window_authority"}
        and facts.get("operation_state") not in {"in_flight", "custody_lost"}
        and not facts.get("pending_invocation_id") and not facts.get("late_result_pending")
    )


PREFLIGHT_STATUSES = ("performed", "not_performed", "skipped")


def preflight_reviewer_error(selector: str) -> str:
    """``commit_reviewed(preflight_reviewer=…)`` names one ENABLED catalog row, a pool member
    or not; an unknown or disabled row is the caller's argument error ("" = the row stands)."""
    from ouroboros import reviewer_slot_config as slots

    try:
        slots.catalog_review_row(None, selector)
    except ValueError as exc:
        return f"preflight_reviewer {selector!r} is not an enabled catalog row ({exc})"
    return ""


def commit_preflight_choice_error(selector: str, *, skipped: bool, continuation: bool,
                                 ctx: Any = None, goal: str = "", scope: str = "", review_rebuttal: str = "") -> str:
    """A commit's preflight is a named row, an explicit skip, or neither; an author
    continuation dispatches no critic, so it names no row either ("" = the choice stands)."""
    if not selector:
        return ""
    if skipped:
        return "preflight_reviewer and skip_advisory_review=True are opposite choices; pass one."
    if continuation:
        return "preflight_reviewer and an author continuation are separate choices (the continuation dispatches no critic)."
    error = preflight_reviewer_error(selector)
    if error and ctx is not None:
        from ouroboros.tools.review_change_custody import pending_preflight_for_call

        if pending_preflight_for_call(ctx, goal=goal, scope=scope, review_rebuttal=review_rebuttal) is not None:
            return ""  # collect its original seat; never admit a replacement
    return error


def compose_commit_panel(ctx: ToolContext, reviewers: Sequence[str], reason: str) -> Any:
    """Freeze the shared own-body panel before staging or paying for review."""
    from ouroboros.tools import review_change as rc
    from ouroboros.reviewer_slot_config import review_strictness
    from ouroboros.tools import git as git_mod

    request = rc.parse_request({"root": "system_repo", "subject": "index", "reviewers": list(reviewers), "reason": reason})
    return rc.compose_panel(request, body=True, strictness=review_strictness(git_mod._current_runtime_mode()))


def release_diagnostics(ctx: ToolContext, paths: Optional[List[str]], source: str) -> Dict[str, Any]:
    """Free release metadata and a separately labelled worktree book balance.

    No sync, staging, tests, provider, readiness pipeline or review state.
    """
    if source not in ("worktree", "index"):
        return {"status": "error", "failure_code": "PREFLIGHT_SOURCE_REQUIRED",
                "message": "deterministic_only requires explicit source=worktree or source=index."}
    from ouroboros import body_candidate
    from ouroboros.commit_admission import release_metadata_diagnostics
    from ouroboros.reference_books import BOOK_ENTRYPOINTS, book_balance_note, ruled_text_paths

    repo = pathlib.Path(ctx.repo_dir)
    note = (book_balance_note(repo, paths if paths is not None
                              else [*BOOK_ENTRYPOINTS.values(), *ruled_text_paths(repo)])
            if (repo / "ouroboros" / "reference_books.py").is_file() else "")
    return {**release_metadata_diagnostics(
        repo, paths, source=source, neutral_allowed=True if source == "index" else body_candidate.is_bound(ctx)),
            "deterministic_only": True, "review_freshness": False,
            "book_balance": {"source": "worktree", "note": note}}


def _neutral_commit_allowed(ctx: ToolContext) -> bool:
    """Prepared body contributions and authorized adoption of the selected official release."""
    from ouroboros import body_candidate
    from ouroboros.tools.registry_guards import _authorized_managed_update_resolver

    return body_candidate.is_bound(ctx) or _authorized_managed_update_resolver(ctx)


def deterministic_preflight(ctx: ToolContext, commit_message: str, paths: Optional[List[str]],
                            *, review_only: bool = False) -> str:
    """The free checks that ran ahead of the retired advisory delivery, kept whole for every
    commit: size headroom, the staged change's diff pulse (the text the panel's seats read
    first) and readiness (information and warnings), release metadata of the staged index,
    and the syntax of staged ``.py`` files. A non-committing cycle reviews the prepared
    contribution form too; it grants no commit authority. Returns the blocking message or ""."""
    from ouroboros.commit_admission import release_metadata_preflight, syntax_preflight_staged_py_files
    from ouroboros.review import diff_pulse
    from ouroboros.tools.review_helpers import check_worktree_readiness
    from ouroboros.utils import append_jsonl, utc_now_iso

    repo_dir = pathlib.Path(ctx.repo_dir)
    information: List[str] = []
    warnings = check_worktree_readiness(repo_dir, paths=paths, information=information)
    information = [line for line in (*information, diff_pulse(repo_dir, staged=True)) if line]
    if information:
        ctx.emit_progress_fn("Size (information):\n" + "\n".join(information))
    if warnings:
        ctx.emit_progress_fn(f"⚠️ Readiness: {'; '.join(warnings)}")
        try:
            append_jsonl(pathlib.Path(ctx.drive_root) / "logs" / "events.jsonl", {
                "ts": utc_now_iso(), "type": "commit_readiness_gate", "warnings": warnings,
                "task_id": str(getattr(ctx, "task_id", "") or "")})
        except Exception:
            pass
    return (release_metadata_preflight(repo_dir, commit_message, paths, source="index",
                                       neutral_allowed=review_only or _neutral_commit_allowed(ctx))
            or syntax_preflight_staged_py_files(repo_dir, list(paths or [])) or "")


def run_commit_preflight(ctx: ToolContext, reviewer: str, *, commit_message: str, goal: str, scope: str,
                         review_rebuttal: str) -> Dict[str, Any]:
    """The author's preflight (decision 3A): ``review_change(subject=worktree, surface=preflight,
    reviewers=[reviewer])`` over the system repository. It informs and never gates (the commit
    panel holds the gate); a refused or crashed call is still ``performed`` with its cause."""
    from ouroboros.budget_pause import BudgetPauseRequested
    from ouroboros.tools import review_change as rc
    from ouroboros.usage_accounting import BudgetExceeded

    ctx.emit_progress_fn(f"Preflight: {reviewer} reads the worktree before the panel...")
    try:
        result = rc.run_review_change(ctx, root="system_repo", subject="worktree", surface="preflight",
                                      reviewers=[reviewer], goal=goal or commit_message, scope=scope,
                                      review_rebuttal=review_rebuttal,
                                      _preflight_execution=getattr(ctx, "_preflight_execution", None))
    except (BudgetExceeded, BudgetPauseRequested):
        raise
    except Exception as exc:
        log.warning("commit preflight did not complete", exc_info=True)
        return {"status": "performed", "record_id": "", "reviewer": reviewer, "aggregate": "",
                "error": f"{type(exc).__name__}: {exc}"}
    findings = dict(result.get("findings") or {})
    fact = {"status": "performed", "record_id": str(result.get("record_id") or ""), "reviewer": reviewer,
            "aggregate": str(result.get("aggregate") or "")}
    critical = list(findings.get("critical_findings") or [])
    ctx.emit_progress_fn(f"Preflight {fact['aggregate'] or 'recorded'} (record {fact['record_id'] or 'unwritten'}); "
                         f"{len(critical)} critical finding(s) for the author to weigh before the panel.")
    return fact


def bind_commit_wake(ctx: ToolContext, wake_on: Any, panel: Any) -> None:
    """Validate the wait choice independently of landing authority, before staging."""
    from ouroboros.review_waiting import select_helpers, bind_panel
    select_helpers(ctx, wake_on)
    if panel is not None:
        bind_panel(ctx, panel)


def _return_commit_feedback(ctx: ToolContext, message: str, started: float, before: dict, after: dict,
                            *, pending: bool = False, reason: str = "author_decision_required", findings: Optional[list] = None,
                            note: str = "") -> dict:
    """Return the received outcome before any Git effect, preserving its original attempt.

    Nonblocking application hands back every outcome of a committing call, a clean PASS
    included; a pending wave keeps its exact index and attempt for a free rejoin. The
    commit is the author's explicit continuation with the returned reference."""
    from dataclasses import asdict
    import time
    from ouroboros.tools import git as git_mod
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.mutation_attribution import selection_disclosure

    gap = selection_disclosure(getattr(ctx, "_commit_attribution", None))  # the reviewed composition's attribution gap
    note = (gap + " " if gap else "") + note
    if not pending:  # a settled wave: no custody stays open behind its handback, in Cyber Pro too
        git_mod._record_commit_attempt(ctx, message, "reviewed", phase="review_only", block_reason=reason,
            late_result_pending=False, duration_sec=time.time() - started, pre_review_fingerprint=before["fingerprint"],
            post_review_fingerprint=after.get("fingerprint", ""), fingerprint_status="matched",
            critical_findings=findings if findings is not None else getattr(ctx, "_last_review_critical_findings", []),
            advisory_findings=getattr(ctx, "_review_advisory", []),
            triad_raw_results=getattr(ctx, "_last_triad_raw_results", []), scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
            degraded_reasons=getattr(ctx, "_review_degraded_reasons", []))
    from ouroboros.tools.review_collection import shown_field

    state = load_state(pathlib.Path(ctx.drive_root))
    row = state.latest_attempt_for(repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
        task_id=str(getattr(ctx, "task_id", "") or ""), tool_name="commit_reviewed", attempt=ctx._current_review_attempt_number)
    reference = {"surface": "commit", **{key: getattr(row, key) for key in ("repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint")},
                 "review_record_id": str(getattr(row, "review_record_id", "") or ""), **shown_field(ctx, row)} if row is not None else {}
    try:
        resolve_commit_review_reference(ctx, reference, state=state)
        continuable = True
    except ValueError:
        continuable = False
    collect = "Re-issue the same unchanged commit call to collect this paid wave at $0 (no second dispatch), "
    choice = ("Inspect it, then revise and review again, stop, or explicitly continue with the same commit tool "
              "passing review_reference and author_disposition {disposition: accepted|rejected|partial|deferred, "
              "rationale: ..., subagent_check optional}; continuation re-checks the current bytes, buys no reviewer "
              "and is no reviewer PASS. subagent_check is author evidence that a helper checked this candidate; "
              "it is not a seat and does not create PASS. ")
    if row is None:
        choice = "The attempt row is unreadable, so no continuation reference exists; retry or stop. "
    elif pending:
        choice = collect + ("or explicitly continue now with review_reference and author_disposition "
                            "(subagent_check optional, not a seat, not a PASS); the pending "
                            "review stays recorded and is no PASS. " if continuable
                            else "or stop; informed continuation needs received feedback. ")
        left = getattr(ctx, "_left_review_wait", None) or {}
        if left.get("surface") == "commit" and left.get("retry_key") == getattr(ctx, "_current_review_retry_key", ""):
            from ouroboros.tools.review_change import left_wave_terms
            event = ("A selected outcome or wake time was ready" if any(row.get("reason") in {"helper_result", "review_result", "wake_at"} for row in
                     getattr(ctx, "_review_detached_operations", []) or []) else "An addressed message reached you")
            note = (event + " while the reviewers ran: this call returned and their paid work continues separately; "
                    + left_wave_terms(commit=True) + " " + note)
    if not pending:
        git_mod._unstage_candidate(ctx)
    from ouroboros.review_continuation import offers_for

    offers = offers_for(ctx, reference.get("review_record_id"))
    if offers:  # a stopped session seat continues only by the author's explicit choice, in the same wave
        choice += ("A reviewer in continuable_seats stopped with work its engine can continue: the same commit tool with "
                   "review_reference and continue_seats (no author_disposition) continues it over its frozen subject. ")
    text = ("Review outcome returned before commit" + (": reviewers are still running. " if pending else ". ") + note + choice +
            "Current files are preserved; no commit, tag or push occurred.\n" + json.dumps({"review_reference": reference,
            "review_outcome": asdict(row) if row is not None else {}, **({"continuable_seats": offers} if offers else {})},
            ensure_ascii=False, default=str))
    return {"status": "reviewed", "message": text, "review_reference": reference,
            "pre_fingerprint": before, "post_fingerprint": after}


def watch_left_commit_review(ctx: ToolContext) -> bool:
    """The author left this commit's paid wave on an addressed message: keep the wait
    intention and one whole-wave settlement notice (``review_pause.watch_left_review``)."""
    from ouroboros import review_pause

    return review_pause.watch_left_review(
        ctx, surface="commit", retry_key=str(getattr(ctx, "_current_review_retry_key", "") or ""),
        attempt=int(getattr(ctx, "_current_review_attempt_number", 0) or 0),
        record_id=str(getattr(ctx, "_current_review_record_id", "") or ""),
        reissue="the same unchanged commit_reviewed call") is not None


def disclose_commit_review_replay(ctx: ToolContext, replay: dict) -> None:
    """Keep the free replay cause separate from authority to create a commit."""
    replay_reason = str(replay.get("replay_reason") or "")
    cyber = cyber_review_authority()
    progress_note = ("Cyber Pro: no new reviewer dispatched; original review facts are retained and the outcome returns first."
                     if cyber else "Max Review Cycles: paid-cycle ceiling exhausted — no review outcome exists for this diff; "
                     "inspect the returned outcome before explicitly choosing an author continuation.")
    disclosure = ("Cyber Pro: no new review was dispatched. " if cyber else
                  f"No new review wave was bought for this commit ({replay_reason}); no fresh automatic preflight was bought. "
                  ) + str(replay.get("advisory_replay") or "")
    advisory_list = getattr(ctx, "_review_advisory", None)
    if isinstance(advisory_list, list):
        advisory_list.append(disclosure)
    try:
        ctx.emit_progress_fn(progress_note)
    except Exception:
        pass


def bind_author_commit_candidate(ctx: ToolContext, commit_message: str, pre_fingerprint: dict) -> Optional[str]:
    """Bind the current author choice, then independently check its staged bytes."""
    from ouroboros.tools import git as git_mod
    author_source = ctx._author_commit_source
    from ouroboros.review_records import build_author_disposition_from_mapping
    from ouroboros.tools.review import _preflight_check, format_name_status_for_preflight
    from ouroboros.review_records import RECORDED_REVIEW_AUTHORITY

    # The critic's own aggregate on ITS subject (the reference's fingerprint), never this candidate's verdict.
    signal, record = "", {}
    if str(getattr(author_source, "review_record_id", "") or ""):
        from ouroboros import review_ledger
        try:
            record = review_ledger.load_record(review_ledger.ledger_root(ctx), author_source.review_record_id) or {}
            signal = str((record.get("verdict") or {}).get("aggregate") or "")
        except Exception:
            log.debug("critic aggregate unavailable for the author continuation", exc_info=True)
    author = build_author_disposition_from_mapping(ctx._author_commit_decision, subject_hash=pre_fingerprint["fingerprint"],
        reviewer_signal=signal or author_source.block_reason or author_source.status, enforcement=RECORDED_REVIEW_AUTHORITY)
    author.update(review_reference=ctx._author_commit_reference, decided_by_task=str(ctx.task_id))  # the acting author
    from ouroboros.review_doctrine import enrich_author_commit
    from ouroboros.utils import utc_now_iso

    author = enrich_author_commit(
        author,
        actors=list(record.get("rows") or []) if getattr(author_source, "raw_stripped", False) else [  # compacted: its record's seats
            *(getattr(author_source, "triad_raw_results", None) or []), getattr(author_source, "scope_raw_result", None) or {}],
        prior_decision=record.get("author_decision") if isinstance(record.get("author_decision"), dict) else None,
        record_id=str(record.get("record_id") or getattr(author_source, "review_record_id", "") or ""),
        revision=int(record.get("revision") or 0),
        drive_root=getattr(ctx, "drive_root", None),
        cyber=cyber_review_authority(),
        observed_at=utc_now_iso(),
        review_reference=ctx._author_commit_reference,
    )
    from ouroboros.tools.review_collection import disclose_subject_delta
    author["subject_delta"] = disclose_subject_delta(ctx, pre_fingerprint)  # published before the tests and Git
    ctx._author_commit_record = author
    staged = git_mod.run_cmd(["git", "diff", "--cached", "--name-status"], cwd=ctx.repo_dir)
    return _preflight_check(commit_message, format_name_status_for_preflight(staged), ctx.repo_dir,
                            neutral_allowed=_neutral_commit_allowed(ctx))


def record_bound_commit_success(ctx: ToolContext, commit_message: str, started_at: float, before: dict, after: dict,
                                commit_sha: str = "") -> None:
    """Persist the already-verified Git effect with its actual review/author facts.

    Landing is written here, after Git succeeded, and nowhere earlier. A failed
    Git command never reaches this function, so it is not a commit. The created
    ``commit_sha`` lands on this call's record with its own tree, read from Git.
    """
    import time
    from ouroboros.review_doctrine import attach_landing
    from ouroboros.tools import git as git_mod

    author = getattr(ctx, "_author_commit_record", None)
    pre_fp, post_fp = before.get("fingerprint", ""), after.get("fingerprint", "")
    if isinstance(author, dict):
        author = attach_landing(author, pre_review_fingerprint=pre_fp, post_review_fingerprint=post_fp)
        ctx._author_commit_record = author
    git_mod._record_commit_attempt(ctx, commit_message, "succeeded",
                           author_disposition=author,
                           duration_sec=time.time() - started_at,
                           phase="commit",
                           pre_review_fingerprint=pre_fp,
                           post_review_fingerprint=post_fp,
                           fingerprint_status="matched",
                           triad_models=getattr(ctx, "_last_triad_models", []),
                           scope_model=getattr(ctx, "_last_scope_model", ""),
                           triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                           scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                           degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []))
    subject = str(author.get("subject_hash") or "") if isinstance(author, dict) else ""
    landing = author.get("landing") if isinstance(author, dict) else None
    record_id = str(getattr(ctx, "_current_review_record_id", "") or "")
    authored = isinstance(landing, dict) and post_fp == pre_fp == subject
    if record_id and (authored or commit_sha):
        from ouroboros import review_ledger as ledger

        try:  # the commit object's own tree, then its parents; unreadable leaves the relation unknown
            tree, *parents = git_mod.run_cmd(["git", "rev-parse", f"{commit_sha}^{{tree}}", f"{commit_sha}^@"],
                                             cwd=ctx.repo_dir).split() if commit_sha else [""]
        except Exception:
            tree, parents = "", []
        ledger.note_commit_landing(ledger.ledger_root(ctx), record_id, landing if authored else None, {
            "commit_sha": commit_sha, "tree_sha": tree, "parent_sha": (parents or [""])[0],
            "author_subject": subject if authored else ""} if commit_sha else None)


def prepare_author_commit_request(ctx: ToolContext, review_reference: Any, author_disposition: Any, review_rebuttal: str,
                                  *, paths_omitted: bool = False) -> Optional[str]:
    """Validate the explicit free continuation before touching a Git candidate; its reviewed subject is read
    once, and omitted paths select that subject's paths (``review_collection.author_scope``)."""
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
    if review_reference is not None or author_disposition is not None:
        from ouroboros.review_records import build_author_disposition_from_mapping
        from ouroboros.tools.review_collection import continuation_refusal, require_shown, without_broken_subagent_check
        try:
            if review_rebuttal:
                raise ValueError("a paid rebuttal and free author continuation are separate choices")
            author_disposition = without_broken_subagent_check(ctx, author_disposition)
            build_author_disposition_from_mapping(author_disposition, subject_hash="pending-current-candidate")
            ctx._author_commit_source = require_shown(ctx, review_reference, resolve_commit_review_reference(ctx, review_reference))
            ctx._author_commit_reference, ctx._author_commit_decision = dict(review_reference), dict(author_disposition)
        except (OSError, ValueError) as exc:  # the text names the next call its cause leaves
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=continuation_refusal(exc)))
        from ouroboros.tools.review_collection import author_scope
        return author_scope(ctx, paths_omitted=paths_omitted)
    return None


# ---- Review ledger hook (ARCHITECTURE §6 "Review ledger record") ------------

def _review_preflight_facts(ctx: ToolContext) -> Dict[str, Any]:
    """THIS attempt's preflight: ``performed`` names the ``surface=preflight`` record of the row
    the author chose, ``skipped`` is an explicit ``skip_advisory_review``, and ``not_performed``
    is the plain fact that none was asked for (never an audited bypass)."""
    fact = getattr(ctx, "_commit_preflight", None)
    if isinstance(fact, dict) and fact.get("status") in PREFLIGHT_STATUSES:
        return dict(fact)
    return {"status": "not_performed", "record_id": ""}


def commit_panel_facts(ctx: ToolContext) -> Dict[str, Any]:
    """THIS attempt's panel composition as ``compose_commit_panel`` stated it (set by the
    cycle, reset per call); ``{}`` is the configured pool, the record's defaults."""
    return dict(getattr(ctx, "_commit_review_panel", None) or {})


def _review_body_facts(ctx: ToolContext) -> Dict[str, Any]:
    """The gate reviews the system repository (``_repo_commit_push`` refuses any other
    root), so its wave runs the body layer; the record states that through the same
    predicate ``review_change`` uses (``review_body_fact.body_fact``), never by assertion.
    The rules are the SERVING body's (``body_candidate.serving_repo_dir_for`` — the root
    the brief tiers them from, ``review._gate_governance_root``): a bound candidate is the
    subject judged against the body that runs, never against its own rewritten copy, and
    the record names that root. An unanswerable predicate leaves the layer facts out (the
    ledger says ``unknown``)."""
    try:
        from ouroboros.body_candidate import serving_repo_dir_for
        from ouroboros.review_body_fact import body_fact, layer_for
        from ouroboros.review_ledger import ledger_root

        serving = str(serving_repo_dir_for(ctx).resolve(strict=False))
        fact = body_fact(ctx.repo_dir, system_repo=serving, data_dir=ledger_root(ctx))
        return {"governance_root": serving, "layer": layer_for(fact), "body_fact": str(fact.body),
                "body_how": str(fact.how)}
    except Exception:
        log.warning("review body fact unavailable for the commit gate record", exc_info=True)
        return {}


def _review_ledger_facts(ctx: ToolContext, commit_message: str, *, goal: str, scope: str, pre_fingerprint: dict,
                         blocked: bool, block_reason: str,
                         combined_findings: Optional[list], dispatch_refusal: Optional[dict], pending: bool) -> Dict[str, Any]:
    """Everything the ledger record states about THIS attempt, read from the gate's own
    forensic fields (no second reading of any reviewer output)."""
    from ouroboros.review_records import RECORDED_REVIEW_AUTHORITY, ReviewRequest, resolve_review_wave
    from ouroboros.tools import git as git_mod

    task_id = str(getattr(ctx, "task_id", "") or "")
    # THIS wave's own execution rows (stashed by the substrate as it recorded them), never the
    # shared last-execution projection another surface may have overwritten meanwhile.
    executions = dict(getattr(ctx, "_last_review_slot_executions", {}) or {})
    try:
        mode = str(git_mod._current_runtime_mode() or "")
    except Exception:
        mode = ""
    tests_passed = getattr(ctx, "_preflight_tests_passed", None)
    panel = commit_panel_facts(ctx)
    return {
        **_review_body_facts(ctx),
        "panel": panel, "composition": str(panel.get("composition") or "full_pool"),
        "composition_reason": str(panel.get("reason") or ""), "chosen_by": str(panel.get("chosen_by") or "owner"),
        "task_id": task_id, "root_task_id": resolve_root_task_id(ctx),
        "review_wave_id": resolve_review_wave(ReviewRequest(
            surface="commit_gate", goal=goal or commit_message, task_id=task_id,
            retry_key=str(getattr(ctx, "_current_review_retry_key", "") or "")), {}, ""),
        "repo_dir": str(ctx.repo_dir), "goal": goal, "scope": scope, "commit_message": commit_message,
        "since_review": str(getattr(ctx, "_review_since", "") or ""),
        "binding": dict(pre_fingerprint.get("binding") or {}),
        "binding_fingerprint": str(pre_fingerprint.get("fingerprint") or ""),
        "review_contract_fingerprint": str(getattr(ctx, "_current_review_contract_fingerprint", "") or ""),
        "rebuttal_sha256": str(getattr(ctx, "_current_review_rebuttal_sha256", "") or ""),
        "enforcement": RECORDED_REVIEW_AUTHORITY, "mode": mode,
        "structured": dict(getattr(ctx, "_last_review_structured", {}) or {}), "slot_executions": executions,
        "triad_raw": list(getattr(ctx, "_last_triad_raw_results", []) or []),
        "blocked": bool(blocked), "block_reason": str(block_reason or ""),
        "dispatch_refusal": dispatch_refusal, "pending": bool(pending),
        "degraded_reasons": list(getattr(ctx, "_review_degraded_reasons", []) or []),
        "critical_findings": list(combined_findings or getattr(ctx, "_last_review_critical_findings", []) or []),
        "advisory_findings": list(getattr(ctx, "_last_review_advisory_findings", []) or []),
        "tests": _review_tests_facts(ctx, tests_passed),
        "test_request": dict(getattr(ctx, "_commit_test_request", None) or {}),
        "preflight": _review_preflight_facts(ctx),
        # The exhibit frozen before this attempt's dispatch, never re-derived at settle.
        "preflight_execution": dict(getattr(ctx, "_preflight_execution", None) or {"status": "unobserved"}),
    }


def name_review_record(ctx: ToolContext, result: Any) -> Any:
    """Every outcome of a commit call that wrote a review record names it (DEVELOPMENT 05):
    passed, blocked, pending, refused after the wave, or failed at the commit itself. The id
    is this call's own (reset at the start of every call); ID-less exits stay ID-less."""
    record_id = str(getattr(ctx, "_current_review_record_id", "") or "")
    if not record_id or not isinstance(result, str) or record_id in result:
        return result
    from ouroboros.review_doctrine import doctrine_result_lines
    from ouroboros.tools.tool_result import append_published_text

    suffix = f"\nreview_record_id: {record_id}"
    lines = doctrine_result_lines(getattr(ctx, "_author_commit_record", None))
    if lines:
        suffix += "\n" + "\n".join(lines)
    if not getattr(ctx, "last_reviewed_commit_sha", ""):  # nothing landed: a stopped seat may still be chosen
        from ouroboros.tools.review_collection import commit_offer
        suffix += commit_offer(ctx, record_id)
    return append_published_text(ctx, result, suffix)


def _review_tests_facts(ctx: ToolContext, tests_passed: Any) -> Dict[str, Any]:
    """Passed requires a proof covering this tree/index/workload, not merely a kept flag.
    Skipped/stale runs are NOT_RUN. The frozen exhibit governs too: deterministic
    refusal means tests never started, even when an older proof still covers the tree.
    """
    if tests_passed is not True:
        return {"policy": "NOT_RUN", "result": "unknown"}
    try:
        from ouroboros.commit_admission import preflight_test_proof_matches

        bound = bool(preflight_test_proof_matches(ctx, ctx.repo_dir, require_full=False))
    except Exception:
        bound = False
    exhibit = getattr(ctx, "_preflight_execution", None)
    if bound and isinstance(exhibit, dict) and exhibit.get("status") == "not_run":
        return {"policy": "NOT_RUN", "result": "unknown", "reason": "tests_not_started"}
    if bound:
        return {"policy": "run", "result": "passed", "proof": "candidate_bound"}
    return {"policy": "NOT_RUN", "result": "unknown", "reason": "tests_proof_not_for_this_candidate"}


def settle_commit_review_ledger(ctx: ToolContext, commit_message: str, *, goal: str = "", scope: str = "",
                                pre_fingerprint: Optional[dict] = None,
                                blocked: bool = False, block_reason: str = "", combined_findings: Optional[list] = None,
                                author_source: Any = None, advisory_replay: Optional[dict] = None) -> str:
    """Write THIS attempt's review ledger record once the gate has aggregated its verdict
    and bind the id to the attempt row and the tool result (``_current_review_record_id``).

    A dispatched wave is a ``settled`` record, or ``pending`` while custody is still
    open; settling that same attempt later raises the record's ``revision``. A free
    replay (exhausted cycles, pending custody) and a wave the budget
    fence declined are ``NOT_DISPATCHED`` records naming their cause. An explicit author
    continuation buys no record: it notes the author's decision on the record it answered.
    Ledger failure never changes the gate's decision; it is logged and the id stays ""."""
    from ouroboros import review_ledger as ledger
    from ouroboros.review_overrides import note_author_continuation
    from ouroboros.tools import git as git_mod

    ctx._current_review_record_id = ""
    try:
        root = ledger.ledger_root(ctx)
        if author_source is not None:
            prior = str(getattr(author_source, "review_record_id", "") or "")
            if prior and ledger.load_record(root, prior) is not None:
                ledger.note_author_decision(root, prior, dict(getattr(ctx, "_author_commit_record", None) or {}))
                ctx._current_review_record_id = prior
            note_author_continuation(ctx, getattr(ctx, "_author_commit_record", None))  # over a FAIL or no verdict
            return ctx._current_review_record_id
        structured = dict(getattr(ctx, "_last_review_structured", {}) or {})
        refusal = None
        if advisory_replay is not None:
            refusal = {"kind": str(advisory_replay.get("replay_reason") or "free_replay"),
                       "message": str(advisory_replay.get("advisory_replay") or "")}
        elif str(getattr(ctx, "_last_review_block_reason", "") or "") == "review_wave_budget_insufficient":
            refusal = {"kind": "review_wave_budget_insufficient", "message": str(structured.get("wave_refusal") or "")}
        pending = advisory_replay is None and bool(git_mod._review_custody_pending(ctx))
        facts = _review_ledger_facts(
            ctx, commit_message, goal=goal, scope=scope, pre_fingerprint=pre_fingerprint or {},
            blocked=blocked, block_reason=block_reason,
            combined_findings=combined_findings, dispatch_refusal=refusal, pending=pending)
        # The exact retry of a pending attempt (same retry key, overlap check bound it)
        # settles THAT attempt's record; the roster release already cleared the
        # reconcile flag by the time the verdict is aggregated.
        attempt, retry_key = getattr(ctx, "_pending_review_attempt", None), str(getattr(ctx, "_current_review_retry_key", "") or "")
        rejoined = attempt is not None and bool(retry_key) and str(getattr(attempt, "review_retry_key", "") or "") == retry_key
        prior = str(getattr(attempt, "review_record_id", "") or "") if rejoined else ""
        existing = ledger.load_record(root, prior) if prior else None
        if existing is not None:
            # Settling in place: the dispatching attempt's provenance (subject, brief with
            # its rules, panel, fingerprints, the seats' prompt refs) stays; this attempt
            # contributes the late answers, the verdict, the cost and the state.
            fresh = ledger.build_commit_gate_record(facts, record_id=prior, drive_root=root).to_dict()
            ledger.revise_record(root, prior, lambda payload: ledger.settle_pending_payload(payload, fresh))
            ctx._current_review_record_id = prior
        else:
            record = ledger.build_commit_gate_record(facts, drive_root=root)
            # The composition facts the panel was built with (requested rows, added critics,
            # the reason) ride the record's panel block as on ``review_change`` (its
            # ``reason_missing`` is the composer's: a panel that names the whole pool owes none).
            record.panel = {**commit_panel_facts(ctx), **dict(record.panel or {})}
            if rejoined and existing is None:
                # Late answers to a wave whose dispatching attempt left no record: the
                # rules and brief the seats saw are not this attempt's — say so.
                ledger.mark_provenance_unknown(record)
            ctx._current_review_record_id = str(ledger.write_record(root, record)["record_id"])
        from ouroboros.reviewer_slot_config import bind_reviewer_slot_record_id

        bind_reviewer_slot_record_id(facts.get("slot_executions") or {}, ctx._current_review_record_id)
        _bind_attempt_record_id(ctx, ctx._current_review_record_id)
    except Exception:
        log.warning("review ledger record could not be written for this commit attempt", exc_info=True)
    return str(ctx._current_review_record_id or "")


def _bind_attempt_record_id(ctx: ToolContext, record_id: str) -> None:
    """Name the record on THIS attempt's custody row as soon as it exists (the terminal
    write repeats the id); only the id changes, no lifecycle field is touched."""
    from ouroboros.review_state import make_repo_key, update_state

    number = getattr(ctx, "_current_review_attempt_number", None)
    if not record_id or not number:
        return

    def _mutate(state):
        row = state.latest_attempt_for(repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)), tool_name=_current_review_tool_name(ctx),
                                       task_id=str(getattr(ctx, "task_id", "") or ""), attempt=int(number))
        if row is not None:
            row.review_record_id = record_id

    update_state(pathlib.Path(ctx.drive_root), _mutate)
