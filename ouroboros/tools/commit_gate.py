"""The commit preflight (the author's named row, decision 3A) and the free deterministic
checks every commit runs before review, durable commit-attempt recording, and the
commit-side Max-Review-Cycles machinery (block classification, the free
identical-diff refusal, the per-root-task paid-cycle ceiling, and the
review-contract fingerprint)."""

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
from ouroboros.tools.review_helpers import (
    REVIEW_POOL_EMPTY_REASON,
    REVIEW_POOL_EMPTY_SENTENCE,
    review_enforcement_blocks,
)
from ouroboros.utils import (
    truncate_review_artifact as _truncate_review_reason,
)

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


def _continuation_source(status: str, *, late_result_pending: bool) -> str:
    if status == "blocked":
        return "blocked_review"
    if late_result_pending:
        return "late_result_pending"
    if status == "failed":
        return "review_failure"
    return ""


def _attempt_accepts_reviewing_update(existing: Any) -> bool:
    if existing is None:
        return False
    return _attempt_has_active_review_custody(existing)


# Max Review Cycles semantics on the commit gate (owner Q12/Q16/Q22/Q23):
# identical bytes are never re-reviewed for pay. From the FIRST genuine
# review-verdict block of a staged diff, resubmitting the same
# pre_review_fingerprint without a NEW rebuttal is refused for FREE — before
# the advisory-freshness gate and before any paid review-wave dispatch —
# quoting the recorded verdict. A rebuttal is identified by CONTENT
# (sha256): a hash new to the current identical-fingerprint streak buys
# exactly ONE paid re-review; a repeated hash is refused free, quoting the
# previous outcome. Infra-blocks (fit overflow, sub-floor window,
# revalidation, transport/no-quorum) are not verdicts: they never build the
# refusal streak and retry freely. The shared OUROBOROS_REVIEW_MAX_CYCLES
# knob (``review_max_cycles()``; ``None`` = unlimited) bounds PAID
# review-wave cycles per ROOT task (the whole task tree shares one ceiling;
# a manual session is its own task; a follow-up task starts a fresh one).
# The ceiling counts MONEY: every attempt that physically dispatched a wave
# (``paid`` recorded at dispatch) counts regardless of how it terminated —
# only UNDISPATCHED attempts (preflight refusals, assembly failures, free
# replays) are outside the count. Exhaustion is a
# free typed refusal plus ``emit_review_cycles_exhausted``. Both refusals
# honor the recorded review-contract fingerprint (roster+routes+enforcement+
# prompt contract): a changed contract lapses the streak. Under ADVISORY
# enforcement returns the prior outcome for an informed author choice;
# explicit continuation buys no reviewer and does not manufacture approval.
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
# of their own: the streak walker's generic terminal break already ends an
# identical-diff streak on them, and their paid dispatch stays counted.
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
        # GATE, never reviewer verdicts: they must not anchor identical-diff
        # refusal quotes nor build a refusal streak, while their dispatched
        # wave (paid=True on the merged row) still counts toward the ceiling.
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
    follow-up task is a FRESH root with its own ceiling. This makes the
    refusal's "leave the remaining work to a follow-up task with its own
    budget" exit real; the cost is that a follow-up can buy new paid cycles
    for the same goal. The cross-task identical-fingerprint refusal remains
    the anti-laundering backstop: byte-identical bytes stay refused whichever
    task resubmits them."""
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        metadata.get("root_task_id")
        or getattr(ctx, "root_task_id", "")
        or getattr(ctx, "task_id", "")
        or ""
    )


def commit_review_contract_fingerprint() -> str:
    """Identity of the commit gate's live review contract (Q22): the one wave's
    seat roster + routes, the ``parts`` each seat is asked (one brief, two parts),
    enforcement, and the shipped prompt-contract text — contract A, contract B
    and the required-source policy version. A changed fingerprint lapses
    free-refusal/replay authority (a new paid review is allowed and refusals
    never quote across the change). Fail-open "" — an unknown contract never
    matches, so nothing is refused on it. The per-row efforts below are the
    RESOLVED efforts (``row_effort`` uses a compound route's encoded effort before
    the configured surface default). Changing a global effort therefore lapses
    only rows that actually inherit it (synthesis F4 — pinned by test)."""
    try:
        from ouroboros.config import get_review_enforcement
        from ouroboros.reviewer_slot_config import commit_triad_delivery
        from ouroboros.tools.review_admission import seat_vectors
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
        # What each seat is ASKED is contract identity: a seat that gains or
        # loses the coupling question is a different review.
        parts = {str(slot_id or ""): list(seat_parts) for slot_id, seat_parts in zip(row_plan["slot_ids"], row_plan["parts"])}
        # Both answer contracts and the required-source policy are part of what a
        # reviewer is asked and owed, so a change to any of them lapses recorded
        # free-replay authority instead of surviving it. Governance-document
        # CONTENTS stay out (docs/development 05).
        prompt_contract = hashlib.sha256(
            "\n".join([
                REVIEW_PREAMBLE, CRITICAL_FINDING_CALIBRATION, REVIEW_JSON_ARRAY_CONTRACT,
                REVIEW_TWO_PART_OBJECT_CONTRACT, SCOPE_REQUIRED_SOURCES_POLICY,
            ]).encode("utf-8")
        ).hexdigest()
        payload = json.dumps(
            {
                "rows": rows,
                "parts": parts,
                "enforcement": str(get_review_enforcement() or ""),
                "prompt_contract": prompt_contract,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        log.debug("commit review contract fingerprint unavailable (fail-open)", exc_info=True)
        return ""


def _quote_verdict_attempt(item: Any) -> str:
    """Render the recorded verdict an identical resubmission is refused with."""
    lines = [
        f"Recorded verdict: attempt #{int(getattr(item, 'attempt', 0) or 0)} "
        f"({getattr(item, 'ts', '') or 'unknown ts'}, block_reason="
        f"{getattr(item, 'block_reason', '') or 'unknown'}"
        + (f", rebuttal_sha256={str(getattr(item, 'rebuttal_sha256', '') or '')[:12]}…" if getattr(item, "rebuttal_sha256", "") else "")
        + ")"
    ]
    findings = [f for f in (getattr(item, "critical_findings", None) or []) if isinstance(f, dict)]
    for finding in findings[:5]:
        label = str(finding.get("item") or finding.get("reason") or "?")
        reason = _truncate_review_reason(str(finding.get("reason", "") or ""), limit=200)
        lines.append(f"  - [CRITICAL] {label}: {reason}")
    if len(findings) > 5:
        lines.append(f"  … and {len(findings) - 5} more critical finding(s) in review_status.")
    details = str(getattr(item, "block_details", "") or "").strip()
    if details and not findings:
        lines.append(_truncate_review_reason(details, limit=600))
    return "\n".join(lines)


def _walk_identical_verdict_streak(
    attempts: List[Any], fp: str, contract_fingerprint: str
) -> tuple[Optional[Any], set]:
    """Trailing verdict-block streak for ``fp``: ``(last_verdict_row,
    rebuttal_hashes_seen)``. ``(None, …)`` = no live streak. Skips in-flight
    rows, refusal records, preflight facts and infra-blocks (none of them is a
    verdict or evidence the diff changed); breaks on any other terminal. A
    verdict row recorded under a DIFFERENT (or unknown) review contract lapses
    the streak (Q22)."""
    last_verdict: Optional[Any] = None
    seen_rebuttals: set = set()
    for item in reversed(attempts):
        status = str(getattr(item, "status", "") or "")
        if status == "reviewing":
            continue  # in-flight marker, not a verdict
        if str(getattr(item, "block_reason", "") or "") in _REFUSAL_BLOCK_REASONS:
            # Free refusals never reset the streak. Their recorded rebuttal
            # hashes are deliberately NOT harvested: a "spent" rebuttal is one
            # that BOUGHT a dispatch — one refused without dispatching stays
            # fresh (e.g. after the owner raises the ceiling).
            continue
        klass = attempt_block_class(item)
        if status == "blocked" and not klass:
            # Preflight facts (stale advisory, tests, protection) inherit the
            # prior fingerprint through the ledger merge; they are neither a
            # review verdict nor evidence the diff changed.
            continue
        if klass == BLOCK_CLASS_INFRA:
            continue  # infra facts never build NOR break the streak
        if status == "failed" and str(getattr(item, "phase", "") or "") in _INFRA_FAILURE_PHASES:
            # Infra failures (lock/stage errors, expired reviewing rows) are
            # transients, not verdicts and not evidence the diff changed —
            # mirroring the ceiling's dispatch accounting, they neither build
            # nor reset the streak (adversarial wave, machine-2).
            continue
        if (
            status == "blocked"
            and klass == BLOCK_CLASS_VERDICT
            and str(getattr(item, "pre_review_fingerprint", "") or "") == fp
        ):
            row_contract = str(getattr(item, "review_contract_fingerprint", "") or "")
            if not contract_fingerprint or row_contract != contract_fingerprint:
                if last_verdict is None:
                    # The streak's HEAD was recorded under another (or unknown)
                    # contract: replay authority lapses, a paid review is due.
                    return None, seen_rebuttals
                # An OLDER row from a previous contract merely ends the streak;
                # the newer same-contract verdict keeps its refusal authority.
                break
            if last_verdict is None:
                last_verdict = item
            # A rebuttal is "spent" only when it BOUGHT this dispatched,
            # verdict-answered wave (machine-4/wording-2): harvest hashes from
            # paid verdict rows only — never from refusal rows or infra facts.
            rebuttal = str(getattr(item, "rebuttal_sha256", "") or "")
            if rebuttal and bool(getattr(item, "paid", False)):
                seen_rebuttals.add(rebuttal)
            continue
        break  # success / pass / different diff / post-review terminal
    return last_verdict, seen_rebuttals


def check_identical_verdict_refusal(
    ctx: ToolContext,
    fingerprint: str,
    *,
    rebuttal_sha256: str = "",
    contract_fingerprint: str = "",
) -> str:
    """Free typed refusal for a byte-identical resubmission whose streak's last
    terminal is a review VERDICT block and no NEW rebuttal is supplied; ""
    allows the attempt. Fires from the FIRST verdict-block — identical bytes
    are never re-reviewed for pay. Deliberately NOT task-scoped: the
    byte-identical diff is the identity, so a new task with the same unchanged
    diff cannot launder a fresh paid review (anti-laundering). Fail-open on
    ledger errors — this is a cost guard, not a safety gate."""
    fp = str(fingerprint or "").strip()
    if not fp:
        return ""
    try:
        from ouroboros.review_state import load_state, make_repo_key

        state = load_state(pathlib.Path(ctx.drive_root))
        attempts = state.filter_attempts(
            repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
            tool_name=_current_review_tool_name(ctx),
        )
        last_verdict, seen_rebuttals = _walk_identical_verdict_streak(
            attempts, fp, str(contract_fingerprint or "")
        )
        if last_verdict is None:
            return ""
        if rebuttal_sha256 and rebuttal_sha256 not in seen_rebuttals:
            return ""  # a NEW rebuttal buys exactly ONE paid re-review
        repeated_note = (
            "\nThe supplied review_rebuttal is byte-identical to one already spent on this "
            "streak — a repeated rebuttal does not buy another review."
            if rebuttal_sha256 else ""
        )
        return (
            "⚠️ IDENTICAL_DIFF_REFUSED: this exact staged diff was already reviewed and "
            "BLOCKED — you appear to have forgotten to change anything. Identical bytes are "
            f"never re-reviewed for pay.{repeated_note}\n"
            f"{_quote_verdict_attempt(last_verdict)}\n"
            "Honest exits: change the code (any change to the staged diff starts a fresh "
            "paid review); supply a NEW review_rebuttal with genuinely new evidence (buys "
            "exactly one paid re-review); escalate the disagreement to the owner; or "
            "finalize honestly without this commit."
        )
    except Exception:
        log.debug("identical-verdict refusal check failed (fail-open)", exc_info=True)
        return ""


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

    state = load_state(pathlib.Path(ctx.drive_root))
    attempts = state.filter_attempts(
        repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
        tool_name=_current_review_tool_name(ctx),
    )
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
        "Honest exits: finalize honestly with what is already reviewed and committed; "
        "escalate to the owner (the ceiling is the owner's Max Review Cycles setting — "
        "3/5/unlimited are one settings change away); or leave the remaining work to a "
        "follow-up task with its own budget. A rebuttal cannot buy past the ceiling — "
        "rebuttal cycles count toward it."
    )
    return {"message": message, "cycles_paid": paid, "cap": cap}


def resolve_commit_review_reference(ctx: ToolContext, reference: Any, *, state: Any = None) -> Any:
    """Resolve the exact prior outcome returned by this task's commit surface."""
    from ouroboros.config import get_review_enforcement
    from ouroboros.review_state import _load_state_unlocked, make_repo_key

    if review_enforcement_blocks(get_review_enforcement()):
        raise ValueError("author continuation requires Advisory enforcement")
    if not isinstance(reference, dict) or reference.get("surface") != "commit":
        raise ValueError("review_reference must name a returned commit review")
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    if reference.get("repo_key") != repo_key or reference.get("task_id") != str(getattr(ctx, "task_id", "") or ""):
        raise ValueError("review_reference belongs to another task or repository")
    if reference.get("tool_name") not in {"commit_reviewed", "vcs_commit_reviewed"} or type(reference.get("attempt")) is not int:
        raise ValueError("review_reference has no exact commit attempt")
    state = state if state is not None else _load_state_unlocked(pathlib.Path(ctx.drive_root), strict_attempt_authority=True)
    source = state.latest_attempt_for(repo_key=repo_key, task_id=reference["task_id"],
        tool_name=reference["tool_name"], attempt=reference["attempt"])
    if (source is None or source.phase not in {"review_only", "late_wait"}
            or not source.pre_review_fingerprint or source.pre_review_fingerprint != reference.get("pre_review_fingerprint")
            or source.raw_stripped):
        raise ValueError("the returned review outcome is missing or no longer has its exact evidence")
    from ouroboros.review_records import review_outcome_received

    if review_enforcement_blocks("blocking") and not review_outcome_received(
        [*source.triad_raw_results, source.scope_raw_result],
        findings=[*source.critical_findings, *source.advisory_findings],
        terminal=source.phase == "review_only" and source.status == "reviewed",
    ):
        raise ValueError("author continuation needs received feedback or a terminal unavailable outcome; reviewers are still running")
    return source


def _record_commit_attempt(
    ctx: ToolContext,
    commit_message: Any = None,
    status: Optional[str] = None,
    **legacy_kwargs: Any,
) -> None:
    """Record a commit attempt; supports positional or keyword commit_message/status."""
    strict = bool(legacy_kwargs.pop("_strict", False))
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
        if status == "blocked" and critical_findings:
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
            except Exception as _synth_exc:
                log.debug("review_synthesis: pre-lock synthesis skipped: %s", _synth_exc)
                _findings_for_attempt = critical_findings

        # C9.3: resolve semantic-dedup redirects for free-text (bug_*/risk_*) obligations
        # from a PRE-LOCK snapshot — the light-model call must stay OUTSIDE the review
        # state lock. Fail-open: any failure yields no redirect (a finding opens a new
        # obligation) and never blocks the gate. Only blocked attempts mint obligations.
        _obligation_redirects: Dict[str, str] = {}
        if status == "blocked" and _findings_for_attempt:
            try:
                from ouroboros.review_state import (
                    compute_obligation_semantic_redirects,
                    load_state as _ls_dedup,
                )
                _obligation_redirects = compute_obligation_semantic_redirects(
                    _ls_dedup(dr), _findings_for_attempt, repo_key=repo_key, drive_root=dr
                )
            except Exception as _dedup_exc:
                log.debug("obligation semantic dedup skipped: %s", _dedup_exc)
                _obligation_redirects = {}

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
                if not _attempt_accepts_reviewing_update(existing):
                    attempt_no = state.next_attempt_number(repo_key, tool_name, task_id)
                    existing = None
                ctx._current_review_attempt_number = attempt_no
            elif attempt_no <= 0:
                existing = state.latest_attempt_for(
                    repo_key=repo_key,
                    tool_name=tool_name,
                    task_id=task_id,
                )
                if existing and existing.status == "reviewing" and not existing.finished_ts:
                    attempt_no = int(existing.attempt or 0)
                else:
                    attempt_no = state.next_attempt_number(repo_key, tool_name, task_id)
                    # A NEW attempt inherits NOTHING from the previous terminal
                    # row (mirrors the reviewing branch above). Leaking the
                    # prior attempt's fields here let every fresh preflight
                    # record inherit paid=True/block_class/fingerprints from
                    # the last real review — inflating the paid-cycle count
                    # on every free refusal (found by the F1 eviction test).
                    existing = None
                ctx._current_review_attempt_number = attempt_no
            else:
                existing = state.latest_attempt_for(
                    repo_key=repo_key,
                    tool_name=tool_name,
                    task_id=task_id,
                    attempt=attempt_no,
                )

            from ouroboros.review_records import validate_author_disposition
            from ouroboros.config import get_review_enforcement

            author_record = getattr(existing, "author_disposition", {}) or {}
            if author_disposition is not None:
                subject = pre_review_fingerprint or str(getattr(existing, "pre_review_fingerprint", "") or "")
                author_record = validate_author_disposition(author_disposition, subject_hash=subject) or {}
                cyber = not review_enforcement_blocks("blocking")
                reference = author_record.get("review_reference")
                try:
                    source = resolve_commit_review_reference(ctx, reference, state=state) if reference else None
                except ValueError:
                    source = None
                same_review = bool(existing and getattr(existing, "paid", False) and subject == existing.pre_review_fingerprint)
                if (not subject or (not cyber and not (source or same_review))
                        or (post_review_fingerprint and post_review_fingerprint != subject)
                        or review_enforcement_blocks(get_review_enforcement())
                        or (not cyber and author_record.get("enforcement") != "advisory")):
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
            if status != "reviewing" and "late_result_pending" not in legacy_kwargs and not review_enforcement_blocks("blocking"):
                attempt.late_result_pending = bool(getattr(existing, "late_result_pending", False)) or _attempt_has_active_review_custody(attempt)
            stamp_paid_review_owner(attempt, paid=bool(paid))
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
                    source = _continuation_source(status, late_result_pending=late_result_pending)
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
        _REVIEW_ATTEMPT_GRACE_SEC,
        _REVIEW_ATTEMPT_TTL_SEC,
        make_repo_key,
        update_state,
        _utc_now,
    )
    from ouroboros.tool_capabilities import REVIEWED_MUTATIVE_TOOLS

    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    expiration_window = _REVIEW_ATTEMPT_TTL_SEC + _REVIEW_ATTEMPT_GRACE_SEC
    ctx._review_resume_pending = False
    ctx._pending_review_attempt = None
    ctx._review_cyber_pending = ""

    def _mutate(state):
        state.expire_stale_attempts(now_ts=_utc_now())
        return [
            item for item in state.get_active_attempts(repo_key=repo_key)
            if item.tool_name in REVIEWED_MUTATIVE_TOOLS
        ]

    try:
        active_attempts = update_state(pathlib.Path(ctx.drive_root), _mutate)
    except Exception as e:
        log.warning("Failed to check overlapping review attempts: %s", e)
        if not review_enforcement_blocks("blocking"):
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
    if not active_attempts:
        return None
    if not review_enforcement_blocks("blocking"):
        ctx._review_cyber_pending = (
            "Existing review custody remains active: "
            + ", ".join(f"{item.tool_name}#{item.attempt}" for item in active_attempts)
            + ". No new reviewer will be dispatched; the original attempts remain collectible."
        )
        return None

    task_id = str(getattr(ctx, "task_id", "") or "")
    tool_name = _current_review_tool_name(ctx)
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


def commit_preflight_choice_error(selector: str, *, skipped: bool, continuation: bool) -> str:
    """A commit's preflight is a named row, an explicit skip, or neither; an author
    continuation dispatches no critic, so it names no row either ("" = the choice stands)."""
    if not selector:
        return ""
    if skipped:
        return "preflight_reviewer and skip_advisory_review=True are opposite choices; pass one."
    if continuation:
        return "preflight_reviewer and an author continuation are separate choices (the continuation dispatches no critic)."
    return preflight_reviewer_error(selector)


def compose_commit_panel(ctx: ToolContext, reviewers: Sequence[str], reason: str) -> Any:
    """This commit's panel through the ONE composer ``review_change`` uses
    (``review_change.compose_panel``, decision 1A). In Cyber Pro the marked pool rows are
    the owner's menu: ``reviewers`` names the counted seats (an enabled catalog row outside
    the pool is an added critic, never the quorum) and ``reason`` says why; below Cyber Pro
    the whole pool judges and ``reviewers`` only add critics outside it. No ``reviewers`` is
    ``None``: the configured pool exactly as the gate reads it today (an empty pool stays the
    gate's own typed ``pool_empty`` refusal). The runtime mode is the gate's own reading (the
    ``mode`` fact of its record). Raises ``ReviewChangeArgumentError`` — before anything is
    staged, reviewed or recorded."""
    from ouroboros.runtime_mode_policy import runtime_mode_at_least
    from ouroboros.tools import git as git_mod
    from ouroboros.tools import review_change as rc

    request = rc.parse_request({"root": "system_repo", "subject": "index", "reviewers": list(reviewers), "reason": reason})
    if not request.reviewers:
        return None
    return rc.compose_panel(request, adds_only=not runtime_mode_at_least(git_mod._current_runtime_mode(), "cyber_pro"))


def release_diagnostics(ctx: ToolContext, paths: Optional[List[str]], source: str) -> Dict[str, Any]:
    """``preflight_review(deterministic_only=True)``: every release-metadata finding of the
    worktree or the index, with no sync, staging, tests, provider or review state."""
    if source not in ("worktree", "index"):
        return {"status": "error", "failure_code": "PREFLIGHT_SOURCE_REQUIRED",
                "message": "deterministic_only requires explicit source=worktree or source=index."}
    from ouroboros import body_candidate
    from ouroboros.commit_admission import release_metadata_diagnostics

    return {**release_metadata_diagnostics(
        ctx.repo_dir, paths, source=source, neutral_allowed=True if source == "index" else body_candidate.is_bound(ctx)),
            "deterministic_only": True, "review_freshness": False}


def deterministic_preflight(ctx: ToolContext, commit_message: str, paths: Optional[List[str]]) -> str:
    """The free checks that ran ahead of the retired advisory delivery, kept whole for every
    commit: size headroom and readiness (information and warnings), release metadata of the
    staged index, and the syntax of staged ``.py`` files. Returns the blocking message or ""."""
    from ouroboros.commit_admission import release_metadata_preflight, syntax_preflight_staged_py_files
    from ouroboros.tools.review_helpers import check_worktree_readiness
    from ouroboros.utils import append_jsonl, utc_now_iso

    repo_dir = pathlib.Path(ctx.repo_dir)
    information: List[str] = []
    warnings = check_worktree_readiness(repo_dir, paths=paths, information=information)
    if information:
        ctx.emit_progress_fn("Size headroom (information):\n" + "\n".join(information))
    if warnings:
        ctx.emit_progress_fn(f"⚠️ Readiness: {'; '.join(warnings)}")
        try:
            append_jsonl(pathlib.Path(ctx.drive_root) / "logs" / "events.jsonl", {
                "ts": utc_now_iso(), "type": "commit_readiness_gate", "warnings": warnings,
                "task_id": str(getattr(ctx, "task_id", "") or "")})
        except Exception:
            pass
    return (release_metadata_preflight(repo_dir, commit_message, paths, source="index")
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
                                      review_rebuttal=review_rebuttal)
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


def _return_commit_feedback(ctx: ToolContext, message: str, started: float, before: dict, after: dict,
                            *, pending: bool = False, reason: str = "author_decision_required", findings: Optional[list] = None) -> dict:
    """Return material criticism before Git effects, preserving its original attempt."""
    from dataclasses import asdict
    import time
    from ouroboros.tools import git as git_mod
    from ouroboros.review_state import load_state, make_repo_key

    if not pending:
        git_mod._record_commit_attempt(ctx, message, "reviewed", phase="review_only", block_reason=reason,
            duration_sec=time.time() - started, pre_review_fingerprint=before["fingerprint"],
            post_review_fingerprint=after.get("fingerprint", ""), fingerprint_status="matched",
            critical_findings=findings if findings is not None else getattr(ctx, "_last_review_critical_findings", []),
            advisory_findings=getattr(ctx, "_review_advisory", []),
            triad_raw_results=getattr(ctx, "_last_triad_raw_results", []), scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
            degraded_reasons=getattr(ctx, "_review_degraded_reasons", []))
    state = load_state(pathlib.Path(ctx.drive_root))
    row = state.latest_attempt_for(repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
        task_id=str(getattr(ctx, "task_id", "") or ""), tool_name="commit_reviewed", attempt=ctx._current_review_attempt_number)
    reference = {"surface": "commit", **{key: getattr(row, key) for key in ("repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint")},
                 "review_record_id": str(getattr(row, "review_record_id", "") or "")}
    choice = ("Inspect the outcome, then revise/request review, stop, or explicitly continue in Advisory using the same "
              "commit tool with review_reference and author_disposition {disposition: accepted|rejected|partial|deferred, "
              "rationale: ...}. Author continuation buys no reviewer cycle. ")
    try:
        resolve_commit_review_reference(ctx, reference, state=state)
    except ValueError:
        choice = "Reviewers are still running without feedback. Collect the existing wave or stop; informed author continuation is not available yet. "
    if not pending:
        git_mod.run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
    text = ("Review outcome returned before commit. " + choice +
            "Current files are preserved; no commit, tag or push occurred.\n" + json.dumps({"review_reference": reference,
            "review_outcome": asdict(row)}, ensure_ascii=False, default=str))
    return {"status": "reviewed", "message": text, "review_reference": reference,
            "pre_fingerprint": before, "post_fingerprint": after}


def disclose_commit_review_replay(ctx: ToolContext, replay: dict) -> None:
    """Keep the free replay cause separate from authority to create a commit."""
    replay_reason = str(replay.get("replay_reason") or "")
    if not review_enforcement_blocks("blocking"):
        progress_note = "Cyber Pro: continuing without a new reviewer dispatch; original review facts are retained."
    elif replay_reason == IDENTICAL_DIFF_BLOCK_REASON:
        progress_note = (
            "Max Review Cycles: identical staged diff — reusing the recorded "
            "review verdict, no paid review-wave dispatch."
        )
    else:
        progress_note = (
            "Max Review Cycles: paid-cycle ceiling exhausted — no review outcome "
            "exists for this diff; inspect the returned outcome before explicitly "
            "choosing Advisory author continuation."
        )
    disclosure = (
        "Review enforcement=Advisory: no new review wave was bought for "
        f"this commit ({replay_reason}); no fresh automatic preflight was bought. "
        + str(replay.get("advisory_replay") or "")
    )
    if not review_enforcement_blocks("blocking"):
        disclosure = "Cyber Pro: proceeding without a new review. " + str(replay.get("advisory_replay") or "")
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
    from ouroboros.config import get_review_enforcement

    author = build_author_disposition_from_mapping(ctx._author_commit_decision, subject_hash=pre_fingerprint["fingerprint"],
        reviewer_signal=author_source.block_reason or author_source.status, enforcement=get_review_enforcement())
    author["review_reference"] = ctx._author_commit_reference
    ctx._author_commit_record = author
    staged = git_mod.run_cmd(["git", "diff", "--cached", "--name-status"], cwd=ctx.repo_dir)
    return _preflight_check(commit_message, format_name_status_for_preflight(staged), ctx.repo_dir)


def record_bound_commit_success(ctx: ToolContext, commit_message: str, started_at: float, before: dict, after: dict) -> None:
    """Persist the already-verified Git effect with its actual review/author facts."""
    import time
    from ouroboros.tools import git as git_mod
    git_mod._record_commit_attempt(ctx, commit_message, "succeeded",
                           author_disposition=getattr(ctx, "_author_commit_record", None),
                           duration_sec=time.time() - started_at,
                           phase="commit",
                           pre_review_fingerprint=before.get("fingerprint", ""),
                           post_review_fingerprint=after.get("fingerprint", ""),
                           fingerprint_status="matched",
                           triad_models=getattr(ctx, "_last_triad_models", []),
                           scope_model=getattr(ctx, "_last_scope_model", ""),
                           triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                           scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                           degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []))


def prepare_author_commit_request(ctx: ToolContext, review_reference: Any, author_disposition: Any, review_rebuttal: str) -> Optional[str]:
    """Validate the explicit free continuation before touching a Git candidate."""
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
    if review_reference is not None or author_disposition is not None:
        from ouroboros.review_records import build_author_disposition_from_mapping
        try:
            if review_rebuttal:
                raise ValueError("a paid rebuttal and free author continuation are separate choices")
            build_author_disposition_from_mapping(author_disposition, subject_hash="pending-current-candidate")
            ctx._author_commit_source = resolve_commit_review_reference(ctx, review_reference)
            ctx._author_commit_reference, ctx._author_commit_decision = dict(review_reference), dict(author_disposition)
        except (OSError, ValueError) as exc:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=f"ERROR: REVIEW_AUTHOR_CONTINUATION_INVALID: {exc}"))
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
    from ouroboros.config import get_review_enforcement
    from ouroboros.review_records import ReviewRequest, resolve_review_wave
    from ouroboros.tools import git as git_mod

    task_id = str(getattr(ctx, "task_id", "") or "")
    enforcement = str(get_review_enforcement() or "")
    # THIS wave's own execution rows (stashed by the substrate as it recorded them), never the
    # shared last-execution projection another surface may have overwritten meanwhile.
    executions = dict(getattr(ctx, "_last_review_slot_executions", {}) or {})
    try:
        mode = str(git_mod._current_runtime_mode() or "")
    except Exception:
        mode = ""
    tests_passed = getattr(ctx, "_preflight_tests_passed", None)
    pre_fingerprint = pre_fingerprint or {}
    panel = commit_panel_facts(ctx)
    return {
        **_review_body_facts(ctx),
        "composition": str(panel.get("composition") or "full_pool"),
        "composition_reason": str(panel.get("reason") or ""), "chosen_by": str(panel.get("chosen_by") or "owner"),
        "task_id": task_id, "root_task_id": resolve_root_task_id(ctx),
        "review_wave_id": resolve_review_wave(ReviewRequest(
            surface="commit_gate", goal=goal or commit_message, task_id=task_id,
            retry_key=str(getattr(ctx, "_current_review_retry_key", "") or "")), {}, ""),
        "repo_dir": str(ctx.repo_dir), "goal": goal, "scope": scope, "commit_message": commit_message,
        "binding": dict(pre_fingerprint.get("binding") or {}),
        "binding_fingerprint": str(pre_fingerprint.get("fingerprint") or ""),
        "review_contract_fingerprint": str(getattr(ctx, "_current_review_contract_fingerprint", "") or ""),
        "rebuttal_sha256": str(getattr(ctx, "_current_review_rebuttal_sha256", "") or ""),
        "enforcement": enforcement, "mode": mode, "enforcement_blocks": bool(review_enforcement_blocks(enforcement)),
        "structured": dict(getattr(ctx, "_last_review_structured", {}) or {}), "slot_executions": executions,
        "triad_raw": list(getattr(ctx, "_last_triad_raw_results", []) or []),
        "blocked": bool(blocked), "block_reason": str(block_reason or ""),
        "dispatch_refusal": dispatch_refusal, "pending": bool(pending),
        "degraded_reasons": list(getattr(ctx, "_review_degraded_reasons", []) or []),
        "critical_findings": list(combined_findings or getattr(ctx, "_last_review_critical_findings", []) or []),
        "advisory_findings": list(getattr(ctx, "_last_review_advisory_findings", []) or []),
        "tests": _review_tests_facts(ctx, tests_passed),
        "preflight": _review_preflight_facts(ctx),
    }


def name_review_record(ctx: ToolContext, result: Any) -> Any:
    """Every outcome of a commit call that wrote a review record names it (DEVELOPMENT 05):
    passed, blocked, pending, refused after the wave, or failed at the commit itself. The id
    is this call's own (reset at the start of every call); ID-less exits stay ID-less."""
    record_id = str(getattr(ctx, "_current_review_record_id", "") or "")
    if not record_id or not isinstance(result, str) or record_id in result:
        return result
    from ouroboros.tools.tool_result import append_published_text

    return append_published_text(ctx, result, f"\nreview_record_id: {record_id}")


def _review_tests_facts(ctx: ToolContext, tests_passed: Any) -> Dict[str, Any]:
    """``passed`` only for THIS candidate: the runner's flag is process state that outlives
    the checkout it tested, so it counts only when the process-held test proof still covers
    the current tree/index/workload (``commit_admission.PreflightTestProof``); a skipped
    or stale run is ``NOT_RUN`` with its reason, never a passed result borrowed from an
    earlier candidate."""
    if tests_passed is not True:
        return {"policy": "NOT_RUN", "result": "unknown"}
    try:
        from ouroboros.commit_admission import preflight_test_proof_matches

        bound = bool(preflight_test_proof_matches(ctx, ctx.repo_dir))
    except Exception:
        bound = False
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
    replay (identical diff, exhausted cycles, pending custody) and a wave the budget
    fence declined are ``NOT_DISPATCHED`` records naming their cause. An explicit author
    continuation buys no record: it notes the author's decision on the record it answered.
    Ledger failure never changes the gate's decision; it is logged and the id stays ""."""
    from ouroboros import review_ledger as ledger
    from ouroboros.tools import git as git_mod

    ctx._current_review_record_id = ""
    try:
        root = ledger.ledger_root(ctx)
        if author_source is not None:
            prior = str(getattr(author_source, "review_record_id", "") or "")
            if prior and ledger.load_record(root, prior) is not None:
                ledger.note_author_decision(root, prior, dict(getattr(ctx, "_author_commit_record", None) or {}))
                ctx._current_review_record_id = prior
            return ctx._current_review_record_id
        structured = dict(getattr(ctx, "_last_review_structured", {}) or {})
        refusal = None
        if advisory_replay is not None:
            refusal = {"kind": str(advisory_replay.get("replay_reason") or "free_replay"),
                       "message": str(advisory_replay.get("advisory_replay") or "")}
        elif str(getattr(ctx, "_last_review_block_reason", "") or "") == "review_wave_budget_insufficient":
            refusal = {"kind": "review_wave_budget_insufficient", "message": str(structured.get("wave_refusal") or "")}
        elif str(getattr(ctx, "_last_review_block_reason", "") or "") == REVIEW_POOL_EMPTY_REASON:
            # An empty pool dispatches nothing under either enforcement: the record
            # names that cause itself, not only the gate's block.
            refusal = {"kind": REVIEW_POOL_EMPTY_REASON, "message": REVIEW_POOL_EMPTY_SENTENCE}
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
        if existing is not None and existing.get("state") == ledger.STATE_PENDING:
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
            record.panel = {**dict(record.panel or {}), **commit_panel_facts(ctx)}
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


def record_commit_gate_refusal(ctx: ToolContext, commit_message: str, *, goal: str = "", scope: str = "",
                               pre_fingerprint: Optional[dict] = None, kind: str, message: str) -> str:
    """A blocking free refusal before any dispatch (identical diff, exhausted cycles) is a
    ``NOT_DISPATCHED`` ledger record: the owner sees WHY nothing was reviewed."""
    return settle_commit_review_ledger(
        ctx, commit_message, goal=goal, scope=scope, pre_fingerprint=pre_fingerprint,
        blocked=True, block_reason=kind, advisory_replay={"replay_reason": kind, "advisory_replay": message})
