"""Advisory freshness gate, durable commit-attempt recording, and the
commit-side Max-Review-Cycles machinery (block classification, the free
identical-diff refusal, the per-root-task paid-cycle ceiling, and the
review-contract fingerprint)."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED, review_max_cycles
from ouroboros.review_owner_custody import stamp_paid_review_owner
from ouroboros.review_state import (
    _attempt_has_active_review_custody,
    infer_review_phase,
)
from ouroboros.tools.registry import ToolContext
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
# the advisory-freshness gate and before any paid triad+scope dispatch —
# quoting the recorded verdict. A rebuttal is identified by CONTENT
# (sha256): a hash new to the current identical-fingerprint streak buys
# exactly ONE paid re-review; a repeated hash is refused free, quoting the
# previous outcome. Infra-blocks (fit overflow, sub-floor window,
# revalidation, transport/no-quorum) are not verdicts: they never build the
# refusal streak and retry freely. The shared OUROBOROS_REVIEW_MAX_CYCLES
# knob (``review_max_cycles()``; ``None`` = unlimited) bounds PAID
# triad+scope cycles per ROOT task (the whole task tree shares one ceiling;
# a manual session is its own task; a follow-up task starts a fresh one).
# The ceiling counts MONEY: every attempt that physically dispatched a wave
# (``paid`` recorded at dispatch) counts regardless of how it terminated —
# only UNDISPATCHED attempts (preflight refusals, assembly failures, free
# replays) are outside the count. Exhaustion is a
# free typed refusal plus ``emit_review_cycles_exhausted``. Both refusals
# honor the recorded review-contract fingerprint (roster+routes+enforcement+
# prompt contract): a changed contract lapses the streak. Under ADVISORY
# enforcement neither refusal hard-blocks a commit — the prior verdict is
# reused, loudly disclosed, and the commit proceeds without buying another
# review.
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
    """Identity of the commit gate's live review contract (Q22): triad roster+
    routes, scope rows, enforcement, and the shipped prompt-contract text. A
    changed fingerprint lapses free-refusal/replay authority (a new paid
    review is allowed and refusals never quote across the change). Fail-open
    "" — an unknown contract never matches, so nothing is refused on it.
    The per-row efforts below are the RESOLVED efforts (``row_effort`` /
    ``scope_reviewer_slots`` use a compound route's encoded effort before the
    configured surface default). Changing a global effort therefore lapses
    only rows that actually inherit it (synthesis F4 — pinned by test)."""
    try:
        from ouroboros.config import get_review_enforcement
        from ouroboros.review_substrate import scope_reviewer_slots
        from ouroboros.reviewer_slot_config import commit_triad_delivery
        from ouroboros.tools.review_helpers import CRITICAL_FINDING_CALIBRATION, REVIEW_PREAMBLE
        from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

        row_plan = commit_triad_delivery()
        triad_rows = [
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
        triad_actor_ids = [str(a or "") for a in (row_plan.get("subagent_ids") or [])]
        if any(triad_actor_ids):
            for row, actor in zip(triad_rows, triad_actor_ids):
                row.append(actor)
        scope_slots = list(scope_reviewer_slots())
        scope_rows = [
            [
                str(getattr(slot, "slot_id", "") or ""),
                str(getattr(slot, "model", "") or ""),
                str(getattr(getattr(slot, "route", None), "value", "") or ""),
                str(getattr(slot, "session_target", "") or ""),
                str(getattr(slot, "session_profile", "") or ""),
                str(getattr(slot, "effort", "") or ""),
            ]
            for slot in scope_slots
        ]
        scope_actor_ids = [str(getattr(slot, "subagent_id", "") or "") for slot in scope_slots]
        if any(scope_actor_ids):
            for row, actor in zip(scope_rows, scope_actor_ids):
                row.append(actor)
        prompt_contract = hashlib.sha256(
            "\n".join([REVIEW_PREAMBLE, CRITICAL_FINDING_CALIBRATION, REVIEW_JSON_ARRAY_CONTRACT]).encode("utf-8")
        ).hexdigest()
        payload = json.dumps(
            {
                "triad": triad_rows,
                "scope": scope_rows,
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
        f"{paid} of {cap} paid triad+scope review cycle(s) "
        "(OUROBOROS_REVIEW_MAX_CYCLES). Refusing to buy another review.\n"
        "Honest exits: finalize honestly with what is already reviewed and committed; "
        "escalate to the owner (the ceiling is the owner's Max Review Cycles setting — "
        "3/5/unlimited are one settings change away); or leave the remaining work to a "
        "follow-up task with its own budget. A rebuttal cannot buy past the ceiling — "
        "rebuttal cycles count toward it."
    )
    return {"message": message, "cycles_paid": paid, "cap": cap}


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
        block_class = _req("block_class")
        rebuttal_sha256 = _req("rebuttal_sha256")
        paid = _req("paid", False)
        review_contract_fingerprint = _req("review_contract_fingerprint")
        review_retry_key = _req("review_retry_key")
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
            )
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
        )
    except Exception:
        pass


def _mark_review_attempt_late(
    ctx: ToolContext,
    *,
    soft_timeout_sec: int,
    duration_sec: float,
) -> None:
    warning = (
        f"Soft timeout exceeded {soft_timeout_sec}s; waiting for a possible late reviewed result."
    )
    _record_commit_attempt(
        ctx,
        commit_message=str(getattr(ctx, "_current_review_commit_message", "") or ""),
        status="reviewing",
        duration_sec=duration_sec,
        readiness_warnings=[warning],
        late_result_pending=True,
        phase="late_wait",
    )


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
        return (
            "⚠️ REVIEW_STATE_UNAVAILABLE: active paid-review custody could not "
            "be verified, so no reviewer dispatch was started. Retry after the "
            "review state store is readable."
        )
    if not active_attempts:
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
    """Classify producer facts only; candidate and owner admission stay separate."""
    return (
        facts.get("failure_phase") in {"context", "delivery", "format", "window_authority"}
        and facts.get("operation_state") not in {"in_flight", "custody_lost"}
        and not facts.get("pending_invocation_id") and not facts.get("late_result_pending")
    )


def _check_advisory_freshness(ctx: ToolContext, commit_message: str,
                              skip_advisory_pre_review: bool = False,
                              paths: Optional[List[str]] = None, *,
                              review_rebuttal: str = "",
                              decision: Optional[Dict[str, Any]] = None) -> Optional[str]:
    from ouroboros.review_state import AdvisoryRunRecord, compute_snapshot_hash, load_state, make_repo_key, update_state, _utc_now
    from ouroboros.config import get_review_enforcement
    from ouroboros.utils import append_jsonl
    drive_root = pathlib.Path(ctx.drive_root)
    repo_dir = pathlib.Path(ctx.repo_dir)
    repo_key = make_repo_key(repo_dir)
    enforcement = get_review_enforcement()

    snapshot_hash = compute_snapshot_hash(repo_dir, commit_message, paths=paths)
    state = load_state(drive_root)
    open_obs = state.get_open_obligations(repo_key=repo_key)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)

    matching_run = state.find_by_hash(snapshot_hash, repo_key=repo_key)
    same_rebuttal = compute_rebuttal_sha256(review_rebuttal) == compute_rebuttal_sha256(
        getattr(matching_run, "review_rebuttal", "")
    )
    fresh = state.is_fresh(snapshot_hash, repo_key=repo_key) and same_rebuttal
    if decision is not None:
        execution = getattr(matching_run, "execution", {}) or {}
        decision["pending"] = bool(execution.get("pending_invocation_id")) or execution.get("operation_state") in {"in_flight", "custody_lost"}
        decision["refresh_required"] = (
            not skip_advisory_pre_review and not fresh
            and str(getattr(matching_run, "status", "")) != "preflight_blocked"
        )

    def _render_obligations() -> list[str]:
        return [
            f"  [{o.obligation_id}] {o.item}: {_truncate_review_reason(o.reason, limit=80)}"
            for o in open_obs
        ]

    def _render_debts() -> list[str]:
        return [
            f"  [{debt.debt_id}] {debt.category}: {_truncate_review_reason(debt.summary, limit=80)}"
            for debt in open_debts
        ]

    if (matching_run is not None and matching_run.status in {"error", "parse_failure"}
            and same_rebuttal and enforcement == "advisory"
            and review_failure_is_technical(matching_run.execution)):
        from ouroboros.tools.review import _record_advisory_override

        warning = (
            f"Preflight {matching_run.status} ({matching_run.execution.get('failure_phase')}): "
            f"{matching_run.raw_result}\nReview enforcement=advisory permits continuing; "
            "this failed preflight is not a PASS. Its full source and findings remain recorded."
        )
        ctx._last_review_block_reason = "advisory_technical_failure"
        _record_advisory_override(ctx, warning)
        ctx._review_advisory = list(getattr(ctx, "_review_advisory", []) or []) + [warning, *matching_run.items]
        return None

    if fresh and not open_obs and not open_debts:
        return None

    if skip_advisory_pre_review:
        task_id = str(getattr(ctx, "task_id", "") or "")
        reason = "skip_advisory_review=True passed to commit_reviewed"
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": _utc_now(), "type": "advisory_review_bypassed",
                "snapshot_hash": snapshot_hash, "commit_message": commit_message,
                "bypass_reason": reason, "task_id": task_id,
            })
        except Exception:
            pass

        def _mutate(bypass_state):
            bypass_state.add_run(AdvisoryRunRecord(
                snapshot_hash=snapshot_hash,
                commit_message=commit_message,
                status="bypassed",
                ts=_utc_now(),
                bypass_reason=reason,
                bypassed_by_task=task_id,
                snapshot_paths=paths,
                repo_key=repo_key,
                tool_name="advisory_review",
                task_id=task_id,
            ))

        update_state(drive_root, _mutate)

        return None  # audited bypass

    if fresh and (open_obs or open_debts):
        if enforcement == "advisory":
            drive_logs = ctx.drive_logs() if callable(getattr(ctx, "drive_logs", None)) else drive_root / "logs"
            event = {
                "ts": _utc_now(),
                "type": "advisory_obligations_acknowledged",
                "snapshot_hash": snapshot_hash,
                "repo_key": repo_key,
                "open_obligations_count": len(open_obs),
                "open_debts_count": len(open_debts),
                "open_obligations": [
                    f"[{o.obligation_id}] {o.item}: {_truncate_review_reason(o.reason, limit=120)}"
                    for o in open_obs
                ],
                "open_debts": [
                    f"[{debt.debt_id}] {debt.category}: {_truncate_review_reason(debt.summary, limit=120)}"
                    for debt in open_debts
                ],
            }
            if append_jsonl(drive_logs / "events.jsonl", event):
                return None
        debt_parts = []
        if open_obs:
            debt_parts.append(f"{len(open_obs)} open obligation(s)")
        if open_debts:
            debt_parts.append(f"{len(open_debts)} commit-readiness debt item(s)")
        lines = [
            f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: Advisory is current (hash={snapshot_hash[:12]}) "
            f"but {' and '.join(debt_parts)} remain unresolved.\n"
        ]
        if open_obs:
            lines.append("Unresolved obligations:")
            lines += _render_obligations()
        if open_debts:
            lines.append("\nCommit-readiness debt:")
            lines += _render_debts()
        lines.append("\nFix the flagged issues and re-run preflight_review so it can verify them PASS.")
        lines.append("Or bypass: commit_reviewed(commit_message='...', skip_advisory_review=True) (audited).")
        return "\n".join(lines)

    matching_run = state.find_by_hash(snapshot_hash, repo_key=repo_key)
    scoped_runs = state.filter_advisory_runs(repo_key=repo_key)
    latest = scoped_runs[-1] if scoped_runs else None

    if matching_run and matching_run.status == "parse_failure":
        obs_section = ""
        if state.get_open_obligations(repo_key=repo_key):
            open_obs = state.get_open_obligations(repo_key=repo_key)
            obs_lines = [f"\nOpen obligations ({len(open_obs)}):"]
            obs_lines += [f"  [{o.obligation_id}] {o.item}: {_truncate_review_reason(o.reason, limit=80)}"
                          for o in open_obs]
            obs_section = "\n".join(obs_lines)
        return (
            f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: Last advisory run for this snapshot returned "
            f"parse_failure (hash={snapshot_hash[:12]}, ts={matching_run.ts}). "
            f"The advisory ran but its output could not be parsed — re-run it.{obs_section}\n"
            "Re-run: preflight_review(commit_message='...')\n"
            "Or bypass: commit_reviewed(commit_message='...', skip_advisory_review=True) (audited)."
        )

    if matching_run and matching_run.status == "preflight_blocked":
        preflight_detail = (matching_run.raw_result or "").strip()
        # H4 (capinv-447): the status is shared by several deterministic checks;
        # name the problem class only when the typed cause is recorded.
        reason_kind = str(getattr(matching_run, "reason_kind", "") or "")
        cause = {
            "syntax": "The advisory delivery was skipped because a staged `.py` file has a SyntaxError.",
            "release_metadata": "The advisory delivery was skipped because the deterministic release metadata preflight failed.",
        }.get(reason_kind, "The advisory delivery was skipped by a deterministic preflight check (exact cause below).")
        return (
            f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: Last advisory run for this snapshot "
            f"was blocked by a preflight check (hash={snapshot_hash[:12]}, "
            f"ts={matching_run.ts}). {cause}\n\n"
            f"{preflight_detail}\n\n"
            "Re-run after fixing: preflight_review(commit_message='...')"
        )

    if latest and latest.status == "stale" and state.last_stale_from_edit_ts:
        stale_reason = (f"Advisory invalidated by worktree edit at "
                        f"{state.last_stale_from_edit_ts}. Re-run advisory after all edits.")
    elif latest:
        stale_reason = (f"Latest run: status={latest.status}, hash={latest.snapshot_hash[:12]}, "
                        f"ts={latest.ts}. Snapshot changed (files edited after advisory ran).")
    else:
        stale_reason = "No advisory runs recorded yet."

    obs_section = ""
    if open_obs:
        lines = [f"\nOpen obligations ({len(open_obs)}):"]
        lines += _render_obligations()
        lines.append("  → preflight_review will verify each obligation is resolved.")
        obs_section = "\n".join(lines)
    debt_section = ""
    if open_debts:
        debt_lines = [f"\nCommit-readiness debt ({len(open_debts)}):"]
        debt_lines += _render_debts()
        debt_lines.append("  → clear or rebut these debt items before the next reviewed attempt.")
        debt_section = "\n".join(debt_lines)

    return (
        f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: No fresh advisory run found for this snapshot "
        f"(hash={snapshot_hash[:12]}).\n"
        f"{stale_reason}\n"
        f"{obs_section}{debt_section}\n\n"
        "Correct workflow:\n"
        "  1. Finish ALL edits first\n"
        "  2. preflight_review(commit_message='your message')       ← run AFTER all edits\n"
        "  3. commit_reviewed(commit_message='your message')       ← run IMMEDIATELY after advisory\n\n"
        "⚠️ Any edit after step 2 makes the advisory stale and requires re-running it.\n\n"
        "To bypass (will be durably audited):\n"
        "  commit_reviewed(commit_message='...', skip_advisory_review=True)"
    )
