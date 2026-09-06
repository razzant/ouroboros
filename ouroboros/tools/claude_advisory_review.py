"""Advisory pre-review gate.

Normally runs a cheap read-only advisory review through the configured route
before multi-model commit review. The LLM may instead choose the audited
advisory-only skip; tests, triad/scope review, exact-snapshot revalidation, and
final commit binding remain authoritative. Any edit after advisory makes it
stale.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import List, Optional

from ouroboros.triad_review import (
    empty_array_is_verified_clean,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    extract_json_array,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)
from ouroboros.skill_review_status import SEVERITY_DRIVEN_ITEMS  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.review_state import (
    AdvisoryRunRecord,
    AdvisoryReviewState,
    compute_snapshot_hash,
    load_state,
    make_repo_key,
    update_state,
    _utc_now,
)
from ouroboros.config import get_review_enforcement as _get_review_enforcement
from ouroboros.config import get_finalization_grace_sec
from ouroboros.deadline_utils import (
    dispatch_window_remaining_sec,
    owner_deadline_exhausted_for_context,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)
from ouroboros.tools.review_helpers import (
    build_advisory_changed_context,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    build_skill_host_context,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    build_blocking_findings_json_section,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    load_checklist_section,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    build_goal_section,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    build_scope_section,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    check_worktree_readiness,
    check_worktree_version_sync as _check_worktree_version_sync_shared,
    CRITICAL_FINDING_CALIBRATION,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    get_advisory_runtime_diagnostics as _get_runtime_diagnostics,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    format_advisory_error as _format_advisory_error,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    load_governance_doc,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    normalize_reviewer_obligation_id,
    strip_obligation_suffix,
    _run_review_preflight_tests,
    emit_review_event,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    emit_review_usage,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)
from ouroboros.utils import (
    append_jsonl,
    utc_now_iso,
    truncate_review_artifact as _truncate_review_artifact,
)
from ouroboros.review_evidence import build_review_projection, build_review_status_payload

log = logging.getLogger(__name__)

# Stable markers of the MANAGED oversize skips: both managed skip messages
# (the 500k delta gate and the prompt-size gate) carry _MANAGED_SKIP_NOTE, and
# _next_step_guidance matches it so the skipped branch never advises the
# impossible "split the commit" for a managed merge.
_MANAGED_SKIP_MARKER = "managed resolution review diff too large"
_MANAGED_SKIP_NOTE = "cannot be split into smaller commits"


ADVISORY_REVIEW_CHOICE_GUIDANCE = (
    "Normally the LLM runs the cheap preflight_review immediately before "
    "commit_reviewed. When advisory review is slow, unhealthy, unavailable, or "
    "low-value, the LLM may deliberately choose skip_advisory_review=True; the "
    "choice is durably audited. This skip bypasses only the requirements for "
    "advisory freshness, advisory obligations, and advisory debt; unresolved "
    "obligation and debt records remain visible, while tests, triad review and "
    "applicable scope review still run (blocking where enforcement makes them "
    "binding), and snapshot/fingerprint revalidation and final commit/tag/SHA "
    "binding still apply."
)


def _json_response(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# Deterministic admission preflights moved to ouroboros/commit_admission.py
# (Q3=A SSOT). The module-level aliases below are this gate's monkeypatch
# seams — the gate calls them through these names.
from ouroboros.commit_admission import (  # noqa: E402
    auto_sync_release_metadata_if_needed as _auto_sync_release_metadata_if_needed,
    release_metadata_preflight as _release_metadata_preflight,
    syntax_preflight_staged_py_files as _syntax_preflight_staged_py_files,  # noqa: F401 -- gate seam; the run leaf reads it through the call-time handle
)


def _mandatory_read_pointer(repo_dir: pathlib.Path, rel_path: str, section: str = "") -> str:
    """One governance doc as a resolvable absolute pointer for the agent_session route.

    Mirrors the plan-review agent_session delivery form (a retrieving row
    receives MANDATORY FULL READS at resolvable locators instead of inlined
    bodies — ``plan_review_runtime`` and the DEVELOPMENT.md "Core Governance
    Artifacts" table are the precedent): the session reads the document itself
    with its own tools; that retrieval is disclosed by the delegated-route
    telemetry and is non-certifying."""
    path = (pathlib.Path(repo_dir) / rel_path).resolve(strict=False)
    target = f"the '## {section}' section of {path}" if section else str(path)
    return (
        f"MANDATORY FULL READ (agent_session route — body not inlined): read {target} "
        "in full with your own file tools BEFORE reviewing; do not review from memory "
        "of this document."
    )


def _same_model_payable_spelling(model: str) -> str:
    """``model`` on a spelling this install can actually pay.

    The given id when its provider has credentials; otherwise the SAME model
    through its direct-provider spelling (``provider/name`` →
    ``provider::name`` — the direct-install class, e.g. an Anthropic-key-only
    install with an OpenRouter catalog id); otherwise the id unchanged — the
    credentials gate then records its loud audited bypass instead of a silent
    one. Never a different model: an unpayable row is bypassed, not swapped.
    """
    from ouroboros.provider_models import model_has_credentials

    model = str(model or "").strip()
    if not model or model_has_credentials(model):
        return model
    provider, _, name = model.partition("/")
    direct = f"{provider}::{name}" if name else ""
    if direct and model_has_credentials(direct):
        return direct
    return model


def _advisory_default_model() -> str:
    """The shipped advisory default on a route this install can actually pay."""
    from ouroboros.provider_models import OPENROUTER_REVIEW_DEFAULTS

    return _same_model_payable_spelling(str(OPENROUTER_REVIEW_DEFAULTS["advisory"]))


def _advisory_native_model() -> str:
    """The routed model the native advisory episode will run on."""
    from ouroboros.reviewer_slot_config import advisory_slot_config

    configured = (advisory_slot_config().target_id or "").strip()
    if configured:
        return _same_model_payable_spelling(configured)
    return _advisory_default_model()


def _advisory_child_timeout(ctx: object) -> Optional[float]:
    metadata = getattr(ctx, "task_metadata", {})
    return dispatch_window_remaining_sec(
        deadline_at=(metadata or {}).get("deadline_at") if isinstance(metadata, dict) else None,
        deadline_ts=getattr(ctx, "deadline_ts", None),
        reserve_sec=get_finalization_grace_sec(),
    )


def _run_advisory_native(
    prompt: str, repo_dir: pathlib.Path, ctx: ToolContext, slot, model: str,
    mandatory_read_corpus_chars: int = 0,
    checkpoint=None,
):
    """The advisory as a bounded native inspection episode, rehydrated into the
    same result structure the retired SDK path produced (only the transport
    changes). Cost: every provider call already rode the usage ledger inside
    the rebound scope, so ``cost_usd`` stays 0.0 here — the ledger is the one
    charge source; the disclosed total rides ``usage`` for forensics.
    ``mandatory_read_corpus_chars`` (wire size of the documents the prompt's
    MANDATORY FULL READ pointers name) declares the episode's mandatory reading
    on ``policy["native_mandatory_read_chars"]`` — task text plus corpus, a
    floor on the episode's bound up to the window — and appends the prompt's
    MANDATORY READ budget (corpus, bound, typed shortfall code)."""
    from dataclasses import replace as _dc_replace
    from types import SimpleNamespace

    from ouroboros.llm import LLMClient
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor, native_episode_transcript_bound
    from ouroboros.review_substrate import ReviewRequest
    from ouroboros.reviewer_slot_config import reviewer_slots
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    _task_metadata = getattr(ctx, "task_metadata", {}) or {}
    deadline_at = (
        str(_task_metadata.get("deadline_at") or "")
        if isinstance(_task_metadata, dict) else ""
    )
    request = ReviewRequest(
        surface="advisory_review",
        goal="Advisory pre-review of the live worktree.",
        task_id=str(getattr(ctx, "task_id", "") or ""),
        session_root=str(repo_dir),
        session_task=prompt,
        policy={"output_contract": (
            "A JSON array of checklist entries: "
            '[{"item": str, "verdict": "PASS"|"FAIL", "severity": '
            '"critical"|"advisory", "reason": str, "obligation_id"?: str}]'
        )},
        no_proxy=True,
        deadline_at=deadline_at,
    )
    # The dispatch builder for api_chat rows (`use_local` off the resolved
    # route): the bound previewed below and the episode's window are ONE route.
    rslot = _dc_replace(
        reviewer_slots([model], effort=slot.effort or "low", role_hint="advisory pre-reviewer",
                       id_prefix="advisory_slot")[0],
        subagent_id=str(getattr(slot, "subagent_id", "") or ""),
    )
    if int(mandatory_read_corpus_chars or 0) > 0:
        # Declared on the request FIRST: the bound the budget section names is
        # the very computation the episode makes from this assignment.
        request.policy["native_mandatory_read_chars"] = len(prompt) + int(mandatory_read_corpus_chars)
        request.session_task += _mandatory_read_budget_section(
            int(mandatory_read_corpus_chars), request.policy["native_mandatory_read_chars"],
            native_episode_transcript_bound(request, rslot),
        )
    from ouroboros.observability import new_call_id
    from ouroboros.review_dispatch import ReviewPaidStamp

    operation_id = new_call_id("advisory_native") if checkpoint else f"advisory:{request.task_id or 'manual'}"
    assignment = ReviewAssignment(
        request=request, slot=rslot, call_id=operation_id,
        dispatch_stamp=(ReviewPaidStamp(lambda: checkpoint(operation_id=operation_id), fail_closed=True)
                        if checkpoint else None),
    )
    executor = NativeToolRoundReviewExecutor(assignment, llm=LLMClient())
    _scope = _dc_replace(
        current_usage_scope() or UsageScope(),
        category="advisory_review", source="advisory_native",
    )
    try:
        with usage_scope(_scope):
            attempt = executor.execute()
    except Exception as exc:
        # The episode's proven facts (rounds, receipts, transcript vs bound,
        # paid ledger) and its typed code survive the failure: the caller
        # classifies on ``failure_code``, never on the message text.
        return _advisory_failure(exc, executor), model
    usage = dict(attempt.usage or {})
    usage["cost_disclosed_usd"] = usage.get("cost")
    source = attempt.message.get("native_transcript") if isinstance(attempt.message, dict) else None
    return SimpleNamespace(
        source_text=source if isinstance(source, str) else str(attempt.raw_text or ""),
        success=True,
        result_text=str(attempt.raw_text or ""),
        session_id="",
        cost_usd=0.0,  # ledger rows are the charge source; never re-emitted
        usage=usage,
        error="",
        stderr_tail="",
    ), str(usage.get("resolved_model") or model)


def _advisory_review_diff(
    repo_dir: pathlib.Path, ctx: ToolContext, paths: Optional[List[str]]
) -> tuple:
    """The advisory review diff and its context path scope, managed-aware (Δ4).

    Returns ``(diff_text, context_paths, early, managed)``. Non-managed callers
    get the byte-identical staged+unstaged capture with their own ``paths``
    scope. The authorized managed resolver gets the disclosed resolution-delta
    artifact (surface="advisory": the worktree candidate — advisory reviews
    work-in-progress by contract) scoped to delta ∪ conflict anchors — and its
    oversize outcome is an honest AUDITED non-blocking skip
    (``early=("skipped", message, chars)``), never the split-the-commit hard
    error: a managed merge stages the whole two-parent tree by contract and
    CANNOT be split into smaller commits. A failed managed capture is
    ``early=("error", message, 0)`` — no placeholder review."""
    from ouroboros.tools.review_subject import managed_review_subject

    # Every advisory pre-review reviews the LIVE worktree afresh: drop any
    # advisory-surface memo entries so a subject built for an earlier
    # pre-review of this attempt can never be served for a changed worktree
    # (gate-surface entries stay — the staged candidate is frozen per attempt).
    memo = getattr(ctx, "_managed_review_subject_memo", None)
    if isinstance(memo, dict):
        for key in [k for k in memo if isinstance(k, tuple) and len(k) >= 4 and k[3] == "advisory"]:
            memo.pop(key, None)
    try:
        subject = managed_review_subject(ctx, repo_dir, surface="advisory")
    except Exception as exc:  # incl. StagedDiffUnavailable
        return "", None, (
            "error", f"⚠️ ADVISORY_ERROR: managed resolution delta unavailable: {exc}", 0
        ), False
    try:
        # Thread the disclosed counters out of THIS subject: the pre-review
        # handler's snapshot summary reads them instead of recomputing a second
        # subject (a full delta recomputation and a display-only TOCTOU).
        ctx._last_advisory_subject_counters = (
            subject.counters_line() if subject is not None else ""
        )
    except Exception:
        pass
    if subject is not None and len(subject.diff) > _MAX_DIFF_CHARS_ERROR:
        # Honest downstream expectation: triad + scope always run, but they
        # gate the commit only under blocking enforcement.
        if _get_review_enforcement() == "blocking":
            gate_note = "Triad and scope review still gate the commit."
        else:
            gate_note = (
                "Triad and scope review still run; enforcement is advisory, so "
                "their findings are recorded rather than blocking."
            )
        warning = (
            f"⚠️ ADVISORY_SKIPPED: {_MANAGED_SKIP_MARKER} "
            f"({len(subject.diff):,} chars > {_MAX_DIFF_CHARS_ERROR:,}). "
            "Advisory review skipped — non-blocking and audited; a managed "
            f"update merge {_MANAGED_SKIP_NOTE}. {gate_note}"
        )
        _stamp_advisory_skip_meta(ctx, None, "managed_diff_too_large")
        return "", None, ("skipped", warning, len(subject.diff)), True
    if subject is not None:
        return subject.render_prompt_diff(), subject.touched_paths(), None, True
    return _get_staged_diff(repo_dir, paths=paths), paths, None, False


def _prompt_oversize_skip_warning(prompt_chars: int, managed: bool) -> str:
    """The 1.6M prompt gate's non-blocking skip text. ``managed=True`` (the
    diff under review is a managed resolution delta) drops the split advice —
    a managed merge stages the whole two-parent tree by contract — and states
    what is actually possible instead."""
    tokens_approx = max(1, prompt_chars // 4)
    remedy = (
        f"A managed update merge {_MANAGED_SKIP_NOTE}; the "
        "skip is audited and non-blocking."
        if managed else "Consider splitting the commit."
    )
    return (
        f"⚠️ ADVISORY_SKIPPED: advisory prompt too large "
        f"({prompt_chars:,} chars, ~{tokens_approx:,} tokens > "
        f"{_ADVISORY_PROMPT_MAX_CHARS:,} char limit). "
        f"Advisory review skipped — non-blocking. {remedy}"
    )


def _api_window_skip_warning(model: str, prompt: str, managed: bool) -> str:
    """The api route's admission verdict against its REAL window, or ``""`` to proceed.

    The window comes from the reviewer-window SSOT
    (``reviewer_window.resolve_reviewer_window``) with the review family's
    existing reserve scaling — never from ``_ADVISORY_PROMPT_MAX_CHARS`` (that
    constant is only the emergency sanity ceiling). An unevidenced route keeps
    the SSOT's full-window assumption, so this gate skips exactly when the
    evidence proves the prompt cannot be admitted; the post-dispatch overflow
    classification stays the honest net for routes without evidence. Oversize
    is the EXISTING typed non-blocking ADVISORY_SKIPPED path, produced BEFORE
    any provider dispatch, naming the window and the measured size."""
    from ouroboros import reviewer_window as _rw
    from ouroboros.tools.review import _review_output_budget
    from ouroboros.utils import estimate_tokens

    window = _rw.resolve_reviewer_window(model).sizing_window()
    output_reserve, tokenizer_margin = _rw.window_scaled_reserves(
        window,
        output_reserve=_review_output_budget(),
        tokenizer_margin=50_000,
    )
    input_limit = max(0, int(window) - int(output_reserve) - int(tokenizer_margin))
    prompt_tokens = estimate_tokens(prompt)
    if prompt_tokens <= input_limit:
        return ""
    remedy = (
        f"A managed update merge {_MANAGED_SKIP_NOTE}; the "
        "skip is audited and non-blocking."
        if managed
        else (
            "Consider splitting the commit, or switch the advisory row to an "
            "agent_session route (its compact pack sends governance docs as "
            "pointers instead of inlining them)."
        )
    )
    return (
        f"⚠️ ADVISORY_SKIPPED: advisory prompt does not fit the api route window "
        f"({len(prompt):,} chars, ~{prompt_tokens:,} estimated tokens > input limit "
        f"{input_limit:,} of the {window:,}-token window for model {model or '(default)'}). "
        f"Advisory review skipped — non-blocking and audited. {remedy}"
    )


def _overflow_failure_text(*texts: object) -> bool:
    """Advisory-only overflow recognition for a DISPATCHED advisory failure.

    Classifies failure text against the ``context_budget`` SSOT: structured
    overflow codes, message markers, and the output/body-size precedence (an
    output-limit rejection is NOT a window overflow). Deliberately NOT a
    generic overflow-classification helper for other tools — the sprint's
    not-build list keeps this advisory-local; other surfaces adopt the SSOT
    themselves when they need it."""
    from ouroboros.context_budget import (
        CONTEXT_OVERFLOW_CODES,
        context_overflow_message,
        output_or_body_size_message,
    )

    combined = " ".join(str(t or "") for t in texts if str(t or "").strip())
    if not combined:
        return False
    if output_or_body_size_message(combined):
        return False
    low = combined.lower()
    return context_overflow_message(combined) or any(
        code in low for code in CONTEXT_OVERFLOW_CODES
    )


def _overflow_skip_warning(route: str, prompt_chars: int, failure_head: str) -> str:
    """Typed non-blocking skip for a provider/harness context-window rejection.

    ``reason=context_window_exceeded``, carrying the delivery route and the
    measured prompt size. No host-side retry or split — advisory is fail-open
    by design; the pre-dispatch gates own prevention, this path owns honesty
    (previously this failure was misfiled as a crashed harness inviting a
    doomed retry of the identical oversize prompt)."""
    tokens_approx = max(1, (int(prompt_chars) + 3) // 4)
    head = " ".join(str(failure_head or "").split())
    head = (head[:200] + "…") if len(head) > 200 else head
    return (
        "⚠️ ADVISORY_SKIPPED: context_window_exceeded — the advisory prompt "
        f"exceeded the {route} route's context window at dispatch "
        f"({prompt_chars:,} chars, ~{tokens_approx:,} estimated tokens). "
        "Advisory review skipped — non-blocking and audited; no host-side retry "
        f"or split. Provider signal: {head}"
    )


def _stamp_advisory_skip_meta(ctx: ToolContext, meta: Optional[dict], skip_reason: str) -> None:
    """Record a typed advisory skip on the ctx meta snapshot (best-effort).

    Pre-dispatch gates pass ``meta=None`` (no run meta exists yet) and stamp a
    minimal skipped snapshot; the post-dispatch classifier passes its full run
    meta so model/session/usage survive alongside the skip."""
    try:
        snapshot = dict(meta) if meta else {}
        snapshot["status"] = "skipped"
        snapshot["skip_reason"] = skip_reason
        setattr(ctx, "_last_claude_advisory_meta", snapshot)
    except Exception:
        pass


def _predispatch_size_skip(
    ctx: ToolContext,
    delegated_route: bool,
    model: str,
    prompt: str,
    managed: bool,
) -> Optional[tuple]:
    """Both pre-dispatch size gates: the typed skip tuple, or ``None`` to dispatch.

    First the emergency sanity ceiling (both routes — see the
    ``_ADVISORY_PROMPT_MAX_CHARS`` note), then, on the api route only, the
    honest admission gate against the REAL route window
    (``_api_window_skip_warning``): the 1.6M constant is far above any real
    route window and used to let oversize prompts die downstream as a false
    "harness crashed / Retry" classification. Every skip stamps the meta
    snapshot with ``status="skipped"`` and a ``skip_reason``."""
    prompt_chars = len(prompt)
    if prompt_chars > _ADVISORY_PROMPT_MAX_CHARS:
        log.warning("Advisory skipped — prompt too large: %d chars", prompt_chars)
        _stamp_advisory_skip_meta(ctx, None, "prompt_ceiling_exceeded")
        return [], _prompt_oversize_skip_warning(prompt_chars, managed), model, prompt_chars
    if delegated_route:
        return None
    window_skip = _api_window_skip_warning(model, prompt, managed)
    if not window_skip:
        return None
    log.warning(
        "Advisory skipped — prompt does not fit the api route window: %d chars",
        prompt_chars,
    )
    _stamp_advisory_skip_meta(ctx, None, "route_window_exceeded")
    return [], window_skip, model, prompt_chars


def _maybe_overflow_skip(
    ctx: ToolContext,
    delegated_route: bool,
    prompt_chars: int,
    model: str,
    meta: Optional[dict],
    failure: object,
    stderr_tail: object = "",
    verb: str = "reported",
    failure_code: str = "",
) -> Optional[tuple]:
    """Post-dispatch overflow classification: the typed skip tuple, or ``None``.

    Runs BEFORE the generic error formatting (``context_budget`` SSOT): a
    prompt the route could not admit is the same typed non-blocking skip the
    pre-dispatch gates produce — never an ADVISORY_ERROR that reads as a
    crashed harness and invites a doomed retry of the identical prompt.
    Serves both dispatched-failure shapes: a returned failure result
    (``verb="reported"``, with its stderr tail and run meta) and a raised
    exception (``verb="raised"``). The native episode's own bound end is keyed
    on its STRUCTURED code (``review_native_episode``:
    ``native_transcript_cap_exceeded``), never on message text, and is NOT a
    provider window refusal — it keeps its own skip reason and the episode's
    numbers (bound, refused chars, paid rounds) from ``failure_custody``."""
    if failure_code == "native_transcript_cap_exceeded":
        facts = dict((meta or {}).get("usage") or {})
        bound, rounds = int(facts.get("native_transcript_bound") or 0), int(facts.get("native_rounds") or 0)
        refused = int(facts.get("native_transcript_refused_chars") or facts.get("native_transcript_chars") or 0)
        log.warning("Advisory skipped — native episode transcript bound exceeded after %d round(s) "
                    "(%d > %d chars)", rounds, refused, bound)
        _stamp_advisory_skip_meta(ctx, meta, "native_transcript_bound_exceeded")
        return [], (
            "⚠️ ADVISORY_SKIPPED: native_transcript_bound_exceeded — the advisory's native "
            f"inspection episode exhausted its window-derived transcript bound after {rounds} paid "
            f"round(s) ({refused:,} chars against the {bound:,}-char bound) before a final answer. "
            "Advisory review skipped — non-blocking and audited; the paid rounds' usage stays on the "
            "advisory meta. Levers: a larger-window advisory row, or "
            "OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS."
        ), model, prompt_chars
    if not _overflow_failure_text(failure, stderr_tail):
        return None
    route_name = "agent_session" if delegated_route else "native"
    log.warning(
        "Advisory skipped — %s route %s context overflow (%d chars)",
        route_name, verb, prompt_chars,
    )
    _stamp_advisory_skip_meta(ctx, meta, "context_window_exceeded")
    return [], _overflow_skip_warning(route_name, prompt_chars, str(failure or "")), model, prompt_chars


def run_advisory_critic(*args, **kwargs):
    """Public cross-module entry for one advisory critic run (skill review).

    A thin typed alias for the module-internal ``_run_claude_advisory`` so
    other surfaces never probe private names with ``hasattr`` (a rename would
    silently no-op their advisory forever). Same signature and return shape:
    ``(items, raw_result, model, prompt_chars)``.
    """
    return _run_claude_advisory(*args, **kwargs)


# -- Audit logging --

def _audit_bypass(ctx: ToolContext, snapshot_hash: str, commit_message: str,
                  bypass_reason: str, task_id: str) -> None:
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "advisory_review_bypassed",
            "snapshot_hash": snapshot_hash,
            "commit_message": commit_message,  # full — no [:200] truncation
            "bypass_reason": bypass_reason,
            "task_id": task_id,
        })
    except Exception:
        pass


def _identical_diff_cap_note() -> str:
    """Schema-build-time NOTE about Max-Review-Cycles semantics on the commit
    gate, derived from the shared OUROBOROS_REVIEW_MAX_CYCLES (never a
    hardcoded number). Identical bytes are never re-reviewed for pay: from the
    FIRST review-verdict block, resubmitting the byte-identical staged diff
    without a NEW rebuttal never buys a new review (identical_diff_refused);
    the knob itself counts PAID triad+scope cycles per task. Whether either
    state blocks the commit follows enforcement (the honest caveat below)."""
    from ouroboros.review_cycles import review_max_cycles

    cap = review_max_cycles()
    base = (
        "NOTE: identical bytes are never re-reviewed for pay — after ANY review-verdict "
        "block, a byte-identical resubmission to commit_reviewed buys no new review "
        "(identical_diff_refused, quoting the recorded verdict) until the diff changes "
        "or a NEW review_rebuttal is supplied (a rebuttal new to the streak buys exactly "
        "one paid re-review; a repeated one buys none)."
    )
    caveat = (
        " Under blocking enforcement an identical resubmission after a recorded "
        "verdict block is refused for free; a pure advisory line never mints verdict "
        "blocks, so its no-new-spend guarantee is the exhaustion free replay — the "
        "commit proceeds with a loud durable disclosure and no new review spend."
    )
    if cap is None:
        return (
            f"{base} OUROBOROS_REVIEW_MAX_CYCLES=unlimited: no per-root-task ceiling on "
            f"paid triad+scope cycles is configured.{caveat}"
        )
    return (
        f"{base} The shared OUROBOROS_REVIEW_MAX_CYCLES cap bounds PAID triad+scope "
        f"cycles per ROOT task (shared across the whole task tree; a follow-up task "
        f"starts its own): after {cap} paid cycle(s) commit_reviewed buys no further "
        "review (typed review_cycles_exhausted event; every dispatched wave counts, "
        f"only undispatched attempts stay outside the count).{caveat}"
    )


def _advisory_run_record(
    snapshot_hash: str,
    commit_message: str,
    status: str,
    *,
    repo_key: str,
    task_id: str,
    **fields,
) -> AdvisoryRunRecord:
    from ouroboros.review_state import _record_from_dict

    # One normalization contract for authored and reloaded advisory records.
    return _record_from_dict({
        **{name: value for name, value in fields.items() if value is not None},
        "snapshot_hash": snapshot_hash, "commit_message": commit_message,
        "status": status, "ts": _utc_now(), "repo_key": repo_key,
        "tool_name": "advisory_review", "task_id": task_id,
    })


def _record_bypass(ctx: ToolContext, state: "AdvisoryReviewState", snapshot_hash: str,
                   commit_message: str, reason: str, task_id: str,
                   drive_root: pathlib.Path,
                   snapshot_paths: Optional[List[str]] = None) -> str:
    """Audit, record, and save a bypassed advisory run. Returns JSON response."""
    _audit_bypass(ctx, snapshot_hash, commit_message, reason, task_id)
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))

    def _mutate(bypass_state: "AdvisoryReviewState") -> None:
        bypass_state.add_run(_advisory_run_record(
            snapshot_hash, commit_message, "bypassed",
            repo_key=repo_key, task_id=task_id,
            bypass_reason=reason, bypassed_by_task=task_id,
            snapshot_paths=snapshot_paths,
        ))

    update_state(drive_root, _mutate)
    # Persistent visibility (same mechanism as advisory-enforcement overrides):
    # review_status surfaces how often the advisory layer was bypassed/absent.
    try:
        from ouroboros.utils import update_json_locked, utc_now_iso as _now_iso

        def _bump(current: dict) -> dict:
            recent = list(current.get("recent") or [])
            recent.append({"ts": _now_iso(), "block_reason": f"advisory_bypass: {reason}"[:200], "message_head": str(commit_message or "")[:200]})
            return {"count": int(current.get("count") or 0) + 1, "recent": recent[-10:]}

        update_json_locked(pathlib.Path(drive_root) / "state" / "advisory_overrides.json", _bump)
    except Exception:
        log.debug("Failed to persist advisory bypass visibility", exc_info=True)
    if "ANTHROPIC_API_KEY" in reason:
        # Route-dependent honesty (plan 5.8 site 4): the key is only the API
        # route's requirement — the owner also has the keyless delegated route.
        msg = (
            "⚠️ ANTHROPIC_API_KEY is not set — advisory review skipped automatically "
            "because the configured advisory route (api) requires it. "
            "Bypass has been durably audited in events.jsonl. "
            "Set ANTHROPIC_API_KEY in Settings, or switch the advisory row to "
            "the delegated subscription route (Review lanes on the Agents tab "
            "— OUROBOROS_REVIEWER_SLOTS advisory kind agent_session), which "
            "needs no API key."
        )
    else:
        msg = "Advisory review bypassed. Bypass has been durably audited."
    return _json_response({
        "status": "bypassed",
        "snapshot_hash": snapshot_hash,
        "bypass_reason": reason,
        "message": msg,
    })


def _resolve_matching_obligations(
    state: "AdvisoryReviewState",
    items: list,
    snapshot_hash: str,
    *,
    repo_key: str | None = None,
) -> None:
    """Resolve obligations only on unambiguous PASS without same-item FAIL."""
    if not items:
        return
    # Build per-item verdict sets to detect contradictions.
    item_verdicts: dict[str, set[str]] = {}
    obligation_verdicts: dict[str, set[str]] = {}
    for i in items:
        if not isinstance(i, dict):
            continue
        verdict = str(i.get("verdict", "")).upper().strip()
        item_name = str(i.get("item", "")).strip()
        if not item_name or not verdict:
            continue
        explicit_obligation_id = normalize_reviewer_obligation_id(i.get("obligation_id", ""))
        normalized_item_name, suffix_obligation_id = strip_obligation_suffix(item_name)
        normalized_item_name = normalized_item_name.strip().lower()
        if normalized_item_name:
            item_verdicts.setdefault(normalized_item_name, set()).add(verdict)
        # Explicit id and suffix id must agree; mismatches are ambiguous and
        # must not clear unrelated obligations/debt.
        if explicit_obligation_id and suffix_obligation_id:
            if explicit_obligation_id.lower() == suffix_obligation_id.lower():
                obligation_verdicts.setdefault(explicit_obligation_id, set()).add(verdict)
            # Mismatch: skip both ids for this entry.
            continue
        if explicit_obligation_id:
            obligation_verdicts.setdefault(explicit_obligation_id, set()).add(verdict)
        elif suffix_obligation_id:
            obligation_verdicts.setdefault(suffix_obligation_id, set()).add(verdict)

    # Only PASS items with no FAIL entry for the same item.
    unambiguous_pass = {
        item_name
        for item_name, verdicts in item_verdicts.items()
        if "PASS" in verdicts and "FAIL" not in verdicts
    }
    unambiguous_pass_ids = {
        obligation_id
        for obligation_id, verdicts in obligation_verdicts.items()
        if "PASS" in verdicts and "FAIL" not in verdicts
    }

    open_obs = state.get_open_obligations(repo_key=repo_key)

    # Item-name fallback is safe only with exactly one open obligation per item.
    from collections import Counter as _Counter
    item_open_count = _Counter(o.item.lower() for o in open_obs)

    resolved = [
        o.obligation_id for o in open_obs
        if o.obligation_id.lower() in unambiguous_pass_ids
        or (
            o.item.lower() in unambiguous_pass
            and item_open_count[o.item.lower()] == 1
        )
    ]
    if resolved:
        state.resolve_obligations(
            resolved,
            resolved_by=f"advisory run {snapshot_hash[:12]}",
            repo_key=repo_key,
        )
        state._sync_commit_readiness_debts(repo_key=repo_key)


def _next_step_guidance(latest: Optional["AdvisoryRunRecord"], state: "AdvisoryReviewState",
                        stale_from_edit: bool, stale_from_edit_ts: Optional[str],
                        open_obs: list, open_debts: list, effective_is_fresh: bool = False,
                        enforcement: str = "blocking", *, advisory_permitted: bool = False) -> str:
    """Return a concrete next-step string based on current advisory state.

    ``enforcement`` keeps the guidance HONEST (O1): under blocking the
    historical wording stands; under advisory the findings are recorded
    durably, the agent decides which to apply, and ``commit_reviewed`` is
    available — the text must never assert a block that will not happen or a
    fix-all-criticals dichotomy that does not exist.

    Snapshot binding of record-derived claims (the v6.74.5 "SyntaxError" stale
    template that cost a release ~25 min) is enforced UPSTREAM by the
    projection: a blocked record whose hash differs from the current tree sets
    ``stale_from_edit`` (review_evidence hash_mismatch), which routes to the
    generic "invalidated" message below instead of asserting the problem class
    — that assertion only ever fires for a record of the CURRENT snapshot. The
    one unbindable case stays as before: an uncomputable current hash cannot
    establish a mismatch either way.
    """
    def _debt_hint() -> str:
        parts = []
        if open_obs:
            parts.append(f"{len(open_obs)} open obligation(s) from previous blocking rounds")
        if open_debts:
            parts.append(f"{len(open_debts)} commit-readiness debt item(s) surfaced by review_status")
        return (" ".join(parts) + ". ") if parts else ""

    regroup = "After the first blocked review, stop patching one finding at a time: re-read the full diff, group obligations by root cause, rewrite the plan, finish all remaining edits, then run preflight_review(commit_message='...')."

    def _with_choices(message: str) -> str:
        return f"{message.rstrip()} {ADVISORY_REVIEW_CHOICE_GUIDANCE}"

    if (advisory_permitted and not stale_from_edit and latest is not None
            and latest.status in {"error", "parse_failure"}):
        return (
            "The current preflight failed technically; its source and findings remain recorded, not PASS. "
            "Advisory enforcement permits commit_reviewed subject to its independent checks."
        )

    if not effective_is_fresh:
        status = str(getattr(latest, "status", "") or "")
        if latest and status in {"tests_preflight_blocked", "preflight_blocked"} and not stale_from_edit:
            if status == "tests_preflight_blocked":
                problem = "test preflight: pytest failed before the paid critic call"
                fix = "Fix the failing tests and re-run preflight_review. Use preflight_review(skip_tests=True) only for intentional WIP code."
            else:
                # H4 (capinv-447): "preflight_blocked" is produced by more than one
                # deterministic check — branch on the typed cause, never assert
                # "SyntaxError" for a release-metadata block (or an unknown one).
                reason_kind = str(getattr(latest, "reason_kind", "") or "")
                if reason_kind == "syntax":
                    problem = "syntax preflight: a staged .py file has a SyntaxError"
                    fix = "See raw_result for file:line:msg, fix it, and re-run preflight_review."
                elif reason_kind == "release_metadata":
                    problem = "release metadata preflight: version/README release carriers failed the deterministic check"
                    fix = "See raw_result for the exact carrier mismatch, fix it, and re-run preflight_review."
                else:
                    problem = "a deterministic preflight check (see raw_result for the exact cause)"
                    fix = "Fix the cause named in raw_result and re-run preflight_review."
            return _with_choices(
                f"Last advisory run was blocked by {problem}. {fix} {_debt_hint()}".strip()
            )
        if latest and status == "parse_failure" and not stale_from_edit:
            suffix = (
                regroup + " Or bypass: commit_reviewed(skip_advisory_review=True) (audited)."
                if (open_obs or open_debts)
                else "Re-run: preflight_review(commit_message='...'), or bypass: commit_reviewed(skip_advisory_review=True) (audited)."
            )
            return _with_choices(
                f"Last advisory run produced unparseable output (parse_failure). {_debt_hint()}{suffix}"
            )
        if open_obs or open_debts:
            prefix = f"Advisory was invalidated by a worktree edit at {stale_from_edit_ts}. " if stale_from_edit else "Advisory is stale or missing for the current snapshot. "
            return _with_choices(prefix + _debt_hint() + regroup)
        if stale_from_edit:
            return _with_choices(
                f"Advisory was invalidated by a worktree edit at {stale_from_edit_ts}. Complete ALL remaining edits, then run: preflight_review(commit_message='...')"
            )
        if not state.advisory_runs:
            return _with_choices("No advisory run yet. Run: preflight_review(commit_message='...')")
        return _with_choices("Advisory is stale (snapshot changed). Run: preflight_review(commit_message='...')")

    # Advisory is effectively fresh — check obligations and findings
    if open_obs or open_debts:
        if enforcement == "blocking":
            return _with_choices(
                f"Advisory is current but unresolved review debt remains. {_debt_hint()}commit_reviewed will be blocked until that debt is cleared. Re-read the full diff, group obligations by root cause, and rewrite the plan. Fix the issues, re-run preflight_review so it marks them PASS, or bypass: commit_reviewed(skip_advisory_review=True) (audited)."
            )
        return _with_choices(
            f"Advisory is current and unresolved review debt remains recorded durably. {_debt_hint()}Enforcement is advisory: you decide which findings to apply — commit_reviewed is available. Re-read the full diff, group obligations by root cause, and rewrite the plan; re-run preflight_review so addressed items are marked PASS."
        )

    if latest and latest.status == "skipped":
        if _MANAGED_SKIP_NOTE in str(getattr(latest, "raw_result", "") or ""):
            # Managed resolution skip: split advice is structurally impossible
            # (the merge stages the whole two-parent tree by contract).
            return (
                "Advisory was skipped — the managed resolution exceeded the "
                "advisory size gate. commit_reviewed may proceed. A managed "
                f"update merge {_MANAGED_SKIP_NOTE}; switch the advisory row "
                "to an agent route or a larger-window model if advisory "
                "coverage is wanted."
            )
        return (
            "Advisory was skipped — the assembled prompt did not fit the advisory "
            "route (window/size gate). commit_reviewed may proceed. Split the "
            "commit into smaller chunks, or switch the advisory row to an "
            "agent_session route, which retrieves context instead of inlining it."
        )

    if latest and latest.status == "bypassed":
        return "Advisory was bypassed (audited). No open obligations — commit_reviewed should proceed. Consider running preflight_review for a proper review."

    fresh_critical = [
        i for i in (latest.items if latest else []) or []
        if isinstance(i, dict) and str(i.get("verdict", "")).upper() == "FAIL"
        and str(i.get("severity", "")).lower() == "critical"
    ]
    if fresh_critical:
        if enforcement == "blocking":
            # Honest blocking-branch wording (no false dichotomy): a FRESH
            # advisory with critical findings already satisfies the commit
            # gate's advisory-freshness requirement, and zero advisory FAILs is
            # not a hard gate — the blocking triad and scope reviews are what
            # can still block. The audited skip bypasses only the advisory
            # freshness/debt checks, never these findings.
            return _with_choices(
                f"Advisory found {len(fresh_critical)} critical issue(s). This fresh advisory already satisfies the commit gate's advisory-freshness requirement; the findings are recorded durably on the advisory run record, and commit_reviewed is available — the blocking triad and scope reviews are the gate that can still block. Fix the critical findings and re-run preflight_review so they are marked PASS; skip_advisory_review=True (audited) bypasses only the freshness/debt checks, not these findings."
            )
        return _with_choices(
            f"Advisory found {len(fresh_critical)} critical issue(s). Findings are recorded durably; enforcement is advisory — you decide which to apply, and commit_reviewed is available. Re-run preflight_review after fixes, or deliberately choose the audited advisory skip."
        )
    return "Advisory is fresh with no critical findings. Proceed with: commit_reviewed(commit_message='...'). ⚠️ Do NOT make any further edits — any edit will make advisory stale."


def _persist_preflight_record(
    ctx: ToolContext,
    snapshot_hash: str,
    commit_message: str,
    record: dict,
) -> None:
    """Persist a preflight fact; strict pre-POST checkpoints propagate failure."""
    record = dict(record or {})
    strict = bool(record.pop("strict", False))
    try:
        drive_root = pathlib.Path(ctx.drive_root)
        record["snapshot_paths"] = record.pop("paths", None)
        status = str(record.pop("status", "error"))
        record.setdefault("snapshot_summary", "preflight execution fact")
        run = _advisory_run_record(
            snapshot_hash, commit_message, status,
            repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
            task_id=str(getattr(ctx, "task_id", "") or ""), **record,
        )
        update_state(drive_root, lambda state: state.add_run(run))
    except Exception:
        if strict:
            raise
        log.debug("_persist_preflight_record failed (non-critical)", exc_info=True)


def _advisory_pre_sdk_gate(
    ctx: ToolContext,
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    snapshot_hash: str,
    commit_message: str,
    paths: Optional[List[str]],
    skip_tests: bool,
    review_rebuttal: str = "",
):
    """Run cheap pre-SDK gates and return warnings/status/early JSON exit."""
    repo_key = make_repo_key(repo_dir)
    task_id = str(getattr(ctx, "task_id", "") or "")
    state = load_state(drive_root)

    # Readiness gate first: reject clean worktree before fresh-run shortcut.
    readiness_warnings = check_worktree_readiness(repo_dir, paths=paths)
    if readiness_warnings and any("no uncommitted changes" in w.lower() for w in readiness_warnings):
        ctx.emit_progress_fn(f"⚠️ Advisory readiness gate: {'; '.join(readiness_warnings)}")
        return readiness_warnings, "", _json_response({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "message": "No uncommitted changes detected — nothing to review.",
            "readiness_warnings": readiness_warnings,
        })

    if readiness_warnings:
        try:
            append_jsonl(drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "advisory_readiness_gate",
                "warnings": readiness_warnings,
                "task_id": task_id,
            })
        except Exception:
            pass

    # Fresh-run shortcut only when no obligations/debt remain.
    existing = state.find_by_hash(snapshot_hash, repo_key=repo_key)
    open_obligations = state.get_open_obligations(repo_key=repo_key)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)
    already_fresh_ok = (
        existing and existing.status in ("fresh", "bypassed", "skipped")
        and str(existing.review_rebuttal or "").strip() == str(review_rebuttal or "").strip()
        and not open_obligations and not open_debts
    )
    if already_fresh_ok:
        return readiness_warnings, "", _json_response({
            "status": "already_fresh",
            "snapshot_hash": snapshot_hash,
            "ts": existing.ts,
            "items": existing.items,
            "readiness_warnings": readiness_warnings,
            "message": "A fresh advisory run already exists for this snapshot. Proceed with commit_reviewed.",
        })

    ctx.emit_progress_fn("Running preflight pre-review (read-only critic)...")
    changed_files = _get_changed_file_list(repo_dir, paths=paths)

    if changed_files.startswith("⚠️ ADVISORY_ERROR"):
        return readiness_warnings, changed_files, _json_response({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "error": changed_files,
            "message": (
                "Advisory review aborted: could not retrieve changed file list. "
                "Fix the error and retry, or use skip_advisory_review=True to bypass (will be audited)."
            ),
        })

    release_preflight_err = _release_metadata_preflight(repo_dir, commit_message, paths)
    if release_preflight_err:
        ctx.emit_progress_fn(release_preflight_err)
        _persist_preflight_record(
            ctx=ctx,
            snapshot_hash=snapshot_hash,
            commit_message=commit_message,
            record={
                "status": "preflight_blocked",
                "reason_kind": "release_metadata",
                "raw_result": release_preflight_err,
                "paths": paths,
                "duration_sec": 0.0,
                "readiness_warnings": readiness_warnings,
            },
        )
        return readiness_warnings, changed_files, _json_response({
            "status": "preflight_blocked",
            "snapshot_hash": snapshot_hash,
            "error": release_preflight_err,
            "readiness_warnings": readiness_warnings,
            "message": (
                "Advisory delivery was skipped: deterministic release metadata preflight "
                "failed before provider budget was spent."
            ),
        })

    # Version-sync check is a non-fatal warning.
    version_sync_warning = _check_worktree_version_sync_shared(repo_dir)
    if version_sync_warning:
        ctx.emit_progress_fn(f"⚠️ Advisory preflight: {version_sync_warning}")

    # Test preflight before the expensive delivery call.
    if not skip_tests:
        ctx.emit_progress_fn("Running tests before the advisory delivery call...")
        from ouroboros.commit_admission import run_tests_preflight_with_proof

        test_err = run_tests_preflight_with_proof(
            ctx, runner=lambda c, **kw: _run_advisory_tests(c, **kw))
        if test_err:
            msg = (
                "⚠️ TESTS_PREFLIGHT_BLOCKED: Tests must pass before advisory review.\n"
                "Fix the failures below, then re-run preflight_review.\n"
                "Use skip_tests=True if this is intentionally incomplete WIP code.\n\n"
                f"{test_err}"
            )
            ctx.emit_progress_fn(msg)
            # Persist non-fresh blocker so review_status can surface it after restart.
            _persist_preflight_record(
                ctx=ctx,
                snapshot_hash=snapshot_hash,
                commit_message=commit_message,
                record={
                    "status": "tests_preflight_blocked",
                    "raw_result": msg,
                    "paths": paths,
                    "duration_sec": 0.0,
                    "readiness_warnings": readiness_warnings,
                },
            )
            return readiness_warnings, changed_files, _json_response({
                "status": "tests_preflight_blocked",
                "snapshot_hash": snapshot_hash,
                "message": msg,
                "readiness_warnings": readiness_warnings,
            })
        # A green run already carries the Q10 managed proof: the shared
        # admission helper records it (commit_admission SSOT).
        ctx.emit_progress_fn(
            "Tests passed ✓ — proceeding with the advisory delivery call."
            if getattr(ctx, "_preflight_tests_passed", False) is True else
            "Tests skipped by configured policy; no green test proof was recorded."
        )

    return readiness_warnings, changed_files, None


def _run_advisory_tests(ctx: ToolContext, *, force: bool = False) -> Optional[str]:
    """Run shared pytest preflight while preserving this monkeypatch seam."""
    return _run_review_preflight_tests(ctx, force=True) if force else _run_review_preflight_tests(ctx)


def _handle_advisory_pre_review(
    ctx: ToolContext,
    commit_message: str = "",
    skip_advisory_review: bool = False,
    skip_advisory_pre_review: bool = False,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
    skip_tests: bool = False,
    review_rebuttal: str = "",
    prepared: bool = False,
) -> str:
    """Run an advisory pre-commit review through the configured read-only route."""
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    repo_dir = pathlib.Path(ctx.repo_dir)
    drive_root = pathlib.Path(ctx.drive_root)

    try:
        execution, pending_run = pending_advisory_execution(
            ctx, commit_message, goal=goal, scope=scope, paths=paths,
            review_rebuttal=review_rebuttal,
        )
    except Exception as exc:
        return _json_response({"status": "pending", "error": str(exc),
                               "message": "Preflight custody must be reconciled before preparing another candidate."})
    resuming = pending_run is not None
    if resuming:
        paths = pending_run.snapshot_paths
    # commit_reviewed already prepared and fingerprinted this candidate.
    # Standalone preflight retains its existing mechanical preparation.
    if not prepared and not resuming:
        auto_synced_paths = _auto_sync_release_metadata_if_needed(ctx, repo_dir, drive_root, paths)
        if paths is not None and auto_synced_paths:
            paths = sorted(set(paths) | set(auto_synced_paths))

    snapshot_hash = compute_snapshot_hash(repo_dir, commit_message, paths=paths)

    # Bypass recording state; the pre-SDK gate derives its own under 8 params.
    repo_key = make_repo_key(repo_dir)
    task_id = str(getattr(ctx, "task_id", "") or "")
    state = load_state(drive_root)

    if not resuming:
        # Auto-bypass a missing Anthropic key ONLY when the configured advisory
        # route actually needs it (plan 5.8 site 3 — the dangerous one): on the
        # delegated route the constitutional gate RUNS instead of recording a
        # routine-looking "auto-bypassed" over a commit the free route could have
        # reviewed. A misconfigured route token is a loud error, not a bypass.
        try:
            _native_route = advisory_review_route() == "api_chat"
            _advisory_enabled = advisory_slot_enabled()
        except ValueError as exc:
            return _json_response({
                "status": "error",
                "snapshot_hash": snapshot_hash,
                "error": f"⚠️ ADVISORY_ERROR: {exc}",
                "message": "Fix the advisory reviewer configuration "
                           "(OUROBOROS_REVIEWER_SLOTS advisory row) and retry.",
            })
        if not _advisory_enabled:
            # The owner switched the advisory slot off (6.2) — or the legacy
            # Claude-SDK target migration force-disabled the row with a typed
            # reason. The constitutional gate still runs — as an AUDITED BYPASS on
            # this exact snapshot, the same durable record an explicit skip makes.
            from ouroboros.reviewer_slot_config import advisory_slot_config as _asc

            _dis = str(getattr(_asc(), "disabled_reason", "") or "")
            return _record_bypass(ctx, state, snapshot_hash, commit_message,
                                   "advisory reviewer disabled in settings — audited bypass"
                                   + (f" ({_dis})" if _dis else ""),
                                   task_id, drive_root,
                                   snapshot_paths=paths)
        if _native_route:
            from ouroboros.provider_models import model_has_credentials

            _m = _advisory_native_model()
            if not model_has_credentials(_m):
                return _record_bypass(ctx, state, snapshot_hash, commit_message,
                                       f"no provider credentials for advisory model {_m} "
                                       "— auto-bypassed (audited)",
                                       task_id, drive_root,
                                       snapshot_paths=paths)

        # Explicit audited bypass.
        if skip_advisory_pre_review:
            return _record_bypass(ctx, state, snapshot_hash, commit_message,
                                   "explicit skip_advisory_review=True", task_id, drive_root,
                                   snapshot_paths=paths)

    readiness_warnings, changed_files = [], ""
    if not resuming:
        readiness_warnings, changed_files, early_exit = _advisory_pre_sdk_gate(
            ctx=ctx,
            repo_dir=repo_dir,
            drive_root=drive_root,
            snapshot_hash=snapshot_hash,
            commit_message=commit_message,
            paths=paths,
            skip_tests=skip_tests,
            review_rebuttal=review_rebuttal,
        )
        if early_exit is not None:
            return early_exit

    # Managed resolutions display the DISCLOSED dual counters instead of one
    # whole-candidate file count (display only — snapshot hashing above stays
    # on the full path set, I2). The counters ride out of the ONE subject
    # _advisory_review_diff builds inside the run below — never a second
    # subject recomputed here (full delta recomputation + display TOCTOU).
    try:
        ctx._last_advisory_subject_counters = ""  # reset: never a stale carry-over
    except Exception:
        pass

    def _snapshot_summary() -> str:
        # counters_line is fallback-aware: with M0 missing it reports the
        # resolution count as n/a instead of masquerading the full list.
        counters = str(getattr(ctx, "_last_advisory_subject_counters", "") or "")
        if counters:
            return counters
        return f"{changed_files.count(chr(10)) + 1} file(s) changed"

    import time as _time
    _advisory_start = _time.monotonic()
    items, raw_result, model_used, prompt_chars = _run_claude_advisory(
        repo_dir,
        commit_message,
        ctx,
        goal=goal,
        scope=scope,
        paths=paths,
        options={"drive_root": drive_root, "review_rebuttal": review_rebuttal,
                 "execution": execution, "snapshot_hash": snapshot_hash},
    )
    _advisory_duration = _time.monotonic() - _advisory_start
    advisory_meta = dict(getattr(ctx, "_last_claude_advisory_meta", {}) or {})
    advisory_session_id = str(advisory_meta.get("session_id") or "")
    execution = dict(advisory_meta.get("execution") or execution)

    # Delivery and deterministic syntax failures share persistence, while their
    # typed status/cause remain separate for admission and diagnostics.
    error_status, reason_kind, failure_message = "", "", ""
    if raw_result.startswith("⚠️ ADVISORY_ERROR"):
        error_status = "error"
        failure_message = "Advisory review failed; the complete source and cause remain recorded."
    elif raw_result.startswith("⚠️ PREFLIGHT_BLOCKED"):
        error_status, reason_kind = "preflight_blocked", "syntax"
        failure_message = (
            "Advisory delivery was skipped: a staged .py file has a SyntaxError. "
            "Fix the syntax error listed above and re-run preflight_review."
        )
    if error_status:
        _persist_preflight_record(ctx, snapshot_hash, commit_message, {
            "status": error_status, "reason_kind": reason_kind, "execution": execution, "strict": True,
            "review_rebuttal": review_rebuttal, "items": items, "raw_result": raw_result,
            "paths": paths, "duration_sec": _advisory_duration,
            "readiness_warnings": readiness_warnings, "prompt_chars": prompt_chars,
            "model_used": model_used, "session_id": advisory_session_id,
        })
        return _json_response({
            "status": error_status, "snapshot_hash": snapshot_hash,
            "error": raw_result, "session_id": advisory_session_id,
            "readiness_warnings": readiness_warnings, "message": failure_message,
        })

    # Prompt too large: persist non-blocking skipped run as fresh for this snapshot.
    if raw_result.startswith("⚠️ ADVISORY_SKIPPED:"):
        snapshot_summary = _snapshot_summary()
        def _mutate_skip(skip_state: AdvisoryReviewState) -> None:
            skip_state.add_run(_advisory_run_record(
                snapshot_hash, commit_message, "skipped",
                repo_key=repo_key, task_id=task_id,
                snapshot_summary=snapshot_summary, raw_result=raw_result,
                review_rebuttal=review_rebuttal, execution=execution,
                snapshot_paths=paths, readiness_warnings=readiness_warnings,
                prompt_chars=prompt_chars, model_used=model_used,
                session_id=advisory_session_id, duration_sec=_advisory_duration,
            ))

        update_state(drive_root, _mutate_skip)
        return _json_response({
            "status": "skipped",
            "snapshot_hash": snapshot_hash,
            "message": raw_result,
            "session_id": advisory_session_id,
            "readiness_warnings": readiness_warnings,
        })

    # Classify findings.
    critical_fails = [i for i in items if isinstance(i, dict)
                      and str(i.get("verdict", "")).upper() == "FAIL"
                      and str(i.get("severity", "")).lower() == "critical"]
    advisory_fails = [i for i in items if isinstance(i, dict)
                      and str(i.get("verdict", "")).upper() == "FAIL"
                      and str(i.get("severity", "")).lower() != "critical"]

    snapshot_summary = _snapshot_summary()

    # An empty array counts as a real "no findings" verdict only when the model
    # emitted the NO_FINDINGS sentinel the prompt asks for (REVIEW_JSON_ARRAY_CONTRACT),
    # or a bare `[]`-only body. A `[]` buried in refusal prose stays parse_failure.
    # Same predicate as triad, so one contract cannot mean two things.
    verified_clean = not items and _is_clean_verdict(raw_result)
    run_status = "fresh" if (items or verified_clean) else "parse_failure"
    if run_status == "parse_failure":
        execution.update(failure_phase="format", failure_code="parse_failure")
    run = _advisory_run_record(
        snapshot_hash, commit_message, run_status,
        repo_key=repo_key, task_id=task_id,
        items=items, snapshot_summary=snapshot_summary, raw_result=raw_result,
        review_rebuttal=review_rebuttal, execution=execution,
        snapshot_paths=paths, readiness_warnings=readiness_warnings,
        prompt_chars=prompt_chars, model_used=model_used,
        session_id=advisory_session_id, duration_sec=_advisory_duration,
    )

    # Locked read-modify-write against the LIVE ledger: the SDK call above runs
    # for minutes, and a state object loaded before it would clobber stale-marks
    # and concurrent runs recorded meanwhile (the pre-SDK `state` snapshot is
    # only used for gating decisions, never persisted from here on).
    def _record_run(live_state: "AdvisoryReviewState") -> None:
        live_state.add_run(run)
        if run_status != "parse_failure" and items:
            _resolve_matching_obligations(live_state, items, snapshot_hash, repo_key=repo_key)

    update_state(drive_root, _record_run)

    # Surface parse failures explicitly.
    if run_status == "parse_failure":
        return _json_response({
            "status": "parse_failure",
            "snapshot_hash": snapshot_hash,
            "error": "Advisory ran but returned no parseable checklist items.",
            "raw_result": _truncate_review_artifact(raw_result),
            "session_id": advisory_session_id,
            "readiness_warnings": readiness_warnings,
            "message": (
                "Advisory output could not be parsed. Re-run preflight_review, "
                "or use skip_advisory_review=True to bypass (will be audited)."
            ),
        })

    # Build human-readable summary.
    findings_summary: List[str] = []
    for item in critical_fails:
        findings_summary.append(f"  CRITICAL [{item.get('item','?')}]: {item.get('reason','')}")
    for item in advisory_fails:
        findings_summary.append(f"  ADVISORY [{item.get('item','?')}]: {item.get('reason','')}")

    result = {
        "status": "fresh",
        "snapshot_hash": snapshot_hash,
        "ts": run.ts,
        "items": items,
        "critical_count": len(critical_fails),
        "advisory_count": len(advisory_fails),
        "snapshot_summary": snapshot_summary,
        "session_id": advisory_session_id,
        "readiness_warnings": readiness_warnings,
        "message": (
            "Advisory review complete. No findings. Run commit_reviewed when ready."
            if verified_clean else
            f"Advisory review complete. {len(critical_fails)} critical, "
            f"{len(advisory_fails)} advisory findings. "
            + (
                "Fix issues and run commit_reviewed when ready."
                if _get_review_enforcement() == "blocking" else
                "Findings are recorded durably; enforcement is advisory — you "
                "decide which to apply. commit_reviewed is available when ready."
            )
        ),
    }
    if findings_summary:
        result["findings"] = findings_summary

    return _json_response(result)


def _handle_review_status(
    ctx: ToolContext,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: Optional[int] = None,
    include_raw: bool = False,
) -> str:
    """Show advisory freshness, review debt, guidance, and optional raw evidence."""
    projection = build_review_projection(
        ctx.drive_root,
        repo_dir=getattr(ctx, "repo_dir", ""),
        repo_key=repo_key,
        tool_name=tool_name,
        task_id=task_id,
        attempt=attempt,
        snapshot_hash_fn=compute_snapshot_hash,
    )
    next_step = _next_step_guidance(
        projection["guidance_run"],
        projection["state"],
        projection["stale_from_edit"],
        projection["stale_from_edit_ts"],
        projection["open_obligations"],
        projection["open_debts"],
        effective_is_fresh=projection["effective_is_fresh"],
        enforcement=_get_review_enforcement(),
        advisory_permitted=bool(projection["repo_commit_ready"]),
    )
    return json.dumps(
        build_review_status_payload(projection, next_step=next_step, include_raw=include_raw),
        ensure_ascii=False,
        indent=2,
    )


_schema_param = lambda param_type, description, **extra: {"type": param_type, "description": description, **extra}


def _preflight_review_params() -> dict:
    """The preflight_review tool's parameter schema (shared with its alias)."""
    return {
        "type": "object",
        "properties": {
            "commit_message": _schema_param("string", "Intended commit message. Used to bind the advisory run to this specific commit."),
            "skip_advisory_review": _schema_param(
                "boolean",
                "Choose the audited advisory-only skip for this call. "
                f"{ADVISORY_REVIEW_CHOICE_GUIDANCE} Default: False.",
                default=False,
            ),
            "goal": _schema_param("string", "High-level goal of this change. Used to judge completeness."),
            "scope": _schema_param("string", "Declared scope boundary. Issues outside scope are advisory-only."),
            "review_rebuttal": _schema_param("string", "Counter-argument to previous review findings, delivered in full to this preflight reviewer."),
            "paths": _schema_param("array", "Explicit list of changed file paths. Auto-detected from git status if omitted.", items={"type": "string"}),
            "skip_tests": _schema_param("boolean", "Skip the preflight pytest run. Default: False (tests run by default). Use True only for intentionally incomplete WIP code where test failures are expected. Tests are run before the paid critic call — in a hermetic worktree, as the same two passes CI runs (parallel 'not serial' then serial) — to catch broken code early and avoid wasting review budget.", default=False),
        },
        "required": ["commit_message"],
    }


def get_tools() -> list:
    return [
        ToolEntry(
            name="preflight_review",
            timeout_sec=1200,
            schema={
                "name": "preflight_review",
                "description": (
                    "Run the preflight pre-commit review (formerly `advisory_review`) "
                    "through the configured read-only route. "
                    "Returns structured JSON findings; any edit afterward makes the result stale. "
                    f"{ADVISORY_REVIEW_CHOICE_GUIDANCE} "
                    f"{_identical_diff_cap_note()}"
                ),
                "parameters": _preflight_review_params(),
            },
            handler=_handle_advisory_pre_review,
        ),
        # Q1 rename compatibility: the organ's old public name stays CALLABLE
        # (saved prompts/memories/configs keep working) but is never
        # advertised — schemas()/available_tools() skip alias entries. Same
        # parameters as the canonical entry so old calls keep their args.
        ToolEntry(
            name="advisory_review",
            timeout_sec=1200,
            alias_for="preflight_review",
            schema={
                "name": "advisory_review",
                "description": "Compatibility alias for `preflight_review`.",
                "parameters": _preflight_review_params(),
            },
            handler=_handle_advisory_pre_review,
        ),
        ToolEntry(
            name="review_status",
            schema={
                "name": "review_status",
                "description": (
                    "Show recent advisory pre-review run history. Read-only diagnostic — use to check advisory freshness before commit_reviewed. Also shows: last commit attempt state (reviewing/blocked/succeeded/failed) with block reason and actionable guidance; whether advisory is stale because of a worktree edit; open obligations from previous blocking rounds; open commit-readiness debt (durable repo-scoped anti-thrashing signal with fields `commit_readiness_debts`, `commit_readiness_debts_count`); `repo_commit_ready` (an advisory-readiness projection only: a fresh/bypassed/skipped advisory and no open advisory obligations or debt, not the full commit gate); `retry_anchor` (non-null, currently `commit_readiness_debt`, when debt is open — start the next retry from that record instead of patching one obligation at a time); and a concrete next_step recommendation. "
                    f"{ADVISORY_REVIEW_CHOICE_GUIDANCE} "
                    "Pass include_raw=true to surface the full per-actor evidence (triad_raw_results, scope_raw_result) for the targeted attempt."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_key": _schema_param("string", "Optional repo identity filter for attempt/advisory history."),
                        "tool_name": _schema_param("string", "Optional tool-name filter (for example commit_reviewed)."),
                        "task_id": _schema_param("string", "Optional task-id filter for attempt/advisory history."),
                        "attempt": _schema_param("integer", "Optional attempt number filter within the selected repo/tool/task scope."),
                        "include_raw": _schema_param("boolean", "If true, append full per-actor evidence (triad_raw_results, scope_raw_result) for the targeted commit attempt to the output. Without this flag the output contains only structured summaries. Defaults to false."),
                    },
                    "required": [],
                },
            },
            handler=_handle_review_status,
        ),
    ]


# v7next F2.3b (D06): moved spans live in their owner leaves; re-exported
# here so this facade stays the single import surface for callers and tests.
from ouroboros.tools.preflight_review_prompt import (  # noqa: E402, F401 -- intentional public re-exports
    _MAX_DIFF_CHARS_ERROR,
    _build_advisory_prompt,
    _build_blocking_history_section,
    _get_changed_file_list,
    _get_staged_diff,
    _mandatory_read_budget_section,
    _mandatory_read_corpus_chars,
)
from ouroboros.tools.preflight_review_run import (  # noqa: E402, F401 -- intentional public re-exports
    _ADVISORY_EXTRACT_CONTRACT,
    _advisory_failure,
    pending_advisory_execution,
    _ADVISORY_PROMPT_MAX_CHARS,
    _ADVISORY_SESSION_MAX_SECONDS,
    _check_expected_items,
    _is_checklist_array,
    _is_clean_verdict,
    _llm_extract_advisory_items,
    _needs_fallback_extraction,
    _note_meta_error,
    _parse_advisory_output,
    _resolve_fallback_model,
    _run_advisory_delegated,
    _run_claude_advisory,
    advisory_gate_unavailability_reason,
    advisory_gate_unavailable,
    advisory_review_route,
    advisory_slot_enabled,
)
