"""Enforcement-aware Atlas-backed scope reviewer for the commit pipeline.

Runs beside triad review and sees touched context plus a generated repo atlas. Critical findings follow
``OUROBOROS_REVIEW_ENFORCEMENT``: blocking enforcement blocks, advisory
enforcement reports them without blocking. Infrastructure failures such as
model errors, empty output, parse failures, and touched-context errors still
fail closed, and so does an oversized prompt. In owner-selected ``low`` context
mode the reviewer is not called at all and a typed skip row is recorded instead.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

from ouroboros.llm import LLMClient
from ouroboros.review_substrate import review_repo_dirs_for, scope_reviewer_slots
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_context_atlas import (
    ReviewContextAtlasRequest,  # noqa: F401
    atlas_assembly_failed,  # noqa: F401
    atlas_assembly_failure_reason,  # noqa: F401
    atlas_hard_budget_overflowed,  # noqa: F401
    atlas_required_beyond_diff,  # noqa: F401
    atlas_unassembled_required,  # noqa: F401
    compile_review_context_atlas,  # noqa: F401
)
from ouroboros.tools.scope_review_contract import (
    SCOPE_REQUIRED_ITEMS,
    TouchedContextStatus as _TouchedContextStatus,
    build_scope_block_message as _build_block_message,
    classify_scope_findings as _classify_scope_findings,
    compute_touched_context_status as _compute_touched_status,  # noqa: F401
    ladder_terminal_cause as _ladder_terminal_cause,
    normalize_scope_items as _normalize_scope_items,
)
from ouroboros.tools.review_binary_context import (
    StagedDiffUnavailable, capture_staged_diff, staged_path_is_binary)  # noqa: F401
from ouroboros.tools.review_synthesis import build_scope_review_prompt  # noqa: F401
from ouroboros.tools.review_helpers import (
    build_goal_section,  # noqa: F401
    build_rebuttal_section as _shared_build_rebuttal_section,  # noqa: F401
    build_scope_section,  # noqa: F401
    build_touched_file_pack,  # noqa: F401
    load_checklist_section,  # noqa: F401
    review_drive_root,
    CRITICAL_FINDING_CALIBRATION,  # noqa: F401
    BINARY_EXTENSIONS,  # noqa: F401
    _SENSITIVE_EXTENSIONS,  # noqa: F401
    _SENSITIVE_NAMES,  # noqa: F401
    load_governance_doc,  # noqa: F401
    _ANTI_THRASHING_RULE_VERDICT,  # noqa: F401
    _CONVERGENCE_RULE_TEXT,  # noqa: F401
    _HISTORY_VERIFICATION_ONLY_RULE,  # noqa: F401
    build_review_history_section as _shared_review_history_section,  # noqa: F401
    format_review_history_entry,  # noqa: F401
    parse_git_name_status,  # noqa: F401
)
from ouroboros.triad_review import REVIEW_JSON_MATRIX_CONTRACT, extract_json_array
from ouroboros.utils import (
    run_cmd,  # noqa: F401
    utc_now_iso,
    append_jsonl,
    estimate_tokens,
    truncate_review_artifact as _truncate_review_artifact,
)
from ouroboros.reviewer_window import ReviewerWindow
from ouroboros.tools.scope_review_budget import (  # noqa: F401 - facade for the extracted budget owner
    _SCOPE_BUDGET_TOKEN_LIMIT,
    _SCOPE_FAILCLOSED_WINDOW,
    _SCOPE_INPUT_TOKEN_LIMIT,
    _SCOPE_MAX_TOKENS,
    _SCOPE_MODEL_CONTEXT_WINDOW,
    _SCOPE_MODEL_DEFAULT,
    _SCOPE_OUTPUT_MARGIN_TOKENS,
    _SCOPE_REVIEW_SLOT_TIMEOUT_SEC,
    _calibrated_input_token_limit,
    _effective_scope_input_limit,
    _get_scope_model,
    _is_provider_oversize_error,
    _provider_error_is_oversize,
    _shared_window_scaled_reserves,
    _window_scaled_reserves,
)
from ouroboros.tools.scope_review_pack import (  # noqa: F401 - facade for the extracted pack owner
    _CANONICAL_CONTEXT_DOCS,
    _CURRENT_TOUCHED_CONTEXT_SKIP_PREFIXES,
    _DELETED_INLINE_MAX_BYTES,
    _SCOPE_CONTEXT_MANIFEST,
    _SCOPE_STABLE_PREFIX_LEN,
    _ScopeAtlasNotAssembled,
    _ScopePromptContext,
    _build_review_history_section,
    _build_scope_history_section,
    _build_scope_prompt,
    _classify_deleted_for_inline,
    _current_scope_context_manifest,
    _degradable_diff_only_paths,
    _gather_scope_packs,
    _inline_deleted_file_pack,
    _load_canonical_context_docs,
    _parse_staged_name_status,
    _record_ladder_steps,
    _render_touched_section,
    _should_skip_current_touched_context,
)

log = logging.getLogger(__name__)
_SCOPE_REQUIRED_ITEMS = SCOPE_REQUIRED_ITEMS  # compatibility export used by tests/review tooling


def _scope_review_skipped_in_low_context() -> bool:
    """Whether the owner's context mode declares scope review out of scope.

    Owner policy coupling (v6.80.0), NOT a structural impossibility: a narrow
    cognitive horizon means whole-repo scope review is declaredly not performed. The
    triad's blocking staged-diff review is unaffected in every mode.

    Reads the OWNER-SELECTED mode, never the effective one: the /api/settings model
    auto-downgrade narrows the effective mode on an agent-reachable path, and honouring
    that here would let the agent switch this gate off for its own commits."""
    try:
        from ouroboros.config import get_owner_context_mode
        return get_owner_context_mode() == "low"
    except Exception:
        return False


# Window authority moved to `tools/scope_window.py` (module-size gate); re-imported
# under the old private aliases so callers/tests keep one patch point on THIS module.
from ouroboros.tools.scope_window import (  # noqa: E402
    WINDOW_ASSERTED as _WINDOW_ASSERTED,  # noqa: F401 (test-read re-export)
    WINDOW_CONFIRMED as _WINDOW_CONFIRMED,  # noqa: F401 (test-read re-export)
    WINDOW_SENTINEL as _WINDOW_SENTINEL,  # noqa: F401 (test-read re-export)
    WINDOW_STALE as _WINDOW_STALE,  # noqa: F401 (test-read re-export)
    WINDOW_UNKNOWN as _WINDOW_UNKNOWN,  # noqa: F401 (test-read re-export)
    scope_window as _scope_window,
    scope_window_provenance as _scope_window_provenance,
    window_provenance_phrase as _window_provenance_phrase,
)

def _low_context_skip_result(scope_model: str) -> "ScopeReviewResult":
    """Typed, non-blocking record of the owner-declared low-context-mode skip.

    Without a durable row a low-mode commit is forensically indistinguishable from
    the bug "scope review silently failed to launch" (BIBLE P1: every significant
    cognitive act stays reconstructible). It rides the SAME review-evidence surface
    that records the fail-closed results (``build_scope_actor_record``)."""
    return ScopeReviewResult(
        blocked=False,
        status="skipped_low_context_mode",
        model_id=scope_model,
        prompt_chars=0,
        prompt_chars_source="not_assembled",
        advisory_findings=[{
            "verdict": "PASS",
            "severity": "advisory",
            "item": "scope_review_skipped_low_context_mode",
            "reason": (
                "ℹ️ SCOPE_REVIEW_SKIPPED_LOW_CONTEXT_MODE: the owner-selected `low` "
                "context mode declares whole-repository scope review not performed, so "
                "no scope reviewer was called and scope did not gate this commit. This "
                "is an owner policy coupling, not a capability limit: the triad's "
                "blocking staged-diff review ran in full, as it does in every mode. "
                "Switch the context mode to `max` to restore the blocking scope gate."
            ),
            "model": scope_model,
        }],
    )


def _scope_sub_floor_finding(
    scope_model: str, window: int, provenance: str = _WINDOW_UNKNOWN, observed_at: str = "",
) -> dict:
    return {
        "verdict": "FAIL",
        "severity": "advisory",
        "item": "scope_review_sub_floor",
        "reason": (
            f"⚠️ SCOPE_REVIEW_SUB_FLOOR: scope reviewer {scope_model} resolves to a "
            f"{_window_provenance_phrase(window, provenance, observed_at)} for authority purposes, "
            "which does not establish the >=1M blocking scope floor with sourced, "
            "current Capability Evidence (BIBLE P3). Its findings are ADVISORY-ONLY "
            "and cannot satisfy the blocking scope gate; connect the provider so the "
            "route can be probed, owner-ack this route's window, or configure a "
            ">=1M-window scope model, to restore an authoritative verdict."
        ),
        "model": scope_model,
    }


@dataclass
class ScopeReviewResult:
    """Structured outcome from ``run_scope_review``."""
    blocked: bool = False
    block_message: str = ""
    parsed_items: List[dict] = field(default_factory=list)
    critical_findings: List[dict] = field(default_factory=list)
    advisory_findings: List[dict] = field(default_factory=list)
    # Canonical per-actor evidence.
    raw_text: str = ""
    model_id: str = ""
    # responded|error|parse_failure|empty_response|budget_exceeded|fixed_overflow|
    # sub_floor|session_advisory|omitted|empty — only `responded` is AUTHORITATIVE
    status: str = "responded"
    prompt_chars: int = 0
    # measured (len(prompt)) | estimated_from_tokens (no prompt was assembled)
    prompt_chars_source: str = "measured"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    context_manifest: dict = field(default_factory=dict)
    prompt_ref: dict = field(default_factory=dict)
    response_ref: dict = field(default_factory=dict)


def _log_scope_result(
    ctx: ToolContext,
    critical_count: int,
    advisory_count: int,
    prompt_chars: int = 0,
    prompt_tokens: int = 0,
    model_id: str = "",
) -> None:
    """Append a scope_review_complete event to events.jsonl.

    Also emits budget headroom metrics so operators can see when the scope
    pack is approaching the gate. ``headroom_tokens`` is a signed delta
    (negative when the prompt exceeds the gate — would have been skipped).
    """
    prompt_tokens = int(prompt_tokens or 0)
    if prompt_tokens <= 0 and prompt_chars:
        prompt_tokens = max(0, int(prompt_chars) // 4)
    input_limit = _effective_scope_input_limit(scope_model=model_id)
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(), "type": "scope_review_complete",
            "task_id": getattr(ctx, "task_id", "") or "",
            "model": model_id or _get_scope_model(),
            "critical_count": critical_count,
            "advisory_count": advisory_count,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_budget": input_limit,
            "headroom_tokens": input_limit - prompt_tokens,
        })
    except Exception:
        pass


def _call_scope_llm(
    prompt: str,
    scope_model: str | None = None,
    ctx: ToolContext | None = None,
    slot_id: str = "",
    route: Any = None,
    session_task: str = "",
    session_root: str = "",
    slot_effort: str = "",
    session_target: str = "",
    session_profile: str = "",
) -> tuple:
    """Execute the scope review call synchronously — api pack or agent session.

    Returns (raw_text, usage, error_msg) — error_msg is non-empty on failure.
    ``usage`` may contain a private ``_review_refs`` entry with durable prompt
    and response refs from the shared review substrate.

    ``slot_id`` is the identity of the configured row this call belongs to,
    supplied by whoever fanned the rows out. ``route`` is the row's configured
    delivery: on ``agent_session`` the substrate's session executor delivers
    ``session_task`` in ``session_root`` and the api pack is never rendered
    (5.2); parsing, classification and blocking above this call are identical
    for both deliveries (5.3)."""
    from ouroboros.config import resolve_effort as _resolve_effort
    from ouroboros.review_execution import ReviewRouteKind

    scope_model = scope_model or _get_scope_model()
    # 6.1/6.3: the row's own effort wins; the global key stays the default.
    scope_effort = slot_effort or _resolve_effort("scope_review")
    delegated = str(getattr(route, "value", route) or "") == "agent_session"
    # Output budget scales with the reviewer window: requesting the absolute
    # 100K reserve on a small-window model would 400 on input+max_tokens.
    _scope_output_tokens, _ = _window_scaled_reserves(
        _scope_window(scope_model).sizing_window(_SCOPE_FAILCLOSED_WINDOW)
    )
    if delegated:
        messages: Any = []
    else:
        # Split at the recorded stable/dynamic boundary: the byte-stable prefix
        # carries the provider cache marker, the per-commit tail stays unmarked.
        from ouroboros.tools.review_helpers import cached_prompt_blocks

        _stable_len = int(_SCOPE_STABLE_PREFIX_LEN.get() or 0)
        if 0 < _stable_len <= len(prompt):
            system_content: Any = cached_prompt_blocks(prompt[:_stable_len], prompt[_stable_len:])
        else:
            # No recorded boundary (e.g. a caller that did not assemble via
            # _build_scope_prompt): send a plain string. Marking the WHOLE prompt —
            # per-commit diff included — as a 1h cache block would pay the extended
            # write premium on content that never repeats.
            system_content = prompt
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": "Review the staged change and context above. Output ONLY a JSON array.",
            },
        ]
    try:
        from ouroboros.review_substrate import ReviewRequest, run_review_request

        request = ReviewRequest(
            surface="scope_review",
            goal="Review the staged change and context above. Output ONLY a JSON array.",
            messages=messages,
            task_id=str(getattr(ctx, "task_id", "") or "scope_review") if ctx is not None else "scope_review",
            call_type="scope_review",
            max_tokens=_scope_output_tokens,
            temperature=0.2,
            no_proxy=True,
            session_task=session_task if delegated else "",
            session_root=session_root if delegated else "",
            # The extraction fallback canonicalizes to the SCOPE contract: required-
            # matrix shape, eight verbatim item ids (D19 — never a looser contract).
            policy=(
                {
                    "output_contract": (
                        REVIEW_JSON_MATRIX_CONTRACT
                        + "\nRequired item ids (verbatim, one entry each): "
                        + ", ".join(sorted(SCOPE_REQUIRED_ITEMS))
                    ),
                }
                if delegated
                else {}
            ),
        )
        # Identity comes from the configured row, never from the row's model.
        row = scope_reviewer_slots([scope_model], effort=scope_effort)[0]
        slot = replace(
            row,
            slot_id=slot_id or row.slot_id,
            timeout_sec=_SCOPE_REVIEW_SLOT_TIMEOUT_SEC,
            max_tokens=_scope_output_tokens,
            temperature=0.2,
            # ROUTE is CARRIED, never re-derived: the one-element slot list above
            # re-reads ROUTES row 1, which sent a mixed config's api row as
            # agent_session — the caller's fanned-out route is the authority (p5x XG).
            route=ReviewRouteKind.AGENT_SESSION if delegated else ReviewRouteKind.API_CHAT,
            # The fanned-out row's own session target (6.1); '' keeps the
            # shared session-route fallback.
            session_target=session_target if delegated else "",
            session_profile=session_profile if delegated else "",
        )
        result = run_review_request(
            request,
            slots=[slot],
            drive_root=review_drive_root(ctx),
            llm=LLMClient(),
            usage_ctx=ctx,
        )
        actor = (result.actors or [{}])[0]
        usage = dict(actor.get("usage") or {})
        usage["_review_refs"] = {
            "prompt_ref": actor.get("prompt_ref") or {},
            "response_ref": actor.get("response_ref") or {},
        }
        if actor.get("status") not in {"ok", "empty"}:
            error_msg = (
                f"⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer ({scope_model}) failed — commit blocked.\n"
                f"Error: {actor.get('error') or actor.get('status') or 'scope reviewer failed'}\n"
                "Retry the commit, or check API key and network connectivity."
            )
            return "", usage, error_msg
        return str(actor.get("raw_text") or ""), usage, ""
    except Exception as e:
        error_msg = (
            f"⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer ({scope_model}) failed — commit blocked.\n"
            f"Error: {type(e).__name__}: {e}\n"
            "Retry the commit, or check API key and network connectivity."
        )
        return "", None, error_msg


def _scope_oversize_result(
    *,
    scope_model_id: str,
    prompt_chars: int,
    prompt_tokens_est: int,
    prompt_ref: dict,
    response_ref: dict,
    provider_detail: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> "ScopeReviewResult":
    """Return a visible, fail-closed oversize result."""
    authority_note = "The blocking scope gate has no authoritative verdict. "
    advisory = {
        "verdict": "FAIL",
        "severity": "advisory",
        "item": "scope_review_skipped",
        "reason": (
            f"⚠️ SCOPE_REVIEW_SKIPPED: the provider rejected the assembled scope prompt "
            f"(~{prompt_tokens_est} estimated tokens) as exceeding the model's real "
            f"context window. {authority_note}"
            "Provider error: "
            + _truncate_review_artifact(str(provider_detail), 1000)
        ),
        "model": scope_model_id,
    }
    return ScopeReviewResult(
        blocked=True,
        block_message=(
            "⚠️ SCOPE_REVIEW_BLOCKED: the provider rejected the scope prompt as "
            "oversized, so the required >=1M blocking scope gate produced no "
            "authoritative verdict. Split the staged change or restore a fitting "
            ">=1M reviewer route."
        ),
        status="fixed_overflow",
        model_id=scope_model_id,
        prompt_chars=prompt_chars,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        context_manifest=_current_scope_context_manifest(),
        prompt_ref=prompt_ref,
        response_ref=response_ref,
        advisory_findings=[advisory],
    )


def _handle_prompt_signals(
    prompt: Optional[str],
    context_status: Optional["_TouchedContextStatus"],
    input_limit: int = _SCOPE_INPUT_TOKEN_LIMIT,
    scope_model: str = "",
) -> Optional[ScopeReviewResult]:
    """Translate touched-context status into an early ScopeReviewResult."""
    if context_status is None:
        return None  # proceed with LLM call

    if context_status.status == "budget_exceeded":
        token_count = context_status.token_count
        # Report the REAL window-scaled reserves, not the 1M constants.
        _resolved = _scope_window(scope_model) if scope_model else ReviewerWindow(
            window_tokens=_SCOPE_MODEL_CONTEXT_WINDOW,
        )
        _window = _resolved.sizing_window(_SCOPE_FAILCLOSED_WINDOW)
        _provenance = _scope_window_provenance(_resolved)
        _output_reserve, _ = _window_scaled_reserves(_window)
        _budget = (f"input budget ({input_limit} tokens, reserving {_output_reserve} for "
                   f"output within its {_window_provenance_phrase(_window, _provenance, _resolved.observed_at)})")
        _cause, _remedy = _ladder_terminal_cause(context_status, input_limit, budget_phrase=_budget)
        log.warning(
            "Scope review pack did not assemble: %s; window=%d provenance=%s (fail-closed).",
            _cause, _window, _provenance,
        )
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                f"⚠️ SCOPE_REVIEW_BLOCKED: {_cause}, so the required >=1M blocking "
                "scope gate has no authoritative verdict."
            ),
            status="sub_floor",
            # No prompt string exists on this path (ladder sentinel): the char count
            # is DERIVED from the token estimate and labelled as such.
            prompt_chars=token_count * 4,
            prompt_chars_source="estimated_from_tokens",
            advisory_findings=[{
                "verdict": "FAIL",
                "severity": "advisory",
                "item": "scope_review_skipped",
                "reason": (
                    f"⚠️ SCOPE_REVIEW_SKIPPED: {_cause}. The blocking scope gate has "
                    f"no authoritative verdict. {_remedy}"
                ),
                "model": scope_model or "scope_reviewer",
            }],
        )

    if context_status.status == "fixed_overflow":
        # The ladder exhausted every degradation step. TWO failures land here — an
        # irreducible overflowing prompt, and a REQUIRED artifact that never
        # assembled — and they can COINCIDE, so the cause(s) are READ from the
        # status and every one that applies is rendered. Fails CLOSED either way.
        token_count = context_status.token_count
        cause, remedy = _ladder_terminal_cause(context_status, input_limit)
        return ScopeReviewResult(
            blocked=True,
            status="fixed_overflow",
            prompt_chars=token_count * 4,
            prompt_chars_source="estimated_from_tokens",
            block_message=(
                f"⚠️ SCOPE_REVIEW_BLOCKED: {cause}. {remedy} "
                "Fail-closed stop — not a skippable budget condition."
            ),
        )

    if context_status.status == "empty":
        return ScopeReviewResult(
            blocked=True,
            status="empty",
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Could not read any touched files — "
                "scope review requires direct file context. Commit blocked."
            ),
        )

    if context_status.status == "omitted":
        omitted_names = ", ".join(context_status.omitted_paths) or "(unknown)"
        return ScopeReviewResult(
            blocked=True,
            status="omitted",
            block_message=(
                f"⚠️ SCOPE_REVIEW_BLOCKED: Some touched file(s) could not be included "
                f"in direct context (binary/oversize/unreadable): {omitted_names}.\n"
                "Scope review requires complete touched-file context. Commit blocked.\n"
                "Possible fixes: reduce file size, commit binary files separately, "
                "or ensure all touched files are readable text."
            ),
        )

    # Unknown status is a programming error; fail closed.
    log.error(
        "Scope review: unrecognised _TouchedContextStatus.status=%r — blocking commit (fail-closed).",
        context_status.status,
    )
    return ScopeReviewResult(
        blocked=True,
        status="error",
        block_message=(
            f"⚠️ SCOPE_REVIEW_BLOCKED: Unexpected context status '{context_status.status}' — "
            "commit blocked (fail-closed). This is a programming error; please report it."
        ),
    )


def _apply_scope_authority(
    critical_findings: List[dict],
    advisory_findings: List[dict],
    *,
    scope_model_id: str,
    result_kwargs: dict,
    delegated: bool = False,
) -> tuple[List[dict], List[dict], Optional[ScopeReviewResult]]:
    """One-pass P3 authority for THIS row's delivery: is the reviewer's window ESTABLISHED
    enough for its verdict to gate a commit? ``api_chat`` must fit the whole assembled pack
    (constitutional >=1M; sub-floor BLOCKS); ``agent_session`` assembles none and needs
    SOURCED window evidence instead (scope_review_session owns that decision). NEITHER is
    waved through — skipping this for sessions let one gate with no window test at all.
    Authority is read from the EVIDENCE, not from the number: a window that is merely
    large enough is not a window that was established — an expired record, an outage-
    carried record, and the unevidenced designated-default sentinel all size a prompt
    at >=1M and all fail here (the BIBLE P3 rule stated in code).

    WHOSE window, for a retrieving row: the ACKED HARNESS ROUTE's. ``scope_model_id``
    is that row's opaque ``harness[=model]`` spec, and `reviewer_window.reviewer_route`
    fingerprints it under its own provider precisely so the owner's ack is recorded
    against the route it travels. It is NOT the model the engine later reports back —
    that arrives only after the run, is absent on telemetry that predates the receipt,
    and would make the authority of a row depend on a fact no pre-flight can know.
    Re-keying this lookup to the reported model was measured: it fails every session
    scope row, closing a delivery path the owner deliberately opened. When the engine
    resolves something other than what the route asked for, that divergence is already
    disclosed on its own axis — ``capability_delta``, reason
    ``session_route_resolves_its_own_model`` — which is where a landing below the ask
    belongs, not in the window predicate."""
    resolved = _scope_window(scope_model_id, session=delegated)
    if delegated:
        from ouroboros.tools.scope_review_session import session_scope_authority

        # EVIDENCE, never the sizing fallback: the session floor is gated on SOURCED
        # provenance, and a fail-closed sizing number handed over as a window would
        # read as evidence for exactly the session-floor number. A STALE record sizes
        # a prompt but authorises nothing (api-row rule): provenance blanked first.
        return session_scope_authority(
            critical_findings, advisory_findings, scope_model=scope_model_id,
            window=int(resolved.window_tokens or 0),
            provenance="" if resolved.stale else str(resolved.status or ""),
            result_kwargs=result_kwargs,
            phrase=_window_provenance_phrase(
                resolved.sizing_window(_SCOPE_FAILCLOSED_WINDOW),
                _scope_window_provenance(resolved), resolved.observed_at),
        )
    if resolved.blocking_authority_allowed:
        return critical_findings, advisory_findings, None
    window = resolved.sizing_window(_SCOPE_FAILCLOSED_WINDOW)
    provenance = _scope_window_provenance(resolved)
    for finding in critical_findings:
        finding["severity"] = "advisory"
        finding["reason"] = "[sub-floor scope reviewer] " + str(finding.get("reason", ""))
    advisory_findings = list(critical_findings) + list(advisory_findings)
    critical_findings = []
    advisory_findings.append(
        _scope_sub_floor_finding(scope_model_id, window, provenance, resolved.observed_at)
    )
    return critical_findings, advisory_findings, ScopeReviewResult(
        blocked=True,
        block_message=(
            f"⚠️ SCOPE_REVIEW_BLOCKED: scope reviewer {scope_model_id} has a "
            f"{_window_provenance_phrase(window, provenance, resolved.observed_at)}, which does not "
            "establish the required >=1M floor with sourced Capability Evidence. Its "
            "advisory findings were preserved, but it cannot supply the authoritative "
            "scope verdict required to commit."
        ),
        critical_findings=critical_findings,
        advisory_findings=advisory_findings,
        status="sub_floor",
        **result_kwargs,
    )


def run_scope_review(
    ctx: ToolContext,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
    review_history: Optional[list] = None,
    scope_review_history: Optional[list] = None,  # prior scope rounds for this commit
    scope_model: Optional[str] = None,
    slot_id: str = "",  # identity of the configured row this call runs (see scope_reviewer_slots)
    route: Any = None,  # the row's configured delivery (ReviewRouteKind); None/api_chat = api
    slot_effort: str = "",  # the row's own effort (6.1); "" = global scope_review effort
    session_target: str = "",  # the row's own harness[=model] target; "" = shared route
    session_profile: str = "",  # optional credential pin (Q2-в); "" = rotation
) -> ScopeReviewResult:
    """Run the blocking scope review, or record the owner-declared low-mode skip."""
    if _scope_review_skipped_in_low_context():
        return _low_context_skip_result(scope_model or _get_scope_model())
    try:
        governance_repo, repo_dir = review_repo_dirs_for(ctx)
    except (TypeError, ValueError) as exc:
        return ScopeReviewResult(
            blocked=True,
            status="error",
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: invalid review roots: {exc}.",
        )
    scope_model_id = scope_model or _get_scope_model()
    delegated = str(getattr(route, "value", route) or "") == "agent_session"

    from ouroboros.tools.registry import _authorized_managed_update_resolver

    try:
        if delegated:
            # Session delivery (5.2): same task/checklist/contract, no assembled
            # pack — the session retrieves with its own tools in the repo root.
            from ouroboros.tools.scope_review_session import ScopeIntentContext as _Intent
            from ouroboros.tools.scope_review_session import build_scope_session_task

            session_task, session_manifest = build_scope_session_task(
                repo_dir, commit_message,
                _Intent(goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                        review_history=review_history,
                        scope_review_history=scope_review_history),
                drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
                governance_repo_dir=governance_repo,
            )
            _SCOPE_CONTEXT_MANIFEST.set(session_manifest)
            prompt, context_status = session_task, None
        else:
            session_task = ""
            prompt, context_status = _build_scope_prompt(
                repo_dir, commit_message,
                goal=goal, scope=scope,
                review_rebuttal=review_rebuttal,
                review_history=review_history,
                scope_review_history=scope_review_history,
                context=_ScopePromptContext(
                    drive_root=(
                        pathlib.Path(ctx.drive_root)
                        if getattr(ctx, "drive_root", None)
                        else None
                    ),
                    scope_model=scope_model_id,
                    governance_repo_dir=governance_repo,
                    represent_binary=_authorized_managed_update_resolver(ctx),
                ),
            )
    except RuntimeError as exc:
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Failed to build review context — commit blocked.\n"
                f"Error: {exc}\n"
                "Ensure git is available and the repository is in a valid state."
            ),
            model_id=scope_model_id,
            status="error",
            context_manifest=_current_scope_context_manifest(),
        )

    # Pack-budget signals belong to an ASSEMBLED pack: a session assembles none, so its
    # context_status is None and this returns None by construction — no route branch.
    signal_result = _handle_prompt_signals(
        prompt, context_status, scope_model=scope_model_id,
        input_limit=_effective_scope_input_limit(scope_model=scope_model_id),
    )
    if signal_result is not None:
        # Keep _handle_prompt_signals as the status SSOT for early exits.
        signal_result.model_id = scope_model_id
        signal_result.context_manifest = _current_scope_context_manifest()
        return signal_result

    _prompt_chars = len(prompt)  # type: ignore[arg-type]
    _prompt_tokens_est = estimate_tokens(prompt)  # type: ignore[arg-type]
    raw_text, usage, llm_error = _call_scope_llm(
        prompt, scope_model=scope_model_id, ctx=ctx, slot_id=slot_id,
        route=route, session_task=session_task, session_root=str(repo_dir),
        slot_effort=slot_effort, session_target=session_target,
        session_profile=session_profile,
    )  # type: ignore[arg-type]
    _usage = dict(usage or {})
    _review_refs = dict(_usage.pop("_review_refs", {}) or {})
    _prompt_ref = dict(_review_refs.get("prompt_ref") or {})
    _response_ref = dict(_review_refs.get("response_ref") or {})
    _tokens_in = int(_usage.get("prompt_tokens", 0) or 0)
    _tokens_out = int(_usage.get("completion_tokens", 0) or 0)
    _cost_usd = float(_usage.get("cost", 0.0) or 0.0)
    if llm_error:
        if _is_provider_oversize_error(llm_error):
            # The estimate-based gate passed but the provider's REAL tokenizer called
            # the prompt oversize: no authoritative verdict, so the >=1M gate fails
            # CLOSED (v6.80.0: not configurable; owner controls only context mode).
            log.warning(
                "Scope reviewer rejected the prompt as oversize "
                "(estimate-gate passed; real tokenizer denser). Failing the "
                "blocking scope gate closed. Error: %s", llm_error,
            )
            return _scope_oversize_result(
                scope_model_id=scope_model_id,
                prompt_chars=_prompt_chars,
                prompt_tokens_est=_prompt_tokens_est,
                prompt_ref=_prompt_ref,
                response_ref=_response_ref,
                provider_detail=llm_error,
                tokens_in=_tokens_in,
                tokens_out=_tokens_out,
                cost_usd=_cost_usd,
            )
        return ScopeReviewResult(
            blocked=True,
            block_message=llm_error,
            model_id=scope_model_id,
            status="error",
            prompt_chars=_prompt_chars,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )
    # Usage emission happens ONCE, inside the shared review substrate
    # (source="review_substrate:scope_review", carrying ledger_attempt_ids). The old
    # job-level re-emit duplicated every scope call without attempt ids, so the pair
    # could not be deduplicated against the monetary ledger (v6.69.0).

    if _provider_error_is_oversize(_usage, _prompt_tokens_est, scope_model_id):
        # Gateway route (openai-compatible/OpenRouter): a real oversize 400 arrives as
        # an EMPTY body + usage['provider_error']{code:400}, not a raised "prompt is
        # too long" error — the llm_error branch above never fires and the empty body
        # would hard-block as empty_response. With INDEPENDENT size evidence, route
        # through the same fail-closed oversize result; non-size 400 stays blocking.
        _pe_msg = str((_usage.get("provider_error") or {}).get("message") or "")
        log.warning(
            "Scope reviewer hit provider_error code=400 oversize (empty body; "
            "estimate-gate passed). Failing the blocking scope gate closed. "
            "provider_error: %s", _pe_msg or "(no message)",
        )
        return _scope_oversize_result(
            scope_model_id=scope_model_id,
            prompt_chars=_prompt_chars,
            prompt_tokens_est=_prompt_tokens_est,
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
            provider_detail=_pe_msg,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
        )

    if not raw_text.strip():
        # Empty model response is distinct from transport/API error.
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer returned empty response — commit blocked.\n"
                "Retry the commit."
            ),
            model_id=scope_model_id,
            status="empty_response",
            prompt_chars=_prompt_chars,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )

    items = extract_json_array(raw_text, normalize=True)
    if items is None:
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Could not parse scope reviewer output as JSON — commit blocked.\n"
                "Full raw response preserved in scope_raw_result (status='parse_failure')."
            ),
            model_id=scope_model_id,
            status="parse_failure",
            raw_text=raw_text,
            prompt_chars=_prompt_chars,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )

    parsed_items, contract_error = _normalize_scope_items(items)
    if contract_error:
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer output violated the "
                "Intent / Scope Review Checklist coverage contract — commit blocked.\n"
                f"{contract_error}\n"
                "Retry the commit so scope review covers all required checklist items."
            ),
            model_id=scope_model_id,
            status="parse_failure",
            raw_text=raw_text,
            parsed_items=parsed_items,
            prompt_chars=_prompt_chars,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )

    critical_findings, advisory_findings = _classify_scope_findings(parsed_items)
    result_kwargs = {
        "parsed_items": parsed_items,
        "model_id": scope_model_id,
        "raw_text": raw_text,
        "prompt_chars": _prompt_chars,
        "tokens_in": _tokens_in,
        "tokens_out": _tokens_out,
        "cost_usd": _cost_usd,
        "context_manifest": _current_scope_context_manifest(),
        "prompt_ref": _prompt_ref,
        "response_ref": _response_ref,
    }
    critical_findings, advisory_findings, authority_block = _apply_scope_authority(
        critical_findings, advisory_findings, scope_model_id=scope_model_id,
        result_kwargs=result_kwargs, delegated=delegated,
    )
    if authority_block is not None:
        return authority_block
    _log_scope_result(
        ctx,
        len(critical_findings),
        len(advisory_findings),
        prompt_chars=_prompt_chars,
        prompt_tokens=_prompt_tokens_est,
        model_id=scope_model_id,
    )

    if critical_findings:
        from ouroboros import config as _cfg
        if _cfg.get_review_enforcement() == "blocking":
            return ScopeReviewResult(
                blocked=True,
                block_message=_build_block_message(critical_findings, advisory_findings),
                critical_findings=critical_findings,
                advisory_findings=advisory_findings,
                status="responded",
                **result_kwargs,
            )
        # Parallel review aggregates advisory findings on the main thread.

    return ScopeReviewResult(
        blocked=False,
        critical_findings=critical_findings,
        advisory_findings=advisory_findings,
        status="responded",
        **result_kwargs,
    )
