"""Pre-dispatch review admission (Q25=A / Q28=A).

Every commit-gate packet — the triad api pack and each scope row's pack — is
ASSEMBLED AND FIT-CHECKED before any reviewer is dispatched, so a
deterministic assembly failure on one side can never spend money on the other
(previously triad and scope dispatched concurrently). A universal reorder with
zero verdict change: the same assembly code runs, the same results come out —
only the ordering moves the spend after the last deterministic gate.

Q28-A oversized outcomes: packet limits gate only the api rows. A panel whose
agent-session rows alone satisfy the quorum proceeds without the api rows
(recorded, never silent); a panel that cannot reach quorum without them gets a
typed ZERO-SPEND terminal, and for the managed resolver that refusal carries
the settings guidance below (the resolver's terminal contract already explains
rollback + retry).

``prepare_scope_review`` is the assembly half of ``run_scope_review`` — moved
here whole; the dispatch half stays in ``scope_review``. Internals are reached
through the module object (``_scope().name``) so test monkeypatching of
``scope_review`` attributes keeps working.

Both packets share ONE cold-start density rung (owner decision 2026-09-05,
answer 1 = A: the commit gate gets the SAME rung the packed deep self-review
has): a pack that would be refused or degraded for SIZE while the reviewer
route has NO fresh exact-model density witness gets one bounded probe send on
the exact model (``capability_evidence.cold_start_density_probe``), the witness is
recorded, the cap recomputed and the pack rebuilt ONCE — never on a warm
store, never retried, never on a commit whose pack fits. A probe the paid
ledger refuses is a typed disclosure on the ladder and in the review events,
and the existing refusal path proceeds unchanged.

Money admission (owner decision 2026-09-05, answer 2 = A) is the last
pre-dispatch gate: ``commit_gate_paid_seats`` prices every PAID seat of the
wave — scope first, packet rows by their exact message pair, native episodes
by their exact first send — each under the usage scope its substrate sends
under, and ``admit_commit_gate_wave`` admits them as ONE wave against the
task's current root fence through the shared ``review_wave_budget_gate``; a
wave that does not fit is a typed $0 refusal naming the shortfall, never a
half-dispatched panel. The scope-first dispatch ORDER stays with the
orchestrator (``parallel_review._await_scope_reservation``).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)

# A managed resolution stages the whole two-parent merge tree by contract: the
# ordinary "split the commit" remedy is structurally impossible for it, so every
# managed oversize terminal REPLACES that clause with these two sentences.
MANAGED_SPLIT_IMPOSSIBLE = (
    "A managed-update resolution stages the whole two-parent merge tree and "
    "cannot be split into smaller commits."
)
MANAGED_OVERSIZE_GUIDANCE = (
    "Switch or add agent-route reviewer rows in "
    "Settings → Agents → Review lanes (packet limits do not apply to them), "
    "or configure larger-window models."
)

# Pack-SIZE terminal statuses (and only those): the Q28-A session-quorum
# override applies to packet limits, never to fail-closed integrity statuses
# (omitted/unreadable touched context, infra errors). Measured set: the
# assembly ladder's sub-floor pack arrives as `sub_floor` and its irreducible
# overflow as `fixed_overflow`; no prepare-time ScopeReviewResult carries a
# `budget_exceeded` status (that name exists only on the TouchedContextStatus
# sentinel, which _handle_prompt_signals translates to `sub_floor`).
#
# Q28-A yield scope (deliberate, admission-time ONLY): the session-quorum
# override applies where these statuses are produced — packet ASSEMBLY, before
# any dispatch. A DISPATCH-TIME oversize (the provider's own tokenizer
# rejecting an assembled prompt, scope_review status `fixed_overflow` from the
# transport) stays a failed row, not an admission-time quorum yield. The commit
# aggregate separately applies owner-selected advisory enforcement to typed
# technical failures; it never promotes that row to an authoritative verdict.
SCOPE_FIT_BLOCK_STATUSES = frozenset({"sub_floor", "fixed_overflow"})


def _scope():
    from ouroboros.tools import scope_review

    return scope_review


# The scope ladder's SIZE terminals (never the integrity ones: empty/omitted).
_SCOPE_SIZE_TERMINALS = frozenset({"fixed_overflow", "budget_exceeded"})

DENSITY_PROBE_CALL_TYPE = "review_density_probe"
DENSITY_PROBE_EVENT = "review_density_probe"


def density_probe_before_size_refusal(ctx: Any, model: str, sample: str, *, surface: str) -> str:
    """The commit gate's cold-start density rung; returns the shared probe's
    typed outcome (``capability_evidence.cold_start_density_probe``) plus
    ``"budget_refused"`` and ``"unavailable"`` (no ctx/drive root to record a
    witness on — a bare fit-check never sends).

    Disclosure mirrors the deep review's: every attempted probe is a progress
    line (``ctx.emit_progress_fn``) and one ``review_density_probe`` review
    event (the scope caller adds the ladder step); the send itself lands in
    custody through ``chat_observed`` and in the paid ledger like any review
    send. A ``BudgetExceeded`` from the ledger is the typed
    ``budget_refused`` outcome — the pack keeps its existing size refusal;
    nothing crashes and nothing is retried. The client comes from the review
    surface's ``LLMClient`` seam and any failure short of the ledger's refusal
    (a client that cannot be constructed included) is the typed ``failed``
    outcome, never an untyped infra block of the whole gate."""
    from ouroboros.capability_evidence import cold_start_density_probe
    from ouroboros.tools import review as _rv
    from ouroboros.tools.review_helpers import emit_review_event, review_drive_root
    from ouroboros.usage_accounting import BudgetExceeded

    if ctx is None or not getattr(ctx, "drive_root", None):
        return "unavailable"

    def _progress(text: str) -> None:
        try:
            ctx.emit_progress_fn(f"{surface}: {text}")
        except Exception:
            pass

    try:
        outcome = cold_start_density_probe(
            review_drive_root(ctx), _rv.LLMClient(), _progress, str(model), sample,
            task_id=str(getattr(ctx, "task_id", "") or "") or "commit_review",
            call_type=DENSITY_PROBE_CALL_TYPE, source="commit_gate_cold_start_probe",
        )
        reason = ""
    except BudgetExceeded as exc:
        outcome, reason = "budget_refused", str(exc)
        _progress(f"density probe refused by the budget ({reason}); the cold input cap stands.")
    except Exception as exc:
        outcome, reason = "failed", f"{type(exc).__name__}: {exc}"
        log.warning("Density probe could not run (%s): %s", surface, exc, exc_info=True)
        _progress(f"density probe failed ({type(exc).__name__}); the cold input cap stands.")
    if outcome not in ("warm", "no_sample"):
        emit_review_event(ctx, {
            "type": DENSITY_PROBE_EVENT, "surface": surface, "model": str(model),
            "outcome": outcome, "reason": reason,
            "task_id": str(getattr(ctx, "task_id", "") or ""),
        })
    return outcome


def fit_triad_prompt(api_models: list, assemble, current_files_section: str,
                     diff_text: str, changed: str, target_repo, ctx=None,
                     subject=None) -> tuple:
    """The api pack's guaranteed-fit ladder (P3 one-pass): drop only evidence
    duplicated by the complete staged diff — full snapshots first, then unchanged
    diff context. Each api slot's limit uses its REAL window from Capability
    Evidence (a hardcoded 1M treated a 200K reviewer as 1M-capable and lost its
    whole review to a deterministic prompt-too-long 400), with sub-1M windows
    scaling their reserves so a small-window slot gets a fit-sized pack, not a
    zero limit; the shared prompt is sized to the review QUORUM — the same SSOT
    plan review uses — so one small slot degrades its OWN seat rather than
    blocking the gate for the whole panel. Session rows are not constrained by
    this pack at all (5.2/5.7): they retrieve with their own tools. Returns
    ``(prompt, stable_prefix_len, block_message_or_empty)``."""
    # Resolved through the review-module namespace on purpose: these names are
    # documented monkeypatch seams pinned by the fit-ladder tests.
    from ouroboros.tools import review as _rv

    def _slot_input_limit(slot_model: str) -> int:
        window = _rv.reviewer_context_window(slot_model)
        output_reserve, tokenizer_margin = _rv.window_scaled_reserves(
            window,
            output_reserve=_rv._review_output_budget(),
            tokenizer_margin=50_000,
        )
        return max(0, _rv.calibrated_input_token_limit(
            slot_model,
            context_window=window,
            output_reserve=output_reserve,
            tokenizer_margin=tokenizer_margin,
            budget_cap=_rv.REVIEW_PROMPT_TOKEN_BUDGET,
        ))

    estimate_tokens = _rv.estimate_tokens
    slot_limits = {m: _slot_input_limit(m) for m in api_models}
    input_limit = _rv._quorum_input_token_limit(api_models, slot_limits)
    prompt, stable_prefix_len = assemble(current_files_section, diff_text)
    if input_limit and estimate_tokens(prompt) > input_limit:
        # Cold-start density rung: every api slot the full prompt overflows and
        # whose route has no fresh witness gets ONE bounded probe on a slice of
        # this very prompt; a recorded witness re-sizes the slots once, then
        # the ladder below runs unchanged on the recalibrated limit.
        from ouroboros.tools.review_helpers import DENSITY_PROBE_SAMPLE_CHARS

        # EVERY overflowing slot is probed (a list, not a short-circuit): the
        # quorum cap is the quorum-th largest slot cap, so one witness alone
        # leaves the other cold slots — and the shared prompt — where they were.
        prompt_tokens = estimate_tokens(prompt)
        outcomes = [
            density_probe_before_size_refusal(
                ctx, m, prompt[:DENSITY_PROBE_SAMPLE_CHARS], surface="triad_review",
            )
            for m in api_models if prompt_tokens > slot_limits.get(m, 0)
        ]
        if "measured" in outcomes:
            slot_limits = {m: _slot_input_limit(m) for m in api_models}
            input_limit = _rv._quorum_input_token_limit(api_models, slot_limits)
    if input_limit and estimate_tokens(prompt) > input_limit:
        touched_paths = [line.strip() for line in changed.splitlines() if line.strip()]
        fit_note = (
            "TRIAD FIT NOTE: Full post-change snapshots were omitted because they "
            "duplicate the complete staged diff and would exceed the strictest "
            "configured reviewer's input limit. Every touched path is listed below; "
            "all added/deleted lines remain in the staged diff.\n\n"
            + ("\n".join(f"- {path}" for path in touched_paths) or "(no paths reported)")
        )
        prompt, stable_prefix_len = assemble(fit_note, diff_text)
        if input_limit and estimate_tokens(prompt) > input_limit:
            from ouroboros.tools.review_binary_context import StagedDiffUnavailable
            from ouroboros.tools.review_subject import capture_review_diff
            try:  # the SAME hardened capture as the primary diff, at zero context
                # A managed subject re-renders ITS OWN pinned trees at -U0: the
                # rung stays bound to the exact subject already under review
                # instead of re-serializing a fresh candidate.
                compact_diff = (
                    subject.render_prompt_diff(unified=0) if subject is not None
                    else capture_review_diff(ctx, target_repo, unified=0)
                )
            except StagedDiffUnavailable:
                compact_diff = ""  # keep the hardened full diff; the gate below blocks if it still overflows
            if compact_diff.strip():
                prompt, stable_prefix_len = assemble(fit_note, compact_diff)
    prompt_tokens = estimate_tokens(prompt)
    if not input_limit or prompt_tokens > input_limit:
        # The split imperative is structurally impossible for a managed
        # resolution — its terminal REPLACES the clause (never appends the
        # managed guidance under a false imperative).
        remedy = (
            f"{MANAGED_SPLIT_IMPOSSIBLE} {MANAGED_OVERSIZE_GUIDANCE} "
            "Reviewer models and evidence authority were not degraded."
            if subject is not None
            else "Split or shrink the staged change; "
            "reviewer models and evidence authority were not degraded."
        )
        return prompt, stable_prefix_len, (
            "⚠️ REVIEW_BLOCKED: The irreducible one-pass triad prompt does not "
            f"fit every configured reviewer ({prompt_tokens:,} estimated input "
            f"tokens; limit {input_limit:,}). {remedy}"
        )
    return prompt, stable_prefix_len, ""


def triad_not_dispatched_records(
    row_plan: dict, reason: str, *, only_api: bool = False
) -> list:
    """Typed $0 ``not_dispatched`` actor records for prepared-but-withheld triad
    seats, in the durable ``ReviewActorRecord.to_dict()`` shape.

    Durable review status must show WHICH configured seats were withheld and
    why — a bare degraded-reason string loses the seat identities. ``only_api``
    restricts the records to the api rows (the Q28-A oversize drop); the
    default covers every row (the Q25-A admission block). ``slot`` keeps each
    seat's ORIGINAL 1-based position in the configured plan."""
    from ouroboros.review_execution import delivery_retrieves

    models = list(row_plan.get("models") or [])
    routes = list(row_plan.get("routes") or [])
    slot_ids = list(row_plan.get("slot_ids") or [])
    actors = list(row_plan.get("subagent_ids") or [])
    records = []
    for index, model in enumerate(models):
        if only_api and (
            # A retrieving row (session, or configured-subagent api row) never
            # received the packet; the packet drop is not its withholding and
            # it keeps its live seat.
            index >= len(routes)
            or delivery_retrieves(routes[index], actors[index] if index < len(actors) else "")
        ):
            continue
        records.append({
            "model_id": str(model),
            "status": "not_dispatched",
            "raw_text": str(reason),
            "parsed_items": [],
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "slot": index + 1,
            "slot_id": str(slot_ids[index]) if index < len(slot_ids) else "",
            "prompt_ref": {},
            "response_ref": {},
            "operation_id": "",
            "operation_state": "not_dispatched",
            "late_result_pending": False,
        })
    return records


def drop_api_rows(row_plan: dict) -> dict:
    """Filter every aligned triad row vector down to the agent-session rows.

    Q28-A: an irreducible oversize packet drops the api subset when the session
    rows alone satisfy the quorum. The caller records the drop loudly."""
    from ouroboros.review_execution import delivery_retrieves

    routes = list(row_plan.get("routes") or [])
    actors = list(row_plan.get("subagent_ids") or [])
    # The RETRIEVES class survives the drop: a configured-subagent api row never
    # received the oversized packet, so packet overflow is not its failure.
    keep = [
        i for i, r in enumerate(routes)
        if delivery_retrieves(r, actors[i] if i < len(actors) else "")
    ]
    filtered = dict(row_plan)
    for key in ("models", "routes", "efforts", "session_targets",
                "session_profiles", "slot_ids", "subagent_ids"):
        rows = list(row_plan.get(key) or [])
        filtered[key] = [rows[i] for i in keep if i < len(rows)]
    return filtered


def _scope_pack_starved(context_status: Any, manifest: dict) -> bool:
    """True when the assembled scope pack was refused or degraded for SIZE —
    the only condition under which the density rung may spend a probe: a size
    terminal of the ladder, or an assembled pack whose ladder trace shows a
    rung taken (touched files degraded to diff-only, or the -U0 diff)."""
    if context_status is not None:
        return str(getattr(context_status, "status", "") or "") in _SCOPE_SIZE_TERMINALS
    return any(
        int(step.get("diff_only_files") or 0) > 0 or bool(step.get("zero_context_diff"))
        for step in (dict(manifest or {}).get("ladder_steps") or []) if isinstance(step, dict)
    )


def prepare_scope_review(
    ctx: Any,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
    review_history: Optional[list] = None,
    scope_review_history: Optional[list] = None,
    scope_model: Optional[str] = None,
    slot_id: str = "",
    route: Any = None,
    slot_effort: str = "",
    session_target: str = "",
    session_profile: str = "",
    subagent_id: str = "",
) -> Tuple[Optional[dict], Optional[Any]]:
    """Assemble ONE scope row's packet without dispatching anything.

    Returns ``(prepared, final)`` — exactly one is non-None. ``final`` is a
    complete ScopeReviewResult (deterministic early exit: low-context skip,
    invalid roots, context-build failure, pack signals); ``prepared`` carries
    everything the dispatch half needs, including the context-manifest and
    stable-prefix values (ContextVars do not cross threads, so they are
    captured here and re-seeded at dispatch).
    """
    sr = _scope()
    if sr._scope_review_skipped_in_low_context():
        return None, sr._low_context_skip_result(scope_model or sr._get_scope_model())
    try:
        governance_repo, repo_dir = sr.review_repo_dirs_for(ctx)
    except (TypeError, ValueError) as exc:
        return None, sr.ScopeReviewResult(
            blocked=True,
            status="error", failure_phase="authority", failure_code="invalid_roots",
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: invalid review roots: {exc}.",
        )
    from ouroboros.review_execution import delivery_retrieves

    scope_model_id = scope_model or sr._get_scope_model()
    delegated = str(getattr(route, "value", route) or "") == "agent_session"
    # RETRIEVES class: a session row and a configured-subagent api row deliver
    # by retrieval — neither assembles the packet/atlas below.
    retrieves = delivery_retrieves(route, subagent_id)

    from ouroboros.tools.review_binary_context import StagedDiffUnavailable
    from ouroboros.tools.review_subject import managed_review_subject

    try:
        subject = managed_review_subject(ctx, repo_dir)
    except (RuntimeError, StagedDiffUnavailable, ValueError) as exc:
        return None, sr.ScopeReviewResult(
            blocked=True, status="error", failure_phase="authority", failure_code="subject_unavailable",
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: review subject could not be established: {exc}",
        )
    try:
        if retrieves:
            # Session delivery (5.2): same task/checklist/contract, no assembled
            # pack — the session retrieves with its own tools in the repo root.
            # For a managed resolution the authoritative delta is inlined.
            from ouroboros.tools.scope_review_session import ScopeIntentContext as _Intent
            from ouroboros.tools.scope_review_session import build_scope_session_task

            session_task, session_manifest = build_scope_session_task(
                repo_dir, commit_message,
                _Intent(goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                        review_history=review_history,
                        scope_review_history=scope_review_history),
                drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
                governance_repo_dir=governance_repo,
                managed_subject=subject,
            )
            sr._SCOPE_CONTEXT_MANIFEST.set(session_manifest)
            prompt, context_status = session_task, None
        else:
            session_task = ""

            def _assemble():
                return sr._build_scope_prompt(
                    repo_dir, commit_message,
                    goal=goal, scope=scope,
                    review_rebuttal=review_rebuttal,
                    review_history=review_history,
                    scope_review_history=scope_review_history,
                    context=sr._ScopePromptContext(
                        drive_root=(
                            pathlib.Path(ctx.drive_root)
                            if getattr(ctx, "drive_root", None)
                            else None
                        ),
                        scope_model=scope_model_id,
                        governance_repo_dir=governance_repo,
                        represent_binary=subject is not None,
                        managed_subject=subject,
                    ),
                )

            prompt, context_status = _assemble()
            if _scope_pack_starved(context_status, sr._current_scope_context_manifest()):
                # Cold-start density rung (the deep review's, shared): the
                # sample is the refused required rows first, then the selected
                # ones; a recorded witness rebuilds the pack ONCE under the
                # recalibrated cap. The manifest is reset by every build, so
                # the probe's ladder step is recorded on the LAST build.
                from ouroboros.tools.review_helpers import density_probe_sample

                outcome = density_probe_before_size_refusal(
                    ctx, scope_model_id,
                    density_probe_sample(repo_dir, sr._current_scope_context_manifest()),
                    surface="scope_review",
                )
                if outcome == "measured":
                    prompt, context_status = _assemble()
                if outcome not in ("warm", "no_sample", "unavailable"):
                    sr._record_ladder_steps(
                        list(sr._current_scope_context_manifest().get("ladder_steps") or [])
                        + [{"step": "density_probe", "model": scope_model_id, "outcome": outcome,
                            "rebuilt": outcome == "measured"}]
                    )
    except (RuntimeError, StagedDiffUnavailable, OSError, ValueError) as exc:
        return None, sr.ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Failed to build review context — commit blocked.\n"
                f"Error: {exc}\n"
                "Ensure git is available and the repository is in a valid state."
            ),
            model_id=scope_model_id,
            status="error", failure_phase="context", failure_code="context_unavailable",
            context_manifest=sr._current_scope_context_manifest(),
        )

    # Pack-budget signals belong to an ASSEMBLED pack: a session assembles none, so its
    # context_status is None and this returns None by construction — no route branch.
    signal_result = sr._handle_prompt_signals(
        prompt, context_status, scope_model=scope_model_id,
        input_limit=sr._effective_scope_input_limit(scope_model=scope_model_id),
        managed=subject is not None,
    )
    if signal_result is not None:
        # Keep _handle_prompt_signals as the status SSOT for early exits.
        signal_result.model_id = scope_model_id
        signal_result.context_manifest = sr._current_scope_context_manifest()
        if (
            signal_result.blocked
            and str(getattr(signal_result, "status", "")) in SCOPE_FIT_BLOCK_STATUSES
            and subject is not None
            and MANAGED_OVERSIZE_GUIDANCE not in str(signal_result.block_message or "")
        ):
            # A fit terminal whose message carries no remedy at all (sub_floor)
            # still gets the managed guidance; a message that already carries
            # the managed remedy (ladder_terminal_cause, managed=True) is left
            # alone — never a duplicate, never an append under a split clause.
            signal_result.block_message = (
                f"{signal_result.block_message}\n"
                f"{MANAGED_SPLIT_IMPOSSIBLE} {MANAGED_OVERSIZE_GUIDANCE}"
            )
        return None, signal_result

    return {
        "prompt": prompt,
        "session_task": session_task,
        "repo_dir": repo_dir,
        "scope_model_id": scope_model_id,
        "delegated": delegated,
        "slot_id": slot_id,
        "route": route,
        "slot_effort": slot_effort,
        "session_target": session_target,
        "session_profile": session_profile,
        "subagent_id": subagent_id,
        "context_manifest": sr._current_scope_context_manifest(),
        "stable_prefix_len": int(sr._SCOPE_STABLE_PREFIX_LEN.get() or 0),
    }, None


def commit_gate_paid_seats(triad_prepared, triad_exited, scope_rows) -> list:
    """The PAID seats of one commit-gate wave, SCOPE FIRST (owner decision
    2026-09-05: the only constitutionally blocking seat takes precedence in
    admission and reservation order). A paid seat is an api row — packet OR
    native episode — whose every send is a ``reserve_attempt`` on the ledger;
    an agent-session row rides the owner's subscription (its ledger row is
    written at settlement, never reserved) and is not priced. Each seat carries
    the exact chars of the send its substrate opens with (the packet's message
    pair; a native episode's first send: instructions, work-order and tool
    schemas — its later rounds reserve themselves) and that send's output
    reservation, so the wave is priced the way ``reserve_attempt`` prices it."""
    import json

    from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves
    from ouroboros.review_native_episode import native_first_send_chars
    from ouroboros.reviewer_slot_config import SCOPE_ROLE_HINT
    from ouroboros.tools.review_multi_model import (
        TRIAD_ROLE_HINT, TRIAD_USER_TURN, _review_output_budget, triad_api_messages,
    )
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

    sr = _scope()

    def _chars(messages) -> int:
        return len(json.dumps({"messages": messages}, ensure_ascii=False, default=str))

    def _session(route) -> bool:
        return str(getattr(route, "value", route) or "") == ReviewRouteKind.AGENT_SESSION.value

    seats = []
    for row in scope_rows or []:
        slot, prepared = row["slot"], row.get("prepared") or {}
        route, slot_id = getattr(slot, "route", None), str(slot.slot_id or "")
        if row.get("final") is not None or _session(route):
            continue
        model = str(prepared.get("scope_model_id") or slot.model or "")
        output_tokens, _ = sr._window_scaled_reserves(
            sr._scope_window(model).sizing_window(sr._SCOPE_FAILCLOSED_WINDOW)
        )
        if delivery_retrieves(route, getattr(slot, "subagent_id", "")):
            chars = native_first_send_chars(
                str(prepared.get("repo_dir") or ""), surface="scope_review", role_hint=SCOPE_ROLE_HINT,
                slot_id=slot_id, session_task=str(prepared.get("session_task") or ""),
                output_contract=sr.SCOPE_RETRIEVING_OUTPUT_CONTRACT,
            )
        else:
            chars = _chars(sr.scope_api_messages(
                str(prepared.get("prompt") or ""), int(prepared.get("stable_prefix_len") or 0)))
        seats.append({"surface": "scope_review", "slot_id": slot_id, "model": model,
                      "prompt_chars": chars, "max_completion_tokens": int(output_tokens)})
    if triad_exited or not triad_prepared:
        return seats
    row_plan = triad_prepared.get("row_plan") or {}
    models = list(triad_prepared.get("models") or row_plan.get("models") or [])
    routes = list(triad_prepared.get("routes") or row_plan.get("routes") or [])
    slot_ids, actors = list(row_plan.get("slot_ids") or []), list(row_plan.get("subagent_ids") or [])
    triad_chars = None
    for index, model in enumerate(models):
        route = routes[index] if index < len(routes) else "api_chat"
        slot_id = str(slot_ids[index] if index < len(slot_ids) else f"slot_{index + 1}")
        if _session(route):
            continue
        if delivery_retrieves(route, actors[index] if index < len(actors) else ""):
            chars = native_first_send_chars(
                str(triad_prepared.get("target_repo") or ""), surface="multi_model_review",
                role_hint=TRIAD_ROLE_HINT, slot_id=slot_id,
                session_task=str(triad_prepared.get("session_task") or ""),
                output_contract=REVIEW_JSON_ARRAY_CONTRACT,
            )
        else:
            if triad_chars is None:
                messages, _ = triad_api_messages(
                    str(triad_prepared.get("prompt") or ""),
                    int(triad_prepared.get("stable_prefix_len") or 0), TRIAD_USER_TURN,
                )
                triad_chars = _chars(messages)
            chars = triad_chars
        seats.append({"surface": "multi_model_review", "slot_id": slot_id, "model": str(model or ""),
                      "prompt_chars": chars, "max_completion_tokens": int(_review_output_budget())})
    return seats


def admit_commit_gate_wave(ctx, seats) -> str | None:
    """All-or-nothing money admission of one commit-gate wave (owner decision
    2026-09-05): every paid seat's reservation upper bound must fit TOGETHER,
    against every fence ``reserve_attempt`` enforces (the global TOTAL_BUDGET
    remainder and the root fence), before ANY seat is dispatched. Returns the
    typed refusal text ($0, nothing dispatched) naming the binding axis, or
    None; fail-open on unknowns like the task-level surfaces that already ride
    ``review_wave_budget_gate``."""
    if not seats:
        return None
    from ouroboros.review_substrate import review_usage_category
    from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate

    # Each seat is priced under the usage scope its substrate will SEND under
    # (surface category + slot), so its bound reads the seat's own observed
    # cache split — never the caller's warm transcript split.
    admission = review_wave_budget_gate(
        ctx, surface="commit_gate",
        models=[seat["model"] for seat in seats],
        prompt_chars=[seat["prompt_chars"] for seat in seats],
        max_completion_tokens=[seat["max_completion_tokens"] for seat in seats],
        categories=[review_usage_category(seat["surface"]) for seat in seats],
        slot_ids=[seat["slot_id"] for seat in seats],
        extra={"seats": [f"{seat['surface']}:{seat['slot_id']}" for seat in seats]},
    )
    if admission is None:
        return None
    usd = lambda value: "unknown" if value is None else f"${float(value):.6f}"  # noqa: E731
    bounds = list(admission.get("slot_bounds") or []) + [None] * len(seats)
    wave, remaining = admission.get("estimated_wave_usd"), admission.get("remaining_usd")
    shortfall = None if wave is None or remaining is None else max(0.0, float(wave) - float(remaining))
    limit, accounted = admission.get("limit_usd"), admission.get("accounted_usd")
    root_remaining = None if limit is None or accounted is None else max(0.0, float(limit) - float(accounted))
    if admission.get("binding_axis") == "global":
        # The refusal names the fence that binds and the knob that moves it — never
        # a per-task fence the wave would have fit.
        fence = (
            f"the global budget TOTAL_BUDGET {usd(admission.get('global_limit_usd'))}: "
            f"accounted={usd(admission.get('global_accounted_usd'))} across every task (of which "
            f"{usd(admission.get('global_reserved_usd'))} is reserved by other in-flight attempts), "
            f"remaining={usd(remaining)}, shortfall={usd(shortfall)}; the per-task budget fence "
            f"{usd(limit)} alone would leave {usd(root_remaining)}"
        )
    else:
        fence = (
            f"the per-task budget fence {usd(limit)}: accounted={usd(accounted)} (of which "
            f"{usd(admission.get('reserved_usd'))} is reserved by other in-flight attempts), "
            f"remaining={usd(remaining)}, shortfall={usd(shortfall)}; the global budget "
            f"{usd(admission.get('global_limit_usd'))} alone would leave {usd(admission.get('global_remaining_usd'))}"
        )
    remedy = review_wave_binding_fence(admission)[1]
    return (
        "⚠️ REVIEW_BLOCKED: commit-gate review wave declined before dispatch ($0 spent). "
        f"The wave's reservation upper bound {usd(wave)} ("
        + "; ".join(f"{s['surface']}:{s['slot_id']} {s['model']} {usd(bounds[i])}" for i, s in enumerate(seats))
        + f") does not fit {fence}. No reviewer seat was dispatched (scope and triad alike): wait for "
        f"in-flight attempts to settle or {remedy}, then retry the same commit."
    )
