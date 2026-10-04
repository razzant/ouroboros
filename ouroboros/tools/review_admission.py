"""Pre-dispatch review admission (Q25=A / Q28=A).

Every commit-gate reviewer is PREPARED before any of them is dispatched — the
triad api pack is assembled and fit-checked, every scope row's brief is built —
so a deterministic assembly failure on one side can never spend money on the
other (previously triad and scope dispatched concurrently). A universal reorder
with zero verdict change: the same assembly code runs, the same results come
out — only the ordering moves the spend after the last deterministic gate.

Q28-A oversized outcomes: packet limits gate only the triad api rows. A panel
whose agent-session rows alone satisfy the quorum proceeds without the api rows
(recorded, never silent); a panel that cannot reach quorum without them gets a
typed ZERO-SPEND terminal, and for the managed resolver that refusal carries
the settings guidance below (the resolver's terminal contract already explains
rollback + retry).

``prepare_scope_review`` is the assembly half of ``run_scope_review`` — moved
here whole; the dispatch half stays in ``scope_review``. Internals are reached
through the module object (``_scope().name``) so test monkeypatching of
``scope_review`` attributes keeps working. Scope review delivers by retrieval
(owner decision 2026-09-17), so a scope row is never fit-checked against a
packet limit: its brief carries the diff, the touched manifest and the
required-source manifest, and the reviewer reads the rest itself.

The triad pack keeps ONE cold-start density rung (owner decision 2026-09-05,
answer 1 = A): a pack that would be refused or degraded for SIZE while the
reviewer route has NO fresh exact-model density witness gets one bounded probe
send on the exact model (``capability_evidence.cold_start_density_probe``), the
witness is recorded, the cap recomputed and the pack rebuilt ONCE — never on a
warm store, never retried, never on a commit whose pack fits. A probe the paid
ledger refuses is a typed disclosure in the review events, and the existing
refusal path proceeds unchanged.

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

import json
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

def _scope():
    from ouroboros.tools import scope_review

    return scope_review


DENSITY_PROBE_CALL_TYPE = "review_density_probe"
DENSITY_PROBE_EVENT = "review_density_probe"


def density_probe_before_size_refusal(ctx: Any, model: str, sample: str, *, surface: str,
                                       window_binding: Optional[dict] = None) -> str:
    """The triad pack's cold-start density rung; returns the shared probe's
    typed outcome (``capability_evidence.cold_start_density_probe``) plus
    ``"budget_refused"`` and ``"unavailable"`` (no ctx/drive root to record a
    witness on — a bare fit-check never sends).

    Every attempted probe is a progress line (``ctx.emit_progress_fn``) and one
    ``review_density_probe`` review event; the send itself lands in custody
    through ``chat_observed`` and in the paid ledger like any review send. A
    ``BudgetExceeded`` from the ledger is the typed ``budget_refused``
    outcome — the pack keeps its existing size refusal; nothing crashes and
    nothing is retried. The client comes from the review surface's
    ``LLMClient`` seam and any failure short of the ledger's refusal (a client
    that cannot be constructed included) is the typed ``failed`` outcome, never
    an untyped infra block of the whole gate."""
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
            model_role=(window_binding or {}).get("model_role", ""),
            model_account_override=(window_binding or {}).get("credential_profile_id"),
        )
        reason = ""
    except BudgetExceeded as exc:
        outcome, reason = "budget_refused", str(exc)
        _progress(f"density probe refused by the budget ({reason}); the cold input cap stands.")
    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
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
                     subject=None, slots: Optional[list] = None) -> tuple:
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
    from ouroboros.reviewer_window import reviewer_window_binding

    if slots is not None and len(slots) != len(api_models):
        raise ValueError("Triad capacity rows must align with the frozen packet models")
    bindings = [reviewer_window_binding(slot) for slot in slots] if slots is not None else [{} for _ in api_models]
    keys = [binding["model_role"].removeprefix("reviewer:") for binding in bindings] if slots is not None else api_models
    if slots is not None and (not all(keys) or len(set(keys)) != len(keys)):
        raise ValueError("Triad capacity requires unique stable slot IDs")

    def _slot_input_limit(index: int) -> int:
        slot_model = api_models[index]
        window = _rv.reviewer_context_window(slot_model, **bindings[index])
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
    slot_limits = {key: _slot_input_limit(index) for index, key in enumerate(keys)}
    input_limit = _rv._quorum_input_token_limit(keys, slot_limits)
    prompt, stable_prefix_len = assemble(current_files_section, diff_text)
    if input_limit and estimate_tokens(prompt) > input_limit and getattr(assemble, "compact_optional_evidence", None):
        assemble.compact_optional_evidence()
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
                window_binding=bindings[index],
            )
            for index, m in enumerate(api_models) if prompt_tokens > slot_limits[keys[index]]
        ]
        from ouroboros.review_records import apply_review_model_override
        from ouroboros.model_wait import current_model_wait
        waiter = current_model_wait()
        updated = [apply_review_model_override(slot, waiter.overrides) for slot in slots] if waiter and slots else slots
        route_changed = updated != slots
        if route_changed:
            slots[:] = updated
            api_models[:] = [slot.model for slot in slots]
            bindings = [reviewer_window_binding(slot) for slot in slots]
        if "measured" in outcomes or route_changed:
            slot_limits = {key: _slot_input_limit(index) for index, key in enumerate(keys)}
            input_limit = _rv._quorum_input_token_limit(keys, slot_limits)
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
    from ouroboros.reviewer_slot_config import row_plan_retrieves

    models = list(row_plan.get("models") or [])
    routes = list(row_plan.get("routes") or [])
    slot_ids = list(row_plan.get("slot_ids") or [])
    records = []
    for index, model in enumerate(models):
        if only_api and (
            # A retrieving row (session, native api row or configured-subagent
            # api row) never received the packet; the packet drop is not its
            # withholding and it keeps its live seat.
            index >= len(routes) or row_plan_retrieves(row_plan, index)
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
    from ouroboros.reviewer_slot_config import row_plan_retrieves

    routes = list(row_plan.get("routes") or [])
    # The RETRIEVES class survives the drop: a native or configured-subagent api
    # row never received the oversized packet, so packet overflow is not its failure.
    keep = [i for i in range(len(routes)) if row_plan_retrieves(row_plan, i)]
    filtered = dict(row_plan)
    for key in ("models", "routes", "efforts", "session_targets",
                "session_profiles", "slot_ids", "subagent_ids", "retrieves", "use_local"):
        rows = list(row_plan.get(key) or [])
        filtered[key] = [rows[i] for i in keep if i < len(rows)]
    return filtered


# One durable marker per install for the scope-delivery migration notice below.
SCOPE_DELIVERY_MIGRATION_FILENAME = "scope_delivery_migration.json"
SCOPE_DELIVERY_MIGRATION_EVENT = "review_scope_delivery_migrated"


def _scope_delivery_migration_path() -> pathlib.Path:
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR) / "state" / SCOPE_DELIVERY_MIGRATION_FILENAME


def disclose_scope_delivery_migration(ctx: Any, slot_id: str, model: str) -> None:
    """Announce ONCE per install that a stored bare api scope row now retrieves.

    A scope row saved before the retrieving delivery carries no actor binding,
    and its delivery changes under it: the row runs a bounded native inspection
    episode on the same model instead of receiving an assembled packet. That is
    a visible change in what the row spends and how long it takes, so it is
    stated in the durable event stream rather than discovered from a bill. The
    marker lives beside the other reviewer-slot projection state in the
    canonical data plane; a marker the host cannot write discloses again next
    time rather than failing the review.
    """
    from ouroboros.tools.review_helpers import emit_review_event
    from ouroboros.utils import utc_now_iso, write_text_atomic

    path = _scope_delivery_migration_path()
    try:
        if path.exists():
            return
    except OSError:
        return
    emit_review_event(ctx, {
        "type": SCOPE_DELIVERY_MIGRATION_EVENT,
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "slot_id": str(slot_id or ""), "model": str(model or ""),
        "delivery": "native_retrieval",
        "reason": (
            "this stored scope row has no actor binding; scope review delivers by "
            "retrieval, so the row now runs a bounded native inspection episode on "
            "its own route instead of receiving an assembled packet"
        ),
    })
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(
            {"disclosed_at": utc_now_iso(), "slot_id": str(slot_id or ""), "model": str(model or "")},
            ensure_ascii=False))
    except OSError:
        log.debug("scope delivery migration marker not written", exc_info=True)


def _scope_source_root(ctx: Any, task_evidence: dict) -> str:
    """The data root a paged brief source is stored under, or ``""``.

    It is the root the row's OWN reader resolves: the canonical task data root,
    which is what the commit-review evidence view already records and what a
    native episode reads as ``policy["native_data_root"]``. A context with no
    resolvable data plane has nowhere to page to, and the brief inlines instead.
    """
    recorded = str((task_evidence or {}).get("data_root") or "").strip()
    if recorded:
        return recorded
    try:
        from ouroboros.tool_access import canonical_data_root

        return str(canonical_data_root(ctx))
    except (AttributeError, OSError, TypeError, ValueError):
        return ""


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
    """Assemble ONE scope row's brief without dispatching anything.

    Returns ``(prepared, final)`` — exactly one is non-None. ``final`` is a
    complete ScopeReviewResult (deterministic early exit: invalid roots,
    an unavailable review subject, a context-build failure); ``prepared``
    carries everything the dispatch half needs, including the required-source
    manifest and the context-manifest value (ContextVars do not cross threads,
    so it is captured here and re-seeded at dispatch).

    Every scope row retrieves (owner decision 2026-09-17): the brief is the same
    for both transports, scope review applies in every context mode, and no
    brief is ever assembled as a packet.
    """
    sr = _scope()
    window_binding = {"model_role": f"reviewer:{slot_id}", "credential_profile_id": session_profile} if slot_id else {}
    try:
        governance_repo, repo_dir = sr.review_repo_dirs_for(ctx)
    except (TypeError, ValueError) as exc:
        return None, sr.ScopeReviewResult(
            blocked=True,
            status="error", failure_phase="authority", failure_code="invalid_roots",
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: invalid review roots: {exc}.",
        )
    scope_model_id = scope_model or sr._get_scope_model()
    from ouroboros.model_wait import current_model_wait
    waiter = current_model_wait()
    override = waiter.overrides.get(f"reviewer:{slot_id}", {}) if waiter else {}
    if override and str(getattr(route, "value", route) or "") != "agent_session":
        scope_model_id, session_profile = override["model"], override["model_account_override"]
        window_binding.update(credential_profile_id=session_profile, use_local=override["use_local"])
    delegated = str(getattr(route, "value", route) or "") == "agent_session"
    from ouroboros.review_evidence import commit_review_evidence_section, materialize_commit_review_session_view

    task_evidence = dict(getattr(ctx, "_commit_review_evidence", None) or {})
    if delegated:
        task_evidence = materialize_commit_review_session_view(task_evidence, repo_dir)
        ctx._commit_review_evidence = task_evidence
    task_evidence_section = commit_review_evidence_section(
        task_evidence, delivery="session" if delegated else "native")

    from ouroboros.tools.review_binary_context import StagedDiffUnavailable
    from ouroboros.tools.review_subject import managed_review_subject

    try:
        subject = managed_review_subject(ctx, repo_dir)
    except (RuntimeError, StagedDiffUnavailable, ValueError) as exc:
        return None, sr.ScopeReviewResult(
            blocked=True, status="error", failure_phase="authority", failure_code="subject_unavailable",
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: review subject could not be established: {exc}",
        )
    required_sources: list = []
    required_ref: dict = {}
    try:
        # Retrieving delivery (5.2): same task/checklist/contract, no assembled
        # pack — the reviewer reads the subject with its own tools in the repo
        # root, and the required-source manifest states what it is owed IN FULL.
        # For a managed resolution the delta is inlined.
        from ouroboros.tools.scope_required_sources import (
            required_sources_ref,
            scope_required_sources, staged_tree_identity, staged_touched_paths,
            touched_manifest,
        )
        from ouroboros.owner_words import owner_words_text
        from ouroboros.tools.scope_review_session import ScopeIntentContext as _Intent
        from ouroboros.tools.scope_review_session import (
            ScopeBriefInputs, build_scope_session_task,
        )

        touched = staged_touched_paths(repo_dir, subject)
        tree_sha = staged_tree_identity(repo_dir, subject)
        manifest_rows = scope_required_sources(
            repo_dir, touched, staged_tree_sha=tree_sha, subject=subject)
        required_ref = required_sources_ref(manifest_rows, staged_tree_sha=tree_sha)
        session_task, session_manifest = build_scope_session_task(repo_dir, ScopeBriefInputs(
            commit_message=commit_message,
            intent=_Intent(goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                           review_history=review_history,
                           scope_review_history=scope_review_history, owner_words=owner_words_text(ctx)),
            drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
            governance_repo_dir=governance_repo,
            managed_subject=subject,
            task_evidence_section=task_evidence_section,
            required_sources=manifest_rows,
            required_sources_ref=required_ref,
            touched_manifest=touched_manifest(repo_dir, touched),
            touched_paths=tuple(path for _status, path in touched),
            delegated=delegated,
            scope_model=scope_model_id,
            slot_id=slot_id,
            session_profile=session_profile,
            use_local=override.get("use_local"),
            task_id=str(getattr(ctx, "task_id", "") or "") or "scope_review",
            source_root=_scope_source_root(ctx, task_evidence),
        ))
        # The brief resolves preimages, inline documents and a paged subject
        # into one final manifest. Missing rows remain diagnostic gaps.
        required_sources = session_manifest["native_required_sources"]
        required_ref = session_manifest["native_required_sources_ref"]
        sr._SCOPE_CONTEXT_MANIFEST.set(session_manifest)
        if not delegated and not str(subagent_id or "").strip():
            disclose_scope_delivery_migration(ctx, slot_id, scope_model_id)
    except (RuntimeError, StagedDiffUnavailable, OSError, ValueError) as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        # Row-local preparation evidence, before any reviewer is dispatched.
        try:
            sr.append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": sr.utc_now_iso(), "type": "scope_review_preparation_failed",
                "task_id": getattr(ctx, "task_id", "") or "", "slot_id": slot_id,
                "model": scope_model_id, "status": "error",
                "failure_phase": "context", "failure_code": "context_unavailable",
                "reason": str(exc),
            })
        except Exception:
            pass
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

    return {
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
        "window_binding": window_binding, "task_evidence": task_evidence,
        "use_local": override.get("use_local"),
        # Every exact source resolves under the root that stored it, whether
        # the brief paged its diff or carries a deleted-file preimage.
        "native_data_root": str(session_manifest.get("native_data_root") or ""),
        "required_sources": required_sources,
        "required_sources_ref": required_ref,
        "context_manifest": sr._current_scope_context_manifest(),
    }, None


def commit_gate_paid_seats(triad_prepared, triad_exited, scope_rows) -> list:
    """The PAID seats of one commit-gate wave, SCOPE FIRST (owner decision
    2026-09-05: the only constitutionally blocking seat takes precedence in
    admission and reservation order). A paid seat is an api row — the triad's
    packet OR a native episode — whose every send is a ``reserve_attempt`` on
    the ledger; an agent-session row rides the owner's subscription (its ledger
    row is written at settlement, never reserved) and is not priced. Each seat
    carries the exact chars of the send its substrate opens with (the triad
    packet's message pair; a native episode's first send: instructions,
    work-order and tool schemas — its later rounds reserve themselves) and that
    send's output reservation, so the wave is priced the way
    ``reserve_attempt`` prices it. Every scope seat is a retrieving one."""

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_native_episode import native_first_send_chars
    from ouroboros.reviewer_slot_config import row_plan_retrieves
    from ouroboros.reviewer_slot_config import SCOPE_ROLE_HINT
    from ouroboros.tools.review_multi_model import (
        TRIAD_ROLE_HINT, TRIAD_USER_TURN, _review_output_budget, triad_api_messages,
    )
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT
    from ouroboros.review_evidence import commit_review_evidence_section

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
        binding = {"model_role": f"reviewer:{slot_id}",
                   "credential_profile_id": prepared.get("session_profile", str(getattr(slot, "session_profile", "") or "")),
                   "use_local": prepared.get("use_local", getattr(slot, "use_local", None)),
                   **(prepared.get("window_binding") or {})}
        output_tokens, _ = sr._window_scaled_reserves(
            sr._scope_window(model, **binding).sizing_window(sr._SCOPE_SIZING_FALLBACK)
        )
        # Every scope row is a native inspection episode, so its seat is priced
        # by its first send — instructions, brief and tool schemas — never by a
        # message pair it does not assemble.
        chars = native_first_send_chars(
            str(prepared.get("repo_dir") or ""), surface="scope_review", role_hint=SCOPE_ROLE_HINT,
            slot_id=slot_id, session_task=str(prepared.get("session_task") or ""),
            output_contract=sr.SCOPE_RETRIEVING_OUTPUT_CONTRACT,
        )
        seats.append({"surface": "scope_review", "slot_id": slot_id, "model": model,
                      "prompt_chars": chars, "max_completion_tokens": int(output_tokens)})
    if triad_exited or not triad_prepared:
        return seats
    row_plan = triad_prepared.get("row_plan") or {}
    models = list(triad_prepared.get("models") or row_plan.get("models") or [])
    routes = list(triad_prepared.get("routes") or row_plan.get("routes") or [])
    slot_ids = list(row_plan.get("slot_ids") or [])
    triad_chars = None
    for index, model in enumerate(models):
        route = routes[index] if index < len(routes) else "api_chat"
        slot_id = str(slot_ids[index] if index < len(slot_ids) else f"slot_{index + 1}")
        if _session(route):
            continue
        if row_plan_retrieves({**row_plan, "routes": routes}, index):
            chars = native_first_send_chars(
                str(triad_prepared.get("target_repo") or ""), surface="multi_model_review",
                role_hint=TRIAD_ROLE_HINT, slot_id=slot_id,
                session_task=str(triad_prepared.get("session_task") or "") + ("\n\n" + commit_review_evidence_section(triad_prepared["task_evidence"], delivery="native") if triad_prepared.get("task_evidence") else ""),
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
