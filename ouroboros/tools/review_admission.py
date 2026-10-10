"""Pre-dispatch review admission (Q25=A / Q28=A) and the one brief's builders.

Every commit-gate seat is PREPARED before any of them is dispatched — the packet
seats' prompt is assembled and fit-checked, every retrieving seat's two-part
brief is built — so a deterministic assembly failure can never spend money. A
universal reorder with zero verdict change: the same assembly code runs, the
same results come out — only the ordering moves the spend after the last
deterministic gate.

Q28-A oversized outcomes: packet limits gate only the packet (api) rows. A panel
whose retrieving rows alone satisfy the quorum proceeds without the packet rows
(recorded, never silent); a panel that cannot reach quorum without them gets a
typed ZERO-SPEND terminal, and for the managed resolver that refusal carries
the settings guidance below (the resolver's terminal contract already explains
rollback + retry).

One brief, two parts (PR-3 B): there is no scope role. Every seat answers the
change (Part 1); a seat that retrieves also answers the eight coupling
questions (Part 2) in the same brief — ``build_two_part_brief`` is the pure
builder over a frozen subject, ``assemble_packet_prompt`` the packet seat's
Part-1 prompt, ``retrieving_brief_for_seat`` the retrieving seat's brief
(``review_brief_coupling``). A retrieving seat is never fit-checked against a
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

Money admission (owner decision 2026-09-05, answer 2 = A, on the known-spend
rule of #1487) is the last pre-dispatch gate: ``commit_gate_paid_seats``
prices every PAID seat of the wave — packet rows by their exact message pair,
native episodes by their exact first send — each under the usage scope its
substrate sends under, and ``admit_commit_gate_wave`` admits them as ONE wave
through the shared ``review_wave_budget_gate`` while known spend is below
every fence. The summed bounds are disclosed; a wave declined there is a typed
$0 refusal naming the binding fence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import subprocess
from typing import Any, Optional, Sequence, Tuple

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
    "Settings → Agents, Reviewer rows (packet limits do not apply to them), "
    "or configure larger-window models."
)

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
                     subject=None, slots: Optional[list] = None, compact_diff=None) -> tuple:
    """The api pack's guaranteed-fit ladder (P3 one-pass): drop only evidence
    duplicated by the complete staged diff — full snapshots first, then unchanged
    diff context. Each api slot's limit uses its REAL window from Capability
    Evidence (a hardcoded 1M treated a 200K reviewer as 1M-capable and lost its
    whole review to a deterministic prompt-too-long 400), with sub-1M windows
    scaling their reserves so a small-window slot gets a fit-sized pack, not a
    zero limit; the shared prompt is sized to the review QUORUM — the same SSOT
    plan review uses — so one small slot degrades its OWN seat rather than
    blocking the gate for the whole panel. Session rows are not constrained by
    this pack at all (5.2/5.7): they retrieve with their own tools.
    ``compact_diff`` is a frozen subject's own -U0 rendering (a zero-argument
    callable); without it the rung re-captures the staged diff of
    ``target_repo``. Returns ``(prompt, stable_prefix_len, block_message_or_empty)``."""
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
            output_reserve=_rv._review_output_budget(), model_id=slot_model, binding=bindings[index],
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
                # A managed or frozen subject re-renders ITS OWN pinned trees at
                # -U0: the rung stays bound to the exact subject already under
                # review instead of re-serializing a fresh candidate.
                compact = (
                    subject.render_prompt_diff(unified=0) if subject is not None
                    else compact_diff() if compact_diff is not None
                    else capture_review_diff(ctx, target_repo, unified=0)
                )
            except StagedDiffUnavailable:
                compact = ""  # keep the hardened full diff; the gate below blocks if it still overflows
            if compact.strip():
                prompt, stable_prefix_len = assemble(fit_note, compact)
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
    vectors = ["models", "routes", "efforts", "session_targets", "session_profiles", "slot_ids",
               "subagent_ids", "retrieves", "use_local"]
    # The per-seat brief vectors and the added-seat bit ride along when the plan
    # already carries them — by the SAME indices, so a critic the author added
    # beside the pool stays an added (uncounted) seat after the drop.
    vectors += [key for key in ("parts", "session_tasks", "session_policies", "brief_shas", "additional")
                if key in row_plan]
    for key in vectors:
        rows = list(row_plan.get(key) or [])
        filtered[key] = [rows[i] for i in keep if i < len(rows)]
    return filtered


def counted_retrieving_seats(row_plan: dict, api_indices: Sequence[int]) -> Tuple[int, int]:
    """``(retrieving counted seats, counted quorum)`` — the Q28-A yield arithmetic over
    the seats that VOTE. A seat the author added beside the pool (``additional``) is
    heard, never counted: it can neither supply the quorum the dropped api rows leave
    behind nor widen the quorum the counted seats owe (``adaptive_quorum``)."""
    from ouroboros.review_model_routes import adaptive_quorum

    extra = list(row_plan.get("additional") or [])
    counted = [i for i in range(len(row_plan.get("models") or [])) if not (i < len(extra) and extra[i])]
    packet = set(api_indices)
    return sum(1 for i in counted if i not in packet), adaptive_quorum(len(counted))


# ---------------------------------------------------------------------------
# One brief, two parts (PR-3 B): the packet seat's prompt and the retrieving
# seat's brief are assembled here, one builder each, and ``build_two_part_brief``
# is the pure entry over a frozen subject that an operator can call outside the
# gate (step R: build the brief of an old subject and hand it to sessions).
# ---------------------------------------------------------------------------

# The packet prompt is assembled STABLE-FIRST for provider prompt caching: fixed
# instructions plus the tier-1 governance rules (the layered Change Review
# Checklist, the standing disclosures, and BIBLE.md through the constitutional
# head) form a byte-stable prefix reused across review rounds AND across commits
# (marked with a cache breakpoint at dispatch). The change-class governance
# selection (`tools/governance_context.py` tiers 2 and 3) and the navigation maps
# open the dynamic tail, ahead of goal/scope/files/diff/history.
PACKET_TEMPLATE_STABLE = """\
{preamble}

## Part 1 — The change

Read the staged diff and the supplied post-change file context (both appear
AFTER the governance documents below). On very large changes, the fit note may
replace duplicated full-file snapshots with a path manifest; in that case the
complete added/deleted lines remain in the staged diff. Review every checklist
item, report every distinct current problem, and make every FAIL actionable
with file/symbol evidence and a concrete remedy — deleting a mechanism or
disclosing a residual are remedies too.

{critical_calibration}

## Answer format

{json_contract}

If an open obligation record below already names an `obligation_id` for this root cause,
reuse that exact `obligation_id`. Do NOT invent a new id when the same root cause persists.

## Anti pattern-lock guard

Run the shared semantic-breadth guard before returning:
{anti_pattern_lock_guard}

{checklist_section}

- Output ONLY a valid JSON array.  No markdown fences, no text outside the JSON.

The governance documents this change activates follow below, then its evidence.
Navigation maps identify sources not delivered to this tool-free packet row;
state uncertainty where the supplied evidence cannot establish a rule.
"""

PACKET_TEMPLATE_DYNAMIC = """\
{goal_section}

{scope_section}

## Current touched files (full content)

{current_files_section}

## Staged diff

{diff_text}

## Changed files

{changed_files}

{rebuttal_section}{review_history_section}
{task_evidence_section}
"""


def assemble_packet_prompt(*, layer: str, checklist_section: str, governance: Any, goal_section: str,
                           scope_section: str, files_section: str, diff_text: str, changed_files: str,
                           rebuttal_section: str = "", review_history_section: str = "",
                           task_evidence_section: str = "") -> Tuple[str, int]:
    """The packet seat's Part-1 prompt: ``(prompt, stable_prefix_len)``. The stable
    governance prefix is byte-identical across rounds and becomes the cache-marked
    block; the change-class governance block opens the dynamic half."""
    from ouroboros.tools.review_helpers import CRITICAL_FINDING_CALIBRATION, anti_pattern_lock_guard, review_preamble
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

    stable_inline = str(getattr(governance, "stable_inline", "") or "")
    tail = "\n\n".join(part for part in (str(getattr(governance, "selected_inline", "") or ""),
                                          str(getattr(governance, "navigation", "") or "")) if part.strip())
    stable = PACKET_TEMPLATE_STABLE.format(
        preamble=review_preamble(layer), critical_calibration=CRITICAL_FINDING_CALIBRATION,
        json_contract=REVIEW_JSON_ARRAY_CONTRACT, anti_pattern_lock_guard=anti_pattern_lock_guard(layer),
        checklist_section=checklist_section,
    ) + (f"\n{stable_inline}\n" if stable_inline.strip() else "")
    dynamic = (f"{tail}\n\n" if tail else "") + PACKET_TEMPLATE_DYNAMIC.format(
        goal_section=goal_section, scope_section=scope_section, current_files_section=files_section,
        rebuttal_section=rebuttal_section, review_history_section=review_history_section,
        diff_text=diff_text, changed_files=changed_files, task_evidence_section=task_evidence_section,
    )
    return stable + "\n" + dynamic, len(stable) + 1


def source_root_for(ctx: Any, task_evidence: dict) -> str:
    """The data root a paged brief source is stored under, or ``""``.

    It is the root the seat's OWN reader resolves: the canonical task data root,
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


def seat_vectors(row_plan: dict) -> dict:
    """The plan with its ``parts`` vector complete: a row without one is asked by
    ``review_ledger.seat_parts`` (retrieving → both parts, packet → ``change``)."""
    from ouroboros.review_ledger import seat_parts
    from ouroboros.reviewer_slot_config import row_plan_retrieves

    plan = dict(row_plan)
    parts = list(plan.get("parts") or [])
    for index in range(len(plan.get("models") or [])):
        if index >= len(parts) or not parts[index]:
            while len(parts) <= index:
                parts.append(())
            parts[index] = seat_parts({"retrieves": row_plan_retrieves(plan, index)})
    plan["parts"] = [tuple(p) for p in parts]
    return plan


def retrieving_brief_for_seat(*, review_root: Any, governance_root: Any, path_subject: Any, managed_subject: Any,
                              diff_text: Optional[str], layer: str, checklist_section: str, commit_message: str,
                              intent: Any, parts: Sequence[str], delegated: bool, model: str, slot_id: str,
                              session_profile: str = "", use_local: Optional[bool] = None,
                              task_evidence_section: str = "", drive_root: Any = None, task_id: str = "",
                              source_root: str = "") -> Tuple[str, dict]:
    """ONE retrieving seat's two-part brief and its manifest, from the subject's
    frozen trees: the touched paths, the candidate tree identity and the
    change-relative required-source manifest are computed here, the text by
    ``review_brief_coupling.build_retrieving_brief``."""
    from ouroboros.tools.review_brief_coupling import BriefInputs, build_retrieving_brief
    from ouroboros.tools.scope_required_sources import (
        required_sources_ref, scope_required_sources, staged_tree_identity, staged_touched_paths, touched_manifest,
    )

    repo_dir = pathlib.Path(review_root)
    touched = staged_touched_paths(repo_dir, path_subject)
    tree_sha = staged_tree_identity(repo_dir, path_subject)
    rows = scope_required_sources(repo_dir, touched, staged_tree_sha=tree_sha, subject=path_subject, layer=layer)
    return build_retrieving_brief(repo_dir, BriefInputs(
        commit_message=commit_message, intent=intent,
        drive_root=pathlib.Path(drive_root) if drive_root else None,
        governance_repo_dir=pathlib.Path(governance_root) if governance_root else None,
        managed_subject=managed_subject, diff_text=diff_text, task_evidence_section=task_evidence_section,
        required_sources=rows, required_sources_ref=required_sources_ref(rows, staged_tree_sha=tree_sha),
        touched_manifest=touched_manifest(repo_dir, touched), touched_paths=tuple(path for _s, path in touched),
        layer=layer, checklist_section=checklist_section, parts=tuple(parts), delegated=delegated, model=model,
        slot_id=slot_id, session_profile=session_profile, use_local=use_local, task_id=task_id, source_root=source_root,
    ))


BRIEF_PREPARATION_FAILED_EVENT = "review_brief_preparation_failed"


def emit_brief_preparation_failure(ctx: Any, *, slot_id: str, model: str, parts: Sequence[str], exc: BaseException) -> None:
    """Seat-local preparation evidence, not a verdict: one durable row in the
    task's events log, whose normal sink forwards it live. Best-effort — a
    logging failure never changes the typed assembly block the gate returns."""
    from ouroboros.utils import append_jsonl, utc_now_iso

    logs = getattr(ctx, "drive_logs", None)
    if not callable(logs):
        return
    try:
        append_jsonl(logs() / "events.jsonl", {
            "ts": utc_now_iso(), "type": BRIEF_PREPARATION_FAILED_EVENT,
            "task_id": str(getattr(ctx, "task_id", "") or ""), "slot_id": str(slot_id), "model": str(model),
            "parts": list(parts), "status": "error", "failure_phase": "context",
            "failure_code": "context_unavailable", "reason": str(exc)})
    except Exception:
        pass


def prepare_retrieving_seats(ctx: Any, row_plan: dict, models: list, row_routes: list, *,
                             target_repo, governance_root, subject, frozen, diff_text: str, layer: str,
                             checklist_section: str, commit_message: str, goal: str, scope: str,
                             review_rebuttal: str, owner_words: str, task_evidence: dict) -> tuple:
    """Every retrieving seat receives ITS OWN two-part brief (Part 1 the change,
    Part 2 the coupling questions; a coupling-only seat answers Part 2 alone),
    sized to the seat's own first-send bound. Returns ``(row_plan,
    retrieving_manifests, brief_texts, failure)``: the row vectors carry the
    brief and its answer policy per seat, ``brief_texts`` keeps every distinct
    brief by sha for the durable wave record, and ``failure`` is ``None`` or
    ``(slot_id, exc)`` — the seat whose brief could not be built, already
    recorded as seat-local preparation evidence; the gate turns it into the
    whole wave's typed assembly block (one wave, so one seat's missing context
    dispatches nothing). Model-control exceptions propagate before logging."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import row_plan_retrieves
    from ouroboros.tools.review_brief_coupling import BriefIntent
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT, REVIEW_TWO_PART_OBJECT_CONTRACT

    row_plan = dict(row_plan)
    session_tasks, session_policies, brief_shas = [""] * len(models), [None] * len(models), [""] * len(models)
    brief_texts: dict = {}
    retrieving_manifests: list = []
    path_subject = subject if subject is not None else (frozen if frozen is not None and not frozen.is_system_index else None)
    intent = BriefIntent(goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                         review_history=list(ctx._review_history or []),
                         coupling_history=list(getattr(ctx, "_coupling_review_history_rounds", None) or []),
                         owner_words=owner_words)
    for i in range(len(models)):
        if not row_plan_retrieves(row_plan, i):
            continue
        parts = tuple(row_plan["parts"][i])
        delegated = row_routes[i] is ReviewRouteKind.AGENT_SESSION
        try:
            text, manifest = retrieving_brief_for_seat(
                review_root=target_repo, governance_root=governance_root, path_subject=path_subject,
                managed_subject=subject, diff_text=diff_text, layer=layer, checklist_section=checklist_section,
                commit_message=commit_message, intent=intent, parts=parts, delegated=delegated,
                model=models[i], slot_id=str(row_plan["slot_ids"][i]),
                session_profile=str(row_plan["session_profiles"][i] or ""), use_local=row_plan["use_local"][i],
                drive_root=getattr(ctx, "drive_root", None),
                task_id=str(getattr(ctx, "task_id", "") or "") or "commit_review",
                source_root=source_root_for(ctx, task_evidence))
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
            from ouroboros.llm_claudexor import propagate_model_error
            propagate_model_error(exc)
            log.error("Review brief assembly failed for %s: %s", row_plan["slot_ids"][i], exc)
            emit_brief_preparation_failure(ctx, slot_id=str(row_plan["slot_ids"][i]), model=str(models[i]),
                                           parts=parts, exc=exc)
            return row_plan, retrieving_manifests, brief_texts, (str(row_plan["slot_ids"][i]), exc)
        session_tasks[i], brief_shas[i] = text, manifest["sha"]["brief"]
        brief_texts[manifest["sha"]["brief"]] = text
        policy = {"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT if "coupling" in parts else REVIEW_JSON_ARRAY_CONTRACT}
        if manifest.get("native_required_sources") is not None:
            policy["native_required_sources"] = manifest["native_required_sources"]
            policy["native_required_sources_ref"] = manifest.get("native_required_sources_ref") or {}
        if manifest.get("native_data_root"):
            policy["native_data_root"] = manifest["native_data_root"]
        session_policies[i] = policy
        retrieving_manifests.append({"slot_id": str(row_plan["slot_ids"][i]), **{
            k: manifest.get(k) for k in ("delivery", "parts", "diff_delivery", "diff_source", "first_send_chars",
                                         "first_send_bound", "brief_chars", "governance_manifest", "repository_index",
                                         "sha")}})
    row_plan.update(session_tasks=session_tasks, session_policies=session_policies, brief_shas=brief_shas)
    return row_plan, retrieving_manifests, brief_texts, None


def build_two_part_brief(frozen_subject: Any, seat: Any, *, layer: Optional[str] = None, goal: str = "",
                         scope: str = "", author_questions: Sequence[str] = (), commit_message: str = "",
                         review_rebuttal: str = "", review_history: Sequence[dict] = (),
                         coupling_history: Sequence[dict] = (), owner_words: str = "",
                         task_evidence_section: str = "", drive_root: Any = None, task_id: str = "",
                         source_root: str = "", coupling_only: bool = False) -> dict:
    """The brief ONE seat would receive for a frozen subject — pure over the
    subject, callable outside the gate (step R).

    ``frozen_subject`` is a ``review_subject.FrozenSubject`` (its ``review_root``
    is read, its ``spec.governance_root`` governs, its ``diff_text`` is the
    change); ``seat`` is a ``ReviewSlot``/``ConfiguredReviewerSlot``-like object or a
    row dict (``slot_id``, ``model``, ``route``, ``retrieves``, ``subagent_id``,
    ``session_profile``, ``use_local``). ``author_questions`` ride the goal
    section as the author's own questions to the panel. Returns ``{"system",
    "user", "parts", "sha", "delivery", "manifest", "stable_prefix_len"}``:
    ``system`` is the whole brief text a seat is sent (the packet prompt behind
    the constitutional head for a packet seat, the two-part brief for a
    retrieving seat), ``user`` the one user turn every seat opens with, ``sha``
    = ``{"brief", "change_prompt_sha", "coupling_brief_sha"}``.
    """
    from ouroboros.review_ledger import seat_parts
    from ouroboros.tools import review as _rv
    from ouroboros.tools.review_brief_coupling import BriefIntent
    from ouroboros.tools.review_helpers import (
        build_goal_section, build_rebuttal_section, build_scope_section, goal_with_author_questions,
        review_history_with_obligations,
    )
    from ouroboros.tools.review_multi_model import TRIAD_USER_TURN, triad_api_messages
    from ouroboros.review_records import ReviewSlot

    def _field(name: str, default: Any = "") -> Any:
        if isinstance(seat, dict):
            return seat.get(name, default)
        return getattr(seat, name, default)

    layer = str(layer or getattr(frozen_subject.spec, "layer", "") or "body")
    review_root = pathlib.Path(frozen_subject.review_root)
    governance_root = pathlib.Path(frozen_subject.spec.governance_root or review_root)
    parts = tuple(seat_parts(seat, coupling_only=coupling_only))
    checklist_section = _rv._load_checklist_section(layer)
    # The author's questions and the prior rounds are rendered by the owners the
    # gate and review_change use (D5-004): the text here IS the text a seat is sent.
    goal_text = goal_with_author_questions(goal, author_questions)
    goal_section = build_goal_section(goal_text, scope, commit_message, owner_words)
    scope_section = build_scope_section(scope)
    rebuttal_section = build_rebuttal_section(review_rebuttal)
    history_section = review_history_with_obligations(review_history, drive_root=drive_root,
                                                      repo_root=frozen_subject.spec.root)
    model, slot_id = str(_field("model") or ""), str(_field("slot_id") or "")
    route = _field("route", None)
    delegated = str(getattr(route, "value", route) or "") == "agent_session"
    # The gate's own subject (the system repository's index at HEAD) is the one
    # subject read on its live root; every other frozen subject hands its own
    # trees and diff to the manifest (the same rule as ``prepare_retrieving_seats``).
    path_subject = frozen_subject.managed if frozen_subject.managed is not None else (
        frozen_subject if not frozen_subject.is_system_index else None)
    intent = BriefIntent(goal=goal_text, scope=scope, review_rebuttal=review_rebuttal, review_history=list(review_history or []),
                         coupling_history=list(coupling_history or []), owner_words=owner_words)
    if "coupling" in parts:
        text, manifest = retrieving_brief_for_seat(
            review_root=review_root, governance_root=governance_root, path_subject=path_subject,
            managed_subject=frozen_subject.managed, diff_text=frozen_subject.diff_text, layer=layer,
            checklist_section=checklist_section, commit_message=commit_message, intent=intent, parts=parts,
            delegated=delegated, model=model, slot_id=slot_id, session_profile=str(_field("session_profile") or ""),
            use_local=_field("use_local", None), task_evidence_section=task_evidence_section, drive_root=drive_root,
            task_id=task_id, source_root=source_root)
        return {"system": text, "user": TRIAD_USER_TURN, "parts": list(parts), "sha": manifest["sha"],
                "delivery": "retrieving", "manifest": manifest, "stable_prefix_len": 0}
    # A packet seat: the assembled change evidence, fit to the seat's window.
    from ouroboros.tools.review_file_pack import build_touched_file_pack, triad_pack_exclusions

    touched_paths = [path for _status, path in frozen_subject.name_status] if frozen_subject.name_status else [
        p.strip() for p in str(_rv.run_cmd(["git", "diff", "--cached", "--name-only"], cwd=review_root) or "").splitlines() if p.strip()]
    slot = ReviewSlot(slot_id=slot_id, model=model, session_profile=str(_field("session_profile") or ""),
                      use_local=_field("use_local", None))
    governance = _rv._triad_governance_context(
        None, touched_paths, checklist_section, [model], [slot], governance_root=governance_root, layer=layer,
        subject_root=review_root if layer != "body" else None)
    managed = frozen_subject.managed
    exclude_paths, exclusion_note = (set(), "") if managed is not None else triad_pack_exclusions(
        review_root, touched_paths, prefix_texts=dict(governance.inline_whole_documents))
    files_section, omitted = build_touched_file_pack(
        review_root, touched_paths, represent_binary=managed is not None,
        m0_tree=getattr(managed, "m0_tree", "") or "", staged_tree=getattr(managed, "staged_tree", "") or "",
        exclude_paths=exclude_paths)
    if omitted:
        files_section += f"\n\n⚠️ OMISSION NOTE: {len(omitted)} file(s) omitted from direct context: {', '.join(omitted)}"
    if exclusion_note:
        files_section += f"\n\n{exclusion_note}"
    changed = "\n".join(touched_paths)

    def _assemble(files: str, diff: str) -> Tuple[str, int]:
        return assemble_packet_prompt(
            layer=layer, checklist_section=checklist_section, governance=governance, goal_section=goal_section,
            scope_section=scope_section, files_section=files or "(no touched files could be read)", diff_text=diff,
            changed_files=changed, rebuttal_section=rebuttal_section, review_history_section=history_section,
            task_evidence_section=task_evidence_section)

    prompt, stable_len, fit_error = fit_triad_prompt(
        [model], _assemble, files_section, frozen_subject.diff_text, changed, str(review_root), ctx=None,
        subject=managed, slots=[slot],
        compact_diff=(lambda: frozen_subject.render_prompt_diff(0)) if not frozen_subject.is_system_index else None)
    if fit_error:
        raise ValueError(fit_error)
    messages, _bible = triad_api_messages(prompt, stable_len, TRIAD_USER_TURN, layer=layer)
    system = "".join(str(block.get("text") or "") if isinstance(block, dict) else str(block)
                     for block in (messages[0]["content"] if isinstance(messages[0]["content"], list) else [messages[0]["content"]]))
    sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    return {"system": system, "user": TRIAD_USER_TURN, "parts": list(parts),
            "sha": {"brief": sha, "change_prompt_sha": sha, "coupling_brief_sha": ""},
            "delivery": "packet", "manifest": {"governance_manifest": list(governance.manifest), "prompt_chars": len(prompt)},
            "stable_prefix_len": stable_len}


def commit_gate_paid_seats(prepared, exited) -> list:
    """The PAID seats of one commit-gate wave, priced before the first paid call.
    A paid seat is an api row — a packet seat OR a native inspection episode —
    whose every send is a ``reserve_attempt`` on the ledger; an agent-session row
    rides the owner's subscription (its ledger row is written at settlement, never
    reserved) and is not priced. Each seat carries the exact chars of the send its
    substrate opens with (the packet's message pair; a native episode's first send:
    instructions, its OWN two-part brief and tool schemas — later rounds reserve
    themselves) and that send's output reservation, so the wave is priced the way
    ``reserve_attempt`` prices it, the exact route's known response maximum included."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_native_episode import native_first_send_chars
    from ouroboros.reviewer_window import reviewer_window_binding
    from ouroboros.reviewer_slot_config import row_plan_retrieves
    from ouroboros.tools.review_multi_model import (
        TRIAD_ROLE_HINT, TRIAD_USER_TURN, review_output_allowance, triad_api_messages,
    )
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT
    from ouroboros.review_evidence import commit_review_evidence_section

    if exited or not prepared:
        return []
    row_plan = prepared.get("row_plan") or {}
    models = list(prepared.get("models") or row_plan.get("models") or [])
    routes = list(prepared.get("routes") or row_plan.get("routes") or [])
    slot_ids = list(row_plan.get("slot_ids") or [])
    tasks = list(row_plan.get("session_tasks") or [])
    profiles, local = list(row_plan.get("session_profiles") or []), list(row_plan.get("use_local") or [])
    seats, packet_chars = [], None
    for index, model in enumerate(models):
        route = routes[index] if index < len(routes) else "api_chat"
        slot_id = str(slot_ids[index] if index < len(slot_ids) else f"slot_{index + 1}")
        if str(getattr(route, "value", route) or "") == ReviewRouteKind.AGENT_SESSION.value:
            continue
        if row_plan_retrieves({**row_plan, "routes": routes}, index):
            task = str(tasks[index] if index < len(tasks) else "") or str(prepared.get("session_task") or "")
            if prepared.get("task_evidence"):
                task += "\n\n" + commit_review_evidence_section(prepared["task_evidence"], delivery="native")
            chars = native_first_send_chars(
                str(prepared.get("target_repo") or ""), surface="multi_model_review", role_hint=TRIAD_ROLE_HINT,
                slot_id=slot_id, session_task=task, output_contract=REVIEW_TWO_PART_OBJECT_CONTRACT)
        else:
            if packet_chars is None:
                messages, _ = triad_api_messages(
                    str(prepared.get("prompt") or ""), int(prepared.get("stable_prefix_len") or 0), TRIAD_USER_TURN,
                    layer=str(prepared.get("layer") or "body"))
                packet_chars = len(json.dumps({"messages": messages}, ensure_ascii=False, default=str))
            chars = packet_chars
        binding = reviewer_window_binding({"slot_id": slot_id, "session_profile": profiles[index] if index < len(profiles) else "",
                                           "use_local": local[index] if index < len(local) else None})
        seats.append({"surface": "multi_model_review", "slot_id": slot_id, "model": str(model or ""), "prompt_chars": chars,
                      "max_completion_tokens": review_output_allowance(str(model or ""), **binding)})
    return seats


def admit_commit_gate_wave(ctx, seats) -> str | None:
    """Money admission of one commit-gate wave (owner decision 2026-09-05, on
    the known-spend rule of #1487): before ANY seat is dispatched, KNOWN spend
    must be below every fence ``reserve_attempt`` enforces (the global
    TOTAL_BUDGET, root and original group fences). The seats' summed reservation
    bounds are disclosed, not an earlier refusal; a fence reached mid-wave
    refuses the remaining seats at their own reservation with truthful custody.
    Returns the typed refusal text ($0, nothing dispatched) naming the binding
    axis, or None; fail-open on unknowns like the task-level surfaces that
    already ride ``review_wave_budget_gate``."""
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
    limit, known = admission.get("limit_usd"), admission.get("known_usd")
    root_remaining = None if limit is None or known is None else max(0.0, float(limit) - float(known))
    if admission.get("binding_axis") == "global":
        # The refusal names the fence that binds and the knob that moves it — never
        # a per-task fence the wave would have fit.
        fence = (
            f"the global budget TOTAL_BUDGET {usd(admission.get('global_limit_usd'))}: "
            f"known spend={usd(admission.get('global_known_usd'))} across every task (plus "
            f"{usd(admission.get('global_reserved_usd'))} of open holds, not counted), "
            f"remaining={usd(remaining)}; the per-task budget fence "
            f"{usd(limit)} alone would leave {usd(root_remaining)}"
        )
    else:
        label = "whole-work billing-group budget fence" if admission.get("binding_axis") == "group" else "per-task budget fence"
        fence = (
            f"the {label} {usd(limit)}: known spend={usd(known)} (plus "
            f"{usd(admission.get('reserved_usd'))} of open holds, not counted), "
            f"remaining={usd(remaining)}; the global budget "
            f"{usd(admission.get('global_limit_usd'))} alone would leave {usd(admission.get('global_remaining_usd'))}"
        )
    remedy = review_wave_binding_fence(admission)[1]
    return (
        "⚠️ REVIEW_BLOCKED: commit-gate review wave declined before dispatch ($0 spent). "
        f"Known spend has reached {fence}. The wave's reservation upper bound would have been {usd(wave)} ("
        + "; ".join(f"{s['surface']}:{s['slot_id']} {s['model']} {usd(bounds[i])}" for i, s in enumerate(seats))
        + f"). No reviewer seat of the wave was dispatched: {remedy}, then retry the same commit."
    )


def managed_update_wave_estimate(remaining_usd: float) -> dict:
    """Disclosure of ONE commit-gate wave for the managed-update resolver: the
    supervisor event to record. Money admission is the known-spend rule the
    caller checked first (#1487), so this estimate never refuses. The pool is
    the one wave's paid seats — every api-route row of the panel (packet or
    native, both parts of the brief ride one seat) — priced at the packs' own
    worst-case caps (the shared 920K-token input SSOT per API row, the review
    output reserve) with the shared reservation math; agent-session rows ride
    subscriptions and are counted, not priced. An estimator error is recorded
    inside the event, never a zero; this function is called explicitly, so a
    broken import propagates to the caller."""
    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET
    from ouroboros.tools.review_multi_model import _review_output_budget
    from ouroboros.usage_admission import review_wave_admission

    rows = review_pool_slots()
    models = [row.model for row in rows if not row.is_session and row.model]
    event: dict = {"type": "managed_update_wave_estimate", "estimated_wave_usd": None,
                   "exceeds_known_remaining": False, "unpriced_slots": 0,
                   "session_slots": sum(1 for row in rows if row.is_session),
                   "remaining_usd": float(remaining_usd)}
    if not models:
        return event
    try:
        estimate = review_wave_admission(
            root_task_id="managed-update-admission", models=models,
            prompt_chars=int(REVIEW_PROMPT_TOKEN_BUDGET) * 4,
            max_completion_tokens=int(_review_output_budget()),
            remaining_usd_override=float(remaining_usd))
    except Exception as exc:
        log.debug("assisted admission wave estimate failed", exc_info=True)
        return {"type": "managed_update_wave_estimate_failed", "remaining_usd": float(remaining_usd),
                "error": f"{type(exc).__name__}: {exc}"}
    total = estimate.get("estimated_wave_usd")
    event["unpriced_slots"] = int(estimate.get("unpriced_slots") or 0) + (len(models) if total is None else 0)
    if total is not None:
        event["estimated_wave_usd"] = round(float(total), 6)
        event["exceeds_known_remaining"] = float(total) > float(remaining_usd) + 1e-9
    return event
