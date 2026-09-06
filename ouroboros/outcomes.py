"""Typed task/loop outcome helpers.

Lifecycle, execution health, artifacts, review, and objective evaluation are
separate axes.  Objective success is never inferred from final text or the
absence of tool errors; only LLM-first task acceptance review can establish
success, while typed runtime evidence may conservatively degrade an otherwise
``not_evaluated`` objective.
"""

from __future__ import annotations

import copy
import json
import logging
import pathlib
from hashlib import sha256
from typing import Any, Dict, List, Optional

from ouroboros import _outcome_receipts
# Tool-call trace vocabulary + execution-axis classifier (leaf module). Re-exported
# here so `from ouroboros.outcomes import _classify_tool_errors/_POLICY_DENIAL_STATUSES/...`
# keeps resolving for every historical import site.
from ouroboros._outcome_tool_errors import (  # explicit re-exports, one statement
    _BLOCKING_TOOL_STATUSES as _BLOCKING_TOOL_STATUSES,
    _classify_tool_errors as _classify_tool_errors,
    _COSMETIC_TOOL_NAMES as _COSMETIC_TOOL_NAMES,
    _is_ignored_readonly_block as _is_ignored_readonly_block,
    _NON_BLOCKING_READONLY_BLOCK_STATUSES as _NON_BLOCKING_READONLY_BLOCK_STATUSES,
    _NON_BLOCKING_RECOVERABLE_STATUSES as _NON_BLOCKING_RECOVERABLE_STATUSES,
    _OK_TOOL_STATUSES as _OK_TOOL_STATUSES,
    _POLICY_DENIAL_STATUSES as _POLICY_DENIAL_STATUSES,
    _RECOVERY_TOOL_NAMES as _RECOVERY_TOOL_NAMES,
    _ROOT_WRITE_TOOLS as _ROOT_WRITE_TOOLS,
    _unresolved_tool_errors as _unresolved_tool_errors,
    _user_file_basenames as _user_file_basenames,
)
from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_READY,
)
from ouroboros.outcome_receipt_store import (  # noqa: F401 - public compatibility re-exports
    append_verification_receipt,
    merge_verification_receipts,
    read_context_verification_receipts,
    read_verification_receipts,
    read_verification_receipts_from_roots,
    verification_receipts_path,
)
from ouroboros.task_results import (
    STATUS_CANCEL_REQUESTED,
    STATUS_REJECTED_DUPLICATE,
    legacy_plan_review_projection,
    validate_task_id,
)
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)


RESULT_SUCCEEDED = "succeeded"
RESULT_FAILED = "failed"
RESULT_INFRA_FAILED = "infra_failed"
RESULT_PARTIAL = "partial"

OBJECTIVE_NOT_EVALUATED = "not_evaluated"
OBJECTIVE_PASS = "pass"
OBJECTIVE_FAIL = "fail"
OBJECTIVE_DEGRADED = "degraded"

EXECUTION_OK = "ok"
EXECUTION_DEGRADED = "degraded"
EXECUTION_FAILED = "failed"
EXECUTION_INFRA_FAILED = "infra_failed"
EXECUTION_CANCELLED = "cancelled"
EXECUTION_INTERRUPTED = "interrupted"
# Forced finalization (deadline/budget/round limit) with a real extracted
# answer is an honest positive shelf, not a failure. The gate is DETERMINISTIC
# runtime facts only: a force-finalization reason code plus a non-empty,
# non-error final text — never prose classification (P5-safe, no whitewash).
EXECUTION_BEST_EFFORT = "best_effort"

OBJECTIVE_BEST_EFFORT = "best_effort"

# Reason codes whose forced finalization may yield a best-effort outcome.
# deadline_local is the loop-local sibling of finalization_grace (v6.33.0 WS2): a
# genuinely-extracted answer at a real deadline must land as best_effort, not an
# agent failure — same as the supervisor finalize_now path.
# provider_unavailable is deliberately NOT here (it was, until the slime-saga
# audit): a provider outage interrupts a task with the objective unmet, and the
# best-effort promotion turned that into "completed" — a lie that hid a real
# outage from the owner. The rail stamps infra_failed instead (loop.py
# _handle_provider_unavailable); salvage text still rides the result body.
# S3 (Q1/Q3=A, 2026-08-15): the typed rail for the owner's "Wrap up"
# graceful stop — one bounded tool-less finalization turn requested through the
# durable stop intent. Distinct from every deadline/budget truncation reason.
# Defined here, above its first consumer set (module-load order).
REASON_OWNER_REQUESTED_FINALIZATION = "owner_requested_finalization"
BEST_EFFORT_REASON_CODES = frozenset({
    "budget_exhausted",
    "round_limit",
    "finalization_grace",
    "deadline_local",
    "children_unabsorbed",
    # S3 (Q1/Q3=A, 2026-08-15): the owner asked the task to summarize and stop.
    # A successful owner-requested finalization is an honest best-effort
    # completion — NEVER recorded as the false ``acceptance_bypassed_deadline``
    # that reusing finalization_grace would persist (CF-02/REASON-001).
    REASON_OWNER_REQUESTED_FINALIZATION,
})

# Typed final-answer protocol marker (machine-readable deliverable payload,
# separate from reasoning prose). Since v6.60.0 the instruction to end a
# short-deliverable answer with this exact line comes from the per-task
# contract (answer_protocol="final_answer_line"), never from prompts/SYSTEM.md.
FINAL_ANSWER_MARKER = "FINAL ANSWER:"

OUTCOME_TIER_SOLVED = "solved"
OUTCOME_TIER_BEST_EFFORT = "best_effort"
OUTCOME_TIER_BLOCKED = "blocked_with_evidence"
_OUTCOME_TIERS = (OUTCOME_TIER_SOLVED, OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED)

REASON_FINAL_MESSAGE = "final_message"
REASON_EMPTY_FINAL_TEXT = "empty_final_text"
REASON_PROVIDER_FAILURE = "provider_failure"
REASON_TASK_EXCEPTION = "task_exception"
REASON_DEEP_SELF_REVIEW_UNAVAILABLE = "deep_self_review_unavailable"
REASON_DEEP_SELF_REVIEW_ERROR = "deep_self_review_error"
REASON_DEEP_SELF_REVIEW_PACK_UNFIT = "deep_self_review_pack_unfit"
REASON_TOOL_FAILURE = "tool_failure"
REASON_DELIVERY_CONTROL_DEGRADED = "delivery_control_degraded"
REASON_CHILD_RESULTS_DEFERRED = "child_results_deferred"
REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE = "review_skipped_deadline_reserve"
# HQ1 (2026-08-15): the owner-hurry acceptance-skip reason. A finite typed
# acceptance-decision reason ONLY — deliberately NOT a member of
# ACCEPTANCE_BYPASS_REASON_BY_RAIL (it is an owner-approved skip, not a forced
# rail), never a lifecycle status, commit-review reason, truncation reason, or
# BEST_EFFORT reason. ``ouroboros/owner_hurry.py`` is the consumer.
REASON_ACCEPTANCE_SKIPPED_OWNER_HURRY = "owner_hurry"
# Owner D10/D27 (2026-08-15): the shared review-cycle cap (``review_cycles.py``)
# spent under BLOCKING enforcement. One typed reason AND event name for both
# gates: task acceptance (Required+Blocking passes exhausted) stamps it as the
# acceptance-decision reason; plan review stamps it on the held wave/event. The
# host objective then terminalizes as BLOCKED (``blocked_with_evidence``) — never
# ``best_effort`` — with the reviewer findings preserved in the review axis.
REASON_REVIEW_CYCLES_EXHAUSTED = "review_cycles_exhausted"

# B2b: a BLOCKING plan review whose recorded wave proves its reviewer quorum is
# STRUCTURALLY unreachable (typed window-exhausted rows leave fewer live slots
# than the quorum). The gate releases finalization for an agent-CHOSEN honest
# blocked terminal — the review stays open, implementation stays held, and the
# objective terminalizes BLOCKED exactly like the spent-cap case above.
REASON_REVIEW_QUORUM_UNREACHABLE = "plan_review_quorum_unreachable"

# A-material (owner ratification 2026-08-30): the agent resubmitted the SAME paid
# acceptance identity — unchanged candidate answer AND no new obligation
# disposition — so the recorded verdict was replayed for free instead of buying
# another panel. The branch only fires when that recorded verdict was NOT a clean
# PASS (a clean PASS replays as `accepted` on its own branch), so the objective
# terminalizes BLOCKED exactly like the spent-cap case above: the deliverable was
# never accepted and no further reviewer round will happen.
REASON_IDENTICAL_ACCEPTANCE_REFUSED = "identical_acceptance_refused"

# The acceptance-decision reasons whose (finalized_unaccepted, reason) PAIR
# terminalizes the objective axis BLOCKED. Value-keyed readers of the acceptance
# decision live here: adding a terminal reason without adding it to the right key
# is the silent false green DEVELOPMENT.md warns about.
_ACCEPTANCE_BLOCKED_TERMINAL_REASONS = frozenset({
    REASON_REVIEW_CYCLES_EXHAUSTED,
    REASON_IDENTICAL_ACCEPTANCE_REFUSED,
})

# CLOSED mapping: forced-finalization rail (the loop's typed reason_code) -> typed
# acceptance-bypass reason, stamped by the loop's common forced-finalization recorder
# when the panel was OWED (eligible) but a rail ended the task first. Both deadline
# rails collapse to ONE reason; an unmapped rail stamps nothing — the enum stays finite
# and no call site may mint a reason by concatenation (ledger vocabulary only, never
# agent-visible text — the v6.61.4 token-parroting class).
ACCEPTANCE_BYPASS_REASON_BY_RAIL = {
    "budget_exhausted": "acceptance_bypassed_budget_exhausted",
    "round_limit": "acceptance_bypassed_round_limit",
    "finalization_grace": "acceptance_bypassed_deadline",
    "deadline_local": "acceptance_bypassed_deadline",
    "provider_unavailable": "acceptance_bypassed_provider_unavailable",
    "context_overflow": "acceptance_bypassed_context_overflow",
    "children_unabsorbed": "acceptance_bypassed_children_unabsorbed",
    # The owner-stop rail bypasses an owed panel because the OWNER asked the
    # task to wrap up now — its own typed reason, never the deadline's (CF-02).
    REASON_OWNER_REQUESTED_FINALIZATION: "acceptance_bypassed_owner_requested_finalization",
}
ACCEPTANCE_BYPASS_REASONS = frozenset(ACCEPTANCE_BYPASS_REASON_BY_RAIL.values())

# v6.78.0 (owner Q23=B): the HOST acceptance decision has exactly three owner-facing
# states, each carrying a typed `reason` drawn from facts the host already computed
# (dialogue_status, pass_reason, panel/degraded reasons, the pacing launch reason).
# This is the HOST decision vocabulary only — the reviewer's own PASS|FAIL|DEGRADED
# verdict vocabulary is a different layer and is deliberately unchanged.
ACCEPTANCE_ACCEPTED = "accepted"
ACCEPTANCE_REVISION_REQUESTED = "revision_requested"
ACCEPTANCE_FINALIZED_UNACCEPTED = "finalized_unaccepted"
ACCEPTANCE_DECISION_STATUSES = (
    ACCEPTANCE_ACCEPTED, ACCEPTANCE_REVISION_REQUESTED, ACCEPTANCE_FINALIZED_UNACCEPTED,
)

# When cosmetic residual errors exist but no acceptance review ran, the
# execution axis is OK yet "did it actually work?" was never judged: surface a
# structural warning so a default-`auto` overclaim isn't displayed as clean.
WARN_RESIDUAL_TOOL_ERRORS_WITHOUT_REVIEW = "residual_tool_errors_without_review"
# FR3: a turn produced real effects and finished cleanly, but the agent recorded
# NO host-attested verification (no verify_and_record receipt and no trivial
# write/edit deliverable). A BINARY transparency flag that keeps the result solved
# (never a downgrade — anti-oscillation), surfaced loudly on the objective axis.
WARN_RECEIPT_ABSENT = "receipt_absent"
# M2 zero-grounding: the task declared a TYPED expected_output, finished cleanly,
# but the agent did literally no tool work and produced no structured FINAL ANSWER —
# a structural overclaim. Advisory flag (keeps solved); conservative so a normal
# text-answer or tool-using task is never false-flagged.
WARN_EXPECTED_OUTPUT_UNGROUNDED = "expected_output_ungrounded"
# Receipt statuses that count as host-attested grounding (suppress receipt_absent):
# a verify_and_record pass, an observed artifact, or an honest no_visible_machine_contract
# declaration. NOT a fail (that is an overclaim signal, not grounding). A trivial write/
# edit deliverable is its OWN grounding via _trace_has_write_edit_grounding (derived from
# the trace, not a receipt), so it needs no receipt status here.
_RECEIPT_GROUNDING_STATUSES = frozenset({"pass", "observed", "declared"})


def _verification_receipt_is_grounding(receipt: Dict[str, Any]) -> bool:
    """Whether one receipt grounds the task's declared deliverable.

    ``delegation_zero_run`` is a typed lifecycle decision: it proves that the
    configured actor deliberately closed the no-leaf branch, not that the
    requested artifact or answer is correct.  It stays visible in the normal
    receipt/acceptance packet, but must not suppress ``receipt_absent`` or turn a
    lifecycle declaration into host-attested deliverable evidence.
    """
    return (
        str(receipt.get("status") or "") in _RECEIPT_GROUNDING_STATUSES
        and str(receipt.get("contract_kind") or "") != "delegation_zero_run"
    )


def _terminal_zero_run_receipt_present(receipts: List[Dict[str, Any]]) -> bool:
    """True when the actor already made its terminal no-leaf decision.

    Such a receipt is intentionally not grounding, but the local-readonly actor
    also no longer exposes the general verification tool after this terminal
    choice.  Do not inject an impossible ``call verify_and_record`` reminder;
    the final outcome/acceptance projection still discloses missing deliverable
    grounding independently.
    """
    from ouroboros.outcome_receipt_store import terminal_zero_run_receipt

    return any(terminal_zero_run_receipt(receipt) for receipt in receipts)

# Historical name of the RED-reconciling statuses; the SSOT now lives next to the
# reconciliation core it parameterizes (see `_outcome_receipts.RED_RECONCILING_STATUSES`).
_RECEIPT_RED_RECONCILING_STATUSES = _outcome_receipts.RED_RECONCILING_STATUSES

# Ledger entry statuses that do NOT count as a failure for ``summary.has_failures``.
# SSOT: the receipt grounding statuses (pass/observed/declared) are folded in so a turn
# that grounded itself via a successful artifact_observation (``observed``) or an honest
# no_visible_machine_contract declaration (``declared``) is NOT mis-read as a ledger
# failure. A plain run-kind verify pass is already ``pass``.
_LEDGER_NON_FAILURE_STATUSES = (
    frozenset({"", "ok", RESULT_SUCCEEDED, "pass", OBJECTIVE_NOT_EVALUATED, "ignored"})
    | _RECEIPT_GROUNDING_STATUSES
    # refused_out_of_scope: an artifact_observation whose path is outside the observable
    # roots is a POLICY refusal (honest telemetry), NOT a verification failure (v6.57.0).
    | frozenset({"refused_out_of_scope"})
    # tool_reported_failure: the tool RAN and answered honestly ({"ok": false} — a
    # diagnostic reporting what it was called to find). A finding, not a failure.
    | frozenset({"tool_reported_failure"})
)


def _merge_objective_warning(objective: Dict[str, Any], code: str) -> None:
    """Add a structural objective warning WITHOUT clobbering an existing one.
    Warnings can co-occur (cosmetic residual + receipt_absent), so ``warning``
    (singular) stays the primary string for back-compat while ``warnings`` (list)
    accumulates every distinct code. Explicit merge semantics (no last-writer-wins)."""
    if not isinstance(objective, dict) or not code:
        return
    existing = objective.get("warnings")
    warnings = list(existing) if isinstance(existing, list) else []
    primary = objective.get("warning")
    if primary and primary not in warnings:
        warnings.append(primary)
    if code not in warnings:
        warnings.append(code)
    objective["warnings"] = warnings
    if not objective.get("warning"):
        objective["warning"] = code


def _trace_has_write_edit_grounding(llm_trace: Dict[str, Any]) -> bool:
    """Host-derived trivial grounding (FR3): a successful non-scratch write_file/
    edit_text IS its own file-exists receipt (the deliverable provably exists), so it
    suppresses receipt_absent without forcing the agent to call verify_and_record for
    a plain write. Derived from the durable trace at finalization, so no per-write
    handler hook is needed."""
    for call in llm_trace.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("is_error"):
            continue
        if str(call.get("status") or "ok") not in _OK_TOOL_STATUSES:
            continue
        if str(call.get("tool") or "") in _ROOT_WRITE_TOOLS:
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if str(args.get("root") or "active_workspace") not in _SCRATCH_ROOTS:
                return True
    return False


def verification_grounding_present(
    llm_trace: Dict[str, Any], drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """True when the turn already carries host-attested grounding — a verify_and_record
    receipt with a grounding status, or a trivial write/edit deliverable. Read-only
    (shared by the one-shot nudge gate and the receipt_absent flag)."""
    receipt_rows = (
        receipts if isinstance(receipts, list)
        else read_verification_receipts(drive_root, task_id)
    )
    if any(_verification_receipt_is_grounding(receipt) for receipt in receipt_rows):
        return True
    return _trace_has_write_edit_grounding(llm_trace)


def should_nudge_verification(
    llm_trace: Dict[str, Any], drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """FR3 one-shot nudge gate: the turn produced real reviewable effects but recorded
    NO host-attested grounding yet — ping the agent ONCE to verify_and_record before it
    finalizes. Binary; the caller latches it so it fires at most once per task."""
    if not turn_has_reviewable_effects(llm_trace):
        return False
    return not verification_grounding_present(
        llm_trace, drive_root, task_id, receipts=receipts,
    )


def latest_unreconciled_failed_receipt(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure core: the most recent RED receipt (``status=="fail"``) with NO later genuine
    grounding receipt for the SAME verification (a passing run-kind check or an observed
    artifact — see ``_RECEIPT_RED_RECONCILING_STATUSES``; a later ``declared`` does NOT
    reconcile). Returns the failing receipt, or ``None``. Structural: the typed receipt
    status decides pass/fail, and identity is ONE typed key: the ``criterion_id`` when
    present, else the canonical ``check`` text, else the observed ``paths`` set (owner
    Q28=B, content-ADDRESSING — never a semantic keyword gate). Kind AND value must match,
    so a green of another check — or one that omits the id — no longer clears a red; a red
    with NO key at all keeps the older any-later-green rule. Advisory, never a gate.
    The NEWEST element of the OUTSTANDING SET (``_outcome_receipts.unreconciled_failed``)
    — never a single latest-pointer, which a newer red would let erase an older still-red
    one. Shared SSOT by the finalize nudge and the acceptance verification_summary so the
    reconciliation rule lives in one place."""
    return _outcome_receipts.latest_unreconciled_failed(receipts, _RECEIPT_RED_RECONCILING_STATUSES)


def latest_unreconciled_failed_verification(
    drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Disk-backed wrapper of ``latest_unreconciled_failed_receipt`` — reads the task's
    durable receipts. Feeds the one-shot red-verification finalization nudge: finalizing over
    your own host-attested red is a self-contradiction (Bible P3/P12), distinct from the
    receipt_absent case."""
    rows = receipts if isinstance(receipts, list) else read_verification_receipts(drive_root, task_id)
    return latest_unreconciled_failed_receipt(rows)


def latest_unreconciled_masked_pass(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure core (v6.52.2): the most recent PASS receipt whose check can MASK the real exit code
    (``check_exit_masking`` flag from the verify sensor — e.g. ``... | tail``, ``|| true``), with
    NO later CLEAN (non-masked) grounding receipt (a pass/observed whose check is not masked).
    Returns the masked passing receipt, or ``None``. Identity is the ``criterion_id`` key when
    the masked receipt carries one, else ANY clean grounding reconciles: its own text
    identity is the MASKED command, which the remediation necessarily changes, so the red
    path's check-text rule would be unclearable (``_outcome_receipts._reconciles_masked``).
    The NEWEST element of the OUTSTANDING SET (``_outcome_receipts.unreconciled_masked``),
    so a cleanly reconciled newer masked check no longer takes an older one with it.
    FLAG-driven (typed receipt field); advisory only. Shared SSOT by the finalize nudge and
    the acceptance verification_summary."""
    return _outcome_receipts.latest_unreconciled_masked(receipts, _RECEIPT_RED_RECONCILING_STATUSES)


def latest_unreconciled_masked_verification(
    drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Disk-backed wrapper of ``latest_unreconciled_masked_pass`` — feeds the one-shot ADVISORY
    masked-check finalization nudge (the agent may still finalize). Distinct from the red nudge:
    that fires on a RED check; this fires on a green check whose exit code may be laundered."""
    rows = receipts if isinstance(receipts, list) else read_verification_receipts(drive_root, task_id)
    return latest_unreconciled_masked_pass(rows)


def latest_agent_defined_verification(
    drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Newest verify receipt whose criterion was AGENT-DEFINED without a stated basis
    (v6.54.4) — feeds the one-shot advisory criterion-provenance nudge: the check
    passed, but the success criterion was synthesized by the agent, so the agent is
    asked once to confirm it is equivalent to what the task actually requires."""
    rows = receipts if isinstance(receipts, list) else read_verification_receipts(drive_root, task_id)
    return _outcome_receipts.latest_agent_defined(rows)


def apply_receipt_absent_flag(
    loop_outcome: Dict[str, Any], llm_trace: Dict[str, Any], drive_root: Any, task_id: str,
    *, expected_output: str = "", receipts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """FR3 flag (+ M2) — run by the host AFTER ``derive_loop_outcome`` and BEFORE the
    verification ledger. Inject durable verify_and_record receipts into the trace so
    the ledger records them, then on a clean turn (execution ok — NOT best_effort/
    degraded/failed) flag one of two structural transparency signals on the objective
    axis: ``receipt_absent`` (real reviewable effects but no host-attested grounding)
    or, when there were no effects at all, the M2 ``expected_output_ungrounded`` zero-
    grounding signal (a TYPED expected_output was declared yet the agent did no tool
    work and produced no structured FINAL ANSWER). Both are BINARY warnings that keep
    the result solved (never a downgrade — anti-oscillation). Applied before
    ``outcome_axes`` is normalized so the persisted axes and the ledger agree."""
    receipt_rows = (
        receipts if isinstance(receipts, list)
        else read_verification_receipts(drive_root, task_id)
    )
    if receipt_rows:
        llm_trace["verification_receipts"] = receipt_rows
    axes = loop_outcome.get("outcome_axes") if isinstance(loop_outcome.get("outcome_axes"), dict) else {}
    objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else None
    execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    if not isinstance(objective, dict):
        return
    if str(execution.get("status") or "") != EXECUTION_OK:
        return
    if not turn_has_reviewable_effects(llm_trace):
        # M2 zero-grounding: a declared deliverable, no tool work, no structured answer.
        if (
            str(expected_output or "").strip()
            and not (llm_trace.get("tool_calls") or [])
            and not str(loop_outcome.get("final_answer") or "").strip()
        ):
            _merge_objective_warning(objective, WARN_EXPECTED_OUTPUT_UNGROUNDED)
        return
    if verification_grounding_present(
        llm_trace, drive_root, task_id, receipts=receipt_rows,
    ):
        return
    _merge_objective_warning(objective, WARN_RECEIPT_ABSENT)


def terminal_outcome_axes(
    *,
    lifecycle: str,
    execution: str,
    reason_code: str,
    review_trigger: str = "runtime_terminal",
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "lifecycle": {"status": str(lifecycle or "")},
        "execution": {"status": str(execution or ""), "reason_code": str(reason_code or "")},
        "artifacts": {"status": "not_applicable"},
        "objective": {"status": OBJECTIVE_NOT_EVALUATED, "source": "none"},
        "review": {"status": "skipped", "trigger": str(review_trigger or "runtime_terminal")},
    }


def infra_failed_axes(reason_code: str, *, lifecycle: str = "failed", review_trigger: str = "runtime_reconciliation") -> Dict[str, Any]:
    return terminal_outcome_axes(
        lifecycle=lifecycle,
        execution=EXECUTION_INFRA_FAILED,
        reason_code=reason_code,
        review_trigger=review_trigger,
    )


# An undisposed own delegated patch is a DEBT, not a failure: the task's own
# derived verdicts (execution, review, objective, artifacts) are what it earned
# and must survive, so the custody fact is ADDED as an objective warning rather
# than replacing the axes with an infrastructure terminal.
WARN_DELEGATED_CUSTODY_UNRECONCILED = "delegated_custody_unreconciled"


def custody_debt_axes(axes: Any) -> Dict[str, Any]:
    """Add the custody-debt warning to derived axes without rewriting them.

    Idempotent: the overlay is applied again when the result row is stored, and
    ``_merge_objective_warning`` already dedups. Nothing is copied onto the
    execution axis — the debt list itself lives on the row as
    ``delegated_runs_unreconciled`` plus the reconciliation envelope."""
    out = copy.deepcopy(axes) if isinstance(axes, dict) and axes else {}
    objective = out.setdefault(
        "objective", {"status": OBJECTIVE_NOT_EVALUATED, "source": "none"})
    _merge_objective_warning(objective, WARN_DELEGATED_CUSTODY_UNRECONCILED)
    return out

# Tools/roots whose successful use means the turn produced reviewable work.
# Root-aware write tools: these take a `root` arg, so the scratch-exclusion rule
# applies directly. (The retired SDK edit gateway was the one cwd-based coding
# tool; delegated coding now rides the subagent lane, whose integration lands
# through integrate_subagent_patch below — D10.)
_EFFECT_COMMIT_TOOLS = frozenset({"commit_reviewed", "vcs_commit_reviewed"})
# Exclusion model: only pure scratch is exempt. Every other root is a real surface
# (deliverable, workspace, repo, skill payload, or a light-mode skill write via
# runtime_data). Excluding by scratch — not enumerating "deliverable" roots —
# keeps the immune gate complete as roots evolve and errs toward reviewing work.
_SCRATCH_ROOTS = frozenset({"task_drive"})
# Process/service tools that produce a registered deliverable when given outputs=[...].
_EFFECT_PROCESS_TOOLS = frozenset({"run_command", "run_script", "start_service"})
# Substantial cwd-based coding tools: none since D10 retired the SDK edit
# gateway; the set stays so the projection shape (and its consumers) hold.
_EFFECT_CODING_TOOLS = frozenset()
# Parent integration of a child's patch stages a repo mutation -> reviewable work.
# The nanny's explicit apply of a delegated run's captured diff (C1) is the same
# class of staged mutation and rides the same gate.
_EFFECT_INTEGRATION_TOOLS = frozenset({"integrate_subagent_patch", "integrate_delegated_patch"})


def reviewable_effect_projection(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the structured tool effects shared by review and delivery binding.

    Reviewable effects are a successful repo commit; a successful write_file/
    edit_text/apply_patch/edit_batch to any non-scratch root; a successful
    run_command/run_script/start_service that declared deliverable outputs; or any
    successful tool that registered a canonical artifact (artifact_registered — a
    stopped service's outputs or a user_files write). Pure scratch (root=task_drive)
    write_file/edit_text does NOT count. Cognitive-memory updates go through
    update_identity/update_scratchpad/knowledge_write (not write tools) and are
    intentionally not effects; a light-mode generic cognitive write is
    advisory-redirected and never succeeds here. This is a P3 deterministic immune
    signal over observable runtime facts, never message-content inspection.
    """
    effects: List[Dict[str, Any]] = []
    for index, call in enumerate(llm_trace.get("tool_calls") or []):
        if not isinstance(call, dict) or call.get("is_error"):
            continue
        if str(call.get("status") or "ok") not in _OK_TOOL_STATUSES:
            continue
        tool = str(call.get("tool") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        is_effect = tool in _EFFECT_COMMIT_TOOLS or tool in _EFFECT_CODING_TOOLS or tool in _EFFECT_INTEGRATION_TOOLS
        if tool in _ROOT_WRITE_TOOLS and str(args.get("root") or "active_workspace") not in _SCRATCH_ROOTS:
            is_effect = True
        if tool in _EFFECT_PROCESS_TOOLS:
            outputs = args.get("outputs")
            if isinstance(outputs, list) and any(str(item or "").strip() for item in outputs):
                is_effect = True
        # Structured flag set from the full (untruncated) tool result at capture time;
        # covers stopped-service outputs and user_files writes regardless of preview length.
        if call.get("artifact_registered"):
            is_effect = True
        if is_effect:
            effects.append({
                "index": index,
                "tool": tool,
                "args": args,
                "status": str(call.get("status") or "ok"),
                "result": call.get("result"),
                "artifact_registered": bool(call.get("artifact_registered")),
            })
    return effects


def turn_has_reviewable_effects(llm_trace: Dict[str, Any]) -> bool:
    """True when the shared structured projection contains a real effect."""
    return bool(reviewable_effect_projection(llm_trace))


def _extract_outcome_tiers(runs: List[Dict[str, Any]]) -> List[str]:
    """Collect per-actor outcome_tier classifications from review runs.

    On a quorum PASS run, only the actors that CONTRIBUTED the PASS lend their
    tier — a single dissenting/degraded slot's pessimistic tier must not poison a
    clean quorum through the objective axis (the same non-surrender rule the
    aggregate-signal quorum already follows). FAIL/DEGRADED runs stay conservative
    and count every parsed tier.
    """
    tiers: List[str] = []
    for run in runs:
        run_pass = str(run.get("aggregate_signal") or "").upper() == "PASS"
        for actor in run.get("actors") or []:
            if not isinstance(actor, dict):
                continue
            parsed = actor.get("parsed")
            if not isinstance(parsed, dict):
                continue
            if run_pass:
                # Prefer the substrate-recorded signal; fall back to the parsed
                # verdict/status so historical runs (pre-`signal` field) still
                # filter correctly.
                sig = str(actor.get("signal") or parsed.get("verdict") or parsed.get("status") or "").upper()
                if sig != "PASS":
                    continue
            tier = str(parsed.get("outcome_tier") or "").strip().lower()
            if tier in _OUTCOME_TIERS:
                tiers.append(tier)
    return tiers


def _aggregate_outcome_tier(tiers: List[str]) -> str:
    """Worst-tier-wins aggregation: blocked > best_effort > solved."""
    for tier in (OUTCOME_TIER_BLOCKED, OUTCOME_TIER_BEST_EFFORT):
        if tier in tiers:
            return tier
    return OUTCOME_TIER_SOLVED if tiers else ""


def _acceptance_decision_projection(acceptance_decision: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "status": str(acceptance_decision.get("status") or ""),
        # v6.78.0: the typed reason carries the distinction the collapsed status no
        # longer spells out (no-quorum vs FAIL-without-capsule vs obligations open
        # vs capsule spent vs deadline skip). Historical records have no reason.
        "reason": str(acceptance_decision.get("reason") or ""),
        "source": str(acceptance_decision.get("source") or ""),
        "rationale": str(acceptance_decision.get("rationale") or "")[:500],
        "agent_disposition": str(acceptance_decision.get("agent_disposition") or ""),
        "agent_rationale": str(acceptance_decision.get("agent_rationale") or "")[:500],
    }
    # v6.54.4: dissent + obligations transparency (blocking review policy).
    if acceptance_decision.get("dissent_noted"):
        out["dissent_noted"] = True
    if acceptance_decision.get("open_obligations"):
        out["open_obligations"] = [str(x) for x in acceptance_decision.get("open_obligations") or []][:10]
    return out


def _trace_mapping(llm_trace: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = llm_trace.get(key)
    return value if isinstance(value, dict) else {}


def _review_axis(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    review_decision = _trace_mapping(llm_trace, "review_decision")
    acceptance_decision = _trace_mapping(llm_trace, "acceptance_decision")
    # The reconciliation helper retains stale failures conservatively while
    # preventing a superseded PASS from accepting an unbound candidate.
    selection = _outcome_receipts.select_current_review_runs(
        llm_trace.get("review_runs"),
        delivery_candidate=_trace_mapping(llm_trace, "delivery_candidate"),
        review_decision=review_decision,
    )
    runs = selection.current_runs
    if not runs:
        axis = {
            "status": "degraded" if selection.superseded_only_acceptance_gap else "skipped",
            "eligibility": str(review_decision.get("eligibility") or "not_eligible"),
            "trigger": str(review_decision.get("trigger") or "not_evaluated"),
            "run_count": 0,
        }
        if selection.superseded_only_acceptance_gap:
            axis["superseded_run_count"] = len(selection.all_runs)
            axis["superseded_aggregate_signals"] = selection.superseded_aggregate_signals
        if acceptance_decision:
            axis["acceptance_decision"] = _acceptance_decision_projection(acceptance_decision)
        _obligations = [o for o in (llm_trace.get("acceptance_obligations") or []) if isinstance(o, dict)]
        if _obligations:
            axis["acceptance_obligations"] = _obligations[:20]
        return axis
    signals = [str(run.get("aggregate_signal") or "").upper() for run in runs]
    if "FAIL" in signals:
        status = "fail"
    elif "DEGRADED" in signals or any(bool(run.get("degraded")) for run in runs):
        status = "degraded"
    elif "PASS" in signals:
        status = "pass"
    else:
        status = "degraded"
    axis = {
        "status": status,
        "eligibility": str(review_decision.get("eligibility") or "eligible"),
        "trigger": str(review_decision.get("trigger") or "review_run"),
        "run_count": len(runs),
        "aggregate_signals": signals,
    }
    tier = _aggregate_outcome_tier(_extract_outcome_tiers(runs))
    if tier:
        axis["outcome_tier"] = tier
    if acceptance_decision:
        axis["acceptance_decision"] = _acceptance_decision_projection(acceptance_decision)
    _obligations = [o for o in (llm_trace.get("acceptance_obligations") or []) if isinstance(o, dict)]
    if _obligations:
        axis["acceptance_obligations"] = _obligations[:20]
    return axis


def _objective_axis(review: Dict[str, Any]) -> Dict[str, Any]:
    status = str(review.get("status") or "skipped")
    tier = str(review.get("outcome_tier") or "")
    decision = review.get("acceptance_decision") if isinstance(review.get("acceptance_decision"), dict) else {}
    _decision_reason = str(decision.get("reason") or "")
    if (
        str(decision.get("status") or "") == ACCEPTANCE_FINALIZED_UNACCEPTED
        and _decision_reason in _ACCEPTANCE_BLOCKED_TERMINAL_REASONS
    ):
        # D27: Required+Blocking acceptance whose shared cap is spent terminalizes
        # BLOCKED, whatever tier the last (failed) review proposed. A-material
        # (2026-08-30) adds the identical-paid-identity refusal on the same key:
        # the deliverable was never accepted and no further round will happen, so
        # the last review's proposed tier must not read as a green objective.
        return {
            "status": OBJECTIVE_FAIL,
            "source": "task_acceptance_review",
            "review_status": status,
            "outcome_tier": OUTCOME_TIER_BLOCKED,
            "reason": _decision_reason,
        }
    if tier:
        # Reviewer tier is the canonical objective lexicon (completion-coach):
        # solved -> pass, best_effort -> best_effort, blocked_with_evidence ->
        # fail. The false-solved veto is structural AND conservative: a solved
        # claim earns PASS only from a clean PASS review; a DEGRADED review
        # (quorum not met / slot failures) keeps objective degraded exactly as
        # before this feature, and a FAIL verdict blocks the claim outright.
        if tier == OUTCOME_TIER_SOLVED and status == "pass":
            objective = OBJECTIVE_PASS
        elif tier == OUTCOME_TIER_SOLVED and status == "fail":
            objective = OBJECTIVE_FAIL
        elif tier == OUTCOME_TIER_SOLVED:
            objective = OBJECTIVE_DEGRADED
        elif tier == OUTCOME_TIER_BEST_EFFORT:
            objective = OBJECTIVE_BEST_EFFORT
        else:
            objective = OBJECTIVE_FAIL
        return {
            "status": objective,
            "source": "task_acceptance_review",
            "review_status": status,
            "outcome_tier": tier,
        }
    if status == "pass":
        objective = OBJECTIVE_PASS
    elif status == "fail":
        objective = OBJECTIVE_FAIL
    elif status == "degraded":
        objective = OBJECTIVE_DEGRADED
    else:
        objective = OBJECTIVE_NOT_EVALUATED
    return {
        "status": objective,
        "source": "task_acceptance_review" if objective != OBJECTIVE_NOT_EVALUATED else "none",
        "review_status": status,
    }


def extract_final_answer(text: str) -> str:
    """Extract the typed FINAL ANSWER payload from the final message.

    Protocol: the LAST line starting with the exact ``FINAL ANSWER:`` marker
    carries the machine-readable deliverable (separate from reasoning prose).
    Returns "" when the protocol is not used.

    Structural invariant: the snake_case outcome-tier identifiers
    (``best_effort``/``blocked_with_evidence``) are internal ledger vocabulary
    from the acceptance-review contract, never a legitimate deliverable — a
    reviewed GAIA run shipped ``FINAL ANSWER: blocked_with_evidence`` verbatim
    after an acceptance downgrade. Such an answer counts as missing so the
    marker nudge / salvage path can recover a real one. ``solved`` is NOT
    rejected: it is an ordinary English word that can be a real answer.
    """
    answer = ""
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(FINAL_ANSWER_MARKER):
            answer = stripped[len(FINAL_ANSWER_MARKER):].strip()
    if answer.strip().lower() in (OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED):
        return ""
    return answer


def _merge_axis(default: Dict[str, Any], value: Any) -> Dict[str, Any]:
    merged = dict(default)
    if isinstance(value, dict):
        merged.update(value)
    return merged


def normalize_outcome_axes(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return canonical axes for new and historical task result records."""

    legacy = str(result.get("result_status") or "").strip().lower()
    reason = str(result.get("reason_code") or "").strip()
    status = str(result.get("status") or "").strip().lower()
    if legacy == RESULT_INFRA_FAILED:
        execution = EXECUTION_INFRA_FAILED
    elif legacy == RESULT_FAILED:
        execution = EXECUTION_FAILED
    elif legacy == RESULT_PARTIAL:
        execution = EXECUTION_DEGRADED
    elif legacy == EXECUTION_BEST_EFFORT:
        execution = EXECUTION_BEST_EFFORT
    elif legacy == RESULT_SUCCEEDED:
        execution = EXECUTION_OK
    elif legacy == EXECUTION_CANCELLED:
        execution = EXECUTION_CANCELLED
        reason = reason or EXECUTION_CANCELLED
    elif legacy == EXECUTION_INTERRUPTED:
        execution = EXECUTION_INTERRUPTED
        reason = reason or EXECUTION_INTERRUPTED
    elif legacy and legacy != RESULT_SUCCEEDED:
        execution = EXECUTION_DEGRADED
        reason = reason or f"unknown_legacy_status:{legacy}"
    else:
        execution = EXECUTION_OK
    if not legacy and status in {EXECUTION_CANCELLED, STATUS_CANCEL_REQUESTED}:
        execution = EXECUTION_CANCELLED
        reason = reason or status or EXECUTION_CANCELLED
    elif not legacy and status == EXECUTION_INTERRUPTED:
        execution = EXECUTION_INTERRUPTED
        reason = reason or EXECUTION_INTERRUPTED
    elif not legacy and status == STATUS_REJECTED_DUPLICATE:
        execution = EXECUTION_OK
        reason = reason or "scheduler_duplicate_rejection"
    elif not legacy and status == "failed":
        execution = EXECUTION_FAILED
        reason = reason or status
    artifact_bundle = result.get("artifact_bundle") if isinstance(result.get("artifact_bundle"), dict) else {}
    explicit_artifact_status = str(artifact_bundle.get("status") or result.get("artifact_status") or "").strip()
    artifact_status = explicit_artifact_status or "not_applicable"
    default_axes = {
        "schema_version": 1,
        "lifecycle": {"status": str(result.get("status") or "")},
        "execution": {"status": execution, "reason_code": reason},
        "artifacts": {"status": artifact_status},
        "objective": {"status": OBJECTIVE_NOT_EVALUATED, "source": "legacy_normalizer" if legacy else "none"},
        "review": {"status": "skipped", "trigger": "legacy" if legacy else "not_evaluated"},
    }
    if legacy and legacy not in {RESULT_SUCCEEDED, RESULT_FAILED, RESULT_INFRA_FAILED, RESULT_PARTIAL}:
        default_axes["execution"]["legacy_status"] = legacy
    axes = result.get("outcome_axes") if isinstance(result.get("outcome_axes"), dict) else {}
    if not axes:
        return default_axes
    normalized = {
        "schema_version": axes.get("schema_version") or 1,
        "lifecycle": _merge_axis(default_axes["lifecycle"], axes.get("lifecycle")),
        "execution": _merge_axis(default_axes["execution"], axes.get("execution")),
        "artifacts": _merge_axis(default_axes["artifacts"], axes.get("artifacts")),
        "objective": _merge_axis(default_axes["objective"], axes.get("objective")),
        "review": _merge_axis(default_axes["review"], axes.get("review")),
    }
    if result.get("status"):
        normalized["lifecycle"]["status"] = str(result.get("status") or "")
    if explicit_artifact_status:
        normalized["artifacts"]["status"] = explicit_artifact_status
    objective = normalized.get("objective") if isinstance(normalized.get("objective"), dict) else {}
    objective_status = str(objective.get("status") or OBJECTIVE_NOT_EVALUATED)
    objective_source = str(objective.get("source") or "none")
    if objective_status != OBJECTIVE_NOT_EVALUATED and objective_source != "task_acceptance_review":
        normalized["objective"] = {
            **objective,
            "status": OBJECTIVE_NOT_EVALUATED,
            "source": "none",
            "ignored_status": objective_status,
            "ignored_source": objective_source,
        }
    for key, value in axes.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def public_task_result(result: Dict[str, Any], *, include_outcome_axes: bool = True) -> Dict[str, Any]:
    """Project persisted/effective task results onto the public task-result contract."""

    if not isinstance(result, dict):
        return {}
    public: Any = {}
    stack: List[tuple[Any, Any, Any]] = [(result, None, None)]
    while stack:
        value, parent, key = stack.pop()
        if isinstance(value, dict):
            clone = {
                item_key: item_value
                for item_key, item_value in value.items()
                if item_key not in {"result_status", "compat_result_status"}
            }
            if parent is None:
                public = clone
            else:
                parent[key] = clone
            for child_key, child_value in list(clone.items()):
                if isinstance(child_value, (dict, list)):
                    stack.append((child_value, clone, child_key))
        elif isinstance(value, list):
            clone = list(value)
            if parent is None:
                public = clone
            else:
                parent[key] = clone
            for child_key, child_value in enumerate(clone):
                if isinstance(child_value, (dict, list)):
                    stack.append((child_value, clone, child_key))
    if not isinstance(public, dict):
        return {}
    # ABI-3 projection boundary: the public contract carries NO retired cost
    # alias — a stored legacy row's pair resolves deprecated-wins and leaves
    # under the honest names only, at the top level and on the nested public
    # cost planes (the subagent envelope with its usage snapshot and the
    # loop-outcome usage snapshot) — ONE shared normalizer with the
    # write_task_result rewrite seam (fix-round-3). Internal planes that
    # merely share the spelling inside evidence blobs (review receipts,
    # ledger rows) are their own schemas and pass through untouched.
    from ouroboros.cost_projection import normalize_task_result_cost_planes

    public = normalize_task_result_cost_planes(public)
    plan_state = public.get("plan_review_state")
    if isinstance(plan_state, dict) and plan_state.get("schema_version") == 1:
        plan_state["legacy_v1_projection"] = legacy_plan_review_projection(plan_state)
    if include_outcome_axes:
        public["outcome_axes"] = normalize_outcome_axes(result)
    return public


# INFRA-failure host-fallback text prefixes: (prefix, failure kind, default reason_code).
_INFRA_TEXT_PREFIXES = (
    ("⚠️ Failed to get a response", "provider", REASON_PROVIDER_FAILURE),
    ("⚠️ All models are down", "provider", REASON_PROVIDER_FAILURE),
    ("⚠️ Error during processing:", "runtime", REASON_TASK_EXCEPTION),
    ("❌ Deep self-review unavailable:", "runtime", REASON_DEEP_SELF_REVIEW_UNAVAILABLE),
    ("⚠️ Deep self-review error:", "runtime", REASON_DEEP_SELF_REVIEW_ERROR),
    ("❌ Deep self-review failed:", "runtime", REASON_DEEP_SELF_REVIEW_ERROR),
    ("❌ Deep self-review pack unfit:", "runtime", REASON_DEEP_SELF_REVIEW_PACK_UNFIT),
)


def _apply_actor_first_terminal_projection(
    outcome: Dict[str, Any], usage: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay the configured actor's unresolved terminal fact on normal outcome truth."""

    actor = usage.get("actor_first_terminal")
    if not isinstance(actor, dict) or not actor:
        return outcome
    actor = dict(actor)
    axes = outcome.get("outcome_axes") if isinstance(outcome.get("outcome_axes"), dict) else {}
    execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    if str(execution.get("status") or "") == EXECUTION_OK:
        actor_status = str(actor.get("status") or "unknown")
        if actor_status not in {"incomplete", "unknown"}:
            actor_status = "unknown"
        reason = f"configured_actor_{actor_status}"
        failure = {
            "kind": "configured_actor", "status": actor_status,
            "reason_code": reason, **actor,
        }
        execution.update({
            "status": EXECUTION_DEGRADED, "reason_code": reason,
            "failure": failure,
        })
        outcome.update({"finish_reason": reason, "reason_code": reason, "failure": failure})
    execution["actor_first_terminal"] = actor
    objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else {}
    if objective.get("status") != OBJECTIVE_FAIL:
        objective.update({
            "status": OBJECTIVE_DEGRADED,
            "source": "configured_actor_terminal",
            "actor_status": str(actor.get("status") or "unknown"),
        })
    outcome["actor_first_terminal"] = actor
    return outcome


def _loop_usage_snapshot(usage: Dict[str, Any], resource_limit: Dict[str, Any]) -> Dict[str, Any]:
    """The loop-outcome's flat usage snapshot (module-size law extraction).

    ABI-3: the loop's own accounted cost rides the honest name — this
    sub-dict reaches the public task-result payload through ``loop_outcome``
    (stored legacy rows still resolve deprecated-wins at the projection
    boundary)."""
    return {
        "accounted_upper_bound_usd": (
            round(float(usage["cost"]), 6)
            if usage.get("cost") is not None else None
        ),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_rounds": int(usage.get("rounds") or 0),
        **({"resource_limit": resource_limit} if resource_limit else {}),
    }


def derive_loop_outcome(final_text: str, usage: Dict[str, Any], llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return a typed LoopOutcome-compatible dict."""

    usage_status = str(usage.get("execution_status") or usage.get("result_status") or "").strip()
    usage_reason = str(usage.get("reason_code") or "").strip()
    resource_limit = dict(usage.get("resource_limit") or {}) if isinstance(usage.get("resource_limit"), dict) else {}
    text = str(final_text or "")
    failure: Dict[str, Any] | None = None
    execution_status = EXECUTION_OK
    reason_code = REASON_FINAL_MESSAGE
    tool_error_state = _classify_tool_errors(llm_trace)
    tool_errors = tool_error_state.get("unresolved") or []
    recovered_tool_errors = tool_error_state.get("recovered") or []
    cosmetic_tool_errors = tool_error_state.get("cosmetic") or []
    # A2: read-only access-policy blocks — recorded for forensics, never degrading
    # and (unlike cosmetic) never a residual-warning trigger.
    ignored_tool_errors = tool_error_state.get("ignored") or []
    # v6.57.0: unrecovered POLICY refusals (write/shell/integration `*_blocked`) —
    # telemetry only, never degrading and never a `tool_failure` headline.
    policy_denials = tool_error_state.get("policy_denials") or []
    delivery_candidate = _trace_mapping(llm_trace, "delivery_candidate")
    acceptance_decision = _trace_mapping(llm_trace, "acceptance_decision")
    review_decision = _trace_mapping(llm_trace, "review_decision")
    mutation_attribution = _trace_mapping(llm_trace, "mutation_attribution")
    # v6.78.0: keyed on the CANONICAL status plus the typed reason (before the
    # three-state collapse the reason literal WAS the status). Missing this pairing
    # would silently stop degrading an eligible-but-skipped panel — a false green;
    # keying on the status alone would degrade honest capsule_spent finalizations.
    # The forced-rail bypass reasons (ACCEPTANCE_BYPASS_REASONS) ride the same key.
    _acceptance_reason = str(acceptance_decision.get("reason") or "")
    acceptance_review_skipped_eligible = (
        str(acceptance_decision.get("status") or "") == ACCEPTANCE_FINALIZED_UNACCEPTED
        and (
            _acceptance_reason == REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE
            or _acceptance_reason in ACCEPTANCE_BYPASS_REASONS
        )
        and str(review_decision.get("eligibility") or "") == "eligible"
    )
    disposition_projection = _trace_mapping(llm_trace, "child_result_dispositions")
    deferred_child_count = int(disposition_projection.get("deferred_count") or 0)
    degraded_reason = str(delivery_candidate.get("degraded_reason") or "")
    deferred_child_suffix = bool(deferred_child_count and degraded_reason == "host_child_status_suffix")
    forced_best_effort_with_deferred_child = bool(
        deferred_child_count
        # provider_unavailable left the best-effort set (2026-08-10 saga:
        # a provider-killed task is failed, not best-effort), but a forced
        # provider rail must still not erase the more specific
        # deferred-child objective below.
        and (degraded_reason in BEST_EFFORT_REASON_CODES or degraded_reason == "provider_unavailable")
    )
    verification_failures: List[Dict[str, Any]] = []
    for event in llm_trace.get("verification_events") or []:
        if not isinstance(event, dict):
            continue
        for service in event.get("services") or []:
            if not isinstance(service, dict):
                continue
            artifact_text = str(service.get("artifact_outputs") or "")
            if bool(service.get("artifact_output_failed")) or artifact_text.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"):
                verification_failures.append({
                    "kind": str(event.get("kind") or "runtime_event"),
                    "service": service.get("name"),
                    "status": "artifact_output_error",
                    "reason": artifact_text[:500],
                })

    if usage_status == RESULT_INFRA_FAILED:
        execution_status = EXECUTION_INFRA_FAILED
        reason_code = usage_reason or REASON_PROVIDER_FAILURE
        failure = {"kind": "provider", "reason_code": reason_code}
        # The overflow salvage keeps `llm_api_error`; a waited-out outage or the unknown
        # no-resend fence may leave the same sticky kind behind under its own reason code,
        # and the published projection must not contradict the terminal that chose it.
        if reason_code == "llm_api_error" and str(usage.get("_last_llm_error_kind") or "") == "context_overflow":
            failure["error_kind"] = "context_overflow"
    elif (
        usage_status == RESULT_FAILED
        and usage_reason in BEST_EFFORT_REASON_CODES
        and bool(usage.get("_best_effort_extracted"))
        and text.strip()
        and not text.lstrip().startswith(("⚠️", "❌"))
    ):
        # Forced finalization (deadline grace / budget / round limit) that
        # actually EXTRACTED a model answer: honest best-effort, not failure.
        # Deterministic structural gate: forced reason code + the loop's typed
        # "model answer extracted" fact + non-empty non-error text. Host
        # fallback strings (e.g. budget rejection notices) never set the
        # extraction fact and stay failed — no text-shape whitewashing.
        execution_status = EXECUTION_BEST_EFFORT
        reason_code = usage_reason
        failure = None
    elif usage_status == RESULT_FAILED:
        execution_status = EXECUTION_FAILED
        reason_code = usage_reason or REASON_EMPTY_FINAL_TEXT
        failure = {"kind": "agent", "reason_code": reason_code}
    elif not text.strip():
        execution_status = EXECUTION_FAILED
        reason_code = REASON_EMPTY_FINAL_TEXT
        failure = {"kind": "agent", "reason_code": reason_code}
    elif (_infra := next(
        (row for row in _INFRA_TEXT_PREFIXES if text.lstrip().startswith(row[0])), None,
    )) is not None:
        execution_status = EXECUTION_INFRA_FAILED
        reason_code = usage_reason or _infra[2]
        failure = {"kind": _infra[1], "reason_code": reason_code}
    elif delivery_candidate.get("degraded") and not deferred_child_suffix:
        execution_status = EXECUTION_DEGRADED
        reason_code = usage_reason or degraded_reason or REASON_DELIVERY_CONTROL_DEGRADED
        failure = {"kind": "finalization_control", "reason_code": reason_code}
    elif deferred_child_count:
        execution_status = EXECUTION_DEGRADED
        reason_code = usage_reason or REASON_CHILD_RESULTS_DEFERRED
        failure = {
            "kind": "child_result_disposition",
            "reason_code": reason_code,
            "deferred_count": deferred_child_count,
        }
    elif verification_failures:
        execution_status = EXECUTION_DEGRADED
        reason_code = usage_reason or REASON_TOOL_FAILURE
        failure = {
            "kind": "verification",
            "reason_code": reason_code,
            "verification_failures": verification_failures[:20],
        }
    elif tool_errors:
        execution_status = EXECUTION_DEGRADED
        reason_code = usage_reason or REASON_TOOL_FAILURE
        failure = {
            "kind": "tool",
            "reason_code": reason_code,
            "tool_errors": tool_errors[:20],
        }

    # A skipped-or-bypassed eligible panel is not a verdict, but cannot remain clean;
    # preserve stronger classifications and degrade only the false-green remainder.
    # Honest reachability (measured, not asserted): the FORCED-rail bypass reasons
    # cannot arrive here on an OK execution — a bypass is stamped only when the rail
    # already wrote `usage.reason_code`, and every writer of that key also writes
    # `execution_status='failed'` (the provider rail upgrades it to 'infra_failed'),
    # so those runs land on the STRONGER failed/infra_failed /
    # best_effort branches above and the owner-visible bypass rides the review axis
    # (see test_forced_rail_axes_are_the_production_shape). What this branch actually
    # decides is the pacing skip (REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE).
    # The bypass keys stay in the condition as a BACKSTOP, not as a live behaviour
    # claim: a future rail that bypasses an owed panel without failing the usage must
    # not be able to come back as a clean green.
    if acceptance_review_skipped_eligible and execution_status == EXECUTION_OK:
        execution_status = EXECUTION_DEGRADED
        reason_code = _acceptance_reason
        failure = {
            "kind": "task_acceptance",
            "reason_code": reason_code,
        }

    review = _review_axis(llm_trace)
    objective = _objective_axis(review)
    plan_gate = _trace_mapping(llm_trace, "force_plan_decision")
    _plan_gate_status = str(plan_gate.get("status") or "")
    if str(plan_gate.get("enforcement") or "") == "blocking" and (
        _plan_gate_status == "cycles_exhausted"
        or (_plan_gate_status == "open" and plan_gate.get("quorum_unreachable"))
    ):
        # D27: a blocking plan review whose cycle cap is spent never closed — the
        # task terminalizes BLOCKED, never best_effort. B2b extends the same honest
        # terminal to a structurally unreachable reviewer quorum (the agent CHOSE to
        # finalize; the review itself stays open and implementation stayed held).
        _quorum_case = _plan_gate_status != "cycles_exhausted"
        objective.update({
            "status": OBJECTIVE_FAIL,
            "source": ("plan_review_quorum_unreachable" if _quorum_case
                       else "plan_review_cycles_exhausted"),
            "outcome_tier": OUTCOME_TIER_BLOCKED,
            "reason": (REASON_REVIEW_QUORUM_UNREACHABLE if _quorum_case
                       else REASON_REVIEW_CYCLES_EXHAUSTED),
        })
    if deferred_child_count and objective.get("status") != OBJECTIVE_FAIL:
        objective.update({
            "status": OBJECTIVE_BEST_EFFORT,
            "source": "child_result_disposition",
            "deferred_count": deferred_child_count,
        })
    # A forced rail already owns the execution reason; its generic candidate
    # degradation must not erase the more specific deferred-child objective.
    # Invalid finalization controls remain degraded through the branch below.
    if (
        delivery_candidate.get("degraded")
        and not deferred_child_suffix
        and not forced_best_effort_with_deferred_child
        and objective.get("status") != OBJECTIVE_FAIL
    ):
        objective.update({
            "status": OBJECTIVE_DEGRADED,
            "source": "delivery_finalization_control",
        })
    if (
        acceptance_review_skipped_eligible
        and objective.get("status") == OBJECTIVE_NOT_EVALUATED
    ):
        objective.update({
            "status": OBJECTIVE_DEGRADED,
            "source": (
                "task_acceptance_deadline_reserve"
                if _acceptance_reason == REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE
                else "task_acceptance_forced_bypass"
            ),
        })
    # Mutation attribution is evidence for the reviewing panels (attached to the
    # failure-evidence projection below), deliberately never a structural veto.
    # T4 honest residual: cosmetic shell errors no longer degrade execution, so
    # when the objective was never judged (default "auto" with no self-call ->
    # objective not_evaluated) a real overclaim could read as clean. Surface a
    # structural warning (not a failure) so the UI escalates it. Gating on the
    # objective being genuinely unjudged is the honest condition: a review that
    # ran (any verdict) already judged it. No review is auto-run, no env knob, no
    # content inference (Bible P5).
    if cosmetic_tool_errors and objective.get("status") == OBJECTIVE_NOT_EVALUATED:
        _merge_objective_warning(objective, WARN_RESIDUAL_TOOL_ERRORS_WITHOUT_REVIEW)
    final_answer_payload = (
        extract_final_answer(text)
        or (
            str(llm_trace.get("best_valid_final_answer") or "")
            if len(llm_trace.get("tool_calls") or []) <= int(llm_trace.get("best_valid_final_answer_tools") or 0)
            else ""
        )
    )
    headline_reason = reason_code
    headline_failure = failure
    if (
        final_answer_payload
        and execution_status == EXECUTION_DEGRADED
        and reason_code == REASON_TOOL_FAILURE
        and text.strip()
        and not text.lstrip().startswith(("⚠️", "❌"))
    ):
        # Keep execution-health honest in outcome_axes.execution, but do not
        # headline a completed answer-bearing task as a top-level tool failure.
        headline_reason = REASON_FINAL_MESSAGE
        headline_failure = None

    outcome_axes = {
        "schema_version": 1,
        "lifecycle": {"status": "completed"},
        "execution": {
            "status": execution_status,
            "reason_code": reason_code,
            "failure": failure,
            **({"resource_limit": resource_limit} if resource_limit else {}),
            "recoveries": recovered_tool_errors[:20],
            "cosmetic_tool_errors": cosmetic_tool_errors[:20],
            "ignored_tool_errors": ignored_tool_errors[:20],
            "policy_denials": policy_denials[:20],
            **({"mutation_attribution": mutation_attribution} if mutation_attribution else {}),
        },
        "artifacts": {"status": "not_applicable"},
        "objective": objective,
        "review": review,
    }
    outcome = {
        "schema_version": 3,
        "outcome_axes": outcome_axes,
        "review_eligibility": str(review.get("eligibility") or "not_eligible"),
        "review_trigger": str(review.get("trigger") or "not_evaluated"),
        "finish_reason": headline_reason,
        "reason_code": headline_reason,
        "final_text": text,
        # Answer precedence: the final text's explicit FINAL ANSWER marker > the latched
        # answer from an earlier round. The latch recovers a produced answer whenever the
        # final text LACKS a marker (whether empty OR marker-less prose — both lose the
        # structured deliverable a downstream extractor needs) AND no NEW tool work
        # happened since it was stamped. The tool-count guard is the key invariant: with
        # no new grounding, a later marker-less round is the model second-guessing its OWN
        # answer under review PRESSURE, which BIBLE Q7 says review must not let DOWNGRADE a
        # produced answer; new grounding (a higher tool count) instead invalidates the latch.
        "final_answer": final_answer_payload,
        # v6.60.0: keyed on the TYPED final_answer payload (extracted OR latch-recovered),
        # not a re-scan of the final text — a task whose earlier-round answer was latched
        # is not "missing" one; marker-free tasks (no answer_protocol) simply read True,
        # which downstream consumers must interpret via the contract, not as a failure.
        "final_answer_missing_sentinel": not final_answer_payload,
        "failure": headline_failure, "degraded": bool(delivery_candidate.get("degraded")), "degraded_reason": degraded_reason,
        "recoveries": recovered_tool_errors[:20],
        "usage": _loop_usage_snapshot(usage, resource_limit),
        "trace_refs": collect_trace_refs(usage, llm_trace),
    }
    return _apply_actor_first_terminal_projection(outcome, usage)


def collect_trace_refs(usage: Dict[str, Any], llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    execution_id = str(usage.get("execution_id") or "").strip()
    if execution_id:
        refs["execution_id"] = execution_id
    llm_refs = [
        {key: item.get(key) for key in (
            "llm_call_id", "execution_id", "round_id", "round", "request_ref",
            "response_ref", "model", "resolved_model", "provider",
        )}
        for item in usage.get("llm_call_refs") or []
        if isinstance(item, dict)
    ]
    if llm_refs:
        refs["llm_call_refs"] = llm_refs
    tool_refs = []
    for item in llm_trace.get("tool_calls") or []:
        if isinstance(item, dict) and item.get("trace_ref"):
            trace = item.get("trace_ref") if isinstance(item.get("trace_ref"), dict) else {}
            tool_refs.append({key: trace.get(key) for key in (
                "call_id", "manifest_ref", "redacted_projection_ref", "redaction",
            )})
    if tool_refs:
        refs["tool_call_refs"] = tool_refs
    return refs


def artifact_bundle_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return v2 ArtifactBundle while preserving old artifact fields."""
    existing_bundle = result.get("artifact_bundle") if isinstance(result.get("artifact_bundle"), dict) else {}
    artifacts = list(result.get("artifacts") or []) if isinstance(result.get("artifacts"), list) else []
    bundle_status = str(existing_bundle.get("status") or "").strip()
    old_status = str(result.get("artifact_status") or "").strip()
    axes = result.get("outcome_axes") if isinstance(result.get("outcome_axes"), dict) else {}
    artifact_axis = axes.get("artifacts") if isinstance(axes.get("artifacts"), dict) else {}
    axis_status = str(artifact_axis.get("status") or "").strip()
    explicit_status = bundle_status or old_status
    if explicit_status in {
        ARTIFACT_STATUS_PENDING,
        ARTIFACT_STATUS_FINALIZING,
        ARTIFACT_STATUS_READY,
        ARTIFACT_STATUS_FAILED,
        "ready_with_changes",
        "ready_no_changes",
        "missing",
        "not_applicable",
    }:
        status = explicit_status
    elif axis_status:
        status = axis_status
    elif artifacts:
        status = ARTIFACT_STATUS_READY
    else:
        status = "not_applicable"
    records: List[Dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        explicit_status = str(item.get("status") or "").strip()
        if explicit_status:
            artifact_status = explicit_status
        elif path and pathlib.Path(path).exists():
            artifact_status = ARTIFACT_STATUS_READY
        elif path:
            artifact_status = "missing"
        elif status in {ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING}:
            artifact_status = status
        else:
            artifact_status = ARTIFACT_STATUS_READY
        record = {
            "kind": str(item.get("kind") or ""),
            "name": str(item.get("name") or pathlib.Path(path).name),
            "path": path,
            "size": int(item.get("size") or 0),
            "sha256": str(item.get("sha256") or ""),
            "status": artifact_status,
            "errors": list(item.get("errors") or []) if isinstance(item.get("errors"), list) else [],
        }
        records.append(record)
    if status != ARTIFACT_STATUS_FAILED and any(str(item.get("status") or "") == "missing" for item in records):
        status = "missing"
    errors = []
    if result.get("artifact_error"):
        errors.append(str(result.get("artifact_error")))
    return {
        "schema_version": 1,
        "status": status,
        "artifacts": records,
        "errors": errors,
    }


def refresh_verification_ledger_artifacts(
    ledger: Dict[str, Any] | None,
    artifact_bundle: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Return ``ledger`` with artifact status synchronized after finalization."""

    if not isinstance(ledger, dict):
        return ledger
    # An omitted-to-artifact stub is a PROJECTION of the artifact file, not a
    # source: it carries no entries, so rebuilding from it would mint "0
    # entries / no failures / execution ok" over the real ledger's summary.
    if ledger.get("omitted_to_artifact"):
        return ledger
    entries = [
        item for item in (ledger.get("entries") or [])
        if not (isinstance(item, dict) and item.get("kind") == "artifact_bundle")
    ]
    artifact_status = str((artifact_bundle or {}).get("status") or "")
    if artifact_status in {ARTIFACT_STATUS_FAILED, ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING, "missing"}:
        entries.append({
            "kind": "artifact_bundle",
            "status": artifact_status,
            "errors": (artifact_bundle or {}).get("errors") or [],
        })
    updated = dict(ledger)
    updated["entries"] = entries
    axes = normalize_outcome_axes({"outcome_axes": updated.get("outcome_axes") if isinstance(updated.get("outcome_axes"), dict) else {}})
    if artifact_status:
        artifact_axis = dict(axes.get("artifacts") or {})
        artifact_axis["status"] = artifact_status
        axes["artifacts"] = artifact_axis
    updated["outcome_axes"] = axes
    updated["summary"] = {
        "entry_count": len(entries),
        "has_failures": any(
            str(item.get("status") or "").lower() not in _LEDGER_NON_FAILURE_STATUSES
            and not (str(item.get("kind") or "") == "task_contract" and str(item.get("status") or "").lower() in {"draft", "recorded"})
            for item in entries
            if isinstance(item, dict)
        ),
    }
    return updated


def build_verification_ledger(
    *,
    task: Dict[str, Any],
    loop_outcome: Dict[str, Any],
    llm_trace: Dict[str, Any],
    artifact_bundle: Dict[str, Any],
    review_evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a task-scoped verification ledger from authoritative runtime facts."""

    entries: List[Dict[str, Any]] = []
    axes = loop_outcome.get("outcome_axes") if isinstance(loop_outcome.get("outcome_axes"), dict) else {}
    execution_axis = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    if str(execution_axis.get("status") or "") not in {"", EXECUTION_OK}:
        entries.append({
            "kind": "loop_outcome",
            "status": execution_axis.get("status"),
            "reason_code": loop_outcome.get("reason_code"),
        })
    objective_axis = axes.get("objective") if isinstance(axes.get("objective"), dict) else {}
    entries.append({
        "kind": "objective_outcome",
        "status": objective_axis.get("status") or OBJECTIVE_NOT_EVALUATED,
        "source": objective_axis.get("source") or "none",
    })
    if isinstance(task.get("task_contract"), dict):
        contract = task.get("task_contract") or {}
        entries.append({
            "kind": "task_contract",
            "status": "recorded",
            "contract_status": str(contract.get("status") or "draft"),
            "objective": str(contract.get("objective") or ""),
            "expected_output": str(contract.get("expected_output") or ""),
        })

    for idx, call in enumerate(llm_trace.get("tool_calls") or [], start=1):
        if not isinstance(call, dict):
            continue
        status = str(call.get("status") or ("error" if call.get("is_error") else "ok"))
        if call.get("is_error") or status not in {"ok", ""}:
            # A2: an ignored read-only access-policy block is recorded transparently but as
            # status="ignored" (its real status kept in blocked_status) so it is NOT counted
            # in summary.has_failures — same classification the execution axis applies.
            ignored = _is_ignored_readonly_block(str(call.get("tool") or ""), status)
            entry = {
                "kind": "tool_call",
                "index": idx,
                "tool": call.get("tool"),
                "status": "ignored" if ignored else status,
                "exit_code": call.get("exit_code"),
                "signal": call.get("signal"),
                "trace_ref": call.get("trace_ref"),
            }
            if ignored:
                entry["blocked_status"] = status
            entries.append(entry)

    for recovery in execution_axis.get("recoveries") or []:
        if isinstance(recovery, dict):
            entries.append({
                "kind": "tool_recovery",
                "status": "ok",
                "tool": recovery.get("tool"),
                "recovered_status": recovery.get("status"),
                "recovered_by_call_index": recovery.get("recovered_by_call_index"),
            })

    for event in llm_trace.get("verification_events") or []:
        if isinstance(event, dict):
            entries.append({"kind": "runtime_event", **event})

    # FR3: host-attested verify_and_record receipts (injected into the trace by
    # _store_task_result before this build) become first-class ledger entries.
    # The row shape is the FIXED projection in `_outcome_receipts` (a new receipt key
    # is silently dropped unless added there).
    for receipt in llm_trace.get("verification_receipts") or []:
        if isinstance(receipt, dict):
            entries.append(_outcome_receipts.verification_receipt_ledger_row(receipt))

    # Agent-invoked child/self review remains in the raw trace for forensics but
    # is advisory evidence, never an objective or verification authority.
    _review_selection = _outcome_receipts.select_current_review_runs(
        llm_trace.get("review_runs"),
        delivery_candidate=_trace_mapping(llm_trace, "delivery_candidate"),
        review_decision=_trace_mapping(llm_trace, "review_decision"),
    )
    for run in _review_selection.all_runs:
        # Selection keeps sole stale failures fail-closed while old PASS is audit-only.
        status, superseded = _outcome_receipts.review_run_ledger_status(run, _review_selection)
        entries.append({
            "kind": "task_acceptance_review",
            "status": status,
            "aggregate_signal": run.get("aggregate_signal"),
            "degraded": run.get("degraded"),
            "superseded": superseded,
            "finding_count": len(run.get("parsed_findings") or []),
        })

    artifact_status = str(artifact_bundle.get("status") or "")
    if artifact_status in {ARTIFACT_STATUS_FAILED, ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING, "missing"}:
        entries.append({
            "kind": "artifact_bundle",
            "status": artifact_status,
            "errors": artifact_bundle.get("errors") or [],
        })

    review = review_evidence or {}
    for key in ("critical_findings", "advisory_findings", "open_obligations"):
        items = review.get(key)
        if isinstance(items, list) and items:
            status = "failed" if key in {"critical_findings", "open_obligations"} else "partial"
            entries.append({
                "kind": "review",
                "category": key,
                "status": status,
                "count": len(items),
                "items": items[:10],
                "omitted": max(0, len(items) - 10),
            })

    return {
        "schema_version": 2,
        "created_at": utc_now_iso(),
        "task_id": str(task.get("id") or task.get("task_id") or ""),
        "task_contract": task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {},
        "outcome_axes": axes,
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "has_failures": any(
                str(item.get("status") or "").lower() not in _LEDGER_NON_FAILURE_STATUSES
                and not (str(item.get("kind") or "") == "task_contract" and str(item.get("status") or "").lower() in {"draft", "recorded"})
                for item in entries
                if isinstance(item, dict)
            ),
        },
    }


def maybe_write_verification_artifact(
    drive_root: pathlib.Path,
    task_id: str,
    ledger: Dict[str, Any],
    *,
    threshold_chars: int = 12_000,
) -> Dict[str, Any]:
    """Inline small ledgers; write large ledgers as task artifacts."""

    raw = json.dumps(ledger, ensure_ascii=False, sort_keys=True, default=str)
    if len(raw) <= threshold_chars:
        return {"inline": ledger, "artifact": None}
    safe_task = validate_task_id(task_id)
    artifact_dir = pathlib.Path(drive_root) / "task_results" / "artifacts" / safe_task
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "verification_ledger.json"
    atomic_write_json(path, ledger, trailing_newline=True)
    data = path.read_bytes()
    return {
        "inline": {
            "schema_version": 1,
            "created_at": ledger.get("created_at"),
            "task_id": ledger.get("task_id"),
            "summary": ledger.get("summary") or {},
            "omitted_to_artifact": True,
        },
        "artifact": {
            "kind": "verification_ledger",
            "name": "verification_ledger.json",
            "path": str(path),
            "size": len(data),
            "sha256": sha256(data).hexdigest(),
            "status": ARTIFACT_STATUS_READY,
            "errors": [],
        },
    }
