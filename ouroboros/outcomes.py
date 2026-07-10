"""Typed task/loop outcome helpers.

Lifecycle, execution health, artifacts, review, and objective evaluation are
separate axes.  Objective success is never inferred from final text or the
absence of tool errors; it is filled only by LLM-first task acceptance review or
remains ``not_evaluated``.
"""

from __future__ import annotations

import json
import pathlib
from hashlib import sha256
from typing import Any, Dict, List, Optional

from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_READY,
)
from ouroboros.task_results import STATUS_CANCEL_REQUESTED, STATUS_REJECTED_DUPLICATE, validate_task_id
from ouroboros.utils import atomic_write_json, utc_now_iso


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
BEST_EFFORT_REASON_CODES = frozenset({
    "budget_exhausted",
    "round_limit",
    "finalization_grace",
    "deadline_local",
    # provider-death terminalization (WA2): a genuinely-extracted final answer
    # after the same-model reroute + fallback exhausted must land as best_effort,
    # not a flat failure — the same honest-shelf semantics as deadline/budget.
    "provider_unavailable",
    "children_unabsorbed",
})

# Typed final-answer protocol marker (machine-readable deliverable payload,
# separate from reasoning prose). The agent is instructed in SYSTEM.md to end
# short-deliverable answers with this exact line.
FINAL_ANSWER_MARKER = "FINAL ANSWER:"

OUTCOME_TIER_SOLVED = "solved"
OUTCOME_TIER_BEST_EFFORT = "best_effort"
OUTCOME_TIER_BLOCKED = "blocked_with_evidence"
_OUTCOME_TIERS = (OUTCOME_TIER_SOLVED, OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED)

REASON_FINAL_MESSAGE = "final_message"
REASON_EMPTY_FINAL_TEXT = "empty_final_text"
REASON_PROVIDER_FAILURE = "provider_failure"
REASON_ARTIFACT_FAILED = "artifact_failed"
REASON_ARTIFACT_PENDING = "artifact_pending"
REASON_TASK_EXCEPTION = "task_exception"
REASON_DEEP_SELF_REVIEW_UNAVAILABLE = "deep_self_review_unavailable"
REASON_DEEP_SELF_REVIEW_ERROR = "deep_self_review_error"
REASON_TOOL_FAILURE = "tool_failure"

_BLOCKING_TOOL_STATUSES = frozenset({
    "artifact_output_error",
    "blocked",
    "claude_code_error",
    "cwd_blocked",
    "data_blocked",
    "edit_text_blocked",
    "elevation_blocked",
    "error",
    "git_via_shell_blocked",
    "heal_mode_blocked",
    "install_error",
    "integration_blocked",
    "light_mode_blocked",
    "non_zero_exit",
    "protected_blocked",
    "resource_constraint_blocked",
    "resource_policy_blocked",
    "run_script_blocked",
    "safety_violation",
    "shell_error",
    "skill_payload_blocked",
    "skill_payload_control_blocked",
    "skill_state_blocked",
    "timeout",
    "unavailable",
    "violation",
    "workspace_blocked",
    "write_file_blocked",
    "root_required_user_files",
    "root_required_active_workspace",
})
_RECOVERY_TOOL_NAMES = frozenset({
    "claude_code_edit",
    "edit_text",
    "run_command",
    "run_script",
    "start_service",
    "stop_service",
    "write_file",
})
# v6.57.0 — POLICY-denial statuses: the runtime/policy said "no" to an action
# (a `*_blocked` refusal, on ANY tool incl. writes/shell). This is telemetry, NOT
# an execution-health failure: the agent either found another way (its work is
# judged on the objective/review axis) or was honestly blocked. Distinct from the
# read-only `_NON_BLOCKING_READONLY_BLOCK_STATUSES` (which already demotes resource
# blocks on read-only tools) — this covers the WRITE/shell/integration blocks that
# previously forced execution=degraded + a false `tool_failure` headline even when
# the deliverable succeeded (the site-presentation incident: integration_blocked +
# LIST_FILES policy → degraded/tool_failure over a shipped site). A structural
# status partition (Bible P5 — never content matching). Genuine tool/exec failures
# (`error`, `*_error`, `non_zero_exit`, `shell_error`, `timeout`, `unavailable`) and
# security-boundary hits (`safety_violation`, `violation`) are intentionally EXCLUDED
# and stay real failures.
_POLICY_DENIAL_STATUSES = frozenset({
    "blocked",
    "cwd_blocked",
    "data_blocked",
    "edit_text_blocked",
    "elevation_blocked",
    "git_via_shell_blocked",
    "heal_mode_blocked",
    "integration_blocked",
    "light_mode_blocked",
    "protected_blocked",
    "resource_constraint_blocked",
    "resource_policy_blocked",
    "run_script_blocked",
    "skill_payload_blocked",
    "skill_payload_control_blocked",
    "skill_state_blocked",
    "workspace_blocked",
    "write_file_blocked",
})
# T4 (v6.35.0): an unrecovered run_command/run_script non-zero exit / shell
# error — e.g. an X11-teardown `exit=1` after "138 passed", or an abandoned
# `find` probe on a nonexistent path — is cosmetic, not a degraded execution.
# NOTE: `non_zero_exit`/`shell_error` ARE in _BLOCKING_TOOL_STATUSES; this branch
# DELIBERATELY demotes them to a non-degrading "cosmetic" bucket (still recorded
# on the execution axis for monitoring) because the owner accepted that an
# ignored shell failure belongs on the LLM-review/objective axis, not the
# execution axis. `timeout` is intentionally EXCLUDED — a stuck/aborted command
# is a real failure. Structural status/tool-name partition, never content
# matching (Bible P5).
_NON_BLOCKING_RECOVERABLE_STATUSES = frozenset({"non_zero_exit", "shell_error"})
_COSMETIC_TOOL_NAMES = frozenset({"run_command", "run_script"})
# A2: an UNRECOVERED access-policy block (resource_policy_blocked /
# resource_constraint_blocked) on a READ-ONLY exploratory tool — e.g. a
# read_file/search_code/query_code refused by the resource policy — is honest
# telemetry, not a degraded execution: the agent simply could not look there.
# DISTINCT from _NON_BLOCKING_RECOVERABLE_STATUSES / _COSMETIC_TOOL_NAMES so this
# never demotes a run_command resource block. Routed to a FULLY-IGNORED bucket (not
# cosmetic) so it raises no WARN_RESIDUAL_TOOL_ERRORS_WITHOUT_REVIEW — the goal is
# honest telemetry, not a new visible warning. The read-only tool whitelist reuses
# the capability SSOT (READ_ONLY_PARALLEL_TOOLS). Write/edit/data/protected/
# light_mode/integration blocks are intentionally NOT demoted here.
_NON_BLOCKING_READONLY_BLOCK_STATUSES = frozenset({"resource_policy_blocked", "resource_constraint_blocked"})
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

# A failed verification receipt (``status=="fail"``) is "reconciled" ONLY by a LATER
# genuine grounding receipt — a passing run-kind check (``pass``) or an observed artifact
# (``observed``). A later ``declared`` (the no_visible_machine_contract escape hatch) does
# NOT reconcile a red: that would let an agent see red, then declare-away the finalization
# nudge. So this is deliberately NARROWER than _RECEIPT_GROUNDING_STATUSES (no ``declared``).
_RECEIPT_RED_RECONCILING_STATUSES = frozenset({"pass", "observed"})

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
)


def _is_ignored_readonly_block(tool: str, status: str) -> bool:
    """A2 (v6.50.2) SSOT predicate: an access-policy block (resource_policy_blocked /
    resource_constraint_blocked) on a READ-ONLY exploratory tool is honest telemetry, not a
    degraded execution NOR a verification-ledger failure — the agent simply could not look
    there. Shared by ``_classify_tool_errors`` (execution axis) and ``build_verification_ledger``
    (has_failures) so both axes classify it identically. Non-read-only/effect tools (e.g. a
    run_command resource block) are NOT matched and stay real failures."""
    from ouroboros.tool_capabilities import READ_ONLY_PARALLEL_TOOLS

    return status in _NON_BLOCKING_READONLY_BLOCK_STATUSES and tool in READ_ONLY_PARALLEL_TOOLS


def _clip(text: Any, cap: int) -> str:
    """Bound a string for a ledger INDEX projection with a DISCLOSED marker (BIBLE P1 —
    never silent). The full content stays durable elsewhere (the receipt store / blobs)."""
    t = str(text or "")
    return t if len(t) <= cap else t[:cap] + f"…[+{len(t) - cap} chars]"


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


def verification_receipts_path(drive_root: Any, task_id: str, *, create: bool = False) -> pathlib.Path:
    """Durable per-task receipt store, a sibling of the verification-ledger artifact
    under the canonical task-artifacts dir (``validate_task_id``-guarded)."""
    safe = validate_task_id(task_id)
    artifact_dir = pathlib.Path(drive_root) / "task_results" / "artifacts" / safe
    if create:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / "verification_receipts.jsonl"


def append_verification_receipt(drive_root: Any, task_id: str, receipt: Dict[str, Any]) -> None:
    """Append a host-attested verification receipt for a task. Advisory: a write
    failure never breaks the tool or task (receipts only shape a transparency flag)."""
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(verification_receipts_path(drive_root, task_id, create=True), receipt)
    except Exception:
        pass


def read_verification_receipts(drive_root: Any, task_id: str) -> List[Dict[str, Any]]:
    try:
        path = verification_receipts_path(drive_root, task_id, create=False)
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out
    except Exception:
        return []


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


def verification_grounding_present(llm_trace: Dict[str, Any], drive_root: Any, task_id: str) -> bool:
    """True when the turn already carries host-attested grounding — a verify_and_record
    receipt with a grounding status, or a trivial write/edit deliverable. Read-only
    (shared by the one-shot nudge gate and the receipt_absent flag)."""
    receipts = read_verification_receipts(drive_root, task_id)
    if any(str(r.get("status") or "") in _RECEIPT_GROUNDING_STATUSES for r in receipts):
        return True
    return _trace_has_write_edit_grounding(llm_trace)


def should_nudge_verification(llm_trace: Dict[str, Any], drive_root: Any, task_id: str) -> bool:
    """FR3 one-shot nudge gate: the turn produced real reviewable effects but recorded
    NO host-attested grounding yet — ping the agent ONCE to verify_and_record before it
    finalizes. Binary; the caller latches it so it fires at most once per task."""
    if not turn_has_reviewable_effects(llm_trace):
        return False
    return not verification_grounding_present(llm_trace, drive_root, task_id)


def latest_unreconciled_failed_receipt(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure core: the most recent RED receipt (``status=="fail"``) with NO later genuine
    grounding receipt (a passing run-kind check or an observed artifact — see
    ``_RECEIPT_RED_RECONCILING_STATUSES``; a later ``declared`` does NOT reconcile). Returns
    the failing receipt, or ``None``. Structural — typed receipt status only, NO content
    matching (Bible P5). Shared SSOT by the finalize nudge and the acceptance
    verification_summary so the reconciliation rule lives in one place."""
    latest_fail: Optional[Dict[str, Any]] = None
    reconciled = False
    for r in receipts:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status") or "")
        if status == "fail":
            latest_fail, reconciled = r, False
        elif latest_fail is not None and status in _RECEIPT_RED_RECONCILING_STATUSES:
            reconciled = True
    return None if (latest_fail is None or reconciled) else latest_fail


def latest_unreconciled_failed_verification(drive_root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """Disk-backed wrapper of ``latest_unreconciled_failed_receipt`` — reads the task's
    durable receipts. Feeds the one-shot red-verification finalization nudge: finalizing over
    your own host-attested red is a self-contradiction (Bible P3/P12), distinct from the
    receipt_absent case."""
    return latest_unreconciled_failed_receipt(read_verification_receipts(drive_root, task_id))


def latest_unreconciled_masked_pass(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure core (v6.52.2): the most recent PASS receipt whose check can MASK the real exit code
    (``check_exit_masking`` flag from the verify sensor — e.g. ``... | tail``, ``|| true``), with
    NO later CLEAN (non-masked) grounding receipt (a pass/observed whose check is not masked).
    Returns the masked passing receipt, or ``None``. A masked PASS is 'reconciled' by a later
    genuine clean grounding. FLAG-driven (typed receipt field), never content matching (Bible P5);
    advisory only. Shared SSOT by the finalize nudge and the acceptance verification_summary."""
    latest_masked: Optional[Dict[str, Any]] = None
    reconciled = False
    for r in receipts:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status") or "")
        masked = bool(r.get("check_exit_masking"))
        if status == "pass" and masked:
            latest_masked, reconciled = r, False
        elif latest_masked is not None and status in _RECEIPT_RED_RECONCILING_STATUSES and not masked:
            reconciled = True
    return None if (latest_masked is None or reconciled) else latest_masked


def latest_unreconciled_masked_verification(drive_root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """Disk-backed wrapper of ``latest_unreconciled_masked_pass`` — feeds the one-shot ADVISORY
    masked-check finalization nudge (the agent may still finalize). Distinct from the red nudge:
    that fires on a RED check; this fires on a green check whose exit code may be laundered."""
    return latest_unreconciled_masked_pass(read_verification_receipts(drive_root, task_id))


def latest_agent_defined_verification(drive_root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """Newest verify receipt whose criterion was AGENT-DEFINED without a stated basis
    (v6.54.4) — feeds the one-shot advisory criterion-provenance nudge: the check
    passed, but the success criterion was synthesized by the agent, so the agent is
    asked once to confirm it is equivalent to what the task actually requires."""
    receipts = read_verification_receipts(drive_root, task_id)
    for receipt in reversed(receipts):
        if not isinstance(receipt, dict):
            continue
        if str(receipt.get("status") or "") not in ("pass", "observed"):
            continue
        # The LATEST passing receipt decides: a later task_stated check or a
        # later agent_defined check WITH a stated basis reconciles the concern.
        if str(receipt.get("criterion_source") or "") != "agent_defined":
            return None
        if str(receipt.get("criterion_basis") or "").strip():
            return None
        return receipt
    return None


def apply_receipt_absent_flag(
    loop_outcome: Dict[str, Any], llm_trace: Dict[str, Any], drive_root: Any, task_id: str, *, expected_output: str = ""
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
    receipts = read_verification_receipts(drive_root, task_id)
    if receipts:
        llm_trace["verification_receipts"] = receipts
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
    if verification_grounding_present(llm_trace, drive_root, task_id):
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

# Tools/roots whose successful use means the turn produced reviewable work.
# Root-aware write tools: these take a `root` arg, so the scratch-exclusion rule
# applies directly. claude_code_edit uses `cwd` (not `root`) and resolves its own
# work dir, so it is NOT root-checked here; its deliverables surface via the
# artifact_registered flag (declared outputs), and workspace/headless claude_code_edit
# is review-eligible anyway because such tasks are not direct chat.
_ROOT_WRITE_TOOLS = frozenset({"write_file", "edit_text"})
_EFFECT_COMMIT_TOOLS = frozenset({"commit_reviewed", "vcs_commit_reviewed"})
# Exclusion model: only pure scratch is exempt. Every other root is a real surface
# (deliverable, workspace, repo, skill payload, or a light-mode skill write via
# runtime_data). Excluding by scratch — not enumerating "deliverable" roots —
# keeps the immune gate complete as roots evolve and errs toward reviewing work.
_SCRATCH_ROOTS = frozenset({"task_drive"})
_OK_TOOL_STATUSES = frozenset({"", "ok", "ok_autocorrected"})
# Process/service tools that produce a registered deliverable when given outputs=[...].
_EFFECT_PROCESS_TOOLS = frozenset({"run_command", "run_script", "start_service"})
# Substantial coding tool (cwd-based, no root arg): any successful run is real
# work. Over-counting a rare scratch edit is the safe direction for an immune
# gate; under-counting a real repo/deliverable edit (no outputs=[...]) is not.
_EFFECT_CODING_TOOLS = frozenset({"claude_code_edit"})
# Parent integration of a child's patch stages a repo mutation -> reviewable work.
_EFFECT_INTEGRATION_TOOLS = frozenset({"integrate_subagent_patch"})


def turn_has_reviewable_effects(llm_trace: Dict[str, Any]) -> bool:
    """True if the turn produced real reviewable work, from a structured trace read.

    Reviewable effects are a successful repo commit; a successful write_file/
    edit_text to any non-scratch root; any successful claude_code_edit (a
    substantial coding tool that uses cwd, not root); a successful
    run_command/run_script/start_service that declared deliverable outputs; or any
    successful tool that registered a canonical artifact (artifact_registered — a
    stopped service's outputs or a user_files write). Pure scratch (root=task_drive)
    write_file/edit_text does NOT count. Cognitive-memory updates go through
    update_identity/update_scratchpad/knowledge_write (not write tools) and are
    intentionally not effects; a light-mode generic cognitive write is
    advisory-redirected and never succeeds here. This is a P3 deterministic immune
    signal over observable runtime facts, never message-content inspection.
    """
    for call in llm_trace.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("is_error"):
            continue
        if str(call.get("status") or "ok") not in _OK_TOOL_STATUSES:
            continue
        tool = str(call.get("tool") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if tool in _EFFECT_COMMIT_TOOLS or tool in _EFFECT_CODING_TOOLS or tool in _EFFECT_INTEGRATION_TOOLS:
            return True
        if tool in _ROOT_WRITE_TOOLS and str(args.get("root") or "active_workspace") not in _SCRATCH_ROOTS:
            return True
        if tool in _EFFECT_PROCESS_TOOLS:
            outputs = args.get("outputs")
            if isinstance(outputs, list) and any(str(item or "").strip() for item in outputs):
                return True
        # Structured flag set from the full (untruncated) tool result at capture time;
        # covers stopped-service outputs and user_files writes regardless of preview length.
        if call.get("artifact_registered"):
            return True
    return False


def _user_file_basenames(args: Dict[str, Any]) -> set[str]:
    """Lowercased file basenames declared in a write call's ``path`` and ``files[]``."""
    candidates = [args.get("path")]
    candidates.extend(
        (entry or {}).get("path") for entry in (args.get("files") or []) if isinstance(entry, dict)
    )
    return {
        pathlib.PurePath(str(candidate or "")).name.lower()
        for candidate in candidates
        if str(candidate or "").strip()
    }


def _tool_error_record(item: Dict[str, Any], *, recovered_by: int | None = None) -> Dict[str, Any]:
    record = {
        "tool": str(item.get("tool") or "unknown"),
        "status": str(item.get("status") or "error"),
        "exit_code": item.get("exit_code"),
        "signal": item.get("signal"),
        "result": str(item.get("result") or "")[:500],
    }
    if recovered_by is not None:
        record["recovered_by_call_index"] = recovered_by
    return record


def _classify_tool_errors(llm_trace: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    calls = [item for item in (llm_trace.get("tool_calls") or []) if isinstance(item, dict)]
    unresolved: List[Dict[str, Any]] = []
    recovered_items: List[Dict[str, Any]] = []
    cosmetic_items: List[Dict[str, Any]] = []
    ignored_items: List[Dict[str, Any]] = []
    policy_denials: List[Dict[str, Any]] = []
    for idx, item in enumerate(calls):
        if not item.get("is_error"):
            continue
        tool = str(item.get("tool") or "unknown")
        status = str(item.get("status") or "error")
        # COGNITIVE_TOOL_REQUIRED is an advisory redirect, not a task failure: the
        # agent is told to use update_identity/update_scratchpad/knowledge_write, but
        # a self-initiated cognitive write through the wrong tool must never fail the
        # task (that was the original "Привет fails" regression). Skip it entirely.
        if status == "cognitive_tool_required":
            continue
        # A2: an access-policy block on a READ-ONLY exploratory tool is honest
        # telemetry, not a degraded execution — fully ignored (recorded for
        # forensics) so it neither sets tool_failure nor raises a residual warning.
        if _is_ignored_readonly_block(tool, status):
            ignored_items.append(_tool_error_record(item))
            continue
        if status not in _BLOCKING_TOOL_STATUSES and tool not in _RECOVERY_TOOL_NAMES:
            continue
        # ROOT_REQUIRED_* redirects name a real misrouted deliverable. Each is
        # recovered ONLY when every blocked file name (path or files[]) is later
        # written via the root the redirect demanded (user_files ↔ active_workspace).
        # These branches are terminal: they never fall through to the generic
        # same-target/artifact_registered recovery, which could otherwise clear
        # them through a write to the wrong root (e.g. a run_command output).
        if status in ("root_required_user_files", "root_required_active_workspace"):
            required_root = (
                "user_files" if status == "root_required_user_files" else "active_workspace"
            )
            blocked_args = item.get("args") if isinstance(item.get("args"), dict) else {}
            blocked_names = _user_file_basenames(blocked_args)
            recovered_names: set[str] = set()
            for later in calls[idx + 1:]:
                if not (isinstance(later, dict) and not later.get("is_error")):
                    continue
                later_args = later.get("args") if isinstance(later.get("args"), dict) else {}
                later_root = str(later_args.get("root") or "")
                # active_workspace is these tools' DEFAULT root: a retry that simply
                # omits root already writes to the demanded place, so it earns the
                # recovery credit too (scope r2 — the explicit-arg-only match left a
                # real recovery marked unresolved → false execution-axis degradation).
                # user_files is never a default and still requires the explicit arg.
                root_matches = later_root == required_root or (
                    required_root == "active_workspace" and not later_root
                )
                if (
                    str(later.get("tool") or "") in _ROOT_WRITE_TOOLS
                    and root_matches
                    and str(later.get("status") or "ok") in _OK_TOOL_STATUSES
                ):
                    recovered_names |= _user_file_basenames(later_args)
            if not (blocked_names and blocked_names <= recovered_names):
                unresolved.append(_tool_error_record(item))
            else:
                recovered_items.append(_tool_error_record(item))
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        target_parts = []
        target_paths = set()
        for key in ("root", "path", "cwd", "cmd", "script", "name", "outputs"):
            if key not in args:
                continue
            value = args.get(key)
            target_parts.append((key, value))
            if key in {"path", "cwd"} and value:
                target_paths.add(str(value))
            if key == "outputs" and isinstance(value, list):
                target_paths.update(str(part) for part in value if str(part or "").strip())
        target_key = json.dumps(target_parts, sort_keys=True, default=str)
        recovered_by: int | None = None
        for later_idx, later in enumerate(calls[idx + 1:], start=idx + 2):
            if later.get("is_error"):
                continue
            later_tool = str(later.get("tool") or "")
            later_status = str(later.get("status") or "ok")
            if later_status not in {"", "ok", "ok_autocorrected"}:
                continue
            later_args = later.get("args") if isinstance(later.get("args"), dict) else {}
            later_parts = []
            later_paths = set()
            for key in ("root", "path", "cwd", "cmd", "script", "name", "outputs"):
                if key not in later_args:
                    continue
                value = later_args.get(key)
                later_parts.append((key, value))
                if key in {"path", "cwd"} and value:
                    later_paths.add(str(value))
                if key == "outputs" and isinstance(value, list):
                    later_paths.update(str(part) for part in value if str(part or "").strip())
            same_target = later_tool == tool and target_key == json.dumps(later_parts, sort_keys=True, default=str)
            same_path = bool(target_paths and later_paths and target_paths.intersection(later_paths))
            # Read the TYPED artifact-registration flag captured from the full result at
            # execution time (loop_tool_execution), not a substring of the (truncatable)
            # trace preview — the same typed signal turn_has_reviewable_effects uses, so
            # the marker is never re-derived from prose on this layer (C9.5).
            artifact_registered = bool(later.get("artifact_registered"))
            if status == "artifact_output_error":
                recovered = artifact_registered and (same_path or not target_paths)
            else:
                recovered = same_target or (artifact_registered and same_path)
            if recovered:
                recovered_by = later_idx
                break
        if recovered_by is not None:
            recovered_items.append(_tool_error_record(item, recovered_by=recovered_by))
            continue
        if status in _NON_BLOCKING_RECOVERABLE_STATUSES and tool in _COSMETIC_TOOL_NAMES:
            # Unrecovered run_command/run_script non-zero exit: cosmetic, not degrading.
            cosmetic_items.append(_tool_error_record(item))
            continue
        if status in _POLICY_DENIAL_STATUSES:
            # v6.57.0 — an unrecovered POLICY refusal (the runtime said "no" to this
            # action) is telemetry, not an execution-health failure. Recorded for
            # forensics; does NOT set execution=degraded nor a `tool_failure` headline.
            policy_denials.append(_tool_error_record(item))
            continue
        unresolved.append(_tool_error_record(item))
    return {
        "unresolved": unresolved,
        "recovered": recovered_items,
        "cosmetic": cosmetic_items,
        "ignored": ignored_items,
        "policy_denials": policy_denials,
    }


def _unresolved_tool_errors(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _classify_tool_errors(llm_trace).get("unresolved") or []


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
    if not tiers:
        return ""
    if OUTCOME_TIER_BLOCKED in tiers:
        return OUTCOME_TIER_BLOCKED
    if OUTCOME_TIER_BEST_EFFORT in tiers:
        return OUTCOME_TIER_BEST_EFFORT
    return OUTCOME_TIER_SOLVED


def _acceptance_decision_projection(acceptance_decision: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "status": str(acceptance_decision.get("status") or ""),
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


def _review_axis(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    review_decision = llm_trace.get("review_decision") if isinstance(llm_trace.get("review_decision"), dict) else {}
    acceptance_decision = llm_trace.get("acceptance_decision") if isinstance(llm_trace.get("acceptance_decision"), dict) else {}
    # A pre-revision acceptance run marked superseded_by_revision is kept in the
    # trace for forensics but must NOT count toward the objective when a REPLACEMENT
    # (non-superseded) review actually landed: the re-reviewed final deliverable's
    # verdict is authoritative, so a stale pre-revision FAIL cannot worst-case-poison
    # a final PASS (the reducer is worst-of-all-runs). BUT if the revision never
    # reached a terminal re-review (provider death, round limit, ...), the superseded
    # run is the SOLE verdict — keep it, never erase a failing verdict (P3 integrity).
    _all_runs = [run for run in (llm_trace.get("review_runs") or []) if isinstance(run, dict)]
    _non_superseded = [run for run in _all_runs if not run.get("superseded_by_revision")]
    runs = _non_superseded if _non_superseded else _all_runs
    if not runs:
        axis = {
            "status": "skipped",
            "eligibility": str(review_decision.get("eligibility") or "not_eligible"),
            "trigger": str(review_decision.get("trigger") or "not_evaluated"),
            "run_count": 0,
        }
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
    if include_outcome_axes:
        public["outcome_axes"] = normalize_outcome_axes(result)
    return public


def derive_loop_outcome(final_text: str, usage: Dict[str, Any], llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return a typed LoopOutcome-compatible dict."""

    usage_status = str(usage.get("execution_status") or usage.get("result_status") or "").strip()
    usage_reason = str(usage.get("reason_code") or "").strip()
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
    elif text.lstrip().startswith("⚠️ Failed to get a response") or text.lstrip().startswith("⚠️ All models are down"):
        execution_status = EXECUTION_INFRA_FAILED
        reason_code = usage_reason or REASON_PROVIDER_FAILURE
        failure = {"kind": "provider", "reason_code": reason_code}
    elif text.lstrip().startswith("⚠️ Error during processing:"):
        execution_status = EXECUTION_INFRA_FAILED
        reason_code = usage_reason or REASON_TASK_EXCEPTION
        failure = {"kind": "runtime", "reason_code": reason_code}
    elif text.lstrip().startswith("❌ Deep self-review unavailable:"):
        execution_status = EXECUTION_INFRA_FAILED
        reason_code = usage_reason or REASON_DEEP_SELF_REVIEW_UNAVAILABLE
        failure = {"kind": "runtime", "reason_code": reason_code}
    elif text.lstrip().startswith("⚠️ Deep self-review error:") or text.lstrip().startswith("❌ Deep self-review failed:"):
        execution_status = EXECUTION_INFRA_FAILED
        reason_code = usage_reason or REASON_DEEP_SELF_REVIEW_ERROR
        failure = {"kind": "runtime", "reason_code": reason_code}
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

    review = _review_axis(llm_trace)
    objective = _objective_axis(review)
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
            "recoveries": recovered_tool_errors[:20],
            "cosmetic_tool_errors": cosmetic_tool_errors[:20],
            "ignored_tool_errors": ignored_tool_errors[:20],
            "policy_denials": policy_denials[:20],
        },
        "artifacts": {"status": "not_applicable"},
        "objective": objective,
        "review": review,
    }
    return {
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
        "failure": headline_failure,
        "recoveries": recovered_tool_errors[:20],
        "usage": {
            "cost_usd": round(float(usage.get("cost") or 0), 6),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_rounds": int(usage.get("rounds") or 0),
        },
        "trace_refs": collect_trace_refs(usage, llm_trace),
    }


def collect_trace_refs(usage: Dict[str, Any], llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    execution_id = str(usage.get("execution_id") or "").strip()
    if execution_id:
        refs["execution_id"] = execution_id
    llm_refs = []
    for item in usage.get("llm_call_refs") or []:
        if not isinstance(item, dict):
            continue
        llm_refs.append({
            "llm_call_id": item.get("llm_call_id"),
            "execution_id": item.get("execution_id"),
            "round_id": item.get("round_id"),
            "round": item.get("round"),
            "request_ref": item.get("request_ref"),
            "response_ref": item.get("response_ref"),
            "model": item.get("model"),
            "resolved_model": item.get("resolved_model"),
            "provider": item.get("provider"),
        })
    if llm_refs:
        refs["llm_call_refs"] = llm_refs
    tool_refs = []
    for item in llm_trace.get("tool_calls") or []:
        if isinstance(item, dict) and item.get("trace_ref"):
            trace = item.get("trace_ref") if isinstance(item.get("trace_ref"), dict) else {}
            tool_refs.append({
                "call_id": trace.get("call_id"),
                "manifest_ref": trace.get("manifest_ref"),
                "redacted_projection_ref": trace.get("redacted_projection_ref"),
                "redaction": trace.get("redaction"),
            })
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
    for receipt in llm_trace.get("verification_receipts") or []:
        if isinstance(receipt, dict):
            entries.append({
                "kind": "verification_receipt",
                "status": str(receipt.get("status") or "unknown"),
                "contract_kind": str(receipt.get("contract_kind") or ""),
                "criterion_id": str(receipt.get("criterion_id") or ""),
                "check": _clip(receipt.get("check"), 300),
                "expected": _clip(receipt.get("expected"), 200),
                # Verification SEMANTICS so a reviewer sees how `expected` was matched
                # (substring-only is weak evidence for a metric-graded task).
                "expected_match": str(receipt.get("expected_match") or "substring"),
                "matched": receipt.get("matched"),
                "returncode": receipt.get("returncode"),
                "summary": _clip(receipt.get("summary"), 300),
                # C: after-only artifact-lifecycle flag (a check that built then deleted a
                # declared deliverable). The receipt entry is a FIXED projection — a new
                # receipt key is silently dropped unless added here. Bounded for ledger size.
                "artifact_lifecycle": (receipt.get("artifact_lifecycle") or [])[:50],
                "artifacts_missing_after": (receipt.get("artifacts_missing_after") or [])[:50],
                # v6.52.2: exit-masking sensor flag — the check's shell pipeline can launder the
                # real exit code (e.g. `... | tail`, `|| true`). FLAG-ONLY (status unchanged).
                "check_exit_masking": bool(receipt.get("check_exit_masking")),
                "check_exit_masking_reasons": (receipt.get("check_exit_masking_reasons") or [])[:10],
                # v6.54.4 criterion provenance (flag-only): task_stated | agent_defined.
                "criterion_source": str(receipt.get("criterion_source") or ""),
                "criterion_basis": _clip(receipt.get("criterion_basis"), 500),
            })

    _accept_runs = [r for r in (llm_trace.get("review_runs") or []) if isinstance(r, dict)]
    _has_replacement = any(not r.get("superseded_by_revision") for r in _accept_runs)
    for run in _accept_runs:
        # A superseded pre-revision run is only forensic (status 'superseded') when a
        # REPLACEMENT review landed; with no replacement it is the sole verdict and
        # must still read as its real failed/ok status (never hide a failing verdict).
        superseded = bool(run.get("superseded_by_revision")) and _has_replacement
        failed = run.get("aggregate_signal") in {"FAIL", "DEGRADED"} or bool(run.get("degraded"))
        entries.append({
            "kind": "task_acceptance_review",
            "status": "superseded" if superseded else ("failed" if failed else "ok"),
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
