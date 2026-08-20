"""Tool-call trace vocabulary and the execution-axis error classifier.

Extracted from ``outcomes.py`` (v6.90.x submarine unwind) so the typed-outcome
module stays under the hard module gate.  This module is a LEAF: it holds the
status/tool frozensets that partition a tool call into blocking / policy-denial /
cosmetic / ignored buckets, plus the classifier that walks a trace and applies
them.  It must never import ``ouroboros.outcomes`` — ``outcomes`` re-exports
every name here, so historical import sites
(``from ouroboros.outcomes import _classify_tool_errors, _POLICY_DENIAL_STATUSES``
in ``agent_task_pipeline`` and the tests) keep working unchanged.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

_BLOCKING_TOOL_STATUSES = frozenset({
    # T1: the statuses below the blank line were produced but partitioned nowhere,
    # so a call could be an honest error while every bucket of the honest breakdown
    # stayed empty. Each is homed by its nearest existing analogue (owner batch #4).
    "artifact_output_error",
    "artifact_output_undeclared",
    "blocked",
    "cwd_blocked",
    "data_blocked",
    "edit_ops_blocked",
    "edit_text_blocked",
    "elevation_blocked",
    "error",
    "git_via_shell_blocked",
    "heal_mode_blocked",
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
    "skill_state_blocked",
    "timeout",
    "unavailable",
    "user_files_path_blocked",
    "violation",
    "workspace_blocked",
    "write_file_blocked",
    "root_required_user_files",
    "root_required_active_workspace",

    "argument_error",            # nearest analogue: `error` (a malformed call)
    "executor_error",            # nearest analogue: `error` (the executor crashed)
    "extension_error",           # nearest analogue: `error` (the extension raised)
    "git_error",                 # nearest analogue: `error`; is_error stays false, so unreachable here
    "mcp_error",                 # nearest analogue: `error` (the provider reported one)
    "review_blocked",            # nearest analogue: `blocked`; is_error stays false, so unreachable here
    "run_script_error",          # nearest analogue: `error`
    # A tool that RAN and answered `{"ok": false}`: walked like a failure so the
    # recovery credit still applies, then routed to `policy_denials` below rather
    # than `unresolved`. It stays is_error=True for the counters and the anti-loop
    # scan, and it does NOT degrade the execution axis — which is what
    # `outcomes._LEDGER_NON_FAILURE_STATUSES` has declared about the SAME status
    # since v6.83.0 ("the tool ran and answered honestly; a finding, not a
    # failure"). Homing it as blocking-only made the two consumers contradict each
    # other on every unrecovered ext_/mcp_/read `{"ok": false}`.
    "tool_reported_failure",
    "unknown_tool",              # nearest analogue: `error` (the tool does not exist)

    # Retired CODES, surviving STATUS names: `root_required`, `resource_blocked`
    # and `safety_error` are no longer published by any code (the merged parents
    # split, and the safety separator count is gone), but a task result written
    # before this change still carries the string, and this classifier reads those
    # traces back. Names stay; nothing new can produce them.
    "resource_blocked",
    "root_required",
    "safety_error",
})
# Buckets deliberately left OUT of every partition, each with its reason. The
# totality assertion in tests/test_tool_classification_differential.py fails if a
# new code acquires a bucket that appears in neither a partition nor this set, so
# an unclassified outcome can no longer arrive silently.
_UNPARTITIONED_BUCKETS = frozenset({
    # `vlm_error` (image too large / no vision model) is unpartitioned TODAY and
    # T1 preserves that: homing it would newly degrade execution health on two
    # image refusals that currently affect nothing, which no owner decision covers.
    "vlm_error",
})
_RECOVERY_TOOL_NAMES = frozenset({
    "edit_text", "apply_patch", "edit_batch",
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
# (`error`, `*_error`, `non_zero_exit`, `shell_error`, `timeout`) and security-
# boundary hits (`safety_violation`, `violation`) are intentionally EXCLUDED and
# stay real failures. `unavailable` moved buckets in v7 (owner decision, spec
# §1.15): see its entry below.
_POLICY_DENIAL_STATUSES = frozenset({
    # v6.90.x (submarine unwind): the three confinement surfaces that leaked past
    # the partition as generic errors are now typed — the user_files path block on
    # READ tools, and the exit_code=0 undeclared-outputs NUDGE (split from the real
    # artifact_output_error registration failure, which stays a genuine failure).
    "artifact_output_undeclared",
    "blocked",
    "cwd_blocked",
    "data_blocked",
    "edit_ops_blocked",
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
    "resource_blocked",
    "review_blocked",
    "skill_payload_blocked",
    "skill_state_blocked",
    # T1 (owner batch #4, operator homing): a provider that RAN and answered
    # `{"ok": false}` is honest telemetry on the execution axis, matching the
    # ledger's v6.83.0 declaration of the same status. It is NOT a runtime "no",
    # so it is named here by its EFFECT — recorded, never degrading — and the
    # bucket-level assertion in tests/test_tool_classification_differential.py
    # pins that effect rather than the membership.
    "tool_reported_failure",
    # v7 (owner §1.15): a target the runtime cannot serve — a control surface that
    # is off, a task id this tree never registered, a dead extension — is the
    # SUBSTRATE's answer, not the agent's mistake. It stays is_error=True and
    # blocking (the counters and anti-loop scan need it), but it must not degrade
    # execution health. `argument_error` deliberately stays OUT: a malformed call
    # is the agent's own defect and feeds reflection.
    "unavailable",
    "user_files_path_blocked",
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


def _is_ignored_readonly_block(tool: str, status: str) -> bool:
    """A2 (v6.50.2) SSOT predicate: an access-policy block (resource_policy_blocked /
    resource_constraint_blocked) on a READ-ONLY exploratory tool is honest telemetry, not a
    degraded execution NOR a verification-ledger failure — the agent simply could not look
    there. Shared by ``_classify_tool_errors`` (execution axis) and ``build_verification_ledger``
    (has_failures) so both axes classify it identically. Non-read-only/effect tools (e.g. a
    run_command resource block) are NOT matched and stay real failures."""
    from ouroboros.tool_capabilities import READ_ONLY_PARALLEL_TOOLS

    return status in _NON_BLOCKING_READONLY_BLOCK_STATUSES and tool in READ_ONLY_PARALLEL_TOOLS

# Shared trace vocabulary.  ``_ROOT_WRITE_TOOLS`` / ``_OK_TOOL_STATUSES`` have TWO
# consumers — ``reviewable_effect_projection`` in ``outcomes`` and the
# root-required recovery branch in ``_classify_tool_errors`` below — so they live
# here, on the leaf, and ``outcomes`` imports them back rather than each side
# keeping its own copy.
_ROOT_WRITE_TOOLS = frozenset({"write_file", "edit_text", "apply_patch", "edit_batch"})  # patch/batch refuse scratch roots: any success is a reviewable effect
# `untyped` is the ok-status a dynamic provider body carries when nothing typed
# it; leaving it out would disqualify a successful extension call from crediting
# a recovery, which is not what "we could not type it" means.
_OK_TOOL_STATUSES = frozenset({"", "ok", "ok_autocorrected", "untyped"})


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


def _call_target_signature(args: Dict[str, Any]) -> tuple[str, set[str]]:
    """One canonical (target_key, target_paths) signature of a tool call's target args —
    shared by the failed call and the later-recovery scan so the two can never diverge."""
    parts: List[tuple[str, Any]] = []
    paths: set[str] = set()
    for key in ("root", "path", "cwd", "cmd", "script", "name", "outputs"):
        if key not in args:
            continue
        value = args.get(key)
        parts.append((key, value))
        if key in {"path", "cwd"} and value:
            paths.add(str(value))
        if key == "outputs" and isinstance(value, list):
            paths.update(str(part) for part in value if str(part or "").strip())
    return json.dumps(parts, sort_keys=True, default=str), paths


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
            # Kept for traces authored before the redirect stopped carrying an
            # error flag at its source; no live producer reaches this branch.
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
        target_key, target_paths = _call_target_signature(args)
        recovered_by: int | None = None
        for later_idx, later in enumerate(calls[idx + 1:], start=idx + 2):
            if later.get("is_error"):
                continue
            later_tool = str(later.get("tool") or "")
            later_status = str(later.get("status") or "ok")
            if later_status not in _OK_TOOL_STATUSES:
                continue
            later_args = later.get("args") if isinstance(later.get("args"), dict) else {}
            later_key, later_paths = _call_target_signature(later_args)
            same_target = later_tool == tool and target_key == later_key
            same_path = bool(target_paths and later_paths and target_paths.intersection(later_paths))
            # Read the TYPED artifact-registration flag captured from the full result at
            # execution time (loop_tool_execution), not a substring of the (truncatable)
            # trace preview — the same typed signal turn_has_reviewable_effects uses, so
            # the marker is never re-derived from prose on this layer (C9.5).
            artifact_registered = bool(later.get("artifact_registered"))
            if status in ("artifact_output_error", "artifact_output_undeclared"):
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
