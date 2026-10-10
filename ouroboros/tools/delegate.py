"""Nanny tools: run a subagent's cognition on an already-paid subscription session.

A delegated subagent is an ORDINARY Ouroboros subagent acting as a NANNY: it lives in
the task tree with its own deadline and authority, but instead of thinking on metered
API tokens it starts a Claudexor run, watches it, and brings the result home. Because
the nanny IS the host, verification receipts stay host-authored and the harness's
output is a claim, not proof.

Five verbs: ``delegate_start``, a time-bounded ``delegate_wait``,
``delegate_cancel``, ``delegate_answer`` (a run's pending interactive question is
answered by its own nanny — owner decision 7=A, poltergeist phase B) and
``delegate_message`` (a live message into the run's running turn, gated by the
route's declared ``liveInput`` capability, never by a harness name). There is still
no ``hurry`` — Claudexor's control verb is ``cancel``, and cancelling a reviewer
destroys the verdict you wanted.

Read-only and mutating children share ONE nanny and ONE transport. The only difference
is the access profile the HOST derives from the calling task's authority (``readonly``
vs the captured mutating profile) and the run shape that follows from it; there is no second
pipeline and no second slot. The child gets a broker tool, never a shell, so it can ask
the host to run something and lower native access, never widen its task authority.

Custody: the daemon token never leaves ``gateways.claudexor``; nothing here puts it in
a ToolContext, a child environment, or a harness sandbox. WHICH run belongs to WHICH
task is decided by ``ouroboros.delegate_custody`` against the durable event log, not by
a dict this process happens to still hold.
"""

from __future__ import annotations

from ouroboros.owner_pause import run_operation, OwnerPauseRefused

import datetime as _dt
import functools
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, NamedTuple

from ouroboros import delegate_custody as custody
from ouroboros import delegate_progress as progress
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.configured_subagents import SESSION_ACCESS_PROFILES, SESSION_ACCESS_LOWERING
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _replace_tool_result
from ouroboros.subagent_work_order import (  # noqa: F401 - compatibility re-export
    assignment_instructions as _assignment_instructions,
)
from ouroboros.delegate_source_coverage import (
    add_terminal_source_verification,  # noqa: F401  (leaf reads it through _delegate() at call time)
    prepare_work_order_start_binding,
    record_started_custody,
)
from ouroboros.delegate_supervision import delegate_wait_entry as _delegate_wait_entry
from ouroboros.delegate_start_instructions import (
    HOST_INSTRUCTIONS as _HOST_INSTRUCTIONS,
    UNPROVEN_BOUNDARY_INSTRUCTION as _UNPROVEN_BOUNDARY_INSTRUCTION,
    access_instruction,
    append_coordination_context,
    apply_execution_binding,
    directory_copy_binding_instruction,
    execution_binding_fingerprint,
)
from ouroboros.subagent_runtime import (  # noqa: F401 - shared primitive re-export
    delegate_start_entry as _delegate_start_entry,
    exact_start,
)
from ouroboros.subagent_runtime import prepare_delegate_start_actor
from ouroboros.subagent_history import session_request_facts
# The staged-output + read-receipt cluster lives in its own module (size gate);
# re-exported here because sibling code, the tests and the convergence census all
# name it on THIS surface, and `_READ_COVERAGE` must stay the same object.
from ouroboros.delegate_output import (  # noqa: F401
    _ARTIFACT_SUBDIR,
    _BULK_FIELDS,
    _PAYLOAD_ENVELOPE_HEADROOM,
    _PREVIEW_PREFIX_SLACK,
    _PREVIEW_STEPS,
    _READ_COVERAGE,
    _READ_COVERAGE_MAX_KEYS,
    _STRUCTURED_FIELDS,
    _covered_whole,
    _preview_payload,
    _resolve_full_primary_output,
    _safe_run_filename,
    _stage_full_output,
    acknowledge_staged_output_read,
)
# The interactive-question cluster (waiting_on_user + delegate_answer) lives in
# its own module too (size gate); re-exported here because the wait loop, the
# tests and sibling code name it on THIS surface, and `_REPORTED_INTERACTIONS`
# must stay the same object.
from ouroboros.delegate_interactions import (  # noqa: F401
    _ANSWER_NOTES,
    _REPORTED_INTERACTIONS,
    _answer_delivery_unknown,
    _bounded_interactions,
    _delegate_answer,
    _delegate_message,
    _interactions_are_news,
    _normalized_answers,
    _waiting_on_user_payload,
)
# The refusal/emit/ownership helpers live in the neutral leaf
# `ouroboros/delegate_shared.py` (moved to break the facade back-edge:
# delegate_interactions needs them, and an extracted module never imports the
# facade back); re-exported here because sibling code, the tests and
# monkeypatch targets name them on THIS surface.
from ouroboros.deadline_utils import deadline_expired
from ouroboros.delegate_directory import blocked_geometry_refusal
from ouroboros.delegate_registration_policy import resolve_registration
from ouroboros.delegate_shared import (  # noqa: F401
    _emit,
    _fail,
    _owned_run,
    delegate_result,
    publish_delegate_result,
    refusal_host_code,
)
# The C1 integration seam (mutation authority, execution snapshots, retry binding,
# terminal patch capture) lives in its own module (size gate); re-exported here
# (same objects) because sibling code and the tests address it on THIS surface.
# `_fail` is NOT re-imported from it — the one shared refusal author is
# `delegate_shared._fail`, which delegate_integration itself imports.
from ouroboros.delegate_continuation import NO_CONTINUATION, replayed_custody, start_binding
from ouroboros.tools.delegate_integration import (  # noqa: F401
    _CAPTURE_DELEGATED_SNAPSHOT,
    _capture_block,
    _capture_terminal_patch,
    _mutation_authority,
    _payload_mutation_authority,
    _payload_selector_refusal,
    _provision_payload_snapshot,
    _provision_snapshot,
    _resolve_retry_invocation,
    _resolved,
    _retry_binding_refusal,
    _validated_invocation,
    claimed_start_request,
    payload_host_instructions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ouroboros.subagents import DelegatedRunShape, DelegationRoute

log = logging.getLogger(__name__)

_TERMINAL_STATES = custody.TERMINAL_STATES

# The containment verifiers moved to `ouroboros/delegate_containment.py` whole (the
# module-size gate); re-exported here because the nanny's seams and the existing
# tests address them through this module.
from ouroboros.delegate_containment import (  # noqa: E402
    _ACCESS_UNVERIFIED,  # noqa: F401  (re-export: tests address it through this module)
    _Breach,
    _home_isolation_breach,  # noqa: F401  (leaf reads it through _delegate() at call time)
    _widened_access,  # noqa: F401  (leaf reads it through _delegate() at call time)
    home_nested_under_operator_home,  # noqa: F401  (leaf reads it through _delegate() at call time)
)
_POLL_INTERVAL_SEC = 3.0
# Claudexor's own schema bound on maxSeconds (packages/schema/src/control.ts).
_CLAUDEXOR_MAX_SECONDS = 604_800

# The process-local memo of the durable custody rows (the authority lives in the module
# above); re-bound here because sibling code and tests name it on this surface.
_CUSTODY = custody._CUSTODY
_RETRY_HINT = ("to retry THIS start call use delegate_start(prompt=..., "
               "retry_of=pending_invocation_id); a plain call starts a NEW run")


def _host_instructions(authority: "DelegatedRunShape", assignment: str = "",
                       payload_skill: str = "", coordination_context: str = "") -> str:
    """The system-prompt text this run's shape earns. One builder, no dialect.

    ``assignment`` is the host-authored contract block (``_assignment_instructions``);
    appended last so the prohibitions stay the opening statement. A payload run
    (``payload_skill`` non-empty) gets the truthful variant: editing the selected
    skill's user-authored files IS the assignment (gate fix 3).
    """
    text = _HOST_INSTRUCTIONS
    if payload_skill:
        text = payload_host_instructions(text, payload_skill)
    if authority.delegated:
        text += (
            " No filesystem sandbox is requested; scoped HOME selects native state "
            "and credentials, not filesystem confinement. Report actual access "
            "honestly and do not claim to be sandboxed."
            if authority.access == "full" else _UNPROVEN_BOUNDARY_INSTRUCTION
        )
    text += access_instruction(authority.access)  # native powers, not a wider assignment
    if assignment:
        text += "\n\n" + assignment
    return append_coordination_context(text, coordination_context)


def _derive_authority(ctx: ToolContext, access: str = "workspace_write") -> "DelegatedRunShape":
    """Derive the run shape from the task's own authority — one question, asked here.

    Mutating eligibility is host-derived; the model may only lower the captured
    native profile. Ouroboros asks for an access PROFILE and lets
    Claudexor pick the mechanism (fs sandbox, tool allowlist, ...) — no harness branch.

    The SHAPE belongs to ``subagents.delegated_run_shape``. An acting child or an
    ordinary root with a selected external workspace/room holds write authority
    there; ``_mutation_authority`` validates that same physical target. The existing
    snapshot capability decides whether it can execute that mutating assignment.
    """
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tool_access import (
        _TOP_LEVEL_PRINCIPAL_PROFILES, active_tool_profile, project_room_lens_dir,
    )

    profile = active_tool_profile(ctx)
    from ouroboros.consciousness_authority import is_observe_origin

    if is_observe_origin(getattr(ctx, "task_metadata", {})):
        # Observe can use the delegated harness for research, but never turns it
        # into an in-place writer merely because the caller omitted/raised access.
        return delegated_run_shape(False, "readonly")
    mutating = profile in ("acting_subagent", "external_workspace_task") or (
        profile in _TOP_LEVEL_PRINCIPAL_PROFILES and project_room_lens_dir(ctx) is not None
    )
    if mutating:
        from ouroboros.presence_authority import presence_ceiling_allows_delegated_surface

        constraint = getattr(ctx, "task_constraint", None)
        surface = str(getattr(constraint, "surface", "") or "external_workspace")
        mutating = presence_ceiling_allows_delegated_surface(ctx, surface)
    return delegated_run_shape(mutating, access)


def _presence_delegate_read_refusal(ctx: ToolContext) -> Optional[ToolResult]:
    from ouroboros.presence_authority import presence_ceiling_allows_delegated_read

    if presence_ceiling_allows_delegated_read(ctx):
        return None
    return _fail(
        "delegate_start",
        "presence_delegate_read_root_unselected",
        "This Presence profile did not select whole-root read access for the active repository, "
        "so a delegated harness cannot honestly be started inside that broader read surface.",
    )


# -- tools --------------------------------------------------------------------


def _start_request(ctx: ToolContext, route: "DelegationRoute", authority: "DelegatedRunShape",
                   root: str, text: str, seconds: int, instructions: str, execution_root: str = "",
                   *, directory_options: Optional[Dict[str, Any]] = None,
                   continuation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The POST body for one delegated run, built from the derived SHAPE.

    Extracted so the caller stays inside the method-size gate, and so the body has ONE
    author: the shape decides the mode and whether the delegated marker rides along,
    and nothing here re-derives either.

    ``seconds`` and ``instructions`` arrive PRE-BUILT rather than being derived here:
    a transport retry of a pending invocation must present a byte-identical body for
    the engine's replay match, and both the deadline-derived bound and the
    contract-derived instructions can change between calls, so the caller decides
    whether to recompute them or replay the recorded ones (the retry path never calls
    this function at all — it replays the stored canonical body verbatim).
    ``continuation`` carries the engine's ``continueFrom`` (+ ``continueCarrier``) keys.
    """
    target = route.resolved_target()  # ABI-4: one typed read; strings only at the wire
    request: Dict[str, Any] = {
        "prompt": text,
        # Built from the SHAPE plus the task contract, so a mutating delegated
        # child is told that its boundary is a request and not a fact — the same
        # disclosure the durable record and the parent's result carry, in the one
        # place the child can read — and the nanny's own objective rides along
        # structurally (`_assignment_instructions`).
        "instructions": instructions,
        # The engine's default authPreference is `auto` = subscription-first WITH
        # policy fallback to a paid API key. That fallback is invisible to us and
        # would be settled at a confident $0.00 — the one shape the ledger must
        # never produce. Ask for the substrate we are actually claiming.
        "authPreference": "subscription",
        # The run SHAPE comes from the derived authority, not from re-deriving it
        # here: one predicate decides what this child may do, and the mode follows it.
        "mode": authority.mode,
        "scope": {"kind": "project", "root": root},
        # PIN, not preference: `primaryHarness` only fronts the engine's
        # auto-pool, which still holds every other doctor-OK harness — the run
        # could fail over onto a route the owner never configured. The
        # explicit one-element `harnesses` pool is the engine's pinning
        # contract (its own MCP surface spells a forced route exactly this
        # way): the child rides THIS route or the start refuses typed.
        "harnesses": [target.provider_route],
        "primaryHarness": target.provider_route,
        "access": authority.access,
    }
    if authority.isolation:
        # `delegated` rides WITH the isolation, from the same record, because they
        # are the same decision: `live` is in-place, and in place is exactly where
        # Claudexor would otherwise hand the harness the operator's real `$HOME`
        # — daemon control token included. Sending one without the other is the
        # containment hole, so neither is assembled separately.
        request["execution"] = {"isolation": authority.isolation, "delegated": authority.delegated,
                                **({"workspaceRoot": execution_root} if execution_root else {})}
    if directory_options:
        request.setdefault("execution", {}).update(directory_options)
    # credentialProfileId is the account pin (D-U5), reviewer-slot wire contract; strict
    # (D-U6). In the stored canonical body, so a retry_of replay stays byte-identical. A
    # level the model slug encodes is the route's identity: no different effort rides beside it.
    from ouroboros.route_spec import harness_model_named_effort

    effort = harness_model_named_effort(target.provider_route, target.model_id) or target.effort
    for key, value in (("model", target.model_id), ("effort", effort), ("credentialProfileId", target.credential_ref)):
        if value:
            request[key] = value
    if seconds:
        request["maxSeconds"] = seconds
    request.update(continuation or {})
    return request


def _processing_start_request(request, actor, gateway, route):
    """Carry only captured actor intent on a route advertising the wire field."""
    preference = str(actor.get("processing_preference") or "")
    if not preference:
        return request, {"requested": ""} if "processing_preference" in actor else {}
    try:
        catalog = gateway.agent_capabilities()
        row = next((item for item in catalog.get("harnesses", [])
                    if isinstance(item, dict) and item.get("id") == route.route_id), {})
    except Exception:
        row = {}
    if preference in (row.get("processingPreferences") or []):
        request["processingPreference"] = preference
    return request, {"requested": preference, "submitted": request.get("processingPreference"),
                     "observed": "unknown", "source": "host_request",
                     "reason": "submitted" if "processingPreference" in request else "processing_not_submitted"}


def _start_argument_refusal(ctx: ToolContext, text: str, selector_root: str, retry_of: Any,
                            bucket: Any, skill_name: Any, continue_from: Any,
                            continue_carrier: Any = None) -> Tuple[str, Optional[ToolResult]]:
    """``(continuation_token, refusal)``: every refusal a start's ARGUMENTS earn before
    the daemon is touched, in their historical order. Each is a definite no-run: an
    empty prompt (a continuation may carry none: the engine resumes the stopped work
    and the host states the facts), a malformed exact-resource selector, a deadline
    already behind the nanny (``definitely_unrun`` = the producer's own no-run
    verdict, P2), and the continuation selector shapes one call cannot combine: a
    retry replays an old key byte-identically while a continuation is a NEW
    intention over a settled run, and a skill-payload selector run keeps its own
    target semantics."""
    if not text.strip() and not str(continue_from or "").strip() and not str(retry_of or "").strip():
        return "", _fail("delegate_start", "empty_prompt", "prompt is required")
    if continue_carrier is not None and (not str(continue_from or "").strip()
                                         or continue_carrier not in ("auto", "packet")):
        return "", _fail("delegate_start", "continuation_carrier_invalid",
                         "continue_carrier is 'auto' or 'packet' and applies only with continue_from.",
                         definitely_unrun=True)
    refusal = _payload_selector_refusal(selector_root, retry_of, bucket, skill_name)
    if refusal:
        return "", refusal
    if deadline_expired(ctx):
        return "", _fail(
            "delegate_start", "task_deadline_expired",
            "This task's deadline has already passed, so a delegated run started now "
            "would outlive it by design. Finalize with what you have — do not start "
            "new work a deadline has already closed.", definitely_unrun=True,
        )
    token = str(continue_from or "").strip()
    if token and str(retry_of or "").strip():
        return token, _fail("delegate_start", "continuation_selector_conflict",
                            "continue_from starts a NEW run bound to a settled predecessor; retry_of replays a "
                            "pending invocation. Supply one of them.", definitely_unrun=True)
    if token and str(selector_root or "").strip():
        return token, _fail("delegate_start", "continuation_resource_conflict",
                            "continue_from applies to ordinary workspace delegation only; a skill-payload "
                            "selector run is started plain.", definitely_unrun=True)
    return token, None


def _delegate_start(ctx: ToolContext, prompt: str, max_seconds: Optional[int] = None,
                    retry_of: Optional[str] = None, root: Optional[str] = None,
                    bucket: Optional[str] = None, skill_name: Optional[str] = None,
                    directory_strategy: Optional[str] = None, scope_paths: Optional[list] = None,
                    continue_from: Optional[str] = None, continue_carrier: Optional[str] = None,
                    _resolved_binding: Any = None,
                    _canonical_work_order_fingerprint: str = "",
                    _work_order_source_request: Any = None,
                    _coordination_context: str = "") -> ToolResult:
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import delegated_execution_workspace_root, resolve_subagent_executor, route_health

    text = str(prompt or "")
    selector_root = str(root or "").strip()
    continuation_token, argument_refusal = _start_argument_refusal(
        ctx, text, selector_root, retry_of, bucket, skill_name, continue_from, continue_carrier)
    if argument_refusal:
        # This argument boundary precedes daemon access, provisioning and any start.
        return _replace_tool_result(argument_refusal, meta_updates={"operation_outcome": "completed_no_effect"})
    seconds_basis = ""
    if not str(retry_of or "").strip():
        # Decided BEFORE the daemon is touched: a spent lifetime or a sub-second
        # deadline is a definite no-run, and the basis rides both custody rows.
        bound = bounded_max_seconds(ctx, max_seconds)
        if bound.refusal_code:
            return _fail("delegate_start", bound.refusal_code, bound.refusal_detail, definitely_unrun=True)
        seconds_basis = bound.basis

    drive = custody.custody_root(ctx)
    owned_project_id, project_persistent = "", False
    invocation_id = snapshot_id = baseline_sha = target_root = authority_source = ""
    binding_fingerprint = ""
    processing_info: Dict[str, Any] = {}
    resource_ref, directory_options, continuation = {}, {}, NO_CONTINUATION
    retry_token = str(retry_of or "").strip()
    source_binding = prepare_work_order_start_binding(
        ctx, drive, retry_token, _canonical_work_order_fingerprint, text,
        _work_order_source_request,
    )
    work_order_source_request = source_binding["request"]
    work_order_fingerprint = source_binding["fingerprint"]
    recovering = source_binding["recovering"]
    actor, actor_refusal = prepare_delegate_start_actor(
        ctx, drive, recovering=recovering, invocation_id=retry_token, continuing=continuation_token,
        work_order_fingerprint=work_order_fingerprint, authority_fingerprint=source_binding["authority_fingerprint"],
    )
    if actor_refusal:
        return actor_refusal
    actor_facts = {key: str(actor.get(key) or "") for key in (
        "selected_subagent_id", "config_fingerprint", "work_order_fingerprint", "authority_fingerprint")}
    actor_facts.update(work_order_coverage=source_binding["coverage"], work_order_source_request=work_order_source_request)
    if recovering:
        binding, refusal = _resolve_retry_invocation(ctx, drive, retry_token, text)
        if refusal:
            return refusal
        (request_body, route, authority, root, key, project_id, owned_project_id,
         project_persistent, seconds, snapshot_id, target_root, baseline_sha,
         authority_source, resource_ref, processing_info, binding_fingerprint) = binding
        invocation_id = retry_token
        # A replay presents the recorded body byte-identically, so its cap
        # basis is the recorded one too — never re-derived from today's clocks.
        recorded = custody.invocation_record(drive, retry_token) or {}
        seconds_basis, continuation = str(recorded.get("max_seconds_basis") or ""), replayed_custody(recorded)
        if directory_strategy is not None or scope_paths is not None:
            return _fail("delegate_start", "retry_selector_conflict",
                         "A retry replays its recorded directory strategy and scope; omit new geometry arguments.")
    else:
        route = actor["route"]
        if selector_root:
            authority, payload_auth, payload_error = _payload_mutation_authority(
                ctx, drive, bucket, skill_name, _resolved_binding, actor.get("access", "workspace_write"))
            if payload_error:
                return payload_error
        else:
            authority = _derive_authority(ctx, actor.get("access", "workspace_write"))
            payload_auth = None
            if refusal := _presence_delegate_read_refusal(ctx):
                return refusal

    if refusal := blocked_geometry_refusal(ctx, authority, selector_root, directory_strategy, scope_paths):
        return refusal
    if not recovering:
        assignment = "" if bool(actor.get("compiled_work_order")) else _assignment_instructions(ctx)
        payload_skill = str(((payload_auth or {}).get("resource_ref") or {}).get("skill_name") or "")
        instructions = _host_instructions(
            authority, assignment, payload_skill=payload_skill, coordination_context="" if continuation_token else _coordination_context,
        )

    access = authority.access
    requested = recovering  # Recovery may already own a physical run.
    try:
        gateway = ensure_owned_gateway()
    except ClaudexorUnavailable as exc:
        resolution = resolve_subagent_executor("harness", route=route, unavailable_reason=exc.code)
        return _fail("delegate_start", exc.code, str(exc), executor=resolution.executor)

    history_facts = session_request_facts(
        request_body if recovering else {"model": route.model, "credentialProfileId": route.profile_id,
                                        "effort": route.effort, "access": authority.access},
        selected_subagent_id=actor_facts["selected_subagent_id"], task_id=str(getattr(ctx, "task_id", "") or ""),
        route=route.route_id, processing=processing_info if recovering else {"requested": actor.get("processing_preference")},
        row_effort=None if recovering else actor.get("row_effort"))
    try:
        # Health checks the stored route/confinement shape on retries, never current
        # environment defaults; blockers stay typed instead of falling through to API spend.
        unavailable, reset_at = route_health(
            gateway, route.route_id, authority, route_model=route.model, pinned_profile=route.profile_id)
        resolution = resolve_subagent_executor(
            "harness", route=route, unavailable_reason=unavailable, reset_at=reset_at,
        )
        if resolution.blocked:
            record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), resolution.reason)
            return _fail(
                "delegate_start", resolution.reason,
                "The delegated route cannot run now. This is a typed blocker: do NOT "
                "silently fall back onto metered API spend — decide explicitly "
                "(wait for the reset, deliver partial work, or ask the parent).",
                executor="blocked", reset_at=resolution.reset_at, route=route.route_id, definitely_unrun=True,
            )

        snapshot = None
        if not recovering:
            if payload_auth is not None:
                record_auth = payload_auth
            else:
                record_auth, root_error = _mutation_authority(ctx, authority)
                if root_error:
                    return root_error
            invocation_id = custody.new_invocation_id()
            root = record_auth["target_root"]
            if continuation_token:  # gated from durable custody, before any snapshot exists
                continuation, refusal = start_binding(
                    ctx, drive, continuation_token, gateway=gateway, actor=actor, route=route, authority=authority,
                    target_root=str(record_auth.get("target_root") or ""), invocation_id=invocation_id, text=text,
                    coordination_context=_coordination_context, carrier=continue_carrier, source_binding=actor_facts,
                    canonical_work_order_fingerprint=str(_canonical_work_order_fingerprint or ""))
                if refusal:
                    return refusal
            if authority.access in SESSION_ACCESS_PROFILES:
                target_root = record_auth["target_root"]
                authority_source = record_auth["source"]
                if authority_source == "skill_payload":
                    snapshot, snap_error = _provision_payload_snapshot(
                        ctx, drive, record_auth, invocation_id)
                elif not (Path(target_root) / ".git").exists():
                    from ouroboros.delegate_directory import prepare_directory_execution
                    try:
                        authority, directory_options, resource_ref = prepare_directory_execution(
                            ctx, gateway, target_root, authority, directory_strategy, scope_paths)
                    except ValueError as exc:
                        return _fail("delegate_start", "directory_execution_unavailable", str(exc), definitely_unrun=True)
                    snapshot, snap_error = None, ""
                else:
                    from ouroboros.delegate_directory import git_directory_options_refusal
                    if error := git_directory_options_refusal(target_root, directory_strategy, scope_paths):
                        return _fail("delegate_start", "directory_execution_unavailable", error, definitely_unrun=True)
                    snapshot, snap_error = ((continuation.snapshot, None) if continuation.snapshot
                                            else _provision_snapshot(ctx, drive, target_root, invocation_id))
                if snap_error:
                    _settle_refused_provision(ctx, gateway, snap_error, invocation_id, history_facts)
                    return snap_error
                if snapshot is not None:
                    snapshot_id, baseline_sha, root = snapshot.snapshot_id, snapshot.baseline_sha, snapshot.path
                    resource_ref = dict(record_auth.get("resource_ref") or {})
            if authority.access == "readonly":
                from ouroboros.delegate_readonly_inputs import prepare_folderless_inputs

                try:
                    readonly_root, input_instruction = prepare_folderless_inputs(ctx, invocation_id)
                except (OSError, ValueError) as exc:
                    return _fail("delegate_start", "readonly_inputs_unavailable", str(exc), definitely_unrun=True)
                if readonly_root:
                    root = readonly_root
                    instructions += input_instruction
            execution_root = (root if directory_options.get("isolation") == "live" else "") if directory_options else delegated_execution_workspace_root(gateway, authority, root)
            scope_root = target_root if execution_root or directory_options else root
            if snapshot is not None:
                binding_fingerprint = execution_binding_fingerprint(
                    execution_root or root, target_root)
                instructions = apply_execution_binding(
                    instructions, execution_root or root, target_root, binding_fingerprint)
            elif directory_options and directory_options.get("isolation") == "envelope":
                binding_fingerprint = execution_binding_fingerprint("", target_root, "directory_copy")
                instructions += directory_copy_binding_instruction(target_root, binding_fingerprint)
            (project_id, owned_project_id, project_persistent) = resolve_registration(
                gateway, scope_root, execution_root, getattr(authority, "access", ""))
            if directory_options:
                project_persistent = True
            if authority.access == "full":
                gateway.ensure_full_access(scope_root)
            seconds = bound.seconds
            request_body = _start_request(ctx, route, authority, scope_root, continuation.prompt or text,
                                          seconds, instructions, execution_root, continuation=continuation.request,
                                          **({"directory_options": directory_options} if directory_options else {}))
            request_body, processing_info = _processing_start_request(request_body, actor, gateway, route)
            history_facts["access"] = request_body["access"]
            key = custody.idempotency_key(getattr(ctx, "task_id", ""), route.route_id,
                                          access, authority.mode, authority.isolation,
                                          root, request_body["prompt"], request_body["instructions"])
        lineage = getattr(ctx, "task_metadata", {}) or {}
        lineage = lineage if isinstance(lineage, dict) else {}
        snapshot_facts = dict(snapshot_id=snapshot_id, baseline_sha=baseline_sha, target_root=target_root,
                              authority_source=authority_source, resource_ref=resource_ref,
                              execution_binding_fingerprint=binding_fingerprint, **continuation.custody)
        requested, claim_refusal = claimed_start_request(
            drive, claim_target=(target_root if not recovering and authority_source == "skill_payload" else ""),
            actor_ctx=ctx, enforce_actor_idle=not recovering,
            run_id="", task_id=str(getattr(ctx, "task_id", "") or ""),
            idempotency_key=key, invocation_id=invocation_id,
            max_seconds=seconds, max_seconds_basis=seconds_basis, request=request_body, project_id=project_id,
            project_owned=bool(owned_project_id), project_persistent=project_persistent, route=route.route_id,
            root_task_id=str(lineage.get("root_task_id") or ""), parent_task_id=str(lineage.get("parent_task_id") or ""),
            **snapshot_facts,
            execution_root=(root if snapshot_id or resource_ref.get("strategy") == "direct" else ""),
            # Recovery proves the original actor and compiled brief before adoption.
            **actor_facts,
            processing=processing_info,
        )
        if not requested and not claim_refusal:
            claim_refusal = {"reason": "start_request_row_unwritable", "detail":
                "The durable start-request row could not be written, so the run was NOT started. "
                "Fix the drive/event log and retry; a run without custody would be unfindable.",
                **({"definitely_unrun": True} if not recovering else {})}
        if claim_refusal:
            reason = str(claim_refusal.get("reason") or "replacement_custody_unknown")
            detail = str(claim_refusal.get("detail") or "Actor start claim unavailable.")
            facts = {key: value for key, value in claim_refusal.items()
                     if key not in {"reason", "detail"}}
            return _fail(
                "delegate_start", reason, detail, **facts,
                **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent, history_facts=history_facts,
                    definite_refusal=not recovering,
                    reason=reason, invocation_id=invocation_id, snapshot_id=("" if recovering else snapshot_id),
                ),
            )
        handle = run_operation(ctx, gateway.start_run, request_body, idempotency_key=invocation_id)
        run_id = str(handle.get("runId") or handle.get("jobId") or "")
        if not run_id:
            return _fail("delegate_start", "queued_without_run_id",
                         f"Claudexor returned a queued handle without a run id: {handle!r}",
                         pending_invocation_id=invocation_id, retry_hint=_RETRY_HINT,
                         **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent, history_facts=history_facts,
                             definite_refusal=False, reason="queued_without_run_id", invocation_id=invocation_id))
    except (ClaudexorUnavailable, OwnerPauseRefused) as exc:
        code = getattr(exc, "code", str(exc))
        # A registration we created BEFORE the start must not outlive a failed start.
        # It used to be left behind with nothing anywhere naming its id.
        status = int(getattr(exc, "status_code", 0) or 0)
        definite = (isinstance(exc, OwnerPauseRefused) and not recovering) or 400 <= status < 500 or not requested
        # An UNKNOWN outcome hands back the retry token: only the caller can say
        # whether the next call is a retry of this intention or a new intention, and
        # without the token every next call is a new one. A definite refusal retires
        # the id, so no token rides a refusal.
        pending = ({} if definite or not invocation_id else
                   {"pending_invocation_id": invocation_id,
                    "retry_hint": _RETRY_HINT})
        return _fail("delegate_start", code, str(exc), executor="blocked",
                     **({"definitely_unrun": True} if not requested else {}),
                     reset_at=getattr(exc, "reset_at", ""), **pending,
                     **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent, history_facts=history_facts,
                         definite_refusal=definite, reason=code,
                         invocation_id=invocation_id, snapshot_id=("" if recovering else snapshot_id)))
    except BaseException as exc:
        # EVERY pre-custody exit leaves a durable disposition, including the ones no
        # typed handler claims (a bug here, a timeout, a signal). NEVER retired: an
        # untyped exit says nothing about whether the POST reached the daemon, so a run
        # may be live against it. Named with a typed reason so the sweep's
        # pending-invocation recovery finds it, then re-raised — disclosure, not a swallow.
        _retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent, history_facts=history_facts,
                                      definite_refusal=False,
                                      reason=f"pre_custody_exit_{type(exc).__name__}",
                                      invocation_id=invocation_id)
        raise
    finally:
        gateway.close()

    # A start whose custody row did not land does not wear the plain name: the run
    # is live and only THIS process knows it exists (the uncustodied-run leak).
    durable = record_started_custody(
        drive, run_id, ctx, route, authority,
        key=key, access=access, root=root, seconds=seconds,
        invocation_id=invocation_id, project_id=project_id,
        project_owned=bool(owned_project_id), project_persistent=project_persistent,
        **actor_facts, **snapshot_facts, processing=processing_info,
        capture_mode=("engine_directory" if resource_ref.get("workspace_kind") == "directory" else
                      _CAPTURE_DELEGATED_SNAPSHOT if snapshot_id else ""),
        max_seconds_basis=seconds_basis, row_effort=None if recovering else actor.get("row_effort"),
    )
    from ouroboros.tools.control import maybe_emit_delegated_run_fanout
    maybe_emit_delegated_run_fanout(ctx, run_id=run_id, route_id=route.route_id, objective=text, durable=durable)
    return _started_payload(handle, run_id, route, access, authority, root,
                            durable=durable, recovering=recovering, invocation_id=invocation_id,
                            snapshot_id=snapshot_id, target_root=target_root, baseline_sha=baseline_sha,
                            resource_ref=resource_ref, processing=processing_info, continuation=continuation.facts,
                            max_seconds=seconds, max_seconds_basis=seconds_basis,
                            engine_version=str(getattr(gateway, "engine_version", "") or ""),
                            snapshot_facts=_snapshot_facts(snapshot),
                            effort_fact=({} if recovering else actor.get("effort_fact")))


def _started_payload(handle: Dict[str, Any], run_id: str, route: Any, access: str,
                     authority: "DelegatedRunShape", root: str, *, durable: bool,
                     recovering: bool, invocation_id: str, snapshot_id: str, target_root: str,
                     baseline_sha: str, engine_version: str = "", resource_ref=None, processing=None,
                     continuation=None, max_seconds: int = 0, max_seconds_basis: str = "", snapshot_facts=None,
                     effort_fact=None) -> ToolResult:
    """The one author of delegate_start's started result (note + payload).

    The AUTHORITY guidance and the CUSTODY warning are independent facts about the same
    start, so both are said. An undurable custody row is the louder one and goes first:
    a nanny that walks away from an uncustodied MUTATING run leaves a live shell in its
    own worktree that nothing outside this process can name.
    """
    note = "" if durable else (
        "CUSTODY IS NOT DURABLE: the run started, but its custody row could not be written, "
        "so nothing outside this worker can wait on, cancel or settle it. Do not walk away "
        "from it — finish it or delegate_cancel it in this session. ")
    note += (
        "You are the nanny and the host. Poll with delegate_wait; the run's own "
        "claims are evidence to check, not a verified result."
        + (
            " This run edits a PRIVATE SNAPSHOT of your write root, not the shared "
            "tree: at terminal its diff is captured for you, and NOTHING lands in "
            "the shared tree until you explicitly integrate_delegated_patch(run_id="
            "...) to apply or reject it — read the captured diff before you claim "
            "it, and never let the run commit. The requested native access profile "
            f"is {authority.access}; scoped HOME selects native state and credentials, "
            "not filesystem confinement. Actual access is read from the run's own "
            "artifacts by delegate_wait."
            if authority.isolation == "live" else
            " This run cannot write anything: it reads and answers."
        )
    )
    payload = {
        "status": "started" if durable else "started_uncustodied",
        "run_id": run_id,
        "run_dir": handle.get("runDir"),
        "engine_version": engine_version,
        "route": route.route_id,
        "model": route.model,
        "effort": route.effort,
        # The effort decision beside the level sent (``settings_scales.choose_effort``).
        **({"effort_requested": str(effort_fact.get("requested") or ""),
            "effort_source": str(effort_fact.get("source") or "")} if effort_fact else {}),
        "access": access,
        "mode": authority.mode,
        "isolation": authority.isolation or "envelope",
        "idempotent_recovery": recovering,
        # ASKED, not applied. The proof arrives with the run's own artifacts and is
        # relayed by delegate_wait; saying "isolated" here would be the exact claim
        # this whole verification exists to stop anyone from making.
        "scoped_home_requested": authority.delegated,
        "root": root,
        "custody_durable": durable,
        "invocation_id": str(invocation_id or ""),
        # The cap and HOW it was decided (#1196): only a `requested` cap's expiry
        # is a finite-leaf expiry a later continue_from may follow.
        "max_seconds": int(max_seconds or 0),
        "max_seconds_basis": str(max_seconds_basis or ""),
        "note": note,
    }
    if not durable:
        payload["pending_invocation_id"] = str(invocation_id or "")
    if processing:
        payload["processing"] = processing
    if continuation:
        # The binding facts, stated where the nanny reads them: which settled run
        # this continues, its cause, its tree and the advice its child was given.
        payload["continuation"] = dict(continuation)
    if snapshot_facts:
        payload["snapshot"] = snapshot_facts
    if snapshot_id:
        # The C1 binding, stated where the nanny can read it: the run edits the
        # EXECUTION snapshot; the authority target receives nothing until apply.
        payload["execution_root"] = root
        payload["authority_target_root"] = target_root
        payload["baseline_id"] = baseline_sha
        if not (snapshot_facts or {}).get("reused_from_run"):  # a reused snapshot's manifest is its first run's
            payload["baseline_manifest_read"] = {"root": "artifact_store", "path": f"delegated_runs/{snapshot_id}/baseline_manifest.json"}
    if isinstance(resource_ref, dict) and resource_ref.get("workspace_kind") == "directory":
        direct = resource_ref.get("strategy") == "direct"
        payload.update(authority_target_root=target_root,
                       execution_root=target_root if direct else None,
                       directory_strategy=resource_ref["strategy"], scope_paths=resource_ref["scopePaths"])
        payload["note"] = (
            ("CUSTODY IS NOT DURABLE. " if not durable else "")
            + ("This run writes directly in the selected folder; effects are already on site, without a full rollback promise. "
               if direct else "The engine prepares a separate copy of the selected inputs; its actual execution directory is not yet observed. ")
            + "Use delegate_wait for its complete file manifest and result. Copy results use integrate_delegated_patch for apply or reject."
        )
    return delegate_result(payload)


def _settle_refused_provision(ctx: ToolContext, gateway: Any, refusal: ToolResult,
                              invocation_id: str, history_facts: Optional[dict]) -> None:
    """A pre-POST provisioning refusal (no start row, no run) still settles its
    invocation durably (START_FAILED), so the refusal exists outside this process —
    the incident's bootstrap refusals left no row at all (#1241)."""
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.subagent_bootstrap import refusal_facts

    payload = delegate_payload(refusal)
    _retire_orphaned_registration(
        ctx, gateway, "", definite_refusal=True, invocation_id=invocation_id,
        reason=str(payload.get("reason") or "execution_snapshot_failed"),
        history_facts={**(history_facts or {}), **refusal_facts(payload),
                       "detail": str(payload.get("detail") or "")})


def _snapshot_facts(handle: Any) -> Dict[str, Any]:
    """Disclosure on the start receipt: how big the private snapshot is and how long
    it took to provision (#1241). Facts only — nothing refuses or truncates on them."""
    if handle is None:
        return {}
    if getattr(handle, "adopted_from", ""):
        return {"reused_from_run": handle.adopted_from}  # a continuation's predecessor snapshot
    file_baseline = getattr(handle, "file_baseline", {}) or {}
    return {"entries": int(getattr(handle, "entry_count", 0) or 0),
            "untracked_files": len(getattr(handle, "untracked_baseline", {}) or {}) + len(file_baseline),
            "file_input_bytes": sum(int(item.get("size", 0) or 0) for item in file_baseline.values()),
            "provisioning_sec": float(getattr(handle, "provisioning_sec", 0.0) or 0.0)}


def _retire_orphaned_registration(ctx: ToolContext, gateway: Any, project_id: str, *,
                                  definite_refusal: bool, reason: str,
                                  project_persistent: bool = False,
                                  invocation_id: str = "",
                                  snapshot_id: str = "", history_facts: Optional[dict] = None) -> Dict[str, Any]:
    """Retire a registration this start created but never bound to a run.

    Only when the daemon gave a DEFINITE negative answer (a 4xx refusal): a transport
    error, a 5xx, or a 2xx handle with no run id all mean the POST's fate is unknown, and
    a run may well be live against this very registration. An unverified outcome is never
    grounds for destroying state — the durable row names the id either way, which is what
    the old code lacked. The caller supplies the verdict, so every failing start reaches
    this one path instead of one branch retiring and its twin abandoning.

    The row also settles the INVOCATION's fate: ``definite: true`` retires the logical
    invocation id (a definitely refused invocation must not be reused — the daemon may
    hold its key against a body a reconfigured route can no longer reproduce, which
    would 409 forever), while an unknown outcome leaves it pending so a transport retry
    presents the same key and lands on whatever the daemon really has. Written even
    with no registration to retire, because the invocation's fate is its own fact.
    """
    if snapshot_id and definite_refusal and not any(  # a continuation's predecessor snapshot stays its run's
            run.snapshot_id == snapshot_id for run in custody.replay(custody.custody_root(ctx)).values()):
        # The C1 execution snapshot THIS attempt provisioned. Only a definite refusal
        # proves no run can be live against it; an unknown outcome keeps it — the
        # pending invocation names it durably, and the startup GC reconciles it.
        try:
            from ouroboros.subagent_worktrees import remove_execution_snapshot

            remove_execution_snapshot(snapshot_id)
        except Exception:
            log.warning("Failed to retire delegated execution snapshot %s", snapshot_id,
                        exc_info=True)
    retired = False
    if project_id and definite_refusal and not project_persistent:
        try:
            gateway.remove_project(project_id)
            retired = True
        except Exception as exc:
            # A registration the daemon does not have is already retired: the same
            # absence-is-discharge fact `retire_project` settles on.
            retired = custody.daemon_says_absent(exc)
            if not retired:
                log.warning("Failed to retire orphaned delegated project %s", project_id, exc_info=True)
    if project_id or invocation_id:
        _emit(ctx, custody.START_FAILED, {"run_id": "", "project_id": project_id,
                                          "project_retired": retired, "reason": reason,
                                          "invocation_id": invocation_id,
                                          "definite": bool(definite_refusal), **(history_facts or {})})
    if not project_id:
        return {"project_retired": False}
    if project_persistent and definite_refusal:
        # #362: a definite refusal still never deletes the user's stable
        # project (the f9356572 skip) — but the invocation's fate row above
        # has landed, so the lane cannot livelock on a forever-pending id.
        return {"project_retired": False, "project_id": project_id,
                "project_retention_reason": "persistent_registration"}
    if retired or definite_refusal:
        return {"project_retired": retired, "project_id": project_id}
    return {"project_retired": False, "project_id": project_id,
            "project_retention_reason": "start_outcome_unknown_run_may_exist"}


class MaxSecondsBound(NamedTuple):
    """One decided ``maxSeconds``: the seconds, HOW they were decided
    (``delegate_registration_policy.CAP_BASIS_*``), or the typed refusal."""

    seconds: int
    basis: str
    refusal_code: str = ""
    refusal_detail: str = ""


def bounded_max_seconds(ctx: ToolContext, requested: Optional[int]) -> MaxSecondsBound:
    """Narrow-only: the delegated run may never outlive the nanny's own deadline
    or its finite lifetime, and the decision is RECORDED beside the number.

    A caller must ask ``deadline_expired`` FIRST: an expired deadline cannot
    produce an honest bound at all. Less than one second of deadline or of
    lifetime is refused typed (``task_deadline_too_close`` /
    ``task_lifetime_exhausted``): ``int()`` would truncate it to 0, which the
    fallback branch read as "no bound" and answered with hours of delegated
    work. An explicit ask is ``requested`` unless the deadline, the lifetime or
    the engine's schema bound narrowed it (each named); an omitted ask derives
    from the tighter of deadline and lifetime, else the operation window.
    """
    from ouroboros.config import get_task_abs_ceiling_sec, operation_window_sec
    from ouroboros.deadline_utils import deadline_remaining_sec, has_deadline
    from ouroboros.delegate_registration_policy import (
        CAP_BASIS_DEADLINE_DERIVED, CAP_BASIS_LIFETIME_DERIVED, CAP_BASIS_OPERATION_WINDOW,
        CAP_BASIS_REQUESTED, CAP_BASIS_REQUESTED_CLAMPED_DEADLINE, CAP_BASIS_REQUESTED_CLAMPED_LIFETIME,
        CAP_BASIS_REQUESTED_CLAMPED_SCHEMA,
    )

    try:
        asked = max(0, int(requested)) if requested is not None else 0
    except (TypeError, ValueError):
        asked = 0
    deadline_left: Optional[float] = None
    if has_deadline(ctx):
        deadline_left = float(deadline_remaining_sec(ctx))
        if deadline_left < 1.0:
            return MaxSecondsBound(0, "", "task_deadline_too_close",
                                   "Less than one second of this task's deadline remains: no delegated run "
                                   "can be started under it. Finalize with what you have.")
    # Seconds of the task's FINITE lifetime still unspent (``None`` = unlimited).
    # Cumulative EXECUTION time decides, never a reset clock: the live model-wait
    # owner's window (elapsed minus the quota union minus the budget-paused
    # carrier) when this task's is bound, else the same arithmetic over the
    # original ``task_started_at`` and the paused carrier the resume handed over
    # (quota waits unknown here count as execution — the narrower direction).
    # No recorded start uses the operation-bounded ceiling; a failed clock read
    # is a typed unknown-lifetime refusal, never a fresh finite window.
    lifetime_left: Optional[float] = get_task_abs_ceiling_sec()
    if lifetime_left is not None:
        try:
            from ouroboros.model_wait import current_model_wait, execution_elapsed_seconds

            waiter = current_model_wait()
            remaining = None
            if (waiter is not None
                    and str(getattr(waiter, "task_id", "") or "") == str(getattr(ctx, "task_id", "") or "")):
                remaining = waiter.execution_window_remaining()
            started = getattr(ctx, "task_started_at", None)
            if remaining is not None:
                lifetime_left = float(remaining)
            elif started:
                paused = getattr(ctx, "_budget_paused_sec", None)
                if paused is None:
                    paused = (getattr(ctx, "budget_pause_resume", None) or {}).get("paused_duration_sec")
                executed = execution_elapsed_seconds(
                    {"started_at": float(started), "budget_paused_sec": float(paused or 0.0),
                     "model_wait_quota_clock": {}}, time.time())
                lifetime_left = max(0.0, float(lifetime_left) - executed)
            else:
                lifetime_left = float(lifetime_left)
        except Exception:
            return MaxSecondsBound(0, "", "task_lifetime_unknown",
                                   "This task's remaining finite lifetime could not be established; no delegated run started.")
    if lifetime_left is not None and lifetime_left < 1.0:
        return MaxSecondsBound(0, "", "task_lifetime_exhausted",
                               "This task's finite lifetime is spent (cumulative execution time, the "
                               "paused interval excluded): no delegated run can be started. Finalize.")
    if asked > 0:
        seconds, basis = asked, CAP_BASIS_REQUESTED
        if deadline_left is not None and int(deadline_left) < seconds:
            seconds, basis = int(deadline_left), CAP_BASIS_REQUESTED_CLAMPED_DEADLINE
        if lifetime_left is not None and int(lifetime_left) < seconds:
            seconds, basis = int(lifetime_left), CAP_BASIS_REQUESTED_CLAMPED_LIFETIME
        if seconds > _CLAUDEXOR_MAX_SECONDS:
            # Clamp HERE too, not only on the fallback below: `max_seconds` is a model-supplied
            # tool argument with no maximum in its schema (control.ts `.max(604_800)`).
            seconds, basis = _CLAUDEXOR_MAX_SECONDS, CAP_BASIS_REQUESTED_CLAMPED_SCHEMA
        return MaxSecondsBound(max(1, seconds), basis)
    candidates = []
    if deadline_left is not None:
        candidates.append((int(deadline_left), CAP_BASIS_DEADLINE_DERIVED))
    if lifetime_left is not None:
        candidates.append((int(lifetime_left), CAP_BASIS_LIFETIME_DERIVED))
    if candidates:
        seconds, basis = min(candidates, key=lambda item: item[0])
        return MaxSecondsBound(max(1, min(_CLAUDEXOR_MAX_SECONDS, seconds)), basis)
    # Neither a deadline nor a finite lifetime: the finite operation window.
    # Omitting `maxSeconds` — the old behavior — handed the run Claudexor's
    # 7-day schema bound; the cap is damage limitation, and custody (the
    # durable start row plus reconciliation) is what actually stops an orphan.
    return MaxSecondsBound(min(_CLAUDEXOR_MAX_SECONDS, int(operation_window_sec(None))), CAP_BASIS_OPERATION_WINDOW)


def _halt_breached_run(ctx: ToolContext, gateway: Any, entry: _RunCustody,
                       breach: _Breach) -> str:
    """Stop a run the engine did not contain as asked, and say exactly what failed.

    The BREACH incident goes through ``custody.record_containment_fault``, the same
    writer an unverified cancel uses, so a breached run also surfaces as the CRITICAL
    health invariant that stays open until a terminal receipt resolves it. Emitting a
    look-alike event here instead left the breach out of the open-fault sweep.

    The stop itself goes through ``custody.cancel_and_verify`` — the ONE cancel path,
    with its four typed outcomes — and the sentence handed back to the agent is built
    from the outcome it returns. The ad-hoc cancel this replaced swallowed every
    exception into a log line and then said "The run was cancelled" unconditionally,
    which is precisely what ``record_containment_fault``'s own contract forbids: an
    incident must never surface "as a reassuring string in a tool result". An
    overpowered run that refused to stop was reported to the agent as stopped.
    """
    run_id = entry.run_id
    drive = custody.custody_root(ctx)
    try:
        cancelled = custody.cancel_and_verify(drive, gateway, entry, breach.code)
    except Exception:
        log.warning("Failed to cancel an uncontained delegated run %s", run_id, exc_info=True)
        cancelled = {"outcome": custody.CANCEL_CONTAINMENT_FAULT}
    custody.record_containment_fault(drive, entry, breach.code, breach.detail,
                                     fault=breach.code, **breach.facts)
    outcome = str(cancelled.get("outcome") or custody.CANCEL_CONTAINMENT_FAULT)
    return _fail(
        "delegate_wait", breach.code,
        f"{breach.detail} {_CANCEL_NOTES.get(outcome, '')} Do not retry it: this is a "
        "containment fault in the transport or the engine, not a task failure — report "
        "it and continue within your own authority.",
        run_id=run_id, cancel_outcome=outcome, **breach.facts,
    ).text


# The typed external-wait lease lives in `delegate_progress` (the wait-liveness
# module); re-bound here because the wait's own seams and the tests name it on
# this surface, exactly like the staged-output cluster above.
_external_wait_lease_until = progress.external_wait_lease_until
_emit_external_wait_lease = progress.emit_external_wait_lease


def _delegate_wait(ctx: ToolContext, run_id: str, wait_sec: Optional[int] = None,
                   since_seq: Optional[int] = None, *, observation_only: bool = False,
                   gateway: Any = None) -> str:
    """Time-bounded, progress-aware wait (docs/DEVELOPMENT.md "Timeout & Wait Control").

    ``gateway`` is a transport BORROWED from the supervision loop: it replaces the
    per-call ``ClaudexorGateway()``, is handshaken here only when it has not been yet
    (``engine_version`` is the handshake receipt), and is never closed here; the
    owner closes it. Absent, the call builds, handshakes and closes its own.

    HOLDS the window it was given. It returns early only on a terminal state or a
    containment fault; a journal-cursor advance past ``since_seq`` is RECORDED and
    streamed to the human live, and the model is woken once, at expiry, with the whole
    sequence in ``advances``. Returning on first advance would buy a model round per
    streaming batch. Only the journal cursor proves progress, never SSE ``: ping``.

    NARROW-ONLY, like ``bounded_max_seconds``: the wait may not outlive the nanny's own
    deadline minus finalization grace. This tool owns its ToolEntry outer bound and is
    absent from ``_DEADLINE_CLAMPED_TOOLS``: clamping here preserves typed ``no_progress``
    instead of an outer thread-kill. Only "no deadline set" is left
    unclamped; a SPENT deadline clamps to the floor, the window is measured from before
    the connection, and every call is BOUNDED by what it has left (``progress.poll_bound``)
    so no read can outrun it as the 60s default could. The internal supervision
    observer instead makes one read under the ordinary transport bound narrowed by
    the real task deadline; its three-second beat is not a network deadline. A transport
    failure that delivered no daemon answer there (typed per class:
    ``observation_read_timeout`` for our own bound expiring, ``daemon_unreachable`` for a
    socket that carried nothing) retains unknown observation and the same run, without
    model wake.
    Legacy caller-sized waits preserve their last-poll expiry contract.
    """
    from ouroboros.config import get_delegate_wait_max_sec, get_delegate_wait_sec
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        ClaudexorUnavailable,
        pending_interactions as _cx_pending,
    )

    rid = str(run_id or "").strip()
    if not rid:
        return _fail("delegate_wait", "missing_run_id", "run_id is required").text
    not_mine, entry = _owned_run(ctx, "delegate_wait", rid)
    if not_mine:
        from ouroboros.tools.delegate_terminal_evidence import retry_terminal_result
        return retry_terminal_result(ctx, rid, gateway=gateway).text
    if not_mine or entry is None:
        return (not_mine or _fail("delegate_wait", "run_ownership_unknown",
                                  "custody unresolved", run_id=rid)).text
    ceiling = get_delegate_wait_max_sec()
    try:
        window = int(wait_sec) if wait_sec is not None else get_delegate_wait_sec()
    except (TypeError, ValueError):
        window = get_delegate_wait_sec()
    from ouroboros.deadline_utils import parse_deadline_ts, window_within_deadline

    window = window_within_deadline(ctx, max(1, min(window, ceiling)))

    # Handshake and first poll spend this call's window too; start its clock before connecting.
    started = time.monotonic()
    deadline = started + window
    from ouroboros.gateways.claudexor import _READ_TIMEOUT_SEC

    def read_window() -> float:
        return (float(window_within_deadline(ctx, int(_READ_TIMEOUT_SEC)))
                if observation_only else deadline - time.monotonic())

    borrowed = gateway is not None

    # The GRANTED shape replays from the durable custody row (R1 item 2): the run
    # was admitted under host-derived authority recorded on its STARTED row, and a
    # top-level payload delegation has no acting/workspace context to re-derive
    # from — re-derivation read `readonly` and cancelled the run as widened on its
    # first wait. Ownership was already proven above (`_owned_run`), and a lost or
    # legacy custody record (no recorded access) keeps the live derivation, so a
    # missing record still cannot become a wider run.
    if entry.access:
        from ouroboros.subagents import DelegatedRunShape as _Shape

        authority = _Shape(access=entry.access, mode=entry.mode,
                           isolation=entry.isolation, delegated=entry.delegated)
    else:
        authority = _derive_authority(ctx)
    # FACTS against premature cancels: how long the run has actually been going and
    # what its cap really is, from the durable start row. A nanny that cannot see
    # these confabulates "exceeded the cap" out of its own impatience. Absent facts
    # (an old row, an unknown run) stay null — never invented.
    _started_ts, _run_max_seconds = custody.run_timing(custody.custody_root(ctx), rid)
    _started_at = parse_deadline_ts(_started_ts)
    # The idle-rail lease for THIS hold: granted before the loop, released in the
    # finally below — the supervisor's idle enforcer spares a leased task while
    # every other rail (deadline, ceiling, budget, cancel) still cuts through.
    # The grant carries a unique lease_id (F5b) and the release names the SAME
    # id, so an abandoned, executor-killed wait thread's late release can never
    # blank a newer grant made by this task's next wait.
    _lease_id = uuid.uuid4().hex
    _emit_external_wait_lease(
        ctx, rid, _external_wait_lease_until(
            ctx, max(window, read_window()) if observation_only else window, _started_at, _run_max_seconds),
        lease_id=_lease_id)
    try:
        if gateway is None:
            gateway = ClaudexorGateway()
        if not getattr(gateway, "engine_version", ""):
            gateway.handshake(timeout_sec=progress.poll_bound(read_window()))
        if observation_only:
            _emit_external_wait_lease(
                ctx, rid, _external_wait_lease_until(ctx, read_window(), _started_at, _run_max_seconds),
                lease_id=_lease_id)
        detail = progress.bounded_poll(gateway, rid, read_window())
        baseline = int(since_seq) if since_seq is not None else int(detail.get("lastSeq") or 0)
        seen = progress.WindowObservations()
        seen.observe_baseline(detail, baseline)
        while True:
            summary = custody.summary_of(detail)
            state = str(summary.get("state") or "")
            last_seq = int(detail.get("lastSeq") or 0)
            breach = _containment_breach(detail, authority)
            if breach:
                return _halt_breached_run(ctx, gateway, entry, breach)
            if last_seq > baseline or state in _TERMINAL_STATES:
                # The STREAM is not collapsed — the TIMER is: every advance reaches the live
                # progress surface the instant this loop sees it (also the frame the idle enforcer
                # reads); only waking the MODEL per batch stops. At the end it drains the unread
                # tail within this wait's read allowance before settlement returns.
                progress.emit(ctx, rid, seen.record(detail, last_seq, int(time.monotonic() - started)), detail=detail,
                              entry=entry, gateway=gateway, after_seq=baseline, read_remaining=read_window,
                              drain_sec=read_window() if state in _TERMINAL_STATES else None)
                baseline = last_seq          # so the NEXT advance is counted once
            if state in _TERMINAL_STATES:
                settlement = custody.settle_run(custody.custody_root(ctx), gateway, entry, detail)
                payload = _delivered_terminal_payload(ctx, rid, detail, authority, entry, gateway)
                payload["settlement"] = settlement
                # C1: a mutating run's changes live in its private snapshot until the
                # nanny explicitly integrates them. Captured HERE, durably, on every
                # terminal observation (idempotent), cancelled runs included — a
                # cancelled run's partial work is salvage material, not garbage.
                capture = _capture_terminal_patch(ctx, entry, **(
                    {"gateway": gateway} if entry.resource_ref.get("workspace_kind") == "directory" else {}))
                if capture is not None:
                    payload["workspace_capture"] = capture
                # D7 made load-bearing: settlement is where "paid for and never read"
                # becomes permanent, so the parent is told in WORDS here — not left to
                # infer it from `output_delivery.consumed`. Re-settling an already
                # settled run reports the CURRENT durable fact, so the line disappears
                # once the read has happened rather than echoing a stale omission.
                if custody.record_settled_unread(custody.custody_root(ctx), entry):
                    payload["result_not_collected"] = (
                        "THIS RESULT IS NOT COLLECTED YET: the run is settled and its "
                        "full output is staged, but nothing has read it to EOF. Read the "
                        "artifact named in output_delivery with read_file "
                        "root='task_drive' until it is covered end to end — a result you "
                        "have not read is not a result you may report."
                    )
                # The containment disclosure is read off what the PARENT was told, so the
                # durable line and the relayed payload cannot disagree. It runs on the
                # PREVIEW path too: a payload big enough to spill is exactly the one whose
                # containment block a reader is least likely to reach.
                _record_containment(ctx, entry, payload)
                return json.dumps(payload, ensure_ascii=False, indent=2)
            pending = _cx_pending(detail)
            if pending and _interactions_are_news(rid, pending):
                # A NEW question returns IMMEDIATELY: the old wait kept only the
                # waitingOnUser boolean and showed it at window expiry, so a paused
                # run burned the rest of the window (up to the engine's whole
                # answer timeout) in dead metered polling. A question the model
                # ALREADY saw does not re-trigger — a nanny that escalated it
                # up the hierarchy keeps holding windows instead of busy-looping, and the
                # engine timeout stays the backstop.
                return _waiting_on_user_payload(ctx, rid, state, last_seq, pending,
                                                seen=seen,
                                                source_request=entry.work_order_source_request,
                                                source_verification=(
                                                    custody.work_order_source_verification(entry)
                                                    if entry.work_order_source_request else {}
                                                ))
            def _expired() -> str:
                # The window payload is a DICT; the supervising wake adds its own
                # whole-sleep facts (``sleep``, one cache-horizon note) as fields at
                # publication, never per tick and never after the rendered JSON.
                payload = progress.window_payload(
                    run_id=rid, state=state, last_seq=last_seq,
                    window=(time.monotonic() - started) if observation_only else window,
                    elapsed_seconds=(None if _started_at is None else max(0, int(
                        (_dt.datetime.now(tz=_dt.timezone.utc) - _started_at).total_seconds()))),
                    max_seconds=_run_max_seconds or None,
                    waiting_on_user=bool(summary.get("waitingOnUser")) or bool(pending),
                    pending_interactions=_bounded_interactions(pending) if pending else None,
                    detail=detail, seen=seen,
                    budget=tool_result_limit("delegate_wait"))
                return json.dumps(payload, ensure_ascii=False, indent=2)

            if observation_only or time.monotonic() >= deadline:
                if observation_only:
                    time.sleep(max(0.0, deadline - time.monotonic()))
                return _expired()
            time.sleep(min(_POLL_INTERVAL_SEC, max(0.0, deadline - time.monotonic())))
            # BOUNDED whether or not the window is spent: a poll STARTED a moment before
            # expiry still carries the client's 60s read default, so the clamp bounded the
            # sleeping and not the waiting. What an UNANSWERED one MEANS is what differs.
            # The last poll of a spent window is bounded and never skipped — terminal
            # state and breach are judged on fresh data or not at all — and a daemon too
            # slow to answer THAT one is this window's expiry. Earlier, the window still
            # has time and there is no expiry to report: the typed refusal propagates to
            # the handler below, because a daemon that died mid-window relayed as a quiet
            # completed wait is a fabricated duration on top of a run nobody is watching.
            left = deadline - time.monotonic()
            fresh = (progress.expiring_poll(gateway, rid) if left <= 0
                     else progress.bounded_poll(gateway, rid, left))
            if fresh is None:
                return _expired()   # unanswered AT expiry: expire on what is already held
            detail = fresh
    except ClaudexorUnavailable as exc:
        if observation_only and exc.observation_timeout:
            # The gateway's per-class reason, not the generic transport code: a
            # read bound that expired against a live daemon is a quiet hole and
            # nothing more, while a socket that carried no answer is the outage
            # the owner is told about once per episode.
            return json.dumps({
                "status": "observation_pending", "run_id": rid,
                "reason": exc.observation_reason or exc.code, "detail": str(exc),
                "waited_sec": time.monotonic() - started,
            })
        return _fail("delegate_wait", exc.code, str(exc), run_id=rid).text
    finally:
        _emit_external_wait_lease(ctx, rid, 0.0, lease_id=_lease_id)
        if gateway is not None and not borrowed:
            gateway.close()


_CANCEL_NOTES = {
    custody.CANCEL_CONFIRMED: "VERIFIED terminal: the run has stopped. Partial artifacts are "
                              "preserved by Claudexor; a cancelled run has no verdict.",
    custody.CANCEL_REQUESTED: "The daemon ACCEPTED the cancel but the run is not terminal yet. "
                              "It is still running. Call delegate_wait to confirm it stops.",
    custody.CANCEL_FAILED: "The daemon REFUSED the cancel and the run is still live and still "
                           "mutating. Escalate — this is not a stopped run.",
    custody.CANCEL_CONTAINMENT_FAULT: "CONTAINMENT FAULT: the cancel could not be verified, so an "
                                      "overpowered mutating run MAY STILL BE LIVE. A durable "
                                      "incident was recorded and is surfaced as a critical health "
                                      "invariant until a terminal receipt clears it.",
}


def _delegate_cancel(ctx: ToolContext, run_id: str, reason: str = "") -> ToolResult:
    """Stop a delegated run. Destructive by nature — a cancelled reviewer has no verdict.

    Reports only what a terminal receipt proves. Saying "cancelled" over an unverified
    control is worse than saying nothing: it retires the operator's attention from a run
    that is still writing to a workspace.
    """
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable

    rid = str(run_id or "").strip()
    if not rid:
        return _fail("delegate_cancel", "missing_run_id", "run_id is required")
    not_mine, entry = _owned_run(ctx, "delegate_cancel", rid)
    if not_mine or entry is None:
        return not_mine or _fail("delegate_cancel", "run_ownership_unknown", "custody unresolved", run_id=rid)
    try:
        gateway = ClaudexorGateway()
        gateway.handshake()
    except ClaudexorUnavailable as exc:
        return _fail("delegate_cancel", exc.code, str(exc), run_id=rid)
    try:
        result = custody.cancel_and_verify(custody.custody_root(ctx), gateway, entry, reason)
    finally:
        gateway.close()
    outcome = str(result["outcome"])
    payload = {
        "status": outcome,
        "run_id": rid,
        "run_may_still_be_live": outcome != custody.CANCEL_CONFIRMED,
        "accepted": result["accepted"],
        "control_status": result["control_status"],
        "state": result["state"],
        "fault_reason": result["fault_reason"],
        "detail": result["detail"],
        "note": _CANCEL_NOTES.get(outcome, ""),
    }
    if outcome in (custody.CANCEL_FAILED, custody.CANCEL_CONTAINMENT_FAULT):
        # The daemon REFUSED the stop, or the stop could not be verified: a call
        # that reports a run which may still be live and mutating is not a
        # successful call. `confirmed` and `requested` stay successful
        # observations — `requested` IS an accepted command, and `confirmed`
        # with accepted=false over an already settled/absent run is a
        # legitimate no-op, not a failure.
        payload.update(ok=False, host_code=refusal_host_code(outcome))
    return delegate_result(payload)


def _published_entry(core: Any) -> Any:
    """The family's five REGISTERED entries, wrapped in their one string boundary.

    Inside the family a result is a native ``ToolResult``; the handler ABI is
    still ``str``. Publication happens HERE, after every decorator the core ran
    (``exact_start``'s actor/source facts, the wake renderer's spill envelope),
    because the registry's equality gate accepts the published result only when
    its text IS the string the handler returned — an earlier publish inside a
    core is silently discarded. ``functools.wraps`` keeps the core's signature,
    which is what the registry binds arguments against.
    """
    @functools.wraps(core)
    def entry(ctx: ToolContext, *args: Any, **kwargs: Any) -> str:
        return publish_delegate_result(ctx, core(ctx, *args, **kwargs))

    return entry


def get_tools() -> List[ToolEntry]:
    from ouroboros.config import EFFORT_SCALE, get_task_abs_ceiling_sec, operation_window_sec

    return [
        ToolEntry("delegate_start", {
            "name": "delegate_start",
            "description": (
                "Start a run on the owner's configured subscription harness; you become its NANNY. "
                "Subscription is requested, not guaranteed free: expired auth, paid routes or auth fallback "
                "may bill. Read delegate_wait's terminal cost (zero, billed, estimated or undisclosed); "
                "time, quota and a worker slot are also spent. Root, route and access inherit your authority "
                "and cannot widen; access only lowers native rights. Read-only runs only read and answer. "
                "Mutating Git runs edit a PRIVATE SNAPSHOT, never commit. Ordinary folders use "
                "directory_strategy=direct (default) or copy with scope_paths. Direct edits are already "
                "on site, without full rollback. Snapshot/copied results reach your tree only through "
                "integrate_delegated_patch(run_id=..., decision='apply'|'reject'); inspect the captured "
                "diff first. delegate_wait.workspace_capture carries the full file manifest and bytes. "
                "A TOP-LEVEL task may select one existing user-managed skill via root='skill_payload', "
                "bucket and skill_name, within existing authority. It edits a private standalone snapshot; "
                "the live payload stays unchanged until integration. Apply stales prior review: run "
                "skill_preflight and skill_review. Create a new skill's manifest first. Seeded native is "
                "system-repo territory; markerless native is logical external. "
                "Returns run_id for delegate_wait/delegate_cancel. Output is a claim to verify; host "
                "verification receipts remain yours. Missing/unavailable routes refuse typed: explicitly "
                "select a configured alternative, wait, narrow or report blocked. Direct starts require "
                "subagent_id. A configured session's exact leaf is already started before your first round: "
                "supervise the startup receipt's run, never duplicate it. Replacement with prompt='' requires "
                "verified cancellation/terminal settlement or a typed refusal proving no run exists. "
                "Recover an unknown start with retry_of, without a new selector; continue settled work "
                "with continue_from (see arguments). This call requests no extra Claudexor review panel; "
                "new ordinary runs on engine 3.9.8+ default to none. The receipt names engine_version; "
                "older engines or a recovered historical run may retain earlier review behavior. Engine review, execution, "
                "integration and Ouroboros review gates remain separate."
            ),
            "parameters": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                "prompt": {"type": "string", "description":
                    "Direct start: a complete standalone task (why the work exists, decisions, material paths); "
                    "the session has none of my memory. The host appends my human's originating words verbatim. "
                    "Configured-session retry/replacement: optional advisory coordination only; "
                    "the host supplies the canonical work order."},
                "subagent_id": {"type": "string", "description":
                    "Exact agent_session id from Available subagents, required for direct starts including continue_from. "
                    "Omit for the configured snapshot and retry_of. Schedule API actors as recursive children instead."},
                "access": {"type": "string", "enum": list(SESSION_ACCESS_LOWERING), "description":
                    "Lower native access to readonly or workspace_write; omit to inherit the captured profile "
                    "(new mutating sessions default full). Explicit readonly authority wins. Omit on retry_of."},
                "effort": {"type": "string", "enum": ["auto", *EFFORT_SCALE], "default": "auto", "description":
                    "The run's reasoning effort inside my human's effort range: auto/omitted = the recommended "
                    "level; outside the range it runs at the nearest bound. A row my human pinned or a level in "
                    "its model name keeps that level outside Cyber Pro; the receipt says so. Omit on retry_of."},
                "root": {"type": "string", "enum": ["active_workspace", "skill_payload"],
                    "default": "active_workspace", "description":
                    "active_workspace (default/omitted): ordinary delegation. skill_payload: one installed "
                    "user-managed payload you can already write, selected by bucket and skill_name."},
                "bucket": {"type": "string", "description":
                    "With root='skill_payload': the payload location "
                    "(external|clawhub|ouroboroshub|user_repo)."},
                "skill_name": {"type": "string", "description":
                    "With root='skill_payload': the exact skill name."},
                "directory_strategy": {"type": "string", "enum": ["direct", "copy"], "description":
                    "Ordinary folders: direct (default) edits the selected folder; copy prepares scope_paths separately "
                    "for explicit application. Choose by task/owner preference. Read-only callers omit this and "
                    "scope_paths; direct without scope equals omission."},
                "scope_paths": {"type": "array", "items": {"type": "string"}, "description":
                    "Relative copy inputs or direct-work capture paths, including future outputs; ['.'] selects the "
                    "whole folder. Copy requires nonempty scope. Read-only callers omit this and directory_strategy. "
                    "Unselected large inputs stay at source. Direct work without selected/observed paths cannot "
                    "claim a complete changed-file list."},
                "max_seconds": {"type": "integer", "description":
                    "Wall-clock cap, narrowed by your remaining deadline. Allow realistic end-to-end time "
                    "(often 3-5+ minutes). An advancing delegate_wait cursor means progress; the run enforces "
                    "its own cap. Do not cancel progressing work just to hurry it."},
                "continue_from": {"type": "string", "description":
                    "Continue a settled run after cap, subscription limit, crash/restart, cancel or input_required. "
                    "Prefer this when work is worth keeping. It must belong to your task line: yours, your confirmed "
                    "retry predecessor's, or the predecessor's task tree in an owner-created Continue root. "
                    "The engine reuses the session where possible, else briefs a new one with retained evidence; "
                    "some work may need repeating. A writer keeps its undisposed private snapshot as one cumulative "
                    "patch, superseding the old capture. prompt carries only new work/answers/corrections and may "
                    "be empty. For resumable pool_exhausted or a limit with resetsAt, use "
                    "await_messages(wake_at=resetsAt) before continuing. Must be settled, not already continued "
                    "(otherwise use the head), with no ambiguous apply and no wider access. Select subagent_id "
                    "as for a start; configured sessions omit it. Never combine with retry_of."},
                "continue_carrier": {"type": "string", "enum": ["auto", "packet"], "description":
                    "With continue_from: auto (default) reuses the session where possible; packet starts a new "
                    "session with retained evidence when the old session has gone astray."},
                "retry_of": {"type": "string", "description":
                    "pending_invocation_id from an unknown start outcome (transport failure/lost response). "
                    "Replays that invocation byte-identically under its original key to retrieve an accepted run "
                    "without duplicating it. Omit subagent_id or receive a typed conflict. Never use for intended "
                    "new work: a plain call creates a new invocation even with identical prompt."},
                },
            },
        }, _published_entry(_delegate_start_entry),
           timeout_sec=120),
        ToolEntry("delegate_wait", {
            "name": "delegate_wait",
            "description": (
                "A host-confirmed retry successor may retrieve a predecessor's proven terminal result; live control stays with its starter. "
                "Sleep on a delegated run until a meaningful event. Quiet transport windows "
                "are renewed by the host with zero model calls; journal progress still streams "
                "to the human but does not wake you. A daemon that cannot be reached is the "
                "same quiet renewal (typed reason daemon_unreachable, told to the owner once per "
                "outage); while such a read hangs, a finalize_now/hurry control is noticed only "
                "when it returns, up to ~60 s later rather than on the 3 s beat. "
                "Terminal settlement, a new interaction, "
                "fault, addressed owner/task message, a direct-child attention/terminal event, "
                "cancel/deadline control, recovery judgment, or an explicit one-shot checkpoint "
                "wakes exactly once. Every wake carries `sleep` (measured over this whole call), "
                "`leaf_live_input` (the route's declared live-input capability, 'unknown' when "
                "unread) and, once the prompt-cache horizon has passed since your last model "
                "response, one cache_horizon_note. A run that asks its "
                "user a question returns IMMEDIATELY as status='waiting_on_user' with "
                "the full question set (interaction/question ids ride WHOLE, never "
                "truncated): answer it with delegate_answer, or raise it with the "
                "escalate verb (parent-first) and keep waiting (a question with a "
                "timeout_at benign-declines "
                "at the engine timeout; timeout_at=null waits until answered); the "
                "payload's continuation=same_session fact means an answer (free_text "
                "included, e.g. a peer's original you relay) resumes THIS session, each "
                "resumed turn a paid round, while an input_required terminal names "
                "continuation=new_physical_run: answer it with delegate_start(subagent_id=..., continue_from=<run_id>, "
                "prompt=<the answers>). An unfinished "
                "terminal carries the engine's resumable block (cause, resetsAt, carriers) and one continuity line "
                "per continued try (carrier, accounts, memory, attested model). A "
                "large terminal result is delivered as a bounded preview plus an "
                "artifact: read output_delivery and finish reading the artifact before "
                "you rely on it. A delegate_message receipt is reconciled HERE: timeline "
                "rows carrying its messageId (message.* receipts and the harness status "
                "row whose outcome reads delivered) carry messageId and outcome."
            ),
            "parameters": {"type": "object", "required": ["run_id"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "since_seq": {"type": "integer", "description": "Event cursor: advances past it are recorded as progress."},
                "checkpoint_after_sec": {"type": "integer", "description":
                    "Optional one-shot future inspection time. Requires checkpoint_reason; "
                    "a real earlier wake consumes it. Omit both for no checkpoint (0 with an empty reason also means none)."},
                "checkpoint_reason": {"type": "string", "description":
                    "Why one proactive inspection is worth a model call. No repeating cadence."},
            }},
        }, _published_entry(_delegate_wait_entry), timeout_sec=operation_window_sec(get_task_abs_ceiling_sec()) + 120),
        ToolEntry("delegate_cancel", {
            "name": "delegate_cancel",
            "description": (
                "Cancel a stuck or misdirected delegated run, never just to hurry it. Partial artifacts "
                "are retained, not a finished verdict/product. Only confirmed proves a terminal receipt; "
                "requested, failed and containment_fault_run_may_still_be_live may still be running. "
                "After verified settlement, preserve useful work with delegate_start(subagent_id=..., continue_from=<run_id>, "
                "prompt=<corrections>), following that tool's selection/authority rules. The engine reuses "
                "the session where possible, else retained evidence; some work may need repeating."
            ),
            "parameters": {"type": "object", "required": ["run_id"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "reason": {"type": "string", "description": "Why you are stopping it."},
            }},
        }, _published_entry(_delegate_cancel), timeout_sec=120),
        ToolEntry("delegate_answer", {
            "name": "delegate_answer",
            "description": (
                "Answer a delegated run's pending interactive question — the "
                "status='waiting_on_user' payload from delegate_wait names the "
                "interaction_id and its questions. Only the task that started the run "
                "may answer. Policy: answer from the task context you already hold; a "
                "question ABOVE your authority (spending money, changing scope, "
                "external actions) is not yours to guess — escalate it with the "
                "escalate verb (parent-first; the reply reaches your mailbox on a "
                "later round and you relay it back here) and keep waiting; an "
                "unanswered question with a "
                "timeout_at benign-declines at the engine timeout (the run continues "
                "on stated assumptions), while timeout_at=null waits until answered. "
                "Typed outcomes: delivered; already_resolved (the run moved on — do "
                "not re-post); not_found; rejected (a definite engine refusal of "
                "these rows — HTTP 400/409/413/422 only — fix them); "
                "subscription_window_exhausted (a distinct outcome carrying reset_at "
                "— the answer did NOT land; retry the SAME answers after reset_at); "
                "delivery_unknown "
                "(transport died mid-answer — re-check with delegate_wait and NEVER "
                "post a different answer for the same interaction). A run on a route "
                "without a mid-run question channel that ENDS needing input "
                "(outcome_facts.reason=input_required) is answered with "
                "delegate_start(subagent_id=..., continue_from=<run_id>, prompt=<the answers>) "
                "— there is no rerun/decision verb, and custody stays with you."
                " For an over-budget work order, pass the host-verified "
                "source_response envelope alongside the ordinary answer; the host "
                "checks its exact canonical range before recording coverage."
            ),
            "parameters": {"type": "object",
                           "required": ["run_id", "interaction_id", "answers"],
                           "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "interaction_id": {"type": "string", "description":
                    "The interaction being answered, from the waiting_on_user payload."},
                "answers": {"type": "array", "items": {"type": "object", "properties": {
                    "question_id": {"type": "string", "description":
                        "The question's id from the waiting_on_user payload."},
                    "selected_labels": {"type": "array", "items": {"type": "string"},
                                        "description": "Labels of the chosen option(s)."},
                    "free_text": {"type": "string", "description":
                        "Free-text answer; omit when options were selected."},
                }, "required": ["question_id"]}, "description":
                    "One row per question you are answering."},
                "source_response": {"type": "object", "description":
                    "Only for a partial work-order run: exact canonical source range "
                    "receipt. The host verifies schema=1, kind=source_response, the "
                    "full brief SHA, canonical selector, and text at start_char:end_char "
                    "before delivering it."},
            }},
        }, _published_entry(_delegate_answer), timeout_sec=120),
        ToolEntry("delegate_message", {
            "name": "delegate_message",
            "description": (
                "Place one live message into a delegated run's RUNNING turn (a "
                "correction, a new fact, a redirection) without cancelling it. Only the "
                "task that started the run may send. Capability-gated, never by harness "
                "name: the route's catalog row declares liveInput (mid_turn / "
                "next_tool_boundary / none) and the engine must list the operation; "
                "otherwise the typed outcome is unsupported and nothing is sent. Typed "
                "outcomes mirror the engine: delivered (the harness consumed it in the "
                "live turn; obedience unproved); accepted (the acceptance boundary was "
                "observed; consumption unproved until a timeline row with this message_id reads outcome=delivered); "
                "rejected (an explicit refusal of THIS submission — see reason); "
                "not_active (no live target: terminal/settled run, turn gap, attempt "
                "mismatch, or a PENDING question — answer that with delegate_answer); "
                "unsupported; delivery_unknown (it MAY have landed); not_found. Every "
                "result returns message_id, the delivery identity: pass it back ONLY to "
                "retry the SAME text after delivery_unknown (the engine replays the "
                "stored receipt instead of delivering twice); after any other outcome a "
                "new message needs a NEW id (omit message_id). A message steers only the current "
                "attempt: the host never re-injects it (a continuation keeps it only as the engine's "
                "session history or evidence), and it "
                "is reconciled on the delegate_wait timeline (message.* rows)."
            ),
            "parameters": {"type": "object", "required": ["run_id", "text"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "text": {"type": "string", "description":
                    "The message, verbatim, as the harness will read it mid-turn "
                    "(non-empty; the engine bounds its length)."},
                "message_id": {"type": "string", "description":
                    "ONLY the message_id a previous delivery_unknown result returned, to "
                    "replay that exact message under its original key. Omit for a new "
                    "message; never invent one."},
            }},
        }, _published_entry(_delegate_message), timeout_sec=120),
    ]


__all__ = ["get_tools"]


# v7next F2 (D07): moved spans live in their owner leaf; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.delegate_terminal_evidence import (  # noqa: E402, F401 -- intentional public re-exports
    _NESTED_HOME_NOTE,
    _NO_BOUNDARY_NOTE,
    _access_evidence,
    _containment_breach,
    _containment_evidence,
    _delivered_terminal_payload,
    _record_containment,
    _reported_cost,
    _terminal_payload,
)
