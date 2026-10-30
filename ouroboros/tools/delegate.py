"""Nanny tools: run a subagent's cognition on an already-paid subscription session.

An ordinary nanny supervises independently custodied subscription leaves. Starts,
waits, cancellation, live interactions and terminal plan answers share this tool
family. Host-derived authority and explicit per-capture integration stay separate
from the harness's claims; a plan may implement only by its recorded starter.

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
    session_design_navigation,
    session_host_references,
)
from ouroboros.subagent_work_order import session_service_directories
from ouroboros.delegate_source_coverage import (
    add_terminal_source_verification,  # noqa: F401  (leaf reads it through _delegate() at call time)
    prepare_work_order_start_binding,
    record_started_custody,
)
from ouroboros.delegate_reply import REPLY_TO_SCHEMA
from ouroboros.delegate_supervision import delegate_wait_entry as _delegate_wait_entry
from ouroboros.delegate_start_instructions import (
    HOST_FACTS_INSTRUCTION as _HOST_FACTS_INSTRUCTION,
    HOST_INSTRUCTIONS as _HOST_INSTRUCTIONS,
    SHARED_ENVIRONMENT_INSTRUCTION as _SHARED_ENVIRONMENT_INSTRUCTION,
    UNPROVEN_BOUNDARY_INSTRUCTION as _UNPROVEN_BOUNDARY_INSTRUCTION,
    access_instruction,
    append_coordination_context,
    directory_copy_binding_instruction,
    execution_binding_fingerprint,
    execution_binding_text,
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
from ouroboros.delegate_directory import (
    blocked_geometry_refusal, git_directory_options_refusal, in_place_refusal, in_place_start, is_in_place_run,
    prepare_directory_execution, snapshot_checkout, started_receipt)
from ouroboros.delegate_registration_policy import resolve_registration
from ouroboros.delegate_documents import NO_DOCUMENTS, attach_selected_documents, attachment_parameters
from ouroboros.subagent_worktrees import registered_checkout
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
from ouroboros.delegate_continuation import NO_CONTINUATION, start_binding, with_run_binding
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
    _snapshot_facts,
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
_RETRY_HINT = ("to retry THIS start call use delegate_start(retry_of=pending_invocation_id, prompt=''), "
               "which replays its recorded request; a plain call starts a NEW run")


def _host_instructions(authority: "DelegatedRunShape", assignment: str = "",
                       payload_skill: str = "", coordination_context: str = "") -> str:
    """The system-prompt text this run's shape earns. One builder, no dialect.

    ``assignment`` is the host-authored contract block (``_assignment_instructions``);
    appended last so the prohibitions stay the opening statement. A payload run
    (``payload_skill`` non-empty) gets the truthful variant: editing the selected
    skill's user-authored files IS the assignment (gate fix 3).
    """
    text = payload_host_instructions(_HOST_INSTRUCTIONS, payload_skill) if payload_skill else _HOST_INSTRUCTIONS
    text += _HOST_FACTS_INSTRUCTION  # the engine's host facts, once, for every delegated session
    text += "\n\n" + _SHARED_ENVIRONMENT_INSTRUCTION
    if authority.delegated:
        text += (
            " No filesystem sandbox is requested; scoped HOME selects native state "
            "and credentials, not filesystem confinement. Report actual access "
            "honestly and do not claim to be sandboxed."
            if authority.access == "full" else _UNPROVEN_BOUNDARY_INSTRUCTION
        )
    text += access_instruction(authority.access, investigation=authority.investigation)  # powers, not a wider assignment
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
    there; ``_mutation_authority`` validates that same physical target. Without
    that authority the captured access still runs, as an evidence-only
    investigation in a private copy (#1609). Observe and a Presence ceiling keep
    their read-only research.
    """
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tool_access import _TOP_LEVEL_PRINCIPAL_PROFILES, active_tool_profile, project_room_lens_dir
    from ouroboros.consciousness_authority import is_observe_origin
    from ouroboros.presence_authority import presence_ceiling_allows_delegated_surface, presence_ceiling_from_context

    profile = active_tool_profile(ctx)
    if is_observe_origin(getattr(ctx, "task_metadata", {})):
        # Observe can use the delegated harness for research, but never turns it
        # into an in-place writer merely because the caller omitted/raised access.
        return delegated_run_shape(False, "readonly")
    mutating = profile in ("acting_subagent", "external_workspace_task") or (
        profile in _TOP_LEVEL_PRINCIPAL_PROFILES and project_room_lens_dir(ctx) is not None
    )
    if mutating:
        constraint = getattr(ctx, "task_constraint", None)
        surface = str(getattr(constraint, "surface", "") or "external_workspace")
        mutating = presence_ceiling_allows_delegated_surface(ctx, surface)
    presence_bound = presence_ceiling_from_context(ctx) is not None
    return delegated_run_shape(mutating, "readonly" if not mutating and presence_bound else access)


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
                   continuation: Optional[Dict[str, Any]] = None, ephemeral: bool = False,
                   service_roots: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """The POST body for one delegated run, built from the derived SHAPE (its ONE author:
    the shape decides the mode and the delegated marker; nothing here re-derives them).

    ``seconds`` and ``instructions`` arrive PRE-BUILT: a transport retry must present a
    byte-identical body for the engine's replay match, so the caller decides whether to
    recompute or replay them (the retry path replays the stored canonical body verbatim).
    ``continuation`` carries the engine's ``continueFrom`` (+ ``continueCarrier``) keys;
    ``service_roots`` the negotiated ``auxiliaryWritableRoots`` grant (``session_service_directories``).
    """
    target = route.resolved_target()  # ABI-4: one typed read; strings only at the wire
    request: Dict[str, Any] = {
        "prompt": text,
        # The compiled instructions disclose requested isolation, never claim confinement.
        "instructions": instructions,
        # Engine `auto` can fall back to a paid key invisibly; request the claimed substrate.
        "authPreference": "subscription",
        # The run shape follows the derived authority: one predicate decides what this child may do.
        "mode": authority.mode,
        "scope": {"kind": "project", "root": root, **({"ephemeral": True} if ephemeral else {})},
        # Pin via one-element `harnesses` pool (`primaryHarness` only fronts the auto-pool):
        # the child rides THIS route or the start refuses typed.
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
    if service_roots and request.get("execution"):  # a grant only ever extends an existing live execution
        request["execution"]["auxiliaryWritableRoots"] = list(service_roots)
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


def _prompt_cache_ttl_request(request, route, gateway, drive, continuation_token):
    """The prompt-cache lifetime this start negotiates; a continuation carries the one its run had."""
    from ouroboros.delegate_custody_current import run as current_custody_run
    from ouroboros.effort_evidence import session_prompt_cache_ttl_request

    return session_prompt_cache_ttl_request(
        request, route_id=route.route_id, gateway=gateway,
        continuation_from=current_custody_run(drive, continuation_token) if continuation_token else None)


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
                            continue_carrier: Any = None, mode=None, override_plan_readiness=None) -> Tuple[str, Optional[ToolResult]]:
    """``(continuation_token, refusal)``: every refusal a start's ARGUMENTS earn before
    the daemon is touched, in their historical order. Each is a definite no-run: an
    empty prompt (a continuation may carry none: the engine resumes the stopped work
    and the host states the facts), a malformed exact-resource selector, a deadline
    already behind the nanny (``definitely_unrun`` = the producer's own no-run
    verdict, P2), and the continuation selector shapes one call cannot combine: a
    retry replays an old key byte-identically while a continuation is a NEW
    intention over a finished run, and a skill-payload selector run keeps its own
    target semantics."""
    if mode not in (None, "ask", "plan", "agent") or (override_plan_readiness is not None and type(override_plan_readiness) is not bool):
        return "", _fail("delegate_start", "mode_invalid", "mode must be ask, plan or agent; override_plan_readiness must be boolean.", definitely_unrun=True)
    if retry_of and (mode is not None or override_plan_readiness is not None):
        return "", _fail("delegate_start", "retry_selector_conflict", "Replay the recorded mode and readiness choice; omit overrides.", definitely_unrun=True)
    if override_plan_readiness is not None and (mode != "agent" or not continue_from):
        return "", _fail("delegate_start", "plan_override_conflict", "override_plan_readiness requires continue_from and mode=agent.", definitely_unrun=True)
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
                            "continue_from starts a NEW run bound to a finished predecessor; retry_of replays a "
                            "pending invocation. Supply one of them.", definitely_unrun=True)
    from ouroboros.delegate_plan import payload_continuation

    if token and str(selector_root or "").strip() and not payload_continuation(ctx, token):
        return token, _fail("delegate_start", "continuation_resource_conflict",
                            "A payload selector can continue its plan thread; ordinary payload runs start fresh.", definitely_unrun=True)
    return token, None


def _start_actor_binding(ctx, drive, retry_token, continuation_token, text, canonical, source_request):
    """Select the actor against the exact source/authority binding used for custody and retry."""
    recovering = bool(retry_token)
    source_binding = prepare_work_order_start_binding(
        ctx, drive, retry_token, canonical, text,
        source_request,
    )
    work_order_source_request = source_binding["request"]
    work_order_fingerprint = source_binding["fingerprint"]
    actor, actor_refusal = prepare_delegate_start_actor(
        ctx, drive, recovering=recovering, invocation_id=retry_token, continuing=continuation_token,
        work_order_fingerprint=work_order_fingerprint, authority_fingerprint=source_binding["authority_fingerprint"],
    )
    if actor_refusal:
        return actor, {}, actor_refusal
    actor_facts = {key: str(actor.get(key) or "") for key in (
        "selected_subagent_id", "config_fingerprint", "work_order_fingerprint", "authority_fingerprint")}
    actor_facts.update(work_order_coverage=source_binding["coverage"], work_order_source_request=work_order_source_request)
    return actor, actor_facts, None


def _delegate_start(ctx: ToolContext, prompt: str, max_seconds: Optional[int] = None,
                    retry_of: Optional[str] = None, root: Optional[str] = None,
                    bucket: Optional[str] = None, skill_name: Optional[str] = None,
                    directory_strategy: Optional[str] = None, scope_paths: Optional[list] = None,
                    continue_from: Optional[str] = None, continue_carrier: Optional[str] = None,
                    workspace: Optional[str] = None, _resolved_binding: Any = None,
                    _canonical_work_order_fingerprint: str = "", _work_order_source_request: Any = None,
                    _coordination_context: str = "", _initial_bootstrap: bool = False, mode: Optional[str] = None,
                    override_plan_readiness: Optional[bool] = None, _plan_answers: Any = None,
                    attachment_paths: Optional[list] = None, _document_intent: str = "", _document_first_start: bool = False) -> ToolResult:
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import delegated_execution_workspace_root, delegated_run_shape, resolve_subagent_executor, route_health
    from ouroboros.delegate_plan import bind_start, plan_facts
    text = str(prompt or "")
    selector_root = str(root or "").strip()
    continuation_token, argument_refusal = _start_argument_refusal(
        ctx, text, selector_root, retry_of, bucket, skill_name, continue_from, continue_carrier, mode, override_plan_readiness)
    if argument_refusal:
        # This argument boundary precedes daemon access, provisioning and any start.
        return _replace_tool_result(argument_refusal, meta_updates={"operation_outcome": "completed_no_effect"})
    retry_token = str(retry_of or "").strip()
    recovering, seconds_basis = bool(retry_token), ""
    if not recovering:
        # Decided BEFORE the daemon is touched: a spent lifetime or a sub-second
        # deadline is a definite no-run, and the basis rides both custody rows.
        bound = bounded_max_seconds(ctx, max_seconds)
        if bound.refusal_code:
            return _fail("delegate_start", bound.refusal_code, bound.refusal_detail, definitely_unrun=True)
        seconds_basis = bound.basis
    drive = custody.custody_root(ctx)
    owned_project_id, project_persistent, readonly_root = "", False, ""
    invocation_id = snapshot_id = baseline_sha = target_root = authority_source = ""
    binding_fingerprint = ""
    processing_info: Dict[str, Any] = {}
    resource_ref, directory_options, continuation, documents = {}, {}, NO_CONTINUATION, NO_DOCUMENTS
    if recovering:  # the owned recorded request is validated before any fingerprint (#1614)
        binding, refusal = _resolve_retry_invocation(
            ctx, drive, retry_token, text, max_seconds=max_seconds, geometry=(directory_strategy, scope_paths))
        if refusal:
            return refusal
        text = text or binding.request_body["prompt"]  # omitted or '' replays the recorded prompt
    actor, actor_facts, actor_refusal = _start_actor_binding(
        ctx, drive, retry_token, continuation_token, text,
        _canonical_work_order_fingerprint, _work_order_source_request)
    if actor_refusal:
        return actor_refusal
    if recovering:
        (request_body, route, authority, root, key, project_id, owned_project_id,
         project_persistent, seconds, snapshot_id, target_root, baseline_sha,
         authority_source, resource_ref, processing_info, binding_fingerprint,
         seconds_basis, continuation, ttl_info) = binding
        invocation_id = retry_token
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
    if refusal := (blocked_geometry_refusal(ctx, authority, selector_root, directory_strategy, scope_paths)
                   or in_place_refusal(authority, selector_root, workspace, mode, directory_strategy, scope_paths)):
        return refusal
    ceiling_authority = authority
    if not recovering and mode is not None:
        authority = delegated_run_shape(not authority.investigation, authority.access, mode)
    if not recovering:
        assignment = "" if bool(actor.get("compiled_work_order")) else _assignment_instructions(ctx)
        payload_skill = str(((payload_auth or {}).get("resource_ref") or {}).get("skill_name") or "")
        instructions = _host_instructions(
            authority, assignment, payload_skill=payload_skill, coordination_context="" if continuation_token else _coordination_context,
        )
    access = authority.access
    start_attempted = recovering  # Recovery may already own a physical run.
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
                record_auth, root_error = _mutation_authority(ctx, ceiling_authority if authority.mode == "plan" else authority)
                if root_error:
                    return root_error
            invocation_id = custody.new_invocation_id()
            root = record_auth["target_root"]
            if continuation_token:  # gated from durable custody, before any snapshot exists
                continuation, refusal = start_binding(
                    ctx, drive, continuation_token, gateway=gateway, actor=actor, route=route, authority=authority,
                    target_root=str(record_auth.get("target_root") or ""), invocation_id=invocation_id, text=text,
                    coordination_context=_coordination_context, carrier=continue_carrier, source_binding=actor_facts,
                    canonical_work_order_fingerprint=str(_canonical_work_order_fingerprint or ""), requested_mode=mode,
                    override_plan_readiness=override_plan_readiness, plan_answers=_plan_answers)
                if refusal:
                    return refusal
            if authority.access == "readonly" or authority.investigation:  # a folderless Project's staged inputs
                from ouroboros.delegate_readonly_inputs import prepare_folderless_inputs
                try:
                    readonly_root, input_instruction = prepare_folderless_inputs(ctx, invocation_id)
                except (OSError, ValueError) as exc:
                    return _fail("delegate_start", "readonly_inputs_unavailable", str(exc), definitely_unrun=True)
                if readonly_root:
                    root = readonly_root
                    instructions += input_instruction
            preparation, run_binding = (ceiling_authority if authority.mode == "plan" else authority), ""
            if preparation.access in SESSION_ACCESS_PROFILES or continuation.snapshot:
                target_root = record_auth["target_root"]
                authority_source = record_auth["source"]
                if workspace == "in_place":
                    snapshot, snap_error, resource_ref, run_binding = in_place_start(continuation.snapshot, target_root)
                elif continuation.snapshot:
                    snapshot, snap_error = continuation.snapshot, None
                elif authority_source == "skill_payload":
                    snapshot, snap_error = _provision_payload_snapshot(
                        ctx, drive, record_auth, invocation_id)
                elif not (checkout := snapshot_checkout(ctx, target_root, preparation)):
                    try:
                        authority, directory_options, resource_ref = prepare_directory_execution(
                            ctx, gateway, target_root, authority, directory_strategy, scope_paths, readonly_root)
                    except ValueError as exc:
                        return _fail("delegate_start", "directory_execution_unavailable", str(exc), definitely_unrun=True)
                    snapshot, snap_error = None, ""
                else:
                    if error := git_directory_options_refusal(checkout, directory_strategy, scope_paths):
                        return _fail("delegate_start", "directory_execution_unavailable", error, definitely_unrun=True)
                    snapshot, snap_error = _provision_snapshot(ctx, drive, checkout, invocation_id, start=target_root)
                if snap_error:
                    _settle_refused_provision(ctx, gateway, snap_error, invocation_id, history_facts)
                    return snap_error
                if snapshot is not None:
                    snapshot_id, baseline_sha, root = snapshot.snapshot_id, snapshot.baseline_sha, snapshot.path
                    resource_ref = dict(continuation.facts.get("resource_ref") or record_auth.get("resource_ref") or {})
            if authority.mode == "plan":
                target_root, authority_source = record_auth["target_root"], "readonly"
            execution_root = (root if directory_options.get("isolation") == "live" else "") if directory_options else delegated_execution_workspace_root(gateway, authority, root)
            scope_root = (readonly_root or target_root) if execution_root or directory_options or (snapshot and authority.mode == "plan") else root
            service = session_service_directories(ctx, gateway, route, authority, snapshot=snapshot, payload=authority_source == "skill_payload", execution_root=execution_root or root)
            if snapshot is not None:
                binding_fingerprint = execution_binding_fingerprint(execution_root or root, target_root)
                run_binding = execution_binding_text(execution_root or root, target_root, binding_fingerprint,
                                                     evidence_root="" if authority_source == "skill_payload" else None, granted_scratch=service.scratch)
            elif directory_options and directory_options.get("isolation") == "envelope":
                binding_fingerprint = execution_binding_fingerprint("", target_root, "directory_copy")
                run_binding = directory_copy_binding_instruction(target_root, binding_fingerprint, investigation=authority.investigation)
            instructions += run_binding  # and the same text in a continuation's prompt, which a resumed session reads (#2513)
            continuation = with_run_binding(continuation, authority, run_binding, execution_root or scope_root)
            project_id, owned_project_id, project_persistent, ephemeral = resolve_registration(
                gateway, scope_root, execution_root, authority.access, minted=bool(readonly_root) or registered_checkout(scope_root))
            if directory_options or is_in_place_run({"resource_ref": resource_ref}):
                project_persistent = True
            if authority.access == "full":
                gateway.ensure_full_access(scope_root)
            seconds = bound.seconds
            documents = attach_selected_documents(ctx, gateway, route.route_id, attachment_paths, _document_intent, _document_first_start)
            instructions += session_host_references(ctx, target_root=record_auth["target_root"], source_root=root, private_copy=snapshot is not None, payload_skill=payload_skill, service=service)
            request_body = _start_request(ctx, route, authority, scope_root, continuation.prompt or text,
                                          seconds, instructions, execution_root, continuation=continuation.request, ephemeral=ephemeral, service_roots=service.roots,
                                          **({"directory_options": directory_options} if directory_options else {}))
            request_body, processing_info = _processing_start_request({**request_body, **documents.request}, actor, gateway, route)
            continuation = bind_start(gateway, request_body, route, invocation_id, continuation, authority,
                                      ceiling_authority, actor_facts, target_root, root if snapshot else "")
            project_persistent = project_persistent or bool(continuation.custody.get("thread_id"))
            ttl_info = _prompt_cache_ttl_request(request_body, route, gateway, drive, continuation_token)
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
            drive, actor_ctx=ctx, initial_bootstrap=_initial_bootstrap and not recovering,
            run_id="", task_id=str(getattr(ctx, "task_id", "") or ""),
            idempotency_key=key, invocation_id=invocation_id,
            max_seconds=seconds, max_seconds_basis=seconds_basis, request=request_body, project_id=project_id,
            project_owned=bool(owned_project_id), project_persistent=project_persistent, route=route.route_id,
            root_task_id=str(lineage.get("root_task_id") or ""), parent_task_id=str(lineage.get("parent_task_id") or ""),
            **snapshot_facts,
            execution_root=(root if snapshot_id or resource_ref.get("strategy") in ("direct", "in_place") else ""),
            # Recovery proves the original actor and compiled brief before adoption.
            **actor_facts,
            processing=processing_info,
            prompt_cache_ttl=ttl_info, **documents.custody,
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
        def submit():
            nonlocal start_attempted
            start_attempted = True  # The durable request is not physical submission.
            return gateway.start_run(request_body, idempotency_key=invocation_id)
        handle = run_operation(ctx, submit)
        run_id = str(handle.get("runId") or handle.get("jobId") or "")
        if not run_id:
            return _fail("delegate_start", "queued_without_run_id",
                         f"Claudexor returned a queued handle without a run id: {handle!r}",
                         pending_invocation_id=invocation_id, retry_hint=_RETRY_HINT,
                         **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent, history_facts=history_facts,
                             definite_refusal=False, reason="queued_without_run_id", invocation_id=invocation_id))
    except BaseException as exc:
        typed = isinstance(exc, (ClaudexorUnavailable, OwnerPauseRefused))
        code = getattr(exc, "code", str(exc)) if typed else f"pre_custody_exit_{type(exc).__name__}"
        status = int(getattr(exc, "status_code", 0) or 0) if typed else 0
        # A request row is not a POST; an entered callable stays unknown unless
        # its typed refusal proves otherwise. Recovery retains its original fate.
        definite = not start_attempted or 400 <= status < 500 or (
            isinstance(exc, OwnerPauseRefused) and not recovering)
        try:
            cleanup = _retire_orphaned_registration(
                ctx, gateway, owned_project_id, project_persistent=project_persistent, history_facts=history_facts,
                definite_refusal=definite, reason=code,
                invocation_id=invocation_id, snapshot_id=("" if recovering else snapshot_id))
        except Exception:
            log.warning("Start cleanup failed for %s", invocation_id, exc_info=True)
            cleanup = {"project_id": owned_project_id, "project_retired": False,
                       "project_retention_reason": "cleanup_unconfirmed"}
        if not typed:
            raise
        # Only the caller identifies a retry: unknown returns the original token;
        # definite refusal retires it, so the next intention gets a new identity.
        pending = ({} if definite or not invocation_id else
                   {"pending_invocation_id": invocation_id, "retry_hint": _RETRY_HINT})
        return _fail("delegate_start", code, str(exc), executor="blocked",
                     **({"definitely_unrun": True} if not start_attempted else {}),
                     reset_at=getattr(exc, "reset_at", ""), **pending, **cleanup)
    finally:
        gateway.close()
    # A start whose custody row did not land does not wear the plain name: the run
    # is live and only THIS process knows it exists (the uncustodied-run leak).
    durable = record_started_custody(
        drive, run_id, ctx, route, authority,
        key=key, access=access, root=root, seconds=seconds,
        invocation_id=invocation_id, project_id=project_id,
        project_owned=bool(owned_project_id), project_persistent=project_persistent,
        **actor_facts, **{k: v for k, v in snapshot_facts.items() if k not in plan_facts(snapshot_facts)},
        plan_facts=plan_facts(snapshot_facts), processing=processing_info, prompt_cache_ttl=ttl_info,
        capture_mode=("engine_directory" if resource_ref.get("workspace_kind") == "directory" else
                      _CAPTURE_DELEGATED_SNAPSHOT if snapshot_id else resource_ref.get("strategy", "")),
        max_seconds_basis=seconds_basis, row_effort=None if recovering else actor.get("row_effort"),
    )
    from ouroboros.tools.control import maybe_emit_delegated_run_fanout
    maybe_emit_delegated_run_fanout(ctx, run_id=run_id, route_id=route.route_id, objective=text, durable=durable)
    return _started_payload(handle, run_id, route, access, authority, root,
                            durable=durable, recovering=recovering, invocation_id=invocation_id,
                            snapshot_id=snapshot_id, target_root=target_root, baseline_sha=baseline_sha,
                            resource_ref=resource_ref, processing=processing_info, prompt_cache_ttl=ttl_info,
                            continuation=continuation.facts, plan=plan_facts(continuation.custody),
                            max_seconds=seconds, max_seconds_basis=seconds_basis,
                            engine_version=str(getattr(gateway, "engine_version", "") or ""),
                            snapshot_facts=_snapshot_facts(snapshot), documents=documents.facts,
                            effort_fact=({} if recovering else actor.get("effort_fact")))


def _started_payload(handle: Dict[str, Any], run_id: str, route: Any, access: str,
                     authority: "DelegatedRunShape", root: str, *, durable: bool,
                     recovering: bool, invocation_id: str, snapshot_id: str, target_root: str,
                     baseline_sha: str, engine_version: str = "", resource_ref=None, processing=None,
                     prompt_cache_ttl=None, continuation=None, max_seconds: int = 0, max_seconds_basis: str = "",
                     snapshot_facts=None, effort_fact=None, plan=None, documents=None) -> ToolResult:
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
            (" This run is an evidence-only investigation in a PRIVATE COPY: its changes are kept as evidence, "
             "never applied (integrate_delegated_patch refuses); never let it commit." if authority.investigation else
             " This run edits a PRIVATE SNAPSHOT of your write root, not the shared "
             "tree: at terminal its diff is captured for you, and NOTHING lands in "
             "the shared tree until you explicitly integrate_delegated_patch(run_id="
             "...) to apply or reject it — read the captured diff before you claim "
             "it, and never let the run commit.") + " The requested native access profile "
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
        **(plan or {}),
    }
    if not durable:
        payload["pending_invocation_id"] = str(invocation_id or "")
    # Optional receipts expose the captured processing, continuation and snapshot
    # facts; copy the continuation binding before publishing it to the nanny.
    for key, value in (("processing", processing), ("prompt_cache_ttl", prompt_cache_ttl),
                       ("continuation", dict(continuation or {})), ("snapshot", snapshot_facts), ("documents", documents)):
        if value:
            payload[key] = value
    if snapshot_id:
        # The C1 binding, stated where the nanny can read it: the run edits the
        # EXECUTION snapshot; the authority target receives nothing until apply.
        payload["workspace"] = "copy"
        payload["execution_root"] = root
        payload["authority_target_root"] = target_root
        payload["baseline_id"] = baseline_sha
        if not (snapshot_facts or {}).get("reused_from_run"):  # a reused snapshot's manifest is its first run's
            payload["baseline_manifest_read"] = {"root": "artifact_store", "path": f"delegated_runs/{snapshot_id}/baseline_manifest.json"}
    if isinstance(resource_ref, dict) and (resource_ref.get("workspace_kind") == "directory"
                                           or is_in_place_run({"resource_ref": resource_ref})):
        started_receipt(payload, resource_ref, target_root, durable=durable, investigation=authority.investigation)
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


def _retire_orphaned_registration(ctx: ToolContext, gateway: Any, project_id: str, *,
                                  definite_refusal: bool, reason: str,
                                  project_persistent: bool = False,
                                  invocation_id: str = "",
                                  snapshot_id: str = "", history_facts: Optional[dict] = None) -> Dict[str, Any]:
    """Retire a registration this start created but never bound to a run.

    Only a proven unsent start or a definite negative daemon answer permits retirement. A transport
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
        # pending invocation names it durably; custody GC reconciles it once the owning task is proven over.
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
    from ouroboros.config import operation_window_sec, resolve_task_execution_timeout_sec
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
    # Seconds of the task's FINITE execution bound still unspent (``None`` = unlimited):
    # the narrower of its API/CLI ``execution_timeout_sec`` and the global ceiling (#939).
    # Cumulative EXECUTION time decides, never a reset clock: the live model-wait
    # owner's window (offset plus elapsed minus the quota union minus the budget-paused
    # carrier) when this task's is bound, else the same arithmetic over the
    # original ``task_started_at`` and the paused carrier the resume handed over
    # (quota waits unknown here count as execution — the narrower direction).
    # No recorded start uses the operation-bounded ceiling; a failed clock read
    # is a typed unknown-lifetime refusal, never a fresh finite window.
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    lifetime_left: Optional[float] = resolve_task_execution_timeout_sec(metadata)
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
                    {"_execution_consumed_sec": metadata.get("_execution_consumed_sec"),
                     "started_at": float(started), "budget_paused_sec": float(paused or 0.0),
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
    socket that carried nothing, the engine's own ``DAEMON_BUSY_CODES`` answer to that GET)
    retains unknown observation and the same run, without model wake.
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
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result
    retained = retained_terminal_result(ctx, rid, retry_money=True)
    if retained is not None:
        return json.dumps(retained, ensure_ascii=False, indent=2)
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
        from ouroboros.delegate_registration_policy import INVESTIGATION_AUTHORITY

        authority = _Shape(access=entry.access, mode=entry.mode,
                           isolation=entry.isolation, delegated=entry.delegated,
                           investigation=entry.authority_source == INVESTIGATION_AUTHORITY)
    else:
        authority = _derive_authority(ctx)
    # FACTS against premature cancels: how long the run has actually been going and
    # what its cap really is, from the durable start row. A nanny that cannot see
    # these confabulates "exceeded the cap" out of its own impatience. Absent facts
    # (an old row, an unknown run) stay null — never invented.
    _started_ts, _run_max_seconds = custody.run_timing(custody.custody_root(ctx), rid)
    _started_at = parse_deadline_ts(_started_ts)
    _elapsed = lambda: None if _started_at is None else max(0, int(  # noqa: E731
        (_dt.datetime.now(tz=_dt.timezone.utc) - _started_at).total_seconds()))
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
        step = None
        if observation_only:
            _emit_external_wait_lease(
                ctx, rid, _external_wait_lease_until(ctx, read_window(), _started_at, _run_max_seconds),
                lease_id=_lease_id)
            from ouroboros import delegate_observe  # speech from the journal, attention from the engine
            step = delegate_observe.step(ctx, gateway, rid, window=window, since_seq=since_seq, read_window=read_window,
                                         elapsed_seconds=_elapsed(), max_seconds=_run_max_seconds or None)
            if step is not None and step.payload is not None:
                return step.payload  # a quiet beat: nothing needed the detail
        detail = step.detail if step is not None else progress.bounded_poll(gateway, rid, read_window())
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
                from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
                return json.dumps(collect_and_deliver_terminal(ctx, rid, detail, authority, entry, gateway),
                                  ensure_ascii=False, indent=2)
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
                    elapsed_seconds=_elapsed(), max_seconds=_run_max_seconds or None,
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
            # BOUNDED whether or not the window is spent (the clamp must bound the waiting,
            # not only the sleeping). The last poll of a spent window is bounded and never
            # skipped — terminal state and breach are judged on fresh data or not at all —
            # and a daemon too slow to answer it is this window's expiry; earlier, the typed
            # refusal propagates — a daemon that died mid-window is no quiet completed wait.
            left = deadline - time.monotonic()
            fresh = (progress.expiring_poll(gateway, rid) if left <= 0
                     else progress.bounded_poll(gateway, rid, left))
            if fresh is None:
                return _expired()   # unanswered AT expiry: expire on what is already held
            detail = fresh
    except ClaudexorUnavailable as exc:
        from ouroboros.tools.delegate_terminal_evidence import retained_terminal_after_gap
        retained = retained_terminal_after_gap(ctx, rid, str(exc.code))
        if retained is not None:
            return json.dumps(retained, ensure_ascii=False, indent=2)
        if observation_only and exc.observation_timeout:
            # The gateway's per-class reason, not the generic transport code: a
            # read bound that expired against a live daemon is a quiet hole and
            # nothing more, while a socket that carried no answer is the outage
            # the owner is told about once per episode.
            return json.dumps({
                "status": "observation_pending", "run_id": rid,
                "reason": exc.observation_reason or exc.code, "detail": str(exc),
                "waited_sec": time.monotonic() - started, "answered": exc.observation_answered,
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
                "Start a run on the owner's configured agent-session (subscription) harness and supervise it as its "
                "NANNY; API actors are scheduled with schedule_subagent instead. "
                "Subscription is requested, not guaranteed free: expired auth, paid routes or auth fallback "
                "may bill. Read delegate_wait's terminal cost (zero, billed, estimated or undisclosed); "
                "time, quota and a worker slot are also spent. Root, route and access inherit your authority and cannot widen; "
                "access only lowers native rights. A run lives in engine-scoped native state: it may not see the owner's own sessions "
                "and is no courier to them. Read-only runs only read and answer. Mutating Git runs edit a PRIVATE SNAPSHOT (or, with workspace='in_place', your own write root), never commit. Ordinary folders use "
                "directory_strategy=direct (default) or copy with scope_paths. Direct edits are already "
                "on site, without full rollback. Snapshot/copied results reach your tree only through "
                "integrate_delegated_patch(run_id=..., decision='apply'|'reject'); first inspect the captured "
                "diff in delegate_wait.workspace_capture (full file manifest and bytes). "
                "A TOP-LEVEL task may select one existing user-managed skill via root='skill_payload', "
                "bucket and skill_name, within existing authority. It edits a private standalone snapshot; "
                "the live payload stays unchanged until integration. Apply stales prior review: run "
                "skill_preflight and skill_review. Create a new skill's manifest first. Seeded native is "
                "system-repo territory; markerless native is logical external. "
                "Returns run_id for delegate_wait/delegate_cancel. Output is a claim to verify; host "
                "verification receipts remain yours. Missing/unavailable routes refuse typed: explicitly "
                "select a configured alternative, wait, narrow or report blocked. Direct starts require "
                "subagent_id. Fresh configured work starts its leaf before the first round; restored work follows "
                "its continuation receipt. Replay an unresolved (unknown-outcome) start with retry_of, without a new "
                "selector; a new prompt is a new run. Replacement with prompt='' requires "
                "verified cancellation/terminal settlement or a typed refusal proving no run exists. "
                "Continue finished work with continue_from (see arguments). This call requests no extra Claudexor review panel; "
                "new ordinary runs on engine 3.9.8+ default to none. The receipt names engine_version; "
                "older engines or a recovered historical run may retain earlier review behavior. Engine review, execution, "
                "integration and Ouroboros review gates remain separate."
            ),
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {
                "prompt": {"type": "string", "description":
                    "Direct start: a complete standalone task (why the work exists, decisions, material paths); "
                    "the session has none of my memory. The host appends my human's originating words verbatim. "
                    "The run starts in my write root; a path named here does not move it: for another "
                    "repository, schedule a child whose workspace is that repository. "
                    "Direct retry_of: omit or pass '' to replay the recorded request; other text must equal "
                    "the original call's. Configured-session retry/replacement: optional advisory coordination "
                    "only, never sent with retry_of; the host supplies the canonical work order."},
                "subagent_id": {"type": "string", "description":
                    "Exact agent_session id from Available subagents, required for direct starts including continue_from. "
                    "Omit for the configured snapshot and retry_of. Schedule API actors as recursive children instead."},
                "mode": {"type": "string", "enum": ["ask", "plan", "agent"], "description":
                    "Omit for access-derived execution. ask/plan are read-only. Plan artifacts return questions to the starter; "
                    "continue_from with mode=plan clarifies, mode=agent implements the frozen plan within its recorded ceiling."},
                "override_plan_readiness": {"type": "boolean", "description":
                    "Explicitly implement despite open or unverified plan questions; only with continue_from and mode=agent. Recorded before send."},
                "access": {"type": "string", "enum": list(SESSION_ACCESS_LOWERING), "description":
                    "Lower native access to readonly or workspace_write; omit to inherit the captured profile (new mutating "
                    "sessions default full; ordinary continue_from keeps the narrower prior access). Plan implementation uses its recorded ceiling. Explicit readonly "
                    "authority wins. Omit on retry_of."},
                "effort": {"type": "string", "enum": ["auto", *EFFORT_SCALE], "default": "auto", "description":
                    "The run's reasoning effort: auto/omitted = this room's primary clamped into its helper "
                    "bounds; outside those bounds it runs at the nearest bound. A row my human pinned or a level in "
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
                "workspace": {"type": "string", "enum": ["copy", "in_place"], "description":
                    "Git writers only. copy (default) gives the run a private copy whose diff you apply or reject. "
                    "in_place lets it work right in your own write root when that is safe, for example while you edit "
                    "nothing there yourself: its edits land at once, nothing is captured for apply or reject, undo is "
                    "through git, and two in-place writers in one folder mix their edits."},
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
                    "Continue a finished run after cap, subscription limit, crash/restart, cancel or input_required. "
                    "Prefer this when work is worth keeping. It must belong to your task line: yours, your confirmed "
                    "retry predecessor's, or the predecessor's task tree in an owner-created Continue root. "
                    "The engine reuses the session where possible, else briefs a new one with retained evidence; "
                    "some work may need repeating. A writer keeps its undisposed private snapshot as one cumulative "
                    "patch, superseding the old capture. prompt carries only new work/answers/corrections and may "
                    "be empty. For resumable pool_exhausted or a limit with resetsAt, use "
                    "await_messages(wake_at=resetsAt) before continuing. Needs an observed terminal (not settled "
                    "money), not superseded, no ambiguous apply. Ordinary continuations never widen; plan implementation requires the starter's recorded ceiling. "
                    "Select subagent_id as for a start; configured sessions omit it. Never combine with retry_of. A continuation inherits no attached documents: pass attachment_paths again."},
                "continue_carrier": {"type": "string", "enum": ["auto", "packet"], "description":
                    "With continue_from: auto (default) reuses the session where possible; packet starts a new "
                    "session with retained evidence when the old session has gone astray. Omit for plan threads."},
                **attachment_parameters("Until a scheduled child's first run starts, omitted attachment_paths keep its parent's selection. A continuation or replacement inherits none: pass them again. Omit both with retry_of."),
                "retry_of": {"type": "string", "description":
                    "pending_invocation_id from an unknown start outcome (transport failure/lost response). "
                    "Replays that invocation's recorded request byte-identically under its original key to retrieve "
                    "an accepted run without duplicating it; prompt may be omitted. Omit subagent_id or receive a "
                    "typed conflict; omit the other arguments too: the recorded ones replay, and a new value is a "
                    "typed conflict. Never use for intended new work: a plain call creates a new invocation even "
                    "with identical prompt."},
                },
            },
        }, _published_entry(_delegate_start_entry),
           timeout_sec=120),
        ToolEntry("delegate_wait", {
            "name": "delegate_wait",
            "description": (
                "Sleep on a delegated run until it ends, asks a question or needs you. Quiet progress costs "
                "no model rounds: the host renews quiet transport windows with zero model calls, and journal "
                "progress streams to the human without waking you. An unreachable daemon is the "
                "same quiet renewal (typed reason daemon_unreachable, told to the owner once per "
                "outage); while such a read hangs, a finalize_now/hurry control is noticed only "
                "when it returns, up to ~60 s later rather than on the 3 s beat. "
                "An observed terminal, a new interaction, "
                "fault, addressed owner/task message, a direct-child attention/terminal event, "
                "cancel/deadline control, recovery judgment, or an explicit one-shot checkpoint "
                "wakes exactly once. "
                "A host-confirmed retry successor may retrieve a predecessor's proven terminal result; live control stays with its starter. "
                "Usable terminal output returns before pending host capture/accounting; "
                "rereading it is immediate, not a wait for that preparation. A wake's leaf_live_input is the "
                "route's declared live-input capability ('unknown' when unread). A run that asks its "
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
                "per continued try. A "
                "large terminal result is delivered as a bounded preview plus an "
                "artifact: read output_delivery and finish reading the artifact before "
                "you rely on it. A delegate_message receipt is reconciled HERE: timeline "
                "rows with its messageId (message.* receipts, or the harness status row "
                "reading delivered) carry its outcome."
            ),
            "parameters": {"type": "object", "properties": {
                "run_id": {"type": "string", "description": "One run id; exclusive with runs."},
                "runs": {"type": "array", "items": {"type": "string"}, "description": "Run ids to supervise; wake on any selected run. Exclusive with run_id."},
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
                "interaction_id and its questions. For terminal plan artifacts omit interaction_id; this creates a new plan answer turn on the same thread. Only the task that started the run "
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
                           "required": ["run_id", "answers"],
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
                "reply_to": REPLY_TO_SCHEMA,
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
                "correction, a new fact, a redirection) without cancelling it. The starter, "
                "ancestors and same-tree tasks may send context; only the starter controls "
                "answers, cancellation, continuation and integration. The host stamps and "
                "retains the frame before sending; peer messages appear in its journal on "
                "the starter's next receipt, without an extra wake. Capability-gated, never by harness "
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
                    "The message, retained verbatim inside a host-stamped JSON frame "
                    "(non-empty; the engine bounds its length)."},
                "reply_to": REPLY_TO_SCHEMA,
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
