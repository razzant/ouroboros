"""Nanny tools: run a subagent's cognition on an already-paid subscription session.

A delegated subagent is an ORDINARY Ouroboros subagent acting as a NANNY: it lives in
the task tree with its own deadline and authority, but instead of thinking on metered
API tokens it starts a Claudexor run, watches it, and brings the result home. Because
the nanny IS the host, verification receipts stay host-authored and the harness's
output is a claim, not proof.

Four verbs: ``delegate_start``, a time-bounded ``delegate_wait``,
``delegate_cancel``, and ``delegate_answer`` (a run's pending interactive question is
answered by its own nanny — owner decision 7=A, poltergeist phase B). There is still
no ``hurry`` — Claudexor's only control verb is ``cancel``, and cancelling a reviewer
destroys the verdict you wanted.

Read-only and mutating children share ONE nanny and ONE transport. The only difference
is the access profile the HOST derives from the calling task's authority (``readonly``
vs ``workspace_write``) and the run shape that follows from it; there is no second
pipeline and no second slot. The child gets a broker tool, never a shell, so it can ask
the host to run something but never choose with what powers.

Custody: the daemon token never leaves ``gateways.claudexor``; nothing here puts it in
a ToolContext, a child environment, or a harness sandbox. WHICH run belongs to WHICH
task is decided by ``ouroboros.delegate_custody`` against the durable event log, not by
a dict this process happens to still hold.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ouroboros import delegate_custody as custody
from ouroboros import delegate_progress as progress
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.subagent_work_order import (  # noqa: F401 - compatibility re-export
    _FIELD_CHARS as _ASSIGNMENT_FIELD_CHARS,
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
    append_coordination_context,
)
from ouroboros.subagent_runtime import (  # noqa: F401 - shared primitive re-export
    delegate_start_entry as _delegate_start_entry,
    exact_start,
)
from ouroboros.subagent_runtime import prepare_delegate_start_actor
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
from ouroboros.delegate_registration_policy import resolve_registration
from ouroboros.delegate_shared import (  # noqa: F401
    _emit,
    _fail,
    _owned_run,
)
# The C1 integration seam (mutation authority, execution snapshots, retry binding,
# terminal patch capture) lives in its own module (size gate); re-exported here
# (same objects) because sibling code and the tests address it on THIS surface.
# `_fail` is NOT re-imported from it — the one shared refusal author is
# `delegate_shared._fail`, which delegate_integration itself imports.
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


def _host_instructions(authority: "DelegatedRunShape", assignment: str = "",
                       payload_skill: str = "") -> str:
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
        text += _UNPROVEN_BOUNDARY_INSTRUCTION
    if assignment:
        text += "\n\n" + assignment
    return text


def _build_start_instructions(
    authority: "DelegatedRunShape", assignment: str = "", payload_skill: str = "",
    coordination_context: str = "",
) -> tuple[str, str]:
    """Build the bounded instruction field for a fresh physical start."""
    return append_coordination_context(
        _host_instructions(authority, assignment, payload_skill), coordination_context,
        instruction_budget_chars=_ASSIGNMENT_FIELD_CHARS,
    )


def _derive_authority(ctx: ToolContext) -> "DelegatedRunShape":
    """Derive the run shape from the task's own authority — one question, asked here.

    Host-derived, never model-supplied: the child asks the host to run something, and
    the host decides with what powers. Ouroboros asks for an access PROFILE and lets
    Claudexor pick the mechanism (fs sandbox, tool allowlist, ...) — no harness branch.

    The SHAPE itself belongs to ``subagents.delegated_run_shape``, which the dispatcher
    also reads: this function only answers "does this task hold a mutating surface",
    which is the one part that needs the live ``ToolContext``. Two authorities qualify
    (B5, owner 2=A): an ACTING CHILD with a valid write surface, and the ROOT of an
    EXTERNAL-WORKSPACE task — the root already holds write+shell inside the project,
    so its delegated runs carry the same mutating shape, bounded by the same
    workspace; ``_mutation_authority`` (``tools.delegate_integration``) validates the
    concrete target either way.
    """
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tool_access import active_tool_profile

    profile = active_tool_profile(ctx)
    mutating = profile in ("acting_subagent", "external_workspace_task")
    if mutating:
        from ouroboros.presence_authority import presence_ceiling_allows_delegated_surface

        constraint = getattr(ctx, "task_constraint", None)
        surface = str(getattr(constraint, "surface", "") or "external_workspace")
        mutating = presence_ceiling_allows_delegated_surface(ctx, surface)
    return delegated_run_shape(mutating)


def _presence_delegate_read_refusal(ctx: ToolContext) -> str:
    from ouroboros.presence_authority import presence_ceiling_allows_delegated_read

    if presence_ceiling_allows_delegated_read(ctx):
        return ""
    return _fail(
        "delegate_start",
        "presence_delegate_read_root_unselected",
        "This Presence profile did not select whole-root read access for the active repository, "
        "so a delegated harness cannot honestly be started inside that broader read surface.",
    )


# -- tools --------------------------------------------------------------------


def _start_request(ctx: ToolContext, route: "DelegationRoute", authority: "DelegatedRunShape",
                   root: str, text: str, seconds: int, instructions: str, execution_root: str = "") -> Dict[str, Any]:
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
    # credentialProfileId is the account pin (D-U5), reviewer-slot wire contract; strict
    # (D-U6). In the stored canonical body, so a retry_of replay stays byte-identical.
    for key, value in (("model", target.model_id), ("effort", target.effort), ("credentialProfileId", target.credential_ref)):
        if value:
            request[key] = value
    if seconds:
        request["maxSeconds"] = seconds
    return request


def _delegate_start(ctx: ToolContext, prompt: str, max_seconds: Optional[int] = None,
                    retry_of: Optional[str] = None, root: Optional[str] = None,
                    bucket: Optional[str] = None, skill_name: Optional[str] = None,
                    _resolved_binding: Any = None,
                    _canonical_work_order_fingerprint: str = "",
                    _work_order_source_request: Any = None,
                    _coordination_context: str = "") -> str:
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import delegated_execution_workspace_root, resolve_subagent_executor, route_health

    text = str(prompt or "").strip()
    if not text:
        return _fail("delegate_start", "empty_prompt", "prompt is required")
    selector_root = str(root or "").strip()
    selector_refusal = _payload_selector_refusal(selector_root, retry_of, bucket, skill_name)
    if selector_refusal:
        return selector_refusal
    if deadline_expired(ctx):
        # EXPIRED pre-daemon; definitely_unrun = the producer's own no-run verdict (P2).
        return _fail(
            "delegate_start", "task_deadline_expired",
            "This task's deadline has already passed, so a delegated run started now "
            "would outlive it by design. Finalize with what you have — do not start "
            "new work a deadline has already closed.", definitely_unrun=True,
        )

    drive = custody.custody_root(ctx)
    owned_project_id, project_persistent = "", False
    invocation_id = ""
    snapshot_id = ""
    baseline_sha = ""
    target_root = ""
    authority_source = ""
    resource_ref: Dict[str, Any] = {}
    selected_subagent_id = ""
    config_fingerprint = ""
    retry_token = str(retry_of or "").strip()
    source_binding = prepare_work_order_start_binding(
        ctx, drive, retry_token, _canonical_work_order_fingerprint, text,
        _work_order_source_request,
    )
    work_order_source_request = source_binding["request"]
    work_order_coverage = source_binding["coverage"]
    authority_fingerprint = source_binding["authority_fingerprint"]
    work_order_fingerprint = source_binding["fingerprint"]
    recovering = source_binding["recovering"]
    actor, actor_refusal = prepare_delegate_start_actor(
        ctx, drive, recovering=recovering, invocation_id=retry_token,
        work_order_fingerprint=work_order_fingerprint, authority_fingerprint=authority_fingerprint,
    )
    if actor_refusal:
        return actor_refusal
    selected_subagent_id = str(actor.get("selected_subagent_id") or "")
    config_fingerprint = str(actor.get("config_fingerprint") or "")
    work_order_fingerprint = str(actor.get("work_order_fingerprint") or "")
    authority_fingerprint = str(actor.get("authority_fingerprint") or "")
    if recovering:
        binding, refusal = _resolve_retry_invocation(ctx, drive, retry_token, text)
        if refusal:
            return refusal
        (request_body, route, authority, root, key, project_id, owned_project_id,
         project_persistent, seconds, snapshot_id, target_root, baseline_sha,
         authority_source, resource_ref) = binding
        invocation_id = retry_token
        payload_auth = None
    else:
        route = actor["route"]
        if selector_root:
            authority, payload_auth, payload_error = _payload_mutation_authority(
                ctx, drive, bucket, skill_name, _resolved_binding)
            if payload_error:
                return payload_error
        else:
            authority = _derive_authority(ctx)
            payload_auth = None
            if refusal := _presence_delegate_read_refusal(ctx):
                return refusal

    instructions = ""
    if not recovering:
        assignment = "" if bool(actor.get("compiled_work_order")) else _assignment_instructions(ctx)
        payload_skill = ""
        if isinstance(payload_auth, dict):
            payload_skill = str(
                (payload_auth.get("resource_ref") or {}).get("skill_name") or ""
            )
        instructions, instruction_error = _build_start_instructions(
            authority,
            assignment,
            payload_skill=payload_skill,
            coordination_context=_coordination_context,
        )
        if instruction_error:
            return instruction_error

    access = authority.access
    try:
        gateway = ensure_owned_gateway()
    except ClaudexorUnavailable as exc:
        resolution = resolve_subagent_executor("harness", route=route, unavailable_reason=exc.code)
        return _fail("delegate_start", exc.code, str(exc), executor=resolution.executor)

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

        if not recovering:
            if payload_auth is not None:
                record_auth = payload_auth
            else:
                record_auth, root_error = _mutation_authority(ctx, authority)
                if root_error:
                    return root_error
            invocation_id = custody.new_invocation_id()
            root = record_auth["target_root"]
            if authority.access == "workspace_write":
                # C1: the run executes in a PRIVATE snapshot of the authority target,
                # never in the shared tree. Provisioned (and durably registered)
                # BEFORE start; 3.8.1+ separates it from the stable project root.
                # A payload target gets the STANDALONE snapshot (the live payload
                # is never initialized as Git); a Git target keeps the worktree
                # snapshot byte-identically.
                target_root = record_auth["target_root"]
                authority_source = record_auth["source"]
                if authority_source == "skill_payload":
                    snapshot, snap_error = _provision_payload_snapshot(
                        ctx, drive, record_auth, invocation_id)
                else:
                    snapshot, snap_error = _provision_snapshot(ctx, drive, target_root, invocation_id)
                if snap_error:
                    return snap_error
                snapshot_id = snapshot.snapshot_id
                baseline_sha = snapshot.baseline_sha
                root = snapshot.path
                resource_ref = dict(record_auth.get("resource_ref") or {})
            execution_root = delegated_execution_workspace_root(gateway, authority, root)
            scope_root = target_root if execution_root else root
            (project_id, owned_project_id, project_persistent) = resolve_registration(
                gateway, scope_root, execution_root, getattr(authority, "access", ""))
            # The canonical ASSIGNMENT — prompt plus host-authored instructions —
            # is digested together: two starts whose prompts agree but whose
            # contract blocks differ are two different logical starts. The digest
            # is the LOOKUP identity only; the wire key stays the invocation id,
            # and a retry replays the STORED body byte-identically regardless.
            key = custody.idempotency_key(getattr(ctx, "task_id", ""), route.route_id,
                                          access, authority.mode, authority.isolation,
                                          root, text, instructions)
            seconds = _bounded_max_seconds(ctx, max_seconds)
            request_body = _start_request(ctx, route, authority, scope_root, text,
                                          seconds, instructions, execution_root)
        lineage = getattr(ctx, "task_metadata", {}) or {}
        lineage = lineage if isinstance(lineage, dict) else {}
        # Fresh payload run: busy check + durable write = ONE atomic claim (fix 5).
        requested, claim_refusal = claimed_start_request(
            drive, claim_target=(target_root if not recovering and authority_source == "skill_payload" else ""),
            actor_ctx=ctx, enforce_actor_idle=not recovering,
            run_id="", task_id=str(getattr(ctx, "task_id", "") or ""),
            idempotency_key=key, invocation_id=invocation_id,
            max_seconds=seconds, request=request_body, project_id=project_id,
            project_owned=bool(owned_project_id), project_persistent=project_persistent, route=route.route_id,
            # Lineage rides the request row so a run RECOVERED from a pending
            # invocation (P34R.2) can still attribute its ledger row to the tree.
            root_task_id=str(lineage.get("root_task_id") or ""),
            parent_task_id=str(lineage.get("parent_task_id") or ""),
            # The C1 isolation binding, durable BEFORE the POST: these name the
            # snapshot, baseline and authority target so a retry reproduces the
            # exact binding and the startup GC can see the pending snapshot.
            snapshot_id=snapshot_id, execution_root=(root if snapshot_id else ""),
            baseline_sha=baseline_sha, target_root=target_root,
            authority_source=authority_source, resource_ref=resource_ref,
            # Same pre-POST row as the request: crash recovery can prove the
            # exact configured actor and exact compiled brief before adopting.
            selected_subagent_id=selected_subagent_id,
            config_fingerprint=config_fingerprint,
            work_order_fingerprint=work_order_fingerprint,
            work_order_coverage=work_order_coverage,
            authority_fingerprint=authority_fingerprint,
            work_order_source_request=work_order_source_request,
        )
        if claim_refusal:
            reason = str(claim_refusal.get("reason") or "replacement_custody_unknown")
            detail = str(claim_refusal.get("detail") or "Actor start claim unavailable.")
            facts = {key: value for key, value in claim_refusal.items()
                     if key not in {"reason", "detail"}}
            return _fail(
                "delegate_start", reason, detail,
                **facts,
                **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent,
                    definite_refusal=True,
                    reason=reason, invocation_id=invocation_id, snapshot_id=snapshot_id,
                ),
            )
        if not requested:
            # The POST is CONDITIONAL on the durable request row: a run started
            # without it is live and unfindable if this worker dies. A fresh
            # start's registration is definitively retirable; a RETRY's project
            # belongs to the original attempt, whose POST may have bound a live
            # run — its fate stays unknown and its invocation stays pending.
            return _fail(
                "delegate_start", "start_request_row_unwritable",
                "The durable start-request row could not be written, so the run was "
                "NOT started: a run launched without its custody trail would be "
                "unfindable if this worker died. Fix the drive/event log and retry.",
                **({"definitely_unrun": True} if not recovering else {}), **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent,
                                                definite_refusal=not recovering,
                                                reason="start_request_row_unwritable",
                                                invocation_id=invocation_id,
                                                snapshot_id=("" if recovering else snapshot_id)))
        handle = gateway.start_run(request_body, idempotency_key=invocation_id)
        # A 202 answers with `jobId` and no `runId` when the run has not bound a run
        # dir inside the daemon's start timeout; `jobId` is a usable GET/control
        # handle — discarding it left a live run nobody could wait on or cancel.
        run_id = str(handle.get("runId") or handle.get("jobId") or "")
        if not run_id:
            # The POST SUCCEEDED, so a run is more likely live here than on the
            # refusal branch beside it — the registration is retained, not abandoned.
            return _fail("delegate_start", "queued_without_run_id",
                         f"Claudexor returned a queued handle without a run id: {handle!r}",
                         pending_invocation_id=invocation_id,
                         retry_hint="to retry THIS start call use "
                                    "delegate_start(prompt=..., "
                                    "retry_of=pending_invocation_id); a plain call starts a NEW run",
                         **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent,
                                                         definite_refusal=False,
                                                         reason="queued_without_run_id",
                                                         invocation_id=invocation_id))
    except ClaudexorUnavailable as exc:
        # A registration we created BEFORE the start must not outlive a failed start.
        # It used to be left behind with nothing anywhere naming its id.
        status = int(getattr(exc, "status_code", 0) or 0)
        definite = 400 <= status < 500
        # An UNKNOWN outcome hands back the retry token: only the caller can say
        # whether the next call is a retry of this intention or a new intention, and
        # without the token every next call is a new one. A definite refusal retires
        # the id, so no token rides a refusal.
        pending = ({} if definite or not invocation_id else
                   {"pending_invocation_id": invocation_id,
                    "retry_hint": "to retry THIS start call use "
                                  "delegate_start(prompt=..., "
                                  "retry_of=pending_invocation_id); a plain call starts a NEW run"})
        return _fail("delegate_start", exc.code, str(exc), executor="blocked",
                     reset_at=getattr(exc, "reset_at", ""), **pending,
                     **_retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent,
                                                     definite_refusal=definite,
                                                     reason=str(getattr(exc, "code", "")),
                                                     invocation_id=invocation_id,
                                                     snapshot_id=("" if recovering else snapshot_id)))
    except BaseException as exc:
        # EVERY pre-custody exit leaves a durable disposition, including the ones no
        # typed handler claims (a bug here, a timeout, a signal). NEVER retired: an
        # untyped exit says nothing about whether the POST reached the daemon, so a run
        # may be live against it. Named with a typed reason so the sweep's
        # pending-invocation recovery finds it, then re-raised — disclosure, not a swallow.
        _retire_orphaned_registration(ctx, gateway, owned_project_id, project_persistent=project_persistent,
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
        selected_subagent_id=selected_subagent_id,
        config_fingerprint=config_fingerprint, work_order_fingerprint=work_order_fingerprint,
        work_order_coverage=work_order_coverage,
        work_order_source_request=work_order_source_request,
        authority_fingerprint=authority_fingerprint, snapshot_id=snapshot_id,
        target_root=target_root, baseline_sha=baseline_sha,
        authority_source=authority_source, resource_ref=resource_ref,
        capture_mode=(_CAPTURE_DELEGATED_SNAPSHOT if snapshot_id else ""),
    )
    from ouroboros.tools.control import maybe_emit_delegated_run_fanout
    maybe_emit_delegated_run_fanout(ctx, run_id=run_id, route_id=route.route_id, objective=text, durable=durable)
    return _started_payload(handle, run_id, route, access, authority, root,
                            durable=durable, recovering=recovering,
                            invocation_id=invocation_id,
                            snapshot_id=snapshot_id, target_root=target_root,
                            baseline_sha=baseline_sha)


def _started_payload(handle: Dict[str, Any], run_id: str, route: Any, access: str,
                     authority: "DelegatedRunShape", root: str, *, durable: bool,
                     recovering: bool, invocation_id: str, snapshot_id: str, target_root: str,
                     baseline_sha: str) -> str:
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
            "it, and never let the run commit. It was ASKED to run under a scoped "
            "HOME and an OS-enforced boundary; whether the engine applied either is "
            "a per-run fact that delegate_wait reads back from the run's own "
            "artifacts. A host with no boundary mechanism runs it anyway and says "
            "so there."
            if authority.isolation == "live" else
            " This run cannot write anything: it reads and answers."
        )
    )
    payload = {
        "status": "started" if durable else "started_uncustodied",
        "run_id": run_id,
        "run_dir": handle.get("runDir"),
        "route": route.route_id,
        "model": route.model,
        "effort": route.effort,
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
        "note": note,
    }
    if not durable:
        payload["pending_invocation_id"] = str(invocation_id or "")
    if snapshot_id:
        # The C1 binding, stated where the nanny can read it: the run edits the
        # EXECUTION snapshot; the authority target receives nothing until apply.
        payload["execution_root"] = root
        payload["authority_target_root"] = target_root
        payload["baseline_id"] = baseline_sha
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _retire_orphaned_registration(ctx: ToolContext, gateway: Any, project_id: str, *,
                                  definite_refusal: bool, reason: str,
                                  project_persistent: bool = False,
                                  invocation_id: str = "",
                                  snapshot_id: str = "") -> Dict[str, Any]:
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
    if snapshot_id and definite_refusal:
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
                                          "definite": bool(definite_refusal)})
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


def _bounded_max_seconds(ctx: ToolContext, requested: Optional[int]) -> int:
    """Narrow-only: the delegated run may never outlive the nanny's own deadline.

    A caller must ask ``deadline_expired`` FIRST: an expired deadline cannot produce an
    honest bound at all, and this function's fallback is for a nanny that has NO
    deadline, never for one whose deadline is behind it.
    """
    from ouroboros.deadline_utils import deadline_remaining_sec

    remaining = int(max(0.0, deadline_remaining_sec(ctx)))
    try:
        asked = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        asked = 0
    candidates = [value for value in (asked, remaining) if value > 0]
    if candidates:
        # Clamp HERE too, not only on the fallback below: `max_seconds` is a model-supplied
        # tool argument with no maximum in its schema, so an explicit ask sailed past the
        # bound the fallback branch was careful about — the same defect, one branch over.
        return min(_CLAUDEXOR_MAX_SECONDS, min(candidates))
    # No positive bound is knowable: either the nanny has no deadline, or its deadline
    # has already passed. Omitting `maxSeconds` — the old behavior — handed the run
    # Claudexor's 7-day schema bound; the cap is damage limitation, and custody (the
    # durable start row plus reconciliation) is what actually stops an orphan.
    from ouroboros.config import get_task_abs_ceiling_sec

    # Claudexor bounds maxSeconds at 7 days (control.ts `.max(604_800)`), and the task
    # ceiling clamps only from BELOW — an owner who raises it past a week would make
    # every deadline-less start send an out-of-schema value.
    return min(_CLAUDEXOR_MAX_SECONDS, int(get_task_abs_ceiling_sec()))


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
    )


# The typed external-wait lease lives in `delegate_progress` (the wait-liveness
# module); re-bound here because the wait's own seams and the tests name it on
# this surface, exactly like the staged-output cluster above.
_external_wait_lease_until = progress.external_wait_lease_until
_emit_external_wait_lease = progress.emit_external_wait_lease


def _delegate_wait(ctx: ToolContext, run_id: str, wait_sec: Optional[int] = None,
                   since_seq: Optional[int] = None) -> str:
    """Time-bounded, progress-aware wait (docs/DEVELOPMENT.md "Timeout & Wait Control").

    HOLDS the window it was given. It returns early only on a terminal state or a
    containment fault; a journal-cursor advance past ``since_seq`` is RECORDED and
    streamed to the human live, and the model is woken once, at expiry, with the whole
    sequence in ``advances``. Returning on the first advance made the caller's window
    meaningless against a healthy run — the only path that ever consulted it was the
    SILENT one, so a streaming run cost a full-context round per event batch (measured:
    18 rounds, 861k prompt tokens, for a run that was doing fine). Progress is the
    JOURNAL cursor, so SSE ``: ping`` keepalives cannot masquerade as it.

    NARROW-ONLY, like ``_bounded_max_seconds``: the wait may not outlive the nanny's own
    deadline, minus the finalization grace it needs to answer at all. This tool is absent
    from ``_DEADLINE_CLAMPED_TOOLS`` (its ToolEntry value IS its outer bound), so nothing
    upstream cuts it — measured, a 2100s window against ten seconds of remaining deadline
    ran the full 2100s and slid the task past its deadline mid-tool, the defect that set
    built for ``web_search``. Clamping HERE keeps the graceful typed ``no_progress``
    return where the outer clamp delivers a thread-kill. Only "no deadline set" is left
    unclamped; a SPENT deadline clamps to the floor, the window is measured from before
    the connection, and every call is BOUNDED by what it has left (``progress.poll_bound``)
    so no read can outrun it as the 60s default could. Only the LAST poll of a spent
    window may go unanswered gracefully; a daemon that fails while the window still has
    time is the typed refusal it was, never a wait reported as quiet.
    """
    from ouroboros.config import get_delegate_wait_max_sec, get_delegate_wait_sec
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        ClaudexorUnavailable,
        pending_interactions as _cx_pending,
    )

    rid = str(run_id or "").strip()
    if not rid:
        return _fail("delegate_wait", "missing_run_id", "run_id is required")
    not_mine, entry = _owned_run(ctx, "delegate_wait", rid)
    if not_mine or entry is None:
        return not_mine or _fail("delegate_wait", "run_ownership_unknown", "custody unresolved", run_id=rid)
    ceiling = get_delegate_wait_max_sec()
    try:
        window = int(wait_sec) if wait_sec is not None else get_delegate_wait_sec()
    except (TypeError, ValueError):
        window = get_delegate_wait_sec()
    from ouroboros.deadline_utils import parse_deadline_ts, window_within_deadline

    window = window_within_deadline(ctx, max(1, min(window, ceiling)))

    # The clock starts HERE, before the connection: the window is a promise about how
    # long this CALL holds, and the opening handshake plus first poll are part of it.
    # Started after them, an unbounded connection could spend the whole deadline before
    # the window it was clamped into had begun.
    started = time.monotonic()
    deadline = started + window
    try:
        gateway = ClaudexorGateway()
        gateway.handshake(timeout_sec=progress.poll_bound(deadline - time.monotonic()))
    except ClaudexorUnavailable as exc:
        return _fail("delegate_wait", exc.code, str(exc), run_id=rid)

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
        ctx, rid, _external_wait_lease_until(ctx, window, _started_at, _run_max_seconds),
        lease_id=_lease_id)
    try:
        detail = progress.bounded_poll(gateway, rid, deadline - time.monotonic())
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
            if state in _TERMINAL_STATES:
                was_settled = bool(entry.settled)
                settlement = custody.settle_run(custody.custody_root(ctx), gateway, entry, detail)
                payload = _delivered_terminal_payload(ctx, rid, detail, authority, entry, gateway)
                payload["settlement"] = settlement
                # C1: a mutating run's changes live in its private snapshot until the
                # nanny explicitly integrates them. Captured HERE, durably, on every
                # terminal observation (idempotent), cancelled runs included — a
                # cancelled run's partial work is salvage material, not garbage.
                capture = _capture_terminal_patch(ctx, entry)
                if capture is not None:
                    payload["workspace_capture"] = capture
                # The «last delegated run» settings receipt (Subagents section):
                # requested vs applied model, written ONLY when THIS call performed
                # a SUCCESSFUL settlement — a later wait re-reading an already-settled
                # run must not re-date it (or replace a newer run as "last"), and a
                # settlement whose durable obligations failed must not mint a receipt
                # it would re-mint on every retry. The delegated REVIEW sessions never
                # pass here — they have their own receipt store
                # (reviewer_slot_last_execution.json).
                if not was_settled and bool(settlement.get("settled")):
                    from ouroboros.subagents import record_last_delegation
                    record_last_delegation(
                        route=entry.route_id, requested_model=entry.model,
                        applied_model=str(payload.get("model") or ""), run_id=rid,
                        selected_subagent_id=entry.selected_subagent_id,
                        # Applied = the settlement receipt's authRoute fact (never invented); requested replays off STARTED.
                        requested_profile=entry.profile_id,
                        applied_profile=str((summary.get("authRoute") or {}).get("profileId") or ""))
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
                from ouroboros.tools.control import cache_horizon_note
                _horizon = cache_horizon_note(ctx, time.monotonic() - started)
                if _horizon:
                    payload["cache_horizon_note"] = _horizon
                return json.dumps(payload, ensure_ascii=False, indent=2)
            if last_seq > baseline:
                # The STREAM is not collapsed — the TIMER is. Every advance reaches the
                # live progress surface the instant this loop sees it, so the human's
                # view stays as rich; what stops is waking the MODEL per event batch.
                # The emit is also the frame the supervisor's idle enforcer reads, which
                # a silently blocking wait would starve.
                progress.emit(ctx, rid, seen.record(detail, last_seq, int(time.monotonic() - started)))
                baseline = last_seq          # so the NEXT advance is counted once
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
                rendered = progress.rendered_window(
                    run_id=rid, state=state, last_seq=last_seq, window=window,
                    elapsed_seconds=(None if _started_at is None else max(0, int(
                        (_dt.datetime.now(tz=_dt.timezone.utc) - _started_at).total_seconds()))),
                    max_seconds=_run_max_seconds or None,
                    waiting_on_user=bool(summary.get("waitingOnUser")) or bool(pending),
                    pending_interactions=_bounded_interactions(pending) if pending else None,
                    detail=detail, seen=seen,
                    budget=tool_result_limit("delegate_wait"))
                from ouroboros.tools.control import cache_horizon_note
                _horizon = cache_horizon_note(ctx, time.monotonic() - started)
                return f"{rendered}\n\n{_horizon}" if _horizon else rendered

            if time.monotonic() >= deadline:
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
        return _fail("delegate_wait", exc.code, str(exc), run_id=rid)
    finally:
        _emit_external_wait_lease(ctx, rid, 0.0, lease_id=_lease_id)
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


def _delegate_cancel(ctx: ToolContext, run_id: str, reason: str = "") -> str:
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
    return json.dumps({
        "status": result["outcome"],
        "run_id": rid,
        "run_may_still_be_live": result["outcome"] != custody.CANCEL_CONFIRMED,
        "accepted": result["accepted"],
        "control_status": result["control_status"],
        "state": result["state"],
        "fault_reason": result["fault_reason"],
        "detail": result["detail"],
        "note": _CANCEL_NOTES.get(result["outcome"], ""),
    }, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    from ouroboros.config import get_task_abs_ceiling_sec

    return [
        ToolEntry("delegate_start", {
            "name": "delegate_start",
            "description": (
                "Start a delegated run on the owner's configured subscription harness and "
                "become its NANNY. Subscription execution is REQUESTED, so the usual case "
                "is no metered API money — but the actual spend is a fact of the finished "
                "run, not a promise of this call: it may come back zero, billed, "
                "estimated, or undisclosed (an expired session, a route that bills by "
                "construction, or an auth fallback all charge real money). Read the "
                "terminal `cost` block from delegate_wait before you treat this as free; "
                "it also costs time, quota and a worker slot. Your working root, "
                "access profile and route come from YOUR task authority; you cannot widen "
                "them, and there is no argument here that would let you try. If you hold "
                "a MUTATING shape the run executes in a PRIVATE SNAPSHOT of your write "
                "root — it never edits the shared tree in place. Its diff is captured at "
                "terminal (delegate_wait's workspace_capture block) and reaches your tree "
                "ONLY when you explicitly call integrate_delegated_patch(run_id=..., "
                "decision='apply'|'reject'); read the captured diff before applying, and "
                "never let the run commit inside its snapshot. If you are read-only it "
                "can only read and answer. "
                "A TOP-LEVEL task may instead select ONE exact installed user-managed "
                "skill payload with root='skill_payload' + bucket + skill_name: the "
                "selector chooses authority you already hold (it grants nothing), the "
                "run edits a private standalone snapshot of that payload, the LIVE "
                "payload stays byte-identical until you explicitly "
                "integrate_delegated_patch, and after an apply the skill's prior "
                "review is stale — run skill_preflight and skill_review as usual. "
                "The payload must already exist (create a NEW skill's manifest first). "
                "Seeded native stays system-repo territory; markerless native is logical external. "
                "Returns a run_id: watch it with delegate_wait, stop it with "
                "delegate_cancel. The run's output is a CLAIM you must check — you are the "
                "host, so verification receipts are still yours to write. If no route is "
                "configured or it is unavailable you get a typed refusal: choose an "
                "explicit configured alternative, wait, narrow, or report blocked. A direct "
                "fresh start requires subagent_id. In a configured session the host already STARTED the exact "
                "leaf before your first round (the startup receipt carries its run id): never start a duplicate — "
                "supervise it; a replacement delegate_start(prompt='') is legal only after verified cancellation/"
                "terminal settlement or a typed refusal proving no run exists. Recovery retries use retry_of without a new selector."
            ),
            "parameters": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                "prompt": {"type": "string", "description":
                    "Complete task for a direct start; for the configured snapshotted session (retry/"
                    "replacement), only optional advisory coordination context — the host supplies the canonical work order."},
                "subagent_id": {"type": "string", "description":
                    "Required for a fresh start made directly: exact agent_session actor id from Available "
                    "subagents. Omit for the current configured snapshotted route and for retry_of. API actor ids are refused here "
                    "and must be scheduled as recursive children."},
                "root": {"type": "string", "enum": ["skill_payload"], "description":
                    "Optional exact-resource selector: 'skill_payload' delegates ONE "
                    "installed user-managed skill payload you can already write. Omit "
                    "for ordinary workspace delegation."},
                "bucket": {"type": "string", "description":
                    "With root='skill_payload': the payload location "
                    "(external|clawhub|ouroboroshub|user_repo)."},
                "skill_name": {"type": "string", "description":
                    "With root='skill_payload': the exact skill name."},
                "max_seconds": {"type": "integer", "description":
                    "Wall-clock cap for the run; narrowed to your own remaining deadline. "
                    "Harness runs routinely need 3-5+ minutes end to end, so do not set a "
                    "tight cap for what feels like a quick edit. While delegate_wait shows "
                    "an advancing cursor the run is WORKING, and it enforces this cap "
                    "itself — cancelling a progressing run discards the whole run's spend."},
                "retry_of": {"type": "string", "description":
                    "EXPLICIT retry token: the pending_invocation_id from a start whose "
                    "outcome was unknown (transport failure, lost response). Replays THAT "
                    "invocation byte-identically under its original key, so the engine "
                    "returns the run it already accepted instead of starting a second one. "
                    "Omit subagent_id on this recovery path; supplying both selectors is a "
                    "typed conflict. "
                    "Never set it for an intended new run — a plain call always starts a "
                    "NEW invocation, even with an identical prompt."},
                },
            },
        }, _delegate_start_entry,
           timeout_sec=120),
        ToolEntry("delegate_wait", {
            "name": "delegate_wait",
            "description": (
                "Sleep on a delegated run until a meaningful event. Quiet transport windows "
                "are renewed by the host with zero model calls; journal progress still streams "
                "to the human but does not wake you. Terminal settlement, a new interaction, "
                "fault, addressed owner/task message, a direct-child attention/terminal event, "
                "cancel/deadline control, recovery judgment, or an explicit one-shot checkpoint "
                "wakes exactly once. A run that asks its "
                "user a question returns IMMEDIATELY as status='waiting_on_user' with "
                "the full question set (interaction/question ids ride WHOLE, never "
                "truncated): answer it with delegate_answer, or raise it with the "
                "escalate verb (parent-first) and keep waiting (a question with a "
                "timeout_at benign-declines "
                "at the engine timeout; timeout_at=null waits until answered). A "
                "large terminal result is delivered as a bounded preview plus an "
                "artifact: read output_delivery and finish reading the artifact before "
                "you rely on it."
            ),
            "parameters": {"type": "object", "required": ["run_id"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "since_seq": {"type": "integer", "description": "Event cursor: advances past it are recorded as progress."},
                "checkpoint_after_sec": {"type": "integer", "description":
                    "Optional one-shot future inspection time. Requires checkpoint_reason; "
                    "a real earlier wake consumes it."},
                "checkpoint_reason": {"type": "string", "description":
                    "Why one proactive inspection is worth a model call. No repeating cadence."},
            }},
        }, _delegate_wait_entry, timeout_sec=get_task_abs_ceiling_sec() + 120),
        ToolEntry("delegate_cancel", {
            "name": "delegate_cancel",
            "description": (
                "Cancel a delegated run. Claudexor keeps partial artifacts, but a cancelled "
                "session has no verdict and no finished work product — cancel a stuck or "
                "misdirected run, never one you merely want to hurry. The result is typed: "
                "only `confirmed` means a verified terminal receipt; `requested`, `failed` "
                "and `containment_fault_run_may_still_be_live` all mean it may still be running."
            ),
            "parameters": {"type": "object", "required": ["run_id"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "reason": {"type": "string", "description": "Why you are stopping it."},
            }},
        }, lambda ctx, run_id, reason="": _delegate_cancel(ctx, run_id, reason), timeout_sec=120),
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
                "post a different answer for the same interaction). Codex-lane runs "
                "have no mid-run questions: a run that ENDS needing input "
                "(outcome_facts.reason=input_required) is answered with a plain NEW "
                "delegate_start(subagent_id=..., prompt=...) whose prompt carries the "
                "assignment plus the answers "
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
        }, lambda ctx, run_id, interaction_id, answers, source_response=None: _delegate_answer(
            ctx, run_id, interaction_id, answers, source_response), timeout_sec=120),
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
