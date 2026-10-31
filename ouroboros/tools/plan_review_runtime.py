"""Reviewer-slot execution and rails for ``plan_task``'s asks.

The planning organ (``tools/plan_review.py`` entry, ``tools/plan_ask.py`` one ask,
``plan_record.py`` the record) owns the plan, the participants and their replies;
this module owns the thin runtime seams under an ask: the deadline rail, the review
pool's rows as ``ReviewSlot`` objects (both delivery kinds), the pre-fan-out lane
health snapshot and fit, one substrate call, the typed row projection of each
seat's actor, and the structural helpers (payload exemption, governance text) that
need a ``ToolContext``. Transport is NEVER chosen here: every slot carries its
route and ``review_execution._review_route_executor`` binds it.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

from ouroboros.config import EFFORT_SCALE
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.llm import LLMClient
from ouroboros.review_execution_projection import review_executions_from_actor_usage
from ouroboros.tools.registry import ToolContext, active_repo_dir_for
from ouroboros.utils import utc_now_iso


PLAN_REVIEW_MAX_TOKENS = 65536
# The one caller-facing strength axis of a review panel: an ask's order decides every
# Auto reviewer row, clamped into the owner's effort range; a pinned row keeps its pin
# outside Cyber Pro and a level in a model name always stays; every ask records the
# effective per-seat effort (``reviewer_slot_config.row_effort``). The entry's schema
# carries the description.
REVIEWER_EFFORT_SCHEMA = {"type": "string", "enum": list(EFFORT_SCALE)}
# ``None`` means no plan-local cognition cutoff.  The substrate settles against
# the owner deadline or shared transport bound, keeping the historical 560s
# number from being reused as an HTTP timeout.
PLAN_REVIEW_SLOT_TIMEOUT_SEC = None
# Per-slot provenance of what the reviewer read (BIBLE P3): an api_chat slot read exactly
# the host-assembled packet; a retrieving slot's reads are observed (host or harness journal)
# or unobserved — a disclosure, never a claim of full-surface coverage.
HOST_FILE_READ_ASSEMBLED = "host_assembled_packet"
HOST_FILE_READ_UNOBSERVED = "unobserved"

log = logging.getLogger(__name__)


def _governance_text(system_root: pathlib.Path, rel_path: str) -> str:
    from ouroboros.tools.review_helpers import load_governance_doc

    text = load_governance_doc(system_root, rel_path, on_missing="explicit")
    return "" if text.startswith("[⚠️ OMISSION") else text


def _session_task_text(system_prompt: str, user_content: str, session_root: str) -> str:
    return (
        "RETRIEVING REVIEWER (agent session): you review "
        f"{session_root or 'the active workspace'} (read-only, or from your own private checking folder "
        "when your seat holds write access; its checking instructions say what it holds). The evidence below is the host's "
        "snapshot — the same bytes every reviewer sees, each item naming its source sha256; a locator you "
        "read yourself may hold later bytes than your peers reviewed. The governance pack's named sources are public "
        "repository documents you MAY read raw, and MUST read in full when the pack marks them "
        "MANDATORY FULL READS (a self-modification plan), even if the agent also declared them as "
        "evidence. Retrieve any OTHER repository context with your own tools.\n\n"
        + "The own-room source handle is the canonical dialogue. Read omitted ranges "
        "from that artifact only; raw chat/mailbox logs do not replace that snapshot.\n\n"
        + system_prompt + "\n\n" + user_content
    )


def plan_deadline_skip(ctx: ToolContext, *, emit: bool = False) -> str:
    """Project the existing deadline rail without starting a paid reviewer panel."""
    from ouroboros.config import get_plan_task_deadline_min_sec

    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    deadline = parse_deadline_ts(metadata.get("deadline_at"))
    if deadline is None:
        return ""
    remaining = (deadline - utc_now()).total_seconds()
    scaled = max(0.0, remaining / 4.0)
    minimum = get_plan_task_deadline_min_sec()
    if remaining > 0 and scaled >= minimum:
        return ""
    if emit:
        try:
            event_queue = getattr(ctx, "event_queue", None)
            if event_queue is not None:
                event_queue.put_nowait({
                    "type": "plan_task_deadline_skip",
                    "task_id": str(getattr(ctx, "task_id", "") or ""),
                    "remaining_sec": round(remaining, 1),
                    "scaled_ceiling_sec": round(scaled, 1),
                    "min_useful_sec": minimum,
                    "ts": utc_now_iso(),
                })
        except Exception:
            pass
    cause = (
        "the task deadline has expired; no reviewer work was started."
        if remaining <= 0 else
        f"insufficient time for useful planning — remaining {int(remaining)}s gives a "
        f"review window of {int(scaled)}s (< {int(minimum)}s useful floor)."
    )
    return (
        f"PLAN_TASK_SKIPPED_DEADLINE: {cause} This call dispatched no reviewer; remaining time "
        f"{max(0, int(remaining))}s. The plan record keeps its state (its revisions and answers stay "
        "recorded); the same call under this deadline returns this rail again. The deadline is the "
        "owner's task bound, not a reviewer verdict."
    )


def plan_review_slots(default_effort: str = "") -> list:
    """The review pool's rows as plan-review ``ReviewSlot`` objects: the shared
    ``triad_delivery_slots`` builder (one reader of the pool for plan, skill
    and acceptance review) with plan review's own slot properties — timeout,
    output budget, temperature — and the envelope's ``reviewer_effort`` as the
    ORDER for this plan under ``row_effort``'s rule: it decides every Auto row,
    clamped into the owner's effort range, a pinned row keeps its pin outside
    Cyber Pro, a level in a model name always stays, and ``''`` leaves every row
    at its own level, else the range's top, exactly like the commit gate. The
    order is an ARGUMENT of this builder only, never a contextvar: the commit
    gate, scope, acceptance and skill review keep reading the untouched rows.
    Both delivery kinds ride; slot ids are the rows' own.
    """
    from ouroboros.reviewer_slot_config import triad_delivery_slots

    return triad_delivery_slots(
        role_hint="plan reviewer", default_effort=str(default_effort or ""),
        timeout_sec=PLAN_REVIEW_SLOT_TIMEOUT_SEC, max_tokens=PLAN_REVIEW_MAX_TOKENS,
        default_temperature=0.2,
    )


def slot_is_session(slot: Any) -> bool:
    """TRANSPORT fact: this slot runs a delegated Claudexor session (daemon
    health, threads, invocation custody). Never use it for delivery class."""
    from ouroboros.review_execution import ReviewRouteKind

    return getattr(slot, "route", None) is ReviewRouteKind.AGENT_SESSION


def slot_retrieves(slot: Any) -> bool:
    """DELIVERY class: this slot reads the subject with its own tools — a
    session row or a configured-subagent native row. Packet sizing, fit and
    retrieval semantics test THIS, not the route name."""
    return bool(getattr(slot, "retrieves", slot_is_session(slot)))


async def run_plan_review_slots(
    ctx: ToolContext,
    slots: list,
    *,
    system_prompt: str,
    user_content: str,
    session_task: str = "",
    session_root: str = "",
    output_contract: str = "",
    slot_messages: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    slot_session_tasks: Optional[Dict[str, str]] = None,
    native_mandatory_read_chars: int = 0,
    request_policy: Optional[dict] = None,
    session_threads: Optional[Dict[str, str]] = None,
    retry_key: str = "",
    reconciliation_identity: Optional[dict] = None,
    release_at_dispatch: bool = False,
) -> list[dict]:
    """ONE ``ReviewRequest`` fanned across the named participants through the substrate.

    ``release_at_dispatch`` returns at the dispatch barrier (drain window 0): slots
    still running come back as typed ``pending_dispatch`` rows and settle through
    process-local custody into the plan record (``plan_ask.settle_released_reply``).

    api_chat rows read ``messages`` (system + user packet); agent_session rows read
    ``session_task``/``session_root``/``policy.output_contract`` and retrieve the
    rest themselves. Returns one raw row per slot, in slot order, carrying the id
    the substrate RAN under, its route, text/error, refs, usage and the
    ``host_file_read_attestation`` fact."""
    from ouroboros.review_substrate import ReviewRequest, run_review_request
    from ouroboros.model_wait import copy_wait_context
    from ouroboros.tools.plan_packet import plan_user_stable_len
    from ouroboros.tools.review_synthesis import build_plan_review_messages
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    request = ReviewRequest(
        surface="plan_review",
        goal="Weigh the plan — an intention, not a verdict — before the work starts.",
        messages=build_plan_review_messages(system_prompt, user_content, plan_user_stable_len(user_content)),
        slot_messages=dict(slot_messages or {}),
        slot_session_tasks=dict(slot_session_tasks or {}),
        task_id=str(getattr(ctx, "task_id", "") or "plan_review"),
        call_type="plan_review",
        max_tokens=PLAN_REVIEW_MAX_TOKENS,
        default_temperature=0.2,
        no_proxy=True,
        session_task=session_task,
        session_root=session_root,
        session_threads=dict(session_threads or {}),
        retry_key=str(retry_key or ""),
        drain_deadline=time.monotonic() if release_at_dispatch else None,
        reconciliation_identity=dict(reconciliation_identity or {}),
        # The ask's identity (task + ask id) owns its cache split: a later ask
        # under the same task/model/slot starts cold.
        usage_attribution={"review_wave_id": str(retry_key or "")} if retry_key else {},
        policy=dict(request_policy) if request_policy is not None else {"output_contract": output_contract,
                "native_data_root": str(getattr(ctx, "budget_drive_root", None) or ctx.drive_root),
                "native_mandatory_read_chars": native_mandatory_read_chars},
    )
    loop = asyncio.get_running_loop()
    wait_context = copy_wait_context()
    parent_usage = current_usage_scope() or UsageScope()

    def run():
        # Carry the admitted root wallet and live wait owner, but no parent
        # Main physical capture/context into the reviewer's separate attempts.
        with usage_scope(parent_usage):
            return run_review_request(
                request, slots=list(slots), drive_root=pathlib.Path(ctx.drive_root),
                llm=LLMClient(), usage_ctx=ctx,
            )

    result = await loop.run_in_executor(None, wait_context.run, run)
    by_id = {str(slot.slot_id): slot for slot in slots}
    rows = [_plan_row_from_actor(actor, by_id.get(str(actor.get("slot_id") or ""))) for actor in result.actors]
    answered = {row["slot_id"] for row in rows}
    for slot in slots:  # a slot the substrate never answered for is still a configured row
        if str(slot.slot_id) not in answered:
            rows.append(_plan_row_from_actor({"slot_id": slot.slot_id, "model": slot.model,
                                              "status": "error", "error": "no actor record",
                                              "failure_code": "review_custody_lost",
                                              "operation_state": "custody_lost",
                                              "late_result_pending": True}, slot))
    return rows


def _typed_facts_from(source: Any, get: Any) -> Dict[str, Any]:
    """The B1 typed failure facts off one source via ``get(source, key)`` — the
    ONE key list is ``review_substrate.TYPED_FAILURE_FACT_KEYS`` (shared with the
    last-execution recorder; sources differ, keys must not). ``http_status`` stays
    raw (int-or-None), ``failure_details`` stays a dict; the rest are strings."""
    from ouroboros.review_substrate import TYPED_FAILURE_FACT_KEYS

    facts = {key: get(source, key) for key in TYPED_FAILURE_FACT_KEYS}
    details = facts.get("failure_details")
    facts["failure_details"] = dict(details) if isinstance(details, dict) else {}
    return {key: value if key in {"http_status", "failure_details"} else str(value or "")
            for key, value in facts.items()}


def _plan_row_from_actor(actor: Dict[str, Any], slot: Any) -> dict:
    usage = actor.get("usage") or {}
    text = actor.get("raw_text") or ""
    error = actor.get("error") or ""
    if actor.get("status") not in {"ok", "empty"} and not error:
        error = str(actor.get("status") or "review failed")
    session = slot_is_session(slot) if slot is not None else False
    physical_attempt_state = str(
        usage.get("physical_attempt_state")
        or actor.get("physical_attempt_state")
        or ""
    )
    provider_status_code = usage.get("provider_status_code")
    if provider_status_code is None:
        provider_status_code = actor.get("provider_status_code")
    if isinstance(provider_status_code, bool):
        provider_status_code = None
    elif provider_status_code is not None:
        try:
            provider_status_code = int(provider_status_code)
        except (TypeError, ValueError, OverflowError):
            provider_status_code = None
    return {
        # Identity is CARRIED, never re-derived: the row keeps the slot_id the
        # substrate ran, so duplicate-model plan rows stay distinguishable.
        "slot_id": str(actor.get("slot_id") or getattr(slot, "slot_id", "") or ""),
        "model": str(usage.get("resolved_model") or actor.get("model") or getattr(slot, "model", "") or ""),
        "request_model": str(getattr(slot, "model", "") or actor.get("model") or ""),
        # The RESOLVED requested effort of this seat (never a provider-observed applied
        # effort: no telemetry carries one), and whether an envelope's order set it.
        "effort": str(getattr(slot, "effort", "") or ""),
        "declared_effort": str(getattr(slot, "declared_effort", "") or ""),
        "route": "agent_session" if session else "api_chat",
        # Delivery-truthful provenance: a native tool-round actor discloses its
        # HOST-OBSERVED reads through its usage; a session whose journal was folded
        # over the room snapshot reads `harness_observed`, else unobserved; a
        # packet row stays host-assembled.
        "host_file_read_attestation": (
            str(usage.get("host_file_read_attestation") or "") or str(usage.get("read_provenance") or "")
            or (HOST_FILE_READ_UNOBSERVED if session else HOST_FILE_READ_ASSEMBLED)
        ),
        **({"room_read_coverage": coverage} if (coverage := _room_read_coverage(usage)) else {}),
        "text": text,
        "error": error or None,
        "prompt_ref": actor.get("prompt_ref") or {},
        "response_ref": actor.get("response_ref") or {},
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "cost": float(usage["cost"]) if usage.get("cost") is not None else None,
        # Presentation-only receipt from the actor's RETURNED usage. Requested
        # slot route/model, money, profile and raw telemetry never enter it.
        "executions": review_executions_from_actor_usage([actor]),
        # B1 typed failure facts, forwarded VERBATIM from the actor record (empty on
        # success and on pre-typed engines) so downstream never regresses to matching
        # the error prose for the code or the reset instant.
        **_typed_facts_from(actor, lambda source, key: source.get(key)),
        "physical_attempt_state": physical_attempt_state,
        "provider_status_code": provider_status_code,
        "capability_delta": usage.get("capability_delta") or [],
        # The engine's own word on the model it ran, kept apart from the requested one.
        "resolved_model": str(usage.get("resolved_model") or ""),
        "review_thread_id": str(usage.get("review_thread_id") or ""),
        "review_turn_id": str(usage.get("review_turn_id") or ""),
        "review_thread_receipt": usage.get("review_thread_receipt") or {},
        "auth_route_receipt": usage.get("auth_route_receipt") or {},
        "profile_continuity_receipt": usage.get("profile_continuity_receipt") or {},
        "applied_profile": str(usage.get("applied_profile") or ""),
        "recovery_binding": dict(actor.get("recovery_binding") or {}),
        # The controller that still owns a released seat (``stamp_review_controller``): a cold
        # collector defers to it while it lives instead of reading the seat as dead.
        "review_controller": dict(usage.get("review_controller") or {}) if isinstance(usage.get("review_controller"), dict) else {},
        "operation_id": str(actor.get("operation_id") or ""),
        "operation_state": str(actor.get("operation_state") or "settled"),
        "late_result_pending": bool(actor.get("late_result_pending")),
        "error_event_logged": actor.get("error_event_logged") is True,  # #1578 append receipt
        "awaiting_since": str(actor.get("awaiting_since") or ""),
        "pending_invocation_id": str(
            usage.get("pending_invocation_id") or actor.get("pending_invocation_id") or ""
        ),
        "delegated_run_id": str(
            usage.get("delegated_run_id") or actor.get("delegated_run_id") or ""
        ),
    }


def _room_read_coverage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """What this reviewer read of the own-room snapshot (observed-source fold): a fact beside the verdict."""
    rows = (usage.get("native_read_coverage") or {}).get("sources") if isinstance(usage.get("native_read_coverage"), dict) else None
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) and rows[0].get("root") == "artifact_store" else {}
    return {"status": str(row.get("status") or ""), "covered_chars": int(row.get("covered_chars") or 0),
            "complete_chars": int(row.get("complete_chars") or 0), "provenance": str(usage.get("read_provenance") or "")} if row else {}


def plan_row_typed_facts(row: Dict[str, Any]) -> Dict[str, Any]:
    """The typed failure and custody facts a participant's settlement carries forward
    from a plan row (``plan_ask.reply_facts_from_row``). Every fact defaults to honest
    absence — typed-fact-free rows (``plan_slot_fit``'s preflight_oversize rows) carry
    none."""
    facts = {
        **_typed_facts_from(row, lambda source, key: source.get(key)),
        "capability_delta": row.get("capability_delta") or [],
        **({"replayed_from": dict(row["replayed_from"])} if isinstance(row.get("replayed_from"), dict) and row["replayed_from"] else {}),
    }
    physical_attempt_state = str(row.get("physical_attempt_state") or "")
    provider_status_code = row.get("provider_status_code")
    if physical_attempt_state or provider_status_code is not None:
        facts.update({
            "physical_attempt_state": physical_attempt_state,
            "provider_status_code": provider_status_code,
        })
    pending_invocation_id = str(row.get("pending_invocation_id") or "")
    delegated_run_id = str(row.get("delegated_run_id") or "")
    if row.get("operation_id") or row.get("late_result_pending") \
            or str(row.get("operation_state") or "settled") != "settled" \
            or pending_invocation_id or delegated_run_id:
        facts.update({
            "recovery_binding": dict(row.get("recovery_binding") or {}),
            "operation_id": str(row.get("operation_id") or ""),
            "operation_state": str(row.get("operation_state") or "settled"),
            "late_result_pending": bool(row.get("late_result_pending")),
            # Since when the host has been waiting; '' whenever it never sent.
            "awaiting_since": str(row.get("awaiting_since") or ""),
            "pending_invocation_id": pending_invocation_id,
            "delegated_run_id": delegated_run_id,
        })
    return facts


def plan_payload_roots(ctx: ToolContext, locators: List[str]) -> list[pathlib.Path]:
    """Skill-payload exemption roots for ``plan_spec.resolve_constitutional``.

    Skill-payload paths live in the DATA plane (``data/skills/<bucket>/…``, not the
    system repo) and never make a plan a self-modification by themselves — the same
    frozen predicate the deleted ``resolve_plan_class`` applied. A recognized locator
    is exempt however it resolves: its data-plane payload root AND its literal
    resolution against the active root are both returned. Classification-only
    (write gates are unrelated code paths); a drive-resolution failure skips the
    exemption, as before."""
    from ouroboros.contracts.skill_payload_policy import resolve_skill_payload_target
    from ouroboros.tool_access import canonical_data_root

    try:
        drive = canonical_data_root(ctx)
        active = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    except Exception:
        return []
    roots: list[pathlib.Path] = []
    for raw in locators or []:
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            target = resolve_skill_payload_target(drive, text)
        except ValueError:
            continue
        roots.append(target.payload_root)
        try:
            candidate = pathlib.Path(text)
            roots.append((candidate if candidate.is_absolute() else active / candidate).resolve(strict=False))
        except (OSError, ValueError):
            continue
    return roots


# ------------------------------------------------------------ panel health (B2b)
#
# ONE health snapshot before fan-out turns slots with POSITIVE structural evidence
# of a spent lane into $0 typed not-dispatched rows the ask records beside its
# participants (nobody silently disappears from the council). Unknown health DISPATCHES
# (fail-open); transient daemon states (`daemon_recovery_only`, a dead socket)
# are never skip evidence (roast pt 9). Side effect,
# disclosed: `ensure_owned_gateway` may lazily SPAWN the owned daemon and runs
# its per-ensure rotation reconcile — both bounded and fail-open (a spawn/probe
# failure returns None here, never an exception).
#
# SCOPE OF THE ANSWER: `subagents.route_health` — the ONE manifest reader — judges
# a ROUTE (harness id + pinned model) NARROWED to the slot's pinned credential
# profile whenever the row pins one, exactly as the dispatcher asks it
# (`review_execution`: a pin is strict, so a healthy sibling account must not vouch
# a spent pin). A row that pins NO profile keeps the route-wide answer: which
# account an unpinned run lands on is Claudexor's rotation business, so a route
# whose pool still holds a live account reads healthy — such a slot dispatches
# (fail-open) and fails typed downstream (B1) if rotation lands it badly. No second
# health oracle is built here. api_chat rows have no route health source at all and
# always dispatch. Cursor lanes without quota snapshots simply dispatch too.


def _structural_skip_code(reason: str, reset_at: str) -> str:
    """POSITIVE structural evidence only: a dated window exhaustion whose reset is
    still ahead, or a typed dead-pool code. An UNDATED exhaustion, a stale reset and
    every other reason (route_disabled, transient daemon states, unknown) dispatch —
    the pre-dispatch admission and the run itself refuse typed downstream at ~$0."""
    from ouroboros.gateways.claudexor import WINDOW_EXHAUSTED_CODES

    if reset_at:
        instant = parse_deadline_ts(reset_at)
        if instant is not None and instant > utc_now():
            return reason or "subscription_window_exhausted"
        return ""
    if reason in WINDOW_EXHAUSTED_CODES and reason != "subscription_window_exhausted":
        # e.g. credential_pool_exhausted: the pool itself is dead. Honest note: NO
        # producer routes this code through `route_health` today — the branch is
        # the contract point for PR-A A5's route_health sync (the typed pool
        # terminal), kept on purpose so the sync lands without an engine change.
        return reason
    return ""


def plan_panel_health_snapshot(slots: list) -> Optional[Dict[str, Dict[str, str]]]:
    """``{slot_id: {failure_code, reset_at}}`` for the slots a pre-fan-out snapshot
    proves structurally dead; ``{}`` when the snapshot ran and found none; ``None``
    when no snapshot could be captured (daemon unprovisioned/unreachable) — unknown
    health dispatches, and a FAILED snapshot must never read as "everything healed"
    at the replay seam."""
    session_slots = [slot for slot in slots if slot_is_session(slot)]
    if not session_slots:
        return {}
    from ouroboros.claudexor_daemon import ensure_owned_gateway, owned_daemon_provisioned
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import delegated_run_shape, route_health
    from ouroboros.review_execution import ReviewRouteUnavailable, session_route_for_review_slot

    if not owned_daemon_provisioned():
        return None
    evidence: Dict[str, Dict[str, str]] = {}
    by_route: Dict[tuple, tuple[str, str]] = {}
    gateway = None
    try:
        gateway = ensure_owned_gateway()
        for slot in session_slots:
            try:
                route = session_route_for_review_slot(slot)
            except ReviewRouteUnavailable:
                # Dispatch owns the typed refusal; unresolvable slots supply
                # no health fact and cannot hide a sibling's actual evidence.
                continue
            # The PIN is part of the subject, so it is part of the memo key: two
            # rows on the same harness+model but different accounts must never
            # share one verdict (a spent pin would otherwise be vouched for by a
            # sibling's health, or vice versa).
            pin = str(getattr(route, "profile_id", "") or "")
            # Rights are part of the subject too: a route may serve read-only seats
            # and refuse a writable one (its access profile or the delegated marker).
            shape = delegated_run_shape(False, str(getattr(slot, "session_access", "") or "readonly"))
            key = (route.route_id, route.model, pin, shape.access)
            if key not in by_route:
                by_route[key] = route_health(gateway, route.route_id, shape,
                                             route_model=route.model,
                                             pinned_profile=pin)
            code = _structural_skip_code(*by_route[key])
            if code:
                evidence[str(getattr(slot, "slot_id", "") or "")] = {
                    "failure_code": code, "reset_at": by_route[key][1]}
    except ClaudexorUnavailable:
        return None  # transient (incl. daemon_recovery_only): never a skip row
    except Exception:
        log.debug("plan panel health snapshot failed (fail-open)", exc_info=True)
        return None
    finally:
        if gateway is not None:
            gateway.close()
    return evidence


def plan_health_skip_rows(slots: list, evidence: Optional[Dict[str, Dict[str, str]]]) -> tuple[list, list[dict]]:
    """``(live_slots, skip_rows)`` — a structurally dead slot becomes a $0 typed row
    shaped like ``plan_slot_fit``'s preflight_oversize rows plus B1's typed facts; the
    ask records it as a named participant that was not dispatched. Live slots keep
    dispatching whatever happened to the dead ones."""
    live, rows = [], []
    for slot in slots:
        ev = (evidence or {}).get(str(getattr(slot, "slot_id", "") or ""))
        if not ev:
            live.append(slot)
            continue
        code, reset = str(ev.get("failure_code") or ""), str(ev.get("reset_at") or "")
        pin = str(getattr(slot, "session_profile", "") or "")
        scope = (
            f"Evidence is scoped to this slot's pinned credential profile {pin!r}."
            if pin else
            "Evidence is route-wide: this slot pins no credential profile, so the "
            "answer covers the route's rotation pool rather than one account."
        )
        rows.append({
            "slot_id": str(getattr(slot, "slot_id", "") or ""),
            "model": str(getattr(slot, "model", "") or ""),
            "request_model": str(getattr(slot, "model", "") or ""),
            "route": "agent_session", "host_file_read_attestation": None, "text": "",
            "effort": str(getattr(slot, "effort", "") or ""), "declared_effort": str(getattr(slot, "declared_effort", "") or ""),
            "error": (
                f"health_skip[{code}]: the pre-fan-out panel health snapshot shows this "
                f"slot's delegated route window spent{f' (resets {reset})' if reset else ''}; "
                f"skipped before dispatch at $0. {scope}"
            ),
            "failure_code": code, "reset_at": reset,
            "prompt_ref": {}, "response_ref": {}, "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "status": "not_dispatched", "operation_state": "not_dispatched",
        })
    return live, rows


def plan_reviewer_config_fingerprint(slots: list) -> str:
    """Identity of one ask's reviewer roster (the ``roster_hash`` of its reconciliation
    identity): slot ids, targets, routes, pinned session targets/profiles AND efforts.
    Effort is identity (it changes what the reviewer actually does); timeouts are
    tuning, not identity; roster ORDER is identity (it is the configured row order)."""
    rows = [
        [str(getattr(s, "slot_id", "") or ""), str(getattr(s, "model", "") or ""),
         str(getattr(getattr(s, "route", None), "value", "") or ""),
         str(getattr(s, "session_target", "") or ""),
         str(getattr(s, "session_profile", "") or ""),
         str(getattr(s, "effort", "") or ""),
         # Rights are identity; a read-only (or legacy) seat keeps its exact bytes.
         *([str(s.session_access)] if str(getattr(s, "session_access", "") or "readonly") != "readonly" else [])]
        for s in slots or []
    ]
    # Actor binding is delivery identity; column added only when some row
    # carries one, so unchanged legacy rosters keep their exact bytes.
    actor_ids = [str(getattr(s, "subagent_id", "") or "") for s in slots or []]
    if any(actor_ids):
        for row, actor in zip(rows, actor_ids):
            row.append(actor)
    # Legacy API rows without actor bindings meant packet. Preserve those
    # historical bytes; a direct native delivery adds explicit identity.
    delivery_changes = [bool(getattr(s, "native_retrieval", False)) != (
                        bool(actor) and str(getattr(getattr(s, "route", None), "value", "")) == "api_chat")
                        for s, actor in zip(slots or [], actor_ids)]
    if any(delivery_changes):
        for row, slot in zip(rows, slots):
            row.append("native" if getattr(slot, "native_retrieval", False) else "packet")
    return sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def effective_plan_slots(slots: list) -> list:
    """Resolve existing task-local owner model choices before sizing or sending."""
    from ouroboros.review_records import apply_review_model_override
    from ouroboros.model_wait import current_model_wait

    waiter = current_model_wait()
    return [apply_review_model_override(slot, waiter.overrides) for slot in slots] if waiter else list(slots)


def plan_slot_fit(slots: list, *, prompt_chars: int, slot_prompt_chars: Optional[dict] = None) -> tuple[list, list[dict]]:
    """``(callable_slots, oversize_rows)`` for one ask fanned across mixed-window slots —
    the review organ's calibrated per-slot input caps
    (`review_synthesis.per_slot_input_token_limits`, Capability Evidence windows) against
    each participant's estimated tokens. An excluded slot is a typed `preflight_oversize`
    row (ok=False, $0) so it is REPORTED as not participating; an ask nobody can take is
    the caller's typed refusal (``PLAN_ASK_NOTHING_SENDABLE``)."""
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits
    slots = effective_plan_slots(slots)

    # Only api_chat rows are sized: a RETRIEVING (agent_session) row's model id is an opaque
    # harness target, not a provider route (`reviewer_window.reviewer_route(session=True)`), and
    # the review organ's convention (triad: "session rows are not constrained by this pack")
    # is that such a row is never fit-excluded — it retrieves with its own tools. Its inline
    # conversation is fitted only to an owner-asserted `reviewer:<slot>` window
    # (`plan_dialogue.dialogue_slot_inputs`); without one the host invents no window.
    api_slots = [slot for slot in slots if not slot_retrieves(slot)]
    api_models = [str(getattr(slot, "model", "") or "") for slot in api_slots]
    limits = per_slot_input_token_limits(
        api_models, output_reserve=PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000, slots=api_slots)
    estimated = max(1, (max(0, int(prompt_chars)) + 3) // 4)  # utils.estimate_tokens on the packet
    callable_slots, oversize = [], []
    for slot in slots:
        estimated = max(1, (int((slot_prompt_chars or {}).get(str(slot.slot_id), prompt_chars)) + 3) // 4)
        cap = 0 if slot_retrieves(slot) else int(limits[str(slot.slot_id)])
        if slot_retrieves(slot) or estimated <= cap:
            callable_slots.append(slot)
            continue
        oversize.append({
            "slot_id": str(getattr(slot, "slot_id", "") or ""), "model": str(getattr(slot, "model", "") or ""),
            "request_model": str(getattr(slot, "model", "") or ""),
            "route": "agent_session" if slot_is_session(slot) else "api_chat",
            "host_file_read_attestation": None, "text": "",
            "effort": str(getattr(slot, "effort", "") or ""), "declared_effort": str(getattr(slot, "declared_effort", "") or ""),
            "error": (f"preflight_oversize: assembled packet ~{estimated:,} estimated tokens exceeds "
                      f"this slot's calibrated input cap {cap:,}"),
            "prompt_ref": {}, "response_ref": {}, "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "status": "not_dispatched", "operation_state": "not_dispatched",
        })
    return callable_slots, oversize


def plan_fanout_inputs(slots: list, *, prompt_chars: int, slot_prompt_chars: Optional[dict] = None) -> dict:
    """One fresh fan-out: the lane health snapshot, then the fit of each live slot."""
    slots = effective_plan_slots(slots)
    health_evidence = plan_panel_health_snapshot(slots)
    live_slots, health_skip_rows = plan_health_skip_rows(slots, health_evidence)
    callable_slots, oversize_rows = plan_slot_fit(live_slots, prompt_chars=prompt_chars, slot_prompt_chars=slot_prompt_chars)
    return {"callable_slots": callable_slots, "health_skip_rows": health_skip_rows,
            "oversize_rows": oversize_rows, "health_evidence": health_evidence}
