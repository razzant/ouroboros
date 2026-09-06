"""Shared multi-review substrate.

This module is the common cognitive primitive for migrated review surfaces and
the contract target for remaining legacy immune-system reviews. Slot identity is
separate from model identity, so duplicate model IDs are valid independent
reviewer slots.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("review_substrate")

from ouroboros.llm import LLMClient
from ouroboros.observability import new_call_id, persist_call, redact_projection  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.provider_models import provider_for_model  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.review_execution_projection import (
    MAX_PROJECTED_ACTOR_FINDINGS, projected_finding_row,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    review_executions_from_actor_usage,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)
from ouroboros.task_results import review_binding_hash  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
# Everything below the seam. Re-exported here because the substrate is the
# historical import site for the api_chat prompt renderers; `review_execution`
# owns them now and must never import this module back.
from ouroboros.review_execution import (  # noqa: F401  (compat re-exports)
    AgentSessionReviewExecutor,
    ApiChatReviewExecutor,
    ReviewAssignment,
    ReviewAttemptResult,
    ReviewRouteKind,
    ReviewRouteUnavailable,
    ReviewSlotExecutor,
    delivery_retrieves,
    _execute_slot_attempt,
    _messages_char_count,
    _render_prompt,
    _render_prompt_parts,
    _request_messages,
    _review_route_executor,
    assert_cache_breakpoint_cap,
)
from ouroboros.review_custody import (
    _ReviewAttemptHistory, _review_exception_projection,
    review_retry_cancelled,
)
# Reviewer-output JSON extraction lives in ONE place beside the array
# extractor it falls back to (the fenced-object and verdict parsers were
# split across two modules for no reason).
from ouroboros.triad_review import parse_review_findings
from ouroboros.usage_accounting import (
    PhysicalAttemptLimitExceeded,
    UsageAccountingError,
    UsageScope,
    current_usage_scope,
    usage_scope,
)
from ouroboros.utils import sanitize_tool_result_for_log, truncate_review_artifact
from ouroboros._outcome_receipts import disclosed_list_projection  # noqa: F401 -- facade import surface; leaves read it through the call-time handle


class _CustodyUsageContext:
    """Forward custody state to the caller while keeping route-owned paid stamps.

    ``review_custody`` retains its standalone pre-fanout stamp contract, but the
    substrate has the more precise landed boundary: typed route refusals are $0,
    sessions stamp before ``START_REQUESTED``, and API calls stamp at the durable
    physical-attempt transition. The route already captured the exact stamp, so
    exposing it again through custody would fire plain callables twice and too
    early. All non-stamp reads and writes still target the original context.
    """

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    @property
    def _review_paid_stamp(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)


def review_repo_dirs_for(ctx: Any) -> tuple[pathlib.Path, pathlib.Path]:
    """Return validated ``(governance, subject)`` roots for plan/scope review."""
    from ouroboros.tools.registry import active_repo_dir_for

    workspace_raw = getattr(ctx, "workspace_root", None)
    workspace = pathlib.Path(workspace_raw) if isinstance(workspace_raw, (str, pathlib.Path)) else None
    if workspace is not None and not str(getattr(ctx, "workspace_mode", "") or "").strip():
        raise ValueError("workspace_root is set without workspace_mode")
    system_raw = getattr(ctx, "system_repo_dir", None)
    system = pathlib.Path(system_raw) if isinstance(system_raw, (str, pathlib.Path)) else None
    governance = (system or pathlib.Path(getattr(ctx, "repo_dir"))).resolve(strict=False)
    subject = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    if not governance.is_dir() or not subject.is_dir():
        raise ValueError(f"unavailable governance/subject root: {governance} / {subject}")
    return governance, subject


# B1 typed failure facts, ONE shared key tuple (row/wave/last-execution projections).


# Thin ReviewProfile hardness levels (Bible P3 DRY): the behavior is carried by
# request.policy; these name the three surfaces so callers/reviewers describe
# hardness consistently without a parallel pipeline.

# Tier vocabulary SSOT lives in outcomes.py; reuse it so a future tier rename
# cannot silently desync the capsule from the objective axis.
from ouroboros.outcomes import OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED, OUTCOME_TIER_SOLVED  # noqa: F401 -- facade import surface; leaves read it through the call-time handle


# v6.74.0 (A5): reviewer-authored dialogue status. The reviewer — not a host
# counter or hash — judges whether the acceptance dialogue is still actionable.


# Historical dispatch names remain re-exported for existing consumers.
from ouroboros.review_dispatch import (  # noqa: E402,F401 — re-exports
    acceptance_slot_fit,
    PLAN_SLOT_ID_PREFIX,
    ReviewPaidStamp,
    SCOPE_SLOT_ID_PREFIX,
    SLOT_ID_PREFIX,
    slot_id_for_row,
    stamp_review_paid_on_dispatch,
    task_acceptance_zero_physical_refusal,
)


# reviewer_slots()/triad_delivery_slots() live in reviewer_slot_config (altitude, P7); re-exported for callers here.
from ouroboros.reviewer_slot_config import SCOPE_ROLE_HINT, reviewer_slots, triad_delivery_slots  # noqa: F401,E402


def scope_reviewer_slots(
    models: List[str] | None = None, *, effort: str | None = None,
) -> List[ReviewSlot]:
    """The configured scope-reviewer rows — the single owner of scope-slot identity.

    Both scope surfaces read their ids from here: the substrate call that produces
    the durable prompt/response refs, and the actor records the commit attempt
    persists. One row therefore carries exactly one identity instead of two that
    disagree. Each row also carries its configured delivery route (D14: every
    scope slot is independently harness-or-API).

    With no explicit ``models`` the rows come from the reviewer-slot SSOT
    (6.1): stable owner ids, per-row route/target/effort. An explicit list
    keeps the historical positional behavior for callers that rebuild one row;
    such rows are pinned ``api_chat`` (the caller that fans out a delegated
    row overrides the route itself — the phase-5 per-row route envs are
    retired, ABI-10).

    An omitted ``effort`` resolves to the configured scope-review effort: the
    legacy path used to take this parameter's old literal default instead,
    silently running the BLOCKING reviewer below configured strength (the
    downgrade class the owner forbade).
    """
    if effort is None:
        from ouroboros.config import resolve_effort

        effort = resolve_effort("scope_review")
    if models is None:
        from ouroboros.reviewer_slot_config import structured_scope_review_slots

        structured = structured_scope_review_slots()
        if structured is not None:
            return structured
        # Resolved at call time so the configured list stays the live authority.
        from ouroboros.config import get_scope_review_models

        models = get_scope_review_models()
    return reviewer_slots(
        models, effort=effort, role_hint=SCOPE_ROLE_HINT, id_prefix=SCOPE_SLOT_ID_PREFIX,
    )


def review_usage_category(surface: str) -> str:
    """The usage-scope category every send of a review surface is attributed
    under — the key the ledger's cache split and the root telemetry's
    reservation identities carry, so the commit gate's admission and its
    scope-first hold name the same scope the substrate sends under."""
    return f"{surface}_review"


class ReviewCoordinator:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        drive_root: pathlib.Path | None = None,
        usage_ctx: Any = None,
    ):
        self.llm = llm or LLMClient()
        if drive_root is not None:
            self.drive_root = pathlib.Path(drive_root)
        else:
            # ISO-DRIP: the default is the ABSOLUTE config SSOT, never the old
            # cwd-relative "../data" — with any cwd under a repo/ that spelling
            # names the LIVE data root's sibling, so default-constructed
            # coordinators dripped synthetic review records into live
            # observability (and on trees with the absolute-root guard they
            # silently LOST the records instead: persist_call refuses relative
            # roots and the coordinator swallows it into empty refs). Same
            # resolution order the review surfaces already use
            # (review_drive_root: ctx → DATA_DIR); read late off the module so
            # test isolation and runtime rebinding are honored.
            from ouroboros import config

            self.drive_root = pathlib.Path(config.DATA_DIR)
        self.usage_ctx = usage_ctx
        paid_stamp = getattr(usage_ctx, "_review_paid_stamp", None)
        self._review_paid_stamp = (
            paid_stamp
            if isinstance(paid_stamp, ReviewPaidStamp) or not callable(paid_stamp)
            else ReviewPaidStamp(
                paid_stamp,
                fail_closed=bool(getattr(paid_stamp, "fail_closed", False)),
            )
        )

    def run(self, request: ReviewRequest, slots: List[ReviewSlot]) -> ReviewRunResult:
        if not slots:
            return ReviewRunResult(
                request=asdict(request),
                actors=[],
                parsed_findings=[],
                aggregate_signal="DEGRADED",
                degraded=True,
                degraded_reasons=["no_review_slots"],
                panel_id=_review_panel_id(request, []),
            )

        base_scope = current_usage_scope() or UsageScope()
        usage_meta = (
            getattr(self.usage_ctx, "task_metadata", {})
            if self.usage_ctx is not None
            else {}
        )
        if not isinstance(usage_meta, dict):
            usage_meta = {}
        if not str(getattr(request, "deadline_at", "") or "").strip():
            inherited_deadline = str(usage_meta.get("deadline_at") or "").strip()
            if inherited_deadline:
                request.deadline_at = inherited_deadline
        if getattr(request, "task_attempt", None) in (None, ""):
            request.task_attempt = getattr(self.usage_ctx, "task_attempt", None)
        task_id = str(request.task_id or base_scope.task_id or "")
        review_meta = request.usage_attribution if isinstance(request.usage_attribution, dict) else {}
        root_task_id = str(
            usage_meta.get("root_task_id") or base_scope.root_task_id or task_id
        )
        budget_root = (
            usage_meta.get("budget_drive_root")
            or getattr(self.usage_ctx, "budget_drive_root", "")
            or base_scope.drive_root
            or self.drive_root
        )
        if base_scope.global_limit_usd is not None:
            global_limit = base_scope.global_limit_usd
        else:
            try:
                from ouroboros.settings_setup_contract import resolve_total_budget_usd
                global_limit = resolve_total_budget_usd()
            except Exception:
                global_limit = None
        if base_scope.root_limit_usd is not None:
            root_limit = base_scope.root_limit_usd
        else:
            try:
                configured_root_limit = float(
                    os.environ.get("OUROBOROS_PER_TASK_COST_USD", "0") or 0
                )
                root_limit = configured_root_limit if configured_root_limit > 0 else None
            except (TypeError, ValueError):
                root_limit = None
        review_usage_scope = UsageScope(
            drive_root=budget_root,
            task_id=task_id,
            root_task_id=root_task_id,
            parent_task_id=str(usage_meta.get("parent_task_id") or base_scope.parent_task_id or ""),
            category=review_usage_category(request.surface),
            source="review_substrate",
            review_skill=str(review_meta.get("review_skill") or base_scope.review_skill or ""),
            review_wave_id=str(review_meta.get("review_wave_id") or base_scope.review_wave_id or ""),
            global_limit_usd=global_limit,
            root_limit_usd=root_limit,
        )

        from ouroboros.review_custody import run_custodied_review_slots

        def _run_slot_with_usage(
            slot: ReviewSlot,
            operation_id: str,
            retry_state: Dict[str, Any],
            deadline: float,
            checkpoint: Any,
        ) -> ReviewActorRecord:
            # Timing and worker lifetime belong to review_custody. This inner
            # scope preserves the landed per-row Skill Review attribution on
            # delegated start, recovery and settlement rows.
            with usage_scope(replace(review_usage_scope, review_slot_id=slot.slot_id)):
                return self._run_slot(
                    request,
                    slot,
                    operation_id=operation_id,
                    retry_state=retry_state,
                    logical_deadline_monotonic=deadline,
                    **(
                        {"pending_invocation_checkpoint": checkpoint}
                        if checkpoint is not None
                        else {}
                    ),
                )

        route_owned_stamp_surface = request.surface in {
            "multi_model_review",
            "scope_review",
            "plan_review",
            "skill_review",
            "task_acceptance",
            "advisory_review",
        }
        route_owned_executor = (
            str(getattr(getattr(self._run_slot, "__func__", None), "__module__", ""))
            == __name__
        )
        custody_usage_ctx = (
            _CustodyUsageContext(self.usage_ctx)
            if (
                self.usage_ctx is not None
                and callable(self._review_paid_stamp)
                and route_owned_stamp_surface
                and route_owned_executor
            )
            else self.usage_ctx
        )
        actors = run_custodied_review_slots(
            request=request, slots=slots,
            usage_ctx=custody_usage_ctx,
            task_id=task_id,
            usage_meta=usage_meta,
            review_usage_scope=review_usage_scope,
            run_slot=_run_slot_with_usage,
            error_actor=lambda slot, error, operation_id="", operation_state="settled": self._error_actor(
                request, slot, error, operation_id=operation_id,
                operation_state=operation_state,
            ),
        )
        slot_order = {slot.slot_id: idx for idx, slot in enumerate(slots)}
        slots_by_id = {slot.slot_id: slot for slot in slots}
        actors.sort(key=lambda actor: slot_order.get(actor.slot_id, len(slot_order)))
        try:
            # «Выполняется как» (D22): beside each saved row the UI shows what
            # the row REALLY ran as last time. Disclosure only; best-effort.
            from ouroboros.reviewer_slot_config import record_reviewer_slot_executions

            record_reviewer_slot_executions(request.surface, actors, slots_by_id)
        except Exception:
            log.debug("reviewer-slot last-execution write failed", exc_info=True)

        from ouroboros.review_actor_aggregation import aggregate_review_actors

        aggregate = aggregate_review_actors(
            request=request,
            slots=slots,
            actors=actors,
            slots_by_id=slots_by_id,
            actor_projection=_review_actor_projection,
            criteria_shape_valid=_criteria_shape_valid,
            advisory_hardness=HARDNESS_ADVISORY_VISIBLE,
        )
        return ReviewRunResult(
            request=asdict(request),
            actors=[asdict(actor) for actor in actors],
            **aggregate,
            panel_id=_review_panel_id(request, actors),
        )

    def _custody_drive_root(self) -> pathlib.Path:
        """Return the delegated slot's canonical custody root."""
        ctx = self.usage_ctx
        if ctx is not None and getattr(ctx, "drive_root", None):
            try:
                from ouroboros.delegate_custody import custody_root

                return custody_root(ctx)
            except Exception:
                log.debug("custody root resolution failed; using coordinator drive", exc_info=True)
        return self.drive_root

    def _error_actor(
        self,
        request: ReviewRequest,
        slot: ReviewSlot,
        error: str,
        *,
        operation_id: str = "",
        operation_state: str = "settled",
        prompt_ref: Dict[str, Any] | None = None,
    ) -> ReviewActorRecord:
        actor_status = "not_dispatched" if operation_state == "not_dispatched" else "error"
        call_id = new_call_id(f"review_{request.surface}_{slot.slot_id}_error")
        base_call_type = request.call_type or f"{request.surface}_review"
        assignment = ReviewAssignment(
            request=request, slot=slot, call_id=call_id, call_type=base_call_type,
            custody_root=self._custody_drive_root(),
        )
        # Best-effort failure evidence uses the route's own projection; a
        # refusal here must degrade the record, never re-raise.
        try:
            prompt_projection = _review_route_executor(assignment, llm=self.llm).prompt_payload()
        except Exception:
            prompt_projection = {}
        response_ref: Dict[str, Any] = {}
        if prompt_ref is None:
            prompt_ref = {}
            try:
                prompt_ref = persist_call(
                    self.drive_root,
                    task_id=request.task_id or "review",
                    call_id=f"{call_id}_prompt",
                    call_type=f"{base_call_type}_prompt",
                    payload={"request": asdict(request), "slot": asdict(slot), **prompt_projection},
                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "synthetic": True},
                )
            except Exception:
                prompt_ref = {}
        try:
            response_ref = persist_call(
                self.drive_root,
                task_id=request.task_id or "review",
                call_id=f"{call_id}_error",
                call_type=f"{base_call_type}_error",
                payload={"error": sanitize_tool_result_for_log(error)},
                manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "status": actor_status, "synthetic": True},
            )
        except Exception:
            response_ref = {}
        return ReviewActorRecord(
            slot_id=slot.slot_id,
            model=slot.model,
            status=actor_status,
            error=sanitize_tool_result_for_log(error),
            prompt_ref=prompt_ref,
            response_ref=response_ref,
            operation_id=str(operation_id or ""),
            operation_state=str(operation_state or "settled"),
            late_result_pending=str(operation_state or "") in {"in_flight", "custody_lost"},
        )

    def _run_slot(
        self,
        request: ReviewRequest,
        slot: ReviewSlot,
        *,
        operation_id: str = "",
        retry_state: Optional[Dict[str, Any]] = None,
        logical_deadline_monotonic: Optional[float] = None,
        pending_invocation_checkpoint: Any = None,
    ) -> ReviewActorRecord:
        call_id = str(operation_id or new_call_id(f"review_{request.surface}_{slot.slot_id}"))
        base_call_type = request.call_type or f"{request.surface}_review"
        assignment = ReviewAssignment(
            request=request, slot=slot, call_id=call_id, call_type=base_call_type,
            custody_root=self._custody_drive_root(),
            dispatch_stamp=self._review_paid_stamp,
        )
        executor = _review_route_executor(assignment, llm=self.llm)
        executor.usage_observer = lambda usage: self._emit_usage(request, slot, usage, prompt_chars=executor.prompt_chars())
        executor._logical_deadline_monotonic = logical_deadline_monotonic
        # The physical session and the logical waiter must share this exact
        # mutable cell.  A fresh cell is normally empty, so ``state or {}``
        # would silently replace it and hide a just-started invocation from a
        # timeout actor.
        executor.restore_custody(
            retry_state if retry_state is not None else {}
        )
        executor.set_pending_invocation_checkpoint(pending_invocation_checkpoint)
        prompt_projection = executor.prompt_payload()
        prompt_ref: Dict[str, Any] = {}
        response_ref: Dict[str, Any] = {}
        start = time.time()
        attempt_history = _ReviewAttemptHistory()
        try:
            prompt_ref = persist_call(
                self.drive_root,
                task_id=request.task_id or "review",
                call_id=f"{call_id}_prompt",
                call_type=f"{base_call_type}_prompt",
                payload={"request": asdict(request), "slot": asdict(slot), **prompt_projection},
                manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model},
            )
        except Exception:
            prompt_ref = {}
        free_refusal = (
            task_acceptance_zero_physical_refusal(request.evidence, retrieving=bool(slot.retrieves))
            if request.surface == "task_acceptance"
            else {}
        )
        if free_refusal:
            # Nothing was sent, so the row must not wear a verdict: it is a
            # typed $0 ``not_dispatched`` actor whose refusal token rides into
            # ``error`` so the aggregate names the real blocker.
            return self._error_actor(
                request, slot, f"{free_refusal['status']}: {free_refusal['summary']}",
                operation_id=call_id, operation_state="not_dispatched", prompt_ref=prompt_ref,
            )
        # Backstop for the quorum-sized packet ceiling: a narrower slot in the
        # same panel can still be handed more than it holds, and refusing it
        # costs nothing while the rest of the panel reviews.
        if request.surface == "task_acceptance" and not slot.retrieves:
            cap, estimated = acceptance_slot_fit(slot, executor, slot_input_caps=request.policy.get("slot_input_caps"))
            if cap and estimated > cap:
                return self._error_actor(
                    request, slot,
                    f"preflight_oversize: assembled acceptance prompt ~{estimated:,} tokens "
                    f"exceeds this slot's calibrated input cap {cap:,}",
                    operation_id=call_id, operation_state="not_dispatched", prompt_ref=prompt_ref,
                )
        owner_deadline = str(getattr(request, "deadline_at", "") or "")
        from ouroboros.config import get_finalization_grace_sec
        from ouroboros.deadline_utils import owner_deadline_exhausted
        # An exact delegated recovery has already crossed the paid boundary and
        # carries both the durable invocation token and its operation id.  It is
        # a settlement join, not a fresh dispatch, so the small custody window
        # selected by ``run_custodied_review_slots`` must remain usable after the
        # owner deadline.  The executor still validates the token and refuses a
        # missing/mismatched durable record before any new POST.
        recovery_token = str(
            (retry_state or {}).get("pending_invocation_id") or ""
        ).strip()
        exact_recovery = bool(recovery_token and str(operation_id or "").strip()
            and str(getattr(slot.route, "value", slot.route) or "") == "agent_session")
        if not exact_recovery and owner_deadline_exhausted(
            deadline_at=owner_deadline, reserve_sec=get_finalization_grace_sec(),
        ):
            return self._error_actor(
                request,
                slot,
                "Owner deadline exhausted before physical review dispatch",
                operation_id=call_id,
                operation_state="not_dispatched",
                prompt_ref=prompt_ref,
            )
        try:
            p3_actor = request.surface in {"multi_model_review", "scope_review"}
            acceptance_actor = request.surface == "task_acceptance"
            actor_attempts = 2 if (p3_actor or acceptance_actor) else 1
            # Acceptance and P3 share one two-send rail: transport/empty retry
            # or same-route format repair for PACKET rows; a retrieving row
            # (native episode, agent session) carries no send count — its
            # executor canonicalizes its own answer, so no repair resend below.
            from ouroboros.review_native_episode import native_or_packet_attempt_rail

            attempt_rail = native_or_packet_attempt_rail(
                slot, acceptance_actor or p3_actor)
            with attempt_rail:
                _prior_msg, _prior_usage, _prior_text = None, None, ""
                _last_msg, _last_usage, _last_text, _has_prior = None, None, "", False
                for actor_attempt in range(actor_attempts):
                    if actor_attempt and review_retry_cancelled(self.usage_ctx):
                        # Every retry shape (transport, malformed output, and
                        # empty output) shares this durable stop fence. The
                        # first response remains the forensic answer; no new
                        # paid physical attempt is authorized after cancel.
                        if _has_prior:
                            msg, usage, raw_text = _last_msg, _last_usage, _last_text
                            break
                        raise UsageAccountingError(
                            "review retry cancelled before physical dispatch"
                        )
                    if (
                        actor_attempt and logical_deadline_monotonic is not None
                        and time.monotonic() >= logical_deadline_monotonic
                    ):
                        if _has_prior:
                            msg, usage, raw_text = _last_msg, _last_usage, _last_text
                            break
                        raise TimeoutError("Review logical deadline expired before retry dispatch")
                    try:
                        # One seam; a null provider message is an empty actor.
                        attempt = _execute_slot_attempt(
                            assignment, llm=self.llm, executor=executor,
                        )
                        msg, usage, raw_text = attempt.message, attempt.usage, attempt.raw_text
                        attempt_history.observe()
                    except UsageAccountingError as exc:
                        attempt_history.observe(exc)
                        if isinstance(exc, PhysicalAttemptLimitExceeded):
                            # The rail raises only after its maximum claims have
                            # already succeeded; the current reservation is the
                            # released, unpaid one and must not erase those sends.
                            attempt_history.dispatched = True
                        # Budget/ledger/rail failures never trigger another send;
                        # retain a prior response, including an empty response,
                        # as forensic evidence.
                        if _has_prior:
                            msg, usage, raw_text = _prior_msg, _prior_usage, _prior_text
                            if _prior_msg is None:
                                msg, usage, raw_text = _last_msg, _last_usage, _last_text
                            break
                        raise
                    except Exception as exc:
                        attempt_history.observe(exc)
                        from ouroboros.review_custody import retryable_review_exception
                        if not retryable_review_exception(exc, self.usage_ctx, attempt_history):
                            raise
                        if actor_attempt + 1 < actor_attempts:
                            if (
                                logical_deadline_monotonic is not None
                                and time.monotonic() >= logical_deadline_monotonic
                            ):
                                raise
                            continue
                        if _has_prior:
                            # The repair RESEND failed (transport, timeout): keep the
                            # first answer, including an empty response, as forensics.
                            msg, usage, raw_text = _last_msg, _last_usage, _last_text
                            break
                        raise
                    _last_msg, _last_usage, _last_text, _has_prior = msg, usage, raw_text, True
                    if raw_text.strip():
                        if (
                            acceptance_actor and not slot.retrieves
                            and actor_attempt + 1 < actor_attempts
                            and parse_review_findings(raw_text)[0] is None
                        ):
                            _prior_msg, _prior_usage, _prior_text = msg, usage, raw_text
                            try:
                                # P1 forensics: a successful repair must not make the
                                # malformed first answer unreconstructible.
                                persist_call(
                                    self.drive_root,
                                    task_id=request.task_id or "review",
                                    call_id=f"{call_id}_attempt1_response",
                                    call_type=f"{base_call_type}_attempt1_response",
                                    payload={"message": msg, "usage": usage},
                                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "repair_attempt": 1},
                                )
                            except Exception:
                                log.warning(
                                    "Failed to persist malformed first acceptance attempt for %s/%s — the repair resend will overwrite it",
                                    request.surface, slot.slot_id, exc_info=True,
                                )
                            if review_retry_cancelled(self.usage_ctx):
                                # Cancellation is a durable owner decision,
                                # not a transport-only concern. Do not issue a
                                # format-repair resend after it linearizes.
                                break
                            continue  # extraction/format repair: one same-route resend
                        break
                    if _prior_text:
                        # Empty repair resend: keep the substantive malformed first
                        # answer instead of degrading to an empty actor.
                        msg, usage, raw_text = _prior_msg, _prior_usage, _prior_text
                        break
                    if actor_attempt + 1 >= actor_attempts:
                        break
            try:
                response_ref = persist_call(
                    self.drive_root,
                    task_id=request.task_id or "review",
                    call_id=f"{call_id}_response",
                    call_type=f"{base_call_type}_response",
                    payload={"message": msg, "usage": usage},
                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model},
                )
            except Exception:
                response_ref = {}
            return ReviewActorRecord(
                slot_id=slot.slot_id,
                model=slot.model,
                status="ok" if raw_text.strip() else "empty",
                raw_text=raw_text,
                usage=usage,
                prompt_ref=prompt_ref,
                response_ref=response_ref,
                duration_sec=round(time.time() - start, 3),
            )
        except Exception as exc:
            error_msg = truncate_review_artifact(str(exc), limit=4000)
            try:
                response_ref = persist_call(
                    self.drive_root,
                    task_id=request.task_id or "review",
                    call_id=f"{call_id}_error",
                    call_type=f"{base_call_type}_error",
                    payload={
                        "error_type": type(exc).__name__,
                        "error": sanitize_tool_result_for_log(error_msg),
                    },
                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "status": "error"},
                )
            except Exception:
                response_ref = {}
            (
                failure_custody, _capture_state, http_status,
                operation_state, failure_code,
            ) = _review_exception_projection(
                exc, executor.failure_custody(), attempt_history, retry_state,
            )
            return ReviewActorRecord(
                slot_id=slot.slot_id,
                model=slot.model,
                status="error",
                error=sanitize_tool_result_for_log(error_msg),
                transport_status=_transport_error_status(exc),
                failure_code=failure_code,
                reset_at=str(getattr(exc, "reset_at", "") or ""),
                http_status=http_status if isinstance(http_status, int) and http_status else None,
                usage=failure_custody,
                prompt_ref=prompt_ref,
                response_ref=response_ref,
                duration_sec=round(time.time() - start, 3),
                operation_state=operation_state,
            )

    def _emit_usage(
        self,
        request: ReviewRequest,
        slot: ReviewSlot,
        usage: Dict[str, Any],
        *,
        prompt_chars: int = 0,
    ) -> None:
        if self.usage_ctx is None:
            return
        try:
            from ouroboros.tools.review_helpers import emit_review_usage

            emit_review_usage(
                self.usage_ctx,
                model=str(usage.get("resolved_model") or slot.model),
                provider=str(usage.get("provider") or ""), usage=usage,
                source=f"review_substrate:{request.surface}",
                prompt_chars=prompt_chars,
                extra={"surface": request.surface, "slot_id": slot.slot_id},
            )
        except Exception:
            pass


def run_review_request(
    request: ReviewRequest,
    *,
    slots: List[ReviewSlot] | None = None,
    drive_root: pathlib.Path | None = None,
    llm: LLMClient | None = None,
    usage_ctx: Any = None,
) -> ReviewRunResult:
    resolved_slots = reviewer_slots(role_hint=request.surface) if slots is None else slots
    coordinator = ReviewCoordinator(llm=llm, drive_root=drive_root, usage_ctx=usage_ctx)
    result = coordinator.run(request, resolved_slots)
    if request.surface == "task_acceptance":
        # D-Q5 annotation-only pass: feeds the clean bit + disclosure, never parse
        # validity/quorum/verdicts. Called UNGUARDED on purpose — the annotator is
        # total and fail-CLOSED (a resolver failure stamps the non-clean row), and
        # `review_evidence` already built this packet, so swallowing an error here
        # could only turn "the host never checked the refs" into a clean PASS.
        from ouroboros.review_evidence import annotate_criteria_evidence_resolution

        annotate_criteria_evidence_resolution(result.actors, request.evidence)
    return result


# v7next F2.3a (D06): moved spans live in their owner leaves; re-exported
# here so this facade stays the single import surface for callers and tests.
from ouroboros.review_records import (  # noqa: E402, F401 -- intentional public re-exports
    HARDNESS_ADVISORY_VISIBLE,
    HARDNESS_HARD_GATE,
    HARDNESS_LABEL_ONLY,
    ReviewActorRecord,
    ReviewRequest,
    ReviewRunResult,
    ReviewSlot,
    TYPED_FAILURE_FACT_KEYS,
)

from ouroboros.review_verdict import (  # noqa: E402, F401 -- intentional public re-exports
    DIALOGUE_CONTINUE,
    DIALOGUE_INCONCLUSIVE,
    DIALOGUE_STABLE_DISAGREEMENT,
    DIALOGUE_STATUS_VALUES,
    DIALOGUE_TERMINAL_STATUSES,
    DIALOGUE_UNREACHABLE,
    DIALOGUE_VOTE_ABSTAIN_INVALID,
    DIALOGUE_VOTE_CONTINUE_WITHOUT_FINDINGS,
    _CRITERION_STATUSES,
    _TIER_ORDER,
    _contributing_actors,
    _criteria_have_supported_evidence,
    _criteria_shape_valid,
    _unresolved_evidence_ref_labels,
    aggregate_dialogue_status,
    aggregate_outcome_tier,
    build_improvement_capsule,
    dissent_findings,
    panel_reason,
    task_acceptance_is_clean,
)

from ouroboros.review_projection import (  # noqa: E402, F401 -- intentional public re-exports
    _public_review_reason,
    _response_ref_projection,
    _review_actor_projection,
    _review_enforcement_impact,
    _review_panel_id,
    _transport_error_status,
    build_review_binding,
    compact_review_projection,
)
