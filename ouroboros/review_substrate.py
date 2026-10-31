"""Shared multi-review substrate.

This module is the common cognitive primitive for migrated review surfaces and
the contract target for remaining legacy immune-system reviews. Slot identity is
separate from model identity, so duplicate model IDs are valid independent
reviewer slots.
"""

from __future__ import annotations

from ouroboros.config import runtime_setting
from ouroboros.model_wait import monotonic_now

from dataclasses import asdict, replace
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("review_substrate")

from ouroboros.llm import LLMClient
from ouroboros.observability import new_call_id, persist_call, read_call_payload, redact_projection  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.provider_models import provider_for_model  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.review_execution_projection import (
    MAX_PROJECTED_ACTOR_FINDINGS, projected_finding_row,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    review_executions_from_actor_usage,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)
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
    review_retry_cancelled, finalize_review_actor,
)
from ouroboros.triad_review import emit_review_run_errors
from ouroboros.usage_accounting import (
    PhysicalAttemptLimitExceeded,
    UsageAccountingError,
    UsageScope,
    current_usage_scope,
    last_physical_attempt_capture,
    usage_scope,
)
from ouroboros.utils import sanitize_tool_result_for_log
from ouroboros.utils import truncate_review_artifact  # noqa: F401 -- facade: review_verdict reads this at call time
from ouroboros._outcome_receipts import disclosed_list_projection  # noqa: F401 -- facade import surface; leaves read it through the call-time handle


class _CustodyUsageContext:
    """Forward custody state; the route owns the physical paid stamp.

    Refusals stay $0, sessions stamp at START_REQUESTED and API calls at
    durable dispatch. Exposing that stamp again would invoke it twice.
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
    # A bound body candidate is the SUBJECT; the adopted (serving) body stays the governance.
    system_raw = getattr(ctx, "serving_repo_dir", None) or getattr(ctx, "system_repo_dir", None)
    system = pathlib.Path(system_raw) if isinstance(system_raw, (str, pathlib.Path)) else None
    governance = (system or pathlib.Path(getattr(ctx, "repo_dir"))).resolve(strict=False)
    from ouroboros import body_candidate

    # The bound candidate is what the context authors and commits: it stays the subject
    # even when the task also carries a workspace/project folder as its active root.
    subject_raw = body_candidate.descriptor(ctx).get("path") if body_candidate.is_bound(ctx) else None
    subject = pathlib.Path(subject_raw or active_repo_dir_for(ctx)).resolve(strict=False)
    if not governance.is_dir() or not subject.is_dir():
        raise ValueError(f"unavailable governance/subject root: {governance} / {subject}")
    return governance, subject


# Shared tier vocabulary for the facade and its projection leaves.
from ouroboros.outcomes import OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED, OUTCOME_TIER_SOLVED  # noqa: F401 -- facade import surface; leaves read it through the call-time handle


# Historical dispatch names remain re-exported for existing consumers.
from ouroboros.review_dispatch import (  # noqa: E402,F401 — re-exports
    PLAN_SLOT_ID_PREFIX,
    ReviewPaidStamp,
    SCOPE_SLOT_ID_PREFIX,
    SLOT_ID_PREFIX,
    slot_id_for_row,
    stamp_review_paid_on_dispatch,
)


# triad_delivery_slots() (the pool under its historical name) lives in reviewer_slot_config (altitude, P7);
# re-exported for its historical callers and their tests here.
from ouroboros.reviewer_slot_config import triad_delivery_slots  # noqa: F401,E402


def review_usage_category(surface: str) -> str:
    """The usage-scope category every send of a review surface is attributed
    under — the key the ledger's cache split and the root telemetry's
    reservation identities carry, so the commit gate's admission names the
    same scope the substrate sends under."""
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
        from ouroboros.review_records import apply_review_model_override
        from ouroboros.model_wait import current_model_wait

        waiter = current_model_wait()
        if waiter and not request.reconcile_only:
            slots = [apply_review_model_override(slot, waiter.overrides) for slot in slots]
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

        if not request.reconcile_only and not request.checking_source:
            from ouroboros.review_checking import wave_checking_source
            request.checking_source = wave_checking_source(request.session_root, slots, data_root=self._custody_drive_root())

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
        if base_scope.root_limit_source or base_scope.root_limit_usd is not None:
            root_limit = base_scope.root_limit_usd
        else:
            try:
                configured_root_limit = float(
                    runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0
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
            non_task_operation=not bool((base_scope.task_id and not base_scope.non_task_operation)
                                        or (getattr(self.usage_ctx, "task_id", "")
                                            and getattr(self.usage_ctx, "task_lifecycle_bound", None) is not False)),
            review_skill=str(review_meta.get("review_skill") or base_scope.review_skill or ""),
            review_wave_id=resolve_review_wave(request, review_meta, base_scope.review_wave_id),
            cache_wave=str(review_meta.get("review_wave_id") or base_scope.cache_wave or ""),
            global_limit_usd=global_limit,
            global_limit_source=(base_scope.global_limit_source if base_scope.global_limit_usd is not None
                                 else "settings_budget_resolver"),
            global_limit_revision=(base_scope.global_limit_revision if base_scope.global_limit_usd is not None else None),
            root_limit_usd=root_limit,
            # A reviewer spends from its task's whole-work group, the original root's included.
            billing_group_id=base_scope.billing_group_id, billing_group_limit_usd=base_scope.billing_group_limit_usd,
            billing_group_limit_source=base_scope.billing_group_limit_source,
            billing_group_limit_revision=base_scope.billing_group_limit_revision,
            root_limit_source=base_scope.root_limit_source,
        )

        from ouroboros.review_custody import run_custodied_review_slots
        from ouroboros.review_dispatch import review_reconciliation_identity

        request.reconciliation_identity = review_reconciliation_identity(
            request, slots, root_task_id=root_task_id,
            contract=getattr(self.usage_ctx, "_current_review_contract_fingerprint", ""))

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
            settle_actor=lambda slot, actor: self._settle_actor(request, slot, actor),
            error_actor=lambda slot, error, operation_id="", operation_state="settled": self._error_actor(
                request, slot, error, operation_id=operation_id,
                operation_state=operation_state,
            ),
        )
        slot_order = {slot.slot_id: idx for idx, slot in enumerate(slots)}
        slots_by_id = {slot.slot_id: slot for slot in slots}
        actors.sort(key=lambda actor: slot_order.get(actor.slot_id, len(slot_order)))
        from ouroboros.review_seat_outcome import aggregate_outcomes

        for actor in actors:
            self._settle_actor(request, slots_by_id[actor.slot_id], actor, producer=False)
        aggregate = aggregate_outcomes(request, slots, actors)
        return ReviewRunResult(
            request=asdict(request),
            actors=[asdict(actor) for actor in actors],
            **aggregate,
            panel_id=_review_panel_id(request, actors),
            slot_roster=[asdict(slot) for slot in slots],
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
        from ouroboros.review_dispatch import review_operation_binding
        binding = review_operation_binding(request, slot, str(operation_id or call_id))
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
        if prompt_ref is None and operation_id:
            from ouroboros.observability import read_call_payload
            try:
                prompt_ref = read_call_payload(self._custody_drive_root(), task_id=request.task_id or "review",
                                               call_id=f"{operation_id}_prompt")[2]
            except (OSError, ValueError):
                pass  # No retained assignment: disclose only the synthetic failure prompt.
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
            recovery_binding=binding,
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
        from ouroboros.review_dispatch import review_operation_binding
        binding = review_operation_binding(request, slot, str(operation_id or call_id))
        assignment = ReviewAssignment(
            request=request, slot=slot, call_id=call_id, call_type=base_call_type,
            custody_root=self._custody_drive_root(),
            dispatch_stamp=self._review_paid_stamp,
        )
        executor = _review_route_executor(assignment, llm=self.llm)
        from ouroboros.review_seat_outcome import answer_received

        executor.answer_observer = lambda: answer_received(self.usage_ctx, request, slot)
        executor.usage_observer = lambda usage: self._emit_usage(request, slot, usage, prompt_chars=executor.prompt_chars())
        executor._logical_deadline_monotonic = logical_deadline_monotonic
        # The physical session and the logical waiter must share this exact
        # mutable cell.  A fresh cell is normally empty, so ``state or {}``
        # would silently replace it and hide a just-started invocation from a
        # timeout actor.
        executor.restore_custody(
            retry_state if retry_state is not None else {}
        )
        invocation = {"id": str((retry_state or {}).get("pending_invocation_id") or "")}
        def checkpoint(invocation_id):
            invocation["id"] = str(invocation_id)
            if callable(pending_invocation_checkpoint):
                pending_invocation_checkpoint(invocation_id)
        executor.set_pending_invocation_checkpoint(checkpoint)
        from ouroboros.review_session_custody import exact_recovery_join
        exact_recovery = exact_recovery_join(executor, self._custody_drive_root(), retry_state, operation_id, slot,
                                             replay=request.reconcile_only)
        prompt_projection = executor.prompt_payload()
        prompt_ref: Dict[str, Any] = {}
        response_ref: Dict[str, Any] = {}
        start = time.time()
        attempt_history = _ReviewAttemptHistory(incoming_capture=last_physical_attempt_capture())
        try:
            # A collector must not replace the original immutable prompt with
            # today's reconstruction or a redacted projection of the raw request.
            prompt_ref = read_call_payload(self._custody_drive_root(), task_id=request.task_id or "review",
                                           call_id=f"{call_id}_prompt")[2] if exact_recovery else persist_call(
                self._custody_drive_root(),
                task_id=request.task_id or "review",
                call_id=f"{call_id}_prompt",
                call_type=f"{base_call_type}_prompt",
                payload={"request": asdict(request), "slot": asdict(slot), **prompt_projection},
                manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model,
                          "review_operation_binding": binding},
            )
        except Exception:
            prompt_ref = {}
        owner_deadline = str(getattr(request, "deadline_at", "") or "")
        from ouroboros.config import get_finalization_grace_sec
        from ouroboros.deadline_utils import owner_deadline_exhausted
        # An exact delegated recovery has already crossed the paid boundary and
        # carries both the durable invocation token and its operation id.  It is
        # a settlement join, not a fresh dispatch, so the small custody window
        # selected by ``run_custodied_review_slots`` must remain usable after the
        # owner deadline.  The executor still validates the token and refuses a
        # missing/mismatched durable record before any new POST.
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
            actor_attempts = 2 if p3_actor else 1
            # P3 PACKET rows share one two-send rail: one transport/empty retry;
            # a retrieving row (native episode, agent session) carries no send
            # count — its executor canonicalizes its own answer.
            from ouroboros.review_native_episode import native_or_packet_attempt_rail

            attempt_rail = native_or_packet_attempt_rail(slot, p3_actor)
            with attempt_rail:
                _last_msg, _last_usage, _last_text, _has_prior = None, None, "", False
                for actor_attempt in range(actor_attempts):
                    if actor_attempt and review_retry_cancelled(self.usage_ctx):
                        # Every retry shape (transport and empty output)
                        # shares this durable stop fence. The
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
                        and monotonic_now() >= logical_deadline_monotonic
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
                                and monotonic_now() >= logical_deadline_monotonic
                            ):
                                raise
                            continue
                        if _has_prior:
                            # The empty-answer RESEND failed (transport, timeout): keep
                            # the first answer, including an empty response, as forensics.
                            msg, usage, raw_text = _last_msg, _last_usage, _last_text
                            break
                        raise
                    _last_msg, _last_usage, _last_text, _has_prior = msg, usage, raw_text, True
                    if raw_text.strip():
                        break
                    if actor_attempt + 1 >= actor_attempts:
                        break
            if raw_text.strip() and not slot.retrieves:
                executor.answer_observer()
            actor = ReviewActorRecord(
                slot_id=slot.slot_id, model=slot.model,
                status="ok" if raw_text.strip() else "empty", raw_text=raw_text,
                usage=usage, prompt_ref=prompt_ref,
                duration_sec=round(time.time() - start, 3),
                recovery_binding={**binding, "pending_invocation_id": invocation["id"]},
            )
            finalize_review_actor(actor, operation_id=call_id)
            self._finalize_outcome(request, slot, actor)
            actor.response_ref = self._persist_producer_outcome(
                request, actor, call_id, base_call_type, {"message": msg, "usage": usage})
            return actor
        except Exception as exc:
            # Preserve the complete safe cause before any presentation bound.
            # A measurement after a long engine sentence must remain recoverable.
            error_msg = sanitize_tool_result_for_log(str(exc))
            (
                failure_custody, _capture_state, http_status,
                operation_state, failure_code,
            ) = _review_exception_projection(
                exc, executor.failure_custody(), attempt_history, retry_state,
            )
            actor = ReviewActorRecord(
                slot_id=slot.slot_id,
                model=slot.model,
                status="error",
                error=sanitize_tool_result_for_log(error_msg),
                transport_status=_transport_error_status(
                    exc, failure_phase=str(failure_custody.get("review_failure_phase") or "")),
                failure_code=failure_code,
                reset_at=str(getattr(exc, "reset_at", "") or ""),
                reported_cause=str(getattr(exc, "reported_cause", "") or ""),
                failure_details=redact_projection(getattr(exc, "failure_details", {}) or {}).value,
                http_status=http_status if isinstance(http_status, int) and http_status else None,
                usage=failure_custody,
                prompt_ref=prompt_ref,
                response_ref=response_ref,
                duration_sec=round(time.time() - start, 3),
                operation_state=operation_state,
                recovery_binding={**binding, "pending_invocation_id": invocation["id"]},
            )
            finalize_review_actor(actor, operation_id=call_id)
            self._finalize_outcome(request, slot, actor)
            actor.response_ref = self._persist_producer_outcome(
                request, actor, call_id, base_call_type,
                {"error_type": type(exc).__name__, "error": actor.error})
            return actor

    def _finalize_outcome(self, request, slot, actor):
        from ouroboros.review_seat_outcome import finalize_seat_outcome

        finalize_seat_outcome(request, slot, actor)
        emit_review_run_errors(self.usage_ctx, request, [actor], drive_root=self.drive_root)

    def _settle_actor(self, request, slot, actor, *, producer=True):
        """Outer worker failures take the same durable path before custody advertises readiness."""
        from ouroboros.review_seat_outcome import ensure_producer_prompt, publish_seat_execution

        complete = bool(actor.outcome and actor.response_ref)
        self._finalize_outcome(request, slot, actor)
        if producer and not complete:
            actor.prompt_ref = ensure_producer_prompt(self._custody_drive_root(), request, slot, actor)
            actor.response_ref = self._persist_producer_outcome(
                request, actor, actor.operation_id, request.call_type or f"{request.surface}_review",
                getattr(actor, "_producer_payload", {"message": {"content": actor.raw_text}, "error": actor.error}))
            if not actor.response_ref:
                raise OSError(f"Review producer outcome is not durable: {actor.operation_id}")
        if producer or not actor.late_result_pending:
            publish_seat_execution(request, slot, actor, self.usage_ctx, fallback_root=self._custody_drive_root())
        else:
            from ouroboros.subagent_metric_events import review_execution
            try:  # a measurement never decides a pending drain
                review_execution(request, slot, actor, self.usage_ctx, fallback_root=self._custody_drive_root())
            except Exception:
                log.debug("review measurement unavailable", exc_info=True)

    def _persist_producer_outcome(self, request, actor, call_id, call_type, payload):
        """Stamp the completed producer beside its existing full response body."""
        actor._producer_payload = payload
        outcome = {key: value for key, value in asdict(actor).items()
                   if key not in {"raw_text", "usage", "prompt_ref", "response_ref"}}
        try:
            return persist_call(
                self._custody_drive_root(), task_id=request.task_id or "review",
                call_id=f"{call_id}_{'error' if actor.status == 'error' else 'response'}",
                call_type=f"{call_type}_{'error' if actor.status == 'error' else 'response'}",
                payload={**payload, "usage": actor.usage, "producer_outcome": outcome},
                manifest={"surface": request.surface, "slot_id": actor.slot_id,
                          "model": actor.model, "producer_complete": True,
                          **({"status": "error"} if actor.status == "error" else {}),
                          "review_operation_binding": actor.recovery_binding},
            )
        except Exception:
            log.warning("Review producer outcome could not be retained: %s", call_id, exc_info=True)
            return {}

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

            session = bool(usage.get("delegated_run_started") or usage.get("delegated_run_id"))
            emit_review_usage(
                self.usage_ctx,
                # A session row names its OBSERVED model or none: the seat's requested route
                # label rides the ``claudexor`` receipt, never this field (#1578).
                model=str(usage.get("resolved_model") or ("" if session else slot.model)),
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
    slots: List[ReviewSlot],
    drive_root: pathlib.Path | None = None,
    llm: LLMClient | None = None,
    usage_ctx: Any = None,
) -> ReviewRunResult:
    """Run ONE review wave on ``slots`` — the rows the caller took from the pool
    (``review_pool_slots`` / ``triad_delivery_slots``) or composed for its wave;
    the substrate never picks reviewers itself."""
    coordinator = ReviewCoordinator(llm=llm, drive_root=drive_root, usage_ctx=usage_ctx)
    return coordinator.run(request, slots)


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
    resolve_review_wave,
)

from ouroboros.review_verdict import (  # noqa: E402, F401 -- intentional public re-exports
    DIALOGUE_STATUS_VALUES,
    _TIER_ORDER,
    _contributing_actors,
    aggregate_outcome_tier,
    panel_reason,
)

from ouroboros.review_projection import (  # noqa: E402, F401 -- intentional public re-exports
    _public_review_reason,
    _response_ref_projection,
    _review_actor_projection,
    _review_enforcement_impact,
    _review_panel_id,
    _transport_error_status,
    compact_review_projection,
)
