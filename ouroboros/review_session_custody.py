"""Recovery and pre-dispatch custody helpers for delegated review sessions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def owned_started_review_custody(
    custody: Any,
    custody_drive: Any,
    record: Dict[str, Any],
    claimant_task_id: str,
) -> tuple[str, Any]:
    """Return existing run custody only after exact owner corroboration."""
    from ouroboros.review_execution import ReviewRouteUnavailable

    run_id = str(record.get("run_id") or "")
    ownership, found = custody.lookup(custody_drive, claimant_task_id, run_id)
    invocation_owner = str(record.get("task_id") or "")
    custody_owner = str(getattr(found, "task_id", "") or "")
    if (
        ownership != custody.OWNED
        or found is None
        or invocation_owner != claimant_task_id
        or custody_owner != claimant_task_id
    ):
        raise ReviewRouteUnavailable(
            "delegated review started-run recovery could not corroborate "
            f"ownership for run {run_id} (lookup={ownership}, "
            f"claimant={claimant_task_id!r}, invocation_owner={invocation_owner!r}, "
            f"custody_owner={custody_owner!r})",
            code="review_recovery_ownership_unverified",
        )
    return run_id, found


def recorded_review_shape(run_request: Any) -> Optional[Any]:
    """The run shape a stored review request recorded, or None when no review runs it.

    Ask/readonly is every request recorded before configured reviewer rights and
    every read-only seat. A seat with configured ``workspace_write``/``full``
    access recorded Agent mode with its private checking copy as the delegated
    live ``workspaceRoot``. Recovery replays exactly this shape — never today's
    configuration — so an old read-only invocation is never relabelled.
    """
    from ouroboros.subagents import DelegatedRunShape

    mode, access = run_request.get("mode"), run_request.get("access")
    execution = run_request.get("execution")
    if mode == "ask" and access == "readonly" and "execution" not in run_request:
        return DelegatedRunShape(access="readonly", mode="ask")
    if mode == "agent" and access in {"workspace_write", "full"} and (run_request.get("_use_thread") is True and (
            "execution" not in run_request) or isinstance(execution, dict) and execution.get("isolation") == "live"
            and execution.get("delegated") is True and str(execution.get("workspaceRoot") or "")):
        return DelegatedRunShape(access=access, mode="agent", isolation="live", delegated=True,
                                 investigation=True)
    return None


def review_recovery_facts(
    record: Dict[str, Any],
    run_request: Any,
    started_custody: Any,
    *,
    prompt: str,
    root: str,
    claimant_task_id: str,
    claimant_surface: str,
    claimant_slot_id: str,
    claimant_operation_id: str,
) -> tuple[Any, str, str, str, bool]:
    """Validate one stored request and restore immutable delivery facts."""
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.subagents import DelegationRoute

    if not isinstance(run_request, dict) or not run_request:
        raise ReviewRouteUnavailable(
            "delegated review recovery has no canonical stored request; "
            "the existing invocation is not re-derived",
            code="review_recovery_request_missing",
        )
    invocation_owner = str(record.get("task_id") or "")
    if invocation_owner != claimant_task_id:
        raise ReviewRouteUnavailable(
            "delegated review pending-invocation recovery could not corroborate "
            f"ownership (claimant={claimant_task_id!r}, "
            f"invocation_owner={invocation_owner!r})",
            code="review_recovery_ownership_unverified",
        )
    stored_binding = (
        str(record.get("surface") or ""), str(record.get("slot_id") or ""),
        str(record.get("operation_id") or ""),
    )
    claimant_binding = (
        str(claimant_surface or ""), str(claimant_slot_id or ""),
        str(claimant_operation_id or ""),
    )
    if stored_binding != claimant_binding:
        raise ReviewRouteUnavailable(
            "delegated review retry token belongs to a different surface, slot, "
            "or physical operation; the recorded invocation is not replayed",
            code="review_recovery_request_mismatch",
        )
    scope = run_request.get("scope")
    route_id = str(run_request.get("primaryHarness") or "")
    if (
        not isinstance(scope, dict)
        or str(scope.get("kind") or "") != "project"
        or not str(scope.get("root") or "")
        or not route_id
        or run_request.get("harnesses") != [route_id]
        or str(record.get("route") or "") != route_id
        or str(run_request.get("authPreference") or "") != "subscription"
        or recorded_review_shape(run_request) is None
    ):
        raise ReviewRouteUnavailable(
            "delegated review recovery has an invalid canonical stored request; "
            "the durable retry token remains untouched and nothing was re-posted",
            code="review_recovery_request_invalid",
        )
    stored_root = str(scope.get("root") or "")
    if str(run_request.get("prompt") or "") != prompt or stored_root != root:
        raise ReviewRouteUnavailable(
            "delegated review retry replays a recorded invocation whose "
            f"{'prompt' if stored_root == root else 'session root'} differs from "
            "this call's; the durable retry token remains untouched and the "
            "recorded invocation is not replayed against a different review",
            code="review_recovery_request_mismatch",
        )
    if started_custody is not None:
        route_id = str(started_custody.route_id or "")
        model = str(started_custody.model or "")
        project_id = str(started_custody.project_id or "")
        project_owned = bool(started_custody.project_owned)
        key = str(started_custody.idempotency_key or "")
    else:
        model = str(run_request.get("model") or "")
        project_id = str(record.get("project_id") or "")
        project_owned = bool(record.get("project_owned"))
        key = str(record.get("idempotency_key") or "")
    route = DelegationRoute(
        route_id=route_id,
        model=model,
        effort=str(run_request.get("effort") or ""),
        profile_id=str(run_request.get("credentialProfileId") or ""),
    )
    existing_project = "" if project_owned else project_id
    return route, project_id, existing_project, key, "outputSchema" in run_request


def checkpoint_pending_invocation(
    *,
    checkpoint: Any,
    invocation_id: str,
    state: Dict[str, Any],
    on_failure: Callable[[], None],
) -> None:
    """Bind a pending invocation to its durable slot before provider POST.

    The binding waits out a busy review-state lock itself. A refusal names its cause: a reserved
    slot that cannot take this token keeps its own reason (``review_custody_slot_refused``); a
    review state that could not be written keeps its lock or I/O facts
    (``review_custody_checkpoint_unwritable``)."""
    if not callable(checkpoint):
        return
    try:
        checkpoint(invocation_id)
    except Exception as exc:
        from ouroboros.review_execution import ReviewRouteUnavailable
        from ouroboros.gateways.claudexor import run_failure_cause
        from ouroboros.review_state import ReservedReviewSlotRefused

        cause = str(getattr(exc, "reported_cause", "") or f"{type(exc).__name__}: {exc}")
        try:
            on_failure()
        except Exception as cleanup_error:
            cause += f"; cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        state.pop("pending_invocation_id", None)
        if isinstance(exc, ReservedReviewSlotRefused):
            refusal = ReviewRouteUnavailable(
                f"the delegated review invocation was refused by its reserved commit-review slot ({exc}); "
                "no run was started", code="review_custody_slot_refused")
        else:
            refusal = ReviewRouteUnavailable(
                "the delegated review invocation could not be bound to its durable "
                "commit-review slot before dispatch; no run was started",
                code="review_custody_checkpoint_unwritable",
            )
        refusal.reported_cause = run_failure_cause({"safeMessage": cause})
        raise refusal from exc


def exact_recovery_join(executor: Any, custody_root: Any, retry_state: Optional[Dict[str, Any]],
                        operation_id: str, slot: Any, *, replay: bool) -> bool:
    """Is this seat an exact delegated recovery? When it only collects, restore its RAW request.

    Such a recovery has already crossed the paid boundary: a session seat carrying
    both its pending invocation token and its operation id. A collection (``replay``)
    sends nothing of its own: the RAW request recorded before the POST is the
    transport source, never a prompt rebuilt from mutable history or the redacted,
    overwritable ``{operation}_prompt`` record (#1582).
    """
    from ouroboros.delegate_custody import invocation_record

    token = str((retry_state or {}).get("pending_invocation_id") or "").strip()
    session = str(getattr(slot.route, "value", slot.route) or "") == "agent_session"
    if not (token and str(operation_id or "").strip() and session):
        return False
    stored = (invocation_record(custody_root, token) or {}).get("request") if replay else None
    if isinstance(stored, dict) and isinstance(stored.get("prompt"), str):
        executor._session_prompt = stored["prompt"]
    return True


def validate_producer_invocation(root: Any, token: str, expected: dict, payload: dict,
                                outcome: dict, *, error_source: bool) -> None:
    """Session-specific proof behind a complete, already-bound producer outcome.

    Only positive preparation provenance needs no START_REQUESTED row. An absent
    token, contradictory start facts or a legacy source never establish that proof.
    """
    from ouroboros.delegate_custody import invocation_record

    usage = payload.get("usage") or {}
    if (error_source and usage.get("delegated_start_unrequested") is True
            and outcome.get("status") == "error" and bool(outcome.get("error"))
            and not (token or usage.get("delegated_run_id") or usage.get("delegated_run_started"))):
        return
    invocation = invocation_record(root, token) if token else None
    if invocation is not None and invocation.get("state") == "legacy_gap":
        raise ValueError(f"legacy_gap: {invocation['reason']}; completed producer custody needs repair")
    # A binding names the facts its request carried: a request without a reconciliation
    # identity binds no root, which is an unbound fact, never a contradiction.
    if not invocation or any(str(invocation.get(k) or "") != str(expected[k] or "")
                             for k in ("task_id", "root_task_id", "surface", "slot_id", "operation_id")
                             if k in expected):
        raise ValueError("completed producer has no exact delegated invocation")
    run_id = str(usage.get("delegated_run_id") or "")
    if run_id and run_id != str(invocation.get("run_id") or ""):
        if not review_continuation_descends_from(root, run_id, invocation):
            raise ValueError("completed producer delegated run mismatch")


def review_continuation_descends_from(root: Any, run_id: str, anchor: dict) -> bool:
    """Prove that a collected successor belongs to the slot's original invocation."""
    from ouroboros import delegate_custody as custody

    seen = set()
    while run_id and run_id not in seen:
        if run_id == str(anchor.get("run_id") or ""):
            return True
        seen.add(run_id)
        ownership, entry = custody.lookup(root, anchor["task_id"], run_id)
        if ownership != custody.OWNED or entry is None or not entry.continuation_of:
            return False
        record = custody.invocation_record(root, entry.invocation_id)
        if not record or any(record.get(k) != anchor.get(k)
                             for k in ("task_id", "root_task_id", "surface", "slot_id", "operation_id")):
            return False
        if record.get("request") != {"continueFrom": entry.continuation_of}:
            return False
        run_id = entry.continuation_of
    return False


def continue_review_run(gateway: Any, root: Any, entry: Any, detail: dict, invocation: Any,
                        *, remaining: float, seen: set[str]) -> Any:
    """One idempotent continuation per stopped run, under the same seat anchor.

    The original slot token stays valid across crashes before or after POST.
    A deterministic successor invocation records the exact continuation body
    before sending; recovery joins its started run or replays that same body.
    No new paid-wave stamp or generic task-delegation authority is introduced.
    """
    from types import SimpleNamespace
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_continuation import engine_continues
    from ouroboros.delegate_collection import record_terminal_money
    from ouroboros.owner_pause import OwnerPauseRefused, run_operation
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    token = custody.idempotency_key("review_continue", entry.invocation_id, entry.run_id)
    record = custody.invocation_record(root, token)
    body = {"continueFrom": entry.run_id}
    if record:
        if (record.get("request") != body or record.get("task_id") != entry.task_id
                or record.get("surface") != invocation.surface or record.get("slot_id") != invocation.slot_id
                or record.get("operation_id") != invocation.operation_id):
            raise ReviewRouteUnavailable("Review continuation custody does not match this seat",
                                         code="review_recovery_request_mismatch")
        if record["state"] == "failed_definite":
            return None
        if record["state"] == "started":
            run_id, successor = owned_started_review_custody(custody, root, record, entry.task_id)
            if run_id in seen or successor.continuation_of != entry.run_id:
                return None
            return successor
    else:
        summary = custody.summary_of(detail)
        resumable = summary.get("resumable") or detail.get("resumable")
        if not isinstance(resumable, dict) or not resumable or not engine_continues(gateway):
            return None
    from ouroboros.budget_pause import dispatch_fenced

    if remaining <= 0 or dispatch_fenced(entry.task_id):
        return None
    if not record:
        requested = custody.record_start_requested(root,
            run_id="", task_id=entry.task_id, invocation_id=token, idempotency_key=token,
            operation_id=invocation.operation_id, request=body, continuation_of=entry.run_id,
            surface=invocation.surface, slot_id=invocation.slot_id, route=entry.route_id,
            max_seconds=entry.max_seconds, project_id=entry.project_id, project_owned=entry.project_owned,
            **{key: getattr(entry, key) for key in ("root_task_id", "parent_task_id", "category", "source",
                "review_skill", "review_wave_id", "review_slot_id", "authority_source", "execution_root",
                "target_root")})
        if not requested:
            raise ReviewRouteUnavailable("Review continuation request could not be recorded; nothing was sent",
                                         code="start_request_row_unwritable")
    # Keep the registration/checking copy alive for the successor. The final
    # seat settlement retires it; this step records only the stopped run's money.
    source = SimpleNamespace(drive_root=root, task_id=entry.task_id, root_task_id=entry.root_task_id)
    try:
        # A continuation is an ordinary start naming the stopped run; the engine picks native/moved/packet.
        handle = run_operation(source, lambda: gateway.start_run({"continueFrom": entry.run_id},
                                                                 idempotency_key=token))
    except (ClaudexorUnavailable, OwnerPauseRefused) as exc:
        if isinstance(exc, OwnerPauseRefused) or 400 <= int(getattr(exc, "status_code", 0) or 0) < 500:
            if not custody.emit(root, custody.START_FAILED, {"invocation_id": token, "task_id": entry.task_id,
                    "reason": str(getattr(exc, "code", "continuation_refused")), "definite": True,
                    "surface": invocation.surface, "slot_id": invocation.slot_id}):
                raise
            return None  # Caller retains the stopped run's original typed failure.
        raise  # Ambiguous acceptance remains replayable from the same token.
    run_id = str(handle.get("runId") or handle.get("jobId") or "")
    if not run_id or run_id in seen:
        raise ReviewRouteUnavailable("Review continuation returned no new run handle",
                                     code="invalid_continuation_handle")
    record_terminal_money(root, entry, detail)
    successor = custody.RunCustody(run_id=run_id, invocation_id=token, idempotency_key=token,
        continuation_of=entry.run_id, ledger_root=str(root),
        **{key: getattr(entry, key) for key in ("task_id", "route_id", "model", "profile_id", "project_id",
            "project_owned", "project_persistent", "root_task_id", "parent_task_id", "category", "source",
            "review_skill", "review_wave_id", "review_slot_id", "authority_source", "execution_root", "target_root")})
    if not custody.record_started(root, successor, shape={
            "access": entry.access, "mode": entry.mode, "isolation": entry.isolation,
            "delegated": entry.delegated, "max_seconds": entry.max_seconds,
            "surface": invocation.surface, "slot_id": invocation.slot_id}):
        raise ReviewRouteUnavailable("Review successor start remains pending in custody",
                                     code="review_continuation_checkpoint_unwritable")
    return successor
