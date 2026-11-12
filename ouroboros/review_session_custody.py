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


def join_recorded_review_continuation(gateway: Any, root: Any, entry: Any, detail: dict, invocation: Any,
                                      *, remaining: float, seen: set[str]) -> Any:
    """Join a successor an older host recorded automatically under this seat's anchor; never create one.

    Continuation is now the author's explicit choice (``review_continuation``): a new operation
    collected through the existing exact recovery. A legacy successor already recorded under
    ``("review_continue", invocation, run)`` is joined when started, or its UNKNOWN POST is replayed
    with the same key and recorded body; a definite refusal or a missing record joins nothing.
    """
    from types import SimpleNamespace
    from ouroboros import delegate_custody as custody
    from ouroboros.budget_pause import dispatch_fenced
    from ouroboros.delegate_collection import record_terminal_money
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.owner_pause import OwnerPauseRefused, run_operation
    from ouroboros.review_execution import ReviewRouteUnavailable

    token = custody.idempotency_key("review_continue", entry.invocation_id, entry.run_id)
    record = custody.invocation_record(root, token)
    body = {"continueFrom": entry.run_id}
    if not record or record["state"] == "failed_definite":
        return None
    if (record.get("request") != body or record.get("task_id") != entry.task_id
            or record.get("surface") != invocation.surface or record.get("slot_id") != invocation.slot_id
            or record.get("operation_id") != invocation.operation_id):
        raise ReviewRouteUnavailable("Review continuation custody does not match this seat",
                                     code="review_recovery_request_mismatch")
    if record["state"] == "started":
        run_id, successor = owned_started_review_custody(custody, root, record, entry.task_id)
        return None if run_id in seen or successor.continuation_of != entry.run_id else successor
    if remaining <= 0 or dispatch_fenced(entry.task_id):
        return None
    source = SimpleNamespace(drive_root=root, task_id=entry.task_id, root_task_id=entry.root_task_id)
    try:  # the recorded legacy POST may have been accepted: replay its exact key and body only
        handle = run_operation(source, lambda: gateway.start_run(body, idempotency_key=token))
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
    record_terminal_money(root, entry, detail)  # the stopped run's money; the final settlement takes the successor's
    successor = custody.RunCustody(run_id=run_id, invocation_id=token, idempotency_key=token,
        continuation_of=entry.run_id, ledger_root=str(root), daemon_store_id=custody.admitted_store_id(handle),
        **{key: getattr(entry, key) for key in ("task_id", "route_id", "model", "profile_id", "project_id",
            "project_owned", "project_persistent", "root_task_id", "parent_task_id", "category", "source",
            "review_skill", "review_wave_id", "review_slot_id", "authority_source", "execution_root", "target_root")})
    successor, landed = custody.bind_started(root, successor, shape={  # converges with any reader of this key
            "access": entry.access, "mode": entry.mode, "isolation": entry.isolation,
            "delegated": entry.delegated, "max_seconds": entry.max_seconds,
            "surface": invocation.surface, "slot_id": invocation.slot_id})[:2]
    if not landed:
        raise ReviewRouteUnavailable("Review successor start remains pending in custody",
                                     code="review_continuation_checkpoint_unwritable")
    return successor


def rejoin_recorded_review(invocation: Any, data_root: Any, gateway: Any, record: Dict[str, Any],
                           run_id: str, started: Any) -> tuple[str, Any, Optional[float]]:
    """A recovered seat's first step on its recorded engine: ``(run_id, started custody, generation end)``.

    A pending start (its key: the seat's own token, ``retry_state["pending_invocation_id"]``) is first
    read for the work the engine already accepted (#3138, ``delegate_accepted.recover``); found, it is
    bound once and joined like a started one. The panel owns this read (the sweep leaves review starts
    to it), after ``review_recovery_facts`` proved the seat owns the record. It precedes the checking-copy restoration
    and the generation window (``prepare_continuation_send``, ``continuation_generation_end``), which
    bound only a new POST: accepted work is collected after its window closed or its released checkout
    went. A read that forbids any replay refuses typed with the start kept; any other miss leaves the
    send exactly as before, an explicit continuation's first send included."""
    if not run_id and record.get("state") == "pending":
        from ouroboros.delegate_accepted import NO_REPLAY, recover

        token = str((getattr(invocation, "retry_state", None) or {}).get("pending_invocation_id") or "")
        held, read = recover(data_root, gateway, {**record, "invocation_id": token})
        if held is not None:
            run_id, started = held.run_id, held
        elif read["reason"] in NO_REPLAY:
            from ouroboros.review_execution import ReviewRouteUnavailable

            raise ReviewRouteUnavailable(f"the engine answered for this review's recorded start ({read['lookup']}) but no "
                                         "run could be bound to it now; nothing was re-sent and its outcome stays unknown",
                                         code=read["reason"])
    prepare_continuation_send(invocation, data_root, record, run_id)  # every continuation POST, replays too
    return run_id, started, continuation_generation_end(invocation, record, run_id)


def continuation_send(invocation: Any) -> bool:
    """Is this send an explicit continuation's successor (``review_continuation``), whose start its decision recorded?"""
    from ouroboros.review_continuation import PREFIX

    return str(getattr(invocation, "operation_id", "") or "").startswith(PREFIX)


def prepare_continuation_send(invocation: Any, data_root: Any, record: Optional[Dict[str, Any]], run_id: str) -> None:
    """Before EVERY POST of an explicit continuation's recorded start (no run bound yet), the first send and a crash or
    restart replay alike: the released read-only checkout its START names is put back from the START row's exact
    mapping (``review_continuation.RESTORE_KEY``, ``review_subject.restore_released_subject``). A failure raises before
    any POST, so the start stays pending, decided and unsent, for a later collection to repair; a start with no
    mapping (a writable seat, a live root, an older row) is untouched."""
    from ouroboros.review_continuation import RESTORE_KEY

    mapping = (record or {}).get(RESTORE_KEY)
    if run_id or not continuation_send(invocation) or not isinstance(mapping, dict) or not mapping.get("root"):
        return
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.tools.review_subject import restore_released_subject

    try:
        restore_released_subject(data_root, mapping)
    except ValueError as exc:
        raise ReviewRouteUnavailable(f"the reviewed checkout this continuation reads is unavailable: {exc}; its "
                                     "recorded start was not POSTed", code="review_subject_unavailable") from exc


def continuation_generation_end(invocation: Any, record: Optional[Dict[str, Any]], run_id: str) -> Optional[float]:
    """When an explicit continuation's successor must stop generating, on this process's monotonic clock: the instant
    its START recorded (``generation_end_at``: the original window narrowed by the owner deadline and task window
    when the projection read it, ``review_continuation._start_requested``), never rebased on the START's own later
    timestamp, whatever cap its immutable bytes carry. The executor polls no longer than that and cancels; custody's
    settlement margin collects receipts only. Past it, a start with no run bound is never POSTed, because that POST
    could begin new work: its outcome stays unknown (``custody_lost``), its key and bytes kept. ``None`` for any
    other send, or a start recorded with no window. Read on custody's clock (``review_custody.utc_now``)."""
    import time

    from ouroboros import review_custody
    from ouroboros.deadline_utils import parse_deadline_ts

    closes = parse_deadline_ts(str((record or {}).get("generation_end_at") or ""))
    if not continuation_send(invocation) or closes is None:
        return None
    end = time.monotonic() + (closes - review_custody.utc_now()).total_seconds()
    if not run_id:
        _refuse_past(end)
    return end


def continuation_send_admitted(invocation: Any, end: Optional[float], request: Dict[str, Any], model: str) -> None:
    """Admit the POST of an explicit continuation's start (``continuation_send``) in the send callback itself, after
    its gateway, START, checkpoint and launch steps, since any POST of a start with no run bound may begin new paid
    work: known spend is below every budget fence, as every review wave is admitted (``review_wave_budget_gate``:
    the global, root and original-group fences, fail-open on an unknown), and its window (``end``,
    ``continuation_generation_end``) has not closed. A refusal keeps its key and bytes: its outcome stays UNKNOWN,
    never a definite absence. Every other send, and every known run's collection, passes untouched."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_substrate import review_usage_category
    from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate

    if not continuation_send(invocation):
        return
    admission = review_wave_budget_gate(
        SimpleNamespace(task_id=invocation.task_id), surface=invocation.surface, models=[model],
        prompt_chars=[len(str(request.get("prompt") or ""))], categories=[review_usage_category(invocation.surface)],
        slot_ids=[invocation.slot_id], extra={"continuation": True, "at_send": True})
    if admission is not None:
        fence, remedy = review_wave_binding_fence(admission)
        raise ReviewRouteUnavailable(f"known spend has reached the {fence}, so this continuation's recorded start was "
                                     "not POSTed: its outcome stays unknown, its key and bytes kept, and no collection "
                                     f"re-posts it. To review again, {remedy} first.", code="continuation_budget_insufficient")
    _refuse_past(end)


def _refuse_past(end: Optional[float]) -> None:
    """Refuse the POST of a continuation start at or after ``end``: UNKNOWN, its key and bytes kept."""
    import time

    from ouroboros.review_execution import ReviewRouteUnavailable

    if end is not None and end <= time.monotonic():
        raise ReviewRouteUnavailable("this continuation's original review window ended before its recorded start was "
                                     "sent; a POST now could begin new work, so its outcome stays unknown and it is "
                                     "never re-posted", code="continuation_window_ended")


def stopped_review_failure(gateway: Any, run_id: str, detail: Dict[str, Any], summary: Dict[str, Any]) -> Any:
    """The typed failure of a run that did not succeed, with its stop facts and whole partial output.

    ``failure_details`` keeps the engine's own failure plus ``resumable`` and ``outcomeFacts`` (what an
    explicit continuation is offered from); ``review_partial_output`` is the run's complete primary
    output when it is readable: retained evidence, never parsed into a verdict."""
    from ouroboros.gateways.claudexor import run_failure_error
    from ouroboros.observability import redact_projection

    run_state = str(summary.get("state") or "")
    failure = run_failure_error(run_id, run_state, summary.get("failure"))
    failure.review_run_state = run_state
    stop = {key: summary.get(key) or detail.get(key) for key in ("resumable", "outcomeFacts")
            if isinstance(summary.get(key) or detail.get(key), dict)}
    if stop:
        failure.failure_details = {**dict(failure.failure_details or {}), **redact_projection(stop).value}
    try:
        from ouroboros.review_execution import _full_session_text

        failure.review_partial_output = _full_session_text(gateway, run_id, detail)
    except Exception as exc:  # an unreadable output is a named gap beside the typed failure
        failure.review_partial_output_gap = f"{type(exc).__name__}: {exc}"
    return failure
