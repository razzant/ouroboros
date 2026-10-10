"""Recovery and pre-dispatch custody helpers for delegated review sessions."""

from __future__ import annotations

from typing import Any, Callable, Dict


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
        or str(run_request.get("mode") or "") != "ask"
        or str(run_request.get("access") or "") != "readonly"
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
    """Bind a pending invocation to its durable slot before provider POST."""
    if not callable(checkpoint):
        return
    try:
        checkpoint(invocation_id)
    except Exception as exc:
        from ouroboros.review_execution import ReviewRouteUnavailable
        from ouroboros.gateways.claudexor import run_failure_cause

        cause = str(getattr(exc, "reported_cause", "") or f"{type(exc).__name__}: {exc}")
        try:
            on_failure()
        except Exception as cleanup_error:
            cause += f"; cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        state.pop("pending_invocation_id", None)
        refusal = ReviewRouteUnavailable(
            "the delegated review invocation could not be bound to its durable "
            "commit-review slot before dispatch; no run was started",
            code="review_custody_checkpoint_unwritable",
        )
        refusal.reported_cause = run_failure_cause({"safeMessage": cause})
        raise refusal from exc
