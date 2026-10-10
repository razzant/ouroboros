"""One host service for owner HTTP and authorized native CLI-maintenance tools.

Claudexor alone owns version selection, installation, custody and job state.
The host connects to its selected owned engine, never stores or retries a job,
and retains every received operation/refusal verbatim for the two consumers.
"""
from __future__ import annotations

import copy

from ouroboros.claudexor_daemon import ensure_owned_gateway, read_owned_gateway
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.gateways.claudexor_maintenance import maintenance_request_id


def maintenance_problem(error: Exception) -> tuple[int, dict]:
    """Keep engine ControlProblem detail, including a retained operation to rejoin."""
    if isinstance(error, ClaudexorUnavailable):
        received = getattr(error, "problem", None)
        problem = copy.deepcopy(received) if isinstance(received, dict) else {
            "code": error.code, "message": str(error), "requiredActions": list(error.required_actions)}
        return (error.status_code if 400 <= error.status_code < 600 else 503), {"error": problem}
    if isinstance(error, ValueError):
        return 400, {"error": {"code": "invalid_request", "message": str(error)}}
    return 503, {"error": {"code": "maintenance_unavailable", "message": "CLI maintenance could not be reached"}}


def _authority(ctx, *, mutate=False, network=False):
    if ctx is None:  # The authenticated owner HTTP surface has its existing ingress authority.
        return
    from ouroboros.consciousness_authority import is_observe_origin
    from ouroboros.presence_authority import presence_ceiling_from_context, presence_ceiling_allows_tool
    from ouroboros.tool_access import active_tool_profile, decide_tool_access
    from ouroboros.tools.registry_guards import _resource_allowed

    name = "maintain_harness" if mutate else "inspect_harness"
    ceiling = presence_ceiling_from_context(ctx)
    decision = decide_tool_access(profile=active_tool_profile(ctx), root="runtime_data",
                                  operation="write" if mutate else "read")
    if (not decision.allow or (ceiling is not None and not presence_ceiling_allows_tool(ceiling, name))
            or (mutate and is_observe_origin(getattr(ctx, "task_metadata", None)))
            or (network and not _resource_allowed(ctx, "network"))):
        raise ClaudexorUnavailable("maintenance_authority_refused",
                                  "This task may not request that CLI maintenance action", status_code=403)


def inspect_harnesses(harnesses=(), *, fresh=False, check_latest=False, ctx=None) -> dict:
    _authority(ctx, network=check_latest)
    if (not isinstance(harnesses, (list, tuple)) or any(not isinstance(item, str) or not item for item in harnesses)
            or type(fresh) is not bool or type(check_latest) is not bool):
        raise ValueError("harnesses must be names and fresh/check_latest must be booleans")
    with read_owned_gateway() as gateway:
        return gateway.maintenance_harnesses(harnesses, fresh=fresh, check_latest=check_latest)


def inspect_operation(operation_id: str, *, ctx=None) -> dict:
    _authority(ctx)
    _operation_id(operation_id)
    with read_owned_gateway() as gateway:
        return gateway.maintenance_operation(operation_id)


def _operation_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("operation_id is required")


def start_maintenance(request: dict, request_id: str, *, ctx=None) -> dict:
    _authority(ctx, mutate=True, network=True)
    maintenance_request_id(request_id)  # Refuse invalid keys before any daemon preparation.
    if not isinstance(request, dict):
        raise ValueError("Maintenance request must be an object")
    request = copy.deepcopy(request)

    def submit():
        with ensure_owned_gateway() as gateway:
            return gateway.maintenance_create(request, request_id)

    if ctx is None:
        return submit()
    from ouroboros.owner_pause import run_operation
    return run_operation(ctx, submit)


def cancel_maintenance(operation_id: str, *, ctx=None) -> dict:
    _authority(ctx, mutate=True)
    _operation_id(operation_id)

    def cancel():
        with read_owned_gateway() as gateway:  # Never wake an engine just to cancel.
            return gateway.maintenance_cancel(operation_id)

    if ctx is None:
        return cancel()
    from ouroboros.owner_pause import run_operation
    return run_operation(ctx, cancel)
