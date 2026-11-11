"""One host service for owner HTTP and authorized native CLI-maintenance tools.

Claudexor owns vendor version selection, installation, custody and job state.
Accepted operations and engine refusals pass through; the host never stores or
retries a job.
"""
from __future__ import annotations

import copy
import os
import re
import time
import uuid
from contextlib import ExitStack

from ouroboros.claudexor_daemon import ensure_owned_gateway, read_owned_gateway
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.gateways.claudexor_maintenance import maintenance_request_id, maintenance_remaining
from ouroboros.harness_program_facts import observe_inventory, operation_observed
from ouroboros.utils import utc_now_iso

_REPORTED_SPAWN_MISSING = re.compile(r"spawn (.+) ENOENT")
_REPORTED_EXECUTABLE_MISSING = "reported_executable_missing"


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


def _explain_missing_executable(error: ClaudexorUnavailable, gateway) -> None:
    """Add what this host sees now to a current inventory failure; the engine's words stay.

    Only for inventory GETs: a retained operation's problem is history, and its path
    stays missing after a healthy replacement, so it must never source this advice.
    "Reported" is deliberate: neither a path nor a version proves whose executable it
    is or why it went. Restarting the engine is the owner's choice, so requiredActions stay as received.
    """
    from ouroboros.review_run_isolation import attach_home

    received = getattr(error, "problem", None)
    if not isinstance(received, dict) or received.get("code") != "ENOENT":
        return
    match = _REPORTED_SPAWN_MISSING.fullmatch(str(received.get("message") or ""))
    if match is None or not os.path.isabs(match.group(1)):
        return
    if attach_home() is not None or os.environ.get("OUROBOROS_CLAUDEXOR_BIN"):
        return  # A selected external engine is not this host's to restart.
    try:
        os.stat(match.group(1))
        return
    except FileNotFoundError:  # Including a symlink whose target went.
        pass
    except OSError:
        return
    context = received.get("context")
    error.problem = {**copy.deepcopy(received), "context": {
        **(copy.deepcopy(context) if isinstance(context, dict) else {}),
        "observation": _REPORTED_EXECUTABLE_MISSING, "observedAt": utc_now_iso(),
        "engineVersion": getattr(gateway, "engine_version", "") or None,
        "engineBuildSha": getattr(gateway, "engine_build_sha", "") or None,
        "remedy": ("The executable reported by this inspection is no longer present. If the app was moved "
                   "or replaced, Restart Claudexor in Accounts may help; it ends every run on the engine, "
                   "and success is not guaranteed."),
    }}


def inspect_harnesses(harnesses=(), *, fresh=False, check_latest=False, ctx=None) -> dict:
    _authority(ctx, network=check_latest)
    if (not isinstance(harnesses, (list, tuple)) or any(not isinstance(item, str) or not item for item in harnesses)
            or type(fresh) is not bool or type(check_latest) is not bool):
        raise ValueError("harnesses must be names and fresh/check_latest must be booleans")
    with read_owned_gateway() as gateway:
        try:
            return observe_inventory(gateway.maintenance_harnesses(harnesses, fresh=fresh, check_latest=check_latest),
                                     complete=not harnesses)
        except ClaudexorUnavailable as exc:
            _explain_missing_executable(exc, gateway)
            raise


def inspect_operation(operation_id: str, *, ctx=None, timeout_sec=None, deadline=None) -> dict:
    _authority(ctx)
    _operation_id(operation_id)
    if deadline is None and timeout_sec is not None:
        deadline = time.monotonic() + timeout_sec
    options = {"timeout_sec": maintenance_remaining(deadline)} if deadline is not None else {}
    with read_owned_gateway(**options) as gateway:
        return operation_observed(gateway, gateway.maintenance_operation(
            operation_id, **({"deadline": deadline} if deadline is not None else {})))


def _operation_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("operation_id is required")


def start_maintenance(request: dict, request_id: str, *, ctx=None) -> dict:
    _authority(ctx, mutate=True, network=True)
    maintenance_request_id(request_id)  # Refuse invalid keys before acquiring the engine.
    if not isinstance(request, dict):
        raise ValueError("Maintenance request must be an object")
    request = copy.deepcopy(request)

    def submit():
        with ExitStack() as stack:
            try:
                gateway = stack.enter_context(ensure_owned_gateway())
            except (ClaudexorUnavailable, OSError) as exc:
                error = exc if isinstance(exc, ClaudexorUnavailable) else ClaudexorUnavailable(
                    "engine_unavailable", f"Could not acquire the engine: {exc}", status_code=503)
                problem = copy.deepcopy(getattr(error, "problem", None)) or {"code": error.code, "message": str(error)}
                problem["context"] = {**problem.get("context", {}), "requestId": request_id,
                                      "acceptance": "not_submitted", "scope": "this_attempt"}
                error.problem = problem
                raise error from (exc if exc is not error else None)
            return operation_observed(gateway, gateway.maintenance_create(request, request_id))

    if ctx is None:
        return submit()
    from ouroboros.owner_pause import run_operation
    return run_operation(ctx, submit)


def cancel_maintenance(operation_id: str, *, ctx=None, timeout_sec=None) -> dict:
    _authority(ctx, mutate=True)
    _operation_id(operation_id)

    def cancel():
        options = {"timeout_sec": timeout_sec} if timeout_sec is not None else {}
        with read_owned_gateway(**options) as gateway:  # Never wake an engine just to cancel.
            return operation_observed(gateway, gateway.maintenance_cancel(operation_id, **options))

    if ctx is None:
        return cancel()
    from ouroboros.owner_pause import run_operation
    return run_operation(ctx, cancel)


def install_for_connect(harness: str) -> dict:
    """One explicit Connect install, bounded by the existing installer wait ceiling.

    Repeat unanswered reads and retryable refusals; another Connect joins an active install.
    The engine keeps custody after this observer's deadline; expiry cancels nothing.
    """
    from ouroboros.config import get_claudexor_harness_install_timeout_sec

    request_id = str(uuid.uuid4())
    try:
        operation = start_maintenance({"harness": harness, "target": {"kind": "latest"}}, request_id)
    except ClaudexorUnavailable as exc:
        identity = ((getattr(exc, "problem", None) or {}).get("context") or {}).get("operationId")
        if exc.code != "maintenance_already_active" or not isinstance(identity, str) or not identity:
            raise
        operation, request_id = None, None
    else:
        identity = operation["id"]
    deadline = time.monotonic() + get_claudexor_harness_install_timeout_sec()
    last_failure = None
    while operation is None or operation["state"] in {"queued", "running"} or operation["termination"] == "unconfirmed":
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            known_running = last_failure is None and operation is not None and operation["state"] in {"queued", "running"}
            error = _connect_install_problem(operation, request_id, "harness_install_timeout",
                                             "The program is still installing" if known_running else "Installation state is unknown",
                                             identity=identity)
            if last_failure is not None:
                error.problem["context"]["lastReadProblem"] = copy.deepcopy(getattr(last_failure, "problem", None)) or {
                    "code": last_failure.code, "message": str(last_failure)}
            raise error
        time.sleep(min(1, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        try:
            operation = inspect_operation(identity, deadline=min(deadline, time.monotonic() + 5))
            request_id = (operation.get("requester") or {}).get("requestId") or request_id
            last_failure = None
        except ClaudexorUnavailable as exc:
            last_failure = exc
            if exc.status_code >= 400 and (getattr(exc, "problem", None) or {}).get("retryable") is not True:
                exc.problem = copy.deepcopy(getattr(exc, "problem", None)) or {"code": exc.code, "message": str(exc)}
                exc.problem["context"] = {**exc.problem.get("context", {}), **_connect_context(identity, operation, request_id)}
                raise
    selected = operation.get("after") or {}
    if (operation["harness"] != harness or operation["state"] != "succeeded"
            or not all(isinstance(selected.get(field), str) and selected[field].strip()
                       for field in ("path", "version", "versionOutput"))):
        problem = operation.get("problem") or {}
        raise _connect_install_problem(operation, request_id, problem.get("code") or "harness_install_unverified",
                                       problem.get("message") or "The selected program did not answer its version", problem=problem)
    return operation


def _connect_context(identity, operation, request_id):
    return {key: copy.deepcopy(value) for key, value in {
        "operationId": identity, "requestId": request_id, "operation": operation}.items() if value is not None}


def _connect_install_problem(operation, request_id, code, message, *, identity=None, problem=None):
    problem = copy.deepcopy(problem or {})
    error = ClaudexorUnavailable(code, message if problem else f"{message}. See this program in Settings, Accounts",
                                status_code=503, required_actions=tuple(problem.get("requiredActions") or ()))
    error.problem = {**problem, "code": code, "message": str(error), "context": {
        **problem.get("context", {}), **_connect_context(identity or operation["id"], operation, request_id)}}
    return error
