"""One host service for owner HTTP and authorized native CLI-maintenance tools.

Claudexor owns vendor version selection, installation, custody and job state.
The host may prepare its Node/npm runtime before submission and ask the owner to
restart the engine (Restart Claudexor in Accounts; the header Restart keeps the
engine). Accepted operations and engine refusals pass through; the host never
stores or retries a job.
"""
from __future__ import annotations

import copy
import os
import re
from contextlib import ExitStack

from ouroboros.claudexor_daemon import ensure_owned_gateway, read_owned_gateway
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.gateways.claudexor_maintenance import maintenance_request_id
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
            return gateway.maintenance_harnesses(harnesses, fresh=fresh, check_latest=check_latest)
        except ClaudexorUnavailable as exc:
            _explain_missing_executable(exc, gateway)
            raise


def inspect_operation(operation_id: str, *, ctx=None) -> dict:
    _authority(ctx)
    _operation_id(operation_id)
    with read_owned_gateway() as gateway:
        return gateway.maintenance_operation(operation_id)


def _operation_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("operation_id is required")


def _prepare_update(gateway, request: dict, request_id: str) -> None:
    """Only a serving engine's npm-missing observation calls for preparation.

    A different next-spawn path proves nothing about the live process. In
    particular an already healthy legacy toolchain needs no restart. These
    optional GETs install nothing; unavailable observations leave POST/rejoin
    to the engine. checkLatest also queries the registry when npm is present.
    """
    harness = request.get("harness")
    target = request.get("target")
    if (set(request) != {"harness", "target"} or not isinstance(harness, str) or not harness
            or not isinstance(target, dict) or not isinstance(target.get("kind"), str)
            or target["kind"] not in {"latest", "version", "previous", "baseline"}
            or set(target) != ({"kind", "version"} if target.get("kind") == "version" else {"kind"})
            or (target.get("kind") == "version" and (not isinstance(target.get("version"), str) or not target["version"].strip()))):
        return  # The engine owns request validation; never broaden an invalid selector.
    try:
        inventory = gateway.maintenance_harnesses([harness], fresh=True)
        row = next((item for item in inventory.get("harnesses", []) if item.get("harness") == harness), {})
        if (not row.get("maintainable") or target["kind"] not in row.get("targets", [])
                or row.get("mechanism") != "managed_npm"):
            return
        checked = gateway.maintenance_harnesses([harness], fresh=True, check_latest=True)
    except (ClaudexorUnavailable, OSError):
        # A failed optional read says nothing about POST acceptance, including
        # an earlier same-key POST. Do not turn observation into an update gate.
        return
    row = next((item for item in checked.get("harnesses", []) if item.get("harness") == harness), {})
    if (row.get("availableProblem") or {}).get("code") != "embedded_npm_missing":
        return  # Unknown or other failures are not evidence that an engine restart is needed.
    from ouroboros.review_run_isolation import attach_home
    from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager

    try:
        if attach_home() is not None or os.environ.get("OUROBOROS_CLAUDEXOR_BIN"):
            raise ClaudexorRuntimeError("runtime_override_selected", "the selected external engine controls its own Node/npm")
        node = get_runtime_manager().prepare_node_toolchain()
        code, preparation = "maintenance_restart_required", "ready"
        message = ("Environment preparation complete; CLI update not started. "
                   "Use Restart Claudexor in Accounts, then repeat Update. It ends every run on the engine; "
                   "its confirmation says how many and whose.")
        cause = None
    except ClaudexorRuntimeError as exc:
        node, code, preparation, cause = "", "maintenance_preparation_failed", "failed", exc.code
        message = f"Environment preparation failed; this attempt did not submit a CLI update: {exc}"
    error = ClaudexorUnavailable(code, message, status_code=409)
    error.problem = {"code": code, "message": message, "requiredActions": ["restart_claudexor"] if node else [],
                     "context": {"requestId": request_id, "acceptance": "not_submitted", "scope": "this_attempt",
                                 "preparation": preparation, "restartRequired": bool(node),
                                 "node": node or None, "cause": cause}}
    raise error


def start_maintenance(request: dict, request_id: str, *, ctx=None) -> dict:
    _authority(ctx, mutate=True, network=True)
    maintenance_request_id(request_id)  # Refuse invalid keys before any daemon preparation.
    if not isinstance(request, dict):
        raise ValueError("Maintenance request must be an object")
    request = copy.deepcopy(request)

    def submit():
        with ExitStack() as stack:
            try:
                gateway = stack.enter_context(ensure_owned_gateway())
                _prepare_update(gateway, request, request_id)
            except (ClaudexorUnavailable, OSError) as exc:
                error = exc if isinstance(exc, ClaudexorUnavailable) else ClaudexorUnavailable(
                    "maintenance_preparation_failed", f"Environment preparation failed: {exc}", status_code=503)
                problem = copy.deepcopy(getattr(error, "problem", None)) or {"code": error.code, "message": str(error)}
                problem["context"] = {**problem.get("context", {}), "requestId": request_id,
                                      "acceptance": "not_submitted", "scope": "this_attempt"}
                error.problem = problem
                raise error from (exc if exc is not error else None)
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
