"""Task-bound, non-generating primary-route observations at natural round boundaries.

Cadence bounds metadata work independently of retry cooldown. The mind chooses
whether to return; only an accepted real response records a route transition.
"""
from __future__ import annotations

import json
import time

OBSERVATION_INTERVAL_SEC = 120.0
OBSERVATION_TIMEOUT_SEC = 3.0


def observe_primary_route(ctx) -> str:
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    from ouroboros.llm_probe import upstream_transport_reachable
    from ouroboros.loop_transport import task_deadline_epoch, transport_repeat_stop_requested
    from ouroboros.model_slots import route_binding, task_model_binding
    from ouroboros.model_wait import current_model_wait, dispatch_deadline_remaining_sec
    from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso

    owner = ctx.tools._ctx
    primary = getattr(owner, "primary_route", None)
    if (ctx.defer_resource_wait is not None or getattr(owner, "exact_model_route", False)
            or not isinstance(primary, dict) or not primary.get("model")):
        return ""
    waiter = current_model_wait()
    overrides = waiter.overrides if waiter else None
    role, _ = task_model_binding({"task_metadata": getattr(owner, "task_metadata", {})},
                                context_fit_plan=ctx.context_fit_plan, overrides=overrides)
    target = route_binding(primary["model"], bool(primary["use_local"]), primary["role"], overrides=overrides)
    acting = route_binding(ctx.active_model, ctx.active_use_local, role, overrides=overrides)
    if target == acting:
        return ""
    previous = getattr(owner, "_primary_route_observation", None) or {}
    now = time.monotonic()
    if previous.get("binding") == target and now - previous["checked_monotonic"] < OBSERVATION_INTERVAL_SEC:
        return ""

    def stopped():
        return transport_repeat_stop_requested(owner) or bool(waiter and waiter.control_reason())

    bounds = [dispatch_window_remaining_sec(deadline_ts=task_deadline_epoch(ctx.tools), reserve_sec=0),
              dispatch_deadline_remaining_sec(), waiter.execution_window_remaining() if waiter else None]
    timeout = min([OBSERVATION_TIMEOUT_SEC] + [value for value in bounds if value is not None])
    if timeout <= 0 or stopped():
        return ""
    if target[1]:
        from ouroboros.local_model import get_manager
        try:
            local = get_manager().serving_context_evidence() or {}
        except Exception:
            local = {}
        observed = {"kind": "local_serving_metadata", "source": local.get("source")} if local.get("confirmed") else {}
    else:
        observed = upstream_transport_reachable(
            ctx.llm, target[0], timeout=timeout, model_role=primary["role"], account_override=target[2],
            observed_after=time.time(), expected_route=None)
    if stopped():
        return ""
    facts = {key: observed[key] for key in ("kind", "status_code", "source", "provenance",
             "credential_profile_id", "account_fingerprint") if key in observed}
    facts.setdefault("kind", "local_not_observed" if target[1] else "unconfirmed")
    stamp = observed.get("observed_at") or utc_now_iso()
    owner._primary_route_observation = {"binding": target, "checked_monotonic": now,
                                        "observed_at": stamp, "facts": facts}
    if previous.get("binding") == target and previous.get("facts") == facts:
        return ""
    row = {"primary_binding": target, "observed_at": stamp, "facts": facts, "generation_tested": False}
    if isinstance(getattr(owner, "_execution_trace", None), dict):
        owner._execution_trace.setdefault("primary_route_observations", []).append(row)
    return sanitize_tool_result_for_log(
        f"[PRIMARY ROUTE OBSERVATION] {json.dumps(row)}. Earlier observations are historical. "
        "Metadata does not establish generation recovery. Stay on the acting route or choose "
        'switch_model(primary="return") / primary="wait". Return reprojects context and may change cache reuse.')


def record_primary_return_request(owner, model: str, use_local: bool, role: str) -> None:
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.utils import utc_now_iso

    waiter = current_model_wait()
    row = {"requested_binding": route_binding(model, use_local, role, overrides=waiter.overrides if waiter else None),
           "status": "requested", "at": utc_now_iso()}
    owner._primary_return_requested = row
    if isinstance(getattr(owner, "_execution_trace", None), dict):
        owner._execution_trace.setdefault("primary_return_requests", []).append(row)


def record_round_route_result(ctx, role: str, *, accepted: bool) -> None:
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.utils import utc_now_iso

    owner = ctx.tools._ctx
    waiter = current_model_wait()
    binding = route_binding(ctx.active_model, ctx.active_use_local, role, overrides=waiter.overrides if waiter else None)
    observed = dict(ctx.accumulated_usage.get("_model_route") or {})
    request = getattr(owner, "_primary_return_requested", None)
    owner._primary_return_requested = None
    if request:
        # Only the requested route's own accepted reply is a return; a reply that a
        # fallback or wait reroute produced in that round answered on another route.
        status = ("no_accepted_response" if not accepted else "accepted_response"
                  if tuple(binding) == tuple(request["requested_binding"]) else "answered_by_other_route")
        outcome = {**request, "actual_binding": binding, "model_route": observed, "at": utc_now_iso(), "status": status}
        ctx.accumulated_usage.setdefault("primary_return_results", []).append(outcome)
        if isinstance(getattr(owner, "_execution_trace", None), dict):
            owner._execution_trace.setdefault("primary_return_results", []).append(outcome)
    if not accepted:
        return
    previous = getattr(owner, "_accepted_route_binding", None)
    previous_observed = getattr(owner, "_accepted_model_route", None)
    owner._accepted_route_binding = binding
    owner._accepted_model_route = observed
    if previous == binding and previous_observed == observed:
        return
    row = {"from_binding": previous, "to_binding": binding, "status": "accepted_response", "at": utc_now_iso(),
           "model_route": observed}
    ctx.accumulated_usage.setdefault("accepted_routes", []).append(row)
    if isinstance(getattr(owner, "_execution_trace", None), dict):
        owner._execution_trace.setdefault("accepted_routes", []).append(row)
