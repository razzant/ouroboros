"""Inspect a configured actor from this task's settings, without starting it."""
from __future__ import annotations

import json

from ouroboros.claudexor_observations import quota_availability, resource_rows, subject_rows
from ouroboros.configured_subagents import roster_handles
from ouroboros.subagent_runtime import (
    SubagentSelectionError, effective_runtime_subagent_settings, offered_subagents, resolve_configured_row,
)


def inspect_subagent(subagent_id: str, *, request_kind: str = "text", drive_root=None) -> dict:
    """Saved row intent and independently dated observations, never selection policy."""
    from ouroboros.config import runtime_settings
    from ouroboros.gateway.claudexor_passive import _quota_payload
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model
    from ouroboros.route_spec import model_named_effort, route_spec_dict
    from ouroboros.utils import utc_now_iso

    if request_kind not in ("text", "image"):
        raise ValueError("request_kind must be text or image")
    settings = effective_runtime_subagent_settings(runtime_settings())
    resolution = offered_subagents(settings)
    if resolution.config is None:
        raise SubagentSelectionError("subagent_catalog_unavailable", "This task has no readable configured catalog")
    config = resolution.config
    row = resolve_configured_row(config, subagent_id, settings)
    route = row.route
    named_effort = model_named_effort(route)
    actor = {
        "subagent_id": roster_handles(config, settings)[row.subagent_id],
        "route": route_spec_dict(route, api_kind="api_model", pin_key="credential_profile_id"),
        "enabled": config.enabled and row.enabled,
        "catalog_enabled": config.enabled, "row_enabled": row.enabled,
        "access": row.access if route.is_session else None,
        "access_source": "configured_session" if route.is_session else "task_write_surface",
        "effort": named_effort or row.effort or "auto", "recommended_use": row.recommended_use,
        **({"effort_source": "model_name"} if named_effort else {}),
    }
    result = {"configured": actor, "inspected_at": utc_now_iso(), "request_kind": request_kind,
              "availability": {"state": "unknown", "reason": "provider_observations_unavailable"},
              "execution_guaranteed": False}
    from ouroboros.config import DATA_DIR
    from ouroboros.subagent_metrics import read
    result["measurements"] = read(drive_root or DATA_DIR, row.subagent_id)
    if route.is_session:
        harness, _, model = route.target_id.partition("=")
    elif provider_for_model(route.target_id) == "claudexor":
        harness, model = parse_claudexor_model(route.target_id)
    else:
        return result
    model = model or None
    profile = route.credential_profile_id
    payload = _quota_payload(model=model, request_kind=request_kind)
    result["engine"] = {key: payload[key] for key in ("reads", "read_errors", "timings_ms")}
    result["engine"]["requested_scope"] = {"model": model, "requestKind": request_kind}
    if "availability_scope" in payload:
        result["engine"]["availability_scope"] = payload["availability_scope"]
    roster = payload.get("profiles", {})
    for source, target in (("quota", "snapshots"), ("quota_absences", "absences")):
        result["engine"][target] = subject_rows(payload.get(source), harness, profile)
    result["engine"]["resources"] = resource_rows(payload.get("resources"), harness, profile)
    result["engine"]["profiles"] = [wrapper for wrapper in roster.get("profiles", [])
        if wrapper.get("profile", {}).get("harness_id") == harness
        and (not profile or wrapper["profile"].get("profile_id") == profile)]
    envelope = {**result["engine"], "availability_scope": payload.get("availability_scope")}
    result["availability"] = quota_availability(
        envelope, harness, model=model, profile=profile, request_kind=request_kind)
    result["availability"]["scope"] = "quota_only"
    return result


def inspect_subagent_tool(ctx, subagent_id: str, request_kind: str = "text") -> str:
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    try:
        body = inspect_subagent(subagent_id, request_kind=request_kind,
                                drive_root=getattr(ctx, "budget_drive_root", None) or ctx.drive_root)
        result = ToolResult(status="ok", code="OK", text=json.dumps(body, ensure_ascii=False))
    except (ValueError, SubagentSelectionError) as exc:
        result = ToolResult(status="error", code="TOOL_ARG_ERROR", text=str(exc))
    return _publish_tool_result(ctx, result)
