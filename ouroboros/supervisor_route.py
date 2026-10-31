"""Frozen cognition for a configured session's supervising Ouroboros child.

The leaf owns ``configured_subagent``; its supervisor owns schema 2 of the
existing ``parent_cognitive_route``. Neither dispatch nor restore reselects a
roster row. An empty account is captured Auto intent, never an observed account.
``follows_role`` names the role a following parent handed down (the nanny
follows it live, ``route_choice``); a frozen model is the parent's own choice.
"""

from __future__ import annotations

import copy
from typing import Any

_INHERIT = object()


def cognitive_parent_context(ctx: Any = None, task: Any = None) -> Any:
    """A live conversation's supervisor follows the room's ordinary cognitive role.

    The transport model is the speaking parent's route, not a cognition choice.
    Use the same role resolver as an ordinary turn; explicit supervisor rows are
    selected separately. Text turns retain their actual send binding unchanged.
    """
    from types import SimpleNamespace
    from ouroboros.route_choice import following, following_route, opening_role

    task = task if isinstance(task, dict) else {}
    metadata = (task.get("task_metadata") or task.get("metadata")
                or getattr(ctx, "task_metadata", None) or {})
    if (metadata.get("execution_transport") or {}).get("kind") != "live_conversation":
        return ctx
    parent = SimpleNamespace(task_metadata=metadata)
    role = opening_role(parent)
    model, local = following_route(role)
    return SimpleNamespace(task_metadata={key: value for key, value in metadata.items()
                                         if key not in {"model", "use_local_model", "execution_transport"}},
        active_model=model, active_use_local=local,
        active_effort=getattr(ctx, "active_effort", "") or metadata.get("reasoning_effort", ""),
        context_fit_plan=SimpleNamespace(model_role=role), route_choice=following(role),
        model_wait_context=getattr(ctx, "model_wait_context", None))  # the owner's wait-card replacement carries over


def validate_cognitive_route(raw: Any) -> dict:
    """Copy historical bindings unchanged; validate the complete v2 carrier."""
    from ouroboros.model_slots import normalize_processing_preference
    from ouroboros.subagent_runtime import SubagentSelectionError, validate_subagent_snapshot

    route = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    if route.get("schema", 1) == 1:
        return route
    required_strings = ("model", "effort", "source_role", "credential_profile_id", "processing_preference")
    try:
        if (type(route.get("schema")) is not int or route["schema"] != 2
                or any(not isinstance(route.get(key), str) for key in required_strings)
                or type(route.get("use_local_model")) is not bool
                or not route["source_role"]
                or route.get("chosen_by") not in ("inherited", "parent_fallback", "parent_switch", "delegator")
                or not isinstance(route.get("follows_role", ""), str)):
            raise ValueError("The complete schema-2 cognitive binding is required")
        route["processing_preference"] = normalize_processing_preference(route["processing_preference"])
        if route["chosen_by"] == "delegator":
            selected = validate_subagent_snapshot(route.get("selected_supervisor_snapshot"))
            if selected["route"]["kind"] != "api_model":
                raise ValueError("A supervisor must be an API-model row")
            route["selected_supervisor_snapshot"] = selected
        elif "selected_supervisor_snapshot" in route:
            raise ValueError("An inherited route cannot claim a selected supervisor")
    except (ValueError, TypeError) as exc:
        raise SubagentSelectionError("invalid_parent_cognitive_route", str(exc)) from exc
    return route


def supervisor_binding(task: dict) -> tuple[dict, str]:
    """The new supervisor's frozen intent and own billing role, or legacy silence."""
    metadata = task.get("task_metadata", task.get("metadata", {}))
    metadata = metadata if isinstance(metadata, dict) else {}
    actor = task.get("configured_subagent", metadata.get("configured_subagent")) or {}
    raw = task.get("parent_cognitive_route", metadata.get("parent_cognitive_route")) or {}
    if (not isinstance(actor, dict) or (actor.get("route") or {}).get("kind") != "agent_session"
            or not isinstance(raw, dict) or raw.get("schema", 1) == 1):
        return {}, ""
    route = validate_cognitive_route(raw)
    selected = route.get("selected_supervisor_snapshot") or actor
    return route, f"subagent:{selected['selected_subagent_id']}"


def capture_cognitive_route(ctx: Any, actor: dict, settings: dict, *, supervisor: Any = _INHERIT) -> dict:
    """Resolve an explicit row or freeze the parent's actual send binding once."""
    from ouroboros.subagent_runtime import SubagentSelectionError, select_subagent_snapshot

    if supervisor is not _INHERIT:
        if not isinstance(supervisor, str) or not supervisor.strip():
            raise SubagentSelectionError("supervisor_selection_required", "supervisor must name an enabled API-model row.")
        if actor["route"]["kind"] != "agent_session":
            raise SubagentSelectionError("supervisor_inapplicable", "Only an Agent-session child has a separate supervisor.")
        try:
            selected, _ = select_subagent_snapshot(settings, subagent_id=supervisor)
        except SubagentSelectionError as exc:
            raise SubagentSelectionError(exc.code.replace("subagent", "supervisor"), exc.detail) from exc
        return _selected_cognition(ctx, selected, settings)
    return _inherited_cognition(ctx, settings)


def _selected_cognition(ctx: Any, selected: dict, settings: dict) -> dict:
    from ouroboros.provider_models import model_has_credentials_in_settings
    from ouroboros.route_spec import RouteSpec, model_named_effort
    from ouroboros.runtime_mode_policy import effort_range_binds
    from ouroboros.settings_scales import choose_effort, effort_range
    from ouroboros.subagent_runtime import SubagentSelectionError

    route = selected["route"]
    if route["kind"] != "api_model":
        raise SubagentSelectionError("supervisor_requires_api_model", "An Agent-session row cannot supply supervisor cognition; select an API-model row.")
    target = route["target_id"]
    if not model_has_credentials_in_settings(target, settings):
        raise SubagentSelectionError("supervisor_credentials_unavailable", "The selected supervisor's exact provider route has no configured credentials.")
    effort, _ = choose_effort(
        "", pin=selected.get("effort"), model_named=model_named_effort(RouteSpec("api_model", target)),
        binds=effort_range_binds(getattr(ctx, "task_metadata", None)), rng=effort_range())
    local = target.endswith(" (local)")
    return validate_cognitive_route({
        "schema": 2, "model": target[:-8].strip() if local else target,
        "effort": effort, "use_local_model": local,
        "source_role": f"subagent:{selected['selected_subagent_id']}",
        "credential_profile_id": str(route.get("credential_profile_id") or ""),
        "processing_preference": selected["processing_preference"],
        "chosen_by": "delegator", "selected_supervisor_snapshot": selected,
    })


def _inherited_cognition(ctx: Any, settings: dict) -> dict:
    """The parent's actual send binding, with its route choice: a parent following its role hands the role
    down (``follows_role``, the nanny follows it live); a parent on its own route (``switch_model``, a
    wait-card choice) hands a frozen ``parent_switch``; an adopted fallback stays ``parent_fallback``."""
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, task_model_binding, task_processing_preference
    from ouroboros.provider_models import provider_for_model
    from ouroboros.route_choice import follows

    ctx = cognitive_parent_context(ctx)
    metadata = getattr(ctx, "task_metadata", None) or {}
    plan = getattr(ctx, "context_fit_plan", None)
    waiter = getattr(ctx, "model_wait_context", None)
    overrides = getattr(waiter, "overrides", None) or {}
    task = {"task_metadata": metadata}
    role, pin = task_model_binding(task, context_fit_plan=plan, overrides=overrides)
    live = overrides.get(role) or {}
    model = str(live.get("model", getattr(ctx, "active_model", "") or metadata.get("model") or ""))
    local = bool(live.get("use_local", getattr(ctx, "active_use_local", metadata.get("use_local_model", False))))
    if pin is None:
        pin = model_role_option(MODEL_ACCOUNTS_KEY, role, settings=settings)
    if local or provider_for_model(model) != "claudexor":
        pin = ""
    choice = getattr(ctx, "route_choice", None)
    followed = follows(choice) if isinstance(choice, dict) else ""
    chosen_by = ("parent_fallback" if role.startswith("fallback:")
                 else "parent_switch" if isinstance(choice, dict) and not followed else "inherited")
    return validate_cognitive_route({
        "schema": 2, "model": model,
        "effort": str(getattr(ctx, "active_effort", "") or metadata.get("reasoning_effort") or ""),
        "use_local_model": local, "source_role": role,
        "credential_profile_id": str(pin or ""),
        "processing_preference": task_processing_preference(task, model_role=role),
        "chosen_by": chosen_by, "follows_role": followed if chosen_by == "inherited" else "",
    })


def cognitive_route_provenance(route: Any) -> str:
    """Only recorded choice facts; model equality never implies a fallback."""
    route = route if isinstance(route, dict) else {}
    chosen = route.get("chosen_by")
    if chosen == "delegator":
        selected = route.get("selected_supervisor_snapshot") or {}
        row = selected.get("selected_subagent_id")
        return f"chosen by the delegator ({row})" if row else "chosen by the delegator"
    if chosen == "parent_fallback":
        return "parent's fallback"
    if chosen == "parent_switch":
        return "parent's own choice"
    if chosen == "inherited":
        followed = str(route.get("follows_role") or "")
        return f"inherited (follows {followed})" if followed else "inherited"
    return ""
