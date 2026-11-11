"""The work's acting lead, shared by the loop, the census and the routing manifest.

The loop records changes, never call attempts. A requested return keeps the prior
fallback fact until the primary accepts a reply. Readers add a follower's pending
route from the live settings view; they never relabel the call already in flight.
Only integrating foci publish here: delegated children, nannies and reviewers keep
their existing observations.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Mapping

from ouroboros import route_choice
from ouroboros.model_slots import MODEL_ROLE_SETTINGS, route_binding, task_model_binding
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def primary_basis(ctx: Any) -> str:
    """Who chose the primary, independently of the route currently acting."""
    choice = route_choice.opening_choice(ctx)
    if role := route_choice.follows(choice):
        return role
    source = choice.get("source")
    if source in route_choice.OWNER_LEAD_SOURCES:
        return source
    if source == "presence":
        metadata = getattr(ctx, "task_metadata", {}) or {}
        return str(metadata.get("model_layer") or "place")
    return str((getattr(ctx, "primary_route", None) or {}).get("role") or "main")


def _publishes(ctx: Any) -> bool:
    metadata = getattr(ctx, "task_metadata", {}) or {}
    return not (metadata.get("delegation_role") == "subagent" or metadata.get("parent_task_id")
                or getattr(ctx, "parent_task_id", None) or metadata.get("review_role")
                or getattr(ctx, "_nanny_route_dispatched", False))


def lead_fact(ctx: Any, *, model: str | None = None, use_local: bool | None = None,
              role: str | None = None) -> dict:
    """Build one fact from the loop's route, retaining the departure's date and cause.

    Bindings include the account policy: a chain row on the same model but another
    account is still a departure. Role is published for that row, never its account.
    ``_lead_episode`` is one bounded observation, retained with the route checkpoint.
    """
    if not _publishes(ctx):
        return {}
    if held := getattr(ctx, "_lead_return_from", None):
        return copy.deepcopy(held)
    primary = getattr(ctx, "primary_route", None) or {}
    model = str(model if model is not None else getattr(ctx, "active_model", "") or "")
    if not model or not primary.get("model"):
        return {}
    local = bool(getattr(ctx, "active_use_local", False) if use_local is None else use_local)
    source = (getattr(ctx, "route_choice", None) or {}).get("source")
    role = role or (getattr(ctx, "_lead_role", None) if source == "fallback" else None) or task_model_binding({"task_metadata": getattr(ctx, "task_metadata", {})},
                                     context_fit_plan=getattr(ctx, "context_fit_plan", None))[0]
    overrides = getattr(getattr(ctx, "model_wait_context", None), "overrides", None)
    home = route_binding(primary["model"], bool(primary.get("use_local")), primary["role"], overrides=overrides)
    acting = route_binding(model, local, role, overrides=overrides)
    basis = primary_basis(ctx)
    departed = acting != home
    fact = {"model": model, "use_local": local, "basis": basis}
    if source == "fallback" and departed:
        fact.update(basis="fallback", role=role)
    elif source == "switch_model":
        fact["basis"] = "mind"
    if departed:
        fact["primary"] = {"model": home[0], "use_local": home[1], "basis": basis}
    if fact["basis"] in {"fallback", "mind"}:
        identity = [model, local, fact["basis"], role, list(acting)]
        episode = getattr(ctx, "_lead_episode", None) or {}
        if episode.get("identity") != identity:
            episode = {"identity": identity, "since": utc_now_iso(),
                       "cause": ("switch_model" if fact["basis"] == "mind" else str(
                           getattr(ctx, "_lead_cause", "") or "configured fallback"))}
            ctx._lead_episode = episode
        fact.update(since=episode["since"], cause=episode["cause"])
    else:
        ctx._lead_episode = None
    return fact


def publish_lead(ctx: Any, **route: Any) -> None:
    """One text-free frame on each change; a failed enqueue stays owed next round."""
    fact = lead_fact(ctx, **route)
    request_id = getattr(ctx, "_owner_model_applied_id", None)
    if not fact or (fact == getattr(ctx, "_lead_last_sent", None)
                    and request_id == getattr(ctx, "_lead_last_request_id", None)):
        return
    bus = getattr(ctx, "event_queue", None)
    if bus is None:
        return
    try:
        bus.put({"type": "work_lead", "task_id": str(ctx.task_id),
                 "task_attempt": int(getattr(ctx, "task_attempt", 1) or 1), "lead": fact,
                 "lead_request_id": request_id})
    except Exception:
        log.warning("Could not publish the work's lead", exc_info=True)
        return
    ctx._lead_last_sent = copy.deepcopy(fact)
    ctx._lead_last_request_id = request_id  # an applied act retires the host projection's pending route


def adopt_fallback(ctx: Any, is_primary: bool, failure: Mapping, role: str) -> None:
    """Publish after adoption, never while the chain is merely trying a row."""
    ctx.route_choice = route_choice.after_fallback(ctx, is_primary)
    ctx._lead_return_from = None
    ctx._lead_role = role
    ctx._lead_cause = str(failure.get("failure_code") or "configured fallback")
    publish_lead(ctx, role=role)


def requested_return(ctx: Any) -> None:
    previous = getattr(ctx, "_lead_last_sent", None)
    if isinstance(previous, dict) and previous.get("basis") == "fallback":
        ctx._lead_return_from = copy.deepcopy(previous)


def accepted_return(ctx: Any, model: str, use_local: bool, role: str, binding: tuple) -> None:
    primary = getattr(ctx, "primary_route", None) or {}
    overrides = getattr(getattr(ctx, "model_wait_context", None), "overrides", None)
    if (getattr(ctx, "_lead_return_from", None) and primary.get("model")
            and binding == route_binding(primary["model"], bool(primary["use_local"]), primary["role"], overrides=overrides)):
        ctx._lead_return_from = None
        publish_lead(ctx, model=model, use_local=use_local, role=role)


def project_lead(fact: Any = None, task: Mapping | None = None, *, queued: bool = False) -> dict:
    """A display-only copy, plus the live route a follower will use next."""
    task = task or {}
    if queued:
        carrier = route_choice.lead_snapshot((task.get("metadata") or {}).get("lead"))
        if not carrier:
            return {}
        if carrier["source"] == "main":
            model, local = route_choice.following_route("main")
            return {"model": model, "use_local": local, "basis": "main"}
        return {"model": carrier["model"], "use_local": carrier["use_local"], "basis": carrier["source"]}
    if not isinstance(fact, dict) or not fact.get("model"):
        return {}
    result = copy.deepcopy(fact)
    result.pop("pending", None)
    basis = result.get("basis")
    control = task.get("owner_model") or {}
    if control.get("request_id") and control["request_id"] != task.get("_lead_request_id"):
        from ouroboros.owner_model import recorded_primary

        primary = recorded_primary(task)
        model, local = primary["model"], primary["use_local"]
    elif basis in MODEL_ROLE_SETTINGS and basis != "fallback":
        model, local = route_choice.following_route(basis)
    else:
        return result
    if (model, local) != (result["model"], result["use_local"]):
        result["pending"] = {"model": model, "use_local": local}
    return result


def lead_fields(fact: Any = None, task: Mapping | None = None, *, queued: bool = False) -> dict:
    lead = project_lead(fact, task, queued=queued)
    return {"lead": lead} if lead else {}


def chooser_words(basis: str) -> str:
    return {"room": "set by my human for this chat", "owner": "set by my human for this work",
            "owner_words": "set by my human's words", "place": "set for this place",
            "circle": "inherited from this circle", "mind": "chosen by me",
            "fallback": "fallback from Settings"}.get(basis, "")


def model_fact_words(ctx: Any, model: str) -> str:
    if ctx is None or not _publishes(ctx):
        return f"model {model}"
    source = (getattr(ctx, "route_choice", None) or {}).get("source")
    basis = {"switch_model": "mind", "fallback": "fallback"}.get(source, primary_basis(ctx))
    words = chooser_words(basis)
    return f"model {model}" + (f" — {words}" if words else "")
