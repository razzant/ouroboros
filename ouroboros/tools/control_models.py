"""Cited model choices: local suggestions and the existing work, room and Main writers."""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from ouroboros.composer_cognition import normalize_model_choice, read_model_choice
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.tools.tool_result import ToolResult, publish_no_effect, _publish_tool_result


def refuse(ctx, text: str) -> str:
    return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))


def cyber_pro(ctx) -> bool:
    from ouroboros.config import get_runtime_mode
    from ouroboros.consciousness_authority import effective_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    return mode_has_unrestricted_agency(effective_runtime_mode(get_runtime_mode(), ctx.task_metadata or {}))


def known_models() -> list[dict]:
    """Configured routes and dated host observations; no provider, engine or HTTP call."""
    from ouroboros.config import get_fallback_models
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, parse_configured_subagents
    from ouroboros.llm_capability_policy import recorded_openrouter_catalog
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS
    from ouroboros.provider_connections import configured_connections
    from ouroboros.provider_models import route_on_connection
    from ouroboros.route_choice import following_route
    from ouroboros.route_spec import ROUTE_KIND_API_MODEL
    from ouroboros.settings_integrity import runtime_setting

    routes = [following_route(role) for role in MODEL_ROLE_SETTINGS if role != "fallback"]
    routes.extend((model, runtime_setting("USE_LOCAL_FALLBACK", "").lower() in {"1", "true"})
                  for model in get_fallback_models())
    local = runtime_setting("LOCAL_MODEL_FILENAME", "") or runtime_setting("LOCAL_MODEL_SOURCE", "")
    if local:
        routes.append((local.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1], True))
    rows = {(model, local): {"model": model, "use_local": local, "source": "local configuration"}
            for model, local in routes if model}
    raw = runtime_setting(SUBAGENTS_SETTING, "")
    try:
        roster = parse_configured_subagents(raw) if raw else None
    except ValueError:
        roster = None  # A broken roster cannot hide other known routes or deny an exact id.
    for row in roster.items if roster else ():
        if row.route.kind == ROUTE_KIND_API_MODEL:
            try:
                choice = normalize_model_choice({"kind": "model", "model": row.route.target_id})
            except ValueError:
                continue
            rows.setdefault((choice["model"], choice["use_local"]), {
                "model": choice["model"], "use_local": choice["use_local"],
                "source": "configured review-pool row" if row.review_eligible else "configured subagent row"})
    connections = [name for kind, name in configured_connections() if kind == "openrouter"]
    if connections:
        models, observed = recorded_openrouter_catalog()
        source = "host's cached OpenRouter catalogue (" + (f"observed {observed}" if observed else "date unavailable") + ")"
        for name in connections:
            for model in models:
                if model:
                    route = route_on_connection(model, "openrouter", name)
                    rows.setdefault((route, False), {"model": route, "use_local": False, "source": source})
    return list(rows.values())


def resolve_model(model: str, *, model_as_typed: bool = False) -> tuple[dict, str]:
    """Known ids pass; suggestions ask for a choice or explicit use of the typed id.

    The chooser searches every query word in a row's route/locality text. A dated
    catalogue is a source of suggestions, not an admission or freshness claim.
    """
    model = str(model or "").strip()
    if not isinstance(model_as_typed, bool):
        raise ValueError("model_as_typed must be a boolean.")
    if model.lower() in {"main", "light"}:
        from ouroboros.route_choice import following_route

        routes = [f"{role.title()}: {route}{' (local)' if local else ''}"
                  for role in ("main", "light") for route, local in [following_route(role)]]
        raise ValueError("Main and Light are roles, not model ids. " + "; ".join(routes)
                         + ". Moving this task between their current routes needs no cite: pass the route to switch_model without owner_words.")
    choice = normalize_model_choice({"kind": "model", "model": model})
    try:
        rows = known_models()
    except (OSError, ValueError, RuntimeError):
        rows = []
    exact = next((row for row in rows if row["model"] == choice["model"]), None)
    if exact:
        return {**choice, "use_local": choice["use_local"] or exact["use_local"]}, (
            f"Route found in {exact.get('source', 'local configuration')}; no live catalog check.")
    words = model.lower().split()
    matches = [row for row in rows if all(word in (row["model"] + (" local" if row["use_local"] else "")).lower()
                                                for word in words)]
    if matches:
        matches.sort(key=lambda row: (
            not row["model"].rsplit("::", 1)[-1].rsplit("/", 1)[-1].lower().startswith(tuple(words)),
            row["model"].lower(), row["use_local"]))
        candidates = [row["model"] + (" (local)" if row["use_local"] else "")
                      + f" [{row.get('source', 'local configuration')}]" for row in matches[:12]]
        more = f"\n{len(matches) - 12} more matched; use a more specific name." if len(matches) > 12 else ""
        suggestions = "Matching known routes:\n- " + "\n- ".join(candidates) + more
        if model_as_typed:
            return choice, "Model id not verified: used as typed. " + suggestions + "\nNo live catalog check."
        raise ValueError("Model id is not verified. " + suggestions
                         + "\nPass an exact known route or model_as_typed=true to use this id as typed. No live catalog check.")
    return choice, "Model id not verified: used as typed. Local configuration and available host-cached catalogues only; no live catalog check."


def start_choice(ctx, model: str, basis: dict | None, project_id: str, *, model_as_typed: bool = False) -> tuple[dict, str]:
    if not model:
        return {}, ""
    if not basis and not cyber_pro(ctx):
        raise ValueError("A model requires your human's verified owner_words. Nothing was started; cite their exact words.")
    choice, note = resolve_model(model, model_as_typed=model_as_typed)
    if basis:
        return {"lead": {"source": "owner_words", "model": choice["model"], "use_local": choice["use_local"],
                         "cited": basis["cited"]}}, note
    room = read_model_choice(canonical_data_root(ctx), project_id)
    if room["kind"] == "model":
        if (room["model"], room["use_local"]) != (choice["model"], choice["use_local"]):
            raise ValueError("Your human's model choice for this room binds. Cite their words to change it for this work.")
        return {"lead": {"source": "room", "model": room["model"], "use_local": room["use_local"]}}, note
    return {"lead": {"source": "mind", "model": choice["model"], "use_local": choice["use_local"]}}, note


def request_work_model(ctx, choice: dict) -> tuple[str, bool]:
    """The worker emits; only the supervisor's durable writer can change the primary."""
    from ouroboros.tools.control_events import _emit_control_event, _PROMOTE_CONFIRM_TIMEOUT_SEC, _PROMOTE_CONFIRM_POLL_SEC
    from ouroboros.tool_call_log import current_tool_call_id
    from ouroboros.task_results import load_task_result
    from ouroboros.owner_model import apply_current, eligibility
    from ouroboros.owner_mailbox import drain_owner_entries, KIND_TASK_MESSAGE

    actor = str(ctx.task_id or "")
    stored = load_task_result(canonical_data_root(ctx), actor)
    if not stored or eligibility(stored):
        return refuse(ctx, "This work has no eligible durable root for a primary change."), False
    call_id = current_tool_call_id() or uuid.uuid4().hex
    nonce = hashlib.sha256(json.dumps([actor, "switch_model", call_id]).encode()).hexdigest()
    event = {"type": "mind_task_control", "action": "switch_model", "task_id": actor,
             "actor_task_id": actor, "action_nonce": nonce, "model": choice["model"],
             "use_local": choice["use_local"], "cited": choice["cited"],
             "reason": "My human's cited words chose this work's primary."}
    mode = _emit_control_event(ctx, event)
    if mode == "serialization_failed":
        return refuse(ctx, "Model request was not emitted. The primary has not changed."), False
    deadline = time.monotonic() + (_PROMOTE_CONFIRM_TIMEOUT_SEC if mode == "live" else 0)
    while True:
        entries = drain_owner_entries(canonical_data_root(ctx), actor, include_acknowledged=True,
                                      kinds={KIND_TASK_MESSAGE})
        for entry in entries:
            if entry.get("msg_id") != "mind-control:" + nonce:
                continue
            outcome = json.loads(entry["text"])["outcome"]
            text = json.dumps(outcome, ensure_ascii=False)
            if not outcome.get("ok"):
                # A failed wake or queue publication may follow the durable primary write.
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=text)), False
            apply_current(ctx)
            return text, True
        if time.monotonic() >= deadline:
            return json.dumps({"ok": True, "status": "requested", "action_nonce": nonce,
                               "note": "Primary change requested; not yet confirmed. The supervisor's outcome arrives in your task mailbox."}), True
        time.sleep(_PROMOTE_CONFIRM_POLL_SEC)


def cited_switch(ctx, *, model: str, owner_words: str, scope: str, project_id: str, effort: str = "",
                 model_as_typed: bool = False) -> str:
    """Only an integrating focus can carry the owner's room or Main choice.

    Presence is an integrating focus too: at Full autonomy its confirmed place-owner
    cite may choose either scope. Dispatch enforces the level before this handler.
    Delegated actors keep their own acting model and effort controls.
    """
    from ouroboros.lead_fact import is_integrating_focus
    from ouroboros.owner_words import cited_owner_basis
    from ouroboros.gateway.ui_preferences import record_model_recent
    from ouroboros.gateway.owner_effort import _write_main_route
    from ouroboros.gateway.owner_settings import settings_document_mutation
    from ouroboros.composer_cognition import update_project_model_choice
    from ouroboros.tools.control_routing import _inherited_project_scope
    from ouroboros.deployment_policy import DeploymentPolicyRefusal

    if scope in {"room", "main"} and not is_integrating_focus(ctx):
        return refuse(ctx, "Only an integrating work can set a room's model or Main. Ask the work that delegated you to carry your human's choice.")
    basis, error = cited_owner_basis(ctx, owner_words, call_speech=True)
    if error or not basis:
        hint = (" In Cyber Pro you may make your own room or Main change through its existing writer; this tool carries your human's words."
                if cyber_pro(ctx) and scope in {"room", "main"} else "")
        return refuse(ctx, "This model choice requires verified owner_words. Cite your human's exact words." + hint)
    def finished(text):
        if effort:
            from ouroboros.tools.control_runtime import _effort_switch

            applied, effort_note = _effort_switch(ctx, effort)
            if applied:
                ctx.active_effort_override = applied
            text += "\n" + (f"effort={applied}. " if applied else "") + effort_note
        return text

    root = canonical_data_root(ctx)
    try:
        choice, note = (({"kind": "main"}, "") if scope == "room" and str(model).strip() in {"", "main"}
                        else resolve_model(model, model_as_typed=model_as_typed))
        if scope == "room":
            project_id = project_id or _inherited_project_scope(ctx)
            if not project_id:
                return refuse(ctx, "This work has no room. Name project_id, or use scope='main' for Main itself.")
    except ValueError as exc:
        return refuse(ctx, str(exc))
    if scope == "work":
        text, accepted = request_work_model(ctx, {**choice, "cited": basis["cited"]})
        return finished(text + "\n" + note) if accepted else text
    result = None
    try:
        with settings_document_mutation():
            if scope == "room":
                previous, saved = update_project_model_choice(root, project_id, choice)
                result = {"ok": True, "project_id": project_id, "model_choice": saved, "previous_model_choice": previous}
            else:
                result = _write_main_route(None, choice, origin={"surface": "tool", "actor": "ouroboros", "actor_task_id": ctx.task_id})
            if choice["kind"] == "model":
                record_model_recent(root, choice["model"], choice["use_local"])
    except DeploymentPolicyRefusal:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
            text="Model write refused by deployment policy; nothing was saved."))
    except Exception as exc:
        # Suggestion failure cannot turn a completed write into a no-effect refusal.
        detail = ("Model choice saved; recent suggestion could not be recorded. " + json.dumps(result)
                  if result else "Model write was not confirmed; read the current choice before retrying.")
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                                                    text=f"{detail} ({type(exc).__name__})"))
    return finished(json.dumps(result, ensure_ascii=False) + ("\n" + note if note else ""))
