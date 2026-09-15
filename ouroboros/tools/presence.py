"""Typed completion for host-admitted presence turns."""

from __future__ import annotations

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

import hashlib
import json
from typing import Any, List

from ouroboros.tools.registry import ToolContext, ToolEntry

PRESENCE_OUTCOMES = ("message", "silent", "tool_delivered", "deferred")


def _finish_presence(ctx: ToolContext, outcome: str, message: str = "") -> str:
    contract = getattr(ctx, "task_contract", {})
    if not isinstance(contract, dict) or not isinstance(contract.get("capability_ceiling"), dict):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=("ERROR: PRESENCE_COMPLETION_UNAVAILABLE: this is not a host-admitted presence turn.")))
    selected = str(outcome or "").strip()
    if selected not in PRESENCE_OUTCOMES:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_OUTCOME_INVALID: choose message, silent, tool_delivered, or deferred.")))
    ctx._presence_completion = {
        "outcome": selected,
        "message": str(message or "").strip(),
    }
    return f"PRESENCE_COMPLETION_RECORDED: {selected}. Finish this turn now."


def _configure_presence(ctx: ToolContext, action: str, **params: Any) -> str:
    from ouroboros.presence_bindings import (
        PresenceBinding,
        PresenceEndpoint,
        list_presence_bindings,
        new_presence_binding_id,
        save_presence_binding,
    )
    from ouroboros.tool_access import canonical_data_root

    root = canonical_data_root(ctx)
    selected = str(action or "").strip()
    if selected in {"inspect", "select", "runtime", "workspace"}:
        from dataclasses import asdict, replace

        from ouroboros.presence_capabilities import (
            PresenceArgumentBinding,
            PresenceResourceTarget,
            PresenceScriptTarget,
            PresenceSelection,
            PresenceToolTarget,
            load_presence_state,
            presence_state_fingerprint,
            save_presence_state,
        )
        from ouroboros.presence_profile import parse_presence_profile, presence_request_fingerprint
        from ouroboros.presence_runtime import PresenceRuntimeOverrides
        from ouroboros.skill_loader import find_skill

        behavior_skill = str(params.get("behavior_skill") or "").strip()
        loaded = find_skill(root, behavior_skill)
        profile = parse_presence_profile(loaded.manifest, loaded.skill_dir) if loaded is not None else None
        if loaded is None or profile is None:
            return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text=("ERROR: PRESENCE_PROFILE_NOT_FOUND")))
        state = load_presence_state(root, loaded.name)
        requests = {
            request.request_id: (request, presence_request_fingerprint(request))
            for request in profile.capability_requests
        }
        if selected == "inspect":
            return json.dumps(
                {
                    "ok": True,
                    "behavior_skill": loaded.name,
                    "requests": [
                        {
                            "id": request.request_id,
                            "kind": request.kind,
                            "required": request.required,
                            "purpose": request.purpose,
                            "fingerprint": fingerprint,
                        }
                        for request, fingerprint in requests.values()
                    ],
                    "selections": [asdict(item) for item in state.selections],
                    "workspace_root": state.workspace_root,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        if selected == "workspace":
            from ouroboros.workspace_admission import validate_workspace_root

            requested = params.get("workspace_root")
            if not isinstance(requested, str):
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text="ERROR: PRESENCE_WORKSPACE_REQUIRED: supply workspace_root, or an empty string to clear it."))
            workspace = validate_workspace_root(
                requested, system_repo_dir=ctx.repo_dir, drive_root=root,
            )
            updated = replace(state, workspace_root=str(workspace) if workspace is not None else "")
            save_presence_state(
                root, loaded.name, updated,
                expected_state_fingerprint=presence_state_fingerprint(state),
            )
            return json.dumps({
                "ok": True, "workspace_root": updated.workspace_root,
                "state_fingerprint": presence_state_fingerprint(updated),
            }, sort_keys=True)
        if selected == "runtime":
            reset = bool(params.get("reset_runtime"))
            overrides = PresenceRuntimeOverrides() if reset else PresenceRuntimeOverrides(
                model_slot=(str(params.get("model_slot") or "").strip() or None),
                inline_max_rounds=params.get("inline_max_rounds"),
            )
            updated = replace(state, runtime_overrides=overrides)
            save_presence_state(
                root,
                loaded.name,
                updated,
                expected_state_fingerprint=presence_state_fingerprint(state),
            )
            return json.dumps(
                {
                    "ok": True,
                    "runtime_overrides": asdict(overrides),
                    "state_fingerprint": presence_state_fingerprint(updated),
                },
                sort_keys=True,
            )
        request_id = str(params.get("request_id") or "").strip()
        request_row = requests.get(request_id)
        if request_row is None:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_REQUEST_NOT_FOUND")))
        request, request_fingerprint = request_row
        target_type = str(params.get("target_type") or "").strip()
        if request.kind == "tool" and target_type == "tool":
            target = PresenceToolTarget(
                str(params.get("tool_kind") or "builtin").strip(),
                str(params.get("target_name") or "").strip(),
                str(params.get("provider") or "").strip(),
            )
        elif request.kind == "script" and target_type == "script":
            target = PresenceScriptTarget(
                str(params.get("target_skill") or "").strip(),
                str(params.get("script") or "").strip(),
            )
        elif request.kind == "resource" and target_type == "resource":
            operations = params.get("operations") if isinstance(params.get("operations"), list) else []
            target = PresenceResourceTarget(
                str(params.get("root") or "").strip(),
                tuple(str(item).strip() for item in operations),
                str(params.get("path_prefix") or ".").strip(),
                str(params.get("bucket") or "").strip(),
                str(params.get("target_skill") or "").strip(),
            )
        else:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_TARGET_KIND_MISMATCH")))
        bindings = []
        for raw in params.get("argument_bindings") or []:
            if not isinstance(raw, dict):
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_ARGUMENT_BINDING_INVALID")))
            if str(raw.get("source") or "").strip() == "resource":
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_RESOURCE_ARGUMENT_BINDING_UNSUPPORTED")))
            resource_request_id = str(raw.get("resource_request_id") or "").strip()
            resource_fp = requests.get(resource_request_id, (None, ""))[1] if resource_request_id else ""
            bindings.append(PresenceArgumentBinding(
                tuple(raw.get("argument_path") or ()),
                str(raw.get("source") or "").strip(),
                tuple(raw.get("source_path") or ()),
                raw.get("static_value"),
                resource_fp,
            ))
        replacement = PresenceSelection(request_fingerprint, target, tuple(bindings))
        selections = [item for item in state.selections if item.request_fingerprint != request_fingerprint]
        selections.append(replacement)
        updated = replace(state, selections=tuple(selections))
        save_presence_state(
            root,
            loaded.name,
            updated,
            expected_state_fingerprint=presence_state_fingerprint(state),
        )
        return json.dumps(
            {"ok": True, "request_id": request_id, "state_fingerprint": presence_state_fingerprint(updated)},
            sort_keys=True,
        )
    if selected == "list":
        bindings = [binding.__dict__ for binding in list_presence_bindings(root)]
        for row in bindings:
            row["origin"] = row["origin"].__dict__
            row["destination"] = row["destination"].__dict__
        return json.dumps({"ok": True, "bindings": bindings}, ensure_ascii=False, sort_keys=True)
    if selected == "disable":
        binding_id = str(params.get("binding_id") or "").strip()
        existing = next(
            (item for item in list_presence_bindings(root) if item.binding_id == binding_id),
            None,
        )
        if existing is None:
            return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text=("ERROR: PRESENCE_BINDING_NOT_FOUND")))
        save_presence_binding(root, PresenceBinding(**{**existing.__dict__, "enabled": False}))
        return json.dumps({"ok": True, "binding_id": binding_id, "enabled": False})
    if selected != "create":
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_BINDING_ACTION_INVALID")))
    origin = PresenceEndpoint(
        str(params.get("transport") or "").strip(),
        str(params.get("account_id") or "").strip(),
        str(params.get("conversation_id") or "").strip(),
        str(params.get("thread_id") or "").strip(),
    )
    destination = PresenceEndpoint(
        str(params.get("destination_transport") or origin.transport).strip(),
        str(params.get("destination_account_id") or origin.account_id).strip(),
        str(params.get("destination_conversation_id") or origin.conversation_id).strip(),
        str(params.get("destination_thread_id") or origin.thread_id).strip(),
    )
    binding = PresenceBinding(
        new_presence_binding_id(),
        str(params.get("transport_skill") or "").strip(),
        str(params.get("behavior_skill") or "").strip(),
        origin,
        destination,
    )
    save_presence_binding(root, binding)
    return json.dumps({"ok": True, "binding_id": binding.binding_id}, sort_keys=True)


def _initiate_presence(
    ctx: ToolContext,
    binding_id: str,
    message: str,
    dedupe_key: str = "",
) -> str:
    """Start one reviewed presence cycle from owner/background cognition."""

    from ouroboros.loop import _resolve_loop_max_rounds
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_bindings import list_presence_bindings
    from ouroboros.presence_runner import PresenceTurnEvent, run_presence_turn
    from ouroboros.tool_access import canonical_data_root

    root = canonical_data_root(ctx)
    selected = next(
        (item for item in list_presence_bindings(root) if item.binding_id == str(binding_id or "")),
        None,
    )
    if selected is None or not selected.enabled:
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text=("ERROR: PRESENCE_BINDING_NOT_FOUND")))
    prompt = str(message or "").strip()
    if not prompt:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_INITIATION_MESSAGE_REQUIRED")))
    admission = admit_presence_turn(
        drive_root=root,
        authenticated_transport_skill=selected.transport_skill,
        binding_id=selected.binding_id,
        global_max_rounds=_resolve_loop_max_rounds(),
    )
    endpoint = selected.destination
    stable = "\0".join((
        selected.binding_id,
        str(getattr(ctx, "task_id", "") or "background"),
        str(dedupe_key or "").strip(),
        prompt,
    ))
    source_event_id = "presence-initiate:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]
    result = run_presence_turn(
        admission=admission,
        event=PresenceTurnEvent(
            source_event_id=source_event_id,
            provider=endpoint.transport,
            account_id=endpoint.account_id,
            conversation_id=endpoint.conversation_id,
            thread_id=endpoint.thread_id,
            conversation_key=":".join(filter(None, (
                endpoint.transport, endpoint.account_id, endpoint.conversation_id, endpoint.thread_id,
            ))),
            actor={"id": "ouroboros", "display_name": "Ouroboros", "kind": "proactive_initiation"},
            conversation={"kind": "configured_presence_destination"},
            message={"kind": "proactive_initiation"},
            text=prompt,
        ),
        repo_dir=ctx.repo_dir,
        drive_root=root,
        event_queue=getattr(ctx, "event_queue", None),
    )
    return json.dumps(
        {
            "ok": True,
            "outcome": result.outcome,
            "delivered": result.outcome == "tool_delivered",
            "text": result.text,
            "turn_ref": result.task_id,
            "work_ref": result.work_ref,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _cancel_presence_work(ctx: ToolContext, work_ref: str, reason: str = "") -> str:
    """Cancel only work correlated to this exact presence binding/conversation."""

    from ouroboros.task_results import load_task_result, validate_task_id
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.join_ledger import _cancel_task

    try:
        task_id = validate_task_id(work_ref)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: PRESENCE_WORK_REF_INVALID: {exc}")))
    current_meta = getattr(ctx, "task_metadata", {})
    current = current_meta.get("presence") if isinstance(current_meta, dict) else None
    stored = load_task_result(canonical_data_root(ctx), task_id) or {}
    target_meta = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    target = target_meta.get("presence") if isinstance(target_meta.get("presence"), dict) else None
    if not isinstance(current, dict) or not isinstance(target, dict):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=("ERROR: PRESENCE_WORK_NOT_CORRELATED")))
    current_event = current.get("event") if isinstance(current.get("event"), dict) else {}
    target_event = target.get("event") if isinstance(target.get("event"), dict) else {}
    if (
        str(current.get("binding_id") or "") != str(target.get("binding_id") or "")
        or str(current_event.get("conversation_key") or "")
        != str(target_event.get("conversation_key") or "")
    ):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=("ERROR: PRESENCE_WORK_NOT_CORRELATED")))
    return _cancel_task(ctx, task_id, reason)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="presence_finish",
            schema={
                "name": "presence_finish",
                "description": (
                    "Finish the current external presence turn with a typed delivery outcome. "
                    "Call exactly once after the useful work is done. Choose message to return "
                    "a conversational reply, silent when no reply is appropriate, tool_delivered "
                    "when an allowed tool already delivered the result, or deferred after long "
                    "work was successfully promoted."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "enum": list(PRESENCE_OUTCOMES)},
                        "message": {
                            "type": "string",
                            "description": "Reply text for message, or an immediate acknowledgement for deferred.",
                        },
                    },
                    "required": ["outcome"],
                },
            },
            handler=_finish_presence,
            timeout_sec=10,
        ),
        ToolEntry(
            name="configure_presence",
            schema={
                "name": "configure_presence",
                "description": (
                    "Create, list, or disable an owner-controlled binding from an authenticated "
                    "transport room to one reviewed behavior skill. Use exact provider account, "
                    "conversation and optional thread ids supplied by the owner or transport UI. "
                    "Use workspace to select an existing working folder for new turns; memory stays shared."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "list", "disable", "inspect", "select", "runtime", "workspace"],
                        },
                        "workspace_root": {
                            "type": "string",
                            "description": "Existing external folder for action=workspace; empty clears the selection. Does not change capability grants or shared memory.",
                        },
                        "binding_id": {"type": "string"},
                        "transport_skill": {"type": "string"},
                        "behavior_skill": {"type": "string"},
                        "transport": {"type": "string"},
                        "account_id": {"type": "string"},
                        "conversation_id": {"type": "string"},
                        "thread_id": {"type": "string"},
                        "destination_transport": {"type": "string"},
                        "destination_account_id": {"type": "string"},
                        "destination_conversation_id": {"type": "string"},
                        "destination_thread_id": {"type": "string"},
                        "request_id": {"type": "string"},
                        "target_type": {"type": "string", "enum": ["tool", "script", "resource"]},
                        "tool_kind": {"type": "string", "enum": ["builtin", "extension", "mcp"]},
                        "target_name": {"type": "string"},
                        "provider": {"type": "string"},
                        "target_skill": {"type": "string"},
                        "script": {"type": "string"},
                        "root": {"type": "string"},
                        "operations": {"type": "array", "items": {"type": "string"}},
                        "path_prefix": {"type": "string"},
                        "bucket": {"type": "string"},
                        "argument_bindings": {"type": "array", "items": {"type": "object"}},
                        "model_slot": {"type": "string", "enum": ["main", "light"]},
                        "inline_max_rounds": {"type": "integer", "minimum": 1},
                        "reset_runtime": {"type": "boolean"},
                    },
                    "required": ["action"],
                },
            },
            handler=_configure_presence,
            timeout_sec=15,
        ),
        ToolEntry(
            name="initiate_presence",
            schema={
                "name": "initiate_presence",
                "description": (
                    "Start one proactive reasoning cycle for an owner-created presence binding. "
                    "The reviewed profile and positive capability ceiling are resolved by the host. "
                    "The cycle must use its selected transport send capability and finish with "
                    "tool_delivered for an external message to have been sent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "binding_id": {"type": "string"},
                        "message": {"type": "string"},
                        "dedupe_key": {"type": "string"},
                    },
                    "required": ["binding_id", "message"],
                },
            },
            handler=_initiate_presence,
            timeout_sec=180,
        ),
        ToolEntry(
            name="presence_cancel_work",
            schema={
                "name": "presence_cancel_work",
                "description": (
                    "Request cancellation of long work previously deferred from this exact "
                    "presence binding and conversation. The opaque work_ref is correlation, "
                    "not general task authority."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "work_ref": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["work_ref"],
                },
            },
            handler=_cancel_presence_work,
            timeout_sec=15,
        ),
    ]


__all__ = ["PRESENCE_OUTCOMES", "get_tools"]
