"""Frozen legacy completion fixture, never registered in the runtime."""
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
PRESENCE_OUTCOMES = ("message", "silent", "tool_delivered", "deferred")

def _finish_presence(ctx: ToolContext, outcome: str, message: str = "", action: str = "finish",
                     rationale: str = "", answer_sha256: str | None = None,
                     pending_review: str | None = None,
                     acceptance_subject: dict | None = None) -> str:
    contract = getattr(ctx, "task_contract", {})
    if not isinstance(contract, dict) or not isinstance(contract.get("capability_ceiling"), dict):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=("ERROR: PRESENCE_COMPLETION_UNAVAILABLE: this is not a host-admitted presence turn.")))
    selected = str(outcome or "").strip()
    if selected not in PRESENCE_OUTCOMES:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: PRESENCE_OUTCOME_INVALID: choose message, silent, tool_delivered, or deferred.")))
    from ouroboros.tools.control_runtime import stage_completion_request
    reply_later = not message and answer_sha256 is None and selected in {"message", "deferred"}
    result = stage_completion_request(ctx, {"action": action, "rationale": rationale, "pending_review": pending_review,
        "acceptance_subject": acceptance_subject,
        **({"answer_sha256": answer_sha256} if answer_sha256 is not None else {"answer": message})},
        source="presence_finish", allow_empty=selected in {"silent", "tool_delivered"}, reply_later=reply_later)
    if not getattr(ctx, "_completion_request", None) or getattr(ctx, "_completion_conflict", False):
        return result
    from ouroboros.presence_continuation import selection_for

    ctx._presence_completion = {
        "outcome": selected,
        "message": str(message or "").strip(),
        # The author's selection revision: its output identity (``presence_continuation``).
        "selection": selection_for(ctx, answer_sha256),
    }
    ctx._presence_completion_accepted = False
    ctx._presence_completion_owner_revision = len(getattr(ctx, "_owner_directives", []) or [])
    return result




def legacy_registry(*args, **kwargs):
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(*args, **kwargs)
    return register_legacy_completion(registry)


def register_legacy_completion(registry):
    from ouroboros.tools.registry import ToolEntry
    registry.register(ToolEntry(name="presence_finish", handler=_finish_presence, safety_policy="skip",
        schema={"name": "presence_finish", "description": "Frozen pre-migration completion fixture.",
                "parameters": {"type": "object", "properties": {"outcome": {"type": "string"},
                    "message": {"type": "string"}, "action": {"type": "string"},
                    "rationale": {"type": "string"}, "answer_sha256": {"type": "string"},
                    "pending_review": {"type": "string"}, "acceptance_subject": {"type": "object"}},
                    "required": ["outcome"]}}))
    return registry


# Frozen callers exercise retained proactive custody without publishing a runtime tool.
import hashlib
import json
_INITIATION_MARGIN_SEC = 10.0
def _initiate_presence(
    ctx: ToolContext,
    binding_id: str,
    message: str,
    dedupe_key: str = "",
) -> str:
    """Start one reviewed presence cycle from owner/background cognition."""

    from ouroboros.loop import _resolve_loop_max_rounds
    from ouroboros.model_wait import dispatch_deadline_remaining_sec
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_bindings import conversation_key, list_presence_bindings
    from ouroboros.presence_continuation import CONTINUATION_VERSION, start_proactive_turn
    from ouroboros.presence_runner import PresenceTurnEvent
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
    # The initiated turn is its own work (#1536): this call waits for its first envelope only
    # within its own operation bound, and the turn never inherits that bound as its lifetime.
    # This call is the turn's version-1 consumer; delivery reporting is negotiated apart.
    remaining = dispatch_deadline_remaining_sec()
    result = start_proactive_turn(
        admission=admission,
        event=PresenceTurnEvent(
            source_event_id=source_event_id,
            provider=endpoint.transport,
            account_id=endpoint.account_id,
            conversation_id=endpoint.conversation_id,
            thread_id=endpoint.thread_id,
            conversation_key=conversation_key(
                endpoint.transport, endpoint.account_id, endpoint.conversation_id, endpoint.thread_id,
            ),
            actor={"id": "ouroboros", "display_name": "Ouroboros", "kind": "proactive_initiation"},
            conversation={"kind": "configured_presence_destination"},
            message={"kind": "proactive_initiation"},
            text=prompt,
            continuation_version=CONTINUATION_VERSION,
        ),
        repo_dir=ctx.repo_dir,
        drive_root=root,
        event_queue=getattr(ctx, "event_queue", None),
        wait_sec=None if remaining is None else max(0.0, remaining - _INITIATION_MARGIN_SEC),
    )
    # One shape for completed, continuing and running: the refs a caller follows are always named.
    return json.dumps(
        {
            "ok": True,
            "status": result.status,
            "outcome": result.outcome,
            "delivered": result.outcome == "tool_delivered",
            "text": result.text,
            "turn_ref": result.task_id,
            "work_ref": result.work_ref,
            "continuation_ref": result.continuation_ref,
            "output_ref": result.output_ref,
            "continuation_version": CONTINUATION_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )




def legacy_tools():
    from ouroboros.tools.presence import get_tools
    from ouroboros.tools.registry import ToolEntry
    return [*get_tools(), ToolEntry(name="initiate_presence", handler=_initiate_presence, safety_policy="skip",
            schema={"name": "initiate_presence", "parameters": {"type": "object", "properties": {}}})]


def focus_response(response):
    """Old lifecycle scripts select the ordinary finish tool for a new focus."""
    import copy
    response = copy.deepcopy(response)
    for tool in response.get("tool_calls", []):
        function = tool["function"]
        if function["name"] == "presence_finish":
            old = json.loads(function["arguments"])
            outcome, message = old.pop("outcome"), old.pop("message", "")
            function.update(name="finish_task", arguments=json.dumps({
                "action": "finish", **old,
                **({"silent": True} if outcome in {"silent", "tool_delivered"} else {"answer": message})}))
    return response
