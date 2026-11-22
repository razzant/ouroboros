"""Materialize the permitted completion schema when finalization needs the front door.

An older schedule's runs keep the frozen Presence ceiling and speaker carrier their template copied by value:
``finish_task`` is outside that ceiling and new registries publish no ``presence_finish``, so a held report could
never be selected (#3687). Exactly such a run, confirmed by its own canonical admission receipt, gets the old name
back as a narrow adapter of the ordinary completion request: the report its author selects becomes the run's quiet
record. It grants no other tool, sends nothing and leaves every other task, chat or Presence turn as it was.
"""
from __future__ import annotations
import json
from typing import Any

SCHEDULE_DOOR = "presence_finish"
_SELECTION_ARGUMENTS = frozenset({"message", "answer_sha256"})  # what the author selects; the rest of the call is the host's
# Ordinary refusals a repeated call cannot pass (an explicit prohibition, a Safety return, a withheld or fixed
# argument), each with the host's own short words: the only explanation a refused run's notice carries. A Safety
# outage, an owner control, a busy store and a correctable argument are not final.
_FINAL_REFUSALS = {
    "RESOURCE_CONSTRAINT_BLOCKED": "this run's constraints do not allow its completion",
    "SAFETY_RETURNED": "the Safety check returned its completion",
    "ACCESS_BLOCKED": "this run's access does not allow its completion",
    "LEGACY_BLOCKED": "its confirmation attempt was blocked",
    "WORKSPACE_BLOCKED": "its workspace does not allow its completion",
}
_SCHEMA = {"name": SCHEDULE_DOOR, "description": (
    "Finish or stop this scheduled run, the completion its older schedule keeps. Select its complete report: message "
    "with the exact bytes, or answer_sha256 of the retained or whole held answer the host offered. The report stays "
    "this run's record in its task result: nothing is posted or sent, whatever outcome says, and an earlier confirmed "
    "send is never repeated. action=stop with a rationale records unfinished work. Review, Safety and your human's "
    "controls apply as to any finish."), "parameters": {"type": "object", "properties": {
        "outcome": {"type": "string", "enum": ["message", "silent", "tool_delivered", "deferred"],
                    "description": "The older call's word; every outcome is recorded the same way."},
        "message": {"type": "string", "description": "The complete report."},
        "answer_sha256": {"type": "string", "description": "Exact offered retained or whole held answer hash."},
        "action": {"type": "string", "enum": ["finish", "stop"], "default": "finish"},
        "rationale": {"type": "string", "description": "For stop, what remains unfinished and why you end now; for "
                                                       "a finish with pending_review=finish, why you accept it now."},
        "pending_review": {"type": "string", "enum": ["wait", "finish"]},
        "acceptance_subject": {"type": "object", "properties": {"owner_source_sha256": {"type": "string"},
            "effective_criteria": {"type": "string"}, "material_tool_indices": {"type": "array", "items": {"type": "integer"}},
        }, "required": ["owner_source_sha256"]}}}}


def completion_schema(tools: Any, schemas: list | None = None) -> str:
    """Materialize the current permitted front door, including cold/Nano holds."""
    from ouroboros.usage_accounting import invalidate_task_cache_splits
    ctx = tools._ctx
    name = "finish_task"
    getter = getattr(tools, "get_schema_by_name", None)
    if not callable(getter):
        return name
    # An already-running pre-upgrade registry can still own its frozen completion
    # tool, and a confirmed run of an older schedule its adapter; new focuses have no ceiling.
    if (getattr(ctx, "task_contract", {}) or {}).get("capability_ceiling") and getter(SCHEDULE_DOOR) is None:
        entry = schedule_door_entry(ctx)  # a busy first read is not a lifetime unbind: the next entry reads again
        if entry is not None and SCHEDULE_DOOR not in getattr(tools, "_entries", {}):
            tools.register(entry, origin="ouroboros.loop_completion_schema.schedule_door_entry")
    if (getattr(ctx, "task_contract", {}) or {}).get("capability_ceiling") and getter(SCHEDULE_DOOR) is not None:
        name = SCHEDULE_DOOR
    schema = getter(name)
    if schema is None:
        return "no currently permitted completion tool"
    if schemas is not None and not any(row.get("function", {}).get("name") == name for row in schemas):
        schemas.append(schema)
        invalidate_task_cache_splits(getattr(ctx, "task_id", ""))
        getattr(ctx, "messages", []).append({"role": "user", "content":
            "[SYSTEM NOTICE] Current permitted completion schema loaded: " + name + ". The schema prefix changed."})
    return name


def _occurrence(ctx: Any) -> dict | None:
    """The work, schedule and occurrence a Presence-carrying run names for itself, or ``None``; reads nothing."""
    metadata = getattr(ctx, "task_metadata", None)
    occurrence = metadata.get("schedule_occurrence") if isinstance(metadata, dict) else None
    if not isinstance(occurrence, dict) or not isinstance(metadata.get("presence"), dict):
        return None
    door = {"task_id": str(getattr(ctx, "task_id", "") or ""), "schedule_id": str(occurrence.get("schedule_id") or ""),
            "token": str(occurrence.get("token") or "")}
    return door if all(value.strip() for value in door.values()) else None


def schedule_door(ctx: Any) -> dict | None:
    """The run this door serves, confirmed by one canonical receipt, else ``None``.

    A root (or its hard-timeout retry: ``resolve_task_lineage``) whose frozen Presence ceiling still decodes and whose
    root record holds the accepted admission receipt of exactly this occurrence (``schedule_admission``: its frozen
    task that root, its schedule and token those of ``metadata.schedule_occurrence``). The receipt is read with the
    store's ordinary waiting read; an unreadable or busy read that never completes is unknown, never a confirmation,
    and the run's next binding reads again.
    """
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.schedule_continuation import admission_basis
    from ouroboros.task_results import load_task_result, resolve_ctx_lineage
    from ouroboros.tool_access import canonical_data_root

    door = _occurrence(ctx)
    lineage = resolve_ctx_lineage(ctx) if door else {}
    if door is None or not lineage["is_root_task"]:
        return None
    root_id = lineage["root_task_id"]
    try:
        if presence_ceiling_from_context(ctx) is None:
            return None
        result = load_task_result(canonical_data_root(ctx), root_id, strict=True)
    except Exception:
        return None
    result = result if isinstance(result, dict) else {}
    if lineage.get("is_retry_root_attempt") and not _retry_chain_reaches(
            ctx, root_id, str(getattr(ctx, "task_id", "") or ""), lineage["timeout_retry_from"]):
        return None  # metadata alone is not admission: the stored chain must reciprocate back to the root
    receipt = result.get("schedule_admission") if isinstance(result.get("schedule_admission"), dict) else {}
    basis = admission_basis(result, root_id)
    if (receipt.get("status") != "accepted" or basis is None or basis["schedule_id"] != door["schedule_id"]
            or basis["occurrence_token"] != door["token"] or str(result.get("root_task_id") or "") != root_id):
        return None
    return {**door, "root_task_id": root_id}


def _retry_chain_reaches(ctx: Any, root_id: str, attempt_id: str, claimed: str) -> bool:
    """Walk stored timeout-retry rows back from this attempt to its root, as the queue's lineage walk does.

    Each attempt names its IMMEDIATE predecessor in all three links, that predecessor names this exact attempt in both
    ``superseded_by`` and ``retry_task_id``, and every intermediate predecessor carries the same logical root; the
    attempt's stored predecessor is the one its own lineage (``claimed``) names. An unreadable row, a one-sided link,
    a cycle or an overlong chain is unconfirmed, never admission."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tool_access import canonical_data_root

    data_root, current_id, seen = canonical_data_root(ctx), attempt_id, {attempt_id}
    try:
        for _ in range(64):
            current = load_task_result(data_root, current_id, strict=True) or {}
            predecessor_id = str(current.get("timeout_retry_from") or "")
            if (not predecessor_id or predecessor_id in seen or (current_id == attempt_id and predecessor_id != claimed)
                    or any(str(current.get(key) or "") != predecessor_id
                           for key in ("supersedes_task_id", "original_task_id"))):
                return False
            predecessor = load_task_result(data_root, predecessor_id, strict=True) or {}
            if (str(predecessor.get("superseded_by") or "") != current_id
                    or str(predecessor.get("retry_task_id") or "") != current_id):
                return False  # the reciprocal pair, never one of its two halves
            if predecessor_id == root_id:
                return True
            if str(predecessor.get("root_task_id") or "") != root_id:
                return False
            seen.add(predecessor_id)
            current_id = predecessor_id
    except Exception:
        return False
    return False


def schedule_door_entry(ctx: Any) -> Any:
    """The adapter's entry for a confirmed run, which ``ctx`` records beside the run it serves; else ``None``.
    Any other context is left as it was, a door it held before excepted (cleared)."""
    from ouroboros.tools.registry import ToolEntry

    door = schedule_door(ctx)
    if door is not None or getattr(ctx, "_schedule_door", None) is not None:
        ctx._schedule_door = door
    return None if door is None else ToolEntry("presence_finish", _SCHEMA, schedule_finish, safety_policy="check")


def legacy_selection_error(ctx: Any, request: dict, text: str | None, calls: list) -> str:
    """Prove that an executed legacy door selected these exact bytes, including after a cold restore."""
    from ouroboros.task_finalization import schedule_report_ctx

    if not schedule_report_ctx(ctx) or request.get("source") != SCHEDULE_DOOR:
        return ""
    if fixed_arguments(ctx):
        return "SCHEDULE_REPORT_FIXED_ARGUMENT: a frozen binding cannot restore the author's selection"
    receipt = next((call for call in reversed(calls) if call.get("tool") == SCHEDULE_DOOR and not call.get("is_error")), None)
    if receipt is None:
        return "selection_unproven: no successful presence_finish receipt for the restored selection"
    args = (receipt or {}).get("args") or {}
    for row in reversed(getattr(ctx, "messages", [])):
        for call in row.get("tool_calls") or []:
            if call.get("id") == receipt.get("tool_call_id"):
                raw = (call.get("function") or {}).get("arguments")
                try:
                    args = json.loads(raw or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                break
        else:
            continue
        break
    if not isinstance(args, dict):
        return "selection_unproven: the saved presence_finish arguments are unavailable"
    if "answer" in request and args.get("message") != text:
        return "selection_mismatch: the restored literal differs from the executed presence_finish message"
    if "answer_sha256" in request and args.get("answer_sha256") != request.get("answer_sha256"):
        return "selection_mismatch: the restored hash differs from the executed presence_finish selector"
    return ""


def _restored_schedule_selection_error(ctx: Any, request: dict, trace: dict) -> str:
    """Check the executed choice against the frozen candidate without taking a new evidence reading."""
    candidate = getattr(ctx, "_delivery_candidate", None)
    if candidate is None:
        return "selection_unavailable: no restored candidate for presence_finish"
    if (request.get("answer") != candidate.full_text if "answer" in request
            else request.get("answer_sha256") != candidate.content_sha256):
        return "selection_mismatch: the restored candidate differs from the selected report"
    observation = request.get("observation") or {}
    calls = (trace.get("tool_calls") or [])[int(observation.get("tool_count") or 0):]
    return legacy_selection_error(ctx, request, candidate.full_text, calls)


def validate_restored_schedule_selection(tools: Any, limit_ctx: Any, request: dict) -> bool:
    """Keep an accepted, proven selection byte-exact; on a gap hold it through the existing control path."""
    from ouroboros.loop_delivery import _arm_delivery_control, observe_completion_hold

    ctx, trace = tools._ctx, limit_ctx.llm_trace
    if not (error := _restored_schedule_selection_error(ctx, request, trace)):
        return True
    ctx._completion_request = ctx._completion_selected = ctx._presence_completion = None
    ctx._presence_completion_accepted = False
    _arm_delivery_control(tools, limit_ctx, trace)
    limit_ctx.messages.append({"role": "user", "content": "[SYSTEM NOTICE] Completion held: " + error})
    observe_completion_hold(tools, limit_ctx, "resume", "selection_refused", error.split(":", 1)[0])
    return False


def resolve_completion_selection(tools: Any, limit_ctx: Any, content: Any, *, emit_progress=None) -> bool | None:
    """Select via the common delivery path; None means a restored scheduled report must wait."""
    from ouroboros.loop_delivery import consume_completion_request
    from ouroboros.task_finalization import schedule_report_ctx

    ctx = tools._ctx
    selected = getattr(ctx, "_completion_selected", None)
    if isinstance(selected, dict) and selected.get("source") == SCHEDULE_DOOR:
        if schedule_report_ctx(ctx):
            if not validate_restored_schedule_selection(tools, limit_ctx, selected):
                return None
        else:  # keep the existing cold selection path of interactive turns
            ctx._completion_request, selected = selected, None
    return bool(selected) or consume_completion_request(tools, limit_ctx, limit_ctx.llm_trace, content,
                                                        emit_progress=emit_progress)


def fixed_arguments(ctx: Any) -> list[str]:
    """The selection arguments the ceiling fixes on this schedule's door: a fixed report is never the author's act.
    Any other frozen binding (the older call's ``outcome``) is applied at dispatch like a Presence call's."""
    from ouroboros.contracts.task_contract import canonical_tool_name
    from ouroboros.presence_authority import presence_ceiling_from_context

    ceiling = presence_ceiling_from_context(ctx)
    return sorted({str(binding.argument_path[0]) for grant in (ceiling.tool_grants if ceiling else ())
                   if canonical_tool_name(grant.name) == SCHEDULE_DOOR for binding in grant.bindings
                   if binding.argument_path and binding.argument_path[0] in _SELECTION_ARGUMENTS})


def fixed_refusal(ctx: Any) -> Any:
    """The typed refusal of a door whose report the ceiling fixes, else ``None``; before Safety and staging."""
    from ouroboros.tools.tool_result import ToolResult

    fixed = fixed_arguments(ctx)
    return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
        "⚠️ SCHEDULE_REPORT_FIXED_ARGUMENT: this run's frozen ceiling fixes " + ", ".join(fixed)
        + " of presence_finish; a fixed value cannot be the author's selection of the report.")) if fixed else None


def schedule_finish(ctx: Any, outcome: str = "", message: str = "", action: str = "finish", rationale: str = "",
                    answer_sha256: str = "", pending_review: str | None = None,
                    acceptance_subject: dict | None = None) -> str:
    """Stage the ordinary completion request of the bound run: the selected report, as its quiet record.

    The author's own literal or offered hash is the selection: an argument the frozen ceiling fixes refuses instead
    of choosing or changing it. Outcome words are the older call's and change no audience. Without a reply carrier
    the quiet Presence completion is staged here too, so no ending makes the record speech."""
    from ouroboros.place_reply import reply_carrier
    from ouroboros.tools.control_runtime import stage_completion_request
    from ouroboros.tools.presence import PRESENCE_OUTCOMES
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    door = getattr(ctx, "_schedule_door", None)
    if not isinstance(door, dict) or {key: door.get(key) for key in ("task_id", "schedule_id", "token")} != _occurrence(ctx):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=(
            "ERROR: SCHEDULE_REPORT_DOOR_UNAVAILABLE: this door serves only the scheduled run it was bound for.")))
    if outcome not in ("", *PRESENCE_OUTCOMES):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "ERROR: COMPLETION_ARGUMENT: outcome is message, silent, tool_delivered or deferred; each is recorded alike.")))
    if (refusal := fixed_refusal(ctx)) is not None:
        return _publish_tool_result(ctx, refusal)
    staged = stage_completion_request(ctx, {
        "action": action, "answer": message or None, "answer_sha256": answer_sha256 or None, "rationale": rationale,
        "pending_review": pending_review, "acceptance_subject": acceptance_subject, "wake_silent": True,
    }, source=SCHEDULE_DOOR)
    request = getattr(ctx, "_completion_request", None)
    if (isinstance(request, dict) and request.get("source") == SCHEDULE_DOOR and not getattr(ctx, "_completion_conflict", False)
            and not reply_carrier(getattr(ctx, "task_metadata", None) or {})):
        ctx._presence_completion, ctx._presence_completion_accepted = {"outcome": "silent", "message": "", "quiet": True}, False
    return staged


def refused_door_end(tools: Any, limit_ctx: Any, *, after_tools: bool = False) -> tuple | None:
    """End a bound run whose held report no permitted door can select; else ``None``.

    Called only where Stop, Pause, time and money have already decided, and before another model call. After a batch
    (``after_tools``) the door was tried and a final ordinary refusal answered every attempt of it. At a held prose
    answer that still needs its selection, the effective prohibition (``registry_guards._disabled_tools``) leaves no
    completion tool to select it with: the same ending, without a call that could only be refused. Owner input
    awaiting a reply and other results of that batch are read first, as for any finish; a round of exempt tools alone
    never ends it. The retained report stays the run's failed, unsent result with no Presence declaration (so no
    ending makes it speech); a host notice names the refusal. No second request, no other door and no send follow."""
    from ouroboros.loop_messages import owner_reply_owed
    from ouroboros.task_finalization import TERMINAL_ORIGIN_MODEL_FINAL, set_terminal_host_notice
    from ouroboros.tool_capabilities import completion_observation_calls
    from ouroboros.tools.registry_guards import _disabled_tools

    ctx, trace = tools._ctx, limit_ctx.llm_trace
    candidate = getattr(ctx, "_delivery_candidate", None)
    if (not getattr(ctx, "_schedule_door", None) or candidate is None or getattr(ctx, "_completion_request", None)
            or getattr(ctx, "_completion_selected", None)):
        return None
    batch = (trace.get("tool_calls") or [])[int((getattr(ctx, "_completion_observation", None) or {}).get("tool_count") or 0):]
    calls = [call for call in batch if isinstance(call, dict) and call.get("tool") == SCHEDULE_DOOR]
    if after_tools:
        final_refusal = bool(calls) and all(call.get("is_error") and call.get("tool_result_code") in _FINAL_REFUSALS
                                            for call in calls)
    else:
        final_refusal = (bool(getattr(ctx, "_delivery_control_required", False)) and not candidate.owes_no_selection
                         and SCHEDULE_DOOR in _disabled_tools(ctx) and tools.get_schema_by_name("finish_task") is None)
    if (not final_refusal
            or any(call.get("tool") != SCHEDULE_DOOR for call in completion_observation_calls(batch))
            or (owner_reply_owed(ctx) and tools.get_schema_by_name("send_user_message") is not None)):
        return None
    from ouroboros.loop_forced_finalization import _close_forced_intake, _finalize_forced_services

    _close_forced_intake(limit_ctx)  # a no-call ending, as every such rail: no new mail, its services finalized
    _finalize_forced_services(limit_ctx, trace)
    code = str(calls[-1]["tool_result_code"]) if after_tools else "RESOURCE_CONSTRAINT_BLOCKED"
    refusal = " ".join(str(calls[-1].get("result") or "").split())[:300] if after_tools else ""
    usage = limit_ctx.accumulated_usage
    usage.update(execution_status="failed", reason_code="tool_failure", terminal_origin=TERMINAL_ORIGIN_MODEL_FINAL,
                 wake_silent=True)
    # The visible line is the host's own: the verified refusal type and where the report is. The refusal's dynamic
    # words stay in this call's history (and the trace below); none of them, its arguments or the report ride here.
    set_terminal_host_notice(usage, str(usage.get("terminal_host_notice") or ""), (
        f"This scheduled run's report was not confirmed: {_FINAL_REFUSALS.get(code, 'its completion was refused')} "
        f"({code}). The report stays in this task's result; this completion sent nothing."))
    ctx._presence_completion, ctx._presence_completion_accepted = None, False
    ctx._presence_forced_declaration = {"status": "missing", "reason": "the scheduled run's report was not confirmed"}
    trace["schedule_report_refused"] = {"code": code, "refusal": refusal, "answer_sha256": candidate.content_sha256}
    return candidate.full_text, usage, trace
