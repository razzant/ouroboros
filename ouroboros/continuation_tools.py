"""Reconcile cold-resume residency only after its frozen prefix was re-rendered.

The answered-send observer owns baseline identity; the route owns its physical
schema limit. This owner stages current contracts, checks history against the
fitted candidate and commits the shared list only when compatible. It never
retires reasoning or edits a recovered source: the send/checkpoint owners do so.
"""

from __future__ import annotations

from typing import Any

from ouroboros import transcript_prefix
from ouroboros.provider_models import tool_schema_limit
from ouroboros.tool_policy import (
    fit_tool_schemas_to_limit,
    format_capability_omissions,
    initial_tool_schemas,
    route_tool_limit_notice,
    usable_selectors,
)


def _by_name(schemas: list) -> dict:
    return {schema["function"]["name"]: schema for schema in schemas}


def history_tool_names(messages: list) -> set[str]:
    """Both canonical calls and native content blocks carry history contracts.

    Private native custody receipts replay only with matching canonical calls;
    those names are already covered here, without decoding opaque receipts.
    """
    names = set()
    for message in messages:
        for call in message.get("tool_calls") or ():
            name = (call.get("function") or {}).get("name")
            if name:
                names.add(name)
        content = message.get("content")
        if isinstance(content, list):
            names.update(block["name"] for block in content if isinstance(block, dict)
                         and block.get("type") == "tool_use" and block.get("name"))
    return names


def reconcile_restored_tools(tools: Any, state: dict, messages: list, mode: str) -> bool:
    """Return whether the caller must re-fit context with the adopted list.

    Called AFTER the ordinary route rebind, which still uses frozen tools. A
    candidate's own context projection must never justify adopting that candidate.
    Current capability notes belong after restore even when residency stays frozen.
    """
    ctx, frozen = tools._ctx, state["tool_schemas"]
    previous, before_tools = transcript_prefix._baseline(ctx)
    frozen_digest = transcript_prefix.tools_digest(frozen)
    intact = (isinstance(previous, list) and bool(previous) and bool(messages)
              and previous[0] == transcript_prefix.message_digest(messages[0])
              and before_tools == frozen_digest)
    changed = False
    if not intact:
        changed = _adopt_current_tools(tools, state, messages, mode, frozen_digest)
    omissions = tools.capability_omissions() if hasattr(tools, "capability_omissions") else []
    if omissions:
        messages.append({"role": "user", "content":
                         "[SYSTEM NOTICE]\n" + "\n".join(format_capability_omissions(omissions))})
    return changed


def _adopt_current_tools(tools: Any, state: dict, messages: list, mode: str, before: str) -> bool:
    ctx, frozen = tools._ctx, state["tool_schemas"]
    candidate = initial_tool_schemas(tools, context_mode=mode)
    old, fresh = _by_name(frozen), _by_name(candidate)
    loaded = getattr(ctx, "_actor_loaded_tool_names", set())
    earlier = set(getattr(ctx, "_route_left_out_tool_names", ()) or ())
    # Saved residency, then what the actor loaded or a ceiling left out: a roomier route brings it back.
    for name in [*old, *sorted((loaded | earlier) - old.keys())]:
        if name in fresh:
            continue
        schema = tools.get_schema_by_name(name)
        if schema is not None:  # current policy, never the saved definition
            candidate.append(schema)
            fresh[name] = schema
    current_names = _by_name(candidate).keys()
    # Legacy residency is evidence of selection, not of an enable_tools action.
    keep = loaded | earlier | (old.keys() & current_names if "actor_loaded_tool_names" not in state else set())
    limit = tool_schema_limit(ctx.active_model, use_local=ctx.active_use_local)
    fitted, left_out = fit_tool_schemas_to_limit(candidate, limit, keep=keep, protect=history_tool_names(messages))
    current = _by_name(fitted)
    after = transcript_prefix.tools_digest(fitted)
    if after == before:
        return False
    missing = (history_tool_names(messages) & old.keys()) - current.keys()
    if missing:
        messages.append({"role": "user", "content": (
            "[SYSTEM NOTICE]\nThe saved tool set was kept: the current tool set after route fitting "
            "cannot represent tools in this task's history: " + ", ".join(sorted(missing)) + ".")})
        return False
    frozen[:] = fitted
    loaded.intersection_update(fresh)  # current policy forgets a revoked tool; a ceiling never forgets a choice
    ctx._route_left_out_tool_names = earlier | set(left_out)
    from ouroboros.usage_accounting import invalidate_task_cache_splits
    from ouroboros.loop_messages import _emit_checkpoint_event

    invalidate_task_cache_splits(getattr(ctx, "task_id", ""))
    added, removed = sorted(current.keys() - old.keys()), sorted(old.keys() - current.keys())
    changed = sorted(name for name in current.keys() & old.keys()
                     if transcript_prefix.tools_digest([current[name]]) != transcript_prefix.tools_digest([old[name]]))
    messages.append({"role": "user", "content": (
        "[SYSTEM NOTICE]\nThe tool set changed since this task's last request: "
        f"+added {', '.join(added) or 'none'}; −removed {', '.join(removed) or 'none'}; "
        f"~changed {', '.join(changed) or 'none'}.")})
    if left_out:
        messages.append({"role": "user", "content": route_tool_limit_notice(
            ctx.active_model, limit, len(candidate), left_out, selectors=usable_selectors(tools, fitted))})
    fact = {"checkpoint_kind": "continuation_tools_changed", "from_tools_digest": before,
            "to_tools_digest": after, "added": added, "removed": removed, "changed": changed}
    trace = getattr(ctx, "_execution_trace", None)
    if isinstance(trace, dict):
        trace.setdefault("continuation_tools", []).append(fact)
    drive_logs = getattr(ctx, "drive_logs", None)
    _emit_checkpoint_event(getattr(ctx, "event_queue", None), getattr(ctx, "task_id", ""),
                           drive_logs() if callable(drive_logs) else drive_logs, fact)
    return True
