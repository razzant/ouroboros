"""Reconcile cold-resume residency only after its frozen prefix was re-rendered.

The answered-send observer owns baseline identity; the route owns its physical
schema limit. This owner stages current contracts, checks history against the
fitted candidate; only the final context frame adopts its list and notes.
Fitting keeps the names history called first. When the saved mode and route cap permit it, a missing history
contract keeps the saved array and says why. A new smaller frame does not preserve the full-mode envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ouroboros import transcript_prefix
from ouroboros.tool_policy import format_capability_omissions, initial_tool_schemas, routine_capability_omissions
from ouroboros.window_learning import RouteSchemas, fit_route_schemas


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


def restore_omissions(tools):
    omissions = routine_capability_omissions(
        tools.capability_omissions() if hasattr(tools, "capability_omissions") else [])
    return ([{"role": "user", "content": "[SYSTEM NOTICE]\n" + "\n".join(format_capability_omissions(omissions))}]
            if omissions else [])


def frozen_tools_intact(tools, frozen, messages, mode, limit):
    previous, before_tools = transcript_prefix._baseline(tools._ctx)
    return (isinstance(previous, list) and bool(previous) and bool(messages)
            and (limit is None or len(frozen) <= limit)
            and previous[0] == transcript_prefix.message_digest(messages[0])
            and before_tools == transcript_prefix.tools_digest(frozen) and not _gateway_due(tools, frozen, mode))


def _gateway_due(tools: Any, frozen: list, mode: str) -> bool:
    """A list saved before the MCP gateway existed lacks it while a fresh start in ``mode`` carries it: adopting
    the current list once (with its notice) costs one tools change instead of one per later MCP load."""
    from ouroboros.tool_capabilities import MCP_GATEWAY_TOOL_NAME as gateway

    return gateway not in _by_name(frozen) and gateway in _by_name(initial_tool_schemas(tools, context_mode=mode))


@dataclass
class RestoredTools:
    fitted: RouteSchemas
    change: dict | None = None

    def price(self):
        return self.fitted.price()

    def adopt(self, tools, frozen, messages):
        """Publish this frame's own rows, plus its unchanged or reconciled catalog."""
        ctx = tools._ctx
        self.fitted.adopt(ctx, frozen, messages, by="continuation_restored")
        if not self.change:
            return False
        loaded = getattr(ctx, "_actor_loaded_tool_names", set())
        loaded.intersection_update(name for name in loaded if tools.get_schema_by_name(name) is not None)
        from ouroboros.loop_messages import _emit_checkpoint_event

        trace = getattr(ctx, "_execution_trace", None)
        if isinstance(trace, dict):
            trace.setdefault("continuation_tools", []).append(self.change)
        drive_logs = getattr(ctx, "drive_logs", None)
        _emit_checkpoint_event(getattr(ctx, "event_queue", None), getattr(ctx, "task_id", ""),
                               drive_logs() if callable(drive_logs) else drive_logs, self.change)
        return True


def reconcile_restored_tools(tools: Any, state: dict, messages: list, mode: str, *, saved_mode, model, limit) -> RestoredTools:
    """Price one permitted catalog and its exact notices without adopting either."""
    ctx, frozen = tools._ctx, state["tool_schemas"]
    old, history = _by_name(frozen), history_tool_names(messages)
    loaded = getattr(ctx, "_actor_loaded_tool_names", set())
    earlier = set(getattr(ctx, "_route_left_out_tool_names", ()) or ())
    retained = (set(old) | earlier) if mode != "nano" or saved_mode == "nano" else set()
    candidate = initial_tool_schemas(tools, context_mode=mode, schema_names=sorted(loaded | history | retained))
    legacy = set(old) if mode == saved_mode and "actor_loaded_tool_names" not in state else set()
    fitted = fit_route_schemas(tools, messages, candidate, model, limit, loaded=loaded | legacy, history=history)
    current = _by_name(fitted.schemas)
    before, after = (transcript_prefix.tools_digest(schemas) for schemas in (frozen, fitted.schemas))
    missing = (history & old.keys()) - current.keys()
    if missing and mode == saved_mode and (limit is None or len(frozen) <= limit):
        return RestoredTools(RouteSchemas(frozen, [{"role": "user", "content": (
            "[SYSTEM NOTICE]\nThe saved tool set was kept: the current tool set after route fitting "
            "cannot represent tools in this task's history: " + ", ".join(sorted(missing)) + ".")}]))
    if after == before:
        return RestoredTools(fitted)
    added, removed = sorted(current.keys() - old.keys()), sorted(old.keys() - current.keys())
    changed = sorted(name for name in current.keys() & old.keys()
                     if transcript_prefix.tools_digest([current[name]]) != transcript_prefix.tools_digest([old[name]]))
    fitted.notice.insert(0, {"role": "user", "content": (
        "[SYSTEM NOTICE]\nThe tool set changed since this task's last request: "
        f"+added {', '.join(added) or 'none'}; −removed {', '.join(removed) or 'none'}; "
        f"~changed {', '.join(changed) or 'none'}.")})
    fact = {"checkpoint_kind": "continuation_tools_changed", "from_tools_digest": before,
            "to_tools_digest": after, "added": added, "removed": removed, "changed": changed}
    return RestoredTools(fitted, fact)
