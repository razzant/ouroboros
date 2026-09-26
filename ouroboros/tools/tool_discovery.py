"""Tool discovery meta-tools: the ONE implementation of catalog listing and enabling.

Tasks start with the full selected capability envelope (Nano with a compact
resident view). ``list_available_tools`` reports the task's current callable
catalog in every context mode — whether a schema is already loaded is an
attribute of a row, never its availability — and ``enable_tools`` loads a
permitted schema that is not resident yet. Neither grants a capability.

The loop binds both handlers to its resident schema list (``bind_resident_schemas``);
without a loop, the module registry set by ``set_registry`` serves as the fallback.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tool_policy import (
    BUILTIN_NAMESPACE,
    CAPABILITY_OMISSION_HEADER,
    catalog_row_lines,
    compact_tool_catalog,
    format_capability_omissions as _format_omissions,
    select_tool_schemas,
    tool_namespace,
)

if TYPE_CHECKING:
    from ouroboros.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Fallback registry for a context where no loop bound the handlers.
_registry: Optional["ToolRegistry"] = None


def set_registry(reg: "ToolRegistry") -> None:
    global _registry
    _registry = reg


def _resident_names(ctx: Any, schemas: List[Dict[str, Any]], resident: Optional[List[Dict[str, Any]]]) -> set:
    if resident is not None:
        return {schema["function"]["name"] for schema in resident}
    # No loop bound a resident list: residency is what this context mode would load.
    from ouroboros.config import get_context_mode

    mode = str(getattr(ctx, "active_context_mode", "") or get_context_mode())
    return set(select_tool_schemas(schemas, context_mode=mode).chosen)


def _catalog_omissions(
    omissions: List[Dict[str, Any]], namespace: str, callable_names: set[str],
) -> List[Dict[str, Any]]:
    """Explain gaps without listing names that this actor cannot call.

    The general capability manifest legitimately names withheld tools, but a
    *callable catalog* must not promote those names into discovery. A scoped MCP
    read also must not print health details from another server.
    """
    visible = []
    for item in omissions:
        names = item.get("tools")
        if isinstance(names, list):
            selected_names = [str(name) for name in names
                              if not namespace or tool_namespace(str(name)) == namespace]
            if not selected_names:
                continue
            allowed = [name for name in selected_names if name in callable_names]
            item = {**item, "tools": allowed}
            if len(allowed) != len(selected_names):
                item["resource"] = f"{len(selected_names) - len(allowed)} withheld tools (names not callable)"
            if not allowed and not item.get("resource"):
                continue
        if namespace:
            if namespace.startswith("mcp_"):
                if item.get("surface") != "mcp" and item.get("reason") not in {"disabled_by_contract", "missing_credential"}:
                    continue
                if not item.get("servers") and not item.get("resource") and not item.get("tools"):
                    # A generic discovery exception has no server identity;
                    # do not ascribe it to the selected server.
                    continue
                servers = [server for server in (item.get("servers") or [])
                           if isinstance(server, dict) and server.get("id") == namespace[4:]]
                if item.get("servers") and not servers:
                    continue
                if item.get("servers"):
                    item = {**item, "servers": servers}
            elif namespace.startswith("ext_"):
                if item.get("surface") != "extensions":
                    continue
            elif item.get("surface") != "tools":
                continue
        visible.append(item)
    return visible


def list_available_tools(
    registry: "ToolRegistry", ctx: Any, resident: Optional[List[Dict[str, Any]]], namespace: str = "",
) -> str:
    """The task's current callable catalog from ``ToolRegistry.schemas()``.

    Without ``namespace``: per-namespace counts plus the built-in view. With one
    (``builtin``, ``mcp_<server>``, an extension prefix): every callable tool of
    that namespace, an MCP row as ``raw MCP name → callable name``. It describes
    the catalog observed now, not what an earlier request carried.
    """
    schemas = registry.schemas()
    omissions = registry.capability_omissions()
    callable_rows = {row["name"]: row for row in registry.callable_rows() or []}
    rows = [
        {**callable_rows[row["name"]], "loaded": row["residency"] == "loaded",
         "description": (" ".join(callable_rows[row["name"]]["raw_description"].split())[:120]
                         if callable_rows[row["name"]].get("raw_name") else row["description"]),
         "description_truncated": (len(" ".join(callable_rows[row["name"]]["raw_description"].split())) > 120
                                   if callable_rows[row["name"]].get("raw_name") else row["description_truncated"])}
        for row in compact_tool_catalog(schemas, schema_names=_resident_names(ctx, schemas, resident))
        if row["name"] in callable_rows
    ]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tool_namespace(row["name"]), []).append(row)
    selected = str(namespace or "").strip()
    header = ("Current callable catalog (observed now); [loaded] = the schema is already in "
              "your tool list, enable_tools loads a not-loaded one.")
    if selected:
        chosen = groups.get(selected, [])
        if chosen:
            pairs = " (raw MCP name → callable name)" if selected.startswith("mcp_") else ""
            lines = [header, f"{selected}: {len(chosen)} callable tools{pairs}:",
                     *catalog_row_lines(chosen, include_purpose=True)]
        else:
            lines = [header, f"No tool in {selected} is currently callable."
                     " Use list_available_tools() for the namespace overview."]
    else:
        lines = [header, f"{len(rows)} callable tools. Namespaces — list_available_tools(namespace=...) lists one completely:"]
        lines += [f"- {name}: {len(items)} tools, {sum(item['loaded'] for item in items)} loaded"
                  for name, items in sorted(groups.items())]
        # Names only here, so the overview stays bounded; namespace="builtin" adds purposes.
        for state, loaded_flag in (("loaded", True), ("not loaded", False)):
            names = [row["name"] for row in groups.get(BUILTIN_NAMESPACE, []) if row["loaded"] is loaded_flag]
            if names:
                lines.append(f"Built-in, {state}: " + ", ".join(names))
    visible_omissions = _catalog_omissions(omissions, selected, set(callable_rows))
    if visible_omissions:
        lines.extend(_format_omissions(visible_omissions, header="\n" + CAPABILITY_OMISSION_HEADER))
    return "\n".join(lines)


def enable_tools(
    registry: "ToolRegistry", ctx: Any, tools: str, resident: Optional[List[Dict[str, Any]]],
    enabled_late: Optional[set] = None,
) -> str:
    """Load permitted schemas that are not resident; answer every other name honestly."""
    names = [n.strip() for n in str(tools or "").split(",") if n.strip()]
    if not names:
        return "No tools specified."
    resident_names = None if resident is None else {schema["function"]["name"] for schema in resident}
    enabled, hidden, not_found = [], [], []
    for name in names:
        schema = registry.get_schema_by_name(name)
        if schema is None:
            # F3 (2026-08-10 saga): a registered tool filtered by policy is not a typo.
            reason = registry.policy_hidden_reason(name)
            if reason:
                hidden.append(f"{name} — {reason}")
            else:
                not_found.append(name)
        elif resident_names is None:
            enabled.append(f"{name} (callable)")
        elif name in resident_names:
            enabled.append(f"{name} (already active)")
        else:
            from ouroboros.usage_accounting import invalidate_task_cache_splits

            resident.append(schema)
            resident_names.add(name)
            if enabled_late is not None:
                enabled_late.add(name)
            invalidate_task_cache_splits(getattr(ctx, "task_id", ""))
            enabled.append(f"{name} (registered late)")
    parts = []
    if enabled:
        parts.append("✅ Tools are registered in the active capability envelope: " + ", ".join(enabled))
    if hidden:
        parts.append("🚫 Hidden by policy (the tool exists but this task cannot use it): " + "; ".join(hidden))
    if not_found:
        pointer = (" list_available_tools shows the current callable names."
                   if registry.get_schema_by_name("list_available_tools") is not None else "")
        parts.append(f"❌ Not found: {', '.join(not_found)}.{pointer}")
    return "\n".join(parts)


def bind_resident_schemas(registry: "ToolRegistry", resident: List[Dict[str, Any]]) -> set:
    """Bind both handlers of ``registry`` to one loop's resident schema list."""
    enabled_late: set = set()
    registry.override_handler(
        "list_available_tools",
        lambda ctx=None, namespace="", **_kwargs: list_available_tools(registry, ctx, resident, namespace),
    )
    registry.override_handler(
        "enable_tools",
        lambda ctx=None, tools="", **_kwargs: enable_tools(registry, ctx, tools, resident, enabled_late),
    )
    return enabled_late


def _list_available_tools(ctx: ToolContext, namespace: str = "", **kwargs) -> str:
    if _registry is None:
        return "Tool discovery not available in this context."
    return list_available_tools(_registry, ctx, None, namespace)


def _enable_tools(ctx: ToolContext, tools: str = "", **kwargs) -> str:
    if _registry is None:
        return "Tool enablement not available in this context."
    return enable_tools(_registry, ctx, tools, None)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="list_available_tools",
            schema={
                "name": "list_available_tools",
                "description": (
                    "List the tools this task can call now, from the current catalog, in every "
                    "context mode: per-namespace counts (builtin, each MCP server mcp_<server>, "
                    "each extension) plus the built-in names, marking which schemas are already "
                    "loaded. Use namespace to list one namespace completely; an MCP namespace "
                    "shows each raw MCP name beside its callable name. Includes the capability "
                    "omission manifest."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace to list completely, e.g. builtin or mcp_<server>.",
                        },
                    },
                    "required": [],
                },
            },
            handler=_list_available_tools,
        ),
        ToolEntry(
            name="enable_tools",
            schema={
                "name": "enable_tools",
                "description": (
                    "Compatibility check for named tools (comma-separated). Tasks start "
                    "with the selected envelope active, so this confirms registration instead "
                    "of granting delayed core tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tools": {
                            "type": "string",
                            "description": "Comma-separated tool names to enable",
                        }
                    },
                    "required": ["tools"],
                },
            },
            handler=_enable_tools,
        ),
    ]
