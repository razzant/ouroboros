"""Tool discovery meta-tools — lets the agent see and enable non-core tools."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ouroboros.tools.registry import ToolContext, ToolEntry

if TYPE_CHECKING:
    from ouroboros.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Module-level registry reference — set by set_registry() after ToolRegistry is created.
# loop.py also overrides these handlers with closures that have access to per-loop state
# (e.g. the _enabled_extra_tools set); the module-level ref serves as a fallback for
# any context where the tool is called without going through run_llm_loop.
_registry: Optional["ToolRegistry"] = None


def set_registry(reg: "ToolRegistry") -> None:
    global _registry
    _registry = reg


def _list_available_tools(ctx: ToolContext, **kwargs) -> str:
    if _registry is None:
        return "Tool discovery not available in this context."
    non_core = _registry.list_non_core_tools()
    # Exclude the meta-tools themselves from the listing
    non_core = [t for t in non_core if t["name"] not in ("list_available_tools", "enable_tools")]
    if not non_core:
        return "All tools are already in your active set."
    lines = [f"**{len(non_core)} additional tools available** (use `enable_tools` to activate):\n"]
    for t in non_core:
        lines.append(f"- **{t['name']}**: {t['description'][:120]}")
    return "\n".join(lines)


def _enable_tools(ctx: ToolContext, tools: str = "", **kwargs) -> str:
    if _registry is None:
        return "Tool enablement not available in this context."
    names = [n.strip() for n in tools.split(",") if n.strip()]
    if not names:
        return "No tools specified."
    found = []
    not_found = []
    for name in names:
        schema = _registry.get_schema_by_name(name)
        if schema:
            found.append(f"{name}: {schema['function'].get('description', '')[:100]}")
        else:
            not_found.append(name)
    parts = []
    if found:
        parts.append("✅ Tools are registered and callable:\n" + "\n".join(f"  - {s}" for s in found))
    if not_found:
        parts.append(f"❌ Not found: {', '.join(not_found)}")
    return "\n".join(parts)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="list_available_tools",
            schema={
                "name": "list_available_tools",
                "description": (
                    "List all additional tools not currently in your active tool set. "
                    "Returns name + description for each. Use this to discover tools "
                    "you might need for specific tasks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
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
                    "Enable specific additional tools by name (comma-separated). "
                    "Their schemas will be added to your active tool set for the "
                    "remainder of this task. Example: enable_tools(tools='multi_model_review,generate_self_portrait')"
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
