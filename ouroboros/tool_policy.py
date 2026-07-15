"""Task-start tool visibility policy.

This module determines which tools are available at the start of a task
without an explicit ``enable_tools`` call.  Tool sets themselves live in
``ouroboros.tool_capabilities`` (the single source of truth).
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol


class ToolSchemaProvider(Protocol):
    """Minimal registry contract needed by the loop/discovery helpers."""

    def schemas(self, core_only: bool = False, compact: bool = False) -> List[Dict[str, Any]]:
        ...


def initial_tool_schemas(registry: ToolSchemaProvider, *, compact: bool = False) -> List[Dict[str, Any]]:
    """Return the full capability envelope that should be present from round 1.

    Visibility is selected by the registry context: normal main/direct/evolution
    tasks expose all available first-party built-ins plus live extension/MCP
    schemas; workspace and local-readonly tasks expose their guarded envelope.
    No enabled schema is silently skipped here.

    When compact=True, non-core tools are returned as name+description stubs
    (no parameter schemas) to reduce context window consumption by ~20K tokens.
    """

    return registry.schemas(compact=compact)


def list_non_core_tools(registry: ToolSchemaProvider) -> List[Dict[str, str]]:
    """Return name+description for tools that require explicit enable_tools."""

    return []
