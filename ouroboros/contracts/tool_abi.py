"""Frozen tool-module ABI: ToolEntry shape and get_tools() signature."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ToolEntryProtocol(Protocol):
    """Structural contract for the public ToolEntry descriptor fields.

    ``timeout_sec`` is the host's outer wait for one call, in seconds. ``None`` (additive)
    is a long-lived entry with no host window: its handler observes the owner's controls
    itself, and only a real deadline can narrow the wait. A numeric value keeps its meaning.
    """

    name: str
    schema: Dict[str, Any]
    handler: Callable[..., str]
    is_code_tool: bool
    timeout_sec: Optional[int]


class GetToolsProtocol(Protocol):
    """Callable contract for ``get_tools()`` exported by every tools module."""

    def __call__(self) -> List[ToolEntryProtocol]: ...


__all__ = ["ToolEntryProtocol", "GetToolsProtocol"]
