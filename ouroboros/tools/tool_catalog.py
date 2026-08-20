"""Intrinsic tool descriptors shared by tool modules and registry dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Collection, Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class ToolEntry:
    """Single tool descriptor."""

    name: str
    schema: Dict[str, Any]
    handler: Callable  # fn(ctx: ToolContext, **args) -> str
    is_code_tool: bool = False
    timeout_sec: int = 360
    # Capability flag: tool can mutate the live repo worktree. The dispatcher
    # snapshots `git status --porcelain` around flagged tools and invalidates
    # advisory freshness when the worktree ACTUALLY changed — covering error
    # and timeout paths uniformly, and never invalidating for read-only runs.
    mutates_worktree: bool = False


class DuplicateToolNameError(ValueError):
    """A first-party or scoped tool attempted to replace an existing name."""

    def __init__(self, name: str, first_origin: str, duplicate_origin: str):
        self.name = str(name)
        self.first_origin = str(first_origin)
        self.duplicate_origin = str(duplicate_origin)
        super().__init__(
            f"duplicate tool name {self.name!r}: first registered by "
            f"{self.first_origin}, then by {self.duplicate_origin}"
        )


@dataclass(frozen=True, init=False)
class ToolCatalog:
    """Immutable first-party tool entries with their registration origins."""

    _entries: Mapping[str, ToolEntry]
    _origins: Mapping[str, str]

    def __init__(self, entries: Iterable[Tuple[str, ToolEntry]]):
        by_name: Dict[str, ToolEntry] = {}
        origins: Dict[str, str] = {}
        for origin, entry in entries:
            name = str(entry.name)
            if name in by_name:
                raise DuplicateToolNameError(name, origins[name], str(origin))
            by_name[name] = entry
            origins[name] = str(origin)
        object.__setattr__(self, "_entries", MappingProxyType(by_name))
        object.__setattr__(self, "_origins", MappingProxyType(origins))

    @property
    def entries(self) -> Mapping[str, ToolEntry]:
        return self._entries

    @property
    def origins(self) -> Mapping[str, str]:
        return self._origins

    def origin_for(self, name: str) -> str:
        return str(self._origins.get(str(name), "unknown"))


def partition_shadowed_tools(
    tools: Iterable[Mapping[str, Any]],
    authoritative_names: Collection[str],
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Split dynamic descriptors without letting them replace catalog names."""

    names = frozenset(str(name) for name in authoritative_names)
    visible: list[Dict[str, Any]] = []
    shadowed: list[Dict[str, Any]] = []
    for tool in tools:
        item = dict(tool)
        target = shadowed if str(item.get("name") or "") in names else visible
        target.append(item)
    return visible, shadowed


__all__ = [
    "DuplicateToolNameError",
    "ToolCatalog",
    "ToolEntry",
    "partition_shadowed_tools",
]
