"""What a tool call does to the world outside Ouroboros: one fact, three consumers.

A connector declares each of its tools in the manifest's ``tools:`` list with
``effect: read | message | write`` (absent = ``write``) and, where it has one,
``target:`` — the argument(s) naming where the effect lands. ``message`` means
words to people at the target; ``write`` is any other change outside. The
declaration is read from the reviewed, published generation the dispatcher
uses, never from the mutable skill folder. MCP tools carry no declaration:
they are ``write`` with ``declared=False``, and their annotations can only
make a call stricter. A mixed tool (one tool that reads or writes depending on
its arguments) is ``write``.

Consumers: the Advises refusal (``place_guard``), Silent/Paused refusal of
writes aimed at a place (``place_guard``), and the Supervisor (``safety``:
declared reads skip Light/Full checks; ``message`` gets the speech semantics).
Built-in tools are governed by their own policy tables and answer ``""``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import threading
from typing import Any, Mapping

EFFECT_READ = "read"
EFFECT_MESSAGE = "message"
EFFECT_WRITE = "write"
EFFECTS = (EFFECT_READ, EFFECT_MESSAGE, EFFECT_WRITE)


@dataclass(frozen=True)
class ToolEffect:
    """The effect fact of one tool name.

    ``effect`` is ``""`` for a built-in tool (not governed here). ``targets``
    names the arguments that carry the destination(s): a list covers e.g.
    to/cc/bcc. ``conversation``/``thread`` name the arguments that address a
    conversation, when the connector declared them. ``declared`` is False when
    the effect is the default (no declaration, MCP, unreadable manifest).
    """

    effect: str
    targets: tuple[str, ...] = ()
    conversation: str = ""
    thread: str = ""
    declared: bool = False
    source: str = ""
    annotations: Mapping[str, Any] = field(default_factory=dict)


BUILTIN = ToolEffect(effect="", source="builtin")
_MCP_ANNOTATIONS: dict[str, dict[str, dict[str, Any]]] = {}
_MCP_LOCK = threading.Lock()


def record_mcp_annotations(server_id: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace one server's discovery hints. They never grant read authority.

    One small entry per tool, replaced on catalog adoption; process exit removes
    the cache. Only the four protocol booleans are retained. The adopter ignores
    the returned hint-free rows and presents the original published items.
    """
    from ouroboros.mcp_client import make_tool_name

    hints = {}
    for tool in tools:
        raw = tool.get("annotations")
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        if isinstance(raw, Mapping):
            hints[make_tool_name(server_id, tool.get("name", ""))] = {
                key: raw[key] for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
                if type(raw.get(key)) is bool
            }
    with _MCP_LOCK:
        _MCP_ANNOTATIONS[server_id] = hints
    return [{key: value for key, value in tool.items() if key != "annotations"} for tool in tools]


def connector_of(name: str, ext_tool: Any = None) -> str:
    """The registry's canonical skill identity (encoded tokens are not names)."""
    from ouroboros import extension_registry_state as registry

    with registry._lock:
        tool = ext_tool if isinstance(ext_tool, Mapping) else registry._tools.get(name)
        return str((tool or {}).get("skill") or "")


def _published_declaration(name: str, ext_tool: Any) -> dict[str, Any]:
    from ouroboros import extension_registry_state as registry
    from ouroboros.extension_surface_names import parse_extension_surface_name

    parsed = parse_extension_surface_name(name)
    if not parsed:
        return {}
    with registry._lock:
        tool = ext_tool if isinstance(ext_tool, Mapping) else registry._tools.get(name)
        if not tool:
            return {}
        bundle = registry._extensions.get(str(tool.get("skill") or ""))
        if bundle is None:
            return {}
        generation = tool.get("extension_generation")
        if generation and generation != bundle.generation_digest:
            return {}  # Never pair an old handler with a new declaration.
        manifest = getattr(bundle.loaded_skill, "manifest", None)
        rows = getattr(manifest, "raw_extra", {}).get("tools", [])
        if not isinstance(rows, list):
            return {}
        matches = [row for row in rows if isinstance(row, Mapping) and row.get("name") == parsed[1]]
        return deepcopy(matches[0]) if len(matches) == 1 else {}


def _argument_names(value: Any) -> tuple[str, ...]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in values):
        return ()
    return tuple(dict.fromkeys(v.strip() for v in values))


def effect_of(name: str, *, ext_tool: Any = None, is_mcp: bool = False, ctx: Any = None) -> ToolEffect:
    """Resolve the published declaration; defaults never relax a call's policy."""
    from ouroboros.extension_surface_names import parse_extension_surface_name

    if is_mcp or name.startswith("mcp_"):
        with _MCP_LOCK:
            hints = next((dict(rows[name]) for rows in _MCP_ANNOTATIONS.values() if name in rows), {})
        return ToolEffect(effect=EFFECT_WRITE, source="mcp", annotations=hints)
    if ext_tool is None and not parse_extension_surface_name(name):
        return BUILTIN
    row = _published_declaration(name, ext_tool)
    effect = row.get("effect")
    declared = isinstance(effect, str) and effect in EFFECTS
    return ToolEffect(
        effect=effect if declared else EFFECT_WRITE, declared=declared, source="extension",
        targets=_argument_names(row.get("target")),
        conversation=row.get("conversation") if isinstance(row.get("conversation"), str) else "",
        thread=row.get("thread") if isinstance(row.get("thread"), str) else "",
    )


def target_values(effect: ToolEffect, args: Mapping[str, Any]) -> dict[str, list[str]]:
    """The declared target argument values of one call, by argument name (strings only)."""
    values: dict[str, list[str]] = {}
    source = args if isinstance(args, Mapping) else {}
    for name in (*effect.targets, effect.conversation, effect.thread):
        if not name or name in values:
            continue
        raw = source.get(name)
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        texts = [str(item).strip() for item in items if type(item) in (str, int) and str(item).strip()]
        if texts:
            values[name] = texts
    return values
