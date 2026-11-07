"""What a tool call does to the world outside Ouroboros: one fact, three consumers.

A connector declares each of its tools in the manifest's ``tools:`` list with
``effect: read | message | write`` (absent = ``write``) and, where it has one,
``target:`` — the argument(s) naming where the effect lands. ``message`` means
words to people at the target; ``write`` is any other change outside. The
declaration is read from the reviewed, published generation the dispatcher
uses, never from the mutable skill folder. An MCP tool is a declared read
when the configured server's catalog marks it ``readOnlyHint: true``;
otherwise it is ``write`` with ``declared=False``. Two widenings follow and
are disclosed: a server that lies about a write runs it unchecked, and
Advises may call such tools. A mixed tool (one tool that reads or writes
depending on its arguments) is ``write``, unless its row declares which of its
calls deliver words: ``message: {when: {<selector>: <value or values>}, target,
conversation, thread}`` makes a call whose selected arguments all hold a listed
value a ``message`` with those targets (a connector-owned fact, e.g. the API
method of a generic Slack tool; core never lists provider methods). Selectors
may be dotted (``body.channel``) to name a nested argument.

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
    ``text`` names the argument holding the words (``""``: the one named
    ``text``); my human's yes on a held message is checked against those bytes.
    """

    effect: str
    targets: tuple[str, ...] = ()
    conversation: str = ""
    thread: str = ""
    declared: bool = False
    source: str = ""
    annotations: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""


BUILTIN = ToolEffect(effect="", source="builtin")
_MCP_ANNOTATIONS: dict[str, dict[str, dict[str, Any]]] = {}
_MCP_LOCK = threading.Lock()


def record_mcp_annotations(server_id: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace one server's discovery hints; ``readOnlyHint: true`` makes its tool a declared read.

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


def argument_value(args: Any, selector: str) -> Any:
    """The argument a (possibly dotted) declared selector names: ``body.channel`` is ``args["body"]["channel"]``."""
    value = args
    for part in selector.split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    return value


def _routed(row: Mapping[str, Any], effect: str, *, declared: bool) -> ToolEffect:
    return ToolEffect(
        effect=effect, declared=declared, source="extension", targets=_argument_names(row.get("target")),
        conversation=row.get("conversation") if isinstance(row.get("conversation"), str) else "",
        thread=row.get("thread") if isinstance(row.get("thread"), str) else "",
        text=row.get("text") if isinstance(row.get("text"), str) else "",
    )


def _declared_send(row: Mapping[str, Any], args: Any) -> bool:
    """Whether the row's ``message.when`` selects this call: every selector holds one of its listed values."""
    send = row.get("message")
    when = send.get("when") if isinstance(send, Mapping) else None
    if not isinstance(when, Mapping) or not when or not isinstance(args, Mapping):
        return False
    for selector, wanted in when.items():
        values = [wanted] if isinstance(wanted, str) else wanted
        if (not isinstance(selector, str) or not selector.strip() or not isinstance(values, list)
                or not all(isinstance(value, str) for value in values) or argument_value(args, selector) not in values):
            return False
    return True


def effect_of(name: str, *, ext_tool: Any = None, is_mcp: bool = False, ctx: Any = None, args: Any = None) -> ToolEffect:
    """Resolve the published declaration; defaults never relax a call's policy. With ``args``, a call the row
    declares as one that delivers words (``message.when``) is a ``message`` with that declaration's targets."""
    from ouroboros.extension_surface_names import parse_extension_surface_name

    if is_mcp or name.startswith("mcp_"):
        with _MCP_LOCK:
            hints = next((dict(rows[name]) for rows in _MCP_ANNOTATIONS.values() if name in rows), {})
        read = hints.get("readOnlyHint") is True
        return ToolEffect(effect=EFFECT_READ if read else EFFECT_WRITE, declared=read, source="mcp", annotations=hints)
    if ext_tool is None and not parse_extension_surface_name(name):
        return BUILTIN
    row = _published_declaration(name, ext_tool)
    if args is not None and _declared_send(row, args):
        return _routed(row["message"], EFFECT_MESSAGE, declared=True)
    effect = row.get("effect")
    declared = isinstance(effect, str) and effect in EFFECTS
    return _routed(row, effect if declared else EFFECT_WRITE, declared=declared)


def mcp_sends_as(name: str) -> dict[str, str]:
    """Whose credential an MCP call acts with, as host facts for the Supervisor's bundle; {} when unknown."""
    from ouroboros import mcp_client, mcp_oauth

    try:
        parsed = mcp_client.parse_tool_name(name)
        cfg = mcp_client.get_manager().server_config(parsed["server_slug"]) if parsed else None
        signed = (mcp_oauth.status(cfg) or {}) if cfg is not None else {}
    except Exception:
        return {}
    if cfg is None:
        return {}
    acts_with = (f"your human's sign-in at {signed.get('issuer') or cfg.name}" if signed.get("state") == mcp_oauth.SIGNED_IN
                 else "credentials saved for it in your human's settings" if cfg.carries_credentials()
                 else "settings passed to its process: " + ", ".join(sorted(cfg.env_from_settings)) if cfg.env_from_settings
                 else "")
    return {"mcp_server": cfg.id, "label": cfg.name, **({"acts_with": acts_with} if acts_with else {})}


def target_values(effect: ToolEffect, args: Mapping[str, Any]) -> dict[str, list[str]]:
    """The declared target argument values of one call, by argument name (strings only; dotted names nested)."""
    values: dict[str, list[str]] = {}
    source = args if isinstance(args, Mapping) else {}
    for name in (*effect.targets, effect.conversation, effect.thread):
        if not name or name in values:
            continue
        raw = argument_value(source, name)
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        texts = [str(item).strip() for item in items if type(item) in (str, int) and str(item).strip()]
        if texts:
            values[name] = texts
    return values
