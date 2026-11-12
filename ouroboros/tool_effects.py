"""External tool effects shared by place policy, Safety and routeless-final observations.

A connector declares each of its tools in the manifest's ``tools:`` list with
``effect: read | message | gesture | write`` (absent = ``write``) and, where it has one,
``target:`` — the argument(s) naming where the effect lands. ``message`` means
words to people at the target; ``gesture`` is an outward act on one message, not speech; ``write`` is any other change outside. The
declaration is read from the reviewed, published generation the dispatcher
uses, never from the mutable skill folder; the loader's descriptor carries it
from its handler's own registry read (``bound_declaration``), so a later
publication never redescribes a bound call. An MCP tool is a declared read
when the configured server's catalog marks it ``readOnlyHint: true``;
otherwise it is ``write`` with ``declared=False``. Two widenings follow and
are disclosed: a server that lies about a write runs it unchecked, and
Advises may call such tools. A mixed tool (one tool that reads or writes
depending on its arguments) is ``write``, unless its row declares which of its
calls deliver words: ``sends: {when: {<selector>: <value or values>}, target,
conversation, thread}`` makes a call whose selected arguments all hold a listed
value a ``message`` with those targets (a connector-owned fact, e.g. the API
method of a generic Slack tool; core never lists provider methods). Selectors
may be dotted (``body.channel``) to name a nested argument.

Consumers: ``place_guard``'s Advises and targeted Silent/Paused refusals,
routeless-note coverage (``place_reply``), and ``safety`` (declared reads skip
the Outgoing and + Commands checks; ``message`` gets speech semantics).
Built-in tools are governed by their own policy tables and answer ``""``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import threading
from typing import Any, Mapping

EFFECT_READ = "read"
EFFECT_MESSAGE = "message"
EFFECT_GESTURE = "gesture"
EFFECT_WRITE = "write"
EFFECTS = (EFFECT_READ, EFFECT_MESSAGE, EFFECT_GESTURE, EFFECT_WRITE)


@dataclass(frozen=True)
class ToolEffect:
    """The effect fact of one tool name.

    ``effect`` is ``""`` for a built-in tool (not governed here). ``targets``
    names the arguments that carry the destination(s): a list covers e.g.
    to/cc/bcc. ``conversation``/``thread`` name the arguments that address a
    conversation, when the connector declared them. ``message`` names the argument
    carrying a gesture's target message id. ``declared`` is False when the effect
    is the default (no declaration, MCP, unreadable manifest). ``text`` names the
    argument holding the words (``""``: the one named ``text``); my human's yes on
    a held message is checked against those bytes.
    """

    effect: str
    targets: tuple[str, ...] = ()
    conversation: str = ""
    thread: str = ""
    message: str = ""
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


def bound_declaration(ext_tool: Any) -> Mapping[str, Any] | None:
    """The reviewed declaration the loader read with this descriptor's handler, if it names the descriptor's own
    generation; ``None`` for a raw, hand-made or child-catalog descriptor (``get_tool_with_generation``)."""
    from ouroboros.extension_registry_state import REVIEWED_DECLARATION

    facts = ext_tool.get(REVIEWED_DECLARATION) if isinstance(ext_tool, Mapping) else None
    generation = str(ext_tool.get("extension_generation") or "") if isinstance(ext_tool, Mapping) else ""
    if (isinstance(facts, Mapping) and generation and facts.get("extension_generation") == generation
            and isinstance(facts.get("tool"), Mapping)):
        return facts
    return None


def _published_declaration(name: str, ext_tool: Any) -> dict[str, Any]:
    from ouroboros import extension_registry_state as registry
    from ouroboros.extension_surface_names import parse_extension_surface_name

    if not parse_extension_surface_name(name):
        return {}
    bound = bound_declaration(ext_tool)
    if bound is not None:  # a bound call keeps its own publication's facts across any later replacement
        return deepcopy(dict(bound["tool"]))
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
        return registry.reviewed_declaration_locked(name, bundle)["tool"]


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
        message=row.get("message") if isinstance(row.get("message"), str) else "",
        text=row.get("text") if isinstance(row.get("text"), str) else "",
    )


def _declared_send(row: Mapping[str, Any], args: Any) -> bool:
    """Whether the row's ``sends.when`` selects this call: every selector holds one of its listed values."""
    send = row.get("sends")
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
    declares as one that delivers words (``sends.when``) is a ``message`` with that declaration's targets."""
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
        return _routed(row["sends"], EFFECT_MESSAGE, declared=True)
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
