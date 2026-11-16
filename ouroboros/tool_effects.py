"""External tool effects shared by place policy, Safety and routeless-final observations.

A connector declares each of its tools in the manifest's ``tools:`` list with
``effect: read | message | gesture | write`` (absent = ``write``) and, where it has one,
``target:`` — the argument(s) naming where the effect lands. ``message`` means
words to people at the target; ``gesture`` is an outward act on one message, not speech; ``write`` is any other change outside. The
declaration is read from the reviewed, published generation the dispatcher
uses, never from the mutable skill folder; the loader's descriptor carries it
from its handler's own registry read (``bound_declaration``), and the call's
result keeps it (``retained_declaration``), so a later publication never
redescribes a bound or executed call. An MCP tool is a declared read
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

A row, or its ``sends`` block for the calls it selects, may add ``touches``: fixed groups of further exact
addresses one call reaches, each ``{conversation, effect, account?, thread?, match?}`` (``touched_addresses``).
The effect is the same vocabulary: ``message`` only beside a message call, ``gesture`` only beside a gesture, and
a read call only reads; ``match`` is ``scope`` (default) or, for a write or read group only, ``exact`` or
``exact_casefold`` (``places.target_place``). A payload using them declares PluginAPI 2.5 or newer, so an older host refuses it
before import; an invalid group, or groups under an older declaration, leave the whole declaration untrusted
(``ToolEffect.problem``).

Consumers: ``place_guard``'s Advises and targeted Silent/Paused refusals,
routeless-note coverage (``place_reply``), and ``safety`` (declared reads skip
the Outgoing and + Commands checks; ``message`` gets speech semantics).
Built-in tools are governed by their own policy tables and answer ``""``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import re
import threading
from typing import Any, Mapping

EFFECT_READ = "read"
EFFECT_MESSAGE = "message"
EFFECT_GESTURE = "gesture"
EFFECT_WRITE = "write"
EFFECTS = (EFFECT_READ, EFFECT_MESSAGE, EFFECT_GESTURE, EFFECT_WRITE)
# The PluginAPI minimum a payload declares to use ``touches`` (contracts/plugin_api.py: 2.5).
TOUCHES_PLUGIN_API = (2, 5)
_TOUCH_KEYS = frozenset({"conversation", "effect", "account", "thread", "match"})


@dataclass(frozen=True)
class TouchGroup:
    """One declared group: the argument naming its conversation id(s), the effect landing there, the arguments
    naming its account (``""``: the call's ``account_id``) and thread (``""``: none), and how its ids meet places
    (``places.target_place``: ``scope``, or ``exact``/``exact_casefold`` for an extra constraint on a write or read
    group)."""

    conversation: str
    effect: str
    account: str = ""
    thread: str = ""
    match: str = "scope"


@dataclass(frozen=True)
class TouchAddress:
    """One exact address a call reaches through a group: what lands there, the argument that named it, and how it
    meets places (its group's ``match``)."""

    effect: str
    conversation: str
    account: str = ""
    thread: str = ""
    argument: str = ""
    match: str = "scope"


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
    a held message is checked against those bytes. ``touches`` are the groups this
    call's effect carries; ``problem`` says why the declaration cannot be trusted
    (the effect is then an undeclared ``write`` and dispatch refuses the call).
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
    touches: tuple[TouchGroup, ...] = ()
    problem: str = ""


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


# A dispatched call's declaration rides in its result facts (``result_meta``, so its trace row) under this key.
RETAINED_DECLARATION = "bound_declaration"
_DECLARATION_KEYS = ("name", "effect", "target", "conversation", "thread", "message", "text", "sends", "touches")


def retained_declaration(ext_tool: Any) -> dict[str, Any] | None:
    """The reviewed declaration a dispatched call ran with, as plain facts its result keeps: readers after the call
    (``place_reply.riding_sends``) judge it by them, never by a later publication. No handler is kept."""
    facts = bound_declaration(ext_tool)
    if facts is None:
        return None
    tool = {key: deepcopy(facts["tool"][key]) for key in _DECLARATION_KEYS if key in facts["tool"]}
    return {"skill": str(ext_tool.get("skill") or ""), "extension_generation": facts["extension_generation"],
            "tool": tool, "provides_place": bool(facts.get("provides_place")),
            "plugin_api": str(facts.get("plugin_api") or "")}


def retained_descriptor(row: Any) -> dict[str, Any] | None:
    """The descriptor a recorded call's retained declaration stands for (``retained_declaration``); None for a
    row recorded without one."""
    from ouroboros.extension_registry_state import REVIEWED_DECLARATION

    facts = row.get(RETAINED_DECLARATION) if isinstance(row, Mapping) else None
    if not isinstance(facts, Mapping) or not facts.get("skill") or not facts.get("extension_generation"):
        return None
    return {"skill": str(facts["skill"]), "extension_generation": str(facts["extension_generation"]),
            REVIEWED_DECLARATION: {key: deepcopy(value) for key, value in facts.items() if key != "skill"}}


def _published_declaration(name: str, ext_tool: Any) -> tuple[dict[str, Any], str]:
    """The tool's reviewed row and the PluginAPI minimum its manifest declares (``""`` when unknown)."""
    from ouroboros import extension_registry_state as registry
    from ouroboros.extension_surface_names import parse_extension_surface_name

    if not parse_extension_surface_name(name):
        return {}, ""
    bound = bound_declaration(ext_tool)
    if bound is not None:  # a bound call keeps its own publication's facts across any later replacement
        return deepcopy(dict(bound["tool"])), str(bound.get("plugin_api") or "")
    with registry._lock:
        tool = ext_tool if isinstance(ext_tool, Mapping) else registry._tools.get(name)
        if not tool:
            return {}, ""
        bundle = registry._extensions.get(str(tool.get("skill") or ""))
        if bundle is None:
            return {}, ""
        generation = tool.get("extension_generation")
        if generation and generation != bundle.generation_digest:
            return {}, ""  # Never pair an old handler with a new declaration.
        facts = registry.reviewed_declaration_locked(name, bundle)
        return facts["tool"], str(facts.get("plugin_api") or "")


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
        touches=_touch_groups(row.get("touches"), effect)[0],
    )


def _touch_groups(raw: Any, operation: str) -> tuple[tuple[TouchGroup, ...], str]:
    """A ``touches`` list as groups, and why it cannot be trusted (``""`` when it can; absent is no groups).

    A group's effect must fit the call it rides: words beside words, a gesture beside a gesture, never a change
    beside a read. An exact ``match`` (``exact``, ``exact_casefold``) marks an extra constraint, so only a write or read
    group takes it: words and gestures keep the ordinary attribution and author lease of their conversation."""
    from ouroboros.places import MATCH_SCOPE, MATCHES

    if raw is None:
        return (), ""
    if not isinstance(raw, list):
        return (), "touches must be a list of groups"
    groups = []
    for index, item in enumerate(raw):
        where = f"touches[{index}]"
        if not isinstance(item, Mapping) or not set(item) <= _TOUCH_KEYS:
            return (), f"{where} may hold only conversation, effect, account, thread and match"
        match = item.get("match", MATCH_SCOPE)
        if match not in MATCHES:
            return (), f"{where}: match is {' or '.join(MATCHES)}"
        if match != MATCH_SCOPE and item.get("effect") in (EFFECT_MESSAGE, EFFECT_GESTURE):
            return (), f"{where}: match {match} is for a write or read group; words and gestures keep their conversation"
        names = {key: item.get(key) for key in ("conversation", "account", "thread") if item.get(key) is not None}
        if any(not isinstance(value, str) or not value.strip() for value in names.values()):
            return (), f"{where}: conversation, account and thread each name one argument"
        if "conversation" not in names:
            return (), f"{where} needs conversation: the argument holding the conversation id or ids it reaches"
        effect = item.get("effect")
        if effect not in EFFECTS:
            return (), f"{where} needs effect: read, message, gesture or write"
        if effect != EFFECT_READ and (operation == EFFECT_READ or (
                effect in (EFFECT_MESSAGE, EFFECT_GESTURE) and effect != operation)):
            return (), f"{where}: a {effect} group cannot ride a {operation} call"
        groups.append(TouchGroup(names["conversation"].strip(), effect, (names.get("account") or "").strip(),
                                 (names.get("thread") or "").strip(), match))
    return tuple(dict.fromkeys(groups)), ""


def _api_minimum(version: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]{1,4})\.([0-9]{1,4})", version.strip())
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _declaration_problem(row: Mapping[str, Any], plugin_api: str) -> str:
    """Why a row's ``touches`` (its own, or its ``sends`` block's) cannot be trusted, judged whole: ``""`` when they
    can or there are none."""
    effect = row.get("effect") if row.get("effect") in EFFECTS else EFFECT_WRITE
    send = row.get("sends") if isinstance(row.get("sends"), Mapping) else {}
    declared = [(raw, operation, where) for raw, operation, where in (
        (row.get("touches"), effect, "touches"), (send.get("touches"), EFFECT_MESSAGE, "sends.touches"))
        if raw is not None]
    if not declared:
        return ""
    if _api_minimum(plugin_api) < TOUCHES_PLUGIN_API:
        needed = ".".join(str(part) for part in TOUCHES_PLUGIN_API)
        return (f"touches need the manifest's plugin_api at \"{needed}\" or newer (this payload binds "
                f"{plugin_api or 'an unknown generation'}), or an older host would run it ignoring them")
    for raw, operation, where in declared:
        problem = _touch_groups(raw, operation)[1]
        if problem:
            return problem.replace("touches", where, 1)
    return ""


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
    declares as one that delivers words (``sends.when``) is a ``message`` with that declaration's targets and
    groups. A declaration whose groups cannot be trusted is an undeclared ``write`` naming its ``problem``."""
    from ouroboros.extension_surface_names import parse_extension_surface_name

    if is_mcp or name.startswith("mcp_"):
        with _MCP_LOCK:
            hints = next((dict(rows[name]) for rows in _MCP_ANNOTATIONS.values() if name in rows), {})
        read = hints.get("readOnlyHint") is True
        return ToolEffect(effect=EFFECT_READ if read else EFFECT_WRITE, declared=read, source="mcp", annotations=hints)
    if ext_tool is None and not parse_extension_surface_name(name):
        return BUILTIN
    row, plugin_api = _published_declaration(name, ext_tool)
    if problem := _declaration_problem(row, plugin_api):
        return ToolEffect(effect=EFFECT_WRITE, source="extension", problem=problem)
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


def thread_pairs(conversations: list[str], threads: list[str], *, same_argument: bool) -> tuple[list[tuple[str, str]], str]:
    """Each conversation id with its own thread: one argument naming both pairs each value with itself; another
    argument's threads all belong to a single conversation, or its one thread to every conversation. Several of
    each pair nothing (a problem): never zipped by position or multiplied into pairs the call never named."""
    if same_argument:
        return [(value, value) for value in conversations], ""
    if len(conversations) > 1 and len(threads) > 1:
        return [], (f"{len(conversations)} conversations beside {len(threads)} threads cannot be paired; name one "
                    "of them per call, or the conversation argument itself as its thread")
    return [(value, thread) for value in conversations for thread in (threads or [""])], ""


def _lookup(args: Any, selector: str) -> tuple[Any, str]:
    """A group argument by its (possibly dotted) selector: an absent or null step is absence; a supplied step of
    another shape is a problem, so a path that cannot be read never reads as missing."""
    value, walked = args, []
    for part in selector.split("."):
        if value is None:
            return None, ""
        if not isinstance(value, Mapping):
            return None, f"{'.'.join(walked)!r} holds a {type(value).__name__}, so {selector!r} cannot be read"
        walked.append(part)
        value = value.get(part)
    return value, ""


def _address_values(args: Mapping[str, Any], selector: str) -> tuple[list[str], str]:
    """A group's conversation ids: absent, null or empty reach nothing; a string or integer, or a flat list of
    them, is its ids; any other shape is a problem (never partly read)."""
    raw, problem = _lookup(args, selector)
    if problem:
        return [], problem
    items = list(raw) if isinstance(raw, (list, tuple)) else [] if raw is None else [raw]
    if any(type(item) not in (str, int) for item in items):
        kind = next(type(item).__name__ for item in items if type(item) not in (str, int))
        return [], f"{selector!r} holds a {kind}; it names a string id or a flat list of string ids"
    return list(dict.fromkeys(text for text in (str(item).strip() for item in items) if text)), ""


def _one_value(args: Mapping[str, Any], selector: str) -> tuple[str, str]:
    """A group's account or (another argument's) thread: one string or integer, or absent; a list, even of one
    value, is a problem, so one value is never inferred from a shape that can carry several."""
    raw, problem = _lookup(args, selector)
    if problem or raw is None or type(raw) in (str, int):
        return ("" if raw is None else str(raw).strip()), problem
    return "", f"{selector!r} holds a {type(raw).__name__}; it names one string id per call"


def touched_addresses(effect: ToolEffect, args: Any) -> tuple[tuple[TouchAddress, ...], str]:
    """The exact addresses this call reaches through its declared groups, and why they cannot be read (``""``
    when they can). Each conversation id takes the group's one account (default: the call's ``account_id``) and
    its thread: the same argument pairs each id with itself, another argument names one thread for all. Beside
    groups a supplied declared target, conversation or thread is read as strictly (a row without groups keeps its
    reading). A call whose addresses cannot be read is refused whole, never partly attributed
    (``place_guard.dispatch_refusal``)."""
    source = args if isinstance(args, Mapping) else {}
    for name in dict.fromkeys((*effect.targets, effect.conversation, effect.thread) if effect.touches else ()):
        if name and (problem := _address_values(source, name)[1]):
            return (), problem
    found: list[TouchAddress] = []
    for group in effect.touches:
        conversations, problem = _address_values(source, group.conversation)
        if problem:
            return (), problem
        if not conversations:
            continue
        same = group.thread == group.conversation
        thread, problem = ("", "") if same or not group.thread else _one_value(source, group.thread)
        account, account_problem = _one_value(source, group.account or "account_id")
        if problem or account_problem:
            return (), problem or account_problem
        pairs = thread_pairs(conversations, [thread] if thread else [], same_argument=same)[0]
        found.extend(TouchAddress(group.effect, conversation, account, paired, group.conversation, group.match)
                     for conversation, paired in pairs)
    return tuple(dict.fromkeys(found)), ""
