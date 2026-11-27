"""The owner's own Telegram bot (A) in groups: what core knows and enforces about it.

The bundled ``telegram`` bridge keeps A's token and is A's only ``getUpdates`` reader. The Official
``telegram-bot`` connector handles groups; for A it reads the bridge's feed and sends through the Host relay
(``gateway/host_telegram_feed.py``), never with A's token. Its manifest keeps one ``place`` declaration: while
the live connector registers the route ``owner_bot_status`` and its installed bytes are an Official release
(``official_connector``), core projects a second declaration ``owner_bot`` from that row (``with_owner_bot``), so
an older core, a fork or an edited copy sees only the connector's own bot (B).

An ``owner_bot`` place is exact: account A and one group conversation. It alone admits A's group events and
actions; a blank, wildcard or account-wide place, or the connector's ``place`` declaration, never does (``admits``,
``presence_admission``). A is the bot id of the saved ``TELEGRAM_BOT_TOKEN`` while the bridge may run; the id is the
token's public prefix, never its secret part.
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

log = logging.getLogger(__name__)

FEED_CONNECTOR = "telegram-bot"
NATIVE_SKILL = "telegram"
OWNER_BOT_CONTRIBUTION = "owner_bot"
BASE_CONTRIBUTION = "place"
OWNER_BOT_STATUS_ROUTE = "owner_bot_status"
CONNECT_EXACT = "exact"
_TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
_STATUS_MOUNT = f"/api/extensions/{FEED_CONNECTOR}/{OWNER_BOT_STATUS_ROUTE}"


_bridge: dict[str, Any] = {"bridge": None, "generation": 0}
_bridge_lock = threading.Lock()


def register_bridge(bridge: Any) -> Callable[[], None]:
    """Install the bundled bridge's feed while its poller runs in this process (``FeedBridge`` of
    ``skills/telegram/lib/telegram_group_feed.py``: ``handle``, ``connect_allowed``); the returned callable removes
    exactly this registration. The Host doors and the exact place writer ask it; without it they refuse."""
    with _bridge_lock:
        _bridge["generation"] += 1
        generation = _bridge["generation"]
        _bridge["bridge"] = bridge

    def unregister() -> None:
        with _bridge_lock:
            if _bridge["generation"] == generation:
                _bridge["bridge"] = None

    return unregister


def current_bridge() -> Any:
    with _bridge_lock:
        return _bridge["bridge"]


# One Telegram bot has one update stream. The bridge (``telegram``) and the connector's own bot (``telegram-bot``) are
# this process's two local readers; each is given a bot once, before it may contact Telegram with it: the bridge
# before its getMe and getUpdates (``claim_bot``); a connector offering ``owner_bot_status`` asks right before its
# deleteWebhook and getUpdates (``claim_connector_poll``, Host ``POST /telegram/feed/poller``) and its other reads
# claim nothing; an older connector, which cannot ask, is given its bot when a settings read hands it the token
# (``hand_settings``). A bot given to one is never given to the other until Ouroboros restarts: a reader's last
# request may still be out (a cancelled worker-thread long poll, a rotated or cleared token, Off), so no switch,
# save or timer moves it. Only bot ids are kept, never a token.
#
# The owner's own bot A is the bridge's in EVERY process, decided once from Settings (``owner_bridge_bot_id``): the
# saved, admitted ``TELEGRAM_BOT_TOKEN`` while the bridge may run. The connector is never given A, whichever version
# reads, whichever reads first and whatever a transient health record says — a duplicate token saved for the
# connector never takes the owner's bot, its private chat or its Panic. This settings decision is the one source of
# truth shared by reader arbitration, settings delivery, admission (``presence_admission``), destination
# (``place_guard``) and the identity door (``host_telegram_feed``); an in-memory holder is never the decider, because
# a carrier re-admits in a worker, and an extension child or companion reads its settings in another process, none
# of which has this process's holder map. The holder map only fences a live switch of a DISTINCT bot until restart,
# where the bridge runs (the server, which also computes a companion's environment).
_holders: dict[str, str] = {}
_holders_lock = threading.Lock()
_TRANSPORT_TOKEN = "TELEGRAM_PUBLIC_BOT_TOKEN"


def _connector_claim(root: Any, bot: str) -> str:
    """Give ``bot`` to the connector unless it is the owner's own (A, the bridge's in every process) or a reader holds
    it. A is decided from Settings, never from a health record or the in-memory holder, so every process agrees."""
    if owner_bridge_bot_id(root) == bot:
        return NATIVE_SKILL
    holder = bot_holder(bot)
    if holder:
        return holder
    with _holders_lock:
        return _holders.setdefault(bot, FEED_CONNECTOR)


def claim_bot(reader: str, bot: str) -> str:
    """Give ``bot`` to ``reader`` unless the other reader holds it; the holder after this (``reader`` when given)."""
    bot = bot_id_of(bot)
    if not bot:
        return reader
    with _holders_lock:
        return _holders.setdefault(bot, reader)


def bot_holder(bot: str) -> str:
    """The local reader this process gave ``bot`` (``telegram`` or ``telegram-bot``), else ``""``."""
    with _holders_lock:
        return _holders.get(bot_id_of(bot), "")


def connector_token_held(root: Any) -> bool | None:
    """Whether the connector's saved token names a bot that is, or would be, the bridge's here (so the connector is
    not given it); ``None`` when Settings cannot be read. A typed fact for the doors that can show it (Host identity,
    the bridge's ``group_feed`` status); an older connector's own Widget cannot."""
    try:
        from ouroboros.config import load_settings
        bot = bot_id_of(load_settings(strict=True).get(_TRANSPORT_TOKEN))
    except Exception:
        return None
    return bool(bot) and owner_bridge_bot_id(root) == bot


def hand_settings(skill: str, values: Mapping[str, Any], root: Any = None, *, claim: bool = True) -> dict:
    """What an authorized settings read hands ``skill`` (``root``: the reader's data root), by one rule in every
    process that reads settings (the server, a worker, an extension child, a companion's environment). A token held
    back reads as not saved: its reader never polls nor deletes its webhook with it.

    The owner's own bot A is never handed to the connector, whichever version reads: the connector reaches A only
    through the relay. When Settings cannot be read, whose bot it is stays unknown and the token waits for a readable
    answer. A distinct bot: a connector that asks before polling (live ``owner_bot_status``) reads it freely; an older
    one, which polls whatever it reads, is given it by the read unless the bridge here already holds it. ``claim``
    false (a passive catalog/discovery read) filters the same way but records no holder: discovery never moves a
    bot's reader."""
    values = dict(values)
    bot = bot_id_of(values.get(_TRANSPORT_TOKEN)) if skill == FEED_CONNECTOR else ""
    if not bot:
        return values
    reader_root = _root(root)
    try:
        own = owner_bridge_bot_id(reader_root, strict=True) == bot
    except Exception:
        own = True  # unknown, never a "not the owner's"
    if own:
        held = True
    elif _live_status_route():
        held = False
    elif claim:
        held = _connector_claim(reader_root, bot) != FEED_CONNECTOR
    else:
        held = bot_holder(bot) == NATIVE_SKILL
    if held:
        values.pop(_TRANSPORT_TOKEN, None)
    return values


def _root(root: Any) -> Path:
    if root is not None:
        return Path(root)
    from ouroboros.config import DATA_DIR
    return Path(DATA_DIR)


def claim_connector_poll(root: Any, bot_id: str) -> tuple[int, dict]:
    """``POST /telegram/feed/poller``: the connector is about to deleteWebhook and getUpdates with bot ``bot_id``,
    which must be the bot of its saved, granted ``TELEGRAM_PUBLIC_BOT_TOKEN``. ``owned`` false: the bridge reads it
    here, so the connector contacts Telegram with it not."""
    bot = bot_id_of(bot_id)
    if not bot or bot != str(bot_id):
        return 400, {"ok": False, "code": "bot_id_invalid"}
    try:
        from ouroboros.config import load_settings
        from ouroboros.skill_loader import find_skill, grant_status_for_skill
        saved = bot_id_of(load_settings(strict=True).get(_TRANSPORT_TOKEN))
        status = grant_status_for_skill(Path(root), find_skill(Path(root), FEED_CONNECTOR))
        own = owner_bridge_bot_id(root, strict=True)  # A, read as strictly as the rest: unknown is never "not A"
    except Exception:
        return 503, {"ok": False, "code": "settings_unknown"}
    readable = (_TRANSPORT_TOKEN in (status.get("granted_keys") or [])
                or _TRANSPORT_TOKEN not in (status.get("requested_keys") or []))
    if saved != bot or not readable:
        return 409, {"ok": False, "code": "account_changed"}
    holder = NATIVE_SKILL if own == bot else _connector_claim(root, bot)
    return 200, {"ok": True, "owned": holder == FEED_CONNECTOR, **({"holder": holder} if holder != FEED_CONNECTOR
                                                                     else {})}


def bot_id_of(token: Any) -> str:
    """The numeric bot id that opens a token (``<id>:<secret>``), or ``""``."""
    head = str(token or "").split(":", 1)[0].strip()
    return head if head.isdigit() and not head.startswith("0") else ""


def owner_bridge_bot_id(root: Any, *, settings: Mapping[str, Any] | None = None, strict: bool = False) -> str:
    """A's id while the bundled bridge may run here (installed, reviewed, enabled and trusted) with a saved token;
    ``""`` otherwise, also when either cannot be read unless ``strict`` (then the read error is raised).
    ``settings``: an already read document."""
    try:
        if settings is None:
            from ouroboros.config import load_settings
            settings = load_settings(strict=strict)
        bot = bot_id_of(settings.get(_TOKEN_KEY))
        if not bot:
            return ""
        from ouroboros.deployment_extensions import admission_refusal
        from ouroboros.skill_loader import find_skill
        return bot if not admission_refusal(find_skill(Path(root), NATIVE_SKILL)) else ""
    except Exception:
        if strict:
            raise
        log.debug("The owner bridge's bot could not be read", exc_info=True)
        return ""


def owner_bot_of(root: Any, connector: str) -> str:
    """The ``owner_bot`` fence a place lookup of ``connector`` passes (``places._owner_bot_eligible``): A's id for
    ``telegram-bot`` while the bridge may run here, else ``""``; other connectors read nothing."""
    return owner_bridge_bot_id(root) if connector == FEED_CONNECTOR else ""


def _live_status_route() -> bool:
    from ouroboros.extension_registry_state import list_routes
    spec = list_routes().get(_STATUS_MOUNT) or {}
    return "GET" in tuple(spec.get("methods") or ())


def official_connector(root: Any, *, content_hash: str | None = None) -> bool:
    """Whether the installed ``telegram-bot`` is exactly an Official OuroborosHub release now: the one answer the
    relay doors and the ``owner_bot`` projection read, from local records only. Its install record
    (``skill_install_record``) names the stock Official source (the shipped default catalog's repository and branch,
    its top folder, the Official part) and that release's digest, which the loader's hash of the bytes installed now
    must equal; a door also passes the hash its authenticated token was issued for (``content_hash``), so both speak
    of one package. Another hub, branch or folder, a same-name fork, a ClawHub copy, an execution domain's trusted
    package without such a record, a local edit or Repair, and an unreadable record read False: unknown is never
    Official."""
    try:
        from ouroboros.config import SETTINGS_DEFAULTS
        from ouroboros.deployment_extensions import admission_refusal
        from ouroboros.skill_install_record import read_install_record
        from ouroboros.skill_loader import find_skill
        from ouroboros.skill_sources import repo_from_catalog_url, source_view
        loaded = find_skill(Path(root), FEED_CONNECTOR)
        if loaded is None or admission_refusal(loaded):
            return False
        digest = str(getattr(loaded, "content_hash", "") or "")
        if not digest or (content_hash is not None and str(content_hash) != digest):
            return False
        record = read_install_record(Path(root), FEED_CONNECTOR) or {}
        source = record.get("source") or {}
        repo, branch = repo_from_catalog_url(SETTINGS_DEFAULTS["OUROBOROS_HUB_CATALOG_URL"])
        return bool(record.get("official") is True
                    and str((record.get("release") or {}).get("digest") or "") == digest
                    and source.get("kind") == "hub" and source.get("part") == "official"
                    and repo and source.get("url") == repo and source_view(source) == (branch, ""))
    except Exception:
        log.debug("The connector's Official release could not be read", exc_info=True)
        return False


def owner_bot_offered(root: Any) -> bool:
    """Whether new core would project ``owner_bot`` now (``with_owner_bot``), read from the live registry."""
    try:
        from ouroboros.extension_contributions import live_place_declarations
        return any(row.get("skill") == FEED_CONNECTOR and row.get("id") == OWNER_BOT_CONTRIBUTION
                   for row in with_owner_bot(live_place_declarations(), root))
    except Exception:
        log.debug("owner_bot availability unreadable", exc_info=True)
        return False


def with_owner_bot(rows: Iterable[Mapping[str, Any]], root: Any) -> list[dict]:
    """``rows`` (published declarations) plus the projected ``owner_bot`` row while the live connector offers it:
    its published ``place`` row, its live ``owner_bot_status`` route, its Official bytes (``official_connector``) and
    the bridge's bot A. The projection copies that row (label, icon, facts, wakes, tools) and changes only what makes
    it A's: id, exact scope and source. A row of that id the connector published itself is never taken for it: only
    this projection, under these conditions, offers your bot's groups."""
    rows = [dict(row) for row in rows
            if not (row.get("skill") == FEED_CONNECTOR and row.get("id") == OWNER_BOT_CONTRIBUTION)]
    base = next((row for row in rows if row.get("skill") == FEED_CONNECTOR and row.get("provides") == "place"
                 and row.get("id") == BASE_CONTRIBUTION), None)
    try:
        if (base is None or not _live_status_route() or not owner_bridge_bot_id(root)
                or not official_connector(root)):
            return rows
    except Exception:
        log.debug("owner_bot projection unavailable", exc_info=True)
        return rows
    row = copy.deepcopy(base)
    refresh = (base.get("source") or {}).get("refresh_s", 15)
    row.update(id=OWNER_BOT_CONTRIBUTION, key=f"{FEED_CONNECTOR}:{OWNER_BOT_CONTRIBUTION}", scope="conversation",
               connect_mode=CONNECT_EXACT, projected_from=BASE_CONTRIBUTION,
               source={"url": _STATUS_MOUNT, "refresh_s": refresh})
    # The copied "Connector on" speaks for the connector; A's readiness is the bridge's own poller, which registers in
    # the server process only. There a stopped or parked bridge is never painted ready; another process cannot know
    # and says nothing more than the connector's readiness (the row's live status names the bridge's state).
    from ouroboros.extension_companion import is_server_process
    if is_server_process() and current_bridge() is None:
        row["readiness"] = {"text": "Bridge not running", "tone": "warn"}
    return [*rows, row]


def is_owner_bot_place(place: Any) -> bool:
    return (getattr(place, "connector", "") == FEED_CONNECTOR
            and getattr(place, "contribution_id", "") == OWNER_BOT_CONTRIBUTION)


def declaring_place(place: Any) -> Any:
    """The place whose published declaration speaks for ``place``'s facts: an ``owner_bot`` place borrows its
    connector's ``place`` row, which its projection copies; every other place speaks for itself."""
    if not is_owner_bot_place(place):
        return place
    from dataclasses import replace
    return replace(place, contribution_id=BASE_CONTRIBUTION)


def exact_owner_bot_identity(place: Any, account_id: str, conversation_id: str, thread_id: str = "") -> bool:
    """Whether an ``owner_bot`` place is exactly this account and conversation, and holds this topic: a chat place
    holds all its topics, a topic place only its own."""
    account, room = str(getattr(place, "account_id", "") or ""), str(getattr(place, "conversation_id", "") or "")
    topic = str(getattr(place, "thread_id", "") or "")
    return (is_owner_bot_place(place) and bool(account) and account not in {"*"} and account == str(account_id)
            and bool(room) and room == str(conversation_id) and (not topic or topic == str(thread_id or "")))


def admits(place: Any, *, account_id: str, conversation_id: str, thread_id: str = "", owner_bot: str = "",
           named: bool = False) -> bool:
    """The one rule for A at the Host: an ``owner_bot`` place admits only its exact account, conversation and topic,
    and only when the caller named it (``named``); an event of the bridge's bot ``owner_bot`` needs such a place.
    Every other place keeps the existing rules (True here)."""
    if is_owner_bot_place(place):
        return named and exact_owner_bot_identity(place, account_id, conversation_id, thread_id)
    return not (owner_bot and getattr(place, "connector", "") == FEED_CONNECTOR and str(account_id) == owner_bot)


def connected_owner_bot_place(places: Iterable[Any], *, account_id: str, conversation_id: str,
                              thread_id: str = "") -> Any:
    """The exact ``owner_bot`` place holding this conversation (and topic), the topic's own place first."""
    found = [place for place in places if exact_owner_bot_identity(place, account_id, conversation_id, thread_id)]
    found.sort(key=lambda place: bool(getattr(place, "thread_id", "")), reverse=True)
    return found[0] if found else None


def owner_bot_binding(drive_root: Any, connector: str, target: Mapping[str, str],
                      binding: Mapping[str, Any] | None, version: int, *, with_place: bool = False) -> dict[str, Any]:
    """``/presence/place?operation=bind|check&contribution_id=owner_bot``: capture or recheck the exact ``owner_bot``
    place of ``target`` (provider, account, group, topic), with no broader, wildcard or B fallback. Same answer shape
    as ``place_guard.outbox_binding_facts``; ``allowed`` only while that very connection is current and Active.
    ``with_place`` adds the loaded ``Place`` under ``_place`` for an in-process caller (the relay), never serialized."""
    from ouroboros import places
    if connector != FEED_CONNECTOR or not target.get("account_id") or not target.get("conversation_id"):
        raise ValueError("An owner_bot binding needs telegram-bot, its account_id and the group's conversation_id.")
    if binding is not None:
        if (not isinstance(binding, Mapping) or binding.get("version") != version or binding.get("connector") != connector
                or any(binding.get(key) != value for key, value in target.items())
                or not isinstance(binding.get("place_id"), str) or not binding["place_id"]
                or type(binding.get("connected_revision")) is not int or binding["connected_revision"] < 0):
            raise ValueError("The retained outbox binding does not match this connector and destination.")
        try:
            place = places.get_place(drive_root, binding["place_id"])
        except places.PlaceError as exc:
            if exc.code != "place_not_found":
                raise
            place = None
        if (place is None or place.connected_revision != binding["connected_revision"]
                or place.provider != target.get("provider")):
            return {"allowed": False, "binding": dict(binding), "state": "disconnected", "code": "place_disconnected",
                    "label": getattr(place, "label", ""), "place_id": binding["place_id"],
                    **({"_place": place} if with_place else {})}
        if not exact_owner_bot_identity(place, target["account_id"], target["conversation_id"], target.get("thread_id", "")):
            return {"allowed": False, "binding": dict(binding), "state": "changed", "code": "place_scope_changed",
                    "label": place.label, "place_id": place.place_id, **({"_place": place} if with_place else {})}
        captured = dict(binding)
    else:
        place = connected_owner_bot_place(
            (p for p in places.places_of_connector(drive_root, connector) if p.provider == target.get("provider")),
            account_id=target["account_id"], conversation_id=target["conversation_id"],
            thread_id=target.get("thread_id", ""))
        if place is None:
            return {"allowed": False, "binding": None, "state": "unknown", "code": "place_not_connected",
                    "label": "", "place_id": ""}
        captured = {"version": version, "connector": connector, **target, "place_id": place.place_id,
                    "connected_revision": place.connected_revision}
    return {"allowed": place.state == places.STATE_ACTIVE, "binding": captured, "state": place.state,
            "code": f"place_{place.state}", "label": place.label, "place_id": place.place_id,
            **({"_place": place} if with_place else {})}


__all__ = [
    "BASE_CONTRIBUTION", "CONNECT_EXACT", "FEED_CONNECTOR", "NATIVE_SKILL", "OWNER_BOT_CONTRIBUTION",
    "OWNER_BOT_STATUS_ROUTE", "admits", "bot_holder", "bot_id_of", "claim_bot", "claim_connector_poll", "connector_token_held",
    "connected_owner_bot_place",
    "current_bridge", "declaring_place", "hand_settings",
    "exact_owner_bot_identity", "is_owner_bot_place", "official_connector", "owner_bot_binding", "owner_bot_offered",
    "owner_bridge_bot_id", "register_bridge", "with_owner_bot",
]
