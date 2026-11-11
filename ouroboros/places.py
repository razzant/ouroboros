"""Host-owned places, circles, atomic revisioned choices and typed wake matching.

The store is installation-local. Its first locked write imports legacy bindings
once; connectors never rewrite an existing place when their declarations change.
Working folders are addresses, not capability fences. Missing state is empty;
malformed state is a repair error and is never replaced with an empty store.

Circles live in the same document beside the place rows (an ordered ``circles``
list and ``place_choices`` by place id: its circle and its description), so one
read resolves a place with the words it inherits and one lock writes both. A
place row keeps its frozen keys: a body predating circles reads every place and
ignores the circles (``PlaceRecords``).
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ouroboros.utils import update_json_locked, utc_now_iso

PLACES_FILENAME = "places.json"
STATE_ACTIVE, STATE_SILENT, STATE_PAUSED = "active", "silent", "paused"
PLACE_STATES = (STATE_ACTIVE, STATE_SILENT, STATE_PAUSED)
FACT_DIRECT_MESSAGE = "direct_message"
FACT_GROUP_DM = "group_dm"
FACT_MENTION = "mention"
FACT_REPLY_TO_ME = "reply_to_me"
FACT_MY_THREAD = "my_thread"
FACT_NEW_ISSUE = "new_issue"
FACT_NEW_PULL_REQUEST = "new_pull_request"
FACT_REVIEW_REQUESTED = "review_requested"
FACT_ASSIGNED = "assigned"
FACT_ANY_MESSAGE = "any_message"
FACT_FROM_BOT = "from_bot"
FACT_LEGACY_ADDRESSED = "legacy_addressed"
WAKE_FACTS = (FACT_DIRECT_MESSAGE, FACT_GROUP_DM, FACT_MENTION, FACT_REPLY_TO_ME,
              FACT_MY_THREAD, FACT_NEW_ISSUE, FACT_NEW_PULL_REQUEST, FACT_REVIEW_REQUESTED,
              FACT_ASSIGNED, FACT_ANY_MESSAGE, FACT_FROM_BOT)
AUTHOR_CONNECTOR, AUTHOR_OWNER, AUTHOR_OUROBOROS, AUTHOR_MIGRATION = (
    "connector", "owner", "ouroboros", "migration")
AUTHORS = (AUTHOR_CONNECTOR, AUTHOR_OWNER, AUTHOR_OUROBOROS, AUTHOR_MIGRATION)
OWNER_VOICE = (AUTHOR_OWNER, AUTHOR_MIGRATION)  # words and acts that are my human's standing choices
CIRCLE_NAME_MAX, DESCRIPTION_MAX = 80, 240
SILENT_DISCLOSURE = ("Writes through this connector's declared tools and delivery queue are blocked. "
                     "Scripts, browser actions and unintegrated code may still write.")


def is_wake_signal(fact: Any) -> bool:
    return isinstance(fact, str) and re.fullmatch(r"signal\.[a-z0-9_-]{1,40}", fact) is not None


def valid_wake_fact(fact: Any) -> bool:
    """Stored signals remain readable when their connector is off or removed."""
    return isinstance(fact, str) and (fact in (*WAKE_FACTS, FACT_LEGACY_ADDRESSED) or is_wake_signal(fact))


def declared_wake_signals(connector: str) -> set[str]:
    """Only the authenticated connector's live publication can vouch for a signal."""
    from ouroboros.extension_registry_state import live_contribution_projection

    return {fact["id"] for row in live_contribution_projection()
            if row.get("provides") == "place" and row.get("skill") == connector
            for fact in row.get("facts", [])
            if isinstance(fact, dict) and is_wake_signal(fact.get("id"))}


class PlaceError(ValueError):
    def __init__(self, code: str, field: str = "place", message: str = "", *, place_id: str = "") -> None:
        self.code, self.field, self.place_id = code, field, place_id  # place_id: the place a conflict names
        super().__init__(message or f"{code}: {field}")


def validate_presence_identity(value: Any) -> dict:
    """One wire shape for the credential facts on status responses and events."""
    def entry(item):
        if (not isinstance(item, dict) or set(item) != {"id", "label", "kind"}
                or any(not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > 200
                       for key in ("id", "label"))
                or item["kind"] not in ("user", "bot")):
            raise ValueError("identity entries need id and label (1–200 characters) and kind user or bot")
        return dict(item)

    if not isinstance(value, dict) or set(value) - {"sends_as", "reads_as"}:
        raise ValueError("identity must be an object with only sends_as and reads_as")
    result = {}
    if "sends_as" in value:
        result["sends_as"] = entry(value["sends_as"])
    if "reads_as" in value:
        if not isinstance(value["reads_as"], list):
            raise ValueError("identity.reads_as must be a list")
        result["reads_as"] = [entry(item) for item in value["reads_as"]]
    return result


@dataclass(frozen=True)
class WakeItem:
    fact: str
    author: str


@dataclass(frozen=True)
class NoteEntry:
    text: str
    author: str
    at: str = ""
    note_id: str = ""


@dataclass(frozen=True)
class OwnerAccount:
    platform_actor_id: str
    label: str = ""
    confirmed_at: str = ""
    account_id: str = ""  # disambiguates people on connector-wide places


@dataclass(frozen=True)
class Place:
    place_id: str
    connector: str
    provider: str
    account_id: str = ""
    conversation_id: str = ""
    label: str = ""
    state: str = STATE_ACTIVE
    wakes: tuple[WakeItem, ...] = ()
    notes: tuple[NoteEntry, ...] = ()
    owner_accounts: tuple[OwnerAccount, ...] = ()
    working_folder: str = ""
    model: str = ""
    context_topics: tuple[str, ...] = ()
    legacy_binding_ids: tuple[str, ...] = ()
    revision: int = 0
    thread_id: str = ""
    destination: Mapping[str, str] | None = None
    scope: str = "account"
    contribution_id: str = ""
    connected_revision: int = 0  # internal connection observation, not part of the stored/public row


def places_path(drive_root: Any) -> Path:
    return Path(drive_root) / "state" / PLACES_FILENAME


def conversation_chat_id(conversation_key: str) -> int:
    digest = hashlib.sha256(f"presence-conversation\0{conversation_key}".encode()).digest()
    return (1 << 40) + (int.from_bytes(digest[:6], "big") & ((1 << 40) - 1))


def _validate(place: Place) -> Place:
    from ouroboros.contracts.skill_manifest import canonical_skill_name
    for key in ("place_id", "connector", "provider", "account_id", "conversation_id", "thread_id", "label", "working_folder", "model"):
        if not isinstance(getattr(place, key), str):
            raise PlaceError("place_invalid_string", key)
    if not place.place_id or not place.connector or canonical_skill_name(place.connector) != place.connector:
        raise PlaceError("place_identity_invalid")
    if not place.provider or place.state not in PLACE_STATES or place.scope not in {"account", "conversation"}:
        raise PlaceError("place_invalid")
    if type(place.revision) is not int or place.revision < 0:
        raise PlaceError("place_revision_invalid")
    if type(place.connected_revision) is not int or place.connected_revision < 0:
        raise PlaceError("place_revision_invalid", "connected_revision")
    if any(not valid_wake_fact(w.fact) or w.author not in AUTHORS for w in place.wakes):
        raise PlaceError("place_wake_invalid", "wakes")
    if any(not n.text.strip() or n.author not in AUTHORS for n in place.notes):
        raise PlaceError("place_note_invalid", "notes")
    if any(not a.platform_actor_id for a in place.owner_accounts):
        raise PlaceError("place_account_invalid", "owner_accounts")
    return place


def _decode(value: Any, connection: Any = None) -> Place:
    try:
        data = dict(value)
        # Older writers preserve unknown document metadata, including after a
        # disconnect/reconnect. Only a matching row revision proves continuity.
        revision = data.get("revision", 0)
        data["connected_revision"] = (connection["connected_revision"]
            if isinstance(connection, Mapping) and connection.get("written_revision") == revision else revision)
        for name, cls in (("wakes", WakeItem), ("notes", NoteEntry), ("owner_accounts", OwnerAccount)):
            data[name] = tuple(cls(**item) for item in data.get(name, ()))
        for name in ("context_topics", "legacy_binding_ids"):
            data[name] = tuple(data.get(name, ()))
        return _validate(Place(**data))
    except (TypeError, AttributeError, KeyError) as exc:
        raise PlaceError("places_invalid_state") from exc


def _document(value: Any) -> dict:
    if value is None:
        return {"schema_version": 1, "revision": 0, "places": {}, "migration": {}}
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or type(value.get("revision")) is not int or not isinstance(value.get("places"), dict)
            or not isinstance(value.get("migration", {}), dict)
            or not isinstance(value.get("connections", {}), dict)):
        raise PlaceError("places_invalid_state")
    for key, row in value["places"].items():
        if _decode(row, value.get("connections", {}).get(key)).place_id != key:
            raise PlaceError("places_invalid_state")
    circles, choices = value.get("circles", []), value.get("place_choices", {})
    if (not isinstance(circles, list) or not all(map(_valid_circle, circles))
            or len({row["circle_id"] for row in circles}) != len(circles)
            or not isinstance(choices, dict) or not all(map(_valid_choice, choices.values()))):
        raise PlaceError("places_invalid_state")
    return value


def _valid_circle(row: Any) -> bool:
    return (isinstance(row, dict) and type(row.get("revision")) is int and row["revision"] >= 0
            and all(isinstance(row.get(key), str) for key in ("circle_id", "name", "text", "text_author", "text_at"))
            and isinstance(row.get("model", ""), str)
            and bool(row["circle_id"].strip() and row["name"].strip()) and row["text_author"] in ("", *AUTHORS))


def _valid_choice(row: Any) -> bool:
    """A place's circle (who placed it, when) and description (who wrote it, when), for one connection."""
    return (isinstance(row, dict) and type(row.get("connected_revision")) is int and row["connected_revision"] >= 0
            and all(isinstance(row.get(key, ""), str) for key in (
                "circle_id", "placed_by", "placed_at", "description", "description_author", "description_at"))
            and (not row.get("circle_id") or row.get("placed_by") in AUTHORS)
            and (not row.get("description") or row.get("description_author") in AUTHORS))


def _read(drive_root: Any) -> dict:
    path = places_path(drive_root)
    if not path.exists():
        return _document(None)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw is None:
            raise PlaceError("places_invalid_state")
        return _document(raw)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise PlaceError("places_unreadable", str(path), f"Place records need repair: {path}: {exc}") from exc


def paused_places_sentence(drive_root: Any) -> str:
    """Name current pauses, including legacy imports in memory, without writing."""
    try:
        current = _read(drive_root)
        migration_pending = not current.get("migration", {}).get("presence_bindings_v1")
        current = _migrated_document(drive_root, current)
        rows = current["places"]
    except (PlaceError, OSError, ValueError):
        return "Paused places: unknown (place records could not be read)."
    labels = [row.get("label") or key for key, row in sorted(rows.items()) if row.get("state") == STATE_PAUSED]
    sentence = "Paused places (no writing there; tasks are not affected): " + ", ".join(labels) + "." if labels else ""
    if migration_pending and current.get("migration", {}).get("problems"):
        sentence += " Paused legacy places: unknown (some old connections could not be read)."
    return sentence.strip()


def _migrated_document(drive_root: Any, current: dict) -> dict:
    """Apply the legacy import in memory for both display and the durable migration."""
    from ouroboros.places_migration import legacy_places

    current = _document(current or None)
    if current.get("migration", {}).get("presence_bindings_v1"):
        return current
    rows = dict(current["places"])
    imported, problems = legacy_places(Path(drive_root))
    known = {bid for row in rows.values() for bid in row.get("legacy_binding_ids", ())}
    for place in imported:
        if not known.intersection(place.legacy_binding_ids):
            rows.setdefault(place.place_id, place_summary(place))
    return {**current, "places": rows, "revision": current["revision"] + 1,
            "migration": {"presence_bindings_v1": True, "at": utc_now_iso(), "problems": problems}}


def migrate_places(drive_root: Any) -> dict:
    """Atomically import each binding once; a failed import leaves all source bytes intact."""
    path = places_path(drive_root)
    try:
        return update_json_locked(path, lambda current: _migrated_document(drive_root, current),
                                  strict_existing_dict=True, reject_existing_empty_dict=True)
    except PlaceError:
        raise
    except (OSError, ValueError, TimeoutError) as exc:
        raise PlaceError("places_migration_failed", str(path), f"Place migration needs repair: {exc}") from exc


@dataclass(frozen=True)
class PlaceRecords:
    """One read of the place store: its places by id, its ordered circles and each place's current choices.

    A choice counts only for the connection it was made for (its ``connected_revision``): a place disconnected and
    connected again under the same id starts outside every circle and without a description, whatever an older
    body left in the document. Admission, FOCUS and its replay, Safety, destination facts, the own-clock map and a
    non-inline root's current guidance all take circle words from one such read (``circle_context``)."""
    revision: int
    places: Mapping[str, Place]
    circles: tuple[Mapping[str, Any], ...]
    choices: Mapping[str, Mapping[str, Any]]

    def circle(self, circle_id: str) -> Mapping[str, Any] | None:
        return next((row for row in self.circles if circle_id and row["circle_id"] == circle_id), None)

    def members(self, circle_id: str) -> list[Place]:
        return [place for key, place in self.places.items() if self.choices.get(key, {}).get("circle_id") == circle_id]

    def circle_context(self, place_id: str) -> dict[str, Any] | None:
        """The circle whose words this place inherits, with both origins: who wrote the words and who placed the place
        there. Its words are my human's standing letter here only when both are theirs; through a placement I made
        their words still restrict, but consent to nothing new. None outside any circle."""
        choice = self.choices.get(place_id) or {}
        circle = self.circle(choice.get("circle_id", ""))
        if circle is None:
            return None
        return {**{key: circle[key] for key in ("circle_id", "name", "text", "text_author", "text_at", "revision")},
                "model": circle.get("model", ""),
                "placed_by": choice["placed_by"], "placed_at": choice.get("placed_at", ""),
                "standing_delegation": bool(circle["text"]) and circle["text_author"] in OWNER_VOICE
                and choice["placed_by"] in OWNER_VOICE}

    def description(self, place_id: str) -> dict[str, str] | None:
        """The place's short purpose line: what it is for, never a rule or a permission."""
        choice = self.choices.get(place_id) or {}
        return ({"text": choice["description"], "author": choice.get("description_author", ""),
                 "at": choice.get("description_at", "")} if choice.get("description") else None)


def place_model_route(place: Place, circle: Mapping[str, Any] | None) -> tuple[str, str]:
    """The place's chosen model and scope; an empty value follows consciousness. Aliases stay stored choices."""
    if place.model:
        return place.model, "place"
    if circle and circle.get("model"):
        return circle["model"], "circle"
    return "", "consciousness"


def _records(document: dict) -> PlaceRecords:
    connections = document.get("connections", {})
    rows = {key: _decode(value, connections.get(key)) for key, value in sorted(document["places"].items())}
    choices = {key: row for key, row in document.get("place_choices", {}).items()
               if key in rows and row["connected_revision"] == rows[key].connected_revision}
    return PlaceRecords(document["revision"], rows, tuple(document.get("circles", ())), choices)


def place_records(drive_root: Any) -> PlaceRecords:
    """The places, circles and choices from one read (the legacy import runs first, as ``load_places``)."""
    current = _read(drive_root)
    if not current.get("migration", {}).get("presence_bindings_v1"):
        current = migrate_places(drive_root)
    return _records(current)


def load_places(drive_root: Any) -> list[Place]:
    return list(place_records(drive_root).places.values())


def migration_problems(drive_root: Any) -> list[dict]:
    load_places(drive_root)
    return list(_read(drive_root).get("migration", {}).get("problems", []))


def place_room_index(drive_root: Any, place_id: str, *, details: bool = False) -> list[dict]:
    from ouroboros.place_arrivals import room_index
    return room_index(drive_root, get_place(drive_root, place_id), details=details)


def get_place(drive_root: Any, place_id: str) -> Place:
    for place in load_places(drive_root):
        if place.place_id == place_id:
            return place
    raise PlaceError("place_not_found", "place_id")


def resolve_place(drive_root: Any, *, connector: str, provider: str = "", account_id: str = "",
                  conversation_id: str = "", binding_id: str = "", thread_id: str = "",
                  places: Iterable[Place] | None = None) -> Place | None:
    """``places``: an already read set (``place_records``), else the store is read now."""
    candidates = []
    for place in (places_of_connector(drive_root, connector) if places is None
                  else [place for place in places if place.connector == connector]):
        if binding_id and binding_id not in (place.place_id, *place.legacy_binding_ids):
            continue
        if provider and provider != place.provider:
            continue
        if account_id and place.account_id and account_id != place.account_id:
            continue
        if conversation_id and place.conversation_id and conversation_id != place.conversation_id:
            continue
        if place.thread_id and (provider or account_id or conversation_id or thread_id) and place.thread_id != thread_id:
            continue
        candidates.append(place)
    candidates.sort(key=lambda p: (bool(p.account_id), bool(p.conversation_id), bool(p.thread_id)), reverse=True)
    if not candidates:
        return None
    first = candidates[0]
    if len(candidates) > 1 and (first.account_id, first.conversation_id, first.thread_id) == (
            candidates[1].account_id, candidates[1].conversation_id, candidates[1].thread_id):
        raise PlaceError("place_ambiguous", "place_id")
    return first


def places_of_connector(drive_root: Any, connector: str) -> list[Place]:
    return [place for place in load_places(drive_root) if place.connector == connector]


def account_in_place_scope(place: Place, account_id: str) -> bool:
    """An account may be addressed by the place itself or its declared destination.

    IDs are exact connector-owned keys, not sender addresses or display labels.
    An absent/wildcard account keeps the existing connector-wide scope.
    """
    return (place.account_id in ("", "*", account_id)
            or bool(place.destination and place.destination.get("account_id", "") in ("", "*", account_id)))


def place_for_target(drive_root: Any, *, connector: str, targets: Mapping[str, list[str]],
                     conversation: str = "", thread: str = "", provider: str = "") -> tuple[Place | None, bool]:
    """Resolve declared target values against stored account/conversation ids.

    Multiple destinations must be checked separately by the dispatch caller. An
    unresolved value inside an account's place scopes stays unknown. An explicit
    account outside every applicable scope is an ordinary, unbound tool target.
    """
    places = places_of_connector(drive_root, connector)
    if provider:
        places = [place for place in places if place.provider == provider]
    return target_place(places, targets=targets, conversation=conversation, thread=thread)


def target_place(places: list[Place], *, targets: Mapping[str, list[str]],
                 conversation: str = "", thread: str = "") -> tuple[Place | None, bool]:
    """Match outbound targets, including declared destinations, in the supplied scopes.

    Dispatch loads the connector's current places; a retained outbox also uses
    this matcher to validate its particular bound place. Ingress stays separate.
    """
    if not places:
        return None, True
    values = {str(value) for key, group in targets.items() if key not in {thread, "thread_id", "account_id"}
              for value in group if str(value)}
    conversations = set(targets.get(conversation, [])) if conversation else values
    threads = set(targets.get(thread or "thread_id", []))
    accounts = set(targets.get("account_id", []))
    values |= accounts
    if not values:
        return None, False
    def matches(account, room, topic):
        return ((not accounts or account in {"", "*"} or account in accounts)
                and (not topic or topic in threads)
                and ((room and room in conversations)
                     or (not room and account and (account in values or (accounts and account == "*")))))
    exact = [p for p in places if matches(p.account_id, p.conversation_id, p.thread_id)
             or (p.destination and matches(p.destination.get("account_id", ""),
                                           p.destination.get("conversation_id", ""), p.destination.get("thread_id", "")))]
    if exact:
        # A caller passing several targets must not hide a blocked one behind an active one.
        exact.sort(key=lambda p: (bool(p.account_id), bool(p.conversation_id), bool(p.thread_id),
                                  p.state != STATE_ACTIVE), reverse=True)
        return exact[0], True
    broad = [p for p in places if not p.account_id and not p.conversation_id and not p.thread_id]
    if len(broad) == 1:
        return broad[0], True
    if (len(places) == 1 and not places[0].conversation_id and not places[0].thread_id
            and (not accounts or places[0].account_id in {"", "*"} or places[0].account_id in accounts)):
        return places[0], True
    if accounts and not accounts.intersection({"", "*"}) and not any(
            account_in_place_scope(place, account) for place in places for account in accounts):
        return None, True
    return None, False


def outbox_place(drive_root: Any, *, connector: str, provider: str, account_id: str,
                 conversation_id: str, thread_id: str = "") -> Place | None:
    """Current outbox scope; an opaque operation retains matching account holds."""
    if not conversation_id:
        restricted = next((p for p in places_of_connector(drive_root, connector)
                           if p.provider == provider and p.account_id in ("", "*", account_id)
                           and p.state in {STATE_SILENT, STATE_PAUSED}), None)
        if restricted is not None:
            return restricted
    return resolve_place(drive_root, connector=connector, provider=provider, account_id=account_id,
                         conversation_id=conversation_id, thread_id=thread_id)


def observed_facts(event: Any) -> tuple[str, ...]:
    """The facts attention reads on an event: the connector's own (an event with none is a legacy addressed
    message), plus ``from_bot`` when the connector names its author a bot."""
    facts = tuple(event.facts) if event.facts is not None else (FACT_LEGACY_ADDRESSED,)
    return (*facts, FACT_FROM_BOT) if event.actor.get("is_bot") is True else facts


def writes_as_owner(place: Place, identity: Mapping[str, Any] | None, account_id: str = "",
                    accounts: Iterable[OwnerAccount] | None = None) -> bool:
    """The place writes from my human's own account (``sends_as`` confirmed as theirs; ``accounts``: the ones that
    count here, ``counted_owner_accounts``): its posts are theirs."""
    sending = (identity or {}).get("sends_as") or {}
    return sending.get("kind") == "user" and is_owner_account(place, str(sending.get("id") or ""), account_id, accounts)


def own_account_event(place: Place, event: Any, accounts: Iterable[OwnerAccount] | None = None) -> bool:
    """Written by the account this place writes as (``identity.sends_as``): by me, through the connector or any other
    tool that acts as that account (my posts; a provider that does not name an edit's or merge's actor reports those
    without one, and its connector keeps them from reading as replies). It stays in the room and my next own wake and
    never wakes me. Not where that account is my human's own (their posts there are their words), and never for an
    actor the connector reports unauthenticated (a mail From proves nothing) or an event without an identity."""
    identity = getattr(event, "identity", None) or {}
    sender = str((identity.get("sends_as") or {}).get("id") or "").strip()
    actor = str(event.actor.get("platform_actor_id") or event.actor.get("id") or "").strip()
    return (bool(sender) and actor == sender and event.actor.get("authenticated") is not False
            and not writes_as_owner(place, identity, event.account_id, accounts))


def wake_matches(place: Place, facts: Iterable[str]) -> tuple[WakeItem, ...]:
    present = set(facts)
    if FACT_FROM_BOT in present and not any(item.fact == FACT_FROM_BOT for item in place.wakes):
        return ()
    return tuple(item for item in place.wakes if item.fact in present or item.fact == FACT_ANY_MESSAGE)


def is_owner_account(place: Place, platform_actor_id: str, account_id: str = "",
                     accounts: Iterable[OwnerAccount] | None = None) -> bool:
    """``accounts``: the ones that count in this place (``counted_owner_accounts``); None, its own rows alone."""
    return owner_account_matches(place.owner_accounts if accounts is None else accounts, platform_actor_id, account_id)


def owner_account_matches(owner_accounts: Iterable[OwnerAccount], platform_actor_id: str, account_id: str) -> bool:
    """Pure account-scoped owner match shared by actors and credential identities."""
    actor = str(platform_actor_id or "").strip()
    return bool(actor) and any(a.platform_actor_id == actor and (not a.account_id or a.account_id == account_id)
                               for a in owner_accounts)


def counted_owner_accounts(places: Iterable[Place]) -> dict[str, tuple[OwnerAccount, ...]]:
    """Each place's owner accounts that count there now: its own rows, then every account confirmed in another place
    of the same connector account (its connector and provider), scoped to the account that confirmation was for (its
    own ``account_id``, else that place's; neither: every account, as inside that place). One confirmation counts on
    the whole bot or app; a note may still say my human's words in one room are not orders, which is my reading,
    never a host gate. Derived at every read from the stored rows: nothing is copied into a place."""
    rows = list(places)
    return {place.place_id: place.owner_accounts + tuple(account for _other, account in confirmed_elsewhere(rows, place))
            for place in rows}


def confirmed_elsewhere(places: Iterable[Place], place: Place) -> list[tuple[Place, OwnerAccount]]:
    """``(other place, account)`` for each account another place of ``place``'s connector account confirms, scoped
    to the account it was confirmed for (``counted_owner_accounts``); the sheet names where each was confirmed."""
    return [(other, replace(account, account_id=account.account_id or other.account_id)) for other in places
            if other.place_id != place.place_id and (other.connector, other.provider) == (place.connector, place.provider)
            for account in other.owner_accounts]


def place_rows(drive_root: Any) -> dict[str, Place]:
    """Every place as stored now, by id: one read, never a migration (``load_places`` may write). An
    unreadable document raises ``PlaceError``: unknown, never "no place"."""
    current = _read(drive_root)
    connections = current.get("connections", {})
    return {key: _decode(value, connections.get(key)) for key, value in current["places"].items()}


class OwnerAccounts(dict):
    """One ``place_owner_accounts`` snapshot: place id -> the owner accounts that count there, and ``sent``, the
    provider identities of messages I posted as an account (``own_sends``), which are never my human's words."""

    def __init__(self, accounts: Mapping[str, tuple[OwnerAccount, ...]] | None = None, sent: Iterable = ()) -> None:
        super().__init__(accounts or {})
        self.sent = frozenset(sent)


def place_owner_accounts(drive_root: Any) -> OwnerAccounts:
    """Every place's counted owner accounts as stored now (``counted_owner_accounts`` over ``place_rows``) with my
    recorded sends (``own_sends.sent_identities``): one read of each for a pass. Unknown, never "no owner"."""
    from ouroboros.own_sends import sent_identities

    return OwnerAccounts(counted_owner_accounts(place_rows(drive_root).values()), sent_identities(drive_root))


def forwarded_message(message: Any) -> bool:
    """A connector's recorded forward carries another author's words, even when my human sent it: the account stays
    theirs (``owner_author``), the words never become theirs (``owner_words.owner_row``, ``place_confirmed_words``,
    ``dialogue_provenance.owner_speech``)."""
    return bool(_fields({"message": message}, "message").get("forward_origin"))


def owner_author(accounts: Mapping[str, Iterable[OwnerAccount]], place_id: str, actor: Any, account_id: str,
                 conversation_id: str = "", message: Any = None) -> bool:
    """A place message as my human's own words now: its actor is an account its place's connector account confirms
    (``accounts``, one ``place_owner_accounts`` snapshot), the connector does not report it unauthenticated, and it is
    not a message I posted as an account (its provider identity among ``accounts.sent``; never compared by its words,
    so my human repeating my words is still my human). Provenance of the words, never whom they are for; who sent a
    forward, never whose words it carries (``forwarded_message``)."""
    from ouroboros.own_sends import message_identity

    actor = actor if isinstance(actor, Mapping) else {}
    actor_id = str(actor.get("platform_actor_id") or actor.get("id") or "").strip()
    return (actor.get("authenticated") is not False
            and owner_account_matches(accounts.get(place_id, ()), actor_id, str(account_id or ""))
            and message_identity(conversation_id, message) not in getattr(accounts, "sent", frozenset()))


def owner_event_now(drive_root: Any, place_id: str, event: Any) -> bool:
    """``owner_author`` for one place event (a captured event, or the facts its record keeps) over the stores as they
    stand now, so a confirmation made after a turn began counts for what that turn reads later and a message I posted
    there never does; without readable places nobody is confirmed."""
    facts = event if isinstance(event, Mapping) else {
        key: getattr(event, key, None) for key in ("actor", "account_id", "conversation_id", "message")}
    try:
        accounts = place_owner_accounts(drive_root) if drive_root is not None else OwnerAccounts()
    except PlaceError:
        return False
    return owner_author(accounts, str(place_id or ""), facts.get("actor"), str(facts.get("account_id") or ""),
                        str(facts.get("conversation_id") or ""), facts.get("message"))


def _fields(value: Any, key: str) -> Mapping[str, Any]:
    found = value.get(key) if isinstance(value, Mapping) else None
    return found if isinstance(found, Mapping) else {}


def row_place_id(row: Mapping[str, Any]) -> str:
    """The place a stored room row came through: its provenance stamp (kept beside the text), else its own field."""
    return str(_fields(row, "presence_provenance").get("place_id") or row.get("place_id") or "")


def owner_confirmed_row(row: Mapping[str, Any], accounts: Mapping[str, Iterable[OwnerAccount]]) -> bool:
    """A place row as my human's own words now (``owner_author`` over one ``place_owner_accounts`` read per pass).

    The answer is the current confirmation, as the room's history shows it (``gateway.place_history``): confirming
    an account names its earlier rows too, and withdrawing a mistaken confirmation unnames them; what caused a work
    is the admission's own snapshot instead (``focus.actor.owner_confirmed``), with the same rule. Only facts a
    stored row keeps beside its text are read (``memory_inventory._META_FIELDS``). An imported earlier message
    (``presence_history``) is never confirmed. A place where I write from my human's own account (``sends_as``
    confirmed, CREATING_SKILLS "Who the connector writes as") confirms their rows as theirs, as admission does; the
    connector filters its own sends, and a message I posted through another tool acting as that account is known by
    its provider identity (``own_sends``), never by its words.
    """
    transport = _fields(row, "transport")
    return not isinstance(row.get("historical_import"), Mapping) and owner_author(
        accounts, row_place_id(row), _fields(transport, "actor"), str(transport.get("account_id") or ""),
        str(transport.get("conversation_id") or ""), _fields(transport, "message"))


def note_identity(place_id: str, index: int, note: NoteEntry) -> str:
    return note.note_id or hashlib.sha256(f"{place_id}:{index}:{note.text}".encode()).hexdigest()[:24]


def place_summary(place: Place) -> Mapping[str, Any]:
    """The public and persisted row stays readable by bodies predating bindings."""
    row = asdict(place)
    row.pop("connected_revision")
    return row


def _expected(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise PlaceError("place_revision_invalid", "expected_revision", "place_revision_invalid: expected_revision must be a whole number.")
    return value


def _transact(drive_root: Any, mutate: Any) -> None:
    """One locked read-modify-write of the place document: ``mutate(document)`` returns the next document, or None
    when nothing changes (nothing is written). The legacy import precedes the first edit, under the same lock file."""
    load_places(drive_root)
    update_json_locked(places_path(drive_root), lambda current: mutate(_document(current)),
                       strict_existing_dict=True, reject_existing_empty_dict=True)


def _with_choices(document: dict, choices: dict) -> dict:
    document = {**document, "place_choices": choices}
    if not choices:
        document.pop("place_choices")
    return document


def _write(drive_root: Any, place_id: str, change: Any, expected_revision: int | None = None,
           choose: Any = None) -> Place | None:
    """One place's change; ``choose(choice, document)`` changes its circle or description in the same transaction.
    A changed choice is a change of the place: its revision moves (its connection's written revision with it), so a
    sheet holding the older revision meets a conflict instead of overwriting."""
    _expected(expected_revision)
    result = None

    def mutate(current: dict) -> dict | None:
        nonlocal result
        connections = current.get("connections", {})
        old = (_decode(current["places"][place_id], connections.get(place_id))
               if place_id in current["places"] else None)
        if expected_revision is not None and (old.revision if old else 0) != expected_revision:
            raise PlaceError("place_revision_conflict", "revision", f"Current revision: {old.revision if old else 0}")
        result = change(old)
        choices = dict(current.get("place_choices", {}))
        stored = choices.get(place_id)
        before = ({key: value for key, value in stored.items() if key != "connected_revision"}
                  if old and stored and stored["connected_revision"] == old.connected_revision else {})
        after = {} if result is None else choose(before, current) if choose else before
        if result == old and after == before:
            return None
        revision = current["revision"] + 1
        rows = dict(current["places"])
        if result is None:
            rows.pop(place_id, None)
        else:
            result = _validate(replace(result, revision=revision,
                                       connected_revision=old.connected_revision if old else revision))
            rows[place_id] = place_summary(result)
        connections = {key: value for key, value in connections.items() if key in rows}
        choices.pop(place_id, None)
        if result is not None:
            connections[place_id] = {"connected_revision": result.connected_revision, "written_revision": result.revision}
            if after:
                choices[place_id] = {**after, "connected_revision": result.connected_revision}
        return _with_choices({**current, "revision": revision, "places": rows, "connections": connections}, choices)

    _transact(drive_root, mutate)
    return result


def connect_place(drive_root: Any, *, connector: str, provider: str, label: str = "", account_id: str = "",
                  conversation_id: str = "", scope: str = "account", wakes: Iterable[str] = (),
                  contribution_id: str = "", place_id: str = "", state: str = STATE_ACTIVE,
                  first: Mapping[str, Any] | None = None, author: str = AUTHOR_OWNER) -> Place:
    """A repeated Connect is idempotent; a refreshed connector cannot replace choices. ``first`` is what an Add says
    about a new place (``note``, ``circle_id``, ``description``), written by ``author`` in the connecting transaction;
    for a place connected already it is refused (``place_exists``), so a repeated Add never writes it twice."""
    identity = (connector, provider, account_id, conversation_id, contribution_id)
    place_id = place_id or uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(identity)).hex
    candidate = _validate(Place(place_id=place_id, connector=connector, provider=provider, label=label or provider,
                               account_id=account_id, conversation_id=conversation_id, scope=scope,
                               contribution_id=contribution_id, state=state,
                               wakes=tuple(WakeItem(fact, AUTHOR_CONNECTOR) for fact in dict.fromkeys(wakes))))
    first = {key: value for key, value in (first or {}).items() if value != ""}
    if set(first) - {"note", "circle_id", "description"}:
        raise PlaceError("place_connect_invalid", "first")
    if first:
        _author(author)
    if "note" in first:
        candidate = _change_place(candidate, "add_note", author, {"text": first["note"]})

    def change(old: Place | None) -> Place:
        if old is not None and first:
            raise PlaceError("place_exists", "place", "place_exists: this place is connected already; change it on its "
                             "sheet.", place_id=old.place_id)
        return old or candidate

    def choose(choice: dict, document: dict) -> dict:
        for action, key in (("set_circle", "circle_id"), ("set_description", "description")):
            if key in first:
                choice = _change_choice(choice, action, author, {key: first[key]}, document)
        return choice
    return _write(drive_root, place_id, change, choose=choose)


_CHOICE_ACTIONS = ("set_circle", "set_description")


def _author(author: str) -> None:
    if author not in {AUTHOR_OWNER, AUTHOR_OUROBOROS}:
        raise PlaceError("place_author_invalid")


def edit_place(drive_root: Any, place_id: str, action: str, *, author: str,
               expected_revision: int | None = None, **values: Any) -> Place | None:
    """One writer shared by owner HTTP and the place verb; author is a host fact. ``update`` applies several changes
    (``values["changes"]``, ``(action, values)`` pairs) in one transaction, the place's circle and description too."""
    _author(author)
    changes = values["changes"] if action == "update" else [(action, values)]
    if author != AUTHOR_OWNER and any(name == "confirm_owner_account" for name, _args in changes):
        raise PlaceError("place_owner_confirmation_required", "owner_accounts",
                         "Confirm this account from the owner's door or the place sheet.")

    def change(old: Place | None) -> Place | None:
        if old is None:
            raise PlaceError("place_not_found", "place_id")
        for name, args in changes:
            if name not in _CHOICE_ACTIONS:
                old = _change_place(old, name, author, args)
        return old

    def choose(choice: dict, document: dict) -> dict:
        for name, args in changes:
            if name in _CHOICE_ACTIONS:
                choice = _change_choice(choice, name, author, args, document)
        return choice
    return _write(drive_root, place_id, change, expected_revision, choose)


def _change_choice(choice: dict, action: str, author: str, values: Mapping[str, Any], document: Mapping[str, Any]) -> dict:
    """A place's circle (``circle_id``, "" for none) or its description ("" clears it). The same circle or line saved
    again keeps an owner's authorship: only my human's deliberate save makes an earlier one of mine theirs."""
    key, by, at = (("circle_id", "placed_by", "placed_at") if action == "set_circle"
                   else ("description", "description_author", "description_at"))
    value = values.get(key, "")
    if not isinstance(value, str):
        raise PlaceError("place_invalid_string", key)
    value = value.strip() if key == "circle_id" else " ".join(value.split())
    if key == "circle_id" and value and not any(row["circle_id"] == value for row in document.get("circles", ())):
        raise PlaceError("circle_not_found", "circle_id", "circle_not_found: no circle has this id; list the circles first.")
    if key == "description" and len(value) > DESCRIPTION_MAX:
        raise PlaceError("place_description_invalid", "description",
                         f"place_description_invalid: a description is one short line of at most {DESCRIPTION_MAX} characters.")
    if not value:
        return {name: item for name, item in choice.items() if name not in (key, by, at)}
    if choice.get(key) == value and choice.get(by) in (*OWNER_VOICE, author):
        return choice
    return {**choice, key: value, by: author, at: utc_now_iso()}


def _circle_index(circles: list, circle_id: Any) -> int:
    index = next((i for i, row in enumerate(circles) if circle_id and row["circle_id"] == circle_id), None)
    if index is None:
        raise PlaceError("circle_not_found", "circle_id", "circle_not_found: no circle has this id; list the circles first.")
    return index


def _circle_name(value: Any, circles: Iterable[Mapping[str, Any]], own_id: str = "") -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlaceError("circle_name_invalid", "name", "circle_name_invalid: a circle needs a name.")
    name = " ".join(value.split())
    if len(name) > CIRCLE_NAME_MAX:
        raise PlaceError("circle_name_invalid", "name", f"circle_name_invalid: a name is at most {CIRCLE_NAME_MAX} characters.")
    if any(row["circle_id"] != own_id and row["name"].casefold() == name.casefold() for row in circles):
        raise PlaceError("circle_name_taken", "name", "circle_name_taken: another circle has this name.")
    return name


def _circle_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise PlaceError("place_invalid_string", field)
    return value.strip()


def create_circle(drive_root: Any, *, name: str, text: str = "", model: str = "", author: str) -> dict:
    """A new circle at the end of the order; its words, when given, are ``author``'s."""
    _author(author)
    words, model, made = _circle_string(text, "text"), _circle_string(model, "model"), {}

    def mutate(current: dict) -> dict:
        circles, revision = list(current.get("circles", ())), current["revision"] + 1
        circle_id = uuid.uuid4().hex[:12]
        while any(row["circle_id"] == circle_id for row in circles):
            circle_id = uuid.uuid4().hex[:12]
        made.clear()
        made.update(circle_id=circle_id, name=_circle_name(name, circles), text=words, model=model,
                    text_author=author if words else "", text_at=utc_now_iso() if words else "", revision=revision)
        return {**current, "revision": revision, "circles": [*circles, dict(made)]}

    _transact(drive_root, mutate)
    return made


def edit_circle(drive_root: Any, circle_id: str, *, author: str, expected_revision: int | None = None,
                name: str | None = None, text: str | None = None, model: str | None = None) -> dict:
    """Change a circle's name, words or model. Name/model edits keep word authorship; new words are ``author``'s.
    The same words saved again stay my human's unless my human saves them (then they become theirs)."""
    _author(author)
    _expected(expected_revision)
    if name is None and text is None and model is None:
        raise PlaceError("circle_update_invalid", "circle", "circle_update_invalid: give a name, words or model.")
    words, saved = None if text is None else _circle_string(text, "text"), {}
    model = None if model is None else _circle_string(model, "model")

    def mutate(current: dict) -> dict | None:
        circles = list(current.get("circles", ()))
        index = _circle_index(circles, circle_id)
        old = circles[index]
        if expected_revision is not None and old["revision"] != expected_revision:
            raise PlaceError("circle_revision_conflict", "revision", f"Current revision: {old['revision']}")
        row = {**old, **({"name": _circle_name(name, circles, circle_id)} if name is not None else {})}
        if model is not None:
            row["model"] = model
        if words is not None and (words != old["text"] or (words and old["text_author"] not in (*OWNER_VOICE, author))):
            row.update(text=words, text_author=author if words else "", text_at=utc_now_iso() if words else "")
        saved.clear()
        saved.update(old)
        if row == old:
            return None
        circles[index] = {**row, "revision": current["revision"] + 1}
        saved.update(circles[index])
        return {**current, "revision": current["revision"] + 1, "circles": circles}

    _transact(drive_root, mutate)
    return saved


def delete_circle(drive_root: Any, circle_id: str, *, expected_revision: int | None = None) -> dict:
    """Remove a circle. Its places stay connected and keep their notes, descriptions and states; they leave the
    circle in the same transaction (each one's revision moves). Returns the removed circle, its words included, and
    the places it detached: there is no trash to restore from."""
    _expected(expected_revision)
    outcome: dict[str, Any] = {}

    def mutate(current: dict) -> dict:
        circles = list(current.get("circles", ()))
        index = _circle_index(circles, circle_id)
        if expected_revision is not None and circles[index]["revision"] != expected_revision:
            raise PlaceError("circle_revision_conflict", "revision", f"Current revision: {circles[index]['revision']}")
        removed, records, revision = circles.pop(index), _records(current), current["revision"] + 1
        rows, connections = dict(current["places"]), dict(current.get("connections", {}))
        choices, detached = dict(current.get("place_choices", {})), []
        for key, choice in list(choices.items()):
            if choice.get("circle_id") != circle_id:
                continue
            if key in records.choices:  # a member now, not a choice an older connection left
                place = replace(records.places[key], revision=revision)
                rows[key] = place_summary(place)
                connections[key] = {"connected_revision": place.connected_revision, "written_revision": revision}
                detached.append({"place_id": key, "label": place.label})
            rest = {name: value for name, value in choice.items() if name not in ("circle_id", "placed_by", "placed_at")}
            if set(rest) - {"connected_revision"}:
                choices[key] = rest
            else:
                choices.pop(key)
        outcome.clear()
        outcome.update(circle=removed, detached=detached)
        return _with_choices({**current, "revision": revision, "places": rows, "connections": connections,
                              "circles": circles}, choices)

    _transact(drive_root, mutate)
    return outcome


def _change_place(old: Place | None, action: str, author: str, values: dict) -> Place | None:
    if action == "confirm_owner_account" and author != AUTHOR_OWNER:
        raise PlaceError("place_owner_confirmation_required", "owner_accounts")
    if old is None:
        raise PlaceError("place_not_found", "place_id")
    if action == "disconnect":
        return None
    if action in {"set_state", "set_working_folder", "set_model"}:
        field = {"set_state": "state", "set_working_folder": "working_folder", "set_model": "model"}[action]
        value = values.get(field, "")
        if not isinstance(value, str):
            raise PlaceError("place_invalid_string", field)
        if field == "state" and value.strip() == STATE_ACTIVE and author != AUTHOR_OWNER:
            raise PlaceError("place_activation_owner_only", "state")
        return replace(old, **{field: value.strip()})
    if action == "set_wakes":
        facts = values.get("wakes")
        if not isinstance(facts, list) or any(not valid_wake_fact(f) for f in facts):
            raise PlaceError("place_wake_invalid", "wakes")
        added_signals = {f for f in facts if is_wake_signal(f)} - {w.fact for w in old.wakes}
        if added_signals and not added_signals <= declared_wake_signals(old.connector):
            raise PlaceError("place_wake_undeclared", "wakes", "place_wake_undeclared: this connector has not declared that signal; turn it on or choose a listed kind.")
        return replace(old, wakes=tuple(next((w for w in old.wakes if w.fact == fact), WakeItem(fact, author)) for fact in dict.fromkeys(facts)))
    if action in {"add_wake", "remove_wake"}:
        fact = values.get("fact", "")
        if not valid_wake_fact(fact):
            raise PlaceError("place_wake_invalid", "fact")
        if action == "add_wake" and is_wake_signal(fact) and fact not in declared_wake_signals(old.connector):
            raise PlaceError("place_wake_undeclared", "fact", "place_wake_undeclared: this connector has not declared that signal; turn it on or choose a listed kind.")
        items = tuple(w for w in old.wakes if w.fact != fact)
        if action == "add_wake":
            items += (WakeItem(fact, author),)
        return replace(old, wakes=items)
    if action == "add_note":
        text = values.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise PlaceError("place_note_invalid", "text")
        return replace(old, notes=old.notes + (NoteEntry(text, author, utc_now_iso(), uuid.uuid4().hex),))
    if action == "remove_note":
        index = values.get("index")
        if values.get("note_id"):
            index = next((i for i, note in enumerate(old.notes) if note_identity(old.place_id, i, note) == values["note_id"]), None)
        if type(index) is not int or not 0 <= index < len(old.notes):
            raise PlaceError("place_note_invalid", "index")
        return replace(old, notes=old.notes[:index] + old.notes[index + 1:])
    if action in {"confirm_owner_account", "revoke_owner_account"}:
        actor = values.get("platform_actor_id", "")
        account = values.get("account_id", old.account_id)
        if not isinstance(actor, str) or not actor.strip():
            raise PlaceError("place_account_invalid", "platform_actor_id/account_id")
        entries = tuple(a for a in old.owner_accounts if (a.platform_actor_id, a.account_id) != (actor, account))
        if action == "confirm_owner_account":
            entries += (OwnerAccount(actor, str(values.get("label", "")), utc_now_iso(), account),)
        return replace(old, owner_accounts=entries)
    raise PlaceError("place_action_invalid", "action")
