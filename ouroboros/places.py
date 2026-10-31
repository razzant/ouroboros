"""Host-owned places, atomic revisioned choices and typed wake matching.

The store is installation-local. Its first locked write imports legacy bindings
once; connectors never rewrite an existing place when their declarations change.
Working folders are addresses, not capability fences. Missing state is empty;
malformed state is a repair error and is never replaced with an empty store.
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
    def __init__(self, code: str, field: str = "place", message: str = "") -> None:
        self.code, self.field = code, field
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
    if any(not valid_wake_fact(w.fact) or w.author not in AUTHORS for w in place.wakes):
        raise PlaceError("place_wake_invalid", "wakes")
    if any(not n.text.strip() or n.author not in AUTHORS for n in place.notes):
        raise PlaceError("place_note_invalid", "notes")
    if any(not a.platform_actor_id for a in place.owner_accounts):
        raise PlaceError("place_account_invalid", "owner_accounts")
    return place


def _decode(value: Any) -> Place:
    try:
        data = dict(value)
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
            or not isinstance(value.get("migration", {}), dict)):
        raise PlaceError("places_invalid_state")
    for key, row in value["places"].items():
        if _decode(row).place_id != key:
            raise PlaceError("places_invalid_state")
    return value


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


def migrate_places(drive_root: Any) -> dict:
    """Atomically import each binding once; a failed import leaves all source bytes intact."""
    from ouroboros.places_migration import legacy_places
    path = places_path(drive_root)

    def migrate(current: dict) -> dict:
        current = _document(current or None)
        if current.get("migration", {}).get("presence_bindings_v1"):
            return current
        rows = dict(current["places"])
        imported, problems = legacy_places(Path(drive_root))
        known = {bid for row in rows.values() for bid in row.get("legacy_binding_ids", ())}
        for place in imported:
            if not known.intersection(place.legacy_binding_ids):
                rows.setdefault(place.place_id, asdict(place))
        return {**current, "places": rows, "revision": current["revision"] + 1,
                "migration": {"presence_bindings_v1": True, "at": utc_now_iso(), "problems": problems}}

    try:
        return update_json_locked(path, migrate, strict_existing_dict=True, reject_existing_empty_dict=True)
    except PlaceError:
        raise
    except (OSError, ValueError, TimeoutError) as exc:
        raise PlaceError("places_migration_failed", str(path), f"Place migration needs repair: {exc}") from exc


def load_places(drive_root: Any) -> list[Place]:
    current = _read(drive_root)
    if not current.get("migration", {}).get("presence_bindings_v1"):
        current = migrate_places(drive_root)
    return [_decode(value) for _key, value in sorted(current["places"].items())]


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
                  conversation_id: str = "", binding_id: str = "", thread_id: str = "") -> Place | None:
    candidates = []
    for place in places_of_connector(drive_root, connector):
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


def place_for_target(drive_root: Any, *, connector: str, targets: Mapping[str, list[str]],
                     conversation: str = "", thread: str = "") -> tuple[Place | None, bool]:
    """Resolve declared target values against stored account/conversation ids.

    Multiple destinations must be checked separately by the dispatch caller. An
    unresolved value is unknown when scoped places exist, never an active default.
    """
    places = places_of_connector(drive_root, connector)
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
                and ((room and room in conversations) or (not room and account and account in values)))
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
    return None, False


def wake_matches(place: Place, facts: Iterable[str]) -> tuple[WakeItem, ...]:
    present = set(facts)
    if FACT_FROM_BOT in present and not any(item.fact == FACT_FROM_BOT for item in place.wakes):
        return ()
    return tuple(item for item in place.wakes if item.fact in present or item.fact == FACT_ANY_MESSAGE)


def is_owner_account(place: Place, platform_actor_id: str, account_id: str = "") -> bool:
    return owner_account_matches(place.owner_accounts, platform_actor_id, account_id)


def owner_account_matches(owner_accounts: Iterable[OwnerAccount], platform_actor_id: str, account_id: str) -> bool:
    """Pure account-scoped owner match shared by actors and credential identities."""
    actor = str(platform_actor_id or "").strip()
    return bool(actor) and any(a.platform_actor_id == actor and (not a.account_id or a.account_id == account_id)
                               for a in owner_accounts)


def note_identity(place_id: str, index: int, note: NoteEntry) -> str:
    return note.note_id or hashlib.sha256(f"{place_id}:{index}:{note.text}".encode()).hexdigest()[:24]


def place_summary(place: Place) -> Mapping[str, Any]:
    return asdict(place)


def _write(drive_root: Any, place_id: str, change: Any, expected_revision: int | None = None) -> Place | None:
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
        raise PlaceError("place_revision_invalid", "expected_revision", "place_revision_invalid: expected_revision must be a whole number.")
    load_places(drive_root)  # migration precedes the first user edit, under the same file lock
    result = None

    def mutate(current: dict) -> dict:
        nonlocal result
        current = _document(current)
        old = _decode(current["places"][place_id]) if place_id in current["places"] else None
        if expected_revision is not None and (old.revision if old else 0) != expected_revision:
            raise PlaceError("place_revision_conflict", "revision", f"Current revision: {old.revision if old else 0}")
        result = change(old)
        if result == old:
            return current
        rows = dict(current["places"])
        if result is None:
            rows.pop(place_id, None)
        else:
            result = _validate(replace(result, revision=current["revision"] + 1))
            rows[place_id] = asdict(result)
        return {**current, "revision": current["revision"] + 1, "places": rows}

    update_json_locked(places_path(drive_root), mutate, strict_existing_dict=True, reject_existing_empty_dict=True)
    return result


def connect_place(drive_root: Any, *, connector: str, provider: str, label: str = "", account_id: str = "",
                  conversation_id: str = "", scope: str = "account", wakes: Iterable[str] = (),
                  contribution_id: str = "", place_id: str = "", state: str = STATE_ACTIVE) -> Place:
    # A repeated Connect is idempotent; a refreshed connector cannot replace choices.
    identity = (connector, provider, account_id, conversation_id, contribution_id)
    place_id = place_id or uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(identity)).hex
    candidate = _validate(Place(place_id=place_id, connector=connector, provider=provider, label=label or provider,
                               account_id=account_id, conversation_id=conversation_id, scope=scope,
                               contribution_id=contribution_id, state=state,
                               wakes=tuple(WakeItem(fact, AUTHOR_CONNECTOR) for fact in dict.fromkeys(wakes))))
    return _write(drive_root, place_id, lambda old: old or candidate)


def edit_place(drive_root: Any, place_id: str, action: str, *, author: str,
               expected_revision: int | None = None, **values: Any) -> Place | None:
    """One writer shared by owner HTTP and the place verb; author is a host fact."""
    if author not in {AUTHOR_OWNER, AUTHOR_OUROBOROS}:
        raise PlaceError("place_author_invalid")
    if action == "confirm_owner_account" and author != AUTHOR_OWNER:
        raise PlaceError("place_owner_confirmation_required", "owner_accounts",
                         "Confirm this account from the owner's door or the Places sheet.")

    return _write(drive_root, place_id, lambda old: _change_place(old, action, author, values), expected_revision)


def _change_place(old: Place | None, action: str, author: str, values: dict) -> Place | None:
    if action == "update":
        for name, args in values["changes"]:
            old = _change_place(old, name, author, args)
        return old
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
