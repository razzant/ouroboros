"""The owner's own bot (A) in groups: what this bridge hands the Official ``telegram-bot`` connector, and how.

This bridge stays A's only ``getUpdates`` reader and keeps its token. The owner turns groups on for the bot they
see (``POST group_feed`` with ``expected_account_id``) once this bridge is bound to them; the consent names that
bot, so another token reads Off, and Off withdraws only the consent of the bot the owner saw. Under the consent and
bound owner read once for each batch the poller receives (``intake``; one changed meanwhile keeps nothing of it), a
group or supergroup update (message, edit, the bot's membership, reactions) is spooled before the owner authorizer
sees it: whole when an exact ``owner_bot`` place holds its chat (and topic), else only what the Connect chooser
needs (a sighting: chat id, type, title, membership, migration; each group's newest one waits). Private chats,
callback queries and channels never enter; without consent or a bound owner nothing is kept (as before). A place
store that cannot be read now is no disconnect: the last view read stands. The spool is durable until the
connector's offset confirms it; a blocked write is a counted gap, not a delay to owner control. Membership
(``groups``) starts again each consent period. Message topics and file offers outlive the consumer offset;
file offers also outlive age while their exact connection stands. Pending feed items keep their bookkeeping.

The connector reads the spool and sends through the Host doors (``ouroboros/gateway/host_telegram_feed.py``),
which call ``FeedBridge.handle``. A read or a call happens only for the bot confirmed now, with its consent and a
bound owner, read off the event loop and checked again right before anything is confirmed, handed or sent; a long
poll keeps the consent record it was let in under (an Off or a new consent meanwhile confirms and hands nothing) and
its idle waits read no disk; ``identity`` with ``groups=1`` adds the groups the bot is in this period
(``group_ids``). A provider call is one request through this bridge's own client
(``TelegramClient.raw_request``, pinned to the token generation it was built for): only listed methods and
arguments, for the exact connected place, revision and topic the caller names (a topic send names that topic; an
action on a message, its own topic; a member's restriction or ban, a chat-wide place), Active for anything but a
file read, files by path inside the connector's own state folder, checked again right before the request.
Telegram's answer comes back as it was; nothing sent is a refusal, an unknown outcome stays unknown, and no answer
carries the token.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import pathlib
import re
import secrets
import sqlite3
import stat
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional

from .telegram_group_events import GROUP_TYPES, REACTION_KINDS, coalesce_sightings, event_key, member_status, record_membership
from .telegram_group_events import group_chat as group_chat
from .telegram_group_events import sighting, topic_of

ALLOWED_UPDATES = ["message", "edited_message", "callback_query", "my_chat_member", "message_reaction",
                   "message_reaction_count"]  # what this bridge reads, and what groups add; callbacks kept
_PROVIDER = "telegram"
_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # what Telegram's cloud serves a bot: the connector's limit, not this bridge's
_UPLOAD_LIMIT = 50 * 1024 * 1024
_POLL_STEP = 0.5
_MAX_POLL = 25
_PUBLISH_WAIT = 0.25  # a busy spool is a gap, never a wait for the owner's next update
_FILE_PATH_TTL = 3600.0  # Telegram keeps a getFile path at least an hour
_SEEN_DAYS = 7  # how long a confirmed item's key still turns its replay away, and bookkeeping outlives its item
_EXPIRE_EVERY = 3600.0
_UNREAD, _NOW = object(), object()  # no place view read yet; ``publish`` reading its consent itself
_USERNAME = re.compile(r"[A-Za-z0-9_]{5,64}")
_CHAT = re.compile(r"-[1-9][0-9]*")
_BOT = re.compile(r"[1-9][0-9]*")
_ARGS: Dict[str, frozenset] = {
    "sendMessage": frozenset({"chat_id", "text", "parse_mode", "message_thread_id", "reply_parameters"}),
    "editMessageText": frozenset({"chat_id", "message_id", "text", "parse_mode", "message_thread_id"}),
    "setMessageReaction": frozenset({"chat_id", "message_id", "reaction", "is_big"}),
    "sendChatAction": frozenset({"chat_id", "action", "message_thread_id"}),
    "deleteMessage": frozenset({"chat_id", "message_id"}),
    "restrictChatMember": frozenset({"chat_id", "user_id", "permissions", "until_date",
                                     "use_independent_chat_permissions"}),
    "banChatMember": frozenset({"chat_id", "user_id", "until_date", "revoke_messages"}),
    "unbanChatMember": frozenset({"chat_id", "user_id", "only_if_banned"}),
    "getFile": frozenset({"file_id"}),
}
_UPLOADS = {"sendPhoto": "photo", "sendDocument": "document", "sendVideo": "video", "sendAudio": "audio",
            "sendVoice": "voice", "sendAnimation": "animation", "sendMediaGroup": "media"}
_UPLOAD_ARGS = frozenset({"chat_id", "caption", "parse_mode", "message_thread_id", "reply_parameters"})
_TOPIC_SENDS = frozenset({"sendMessage", "sendChatAction", *_UPLOADS})  # a topic's speech names its topic
_SENT_MESSAGES = frozenset({"sendMessage", *_UPLOADS})  # their answer is the message (an album: its messages)
_MESSAGE_TARGETED = frozenset({"editMessageText", "deleteMessage", "setMessageReaction"})  # scope: the message's topic
_CHAT_WIDE = frozenset({"restrictChatMember", "banChatMember", "unbanChatMember"})  # scope: the whole group
_MEDIA_KEYS = frozenset({"type", "media", "caption", "parse_mode"})
_STORE = "group_feed.sqlite3"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS feed(
    feed_id INTEGER PRIMARY KEY AUTOINCREMENT, account_id TEXT NOT NULL, dedupe TEXT NOT NULL, body TEXT NOT NULL,
    received_at REAL NOT NULL, UNIQUE(account_id, dedupe));
CREATE TABLE IF NOT EXISTS seen(
    account_id TEXT NOT NULL, dedupe TEXT NOT NULL, seen_at REAL NOT NULL, PRIMARY KEY(account_id, dedupe));
CREATE INDEX IF NOT EXISTS seen_age ON seen(seen_at);
CREATE TABLE IF NOT EXISTS offers(
    account_id TEXT NOT NULL, place_id TEXT NOT NULL, connected_revision INTEGER NOT NULL, thread_id TEXT NOT NULL,
    file_id TEXT NOT NULL, offered_at REAL NOT NULL,
    PRIMARY KEY(account_id, place_id, connected_revision, thread_id, file_id));
CREATE INDEX IF NOT EXISTS offers_age ON offers(offered_at);
CREATE TABLE IF NOT EXISTS groups(
    account_id TEXT NOT NULL, chat_id TEXT NOT NULL, chat_type TEXT NOT NULL, membership TEXT NOT NULL,
    update_id INTEGER NOT NULL, PRIMARY KEY(account_id, chat_id));
CREATE TABLE IF NOT EXISTS topics(
    account_id TEXT NOT NULL, chat_id TEXT NOT NULL, message_id INTEGER NOT NULL, thread_id TEXT NOT NULL,
    seen_at REAL NOT NULL, PRIMARY KEY(account_id, chat_id, message_id));
CREATE INDEX IF NOT EXISTS topics_age ON topics(seen_at);
"""
_SCHEMA_OBJECTS = frozenset({"feed", "seen", "seen_age", "offers", "offers_age", "groups", "topics", "topics_age"})
_TOPIC_RECORD = """
INSERT INTO topics VALUES(?,?,?,?,?) ON CONFLICT(account_id, chat_id, message_id) DO UPDATE SET
    thread_id=excluded.thread_id, seen_at=excluded.seen_at"""
_OFFER_RECORD = """
INSERT INTO offers VALUES(?,?,?,?,?,?) ON CONFLICT(account_id, place_id, connected_revision, thread_id, file_id)
    DO UPDATE SET offered_at=excluded.offered_at"""


class FeedRefusal(Exception):
    """Nothing was sent: the relay refuses before any provider call."""

    def __init__(self, code: str, description: str, retry_after: int = 0):
        super().__init__(description)
        self.code, self.description, self.retry_after = code, description, retry_after


def _file_ids(value: Any, found: set) -> set:
    if isinstance(value, Mapping):
        if isinstance(value.get("file_id"), str) and value["file_id"]:
            found.add(value["file_id"])
        for item in value.values():
            _file_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _file_ids(item, found)
    return found


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_precise() -> str:
    """This moment in UTC to the microsecond: what tells one consent write from the next, a quick Off and On of one
    bot included (each write also waits for its own sync)."""
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1_000_000):06d}Z"


class FeedBridge:
    """One bridge's feed: consent, spool and relay. ``client_factory()`` builds this bridge's client for the token in
    force; ``verified()`` is ``(bot_id, username)`` once getMe confirmed that token, else None; ``owner_bound()``
    whether this bridge is bound to its owner."""

    def __init__(self, state_dir: pathlib.Path, data_dir: pathlib.Path, *, client_factory: Callable[[], Any],
                 verified: Callable[[], Optional[tuple[str, str]]], owner_bound: Callable[[], bool],
                 log: Callable[..., Any] = lambda *_args, **_kw: None, capable: Optional[Callable[[], bool]] = None):
        self.state_dir, self.data_dir = pathlib.Path(state_dir), pathlib.Path(data_dir)
        self.client_factory, self.verified, self.owner_bound, self.log = client_factory, verified, owner_bound, log
        self.capable = capable or self._offered
        self._lock = threading.Lock()
        self._gaps = {"count": 0, "at": "", "reason": ""}
        self._schema_ready = False
        self._last_expire = 0.0
        # A relayed getFile's path behind the opaque reference the connector got instead (a --local server's is an
        # absolute path): reference -> (account, place, revision, topic, path, expiry, consent period).
        self._paths: Dict[str, tuple] = {}
        self._places_key: Any = _UNREAD  # the place store's file as last read (None: no file), _UNREAD before any read
        self._places: list = []
        self._places_recorded = False
        self._published = 0  # items this process spooled: what wakes an idle long poll without reading the disk
        self.accepted_poll: Optional[dict[str, Any]] = None  # this reader's last established group subscription

    def _offered(self) -> bool:
        """Whether core offers your bot's groups now: the live, Official connector publishes ``owner_bot``."""
        from ouroboros.telegram_owner_bot import owner_bot_offered
        return owner_bot_offered(self.data_dir)

    # -- consent ------------------------------------------------------------------------------------------------
    def _consent_path(self) -> pathlib.Path:
        return self.state_dir / "group_feed.json"

    def _consent_record(self) -> tuple[str, Optional[dict[str, Any]]]:
        """The saved consent as it reads now: ``missing`` (never saved), ``unreadable`` (a read error), ``unusable``
        (bytes that name no one consent) or ``ok`` with ``{"enabled", "account_id", "at"}``."""
        try:
            raw = self._consent_path().read_bytes()
        except FileNotFoundError:
            return "missing", None
        except OSError:
            return "unreadable", None
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return "unusable", None
        if not isinstance(data, dict) or type(data.get("enabled")) is not bool:
            return "unusable", None
        account = data.get("account_id", "")
        if not isinstance(account, str) or (data["enabled"] and not _BOT.fullmatch(account)):
            return "unusable", None
        return "ok", {"enabled": data["enabled"], "account_id": account if data["enabled"] else "",
                      "at": str(data.get("at") or "")}

    def consent(self) -> dict[str, Any]:
        """The consent in force: anything but a readable saved one reads Off (unknown never authorizes)."""
        state, record = self._consent_record()
        return dict(record) if state == "ok" else {"enabled": False, "account_id": "", "at": ""}

    def consent_account_id(self) -> Optional[str]:
        """The owner's own door's reading of the saved consent: its bot's id while it is on; "" when it is off or was
        never saved; None when it cannot be read or names no one consent, which is unknown, never Off. Never part of
        the connector's doors or of the verified identity."""
        state, record = self._consent_record()
        if state == "ok":
            return record["account_id"] if record["enabled"] else ""
        return "" if state == "missing" else None

    def enabled_for(self, bot_id: str) -> bool:
        consent = self.consent()
        return consent["enabled"] and bool(bot_id) and consent["account_id"] == bot_id

    def _held(self) -> Optional[dict[str, Any]]:
        """The consent in force as one record, its bot and the write that saved it (``at``), or None while none is.
        A persist, an acknowledgement or a request let in under one record goes on only while it is still this one:
        an Off, or an Off and On again of the same bot, is another record."""
        state, record = self._consent_record()
        return record if state == "ok" and record["enabled"] else None

    def intake(self, bot_id: str) -> Optional[dict[str, Any]]:
        """The consent record one batch of ``bot_id``'s updates is received under, read once for the batch: the one
        in force while it names that bot and the owner is bound, else None (nothing of the batch is kept, and no
        group is gathered without a bound owner)."""
        verified = self.verified()
        held = self._held()
        if (verified and verified[0] == bot_id and held and held["account_id"] == bot_id and self.owner_bound()
                and self.verified() == verified):
            return held
        return None

    def _write_consent(self, enabled: bool, account: str) -> None:
        """The choice, written by the core's atomic replace of a synced copy; ``at`` names this very write."""
        from ouroboros.utils import atomic_write_json
        atomic_write_json(self._consent_path(), {"enabled": enabled, "account_id": account if enabled else "",
                                                 "at": _now_precise()}, fsync=True)

    def _begin_period(self, expected: str, db: sqlite3.Connection) -> bool:
        """On, first step: whether a new consent period starts for ``expected`` (it is not on for it already); if so
        the membership evidence of every earlier period, which may have changed unseen, goes in this transaction."""
        held = self._held()
        if held is not None and held["account_id"] == expected:
            return False
        db.execute("DELETE FROM groups")
        return True

    def _publish_on(self, expected: str) -> None:
        """On, second step, once the old membership is gone for good: the consent itself (another On of this bot
        that finished meanwhile stands as it is)."""
        held = self._held()
        if held is None or held["account_id"] != expected:
            self._write_consent(True, expected)

    def _withdraw(self, expected: str) -> None:
        """Off of ``expected``'s consent only, decided on the saved record."""
        state, record = self._consent_record()
        if state == "unreadable":
            raise FeedRefusal("consent_unknown", "Your groups choice could not be read now; nothing was changed.",
                              retry_after=5)
        if state == "missing" or (state == "ok" and not record["enabled"]):
            return  # nothing to withdraw
        if state == "ok" and record["account_id"] != expected:
            raise FeedRefusal("account_changed", "Groups are on for another bot now; nothing was turned off.")
        self._write_consent(False, "")  # also replaces unusable bytes: no consent stands in them either way

    def set_consent(self, enabled: bool, expected_account_id: str = "") -> None:
        """The owner's choice for the bot they saw. On: that bot must be the one confirmed now, its owner bound and
        the connector offering its groups. Off: only the saved consent of the bot they saw is withdrawn, from this
        local record, without asking Telegram (which may be away); a consent saved meanwhile for another bot stays
        (``account_changed``). Off, On, every persist and every acknowledgement take one serialization, the feed's
        write lock: On first commits the removal of the old membership, then publishes the consent under the lock
        again, so nothing reads the new consent beside stale evidence. A store or file that cannot take the choice
        safely raises (``sqlite3.Error``, ``OSError``) and nothing is claimed."""
        expected = str(expected_account_id or "")
        if enabled:
            verified = self.verified()  # one reading: the bot this consent names
            if verified is None or verified[0] != expected:
                raise FeedRefusal("account_changed", "This bot is not the one confirmed for this token now.")
            if not self.owner_bound():
                raise FeedRefusal("owner_not_bound", "Open your bot in Telegram first, so it is bound to you.")
            if not self.capable():  # nothing would read the groups: keep nothing for them
                raise FeedRefusal("connector_unavailable", "Turn on the Telegram Bot connector first.")
        else:
            state, record = self._consent_record()
            if state == "missing" or (state == "ok" and not record["enabled"]):
                return  # already off: nothing to write, no store to open

        def decide(db):
            if not enabled:
                self._transaction(db, lambda: self._withdraw(expected))
            elif self._transaction(db, lambda: self._begin_period(expected, db)):
                self._transaction(db, lambda: self._publish_on(expected))
        self._use(decide)

    # -- status -------------------------------------------------------------------------------------------------
    def facts(self, *, groups: bool = False) -> dict[str, Any]:
        """The verified bot, its invite and the consent: the native door's status and the Host identity's facts;
        ``groups`` (the identity door's ``groups=1`` only) adds ``group_ids``."""
        verified = self.verified()
        bot, name = verified if verified else ("", "")
        bound = bool(self.owner_bound())
        state, record = self._consent_record()  # one reading: the header and its group_ids speak of the same
        held = record if state == "ok" and record["enabled"] else None
        enabled = bool(bot) and held is not None and held["account_id"] == bot and bound
        facts: dict[str, Any] = {"account_id": bot, "owner_bound": bound, "enabled": enabled,
                                 "invite_url": f"https://t.me/{name}?startgroup=true" if _USERNAME.fullmatch(name)
                                 else "", "reason": "" if enabled else "identity_unconfirmed" if not bot else "feed_off"}
        if bot and _USERNAME.fullmatch(name):
            facts["identity"] = {"sends_as": {"id": bot, "label": f"@{name}", "kind": "bot"}}
        if groups:
            facts["group_ids"] = self._group_ids(verified, bound, state, held)
        return facts

    def _group_ids(self, verified: Optional[tuple[str, str]], bound: bool, state: str,
                   held: Optional[dict[str, Any]]) -> Optional[list[str]]:
        """``group_ids`` for the identity ``facts`` observed (its verified bot, bound owner and consent reading): the
        groups and supergroups that bot is in as this consent period has seen them (canonical negative ids from the
        membership index); ``[]`` when none, or when no period is in force for it (groups Off or saved for another
        bot, no bound owner: never an id then); None when unknown (no verified bot, the consent or store unreadable,
        or that observation changed before the index read ended: another bot, owner or consent period)."""
        if verified is None or state not in {"ok", "missing"}:
            return None
        if held is None or held["account_id"] != verified[0] or not bound:
            return []
        try:
            rows = (self._use(lambda db: db.execute(
                "SELECT chat_id FROM groups WHERE account_id=? AND membership='present' AND chat_type IN "
                "('group','supergroup') ORDER BY chat_id", (verified[0],)).fetchall())
                if self._store_path().exists() else [])
        except sqlite3.Error:
            return None
        if self.verified() != verified or bool(self.owner_bound()) != bound or self._held() != held:
            return None
        return [row[0] for row in rows if _CHAT.fullmatch(str(row[0]))]

    def status(self) -> dict[str, Any]:
        """The owner's own door: the facts, and the saved consent's bot (``consent_account_id``) for its Off."""
        try:
            pending = (self._use(lambda db: db.execute("SELECT COUNT(*) FROM feed").fetchone()[0])
                       if self._store_path().exists() else 0)
        except sqlite3.Error:
            pending = None  # unknown, never zero
        with self._lock:
            gaps = dict(self._gaps)
        facts = self.facts()
        return {"v": 1, **facts,
                "intake_ready": facts["enabled"] and self.accepted_poll is not None
                and self.intake(facts["account_id"]) == self.accepted_poll,
                "consent_account_id": self.consent_account_id(),
                "capable": bool(self.capable()), "pending": pending, "write_gaps": gaps["count"],
                "last_write_gap_at": gaps["at"]}

    # -- spool --------------------------------------------------------------------------------------------------
    def _store_path(self) -> pathlib.Path:
        return self.state_dir / _STORE

    def _use(self, work: Callable[[sqlite3.Connection], Any], *, wait: float = 5.0) -> Any:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._store_path(), timeout=wait, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            if not self._schema_ready:  # the schema is written once, never by a read of a store that has it
                present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')")}
                if not _SCHEMA_OBJECTS <= present:
                    db.execute("PRAGMA journal_mode=WAL")
                    db.executescript(_SCHEMA)
                self._schema_ready = True
            return work(db)
        finally:
            db.close()

    @staticmethod
    def _transaction(db: sqlite3.Connection, work: Callable[[], Any]) -> Any:
        db.execute("BEGIN IMMEDIATE")
        try:
            result = work()
            db.execute("COMMIT")
            return result
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def _gap(self, reason: str) -> None:
        with self._lock:
            first = self._gaps["reason"] != reason
            self._gaps = {"count": self._gaps["count"] + 1, "reason": reason, "at": _now_iso()}
        if first:
            self.log("warning", f"Telegram group feed could not keep an update ({reason}); it was confirmed anyway.")

    def _current_places(self, *, strict: bool = False) -> list:
        """The place store's places, read again only when its file changed; no file is no place. A store that cannot
        be read now is no disconnect: the view last read stands (and is read again next time); before any, it raises."""
        from ouroboros.places import place_records, places_path
        try:
            info = places_path(self.data_dir).stat()
            key = (info.st_ino, info.st_mtime_ns, info.st_size)
            if key != self._places_key:
                records = place_records(self.data_dir, persist_import=False)
                self._places, self._places_recorded, self._places_key = list(records.places.values()), records.recorded, key
        except FileNotFoundError:
            self._places, self._places_recorded, self._places_key = [], False, None
        except Exception:
            if strict or self._places_key is _UNREAD:
                raise
        return self._places

    def _expire(self, db: sqlite3.Connection, now: float) -> None:
        """Retire old keys and topics; retire old file offers only with readable evidence their connection ended."""
        from ouroboros.telegram_owner_bot import is_owner_bot_place

        cutoff = now - _SEEN_DAYS * 86400
        db.execute("DELETE FROM seen WHERE seen_at<?", (cutoff,))
        try:
            connected = self._current_places(strict=True)
        except (ValueError, OSError):
            connected = None
        if connected is not None and self._places_recorded:
            revisions = json.dumps([(p.account_id, p.place_id, p.connected_revision) for p in connected
                                    if is_owner_bot_place(p)])
            db.execute("DELETE FROM offers WHERE offered_at<? AND (account_id,place_id,connected_revision) NOT IN "
                       "(SELECT json_extract(value,'$[0]'),json_extract(value,'$[1]'),json_extract(value,'$[2]') "
                       "FROM json_each(?)) AND NOT EXISTS (SELECT 1 FROM feed WHERE "
                       "feed.account_id=offers.account_id AND feed.received_at<=offers.offered_at)", (cutoff, revisions))
        db.execute("DELETE FROM topics WHERE seen_at<? AND NOT EXISTS (SELECT 1 FROM feed WHERE feed.account_id="
                   "topics.account_id AND feed.received_at<=topics.seen_at)", (cutoff,))
        self._last_expire = now

    def publish(self, update: Mapping[str, Any], bot_id: str, held: Any = _NOW) -> str:
        """Spool one group update of bot ``bot_id``: ``stored``, ``sighting``, ``duplicate``, ``off``, ``gap``.
        ``held``: the consent record its batch was received under (``intake``; default: read now). That record is
        checked again at the write itself, inside the feed's write transaction: an Off, or an Off and On again, since
        the batch arrived keeps nothing of it."""
        found = group_chat(update)
        if found is None:
            return "ignored"
        held = self.intake(bot_id) if held is _NOW else held
        if held is None or not bot_id or held["account_id"] != bot_id:
            return "off"
        kind, chat = found
        body, chat_id = update[kind], str(chat.get("id"))
        try:
            known = self._current_places()
        except Exception as exc:  # no place view could be read yet: nothing it cannot judge is kept
            self._gap(f"places_{type(exc).__name__}")
            return "gap"
        now = time.time()

        def write(db):
            from ouroboros.telegram_owner_bot import connected_owner_bot_place
            own = lookup = topic_of(body)
            if kind in REACTION_KINDS and type(body.get("message_id")) is int:
                # A reaction names only its message: the place is chosen by that message's topic, as this feed saw
                # or sent it, so a topic's own place gets it.
                row = db.execute("SELECT thread_id FROM topics WHERE account_id=? AND chat_id=? AND message_id=?",
                                 (bot_id, chat_id, body["message_id"])).fetchone()
                lookup = str(row[0]) if row else ""
            place = connected_owner_bot_place(known, account_id=bot_id, conversation_id=chat_id, thread_id=lookup)
            # The item names a topic place's own topic, else the topic the update is in: the one it names itself,
            # or for a reaction its message's topic as this feed knows it, a chat-wide place's included.
            topic = str(getattr(place, "thread_id", "") or "") or lookup
            seen = sighting(update, kind, chat, bot_id)
            key = event_key(update, kind, body, chat_id)
            if place is None:
                item: dict[str, Any] = {"sighting": seen}
            else:  # the topic this update is in, a chat-wide place's too: replies and files keep it
                item = {"update": dict(update), "place": {"place_id": place.place_id,
                                                          "connected_revision": place.connected_revision,
                                                          "thread_id": topic}}
            offers = sorted(_file_ids(body, set())) if place is not None else []

            def work():
                if self.intake(bot_id) != held:
                    return None  # the consent changed while this update waited: no period keeps it
                if db.execute("SELECT 1 FROM seen WHERE account_id=? AND dedupe=?", (bot_id, key)).fetchone():
                    return 0
                inserted = db.execute("INSERT OR IGNORE INTO feed(account_id,dedupe,body,received_at) VALUES(?,?,?,?)",
                                      (bot_id, key, json.dumps(item, ensure_ascii=False), now)).rowcount
                if not inserted:
                    return 0  # the same update still waits: its evidence is already in
                if place is None:
                    coalesce_sightings(db, bot_id, chat_id, key, now)
                for file_id in offers:  # a file offered again is offered from now on
                    db.execute(_OFFER_RECORD, (bot_id, place.place_id, place.connected_revision, topic, file_id, now))
                record_membership(db, bot_id, seen, kind)
                if kind in {"message", "edited_message"} and type(body.get("message_id")) is int:
                    db.execute(_TOPIC_RECORD, (bot_id, chat_id, body["message_id"], own, now))
                if now - self._last_expire >= _EXPIRE_EVERY:
                    self._expire(db, now)
                return inserted
            inserted = self._transaction(db, work)
            if inserted is None:
                return "off"
            if inserted:
                with self._lock:
                    self._published += 1  # committed: a long poll waiting here reads it now
            return ("stored" if place is not None else "sighting") if inserted else "duplicate"
        try:
            return self._use(write, wait=_PUBLISH_WAIT)
        except sqlite3.Error as exc:
            self._gap(f"store_{type(exc).__name__}")
            return "gap"

    def read(self, account_id: str, offset: int, limit: int = 100, *,
             held: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Confirm every item below ``offset`` (getUpdates' rule) and return up to ``limit`` from there. ``held``: the
        consent the caller was let in under (default: the one in force now). A read with nothing to confirm writes
        nothing; a confirmation checks that consent inside the feed's write transaction, where Off and On are decided
        too, so a read overtaken by an Off, or an Off and On again, confirms and returns nothing."""
        if not self._store_path().exists():
            return []
        held = self.intake(account_id) if held is None else held
        if held is None or held["account_id"] != account_id:
            return []
        now = time.time()

        def work(db):
            select = ("SELECT feed_id, body FROM feed WHERE account_id=? AND feed_id>=? ORDER BY feed_id LIMIT ?",
                      (account_id, offset, limit))
            below = db.execute("SELECT 1 FROM feed WHERE account_id=? AND feed_id<? LIMIT 1",
                               (account_id, offset)).fetchone()
            if below is None:
                rows = db.execute(*select).fetchall()
                # Changed since the caller's check: nothing handed (an empty read has nothing to check).
                return rows if not rows or self.intake(account_id) == held else []

            def confirm():
                if self.intake(account_id) != held:
                    return None  # Off (or Off and On again) overtook this read: its offset stays unconfirmed
                db.execute("INSERT OR IGNORE INTO seen SELECT account_id, dedupe, ? FROM feed "
                           "WHERE account_id=? AND feed_id<?", (now, account_id, offset))
                db.execute("DELETE FROM feed WHERE account_id=? AND feed_id<?", (account_id, offset))
                return db.execute(*select).fetchall()
            return self._transaction(db, confirm) or []
        return [{"feed_id": feed_id, **json.loads(body)} for feed_id, body in self._use(work)]

    def _membership(self, account_id: str, chat_id: str) -> Optional[str]:
        """No observation is different from an observed departure, and a store error is neither."""
        if not self._store_path().exists():
            return None
        row = self._use(lambda db: db.execute("SELECT chat_type, membership FROM groups WHERE account_id=? AND "
                                             "chat_id=?", (account_id, chat_id)).fetchone())
        return str(row[1]) if row and row[0] in GROUP_TYPES else None

    def member_of(self, account_id: str, chat_id: str) -> bool:
        try:
            return self._membership(account_id, chat_id) == "present"
        except (OSError, sqlite3.Error):
            return False

    def connect_allowed(self, account_id: str, chat_id: str) -> bool:
        """The exact place writer's check: the confirmed bot, its consent, a bound owner, in this group now."""
        account, chat = str(account_id), str(chat_id)
        held = self.intake(account)
        return (held is not None and bool(_CHAT.fullmatch(chat)) and self.member_of(account, chat)
                and self.intake(account) == held)

    def offered(self, account_id: str, place_id: str, revision: int, thread_id: str, file_id: str) -> bool:
        if not self._store_path().exists():
            return False
        try:
            return bool(self._use(lambda db: db.execute(
                "SELECT 1 FROM offers WHERE account_id=? AND place_id=? AND connected_revision=? AND thread_id=? "
                "AND file_id=?", (account_id, place_id, revision, thread_id, file_id)).fetchone()))
        except sqlite3.Error:
            raise FeedRefusal("feed_unavailable", "The feed could not be read now.", retry_after=5) from None

    def _message_thread(self, account_id: str, chat_id: str, message_id: Any) -> Optional[str]:
        """The forum topic of a message of this chat the feed saw or this relay sent ("" for General or no forum);
        None when it knows none."""
        if type(message_id) is not int or not self._store_path().exists():
            return None
        try:
            row = self._use(lambda db: db.execute(
                "SELECT thread_id FROM topics WHERE account_id=? AND chat_id=? AND message_id=?",
                (account_id, chat_id, message_id)).fetchone())
        except sqlite3.Error:
            raise FeedRefusal("feed_unavailable", "The feed could not be read now.", retry_after=5) from None
        return str(row[0]) if row else None

    def _remember_sent(self, account_id: str, chat_id: str, body: Any) -> None:
        """The topic of each message this relay just sent, so an edit, deletion or reaction of the bot's own reply
        finds it as it finds any message the feed saw. Best effort and never raising: it runs after the send, whose
        answer stays as Telegram gave it (a failure here must never read as "nothing sent")."""
        try:
            result = body.get("result") if isinstance(body, dict) and body.get("ok") is True else None
            now = time.time()
            rows = [(account_id, chat_id, message["message_id"], topic_of(message), now)
                    for message in (result if isinstance(result, list) else [result])
                    if isinstance(message, dict) and type(message.get("message_id")) is int]
            if rows:
                self._use(lambda db: db.executemany(_TOPIC_RECORD, rows), wait=_PUBLISH_WAIT)
        except Exception:
            self.log("warning", "Telegram group feed could not note a sent message's topic.")

    # -- Host doors ---------------------------------------------------------------------------------------------
    async def handle(self, operation: str, request: Mapping[str, Any]) -> tuple[int, Any, str]:
        try:
            if operation == "identity":  # Settings, the consent and the index are disk reads: off the event loop
                return 200, await asyncio.to_thread(self.facts, groups=str(request.get("groups") or "") == "1"), ""
            if operation == "updates":
                return await self._updates(request)
            if operation in {"call", "upload"}:
                return await self._relay(operation, request)
            if operation == "file":
                return await self._file(request)
            raise FeedRefusal("method_not_allowed", "Unknown operation.")
        except FeedRefusal as refusal:
            body: dict[str, Any] = {"ok": False, "error_code": 403, "description": refusal.description,
                                    "relay_refusal": refusal.code}
            if refusal.retry_after:
                body["parameters"] = {"retry_after": refusal.retry_after}
            return 403, body, ""
        except (ValueError, OSError, sqlite3.Error):
            # A local read or argument failure before any provider call is a typed refusal, never an unshaped 5xx the
            # connector would read as an unknown outbound outcome (``PlaceError`` is a ``ValueError``). A failure once
            # a request may have left is ``_send``'s unknown outcome, never this.
            return 403, {"ok": False, "error_code": 403, "description": "The feed could not read this request now.",
                         "relay_refusal": "feed_unavailable", "parameters": {"retry_after": 5}}, ""

    def _holding(self, request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        """Who may read or act now, and under which consent record: the confirmed bot the caller names, its consent
        and a bound owner."""
        verified = self.verified()
        account = str(request.get("account_id") or "")
        if verified is None:
            raise FeedRefusal("bridge_not_ready", "Your own bot is not confirmed yet.", retry_after=5)
        if account != verified[0]:
            raise FeedRefusal("account_mismatch", "This is not the bot this bridge reads now.")
        held = self._held()
        if not self.owner_bound() or held is None or held["account_id"] != account:
            raise FeedRefusal("feed_off", "Groups are off for your own bot.")
        return account, held

    def _still(self, request: Mapping[str, Any], held: dict[str, Any], nothing: str = "nothing was sent") -> None:
        """Right before a provider request or a hand-over: the consent this request was let in under is still the one
        in force. An Off meanwhile refuses as Off; an Off and On again of the same bot asks again, under the new one."""
        if self._holding(request)[1] != held:
            raise FeedRefusal("feed_unavailable", f"Groups were turned off and on again meanwhile; {nothing}.",
                              retry_after=1)

    async def _updates(self, request: Mapping[str, Any]) -> tuple[int, Any, str]:
        """getUpdates over the spool. One long poll is let in under one consent record: its reads confirm and hand
        only under it, so an Off or a new consent meanwhile confirms no old offset. Waiting reads no disk: it watches
        this process's own spooling (``_published``) and the verified bot, both in memory."""
        try:
            offset = max(0, int(request.get("offset") or 0))
            timeout = min(_MAX_POLL, max(0, int(request.get("timeout") or 0)))
            limit = min(100, max(1, int(request.get("limit") or 100)))
        except (TypeError, ValueError):
            raise FeedRefusal("argument_not_allowed", "offset, timeout and limit are integers.") from None
        deadline = time.monotonic() + timeout
        account, held = await asyncio.to_thread(self._holding, request)
        while True:
            mark = self._published
            try:
                items = await asyncio.to_thread(self.read, account, offset, limit, held=held)
            except sqlite3.Error:
                raise FeedRefusal("feed_unavailable", "The feed could not be read now.", retry_after=5) from None
            if items:  # the same consent, bot and bound owner, right before anything is handed
                await asyncio.to_thread(self._still, request, held, "nothing was handed")
                return 200, {"ok": True, "result": items}, ""
            while self._published == mark and time.monotonic() < deadline:
                if (self.verified() or ("",))[0] != account:  # another token, or none confirmed: refused as at the door
                    await asyncio.to_thread(self._holding, request)
                await asyncio.sleep(_POLL_STEP)
            if self._published == mark:
                return 200, {"ok": True, "result": []}, ""

    def _client(self, account: str) -> Any:
        """This bridge's client for the token in force, which must still be the confirmed bot ``account``."""
        try:
            client = self.client_factory()
            client.check_binding()
        except Exception:
            raise FeedRefusal("bridge_not_ready", "Your own bot's token is changing; nothing was sent.",
                              retry_after=5) from None
        if str(client.bot_id) != account:
            raise FeedRefusal("account_mismatch", "This is not the bot this bridge reads now.")
        return client

    def _bound(self, account: str, request: Mapping[str, Any], chat_id: str, *, write: bool) -> Any:
        """The exact owner_bot place the caller names, at its connected revision and topic, current now."""
        from ouroboros.place_guard import PLACE_BINDING_VERSION
        from ouroboros.places import PlaceError
        from ouroboros.telegram_owner_bot import FEED_CONNECTOR, owner_bot_binding
        revision = request.get("connected_revision")
        place_id = request.get("place_id")
        if not isinstance(place_id, str) or not place_id or type(revision) is not int or revision < 0:
            raise FeedRefusal("binding_changed", "place_id and connected_revision name the bound place.")
        target = {"provider": _PROVIDER, "account_id": account, "conversation_id": chat_id,
                  "thread_id": str(request.get("thread_id") or "")}
        try:
            decision = owner_bot_binding(self.data_dir, FEED_CONNECTOR, target,
                                         {"version": PLACE_BINDING_VERSION, "connector": FEED_CONNECTOR, **target,
                                          "place_id": place_id, "connected_revision": revision},
                                         PLACE_BINDING_VERSION, with_place=True)
        except PlaceError as exc:
            if exc.code == "place_not_found":
                raise FeedRefusal("place_not_connected", "This group is not connected.") from None
            raise FeedRefusal("feed_unavailable", "The place store could not be read now.", retry_after=5) from None
        except ValueError:
            raise FeedRefusal("binding_changed", "This group is no longer connected as it was.") from None
        if decision["state"] in {"disconnected", "changed"}:
            raise FeedRefusal("binding_changed", "This group is no longer connected as it was.")
        if write and not decision["allowed"]:
            raise FeedRefusal("place_not_active", "This group's place is not Active.")
        return decision["_place"]

    def _require_member(self, account: str, chat_id: str) -> None:
        membership = self._membership(account, chat_id)
        if membership is None:
            raise FeedRefusal("feed_unavailable", "Your bot's group membership could not be confirmed now.",
                              retry_after=5)
        if membership != "present":
            raise FeedRefusal("not_connected", "Your own bot is not in this group now.")

    async def _refresh_member(self, client, account: str, held: dict, request: Mapping[str, Any], chat_id: str,
                              *, write: bool) -> None:
        """A quiet, already connected group can restore evidence after Off→On; this cannot discover new places."""
        if not _CHAT.fullmatch(chat_id) or await asyncio.to_thread(self._membership, account, chat_id) is not None:
            return
        await asyncio.to_thread(self._bound, account, request, chat_id, write=write)
        await asyncio.to_thread(self._still, request, held)
        code, body, _ = await self._send(client, "getChatMember", {"chat_id": chat_id, "user_id": account}, {})
        result = body.get("result") if code == 200 and body.get("ok") is True else None
        user = result.get("user") if isinstance(result, dict) else None
        status = member_status(result) if isinstance(user, dict) and str(user.get("id")) == account else ""
        if not status:
            return  # read failure is not absence

        def keep(db):
            def bound():
                self._bound(account, request, chat_id, write=write)
                client.check_binding()
                self._still(request, held)
                # Poll evidence that arrived while the request was reading wins over its answer.
                db.execute("INSERT OR IGNORE INTO groups VALUES(?,?,?,?,0)", (account, chat_id, "group", status))
            self._transaction(db, bound)
        await asyncio.to_thread(self._use, keep, wait=_PUBLISH_WAIT)

    async def refresh_membership(self, request: Mapping[str, Any], chat_id: str) -> bool:
        """Recheck a previously bound inbox arrival without granting Active to a Silent place."""
        from .telegram_api import TelegramBindingChanged
        try:
            account, held = await asyncio.to_thread(self._holding, request)
            await self._refresh_member(self._client(account), account, held, request, chat_id, write=False)
            return await asyncio.to_thread(self.member_of, account, chat_id)
        except (FeedRefusal, OSError, ValueError, sqlite3.Error, TelegramBindingChanged):
            return False  # the existing arrival retry and settled-reader contract remains

    def _in_scope(self, account: str, chat_id: str, place: Any, method: str, params: Mapping[str, Any]) -> None:
        """What this place may touch: a member's restriction or ban acts on the whole group, so it needs a chat-wide
        place; an edit, deletion or reaction acts in its message's own topic, which a topic place must know to be its
        own; a reply in a topic never quotes a message the feed knows to be in another one."""
        topic = str(getattr(place, "thread_id", "") or "")
        if method in _CHAT_WIDE:
            if topic:
                raise FeedRefusal("argument_not_allowed", "This acts on the whole group; this place is one topic.")
            return
        if not topic:
            return  # a chat-wide place holds every topic of its chat
        if method in _MESSAGE_TARGETED and self._message_thread(account, chat_id, params.get("message_id")) != topic:
            raise FeedRefusal("argument_not_allowed", "This action targets a message of another topic.")
        reply = params.get("reply_parameters")
        reply = json.loads(reply) if isinstance(reply, str) else reply  # already checked: one object
        if isinstance(reply, Mapping):
            quoted = self._message_thread(account, chat_id, reply.get("message_id"))
            if quoted is not None and quoted != topic:
                raise FeedRefusal("argument_not_allowed", "This reply quotes a message of another topic.")

    def _place_of(self, account: str, request: Mapping[str, Any]) -> Any:
        """A file read's place: the one named, connected now as it was (any state; reads never speak). A place store
        that cannot be read now is a retry, never a disconnect."""
        from ouroboros.places import PlaceError, get_place
        try:
            place = get_place(self.data_dir, str(request.get("place_id") or ""))
        except PlaceError as exc:
            if exc.code == "place_not_found":
                raise FeedRefusal("not_connected", "This group is not connected.") from None
            raise FeedRefusal("feed_unavailable", "The place store could not be read now.", retry_after=5) from None
        return self._bound(account, request, place.conversation_id, write=False)

    async def _relay(self, operation: str, request: Mapping[str, Any]) -> tuple[int, Any, str]:
        account, held = await asyncio.to_thread(self._holding, request)
        method, params = str(request.get("method") or ""), request.get("params")
        allowed = _ARGS.get(method) if operation == "call" else (_UPLOAD_ARGS | {_UPLOADS[method]}
                                                                 if method in _UPLOADS else None)
        if allowed is None or not isinstance(params, Mapping):
            raise FeedRefusal("method_not_allowed", f"{method or 'This method'} is not available through the relay.")
        if set(params) - allowed:
            raise FeedRefusal("argument_not_allowed", "Unlisted arguments: " + ", ".join(sorted(set(params) - allowed)))
        client = self._client(account)  # pinned: a token change after this sends nothing
        if method == "getFile":
            return await self._get_file(client, account, held, request, str(params.get("file_id") or ""))
        chat_id, thread = str(params.get("chat_id") or ""), str(request.get("thread_id") or "")
        if not _CHAT.fullmatch(chat_id):
            raise FeedRefusal("argument_not_allowed", "chat_id names one group by its numeric id.")
        named = str(params["message_thread_id"]) if "message_thread_id" in params else ""
        if named != thread and (named or method in _TOPIC_SENDS):
            raise FeedRefusal("argument_not_allowed", "message_thread_id must be the bound topic.")
        data = self._checked(params)
        if operation == "upload" and method != "sendMediaGroup":
            data.pop(_UPLOADS[method], None)  # one file part only: the attach:// text field would duplicate it
        place = await asyncio.to_thread(self._bound, account, request, chat_id, write=True)
        await asyncio.to_thread(self._in_scope, account, chat_id, place, method, params)
        await self._refresh_member(client, account, held, request, chat_id, write=True)
        await asyncio.to_thread(self._require_member, account, chat_id)
        files = await asyncio.to_thread(self._files, method, params, request.get("files")) if operation == "upload" else {}
        try:
            # Checked again after the files were read, the consent last, right before the request: an Off (or an Off
            # and On again), a disconnect or Silent meanwhile sends nothing.
            await asyncio.to_thread(self._bound, account, request, chat_id, write=True)
            await asyncio.to_thread(self._require_member, account, chat_id)
            await asyncio.to_thread(self._still, request, held)
            status, body, media = await self._send(client, method, data, files)
        finally:
            for _name, handle in files.values():
                with contextlib.suppress(OSError):  # after a send, a closing failure never reads as "nothing sent"
                    handle.close()
        if status == 200 and method in _SENT_MESSAGES:
            await asyncio.to_thread(self._remember_sent, account, chat_id, body)
        return status, body, media

    async def _get_file(self, client, account: str, held: dict[str, Any], request: Mapping[str, Any],
                        file_id: str) -> tuple[int, Any, str]:
        place = await asyncio.to_thread(self._place_of, account, request)
        thread = str(request.get("thread_id") or "")
        if not await asyncio.to_thread(self.offered, account, place.place_id, place.connected_revision, thread, file_id):
            raise FeedRefusal("file_not_offered", "This file was not offered in this group's connection and topic.")
        await asyncio.to_thread(self._place_of, account, request)
        await asyncio.to_thread(self._still, request, held)
        status, body, media = await self._send(client, "getFile", {"file_id": file_id}, {})
        result = body.get("result") if status == 200 and isinstance(body, dict) else None
        if isinstance(result, dict) and isinstance(result.get("file_path"), str) and result["file_path"]:
            await asyncio.to_thread(self._place_of, account, request)
            await asyncio.to_thread(self._still, request, held, "no file reference was handed")
            reference = f"feed/{secrets.token_urlsafe(18)}"  # URL-safe, no leading slash: sent back as it is
            with self._lock:
                now = time.monotonic()
                self._paths = {item: held for item, held in self._paths.items() if held[5] > now}
                self._paths[reference] = (account, place.place_id, place.connected_revision, thread,
                                          result["file_path"], now + _FILE_PATH_TTL, held)
            body = {**body, "result": {**result, "file_path": reference}}
        return status, body, media

    @staticmethod
    def _checked(params: Mapping[str, Any]) -> dict[str, Any]:
        """Form fields for Telegram, every nested destination checked: a reply stays in this chat."""
        data: dict[str, Any] = {}
        for key, value in params.items():
            if key == "reply_parameters":
                try:
                    reply = json.loads(value) if isinstance(value, str) else value
                except (ValueError, RecursionError):
                    raise FeedRefusal("argument_not_allowed", "reply_parameters is one JSON object.") from None
                if not isinstance(reply, Mapping) or set(reply) != {"message_id"} or type(reply["message_id"]) is not int:
                    raise FeedRefusal("argument_not_allowed", "reply_parameters names only a message of this chat.")
                value = reply
            elif key == "media":
                try:
                    media = json.loads(value) if isinstance(value, str) else value
                except (ValueError, RecursionError):
                    raise FeedRefusal("argument_not_allowed", "media is one JSON array.") from None
                if (not isinstance(media, list) or not 2 <= len(media) <= 10
                        or any(not isinstance(item, Mapping) or set(item) - _MEDIA_KEYS
                               or not str(item.get("media") or "").startswith("attach://") for item in media)):
                    raise FeedRefusal("argument_not_allowed", "An album lists 2 to 10 attached files only.")
                value = media
            data[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        return data

    def _files(self, method: str, params: Mapping[str, Any], declared: Any) -> Dict[str, Any]:
        """The connector's own files, opened once and checked on the open handle that is then streamed."""
        root = (self.data_dir / "state" / "skills" / "telegram-bot").resolve()
        if method == "sendMediaGroup":
            media = params.get("media")
            try:
                media = json.loads(media) if isinstance(media, str) else media
            except (ValueError, RecursionError):
                raise FeedRefusal("argument_not_allowed", "media is one JSON array.") from None
            wanted = {str(item.get("media"))[len("attach://"):] for item in media or []}
        else:
            wanted = {_UPLOADS[method]}
            if str(params.get(_UPLOADS[method]) or "") not in {"", f"attach://{_UPLOADS[method]}"}:
                raise FeedRefusal("argument_not_allowed", "A file travels as an attachment only.")
        if not isinstance(declared, list) or not declared:
            raise FeedRefusal("argument_not_allowed", "files lists the attached files.")
        opened: Dict[str, Any] = {}
        try:
            for row in declared:
                if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                    raise FeedRefusal("argument_not_allowed", "Each file names its path.")
                field, name, path = str(row.get("field") or ""), str(row.get("name") or ""), pathlib.Path(row["path"])
                if field not in wanted or field in opened or not name or not path.is_absolute():
                    raise FeedRefusal("argument_not_allowed", "Each attached field names one file of this send.")
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise FeedRefusal("argument_not_allowed", "A file must be the connector's own copy.")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
                try:
                    handle = os.fdopen(os.open(resolved, flags), "rb")
                except OSError:
                    raise FeedRefusal("file_missing", "A file to send is missing.") from None
                opened[field] = (name, handle)
                info = os.fstat(handle.fileno())
                # Type and size before any read or hash: a directory, a device or an oversized file is refused at once.
                if (not stat.S_ISREG(info.st_mode) or type(row.get("size")) is not int
                        or info.st_size != row["size"] or info.st_size > _UPLOAD_LIMIT):
                    raise FeedRefusal("file_changed", "A file does not match its recorded size and SHA-256.")
                digest = hashlib.sha256()
                try:
                    for piece in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(piece)
                except OSError:
                    raise FeedRefusal("file_missing", "A file to send could not be read.") from None
                handle.seek(0)
                if digest.hexdigest() != row.get("sha256"):
                    raise FeedRefusal("file_changed", "A file does not match its recorded size and SHA-256.")
            if set(opened) != wanted:
                raise FeedRefusal("argument_not_allowed", "Every attached field needs its file.")
        except BaseException:
            for _name, handle in opened.values():
                handle.close()
            raise
        return opened

    async def _send(self, client, method: str, data: dict[str, Any], files: Dict[str, Any]) -> tuple[int, Any, str]:
        """One provider request through the pinned client: Telegram's whole answer, or a typed outcome. Only a
        failure the client proves came before any request byte (a token change before it, a refused connection) is
        ``not_sent``; anything else once the request may have started (a reset, a file read failing mid-upload, an
        unexpected client error) is ``unknown``, never a refusal the connector could send again."""
        from .telegram_api import TelegramBindingChanged, TelegramTransportError
        try:
            status, body = await client.raw_request(method, data=data, files=files, timeout=120 if files else 35)
        except TelegramBindingChanged as exc:
            if exc.unsent:
                raise FeedRefusal("bridge_not_ready", "Your own bot's token changed; nothing was sent.",
                                  retry_after=5) from None
            return 502, {"relay_outcome": "unknown"}, ""
        except TelegramTransportError as exc:
            return (503, {"relay_outcome": "not_sent"}, "") if exc.unsent else (502, {"relay_outcome": "unknown"}, "")
        except Exception:
            return 502, {"relay_outcome": "unknown"}, ""
        if body is None or "ok" not in body:
            return 502, {"relay_outcome": "unknown"}, ""
        if isinstance(body.get("description"), str) and client.token:
            from urllib.parse import quote
            description = body["description"].replace(client.token, "<token>")
            body["description"] = description.replace(quote(client.token, safe=""), "<token>")
        return status, body, ""

    async def _file(self, request: Mapping[str, Any]) -> tuple[int, Any, str]:
        from .telegram_api import (
            TelegramBindingChanged, TelegramDownloadTooLarge, TelegramFileReadTimeout, TelegramFileRefused,
        )
        account, consent = await asyncio.to_thread(self._holding, request)
        place = await asyncio.to_thread(self._place_of, account, request)
        reference = str(request.get("file_path") or "")
        scope = (account, place.place_id, place.connected_revision, str(request.get("thread_id") or ""))
        with self._lock:
            self._paths = {item: held for item, held in self._paths.items() if held[5] > time.monotonic()}
            held = self._paths.get(reference)
        if held is None or held[:4] != scope or held[6] != consent:
            return 404, {"ok": False, "error_code": 404, "description": "Not Found: run getFile again"}, ""
        client = self._client(account)
        await asyncio.to_thread(self._still, request, consent)
        try:  # the bridge's own download of the path Telegram named: its server, its --local store, the byte limit
            content = await client._download_bytes(held[4], limit=_DOWNLOAD_LIMIT)
            await asyncio.to_thread(self._place_of, account, request)
            await asyncio.to_thread(self._still, request, consent, "no file bytes were handed")
            client.check_binding()
        except TelegramBindingChanged:
            raise FeedRefusal("bridge_not_ready", "Your own bot's token changed.", retry_after=5) from None
        except TelegramDownloadTooLarge:
            return 400, {"ok": False, "error_code": 400, "description": "Bad Request: file is too big"}, ""
        except TelegramFileRefused as exc:
            return exc.status_code, {"ok": False, "error_code": exc.status_code,
                                     "description": f"HTTP {exc.status_code}"}, ""
        except TelegramFileReadTimeout:
            return 504, {"relay_outcome": "file_read_timeout"}, ""
        except RuntimeError:
            return 502, {"relay_outcome": "unknown"}, ""
        return 200, bytes(content), "application/octet-stream"
