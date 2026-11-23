"""The read-only archive of saved rooms while Main's own history refuses (#3294).

Main is every chat no Project reserves, so while a row of the Project registry cannot be read (the strict
identity read fails: the file lost beside its commit witness, unreadable, or one row whose identity cannot be
read) Main's history answers the typed refusal (``history_paging.RegistryUnavailable``, 503
``project_registry: "unavailable"``). This archive shows what positive evidence alone places, for reading only.

Placement is the ordinary rule (``room_membership.room_membership``) over what the saved records prove: the
rooms ``recognized_rooms`` offers and the room each task's own binding names (a binding is immutable: that work's
placement, when its project and chat are a recognized room; never its lineage, a scope or a shape). Before the
rule, a candidate must be one the archive reads at all (its ``readable`` gate): an exact stored room that is no
hidden, A2A, private, Presence or place row; plain, or a fixed notice (``project_dialogue.ROOM_FIXED_ROW_TYPES``
but Project question pointers); an owner's own words, or
the plain words of work a readable binding places. So Main shows the owner's plain messages typed there and the
fixed notices written there; a recognized room the owner's words typed in it, the fixed notices written to it,
its bound work's plain words and notices, an authorized relay's answer to that work in its source room or that
work's room, and the owner message each binding names (by its client id, or as the one row its exact chat,
time and digest identify).

Nothing else is shown, and its absence says nothing. Most of Ouroboros's ordinary replies carry no typed
placement of their own: a reply of work no readable binding places stays unshown (a missing binding is no
placement), as do every typed row but the fixed notices (questions, answers, closures, delivered media, files
and links, cards, summaries), task children and rows without an exact stored room. ``complete`` is therefore
always false; the ``limits`` of a page are facts of its read, never counts.

Room discovery is bounded (``_EVIDENCE_ROWS`` notices, ``_STORES`` shelves), and a bound is not a proof: a
proposed room standing in another chat than its project's derived address survives a truncated window only
withheld, named in the page's ``rooms_withheld_unverified`` limit -- an unseen notice or marker of that id
would claim the derived address and contradict the proposal. A room at its project's derived address needs
no such check: only the fully read registry and bindings, or a ``project_chat_id`` collision, could
contradict it, and unseen bounded records always claim derived addresses.

Registry and binding reads and directory discovery precede the page's one ``dialogue_history_evidence.saved_history_snapshot``;
only marker files selected by the snapshot's activity and claims are read inside it. A read grants no credit and
starts no cutover or repair. A row carries no row address, task, quiz or reply identity, so it can be read,
never answered, continued, moved or linked; a room read here is not reopened for work. Pages read the
chat stream's coordinates newest first through indexed probes, at most ``_CANDIDATE_CEILING`` candidates a
request: a stretch with nothing to show answers its continuation like any other, and a page never reads below the
archive import's frontier. A page bounds the bytes it decodes, sharing the allowance with room discovery and
legacy origins. Only the last page body may cross it, whole; the next cursor continues before the last examined
candidate. A cursor holds its room's view: the facts that select its rows (the room's project,
its bound tasks, their origins). When they changed, a continuation answers ``ArchiveCursorChanged`` and the reader
starts again from the newest page.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import pathlib
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

from ouroboros import dialogue_history_evidence
from ouroboros import dialogue_store as ds
from ouroboros import store
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID, is_a2a_chat_id, project_chat_id
from ouroboros.project_facts import sanitize_project_id
from ouroboros.project_registry_recovery import history_generation
from ouroboros.store import StoreError, StoreLost, StoreMigrating
from ouroboros.store_capture import capture

MAIN = "main"
PAGE_ROWS = 50  # shown rows a page holds at most
_PROBE_ROWS = 200  # candidates one indexed probe reads
_CANDIDATE_CEILING = 2000  # candidates one request examines; its cursor continues where it stopped
_EVIDENCE_ROWS = 200  # Main's newest pinned host rows read to recognize rooms
_STORES = 2000  # project stores whose registry-authored marker is read
_ORIGIN_BUDGET = 2000  # indexed seeks and stored rows inspected for origins predating client ids
_READ_BYTES = 512 * 1024  # like ordinary history search; actual JSON, not allocated byte coordinates
_PRELUDE_SHARE = 4  # discovery and legacy proof can each spend at most a quarter; pagination spends what's left
_CURSOR_CHARS = 256
_NAME_MAX = 80
_TASK_KEYS = ("task_id", "parent_task_id", "root_task_id", "subagent_task_id")


class ArchiveRoomUnknown(LookupError):
    """The room the read names is not one the evidence recognizes."""


class ArchiveCursorError(ValueError):
    """A cursor this archive never gave for that room."""


class ArchiveCursorChanged(LookupError):
    """A cursor this archive gave, whose room's selecting evidence has changed since: start again."""


def registry_reservations(drive_root: Any) -> Optional[Set[int]]:
    """The one readiness answer Main's history and this archive decide by: the chats the registry's strict
    identity read reserves, or ``None`` when that read fails on a confirmed profile (lost or unreadable).
    Dialogue history still importing, which first use waits for, raises ``StoreMigrating``; an unconfirmed
    profile generation (a Reset still cleaning up, an unreadable witness) raises ``StoreError``."""
    from ouroboros.projects_registry import reserved_project_chat_ids

    try:
        return reserved_project_chat_ids(drive_root, strict=True)
    except (dialogue_history_evidence.ProjectHistoryImportPending, StoreMigrating) as exc:
        raise StoreMigrating(str(exc)) from exc
    except Exception:
        if history_generation(drive_root) != "current":
            raise StoreError("Saved history is unavailable while profile initialization is unconfirmed") from None
        return None


def archive_needed(drive_root: Any) -> bool:
    """Whether Main's history refuses for the registry (``registry_reservations``)."""
    return registry_reservations(drive_root) is None


def _probe(conn: sqlite3.Connection, keys, *, limit: int, before: Optional[int] = None, lower: int = 0) -> List[int]:
    return sorted(ds._probe_seqs(conn, "chat", keys, before=before, lower=lower, limit=limit, newest_first=True),
                  reverse=True)[:limit]


_AT_SEQS = "FROM dialogue WHERE stream='chat' AND seq IN (SELECT value FROM json_each(?))"
# 3.43+ can read a TEXT column's byte count from the record header. Keep the older SQLite floor.
_BODY_SIZE = "octet_length(body)" if sqlite3.sqlite_version_info >= (3, 43) else "length(CAST(body AS BLOB))"


def _body_sizes(conn: sqlite3.Connection, seqs: List[int]) -> Dict[int, int]:
    if not seqs:
        return {}
    # ``length`` is the original allocated span and does not change when text is revised. Read only the
    # actual body sizes here, even for records the byte cut holds for the next page.
    return dict(capture(conn, f"SELECT seq, {_BODY_SIZE} AS size " + _AT_SEQS,
                        (json.dumps(seqs),), ("seq", "size")).tuples())


def _bodies(conn: sqlite3.Connection, seqs: List[int]):
    if seqs:
        bodies = dict(capture(conn, "SELECT seq, body " + _AT_SEQS, (json.dumps(seqs),), ("seq", "body")).tuples())
        for seq in seqs:
            # Only the saved row is used for placement or display, not dispatch inputs or source-pointer projections.
            yield seq, ds.decode(bodies[seq])


def _project_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and sanitize_project_id(value) == value


def _written(row: Dict[str, Any]) -> Optional[int]:
    """The room a row was written to, only as an exact stored integer: a missing or other value is none."""
    chat = row.get("chat_id")
    return chat if type(chat) is int else None


def _kind(row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """``(typed as stored, kind)``: a plain row has no ``type`` (absent, null or empty); a typed row a non-empty
    string. Any other stored value is neither plain nor a kind, so no reader treats it as either."""
    kind = row.get("type")
    if kind is None or kind == "":
        return True, None
    return (True, kind) if isinstance(kind, str) else (False, None)


def _foreign(row: Dict[str, Any]) -> bool:
    """Private contact, a place's row or a row carrying contact: another audience's, never an archive row."""
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.presence_owner_contact import private_main_row

    return bool(row.get("presence_contact") or row.get("place_origin")
                or private_main_row(dict(row)) or row_provenance(row) == "place")


# The pinned notice's address and label, and every field the existing audience checks read. Text,
# attachments and completion payloads are not discovery evidence.
_NOTICE_FIELDS = ("chat_id", "role", "type", "project_id", "project_name", "presence_contact", "place_origin",
                  "presence_provenance", "presence", "source", "quiz", "task_id")
_JSON_FIELD = ("CASE type WHEN 'object' THEN json(value) WHEN 'array' THEN json(value) "
               "WHEN 'true' THEN json('true') WHEN 'false' THEN json('false') ELSE atom END")


def _notice_rows(conn: sqlite3.Connection, seqs: List[int]):
    """Only notice metadata crosses into Python. Preserve JSON types and legacy duplicate-key order:
    the last stored field wins, as it does in the full body's decoder."""
    if not seqs:
        return
    marks = ",".join("?" for _ in _NOTICE_FIELDS)
    # A legacy contact can live in the quiz block. Keep the contact but not the question, choices or materials.
    values = (f"CASE WHEN key='quiz' AND type='object' THEN (SELECT json_group_object(key, {_JSON_FIELD}) FROM ("
              "SELECT key, type, value, atom FROM json_each(notice.value) WHERE key='presence_contact' ORDER BY id"
              ")) ELSE notice.value END")
    sql = (f"SELECT CAST((SELECT json_group_object(key, {_JSON_FIELD}) FROM ("
           f"SELECT key, type, {values} AS value, atom FROM json_each(dialogue.body) AS notice "
           f"WHERE key IN ({marks}) ORDER BY id)) AS BLOB) {_AT_SEQS} ORDER BY seq DESC")
    remaining = _READ_BYTES // _PRELUDE_SHARE
    for (body,) in conn.execute(sql, (*_NOTICE_FIELDS, json.dumps(seqs))):
        size = len(body)
        remaining -= size
        if remaining < 0:
            # Missing evidence could be a competing claim. Do not infer room membership from a cut.
            raise StoreError("saved-room discovery metadata exceeds its read allowance")
        yield ds.decode(body), size


class _Evidence(NamedTuple):
    """What the saved records state, each fact read by itself: ``claims`` (project id, chat, kind, name) of
    exact addresses, ``refused`` ids a record withholds and ``contested`` chats a record whose id cannot be read
    reserves."""

    claims: List[Tuple[str, int, str, str]]
    refused: Set[str]
    contested: Set[int]


def _registry_evidence(root: pathlib.Path) -> _Evidence:
    """The raw registry rows' own facts, any lifecycle, none when the file is lost or unreadable.

    Each row is read field by field, so a fact it states survives a malformed neighbouring field: an identity
    marked deleted, or named twice, is withheld whatever else its row holds. Only an exact address confers a
    claim (an absent ``chat_id`` is the registry's own default, ``project_chat_id``); an identity whose own
    address, or lifecycle, cannot be read is withheld; a reservation whose identity cannot be read contests its
    chat for every id. Nothing is normalized or written back."""
    from ouroboros.project_admission import _registry_snapshot
    from ouroboros.projects_registry import PROJECT_ACTIVE, PROJECT_LIFECYCLES, PROJECT_TOMBSTONED

    found = _Evidence([], set(), set())
    try:
        rows = _registry_snapshot(root, allow_missing=True)[0]["projects"]
    except Exception:
        return found
    seen: Set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue  # nothing it says ties to an identity or a chat
        pid = raw.get("id")
        chat = raw.get("chat_id", project_chat_id(pid) if _project_id(pid) else None)
        exact = type(chat) is int and chat > 0
        if not _project_id(pid):
            if exact:
                found.contested.add(chat)
            continue
        lifecycle = raw.get("lifecycle", PROJECT_ACTIVE)
        known = isinstance(lifecycle, str) and lifecycle in PROJECT_LIFECYCLES  # a non-string withholds this id alone
        if pid in seen or not exact or not known or lifecycle == PROJECT_TOMBSTONED:
            found.refused.add(pid)
        seen.add(pid)
        if exact:
            found.claims.append((pid, chat, "registry", raw.get("name") if isinstance(raw.get("name"), str) else ""))
    return found


def _bindings(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The bindings file when it exists and parses; ``None`` otherwise: then no binding places anything."""
    from ouroboros.projects_registry import _bindings_path
    from ouroboros.utils import json_snapshot

    path = _bindings_path(root)
    try:
        data = json_snapshot(path, strict=True) if path.is_file() else None
    except (ValueError, OSError):
        return None
    return data["bindings"] if data and isinstance(data.get("bindings"), dict) else None


def _binding_target(row: Any) -> Optional[Tuple[str, int]]:
    if not isinstance(row, dict) or not _project_id(row.get("project_id")) or type(row.get("project_chat_id")) is not int:
        return None
    return row["project_id"], row["project_chat_id"]


def _binding_evidence(bindings: Optional[Dict[str, Any]], found: _Evidence) -> Set[str]:
    """Each binding claims its project's chat when both are exact; a binding naming an id with an unreadable
    chat withholds that id, and one naming a chat for an unreadable id contests that chat. The ids claimed."""
    bound = set()
    for row in (bindings or {}).values():
        if not isinstance(row, dict):
            continue
        if target := _binding_target(row):
            found.claims.append((*target, "binding", ""))
            bound.add(target[0])
        elif _project_id(row.get("project_id")):
            found.refused.add(row["project_id"])
        elif type(row.get("project_chat_id")) is int:
            found.contested.add(row["project_chat_id"])
    return bound


def _store_candidates(root: pathlib.Path) -> Tuple[Dict[str, int], bool]:
    """Candidate marker names and their derived chats, and whether the listing outran that bound (then an
    unscanned shelf's claims are unseen). A directory name alone never claims a room."""
    found: Dict[str, int] = {}
    try:
        with os.scandir(root / "projects") as entries:
            for entry in itertools.islice(entries, _STORES):
                if not _project_id(entry.name):
                    continue
                try:
                    if entry.is_dir():
                        found[entry.name] = project_chat_id(entry.name)
                except OSError:
                    continue
            return found, next(entries, None) is not None
    except FileNotFoundError:
        return found, False  # no shelves at all: the discovery is complete
    except OSError:
        return found, True  # an unreadable listing may have missed shelves


class _RoomFiles(NamedTuple):
    bindings: Optional[Dict[str, Any]]
    registry: _Evidence
    stores: Dict[str, int]
    stores_truncated: bool


@contextmanager
def _saved_room_snapshot(
    root: pathlib.Path, *, bindings: Optional[Dict[str, Any]] = None, read_bindings: bool = False,
) -> Iterator[Tuple[sqlite3.Connection, _RoomFiles]]:
    """Stage file evidence before borrowing the dialogue snapshot, never across a store replacement."""
    path = ds.path(root)
    identity = store.file_identity(path)
    stores, stores_truncated = _store_candidates(root)
    files = _RoomFiles(_bindings(root) if read_bindings else bindings, _registry_evidence(root), stores,
                       stores_truncated)
    if store.file_identity(path) != identity:
        raise StoreLost("the dialogue store changed during saved-room discovery")
    with dialogue_history_evidence.saved_history_snapshot(root) as conn:
        if store.file_identity(path) != identity:
            raise StoreLost("the dialogue store changed before the archive snapshot")
        yield conn, files


def recognized_rooms(drive_root: Any, *, bindings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Historical Project rooms the saved records positively name, newest activity first.

    Each record claims a project id in one chat: a registry row (any lifecycle) the chat it reserves; a binding
    the chat it names; a registry-authored store marker and one of Main's pinned host rows
    (``MAIN_PINNED_ROW_TYPES``, newest ``_EVIDENCE_ROWS``) the id's ``project_chat_id``. A room is offered only for
    one id claimed in exactly one chat that no other id claims and no record whose id cannot be read reserves,
    whose id no record withholds (``_registry_evidence``, ``_binding_evidence``): a contradiction leaves the
    affected identities out, and no marker or binding outweighs it; uncontradicted neighbours stay. A room with
    nothing to read is left out too. Neither a folder alone, a chat id's size, a task's project scope nor a link in
    a message's words names a room. The list may lack rooms whose records are lost, and may name a room deleted
    while the registry that said so is unreadable."""
    root = pathlib.Path(drive_root)
    with _saved_room_snapshot(root, bindings=bindings) as (conn, files):
        return _recognized_rooms(conn, root, files)[0]


def _recognized_rooms(conn, root, files: _RoomFiles):
    from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES
    from ouroboros.projects_registry import _PROJECT_MARKER_NAME
    from ouroboros.utils import json_snapshot

    evidence = files.registry
    bound = _binding_evidence(files.bindings, evidence)
    claims = evidence.claims
    kinds = [("kind", kind) for kind in sorted(MAIN_PINNED_ROW_TYPES)]
    spent = 0
    probed = _probe(conn, kinds, limit=_EVIDENCE_ROWS + 1)  # one row of lookahead: is the window complete?
    notices_truncated = len(probed) > _EVIDENCE_ROWS
    for row, size in _notice_rows(conn, probed[:_EVIDENCE_ROWS]):
        spent += size
        kind = _kind(row)[1]
        if (_written(row) == WEB_UI_CHAT_ID and row.get("role") == "system" and kind in MAIN_PINNED_ROW_TYPES
                and _project_id(row.get("project_id")) and not _foreign(row)):
            claims.append((row["project_id"], project_chat_id(row["project_id"]), "main_record",
                           row.get("project_name") if isinstance(row.get("project_name"), str) else ""))
    stores = files.stores
    claimed_ids = {pid for pid, _, _, _ in claims}
    claimed_chats = {chat for _, chat, _, _ in claims}
    candidates = json.dumps(list(claimed_chats | set(stores.values())))
    activity = dict(conn.execute("SELECT chat_id, activity_at FROM rooms WHERE chat_id IN "
                                 "(SELECT value FROM json_each(?))", (candidates,)))
    for pid, chat in stores.items():
        if chat not in activity and chat not in claimed_chats and pid not in claimed_ids:
            continue  # no rows to show and neither side of another claim to confirm or contest
        path = root / "projects" / pid / _PROJECT_MARKER_NAME
        try:
            marker = json_snapshot(path) if path.is_file() else None
        except OSError:
            continue
        if marker and marker.get("id") == pid:
            claims.append((pid, chat, "marker", str(marker.get("name") or "")))
    chats_of: Dict[str, set] = {}
    owners: Dict[int, set] = {}
    names: Dict[str, Dict[str, str]] = {}
    for pid, chat, kind, name in claims:
        chats_of.setdefault(pid, set()).add(chat)
        owners.setdefault(chat, set()).add(pid)
        names.setdefault(pid, {}).setdefault(kind, " ".join(name.split()))
    found = []
    withheld = 0
    bounded = notices_truncated or files.stores_truncated
    for pid, chats in chats_of.items():
        chat = next(iter(chats))
        if (pid in evidence.refused or len(chats) != 1 or owners[chat] != {pid} or chat in evidence.contested
                or chat in (WEB_UI_CHAT_ID, HIDDEN_CHAT_ID) or is_a2a_chat_id(chat)
                or (chat not in activity and pid not in bound)):
            continue
        if bounded and chat != project_chat_id(pid):
            # A notice or marker of this id outside the window would claim its derived address and
            # contradict this proposal; the window cannot prove there is none. Withhold, and say so.
            withheld += 1
            continue
        said = names[pid]
        label = next((said[kind] for kind in ("registry", "marker", "main_record") if said.get(kind)), pid)
        found.append({"room": str(chat), "chat_id": chat, "project_id": pid,
                      "name": label if len(label) <= _NAME_MAX else label[:_NAME_MAX - 1].rstrip() + "…",
                      "evidence": sorted(said)})
    found.sort(key=lambda room: (room["name"].casefold(), room["project_id"]))
    found.sort(key=lambda room: activity.get(room["chat_id"], ""), reverse=True)
    return found, spent, withheld


def _origin_refs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The owner messages bindings name by an exact relation, as the host's typed identity: an exact chat, the
    words' digest and the message's client id or, for a binding older than client ids, its time (each a string
    as the host stores it). Only those four fields travel on. A digest alone would select by coincident words,
    and a field of another type is no identity, so either ref names nothing."""
    refs = []
    for row in rows:
        ref = row.get("source_ref")
        if (not isinstance(ref, dict) or type(ref.get("chat_id")) is not int
                or not 0 < ref["chat_id"] < (1 << 63)):
            continue
        fields = {key: ref.get(key) for key in ("client_message_id", "ts", "text_sha256")}
        if any(value is not None and not isinstance(value, str) for value in fields.values()):
            continue
        client, ts, digest = (str(value or "") for value in fields.values())
        if digest and (client or ts):
            refs.append({"chat_id": ref["chat_id"], "client_message_id": client, "ts": ts, "text_sha256": digest})
    return refs


class _Reader(NamedTuple):
    """One room's read: its indexed probe ``keys``, the ``extra`` candidates an identity lookup found (a
    binding's origin named by chat and time), the rule that admits a candidate record, and the room's ``view``."""

    keys: List[Tuple[str, Any]]
    extra: Set[int]
    admit: Callable[[int, Dict[str, Any]], bool]
    view: str
    origin_bytes: int = 0
    origins_limited: bool = False


def _view(*facts: Any) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]


def _legacy_origins(conn, refs, readable) -> Tuple[Set[int], int, bool]:
    """Prove uniqueness over the ordinary owner's full indexed address scope before filtering for display."""
    from ouroboros.project_dialogue import _entry_source_identities, legacy_owner_source_chat_keys

    places: Dict[Tuple[int, str], set] = {}
    for ref in refs:
        if not ref["client_message_id"]:
            places.setdefault((ref["chat_id"], ref["ts"]), set()).add(ref["text_sha256"])
    if places and conn.execute(
        "SELECT EXISTS(SELECT 1 FROM import_ledger INDEXED BY ledger_state WHERE stream='chat' "
        "AND state IN ('pending','mismatch')) OR EXISTS(SELECT 1 FROM skipped WHERE stream='chat')"
    ).fetchone()[0]:
        return set(), 0, False  # unread chat rows can still share an origin's time and words
    origins: Set[int] = set()
    remaining, spent, limited = _ORIGIN_BUDGET, 0, False
    allowance = _READ_BYTES // _PRELUDE_SHARE
    seek = "SELECT seq FROM dialogue INDEXED BY dialogue_room_ts WHERE stream='chat' AND chat_id IS ? AND ts=?"
    for (chat, ts), digests in sorted(places.items()):
        if remaining < 2:
            limited = True
            break
        chats = legacy_owner_source_chat_keys(chat)
        limit = min(_PROBE_ROWS + 1, remaining - len(chats))
        if limit <= 0:  # this address has more seeks than the remaining allowance can prove
            limited = True
            continue
        # Each arm is an exact index seek; one lookahead covers their union, including the legacy NULL bucket.
        query = " UNION ALL ".join([seek] * len(chats)) + " LIMIT ?"
        params = [part for key in chats for part in (key, ts)]
        seqs = [seq for (seq,) in conn.execute(query, (*params, limit))]
        remaining -= len(chats) + len(seqs)
        size = sum(_body_sizes(conn, seqs).values()) if len(seqs) < limit else 0
        if len(seqs) == limit or spent + size > allowance:
            limited = True
            continue  # an unseen row can have exactly the same identity
        spent += size
        matches = {}
        for seq, row in _bodies(conn, seqs):
            # One row contributes its legacy identity once, even if it has a newer physical Reply key.
            identities = {key[:4] for key in _entry_source_identities(row)}
            for cid, client, stamp, digest in identities:
                if cid == chat and not client and stamp == ts and digest in digests:
                    matches[digest] = None if digest in matches else (seq, row)
        for match in matches.values():
            if match is None:
                continue
            seq, row = match
            if (_written(row) == chat and row.get("ts") == ts and _kind(row) == (True, None)
                    and row.get("direction") == "in" and readable(row, chat)):
                origins.add(seq)
    return origins, spent, limited


def _room_reader(conn: sqlite3.Connection, room: str, rooms: List[Dict[str, Any]],
                 bindings: Optional[Dict[str, Any]]) -> _Reader:
    """How one room is read: candidates by index, then its ``readable`` gate and the ordinary membership rule over the
    recognized rooms and the bindings whose (project, chat) is a recognized room. The view holds only what selects
    this room's rows: Main's never depends on a binding (its work speech joins Main by its written address, its
    bound work leaves it); a Project's is its project, the tasks bound to it, the origins their bindings name and
    the rows those origins identify."""
    from ouroboros.project_dialogue import (
        ROOM_FIXED_ROW_TYPES,
        WORK_SPEECH_ELSEWHERE_TYPES,
        entry_matches_source_ref,
        project_origin_identity,
    )
    from ouroboros.room_membership import room_membership
    from ouroboros.room_messages import counts

    fixed = ROOM_FIXED_ROW_TYPES - {"project_question_pointer"}
    pairs = {(entry["project_id"], entry["chat_id"]) for entry in rooms}
    placed = {str(task): target[1] for task, row in (bindings or {}).items()
              if (target := _binding_target(row)) in pairs}

    def resolve(task: Any = None, *_lineage: Any) -> int:  # the room rule over direct, recognized bindings alone
        return placed.get(task, 0) if isinstance(task, str) else 0

    def readable(row: Dict[str, Any], written: Optional[int]) -> bool:
        typed, kind = _kind(row)
        if not typed or written is None or written == HIDDEN_CHAT_ID or is_a2a_chat_id(written) or _foreign(row):
            return False
        if kind:
            return kind in fixed
        if not any(row.get(key) for key in _TASK_KEYS):
            return row.get("direction") == "in"  # the owner's own words; an untasked reply names no placement
        return bool(resolve(row.get("task_id")) and counts(row))

    chats = {entry["chat_id"] for entry in rooms}
    if room == MAIN:
        member = room_membership(WEB_UI_CHAT_ID, chats, [], resolve)

        def admit_main(_seq: int, row: Dict[str, Any]) -> bool:
            written = _written(row)
            return written == WEB_UI_CHAT_ID and readable(row, written) and member(written, row)

        return _Reader([("chat_id", WEB_UI_CHAT_ID)], set(), admit_main, _view(MAIN))
    match = next((entry for entry in rooms if entry["room"] == room), None)
    if match is None:
        raise ArchiveRoomUnknown(room)
    chat = match["chat_id"]
    bound = sorted(task for task, row in (bindings or {}).items() if _binding_target(row) == (match["project_id"], chat))
    refs = _origin_refs([bindings[task] for task in bound])
    named: Dict[str, list] = {}
    for ref in refs:
        if ref["client_message_id"]:
            named.setdefault(ref["client_message_id"], []).append(ref)
    origins, origin_bytes, origins_limited = _legacy_origins(conn, refs, readable)
    member = room_membership(chat, chats, [], resolve)

    def admit(seq: int, row: Dict[str, Any]) -> bool:
        written = _written(row)
        if not readable(row, written):
            return False
        if seq in origins:
            return True
        kind = _kind(row)[1]
        client = row.get("client_message_id")
        if (kind is None and row.get("direction") == "in" and isinstance(client, str)
                and client in named and entry_matches_source_ref(row, named[client])):
            return True
        # A direct binding moves plain work from any owner transport. Only work speech can move a fixed notice.
        placed = resolve(row.get("task_id")) == chat and (kind is None or kind in WORK_SPEECH_ELSEWHERE_TYPES)
        return (written == chat or placed) and member(written, row)

    keys = [("chat_id", chat), *(("task_id", task) for task in bound),
            *(("client_message_id", client) for client in named)]
    view = _view(chat, match["project_id"], bound, sorted(project_origin_identity(ref) for ref in refs), sorted(origins))
    return _Reader(keys, origins, admit, view, origin_bytes, origins_limited)


def _archive_row(seq: int, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return inert words and attachment names; a pinned completion keeps its saved label, status and detail."""
    from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES
    from ouroboros.utils import strip_markdown

    direction = str(row.get("direction") or "").lower()
    role = "system" if row.get("role") == "system" else {"in": "user", "out": "assistant", "system": "system"}.get(direction)
    if role is None:
        return None
    text = "" if row.get("text_placeholder") is True else str(row.get("text") or "")
    if row.get("type") in MAIN_PINNED_ROW_TYPES:
        text = strip_markdown(text)
    names = [str(ref.get("name") or "file") for ref in row.get("attachments") or [] if isinstance(ref, dict)] \
        if role == "user" and isinstance(row.get("attachments"), list) else []
    if not text.strip() and not names:
        return None
    return {"id": f"archive:chat:{seq}", "role": role, "text": text, "ts": str(row.get("ts") or ""),
            **({"kind": row["type"]} if row.get("type") in MAIN_PINNED_ROW_TYPES else {}),
            **({"attachments": names} if names else {})}


def _cursor(room: str, before: int, view: str) -> str:
    raw = json.dumps({"v": 1, "room": room, "view": view, "before": int(before)}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _cursor_token(cursor: Any, room: str) -> Optional[Tuple[int, str]]:
    """``(before, view)`` of a cursor this archive gave for ``room``; ``None`` for none."""
    if cursor is None or cursor == "":
        return None
    try:
        if not isinstance(cursor, str) or len(cursor) > _CURSOR_CHARS:
            raise ValueError("cursor")
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except ValueError:
        raise ArchiveCursorError("archive_cursor_invalid") from None
    if (not isinstance(value, dict) or value.get("v") != 1 or value.get("room") != room
            or type(value.get("before")) is not int or value["before"] < 0
            or not isinstance(value.get("view"), str) or not value["view"]):
        raise ArchiveCursorError("archive_cursor_invalid")
    return value["before"], value["view"]


def _candidates(conn: sqlite3.Connection, reader: _Reader, position: int, floor: int,
                limit: int) -> Tuple[List[int], bool]:
    """Indexed coordinates and proven origins, merged before any body is fetched. One extra coordinate tells
    whether the probe reached the floor even when the last available row fills the work allowance."""
    batch = _probe(conn, reader.keys, before=position, lower=floor, limit=limit + 1)
    reached = len(batch) <= limit
    bottom = floor if reached else batch[limit - 1]
    merged = sorted(set(batch[:limit]) | {seq for seq in reader.extra if bottom <= seq < position}, reverse=True)
    return merged[:limit], reached and len(merged) <= limit


def archive_page(drive_root: Any, room: str = MAIN, cursor: Any = None) -> Dict[str, Any]:
    """One page of a room's archive, oldest first, read below ``cursor`` (the newest when none).

    ``{"state": "not_needed"}`` while the registry's rooms read (Main's own history answers). A store that
    cannot answer raises its typed ``StoreError``; a store not prepared yet, or a cursor at the import frontier,
    raises ``StoreMigrating``; a cursor whose room's view changed (or whose room is no longer recognized) raises
    ``ArchiveCursorChanged``."""
    root = pathlib.Path(drive_root)
    if not archive_needed(root):
        return {"state": "not_needed"}
    with _saved_room_snapshot(root, read_bindings=True) as (conn, files):
        return _page(conn, root, room, cursor, files)


def _page(conn, root, room, cursor, files: _RoomFiles):
    from ouroboros.gateway.history_paging import import_floor

    room = str(room or MAIN)
    token = _cursor_token(cursor, room)
    stream = conn.execute("SELECT next_seq FROM streams WHERE stream='chat'").fetchone()
    created = ds._meta(conn, "created_at")
    if stream is None and (created is not None or ds._meta(conn, "cutover") is not None):
        raise StoreLost("the dialogue store has no chat coordinates")
    end, floor = (stream[0] if stream else 0), import_floor(root, "chat", snapshot=conn)
    rooms, discovery_bytes, withheld = _recognized_rooms(conn, root, files)
    try:
        reader = _room_reader(conn, room, rooms, files.bindings)
    except ArchiveRoomUnknown:
        if token is not None:
            raise ArchiveCursorChanged(room) from None
        raise
    view = _view(created, reader.view)
    before = None if token is None else token[0]
    if token is not None and token[1] != view:
        raise ArchiveCursorChanged(room)
    if before is not None and before > end:
        raise ArchiveCursorError("archive_cursor_invalid")
    if before is not None and floor and before <= floor:
        raise StoreMigrating("older saved history is still being imported")
    position = end if before is None else before
    shown: List[Dict[str, Any]] = []
    examined, exhausted = 0, position <= floor
    spent = discovery_bytes + reader.origin_bytes
    while not exhausted and len(shown) < PAGE_ROWS and examined < _CANDIDATE_CEILING and spent < _READ_BYTES:
        batch, reached = _candidates(conn, reader, position, floor, min(_PROBE_ROWS, _CANDIDATE_CEILING - examined))
        sizes = _body_sizes(conn, batch)
        selected = []
        for seq in batch:
            if spent >= _READ_BYTES:
                break
            selected.append(seq)
            spent += sizes[seq]
        for seq, row in _bodies(conn, selected):
            examined, position = examined + 1, seq
            message = _archive_row(seq, row) if reader.admit(seq, row) else None
            if message:
                shown.append(message)
            if len(shown) >= PAGE_ROWS:
                break
        exhausted = reached and (not batch or position == batch[-1])
    importing = exhausted and floor > 0
    has_more = not exhausted or importing
    limits = ["placement_unproven", *(["bindings_unavailable"] if files.bindings is None else []),
              *(["rooms_withheld_unverified"] if withheld else []),
              *(["legacy_origins_limited"] if reader.origins_limited else []),
              *(["importing"] if importing else [])]
    return {"state": "available", "room": room, "rooms": [{"room": MAIN, "name": "Main"}, *rooms],
            "messages": shown[::-1], "has_more": has_more,
            "next_cursor": _cursor(room, floor if importing else position, view) if has_more else None,
            "complete": False, "limits": limits}


__all__ = ["MAIN", "PAGE_ROWS", "ArchiveCursorChanged", "ArchiveCursorError", "ArchiveRoomUnknown", "archive_needed",
           "archive_page", "recognized_rooms", "registry_reservations"]
