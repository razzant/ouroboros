"""Attention facts on accepted room rows, and indexed place activity projections.

No event ledger: the existing dialogue dedup record owns the intake decision.
The task store continues to own every focus that actually started.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

from ouroboros import dialogue_store, store

_PLACE = "json_extract(body, '$.place_id')"
_ARRIVAL = "stream='chat' AND kind='place_arrival'"
_OPEN = "status NOT IN ('completed','failed','cancelled')"
# Imported earlier messages (``presence_history``) are arrivals carrying an immutable marker.
_IMPORTED = _ARRIVAL + " AND json_extract(body, '$.historical_import') IS NOT NULL"
# The reading order of a room holding imports (``gateway.history_chronology``): a row's provider send time and
# order from its immutable ``source_order`` block, else its ``ts`` and no order (-1, before every order); then
# its stored position. ``dialogue_room_reading`` keys these very expressions; ``reading_key`` mirrors them.
READING_TIME = "COALESCE(json_extract(body,'$.source_order.time'),ts)"
READING_ORDER = "COALESCE(json_extract(body,'$.source_order.order'),-1)"


def reading_key(row, position):
    """``(time, order, position)`` of a stored row, as ``READING_TIME`` and ``READING_ORDER`` read it (-1: no order)."""
    block = row.get('source_order') if isinstance(row.get('source_order'), dict) else {}
    time, order = block.get('time'), block.get('order')
    return (dialogue_store.facts(row)['ts'] if time is None else time, -1 if order is None else order, position)


class SourceProjection(dict):
    """A reader projection retains its original immutable row for source addressing."""
    def __init__(self, body, source):
        super().__init__(body, context_source_ref=source)
        self.source_body = body


def project_source(record):
    """Project a deferred source pointer without mutating the accepted source bytes."""
    body = dialogue_store.decode(record["body"])
    if body.get("type") != "place_arrival" or "dispatch_input" not in record.keys():
        return body
    extra = json.loads(record["dispatch_input"] or "{}")
    source = extra.get("place_context_source_ref") if isinstance(extra, dict) else None
    return SourceProjection(body, source) if isinstance(source, dict) else body


@contextmanager
def intake_lock(root, conversation_key):
    from ouroboros.presence_runner import _configured_gate
    gate = _configured_gate(Path(root))
    key, lock = gate._conversation("intake:" + conversation_key)
    with lock, gate._file_lock(gate._conversation_file(key)):
        yield


def _record(conn, task_id):
    return conn.execute("SELECT seq, body, dispatch_input FROM dialogue INDEXED BY dialogue_task "
                        "WHERE stream='chat' AND task_id=? AND direction='in' ORDER BY seq LIMIT 1",
                        (task_id,)).fetchone()


def heard_replay(root, task_id, identity="", continuation_version=0):
    """A heard arrival's final silent answer. An imported earlier message (its immutable
    ``historical_import`` marker, ``presence_history``) is heard too: a live replay of that
    revision answers silent before any attention, author mail or runner."""
    from ouroboros.presence_runner import PresenceTurnError, PresenceTurnResult
    if not dialogue_store.present(root):
        return None
    with dialogue_store.transaction(root, write=False) as conn:
        row = _record(conn, task_id)
    if row is None:
        return None
    facts = json.loads(row['dispatch_input'] or '{}').get('place_attention', {})
    body = dialogue_store.decode(row['body'])
    imported = isinstance(body.get('historical_import'), dict)
    if facts.get('decision') != 'heard' and not imported:
        return None
    if identity and body.get('presence_event_identity') != identity:
        raise PresenceTurnError('presence_event_identity_conflict', 'source_event_id', turn_ref=task_id)
    return PresenceTurnResult(outcome='silent', text='', task_id=task_id,
                              delivery_reporting_version=0 if imported else facts.get('delivery_reporting_version', 0),
                              continuation_version=continuation_version)


def record_heard(root, task, event, reason, *, captured_task_id=None):
    from ouroboros.presence_runner import _stored_turn, presence_result_from_stored
    from dataclasses import replace
    identity = task['metadata']['presence_event_identity']
    arrival_id = captured_task_id or task['id']
    with intake_lock(root, event.conversation_key):
        # A certified retry has a new physical id, but custody still belongs to
        # that attempt while the immutable arrival keeps its original id.
        stored = _stored_turn(root, task['id'], identity)
        if stored:
            if stored.get('status') not in {'completed', 'failed'}:
                return None  # admitted custody is never overwritten by an attention decision
            return replace(presence_result_from_stored(stored, task['id']), continuation_version=event.continuation_version)
        with dialogue_store.transaction(root, write=True) as conn:
            row = _record(conn, arrival_id)
            if row is None or dialogue_store.decode(row['body']).get('presence_event_identity') != identity:
                raise ValueError('Place attention requires its captured event')
            facts = json.loads(row['dispatch_input'] or '{}')
            facts.setdefault('place_attention', {'decision': 'heard', 'reason': reason,
                             'facts': list(event.facts or ()),
                             'delivery_reporting_version': event.delivery_reporting_version})
            conn.execute("UPDATE dialogue SET dispatch_input=? WHERE stream='chat' AND seq=?",
                         (json.dumps(facts), row['seq']))
        return heard_replay(root, arrival_id, identity, event.continuation_version)


def retain_focus_source(root, task_id):
    """The immutable context-source blob of an admitted focus, or of an arrival whose author's mail leaves
    facts out (``place_intake.offer_to_author``); heard-only arrivals need none. Returns its reference,
    the one already retained when there is one.

    Its pointer lives in the existing dedup record, so source row bytes and byte
    coordinates stay unchanged. dialogue_store projects it for every row reader.
    Caller holds intake_lock through this operation (and a focus's running-task write).
    """
    from ouroboros import chat_chain
    with dialogue_store.transaction(root, write=False) as conn:
        row = _record(conn, task_id)
    if row is None:
        raise ValueError('Focus requires its captured event')
    body = dialogue_store.decode(row['body'])
    facts = json.loads(row['dispatch_input'] or '{}')
    if body.get('context_source_ref') or facts.get('place_context_source_ref'):
        return body.get('context_source_ref') or facts['place_context_source_ref']
    source = {'recorded_at': body['ts'], 'source_event_id': body['client_message_id'],
              'text': body['text'], 'transport': body['transport']}
    ref = chat_chain.retain_memory_source(SimpleNamespace(drive_root=root, task_id=task_id), 'presence-event',
        json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode(), 'json')
    with dialogue_store.transaction(root, write=True) as conn:
        conn.execute("UPDATE dialogue SET dispatch_input=json_set(COALESCE(dispatch_input,'{}'), "
                     "'$.place_context_source_ref',json(?)) WHERE stream='chat' AND seq=?",
                     (json.dumps(ref), row['seq']))
    return ref


def _dialogue_indexes(root):
    if not dialogue_store.present(root):
        return False
    with dialogue_store.transaction(root, write=False) as conn:
        found = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name IN "
                                          "('dialogue_place_rooms','dialogue_place_recent','dialogue_place_imported',"
                                          "'dialogue_room_reading')")}
    if len(found) != 4:
        # One-time index build, including pre-upgrade captured arrivals and imported rows a
        # store cutover brought back. No body backfill and no second ledger; later reads only
        # seek these B-trees.
        with dialogue_store.transaction(root, write=True) as conn:
            conn.execute(f"CREATE INDEX IF NOT EXISTS dialogue_place_rooms ON dialogue({_PLACE},chat_id,seq DESC) WHERE {_ARRIVAL}")
            conn.execute(f"CREATE INDEX IF NOT EXISTS dialogue_place_recent ON dialogue({_PLACE},ts DESC) WHERE {_ARRIVAL}")
            conn.execute(f"CREATE INDEX IF NOT EXISTS dialogue_place_imported ON dialogue(chat_id,ts,seq) WHERE {_IMPORTED}")
            conn.execute(f"CREATE INDEX IF NOT EXISTS dialogue_room_reading ON dialogue(stream,chat_id,{READING_TIME},"
                         f"{READING_ORDER},seq) WHERE stream='chat'")
    return True


def holds_history_import(root, chat_id):
    """Whether a room holds imported earlier messages: one seek of their partial index.

    Its room then reads chronologically (``gateway.history_chronology.ChronologicalSource``), which
    seeks ``dialogue_room_reading``: this endpoint read builds the indexes first, as place reads do.
    """
    return _dialogue_indexes(root) and int(chat_id) in history_import_rooms(root, (chat_id,))


def history_import_rooms(root, chat_ids):
    """The rooms among ``chat_ids`` holding imported earlier messages, in one read that writes nothing:
    a seek per room. Until an import or a place read has built their index (``_dialogue_indexes``; a
    store cutover leaves it unbuilt) the store's place arrivals are walked by kind instead, never
    answering "none" for want of an index.

    A fact of the room, whichever of its rows a reader selected: memory reads these rooms by time
    as their endpoint does (``dialogue_provenance.in_reading_order``). Main (1, or no chat id)
    never reads by time; its reads leave arrivals out.
    """
    chats = sorted({int(chat) for chat in chat_ids if chat is not None} - {1})
    if not chats or not dialogue_store.present(root):
        return set()
    with dialogue_store.transaction(root, write=False) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='dialogue_place_imported'").fetchone():
            return {chat for chat in chats if conn.execute(f"SELECT 1 FROM dialogue INDEXED BY dialogue_place_imported "
                                                           f"WHERE {_IMPORTED} AND chat_id=? LIMIT 1", (chat,)).fetchone()}
        held = {row[0] for row in conn.execute(
            f"SELECT DISTINCT chat_id FROM dialogue INDEXED BY dialogue_kind WHERE {_IMPORTED}")}
    return set(chats) & held


def place_room_chats(root):
    """Every room holding captured arrivals: a loose scan of their index, one seek per place and per
    room, whatever the rooms hold. Main never shows an arrival (``room_membership``), so its
    conversation reads leave these rooms out (``gateway.history_paging.room_lens``). A Main read never
    builds the index: until a place read or an import has (``presence_history.record_history``), none
    is left out and membership alone keeps arrivals out of Main."""
    if not dialogue_store.present(root):
        return set()
    rooms, place = set(), None
    with dialogue_store.transaction(root, write=False) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='dialogue_place_rooms'").fetchone() is None:
            return rooms
        while True:
            row = conn.execute(f"SELECT {_PLACE} FROM dialogue INDEXED BY dialogue_place_rooms WHERE {_ARRIVAL} AND "
                               f"{_PLACE} {'IS NOT NULL' if place is None else '>?'} ORDER BY {_PLACE} LIMIT 1",
                               () if place is None else (place,)).fetchone()
            if row is None:
                return rooms
            place, chat = row[0], None
            while True:
                bound = '' if chat is None else ' AND chat_id>?'
                found = conn.execute(f"SELECT chat_id FROM dialogue INDEXED BY dialogue_place_rooms WHERE {_ARRIVAL} "
                                     f"AND {_PLACE}=? AND chat_id IS NOT NULL{bound} ORDER BY chat_id LIMIT 1",
                                     (place,) if chat is None else (place, chat)).fetchone()
                if found is None:
                    break
                chat = int(found[0])
                rooms.add(chat)


def _open_counts(root, place):
    """One aggregate over indexed nonterminal focuses, never task-document loads."""
    if not (Path(root) / store.STORE_REL).exists() or store.migrating(root):
        return None
    with store.open_store(root, write=False) as txn:
        exists = txn.conn.execute("SELECT 1 FROM sqlite_master WHERE name='tasks_place_open'").fetchone()
    if not exists:
        with store.open_store(root, write=True) as txn:
            txn.conn.execute(f"CREATE INDEX IF NOT EXISTS tasks_place_open ON tasks(presence_binding_id,task_id) WHERE {_OPEN}")
    bindings = tuple(dict.fromkeys((place.place_id, *place.legacy_binding_ids)))
    with store.open_store(root, write=False) as txn:
        rows = txn.conn.execute(
            "SELECT json_extract(b.body,'$.chat_id') AS room,COUNT(*) AS count FROM tasks t INDEXED BY tasks_place_open "
            "JOIN task_bodies b ON b.task_id=t.task_id WHERE " + _OPEN +
            " AND t.presence_binding_id IN (" + ','.join('?' for _ in bindings) + ") "
            "AND json_extract(b.body,'$.metadata.wake_cause')='focus' "
            "AND COALESCE(t.parent_task_id,'')='' GROUP BY room", bindings).fetchall()
    return {int(r['room']): int(r['count']) for r in rows if r['room'] is not None}


def _room_details(latest):
    """The room's words from its latest captured arrival; a missing title/name stays null."""
    transport = dialogue_store.decode(latest['body']).get('transport') or {}
    if not isinstance(transport, dict):
        transport = {}
    conversation = transport.get('conversation') or {}
    if not isinstance(conversation, dict):
        conversation = {}
    label = conversation.get('title') or conversation.get('name')
    return {'label': label if isinstance(label, str) else None, 'last_at': str(latest['ts'] or ''),
            'conversation_id': str(transport.get('conversation_id') or ''),
            'thread_id': str(transport.get('thread_id') or '')}


def room_index(root, place, *, details=False):
    """Polls (/api/state, consciousness) read index entries only. ``details`` is the owner's
    rooms read: the room's latest arrival by time, one seek per room (and, in a room holding
    imported messages, a walk of its time index down to the newest one, past only the room's
    own non-arrival rows), the body decoded after the transaction closes."""
    if not _dialogue_indexes(root):
        return []
    counts = _open_counts(root, place)
    rooms, latest = [], []
    # A loose index scan: one seek per captured room, independent of arrival
    # history size. A busy room with a million heard rows still takes one seek.
    with dialogue_store.transaction(root, write=False) as conn:
        last = None
        while True:
            bound = '' if last is None else ' AND chat_id>?'
            args = (place.place_id,) if last is None else (place.place_id, last)
            row = conn.execute(f"SELECT chat_id FROM dialogue INDEXED BY dialogue_place_rooms WHERE {_ARRIVAL} "
                               f"AND {_PLACE}=? AND chat_id IS NOT NULL{bound} ORDER BY chat_id LIMIT 1", args).fetchone()
            if row is None:
                break
            last = int(row[0])
            rooms.append({'room_chat_id': last, 'unanswered_count': None if counts is None else counts.get(last, 0),
                          'address': f'chat:{last}'})
            if details:
                # Imported earlier messages are stored after the live ones they precede, so the
                # room's latest by time is its newest import or a live arrival above it in time.
                imported = conn.execute(f"SELECT body,ts FROM dialogue INDEXED BY dialogue_place_imported WHERE {_IMPORTED} "
                                        f"AND chat_id=? AND {_PLACE}=? ORDER BY ts DESC LIMIT 1",
                                        (last, place.place_id)).fetchone()
                if imported is None:
                    latest.append(conn.execute(f"SELECT body,ts FROM dialogue INDEXED BY dialogue_place_rooms WHERE {_ARRIVAL} "
                                               f"AND {_PLACE}=? AND chat_id=? ORDER BY seq DESC LIMIT 1",
                                               (place.place_id, last)).fetchone())
                else:
                    latest.append(conn.execute(f"SELECT body,ts FROM dialogue INDEXED BY dialogue_room_ts WHERE {_ARRIVAL} "
                                               f"AND chat_id=? AND ts>? AND {_PLACE}=? ORDER BY ts DESC LIMIT 1",
                                               (last, imported['ts'], place.place_id)).fetchone() or imported)
    for room, row in zip(rooms, latest):
        room.update(_room_details(row))
    return rooms


def activity(root, place):
    if not _dialogue_indexes(root):
        return {'unanswered': 0, 'last_activity_at': ''}
    counts = _open_counts(root, place)
    # O(log arrivals + open focuses), not O(all historical arrivals). The
    # newest row is an index-only LIMIT 1; counts decode no task documents.
    with dialogue_store.transaction(root, write=False) as conn:
        row = conn.execute(f"SELECT ts FROM dialogue INDEXED BY dialogue_place_recent WHERE {_ARRIVAL} "
                           f"AND {_PLACE}=? ORDER BY ts DESC LIMIT 1", (place.place_id,)).fetchone()
    return {'unanswered': None if counts is None else sum(counts.values()),
            'last_activity_at': row[0] if row else ''}


def recent_identity(root, place_id, account_id=""):
    """First valid credential declaration among the place's newest 50 captured arrivals (of one account when the call
    names it); None when nothing was declared there or the room cannot be read."""
    import logging
    from ouroboros.places import validate_presence_identity

    try:
        if not _dialogue_indexes(root):
            return None
        with dialogue_store.transaction(root, write=False) as conn:
            # The window is the place's newest 50 arrivals whoever wrote them; a named account is picked out of it
            # after the transaction, so the read never walks further back for a quiet account.
            rows = conn.execute(f"SELECT json_type(body,'$.transport.identity'), json_extract(body,'$.transport.identity'), "
                                f"json_extract(body,'$.transport.account_id') FROM dialogue INDEXED BY dialogue_place_recent "
                                f"WHERE {_ARRIVAL} AND {_PLACE}=? ORDER BY ts DESC LIMIT 50", (place_id,)).fetchall()
    except (store.StoreError, OSError) as exc:
        # A supplementary fact: an own wake or a Safety check goes on without it rather than failing on the store.
        logging.getLogger(__name__).warning("Captured identity unavailable for %s: %s", place_id, exc)
        return None
    for kind, declared, account in rows:
        # json_extract hands back a stored string unquoted and a number natively: only a stored object is a declaration.
        if kind != 'object' or (account_id and account != account_id):
            continue
        try:
            return validate_presence_identity(json.loads(declared))
        except ValueError:
            continue
    return None


def account_candidates(root, place_id):
    """At most 20 people from the newest 50 arrivals: one index seek + 50 rows.

    This is a recent UI suggestion, never owner identity evidence or a history
    scan. It uses the existing place/time index and stores no second actor list.
    The connected account's own imported messages among them suggest no person.
    """
    if not _dialogue_indexes(root):
        return []
    with dialogue_store.transaction(root, write=False) as conn:
        rows = conn.execute(f"SELECT json_extract(body,'$.sender_session_id') AS actor, "
                            "json_extract(body,'$.sender_label') AS label, "
                            "json_extract(body,'$.transport.actor.is_bot') AS bot, "
                            "json_extract(body,'$.historical_import.authorship') AS authorship "
                            f"FROM dialogue INDEXED BY dialogue_place_recent WHERE {_ARRIVAL} "
                            f"AND {_PLACE}=? ORDER BY ts DESC LIMIT 50", (place_id,)).fetchall()
    people = {}
    for row in rows:
        actor = str(row['actor'] or '')
        if actor and not row['bot'] and row['authorship'] != 'platform_account' and actor not in people:
            people[actor] = {'platform_actor_id': actor, 'label': str(row['label'] or actor)}
    return list(people.values())[:20]
