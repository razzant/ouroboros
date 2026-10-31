"""Small owner-history projection of retained Place facts, never an attention decision.

Source rows/addresses stay unchanged. Heard decisions live on their accepted
dialogue rows; an admitted focus lives on its exact task. No wake match is inferred.
An imported earlier message shows its import facts, never owner confirmation or attention.
A row whose arrival recorded its provider send time shows it (``sent_at``); ``ts`` stays the row's own.
"""
from __future__ import annotations

import json

from ouroboros import dialogue_store
from ouroboros.places import PlaceError, get_place, is_owner_account
from ouroboros.store import StoreError


def record_facts(entry):
    kind, direction = entry.get('type', ''), entry.get('direction')
    transport = entry.get('transport') or {}
    if not isinstance(transport, dict):
        return None
    if kind in {'place_notice', 'presence_delivery'}:
        facts = {'kind': 'notice' if kind == 'place_notice' else 'delivery'}
    elif (direction in {'in', 'out'} and kind in {'', 'place_arrival', 'message', 'document', 'photo', 'video', 'links', 'card'}
          and (entry.get('place_id') or entry.get('presence_provenance')
               or str(entry.get('source', '')).startswith('presence:'))):
        facts = {'kind': 'message'}
    else:
        return None  # Progress, task evidence and private Main contact are not room speech.
    delivery = transport.get('delivery') or {}
    if isinstance(delivery, dict) and delivery.get('state') in {'authored', 'accepted', 'delivered', 'failed', 'uncertain'}:
        facts['delivery_state'] = delivery['state']
    for key in ('identity', 'sends_as', 'sends_as_owner'):
        if (value := transport.get(key)) is not None:
            facts[key] = value
    if kind == 'place_notice':
        for key in ('held_text', 'held', 'checked'):
            if (value := entry.get(key)) is not None:
                facts[key] = value
    return facts


def annotate_place_history(messages, entries, root, reads):
    """Only the returned rows, using indexed accepted-row lookups and the page's batch task reader."""
    sources = {entry.get('history_id'): entry for entry in entries or () if record_facts(entry)}
    candidates = [(message, sources[message['history_id']]) for message in messages
                  if message.get('history_id') in sources and not message.get('is_progress')]
    arrivals = {entry['history_position']['offset']: entry for _, entry in candidates
                if entry.get('type') == 'place_arrival' and entry.get('history_position', {}).get('source') == 'chat'
                and not isinstance(entry.get('historical_import'), dict)}
    retained = []
    if arrivals:
        with dialogue_store.transaction(root, write=False) as conn:
            # A page is bounded, but may exceed SQLite's parameter floor.
            offsets = list(arrivals)
            for start in range(0, len(offsets), 500):
                batch = offsets[start:start + 500]
                retained += conn.execute("SELECT seq,dispatch_input FROM dialogue WHERE stream='chat' AND seq IN ("
                                         + ','.join('?' for _ in batch) + ')', batch).fetchall()
    # Decoded after the read transaction: JSON work never extends the transaction or its lock.
    heard = {row['seq']: json.loads(row['dispatch_input'] or '{}').get('place_attention', {}) for row in retained}
    places = {}
    for message, entry in candidates:
        facts = record_facts(entry)
        block = entry.get('source_order')
        if isinstance(block, dict) and isinstance(block.get('time'), str) and block['time']:
            # The provider send time the arrival recorded (``source_order``); a row without one has none.
            facts['sent_at'] = block['time']
        imported = entry.get('historical_import')
        if isinstance(imported, dict):
            # An earlier message: its sender stays the platform person, with no owner-account
            # claim and no attention decision, which history never had.
            facts['historical_import'] = {'authorship': str(imported.get('authorship') or ''),
                                          'received_at': str(imported.get('received_at') or '')}
        elif message['role'] == 'user':
            place_id = entry.get('place_id')
            if place_id and place_id not in places:
                try:
                    places[place_id] = get_place(root, place_id)
                except (PlaceError, OSError, ValueError, StoreError):
                    places[place_id] = None
            transport = entry.get('transport') or {}
            actor = transport.get('actor') or {}
            authenticated = actor.get('authenticated') if isinstance(actor, dict) else None
            if places.get(place_id) is not None:
                facts['owner_confirmed'] = authenticated is not False and is_owner_account(
                    places[place_id], str(entry.get('sender_session_id') or ''), str(transport.get('account_id') or ''))
            attention = heard.get(entry.get('history_position', {}).get('offset'), {})
            if attention.get('decision') == 'heard':
                facts['attention'] = {'decision': 'heard', 'reason': str(attention.get('reason') or '')}
                if attention.get('facts') is not None:
                    facts['facts'] = attention['facts']
            elif entry.get('type') == 'place_arrival' and entry.get('task_id'):
                try:
                    task = reads.load(root, entry['task_id']) or {}
                except (OSError, ValueError, StoreError):
                    task = {}  # Missing task knowledge proves no attention outcome.
                metadata = task.get('metadata') or {}
                if (metadata.get('wake_cause') == 'focus' and entry.get('presence_event_identity')
                        and metadata.get('presence_event_identity') == entry['presence_event_identity']
                        and task.get('execution_owner')):
                    facts['attention'] = {'decision': 'focus_started'}
        message['place_history'] = facts
