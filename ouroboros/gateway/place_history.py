"""Small owner-history projection of retained Place facts, never an attention decision.

Source rows/addresses stay unchanged. Heard decisions live on their accepted
dialogue rows; an admitted focus lives on its exact task. No wake match is inferred.
An imported earlier message shows its import facts, never owner confirmation or attention.
A row whose arrival recorded its provider send time shows it (``sent_at``); ``ts`` stays the row's own.
"""
from __future__ import annotations

from ouroboros.dialogue_provenance import reaction_action_words, reaction_fact, reaction_words
from ouroboros.place_arrivals import heard_attention
from ouroboros.places import PlaceError, owner_confirmed_row, place_owner_accounts, row_place_id
from ouroboros.presence_call import call_utterance
from ouroboros.presence_message_facts import non_text_items
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
    # The connector's message facts are this row's own only on an arrival or a delivery receipt: a v0 reply
    # (``presence_runner._log_dialogue``) keeps the event it answered in its transport, and a call's utterance its
    # opening's (``presence_call.call_utterance``), never facts about itself.
    own = (direction == 'in' and not call_utterance(entry)) or kind == 'presence_delivery'
    message = transport.get('message') if own else None
    if non_text := non_text_items(message):  # connector facts beside the words, never speech
        facts['non_text'] = [{'kind': item_kind, 'description': description} for item_kind, description in non_text]
    reaction = reaction_fact(message)
    if reaction:  # English host words for the act; words the row sent stay its text
        state = delivery.get('state') if isinstance(delivery, dict) else ''
        facts['reaction'] = {'summary': reaction_action_words(reaction, state) if kind == 'presence_delivery'
                             else reaction_words(reaction)}
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
    heard = heard_attention(root, (entry['history_position']['offset'] for _, entry in candidates
                                   if entry.get('type') == 'place_arrival'
                                   and entry.get('history_position', {}).get('source') == 'chat'
                                   and not isinstance(entry.get('historical_import'), dict)))
    accounts = None  # the places' current owner accounts, read once for the page when a person's row needs them
    for message, entry in candidates:
        facts = record_facts(entry)
        # The provider send time the arrival recorded (``source_order``); a row without one has none, nor has a call's
        # utterance. One stored before heard rows stopped carrying the opening's message holds the opening's time and,
        # never rewritten, still reads there in a room ordered by time (``place_arrivals.READING_TIME``).
        block = None if call_utterance(entry) else entry.get('source_order')
        if isinstance(block, dict) and isinstance(block.get('time'), str) and block['time']:
            facts['sent_at'] = block['time']
        imported = entry.get('historical_import')
        if isinstance(imported, dict):
            # An earlier message: its sender stays the platform person, with no owner-account
            # claim and no attention decision, which history never had.
            facts['historical_import'] = {'authorship': str(imported.get('authorship') or ''),
                                          'received_at': str(imported.get('received_at') or '')}
        elif message['role'] == 'user':
            if accounts is None:
                try:
                    accounts = place_owner_accounts(root)
                except PlaceError:
                    accounts = {}  # unreadable places: the confirmation stays unknown, never "not the owner"
            if row_place_id(entry) in accounts:
                facts['owner_confirmed'] = owner_confirmed_row(entry, accounts)
            attention = heard.get(entry.get('history_position', {}).get('offset'), {})
            if attention.get('decision') == 'heard':
                facts['attention'] = {'decision': 'heard', 'reason': str(attention.get('reason') or ''),
                                      **({'recipient': str(attention['recipient'])} if attention.get('recipient') else {})}
                if attention.get('facts') is not None:
                    facts['facts'] = attention['facts']
            elif entry.get('type') == 'place_arrival' and entry.get('task_id') and not call_utterance(entry):
                try:  # a call's utterances share its identity: only the opening's row started the focus
                    task = reads.load(root, entry['task_id']) or {}
                except (OSError, ValueError, StoreError):
                    task = {}  # Missing task knowledge proves no attention outcome.
                metadata = task.get('metadata') or {}
                if (metadata.get('wake_cause') == 'focus' and entry.get('presence_event_identity')
                        and metadata.get('presence_event_identity') == entry['presence_event_identity']
                        and task.get('execution_owner')):
                    facts['attention'] = {'decision': 'focus_started'}
        message['place_history'] = facts
