"""Small source witnesses from actual renderers, checked again on the prepared request.

Positions are Unicode text coordinates, not provider tokens or JSON byte offsets.
No text or guessed provider acceptance is stored by this trace.
"""
from __future__ import annotations

from functools import wraps
from hashlib import sha256
import logging
import re

_ROW_ADDRESS = re.compile(r'row:-?\d+@[-0-9T:.+Z]+#[0-9a-f]{12}')
_MAX_GROUPS_PER_BLOCK = 256


def _optional_observation(result_arg=None):
    """Keep optional diagnostics out of the producer's control path."""
    def decorate(observe):
        @wraps(observe)
        def guarded(*args, **kwargs):
            try:
                return observe(*args, **kwargs)
            except Exception:
                logging.getLogger(__name__).debug('Context observation %s unavailable', observe.__name__, exc_info=True)
                return args[result_arg] if result_arg is not None else None
        return guarded
    return decorate


class AttachmentSelection:
    """Bounded decisions recorded at the existing read/prepare loop, not a second file scan."""

    limit = 32

    def __init__(self, facts):
        self.facts = facts

    def omit_remaining(self, count):
        if self.facts is not None:
            self.facts['attachment_items_omitted'] = self.facts.get('attachment_items_omitted', 0) + count

    def record(self, ordinal, decision):
        if self.facts is None:
            return
        if ordinal >= self.limit:
            self.omit_remaining(1)
        else:
            self.facts.setdefault('attachment_items', []).append({'number': ordinal + 1, 'decision': decision})


def digest(text: str) -> str:
    return sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


class RoomWitness:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.end = 0
        self.pieces = []
        self._pieces_complete = True
        self.groups = {group['key']: group for group in getattr(snapshot, 'conversations', ())}
        self.forms = {}
        self.refs = []

    def selections(self):
        return {**({'rendered_forms': self.forms} if self.forms else {}),
                **({'sample_row_addresses': self.refs} if self.refs else {})}

    @_optional_observation(1)
    def part(self, lines, kind='framing', sources=(), form='whole', room_id=None):
        """Selected render only. Record counted forms and a bounded sample of canonical addresses."""
        if lines and kind != 'framing' and isinstance(sources, (tuple, list)):
            key = kind + ':' + form
            self.forms[key] = self.forms.get(key, 0) + len(sources)
            for source in sources[:max(0, 6 - len(self.refs))]:
                if not isinstance(source, dict):
                    continue
                for field in ('address', 'first', 'last'):
                    value = source.get(field)
                    if len(self.refs) < 6 and isinstance(value, str) and _ROW_ADDRESS.fullmatch(value):
                        self.refs.append(value)
        return lines

    @_optional_observation(2)
    def observe(self, key, text):
        try:
            if text and self._pieces_complete:
                start = self.end + (2 if self.end else 0)
                self.end = start + len(text)
                fact = {**self.piece_fact(key), **self.selections()}
                self.pieces.append({**fact, 'start': start, 'chars': len(text)})
        except Exception:
            # An unobserved room piece is not a fit note. Fall back to the whole room.
            self.pieces.clear()
            self._pieces_complete = False
            raise
        finally:
            self.forms, self.refs = {}, []
        return text

    def piece_fact(self, key):
        if key[0] != 'conversations':
            steps = {step: len(ids) for step, ids in key[1] if ids} if key[0] == 'room' else {}
            return {'family': key[0], 'form': 'shorter' if steps else 'rendered', 'choices': steps}
        if key[1] == 'group':
            _, _, group, form, words, owners, events = key
            rooms = self.groups.get(group, {}).get('rooms', ())
            room = str(rooms[0].get('room_id', '')) if rooms else ''
            return {'family': 'other_conversation', 'form': 'summary' if form else 'rendered',
                    'room_count': len(rooms), **({'sample_room': room} if re.fullmatch(r'\d{1,16}', room) else {}),
                    'choices': {'group_form': form, 'words_reduced': len(words),
                                'owner_words_reduced': len(owners), 'events_counted': events}}
        if key[1] == 'counted':
            return {'family': 'other_conversations', 'form': 'count',
                    'choices': {'conversations_counted': len(key[2]), 'owner_words_reduced': len(key[3]),
                                'unread_counted': key[4]}}
        return {'family': 'other_conversations', 'form': 'framing'}


def source_rows(block, index, groups, *, changing=False):
    """Address groups from the renderer's actual strings; no reread or body in the receipt.

    The carrier digest binds its placement. Child spans refer to the same text,
    not a reconstruction from the current memory or a redacted JSON document.
    """
    if not block or not groups:
        return None
    return {'source_kind': 'system_block', 'block': index, 'carrier_chars': len(block),
            'carrier_sha256': digest(block), 'groups': groups[:_MAX_GROUPS_PER_BLOCK], 'changing': changing,
            'groups_omitted': max(0, len(groups) - _MAX_GROUPS_PER_BLOCK)}


def span_groups(parts):
    """Two newlines separate the same nonempty parts the renderer joins."""
    end = 0
    for fact, text in parts:
        if not text:
            continue
        start = end + (2 if end else 0)
        end = start + len(text)
        yield {**fact, 'start': start, 'chars': len(text)}


def section_fact(part, addressed, receipts, books):
    kind = part.section_id
    family = 'held_note' if kind.startswith('held_note:') else kind
    fact = {'family': family, 'form': 'shorter' if kind in addressed else 'rendered'}
    receipt = receipts.get(kind, {})
    if receipt:
        fact['reason'] = 'mode_or_start_allowance' if receipt.get('start_room') else 'physical_window'
        ref = receipt.get('source_ref', {})
        path = ref.get('path')
        if family != 'held_note' and isinstance(path, str) and re.fullmatch(r'source_handles/context_checkpoints/host-[a-zA-Z0-9_.-]+\.md', path):
            fact['retained_source'] = {'path': path, 'sha256': ref.get('sha256')}
    if book := books.get(kind.removesuffix('_map')):
        fact['family'] = book.book_id
        fact['form'] = ('chapter_links' if kind in addressed else 'chapter_map') if kind.endswith('_map') else 'whole_book'
        fact['sources'] = [{'path': source.source_path, 'sha256': source.sha256,
                            'bytes': len(source.raw)} for source in (book.entrypoint, *book.chapters)]
    return fact


def layout_witnesses(core, governance, addressed, receipts, live, *, story_whole, room_pieces, story_selection=None):
    """Coordinates of the renderer's already-joined blocks; group names never come from text."""
    common, handbook, shared, story, head, room, dynamic, stable, changing = live
    books = {book.book_id: book for book in core.reference_books}
    own_block = core.docs_need_development or core.development_required
    receipts = {receipt['section_id']: receipt for receipt in receipts}

    def typed(parts):
        return [(section_fact(part, addressed, receipts, books), addressed.get(part.section_id, part.full_text))
                for part in parts]

    def fragment(groups, family, text, parts):
        if not text:
            return
        prior = groups[-1]['start'] + groups[-1]['chars'] if groups else 0
        offset = prior + (2 if prior else 0)
        subgroups = list(span_groups(parts))
        # The ordinary strings are joined even when a captured section is empty. In that
        # unusual case do not claim fine positions using the nonempty join's geometry.
        if subgroups and subgroups[-1]['start'] + subgroups[-1]['chars'] == len(text):
            groups.extend({**row, 'start': row['start'] + offset} for row in subgroups)
        else:
            groups.append({'family': family, 'form': 'rendered', 'start': offset, 'chars': len(text)})

    def handbook_group():
        part = governance[1]
        if hasattr(part, 'section_id'):
            return section_fact(part, addressed, receipts, books)
        return {'family': 'development', 'form': 'rendered'}

    common_groups, shared_groups, changing_groups = [], [], []
    source_parts = governance[0]
    fragment(common_groups, 'governance', common, () if isinstance(source_parts, str) else typed(source_parts))
    if not own_block and handbook:
        shared_groups.extend(span_groups([(handbook_group(), handbook[0])]))
    fragment(shared_groups, 'stable_context', shared[0], typed(core.semi_stable_sections))
    fragment(shared_groups, 'connections', shared[1], ())
    story_fact = {'family': 'story', 'form': 'whole_view' if story_whole else 'shorter', **(story_selection or {})}
    fragment(shared_groups, 'story', story, [(story_fact, story)])
    fragment(changing_groups, 'knowledge', head, typed(core.dynamic_head_sections))
    if room:
        room_start = len(head) + 2 if head else 0
        if room_pieces and room_pieces[-1]['start'] + room_pieces[-1]['chars'] <= len(room):
            changing_groups.extend({**row, 'start': row['start'] + room_start} for row in room_pieces)
            start = room_pieces[-1]['start'] + room_pieces[-1]['chars']
            if start < len(room):
                changing_groups.append({'family': 'memory_fit_notes', 'form': 'rendered',
                                        'start': room_start + start, 'chars': len(room) - start})
        else:
            fragment(changing_groups, 'room', room, ())
    fragment(changing_groups, 'runtime', dynamic, typed(core.dynamic_sections))

    groups = [common_groups, *([list(span_groups([(handbook_group(), handbook[0])]))] if own_block and handbook else ()), shared_groups]
    selected = [(block, parts) for block, parts in zip(stable, groups) if block.strip()]
    return tuple(row for i, (block, parts) in enumerate([*selected, (changing, changing_groups)])
                 if (row := source_rows(block, i, parts, changing=i == len(selected))) is not None)


def _text_block(content, index, path):
    if isinstance(content, str) and index == 0:
        yield path, content
    elif isinstance(content, list) and 0 <= index < len(content):
        item = content[index]
        if isinstance(item, dict) and isinstance(item.get('text'), str):
            yield f'{path}[{index}].text', item['text']


def _layout_carriers(candidate, row):
    """Only route-owned coordinates: a source repeated later in chat is not a witness."""
    from ouroboros.llm_messages import HOST_CONTEXT_NOTICE_BEFORE_TASK, HOST_CONTEXT_NOTICE_AFTER_TASK

    index = row['block']
    for key in ('system', 'instructions'):
        for path, text in _text_block(candidate.get(key), index, f'$.{key}'):
            yield path, text, 0
    for key in ('messages', 'input'):
        messages = candidate.get(key)
        if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
            continue
        role = ('system', 'developer') if key == 'input' else ('system',)
        if messages[0].get('role') in role:
            for path, text in _text_block(messages[0].get('content'), index, f'$.{key}[0].content'):
                yield path, text, 0
        if not index or index >= len(messages):
            continue
        current = messages[index]
        if not all(isinstance(m, dict) and m.get('role') in role for m in messages[:index]):
            continue
        if isinstance(current, dict) and current.get('role') in role:
            for path, text in _text_block(current.get('content'), 0, f'$.{key}[{index}].content'):
                yield path, text, 0
        if row['changing']:
            for pos, notice_role, prefix in ((index, 'user', HOST_CONTEXT_NOTICE_BEFORE_TASK),
                                             (index + 1, 'developer', HOST_CONTEXT_NOTICE_AFTER_TASK)):
                if pos >= len(messages) or not isinstance(notice := messages[pos], dict) or notice.get('role') != notice_role:
                    continue
                marker = '[SYSTEM NOTICE]\n' + prefix + '\n\n'
                for path, text in _text_block(notice.get('content'), 0, f'$.{key}[{pos}].content'):
                    if text.startswith(marker):
                        yield path, text, len(marker)


def verified_layout(candidate, witnesses):
    """Report every selected logical block, including an ambiguous/changed layout."""
    blocks, bound_paths, measured = [], set(), {}
    for row in witnesses:
        if row.get('source_kind') != 'system_block':
            continue
        matches = set()
        for path, text, offset in _layout_carriers(candidate, row):
            if len(text) - offset != row['carrier_chars']:
                continue
            key = id(text), offset
            if key not in measured:
                measured[key] = digest(text[offset:])
            if measured[key] == row['carrier_sha256']:
                matches.add((path, offset))
        path, offset = next(iter(matches)) if len(matches) == 1 else (None, 0)
        block = {key: row[key] for key in ('block', 'groups', 'groups_omitted')}
        block['chars'] = row['carrier_chars']
        if path is not None and path not in bound_paths:
            bound_paths.add(path)
            block.update(status='located', prepared_text={'path': path, 'start_char': offset,
                                                          'end_char': offset + row['carrier_chars']})
        else:
            block['status'] = 'position_unverified'
        blocks.append(block)
    return blocks


def prepared_inventory(candidate, layout, contributions=(), *, verified_ranges=()):
    """A bounded shallow census of the prepared request, independent of the logical trace.

    Counts exist even when the body is too large to read. No provider-specific names or
    values enter the record. Only a located anchor credits part of a carrier, never the whole row.
    """
    fields = ('messages', 'input', 'system', 'instructions', 'tools', 'functions')
    roles = ('system', 'developer', 'user', 'assistant', 'tool', 'function')
    paths = [block['prepared_text']['path'] for block in layout if block['status'] == 'located']
    paths.extend(row['prepared_text']['path'] for row in verified_ranges if 'prepared_text' in row)
    paths.extend(row['path'] for row in contributions if row.get('status') == 'located')
    result = {'other_fields': sum(key not in fields for key in candidate), 'fields': []}
    for field in fields:
        if field not in candidate:
            continue
        value = candidate[field]
        items = value if isinstance(value, list) else [value]
        typed = {}
        if field in ('messages', 'input'):
            for item in items[:512]:
                role = item.get('role') if isinstance(item, dict) else None
                kind = role if role in roles else 'other'
                typed[kind] = typed.get(kind, 0) + 1
        prefix = '$.' + field
        anchored = set()
        for path in paths:
            if path == prefix or path.startswith(prefix + '['):
                anchored.add(path[len(prefix):].partition(']')[0])
        result['fields'].append({'field': field, 'items': len(items), 'items_with_origin': len(anchored),
                                 **({'roles': typed, 'uninspected_roles': max(0, len(items) - 512)} if typed else {})})
    return result


# Keep the disclosure bounded even on a very long working transcript. The inventory
# still counts the request itself. A bounded search cannot certify unique positions.
_CONTRIBUTION_ROWS = 256
_CONTRIBUTION_PARTS = 32


@_optional_observation(0)
def bind_contribution_input(context, kwargs, canonical, slot=None):
    from dataclasses import replace

    # References to this call's existing rows; no copy of the request body. Not a
    # dataclass field, and never in the logical rows or provider payload.
    rows = kwargs.get('messages') or ()
    tail = tuple(rows[-_CONTRIBUTION_ROWS:])
    original = tail if canonical is rows else (
        tuple(canonical[-_CONTRIBUTION_ROWS:]) if canonical is not None and len(canonical) == len(rows) else ())
    observations = getattr(slot, '_context_contributions', {})
    if observations:
        live = {id(row) for row in (original or tail)}
        for retired in observations.keys() - live:
            del observations[retired]
    notes = tuple(observations.get(id(row), (None, (), 0))[1:] for row in (original or tail))
    tools = tuple(kwargs.get('tools') or ())
    schema_notes = getattr(slot, '_context_schema_origins', {})
    origins = tuple(schema_notes.get(id(schema), (None, None))[1] for schema in tools[:_CONTRIBUTION_ROWS])
    projected = getattr(slot, '_context_image_projections', {})
    image_forms = tuple(_image_forms(row, projected) for row in tail)
    return replace(context, contribution_input=(tail, tools, len(rows), notes, origins, image_forms))


def _image_forms(row, projected):
    if not projected or not isinstance(row, dict) or not isinstance(content := row.get('content'), list):
        return {}
    return {i: held[1] for i, part in enumerate(content[:_CONTRIBUTION_PARTS])
            if (held := projected.get(id(part))) is not None}


@_optional_observation()
def note_image_projection(block, form):
    """Witness the actual send-copy part, never stamp the message or reattribute a copy."""
    from ouroboros.model_wait import current_model_wait

    slot = getattr(current_model_wait(), 'tool_context', None)
    if slot is None:
        return
    notes = getattr(slot, '_context_image_projections', None)
    if notes is None:
        notes = slot._context_image_projections = {}
    notes[id(block)] = (block, form)  # retain identity until eviction, not just its reusable Python id
    if len(notes) > _CONTRIBUTION_ROWS:
        del notes[next(iter(notes))]


@_optional_observation()
def note_contribution(slot, row, family, *, source='', form='typed_row', observation=None):
    """A producer marks the selected occurrence off transcript, including a containing merged row.

    Copy/recovery does not preserve object identity. A missing note is not proof
    there was no source. Producer IDs remain private; persist their hashes only.
    """
    if slot is None:
        return
    notes = getattr(slot, '_context_contributions', None)
    if notes is None:
        notes = slot._context_contributions = {}
    _, prior, omitted = notes.get(id(row), (None, (), 0))
    fact = {'family': family, 'form': form}
    if isinstance(source, str) and _ROW_ADDRESS.fullmatch(source):
        fact['row_address'] = source
    elif isinstance(source, str) and source:
        fact['source_id_sha256'] = digest(source)
    if observation:
        for key in ('listed', 'omitted', 'deferred_to_mail', 'history_gaps',
                    'manifest_items', 'manifest_path_only', 'manifest_pixels', 'attachment_items_omitted'):
            value = observation.get(key)
            if type(value) is int and value >= 0:
                fact[key] = value
        for key in ('mailbox_read', 'manifest_unavailable'):
            if type(value := observation.get(key)) is bool:
                fact[key] = value
        if items := observation.get('attachment_items'):
            fact['attachment_items'] = items[:AttachmentSelection.limit]
        if isinstance(ref := observation.get('presence_reentry_sha256'), str) and re.fullmatch('[0-9a-f]{64}', ref):
            fact['presence_reentry_sha256'] = ref
    notes[id(row)] = (row, (*prior[-(_CONTRIBUTION_PARTS - 1):], fact),
                      omitted + int(len(prior) == _CONTRIBUTION_PARTS))
    if len(notes) > _CONTRIBUTION_ROWS:
        del notes[next(iter(notes))]


@_optional_observation(1)
def note_schema_origin(slot, schema, origin):
    """The registry records its materialized objects, not name or prefix guesses."""
    notes = getattr(slot, '_context_schema_origins', None)
    if notes is None:
        notes = slot._context_schema_origins = {}
    notes[id(schema)] = (schema, origin)
    if len(notes) > 2 * _CONTRIBUTION_ROWS:
        del notes[next(iter(notes))]
    return schema


@_optional_observation()
def carry_schema_origins(slot, originals, copies):
    """The selection owner knows which schema was copied; none is reattributed by name or bytes."""
    notes = getattr(slot, '_context_schema_origins', {})
    observed = [(copy, notes[id(original)][1]) for original, copy in zip(originals, copies) if id(original) in notes]
    for copy, origin in observed:
        note_schema_origin(slot, copy, origin)


@_optional_observation()
def move_contributions(slot, before, after):
    """The append owner just replaced an unsent row with its merged successor."""
    notes = getattr(slot, '_context_contributions', None)
    if notes and (held := notes.pop(id(before), None)) is not None:
        notes[id(after)] = (after, *held[1:])


def _media(row):
    content = row.get('content')
    if not isinstance(content, list):
        return {}
    return {i: 'pixels' if part.get('type') in ('image', 'image_url', 'input_image') else 'evicted'
            for i, part in enumerate(content[:_CONTRIBUTION_PARTS]) if isinstance(part, dict)
            and (part.get('type') in ('image', 'image_url', 'input_image') or part.get('_evicted_image') is True)}


def _carry(row):
    """Only typed carriers with an exact comparable cell; no parsing of notice, caption or call text."""
    role, content = row.get('role'), row.get('content')
    if role == 'tool' or row.get('type') in {'function_call_output', 'custom_tool_call_output', 'tool_result'}:
        from ouroboros.tool_result_record import logical_tool_result_text

        identity = row.get('tool_call_id', row.get('call_id', row.get('tool_use_id')))
        value = logical_tool_result_text(row.get('output', content))
        return ('tool', identity, value) if isinstance(identity, str) and isinstance(value, str) else None
    if role not in {'user', 'assistant'}:
        return None
    if isinstance(content, str):
        content = ({'type': 'text', 'text': content},)
    if not isinstance(content, (list, tuple)) or not content or len(content) > _CONTRIBUTION_PARTS:
        return None
    parts = []
    for part in content:
        if not isinstance(part, dict):
            return None
        kind = part.get('type')
        if kind in {'text', 'input_text', 'output_text'} and isinstance(part.get('text'), str):
            parts.append(('text', part['text']))
        elif kind in {'image_url', 'input_image'}:
            url = part.get('image_url')
            if isinstance(url, dict):
                url = url.get('url')
            if not isinstance(url, str):
                return None
            parts.append(('image', url))
        else:
            return None
    return (role, *parts)


def _indexed_contributions(candidate):
    from collections import defaultdict

    field = 'messages' if 'messages' in candidate else 'input'
    rows = candidate.get(field)
    index = defaultdict(list)
    if not isinstance(rows, list) or len(rows) > _CONTRIBUTION_ROWS + 8:
        return index, False, len(rows) if isinstance(rows, list) else None
    omitted_parts = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if key := _carry(row):
            index[key].append(f'$.{field}[{i}]')
        content = row.get('content')
        if isinstance(content, list):
            omitted_parts += max(0, len(content) - _CONTRIBUTION_PARTS)
            for j, part in enumerate(content[:_CONTRIBUTION_PARTS]):
                if isinstance(part, dict) and part.get('type') == 'tool_result' and (key := _carry(part)):
                    index[key].append(f'$.{field}[{i}].content[{j}]')
    return index, omitted_parts == 0, omitted_parts


def _row_contributions(row, image_forms):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.tool_result_record import read_tool_result_record

    out = []
    if family := _ROW_FAMILY.get(row.get(HOST_CONTEXT_KIND_KEY)):
        out.append({'family': family, 'form': 'typed_row'})
    if row.get('role') == 'tool':
        record = read_tool_result_record(row)
        proved = record['state'] == 'recorded'
        out.append({'family': 'tool_result', 'form': 'result_view' if proved else 'unrecorded',
                    **({'invocation_sha256': digest(str(record['invocation']['invocation_id'])),
                        'view': 'retained_source' if record.get('source_ref') else 'extent_unknown'} if proved else {})})
    content = row.get('content')
    if isinstance(content, list):
        for part in content[:_CONTRIBUTION_PARTS]:
            capsule = part.get('_context_capsule') if isinstance(part, dict) else None
            if (isinstance(capsule, dict) and isinstance(part.get('text'), str)
                    and capsule.get('visible_sha256') == digest(part['text'])
                    and capsule.get('retention') in {'source_view', 'summarized'}):
                out.append({'family': 'checkpoint', 'form': capsule['retention']})
        media = {**_media(row), **image_forms}
        for form in dict.fromkeys(media.values()):
            out.append({'family': 'attachment_image', 'form': form, 'count': sum(v == form for v in media.values())})
    return out


_ROW_FAMILY = {'owner_words': 'protected_input', 'owner_dialogue': 'outward_reply', 'reply_citation': 'reply_context',
               'return_facts': 'return_context', 'skill_context_snapshot': 'skill_snapshot'}


def _schema_occurrences(tools):
    rows = {}
    for i, schema in enumerate(tools[:_CONTRIBUTION_ROWS]):
        if not isinstance(schema, dict):
            continue
        definition = schema.get('function', schema)
        name = definition.get('name') if isinstance(definition, dict) else None
        if isinstance(name, str):
            rows.setdefault(name, []).append((i, definition))
    return rows


def verified_contributions(candidate, captured):
    """A logical producer observation and a verified prepared location are independent."""
    from collections import Counter

    rows, tools, total, notes, origins, image_forms = captured
    physical, inspected, omitted = _indexed_contributions(candidate)
    keys = [_carry(row) if isinstance(row, dict) else None for row in rows]
    counts = Counter(key for key in keys if key is not None)
    entries = []
    row_notes_omitted = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key, facts = keys[i], _row_contributions(row, image_forms[i])
        row_notes, row_omitted = notes[i]
        row_notes_omitted += row_omitted
        if row_notes:
            superseded = {note['family'] for note in row_notes}
            if superseded & {'place_arrival', 'task_mail', 'owner_input'}:
                superseded.add('protected_input')
            facts = [*row_notes, *(fact for fact in facts if fact['family'] not in superseded)]
        matches = physical.get(key, ()) if key and inspected and len(rows) == total else ()
        path = matches[0] if len(matches) == 1 and counts[key] == 1 else None
        entries.extend({**fact, **({'path': path} if path else {}), 'status': 'carrier_located' if path else 'position_unverified'}
                       for fact in facts)
    logical_catalog = _schema_occurrences(tools)
    field = 'tools' if 'tools' in candidate else 'functions'
    prepared = candidate.get(field)
    selected = _schema_occurrences(prepared) if isinstance(prepared, list) else {}
    for name, pairs in logical_catalog.items():
        matches = selected.get(name, ())
        path = None
        if len(tools) <= _CONTRIBUTION_ROWS and len(prepared or ()) <= _CONTRIBUTION_ROWS and len(pairs) == len(matches) == 1:
            (i, definition), = matches
            if definition == pairs[0][1]:
                path = f'$.{field}[{i}]'
        entries.append({'family': 'tool_schema', 'form': 'selected_schema',
                        **({'origin': origin} if len(pairs) == 1 and (origin := origins[pairs[0][0]]) else {}),
                        **({'path': path} if path else {}), 'status': 'located' if path else 'position_unverified'})
    return {'entries': entries[:4 * _CONTRIBUTION_ROWS], 'entries_omitted': max(0, len(entries) - 4 * _CONTRIBUTION_ROWS),
            'logical_rows_omitted': max(0, total - len(rows)), 'logical_tools_omitted': max(0, len(tools) - _CONTRIBUTION_ROWS),
            'row_notes_omitted': row_notes_omitted,
            'physical_position_scan': 'complete' if inspected else 'incomplete_or_unsupported',
            'uninspected_physical_items': omitted}



def observe_layout(core, blocks, addressed, rows, whole, shown, witnesses):
    """Observe the selected rendering without owning its projection or selection."""
    @_optional_observation()
    def trace(live):
        witnesses.extend(layout_witnesses(core, blocks, addressed, rows, live, story_whole=whole,
            room_pieces=getattr(shown, 'room_pieces', ()), story_selection=getattr(shown, 'story_selection', None)))
    return trace
