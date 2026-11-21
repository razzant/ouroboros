"""A holding unit and a folded line read the source rooms of their counted words."""
from dataclasses import replace
import re

import pytest

from ouroboros import chat_chain, dialogue_store as ds, memory_forms as forms, memory_view as mv
from tests import _memory_inventory_shared as shared
from tests.test_memory_waiting import _ctx, _event_group, _event_row, _memory_read, view


def _read_unit(root, description):
    call = description.split('memory_read(', 1)[1].split(')', 1)[0]
    args = dict(re.findall(r"(\w+)='([^']*)'", call))
    return _memory_read(_ctx(root, 'root0001'), rows=True, **args), args


@pytest.mark.parametrize('rooms', [1, 3])
@pytest.mark.parametrize('kind', ['untold in other conversations', 'owner words in other rooms',
                                 'work gists', 'unread host facts', 'read and folded'])
def test_holding_reader_returns_its_sources_instead_of_the_newest_room(tmp_path, rooms, kind):
    from ouroboros import places
    from ouroboros.memory_floor import view_tokens

    shared.world(tmp_path)
    if kind == 'owner words in other rooms':
        place = places.connect_place(tmp_path, connector='bridge', provider='harbor', account_id='account', label='Garden')
        places.edit_place(tmp_path, place.place_id, 'confirm_owner_account', author='owner',
                          platform_actor_id='Ann', account_id='account')
    incoming = kind in ('owner words in other rooms', 'unread host facts')
    rows = [_event_row('harbor', 'older', i, 'Ann', direction='in' if incoming else 'out',
                       task_id='read0001', text=f'Source word {i}. ' + 'The older exchange stays addressable. ' * 20)
            for i in (1, 2)]
    if kind == 'owner words in other rooms':
        for row in rows:
            row.update(place_id=place.place_id, presence_provenance={'place_id': place.place_id})
    for row in rows:
        ds.append(tmp_path, 'chat', row)
    if kind in ('work gists', 'read and folded'):
        ds.append(tmp_path, 'chat', _event_row('harbor', 'older', 3, direction='system', type='task_summary',
            task_id='read0001', status='completed', wake_summary='I kept the older exchange.',
            fold_read=kind == 'read and folded'))
    for i in range(1, rooms):
        ds.append(tmp_path, 'chat', _event_row('harbor', f'newer-{i}', 10 + i, text=f'Unrelated arrival {i}.'))
    captured = view(tmp_path)
    group = _event_group(captured)
    snap = replace(captured, story=[], room=None, conversations=[group])
    level = mv.FloorLevel((('F6g', (group['key'],)),)) if kind in ('work gists', 'unread host facts') else mv.FULL_VIEW
    held = forms.holding_room(snap, level)
    units = [(n, description) for n, description in held['largest'] if description.startswith(kind + ',')]
    assert len(units) == 1, held
    tokens, description = units[0]
    assert tokens == held['tokens'][kind] and group['label'] in description
    returned, reader = _read_unit(tmp_path, description)
    assert reader['room_id'] == str(rows[0]['chat_id'])
    assert all(row['text'] in returned for row in rows)
    assert 'Unrelated arrival' not in returned
    assert reader['from'] == chat_chain.format_address(chat_chain.row_address(rows[0]))
    assert reader['to'] == chat_chain.format_address(chat_chain.row_address(rows[-1]))
    if kind in ('untold in other conversations', 'owner words in other rooms'):
        words = group['rooms'][0]['words']
        assert held['tokens'][kind] == view_tokens('\n'.join(w['line'] for w in words))
    assert mv.snapshot_json(snap) == mv.snapshot_json(replace(captured, story=[], room=None, conversations=[group]))
