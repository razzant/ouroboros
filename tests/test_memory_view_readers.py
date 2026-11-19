"""A folded row's printed reader uses its room membership, not an endpoint's room."""
from dataclasses import replace
import re

import pytest

from ouroboros import chat_chain, dialogue_store as ds, memory_forms as forms, memory_view as mv
from ouroboros.memory_view_conversation_forms import _fold_line
from tests import _memory_inventory_shared as shared
from tests.test_memory_waiting import _ctx, _memory_read


def _read_unit(root, description):
    call = description.split('memory_read(', 1)[1].split(')', 1)[0]
    args = dict(re.findall(r"(\w+)='([^']*)'", call))
    return _memory_read(_ctx(root, 'root0001'), rows=True, **args), args


@pytest.mark.parametrize('inherited', [False, True])
def test_folded_line_reads_all_related_rows_and_no_unrelated_room_words(tmp_path, inherited):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project

    rooms = shared.world(tmp_path)
    project = create_project(tmp_path, 'fold-range', name='Fold range')
    room_id = int(project['chat_id'])
    owner = shared.msg('2026-09-10T00:00:00+00:00', 'Keep my own words whole.', chat_id=room_id,
                       client_message_id='fold-owner')
    ds.append(tmp_path, 'chat', owner)
    ref = build_owner_message_ref(chat_id=room_id, client_message_id='fold-owner', ts=owner['ts'], text=owner['text'])
    bind_task_to_project(tmp_path, 'fold0001', project['id'], origin={'ref': ref, 'text': owner['text']})
    source_rooms = [1, room_id, 777] if inherited else [room_id] * 3
    rows = []
    for n, source_room in enumerate(source_rooms, 1):
        row = shared.msg(f'2026-09-10T00:0{n}:00+00:00', f'Related answer {n}.',
                         chat_id=source_room, direction='out', task_id='fold0001')
        ds.append(tmp_path, 'chat', row)
        rows.append(row)
        if n == 1:
            ds.append(tmp_path, 'chat', shared.msg('2026-09-10T00:01:30+00:00', 'Unrelated Main answer.',
                                                  direction='out', task_id='other001'))
    ds.append(tmp_path, 'chat', shared.msg('2026-09-10T00:04:00+00:00', 'Finished.', chat_id=room_id,
        direction='system', type='task_summary', task_id='fold0001', wake_summary='The related exchange is settled.',
        fold_read=True, status='completed', summary_kind='terminal_root_projection'))
    task = {'id': 'reader01', 'chat_id': rooms['beta']}
    captured = mv.capture_memory_view(tmp_path, task, replace(mv.ROLE_DEFAULTS['integrator'], room_id=str(rooms['beta'])))
    group = next(g for g in captured.conversations if any(r['room_id'] == str(room_id) for r in g['rooms']))
    room = next(r for r in group['rooms'] if r['room_id'] == str(room_id))
    fold = room['folded']
    assert fold['count'] == 3
    shown = _fold_line(room)
    assert shown.count(' → ') == (0 if inherited else 1)
    if inherited:
        assert 'memory_read(' in shown  # the two endpoints name different rooms; the membership room is explicit
        returned, reader = _read_unit(tmp_path, shown)
    else:
        assert shown == f"[{fold['from']} → {fold['to']}] «The related exchange is settled.»: 3 words read and folded"
        reader = {'room_id': str(chat_chain.parse_address(fold['from'])['chat_id']),
                  'from': fold['from'], 'to': fold['to']}
        returned = _memory_read(_ctx(tmp_path, 'root0001'), rows=True, **reader)
    assert reader == {'room_id': str(room_id), 'from': fold['from'], 'to': fold['to']}
    assert all(row['text'] in returned for row in rows)
    assert owner['text'] not in returned and 'Unrelated Main answer.' not in returned
    snap = replace(captured, story=[], room=None, conversations=[group])
    assert owner['text'] in mv.render_room(snap)
    held = forms.holding_room(snap, mv.FULL_VIEW)
    unit = next(text for _tokens, text in held['largest'] if text.startswith('read and folded,'))
    held_text, held_reader = _read_unit(tmp_path, unit)
    assert held_reader == reader and held_text == returned


def test_map_separates_display_order_stored_bounds_and_the_finish_that_supplied_the_sentence(tmp_path):
    from ouroboros.memory_form_facts import memory_composition

    rooms = shared.world(tmp_path)
    room = 87436
    def append(minute, words, direction, **facts):
        row = shared.msg(f'2026-09-10T00:{minute:02d}:00+00:00', words, chat_id=room, direction=direction,
                         **{"source": "presence:test", **facts})
        ds.append(tmp_path, 'chat', row)
        return chat_chain.format_address(chat_chain.row_address(row))

    first_task = append(8, 'First task fact.', 'system', type='task_summary', task_id='work-a')
    first_reply = append(7, 'First stored reply.', 'out', task_id='work-a')
    append(6, 'Foreign task fact.', 'system', type='task_summary', task_id='work-b')
    append(5, 'Keep these owner words whole.', 'in', source='web')
    last_reply = append(4, 'Second stored reply.', 'out', task_id='work-a')
    append(3, 'Child task fact.', 'system', type='task_summary', task_id='child-a', root_task_id='work-a')
    finish = append(2, 'Terminal row.', 'system', type='task_summary', task_id='work-a',
                    summary_kind='terminal_root_projection', fold_read=True, wake_summary='Both replies settled.')
    # An imported arrival makes this room read by send time, even for the other rows.
    append(1, 'Imported arrival.', 'in', type='place_arrival', historical_import={'authorship': 'platform_account'})
    def capture(focus):
        task = {'id': 'reader01', 'chat_id': focus}
        snapshot = mv.capture_memory_view(tmp_path, task, replace(mv.ROLE_DEFAULTS['integrator'], room_id=str(focus)))
        if focus == rooms['beta']:
            assert f'[{last_reply} → {first_reply}] «Both replies settled.»: 2 words read and folded' in mv.render_room(snapshot)
        return memory_composition(mv.snapshot_from_json(mv.snapshot_json(snapshot)), mv.FULL_VIEW)['blocks']['room']['parts']

    parts = capture(rooms['beta'])
    fold = next(s for p in parts if p['kind'] == 'conversation_fold' for s in p['sources'] if s['room_id'] == str(room))
    assert (fold['from'], fold['to'], fold['count']) == (last_reply, first_reply, 2)
    args = fold['range_read']['arguments']
    assert args == {'room_id': str(room), 'rows': True, 'from': first_reply, 'to': last_reply}
    ctx = _ctx(tmp_path, 'reader01')
    bounded = _memory_read(ctx, **args)
    assert all(text in bounded for text in ('First stored reply.', 'Second stored reply.', 'Keep these owner words whole.'))
    assert 'excluding owner speech' in fold['selection'] and fold['row_membership_gap']
    gist = fold['gist_source']['read']['arguments']
    assert gist == {'room_id': str(room), 'rows': True, 'from': finish, 'to': finish}
    assert 'Terminal row.' in _memory_read(ctx, **gist) and 'Terminal row.' not in bounded
    for source_parts, kind in ((parts, 'conversation_task'), (capture(room), 'task_fact')):
        task = next(s for p in source_parts if p['kind'] == kind for s in p['sources'] if s['task'] == 'work-a')
        assert (task['first'], task['last']) == (finish, first_task)
        args = task['range_read']['arguments']
        assert args == {'room_id': str(room), 'rows': True, 'from': first_task, 'to': finish, 'task_id': 'work-a'}
        rows = _memory_read(ctx, **args)
        assert all(text in rows for text in ('First task fact.', 'Child task fact.', 'Terminal row.', 'First stored reply.'))
        assert 'Foreign task fact.' not in rows and 'lane-2' in task['selection'] and task['row_membership_gap']
