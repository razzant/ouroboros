"""Owner-shaped counts and measured text lengths; no private owner content."""
from dataclasses import replace
import json
from types import SimpleNamespace

from ouroboros import context_fit as cf, memory_view as mv
from ouroboros.section_floor import HostSection
from tests import _memory_view_synthetic as syn

PROTECTED = {'system': 34800, 'bible': 51900, 'identity': 46000,
             'shared_understanding': 38000, 'scratchpad': 14400, 'memory_marks': 7700}
HOST = {'knowledge_index': 20000, 'runtime': 15200, 'subagents': 10400,
        'review_continuity': 7200, 'installed_skills': 3200, 'review_ledger': 3200}


def text(label, size):
    return (label + ' ' + 'x' * size)[:size]


def snapshot(rows=158, pages=113):
    mine = round(rows * 115 / 158)
    lane = [syn.spoken(i, 'human' if i < 2 * (rows - mine) and i % 2 == 0 else 'ouroboros', 1900) for i in range(rows)]
    for row in lane:
        if row['kind'] == 'human':
            row.update(owner_side=True, carried=False)
    lane += [{**syn.spoken(rows + i, 'human', 350), 'owner_side': True, 'immune': True}
             for i in range(29)]
    room = syn.room('1', lane1=lane, lane2=163, notes=6, under=5)
    for note in room['notes']:
        note['text'] = text('Protected own unsealed note.', 3700)
    live = [syn.live_room(i, notes=(1 + (i < 7)) if i < 66 else 0) for i in range(228)]
    for other in live:
        other['label'] = text('Project room ' + other['room_id'], 220)
        for note in other['notes']:
            note['text'] = text('Unsealed project note.', 1600)
    story = syn.pages(pages, 1800)
    if len(story) >= 3:
        story[0]['kind'] = 'part'
        story[1].update(kind='account', selection='selection-001', signer='root', written='2026-09-01',
            sources=[{'id': story[2]['id'], 'revision': story[2]['id'], 'revision_known': True}])
    for entry in story:
        entry['label'] = text('Main and project experience', 60)
    snap = syn.snapshot(room_facts=room, live=live, story=story)
    return replace(snap, legacy_blocks={**snap.legacy_blocks, 'folded': 23, 'total': 23,
                                       'open_records': 0, 'open_rows': 0, 'open_chars': 0})


def core(root, snap=None):
    snap = snap or snapshot()
    stable = (HostSection('identity', text('Identity', PROTECTED['identity'])),)
    head = (HostSection('shared_understanding', text('Overview', PROTECTED['shared_understanding'])),
            HostSection('knowledge_index', text('Knowledge full', 40000), text('Knowledge outline', HOST['knowledge_index'])))
    dynamic = tuple(HostSection(name, text(name, size)) for name, size in PROTECTED.items()
                    if name in {'scratchpad', 'memory_marks'}) + tuple(
        HostSection(name, text(name + ' full', size * 2), text(name + ' narrow', size))
        if name in {'runtime', 'subagents', 'installed_skills'} else HostSection(name, text(name, size))
        for name, size in HOST.items() if name != 'knowledge_index')
    return cf.ContextCore(base_prompt=text('SYSTEM', PROTECTED['system']), bible_md=text('BIBLE', PROTECTED['bible']),
        architecture_md='# Architecture\n\n' + 'book ' * 80000, development_md='', docs_need_development=False,
        semi_stable_text='\n\n'.join(stable), dynamic_head_text='\n\n'.join(head), dynamic_text='\n\n'.join(dynamic),
        semi_stable_sections=stable, dynamic_head_sections=head, dynamic_sections=dynamic,
        memory_view_json=mv.snapshot_json(snap), user_content_json=json.dumps('Continue in this room.'),
        section_root=str(root), section_task_id='scale-fit')


def tools():
    return [{'type': 'function', 'function': {'name': name, 'description': text(name, 1460),
             'parameters': {'type': 'object', 'properties': {}}}}
            for name in ('enable_tools', 'list_available_tools', 'compact_context', 'task', 'read_file')]


def build(root, monkeypatch, *, window=92000, reserve=10000, snap=None, mode='max', reply_limit=None):
    captured = core(root, snap)
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *a, **kw: 1.0)
    evidence = SimpleNamespace(window_tokens=window, status='confirmed', stale=False, route_fp='', source='fixture',
        response_limit={'max_output_tokens': reply_limit} if reply_limit else {},
        route_facts={'working_reserve_tokens': reserve, 'basis': 'provider'})
    return cf.build_context_fit_plan(SimpleNamespace(drive_root=root), captured,
        {'id': 'scale-fit', 'chat_id': 1, '_is_direct_chat': True}, preferred_mode=mode, tool_schemas=tools(),
        route_resolver=lambda *a, **kw: ({'provider': 'fixture', 'model': 'fixture'}, evidence))


def write_root(root):
    """The same population in real dialogue/chronicle stores for read-only tooling.

    Authored files carry the measured sizes; runtime host catalogs come from the
    ordinary producers on this synthetic install, not fabricated capture caches.
    """
    from ouroboros import dialogue_store as ds, usage_store
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.projects_registry import create_project
    from tests.test_chronicle_store import MIND, page
    from tests.test_budget_fence_liveness import _publish

    root.mkdir(parents=True, exist_ok=True)
    _publish(root)
    usage_store.migrate_from_journal(root)
    memory = root / 'memory'
    memory.mkdir(exist_ok=True)
    (memory / 'identity.md').write_text(text('Identity', PROTECTED['identity']))
    (memory / 'scratchpad.md').write_text(text('Scratchpad', PROTECTED['scratchpad']))
    knowledge = memory / 'knowledge'
    knowledge.mkdir(exist_ok=True)
    (knowledge / 'overview.md').write_text('# Overview\n' + text('Overview', PROTECTED['shared_understanding']))
    store = ChronicleStore(root)
    store.ensure_activated()
    source = snapshot()
    for other in source.live_rooms:
        project = create_project(root, 'scale-' + other['room_id'], name=other['label'][:80])
        ds.append(root, 'chat', {'chat_id': project['chat_id'], 'direction': 'in',
            'ts': '2026-09-01T00:00:00+00:00', 'text': 'Project room is open.'}, emit_live=False)
        for note in other['notes']:
            assert store.write_note(room_id=str(project['chat_id']), task_id='scale-note', text=note['text'], author=MIND).ok
    for row in source.room['lane1']:
        ds.append(root, 'chat', {'chat_id': 1, 'direction': 'out' if row['kind'] == 'ouroboros' else 'in',
            'ts': row['ts'], 'text': row['line'].split('] ', 1)[-1], 'task_id': f"speech-{row['pos']}"}, emit_live=False)
    for i, row in enumerate(source.room['lane2']):
        ds.append(root, 'chat', {'chat_id': 1, 'direction': 'system', 'type': 'task_summary',
            'ts': row['ts'], 'text': row['line'], 'task_id': f'scale-task-{i}', 'status': 'completed'}, emit_live=False)
    for note in source.room['notes']:
        assert store.write_note(room_id='1', task_id='scale-note', text=note['text'], author=MIND).ok
    pages = [page(store, f'sealed-source-{i}', room='1', at=i, text=text(f'Page {i}', 1800)).record for i in range(117)]
    assert store.publish_part(room_id='1', text='Five pages folded into a part.',
        member_ids=[p['id'] for p in pages[:5]], author=MIND, expected_sequence=store.room_head('1')).ok
    account = store.publish_account(room_id='1', text=text('Selected account', 1800),
                                    sources=[pages[-1]['id']], author=MIND)
    assert account.ok
    assert store.select_account(account.record['id'], replaces=[pages[-1]['id']], author=MIND,
                                reason='This account carries the last page.').ok
    assert store.mark({'node_id': pages[-1]['id']}, text('Protected marks', PROTECTED['memory_marks']),
                      MIND, room_id='1', scope='global').ok
    return root
