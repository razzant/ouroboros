"""Coverage uses delivered coordinate boundaries of the real history endpoint."""
from ouroboros import dialogue_import
from ouroboros.gateway.history_paging import decode_cursor
from tests.test_chat_history_paging import isolated_runtime, write, row, request, pages  # noqa: F401


def test_coverage_excludes_quota_deferred_backdated_row(tmp_path):
    write(tmp_path / 'logs/chat.jsonl', [row(0), row(1), row(2, direction='system',
        type='task_summary', task_id='old', ts='2020-01-01T00:00:00Z', text='deferred')])
    loaded = list(pages(tmp_path, n_human='1'))
    recent = loaded[0]['coverage']
    cursor = decode_cursor(loaded[0]['next_cursor'], 1, recent['view'])
    assert recent['spans']['chat']['from'] == cursor['before']['chat']
    assert recent['spans']['chat']['from'] == recent['spans']['chat']['to'], 'last physical row was deferred'
    assert any(item['text'] == 'deferred' for page in loaded[1:] for item in page['messages'])
    assert loaded[-1]['coverage']['spans']['chat']['from'] == 0


def test_coverage_discloses_disabled_source_and_parse_gap(tmp_path):
    write(tmp_path / 'logs/chat.jsonl', [row(1)])
    with (tmp_path / 'logs/chat.jsonl').open('a') as stream:
        stream.write('malformed\n')
    _, page = request(tmp_path, n_progress='0')
    assert page['coverage']['spans']['progress'] is None
    assert page['coverage']['spans']['chat']['gaps']


def test_the_chain_witness_is_one_durable_coordinate_space(tmp_path):
    """The store's coordinates survive every append and an older body's rotation: a page read
    before stays compatible with every page read after (the client's rule in
    web/modules/chat_history.js historyCoverage: a span's witness is listed by the newer read)."""
    live = tmp_path / 'logs/chat.jsonl'
    write(live, [row(1)])
    _, first = request(tmp_path)
    write(live, [row(2)])  # a store-unaware body appended; the next cutover imports it at its place
    _, second = request(tmp_path)
    assert _compatible(first, second) and _compatible(second, first)
    assert [message['text'] for message in second['messages']] == ['human-1', 'human-2']
    assert second['coverage']['spans']['chat']['to'] > first['coverage']['spans']['chat']['to']


def _compatible(page, head):
    # The client's rule: a span's own last prefix witness must still be listed
    # by the newer read (web/modules/chat_history.js historyCoverage).
    return page['coverage']['spans']['chat']['chain'].split('.')[-1] in head['coverage']['spans']['chat']['chain'].split('.')


def test_a_frozen_segment_that_changed_before_its_import_is_a_disclosed_gap_never_other_rows(tmp_path):
    """An archive segment rewritten under the import (its size is not the ledger's) is never
    imported: the pages spanning it name the gap (``chat_segment_unimportable``) and page on
    past it; an untouched segment imports and leaves no gap."""
    for root in (tmp_path / 'changed', tmp_path / 'frozen'):
        archives = [root / f'archive/chat_2026090{day}T000000.jsonl' for day in (1, 2)]
        for index, path in enumerate(archives):
            write(path, [row(index)])
        write(root / 'logs/chat.jsonl', [row(2)])
        dialogue_import.cutover(root)  # the archive is planned, not imported yet
        if root.name == 'changed':
            archives[1].write_bytes(archives[1].read_bytes() + b'{"direction": "in", "text": "written after the plan"}\n')
    loaded = list(pages(tmp_path / 'changed', n_human='1'))
    texts = [message['text'] for page in loaded for message in page['messages']]
    assert texts == ['human-2', 'human-0'], 'the changed segment holds no rows the store can vouch for'
    # The segment that changed and every segment whose chain place it shifted are gaps on the pages
    # spanning their coordinates; the rows the store already held keep theirs and are still delivered.
    shifted_from = (tmp_path / 'changed' / 'archive' / 'chat_20260901T000000.jsonl').stat().st_size
    spanning = [page for page in loaded if page['coverage']['spans']['chat']['to'] > shifted_from]
    assert spanning and all(page['coverage']['spans']['chat']['gaps'] == ['segment_unimportable'] for page in spanning)
    assert all('chat_segment_unimportable' in page['window']['truncated_by'] for page in spanning)
    assert all(page['coverage']['spans']['chat']['gaps'] == [] for page in loaded if page not in spanning)
    assert loaded[-1]['coverage']['spans']['chat']['from'] == 0 and loaded[-1]['has_more'] is False
    frozen = list(pages(tmp_path / 'frozen', n_human='1'))
    assert [message['text'] for page in frozen for message in page['messages']] == ['human-2', 'human-1', 'human-0']
    assert all(page['coverage']['spans']['chat']['gaps'] == [] for page in frozen)
    assert not any('segment_unimportable' in cause for page in frozen for cause in page['window']['truncated_by'])


def test_sparse_project_opens_with_its_answer_and_retained_origin_in_one_read(tmp_path):
    """The Nova shape (owner decision 2026-10-05, 1A): other rooms wrote five archives since
    the Project's last answer. Its first read still holds that answer beside the retained
    origin, covers the whole chain, and leaves no empty page to walk."""
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.project_dialogue import build_owner_message_ref

    project = create_project(tmp_path, 'sparse', name='Sparse continuity')
    origin = 'Original request'
    bind_task_to_project(tmp_path, 'sparse-root', project['id'], origin={
        'ref': build_owner_message_ref(chat_id=1, client_message_id='origin',
            ts='2026-09-01T10:00:00Z', text=origin), 'text': origin})
    write(tmp_path / 'archive/chat_20260902T000000.jsonl', [row(0, chat_id=project['chat_id'],
        direction='out', text='Later answer', ts='2026-09-02T10:00:00Z')])
    for index in range(5):
        write(tmp_path / f'archive/chat_20260903T00000{index}.jsonl', [row(i, text='foreign' + 'x' * 2000) for i in range(350)])
    write(tmp_path / 'logs/chat.jsonl', [row(9)])
    loaded = list(pages(tmp_path, chat_id=str(project['chat_id'])))
    assert len(loaded) == 1
    first, later = loaded[0]['messages']
    assert first['origin_projected'] and first['text'] == origin
    assert later['text'] == 'Later answer' and later['ts'] > first['ts']
    assert loaded[0]['coverage']['spans']['chat']['from'] == 0


def test_legacy_origin_identity_survives_source_label_and_cmid_absence(tmp_path):
    import json
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.project_dialogue import build_owner_message_ref

    project = create_project(tmp_path, 'origin', name='Retained context')
    canonical = row(1, direction='in', source='web', text='Original request', client_message_id='')
    ref = build_owner_message_ref(chat_id=1, client_message_id='', ts=canonical['ts'], text=canonical['text'])
    # Current admission requires a client id; retained legacy bindings predate it.
    bind_task_to_project(tmp_path, 'origin-root', project['id'], origin={
        'ref': {**ref, 'client_message_id': 'before-legacy-fixture'}, 'text': canonical['text']})
    path = tmp_path / 'state/project_task_bindings.json'
    bindings = json.loads(path.read_text())
    bindings['bindings']['origin-root']['source_ref']['client_message_id'] = ''
    path.write_text(json.dumps(bindings))
    _, first = request(tmp_path, chat_id=str(project['chat_id']))
    fallback = first['messages'][0]
    assert fallback['origin_projected'] and fallback['origin_id']
    write(tmp_path / 'logs/chat.jsonl', [canonical])
    _, second = request(tmp_path, chat_id=str(project['chat_id']))
    assert len(second['messages']) == 1
    adopted = second['messages'][0]
    assert adopted['history_id'] and not adopted.get('origin_projected')
    assert adopted['origin_id'] == fallback['origin_id']
