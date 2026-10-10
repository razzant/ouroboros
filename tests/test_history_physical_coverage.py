"""Coverage uses delivered physical boundaries of the real history endpoint."""
from tests.test_chat_history_paging import isolated_runtime, write, row, request, pages  # noqa: F401
from ouroboros.gateway.history_paging import decode_cursor


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


def test_root_witness_survives_rotation_but_not_prefix_replacement(tmp_path):
    live = tmp_path / 'logs/chat.jsonl'
    write(live, [row(1)])
    _, first = request(tmp_path)
    archive = tmp_path / 'archive/chat_20260901T000000.jsonl'
    archive.parent.mkdir(); live.rename(archive)
    write(live, [row(2)])
    _, second = request(tmp_path)
    assert _compatible(first, second), 'rotation keeps the renamed live segment and its base'
    replacement = archive.with_suffix('.replacement')
    write(replacement, [row(3)])
    replacement.replace(archive)
    _, third = request(tmp_path)
    assert not _compatible(first, third) and not _compatible(second, third)


def _compatible(page, head):
    # The client's rule: a span's own last prefix witness must still be listed
    # by the newer read (web/modules/chat_history.js historyCoverage).
    return page['coverage']['spans']['chat']['chain'].split('.')[-1] in head['coverage']['spans']['chat']['chain'].split('.')


def test_replacing_a_later_archive_invalidates_every_page_read_before(tmp_path):
    archives = [tmp_path / f'archive/chat_2026090{day}T000000.jsonl' for day in (1, 2)]
    for index, path in enumerate(archives):
        write(path, [row(index)])
    write(tmp_path / 'logs/chat.jsonl', [row(2)])
    loaded = list(pages(tmp_path, n_human='1'))
    assert len(loaded) == 3 and all(_compatible(page, loaded[0]) for page in loaded)
    # Same bytes and size under a new inode: offsets would still line up, but
    # the rows read from the old file are no longer this chain's rows.
    replacement = archives[1].with_suffix('.replacement')
    write(replacement, [row(1)])
    replacement.replace(archives[1])
    _, head = request(tmp_path, n_human='1')
    assert [_compatible(page, head) for page in loaded] == [False, False, False]
    assert head['coverage']['spans']['chat']['chain'].split('.')[0] == loaded[0]['coverage']['spans']['chat']['chain'].split('.')[0], \
        'the untouched first archive keeps its own prefix witness'


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
