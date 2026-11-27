"""The local room picker joins recent root rows to exact saved physical attempts."""
import json

import pytest

from ouroboros import context_report_archive as archive, dialogue_store
from ouroboros.launcher_context_report import ContextArchivePage
from tests._context_reports import context_reports, _dispatch


def test_recent_archive_reads_only_selected_room_attempts(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'local'
    reports = context_reports(root, monkeypatch, capsys)
    monkeypatch.setattr(archive.projects_registry, 'list_sidebar_projects', lambda *a, **kw: [
        {'chat_id': 1234, 'name': 'Tide tables'}, {'chat_id': 1235, 'name': 'Other work'},
    ])
    monkeypatch.setattr(archive.projects_registry, 'list_reserved_projects', lambda *a, **kw: [
        {'id': 'tide', 'chat_id': 1234, 'name': 'Tide tables'}, {'id': 'other', 'chat_id': 1235, 'name': 'Other work'},
    ])
    dialogue_store.append(root, 'chat', {'chat_id': 1234, 'task_id': 'report-task', 'direction': 'out'})
    dialogue_store.append(root, 'chat', {'chat_id': 1234, 'task_id': 'telegram-turn',
                                         'direction': 'out', 'logical_room_chat_id': 1}, appearances=['room:1'])
    telegram = _dispatch(root, 'telegram-turn')['attempt_id']
    dialogue_store.append(root, 'chat', {'chat_id': 1, 'task_id': 'unknown-room', 'direction': 'out'})
    uncertain = _dispatch(root, 'unknown-room')['attempt_id']
    resolve = archive.projects_registry.RoomResolver.__call__

    def unavailable(self, task, *args):
        if task == 'unknown-room':
            assert self.strict
            raise ValueError('project_source_unknown')
        return resolve(self, task, *args)

    monkeypatch.setattr(archive.projects_registry.RoomResolver, '__call__', unavailable)
    dialogue_store.append(root, 'progress', {'chat_id': 1234, 'task_id': 'retry-task', 'parent_task_id': 'report-task'})
    dialogue_store.append(root, 'chat', {'chat_id': 1235, 'task_id': 'retry-task', 'direction': 'out'})
    dialogue_store.append(root, 'chat', {'chat_id': 1, 'task_id': 'retry-task',
                                         'direction': 'out', 'type': 'project_completion_summary'})
    dialogue_store.append(root, 'chat', {'chat_id': 1, 'task_id': 'retry-task', 'type': 'project_started'})
    _dispatch(root, 'report-task', category='review')
    assert archive.rooms(root)[1]['name'] == 'Tide tables'
    main_selection, _ = archive.recent_requests(root, 1)
    assert set(main_selection) == {telegram}
    with pytest.raises(ValueError, match='not offered'):
        archive.document(root, main_selection, uncertain)
    with pytest.raises(ValueError):
        archive.recent_requests(root, 0)

    page = ContextArchivePage(root)
    assert not page.rooms()['ok']
    page._enabled.set()
    first, second = reports['attempts']
    noisy = json.loads(reports['noisy'])['facts']['attempt_id']
    assert [row['attempt_id'] for row in page.requests(1234)['value']] == [noisy, second, first]
    assert page.report(first)['value'].encode() == reports['first']
    assert not page.report(reports['retried'][0])['ok']
    assert not page.report(telegram)['ok']
    assert page.requests(1235)['ok']
    assert not page.report(first)['ok']
    assert page.report(reports['retried'][0])['ok']
    page._enabled.clear()
    assert not page.report(reports['retried'][0])['ok']


def test_picker_reuses_strict_registry_and_defers_legacy_origin_lookup(tmp_path, monkeypatch):
    from hashlib import sha256
    from ouroboros import chat_chain, project_dialogue, projects_registry

    project = {'id': 'tide', 'chat_id': 1234, 'name': 'Tide tables'}
    origin = {'chat_id': 1, 'ts': '2026-01-01T00:00:00Z', 'user_id': 1, 'source': 'web',
              'direction': 'in', 'text': 'an older originating request'}
    saved = {'chat_id': 1, 'text_sha256': sha256(origin['text'].encode()).hexdigest()}
    dialogue_store.append(tmp_path, 'chat', origin)
    bindings = {}
    for task in ('work-a', 'work-b'):
        bindings[task] = {'task_id': task, 'project_id': 'tide', 'project_chat_id': 1234, 'source_ref': saved}
        dialogue_store.append(tmp_path, 'chat', {'task_id': task, 'chat_id': 1, 'direction': 'out'})
    registry_reads, origin_reads = [], []
    read_origins = project_dialogue.owner_message_entries

    def registry(_root, *, strict):
        registry_reads.append(strict)
        return [project]

    def lookup(root, refs):
        origin_reads.append(tuple(refs))
        return read_origins(root, refs)

    monkeypatch.setattr(projects_registry, 'list_reserved_projects', registry)
    monkeypatch.setattr(projects_registry, '_load_bindings', lambda *a, **kw: {'bindings': bindings})
    monkeypatch.setattr(projects_registry, 'get_reserved_project',
                        lambda *a, **kw: pytest.fail('strict resolver reopened the registry'))
    monkeypatch.setattr(project_dialogue, 'owner_message_entries', lookup)
    assert chat_chain.recent_outbound_task_ids(tmp_path, 1234, row_limit=3, task_limit=2) == ['work-b', 'work-a']
    assert registry_reads == [True] and origin_reads == []

    _, origins, admits, _ = chat_chain.room_keys(tmp_path, 1234, strict=True, include_origin_rows=False)
    assert not origins and not origin_reads
    assert not admits({'chat_id': 1, 'direction': 'out', 'text': 'not an owner source'})
    assert not origin_reads
    assert admits(origin) and admits(origin)  # the same checksum-only source still joins the project
    assert len(origin_reads) == 1
