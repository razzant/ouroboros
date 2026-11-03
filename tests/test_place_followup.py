"""Integrator seams: connector custody, live authors and bounded room intake."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import dialogue_store, places
from ouroboros.gateway import host_service
from ouroboros.presence_runner import capture_presence_event, presence_turn_replay
from ouroboros.place_intake import decide_attention, settle_heard
from ouroboros.task_results import write_task_result, load_task_result
from tests.test_place_focus import focus as focus


@pytest.fixture
def client(focus, monkeypatch):
    async def authenticated(*_args):
        return 'bridge', {}
    monkeypatch.setattr(host_service, '_authenticated', authenticated)
    with TestClient(host_service.create_host_service_app(focus[0])) as value:
        yield value


@pytest.mark.parametrize('account', ['a', '', '*'])
@pytest.mark.parametrize('state', ['silent', 'paused'])
def test_opaque_outbox_restricts_only_owned_matching_places(focus, client, account, state):
    root, active, _, _ = focus
    restricted = places.connect_place(root, connector='bridge', provider='chat', account_id=account, conversation_id='closed')
    places.edit_place(root, restricted.place_id, 'set_state', author='owner', state=state)
    query = {'provider': 'chat', 'account_id': 'a', 'conversation_id': ''}
    assert client.get('/presence/place', params=query).json()['state'] == state
    assert client.get('/presence/place', params={**query, 'conversation_id': 'r'}).json()['state'] == 'active'
    places.edit_place(root, restricted.place_id, 'set_state', author='owner', state='active')
    assert client.get('/presence/place', params=query).json()['state'] == 'active'
    for connector, provider, owner_account in [('other', 'chat', 'a'), ('bridge', 'other', 'a'), ('bridge', 'chat', 'other')]:
        wrong = places.connect_place(root, connector=connector, provider=provider, account_id=owner_account, conversation_id='foreign')
        places.edit_place(root, wrong.place_id, 'set_state', author='owner', state=state)
    assert client.get('/presence/place', params=query).json()['state'] == 'active'


@pytest.mark.parametrize('binding', [None, '', 'legacy'])
def test_work_get_and_queue_post_bind_authenticated_retained_owner(focus, client, binding):
    root, place, _, event = focus
    presence = {'binding_id': 'legacy', 'place_id': place.place_id, 'transport_skill': 'bridge',
                'event': {'conversation_key': event.conversation_key, 'source_event_id': event.source_event_id}}
    write_task_result(root, 'owned', 'completed', metadata={'presence': presence}, result='done')
    write_task_result(root, 'foreign', 'completed', metadata={'presence': {**presence, 'transport_skill': 'other'}}, result='private')
    write_task_result(root, 'ordinary', 'completed', result='private')
    params = {} if binding is None else {'binding_id': binding}
    snapshot = {'schema_version': 1, 'source': 'bridge queue', 'conversation_key': event.conversation_key,
                'after_source_event_id': event.source_event_id, 'observed_at': '2026-10-21T12:00:00+00:00',
                'complete': True, 'pending_count': 0, 'omitted_count': 0, 'events': []}
    body = {**params, 'transport_queue': snapshot}
    assert client.get('/presence/work/owned', params=params).status_code == 200
    assert client.post('/presence/work/owned', json=body).json()['status'] == 'recorded'
    for ref in ['foreign', 'ordinary', 'absent']:
        assert client.get('/presence/work/' + ref, params=params).status_code == 404
        assert client.post('/presence/work/' + ref, json=body).status_code == 404
    assert client.get('/presence/work/owned', params={'binding_id': 'wrong'}).status_code == 404
    assert client.post('/presence/work/owned', json={**body, 'binding_id': 'wrong'}).status_code == 404
    for invalid in ({}, {'binding_id': 'legacy'}, {**body, 'extra': True}):
        assert client.post('/presence/work/owned', json=invalid).status_code == 400


@pytest.mark.parametrize('crash', [False, True])
def test_lent_author_remains_in_remote_stop_census_until_stack_ends(tmp_path, crash):
    from ouroboros.conversation_lease import live_authors, live_author
    code = '''
import sys
from pathlib import Path
from ouroboros.conversation_lease import PresenceTurnGate
lease = PresenceTurnGate(state_root=Path(sys.argv[1])/'state').acquire('room')
lease.bind_author('parked-focus')
lease.lend()
print('parked', flush=True)
sys.stdin.readline()
lease.release()
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'parked'
        assert live_authors(drive_root=tmp_path) == ['parked-focus']
        assert live_author('room', drive_root=tmp_path) == ''
        if crash:
            child.kill()
        child.communicate('end\n', timeout=15)
        assert live_authors(drive_root=tmp_path) == []
        assert not list((tmp_path/'state'/'presence_turn_gate').glob('author-*.live'))
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)


@pytest.mark.parametrize('enabled', [False, True])
def test_continuing_focus_checks_autonomous_gate_before_generation(focus, monkeypatch, enabled):
    from ouroboros.presence_continuation import ReviewWaitBinding, presence_review_wait, conversation_lost
    from ouroboros.conversation_lease import PresenceTurnGate
    import supervisor.consciousness_carriers as carriers
    root, _, admission, event = focus
    task = capture_presence_event(admission, event, drive_root=root)
    lease = PresenceTurnGate(state_root=root/'state').acquire(event.conversation_key)
    lease.bind_author(task['id'])
    lease.lend()
    ctx = SimpleNamespace(task_id=task['id'], task_metadata=task['metadata'])
    calls = []
    monkeypatch.setattr(carriers, 'autonomous_work_enabled', lambda t, r: calls.append((t, r)) or enabled, raising=False)
    monkeypatch.setattr('ouroboros.tool_access.canonical_data_root', lambda _ctx: root)
    monkeypatch.setattr('ouroboros.presence_continuation._yield_conversation', lambda *_: True)
    monkeypatch.setattr('ouroboros.owner_wait.direct_owner_wait', lambda *_: 'messages')
    monkeypatch.setattr('ouroboros.presence_continuation.reentry_note', lambda *_: 'resume')
    messages = []
    try:
        presence_review_wait(ReviewWaitBinding(lease, root, task['id'], '', event), ctx,
                           {'review_binding': 'left-review'}, messages)
        assert calls and calls[0][0]['metadata']['wake_cause'] == 'focus'
        assert lease.held() is enabled
        assert bool(messages) is enabled
        assert conversation_lost(ctx) == ('' if enabled else 'cancelled')
    finally:
        lease.release()


@pytest.mark.parametrize('state,enabled,facts,reason', [
    ('paused', True, ('mention',), 'paused'), ('active', False, ('mention',), 'off'),
    ('active', True, (), 'no_wake_match'),
])
def test_heard_only_replay_has_no_task_or_source_blob(focus, monkeypatch, state, enabled, facts, reason):
    root, place, admission, event = focus
    places.edit_place(root, place.place_id, 'set_state', author='owner', state=state)
    monkeypatch.setattr('ouroboros.presence_admission.focus_enabled', lambda _: enabled)
    event = replace(event, facts=facts)
    task = capture_presence_event(admission, event, drive_root=root)
    decision, details = decide_attention(root, admission, event)
    assert decision == 'silent' and details['reason'] == reason
    original = list(dialogue_store.iter_rows(root, 'chat'))[0]
    assert 'context_source_ref' not in original.row
    assert settle_heard(root, task, event, reason).outcome == 'silent'
    assert load_task_result(root, task['id']) is None
    assert not list((root/'task_results').rglob('presence-event-*.json'))
    with dialogue_store.transaction(root, write=False) as conn:
        row = conn.execute('SELECT body,dispatch_input FROM dialogue WHERE stream=? AND seq=?', ('chat', original.seq)).fetchone()
    assert json.loads(row['dispatch_input'])['place_attention']['reason'] == reason
    assert json.loads(row['body']) == original.row
    places.edit_place(root, place.place_id, 'set_state', author='owner', state='active')
    monkeypatch.setattr('ouroboros.presence_admission.focus_enabled', lambda _: True)
    assert presence_turn_replay(root, task['id'], event.conversation_key, task['metadata']['presence_event_identity']).outcome == 'silent'
    next_event = replace(event, source_event_id='next', facts=('mention',))
    assert decide_attention(root, admission, next_event)[0] == 'focus'
    next_task = capture_presence_event(admission, next_event, drive_root=root)
    from ouroboros.place_arrivals import intake_lock, retain_focus_source
    from ouroboros import chat_chain
    before_source = list(dialogue_store.iter_rows(root, 'chat'))[-1]
    address = chat_chain.row_address(before_source.row)
    with intake_lock(root, next_event.conversation_key):
        write_task_result(root, next_task['id'], 'running', metadata=next_task['metadata'], chat_id=next_task['chat_id'])
        retain_focus_source(root, next_task['id'])
    assert settle_heard(root, next_task, next_event, 'off') is None
    assert load_task_result(root, next_task['id'])['status'] == 'running'
    assert list(dialogue_store.iter_rows(root, 'chat'))[-1].row['context_source_ref']
    assert chat_chain.row_address(list(dialogue_store.iter_rows(root, 'chat'))[-1].row) == address
    assert list((root/'task_results').rglob('presence-event-*.json'))


def test_room_and_activity_indexes_are_independent_of_heard_history(focus, monkeypatch):
    from ouroboros.place_projection import place_activity
    from ouroboros import place_arrivals, task_store
    root, place, admission, event = focus
    assert places.place_room_index(root, place.place_id) == []
    captured_at = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    monkeypatch.setattr('ouroboros.presence_runner.utc_now_iso', lambda: captured_at.isoformat())
    task = capture_presence_event(admission, event, drive_root=root)
    assert place_activity(root, place)['last_activity_at'] == captured_at.isoformat()
    assert places.place_room_index(root, place.place_id)[0]['unanswered_count'] is None  # no task store yet
    write_task_result(root, task['id'], 'running', metadata=task['metadata'], chat_id=task['chat_id'])
    assert places.place_room_index(root, place.place_id) == [{'room_chat_id': task['chat_id'], 'unanswered_count': 1,
                                                            'address': f"chat:{task['chat_id']}"}]
    assert place_activity(root, place)['unanswered'] == 1
    write_task_result(root, task['id'], 'completed')
    assert place_activity(root, place)['unanswered'] == 0
    # Prebuild the indexes, then add history directly inside one fixture transaction.
    with dialogue_store.transaction(root, write=True) as conn:
        seq = conn.execute("SELECT MAX(seq+length) FROM dialogue WHERE stream='chat'").fetchone()[0]
        for i in range(2000):
            row = {'type': 'place_arrival', 'place_id': place.place_id, 'chat_id': task['chat_id'],
                   'ts': (captured_at + timedelta(seconds=i+1)).isoformat(), 'text': 'heard', 'direction': 'in'}
            body = dialogue_store.encode(row)
            dialogue_store.insert_row(conn, 'chat', seq+i*1000, row, body, length=len(body)+1, segment=1, ordinal=i+1, pos=None)
    from contextlib import contextmanager
    original = dialogue_store.transaction
    @contextmanager
    def bounded(*args, **kw):
        with original(*args, **kw) as conn:
            calls = 0
            def progress():
                nonlocal calls
                calls += 1
                return int(calls > 15)
            conn.set_progress_handler(progress, 100)  # each transaction has its own ~1,500-op budget
            try:
                yield conn
            finally:
                conn.set_progress_handler(None, 0)
    monkeypatch.setattr(dialogue_store, 'transaction', bounded)
    monkeypatch.setattr(task_store, 'read_document', lambda *a, **kw: pytest.fail('per-row task document load'))
    assert place_activity(root, place) == {'unanswered': 0,
                                         'last_activity_at': (captured_at + timedelta(seconds=2000)).isoformat()}
    assert len(places.place_room_index(root, place.place_id)) == 1
    assert place_arrivals.account_candidates(root, place.place_id) == []  # bounded despite 2,000 old arrivals
    monkeypatch.setattr(place_arrivals, '_open_counts', lambda *_: None)
    assert place_activity(root, place)['unanswered'] is None
    assert places.place_room_index(root, place.place_id)[0]['unanswered_count'] is None


def test_partial_migration_preserves_successes_failures_and_source_bytes(tmp_path, monkeypatch):
    from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
    from tests.frozen_presence_bindings import save_presence_binding
    from ouroboros.place_projection import places_view
    good = SimpleNamespace(skill_dir=tmp_path, manifest=SimpleNamespace(raw_extra={'presence': {'instructions': 'Exact words'}}, contributes=[]))
    bad = SimpleNamespace(skill_dir=tmp_path, manifest=SimpleNamespace(raw_extra={'presence': 'invalid'}, contributes=[]))
    empty = SimpleNamespace(skill_dir=tmp_path, manifest=SimpleNamespace(raw_extra={'presence': {}}, contributes=[]))
    skills = {'good': good, 'bad': bad, 'empty': empty}
    monkeypatch.setattr('ouroboros.skill_loader.find_skill', lambda _r, n: skills.get(n))
    for i, name in enumerate(['good', 'bad', 'empty', 'missing']):
        save_presence_binding(tmp_path, PresenceBinding(f'{i+1:032x}', 'bridge', name,
            PresenceEndpoint('chat', 'a', name), PresenceEndpoint('chat', 'a', name), True))
    source = tmp_path/'state'/'presence_bindings.json'
    before = source.read_bytes()
    imported = places.load_places(tmp_path)
    assert len(imported) == 1 and imported[0].notes[0].text == 'Exact words'
    problems = places.migration_problems(tmp_path)
    assert len(problems) == 3 and {p['binding_id'] for p in problems} == {f'{i:032x}' for i in (2, 3, 4)}
    assert {p['code']: p['detail'] for p in problems} == {
        'places_migration_source_invalid': 'its presence profile could not be read',
        'places_migration_instructions_missing': 'the behavior skill empty has no instructions',
        'places_migration_source_missing': 'the behavior skill missing is not installed',
    }
    assert source.read_bytes() == before
    monkeypatch.setattr('ouroboros.skill_loader.find_skill', lambda *_: pytest.fail('migration repeated'))
    assert places.load_places(tmp_path) == imported
    monkeypatch.setattr('ouroboros.place_projection.place_declarations', lambda _: [])
    monkeypatch.setattr('ouroboros.presence_admission.focus_allowance', lambda _: {})
    assert places_view(tmp_path)['migration_problems'] == problems
    from starlette.applications import Starlette
    from ouroboros.gateway.places_api import places_routes
    monkeypatch.setattr('ouroboros.gateway.places_api.request_drive_root', lambda _: tmp_path)
    with TestClient(Starlette(routes=places_routes())) as owner:
        assert owner.get('/api/places').json()['migration_problems'] == problems
    new = places.connect_place(tmp_path, connector='bridge', provider='chat', conversation_id='missing')
    assert new in places.load_places(tmp_path)


@pytest.mark.parametrize('error,code,detail', [
    (OSError('private filesystem path'), 'places_migration_source_invalid', 'its presence profile could not be read'),
    (ValueError('raw JSON bytes'), 'places_migration_source_invalid', 'its presence profile could not be read'),
    (places.PlaceError('unexpected_import_error', 'internal field'), 'unexpected_import_error',
     'the old connection could not be imported'),
])
def test_migration_problem_uses_owner_fragment_without_losing_code(tmp_path, monkeypatch, error, code, detail):
    from ouroboros import places_migration
    good = places.Place('good', 'bridge', 'chat')
    monkeypatch.setattr('ouroboros.presence_bindings.list_presence_bindings', lambda _: [
        SimpleNamespace(binding_id='bad', behavior_skill='behavior'), SimpleNamespace(binding_id='good')])
    def migrate(_root, binding):
        if binding.binding_id == 'bad':
            raise error
        return good
    monkeypatch.setattr(places_migration, '_legacy_place', migrate)
    imported, problems = places_migration.legacy_places(tmp_path)
    assert imported == [good]
    assert problems == [{'binding_id': 'bad', 'code': code, 'detail': detail}]


def test_author_lifetime_closes_its_handle_before_removing_the_file(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from ouroboros.conversation_lease import PresenceTurnGate, live_authors
    lease = PresenceTurnGate(state_root=tmp_path/'state').acquire('room')
    lease.bind_author('author')
    path, fd = lease._live_path, lease._live_fd
    unlink = Path.unlink
    def windows_unlink(candidate, *args, **kwargs):
        if candidate == path:
            try:
                os.fstat(fd)
            except OSError:
                pass
            else:
                raise PermissionError('cannot remove an open author handle')
        return unlink(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', windows_unlink)
    assert path.exists() and live_authors(drive_root=tmp_path) == ['author']
    lease.release()
    assert not path.exists() and live_authors(drive_root=tmp_path) == []
