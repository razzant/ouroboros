"""Integrator seams: connector custody, live authors and bounded room intake."""
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import places
from ouroboros.gateway import host_service
from ouroboros.task_results import write_task_result
from tests.test_place_focus import focus as focus


@pytest.fixture
def client(focus, monkeypatch):
    async def authenticated(*_args):
        return 'bridge', {}
    monkeypatch.setattr(host_service, '_authenticated', authenticated)
    with TestClient(host_service.create_host_service_app(focus[0])) as value:
        yield value


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
