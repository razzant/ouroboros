"""Notification clicks resolve canonical room identity beyond the sidebar cap."""
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.projects import api_projects_list
from ouroboros.projects_registry import begin_project_deletion, create_project


def _client(root):
    app = Starlette(routes=[Route('/api/projects', api_projects_list)])
    app.state.drive_root = root
    return TestClient(app)


def test_addressed_notification_destination_beyond_both_list_caps(tmp_path, monkeypatch):
    from ouroboros import projects_registry

    old = create_project(tmp_path, 'older', name='Old Project')
    for index in range(205):
        create_project(tmp_path, f'newer-{index}')
    with _client(tmp_path) as client:
        sidebar = client.get('/api/projects').json()['projects']
        assert len(sidebar) == 200
        assert old['id'] not in {row['id'] for row in sidebar}
        monkeypatch.setattr(projects_registry, 'list_sidebar_projects',
                            lambda *_args, **_kw: (_ for _ in ()).throw(AssertionError('sidebar lookup')))
        response = client.get('/api/projects', params={'chat_id': old['chat_id']})
        assert response.status_code == 200
        [room] = response.json()['projects']
        assert set(room) == set(sidebar[0]), 'addressed and sidebar replies share the same projection'
        assert room['id'] == old['id']
        assert room['chat_id'] == old['chat_id']
        assert room['lifecycle'] == 'active'


def test_missing_or_inactive_destination_never_substitutes_main(tmp_path):
    old = create_project(tmp_path, 'older')
    begin_project_deletion(tmp_path, old['id'])
    with _client(tmp_path) as client:
        for address in (old['chat_id'], -999, 1, 0):
            response = client.get('/api/projects', params={'chat_id': address})
            assert response.status_code == 200
            assert response.json() == {'projects': []}


def test_unreadable_registry_is_explicit_failure_and_read_does_not_repair_it(tmp_path):
    project = create_project(tmp_path, 'p')
    registry = tmp_path / 'state' / 'projects.json'
    registry.write_text('{broken')
    with _client(tmp_path) as client:
        response = client.get('/api/projects', params={'chat_id': project['chat_id']})
        assert response.status_code == 500
        assert 'error' in response.json()
    assert registry.read_text() == '{broken'


def test_addressed_project_distinguishes_an_import_in_progress_from_a_missing_room(tmp_path):
    from tests._dialogue import import_all, seed

    project = create_project(tmp_path, 'p', origin='reconcile')
    seed(tmp_path, archives={'chat_20260101T000000.jsonl': [{'chat_id': project['chat_id'], 'text': 'older words'}]})
    with _client(tmp_path) as client:
        response = client.get('/api/projects', params={'id': project['id']})
        assert response.status_code == 503 and response.json()['reason_code'] == 'migrating'
        assert client.get('/api/projects', params={'id': 'missing'}).json() == {'projects': []}
        import_all(tmp_path)
        assert client.get('/api/projects', params={'id': project['id']}).json()['projects'][0]['id'] == project['id']
