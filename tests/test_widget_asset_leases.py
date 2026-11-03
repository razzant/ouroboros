"""Scope, captured bytes, auth and lifecycle negatives for the real lease handlers."""
from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import extension_loader as loader
from ouroboros import server_auth
from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_child_catalog import _stage_out_of_process_surfaces
from ouroboros.extension_plugin_api import PluginAPIImpl
from ouroboros.extension_registry_state import _PluginAPIConfig
from ouroboros.extension_ui_validation import validate_ui_render
from ouroboros.extension_widget_assets import capture_widget_sources
from ouroboros.gateway.widget_assets import api_widget_asset_lease
from ouroboros.gateway.widgets import api_widgets
from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state
from tests._shared import clean_extension_runtime_state
from tests.widget_esm_fixture import BINARY, NAME, RENDER, install


@pytest.fixture
def env(tmp_path, monkeypatch):
    clean_extension_runtime_state()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "fixture-password")
    skill_dir = install(tmp_path)
    skill = find_skill(tmp_path, NAME)
    assert loader.load_extension(skill, lambda: {}, drive_root=tmp_path) is None
    app = Starlette(routes=[Route('/api/widgets', api_widgets),
        Route('/api/widgets/asset-leases', api_widget_asset_lease, methods=['POST']),
        Route('/api/widgets/asset-leases/{op}', api_widget_asset_lease, methods=['POST'])])
    app.state.drive_root = tmp_path
    gate = server_auth.NetworkAuthGate(app)
    client = TestClient(gate, client=('192.0.2.8', 12345))
    assert client.get('/api/widgets').status_code == 401
    assert client.post('/auth/login', json={'password': 'fixture-password'}).status_code == 200
    try:
        yield client, gate, skill_dir, skill, tmp_path
    finally:
        client.close()
        clean_extension_runtime_state()


def open_lease(client, tab='main'):
    row = next(t for t in client.get('/api/widgets').json()['ui_tabs'] if t['tab_id'] == tab)
    response = client.post('/api/widgets/asset-leases', json={k: row[k] for k in ('skill', 'tab_id', 'generation')})
    assert response.status_code == 200, response.text
    return response.json()


def test_only_declared_bytes_copyable_but_not_an_api_credential(env):
    client, gate, _, _, _ = env
    lease = open_lease(client)
    other = open_lease(client, 'other')
    with TestClient(gate, client=('192.0.2.9', 12345)) as reader:
        response = reader.get(lease['base_url'] + 'data.bin', headers={'Origin': 'null'})
        assert response.status_code == 200 and response.content == BINARY
        assert response.headers['content-type'] == 'application/octet-stream'
        assert response.headers['access-control-allow-origin'] == 'null'
        assert response.headers['x-content-type-options'] == 'nosniff'
        assert response.headers['cache-control'] == 'no-store'
        assert 'access-control-allow-credentials' not in response.headers
        assert reader.head(lease['base_url'] + 'data.bin').headers['content-length'] == '256'
        for path in ('other.js', 'unlisted.txt', 'plugin.py', 'SKILL.md', 'settings.json', 'api/settings',
                     '%2e%2e/data.bin', 'app%5clib.js', 'app//lib.js', '.hidden.js'):
            assert reader.get(lease['base_url'] + path).status_code in {400, 404}, path
        assert reader.get(other['base_url'] + 'data.bin').status_code == 404
        assert reader.post(lease['base_url'] + 'data.bin', json={}).status_code == 405
        for path in ('/api/widgets', '/api/settings', '/api/chat/history', '/api/extensions/neighbor/ping',
                     '/api/widgets/asset-leases'):
            assert reader.get(path, headers={'Authorization': f"Bearer {lease['token']}"}).status_code == 401
        assert reader.post('/api/widgets/asset-leases/renew', json={'token': lease['token']}).status_code == 401
    assert client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']},
                       headers={'Origin': 'null'}).status_code == 403
    assert client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']},
                       headers={'Origin': 'https://elsewhere.invalid'}).status_code == 403
    assert client.get('/api/widget-assets/' + 'x' * 43 + '/data.bin').status_code == 404


def test_controls_take_the_browsers_fetch_metadata_behind_a_host_rewriting_proxy(env):
    """A proxy that rewrites Host or terminates TLS changes the URL this server sees, not the browser's own
    Sec-Fetch-Site. Opaque, same-site and cross-site initiators stay refused; without Fetch Metadata, Origin must
    still name the URL this server sees."""
    client = env[0]
    row = next(t for t in client.get('/api/widgets').json()['ui_tabs'] if t['tab_id'] == 'main')
    body = {k: row[k] for k in ('skill', 'tab_id', 'generation')}
    page = 'https://ouro.example'  # the proxy's origin; this server sees http://testserver
    lease = client.post('/api/widgets/asset-leases', json=body, headers={'Origin': page, 'Sec-Fetch-Site': 'same-origin'})
    assert lease.status_code == 200, lease.text
    renew = client.post('/api/widgets/asset-leases/renew', json={'token': lease.json()['token']},
                        headers={'Origin': page, 'Sec-Fetch-Site': 'same-origin'})
    assert renew.status_code == 200, renew.text
    for headers in ({'Origin': page}, {'Origin': 'null', 'Sec-Fetch-Site': 'same-origin'},
                    {'Origin': 'null', 'Sec-Fetch-Site': 'cross-site'}, {'Origin': page, 'Sec-Fetch-Site': 'same-site'},
                    {'Origin': 'https://elsewhere.invalid', 'Sec-Fetch-Site': 'cross-site'}):
        assert client.post('/api/widgets/asset-leases', json=body, headers=headers).status_code == 403, headers


def test_reload_even_same_hash_and_drift_do_not_mix_publications(env):
    client, _, skill_dir, skill, root = env
    old = open_lease(client)
    (skill_dir / 'data.bin').write_bytes(b'changed unreviewed bytes')
    assert client.get(old['base_url'] + 'data.bin').content == BINARY
    assert client.post('/api/widgets/asset-leases/renew', json={'token': old['token']}).status_code == 200
    (skill_dir / 'data.bin').write_bytes(BINARY)
    loader.unload_extension(NAME)
    assert loader.load_extension(skill, lambda: {}, drive_root=root) is None
    assert client.get(old['base_url'] + 'data.bin').status_code == 404
    fresh = open_lease(client)
    assert fresh['token'] != old['token']
    assert client.get(fresh['base_url'] + 'data.bin').content == BINARY


@pytest.mark.parametrize('end', ['revoke', 'disable', 'review', 'grant', 'logout', 'password', 'expiry'])
def test_lifecycle_end_cannot_be_renewed_or_resurrected(env, monkeypatch, end):
    client, gate, _, _, root = env
    lease = open_lease(client)
    if end == 'revoke':
        assert client.post('/api/widgets/asset-leases/revoke', json={'token': lease['token']}).status_code == 200
    elif end == 'disable':
        save_enabled(root, NAME, False)
    elif end == 'review':
        save_review_state(root, NAME, SkillReviewState(status='fail', content_hash='changed'))
    elif end == 'grant':
        monkeypatch.setattr('ouroboros.extension_liveness.grant_status_for_skill', lambda *a: {'usable': False})
    elif end == 'logout':
        client.get('/auth/logout', follow_redirects=False)
    elif end == 'password':
        monkeypatch.setattr(server_auth, 'get_configured_network_password', lambda: 'different-password')
    elif end == 'expiry':
        # Time control belongs to the unit test; real mounts use the production TTL.
        gate.widget_assets._leases[lease['token']] = dataclasses.replace(
            gate.widget_assets._leases[lease['token']], expires=0)
    assert client.get(lease['base_url'] + 'data.bin').status_code == 404
    assert client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']}).status_code in {401, 409}
    if end == 'disable':
        save_enabled(root, NAME, True)
        assert client.get(lease['base_url'] + 'data.bin').status_code == 404


def test_deadline_runs_without_contact_and_only_parent_renewal_extends_it(env, monkeypatch):
    client, gate, _, _, _ = env
    clock = [1000.0]
    monkeypatch.setattr('ouroboros.gateway.widget_assets.time', SimpleNamespace(monotonic=lambda: clock[0]))
    lease = open_lease(client)
    assert gate.widget_assets._leases[lease['token']].expires == 1120.0
    clock[0] = 1040.0
    assert client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']}).status_code == 200
    assert gate.widget_assets._leases[lease['token']].expires == 1160.0
    clock[0] = 1159.9
    assert client.get(lease['base_url'] + 'data.bin').status_code == 200
    # No timer, websocket or explicit disposal needs to reach the server.
    clock[0] = 1160.0
    assert client.get(lease['base_url'] + 'data.bin').status_code == 404
    assert client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']}).status_code == 409


@pytest.mark.parametrize('end', ['logout', 'password_cycle'])
def test_revocation_cancels_a_mint_already_admitted_but_queued_for_a_worker(env, monkeypatch, end):
    client, gate, _, _, _ = env
    row = next(t for t in client.get('/api/widgets').json()['ui_tabs'] if t['tab_id'] == 'main')
    fields = {k: row[k] for k in ('skill', 'tab_id', 'generation')}
    admitted, release = threading.Event(), threading.Event()
    original_open = gate.widget_assets.open

    def queued_open(*args):
        admitted.set()
        assert release.wait(10), 'test did not release the admitted mint'
        return original_open(*args)

    monkeypatch.setattr(gate.widget_assets, 'open', queued_open)
    with TestClient(gate, client=('192.0.2.8', 54321)) as pending, ThreadPoolExecutor(max_workers=1) as pool:
        pending.cookies.update(client.cookies)
        future = pool.submit(pending.post, '/api/widgets/asset-leases', json=fields)
        try:
            assert admitted.wait(10), 'mint did not reach the worker'
            if end == 'logout':
                assert client.get('/auth/logout', follow_redirects=False).status_code == 303
            else:
                monkeypatch.setattr(server_auth, 'get_configured_network_password', lambda: 'changed-password')
                assert client.get('/api/widgets').status_code == 401
                monkeypatch.setattr(server_auth, 'get_configured_network_password', lambda: 'fixture-password')
                assert client.get('/api/widgets').status_code == 200
        finally:
            release.set()
        assert future.result(timeout=10).status_code == 409
    assert gate.widget_assets._leases == {}
    assert not gate.widget_assets._admissions
    # Fresh authenticated admission still works: no new global session policy.
    assert client.post('/auth/login', json={'password': 'fixture-password'}).status_code == 200
    assert client.post('/api/widgets/asset-leases', json=fields).status_code == 200


def test_withdrawn_persisted_grant_refuses_old_reads_and_new_leases(env):
    from ouroboros.skill_loader import compute_content_hash, save_skill_grants, skill_state_dir_path

    client, _, skill_dir, _, root = env
    loader.unload_extension(NAME)
    manifest = skill_dir / 'SKILL.md'
    manifest.write_text(manifest.read_text().replace('[widget, route]', '[widget, route, presence]'))
    revision = compute_content_hash(skill_dir)
    save_review_state(root, NAME, SkillReviewState(status='pass', content_hash=revision))
    save_skill_grants(root, NAME, [], content_hash=revision, requested_keys=[],
                      granted_permissions=['presence'], requested_permissions=['presence'])
    assert loader.load_extension(find_skill(root, NAME), dict, drive_root=root) is None
    lease = open_lease(client)
    row = next(t for t in client.get('/api/widgets').json()['ui_tabs'] if t['tab_id'] == 'main')
    (skill_state_dir_path(root, NAME) / 'grants.json').unlink()
    assert client.get(lease['base_url'] + 'data.bin').status_code == 404
    assert client.post('/api/widgets/asset-leases', json={k: row[k] for k in
                       ('skill', 'tab_id', 'generation')}).status_code == 409


def test_access_and_fault_logs_redact_the_read_link(env):
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.widget_assets import widget_asset_response
    from ouroboros.gateway.widgets import _fault_record

    client, gate, _, skill, _ = env
    lease = open_lease(client)
    path = lease['base_url'] + 'data.bin'
    scope = {'type': 'http', 'method': 'GET', 'path': path, 'raw_path': path.encode(),
             'query_string': lease['token'].encode(), 'headers': []}
    response = asyncio.run(widget_asset_response(Request(scope), gate.widget_assets))
    assert response.body == BINARY
    assert lease['token'] not in str(scope)
    record = _fault_record({'skill': NAME, 'tab_id': 'main', 'revision': skill.content_hash,
                            'where': 'board', 'kind': 'error', 'message': path, 'source': path, 'line': 1})
    assert record is not None and lease['token'] not in str(record)
    assert '[redacted]' in record['source']


def test_capture_rejects_symlinks_missing_and_unreviewed_changes(env):
    _, _, skill_dir, skill, _ = env
    source = skill_dir / 'data.bin'
    source.unlink()
    source.symlink_to(skill_dir / 'data.json')
    with pytest.raises(ExtensionRegistrationError, match='symlink'):
        capture_widget_sources(skill, RENDER)
    source.unlink()
    with pytest.raises(ExtensionRegistrationError):
        capture_widget_sources(skill, RENDER)
    source.write_bytes(b'changed')
    with pytest.raises(ExtensionRegistrationError, match='payload changed'):
        capture_widget_sources(skill, RENDER)
    (skill_dir / 'app').rename(skill_dir / 'real-app')
    (skill_dir / 'app').symlink_to(skill_dir / 'real-app', target_is_directory=True)
    with pytest.raises(ExtensionRegistrationError, match='symlink'):
        capture_widget_sources(skill, RENDER)


def test_out_of_process_catalog_captures_the_same_finite_files(env):
    _, _, _, skill, root = env
    loader.unload_extension(NAME)
    api = PluginAPIImpl(_PluginAPIConfig(skill_name=NAME, permissions=['widget'], env_allowlist=[],
        state_dir=root / 'state', settings_reader=dict, drive_root=root, skill_dir=skill.skill_dir, loaded_skill=skill))
    _stage_out_of_process_surfaces(api, skill, {'ui_tabs': [
        {'key': f'{NAME}:main', 'tab_id': 'main', 'title': 'Module files', 'render': RENDER}]})
    assert set(api._staged.widget_sources) == {RENDER['entry'], *RENDER['assets']}
    assert api._staged.widget_sources['data.bin'] == BINARY
    assert loader.snapshot()['ui_tabs'] == []  # still staged, no partial publication


@pytest.mark.parametrize('changes', [
    {'entry_type': 'esm'}, {'assets': ['../escape.js']}, {'assets': ['*.js']}, {'assets': ['page.html']},
    {'assets': ['.private.js']}, {'assets': ['app/main.js']}, {'assets': ['x.js', 'x.js']},
    {'assets': 'data.bin'}, {'assets': ['a.js'] * 256}, {'kind': 'iframe'},
])
def test_schema_refuses_implicit_or_unbounded_files(changes):
    with pytest.raises(ExtensionRegistrationError):
        validate_ui_render({**RENDER, **changes})


def test_default_mjs_is_still_classic():
    assert validate_ui_render({'kind': 'module', 'entry': 'classic.mjs'}).get('entry_type') is None
    with pytest.raises(ExtensionRegistrationError, match='require entry_type'):
        validate_ui_render({'kind': 'module', 'entry': 'classic.mjs', 'assets': []})


def test_reads_never_renew_and_copied_tokens_cannot_control_a_mount(env):
    client, gate, _, _, _ = env
    lease = open_lease(client)
    before = gate.widget_assets._leases[lease['token']].expires
    assert client.get(lease['base_url'] + 'data.bin').status_code == 200
    assert gate.widget_assets._leases[lease['token']].expires == before
    with TestClient(gate, client=('192.0.2.9', 12345)) as other:
        other.cookies.set(server_auth.AUTH_COOKIE_NAME, server_auth._session_value('fixture-password',
                          issued_at=int(time.time()) - 10))
        assert other.get('/api/widgets').status_code == 200
        assert other.post('/api/widgets/asset-leases/renew', json={'token': lease['token']}).status_code == 409
        assert other.post('/api/widgets/asset-leases/revoke', json={'token': lease['token']}).status_code == 200
    assert client.get(lease['base_url'] + 'data.bin').status_code == 200
    renewed = client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']}).json()
    assert renewed['base_url'] == lease['base_url']
    assert gate.widget_assets._leases[lease['token']].expires > before


def test_unpassworded_access_uses_the_same_lease_protocol(env, monkeypatch):
    client, _, _, _, _ = env
    monkeypatch.setattr(server_auth, 'get_configured_network_password', lambda: '')
    client.cookies.clear()
    lease = open_lease(client)
    assert client.get(lease['base_url'] + 'data.bin').content == BINARY
    assert client.post('/api/widgets/asset-leases/renew', json={'token': lease['token']},
                       headers={'Origin': 'null'}).status_code == 403
    assert client.post('/api/widgets/asset-leases/open', json={'token': lease['token']}).status_code == 404


def test_capture_size_bound_is_enforced_on_actual_bytes(env, monkeypatch):
    _, _, _, skill, _ = env
    monkeypatch.setattr('ouroboros.extension_widget_assets.MAX_WIDGET_BYTES', 16)
    with pytest.raises(ExtensionRegistrationError, match='exceed'):
        capture_widget_sources(skill, RENDER)
