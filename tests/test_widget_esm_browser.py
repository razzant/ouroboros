"""Native modules/resources on the real builtin board, behind the real network password.

The core always sees a non-loopback browser and gates it. By default that is the
documented reverse-proxy topology: the real server.py on loopback behind this
test's own forwarding proxy (``_SameOriginProxy``, tests/test_ui_session_recovery_browser.py),
which overwrites X-Forwarded-For with a TEST-NET-2 visitor as a deployed proxy
does; no LAN listener opens. WIDGET_TEST_HOST, an authorized local non-loopback
address, instead binds the core there, so the browser's own origin is that LAN
address. No wildcard, loopback exemption or public listener is used. Run under
safe_test with Chromium and WebKit installed.
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import time
from pathlib import Path

import httpx
import pytest

from ouroboros.test_environment import isolated_environment
from ouroboros.test_evidence import diagnostic_directory
from tests.candidate_checkout import candidate_checkout, require_candidate_interpreter, verify_checkout
from tests.separate_ui_support import ModelStub, OwnedProcess
from tests.widget_esm_fixture import BINARY, NAME, install

pytestmark = pytest.mark.ui_browser
PASSWORD = 'widget-fixture-password'


@pytest.fixture
def network_server(tmp_path, request):
    from tests.test_ui_session_recovery_browser import LAN_CLIENT, _SameOriginProxy

    host = os.environ.get('WIDGET_TEST_HOST', '')
    if host:
        address = ipaddress.ip_address(host)
        assert not address.is_loopback and not address.is_unspecified and address.is_private
    evidence = diagnostic_directory(tmp_path / 'evidence', 'widget-esm-' + request.node.callspec.params['engine'])
    source = Path(__file__).resolve().parents[1]
    with candidate_checkout(source, tmp_path / 'repo', origin_proof=True) as candidate:
        env = isolated_environment(tmp_path, tmp_path / 'repo')
        data = Path(env['OUROBOROS_DATA_DIR'])
        skill = install(data)
        model = ModelStub()
        settings = {'OUROBOROS_MODEL': 'openai-compatible::mock-model',
            'OUROBOROS_MODEL_LIGHT': 'openai-compatible::mock-model',
            'OPENAI_COMPATIBLE_API_KEY': 'fixture-only', 'OPENAI_COMPATIBLE_BASE_URL': model.base_url,
            'OUROBOROS_MAX_WORKERS': 1, 'OUROBOROS_RUNTIME_MODE': 'advanced',
            'OUROBOROS_STARTER_SKILLS': '[]', 'OUROBOROS_NETWORK_PASSWORD': PASSWORD}
        (data / 'settings.json').write_text(json.dumps(settings))
        with socket.socket() as sock:
            sock.bind((host or '127.0.0.1', 0))
            port = sock.getsockname()[1]
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            service_port = sock.getsockname()[1]
        env.pop('OUROBOROS_NETWORK_PASSWORD', None)
        env.update(OUROBOROS_SERVER_HOST=host or '127.0.0.1', OUROBOROS_SERVER_PORT=str(port),
                   OUROBOROS_HOST_SERVICE_PORT=str(service_port), OUROBOROS_DISABLE_MANAGED_UPDATES='1')
        if not host:
            env['FORWARDED_ALLOW_IPS'] = '127.0.0.1'  # only this test's own proxy names the visitor
        require_candidate_interpreter(sys.executable, env, tmp_path / 'repo')
        verify_checkout(tmp_path / 'repo', candidate)
        server = OwnedProcess('widget-server', [sys.executable, 'server.py'], cwd=tmp_path / 'repo',
                              env=env, log_path=evidence / 'server.log')
        proxy = None if host else _SameOriginProxy(port, port)  # one upstream: every path reaches the core
        url = f'http://{host}:{port}' if host else proxy.url
        candidate.hold()
        try:
            server.start()
            with httpx.Client(base_url=url, trust_env=False, timeout=5) as client:
                deadline = time.monotonic() + 60
                while True:
                    try:
                        health = client.get('/api/health')
                        if health.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        pytest.fail('isolated server did not answer: ' + server.log_text()[-4000:])
                    time.sleep(.2)
                assert client.get('/api/widgets').status_code == 401, 'must exercise the real network gate'
                assert client.get('/static/modules/widget_module.js').status_code == 401
                assert client.post('/auth/login', json={'password': PASSWORD}).status_code == 200
                assert client.get('/static/candidate-sentinel.txt').content == candidate.sentinel_bytes
                assert health.json()['runtime_version'] == candidate.version_text
                enabled = client.post(f'/api/skills/{NAME}/toggle', json={'enabled': True}, timeout=45)
                assert enabled.status_code == 200, enabled.text
                assert len(client.get('/api/widgets').json()['ui_tabs']) == 3, enabled.text
                yield url, client, skill, evidence
        finally:
            server.stop()
            if proxy is not None:
                proxy.close()
            candidate.release()
            model.close()
            (evidence / 'process.json').write_text(json.dumps({'host': host, 'port': port, 'url': url,
                'topology': 'lan' if host else f'proxy, visitor {LAN_CLIENT}',
                'pids': server.pids, 'exits': server.exits, 'candidate': candidate.identity}, default=str, indent=2))


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_password_network_esm_files(network_server, engine):
    from playwright.sync_api import sync_playwright

    url, client, skill, evidence = network_server
    observations = {}
    errors = []
    asset_responses = []

    def resource_records():
        records = []
        for response in list(asset_responses):
            request = response.request
            headers = request.all_headers()
            records.append({'url': response.url, 'status': response.status, 'kind': request.resource_type,
                            'has_cookie': bool(headers.get('cookie', '')), 'origin': headers.get('origin', '')})
        return records

    card = f'[data-widget-key="{NAME}:main"]'
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        observations.update(browser_version=browser.version, headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 900})
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        # Collect without nested protocol calls from inside a network callback.
        # Body completion is proven by the widget's values, not event ordering.
        page.on('response', lambda r: asset_responses.append(r) if '/api/widget-assets/' in r.url else None)
        try:
            page.goto(url)
            page.locator('#password').fill(PASSWORD)
            page.get_by_role('button', name='Unlock').click()
            page.locator('[data-nav-page="widgets"]').click()
            frame = page.frame_locator(card + ' iframe')
            frame.locator('#root[data-ready="1"]').wait_for(timeout=30000)
            actual = page.locator(card + ' iframe').element_handle().content_frame()
            proof = actual.evaluate('window.proof')
            assert {k: proof[k] for k in ('value', 'cycle', 'size', 'last', 'json', 'api', 'origin', 'top', 'cookie')} == {
                'value': 'Module ready', 'cycle': 'Module ready', 'size': 256, 'last': 255,
                'json': 'declared', 'api': True, 'origin': 'null', 'top': False, 'cookie': False}
            actual.wait_for_function("document.querySelector('#pixel').naturalWidth === 1 && getComputedStyle(document.body).color === 'rgb(242, 243, 245)'")
            assert actual.evaluate("getComputedStyle(document.querySelector('#summary')).backgroundImage") == (
                f'url("{proof["base"]}images/pixel.png")')
            assert 'allow-same-origin' not in page.locator(card + ' iframe').get_attribute('sandbox')
            other = page.locator(f'[data-widget-key="{NAME}:other"] iframe').element_handle().content_frame()
            other.wait_for_function('window.proof')
            other_proof = other.evaluate('window.proof')
            assert other_proof['origin'] == 'null'
            assert proof['base'] not in other_proof['url']
            classic = page.locator(f'[data-widget-key="{NAME}:classic"] iframe').element_handle().content_frame()
            classic.wait_for_function('window.classicThis === true')
            assert classic.evaluate('window.classicGlobal') == 'kept'
            # Ordinary links are copyable without the session, limited to this tab's files.
            with httpx.Client(trust_env=False) as stranger:
                assert stranger.get(proof['base'] + 'data.bin').content == BINARY
                assert stranger.get(proof['base'] + 'other.js').status_code == 404
                assert stranger.get(url + '/api/settings').status_code == 401
            for target in ('/api/widgets/asset-leases', '/api/settings', other_proof['url']):
                assert actual.evaluate("async url => {try {await fetch(url); return false;} catch {return true;}}", target)
            page.screenshot(path=str(evidence / 'modules.png'), full_page=True)
            observations['mount'] = proof
            # Observe an actual parent renewal before reloading the publication.
            # No browser clock or server TTL override: delayed imports keep the same base URL.
            with page.expect_response(lambda r: r.url.endswith('/api/widgets/asset-leases/renew')
                                      and r.request.post_data_json.get('token') == proof['base'].split('/')[-2],
                                      timeout=50000) as renewal:
                pass
            assert renewal.value.status == 200
            observations['renewal'] = renewal.value.json()['expires_in']
            # Immutable loaded bytes survive disk drift; delayed native imports
            # retain their relative base after a real lease renewal.
            (skill / 'app/late.mjs').write_text("export const late = 'UNREVIEWED';")
            frame.locator('#late').click()
            frame.locator('#root[data-late="loaded"]').wait_for()
            (skill / 'app/late.mjs').write_text("export const late = 'loaded';")
            # WebKit may attach a cookie to passive loads; opacity protects
            # its visibility, and the asset gate never uses it as authority.
            assert page.request.get(url + '/api/widget-assets/' + 'x' * 43 + '/data.bin').status == 404
            asset_requests = resource_records()
            for path in ('app/main.js', 'app/lib.js', 'app/cycle.js', 'app/late.mjs',
                         'style.css', 'nested/palette.css', 'images/pixel.png', 'data.bin', 'data.json'):
                assert any(r['url'] == proof['base'] + path and r['status'] == 200 for r in asset_requests), path
            assert all(r['origin'] == 'null' for r in asset_requests if r['kind'] == 'script')
            observations['asset_requests'] = asset_requests[:]
            # Same-content reload changes publication identity and the board remounts.
            old_generation = next(t for t in client.get('/api/widgets').json()['ui_tabs'] if t['tab_id'] == 'main')['generation']
            # Reopening unchanged bytes uses the ordinary skill lifecycle endpoints.
            assert client.post(f'/api/skills/{NAME}/toggle', json={'enabled': False}).status_code == 200
            page.locator(card).wait_for(state='detached')
            assert client.post(f'/api/skills/{NAME}/toggle', json={'enabled': True}).status_code == 200
            frame.locator('#root[data-ready="1"]').wait_for(timeout=30000)
            new_generation = next(t for t in client.get('/api/widgets').json()['ui_tabs'] if t['tab_id'] == 'main')['generation']
            assert new_generation != old_generation
            observations['generations'] = [old_generation, new_generation]
            assert client.get(proof['base'] + 'data.bin').status_code == 404
            actual = page.locator(card + ' iframe').element_handle().content_frame()
            fresh = actual.evaluate('window.proof')
            # Stop completes the existing dispose hook and revokes this mount only.
            page.locator(card + ' [data-widget-power]').click()
            page.locator(card + ' iframe').wait_for(state='detached')
            page.wait_for_function("async base => (await fetch(base + 'data.bin')).status === 404", arg=fresh['base'])
            assert client.get(f'/api/extensions/{NAME}/count').json()['hits'] >= 1
            page.locator(card + ' [data-widget-power]').click()
            frame.locator('#root[data-ready="1"]').wait_for()
            mounted = page.locator(card + ' iframe').element_handle().content_frame().evaluate('window.proof')
            assert client.post(f'/api/skills/{NAME}/toggle', json={'enabled': False}).status_code in (200, 202)
            page.locator(card).wait_for(state='detached', timeout=30000)
            assert client.get(mounted['base'] + 'data.bin').status_code == 404
            assert client.post(f'/api/skills/{NAME}/toggle', json={'enabled': True}).status_code in (200, 202)
            frame.locator('#root[data-ready="1"]').wait_for(timeout=30000)
            mounted = page.locator(card + ' iframe').element_handle().content_frame().evaluate('window.proof')
            # Wait for the existing start ledger, so signing back in exercises
            # automatic mounts rather than a correctly held unfinished start.
            page.wait_for_function("""() => Object.values(JSON.parse(
                localStorage.getItem('ouroboros.holds') || '{}').entries || {})
                .every(entry => entry.reason !== 'start')""")
            page.goto(url + '/auth/logout')
            assert client.get(mounted['base'] + 'data.bin').status_code == 404
            page.locator('#password').fill(PASSWORD)
            page.get_by_role('button', name='Unlock').click()
            page.locator('[data-nav-page="widgets"]').click()
            frame.locator('#root[data-ready="1"]').wait_for(timeout=30000)
            mounted = page.locator(card + ' iframe').element_handle().content_frame().evaluate('window.proof')
            # Settings validates its saved host even when the active socket came
            # from the manual LAN override. This does not rebind the running server.
            response = client.post('/api/settings', json={'OUROBOROS_NETWORK_PASSWORD': 'changed-fixture-password',
                                                         'OUROBOROS_SERVER_HOST': '127.0.0.1'})
            assert response.status_code == 200, response.text
            assert client.get(mounted['base'] + 'data.bin').status_code == 404
            assert client.get('/api/widgets').status_code == 401
            observations.update(reload=True, dispose=True, disable=True, logout=True, password_change=True,
                                errors=errors, engines=engine)
            assert errors == []
        finally:
            observations['asset_requests'] = resource_records()
            if not observations.get('password_change'):
                page.screenshot(path=str(evidence / 'last-state.png'), full_page=True)
            (evidence / 'consumer.json').write_text(json.dumps(observations, indent=2))
            context.close()
            browser.close()
