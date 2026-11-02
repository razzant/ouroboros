"""Fixture oracles only: these tests never claim a native Linux/AppImage PASS."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest

from tests import test_onboarding_complete_endpoint as _onboarding_endpoint


ROOT = Path(__file__).resolve().parents[1]
# The REAL onboarding transaction (isolated settings file) from its own suite.
onboarding = _onboarding_endpoint.onboarding


@pytest.fixture
def qualifier():
    return runpy.run_path(str(ROOT / 'scripts/qualify_linux_appimage.py'))


def test_inert_provider_lists_one_model_and_refuses_records_generation(qualifier, tmp_path):
    receiver, received, unexpected = qualifier['fixture_receiver'](tmp_path, 'nonce')
    base = f'http://127.0.0.1:{receiver.server_port}'
    try:
        with urllib.request.urlopen(base + '/v1/models') as response:
            assert json.load(response)['data'][0]['id'] == 'native-qualification-inert'
        request = urllib.request.Request(base + '/v1/chat/completions', data=b'{"model":"inert"}')
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request)
        assert failure.value.code == 503
        assert json.loads(failure.value.read()) == {'error': 'native qualification never generates'}
        assert unexpected == [json.loads((tmp_path / 'unexpected-generation.jsonl').read_text())]
        assert unexpected[0]['refused'] is True and unexpected[0]['body'] == '{"model":"inert"}'
        bad = urllib.request.Request(base + '/bridge', data=b'{"nonce":"wrong"}')
        with pytest.raises(urllib.error.HTTPError) as wrong_nonce:
            urllib.request.urlopen(bad)
        assert wrong_nonce.value.code == 403 and received == []
        with urllib.request.urlopen(urllib.request.Request(base + '/bridge', data=b'{"nonce":"nonce"}')):
            pass
        assert received == [{'nonce': 'nonce'}]
    finally:
        receiver.shutdown()
        receiver.server_close()


def test_bridge_rejects_wrong_source_revision_origin_and_incomplete_readiness(qualifier):
    valid = {'info': {'source_launcher': True, 'source_revision': 'revision'},
             'href': 'http://127.0.0.1:1234/', 'overlay_absent': True, 'main_visible': True}
    check = qualifier['assert_bridge']
    check(valid, True, 'revision', 1234)
    for key, value in [('source_launcher', False), ('source_revision', 'other')]:
        with pytest.raises(AssertionError):
            check({**valid, 'info': {**valid['info'], key: value}}, True, 'revision', 1234)
    for key, value in [('href', 'http://127.0.0.1:9999/'), ('overlay_absent', False), ('main_visible', False)]:
        with pytest.raises(AssertionError):
            check({**valid, key: value}, True, 'revision', 1234)
    with pytest.raises(AssertionError):
        check({k: v for k, v in valid.items() if k != 'overlay_absent'}, True, 'revision', 1234)


def test_cold_endpoint_refresh_changes_only_fixture_url(qualifier, tmp_path):
    settings = tmp_path / 'settings.json'
    original = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:1/v1',
                'OUROBOROS_ONBOARDING_COMPLETED_AT': 'existing', 'OUROBOROS_MODEL': 'inert',
                'OUROBOROS_RUNTIME_MODE': 'advanced', 'other': {'preserved': True}}
    settings.write_text(json.dumps(original))
    qualifier['refresh_fixture_endpoint'](settings, 'http://127.0.0.1:2/v1', tmp_path)
    assert json.loads(settings.read_text()) == {**original, 'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:2/v1'}


def test_witness_waits_for_overlay_absence_visible_dom_and_post_paint_recheck(qualifier, tmp_path):
    node = shutil.which('node')
    assert node, 'Node is required to execute the actual fixture witness'
    page = tmp_path / 'index.html'
    original = '<html>unchanged product UI</html>'
    page.write_text(original)
    assert qualifier['install_witness'](page, 4321, 'nonce', tmp_path) == original
    assert page.read_text().startswith(original)
    suffix = (tmp_path / 'bridge-instrumentation.html').read_text()
    script = suffix.removeprefix('<script>').removesuffix('</script>')
    harness = r'''
const vm = require('node:vm'), assert = require('node:assert/strict');
let overlay = true, painted = true, bridge = 0, sent = [], frame = 0, afterBridge = () => {};
const nodes = {
    '#primary-sidebar': {innerText:'Main Chat', getBoundingClientRect:() => ({width:240,height:800,top:0,left:0,bottom:800,right:240})},
    '#content': {innerText:'Main', getBoundingClientRect:() => ({width:painted?800:0,height:800,top:0,left:240,bottom:800,right:1040})}
};
const context = {
    document: {readyState:'complete', fonts:{ready:Promise.resolve()}, title:'Ouroboros', body:{innerText:'Main Chat Main'},
        querySelector: selector => selector === '#onboarding-overlay' ? (overlay ? {} : null) : nodes[selector]},
    window:{pywebview:{api:{shell_info:async () => {bridge++; afterBridge(); return {source_launcher:true};}}}},
    location:{href:'http://127.0.0.1:1234/'}, innerWidth:1100, innerHeight:800,
    getComputedStyle:() => ({display:'block',visibility:'visible',opacity:'1'}),
    setInterval:callback => {context.tick=callback; return 1;}, clearInterval:() => {},
    requestAnimationFrame:callback => {frame++; callback();},
    fetch:async (url, args) => {sent.push(JSON.parse(args.body));}
};
vm.runInNewContext(SCRIPT, context);
(async () => {
    await context.tick(); assert.equal(bridge,0); assert.equal(sent.length,0); // blocking shell, even empty
    overlay=false; painted=false; await context.tick(); assert.equal(bridge,0); // hidden/zero-size content
    painted=true; afterBridge=() => {overlay=true;};
    await context.tick(); assert.equal(bridge,1); assert.equal(sent.length,0); // readiness changed during bridge
    overlay=false; afterBridge=() => {};
    await context.tick(); assert.equal(sent.length,1); assert.equal(frame,4);
    assert.equal(sent[0].overlay_absent,true); assert.equal(sent[0].main_visible,true);
    assert.equal(sent[0].nonce,'nonce');
})().catch(error => {console.error(error); process.exitCode=1;});
'''.replace('SCRIPT', json.dumps(script))
    result = subprocess.run([node, '-e', harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_configured_launch_witness_is_installed_before_popen_and_does_not_wait_for_setup(qualifier):
    import ast
    tree = ast.parse((ROOT / 'scripts/qualify_linux_appimage.py').read_text())
    one_case = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'one_case')
    calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(one_case) if isinstance(n, ast.Call)]
    popen = next(line for line, name in calls if name == 'subprocess.Popen')
    assert any(line < popen for line, name in calls if name == 'install_witness')
    # Setup observation/completion must stay inside the fresh-only branch.
    fresh = next(n for n in ast.walk(one_case) if isinstance(n, ast.If) and ast.unparse(n.test) == 'not configured')
    fresh_calls = [ast.unparse(n.func) for n in ast.walk(fresh) if isinstance(n, ast.Call)]
    assert 'find_window' in fresh_calls and 'complete_setup' in fresh_calls
    assert 'install_witness' in fresh_calls


@pytest.mark.parametrize('body,status,readiness', [({'ok': False, 'restart_required': False}, 200, 204),
                                                   ({'ok': True, 'restart_required': True}, 200, 204),
                                                   ({'ok': True, 'restart_required': False}, 200, 200),
                                                   ({'error': 'refused'}, 400, 204)])
def test_completion_never_accepts_unsaved_restart_pending_or_still_blocked_setup(qualifier, tmp_path, monkeypatch,
                                                                             body, status, readiness):
    from types import SimpleNamespace

    class Response(SimpleNamespace):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    replies = [Response(status=status, read=lambda: json.dumps(body).encode()),
               Response(status=readiness)]
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: replies.pop(0))
    with pytest.raises(AssertionError):
        qualifier['complete_setup'](1234, 'http://127.0.0.1:4321/v1', tmp_path)
    assert json.loads((tmp_path / 'setup-payload.json').read_text())['starter_skills'] == []
    assert json.loads((tmp_path / 'setup-response.json').read_text())['status'] == status


def test_native_qualification_inert_payload_completes_without_subscription_or_starter_install(onboarding, qualifier):
    """The native fixture uses a real transaction, not a readiness bypass."""
    from ouroboros.server_runtime import has_startup_ready_provider
    from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY

    payload = qualifier['setup_payload']('http://127.0.0.1:4321/v1')
    response = onboarding.client.post('/api/onboarding/complete', json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['ok'] is True
    assert response.json()['restart_required'] is False
    saved = onboarding.saved()
    assert saved[ONBOARDING_COMPLETED_KEY]
    assert has_startup_ready_provider(saved)
    assert saved['OUROBOROS_MODEL'] == payload['OUROBOROS_MODEL']
    assert saved['OPENAI_COMPATIBLE_BASE_URL'] == payload['OPENAI_COMPATIBLE_BASE_URL']
    assert not saved.get('OPENAI_COMPATIBLE_API_KEY')
    assert onboarding.calls['snapshot'] == 0
    assert onboarding.calls['starter'][0][1] == []
