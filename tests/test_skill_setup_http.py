"""Skill setup crosses the real HTTP, worker-thread and lifecycle-lane boundaries."""
from __future__ import annotations

import asyncio
import json
import threading

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from ouroboros import skill_lifecycle_actions as actions, skill_lifecycle_queue as queue
from ouroboros.gateway import extensions
from ouroboros.skill_loader import find_skill, load_enabled, save_review_state, SkillReviewState
from tests.test_skill_enable_dependencies import make_skill, request_turn_on


@pytest.fixture
def setup_http(tmp_path, monkeypatch):
    drive = tmp_path / 'data'
    drive.mkdir()
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setenv('OUROBOROS_REVIEW_ENFORCEMENT', 'advisory')
    monkeypatch.setenv('OUROBOROS_SKILLS_REPO_PATH', str(tmp_path / 'absent'))
    monkeypatch.setattr(actions, '_reconcile', lambda *a, **k: {})
    monkeypatch.setattr(extensions, '_broadcast_extension_lifecycle', lambda *a: None)
    monkeypatch.setattr(queue, 'JOB_ANSWER_WAIT_SEC', 0.03)
    make_skill(drive)
    loaded = find_skill(drive, 'demo', repo_path='')
    save_review_state(drive, 'demo', SkillReviewState(status='clean', content_hash=loaded.content_hash))
    app = Starlette(routes=[
        Route('/api/skills/{skill}/toggle', extensions.api_skill_toggle, methods=['POST']),
        Route('/api/skills/{skill}/grants', extensions.api_skill_grants, methods=['POST']),
    ])
    app.state.drive_root, app.state.repo_dir = drive, tmp_path / 'repo'
    return app, drive


def test_mixed_source_launcher_is_told_to_restart_before_any_effect(setup_http, monkeypatch):
    from types import SimpleNamespace

    _app, drive = setup_http
    before = queue.queue_snapshot()

    def forbidden(*args, **kwargs):
        pytest.fail("The legacy launcher must not execute lifecycle work or its callback")

    monkeypatch.setattr(actions, 'run_lifecycle_job_blocking', forbidden)
    monkeypatch.setattr(actions, 'save_skill_grants', forbidden)
    monkeypatch.setattr(actions, '_reconcile', forbidden)
    monkeypatch.setattr('ouroboros.skill_dependencies.ensure_skill_dependencies', forbidden)
    result = actions.run_skill_action(SimpleNamespace(drive_root=drive), 'demo', 'grant',
        items=['inject_chat'], expected_content_hash='old-preview',
        _owner_actor='owner_launcher', _reconcile_grant=forbidden)
    assert result['ok'] is False and result['status_code'] == 409
    assert result['code'] == 'launcher_restart_required'
    assert 'Restart' in result['error'] and queue.queue_snapshot() == before
    assert not (drive / 'state/skills/demo/grants.json').exists()


def test_legacy_reconcile_does_not_reenter_the_callers_lifecycle_lane(setup_http, monkeypatch):
    app, drive = setup_http
    app.router.routes.append(Route('/api/skills/{skill}/reconcile', extensions.api_skill_reconcile, methods=['POST']))
    monkeypatch.setattr('ouroboros.extension_loader.reconcile_extension', lambda *a, **k: {'action': 'unchanged'})
    entered, release = threading.Event(), threading.Event()

    def held():
        entered.set()
        assert release.wait(5)
        return {}

    thread = threading.Thread(target=lambda: queue.run_lifecycle_job_blocking(
        kind='grant', target='demo', source='owner_launcher', runner=held,
        options=queue.LifecycleJobOptions(drive_root=drive)))
    thread.start()

    async def journey():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            assert entered.wait(3)
            response = await asyncio.wait_for(client.post('/api/skills/demo/reconcile'), 1)
            assert response.status_code == 200 and not release.is_set()

    try:
        asyncio.run(journey())
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


@pytest.mark.parametrize('install_failure', [False, True])
def test_held_lane_returns_a_real_job_then_terminal_result(setup_http, monkeypatch, install_failure):
    app, drive = setup_http
    calls = []

    def install(packages, *a, **k):
        calls.append(packages)
        if install_failure:
            raise RuntimeError('controlled package failure')
        return []

    monkeypatch.setattr('ouroboros.marketplace.isolated_deps._install_python_packages', install)
    lock = queue._get_lock()
    lock.acquire()

    async def journey():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            try:
                reply = await asyncio.wait_for(client.post('/api/skills/demo/toggle', json={'enabled': True},
                    headers={'Prefer': 'respond-async'}), 3)
                body = reply.json()
                assert reply.status_code == 200 and body['pending'] is True
                assert body['job_id'].startswith('skill-job-') and 'enabled' not in body
                jobs = queue.queue_snapshot()['events']
                assert any(j['id'] == body['job_id'] and j['status'] == 'queued' for j in jobs)
                assert calls == [] and not load_enabled(drive, 'demo')
            finally:
                lock.release()
            work = list(queue._BACKGROUND)
            await asyncio.wait_for(asyncio.gather(*work), 5)
            terminal = [j for j in queue.queue_snapshot()['events'] if j['id'] == body['job_id']][-1]
            assert terminal['status'] == ('failed' if install_failure else 'succeeded')
            assert ('packages did not install' in terminal['error']) is install_failure
            assert load_enabled(drive, 'demo') is not install_failure
            assert calls == [['wheel']]
            # The same endpoint still gives its ordinary final shape for a fast disable.
            monkeypatch.setattr(queue, 'JOB_ANSWER_WAIT_SEC', 5)
            result = (await client.post('/api/skills/demo/toggle', json={'enabled': False})).json()
            assert result['enabled'] is False and not result.get('pending')

    asyncio.run(journey())


@pytest.mark.parametrize('prefer', ['', 'return=representation'])
@pytest.mark.parametrize('install_failure', [False, True])
def test_legacy_toggle_waits_for_the_terminal_result(setup_http, monkeypatch, prefer, install_failure):
    """Old UI and dictation do not mistake accepted work for enabled state."""
    app, drive = setup_http
    bound = threading.Event()
    original_bind = queue.JobHandle.bind
    calls = []

    def bind(handle, job):
        original_bind(handle, job)
        bound.set()

    def install(packages, *args, **kwargs):
        calls.append(packages)
        if install_failure:
            raise RuntimeError('controlled offline')
        return []

    monkeypatch.setattr(queue.JobHandle, 'bind', bind)
    monkeypatch.setattr('ouroboros.marketplace.isolated_deps._install_python_packages', install)
    lock = queue._get_lock()
    lock.acquire()

    async def journey():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            request = asyncio.create_task(client.post('/api/skills/demo/toggle',
                json={'enabled': True}, headers={'Prefer': prefer} if prefer else {}))
            try:
                assert await asyncio.to_thread(bound.wait, 3)
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(request), 0.12)
                assert not load_enabled(drive, 'demo') and calls == []
            finally:
                lock.release()
                await asyncio.gather(request, *list(queue._BACKGROUND), return_exceptions=True)
            reply = request.result()
            body = reply.json()
            assert not body.get('pending')
            assert calls == [['wheel']]
            assert load_enabled(drive, 'demo') is not install_failure
            if install_failure:
                assert reply.status_code >= 400 and body['error']
            else:
                assert reply.status_code == 200 and body['enabled'] is True

    asyncio.run(journey())


def test_delayed_validation_never_advertises_an_empty_job(monkeypatch):
    monkeypatch.setattr(queue, 'JOB_ANSWER_WAIT_SEC', 0.01)
    entered, release = threading.Event(), threading.Event()

    def validation(handle):
        entered.set()
        assert release.wait(3)
        return {'ok': False, 'error': 'selected revision changed'}

    async def journey():
        task = asyncio.create_task(queue.answer_or_job(validation, allow_pending=True))
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.sleep(0.04)
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, 2) == {'ok': False, 'error': 'selected revision changed'}

    try:
        asyncio.run(journey())
    finally:
        release.set()


def test_grant_keeps_its_fact_and_exposes_one_failed_setup(setup_http, monkeypatch):
    _app, drive = setup_http
    payload = drive / 'skills' / 'external' / 'demo' / 'SKILL.md'
    payload.write_text(payload.read_text(encoding='utf-8').replace('timeout_sec: 30', 'timeout_sec: 30\npermissions: [inject_chat]'), encoding='utf-8')
    skill = find_skill(drive, 'demo', repo_path='')
    save_review_state(drive, 'demo', SkillReviewState(status='clean', content_hash=skill.content_hash))
    request_turn_on(drive)
    attempts = []

    def fail(*a, **k):
        attempts.append(1)
        raise RuntimeError('controlled offline')

    monkeypatch.setattr('ouroboros.marketplace.isolated_deps._install_python_packages', fail)
    skill = find_skill(drive, 'demo', repo_path='')
    result = actions._grant(skill, drive, '', ['inject_chat'])
    assert result['ok'] is True and result['granted_permissions'] == ['inject_chat']
    assert result['setup_error'] == 'Its packages did not install.'
    assert result['turn_on_when_ready'] == 'not_ready' and attempts == [1]
    assert not load_enabled(drive, 'demo')


@pytest.mark.parametrize('enable_after_grant,install_failure', [(False, False), (True, False), (True, True)])
def test_queued_grant_preserves_only_the_explicit_enable_intent(setup_http, monkeypatch, enable_after_grant, install_failure):
    """No install-time turn-on request: the HTTP request must carry the owner's Enable."""
    app, drive = setup_http
    payload = drive / 'skills' / 'external' / 'demo' / 'SKILL.md'
    payload.write_text(payload.read_text(encoding='utf-8').replace('timeout_sec: 30',
        'timeout_sec: 30\npermissions: [inject_chat]'), encoding='utf-8')
    current = find_skill(drive, 'demo', repo_path='')
    save_review_state(drive, 'demo', SkillReviewState(status='clean', content_hash=current.content_hash))
    assert not (drive / 'state' / 'skills' / 'demo' / 'install.json').exists()
    calls = []

    def install(packages, *a, **k):
        calls.append(packages)
        if install_failure:
            raise RuntimeError('controlled package failure')
        return []

    monkeypatch.setattr('ouroboros.marketplace.isolated_deps._install_python_packages', install)
    lock = queue._get_lock()
    lock.acquire()

    async def journey():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            try:
                reply = await asyncio.wait_for(client.post('/api/skills/demo/grants', json={
                    'items': ['inject_chat'], 'enable_after_grant': enable_after_grant},
                    headers={'Prefer': 'respond-async'}), 3)
                body = reply.json()
                assert reply.status_code == 200 and body['pending'] is True and body['job_id']
                assert not load_enabled(drive, 'demo') and calls == []
            finally:
                lock.release()
            await asyncio.wait_for(asyncio.gather(*list(queue._BACKGROUND)), 5)
            terminal = [j for j in queue.queue_snapshot()['events'] if j['id'] == body['job_id']][-1]
            assert terminal['status'] == 'succeeded', terminal  # the grant itself was saved
            assert ('Access granted; Its packages did not install.' in terminal['message']) is install_failure
            assert load_enabled(drive, 'demo') == (enable_after_grant and not install_failure)
            assert calls == ([['wheel']] if enable_after_grant else [])
            from ouroboros.skill_loader import grant_status_for_skill
            assert grant_status_for_skill(drive, find_skill(drive, 'demo', repo_path=''))['all_granted']

    asyncio.run(journey())


def test_grant_rejects_non_boolean_enable_intent(setup_http):
    app, drive = setup_http

    async def journey():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            reply = await client.post('/api/skills/demo/grants', json={'grant_all': True, 'enable_after_grant': 'false'})
            assert reply.status_code == 400 and 'boolean' in reply.json()['error']
    asyncio.run(journey())
    assert not load_enabled(drive, 'demo')


def test_packageless_requested_skill_and_load_error_revert_are_retried(setup_http):
    from ouroboros.skill_builtin import _dependency_pass
    from ouroboros.skill_loader import save_enabled
    import logging

    _app, drive = setup_http
    make_skill(drive, 'plain', specs=False)
    request_turn_on(drive, 'plain')
    skill = find_skill(drive, 'plain', repo_path='')
    save_review_state(drive, 'plain', SkillReviewState(status='clean', content_hash=skill.content_hash))
    save_enabled(drive, 'plain', False, actor='load_error_revert')
    assert _dependency_pass(drive, ['plain'], logging.getLogger(__name__)) == ['plain']
    assert load_enabled(drive, 'plain') is True
    save_enabled(drive, 'plain', False, actor='owner_ui')
    assert _dependency_pass(drive, ['plain'], logging.getLogger(__name__)) == []
    assert not load_enabled(drive, 'plain')


@pytest.mark.parametrize('invalid_after_wait', [False, True])
def test_author_finish_reads_the_payload_after_lane_admission(tmp_path, monkeypatch, invalid_after_wait):
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros import skill_source_install
    from ouroboros.skill_loader import load_review_state, compute_content_hash
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec
    from tests.test_skill_exec import _make_ctx, _build_skill

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    monkeypatch.setenv('OUROBOROS_REVIEW_ENFORCEMENT', 'advisory')
    ctx = _make_ctx(tmp_path)
    directory = _build_skill(ctx.drive_root / 'skills' / 'external', 'demo')
    old_hash = compute_content_hash(directory)
    save_review_state(ctx.drive_root, 'demo', SkillReviewState(status='clean', content_hash=old_hash,
        raw_actor_records=[{'slot_id': 'original-critic', 'status': 'ok'}]))
    binding = ResolvedResourceBinding(profile='self_modification', root='skill_payload', operation='review',
        base_path=directory, target_path=directory, source='test', skill_name='demo', state_drive_root=ctx.drive_root)
    monkeypatch.setattr(skill_exec, '_skill_tool_preflight', lambda *a, **k: '')
    entered = threading.Event()
    real_lane = skill_source_install.run_in_lane

    def enter_lane(*a, **k):
        entered.set()
        return real_lane(*a, **k)

    monkeypatch.setattr(skill_source_install, 'run_in_lane', enter_lane)
    prepared = []
    monkeypatch.setattr('ouroboros.skill_dependencies.ensure_skill_dependencies',
        lambda drive, skill, **k: prepared.append(skill.content_hash) or ('not_required', ''))
    lock = queue._get_lock()
    lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(skill_exec._handle_review_skill, ctx, skill='demo', _resolved_binding=binding,
            author_disposition='accepted', author_rationale='Accept the current candidate after verification.')
        try:
            assert entered.wait(3), 'author finish did not enter the lifecycle lane'
            assert not future.done() and not prepared
            assert not load_review_state(ctx.drive_root, 'demo').author_disposition
            (directory / 'scripts' / 'hello.py').write_text('def invalid(:\n' if invalid_after_wait else "print('new payload')\n", encoding='utf-8')
            new_hash = compute_content_hash(directory)
            assert new_hash != old_hash
        finally:
            lock.release()
        result = future.result(timeout=20)
    current = load_review_state(ctx.drive_root, 'demo')
    if invalid_after_wait:
        assert 'deterministic preflight did not pass' in result and not prepared
        assert not current.author_disposition
    else:
        assert 'Author finish recorded' in result and prepared == [new_hash]
        assert current.author_disposition['subject_hash'] == new_hash


@pytest.mark.parametrize('stage', ['grants', 'toggle', 'grant_setup_failure'])
def test_colab_does_not_claim_pending_or_failed_setup_enabled(stage):
    from ouroboros.colab_bootstrap import ensure_native_telegram_live
    from tests.test_colab_bootstrap import _native_telegram_index
    calls = []

    def request(method, path, body=None, timeout=None):
        calls.append(path)
        if path == '/api/health':
            return 200, {'ok': True}
        if path == '/api/extensions':
            return 200, _native_telegram_index(missing=['TELEGRAM_BOT_TOKEN'] if stage != 'toggle' else [])
        if path.endswith('/grants') and stage == 'grant_setup_failure':
            return 200, {'ok': True, 'granted_keys': ['TELEGRAM_BOT_TOKEN'], 'setup_error': 'Its packages did not install.'}
        if path.endswith('/' + stage):
            return 200, {'ok': True, 'pending': True, 'job_id': 'actual-job', 'message': 'Installing packages…'}
        raise AssertionError('Unexpected continuation after incomplete setup: ' + path)

    status = ensure_native_telegram_live(settings={'TELEGRAM_BOT_TOKEN': 'synthetic-not-real',
        'OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS': True}, request=request, timeout=5)
    assert not status['ok'] and 'enabled' not in status['steps']
    assert not any(path.endswith('/settings/save') for path in calls)
    if stage == 'grant_setup_failure':
        assert status['error'] == 'Its packages did not install.'
        assert not any(path.endswith('/toggle') for path in calls)
    else:
        assert status['pending'] and status['job_id'] == 'actual-job'


def _launcher_grant_skill(tmp_path, monkeypatch, *, name="demo", skill_type="script", permissions=(), keys=("OPENROUTER_API_KEY",)):
    import launcher
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_review_state

    drive = tmp_path / "data"
    payload = drive / "skills" / "external" / name
    payload.mkdir(parents=True)
    declaration = {
        "script": "runtime: python3\nscripts:\n  - name: run.py\n",
        "extension": "entry: plugin.py\nplugin_api: '2.0'\n",
        "instruction": "",
    }[skill_type]
    (payload / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Launcher grant fixture.\nversion: 1.0.0\ntype: {skill_type}\n"
        f"permissions: {json.dumps(list(permissions))}\nenv_from_settings: {json.dumps(list(keys))}\n"
        "subscribe_events: [chat.outbound]\n" + declaration + "---\n"
    )
    if skill_type == "script":
        (payload / "scripts").mkdir()
        (payload / "scripts" / "run.py").write_text("print('ok')\n")
    if skill_type == "extension":
        (payload / "plugin.py").write_text("def register(api):\n    return None\n")
    loaded = load_skill(payload, drive)
    save_review_state(drive, name, SkillReviewState(status="clean", content_hash=loaded.content_hash))
    monkeypatch.setattr(launcher, "DATA_DIR", drive)
    monkeypatch.setattr(launcher, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_SKILLS_REPO_PATH": ""})
    monkeypatch.setattr(launcher, "_read_port_file", lambda: 8765)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    # Native confirmation reaches the real HTTP action owner, not a second local grant writer.
    import asyncio
    import io
    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route
    from ouroboros.gateway import extensions
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions._reconcile", lambda *a, **k: {})
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *a: None)
    app = Starlette(routes=[Route("/api/skills/{skill}/grants", extensions.api_skill_grants, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = drive, tmp_path / "repo"
    def request(req, timeout=0):
        assert req.full_url == f"http://127.0.0.1:8765/api/skills/{name}/grants"
        assert req.get_header('Prefer') in (None, 'respond-async')
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                return await client.post(f"/api/skills/{name}/grants", json=json.loads(req.data), headers=dict(req.header_items()))
        return io.BytesIO(asyncio.run(send()).content)
    monkeypatch.setattr(launcher.urllib.request, "urlopen", request)
    return loaded, drive


def test_launcher_skill_key_grant_validates_review_and_manifest(monkeypatch, tmp_path):
    import launcher
    from ouroboros.skill_loader import load_skill_grants

    loaded, drive = _launcher_grant_skill(tmp_path, monkeypatch)
    result = launcher._request_skill_key_grant("demo", ["OPENROUTER_API_KEY"], lambda *a: True)
    assert result["ok"] is True
    grants = load_skill_grants(drive, "demo")
    assert grants["granted_keys"] == ["OPENROUTER_API_KEY"]
    assert grants["requested_keys"] == ["OPENROUTER_API_KEY"]
    assert grants["content_hash"] == loaded.content_hash
    assert result.get("extension_action") is None
    assert result.get("extension_reason") is None


@pytest.mark.parametrize('pending_aware', [False, True])
def test_launcher_negotiates_only_for_a_pending_aware_client(monkeypatch, tmp_path, pending_aware):
    import launcher

    _loaded, drive = _launcher_grant_skill(tmp_path, monkeypatch)
    original = launcher.urllib.request.urlopen
    observed = []

    def request(req, timeout=0):
        observed.append((req.get_header('Prefer'), timeout))
        assert json.loads(req.data)['enable_after_grant'] is True
        return original(req, timeout=timeout)

    monkeypatch.setattr(launcher.urllib.request, 'urlopen', request)
    result = launcher._request_skill_key_grant('demo', ['OPENROUTER_API_KEY'], lambda *args: True,
        enable_after_grant=True, allow_pending=pending_aware)
    assert result['enabled'] is True and load_enabled(drive, 'demo')
    assert observed[0][0] == ('respond-async' if pending_aware else None)
    assert (observed[0][1] == 20) is pending_aware


def test_launcher_skill_grant_supports_permission_grants(monkeypatch, tmp_path):
    import launcher
    from ouroboros.skill_loader import load_skill_grants

    _loaded, drive = _launcher_grant_skill(tmp_path, monkeypatch, name="bridge", skill_type="extension",
                                         keys=(), permissions=("inject_chat", "subscribe_event"))
    result = launcher._request_skill_key_grant("bridge", ["inject_chat", "subscribe_event:chat.outbound"], lambda *a: True)
    assert result["ok"] is True
    assert result["granted_permissions"] == ["inject_chat", "subscribe_event:chat.outbound"]
    grants = load_skill_grants(drive, "bridge")
    assert grants["granted_keys"] == []
    assert grants["granted_permissions"] == result["granted_permissions"]


def test_launcher_skill_key_grant_supports_extensions(monkeypatch, tmp_path):
    """The server persists the grant before reconciling; the launcher returns its outcome."""
    import launcher
    from ouroboros.skill_loader import load_skill_grants

    _loaded, drive = _launcher_grant_skill(tmp_path, monkeypatch, name="demo_ext", skill_type="extension")
    calls = []

    def reconcile(skill, root, repo):
        calls.append(skill.name)
        assert load_skill_grants(drive, "demo_ext")["granted_keys"] == ["OPENROUTER_API_KEY"]
        return {"action": "extension_loaded", "reason": "ready", "live_loaded": True}

    monkeypatch.setattr("ouroboros.skill_lifecycle_actions._reconcile", reconcile)
    result = launcher._request_skill_key_grant("demo_ext", ["OPENROUTER_API_KEY"], lambda *a: True)
    assert result["ok"] is True
    assert calls == ["demo_ext"]
    assert result["extension_action"] == "extension_loaded"


def test_launcher_skill_key_grant_handles_http_error_without_local_writes(monkeypatch, tmp_path):
    """A failed request proves no grant; only the server writes that fact."""
    import launcher
    from ouroboros.skill_loader import load_skill_grants

    _loaded, drive = _launcher_grant_skill(tmp_path, monkeypatch, name="demo_ext", skill_type="extension")

    def unavailable(*a, **k):
        raise ConnectionError("server not reachable")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    result = launcher._request_skill_key_grant("demo_ext", ["OPENROUTER_API_KEY"], lambda *a: True)
    assert result["ok"] is False and "server not reachable" in result["error"]
    assert load_skill_grants(drive, "demo_ext")["granted_keys"] == []


def test_launcher_skill_key_grant_rejects_instruction_skill(monkeypatch, tmp_path):
    import launcher
    from ouroboros.skill_loader import load_skill_grants

    _loaded, drive = _launcher_grant_skill(tmp_path, monkeypatch, name="instr", skill_type="instruction")
    result = launcher._request_skill_key_grant("instr", ["OPENROUTER_API_KEY"], lambda *a: True)
    assert result["ok"] is False
    assert "script and extension" in result["error"]
    assert load_skill_grants(drive, "instr")["granted_keys"] == []
