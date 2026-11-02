"""Production controls and real admission/census APIs on a disposable local stand.

The fixture serves candidate modules, injects only a queue persistence failure,
then reloads the browser. It does not emulate Continue acknowledgements.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread

import pytest

from tests._shared import producer_evidence_dir
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import _interrupted
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / 'web'
HTML = '''<!doctype html><html class="ouro-ui"><head><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css"></head><body>
<main><div id="card" class="chat-live-card"><h3>Interrupted report</h3></div>
<button id="header-restart" class="btn">Restart</button><button id="settings-restart" class="btn">Restart now</button>
<div id="activity"></div></main><script type="module">
import {syncContinueAction} from '/modules/task_continue.js';
import {confirmAndSendRestart} from '/modules/chat_activity.js';
import {openConfirmDialog} from '/modules/confirm_dialog.js';
import {initActivity} from '/modules/activity.js';
const ws={on(){return ()=>{};},send(){throw Error('test must cancel Restart');}};
const detail=await (await fetch('/api/tasks/pred-1')).json();
syncContinueAction({root:document.getElementById('card'),groupId:'pred-1'},detail);
window.activity=initActivity({mount:document.getElementById('activity'),ws});
await window.activity.refresh();
for(const id of ['header-restart','settings-restart']) document.getElementById(id).onclick=()=>confirmAndSendRestart({openConfirmDialog,ws});
window.ready=true;
</script></body></html>'''


def test_failed_admission_reload_retry_and_tree_pause_consumers(tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright, expect
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.task_continue import api_task_continue
    from tests._work_projection_shared import state_fields
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.owner_pause import install_fence, set_fence_state
    from ouroboros.task_results import load_task_result, write_task_result

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    write_task_result(tmp_path, 'pausing-root', 'running', root_task_id='pausing-root', chat_id=1)
    fence, _ = install_fence(tmp_path, 'pausing-root', request_id='tree-pause')
    workers.PENDING.append({'id': 'pausing-root', 'root_task_id': 'pausing-root', 'type': 'task', 'chat_id': 1,
                            'title': 'Saved root waiting for child', '_budget_pause': {'reason': 'owner'}})
    workers.RUNNING['child'] = {'task': {'id': 'child', 'root_task_id': 'pausing-root', 'parent_task_id': 'pausing-root', 'chat_id': 1}}
    write_task_result(tmp_path, 'child', 'running', root_task_id='pausing-root', parent_task_id='pausing-root', chat_id=1)
    async def detail(_request):
        row = load_task_result(tmp_path, 'pred-1')
        return JSONResponse({**row, 'continuation_offer': continuation_offer(row, 'pred-1')})
    async def state(_request):
        return JSONResponse({**state_fields(tmp_path),
                             'active_chat_activities_complete': True, 'bg_consciousness_enabled': False})
    async def tasks(_request):
        return JSONResponse({'queue': {'pending': [{'id': t['id'], 'task': t} for t in workers.PENDING],
                                      'running': [{'id': tid, **r} for tid, r in workers.RUNNING.items()]}})
    async def schedules(_request):
        return JSONResponse({'tasks': []})
    app = Starlette(routes=[Route('/api/tasks/pred-1', detail), Route('/api/state', state),
                           Route('/api/tasks', tasks), Route('/api/schedules', schedules),
                           Route('/api/tasks/{task_id}/continue', api_task_continue, methods=['POST'])])
    client = TestClient(app)
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            if self.path == '/fixture':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(HTML.encode())
            else:
                super().do_GET()
    requests = []
    def respond(route):
        req = route.request
        path = req.url.split('/api/', 1)[1]
        if req.method == 'POST':
            requests.append(req.post_data_json)
            response = client.post('/api/' + path, json=req.post_data_json)
        else:
            response = client.get('/api/' + path)
        route.fulfill(status=response.status_code, content_type='application/json', body=response.content)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = producer_evidence_dir(tmp_path / 'screenshots', "batch4-repair-controls")
    persist = q.persist_queue_snapshot
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={'width': 1200, 'height': 850})
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.route('**/api/**', respond)
                page.goto(f'http://127.0.0.1:{server.server_port}/fixture')
                page.wait_for_function('window.ready')
                child = page.locator('.activity-row').filter(has=page.locator('[data-id="child"]'))
                root = page.locator('.activity-row').filter(has=page.locator('[data-id="pausing-root"]'))
                expect(child.locator('.activity-sub')).to_have_text('Pausing…')
                expect(root.locator('.activity-sub')).to_have_text('Paused · 1 helper still working')
                for door in ('header-restart', 'settings-restart'):
                    page.locator('#' + door).click()
                    expect(page.get_by_text('1 task is still pausing', exact=False)).to_be_visible()
                    page.screenshot(path=str(evidence / f'{door}-pausing.png'))
                    page.get_by_role('button', name='Cancel', exact=True).click()
                monkeypatch.setattr(q, 'persist_queue_snapshot', lambda **_k: False)
                page.get_by_role('button', name='Continue', exact=True).click()
                expect(page.get_by_role('button', name='Continue', exact=True)).to_be_enabled()
                claim = load_task_result(tmp_path, 'pred-1')['continued_by']
                assert claim['state'] == 'bound' and load_task_result(tmp_path, claim['successor_task_id']) is None
                # Lose all page/local identity; reload restores the exact server action.
                page.evaluate('localStorage.clear()')
                page.reload()
                page.wait_for_function('window.ready')
                retry = page.get_by_role('button', name='Retry Continue', exact=True)
                expect(retry).to_be_enabled()
                page.screenshot(path=str(evidence / 'bound-retry.png'))
                monkeypatch.setattr(q, 'persist_queue_snapshot', persist)
                retry.click()
                expect(page.locator('[data-continue-successor]')).to_have_text('Continued')
                expect(page.locator('[data-continue-successor]')).to_have_attribute('data-continue-successor', claim['successor_task_id'])
                assert [r['action_nonce'] for r in requests] == [claim['action_nonce']] * 2
                assert sum(t['id'] == claim['successor_task_id'] for t in workers.PENDING) == 1
                workers.RUNNING.clear()
                set_fence_state(tmp_path, 'pausing-root', fence_id=fence['fence_id'], state='paused')
                page.reload()
                page.wait_for_function('window.ready')
                expect(root.locator('.activity-sub')).to_have_text('Paused')
                expect(child).to_have_count(0)
                page.locator('#header-restart').click()
                expect(page.get_by_text('1 task is still pausing', exact=False)).to_have_count(0)
                page.get_by_role('button', name='Cancel', exact=True).click()
                page.screenshot(path=str(evidence / 'admitted-and-paused.png'))
                assert not errors
            finally:
                browser.close()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
    print('BATCH4_REPAIR_UI_EVIDENCE', evidence)


CONTINUE_HTML = '''<!doctype html><html class="ouro-ui"><head><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">
<script src="/purify.min.js"></script><script src="/marked.min.js"></script>
</head><body><main id="chat-mount" style="height:100vh;width:100%"></main><script type="module">
import {createChatInstance} from '/modules/chat.js';
const room=Number(new URLSearchParams(location.search).get('room')||1);
const handlers=new Map();
const ws={on(type,fn){const list=handlers.get(type)||[];list.push(fn);handlers.set(type,list);return ()=>{};},
    isConnected:()=>true,send(){throw Error('No model request belongs to this display test');}};
const emit=(type,value)=>(handlers.get(type)||[]).forEach(fn=>fn(value));
const state={activePage:'chat',projectChatIds:new Set([7]),unreadCount:0};
window.chat=createChatInstance({ws,state,updateUnreadBadge(){},chatId:room,projectId:room===7?'report-project':'',
    idPrefix:'fixture-chat',asPanel:true,mountEl:document.getElementById('chat-mount'),
    stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),gate(){return Promise.resolve(this.begin());},
        isCurrent:()=>true,apply(){}}});
window.replay=async()=>{
    for(const kind of ['budget','expired','ready']){
        const id=kind+'-'+room;
        const detail=await (await fetch('/api/tasks/'+id)).json();
        emit('chat',{task_id:id,chat_id:room,role:'assistant',is_progress:true,
            text:'Preparing the report',ts:'2026-09-26T11:59:59Z'});
        emit('chat',{...detail,task_id:id,chat_id:room,role:'system',system_type:'task_summary',
            task_terminal_status:detail.status,text:'Report preparation ended.',ts:'2026-09-26T12:00:00Z'});
    }
};
await window.replay();window.fixtureReady=true;
</script></body></html>'''


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('width', [1200, 390])
def test_hard_bound_continue_reason_on_real_collapsed_chat_cards(tmp_path, monkeypatch, engine, width):
    """Real host offers + Chat's settled-card consumer, with no model or server runtime."""
    from playwright.sync_api import sync_playwright, expect
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros import dialogue_store
    from ouroboros.task_results import load_task_result, write_task_result

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr('ouroboros.llm.LLMClient.chat', lambda *_a, **_k: pytest.fail('paid explanation'))
    for room in (1, 7):
        for kind, reason in [('budget', 'budget_exhausted'), ('expired', 'provider_unavailable'),
                             ('ready', 'provider_unavailable')]:
            tid = f'{kind}-{room}'
            _interrupted(tmp_path, tid, reason_code=reason)
            write_task_result(tmp_path, tid, 'failed', chat_id=room,
                              deadline_at='2000-01-01T00:00:00Z' if kind == 'expired' else '2099-01-01T00:00:00Z')
            dialogue_store.append(tmp_path, 'chat', {
                'task_id': tid, 'chat_id': room, 'direction': 'out', 'role': 'system',
                'system_type': 'task_summary', 'task_terminal_status': 'failed', 'is_progress': True,
                'text': 'Report preparation ended.', 'ts': '2026-09-26T12:00:00Z'})
    app = Starlette(routes=[Route('/api/tasks/{task_id}', api_task_get),
                           Route('/api/chat/history', make_chat_history_endpoint(tmp_path)),
                           Route('/api/tasks/{task_id}/continue', api_task_continue, methods=['POST'])])
    app.state.drive_root, app.state.repo_dir = tmp_path, WEB.parent
    client = TestClient(app)
    requests = []

    def respond(route):
        request = route.request
        path = request.url.split('/api/', 1)[1]
        if path.startswith(('tasks/', 'chat/history')):
            if request.method == 'POST':
                requests.append((path, request.post_data_json))
                response = client.post('/api/' + path, json=request.post_data_json)
            else:
                response = client.get('/api/' + path)
            route.fulfill(status=response.status_code, content_type='application/json', body=response.content)
        else:
            route.fulfill(content_type='application/json', body=json.dumps({
                'messages': [], 'entries': [], 'active_chat_activities': [],
                'active_chat_activities_complete': True, 'supervisor_ready': True}))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith('/fixture'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(CONTINUE_HTML.encode('utf-8'))
            else:
                super().do_GET()

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path / 'screenshots')))
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={'width': width, 'height': 900}, has_touch=width < 980)
                errors = PageErrors(page, engine)
                page.route('**/api/**', respond)
                for room in (1, 7):
                    page.goto(f'http://127.0.0.1:{server.server_port}/fixture?room={room}')
                    page.wait_for_function('window.fixtureReady === true')
                    cards = {kind: page.locator(f'.chat-live-card[data-task-id="{kind}-{room}"]')
                             for kind in ('budget', 'expired', 'ready')}
                    for card in cards.values():
                        expect(card).to_have_count(1)
                        if card.get_attribute('data-expanded') == '1':
                            card.locator('[data-live-disclosure]').click()
                    page.screenshot(path=str(evidence / f'continue-{engine}-{width}-room{room}.png'))
                    for kind in ('budget', 'expired'):
                        cards[kind].screenshot(path=str(evidence / f'continue-{engine}-{width}-room{room}-{kind}.png'))
                    before_posts = len(requests)
                    for kind, phrase in [('budget', 'budget'), ('expired', 'deadline')]:
                        card = cards[kind]
                        reason = card.locator(':scope > .chat-live-foot > .chat-live-foot-end > .chat-live-actions > [data-continue-refusal]')
                        expect(reason).to_be_visible()
                        expect(reason).to_contain_text('Continue unavailable.')
                        expect(reason).to_contain_text(phrase)
                        expect(reason).to_have_attribute('data-tone', 'muted')  # shared neutral alias
                        assert reason.get_attribute('aria-live') is None
                        assert reason.evaluate('el => el.tagName') == 'SPAN'
                        expect(card.locator('[data-continue-task]')).to_have_count(0)
                        expect(card.locator('[data-live-phase]')).to_have_text('Failed')
                        assert card.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
                    expect(cards['expired'].locator('[data-live-activity]')).to_contain_text('could not get a usable model answer')
                    page.evaluate('window.savedRefusal=document.querySelector("[data-continue-refusal]")')
                    page.evaluate('window.replay()')
                    assert page.evaluate('window.savedRefusal===document.querySelector("[data-continue-refusal]")')
                    expect(cards['budget'].locator('[data-continue-refusal]')).to_have_count(1)
                    assert len(requests) == before_posts
                    page.reload()
                    page.wait_for_function('window.fixtureReady === true')
                    expect(cards['budget'].locator('[data-continue-refusal]')).to_be_visible()
                    expect(cards['expired'].locator('[data-continue-refusal]')).to_contain_text('deadline')
                    assert len(requests) == before_posts
                    # A stale explicit press still refuses under the same hard rail.
                    denied = client.post(f'/api/tasks/budget-{room}/continue', json={'action_nonce': 'stale-press-123'})
                    assert denied.status_code == 409 and denied.json()['reason_code'] == 'hard_limit_reached'
                    assert not load_task_result(tmp_path, f'budget-{room}').get('continued_by')
                    cards['ready'].get_by_role('button', name='Continue', exact=True).click()
                    expect(cards['ready'].locator('[data-continue-task]')).to_have_text('Continued')
                    assert len(requests) == before_posts + 1
                    successor = load_task_result(tmp_path, f'ready-{room}')['continued_by']['successor_task_id']
                    assert sum(task['id'] == successor for task in workers.PENDING) == 1
                    page.screenshot(path=str(evidence / f'continue-{engine}-{width}-room{room}-continued.png'))
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
    print('BATCH4_CONTINUE_DISCLOSURE_UI_EVIDENCE', evidence)
