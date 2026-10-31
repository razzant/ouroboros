"""Production Main header/composer with stubbed transport on an owned static listener.

This checks browser delivery and presentation, not packaged native tray/WebView behavior.
"""
import json
import os
from pathlib import Path

import pytest
from tests.test_subscription_setup_browser import subscription_ui  # noqa: F401

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
FAILURE = 'Panic was not confirmed. Check Ouroboros on the host computer.'
DESKTOP_FAILURE = 'Panic was not confirmed. Use Panic in the tray or menu bar.'


@pytest.fixture(params=['chromium', 'webkit'])
def panic_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


def _typed(page):
    page.locator('#chat-input').fill('/panic')
    page.locator('#chat-input').press('Enter')


def _facts(page):
    return page.evaluate('''() => {
        // No veil covers the app while the socket comes back (owner 20.10, Q4): the room's own Panic stays its control.
        const button = document.querySelector('#page-chat [data-chat-command="panic"]');
        const box = button?.getBoundingClientRect();
        const hit = box && document.elementFromPoint(box.x + box.width/2, box.y + box.height/2);
        const toast = document.querySelector('#toast-stack .toast');
        const rect = toast?.getBoundingClientRect();
        return {
            theme: document.documentElement.dataset.theme,
            button: box ? {width: box.width, height: box.height, visible: box.left >= 0 && box.right <= innerWidth,
                     hit: hit === button || button.contains(hit),
                     touch_hit: [[-21,0],[21,0],[0,-21],[0,21]].every(([x,y]) => {
                        const el = document.elementFromPoint(box.x + box.width/2+x, box.y + box.height/2+y);
                        return el === button || button.contains(el);
                     })} : null,
            user_bubbles: document.querySelectorAll('#chat-messages .chat-bubble.user').length,
            draft: document.querySelector('#chat-input').value,
            attachments: document.querySelector('#chat-attachment-preview').innerText,
            dialog: Boolean(document.querySelector('.confirm-dialog')),
            queued: window.__ouroWs._pendingMessages,
            resend: [...document.querySelectorAll('[data-unconfirmed-action="retry"]')].map(n => n.textContent),
            kept: Object.entries(sessionStorage).filter(([key]) => key.includes('unconfirmed')),
            toast: toast ? {text: toast.innerText, clipped: toast.scrollWidth > toast.clientWidth + 1,
                on_screen: rect.left >= 0 && rect.right <= innerWidth && rect.bottom <= innerHeight} : null,
            page_overflow: document.documentElement.scrollWidth > innerWidth + 1,
            veil: Boolean(document.querySelector('#reconnect-overlay')),
        };
    }''')


@pytest.mark.parametrize('theme,width,height', [('dark', 1280, 800), ('light', 390, 844)])
def test_real_controls_bridge_first_offline_failure_and_no_successor_replay(panic_ui, theme, width, height):
    from playwright.sync_api import expect

    ui, messages, generations = panic_ui, [], []
    ui['page'].close()
    page = ui['new_page'](has_touch=width < 720, device_scale_factor=2)
    page.set_default_timeout(10000)
    page.set_viewport_size({'width': width, 'height': height})
    page.emulate_media(color_scheme=theme)
    console_errors = []
    page.on('console', lambda msg: console_errors.append(msg.text) if msg.type == 'error' else None)

    def connected(socket):
        generation = len(generations)
        generations.append(socket)
        socket.send(json.dumps({'type': 'heartbeat'}))
        def received(raw):
            msg = json.loads(raw)
            messages.append({'generation': generation, 'frame': msg})
            if msg.get('cmd') == '/panic':
                socket.send(json.dumps({'type': 'panic_accepted'}))
        socket.on_message(received)
    page.route('**/api/state', lambda r: r.fulfill(json={
        'supervisor_ready': True, 'active_chat_activities': [], 'projects': [], 'sha': ''}))
    page.route('**/api/chat/history*', lambda r: r.fulfill(json={
        'messages': [], 'progress': [], 'has_more': False, 'page_cursor': 'panic-fixture',
        'coverage': {'v': 1, 'view': 'panic-fixture', 'upper': {'chat': 0, 'progress': 0},
                     'spans': {key: {'from': 0, 'to': 0, 'chain': 'empty', 'gaps': []}
                               for key in ('chat', 'progress')}}}))
    page.route_web_socket('**/ws', connected)
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input')
    page.wait_for_function('window.__ouroWs?.ws?.readyState === 1')
    page.wait_for_function('(theme) => document.documentElement.dataset.theme === theme', arg=theme)
    button = page.locator('#page-chat [data-chat-command="panic"]')  # every Project bar has its own
    expect(button).to_be_visible()
    panic_messages = lambda: [m for m in messages if m['frame'].get('cmd') == '/panic']

    button.click()
    page.wait_for_function('window.__ouroWs.ws.bufferedAmount === 0')
    expect(page.locator('.confirm-dialog')).to_have_count(0)
    _typed(page)
    page.wait_for_timeout(100)  # routed socket messages finish; no server shutdown is performed
    assert [m['frame']['client_surface']['input'] for m in panic_messages()] == ['command', 'typed_chat']
    # A staged file must not put a typed emergency behind the uploader.
    page.locator('#chat-file-input').set_input_files({
        'name': 'note.txt', 'mimeType': 'text/plain', 'buffer': b'Owned disposable upload fixture.'})
    _typed(page)
    page.wait_for_timeout(100)
    assert len(panic_messages()) == 3
    assert not [post for post in ui['posts'] if 'upload' in post[0]]
    expect(page.locator('#chat-messages .chat-bubble.user')).to_have_count(0)
    expect(page.locator('#chat-input')).to_have_value('')
    assert page.evaluate("sessionStorage.getItem('ouro_chat_draft')") is None
    expect(page.locator('#chat-attachment-preview')).to_contain_text('note.txt')
    before = len(panic_messages())

    # The production caller feature-detects the independent bridge at each press.
    page.evaluate('''() => {
        window.__bridgeCalls = 0;
        window.pywebview = {api: {panic_stop: async () => { window.__bridgeCalls++; return {status:'stopping'}; }}};
    }''')
    button.click()
    _typed(page)
    page.wait_for_function('window.__bridgeCalls === 2')
    expect(page.locator('#chat-input')).to_have_value('')
    expect(page.locator('#chat-attachment-preview')).to_contain_text('note.txt')
    assert len(panic_messages()) == before
    page.evaluate('delete window.pywebview')

    # Hold the ordinary reconnect timer so both actual controls see an unavailable server.
    page.evaluate('window.__ouroWs.reconnectDelay = 60000; window.__ouroWs.ws.close()')
    page.wait_for_function('!window.__ouroWs.ws')
    old_generations = len(generations)
    expect(page.locator('#reconnect-overlay')).to_have_count(0)
    _typed(page)
    expect(page.get_by_text(FAILURE, exact=True).first).to_be_visible()
    expect(page.locator('.confirm-dialog')).to_have_count(0)
    page.wait_for_function("[...document.querySelectorAll('.toast')].every(n => Number(getComputedStyle(n).opacity) > .95)")
    evidence = Path(os.environ['OUROBOROS_UI_EVIDENCE_DIR'])
    evidence.mkdir(parents=True, exist_ok=True)
    name = f'production-panic-{ui["engine"]}-{theme}-{width}'
    page.screenshot(path=str(evidence / f'{name}.png'), scale='css')
    page.locator('.toast').first.screenshot(path=str(evidence / f'{name}-toast.png'))
    facts = _facts(page)
    assert facts['queued'] == [] and len(generations) == old_generations
    assert facts['toast'] == {'text': FAILURE, 'clipped': False, 'on_screen': True}
    assert facts['button']['hit'] and not facts['veil'] and not facts['page_overflow'], facts
    assert facts['user_bubbles'] == 0 and facts['draft'] == '/panic' and 'note.txt' in facts['attachments']
    assert not facts['resend'] and all('/panic' not in value for _, value in facts['kept'])

    # With no veil, the room's own Panic reaches the independent desktop bridge while the server is unavailable.
    page.evaluate('''() => {
        window.pywebview = {api: {panic_stop: async () => { window.__bridgeCalls++; return {status:'stopping'}; }}};
        window.dispatchEvent(new Event('pywebviewready'));
    }''')
    button.click()
    page.wait_for_function('window.__bridgeCalls === 3')
    assert len(panic_messages()) == before
    page.screenshot(path=str(evidence / f'{name}-desktop.png'), scale='css')
    desktop_facts = _facts(page)
    assert desktop_facts['button']['visible'] and desktop_facts['button']['hit']
    if width < 720:
        assert desktop_facts['button']['touch_hit']
    # A bridge refusal delayed across reconnection retains the ORIGINAL absent socket.
    page.evaluate('''() => {
        window.pywebview.api.panic_stop = () => new Promise((_, reject) => {window.__bridgeReject = reject});
    }''')
    button.click()
    page.wait_for_function('Boolean(window.__bridgeReject)')
    # The normal connection owner now opens a successor; offline gestures are never replayed.
    page.evaluate('window.__ouroWs.connect()')
    page.wait_for_function('window.__ouroWs.ws?.readyState === 1')
    page.evaluate("window.__bridgeReject(Error('fixture bridge refusal'))")
    page.wait_for_timeout(100)
    expect(page.get_by_text(DESKTOP_FAILURE, exact=True).first).to_be_visible()
    assert len(generations) == old_generations + 1 and len(panic_messages()) == before
    assert not [post for post in ui['posts'] if post[0] == '/api/command']
    assert not ui['errors'] and not console_errors
    (evidence / f'{name}.json').write_text(json.dumps({
        'facts': facts, 'desktop_facts': desktop_facts, 'messages': messages, 'generations': len(generations),
        'page_errors': ui['errors'], 'console_errors': console_errors,
        'fixture': 'production SPA; fake API, WebSocket receiver and desktop bridge',
    }, indent=2) + '\n')
