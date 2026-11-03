"""Production SPA on a phone with controlled APIs; no runtime or native keyboard.

The viewport signal is synthetic; Apple writing assistance still needs a device.
A fresh page in the same browser context loses sessionStorage but keeps the text
preview. Canonical reads, not that preview, own history and actions.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.test_subscription_setup_browser import subscription_ui
from tests.test_shell_interactions_browser import shell_ui, open_project
from tests.test_design_system_gate_browser import seed_page, settle
from tests.test_ui_smoke_playwright import _install_controlled_visual_viewport

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ['subscription_ui', 'shell_ui']


def shot(page, name):
    root = os.environ.get('OUROBOROS_UI_EVIDENCE_DIR')
    if root:
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        options = {'clip': {'x': 0, 'y': 0, 'width': 390, 'height': 250}} if 'keyboard' in name else {}
        page.screenshot(path=str(target / (name + '.png')), **options)


def history(texts):
    return {'messages': [{'role': 'assistant', 'text': text,
             'ts': f'2026-11-03T17:00:{i:02d}Z', 'history_id': f'chat:{i}',
             'history_position': {'source': 'chat:fixture', 'offset': i}}
            for i, text in enumerate(texts)], 'page_cursor': 'page:recent',
            'next_cursor': None, 'after_cursor': 'after:recent', 'has_more': False,
            'window': {'complete': True, 'truncated_by': []},
            'coverage': {'v': 1, 'view': 'phone', 'upper': {'chat': 100},
                'spans': {'chat': {'chain': 'fixture', 'from': 0, 'to': 100, 'gaps': []}, 'progress': None}}}


@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_short_phone_composer_fits_and_keeps_assistance(shell_ui, theme, project_room):
    phone = {**shell_ui, 'new_page': lambda **options: shell_ui['new_page'](is_mobile=True, **options)}
    page = seed_page(phone, 390, 844, telegram=True)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([
        'The conversation stays here while you write. This is a saved message, not a loading placeholder.'
    ] * 12)))
    _install_controlled_visual_viewport(page, 844)
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    page.goto(shell_ui['url'])
    page.locator('#chat-input').wait_for(state='visible')
    if project_room:
        open_project(page)
    field = page.locator('textarea.chat-input:visible' if project_room else '#chat-input')
    field.wait_for(state='visible')
    field_id = field.get_attribute('id')
    assert field.get_attribute('autocorrect') == 'on'
    assert field.get_attribute('autocapitalize') == 'sentences'
    assert field.get_attribute('spellcheck') == 'true'
    # No focused editable is not a keyboard, even in a short WebView.
    page.evaluate('window.__setTestVisualViewportHeight(250)')
    assert not page.locator('body').evaluate("e => e.classList.contains('keyboard-open')")
    page.evaluate('window.__setTestVisualViewportHeight(844)')
    settle(page)
    field.focus()
    page.evaluate('window.__setTestVisualViewportHeight(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    for text in ['A normal message', '\n'.join(['A long draft with wrapped words'] * 12), 'Short again']:
        field.fill(text)
        settle(page)
        facts = page.evaluate("""id => {
            const f = document.getElementById(id), area = document.getElementById(id.replace(/input$/, 'input-area'));
            const feed = document.getElementById(id.replace(/input$/, 'messages'));
            return {bottom: area.getBoundingClientRect().bottom, input: f.getBoundingClientRect().height,
                cap: parseFloat(getComputedStyle(f).maxHeight), feed: feed.clientHeight,
                overflow: feed.scrollWidth - feed.clientWidth};
        }""", field_id)
        assert facts['bottom'] <= 251 and facts['feed'] > 0, facts
        assert facts['input'] <= facts['cap'] + 1 and facts['overflow'] <= 1, facts
    shot(page, f'phone-keyboard-{project_room}-{shell_ui["engine"]}-{theme}')
    page.evaluate('window.__setTestVisualViewportHeight(844)')
    field.blur()
    settle(page)
    shot(page, f'phone-room-{project_room}-{shell_ui["engine"]}-{theme}')
    page.close()


@pytest.mark.parametrize('mode', ['newest', 'reference', 'selected-empty'])
def test_new_document_shows_saved_text_before_history_and_reconciles(shell_ui, mode):
    ui = shell_ui
    page = seed_page(ui, 390, 844, telegram=True)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['Saved answer', 'Obsolete preview row'])))
    page.goto(ui['url'])
    page.locator('#chat-messages .chat-bubble').filter(has_text='Saved answer').wait_for()
    page.wait_for_function("localStorage.getItem('ouroboros.chat-preview.v1')?.includes('Saved answer')")
    # A recreated browser context retains the same origin's persistent store,
    # not sessionStorage. Telegram itself may discard storage; that case opens cold.
    context = page.context.browser.new_context(storage_state=page.context.storage_state(),
        viewport={'width': 390, 'height': 844}, has_touch=True)
    other = context.new_page()
    other.route_web_socket('**/ws', lambda ws: ws.send('{"type":"heartbeat"}'))
    other.route('**/api/**', lambda route: route.fulfill(json={}))
    other.route('**/api/onboarding', lambda route: route.fulfill(status=204))
    other.route('**/api/state', lambda route: route.fulfill(json={'supervisor_ready': True, 'projects': [], 'active_chat_activities': []}))
    other.route('**/api/projects', lambda route: route.fulfill(json={'projects': []}))
    pending = []
    other.route('**/api/chat/history**', lambda route: pending.append(route))
    ui['errors'].watch(other, navigates=True)
    other.goto(ui['url'])
    other.locator('[data-saved-preview]').wait_for(state='visible')
    assert 'Saved answer' in other.locator('[data-saved-preview]').inner_text()
    assert other.locator('[data-saved-preview] button, [data-saved-preview] a').count() == 0
    assert other.locator('[data-saved-preview] [data-history-id]').count() == 0
    assert other.locator('#chat-messages').get_attribute('data-history-hydrated') is None
    other.locator('#chat-input').fill('Unsent draft')
    shot(other, f'phone-preview-{ui["engine"]}')
    other.wait_for_function('document.querySelector("#chat-input") !== null')
    # Wait for the actual read to be held, not for a guessed duration.
    from time import monotonic
    until = monotonic() + 10
    while not pending and monotonic() < until:
        other.wait_for_timeout(20)
    assert pending
    if mode == 'reference':
        pending.pop(0).fulfill(status=500, json={'error': 'Controlled history failure'})
        other.get_by_role('button', name='Retry loading messages').wait_for()
        target = history(['**Canonical formatted**'])
        target['anchor'] = {'state': 'found', 'task_id': 'preview-target', 'history_id': 'chat:0', 'chat_id': 1}
        other.route('**/api/chat/history**around=**', lambda route: route.fulfill(json=target))
        other.evaluate("window.dispatchEvent(new CustomEvent('ouro:open-task', {detail: {task_id: 'preview-target', chat_id: 1}}))")
        other.locator('#chat-messages strong').filter(has_text='Canonical formatted').wait_for()
        assert other.locator('[data-saved-preview]').count() == 0
        assert 'Obsolete preview row' not in other.locator('#chat-messages').inner_text()
        assert other.locator('#chat-messages .chat-bubble button').count() > 0
        context.close(); page.close()
        return
    if mode == 'selected-empty':
        other.evaluate("""() => {
            const text = document.querySelector('[data-saved-preview] .chat-text');
            const range = document.createRange(); range.selectNodeContents(text);
            getSelection().removeAllRanges(); getSelection().addRange(range);
        }""")
        pending.pop(0).fulfill(json=history([]))
        other.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
        assert other.locator('[data-saved-preview]').count() == 1
        assert 'retained until selection is released' in other.locator('[data-saved-preview]').inner_text()
        assert other.evaluate('getSelection().toString()') == 'Saved answer'
        other.evaluate("getSelection().removeAllRanges(); document.dispatchEvent(new Event('selectionchange'))")
        other.locator('[data-saved-preview]').wait_for(state='detached')
        context.close(); page.close()
        return
    pending.pop(0).fulfill(json=history(['Saved answer', 'Fresh answer']))
    other.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    assert other.locator('[data-saved-preview]').count() == 0
    for text in ['Saved answer', 'Fresh answer']:
        assert other.locator('#chat-messages .chat-bubble').filter(has_text=text).count() == 1
    assert other.locator('#chat-input').input_value() == 'Unsent draft'
    shot(other, f'phone-reconciled-{ui["engine"]}')
    context.close()
    page.close()
