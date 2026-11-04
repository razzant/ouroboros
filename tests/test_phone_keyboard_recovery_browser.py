"""Production viewport recovery order, not a native iPhone keyboard claim."""
from __future__ import annotations

import pytest
from tests.test_subscription_setup_browser import subscription_ui
from tests.test_shell_interactions_browser import shell_ui, open_project
from tests.test_design_system_gate_browser import seed_page, settle
from tests.test_phone_chat_reopen_browser import history, shot

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ['subscription_ui', 'shell_ui']

VIEWPORT = """(() => {
    let height = 844, pinched = false;
    const viewport = new EventTarget();
    Object.defineProperties(viewport, {height: {get: () => height},
        width: {get: () => document.documentElement.clientWidth / (pinched ? 2 : 1)}, scale: {get: () => pinched ? 2 : 1}});
    Object.defineProperty(window, 'visualViewport', {configurable:true, value:viewport});
    window.__viewport = (h, emit = true) => {height=h; if(emit) viewport.dispatchEvent(new Event('resize'));};
    window.__pinch = () => {pinched=true; viewport.dispatchEvent(new Event('resize'));};
})()"""

@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
@pytest.mark.parametrize('dismiss', ['blur', 'host-collapse', 'closing'])
def test_blur_reconciles_restored_viewport_without_another_resize(shell_ui, project_room, dismiss):
    phone = {**shell_ui, 'new_page': lambda **kw: shell_ui['new_page'](is_mobile=True, **kw)}
    page = seed_page(phone, 390, 844, telegram=True)
    page.add_init_script(VIEWPORT)
    page.route('**/api/chat/history**', lambda r:r.fulfill(json=history(['Long saved line. '*25]*20)))
    page.goto(shell_ui['url'], wait_until='domcontentloaded')
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    if project_room: open_project(page)
    field = page.locator('textarea.chat-input:visible' if project_room else '#chat-input')
    field.fill('Keep this draft.'); field.focus()
    page.evaluate('window.__viewport(320)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    # Native focusout and viewport restoration can be one task. The final
    # measurement is available by the focusout frame, without another resize.
    page.evaluate("""({id, mode}) => {
        if(mode==='host-collapse') window.__viewport(60);
        else document.getElementById(id).blur();
        if(mode!=='closing') window.__viewport(844, false);
    }""", {'id':field.get_attribute('id'), 'mode':dismiss})
    settle(page)
    if dismiss == 'closing':
        assert field.evaluate('f=>f.getBoundingClientRect().bottom') <= 321
        page.evaluate('window.__viewport(844)'); settle(page)
    facts = field.evaluate("""f=>({focus:document.activeElement===f,
        height:parseFloat(getComputedStyle(document.body).height),
        top:f.getBoundingClientRect().top,bottom:f.getBoundingClientRect().bottom,
        keyboard:document.body.classList.contains('keyboard-open')})""")
    shot(page, f'recovery-{project_room}-{dismiss}-{shell_ui["engine"]}')
    assert facts['height'] == 844 and not facts['keyboard'], facts
    assert 0 <= facts['top'] < facts['bottom'] <= 844, facts
    assert field.input_value() == 'Keep this draft.'
    field.focus(); page.evaluate('window.__viewport(320)'); settle(page)
    assert page.locator('body').evaluate("e=>e.classList.contains('keyboard-open')")
    assert field.evaluate('f=>f.getBoundingClientRect().bottom') <= 321
    page.close()


def test_blur_during_pinch_releases_keyboard_state_without_resizing_shell(shell_ui):
    phone = {**shell_ui, 'new_page': lambda **kw: shell_ui['new_page'](is_mobile=True, **kw)}
    page = seed_page(phone, 390, 844, telegram=True)
    page.add_init_script(VIEWPORT)
    page.goto(shell_ui['url'], wait_until='domcontentloaded')
    field = page.locator('#chat-input'); field.wait_for(state='visible'); field.focus()
    page.evaluate('window.__viewport(320)'); settle(page)
    assert page.locator('body').evaluate("e=>e.classList.contains('keyboard-open')")
    before = page.locator('body').evaluate('e=>e.getBoundingClientRect().height')
    page.evaluate('window.__pinch()'); field.blur(); settle(page)
    assert not page.locator('body').evaluate("e=>e.classList.contains('keyboard-open')")
    assert page.locator('body').evaluate('e=>e.getBoundingClientRect().height') == before
    page.close()
