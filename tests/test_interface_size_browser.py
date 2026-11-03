"""Settings → Appearance → Interface size on the real SPA (owner 23.10: one slider, smaller than 90 % too).

Static-server fixture with synthetic API rows (test_subscription_setup_browser); no runtime starts. Three surfaces:
a browser on a computer (the slider gives way to the browser's own zoom keys), the desktop app (a recorder stands in
for its bridge: the slider and ⌘ + − 0 set the window's size there) and a phone (mobile Chromium honours the viewport
meta, so the page is laid out at another width). The engines' own page zoom was measured on a real WKWebView for the
design, not here.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import open_nav, rest, seed_page, settle

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

BRIDGE = """(() => {
    window.__sizes = [];
    window.pywebview = {api: {
        shell_info: async () => ({interface_size: {supported: true, min: 50, max: 200, step: 5, percent: 100}}),
        set_interface_size: async (percent, page, sequence) => { window.__sizes.push(percent); return {ok: true, percent}; },
    }};
    addEventListener('DOMContentLoaded', () => setTimeout(() => dispatchEvent(new Event('pywebviewready')), 50));
})();"""
FACTS = """() => {
    const block = document.querySelector('[data-interface-size]');
    const input = block.querySelector('[data-interface-size-input]');
    return {rowHidden: block.querySelector('[data-interface-size-row]').hidden, min: input.min, max: input.max,
            value: input.value, shown: block.querySelector('[data-interface-size-value]').textContent,
            note: block.querySelector('[data-interface-size-note]').textContent, innerWidth,
            meta: document.querySelector('meta[name="viewport"]').getAttribute('content')};
}"""


def _appearance(page):
    open_nav(page, 'settings')
    page.locator('#page-settings [data-settings-tab="appearance"]').click()
    page.locator('[data-interface-size]').scroll_into_view_if_needed()
    rest(page)


def _slide_to(page, value):
    page.locator('[data-interface-size-input]').evaluate(
        """(el, value) => { el.value = String(value); el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true})); }""", value)


@pytest.fixture(params=['chromium', 'webkit'])
def size_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


def test_a_browser_keeps_its_own_zoom_and_the_control_names_its_keys(size_ui):
    page = seed_page(size_ui, 1280, 860)
    try:
        page.goto(size_ui['url'])
        _appearance(page)
        facts = page.evaluate(FACTS)
        assert facts['rowHidden'], facts
        assert 'This browser sets the size of the page' in facts['note'], facts
    finally:
        page.close()


def test_the_desktop_app_slider_and_keys_set_the_windows_own_size(size_ui):
    page = seed_page(size_ui, 1280, 860)
    page.add_init_script(BRIDGE)
    try:
        page.goto(size_ui['url'])
        _appearance(page)
        page.wait_for_function("!document.querySelector('[data-interface-size-row]').hidden")
        facts = page.evaluate(FACTS)
        assert facts['min'] == '50' and facts['max'] == '200' and facts['value'] == '100', facts
        assert 'For this window' in facts['note'], facts
        _slide_to(page, 80)
        page.wait_for_function("window.__sizes.length === 1")
        assert page.evaluate('window.__sizes') == [80], 'below 90 % (owner 23.10)'
        modifier = 'Meta' if page.evaluate("/Mac/.test(navigator.platform)") else 'Control'
        page.keyboard.press(f'{modifier}+Equal')
        page.wait_for_function("window.__sizes.length === 2")
        page.keyboard.press(f'{modifier}+Digit0')
        page.wait_for_function("window.__sizes.length === 3")
        assert page.evaluate('window.__sizes') == [80, 90, 100]
        assert page.evaluate(FACTS)['shown'] == '100%'
    finally:
        page.close()


def test_a_phone_lays_the_page_out_at_another_width(subscription_ui, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', 'chromium')
    page = subscription_ui['new_page'](viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
    try:
        page.goto(subscription_ui['url'])
        _appearance(page)
        page.wait_for_function("!document.querySelector('[data-interface-size-row]').hidden")
        facts = page.evaluate(FACTS)
        assert facts['min'] == '75' and facts['max'] == '120' and facts['innerWidth'] == 390, facts
        _slide_to(page, 120)
        page.wait_for_function("innerWidth === 325")
        facts = page.evaluate(FACTS)
        assert facts['meta'].startswith('width=325, initial-scale=1.2'), facts
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), 'nothing scrolls sideways'
        # A split screen: narrower, still portrait; 360 px allows 110 % at most, and 390 px brings back the owner's 120 %.
        page.set_viewport_size({'width': 360, 'height': 844})
        page.wait_for_function("document.documentElement.clientWidth === 327 && ouroViewportScale.percent === 110")
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function("document.documentElement.clientWidth === 325 && ouroViewportScale.percent === 120")
        assert page.evaluate(FACTS)['value'] == '120'
        _slide_to(page, 80)  # smaller than 90 % (owner 23.10): 488 px laid out, all of them on the screen
        page.wait_for_function("document.documentElement.clientWidth === 488")
        _expect_room_fills_screen(page)
        assert page.evaluate(FACTS)['meta'].startswith('width=488, initial-scale=0.79918'), page.evaluate(FACTS)
        subscription_ui['errors'].watch(page)  # the reload may cut off this page's own reads; other pages stay strict
        page.reload()
        page.wait_for_function("document.documentElement.clientWidth === 488")  # the stored size is the first layout
        _expect_room_fills_screen(page)
    finally:
        page.close()


def _expect_room_fills_screen(page):
    """The room is as tall as the screen shows; a size below 100 % once read as a pinch and kept the old height."""
    try:
        page.wait_for_function("Math.abs(document.getElementById('app').getBoundingClientRect().bottom"
                               " - visualViewport.height) <= 1.5", timeout=5000)
    except Exception:
        pass
    room = page.evaluate("() => ({bottom: document.getElementById('app').getBoundingClientRect().bottom,"
                         " screen: visualViewport.height, vvh: getComputedStyle(document.documentElement)"
                         ".getPropertyValue('--vvh')})")
    assert abs(room['bottom'] - room['screen']) <= 1.5, room


@pytest.mark.parametrize('theme,width,height', [('light', 1280, 860), ('dark', 390, 844)])
@pytest.mark.parametrize('mode', ['pending', 'missing', 'error', 'unsupported', 'supported'])
def test_window_capability_facts_survive_the_real_settings_flow(subscription_ui, theme, width, height, mode):
    """Real page and controls, recording bridge only; no OS permission or update is invoked."""
    import json

    page = seed_page(subscription_ui, width, height, touch=False)
    page.add_init_script("localStorage.setItem('ouroboros.theme', " + json.dumps(theme) + ");")
    page.add_init_script("""(() => {
        window.__factsMode = MODE;
        window.__sizes = [];
        window.__factsApi = {
            shell_info: async () => {
                if (window.__factsMode === 'error') throw new Error('controlled bridge read failure');
                return {shell_version: '7.7.0', persistent_storage: true,
                    interface_size: {supported: window.__factsMode !== 'unsupported', min: 50, max: 200, percent: 100}};
            },
            set_interface_size: async percent => { window.__sizes.push(percent); return {ok: true, percent}; },
        };
        window.pywebview = window.__factsMode === 'pending' ? {} : {api:
            window.__factsMode === 'missing' ? {request_attention: async () => ({})} : window.__factsApi};
    })();""".replace('MODE', json.dumps(mode)))
    expected = {
        'pending': '',
        'missing': 'This desktop app cannot change its size.',
        'error': 'Could not read whether this window can change its size.',
        'unsupported': 'This window cannot change its size.',
        'supported': 'For this window;',
    }
    try:
        page.goto(subscription_ui['url'])
        _appearance(page)
        page.wait_for_function("expected => document.querySelector('[data-interface-size-note]').textContent.startsWith(expected)",
                               arg=expected[mode])
        facts = page.evaluate(FACTS)
        assert facts['note'].startswith(expected[mode]), facts
        assert facts['rowHidden'] == (mode != 'supported'), facts
        assert not any(word in facts['note'].lower() for word in ('update', 'restart', 'reinstall')), facts
        assert page.locator('[data-interface-size] a').count() == 0
        setup_browser.capture(page, f'window-facts-{mode}-{theme}')
        if mode in {'pending', 'error'}:
            page.evaluate("() => {window.__factsMode = 'supported'; window.pywebview.api = window.__factsApi;"
                          "dispatchEvent(new Event('pywebviewready'));}")
            page.wait_for_function("!document.querySelector('[data-interface-size-row]').hidden")
        if mode in {'pending', 'error', 'supported'}:
            _slide_to(page, 80)
            page.wait_for_function("window.__sizes.includes(80)")
            page.evaluate("() => {window.__factsMode = 'error'; dispatchEvent(new Event('pywebviewready'));}")
            rest(page)
            facts = page.evaluate(FACTS)
            assert not facts['rowHidden'] and facts['value'] == '80', facts
            assert facts['note'].startswith('For this window;'), facts
            setup_browser.capture(page, f'window-facts-retained-{mode}-{theme}')
        assert page.locator('#page-settings .settings-scroll').evaluate(
            'e => e.scrollWidth <= e.clientWidth + 1'), 'the actual Settings scroller contains its controls'
        forbidden = {'/api/settings', '/api/update/prepare', '/api/update/apply', '/api/command'}
        assert not any(path in forbidden for path, _ in subscription_ui['posts']), \
            'viewing capability facts must not submit settings or update work'
    finally:
        page.close()


# A recording bridge whose calls go as the test says: `__reads`/`__sets` are 'answer', 'fail' or 'hold' (the call waits
# in `__held` until the test settles it). Nothing reaches an operating system.
RACE_BRIDGE = """(() => {
    window.__sizes = [];
    window.__answers = [];
    window.__held = [];
    window.__readCount = 0;
    window.__reads = READS;
    window.__sets = 'answer';
    window.__permission = 'not_determined';
    window.__percent = 100;
    // The window answers a call when it is made, as launcher_page_zoom.PageZoom does; 'hold' delays only the delivery.
    const go = (how, answer) => {
        if (how === 'fail') return Promise.reject(new Error('controlled bridge failure'));
        const value = answer();
        return how === 'hold' ? new Promise((resolve, reject) => { window.__held.push({resolve: () => resolve(value), reject}); })
            : Promise.resolve(value);
    };
    window.pywebview = {api: {
        shell_info: () => { window.__readCount += 1; return go(window.__reads, () => ({shell_version: '8.0.0',
            bundle_version: '7.7.0', persistent_storage: true, source_launcher: true, macos_source_entry: false,
            native_notifications: {available: true, status: window.__permission, platform: 'macos', reason: ''},
            interface_size: {supported: true, min: 50, max: 200, step: 5, percent: window.__percent}})); },
        // 'refuse' is PageZoom.request's 'unavailable': the size is stored, the window's own write was not scheduled.
        set_interface_size: percent => { window.__sizes.push(percent);
            const refused = window.__sets === 'refuse';
            return go(refused ? 'answer' : window.__sets, () => { window.__percent = percent;
                const answer = refused ? {ok: false, state: 'unavailable', percent} : {ok: true, state: 'scheduled', percent};
                window.__answers.push(answer);
                return answer; }); },
        show_native_notification: async () => ({ok: true, status: 'submitted'}),
        request_native_notifications: async () => { window.__permission = 'authorized';
            return {available: true, status: 'authorized', platform: 'macos', reason: ''}; },
    }};
})();"""
LINES = """() => ({storage: document.querySelector('#page-settings [data-shell-storage-status]').textContent,
    notify: document.querySelector('#page-settings [data-notify-status]').textContent})"""
# What the next chat frame would carry: the page's own controller (the same `ouroInterfaceSize`), read at send time.
SENT_SIZE = """async () => {
    const before = window.ouroInterfaceSize;
    const { clientSurfaceSnapshot } = await import('/static/modules/client_surface.js');
    return {size: clientSurfaceSnapshot().interface_size ?? null, same: window.ouroInterfaceSize === before};
}"""


@pytest.mark.parametrize('theme,width,height', [('light', 1280, 860), ('dark', 390, 844)])
@pytest.mark.parametrize('race', ['health', 'permission', 'size'])
def test_window_fact_races_keep_the_windows_newest_answer(subscription_ui, theme, width, height, race):
    """The three reviewed sequences on the real Settings page; a recording bridge, a held /api/health.

    health: the first shell read fails while /api/health is slow, the window answers meanwhile, and the stamped
    version arriving later must not replace that answer. permission: a shell kept through a failed read must not
    undo the permission the owner granted while that read was pending. size: a size the window never reported is
    the owner's choice on screen, not the window's size in the send-time fact, and choosing it again retries."""
    import json
    import time

    page = seed_page(subscription_ui, width, height, touch=False)
    page.add_init_script("localStorage.setItem('ouroboros.theme', " + json.dumps(theme) + ");")
    page.add_init_script("localStorage.setItem('ouroboros.notifications', "
                         + json.dumps(json.dumps({'enabled': True})) + ");")
    page.add_init_script(RACE_BRIDGE.replace('READS', json.dumps('fail' if race == 'health' else 'answer')))
    held, finished = [], []
    if race == 'health':
        page.route('**/api/health', lambda route: held.append(route))
        page.on('requestfinished', lambda request: finished.append(request.url) if '/api/health' in request.url else None)
    note = "expected => document.querySelector('[data-interface-size-note]').textContent.startsWith(expected)"
    try:
        page.goto(subscription_ui['url'])
        _appearance(page)
        if race == 'health':
            assert held, 'the shell read failed, so the stamped version was asked of the server'
            assert page.evaluate(LINES)['storage'] == '', 'nothing is claimed while the window has not answered'
            page.evaluate("() => { window.__reads = 'answer'; dispatchEvent(new Event('pywebviewready')); }")
            page.wait_for_function("() => document.querySelector('#page-settings [data-shell-storage-status]')"
                                   ".textContent.includes('source launcher')")
            answered = page.evaluate(LINES)
            waiting = list(held)
            page.unroute('**/api/health')
            for route in waiting:
                route.fulfill(json={'ok': True, 'version': 'fixture', 'app_version': '6.82.0'})
            deadline = time.monotonic() + 5
            while len(finished) < len(waiting) and time.monotonic() < deadline:
                page.wait_for_timeout(20)
            rest(page)
            settle(page)
            lines = page.evaluate(LINES)
            assert lines == answered, lines
            assert 'running without the app entry' in lines['storage'] and '6.82.0' not in lines['storage'], lines
        elif race == 'permission':
            page.wait_for_function("() => document.querySelector('#page-settings [data-notify-status]')"
                                   ".textContent.includes('has not been asked yet')")
            page.evaluate("() => { window.__reads = 'hold'; dispatchEvent(new Event('pywebviewready')); }")
            page.locator('#page-settings [data-notify-test]').click()  # the owner's gesture: the system asks once
            page.wait_for_function("() => document.querySelector('#page-settings [data-notify-status]')"
                                   ".textContent.startsWith('System notifications are on')")
            page.evaluate("() => { window.__reads = 'fail';"
                          " window.__held.splice(0).forEach((read) => read.reject(new Error('controlled late failure'))); }")
            rest(page)
            settle(page)
            lines = page.evaluate(LINES)
            assert lines['notify'].startswith('System notifications are on'), lines
        else:
            page.wait_for_function("!document.querySelector('[data-interface-size-row]').hidden")
            assert page.evaluate(SENT_SIZE) == {'size': 100, 'same': True}
            page.evaluate("() => { window.__sets = 'fail'; window.__reads = 'fail'; }")
            _slide_to(page, 120)
            page.wait_for_function(note, arg='This window did not confirm')
            facts = page.evaluate(FACTS)
            assert facts['note'].startswith('This window did not confirm 120%.'), facts
            assert not facts['rowHidden'] and facts['value'] == '120' and facts['shown'] == '120%', facts
            assert page.locator('[data-interface-size-input]').is_enabled(), 'the control stays to try again'
            assert page.locator('[data-interface-size-reset]').is_enabled()
            assert page.evaluate(SENT_SIZE) == {'size': None, 'same': True}, 'an unreported size is no send-time fact'
        setup_browser.capture(page, f'window-race-{race}-{theme}')
        if race == 'size':
            page.evaluate("() => { window.__sets = 'answer'; window.__reads = 'answer'; }")
            _slide_to(page, 120)  # choosing the size again tries again
            page.wait_for_function(note, arg='For this window;')
            assert page.evaluate('window.__sizes') == [120, 120]
            assert page.evaluate(SENT_SIZE) == {'size': 120, 'same': True}
        assert page.locator('#page-settings .settings-scroll').evaluate(
            'e => e.scrollWidth <= e.clientWidth + 1'), 'the actual Settings scroller contains its controls'
        forbidden = {'/api/settings', '/api/update/prepare', '/api/update/apply', '/api/command'}
        assert not any(path in forbidden for path, _ in subscription_ui['posts']), \
            'the window facts must not submit settings or update work'
    finally:
        page.close()


@pytest.mark.parametrize('theme,width,height', [('light', 1280, 860), ('dark', 390, 844)])
@pytest.mark.parametrize('case', ['pending', 'refused', 'older-read'])
def test_a_size_is_the_windows_only_once_the_window_takes_it(subscription_ui, theme, width, height, case):
    """The boundary between a size asked and one the window took, on the real Settings page (recording bridge).

    pending: a held call shows the choice as waiting, the slider and keys stay live, nothing is sent as the window's
    size, and the window's answer makes it its size. refused: PageZoom's 'unavailable' carries the size it stored
    without applying it; that is no fact of the window's, nothing is polled, and a retry the window takes is one.
    older-read: a pywebviewready read answered at 100 % and delivered after 120 % was taken does not bring 100 % back,
    while a read begun later still speaks for the window."""
    import json

    page = seed_page(subscription_ui, width, height, touch=False)
    page.add_init_script("localStorage.setItem('ouroboros.theme', " + json.dumps(theme) + ");")
    page.add_init_script(RACE_BRIDGE.replace('READS', json.dumps('answer')))
    note = "expected => document.querySelector('[data-interface-size-note]').textContent.startsWith(expected)"
    taken = ("percent => document.querySelector('[data-interface-size-value]').textContent === percent + '%'"
             " && document.querySelector('[data-interface-size-note]').textContent.startsWith('For this window;')")
    try:
        page.goto(subscription_ui['url'])
        _appearance(page)
        page.wait_for_function("!document.querySelector('[data-interface-size-row]').hidden")
        assert page.evaluate(SENT_SIZE) == {'size': 100, 'same': True}
        if case == 'pending':
            page.evaluate("() => { window.__sets = 'hold'; }")
            _slide_to(page, 110)
            page.wait_for_function(note, arg='Waiting for this window to confirm 110%.')
            modifier = 'Meta' if page.evaluate("/Mac/.test(navigator.platform)") else 'Control'
            page.keyboard.press(f'{modifier}+Equal')  # the keys step on from the size on its way
            page.wait_for_function(note, arg='Waiting for this window to confirm 120%.')
            facts = page.evaluate(FACTS)
            assert not facts['rowHidden'] and facts['value'] == '120' and facts['shown'] == '120%', facts
            assert page.locator('[data-interface-size-input]').is_enabled(), 'the slider stays live'
            assert page.evaluate('window.__sizes') == [110, 120]
            assert page.evaluate(SENT_SIZE) == {'size': None, 'same': True}, 'a size on its way is no send-time fact'
            setup_browser.capture(page, f'window-size-pending-{theme}')
            page.evaluate("() => { window.__sets = 'answer'; window.__held.splice(0).forEach((call) => call.resolve()); }")
            page.wait_for_function(taken, arg=120)
            assert page.evaluate(SENT_SIZE) == {'size': 120, 'same': True}
        elif case == 'refused':
            reads = page.evaluate('window.__readCount')
            page.evaluate("() => { window.__sets = 'refuse'; }")
            _slide_to(page, 120)
            page.wait_for_function(note, arg='This window did not confirm 120%.')
            assert page.evaluate('window.__answers.at(-1)') == {'ok': False, 'state': 'unavailable', 'percent': 120}
            assert page.evaluate(SENT_SIZE) == {'size': None, 'same': True}, 'a refusal\'s percent is no fact'
            assert page.locator('[data-interface-size-input]').is_enabled(), 'the control stays to try again'
            rest(page)
            settle(page)
            assert page.evaluate('window.__readCount') == reads, 'nothing is polled after a refusal'
            setup_browser.capture(page, f'window-size-refused-{theme}')
            page.evaluate("() => { window.__sets = 'answer'; }")
            _slide_to(page, 120)  # the same choice again
            page.wait_for_function(taken, arg=120)
            assert page.evaluate('window.__answers.at(-1)') == {'ok': True, 'state': 'scheduled', 'percent': 120}
            assert page.evaluate(SENT_SIZE) == {'size': 120, 'same': True}
        else:
            page.evaluate("() => { window.__reads = 'hold'; dispatchEvent(new Event('pywebviewready'));"
                          " window.__reads = 'answer'; }")  # answered now, at 100 %, and delivered late
            _slide_to(page, 120)
            page.wait_for_function(taken, arg=120)
            page.evaluate("() => window.__held.splice(0).forEach((read) => read.resolve())")
            rest(page)
            settle(page)
            facts = page.evaluate(FACTS)
            assert facts['value'] == '120' and facts['note'].startswith('For this window;'), facts
            assert page.evaluate(SENT_SIZE) == {'size': 120, 'same': True}, 'the older 100 % does not come back'
            setup_browser.capture(page, f'window-size-older-read-{theme}')
            page.evaluate("() => { window.__percent = 140; dispatchEvent(new Event('pywebviewready')); }")
            page.wait_for_function(taken, arg=140)  # a read begun later still speaks for the window
            assert page.evaluate(SENT_SIZE) == {'size': 140, 'same': True}
        forbidden = {'/api/settings', '/api/update/prepare', '/api/update/apply', '/api/command'}
        assert not any(path in forbidden for path, _ in subscription_ui['posts']), \
            'the window facts must not submit settings or update work'
    finally:
        page.close()
