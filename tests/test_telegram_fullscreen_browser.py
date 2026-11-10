"""Telegram fullscreen and the screen's edges on the production SPA (web/modules/host_telegram.js; style.css
"Telegram's edges"), in WebKit and Chromium.

The Telegram SDK is a stand-in (tests/_phone_room_shared.py ``telegram_sdk``) answering as Bot API 8.0 documents:
a request is acknowledged later by ``fullscreenChanged`` with ``isFullscreen``, or refused by ``fullscreenFailed``.
Its insets are the test's numbers, not a device's; no Telegram client or iPhone is involved. Uses the
static-server/API fixture; never starts the runtime.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests._phone_room_shared import VIEWPORT, facts, history, shot, telegram_sdk
from tests.test_design_system_gate_browser import rest, seed_page, settle
from tests.test_shell_interactions_browser import open_project, shell_ui
from tests.test_subscription_setup_browser import subscription_ui

pytestmark = [pytest.mark.ui_browser]
__all__ = ['shell_ui', 'subscription_ui']
WEB = Path(__file__).resolve().parents[1] / 'web'


def telegram_page(ui, width, height, sdk, *, stamp=True, script_tag=False):
    """The SPA as the sidecar serves it (`stamp`: the proxy's host attribute), with the SDK stand-in; `script_tag`
    puts the async SDK tag a selected interface carries, without the stamp."""
    page = seed_page(ui, width, height, telegram=False)
    page.add_init_script(sdk)
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    if stamp:
        html = html.replace('<html', '<html data-ouroboros-host="telegram"', 1)
    if script_tag:
        html = html.replace('<head>', '<head><script async src="/telegram-web-app.js"></script>', 1)
        page.route('**/telegram-web-app.js', lambda route: route.fulfill(content_type='text/javascript', body='/* SDK tag */'))
    page.route(ui['url'] + '/', lambda route: route.fulfill(content_type='text/html', body=html))
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['A saved reply in a fullscreen room.'] * 3)))
    return page


def boot(page, ui):
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input', state='attached')
    settle(page)


MODE = "() => document.documentElement.dataset.telegramFullscreen ?? null"
CALLS = "() => window.__tg ? [...window.__tg.calls] : null"
EDGES = """() => {
    const px = (el, prop) => el ? parseFloat(getComputedStyle(el)[prop]) : null;
    const rect = (sel) => { const el = document.querySelector(sel); if (!el || !el.getClientRects().length) return null;
        const r = el.getBoundingClientRect(); return {left: r.left, right: innerWidth - r.right, top: r.top}; };
    return {app: {left: px(document.getElementById('app'), 'marginLeft'), right: px(document.getElementById('app'), 'marginRight')},
        header: px(document.querySelector('#page-chat .app-page-header'), 'paddingTop'),
        sidebarHead: px(document.querySelector('.sidebar-head'), 'paddingTop'),
        composer: px(document.getElementById('chat-input-area'), 'paddingBottom'),
        room: rect('#content'), drawer: rect('#primary-sidebar'), project: rect('.project-panel.open'),
        projectBar: px(document.querySelector('.project-panel-bar'), 'paddingTop'),
        sideways: document.scrollingElement.scrollWidth - innerWidth};
}"""
# Where the closed drawer's painted part begins on screen (its translate and its clip): never beside an inset room.
DRAWER_SHOWS_FROM = r"""() => {
    const s = document.getElementById('primary-sidebar'), r = s.getBoundingClientRect();
    const clip = /^inset\(\S+ \S+ \S+ ([\d.]+)(%|px)\)$/.exec(getComputedStyle(s).clipPath);
    return r.left + (clip ? (clip[2] === '%' ? r.width * clip[1] / 100 : Number(clip[1])) : 0);
}"""


def test_fullscreen_is_asked_once_acknowledged_only_by_telegram_and_an_exit_is_never_undone(shell_ui):
    sdk = telegram_sdk(fullscreen='grant', insets={'top': 47, 'bottom': 34, 'left': 20, 'right': 12}, content={'top': 46})
    page = telegram_page(shell_ui, 390, 844, sdk)
    boot(page, shell_ui)
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'")
    calls = page.evaluate(CALLS)
    assert calls.count('requestFullscreen') == 1 and calls.index('expand') < calls.index('requestFullscreen'), calls
    edges = page.evaluate(EDGES)
    # The stack header clears the device's and Telegram's top (5 px chrome inset + 47 + 46); the sides are the shell's.
    assert edges['app'] == {'left': 20, 'right': 12} and edges['header'] == 98 and edges['sideways'] <= 0, edges
    assert edges['room']['left'] == 20 and edges['room']['right'] == 12, edges
    assert edges['composer'] == 34, 'the composer keeps the bottom inset'
    shot(page, f'telegram-fullscreen-main-{shell_ui["engine"]}')  # before the edge check: a failure keeps its frame
    assert page.evaluate(DRAWER_SHOWS_FROM) >= 19.5, 'the closed drawer never shows beside the inset room'
    page.locator('[data-mobile-nav-toggle]:visible').first.click()
    page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
    rest(page)
    drawer = page.evaluate(EDGES)
    assert drawer['drawer']['left'] == 20, drawer
    shot(page, f'telegram-fullscreen-drawer-{shell_ui["engine"]}')
    page.locator('.nav-drawer-backdrop').click()
    page.wait_for_function("() => window.__ouroLayers.size() === 0")
    rest(page)
    assert page.evaluate(DRAWER_SHOWS_FROM) >= 19.5, 'closed again, it is unseen again'
    open_project(page)
    rest(page)
    project = page.evaluate(EDGES)
    assert project['project']['left'] == 20 and project['project']['right'] == 12, project
    assert page.evaluate(DRAWER_SHOWS_FROM) >= 19.5
    assert page.evaluate("getComputedStyle(document.querySelector('.project-panel')).boxShadow") == 'none', \
        'no shadow of the docked panel falls into the inset'
    shot(page, f'telegram-fullscreen-project-{shell_ui["engine"]}')
    # The keyboard's fixed room keeps the same edges.
    page.locator('textarea.chat-input:visible').focus()
    page.evaluate("() => { const vv = window.visualViewport; Object.defineProperty(vv, 'height', {configurable: true, value: 420}); vv.dispatchEvent(new Event('resize')); }")
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    settle(page)
    keyboard = page.evaluate(EDGES)
    assert keyboard['project']['left'] == 20 and keyboard['project']['right'] == 12 and keyboard['sideways'] <= 0, keyboard
    # The owner leaves fullscreen in Telegram: the page follows and asks nothing again.
    page.evaluate('window.__tg.exit()')
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'expanded'")
    page.wait_for_timeout(300)
    assert page.evaluate(CALLS).count('requestFullscreen') == 1
    facts(f'telegram-fullscreen-edges-{shell_ui["engine"]}', {'main': edges, 'drawer': drawer, 'project': project,
                                                              'keyboard': keyboard, 'calls': page.evaluate(CALLS)})


# Two frames into a close (Opus R1 probe): what the drawer paints and whether its left strip (×, icons) still shows.
CLOSE_START = """async () => {
    const sidebar = document.getElementById('primary-sidebar');
    document.querySelector('.nav-drawer-backdrop').click();
    await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
    return {open: sidebar.classList.contains('open'), clip: getComputedStyle(sidebar).clipPath,
        showsFrom: (""" + DRAWER_SHOWS_FROM + """)()};
}"""


@pytest.mark.parametrize('side', [0, 20], ids=['no-side-inset', 'left-inset'])
def test_a_closing_drawer_keeps_its_whole_face_and_rests_unseen_beside_an_inset_room(shell_ui, side):
    """The closed drawer's clip hides the slice an inset room no longer covers; it waits until the close has run (the
    scrim app.js puts away then), so the drawer going away keeps its ×, icons and version line, with or without an
    inset."""
    sdk = telegram_sdk(fullscreen='grant', insets={'top': 47, 'bottom': 34, 'left': side}, content={'top': 46})
    page = telegram_page(shell_ui, 390, 844, sdk)
    boot(page, shell_ui)
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'")
    page.locator('[data-mobile-nav-toggle]:visible').first.click()
    page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
    rest(page)
    closing = page.evaluate(CLOSE_START)
    name = f'drawer-close-start-{side}-{shell_ui["engine"]}'
    shot(page, name)
    page.wait_for_function("() => window.__ouroLayers.size() === 0")
    rest(page)
    rested = {'clip': page.evaluate("getComputedStyle(document.getElementById('primary-sidebar')).clipPath"),
              'showsFrom': page.evaluate(DRAWER_SHOWS_FROM)}
    facts(name, {'closing': closing, 'rested': rested})
    shot(page, f'drawer-rested-{side}-{shell_ui["engine"]}')
    assert not closing['open'] and closing['clip'] == 'none', closing
    assert rested['showsFrom'] >= side - 0.5, f'at rest it never shows beside the inset room: {rested}'
    page.close()


@pytest.mark.parametrize('case', ['refused', 'old-sdk', 'no-method'])
def test_a_fullscreen_telegram_cannot_give_keeps_the_expanded_shell_and_asks_once_at_most(shell_ui, case):
    sdk = telegram_sdk(fullscreen='absent' if case == 'no-method' else 'refuse' if case == 'refused' else 'grant',
                       version='7.10' if case == 'old-sdk' else '9.1')
    page = telegram_page(shell_ui, 390, 844, sdk)
    boot(page, shell_ui)
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'expanded'")
    page.wait_for_timeout(400)
    calls = page.evaluate(CALLS)
    assert page.evaluate(MODE) == 'expanded', 'a request or a failure is never success'
    assert calls.count('requestFullscreen') == (1 if case == 'refused' else 0), calls
    assert 'expand' in calls and 'ready' in calls
    edges = page.evaluate(EDGES)
    assert edges['app'] == {'left': 0, 'right': 0} and edges['header'] == 5, edges


def test_an_ordinary_browser_without_telegram_is_untouched(shell_ui):
    page = seed_page(shell_ui, 390, 844, telegram=False)
    boot(page, shell_ui)
    page.wait_for_timeout(600)
    assert page.evaluate(MODE) is None
    assert page.evaluate("() => typeof window.Telegram") == 'undefined'
    assert page.evaluate(EDGES)['app'] == {'left': 0, 'right': 0}


PNG = ('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5E'
       'rkJggg==')
# What stands at the screen's top outside the shell on a phone: the document reader's head and an image scene (Places'
# and a skill's sheets are reader dialogs too), the page's notices, and a dialog whose keyboard is up (it stands at the
# top then, M12).
READER = """async ([kind, png]) => {
    const frame = () => new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
    const top = el => el.getBoundingClientRect().top;
    const {createDocumentReader} = await import('/static/modules/document_reader.js');
    window.__reader ||= createDocumentReader({});
    window.__reader.open(kind === 'image' ? {kind, src: png, filename: 'chart.png', meta: 'PNG'}
        : {kind, reader: {base64: btoa('# Notes\\n\\nSome saved text.')}, filename: 'notes.md', meta: 'Markdown'});
    for (let n = 0; n < 60 && kind === 'image' && !document.querySelector('dialog.document-reader .is-image'); n += 1) await frame();
    await frame();
    const sheet = document.querySelector('dialog.document-reader'), body = getComputedStyle(sheet.querySelector('.document-reader-body'));
    return {close: top(sheet.querySelector('.document-reader-close')), title: top(sheet.querySelector('.document-reader-title')),
        bodyTopPad: parseFloat(body.paddingTop), bodyBottomPad: parseFloat(body.paddingBottom)};
}"""
TOAST = """async () => {
    window.__reader?.close();
    const {showToast} = await import('/static/modules/toast.js');
    showToast('A notice over the room', 'info', {ttl: 0});
    await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
    // The stack's own top: a new toast still slides in from 8 px lower for 160 ms (style.css toast-in).
    return {toast: document.getElementById('toast-stack').getBoundingClientRect().top,
        headerBottom: document.querySelector('#page-chat .app-page-header').getBoundingClientRect().bottom};
}"""
DIALOG = """async () => {
    const {openConfirmDialog} = await import('/static/modules/confirm_dialog.js');
    openConfirmDialog({title: 'Rename the file', body: 'A new name for it.', input: true});
    await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
    document.querySelector('.confirm-dialog [data-confirm-input]').focus();
}"""


@pytest.mark.parametrize('host', ['telegram', 'browser'])
def test_overlays_at_the_top_clear_telegrams_fullscreen_band_and_stay_put_in_a_browser(shell_ui, host):
    """Fullscreen puts Telegram's Close and ⋯ over the WebView's top (safe 47 + content 46 here): a reader's Close under
    them would close the Mini App. The shell's own token (ui.css --inset-top) is env() outside Telegram, so an
    ordinary browser keeps every position it had."""
    if host == 'telegram':
        page = telegram_page(shell_ui, 390, 844, telegram_sdk(fullscreen='grant', insets={'top': 47, 'bottom': 34},
                                                              content={'top': 46}))
    else:
        page = seed_page(shell_ui, 390, 844, telegram=False)
    page.add_init_script(VIEWPORT)
    boot(page, shell_ui)
    if host == 'telegram':
        page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'")
    band, bottom = (93, 34) if host == 'telegram' else (0, 0)
    name = f'fullscreen-overlays-{host}-{shell_ui["engine"]}'
    measured = {'reader': page.evaluate(READER, ['markdown', PNG])}
    shot(page, f'{name}-reader')
    measured['image'] = page.evaluate(READER, ['image', PNG])
    shot(page, f'{name}-image')
    measured.update(page.evaluate(TOAST))
    shot(page, f'{name}-toast')
    page.evaluate(DIALOG)
    page.evaluate('window.__viewport(420)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    settle(page)
    box = page.locator('.confirm-dialog').bounding_box()
    measured['dialog'] = {'top': box['y'], 'bottom': box['y'] + box['height']}
    facts(name, measured)
    shot(page, f'{name}-dialog-keyboard', {'x': 0, 'y': 0, 'width': 390, 'height': 420})
    reader, image = measured['reader'], measured['image']
    # The heads clear the band by the padding they always had; the bottoms clear the home indicator.
    assert reader['close'] == reader['title'] == band + 8 and reader['bodyBottomPad'] == 24 + bottom, measured
    assert image['close'] == band + 8 and image['bodyTopPad'] == band + 12 and image['bodyBottomPad'] == 16 + bottom, measured
    assert measured['toast'] == 76 + band and measured['toast'] >= measured['headerBottom'] + 8, measured
    assert measured['dialog']['top'] >= band + 33 and measured['dialog']['bottom'] <= 420, measured
    page.close()


SHEET = """() => { const r = document.querySelector('dialog.document-reader').getBoundingClientRect();
    return {top: r.top, bottom: r.bottom}; }"""


def test_a_docked_telegram_reader_and_dialog_stand_below_the_top_band(shell_ui):
    """A tablet (or a landscape phone) docks: the reader is a centred panel, not a full sheet. In fullscreen it stands
    below Telegram's band (safe 24 + content 46 here) with the panel's own margin, and a dialog and the notices keep
    clear of the band too; nothing else moves (ui.css --inset-top is env() outside Telegram)."""
    page = telegram_page(shell_ui, 1024, 768, telegram_sdk(fullscreen='grant', insets={'top': 24, 'bottom': 20},
                                                           content={'top': 46}))
    boot(page, shell_ui)
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'")
    assert page.locator('html').get_attribute('data-layout') == 'docked'
    band = 70
    name = f'fullscreen-overlays-docked-{shell_ui["engine"]}'
    measured = {'reader': {**page.evaluate(READER, ['markdown', PNG]), **page.evaluate(SHEET)}}
    shot(page, f'{name}-reader')
    measured.update(page.evaluate(TOAST))
    page.evaluate(DIALOG)
    box = page.locator('.confirm-dialog').bounding_box()
    measured['dialog'] = {'top': box['y'], 'bottom': box['y'] + box['height']}
    facts(name, measured)
    shot(page, f'{name}-dialog')
    reader = measured['reader']
    assert reader['top'] == band + 24 and reader['bottom'] == 768 - 24, measured  # centred in what the band leaves
    assert measured['toast'] == 76 + band and measured['dialog']['top'] >= band, measured
    page.close()


@pytest.mark.parametrize('case', ['stamped-late', 'unstamped-telegram', 'unstamped-browser'])
def test_a_late_sdk_binds_only_on_telegram_launch_data(shell_ui, case):
    """The sidecar's stamp waits for a late SDK; a selected interface's own SDK tag waits too, but binds only when
    Telegram's launch data arrives, never because the script exists (an ordinary browser loads it as well)."""
    sdk = telegram_sdk(fullscreen='grant', late_ms=1500, init_data='' if case == 'unstamped-browser' else 'phone-fixture')
    page = telegram_page(shell_ui, 390, 844, sdk, stamp=case == 'stamped-late', script_tag=case != 'stamped-late')
    boot(page, shell_ui)
    page.wait_for_function('() => window.__tg.installedAt != null', timeout=5000)
    if case == 'unstamped-browser':
        page.wait_for_timeout(1500)  # six polls past the SDK's arrival
        assert page.evaluate(MODE) is None
        assert page.evaluate(CALLS) == [], 'an ordinary browser that loads the script gets no Telegram calls'
        return
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'", timeout=5000)
    timing = page.evaluate('() => ({installed: window.__tg.installedAt, bound: window.__tg.boundAt})')
    assert timing['bound'] >= timing['installed'], f'nothing binds before the SDK is there: {timing}'
    calls = page.evaluate(CALLS)
    assert calls.count('requestFullscreen') == 1 and calls.count('ready') == 1, calls


@pytest.mark.parametrize('width,height', [(844, 390), (1024, 768)], ids=['phone-landscape', 'tablet'])
def test_a_docked_telegram_shell_keeps_its_sides_and_puts_the_top_inside_its_bars(shell_ui, width, height):
    insets = {'left': 47, 'right': 47, 'bottom': 21} if width == 844 else {'top': 24, 'bottom': 20}
    content = {} if width == 844 else {'top': 46}
    page = telegram_page(shell_ui, width, height, telegram_sdk(fullscreen='grant', insets=insets, content=content))
    boot(page, shell_ui)
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'")
    assert page.locator('html').get_attribute('data-layout') == 'docked'
    edges = page.evaluate(EDGES)
    top = insets.get('top', 0) + content.get('top', 0)
    assert edges['app'] == {'left': insets.get('left', 0), 'right': insets.get('right', 0)}, edges
    assert edges['header'] == 10 + top and edges['sidebarHead'] == 16 + top and edges['sideways'] <= 0, edges
    open_project(page)
    rest(page)
    edges['projectOpen'] = page.evaluate(EDGES)
    assert edges['projectOpen']['projectBar'] == 10 + top, edges
    facts(f'telegram-docked-edges-{width}x{height}-{shell_ui["engine"]}', edges)
    shot(page, f'telegram-docked-{width}x{height}-{shell_ui["engine"]}')


SIDE_TOAST = """async () => {
    const {showToast} = await import('/static/modules/toast.js');
    showToast('A notice long enough to take the widest a page notice may be, so both of its edges are the ones its '
        + 'place and its width keep from the screen and from Telegram\\'s sides.', 'info', {ttl: 0});
    await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
    const r = document.querySelector('#toast-stack .toast').getBoundingClientRect();
    return {left: r.left, right: innerWidth - r.right, width: r.width};
}"""


@pytest.mark.parametrize('width,side', [(844, 47), (480, 59)], ids=['landscape', 'narrow'])
def test_page_notices_keep_their_margin_inside_telegrams_side_insets(shell_ui, width, side):
    """Astra R2: the page's notices kept 18 px from the screen's right edge, inside a fullscreen Telegram's side inset
    (a landscape notch). They keep it from the inset, and a notice as wide as it may be keeps its left edge out of the
    other inset too (the narrow window: a right-only shift would push it 35 px into the left one)."""
    page = telegram_page(shell_ui, width, 390, telegram_sdk(fullscreen='grant',
                                                            insets={'left': side, 'right': side, 'bottom': 21}))
    boot(page, shell_ui)
    page.wait_for_function("() => document.documentElement.dataset.telegramFullscreen === 'active'")
    measured = page.evaluate(SIDE_TOAST)
    name = f'notice-side-insets-{width}-{shell_ui["engine"]}'
    facts(name, measured)
    shot(page, name)
    assert measured['right'] >= side + 18 - 0.5 and measured['left'] >= side - 0.5, measured
    page.close()
