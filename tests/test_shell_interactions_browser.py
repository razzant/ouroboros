"""Phone shell interactions: layer inert, Back, Escape, focus, the drawer drag, inline links, a short phone.

Each case turns a CR1 reviewer probe around: it asserts the fixed behaviour and, where a guard is involved, that
the guard still fires. Uses the static-server/API fixture (test_subscription_setup_browser); never starts the
runtime or reads live data. Synthetic CDP touch is Chromium's input pipeline, not a claim about an iPhone WebView.
"""
from __future__ import annotations

import json

import pytest

from ouroboros.launcher_window_chrome import DOCUMENT_SCRIPT
from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import (PROJECTS, open_nav, rest, save_evidence, seed_page, settle,
                                                   sidebar_metrics)
from tests._work_projection_shared import work_fields
from tests.ui_motion_samples import SAMPLE_TRANSITION

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(params=['webkit', 'chromium'])
def shell_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


@pytest.fixture
def chromium_ui(request, monkeypatch):
    # CDP Input.dispatchTouchEvent exists only in Chromium; WebKit has no equivalent, so these cases run there only.
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', 'chromium')
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = 'chromium'
    return ui


def evidence(tmp_path, name, facts):
    save_evidence(tmp_path, name, facts)
    print(name, json.dumps(facts, indent=1))


def boot(page, ui):
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input', state='attached')
    page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
    settle(page)


def ids(page):
    return page.evaluate('window.__ouroLayers.ids()')


def wait_quietly(page, expression, timeout=3_000):
    """Wait for a condition; a defect that never meets it is recorded by the facts that follow, not a timeout."""
    try:
        page.wait_for_function(expression, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 - the facts read next are the observation
        return False


def open_project(page, index=0):
    page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))",
                  PROJECTS[index])
    page.wait_for_function("() => window.__ouroLayers.top() === 'project'")
    page.wait_for_timeout(300)


def open_drawer(page):
    page.locator('[data-mobile-nav-toggle]:visible').first.click()
    page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
    page.wait_for_timeout(250)


def open_confirm(page):
    page.evaluate("""async () => {
        const m = await import('/static/modules/confirm_dialog.js');
        window.__confirmResult = 'pending';
        m.openConfirmDialog({title: 'Stop this work?', body: 'shell fixture', danger: true})
            .then((r) => { window.__confirmResult = JSON.stringify(r); });
    }""")
    page.wait_for_selector('.confirm-dialog-backdrop')


def open_reader(page):
    page.evaluate("""async () => {
        const m = await import('/static/modules/document_reader.js');
        window.__reader = m.createDocumentReader({});
        window.__reader.open({filename: 'notes.md', kind: 'markdown', meta: '1 KB',
                              reader: {url: '/api/files/read?path=notes.md'}});
    }""")
    page.wait_for_selector('dialog.document-reader[open]')
    page.wait_for_timeout(300)


def sync_project_rows(page):
    """The next /api/state snapshot changes one row (Project 5 has three new messages). The 20 s poll would
    deliver it; the projects_changed frame (a backend-created project) asks for the same snapshot now."""
    rows = [dict(p) for p in PROJECTS]
    rows[5]['visible_revision'] = 3
    state = {'supervisor_ready': True, 'active_chat_activities': [], 'projects': rows,
             'project_chat_ids': [p['chat_id'] for p in rows]}
    page.route('**/api/state', lambda route: route.fulfill(json=state))
    page.evaluate("window.__ouroWs.emit('projects_changed', {})")
    page.wait_for_function("() => document.querySelector('#nav-projects-count')?.textContent === '3'")
    page.wait_for_timeout(300)


SURFACES = {
    'confirm': {'surface': '.confirm-dialog-backdrop',
                'control': '.confirm-dialog .marketplace-modal-actions [data-confirm-cancel]'},
    'reader': {'surface': 'dialog.document-reader', 'control': 'dialog.document-reader [data-reader-action="close"]'},
}
OVER_ROOM = """sel => {
    const s = document.querySelector(sel.surface), c = document.querySelector(sel.control);
    const room = {contentInert: Boolean(document.getElementById('content').closest('[inert]')),
                  panelInert: Boolean(document.getElementById('project-panel').closest('[inert]')),
                  layers: window.__ouroLayers.ids()};
    if (!s || !c) return {...room, open: false};
    const r = c.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return {...room, open: true, inert: Boolean(s.closest('[inert]')),
            hitIsControl: Boolean(hit) && (hit === c || c.contains(hit)),
            hit: hit ? hit.tagName + '.' + [...hit.classList].join('.') : null};
}"""


@pytest.mark.parametrize('surface', list(SURFACES))
def test_a_dialog_over_a_phone_project_stays_live_after_a_state_sync(shell_ui, tmp_path, surface):
    """Layer inert stops at the app shell: a body-level dialog survives the row change; the room stays inert."""
    page = seed_page(shell_ui, 390, 844)
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        open_project(page)
        open_confirm(page) if surface == 'confirm' else open_reader(page)
        facts['before_sync'] = page.evaluate(OVER_ROOM, SURFACES[surface])
        sync_project_rows(page)
        facts['after_sync'] = page.evaluate(OVER_ROOM, SURFACES[surface])
        try:
            page.locator(SURFACES[surface]['control']).click(timeout=2_500)
            facts['click'] = 'clicked'
        except Exception as exc:  # noqa: BLE001 - the observation is the point
            facts['click'] = f'failed: {type(exc).__name__}'
        page.wait_for_timeout(300)
        facts['after_click'] = page.evaluate(OVER_ROOM, SURFACES[surface])
        facts['result'] = page.evaluate('window.__confirmResult ?? null')
    finally:
        evidence(tmp_path, f'shell-dialog-after-sync-{surface}-{shell_ui["engine"]}', facts)
        page.close()
    for phase in ('before_sync', 'after_sync'):
        assert facts[phase]['open'] and not facts[phase]['inert'] and facts[phase]['hitIsControl'], facts
    # The guard still fires: the room outside the open Project layer stays inert, the panel stays live.
    assert facts['after_sync']['contentInert'] and not facts['after_sync']['panelInert'], facts
    assert facts['click'] == 'clicked' and not facts['after_click']['open'], facts
    assert facts['after_click']['layers'] == ['project'] and facts['after_click']['contentInert'], facts
    if surface == 'confirm':
        assert facts['result'] == 'false', facts


# A Back that leaves the app lands on about:blank: the facts say so instead of throwing.
BACK_FACTS = """() => ({layers: window.__ouroLayers?.ids() ?? 'left the app',
    drawer: Boolean(document.getElementById('primary-sidebar')?.classList.contains('open')),
    dialog: Boolean(document.querySelector('.new-project-dialog, .confirm-dialog')),
    result: window.__confirmResult ?? null})"""


def press_back(page, via):
    page.evaluate('Telegram.WebApp.BackButton.callback()' if via == 'telegram' else 'history.back()')


@pytest.mark.parametrize('via', ['history', 'telegram'])
@pytest.mark.parametrize('case', ['new-project-over-drawer', 'confirm-over-project'])
def test_back_closes_the_topmost_dialog_before_its_layer(shell_ui, tmp_path, case, via):
    layer = 'drawer' if case == 'new-project-over-drawer' else 'project'
    page = seed_page(shell_ui, 390, 844, telegram=via == 'telegram')
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        if layer == 'drawer':
            open_drawer(page)
            page.locator('#nav-projects-add').click()
            page.locator('[data-np-name]').fill('Retained draft')
        else:
            open_project(page)
            open_confirm(page)
        wait_quietly(page, 'window.__ouroLayers?.historyArmed()')
        facts['open'] = page.evaluate(BACK_FACTS)
        press_back(page, via)
        wait_quietly(page, "() => !document.querySelector('.new-project-dialog, .confirm-dialog') "
                           "|| !window.__ouroLayers?.size()")
        page.wait_for_timeout(200)
        facts['first_back'] = page.evaluate(BACK_FACTS)
        wait_quietly(page, 'window.__ouroLayers?.historyArmed()')
        press_back(page, via)
        wait_quietly(page, '() => !window.__ouroLayers?.size()')
        page.wait_for_timeout(200)
        facts['second_back'] = page.evaluate(BACK_FACTS)
    finally:
        evidence(tmp_path, f'shell-back-{case}-{via}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['open']['layers'] == [layer] and facts['open']['dialog'], facts
    assert facts['first_back']['layers'] == [layer] and not facts['first_back']['dialog'], facts
    assert facts['first_back']['drawer'] is (layer == 'drawer'), facts
    if layer == 'project':
        assert facts['first_back']['result'] == 'false', facts
    assert facts['second_back']['layers'] == [] and not facts['second_back']['drawer'], facts


FILES_BACK_FACTS = """() => {
    const shown = el => Boolean(el && el.getClientRects().length);
    const active = document.activeElement;
    return {layers: window.__ouroLayers?.ids() ?? 'left the app', armed: window.__ouroLayers?.historyArmed() ?? null,
        view: document.querySelector('.files-layout')?.dataset.filesView ?? null,
        list: shown(document.querySelector('.files-sidebar')), file: shown(document.querySelector('.files-preview')),
        editor: document.querySelector('.files-editor')?.value ?? null,
        unsaved: [...document.querySelectorAll('.files-entry')].filter(row => shown(row.querySelector('.files-entry-unsaved')))
            .map(row => row.dataset.path || ''),
        focus: active?.closest?.('.files-entry')?.dataset.path ?? (active?.matches?.('[data-files-back]') ? 'back' : active?.tagName),
        telegramBack: window.Telegram?.WebApp?.BackButton?.isVisible ?? null};
}"""


@pytest.mark.parametrize('via', ['history', 'telegram'])
def test_back_returns_a_narrow_files_page_from_its_file_to_the_list(shell_ui, tmp_path, via):
    """#7: a file shown alone on a phone is one dismissable layer, so the system Back and Telegram's Back close it the
    way Back to folder does: the list, focus on the file's row, its unsaved draft kept for that row. Escape in the
    editor closes it the same way; the desk split registers no layer, and a page narrow again registers it again."""
    page = seed_page(shell_ui, 390, 844, telegram=via == 'telegram')
    page.route('**/api/files/read*', lambda route: route.fulfill(json={
        'path': 'notes.md', 'name': 'notes.md', 'size': 11, 'display_path': '/workspace/notes.md',
        'is_text': True, 'content': 'Saved text\n', 'truncated': False}))
    facts = {'engine': shell_ui['engine'], 'via': via}
    try:
        boot(page, shell_ui)
        open_nav(page, 'files')
        row = page.locator('.files-entry', has_text='project_notes_with_a_long_descriptive_name.md')
        facts['list'] = page.evaluate(FILES_BACK_FACTS)
        row.click()
        page.locator('.files-editor').wait_for()
        wait_quietly(page, 'window.__ouroLayers?.historyArmed()')
        facts['open'] = page.evaluate(FILES_BACK_FACTS)
        page.locator('.files-editor').fill('Draft kept by Back')
        press_back(page, via)
        wait_quietly(page, "() => document.querySelector('.files-layout').dataset.filesView === 'list'")
        page.wait_for_timeout(200)
        facts['back'] = page.evaluate(FILES_BACK_FACTS)
        row.click()
        wait_quietly(page, 'window.__ouroLayers?.historyArmed()')
        facts['reopened'] = page.evaluate(FILES_BACK_FACTS)
        page.locator('.files-editor').press('Escape')
        wait_quietly(page, "() => document.querySelector('.files-layout').dataset.filesView === 'list'")
        page.wait_for_timeout(200)
        facts['escape'] = page.evaluate(FILES_BACK_FACTS)
        row.click()
        page.set_viewport_size({'width': 1280, 'height': 860})
        wait_quietly(page, '() => !window.__ouroLayers?.size()')
        page.wait_for_timeout(200)
        facts['desk'] = page.evaluate(FILES_BACK_FACTS)
        page.set_viewport_size({'width': 390, 'height': 844})
        wait_quietly(page, '() => window.__ouroLayers?.size() === 1')
        page.wait_for_timeout(200)
        facts['narrow_again'] = page.evaluate(FILES_BACK_FACTS)
    finally:
        evidence(tmp_path, f'shell-files-back-{via}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['list']['layers'] == [] and facts['list']['view'] == 'list', facts
    opened = facts['open']
    assert opened['layers'] == ['files-file'] and opened['armed'] and opened['file'] and not opened['list'], facts
    assert opened['focus'] == 'back', facts
    back = facts['back']
    assert back['layers'] == [] and back['view'] == 'list' and back['list'] and not back['file'], facts
    assert back['focus'] == 'notes.md' and back['unsaved'] == ['notes.md'], facts
    assert back['editor'] == 'Draft kept by Back', facts
    assert facts['reopened']['layers'] == ['files-file'] and facts['reopened']['editor'] == 'Draft kept by Back', facts
    escape = facts['escape']
    assert escape['layers'] == [] and escape['view'] == 'list' and escape['focus'] == 'notes.md', facts
    assert escape['editor'] == 'Draft kept by Back' and escape['unsaved'] == ['notes.md'], facts
    assert facts['desk']['layers'] == [] and facts['desk']['list'] and facts['desk']['file'], facts
    assert facts['narrow_again']['layers'] == ['files-file'] and facts['narrow_again']['view'] == 'file', facts
    if via == 'telegram':
        assert (opened['telegramBack'], back['telegramBack'], facts['desk']['telegramBack']) == (True, False, False), facts


@pytest.mark.parametrize('width,height', [(1280, 860), (390, 844)])
def test_escape_in_a_composer_capsule_closes_only_the_capsule(shell_ui, tmp_path, width, height):
    page = seed_page(shell_ui, width, height)
    facts = {'engine': shell_ui['engine'], 'viewport': [width, height]}
    try:
        boot(page, shell_ui)
        open_project(page)
        head = page.locator('#project-panel .chat-capsule-head').first
        head.click()
        page.wait_for_timeout(250)
        facts['capsule_before'] = head.get_attribute('aria-expanded')
        page.keyboard.press('Escape')
        page.wait_for_timeout(400)
        facts['capsule_after'] = head.get_attribute('aria-expanded') if head.count() else 'gone'
        facts['layers_after'] = ids(page)
        # Nothing inside takes the next Escape: it is the room's again (the guard is not a swallow-all).
        page.keyboard.press('Escape')
        page.wait_for_timeout(400)
        facts['layers_second'] = ids(page)
    finally:
        evidence(tmp_path, f'shell-escape-capsule-{shell_ui["engine"]}-{width}', facts)
        page.close()
    assert facts['capsule_before'] == 'true', facts
    assert facts['capsule_after'] == 'false' and facts['layers_after'] == ['project'], facts
    assert facts['layers_second'] == [], facts


WHO = "(() => { const a = document.activeElement; return a ? a.tagName + '#' + a.id + '.' + [...a.classList].slice(0, 3).join('.') : null; })()"


@pytest.mark.parametrize('how', ['enter', 'tap'])
def test_a_project_opened_from_the_phone_drawer_takes_focus(shell_ui, tmp_path, how):
    page = seed_page(shell_ui, 390, 844)
    facts = {'engine': shell_ui['engine'], 'how': how}
    def press(locator):  # a keyboard user or a finger
        if how == 'tap':
            return locator.tap()
        locator.focus()
        page.keyboard.press('Enter')

    try:
        boot(page, shell_ui)
        press(page.locator('[data-mobile-nav-toggle]:visible').first)
        page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
        page.wait_for_timeout(250)
        press(page.locator('#nav-projects-list [data-project-id]').first)
        page.wait_for_function("() => window.__ouroLayers.top() === 'project'")
        # The panel lay in the room the drawer made inert: focus goes in once the card has come home.
        rest(page)
        facts['focus'] = page.evaluate(WHO)
        facts['inside_panel'] = page.evaluate("document.getElementById('project-panel').contains(document.activeElement)")
        facts['layers'] = ids(page)
    finally:
        evidence(tmp_path, f'shell-focus-{how}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['layers'] == ['project'] and facts['inside_panel'] is True, facts


def swipe(page, start, end, steps=12):
    cdp = page.context.new_cdp_session(page)
    try:
        for i in range(steps + 1):
            point = {'x': start[0] + (end[0] - start[0]) * i / steps,
                     'y': start[1] + (end[1] - start[1]) * i / steps, 'id': 1}
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove', 'touchPoints': [point]})
            page.wait_for_timeout(16)
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        page.wait_for_timeout(450)
    finally:
        cdp.detach()


DRAWER = """() => ({open: document.getElementById('primary-sidebar').classList.contains('open'),
                    layers: window.__ouroLayers.ids()})"""
UNDER = "([x, y]) => { const e = document.elementFromPoint(x, y); return e ? e.tagName + '#' + e.id + '.' + e.className : null; }"


@pytest.mark.parametrize('room', ['main', 'project'])
def test_the_drawer_follows_a_finger_on_a_phone_room(chromium_ui, tmp_path, room):
    page = seed_page(chromium_ui, 390, 844)
    under = [] if room == 'main' else ['project']  # the drawer opens over an open Project room
    facts = {'room': room}
    try:
        boot(page, chromium_ui)
        if room == 'project':
            open_project(page)
        facts['coarse'] = page.evaluate("matchMedia('(pointer: coarse)').matches")
        box = page.locator('#chat-messages' if room == 'main' else '#project-panel-body').bounding_box()
        y = box['y'] + box['height'] / 2
        facts['under'] = page.evaluate(UNDER, [60, y])
        for name, start, end in [('vertical_up', (200, y + 120), (200, y - 120)),
                                 ('vertical_down', (200, y - 120), (200, y + 120))]:
            swipe(page, start, end)
            facts[name] = page.evaluate(DRAWER)
            if facts[name]['open']:
                page.evaluate('window.__ouroLayers.pop()')  # keep the next case independent
                page.wait_for_timeout(300)
        swipe(page, (60, y), (330, y))
        facts['open_drag'] = page.evaluate(DRAWER)
        if facts['open_drag']['open']:
            swipe(page, (300, y), (20, y))  # starts on the open drawer
            facts['close_drag'] = page.evaluate(DRAWER)
            swipe(page, (60, y), (330, y))
            swipe(page, (370, y), (60, y))  # starts on the scrim's visible strip
            facts['scrim_close_drag'] = page.evaluate(DRAWER)
    finally:
        evidence(tmp_path, f'shell-drawer-drag-{room}', facts)
        page.close()
    assert facts['coarse'] is True, facts
    # The guard: a vertical finger scrolls the feed and never opens the drawer, in either direction.
    assert facts['vertical_up'] == facts['vertical_down'] == {'open': False, 'layers': under}, facts
    assert facts['open_drag'] == {'open': True, 'layers': under + ['drawer']}, facts
    assert facts['close_drag'] == {'open': False, 'layers': under}, facts
    assert facts['scrim_close_drag'] == {'open': False, 'layers': under}, facts


WORD = """bar => {
    const t = document.querySelector(bar + ' .rw-trigger:not([hidden])');
    const b = t?.querySelector('.status-badge');
    if (!t || !b) return null;
    const shown = getComputedStyle(b).display !== 'none';
    return {shown, cut: shown && b.scrollWidth > b.clientWidth + 1, word: t.dataset.word || '', title: t.title,
            mark: [...t.querySelectorAll('.status-mark')].some((m) => getComputedStyle(m).display !== 'none')};
}"""


@pytest.mark.parametrize('room,width', [('main', 390), ('project', 390), ('main', 1280)])
def test_the_status_word_is_whole_or_absent(shell_ui, tmp_path, room, width):
    page = seed_page(shell_ui, width, 844)
    facts = {'engine': shell_ui['engine'], 'room': room, 'width': width}
    try:
        boot(page, shell_ui)
        if room == 'project':
            open_project(page)  # the fixture's long Project name leaves the word no room on a phone
        page.wait_for_timeout(300)
        facts['word'] = page.evaluate(WORD, '#project-panel .project-panel-bar' if room == 'project' else '.chat-page-header')
    finally:
        evidence(tmp_path, f'shell-status-word-{room}-{width}-{shell_ui["engine"]}', facts)
        page.close()
    word = facts['word']
    assert word and word['mark'] and word['title'], facts
    assert word['cut'] is False, facts  # never a fragment such as "Wor…"
    if width >= 720:
        assert word['shown'] is True, facts  # a bar with room keeps the word


INLINE_LINK = """() => {
    const a = document.querySelector('.settings-section-copy a[href="https://clawhub.ai"]');
    const range = document.createRange();
    range.selectNodeContents(a.parentElement);
    const tops = [...range.getClientRects()].filter((q) => q.width && q.height).map((q) => q.top).sort((x, y) => x - y);
    const lines = tops.reduce((acc, t) => (acc.length && t - acc[acc.length - 1] < 3 ? acc : [...acc, t]), []);
    return {display: getComputedStyle(a).display, height: Math.round(a.getBoundingClientRect().height),
            lines: lines.length, gaps: lines.slice(1).map((t, i) => Math.round((t - lines[i]) * 10) / 10)};
}"""
TARGET = """sel => {
    const e = document.querySelector(sel);
    if (!e) return null;
    const r = e.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const hit = [[cx - 21, cy], [cx + 21, cy], [cx, cy - 21], [cx, cy + 21]].every(([x, y]) => {
        const t = document.elementFromPoint(x, y);
        return Boolean(t) && (t === e || e.contains(t));
    });
    return {width: Math.round(r.width * 10) / 10, height: Math.round(r.height * 10) / 10, hit,
            display: getComputedStyle(e).display};
}"""


def open_settings_panel_of(page, selector):
    panel = page.evaluate("s => document.querySelector(s)?.closest('[data-settings-panel]')?.dataset.settingsPanel || ''",
                          selector)
    page.locator(f'[data-settings-tab="{panel}"]').first.click()
    page.locator(selector).first.scroll_into_view_if_needed()
    settle(page)
    return panel


def test_an_inline_link_in_settings_copy_reads_as_text_while_a_standalone_link_keeps_a_finger_target(shell_ui, tmp_path):
    page = seed_page(shell_ui, 390, 844)
    facts = {'engine': shell_ui['engine']}
    standalone = '.about-links a[href]'
    try:
        boot(page, shell_ui)
        facts['coarse'] = page.evaluate("matchMedia('(pointer: coarse)').matches")
        open_drawer(page)
        page.locator('[data-nav-page="settings"]').click()
        page.locator('#page-settings').wait_for(state='visible')
        facts['inline_panel'] = open_settings_panel_of(page, '.settings-section-copy a[href="https://clawhub.ai"]')
        facts['inline'] = page.evaluate(INLINE_LINK)
        facts['standalone_panel'] = open_settings_panel_of(page, standalone)
        facts['standalone'] = page.evaluate(TARGET, standalone)
    finally:
        evidence(tmp_path, f'shell-inline-link-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['coarse'] is True, facts
    inline = facts['inline']
    assert inline['display'] == 'inline', facts
    assert inline['lines'] >= 3 and max(inline['gaps']) - min(inline['gaps']) <= 2, facts
    target = facts['standalone']
    assert target and (target['hit'] or (target['width'] >= 43.9 and target['height'] >= 43.9)), facts


def test_a_short_phone_drawer_has_one_scroller_the_whole_column(shell_ui, tmp_path):
    page = seed_page(shell_ui, 390, 570)  # the 30-Project fixture of the gate
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        facts['coarse'] = page.evaluate("matchMedia('(pointer: coarse)').matches")
        facts['projects'] = page.locator('#nav-projects-list [data-project-id]').count()
        facts['scrollers'] = sidebar_metrics(page)  # opens the drawer
        facts['layers'] = ids(page)
    finally:
        evidence(tmp_path, f'shell-short-phone-sidebar-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['coarse'] is True and facts['projects'] == len(PROJECTS) and facts['layers'] == ['drawer'], facts
    assert [row['node'] for row in facts['scrollers']] == ['primary-sidebar'], facts


def saved_history(tmp_path):
    """The real history endpoint over two saved rows, so the feed draws bubbles with Copy and Reply."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import dialogue_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    root = tmp_path / 'history'
    root.mkdir()
    for minute, (direction, role, text) in enumerate([('in', 'user', 'Please check the build before the release.'),
                                                       ('out', 'assistant', 'The build is green; nothing to fix.')]):
        dialogue_store.append(root, 'chat', {'chat_id': 1, 'user_id': 1, 'direction': direction, 'role': role,
                                             'text': text, 'ts': f'2026-10-18T10:0{minute}:00Z'})
    client = TestClient(Starlette(routes=[Route('/api/chat/history', make_chat_history_endpoint(root))]))

    def respond(route):
        response = client.get('/' + route.request.url.split('/', 3)[3])
        route.fulfill(status=response.status_code, content_type='application/json', body=response.content)
    return client, respond


ACTION_ROWS = """() => [...document.querySelectorAll('#chat-messages .chat-message-actions')]
    .filter((row) => row.offsetParent && row.children.length >= 2)
    .map((row) => [...row.children].filter((c) => c.offsetParent).map((c) => {
        const r = c.getBoundingClientRect();
        return {name: c.className, top: Math.round(r.top * 10) / 10, bottom: Math.round(r.bottom * 10) / 10};
    }))"""


@pytest.mark.parametrize('width', [1280, 390])
def test_copy_and_reply_sit_side_by_side_on_one_line(shell_ui, tmp_path, width):
    page = seed_page(shell_ui, width, 860)
    client, respond = saved_history(tmp_path)
    facts = {'engine': shell_ui['engine'], 'width': width}
    try:
        page.route('**/api/chat/history**', respond)
        boot(page, shell_ui)
        page.wait_for_selector('#chat-messages .chat-message-actions .chat-message-copy', timeout=20_000)
        facts['rows'] = page.evaluate(ACTION_ROWS)
    finally:
        evidence(tmp_path, f'shell-copy-reply-{width}-{shell_ui["engine"]}', facts)
        page.close()
        client.close()
    assert facts['rows'], facts
    for row in facts['rows']:
        assert len(row) >= 2 and any('copy' in c['name'] for c in row) and any('reply' in c['name'] for c in row), facts
        assert max(c['top'] for c in row) - min(c['top'] for c in row) <= 0.5, facts
        assert max(c['bottom'] for c in row) - min(c['bottom'] for c in row) <= 0.5, facts


TITLE_BAND = """({band, allowed}) => {
    // Exercise the launcher's real CONTROLS/cursor exclusion, not a copied predicate.
    const left = document.getElementById('primary-sidebar').getBoundingClientRect().right;
    const controls = new Set(), unexpected = [];
    let empty = null, drags = 0;
    for (let y = 4; y < band; y += 6) {
        for (let x = left + 4; x < innerWidth - 8; x += 12) {
            const t = document.elementFromPoint(x, y);
            const control = t?.closest(allowed);
            window.__chromeMessages.length = 0;
            const press = new MouseEvent('mousedown', {bubbles: true, cancelable: true, clientX: x, clientY: y, detail: 1});
            t.dispatchEvent(press);
            const messages = window.__chromeMessages.map(m => m.type);
            if (control) {
                controls.add(control.id || control.textContent.trim());
                if (messages.length || press.defaultPrevented) unexpected.push({x, y, messages, control: control.outerHTML});
            } else if (messages.join() === 'drag' && press.defaultPrevented) {
                drags++;
                empty ||= {x, y};
            } else unexpected.push({x, y, messages, target: t.outerHTML.slice(0, 200)});
        }
    }
    const tab = document.querySelector('.page.active .app-tab');
    const header = document.querySelector('.page.active .app-page-header');
    return {controls: [...controls], unexpected, empty, drags, height: header.getBoundingClientRect().height,
        firstTabTop: tab ? Math.round(tab.getBoundingClientRect().top) : null};
}"""


@pytest.mark.parametrize('page_name', ['dashboard', 'settings', 'skills', 'files', 'widgets'])
def test_the_desktop_title_band_over_a_page_moves_the_window(shell_ui, tmp_path, page_name):
    """Composition A: only intended navigation/actions occupy the band; plain space still requests native drag."""
    page = seed_page(shell_ui, 1280, 860)
    # The production document-start script and its messages, with only the native bridge replaced by a recorder.
    # AppKit's window movement itself is covered separately; no owner window is used here.
    page.add_init_script("""window.__chromeMessages = [];
        window.webkit = {messageHandlers: {ouroborosWindowChrome: {postMessage: m => window.__chromeMessages.push(m)}}};
    """ + DOCUMENT_SCRIPT)
    facts = {'engine': shell_ui['engine'], 'page': page_name}
    try:
        boot(page, shell_ui)
        page.evaluate("window.__ouroWindowChrome({doc: performance.timeOrigin, top: 52, start: 78})")
        page.locator(f'[data-nav-page="{page_name}"]').click()
        page.locator(f'#page-{page_name}').wait_for(state='visible')
        settle(page)
        allowed = ('.app-page-lead .app-tab, .app-page-lead .files-crumb, '
                   '.app-page-toolbar :is(#skills-refresh, #skills-update-all, #files-refresh, '
                   '#files-new-file, #files-new-dir, #files-paste)')
        facts['band'] = page.evaluate(TITLE_BAND, {'band': 52, 'allowed': allowed})
        band = facts['band']
        assert band['height'] == 52 and band['drags'] > 0 and not band['unexpected'], facts
        if page_name == 'widgets':
            assert band['controls'] == [] and band['firstTabTop'] is None, facts
        else:
            assert band['controls'], facts
            if page_name != 'files':
                assert 0 <= band['firstTabTop'] < 52, facts
        # Real pointer presses on empty band: one drag, then the Mac's double-click request.
        page.evaluate('window.__chromeMessages.length = 0')
        page.mouse.click(**band['empty'])
        assert page.evaluate('window.__chromeMessages.map(m => m.type)') == ['drag']
        page.evaluate('window.__chromeMessages.length = 0')
        page.mouse.dblclick(**band['empty'])
        assert page.evaluate('window.__chromeMessages.map(m => m.type)') == ['drag', 'double-click']
        # Controls keep their normal behaviour and never send a native-window request.
        page.evaluate('window.__chromeMessages.length = 0')
        if page_name == 'files':
            page.locator('#files-new-file').click()
            page.locator('#page-files .files-editor').wait_for()
        elif page_name != 'widgets':
            tab = page.locator(f'#page-{page_name} .app-page-lead .app-tab').nth(1)
            tab.click()
            assert tab.get_attribute('aria-selected') == 'true'
        assert page.evaluate('window.__chromeMessages.map(m => m.type)') == []
    finally:
        setup_browser.capture(page, f'shell-title-band-{page_name}-{shell_ui["engine"]}')
        evidence(tmp_path, f'shell-title-band-{page_name}-{shell_ui["engine"]}', facts)
        page.close()


def live_state(project_ids, pins=()):
    """The /api/state snapshot with work running in these Projects: each census row in its Project's room and the
    server's own work projection of the rows, which the sidebar's marks read; ``pins`` are pinned Projects."""
    chats = {p['id']: p['chat_id'] for p in PROJECTS}
    rows = [{'activity_id': f'work-{pid}', 'root_task_id': f'work-{pid}', 'project_id': pid, 'chat_id': chats[pid],
             'phase': 'working'} for pid in project_ids]
    return {'supervisor_ready': True, 'projects': [dict(p, pinned=p['id'] in pins) for p in PROJECTS],
            'project_chat_ids': [p['chat_id'] for p in PROJECTS],
            'active_chat_activities_complete': True, 'active_chat_activities': rows, **work_fields(rows)}


MOVING = "n => document.querySelectorAll('#nav-projects-list .status-mark[data-shape=\"moving\"]').length === n"
SIDEBAR_FACTS = """() => {
    const side = document.getElementById('primary-sidebar');
    const rows = [...side.querySelectorAll('#nav-projects-list > .nav-project-item')].filter((e) => !e.hidden);
    const zone = document.getElementById('nav-projects').getBoundingClientRect();
    const files = side.querySelector('[data-nav-page="files"]').getBoundingClientRect();
    const mark = document.querySelector('.nav-projects-facts .status-mark');
    return {rows: rows.map((e) => e.querySelector('[data-project-id]').dataset.projectId),
            gapToTools: Math.round(files.top - zone.bottom), filesTop: Math.round(files.top),
            sidebarBottom: Math.round(side.getBoundingClientRect().bottom),
            // The foot: how far the version row's bottom sits above the column's content edge (0 = on the edge), and
            // whether the column itself scrolls (the short-window fallback, where the foot is pinned over it).
            footGap: Math.round(side.getBoundingClientRect().bottom - parseFloat(getComputedStyle(side).paddingBottom)
                - side.querySelector('.sidebar-foot-meta').getBoundingClientRect().bottom),
            footBelowTools: side.querySelector('.sidebar-foot-meta').getBoundingClientRect().top
                >= side.querySelector('[data-nav-page="settings"]').getBoundingClientRect().bottom - 0.5,
            columnScrolls: side.scrollHeight > side.clientHeight + 1,
            showAll: Boolean(document.getElementById('nav-projects-show-all')),
            header: {shape: mark.dataset.shape, hue: mark.dataset.hue, drawn: getComputedStyle(mark).display !== 'none'},
            expanded: document.getElementById('nav-projects-toggle').getAttribute('aria-expanded')};
}"""


@pytest.mark.parametrize('width,height', [(1108, 680), (390, 844)])
def test_collapsed_projects_keep_only_the_pinned_ones_and_the_tools_follow(shell_ui, tmp_path, width, height):
    page = seed_page(shell_ui, width, height)
    live = [PROJECTS[3]['id'], PROJECTS[7]['id']]
    pins = [PROJECTS[1]['id'], PROJECTS[7]['id']]
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    try:
        page.route('**/api/state', lambda route: route.fulfill(json=live_state(live, pins)))
        boot(page, shell_ui)
        page.wait_for_function(MOVING, arg=2, timeout=20_000)
        if width < 720:
            open_drawer(page)
        facts['open'] = page.evaluate(SIDEBAR_FACTS)
        page.locator('#nav-projects-toggle').click()
        settle(page)
        facts['collapsed'] = page.evaluate(SIDEBAR_FACTS)
        page.locator('#nav-projects-toggle').click()
        settle(page)
        facts['expanded'] = page.evaluate(SIDEBAR_FACTS)
    finally:
        evidence(tmp_path, f'shell-sidebar-collapse-{width}-{shell_ui["engine"]}', facts)
        page.close()
    collapsed = facts['collapsed']
    # All or nothing (owner 19.10, 2A): only the pinned Projects stay, in their order; work elsewhere keeps no row and
    # no "Show all" offers the rest. The header's one mark still speaks for them, in both states.
    assert collapsed['expanded'] == 'false' and collapsed['rows'] == pins, facts
    assert collapsed['showAll'] is False, facts
    assert collapsed['header'] == facts['open']['header'] == {'shape': 'moving', 'hue': 'ink', 'drawn': True}, facts
    # The tools follow the Projects zone: no empty room between them (owner 18.10). The version and the update rest on
    # the bottom edge in every state (owner 22.10): the free room lies between the tools and the foot.
    assert collapsed['gapToTools'] <= 24 and collapsed['filesTop'] < collapsed['sidebarBottom'] - 200, facts
    for state in ('open', 'collapsed', 'expanded'):
        side = facts[state]
        assert side['footBelowTools'], (state, facts)
        if not side['columnScrolls']:
            assert abs(side['footGap']) <= 1, (state, facts)
    assert facts['open']['expanded'] == 'true' and len(facts['open']['rows']) == len(PROJECTS), facts
    assert facts['expanded']['expanded'] == 'true' and len(facts['expanded']['rows']) == len(PROJECTS), facts


DOT_FACTS = """() => {
    const ink = (el) => el && getComputedStyle(el).color;
    const fill = (mark) => mark && getComputedStyle(mark.querySelector('i')).backgroundColor;
    const token = (name) => { const probe = document.createElement('span'); probe.style.color = `var(${name})`;
        document.querySelector('#nav-projects-list').append(probe); const c = getComputedStyle(probe).color; probe.remove(); return c; };
    const unread = document.querySelector('#nav-projects-list .nav-project-row[data-unread]');
    const working = document.querySelector('#nav-projects-list .status-mark[data-shape="moving"]')?.closest('.nav-project-row');
    const read = document.querySelector('#nav-projects-list .nav-project-row:not([data-unread]):not(.active)');
    const mark = (row) => row?.querySelector('.status-mark');
    return {unread: unread && {shape: mark(unread).dataset.shape, hue: mark(unread).dataset.hue, fill: fill(mark(unread)),
                ring: getComputedStyle(mark(unread).querySelector('i')).boxShadow, size: mark(unread).querySelector('i').getBoundingClientRect().width,
                ink: ink(unread), name: unread.getAttribute('aria-label')},
            working: working && {hue: mark(working).dataset.hue, fill: fill(mark(working)), ink: ink(working)},
            readInk: ink(read), crimson: token('--mark-unread'),
            retired: document.querySelectorAll('#primary-sidebar :is(.nav-unread-dot, .nav-activity-marker)').length};
}"""


def test_sidebar_marks_say_new_messages_in_crimson_and_work_in_the_row_ink(shell_ui, tmp_path):
    page = seed_page(shell_ui, 1108, 680)
    unread = [dict(p, visible_revision=3) if i == 2 else p for i, p in enumerate(PROJECTS)]
    state = dict(live_state([PROJECTS[5]['id']]), projects=unread)
    facts = {'engine': shell_ui['engine']}
    try:
        page.route('**/api/state', lambda route: route.fulfill(json=state))
        page.route('**/api/projects', lambda route: route.fulfill(json={'projects': unread}))
        boot(page, shell_ui)
        page.wait_for_function("() => document.querySelector('#nav-projects-list .nav-project-row[data-unread]') && "
                               "document.querySelector('#nav-projects-list .status-mark[data-shape=\"moving\"]')",
                               timeout=20_000)
        facts.update(page.evaluate(DOT_FACTS))
    finally:
        evidence(tmp_path, f'shell-sidebar-dots-{shell_ui["engine"]}', facts)
        page.close()
    # A room at rest with new messages is one plain 7px crimson dot, no ring around it; its name keeps every row's ink.
    assert facts['unread']['shape'] == 'dot' and facts['unread']['hue'] == 'unread', facts
    assert facts['unread']['fill'] == facts['crimson'] and facts['unread']['ring'] == 'none', facts
    assert facts['unread']['size'] == 7 and facts['unread']['ink'] == facts['readInk'], facts
    assert facts['unread']['name'].endswith(' · new messages'), facts
    assert facts['working']['hue'] == 'ink' and facts['working']['fill'] == facts['working']['ink'], facts
    assert facts['retired'] == 0, facts


PANEL_FACTS = """() => {
    const panel = document.getElementById('project-panel').getBoundingClientRect();
    const side = document.getElementById('primary-sidebar').getBoundingClientRect();
    const handle = document.querySelector('#project-panel > .resize-handle');
    return {panel: Math.round(panel.width), left: Math.round(panel.left), sidebarRight: Math.round(side.right),
            handle: Boolean(handle && handle.getClientRects().length),
            fill: document.documentElement.hasAttribute('data-project-fill')};
}"""


@pytest.mark.parametrize('width', [1108, 800])
def test_the_project_panel_keeps_its_own_width_beside_main(shell_ui, tmp_path, width):
    """The owner's widths (sidebar 282, panel 581): beside Main at its own width with its handle; a window too
    narrow for it gets a panel as wide as the content area, never over the sidebar."""
    page = seed_page(shell_ui, width, 680)
    facts = {'engine': shell_ui['engine'], 'width': width}
    try:
        page.route('**/api/ui/preferences', lambda route: route.fulfill(
            json={'sidebar_width': 282, 'project_panel_width': 581, 'project_seen_revision': {}}))
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--project-panel-width').trim() === '581px'",
                               timeout=20_000)
        open_project(page)
        settle(page)
        facts.update(page.evaluate(PANEL_FACTS))
    finally:
        evidence(tmp_path, f'shell-project-panel-{width}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['fill'] is False and facts['handle'] is True, facts
    content = width - facts['sidebarRight']
    assert facts['panel'] == min(581, content), facts
    assert facts['left'] >= facts['sidebarRight'], facts


# The real handle dragged one step per frame (out, then back), sampled every frame after the handle's own write.
DRAG = """async ([handleSelector, steps, dx, out]) => {
    const root = document.documentElement, side = document.getElementById('primary-sidebar');
    const panel = document.getElementById('project-panel'), dim = document.getElementById('project-panel-backdrop');
    const handle = document.querySelector(handleSelector), frame = () => new Promise(r => requestAnimationFrame(r));
    const box = handle.getBoundingClientRect(), y = box.top + Math.min(200, box.height / 2);
    let x = box.left + box.width / 2;
    const send = type => handle.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true, clientX: x,
        clientY: y, pointerId: 9, pointerType: 'mouse', isPrimary: true, button: 0, buttons: type === 'pointerup' ? 0 : 1}));
    const stored = () => [root.style.getPropertyValue('--sidebar-width'), root.style.getPropertyValue('--project-panel-width')];
    const before = stored(), seen = [];
    send('pointerdown');
    for (let i = 0; i < steps; i += 1) {
        x += i < steps * out ? dx : -dx;
        send('pointermove');
        await frame();
        const s = side.getBoundingClientRect(), p = panel.getBoundingClientRect(), d = dim.getBoundingClientRect();
        seen.push({side: s.right, dim: d.left, dimRight: d.right, panelLeft: p.left, panel: p.width, stored: stored()});
    }
    send('pointerup');
    await frame();
    const own = (box) => box.style.getPropertyValue('--drag-width');
    return {before, seen, after: stored(), inline: [own(side), own(panel)],
            shown: [Math.round(side.getBoundingClientRect().width), Math.round(panel.getBoundingClientRect().width)],
            content: innerWidth - side.getBoundingClientRect().right};
}"""


@pytest.mark.parametrize('which', ['sidebar', 'panel'])
def test_a_handle_drag_moves_its_box_and_the_room_follows_it_frame_by_frame(shell_ui, tmp_path, which):
    """With a Project open, the owner's widths: a drag writes only the dragged box's width, so a frame restyles one
    element instead of the whole document; the dimmed room and the panel's cap follow the sidebar's box by layout on
    every frame, the stored widths change once, at the end, to the width shown, and the box keeps no width of its
    own. The panel is dragged past the content area's edge and stops at it, never over the sidebar."""
    page = seed_page(shell_ui, 1108, 680)
    facts = {'engine': shell_ui['engine'], 'which': which}
    saved = []

    def preferences(route):
        if route.request.method != 'GET':
            saved.append(json.loads(route.request.post_data or '{}'))
        route.fulfill(json={'sidebar_width': 282, 'project_panel_width': 581, 'project_seen_revision': {}})
    try:
        page.route('**/api/ui/preferences', preferences)
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--project-panel-width').trim() === '581px'",
                               timeout=20_000)
        open_project(page)
        settle(page)
        # The sidebar goes out and part of the way back; the panel goes on past the content area's edge.
        drag = ['#primary-sidebar > .resize-handle', 30, 6, 0.7] if which == 'sidebar' else ['#project-panel > .resize-handle', 30, -16, 1]
        facts.update(page.evaluate(DRAG, drag))
        key = 'sidebar_width' if which == 'sidebar' else 'project_panel_width'
        for _ in range(30):  # the save is debounced
            if any(key in patch for patch in saved):
                break
            page.wait_for_timeout(100)
        facts['saved'] = [patch for patch in saved if key in patch]
    finally:
        evidence(tmp_path, f'shell-handle-drag-{which}-{shell_ui["engine"]}', facts)
        page.close()
    seen = facts['seen']
    moved = {round(f['side'] if which == 'sidebar' else f['panel']) for f in seen}
    assert len(moved) > 8, ('the box followed the pointer', facts)
    assert all(f['stored'] == facts['before'] for f in seen), ('the stored widths wait for the end of the drag', facts)
    for f in seen:
        assert abs(f['dim'] - f['side']) <= 0.5 and f['panelLeft'] >= f['side'] - 0.5, ('the room follows the sidebar', f)
    index = 0 if which == 'sidebar' else 1
    assert facts['after'][index] == f"{facts['shown'][index]}px" and facts['after'][1 - index] == facts['before'][1 - index], facts
    assert facts['inline'] == ['', ''], facts
    assert facts['saved'] == [{key: facts['shown'][index]}], ('the width shown is saved once', facts)
    if which == 'panel':
        assert 581 + 16 * 30 > facts['content'], facts
        assert max(f['panel'] for f in seen) <= facts['content'] + 0.5, ('it stops at the content area', facts)
        assert abs(facts['shown'][1] - facts['content']) <= 1, ('and keeps the width shown there', facts)


def test_the_project_panel_sliding_in_never_widens_the_window(shell_ui, tmp_path):
    """Docked, the panel slides in from past the window's right edge. Placed in #app (an absolute grid item), that
    offset would widen the page, and a scrollIntoView during the slide (a Project opened on a question) would
    scroll the whole window sideways; #app clips it. The slide is slowed so the check lands inside it."""
    page = seed_page(shell_ui, 1108, 680)
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        page.add_style_tag(content='.project-panel { transition-duration: 4s !important; }')
        open_project(page)
        facts.update(page.evaluate("""() => {
            const panel = document.getElementById('project-panel'), root = document.scrollingElement;
            const sliding = getComputedStyle(panel).transform;
            panel.querySelector('.project-panel-bar')?.scrollIntoView({block: 'center', inline: 'nearest'});
            return {sliding, pageWidth: root.scrollWidth, window: innerWidth, scrollLeft: root.scrollLeft,
                    sidebarLeft: document.getElementById('primary-sidebar').getBoundingClientRect().left};
        }"""))
    finally:
        evidence(tmp_path, f'shell-panel-slide-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['sliding'] not in ('none', 'matrix(1, 0, 0, 1, 0, 0)'), ('checked while the panel slides', facts)
    assert facts['pageWidth'] <= facts['window'], facts
    assert facts['scrollLeft'] == 0 and facts['sidebarLeft'] == 0, facts


# Seek the real slide: a busy runner must not turn missed compositor frames into a missing-motion failure.
PANEL_FRAMES = """({project, opening, answers}) => { const panel = document.getElementById('project-panel');
    const sample = """ + SAMPLE_TRANSITION + """;
    return sample({root: panel, target: () => panel, property: 'transform',
        ready: () => panel.classList.contains('open') === opening,
        act: () => { if (opening) window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project}}));
                     else panel.querySelector('#project-panel-close').click(); },
        measure: () => ({left: panel.getBoundingClientRect().left, hidden: panel.hidden,
                        rows: answers.filter(answer => [...panel.querySelectorAll('.chat-messages .chat-bubble')]
                            .some(row => row.textContent.includes(answer))).length})}); }"""
# A room's read takes 150 ms, as on a real link: a panel that set off at the click would cross the window empty.
SLOW_HISTORY = """(() => { const real = window.fetch; window.fetch = (input, init) => { const answer = real(input, init);
    return String(input?.url || input).includes('/api/chat/history')
        ? answer.then((r) => new Promise((resolve) => setTimeout(() => resolve(r), 150))) : answer; }; })()"""


def test_docked_the_project_panel_comes_in_with_its_room_and_goes_back_with_it(shell_ui, tmp_path):
    """Owner 20.10 (Q2 A): docked, the panel comes in from the window's right edge once its room has painted and goes
    back there whole: samples along both real transitions show its rows, from the edge to Main and back."""
    page = seed_page(shell_ui, 1143, 790)
    page.emulate_media(reduced_motion='no-preference')
    rows = [{'role': 'assistant', 'text': f'Answer {i}. ' + 'Ordinary context. ' * 8,
             'content': f'Answer {i}.', 'ts': f'2026-10-14T09:{i:02d}:00Z'} for i in range(6)]
    answers = [row['content'] for row in rows]  # A startup System bubble is not the room's history.
    page.route('**/api/chat/history*', lambda route: route.fulfill(json={'messages': rows, 'progress': []}))
    page.add_init_script(SLOW_HISTORY)
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        width = page.evaluate('innerWidth')
        facts['opening'] = page.evaluate(PANEL_FRAMES, {'project': PROJECTS[0], 'opening': True, 'answers': answers})
        rest(page)
        page.locator('#project-panel-close').click(trial=True)
        facts['closing'] = page.evaluate(PANEL_FRAMES, {'opening': False, 'answers': answers})
        page.wait_for_function("() => document.getElementById('project-panel').hidden")
    finally:
        evidence(tmp_path, f'shell-panel-arrival-{shell_ui["engine"]}', facts)
        page.close()
    opening, closing = facts['opening']['values'], facts['closing']['values']
    home = opening[-1]['left']
    assert home < width - 300, ('it ends beside Main', home, width)
    assert facts['opening']['duration'] > 0 and facts['closing']['duration'] > 0, facts
    assert abs(opening[1]['left'] - width) <= 1 and abs(closing[-1]['left'] - width) <= 1, facts

    def moving(frames):
        return [f for f in frames if not f['hidden'] and home + 1 < f['left'] < width - 1]
    assert len(moving(opening)) >= 3, ('the arrival is seen moving', opening)
    arriving = opening + facts['opening']['live']
    leaving = closing + facts['closing']['live']
    assert all(f['rows'] == len(answers) for f in moving(arriving)), ('it sets off with its room painted', arriving)
    assert len(moving(closing)) >= 3, ('the way back is seen moving', closing)
    assert all(f['rows'] == len(answers) for f in leaving if not f['hidden']), ('it goes back with its room', leaving)


def test_a_project_closed_mid_drag_keeps_its_stored_width(shell_ui, tmp_path):
    """Escape while holding the Project panel's handle closes the panel; the release then finds no width shown,
    and the stored width and the saved preference stay what they were (no 0px panel when it opens again)."""
    page = seed_page(shell_ui, 1108, 680)
    facts = {'engine': shell_ui['engine']}
    saved = []

    def preferences(route):
        if route.request.method != 'GET':
            saved.append(json.loads(route.request.post_data or '{}'))
        route.fulfill(json={'sidebar_width': 282, 'project_panel_width': 581, 'project_seen_revision': {}})
    try:
        page.route('**/api/ui/preferences', preferences)
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--project-panel-width').trim() === '581px'",
                               timeout=20_000)
        open_project(page)
        settle(page)
        page.evaluate("""async () => {
            const handle = document.querySelector('#project-panel > .resize-handle');
            const frame = () => new Promise(r => requestAnimationFrame(r));
            const box = handle.getBoundingClientRect(), y = box.top + 200;
            let x = box.left + box.width / 2;
            const send = type => handle.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true,
                clientX: x, clientY: y, pointerId: 9, pointerType: 'mouse', isPrimary: true, button: 0,
                buttons: type === 'pointerup' ? 0 : 1}));
            send('pointerdown');
            for (let i = 0; i < 6; i += 1) { x -= 10; send('pointermove'); await frame(); }
            window.__dragHandle = {send};
        }""")
        page.keyboard.press('Escape')
        page.wait_for_function("() => document.getElementById('project-panel').hidden", timeout=5_000)
        page.evaluate("() => window.__dragHandle.send('pointerup')")
        page.wait_for_timeout(700)  # the save is debounced
        facts['stored'] = page.evaluate("() => document.documentElement.style.getPropertyValue('--project-panel-width')")
        facts['saved'] = [patch for patch in saved if 'project_panel_width' in patch]
        open_project(page)
        settle(page)
        facts['reopened'] = page.evaluate("() => Math.round(document.getElementById('project-panel').getBoundingClientRect().width)")
    finally:
        evidence(tmp_path, f'shell-panel-closed-mid-drag-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['stored'] == '581px' and facts['saved'] == [] and facts['reopened'] == 581, facts


FOCUS = """() => { const a = document.activeElement;
    return a === document.body ? 'body' : a.id || (a.dataset.projectId ? 'project:' + a.dataset.projectId
        : a.dataset.navPage ? 'page:' + a.dataset.navPage : a.className); }"""


def test_a_row_that_hides_gives_the_keyboard_to_the_projects_toggle(shell_ui, tmp_path):
    page = seed_page(shell_ui, 1108, 680)
    pins = [PROJECTS[3]['id'], PROJECTS[7]['id']]
    current = {'state': live_state([], pins)}
    facts = {'engine': shell_ui['engine']}
    hidden = "id => document.querySelector(`[data-project-id=\"${id}\"]`).closest('[hidden]') !== null"

    def unpin(still):  # another window unpins: the next snapshot says so
        current['state'] = live_state([], still)
        page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")

    try:
        page.route('**/api/state', lambda route: route.fulfill(json=current['state']))
        boot(page, shell_ui)
        page.locator('#nav-projects-toggle').click()
        settle(page)
        facts['collapsed'] = page.evaluate(ROWS)
        # The keyboard in a pinned row's menu, then that Project is unpinned: its row and its menu go, the keyboard
        # lands on the Projects toggle instead of the page.
        page.locator(f'[data-nav-project-id="{pins[0]}"] .nav-project-kebab').focus()
        page.keyboard.press('Enter')
        page.locator('body > .project-row-menu:not([data-leaving])').wait_for()
        facts['in_menu'] = page.evaluate("() => document.activeElement.dataset.prm || ''")
        unpin(pins[1:])
        page.wait_for_function(hidden, arg=pins[0], timeout=20_000)
        settle(page)
        facts['menus_after'] = page.locator('.project-row-menu:not([data-leaving])').count()
        facts['focus_after_menu'] = page.evaluate(FOCUS)
        # The keyboard on the other pinned row, which is unpinned too: the list empties, the toggle keeps the keyboard.
        page.locator(f'[data-project-id="{pins[1]}"]').focus()
        unpin([])
        page.wait_for_function(hidden, arg=pins[1], timeout=20_000)
        settle(page)
        facts['focus_after_row'] = page.evaluate(FOCUS)
        facts['list_hidden'] = page.evaluate("() => document.getElementById('nav-projects-list').hidden")
        page.keyboard.press('Enter')
        settle(page)
        facts['expanded_rows'] = len(page.evaluate(ROWS))
    finally:
        evidence(tmp_path, f'shell-sidebar-collapse-focus-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['collapsed'] == pins and facts['in_menu'] == 'pin', facts
    assert facts['menus_after'] == 0 and facts['focus_after_menu'] == 'nav-projects-toggle', facts
    assert facts['focus_after_row'] == 'nav-projects-toggle' and facts['list_hidden'] is True, facts
    assert facts['expanded_rows'] == len(PROJECTS), 'the keyboard expands the list again from where it was left'


ROWS = "() => [...document.querySelectorAll('#nav-projects-list > .nav-project-item')].filter((e) => !e.hidden).map((e) => e.querySelector('[data-project-id]').dataset.projectId)"


def test_collapsed_projects_ignore_the_census_and_the_open_one(shell_ui, tmp_path):
    page = seed_page(shell_ui, 1108, 680)
    current = {'state': live_state([PROJECTS[3]['id']])}
    facts = {'engine': shell_ui['engine']}
    header = "() => document.querySelector('.nav-projects-facts .status-mark').dataset.shape"
    try:
        page.route('**/api/state', lambda route: route.fulfill(json=current['state']))
        boot(page, shell_ui)
        page.wait_for_function(MOVING, arg=1, timeout=20_000)
        page.locator('#nav-projects-toggle').click()
        facts['one'] = page.evaluate(ROWS)
        facts['one_header'] = page.evaluate(header)
        for name, ids_now in (('three', [PROJECTS[3]['id'], PROJECTS[5]['id'], PROJECTS[9]['id']]), ('none', [])):
            current['state'] = live_state(ids_now)
            page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
            page.wait_for_function(MOVING, arg=len(ids_now), timeout=20_000)
            settle(page)
            facts[name] = page.evaluate(ROWS)
            facts[f'{name}_header'] = page.evaluate(header)
        # The open Project keeps no row in a collapsed list either: the room itself is open beside it.
        open_project(page, 12)
        settle(page)
        facts['open'] = page.evaluate(ROWS)
        facts['list_hidden'] = page.evaluate("() => document.getElementById('nav-projects-list').hidden")
    finally:
        evidence(tmp_path, f'shell-sidebar-collapse-census-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['one'] == facts['three'] == facts['none'] == facts['open'] == [], facts
    assert facts['list_hidden'] is True, facts
    # Work under the collapsed header shows as its one mark: moving while any works, nothing once none does.
    assert (facts['one_header'], facts['three_header'], facts['none_header']) == ('moving', 'moving', 'none'), facts
