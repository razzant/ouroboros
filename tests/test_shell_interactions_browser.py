"""Phone shell interactions: layer inert, Back, Escape, focus, the drawer drag, inline links, a short phone.

Each case turns a CR1 reviewer probe around: it asserts the fixed behaviour and, where a guard is involved, that
the guard still fires. Uses the static-server/API fixture (test_subscription_setup_browser); never starts the
runtime or reads live data. Synthetic CDP touch is Chromium's input pipeline, not a claim about an iPhone WebView.
"""
from __future__ import annotations

import json
import re

import pytest

from ouroboros.launcher_window_chrome import DOCUMENT_SCRIPT
from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_settings
from tests.test_design_system_gate_browser import (PROJECTS, open_nav, rest, save_evidence, seed_page, settle,
                                                   sidebar_metrics)
from tests._work_projection_shared import work_fields
from tests._shared import answer_projects
from tests.ui_motion_samples import SAMPLE_TRANSITION

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]


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


# A Project reference in Main is a jump (DESIGN "Layers and dismissal"): its room lies over the return to Main, and
# closing the room consumes that return with it. A row of the sidebar or drawer leaves the room alone (the drawer case).
PROJECT_LAYERS = ['page-return-1', 'project']


def open_project(page, index=0):
    """Open a Project the way its reference in Main does."""
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
    page.wait_for_function("""() => {
        const row = document.querySelector('[data-project-id="gate-5"]');
        return row?.hasAttribute('data-unread') && row.querySelector('.status-mark')?.dataset.hue === 'unread';
    }""")
    assert page.locator('#nav-projects-count').count() == 0
    page.wait_for_timeout(300)


SURFACES = {
    'confirm': {'surface': '.confirm-dialog-backdrop',
                'control': '.confirm-dialog .ui-modal-actions [data-confirm-cancel]'},
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
    assert facts['after_click']['layers'] == PROJECT_LAYERS and facts['after_click']['contentInert'], facts
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
    under = ['drawer'] if layer == 'drawer' else PROJECT_LAYERS
    assert facts['open']['layers'] == under and facts['open']['dialog'], facts
    assert facts['first_back']['layers'] == under and not facts['first_back']['dialog'], facts
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
        focus: active?.closest?.('.files-entry')?.dataset.path ?? (active?.matches?.('[data-shell-back]') ? 'back'
            : active?.matches?.('.files-preview') ? 'file' : active?.tagName),
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
    # Telegram owns Back in its native row; the page gives focus to the visible file instead.
    assert opened['focus'] == ('file' if via == 'telegram' else 'back'), facts
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
    assert facts['capsule_after'] == 'false' and facts['layers_after'] == PROJECT_LAYERS, facts
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
    under = [] if room == 'main' else PROJECT_LAYERS  # the drawer opens over an open Project room
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


# Settings copy carries no inline link any more (the ClawHub sentence that held one is a row's meta now); a standalone
# link, such as About's, keeps a finger's target on a phone.
def test_a_standalone_link_in_settings_keeps_a_finger_target(shell_ui, tmp_path):
    page = seed_page(shell_ui, 390, 844)
    facts = {'engine': shell_ui['engine']}
    standalone = '#page-settings .about-links a[href]'
    try:
        boot(page, shell_ui)
        facts['coarse'] = page.evaluate("matchMedia('(pointer: coarse)').matches")
        open_drawer(page)
        page.locator('[data-nav-page="settings"]').click()
        page.locator('#page-settings').wait_for(state='visible')
        open_settings(page, 'computer/about')
        page.locator(standalone).first.scroll_into_view_if_needed()
        settle(page)
        facts['standalone'] = page.evaluate(TARGET, standalone)
    finally:
        evidence(tmp_path, f'shell-standalone-link-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['coarse'] is True, facts
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
                controls.add(control.matches('[data-shell-back]') ? 'back'
                    : control.id || control.querySelector('input')?.id || control.textContent.trim());
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
        # Extensions opens through Settings' door (owner 24.11), so its band leads with the ordinary Back to Settings.
        open_nav(page, page_name)
        settle(page)
        allowed = ('.app-page-lead .app-tab, .app-page-lead .files-crumb, .settings-search, '
                   '.app-page-header #log-filters button, .app-page-header #btn-clear-logs, '
                   '.app-page-header [data-shell-back], '
                   '.app-page-toolbar :is(#skills-refresh, #skills-update-all, #files-refresh, '
                   '#files-new-file, #files-new-dir, #files-paste)')
        facts['band'] = page.evaluate(TITLE_BAND, {'band': 52, 'allowed': allowed})
        band = facts['band']
        assert band['height'] == 52 and band['drags'] > 0 and not band['unexpected'], facts
        if page_name == 'widgets':
            assert band['controls'] == [] and band['firstTabTop'] is None, facts
        elif page_name == 'dashboard':  # Logs' filters and Clear occupy the one band.
            assert 'btn-clear-logs' in band['controls'] and band['firstTabTop'] is None, facts
        elif page_name == 'settings':
            assert band['controls'] == ['settings-search'] and band['firstTabTop'] is None, facts
        elif page_name == 'skills':  # Extensions (#3921): its pages are rail rows below the band, no tab strip in it.
            assert sorted(band['controls']) == ['back', 'skills-refresh'] and band['firstTabTop'] is None, facts
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
        elif page_name == 'settings':
            page.locator('#settings-search').fill('models')
            page.locator('[data-settings-result="home:models"]').wait_for(state='visible')
            assert page.locator('#settings-search').input_value() == 'models'
        elif page_name == 'dashboard':
            row = page.locator('[data-dashboard-tab="activity"]')
            row.click()
            assert row.get_attribute('aria-selected') == 'true'
            page.locator('#dashboard-panel-activity.active').wait_for(state='visible')
        elif page_name == 'skills':  # the band's one control there
            page.locator('#skills-refresh').click()
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
    const box = (sel) => side.querySelector(sel).getBoundingClientRect();
    // The column's content edge: the Projects zone reaches it, and the rail's version row ends on it.
    const edge = side.getBoundingClientRect().bottom - parseFloat(getComputedStyle(side).paddingBottom);
    return {rows: rows.map((e) => e.querySelector('[data-project-id]').dataset.projectId),
            zoneGap: Math.round(edge - box('#nav-projects').bottom),
            railTop: Math.round(box('.nav-rail').top), sectionsTop: Math.round(box('.nav-rail [data-nav-page="circles"]').top),
            settingsTop: Math.round(box('.nav-rail [data-nav-page="settings"]').top),
            versionGap: Math.round(edge - box('.sidebar-foot-meta').bottom),
            versionBelowSettings: box('.sidebar-foot-meta').top >= box('.nav-rail [data-nav-page="settings"]').bottom - 0.5,
            columnScrolls: side.scrollHeight > side.clientHeight + 1,
            retired: side.querySelectorAll('#nav-projects-toggle, #nav-projects-count, .nav-projects-facts, .sidebar-foot, .nav-card').length,
            moving: rows.filter(e => e.querySelector('.status-mark')?.dataset.shape === 'moving')
                .map(e => e.dataset.navProjectId)};
}"""


@pytest.mark.parametrize('width,height', [(1108, 680), (390, 844)])
def test_collapsed_projects_keep_only_the_pinned_ones_and_the_rail_holds_still(shell_ui, tmp_path, width, height):
    """Navigation has no collapse: Find narrows the list; its pins lead, the Projects keep the column down to its bottom
    edge, and the rail beside them holds its places and its version still (owner 24.11)."""
    page = seed_page(shell_ui, width, height)
    live = [PROJECTS[3]['id'], PROJECTS[7]['id']]
    pins = [PROJECTS[1]['id'], PROJECTS[7]['id']]
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    state = live_state(live, pins)
    for project in state['projects']:
        if project['id'] in pins:
            project['name'] = 'Matched pin: ' + project['name']
    try:
        page.route('**/api/state', lambda route: route.fulfill(json=state))
        page.route('**/api/projects*', lambda route: answer_projects(route, state['projects']))
        boot(page, shell_ui)
        page.wait_for_function(MOVING, arg=2, timeout=20_000)
        if width < 720:
            open_drawer(page)
        facts['open'] = page.evaluate(SIDEBAR_FACTS)
        page.locator('.nav-find-input').fill('matched pin')
        page.wait_for_function('wanted => JSON.stringify((' + ROWS + ')()) === JSON.stringify(wanted)', arg=pins)
        settle(page)
        facts['filtered'] = page.evaluate(SIDEBAR_FACTS)
        setup_browser.capture(page, f'shell-sidebar-filtered-{width}-{shell_ui["engine"]}')
        page.locator('.nav-find-clear').click()
        page.wait_for_function('n => (' + ROWS + ')().length === n', arg=len(PROJECTS))
        settle(page)
        facts['cleared'] = page.evaluate(SIDEBAR_FACTS)
    finally:
        evidence(tmp_path, f'shell-sidebar-collapse-{width}-{shell_ui["engine"]}', facts)
        page.close()
    expected = pins + [p['id'] for p in PROJECTS if p['id'] not in pins]
    assert facts['open']['rows'] == facts['cleared']['rows'] == expected, facts
    assert facts['filtered']['rows'] == pins, facts
    assert set(facts['open']['moving']) == set(facts['cleared']['moving']) == set(live), facts
    assert facts['filtered']['moving'] == [pins[1]], facts
    # DESIGN Navigation: the Projects take the column down to its bottom edge; the rail's places and version hold still.
    for phase in ('open', 'filtered', 'cleared'):
        side = facts[phase]
        assert side['retired'] == 0 and abs(side['zoneGap']) <= 1, (phase, facts)
        for key in ('railTop', 'sectionsTop', 'settingsTop'):
            assert abs(side[key] - facts['open'][key]) <= 1, (phase, key, facts)
        assert side['versionBelowSettings'], (phase, facts)
        if not side['columnScrolls']:
            assert abs(side['versionGap']) <= 1, ('the version row ends on the bottom inset', phase, facts)


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
    // What the rail adds beside the sidebar's column (style.css --nav-rail-extra): the width the handle owns is the
    // column's, the box less that share.
    const extra = parseFloat(getComputedStyle(side).getPropertyValue('--nav-rail-extra')) || 0;
    return {before, seen, after: stored(), inline: [own(side), own(panel)], extra,
            shown: [Math.round(side.getBoundingClientRect().width - extra), Math.round(panel.getBoundingClientRect().width)],
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
    assert facts['extra'] == 40, ('the rail adds its 40px beside the column the handle owns', facts)
    assert facts['saved'] == [{key: facts['shown'][index]}], ('the width shown is saved once', facts)
    if which == 'panel':
        assert 581 + 16 * 30 > facts['content'], facts
        assert max(f['panel'] for f in seen) <= facts['content'] + 0.5, ('it stops at the content area', facts)
        assert abs(facts['shown'][1] - facts['content']) <= 1, ('and keeps the width shown there', facts)


# The real sidebar handle pressed and released in place, then dragged out and back by the same step, several times: what
# each drag stored, and the column and the box after it.
STILL_AND_BACK = """async ([step, rounds]) => {
    const root = document.documentElement, side = document.getElementById('primary-sidebar');
    const handle = side.querySelector(':scope > .resize-handle'), frame = () => new Promise(r => requestAnimationFrame(r));
    const drag = async (dx) => {
        const box = handle.getBoundingClientRect(), y = box.top + Math.min(200, box.height / 2);
        let x = box.left + box.width / 2;
        const send = type => handle.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true, clientX: x,
            clientY: y, pointerId: 9, pointerType: 'mouse', isPrimary: true, button: 0, buttons: type === 'pointerup' ? 0 : 1}));
        send('pointerdown');
        if (dx) { x += dx; send('pointermove'); await frame(); }
        send('pointerup');
        await frame();
        await frame();
        return {stored: root.style.getPropertyValue('--sidebar-width'), box: side.getBoundingClientRect().width,
                column: side.querySelector('.sidebar-head').getBoundingClientRect().width};
    };
    const seen = [await drag(0)];
    for (let i = 0; i < rounds; i += 1) { seen.push(await drag(step)); seen.push(await drag(-step)); }
    seen.push(await drag(0));
    return seen;
}"""


def test_a_sidebar_drag_stores_its_column_and_repeats_without_growing(shell_ui, tmp_path):
    """The rail stands beside the column and the room pays for it (owner 24.11): the handle owns the column's width, so
    a press released in place stores the width it found, and drags out and back by the same step return to it however
    many times, the box always the column and the rail's 40px (Neon's static review, 09a6e44d: a box measured whole
    would add the rail at every drag)."""
    page = seed_page(shell_ui, 1108, 680)
    facts = {'engine': shell_ui['engine']}
    saved = []

    def preferences(route):
        if route.request.method != 'GET':
            saved.append(json.loads(route.request.post_data or '{}'))
        route.fulfill(json={'sidebar_width': 282, 'project_seen_revision': {}})
    try:
        page.route('**/api/ui/preferences', preferences)
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width').trim() === '282px'",
                               timeout=20_000)
        facts['seen'] = page.evaluate(STILL_AND_BACK, [30, 3])
        for _ in range(30):  # the save is debounced
            if saved:
                break
            page.wait_for_timeout(100)
        facts['saved'] = saved
    finally:
        evidence(tmp_path, f'shell-sidebar-drag-repeat-{shell_ui["engine"]}', facts)
        page.close()
    seen = facts['seen']
    assert [f['stored'] for f in seen] == ['282px'] + ['312px', '282px'] * 3 + ['282px'], facts
    for f in seen:
        width = float(f['stored'][:-2])
        assert abs(f['box'] - (width + 40)) <= 0.5 and abs(f['column'] - (width - 25)) <= 0.5, ('the box and its column', f)
    assert facts['saved'] and facts['saved'][-1] == {'sidebar_width': 282}, ('the last width shown is saved', facts)


SIDEBAR_SNAPSHOT = """() => {
    const root = document.documentElement, side = document.getElementById('primary-sidebar');
    const rawExtra = getComputedStyle(side).getPropertyValue('--nav-rail-extra').trim();
    return {
        stored: root.style.getPropertyValue('--sidebar-width'),
        box: side.getBoundingClientRect().width,
        column: side.querySelector('.sidebar-head').getBoundingClientRect().width,
        content: document.getElementById('content').getBoundingClientRect().width,
        rawExtra,
        extraUnparsed: Number.isNaN(Number.parseFloat(rawExtra)),
    };
}"""

NO_MOTION_DRAGS = """async () => {
    const handle = document.querySelector('#primary-sidebar > .resize-handle');
    const frame = () => new Promise(r => requestAnimationFrame(r));
    const box = handle.getBoundingClientRect(), x = box.left + box.width / 2, y = box.top + Math.min(200, box.height / 2);
    const send = (type) => handle.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true, clientX: x,
        clientY: y, pointerId: 11, pointerType: 'mouse', isPrimary: true, button: 0, buttons: type === 'pointerup' ? 0 : 1}));
    send('pointerdown');
    send('pointerup');
    await frame();
    await frame();
    send('pointerdown');
    send('pointermove');
    await frame();
    send('pointerup');
    await frame();
    await frame();
}"""


@pytest.mark.parametrize('raw_calc', [False, True], ids=['registered', 'raw-calc'])
def test_preferred_sidebar_width_survives_narrow_docked_boot_no_motion_and_raw_calc_drags(shell_ui, tmp_path, raw_calc):
    """Booting at docked 720 px with a valid 400 px desktop preference keeps CSS's display clamp (400 px box, 320 px
    room) without posting a narrower preference; no-motion press/release and window widening never mutate it and
    widening recovers 400 px (440 px box); repeated real drags keep the column width accurately even when
    --nav-rail-extra is an unregistered calc() expression."""
    page = seed_page(shell_ui, 720, 680)
    facts = {'engine': shell_ui['engine'], 'raw_calc': raw_calc}
    saved = []

    def preferences(route):
        if route.request.method != 'GET':
            saved.append(json.loads(route.request.post_data or '{}'))
        route.fulfill(json={'sidebar_width': 400, 'project_seen_revision': {}})

    try:
        if raw_calc:
            raw_css, removed = re.subn(
                r'@property\s+--nav-rail-extra\s*\{[^}]*\}', '',
                (setup_browser.WEB / 'style.css').read_text(encoding='utf-8'), count=1,
            )
            assert removed == 1, 'style.css must declare @property --nav-rail-extra for raw-calc testing'
            page.route('**/static/style.css*', lambda route: route.fulfill(content_type='text/css', body=raw_css))
        page.route('**/api/ui/preferences', preferences)
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width').trim() === '400px'",
                               timeout=20_000)
        page.wait_for_timeout(600)  # past the 400 ms persist debounce
        facts['boot_720'] = page.evaluate(SIDEBAR_SNAPSHOT)
        facts['saved_after_boot'] = list(saved)

        page.evaluate(NO_MOTION_DRAGS)
        page.wait_for_timeout(600)
        facts['after_no_motion'] = page.evaluate(SIDEBAR_SNAPSHOT)
        facts['saved_after_no_motion'] = list(saved)

        page.set_viewport_size({'width': 1280, 'height': 680})
        settle(page)
        page.wait_for_timeout(600)
        facts['widened_1280'] = page.evaluate(SIDEBAR_SNAPSHOT)
        facts['saved_after_widen'] = list(saved)

        facts['drags'] = page.evaluate(STILL_AND_BACK, [30, 2])
        for _ in range(30):
            if saved:
                break
            page.wait_for_timeout(100)
        facts['saved_after_drags'] = list(saved)
    finally:
        tag = 'raw-calc' if raw_calc else 'registered'
        evidence(tmp_path, f'shell-sidebar-pref-widen-{tag}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['boot_720']['extraUnparsed'] is raw_calc, facts
    assert facts['boot_720']['stored'] == '400px' and facts['saved_after_boot'] == [], facts
    assert abs(facts['boot_720']['box'] - 400) <= 1 and abs(facts['boot_720']['content'] - 320) <= 1, facts
    assert facts['after_no_motion']['stored'] == '400px' and facts['saved_after_no_motion'] == [], facts
    assert facts['widened_1280']['stored'] == '400px' and facts['saved_after_widen'] == [], facts
    assert abs(facts['widened_1280']['box'] - 440) <= 0.5 and abs(facts['widened_1280']['column'] - 375) <= 0.5, facts
    assert [f['stored'] for f in facts['drags']] == ['400px', '430px', '400px', '430px', '400px', '400px'], facts
    for f in facts['drags']:
        width = float(f['stored'][:-2])
        assert abs(f['box'] - (width + 40)) <= 0.5 and abs(f['column'] - (width - 25)) <= 0.5, ('column + 40 px rail', f)
    assert facts['saved_after_drags'] and facts['saved_after_drags'][-1] == {'sidebar_width': 400}, facts


TG_INSET_INIT = """(() => {
    const app = window.Telegram?.WebApp;
    if (!app) return;
    const handlers = {}, origOn = app.onEvent.bind(app);
    app.onEvent = (name, fn) => ((handlers[name] ||= []).push(fn), origOn(name, fn));
    window.__tgEmit = (name) => (handlers[name] || []).forEach(fn => fn());
    app.isFullscreen = true;
    app.safeAreaInset = {top: 0, bottom: 21, left: 47, right: 47};
    app.contentSafeAreaInset = {top: 0, bottom: 0, left: 0, right: 0};
})()"""

DRAG_ONCE = """async (dx) => {
    const root = document.documentElement, side = document.getElementById('primary-sidebar');
    const handle = side.querySelector(':scope > .resize-handle'), frame = () => new Promise(r => requestAnimationFrame(r));
    const b = handle.getBoundingClientRect(), y = b.top + Math.min(200, b.height / 2);
    let x = b.left + b.width / 2;
    const send = type => handle.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true, clientX: x,
        clientY: y, pointerId: 12, pointerType: 'mouse', isPrimary: true, button: 0, buttons: type === 'pointerup' ? 0 : 1}));
    const widths = () => ({sidebarWidth: side.getBoundingClientRect().width, contentWidth: document.getElementById('content').getBoundingClientRect().width});
    send('pointerdown'); x += dx; send('pointermove'); await frame();
    const during = {dragWidth: side.style.getPropertyValue('--drag-width'), ...widths()};
    send('pointerup'); await frame(); await frame();
    return {during, stored: root.style.getPropertyValue('--sidebar-width'), ...widths()};
}"""


def _route_sidebar_prefs(page, width, *, raw_calc=False):
    saved = []
    if raw_calc:
        raw_css, removed = re.subn(r'@property\s+--nav-rail-extra\s*\{[^}]*\}', '',
                                   (setup_browser.WEB / 'style.css').read_text(encoding='utf-8'), count=1)
        assert removed == 1, 'style.css must declare @property --nav-rail-extra for raw-calc testing'
        page.route('**/static/style.css*', lambda route: route.fulfill(content_type='text/css', body=raw_css))

    def preferences(route):
        if route.request.method != 'GET':
            saved.append(json.loads(route.request.post_data or '{}'))
        route.fulfill(json={'sidebar_width': width, 'project_seen_revision': {}})

    page.route('**/api/ui/preferences', preferences)
    return saved


@pytest.mark.parametrize('raw_calc', [False, True], ids=['registered', 'raw-calc'])
def test_telegram_paired_side_insets_cap_docked_sidebar_to_actual_app_width(shell_ui, tmp_path, raw_calc):
    """844x390 Telegram fullscreen with paired 47 px insets caps 420 px sidebar to 430 px inside 750 px #app
    (320 px room, zero inward pull/POST on capped drags), recovers 460 px on clear, and saves inward drags."""
    page = seed_page(shell_ui, 844, 390, telegram=True)
    page.add_init_script(TG_INSET_INIT)
    facts, saved = {'engine': shell_ui['engine'], 'raw_calc': raw_calc}, _route_sidebar_prefs(page, 420, raw_calc=raw_calc)
    snap = """() => {
        const r = el => { const b = el.getBoundingClientRect(); return {left: b.left, right: b.right, width: b.width}; };
        const side = document.getElementById('primary-sidebar'), rawExtra = getComputedStyle(side).getPropertyValue('--nav-rail-extra').trim();
        return {layout: document.documentElement.dataset.layout || null, fullscreen: document.documentElement.dataset.telegramFullscreen || null,
            stored: document.documentElement.style.getPropertyValue('--sidebar-width'), app: r(document.getElementById('app')),
            sidebar: r(side), content: r(document.getElementById('content')), rawExtra, extraUnparsed: Number.isNaN(Number.parseFloat(rawExtra))};
    }"""
    set_insets = "insets => { window.Telegram.WebApp.safeAreaInset = insets; window.__tgEmit('safeAreaChanged'); }"
    try:
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width').trim() === '420px'", timeout=20_000)
        page.wait_for_timeout(600)
        facts['boot_inset'], facts['saved_after_boot'] = page.evaluate(snap), list(saved)
        page.evaluate(NO_MOTION_DRAGS)
        page.wait_for_timeout(600)
        facts['right_into_cap'] = page.evaluate(DRAG_ONCE, 80)
        page.wait_for_timeout(600)
        facts['saved_after_capped_gestures'] = list(saved)
        page.evaluate(set_insets, {'top': 0, 'bottom': 0, 'left': 0, 'right': 0})
        settle(page)
        page.wait_for_timeout(600)
        facts['cleared_insets'], facts['saved_after_clear'] = page.evaluate(snap), list(saved)
        page.evaluate(set_insets, {'top': 0, 'bottom': 21, 'left': 47, 'right': 47})
        settle(page)
        facts['inward_drag'] = page.evaluate(DRAG_ONCE, -30)
        for _ in range(30):
            if saved:
                break
            page.wait_for_timeout(100)
        facts['saved_after_inward'] = list(saved)
    finally:
        evidence(tmp_path, f'shell-telegram-inset-cap-{"raw-calc" if raw_calc else "registered"}-{shell_ui["engine"]}', facts)
        page.close()
    b, cap, clr, inw = facts['boot_inset'], facts['right_into_cap'], facts['cleared_insets'], facts['inward_drag']
    assert b['extraUnparsed'] is raw_calc and b['layout'] == 'docked' and b['fullscreen'] == 'active', facts
    assert b['stored'] == '420px' and facts['saved_after_boot'] == [], facts
    assert abs(b['app']['left'] - 47) <= 1 and abs(b['app']['width'] - 750) <= 1, facts
    assert abs(b['sidebar']['width'] - 430) <= 1 and b['content']['width'] >= 319.5 and abs(b['content']['width'] - 320) <= 1, facts
    assert cap['during']['dragWidth'] == '390px' and cap['during']['sidebarWidth'] >= 429.5 and cap['during']['contentWidth'] >= 319.5, facts
    assert cap['stored'] == '420px' and facts['saved_after_capped_gestures'] == [], facts
    assert abs(clr['app']['width'] - 844) <= 1 and abs(clr['sidebar']['width'] - 460) <= 1 and abs(clr['content']['width'] - 384) <= 1, facts
    assert clr['stored'] == '420px' and facts['saved_after_clear'] == [], facts
    assert inw['stored'] == '360px' and abs(inw['sidebarWidth'] - 400) <= 1 and abs(inw['contentWidth'] - 350) <= 1, facts
    assert facts['saved_after_inward'] and facts['saved_after_inward'][-1] == {'sidebar_width': 360}, facts


def _route_call_mark_assets(page):
    call_ui_dir = setup_browser.WEB.parent / 'skills' / 'live_conversation' / 'ui'

    def serve_call_assets(route):
        name = route.request.url.split('?', 1)[0].rsplit('/', 1)[-1]
        if name not in ('call.css', 'call_mark.js'):
            return route.fulfill(status=404, body='not found')
        route.fulfill(content_type='text/css' if name.endswith('.css') else 'text/javascript',
                      body=(call_ui_dir / name).read_text(encoding='utf-8'))

    page.route('**/api/extensions/live_conversation/ui/test/*', serve_call_assets)


def test_off_room_call_mark_stays_inside_capped_docked_sidebar_and_leaves_composer_controls_hittable(shell_ui, tmp_path):
    """At desktop 720 px with saved sidebar_width=540 (sidebar capped at 400 px), an off-room .call-mark stays
    inside the sidebar's column bounds without covering .nav-rail or composer controls, and follows settled widening."""
    mount_js = """async () => {
        await new Promise((ok, err) => document.head.append(Object.assign(document.createElement('link'),
            {rel: 'stylesheet', href: '/api/extensions/live_conversation/ui/test/call.css', onload: ok, onerror: () => err(new Error('call.css'))})));
        const { trackCallMark } = await import('/api/extensions/live_conversation/ui/test/call_mark.js');
        const entry = trackCallMark({session: {active: true, liveSince: Date.now() - 12_000, end() {}},
            label: () => 'Project 0: a descriptive project name', onScreen: () => false, reveal() {}, page: document.getElementById('page-chat')});
        entry.changed();
        window.__testCallMarkEntry = entry;
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    }"""
    probe_js = """() => {
        const r = el => { const b = el.getBoundingClientRect(); return {left: b.left, right: b.right, width: b.width}; };
        const markEl = document.querySelector('.call-mark');
        const hit = sel => {
            const el = document.querySelector(sel);
            if (!el) return {present: false};
            const b = el.getBoundingClientRect(), h = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
            return {present: true, hitSelf: Boolean(h && (h === el || el.contains(h))),
                hitCallMark: Boolean(h && markEl && (h === markEl || markEl.contains(h)))};
        };
        return {layout: document.documentElement.dataset.layout || null, stored: document.documentElement.style.getPropertyValue('--sidebar-width'),
            sidebar: r(document.getElementById('primary-sidebar')), rail: r(document.querySelector('#primary-sidebar .nav-rail')),
            content: r(document.getElementById('content')), mark: markEl && !markEl.hidden ? r(markEl) : null,
            controls: {add: hit('#chat-plus'), model: hit('#chat-model-mode .chat-model-head'),
                subagents: hit('#chat-subagents-mode .chat-subagents-head'),
                railSettings: hit('#primary-sidebar .nav-rail [data-nav-page="settings"]'),
                railDashboard: hit('#primary-sidebar .nav-rail [data-nav-page="dashboard"]'),
                markOpen: hit('.call-mark .call-mark-open'), markEnd: hit('.call-mark .call-mark-end')}};
    }"""

    page = seed_page(shell_ui, 720, 680)
    facts, saved = {'engine': shell_ui['engine']}, _route_sidebar_prefs(page, 540)
    try:
        _route_call_mark_assets(page)
        boot(page, shell_ui)
        page.wait_for_function("() => getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width').trim() === '540px'", timeout=20_000)
        for sel in ('#chat-plus', '#chat-model-mode .chat-model-head', '#chat-subagents-mode .chat-subagents-head'):
            page.wait_for_selector(sel)
        page.evaluate(mount_js)
        page.wait_for_timeout(600)
        facts['desktop_720'], facts['desktop_saved'] = page.evaluate(probe_js), list(saved)
        page.set_viewport_size({'width': 1280, 'height': 680})
        settle(page)
        facts['desktop_1280'], facts['desktop_saved_after_widen'] = page.evaluate(probe_js), list(saved)
    finally:
        evidence(tmp_path, f'shell-call-mark-geometry-{shell_ui["engine"]}', facts)
        try:
            page.evaluate('() => window.__testCallMarkEntry?.destroy?.()')
        except Exception:  # noqa: BLE001 - best-effort cleanup before closing
            pass
        page.close()
    d720, d1280 = facts['desktop_720'], facts['desktop_1280']
    assert d720['layout'] == 'docked' and d720['stored'] == '540px' and facts['desktop_saved'] == [], facts
    assert abs(d720['sidebar']['width'] - 400) <= 1 and abs(d720['content']['width'] - 320) <= 1, facts
    assert d720['mark'] and d720['mark']['left'] >= d720['rail']['right'] - 0.5, facts
    assert d720['mark']['right'] <= d720['sidebar']['right'] - 0.5 and d720['mark']['right'] <= d720['content']['left'] - 0.5, facts
    assert abs(d720['mark']['width'] - 336) <= 1, facts
    for key in ('add', 'model', 'subagents', 'railSettings', 'railDashboard', 'markOpen', 'markEnd'):
        assert d720['controls'][key]['present'] and d720['controls'][key]['hitSelf'], (key, d720['controls'][key])
    for key in ('add', 'model', 'subagents', 'railSettings', 'railDashboard'):
        assert not d720['controls'][key]['hitCallMark'], (key, d720['controls'][key])
    assert d1280['stored'] == '540px' and facts['desktop_saved_after_widen'] == [], facts
    assert abs(d1280['sidebar']['width'] - 580) <= 1 and abs(d1280['mark']['width'] - 516) <= 1, facts
    assert d1280['mark']['left'] >= d1280['rail']['right'] - 0.5 and d1280['mark']['right'] <= d1280['sidebar']['right'] - 0.5, facts


def test_off_room_call_mark_keeps_last_project_row_and_menu_reachable_without_permanent_clearance(shell_ui, tmp_path):
    """At 1100x646 with 10 projects (and in a 1100x480 short window where #primary-sidebar owns the one scroll),
    an active docked off-room .call-mark dynamically reserves clearance in .nav-projects so the last project row and
    its actions menu scroll cleanly above the badge and open via real pointer activation without hitting End call;
    stack layout, hide and destroy reserve zero extra clearance."""
    ten = PROJECTS[:10]
    mount_js = """async () => {
        await new Promise((ok, err) => document.head.append(Object.assign(document.createElement('link'),
            {rel: 'stylesheet', href: '/api/extensions/live_conversation/ui/test/call.css', onload: ok, onerror: () => err(new Error('call.css'))})));
        const { trackCallMark } = await import('/api/extensions/live_conversation/ui/test/call_mark.js');
        const ctrl = {onScreen: false, ended: 0, revealed: 0};
        ctrl.session = {active: true, liveSince: Date.now() - 12_000, end() { ctrl.ended += 1; }};
        ctrl.entry = trackCallMark({session: ctrl.session, label: () => 'Project 0: a descriptive project name',
            onScreen: () => ctrl.onScreen, reveal() { ctrl.revealed += 1; }, page: document.getElementById('page-chat')});
        ctrl.entry.changed();
        window.__testCallMark = ctrl;
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    }"""
    synced_height_js = """() => {
        const mark = document.querySelector('.call-mark');
        if (!mark || mark.hidden) return false;
        const actual = Math.ceil(mark.getBoundingClientRect().height);
        const cssVar = Number.parseFloat(document.documentElement.style.getPropertyValue('--call-mark-height'));
        return actual > 0 && Math.abs(cssVar - actual) <= 1;
    }"""
    probe_js = """() => {
        const box = el => { const b = el.getBoundingClientRect();
            return {top: b.top, bottom: b.bottom, left: b.left, right: b.right, width: b.width, height: b.height}; };
        const side = document.getElementById('primary-sidebar');
        const zone = document.getElementById('nav-projects');
        const list = document.getElementById('nav-projects-list');
        const markEl = document.querySelector('.call-mark');
        const markVisible = Boolean(markEl && !markEl.hidden && markEl.getClientRects().length);
        const item = list.querySelector('.nav-project-item[data-nav-project-id="gate-9"]');
        const kebab = item?.querySelector('.nav-project-kebab') || null;
        const hit = el => {
            if (!el) return {present: false};
            const b = el.getBoundingClientRect(), h = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
            return {present: true, hitSelf: Boolean(h && (h === el || el.contains(h))),
                hitCallMark: Boolean(h && markEl && (h === markEl || markEl.contains(h))),
                hit: h ? (h.tagName + (h.className ? '.' + String(h.className).trim().replace(/\\s+/g, '.') : '')) : null};
        };
        return {
            layout: document.documentElement.dataset.layout || null,
            rows: list.querySelectorAll('.nav-project-item').length,
            heightVar: document.documentElement.style.getPropertyValue('--call-mark-height').trim(),
            paddingBottom: Number.parseFloat(getComputedStyle(zone).paddingBottom) || 0,
            scrollPaddingBottom: Number.parseFloat(getComputedStyle(side).scrollPaddingBottom) || 0,
            listClientHeight: list.clientHeight,
            listScrollHeight: list.scrollHeight,
            listScrollTop: list.scrollTop,
            sideClientHeight: side.clientHeight,
            sideScrollHeight: side.scrollHeight,
            sideScrollTop: side.scrollTop,
            mark: markVisible ? box(markEl) : null,
            lastItem: item ? box(item) : null,
            lastKebab: kebab ? box(kebab) : null,
            hitRow: hit(item?.querySelector('.nav-project-row')),
            hitKebab: hit(kebab),
            ended: window.__testCallMark?.ended ?? 0,
            revealed: window.__testCallMark?.revealed ?? 0,
        };
    }"""

    def open_and_check_last_menu(page):
        last_item = page.locator('#nav-projects-list .nav-project-item[data-nav-project-id="gate-9"]')
        last_kebab = last_item.locator('.nav-project-kebab')
        last_item.hover()
        last_kebab.click()
        menu = page.locator('.project-row-menu[role="menu"]')
        menu.wait_for(state='visible')
        info = page.evaluate("""() => {
            const menu = document.querySelector('.project-row-menu[role="menu"]');
            const pin = menu?.querySelector('[data-prm="pin"]');
            const mark = document.querySelector('.call-mark');
            const b = pin?.getBoundingClientRect();
            const h = b ? document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2) : null;
            return {
                label: menu?.getAttribute('aria-label') || '',
                actions: [...(menu?.querySelectorAll('[data-prm]') || [])].map(el => el.dataset.prm),
                pinHit: Boolean(h && pin && (h === pin || pin.contains(h))),
                pinHitMark: Boolean(h && mark && (h === mark || mark.contains(h))),
                ended: window.__testCallMark?.ended ?? 0,
                revealed: window.__testCallMark?.revealed ?? 0,
            };
        }""")
        page.keyboard.press('Escape')
        menu.wait_for(state='detached')
        return info

    page = seed_page(shell_ui, 1100, 646)
    page.route('**/api/projects', lambda route: route.fulfill(json={'projects': ten}))
    page.route('**/api/state', lambda route: route.fulfill(json={
        'supervisor_ready': True, 'active_chat_activities': [], 'projects': ten,
        'project_chat_ids': [p['chat_id'] for p in ten]}))
    facts = {'engine': shell_ui['engine']}
    try:
        _route_call_mark_assets(page)
        boot(page, shell_ui)
        facts['before_call'] = page.evaluate(probe_js)

        page.evaluate(mount_js)
        page.wait_for_function(synced_height_js, timeout=5_000)
        settle(page)
        facts['docked_scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        page.evaluate("() => { const el = document.getElementById('nav-projects-list'); el.scrollTop = el.scrollHeight; }")
        settle(page)
        facts['docked_scrolled'] = page.evaluate(probe_js)
        facts['docked_menu'] = open_and_check_last_menu(page)

        page.set_viewport_size({'width': 1100, 'height': 480})
        page.wait_for_function(synced_height_js, timeout=5_000)
        settle(page)
        facts['short_scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        page.evaluate("() => { const el = document.getElementById('primary-sidebar'); el.scrollTop = el.scrollHeight; }")
        settle(page)
        facts['short_scrolled'] = page.evaluate(probe_js)
        facts['short_menu'] = open_and_check_last_menu(page)

        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function("() => document.documentElement.dataset.layout === 'stack'", timeout=5_000)
        settle(page)
        facts['stack_390'] = page.evaluate(probe_js)

        page.set_viewport_size({'width': 1100, 'height': 646})
        page.wait_for_function("() => document.documentElement.dataset.layout === 'docked'", timeout=5_000)
        page.wait_for_function(synced_height_js, timeout=5_000)
        settle(page)
        facts['restored_docked'] = page.evaluate(probe_js)

        page.evaluate("() => { window.__testCallMark.onScreen = true; window.__testCallMark.entry.changed(); }")
        settle(page)
        facts['after_hide'] = page.evaluate(probe_js)

        page.evaluate("() => { window.__testCallMark.onScreen = false; window.__testCallMark.entry.changed(); }")
        page.wait_for_function(synced_height_js, timeout=5_000)
        settle(page)
        page.evaluate("() => window.__testCallMark.entry.destroy()")
        settle(page)
        facts['after_destroy'] = page.evaluate(probe_js)
    finally:
        evidence(tmp_path, f'shell-call-mark-clearance-{shell_ui["engine"]}', facts)
        try:
            page.evaluate('() => window.__testCallMark?.entry?.destroy?.()')
        except Exception:  # noqa: BLE001 - best-effort cleanup before closing
            pass
        page.close()

    before, docked, short, stack = facts['before_call'], facts['docked_scrolled'], facts['short_scrolled'], facts['stack_390']
    assert before['rows'] == 10 and before['heightVar'] == '' and before['paddingBottom'] == 0, facts
    assert facts['docked_scrollers'] == ['nav-projects-list'], facts
    assert docked['mark'] and docked['paddingBottom'] >= docked['mark']['height'] + 43, facts
    assert docked['listScrollTop'] > 0 and docked['sideScrollTop'] == 0, facts
    assert docked['lastItem']['bottom'] <= docked['mark']['top'] - 6, facts
    assert docked['lastKebab']['bottom'] <= docked['mark']['top'] - 6, facts
    assert docked['hitRow']['hitSelf'] and not docked['hitRow']['hitCallMark'], facts
    assert docked['hitKebab']['hitSelf'] and not docked['hitKebab']['hitCallMark'], facts
    assert facts['docked_menu']['label'].startswith('Actions for Project 9: a descriptive project name'), facts
    assert facts['docked_menu']['actions'] == ['pin', 'rename', 'folder', 'delete'], facts
    assert facts['docked_menu']['pinHit'] and not facts['docked_menu']['pinHitMark'], facts
    assert facts['docked_menu']['ended'] == 0 and facts['docked_menu']['revealed'] == 0, facts

    assert facts['short_scrollers'] == ['primary-sidebar'], facts
    assert short['mark'] and short['paddingBottom'] >= short['mark']['height'] + 43, facts
    assert short['scrollPaddingBottom'] >= short['mark']['height'] + 43 and short['sideScrollTop'] > 0, facts
    assert short['lastItem']['bottom'] <= short['mark']['top'] - 6, facts
    assert short['lastKebab']['bottom'] <= short['mark']['top'] - 6, facts
    assert short['hitKebab']['hitSelf'] and not short['hitKebab']['hitCallMark'], facts
    assert facts['short_menu']['label'].startswith('Actions for Project 9: a descriptive project name'), facts
    assert facts['short_menu']['pinHit'] and facts['short_menu']['ended'] == 0, facts

    assert stack['layout'] == 'stack' and stack['mark'] and stack['mark']['top'] < 100, facts
    assert stack['paddingBottom'] == 0 and stack['scrollPaddingBottom'] == 0, facts

    assert facts['restored_docked']['mark'] and facts['restored_docked']['paddingBottom'] >= facts['restored_docked']['mark']['height'] + 43, facts
    for cleared in (facts['after_hide'], facts['after_destroy']):
        assert cleared['mark'] is None and cleared['heightVar'] == '', facts
        assert cleared['paddingBottom'] == 0 and cleared['scrollPaddingBottom'] == 0, facts
        assert abs(cleared['listClientHeight'] - before['listClientHeight']) <= 1, facts


DRAWER_CLOSE_ORDER = """() => {
    const side = document.getElementById('primary-sidebar');
    const close = document.getElementById('nav-drawer-close');
    const main = side?.querySelector('[data-nav-page="chat"]');
    const projects = document.getElementById('nav-projects');
    const follows = (a, b) => Boolean(a && b && (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING));
    const focusables = [...side.querySelectorAll('button, input, a[href], [tabindex]:not([tabindex="-1"])')]
        .filter((el) => !el.disabled && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden');
    const first = focusables[0] || null;
    return {
        closeCount: document.querySelectorAll('#nav-drawer-close').length,
        firstInRail: Boolean(side?.querySelector('.nav-rail > #nav-drawer-close:first-child')),
        firstFocusable: first ? (first.id || first.dataset.navPage || first.className) : null,
        beforeMain: follows(close, main),
        beforeProjects: follows(close, projects),
        landing: document.activeElement?.id || null,
        visible: Boolean(close && getComputedStyle(close).display !== 'none' && close.getClientRects().length > 0),
    };
}"""


def test_phone_drawer_close_leads_dom_and_focus_order_and_yields_to_telegram_fullscreen(shell_ui, tmp_path):
    """The phone drawer's one #nav-drawer-close sits first in DOM/AT and keyboard Tab order, closes the drawer and
    restores focus to its opener while keeping Escape and Back intact; in Telegram fullscreen CSS hides #nav-drawer-close
    so Telegram's own native Back owns closing the drawer."""
    page = seed_page(shell_ui, 390, 844)
    tg_page = None
    facts = {'engine': shell_ui['engine']}

    def open_from_toggle():
        toggle = page.locator('[data-mobile-nav-toggle]:visible').first
        toggle.focus()
        page.keyboard.press('Enter')
        page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
        rest(page)

    def closed_state():
        wait_quietly(page, "() => !document.getElementById('primary-sidebar').classList.contains('open') && window.__ouroLayers.size() === 0")
        wait_quietly(page, "() => Boolean(document.activeElement?.matches?.('[data-mobile-nav-toggle]'))")
        return page.evaluate("""() => ({
            open: document.getElementById('primary-sidebar').classList.contains('open'),
            layers: window.__ouroLayers.ids(),
            openerFocused: Boolean(document.activeElement?.matches?.('[data-mobile-nav-toggle]')),
        })""")

    try:
        boot(page, shell_ui)
        open_from_toggle()
        facts['order'] = page.evaluate(DRAWER_CLOSE_ORDER)
        if shell_ui['engine'] == 'chromium':
            page.keyboard.press('Tab')
            facts['first_tab'] = page.evaluate("() => document.activeElement?.id || null")
            page.keyboard.press('Enter')
            facts['after_key_close'] = closed_state()
            open_from_toggle()
        page.locator('#nav-drawer-close').click()
        facts['after_click_close'] = closed_state()

        open_from_toggle()
        page.keyboard.press('Escape')
        facts['after_escape'] = closed_state()

        open_from_toggle()
        wait_quietly(page, 'window.__ouroLayers?.historyArmed()')
        press_back(page, 'history')
        facts['after_back'] = closed_state()

        tg_page = seed_page(shell_ui, 390, 844, telegram=True)
        tg_page.add_init_script("""(() => {
            if (!window.Telegram?.WebApp) return;
            window.Telegram.WebApp.isFullscreen = true;
            window.Telegram.WebApp.safeAreaInset = {top: 47, bottom: 34, left: 0, right: 0};
            window.Telegram.WebApp.contentSafeAreaInset = {top: 44, bottom: 0, left: 0, right: 0};
        })()""")
        boot(tg_page, shell_ui)
        open_drawer(tg_page)
        settle(tg_page)
        facts['telegram_open'] = tg_page.evaluate("""() => {
            const close = document.getElementById('nav-drawer-close');
            return {
                fullscreen: document.documentElement.dataset.telegramFullscreen || null,
                closeDisplay: close ? getComputedStyle(close).display : null,
                closeRects: close ? close.getClientRects().length : -1,
                telegramBack: window.Telegram?.WebApp?.BackButton?.isVisible ?? null,
                open: document.getElementById('primary-sidebar').classList.contains('open'),
            };
        }""")
        press_back(tg_page, 'telegram')
        wait_quietly(tg_page, '() => window.__ouroLayers.size() === 0')
        settle(tg_page)
        facts['telegram_after_back'] = tg_page.evaluate("""() => ({
            open: document.getElementById('primary-sidebar').classList.contains('open'),
            layers: window.__ouroLayers.ids(),
            telegramBack: window.Telegram?.WebApp?.BackButton?.isVisible ?? null,
        })""")
    finally:
        evidence(tmp_path, f'shell-drawer-close-focus-{shell_ui["engine"]}', facts)
        page.close()
        if tg_page:
            tg_page.close()
    assert facts['order'] == {
        'closeCount': 1, 'firstInRail': True, 'firstFocusable': 'nav-drawer-close',
        'beforeMain': True, 'beforeProjects': True, 'landing': 'primary-sidebar', 'visible': True,
    }, facts
    if shell_ui['engine'] == 'chromium':
        assert facts['first_tab'] == 'nav-drawer-close', facts
        assert facts['after_key_close'] == {'open': False, 'layers': [], 'openerFocused': True}, facts
    for step in ('after_click_close', 'after_escape', 'after_back'):
        assert facts[step] == {'open': False, 'layers': [], 'openerFocused': True}, (step, facts)
    assert facts['telegram_open'] == {
        'fullscreen': 'active', 'closeDisplay': 'none', 'closeRects': 0, 'telegramBack': True, 'open': True,
    }, facts
    assert facts['telegram_after_back'] == {'open': False, 'layers': [], 'telegramBack': False}, facts


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


# What the panel shows: how many of its room's answers, whether its loading row is up (chat_render_batch.js), and how
# many rows of this browser's saved text of the room stand (chat_preview.js).
PANEL_SHOWS = """(panel, answers) => ({
    rows: answers.filter(answer => [...panel.querySelectorAll('.chat-messages .chat-bubble')]
        .some(row => row.textContent.includes(answer))).length,
    loading: [...panel.querySelectorAll('.chat-load-older[aria-busy="true"]')]
        .some(row => !row.hidden && row.getClientRects().length > 0),
    saved: panel.querySelectorAll('[data-saved-preview] .chat-bubble').length})"""
# Seek the real slide: a busy runner must not turn missed compositor frames into a missing-motion failure. `setOff`
# is the time from just before the open was asked to the slide's start: the panel's own wait begins after the ask.
PANEL_FRAMES = """({project, opening, answers}) => { const panel = document.getElementById('project-panel');
    const sample = """ + SAMPLE_TRANSITION + """, shows = """ + PANEL_SHOWS + """;
    let asked = 0, setOff = null;
    return sample({root: panel, target: () => panel, property: 'transform',
        ready: () => { const ready = panel.classList.contains('open') === opening;
                       if (ready && setOff === null) setOff = performance.now() - asked;
                       return ready; },
        act: () => { asked = performance.now();
                     if (opening) window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project}}));
                     else panel.querySelector('#project-panel-close').click(); },
        measure: () => ({left: panel.getBoundingClientRect().left, hidden: panel.hidden, ...shows(panel, answers)})})
        .then(seen => ({...seen, setOff})); }"""
# A room's read takes 150 ms, as on a real link: a panel that set off at the click would cross the window empty. A
# held read lands only when the test releases it, so the panel's bounded wait for its room has to end on its own.
HISTORY_READS = """(() => { const real = window.fetch, waiting = [];
    window.__historyReads = {hold: false, held: () => waiting.length, release: () => waiting.splice(0).forEach(go => go())};
    window.fetch = (input, init) => { const answer = real(input, init);
        if (!String(input?.url || input).includes('/api/chat/history')) return answer;
        const hold = window.__historyReads.hold;
        return answer.then((r) => new Promise((resolve) => hold ? waiting.push(() => resolve(r))
            : setTimeout(() => resolve(r), 150))); }; })()"""


@pytest.mark.parametrize('read', ['slow', 'held', 'saved'])
def test_docked_the_project_panel_comes_in_with_its_room_and_goes_back_with_it(shell_ui, tmp_path, read):
    """Owner 20.10 (Q2 A): docked, the panel comes in from the window's right edge once its room has painted (or at
    its read's bound, with its loading row) and goes back there whole: samples along both real transitions show where
    it is and what it shows, from the edge to Main and back. A 150 ms read normally paints first, though a busy runner
    may reach the bound first, which is the designed branch, not a defect; a held read always reaches the bound. A
    room this browser kept the saved text of (owner 25.11, 2A) paints it at once: with its read held, the panel sets
    off at once with those words (chat_preview.js, app.js arrivePanel)."""
    app_js = (setup_browser.WEB / 'app.js').read_text(encoding='utf-8')
    bound = int(re.search(r'PANEL_ARRIVAL_BOUND_MS = (\d+);', app_js).group(1))
    page = seed_page(shell_ui, 1143, 790)
    page.emulate_media(reduced_motion='no-preference')
    rows = [{'role': 'assistant', 'text': f'Answer {i}. ' + 'Ordinary context. ' * 8,
             'content': f'Answer {i}.', 'ts': f'2026-10-14T09:{i:02d}:00Z'} for i in range(6)]
    answers = [row['content'] for row in rows]  # A startup System bubble is not the room's history.
    # The document proves its data profile (the shell fixture's meta): a read of another profile would never paint.
    page.route('**/api/chat/history*', lambda route: route.fulfill(json={'messages': rows, 'progress': [],
                                                                          'profile_scope': setup_browser.PREVIEW_SCOPE}))
    page.add_init_script(HISTORY_READS)
    if read == 'saved':
        room_key = json.dumps([setup_browser.PREVIEW_SCOPE, [PROJECTS[0]['chat_id'], PROJECTS[0]['id']]], separators=(',', ':'))
        entry = json.dumps({'read': 1, 'rows': [{'id': 'chat:s1', 'role': 'assistant', 'text': 'Saved answer words',
                                                 'speaker': 'Ouroboros'}]}, separators=(',', ':'))
        page.add_init_script(f"""(() => {{ try {{ localStorage.setItem('ouroboros.chat-preview.v1', '{{"v":3}}');
            localStorage.setItem({json.dumps('ouroboros.chat-preview.v1:' + room_key)}, {json.dumps(entry)}); }}
            catch {{ /* storage refused: the case then fails on its saved rows */ }} }})()""")
    facts = {'engine': shell_ui['engine'], 'read': read, 'bound': bound}
    room = f"() => ({PANEL_SHOWS})(document.getElementById('project-panel'), {json.dumps(answers)})"
    try:
        boot(page, shell_ui)
        width = page.evaluate('innerWidth')
        page.evaluate('hold => { window.__historyReads.hold = hold; }', read != 'slow')
        facts['opening'] = page.evaluate(PANEL_FRAMES, {'project': PROJECTS[0], 'opening': True, 'answers': answers})
        facts['held'] = page.evaluate('window.__historyReads.held()')
        page.evaluate('window.__historyReads.release()')
        wait_quietly(page, f'() => {{ const shows = ({room})(); return shows.rows === {len(answers)} && !shows.loading; }}')
        facts['room'] = page.evaluate(room)
        rest(page)
        page.locator('#project-panel-close').click(trial=True)
        facts['closing'] = page.evaluate(PANEL_FRAMES, {'opening': False, 'answers': answers})
        page.wait_for_function("() => document.getElementById('project-panel').hidden")
    finally:
        evidence(tmp_path, f'shell-panel-arrival-{read}-{shell_ui["engine"]}', facts)
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
    # The slide's first sample is the panel as it set off: its room already painted, or not (then it waited the bound).
    painted = opening[1]['rows'] == len(answers)
    if read != 'slow':
        assert facts['held'] >= 1 and not painted, ('its read was still held as it came in', facts)
    if read == 'saved':
        assert facts['opening']['setOff'] < bound, ('with its saved text painted it sets off at once', facts['opening'])
        assert all(f['saved'] == 1 and not f['rows'] for f in moving(arriving)), ('it comes in with that text', arriving)
    elif painted:
        assert all(f['rows'] == len(answers) for f in moving(arriving)), ('it sets off with its room painted', arriving)
    else:
        assert facts['opening']['setOff'] >= bound, ('without its room it waits until the bound', facts['opening'])
        seen = ''.join('R' if f['rows'] == len(answers) else 'L' if f['loading'] and not f['rows'] else '?'
                       for f in moving(arriving))
        assert re.fullmatch('L+R*', seen), ('at the bound it comes in with its loading row, then its room', seen, arriving)
    assert facts['room'] == {'rows': len(answers), 'loading': False, 'saved': 0}, ('its read lands in the room', facts['room'])
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
    """The retired toggle's keyboard home is Find; an unpin keeps its row, an absent row hands focus there."""
    page = seed_page(shell_ui, 1108, 680)
    pins = [PROJECTS[3]['id'], PROJECTS[7]['id']]
    current = {'state': live_state([], pins)}
    facts = {'engine': shell_ui['engine']}
    gone = "id => !document.querySelector(`[data-project-id=\"${id}\"]`)"

    def snapshot(projects):  # another window changes the feed
        current['state'] = dict(live_state([]), projects=projects)
        page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")

    try:
        page.route('**/api/state', lambda route: route.fulfill(json=current['state']))
        boot(page, shell_ui)
        assert page.locator('#nav-projects-toggle').count() == 0
        facts['initial'] = page.evaluate(ROWS)
        # Unpinning no longer hides a row: both the node and its open menu keep the keyboard.
        page.locator(f'[data-nav-project-id="{pins[0]}"] .nav-project-kebab').focus()
        page.keyboard.press('Enter')
        page.locator('body > .project-row-menu:not([data-leaving])').wait_for()
        facts['in_menu'] = page.evaluate("() => document.activeElement.dataset.prm || ''")
        page.evaluate('id => { window.keptProjectRow = document.querySelector(`[data-project-id="${id}"]`); }', pins[0])
        snapshot(live_state([], pins[1:])['projects'])
        page.wait_for_function('id => !document.querySelector(`[data-nav-project-id="${id}"]`).hasAttribute("data-pinned")', arg=pins[0])
        facts['unpin'] = page.evaluate("""() => ({same: window.keptProjectRow.isConnected,
            menu: document.querySelectorAll('.project-row-menu:not([data-leaving])').length,
            focus: document.activeElement.dataset.prm})""")
        # Once its row really leaves the feed, the menu closes and hands the keyboard to the list's first control.
        snapshot([p for p in live_state([], pins[1:])['projects'] if p['id'] != pins[0]])
        page.wait_for_function(gone, arg=pins[0], timeout=20_000)
        settle(page)
        facts['menus_after'] = page.locator('.project-row-menu:not([data-leaving])').count()
        facts['focus_after_menu'] = page.locator('.nav-find-input').evaluate('el => el === document.activeElement')
        # The keyboard on a row itself: an empty feed must hand it to Find too, never drop it on the body.
        page.locator(f'[data-project-id="{pins[1]}"]').focus()
        snapshot([])
        page.wait_for_function(gone, arg=pins[1], timeout=20_000)
        settle(page)
        facts['focus_after_row'] = page.locator('.nav-find-input').evaluate('el => el === document.activeElement')
        facts['empty_rows'] = page.evaluate(ROWS)
        setup_browser.capture(page, f'shell-sidebar-empty-focus-{shell_ui["engine"]}')
        snapshot(live_state([])['projects'])
        page.wait_for_function('n => (' + ROWS + ')().length === n', arg=len(PROJECTS))
        page.keyboard.press('ArrowDown')
        settle(page)
        facts['restored_rows'] = page.evaluate(ROWS)
        facts['reentered'] = page.evaluate(FOCUS)
        # A state read with the keyboard elsewhere must not steal it into the list.
        page.locator('#chat-input').focus()
        snapshot([])
        page.wait_for_function('() => (' + ROWS + ')().length === 0')
        facts['elsewhere'] = page.evaluate(FOCUS)
    finally:
        evidence(tmp_path, f'shell-sidebar-collapse-focus-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['initial'] == pins + [p['id'] for p in PROJECTS if p['id'] not in pins], facts
    assert facts['in_menu'] == 'pin' and facts['unpin'] == {'same': True, 'menu': 1, 'focus': 'pin'}, facts
    assert facts['menus_after'] == 0 and facts['focus_after_menu'] is True, facts
    assert facts['focus_after_row'] is True and facts['empty_rows'] == [], facts
    assert facts['restored_rows'] == [p['id'] for p in PROJECTS] and facts['reentered'] == 'project:gate-0', facts
    assert facts['elsewhere'] == 'chat-input', facts


ROWS = "() => [...document.querySelectorAll('#nav-projects-list > .nav-project-item')].filter((e) => !e.hidden).map((e) => e.querySelector('[data-project-id]').dataset.projectId)"


def test_collapsed_projects_ignore_the_census_and_the_open_one(shell_ui, tmp_path):
    """Find keeps only matching names through census changes and opening another room; their marks still update."""
    page = seed_page(shell_ui, 1108, 680)
    matched = [PROJECTS[3]['id'], PROJECTS[9]['id']]

    def state_for(live):
        state = live_state(live)
        for project in state['projects']:
            if project['id'] in matched:
                project['name'] = 'Match fixture: ' + project['name']
        return state

    current = {'state': state_for([matched[0]])}
    facts = {'engine': shell_ui['engine']}
    try:
        page.route('**/api/state', lambda route: route.fulfill(json=current['state']))
        page.route('**/api/projects*', lambda route: answer_projects(route, current['state']['projects']))
        boot(page, shell_ui)
        page.wait_for_function(MOVING, arg=1, timeout=20_000)
        page.locator('.nav-find-input').fill('match fixture')
        page.wait_for_function('wanted => JSON.stringify((' + ROWS + ')()) === JSON.stringify(wanted)', arg=matched)
        facts['one'] = page.evaluate(ROWS)
        for name, ids_now in (('three', [PROJECTS[3]['id'], PROJECTS[5]['id'], PROJECTS[9]['id']]), ('none', [])):
            current['state'] = state_for(ids_now)
            page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
            page.wait_for_function(MOVING, arg=len(set(ids_now) & set(matched)), timeout=20_000)
            settle(page)
            facts[name] = page.evaluate(ROWS)
        # Opening a room by id does not change the field's words or force its row into the results.
        open_project(page, 12)
        settle(page)
        facts['open'] = page.evaluate(ROWS)
        facts['query'] = page.locator('.nav-find-input').input_value()
        facts['retired'] = page.locator('#nav-projects-toggle, #nav-projects-count, .nav-projects-facts').count()
        assert page.locator('#project-panel').is_visible()
        page.locator('.nav-find-clear').click()
        page.wait_for_function('n => (' + ROWS + ')().length === n', arg=len(PROJECTS))
        facts['cleared'] = page.evaluate(ROWS)
    finally:
        evidence(tmp_path, f'shell-sidebar-collapse-census-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['one'] == facts['three'] == facts['none'] == facts['open'] == matched, facts
    assert facts['query'] == 'match fixture' and facts['retired'] == 0, facts
    assert facts['cleared'] == [p['id'] for p in PROJECTS], facts


# Selection dismissal regression coverage: kept in this module so browser fixtures remain
# owned by the shell-interaction test surface rather than imported from another test module.
def _select_first_bubble_for_dismissal(page):
    return page.evaluate("""() => {
        const bubble = document.querySelector('#chat-messages .chat-bubble');
        const text = [...(bubble?.childNodes || [])].flatMap((node) => {
            const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
            const out = []; while (walker.nextNode()) out.push(walker.currentNode); return out;
        }).find((node) => (node.textContent || '').length > 8);
        if (!text) return {length: 0, error: 'no selectable chat text'};
        const range = document.createRange(); range.setStart(text, 0);
        range.setEnd(text, Math.min(8, text.length));
        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
        return {length: selection.toString().length, text: selection.toString()};
    }""")


@pytest.mark.parametrize('target', ['sidebar', 'composer'])
def test_primary_press_dismisses_document_selection_without_breaking_controls(shell_ui, tmp_path, target):
    page = seed_page(shell_ui, 1100, 750)
    client, respond = saved_history(tmp_path)
    try:
        page.route('**/api/chat/history**', respond)
        boot(page, shell_ui)
        page.wait_for_selector('#chat-messages .chat-bubble')
        selected = _select_first_bubble_for_dismissal(page)
        assert selected['length'] > 0, selected
        if target == 'sidebar':
            page.locator('#chat-messages .chat-message-copy').last.click()
        else:
            page.locator('#chat-input').click()
            assert page.locator('#chat-input').evaluate('node => document.activeElement === node')
        page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
        assert page.evaluate('() => getSelection()?.toString() || ""') == ''
    finally:
        page.close()
        client.close()


def test_control_inside_selected_range_dismisses_before_activation(shell_ui, tmp_path):
    page = seed_page(shell_ui, 1100, 750)
    client, respond = saved_history(tmp_path)
    try:
        page.route('**/api/chat/history**', respond)
        boot(page, shell_ui)
        page.wait_for_selector('#chat-messages .chat-bubble')
        selected = page.evaluate("""() => {
            const bubble = document.querySelector('#chat-messages .chat-bubble');
            const button = document.createElement('button');
            button.type = 'button'; button.dataset.selectionTest = '1'; button.textContent = 'control';
            bubble.appendChild(button);
            const range = document.createRange(); range.selectNodeContents(bubble);
            const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
            return selection.toString().length;
        }""")
        assert selected > 0
        page.locator('[data-selection-test]').click()
        assert page.evaluate('() => getSelection()?.toString() || ""') == ''
    finally:
        page.close()
        client.close()


def test_touch_press_preserves_selection_for_native_scroll_and_handles(shell_ui, tmp_path):
    page = seed_page(shell_ui, 390, 844)
    client, respond = saved_history(tmp_path)
    try:
        page.route('**/api/chat/history**', respond)
        boot(page, shell_ui)
        page.wait_for_selector('#chat-messages .chat-bubble')
        selected = _select_first_bubble_for_dismissal(page)
        assert selected['length'] > 0, selected
        page.locator('#primary-sidebar').dispatch_event(
            'pointerdown', {'button': 0, 'isPrimary': True, 'pointerType': 'touch', 'clientX': 8, 'clientY': 200})
        assert page.evaluate('() => getSelection()?.toString() || ""') == selected['text']
    finally:
        page.close()
        client.close()
