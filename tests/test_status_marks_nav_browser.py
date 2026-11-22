"""The sidebar's status marks, the remembered Projects collapse and the Project row menu in the real SPA.

docs/DESIGN.md "Status marks" and "Shell": Main's row carries its room's mark in the Project rows' column, the ☰ says
what waits behind it with one dot, a Project row's menu opens by right-click, Shift+F10 or a long press and starts with
the row's mark and its words, and the collapse is the owner's choice on this device. Uses the static-server/API fixture
(test_subscription_setup_browser); never starts the runtime or reads live data. Synthetic CDP touch is Chromium's input
pipeline, not a claim about an iPhone WebView.
"""
from __future__ import annotations

import json
import re

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests._phone_touch import touch as phone_touch
from tests._work_projection_shared import work_fields
from tests.test_design_system_gate_browser import PROJECTS, rest, save_evidence, seed_page, settle

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]
CHATS = {p['id']: p['chat_id'] for p in PROJECTS}
MENU = 'body > .project-row-menu:not([data-leaving])'  # the open one: a closed menu's picture fades a moment


@pytest.fixture(params=['webkit', 'chromium'])
def nav_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


@pytest.fixture
def chromium_ui(request, monkeypatch):
    # CDP Input.dispatchTouchEvent exists only in Chromium; the WebKit long press is the pin suite's synthetic finger.
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', 'chromium')
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = 'chromium'
    return ui


def expect():
    return pytest.importorskip('playwright.sync_api').expect


def snapshot(work=(), asking=(), unread=(), pins=(), main='', main_revision=None, main_asking=False, question=''):
    """An /api/state body: work running in the ``work`` Projects, a question waiting for the owner in ``asking`` (with
    ``question``, an addressed one whose words the body's ``current_questions`` carry), three new messages in each
    ``unread`` Project, ``pins`` pinned, Main's own work in ``main`` (a census phase), a question of Main's own waiting
    for the owner (``main_asking``) and Main's room revision ``main_revision`` (its read cursor starts at 0, see
    ``open_app``); the work fields are the server's own projection of those rows."""
    rows = [dict(activity_id=f'work-{pid}', root_task_id=f'work-{pid}', project_id=pid, chat_id=CHATS[pid],
                 kind='managed_task', phase='working') for pid in work]
    rows += [dict(activity_id=f'ask-{pid}', root_task_id=f'ask-{pid}', project_id=pid, chat_id=CHATS[pid],
                  kind='managed_task', phase='working',
                  required_question=dict(wait_for_answer=True, owner_wait_state='waiting',
                                         **({'quiz_id': f'quiz-{pid}'} if question else {}))) for pid in asking]
    if main:
        rows.append(dict(activity_id='main-work', root_task_id='main-work', chat_id=1, kind='direct_chat', phase=main))
    if main_asking:
        rows.append(dict(activity_id='main-ask', root_task_id='main-ask', chat_id=1, kind='managed_task', phase='working',
                         required_question=dict(wait_for_answer=True, owner_wait_state='waiting')))
    projects = [dict(p, pinned=p['id'] in pins, visible_revision=3 if p['id'] in unread else 0) for p in PROJECTS]
    main_unread = {} if main_revision is None else {'main_visible_revision': main_revision}
    if question:
        names = {p['id']: p for p in PROJECTS}
        main_unread['current_questions_complete'] = True
        main_unread['current_questions'] = [dict(
            role='system', system_type='project_question_pointer', task_id=f'ask-{pid}', quiz_id=f'quiz-{pid}',
            quiz_state='open', wait_for_answer=True, project_id=pid, project_name=names[pid]['name'],
            project_chat_id=CHATS[pid], chat_id=1, ts='2026-11-15T00:00:00+00:00', question=question, options=[],
            text=f'Waiting for your answer in {names[pid]["name"]}') for pid in asking]
    return {'supervisor_ready': True, 'projects': projects, 'project_chat_ids': list(CHATS.values()),
            'active_chat_activities_complete': True, 'active_chat_activities': rows, **work_fields(rows), **main_unread}


def open_app(ui, width, height, body, touch=None, init=()):
    page = seed_page(ui, width, height, touch=touch)
    current = {'state': body}
    page.route('**/api/state', lambda route: route.fulfill(json=current['state']))
    # The shared read cursors, as the server keeps them: Main's starts read at 0, a write is clamped and monotonic.
    cursors = {'main_seen_revision': 0}

    def preferences(route):
        asked = route.request.post_data_json if route.request.method == 'POST' else None
        if isinstance((asked or {}).get('main_seen_revision'), int):
            held = int(current['state'].get('main_visible_revision') or 0)
            cursors['main_seen_revision'] = max(cursors['main_seen_revision'], min(asked['main_seen_revision'], held))
        route.fulfill(json={'project_seen_revision': {}, **cursors})
    page.route('**/api/ui/preferences', preferences)
    for script in init:
        page.add_init_script(script)
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input', state='attached')
    page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
    settle(page)

    def sync(body_now):
        """The next state snapshot (the 20 s poll would bring it; projects_changed asks for it now)."""
        current['state'] = body_now
        with page.expect_response(lambda response: response.url.endswith('/api/state')):
            page.evaluate("window.__ouroWs.emit('projects_changed', {})")
        settle(page)

    return page, sync


def mark_of(page, selector):
    return page.locator(selector).evaluate("""el => {
        const m = el.matches('.status-mark') ? el : el.querySelector('.status-mark');
        const i = m.querySelector('i'), r = i.getBoundingClientRect(), s = getComputedStyle(i);
        return {shape: m.dataset.shape, hue: m.dataset.hue, drawn: getComputedStyle(m).display !== 'none',
                fill: s.backgroundColor, ring: s.boxShadow, size: [r.width, r.height],
                box: (({left, right, top, bottom}) => ({left, right, top, bottom}))(m.getBoundingClientRect())};
    }""")


def picture_dot(page):
    """The dot on the picture of Ouroboros's card: its mark, the ring the sheet cuts it out with, the card's fill and
    the picture's box."""
    return page.locator('.nav-row-main').evaluate("""card => {
        const picture = card.querySelector('.nav-ouro-ava'), m = picture.querySelector('.status-mark');
        const i = m.querySelector('i'), r = i.getBoundingClientRect();
        const box = (el) => (({left, right, top, bottom}) => ({left, right, top, bottom}))(el.getBoundingClientRect());
        return {shape: m.dataset.shape, hue: m.dataset.hue, drawn: getComputedStyle(m).display !== 'none',
                fill: getComputedStyle(i).backgroundColor, ring: getComputedStyle(m).boxShadow, size: [r.width, r.height],
                card: getComputedStyle(card).backgroundColor, box: box(m), picture: box(picture)};
    }""")


def token(page, name):
    return page.evaluate("""name => { const probe = document.createElement('span'); probe.style.color = `var(${name})`;
        document.body.append(probe); const color = getComputedStyle(probe).color; probe.remove(); return color; }""", name)


def test_the_menu_button_says_what_waits_behind_it_in_both_places(nav_ui, tmp_path):
    """The ☰ (owner 19.10, 3A): an amber dot while Main or a Project waits for the owner, else a crimson dot while one
    has new messages, else nothing; its name says the same. A room's composer ☰ and a page header's ☰ alike."""
    ui = nav_ui
    page, sync = open_app(ui, 390, 844, snapshot())
    composer = '#page-chat .composer-menu[data-mobile-nav-toggle]'
    header = '#page-files .app-page-header [data-mobile-nav-toggle]'
    facts = {'engine': ui['engine']}

    def toggle(selector):
        button = page.locator(selector)
        return {**mark_of(page, selector), 'name': button.get_attribute('aria-label'),
                'button': button.evaluate("el => (({left, right, top, bottom}) => ({left, right, top, bottom}))"
                                          "(el.getBoundingClientRect())")}

    try:
        facts['amber'], facts['crimson'] = token(page, '--mark-owner'), token(page, '--mark-unread')
        facts['rest'] = toggle(composer)
        sync(snapshot(unread=[PROJECTS[4]['id']]))
        facts['unread'] = toggle(composer)
        sync(snapshot(unread=[PROJECTS[4]['id']], asking=[PROJECTS[6]['id']]))
        facts['owner'] = toggle(composer)
        # A page header's ☰ (Files): the same dot on its own corner.
        page.locator('#page-chat [data-mobile-nav-toggle]').click()
        page.locator('#primary-sidebar [data-nav-page="files"]').click()
        page.locator('#page-files.active').wait_for()
        rest(page)  # the card has come home: its ☰ is measured at its own size, not mid-glide
        facts['header_owner'] = toggle(header)
        # Main's own new messages (its room revision past the shared cursor), away from Main, light the ☰ too.
        sync(snapshot(main_revision=1))
        facts['main_unread'] = toggle(header)
        facts['main_badge'] = page.locator('[data-nav-page="chat"] .unread-badge').count()
        setup_browser.capture(page, f'nav-menu-button-{ui["engine"]}')
    finally:
        save_evidence(tmp_path, f'nav-menu-button-{ui["engine"]}', facts)
        page.close()
    assert facts['rest']['shape'] == 'none' and not facts['rest']['drawn'], facts
    assert facts['rest']['name'] == 'Open navigation', facts
    assert (facts['unread']['shape'], facts['unread']['hue']) == ('dot', 'unread'), facts
    assert facts['unread']['name'] == 'Open navigation · new messages', facts
    # The owner's move outranks new messages.
    assert (facts['owner']['shape'], facts['owner']['hue']) == ('dot', 'owner'), facts
    assert facts['owner']['name'] == 'Open navigation · something waits for you', facts
    assert (facts['header_owner']['shape'], facts['header_owner']['hue']) == ('dot', 'owner'), facts
    assert (facts['main_unread']['shape'], facts['main_unread']['hue'], facts['main_badge']) == ('dot', 'unread', 1), facts
    for case in ('unread', 'owner', 'header_owner', 'main_unread'):
        mark, button = facts[case], facts[case]['button']
        # One plain 7px dot, no ring around it, on the control's own top-right corner, clear of its middle.
        assert mark['size'] == [7, 7] and mark['ring'] == 'none', (case, mark)
        assert button['right'] - 9 <= mark['box']['right'] <= button['right'] + 1, (case, mark, button)
        assert button['top'] - 1 <= mark['box']['top'] <= button['top'] + 9, (case, mark, button)
    assert facts['unread']['fill'] == facts['crimson'] and facts['owner']['fill'] == facts['amber'], facts


def test_the_closed_drawer_rests_its_moving_dots_and_opening_it_starts_them(nav_ui, tmp_path):
    """On a phone the room card covers the closed drawer: its rows' moving dots rest instead of running unseen, and
    run again, on the page's one wave, while the drawer is open. Docked they always run."""
    ui = nav_ui
    working = PROJECTS[0]['id']
    page, _sync = open_app(ui, 390, 844, snapshot(work=[working]))
    dots = f'#primary-sidebar [data-project-id="{working}"] .status-mark > i'
    running = f"""() => [...document.querySelectorAll('{dots}')].map((i) => getComputedStyle(i).animationName)"""
    facts = {'engine': ui['engine']}
    try:
        page.wait_for_function(f"() => document.querySelector('{dots}')?.parentElement.dataset.shape === 'moving'")
        facts['closed'] = page.evaluate(running)
        page.locator('#page-chat [data-mobile-nav-toggle]').click()
        page.wait_for_function("() => document.getElementById('primary-sidebar').classList.contains('open')")
        settle(page)
        facts['open'] = page.evaluate(running)
        page.set_viewport_size({'width': 1280, 'height': 844})
        page.wait_for_function("() => document.documentElement.dataset.layout === 'docked'")
        settle(page)
        facts['docked'] = page.evaluate(running)
    finally:
        save_evidence(tmp_path, f'nav-drawer-dots-{ui["engine"]}', facts)
        page.close()
    assert facts['closed'] == ['none'] * 3, facts
    assert facts['open'] == ['status-mark-bounce'] * 3, facts
    assert facts['docked'] == ['status-mark-bounce'] * 3, facts


def test_a_closed_project_whose_row_the_collapse_hides_gives_the_focus_to_the_toggle(nav_ui, tmp_path):
    """The Projects list never collapses: closing a Project returns focus to its row, and no retired toggle exists."""
    ui = nav_ui
    facts = {'engine': ui['engine']}
    page, sync = open_app(ui, 1280, 844, snapshot())
    row = f'.nav-project-row[data-project-id="{PROJECTS[2]["id"]}"]'
    focused = "() => { const a = document.activeElement; return a?.id || a?.className || a?.tagName || ''; }"
    try:
        page.locator(row).focus()
        page.keyboard.press('Enter')
        page.locator('#project-panel:not([hidden])').wait_for()
        page.locator('#project-panel .chat-messages').first.focus()
        page.keyboard.press('Escape')
        page.wait_for_function("() => !document.body.classList.contains('project-panel-open')")
        settle(page)
        facts['closed'] = page.evaluate(focused)
        facts['toggle_count'] = page.locator('#nav-projects-toggle').count()
    finally:
        save_evidence(tmp_path, f'nav-close-focus-{ui["engine"]}', facts)
        page.close()
    assert 'nav-project-row' in facts['closed'] and facts['toggle_count'] == 0, facts


def test_mains_unread_count_sits_beside_its_mark_never_on_it(nav_ui, tmp_path):
    """Main's count (its own question to the owner is still open) never covers Main's mark or its label: at three
    digits, with a mouse, in the narrowest sidebar and on touch (where no ⋯ column separates them), and it keeps its
    place when the mark goes."""
    ui = nav_ui
    facts = {'engine': ui['engine']}
    for case, width, touch, sidebar in (('mouse', 1280, False, ''), ('narrow', 1280, False, '180px'),
                                        ('touch', 390, True, '')):
        page, sync = open_app(ui, width, 844, snapshot(main='thinking'), touch=touch)
        try:
            if sidebar:
                page.evaluate("w => document.documentElement.style.setProperty('--sidebar-width', w)", sidebar)
            drawer = "() => window.__ouroLayers.top() === 'drawer'"
            if touch:
                page.locator('#page-chat [data-mobile-nav-toggle]').click()
                page.wait_for_function(drawer)
            page.locator('#primary-sidebar [data-nav-page="files"]').click()
            page.locator('#page-files.active').wait_for()
            # Away from Main its unread messages count: a hundred and twenty read 99+.
            sync(snapshot(main='thinking', main_revision=120))
            if touch:
                page.locator('#page-files [data-mobile-nav-toggle]').click()
                page.wait_for_function(drawer)
                page.wait_for_timeout(300)
            read = """() => { const row = document.querySelector('.nav-row-main'), badge = row.querySelector('.unread-badge');
                const box = (el) => (({left, right, top, bottom}) => ({left, right, top, bottom}))(el.getBoundingClientRect());
                const mark = row.querySelector('.nav-mark');
                return {text: badge.textContent, badge: box(badge), label: box(row.querySelector('.nav-row-label')),
                        mark: mark.hidden ? null : box(mark)}; }"""
            facts[case] = page.evaluate(read)
            sync(snapshot())
            settle(page)
            facts[case]['rest'] = page.evaluate(read)
        finally:
            save_evidence(tmp_path, f'nav-main-count-{ui["engine"]}', facts)
            page.close()
    for case in ('mouse', 'narrow', 'touch'):
        seen, rest = facts[case], facts[case]['rest']
        assert seen['text'] == '99+' and seen['mark'], facts
        if case in ('mouse', 'narrow'):
            # The narrow card keeps Main Chat on its second line (783b1cf34, c374668b7).
            assert seen['label']['top'] >= seen['badge']['bottom'], (case, facts)
            assert seen['label']['right'] == pytest.approx(seen['mark']['right'], abs=0.5), (case, facts)
        else:
            assert seen['label']['right'] <= seen['badge']['left'] + 0.5, (case, facts)
        assert seen['badge']['right'] <= seen['mark']['left'] - 2, (case, facts)
        assert rest['mark'] is None, facts
        assert all(abs(rest['badge'][side] - seen['badge'][side]) <= 0.5 for side in seen['badge']), (case, facts)


COLUMNS = """work => {
    const box = (selector) => { const el = document.querySelector(selector); if (!el) return null;
        const r = el.getBoundingClientRect(); return {left: r.left, right: r.right, text: el.textContent.trim()}; };
    return {main_count: box('[data-nav-page="chat"] .unread-badge'), main_mark: box('[data-nav-page="chat"] .nav-mark'),
            count: box('#nav-projects-count'), header_mark: box('.nav-projects-facts .status-mark'),
            row_mark: box(`[data-project-id="${work}"] .status-mark`), kebab: box(`[data-project-id="${work}"] ~ .nav-project-kebab`),
            add: box('#nav-projects-add'), chevron: box('#nav-projects-toggle .nav-chevron')};
}"""


def test_main_and_the_projects_header_share_the_rows_columns(nav_ui, tmp_path):
    """Main has its own count; each Project row carries its own mark, and the removed Projects header has no total."""
    ui = nav_ui
    facts = {'engine': ui['engine']}
    work, asking, unread = PROJECTS[2]['id'], PROJECTS[6]['id'], [PROJECTS[1]['id'], PROJECTS[4]['id']]
    cases = (('mouse', 1280, False, '', 2), ('narrow', 1280, False, '180px', 120), ('touch', 390, True, '', 2))
    for case, width, touch, sidebar, main_revision in cases:
        page, sync = open_app(ui, width, 844, snapshot(work=[work], asking=[asking], main='thinking'), touch=touch)
        try:
            if sidebar:
                page.evaluate("w => document.documentElement.style.setProperty('--sidebar-width', w)", sidebar)
            drawer = "() => window.__ouroLayers.top() === 'drawer'"
            if touch:
                page.locator('#page-chat [data-mobile-nav-toggle]').click()
                page.wait_for_function(drawer)
            page.locator('#primary-sidebar [data-nav-page="files"]').click()
            page.locator('#page-files.active').wait_for()
            sync(snapshot(work=[work], asking=[asking], unread=unread, main='thinking', main_revision=main_revision))
            if touch:
                page.locator('#page-files [data-mobile-nav-toggle]').click()
                page.wait_for_function(drawer)
                page.wait_for_timeout(300)
            facts[case] = page.evaluate(COLUMNS, work)
            setup_browser.capture(page, f'nav-columns-{case}-{ui["engine"]}')
        finally:
            save_evidence(tmp_path, f'nav-columns-{ui["engine"]}', facts)
            page.close()
    expected_main = {'mouse': '2', 'narrow': '99+', 'touch': '2'}
    for case in ('mouse', 'narrow', 'touch'):
        seen = facts[case]
        assert seen['main_count'] and seen['main_count']['text'] == expected_main[case], (case, facts)
        assert seen['count'] is None and seen['header_mark'] is None, (case, facts)
        assert seen['main_mark'] and seen['row_mark'], (case, facts)
        middle = (seen['main_mark']['left'] + seen['main_mark']['right']) / 2
        assert (seen['row_mark']['left'] + seen['row_mark']['right']) / 2 == pytest.approx(middle, abs=0.5), (case, facts)
    for case in ('mouse', 'narrow'):  # + keeps the ⋯ column whole, the narrowest sidebar included
        seen = facts[case]
        assert seen['add'] and seen['kebab'], (case, facts)
        assert (seen['add']['left'] + seen['add']['right']) / 2 == pytest.approx(
            (seen['kebab']['left'] + seen['kebab']['right']) / 2, abs=0.5), (case, facts)
    assert facts['touch']['add'] and facts['touch']['chevron'] is None, facts


QUESTION = '**The line is up.**\n\nOne step is `yours`: switch the local test on in Circles, then I run the trial and finish the merge.'
QUESTION_WORDS = 'The line is up. One step is yours: switch the local test on in Circles, then I run the trial and finish the merge.'


def test_the_picture_on_ouroboross_card_says_the_owners_move_for_main_and_the_projects(nav_ui, tmp_path):
    """Ouroboros's picture on his card stands for the rooms the list may hold out of sight (owner 15.11, 6A): while a
    Project asks, the 7px amber fact dot stands on the picture's top-right rim, ringed in the card's own fill, and the
    card's words for readers say a Project waits and its tooltip that something does; work or new messages alone
    never light it; Main's own question lights it beside Main's own mark. The waiting Project's row says its
    question in the question's own opening words. Docked and in the phone's open drawer alike."""
    ui = nav_ui
    facts = {'engine': ui['engine']}
    asking = PROJECTS[2]['id']
    for width, touch in ((1280, False), (390, True)):
        page, sync = open_app(ui, width, 844, snapshot(work=[asking], main='thinking'), touch=touch)
        try:
            if width < 720:
                page.locator('#page-chat [data-mobile-nav-toggle]').click()
                page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
                page.wait_for_timeout(300)
            main = page.locator('.nav-row-main')
            words = page.locator('.nav-row-main .nav-row-words')
            picture, own = '.nav-row-main .nav-ava-mark', '.nav-row-main .nav-mark'
            facts[width] = {'amber': token(page, '--mark-owner'), 'work': mark_of(page, picture)}
            sync(snapshot(asking=[asking]))
            facts[width]['asks'] = picture_dot(page)
            facts[width]['asksMain'] = mark_of(page, own)
            facts[width]['asksWords'] = words.text_content()
            facts[width]['asksTitle'] = main.get_attribute('title')
            expect()(main).to_have_accessible_name(
                re.compile(r'^Ouroboros Main Chat\s*·\s*Something in a project waits for you$'))
            # The waiting Project's own line: the sentence of its state while the read carries no words of the
            # question, the question's opening words once it does (one line, cut by the row's width, the room's
            # primary ink), and the row's name ends with the same words.
            row = page.locator(f'[data-project-id="{asking}"]')
            line = """el => ({text: el.textContent, kind: el.dataset.kind, ink: getComputedStyle(el).color,
                           oneLine: el.getBoundingClientRect().height <= parseFloat(getComputedStyle(el).lineHeight) + 1,
                           inside: el.getBoundingClientRect().right <= el.closest('.nav-project-row').getBoundingClientRect().right})"""
            facts[width]['sentence'] = row.locator('.nav-project-says').evaluate(line)
            sync(snapshot(asking=[asking], question=QUESTION))
            facts[width]['asked'] = row.locator('.nav-project-says').evaluate(line)
            facts[width]['askedName'] = row.get_attribute('aria-label')
            facts[width]['ink'] = token(page, '--text-primary')
            # A read that vouches for its questions drops the one it no longer lists (a read that does not keeps it).
            answered = dict(current_questions=[], current_questions_complete=True)
            sync({**snapshot(unread=[asking]), **answered})
            facts[width]['answeredLine'] = row.locator('.nav-project-says').text_content()
            facts[width]['unread'] = mark_of(page, picture)
            facts[width]['unreadWords'] = words.text_content()
            facts[width]['unreadTitle'] = main.get_attribute('title')
            expect()(main).to_have_accessible_name('Ouroboros Main Chat')
            sync({**snapshot(main_asking=True), **answered})
            facts[width]['mainAsks'] = {'picture': mark_of(page, picture), 'main': mark_of(page, own),
                                        'words': words.text_content(), 'title': main.get_attribute('title')}
            sync({**snapshot(), **answered})
            facts[width]['rest'] = {'picture': mark_of(page, picture), 'title': main.get_attribute('title')}
        finally:
            save_evidence(tmp_path, f'nav-card-picture-{ui["engine"]}', facts)
            page.close()
    for width in (1280, 390):
        seen = facts[width]
        assert not seen['work']['drawn'], facts
        asks = seen['asks']
        assert (asks['shape'], asks['hue'], asks['drawn'], asks['size']) == ('dot', 'owner', True, [7, 7]), facts
        assert asks['fill'] == seen['amber'] and asks['ring'] == f"{asks['card']} 0px 0px 0px 2px", facts
        assert abs(asks['box']['right'] - asks['picture']['right'] - 1) <= 0.5, facts
        assert abs(asks['picture']['top'] - asks['box']['top'] - 1) <= 0.5, facts
        assert not seen['asksMain']['drawn'], facts
        assert seen['asksWords'].strip() == '· Something in a project waits for you', facts
        assert seen['asksTitle'] == 'Main Ouroboros chat · something waits for you', facts
        assert (seen['sentence']['text'], seen['sentence']['kind']) == ('Waiting for your answer', 'ask'), facts
        asked = seen['asked']
        assert (asked['text'], asked['kind']) == (QUESTION_WORDS, 'ask'), facts
        assert asked['oneLine'] and asked['inside'] and asked['ink'] == seen['ink'], facts
        assert seen['askedName'].endswith(f' · {QUESTION_WORDS}'), facts
        assert seen['answeredLine'] == '', facts
        assert not seen['unread']['drawn'] and seen['unreadWords'] == '', facts
        assert seen['unreadTitle'] == 'Main Ouroboros chat', facts
        mine = seen['mainAsks']
        assert (mine['picture']['shape'], mine['picture']['hue'], mine['picture']['drawn']) == ('dot', 'owner', True), facts
        assert mine['main']['hue'] == 'owner' and mine['main']['drawn'], facts
        assert 'in a project' not in mine['words'] and mine['title'] == 'Main Ouroboros chat · something waits for you', facts
        assert not seen['rest']['picture']['drawn'] and seen['rest']['title'] == 'Main Ouroboros chat', facts


def test_mains_row_carries_its_rooms_mark_in_the_project_rows_column(nav_ui, tmp_path):
    ui = nav_ui
    facts = {'engine': ui['engine']}
    for width, touch in ((1280, False), (390, True)):
        page, sync = open_app(ui, width, 844, snapshot(work=[PROJECTS[2]['id']], main='thinking'), touch=touch)
        try:
            if width < 720:
                page.locator('#page-chat [data-mobile-nav-toggle]').click()
                page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
                page.wait_for_timeout(300)
            main = page.locator('.nav-row-main')
            project = f'[data-project-id="{PROJECTS[2]["id"]}"]'
            # The chosen row's ink fades in (the 120 ms colour motion), and its mark inherits every step of it: read
            # both once the row's own transition has finished.
            page.wait_for_function("() => { const row = document.querySelector('.nav-row-main');"
                                   " return row && row.getAnimations().length === 0; }")
            facts[width] = {'main': mark_of(page, '.nav-row-main .nav-mark'), 'project': mark_of(page, project),
                            'mainInk': main.evaluate('el => getComputedStyle(el).color'),
                            'words': page.locator('.nav-row-main .nav-row-words').text_content()}
            expect()(main).to_have_accessible_name(re.compile(r'^Ouroboros\s+Main Chat\s*·\s*Thinking'))
            # Main's own mark rests while only a Project asks (the picture's dot says that wait: the test above).
            sync(snapshot(asking=[PROJECTS[2]['id']]))
            facts[width]['rest'] = mark_of(page, '.nav-row-main .nav-mark')
        finally:
            save_evidence(tmp_path, f'nav-main-mark-{ui["engine"]}', facts)
            page.close()
    for width in (1280, 390):
        main, project = facts[width]['main'], facts[width]['project']
        assert (main['shape'], main['hue']) == ('moving', 'ink'), facts
        # The row's own ink: the chosen Main row's crimson.
        assert main['fill'] == facts[width]['mainInk'], facts
        # Both marks end at one x by construction, whether a ⋯ column follows (a mouse) or not (touch).
        assert abs(main['box']['right'] - project['box']['right']) <= 0.5, facts
        assert abs(main['box']['left'] - project['box']['left']) <= 0.5, facts
        assert facts[width]['words'].strip() == '· Thinking', facts
        assert not facts[width]['rest']['drawn'], facts


def test_the_collapse_is_remembered_on_this_device_and_only_there(nav_ui, tmp_path):
    ui = nav_ui
    body = snapshot(pins=[PROJECTS[5]['id']])
    facts = {'engine': ui['engine']}
    rows = "() => [...document.querySelectorAll('#nav-projects-list > .nav-project-item')].filter(e => !e.hidden).length"
    expanded = "() => document.getElementById('nav-projects-toggle')?.getAttribute('aria-expanded') || null"
    page, _sync = open_app(ui, 1280, 800, body)
    try:
        facts['first'] = [page.evaluate(expanded), page.evaluate(rows)]
        facts['toggle'] = page.locator('#nav-projects-toggle').count()
        ui['errors'].watch(page)  # the reload may cut off this page's own reads; the other pages stay strict
        page.reload()
        page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
        settle(page)
        facts['reloaded'] = [page.evaluate(expanded), page.evaluate(rows)]
    finally:
        page.close()
    # Another browser (its own storage) never chose: it starts expanded.
    other, _ = open_app(ui, 1280, 800, body)
    try:
        facts['other'] = [other.evaluate(expanded), other.evaluate(rows)]
    finally:
        other.close()
    # Storage that refuses (a private window, a locked-down WebView) has the same permanent feed.
    refusing = ("for (const name of ['getItem', 'setItem', 'removeItem']) Storage.prototype[name] = function () {"
                " throw new DOMException('storage denied', 'SecurityError'); };")
    locked, _ = open_app(ui, 1280, 800, body, init=[refusing])
    try:
        facts['locked'] = [locked.evaluate(expanded), locked.evaluate(rows), locked.locator('#nav-projects-toggle').count()]
    finally:
        save_evidence(tmp_path, f'nav-collapse-memory-{ui["engine"]}', facts)
        locked.close()
    everything = len(PROJECTS)
    assert facts['first'] == [None, everything], facts
    assert facts['toggle'] == 0, facts
    assert facts['reloaded'] == [None, everything], facts
    assert facts['other'] == [None, everything], facts
    assert facts['locked'] == [None, everything, 0], facts
    assert not ui['errors'].unexpected(), ui['errors'].report()


STATUS = """menu => {
    const line = menu.firstElementChild;
    const mark = line.querySelector('.status-mark');
    return {role: line.getAttribute('role'), words: line.textContent, word: line.querySelector('.project-row-menu-word')?.textContent,
            shape: mark?.dataset.shape, hue: mark?.dataset.hue, name: menu.getAttribute('aria-label'),
            items: [...menu.querySelectorAll('[role="menuitem"]')].map(i => i.dataset.prm),
            focused: document.activeElement.dataset.prm || document.activeElement.className,
            box: (({left, top}) => ({left, top}))(menu.getBoundingClientRect())};
}"""
ARIA = """sel => { const row = document.querySelector(sel), kebab = row.parentElement.querySelector('.nav-project-kebab');
    return {row: [row.getAttribute('aria-haspopup'), row.getAttribute('aria-expanded')],
            kebab: [kebab.getAttribute('aria-haspopup'), kebab.getAttribute('aria-expanded')]}; }"""


def test_a_rows_menu_opens_by_right_click_or_keyboard_and_says_the_rows_state(nav_ui, tmp_path):
    ui = nav_ui
    working, asking = PROJECTS[1]['id'], PROJECTS[3]['id']
    page, _sync = open_app(ui, 1280, 800, snapshot(work=[working], asking=[asking], unread=[working, asking]))
    row = f'.nav-project-row[data-project-id="{working}"]'
    facts = {'engine': ui['engine']}
    try:
        box = page.locator(row).bounding_box()
        point = (box['x'] + 40, box['y'] + box['height'] / 2)
        page.mouse.click(*point, button='right')
        page.locator(MENU).wait_for()
        facts['right'] = page.locator(MENU).evaluate(STATUS)
        facts['right_aria'] = page.evaluate(ARIA, row)
        page.keyboard.press('Escape')
        page.locator(MENU).wait_for(state='detached')
        facts['right_back'] = page.evaluate("() => document.activeElement.dataset.projectId || ''")
        # Shift+F10 on the focused row: under the row, the owner's move outranking its new messages.
        asking_row = f'.nav-project-row[data-project-id="{asking}"]'
        page.locator(asking_row).focus()
        page.keyboard.press('Shift+F10')
        page.locator(MENU).wait_for()
        facts['keyboard'] = page.locator(MENU).evaluate(STATUS)
        facts['keyboard_aria'] = page.evaluate(ARIA, asking_row)
        facts['anchor'] = page.locator(asking_row).bounding_box()
        page.keyboard.press('Escape')
        # The ⋯ is the one control that says it has a popup.
        page.locator(row).hover()
        page.locator(f'[data-nav-project-id="{working}"] .nav-project-kebab').click()
        page.locator(MENU).wait_for()
        facts['kebab_aria'] = page.evaluate(ARIA, row)
        page.keyboard.press('Escape')
        facts['kebab_closed'] = page.evaluate(ARIA, row)
        setup_browser.capture(page, f'nav-row-menu-{ui["engine"]}')
    finally:
        save_evidence(tmp_path, f'nav-row-menu-{ui["engine"]}', facts)
        page.close()
    name = PROJECTS[1]['name']
    right = facts['right']
    assert right['role'] == 'none' and right['items'] == ['pin', 'rename', 'folder', 'delete'] and right['focused'] == 'pin', facts
    assert (right['shape'], right['hue'], right['word']) == ('moving', 'unread', 'Working'), facts
    assert right['words'] == 'Working · new messages', facts
    assert right['name'] == f'Actions for {name} · Working · new messages', facts
    assert abs(right['box']['left'] - point[0]) <= 1 and abs(right['box']['top'] - point[1]) <= 1, facts
    assert facts['right_aria']['row'] == [None, None] and facts['right_back'] == working, facts
    keyboard = facts['keyboard']
    assert (keyboard['shape'], keyboard['hue'], keyboard['word']) == ('dot', 'owner', 'Waiting for your answer'), facts
    assert keyboard['words'] == 'Waiting for your answer · new messages', facts
    assert keyboard['box']['top'] >= facts['anchor']['y'] + facts['anchor']['height'], 'opened under the row'
    assert facts['keyboard_aria']['row'] == [None, None], facts
    assert facts['kebab_aria']['kebab'] == ['menu', 'true'] and facts['kebab_aria']['row'] == [None, None], facts
    assert facts['kebab_closed']['kebab'] == ['menu', 'false'], facts


def test_a_long_press_opens_a_rows_menu_on_touch_where_the_dots_take_no_column(chromium_ui, tmp_path):
    ui = chromium_ui
    sending = PROJECTS[2]['id']
    page, _sync = open_app(ui, 390, 844, snapshot(work=[sending], unread=[sending]))
    row = f'.nav-project-row[data-project-id="{sending}"]'
    facts = {}
    try:
        page.locator('#page-chat [data-mobile-nav-toggle]').click()
        page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
        page.wait_for_timeout(300)
        facts['coarse'] = page.evaluate("matchMedia('(pointer: coarse)').matches")
        facts['layout'] = page.locator(row).evaluate("""el => { const item = el.parentElement,
            kebab = item.querySelector('.nav-project-kebab'), k = kebab.getBoundingClientRect();
            return {kebab: [k.width, k.height], rowRight: el.getBoundingClientRect().right,
                    itemRight: item.getBoundingClientRect().right, select: getComputedStyle(item).userSelect}; }""")
        box = page.locator(row).bounding_box()
        x, y = box['x'] + 60, box['y'] + box['height'] / 2
        # A finger that moves is a scroll or a drag, never a long press.
        phone_touch(page, x, y, 600, move=(0, 30))
        facts['moved'] = page.locator(MENU).count()
        phone_touch(page, x, y, 650)
        page.locator(MENU).wait_for()
        facts['menu'] = page.locator(MENU).evaluate(STATUS)
        facts['aria'] = page.evaluate(ARIA, row)
        update = f'/api/projects/{sending}/update'
        facts['pressed_by_lift'] = [body for path, body in ui['posts'] if path == update]
        setup_browser.capture(page, 'nav-long-press-menu')
        page.wait_for_timeout(500)
        with page.expect_request(lambda request: request.url.endswith(update)) as pressed:
            page.locator(f'{MENU} [data-prm="pin"]').tap()
        facts['pressed'] = pressed.value.post_data_json
        # The shared fake returns {} for this POST, so the save settles as unconfirmed.
        # While it is pending, Pin is disabled and Rename correctly receives the menu's focus.
        expect()(page.locator('#toast-stack')).to_contain_text('Pin state was not confirmed')
        # The keyboard still has the ⋯: focused, it is drawn again with its column.
        kebab = page.locator(f'[data-nav-project-id="{sending}"] .nav-project-kebab')
        kebab.focus()
        drawn = """el => { const k = el.getBoundingClientRect();
            return {size: [k.width, k.height], rowRight: el.previousElementSibling.getBoundingClientRect().right,
                    left: k.left}; }"""
        facts['focused_kebab'] = kebab.evaluate(drawn)
        # It opens the menu by the keyboard and stays drawn under it while the focus is in the menu.
        page.keyboard.press('Enter')
        page.locator(MENU).wait_for()
        expect()(kebab).to_have_attribute('aria-expanded', 'true')  # the menu can paint a frame before its trigger says so
        facts['kebab_under_menu'] = {**kebab.evaluate(drawn), 'expanded': kebab.get_attribute('aria-expanded'),
                                     'focus': page.evaluate("() => document.activeElement.dataset.prm || ''")}
    finally:
        save_evidence(tmp_path, 'nav-long-press', facts)
        page.close()
    assert facts['coarse'] is True, facts
    assert facts['layout']['kebab'] == [0, 0] and facts['layout']['rowRight'] == facts['layout']['itemRight'], facts
    assert facts['layout']['select'] == 'none', facts
    assert facts['moved'] == 0, facts
    menu = facts['menu']
    assert menu['role'] == 'none' and menu['words'] == 'Working · new messages' and menu['focused'] == 'pin', facts
    assert (menu['shape'], menu['hue']) == ('moving', 'unread'), facts
    assert abs(menu['box']['left'] - x) <= 1 or menu['box']['left'] < x, 'at the finger, kept inside the window'
    assert facts['aria']['row'] == [None, None] and facts['aria']['kebab'] == [None, None], facts
    assert facts['pressed_by_lift'] == [], 'the lifting finger pressed nothing under it'
    assert facts['pressed'] == {'pinned': True}, 'a deliberate tap afterwards presses the item'
    assert facts['focused_kebab']['size'] == [44, 44], facts
    assert facts['focused_kebab']['rowRight'] - (
        facts['focused_kebab']['left'] + facts['focused_kebab']['size'][0]) == pytest.approx(2, abs=0.5), facts
    under = facts['kebab_under_menu']
    assert under['size'] == [44, 44] and under['expanded'] == 'true' and under['focus'] == 'pin', facts


def test_the_consciousness_chip_says_a_campaign_by_its_word_and_paints_no_tone(nav_ui, tmp_path):
    ui = nav_ui
    # The host's own sentence for this state (consciousness_view.py): the chip's label says it.
    body = {**snapshot(), 'evolution_enabled': True, 'bg_consciousness_enabled': False,
            'bg_consciousness_state': {'status': 'disabled', 'detail': 'Off: no own wakes.'}}
    page, _sync = open_app(ui, 1280, 800, body)
    try:
        chip = page.locator('#page-chat .cx-chip')
        expect()(chip).to_contain_text('Evolving')
        facts = chip.evaluate("""el => ({after: getComputedStyle(el, '::after').content,
            background: getComputedStyle(el).backgroundColor, mark: el.querySelector('.status-mark').dataset.shape,
            label: el.getAttribute('aria-label')})""")
    finally:
        page.close()
    assert facts['after'] in ('none', 'normal', ''), facts
    assert facts['background'] not in ('rgba(0, 0, 0, 0)', 'transparent'), 'a capsule of the bar'
    assert facts['mark'] == 'none', 'a campaign that goes on by itself draws no mark, never amber'
    assert facts['label'] == 'Consciousness: Off: no own wakes. · evolution campaign running', facts
    print(json.dumps(facts))
