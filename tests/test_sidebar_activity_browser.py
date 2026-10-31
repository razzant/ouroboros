"""Real SPA sidebar status marks over the server's work projection, socket lifecycle, CSS and navigation preservation.

Uses the existing static-document fixture; never starts or bootstraps Ouroboros.
All runtime replies are synthetic: each census row lands in its Project's room, and the work fields are the
server's own projection of those rows. Run with either fixture browser engine.
"""
from __future__ import annotations

import copy
import json
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests._work_projection_shared import work_fields
from tests.test_design_system_gate_browser import rest

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WORK = '.nav-project-row[data-project-id="p-work"]'
FRAMES = '() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))'
OBSERVE = """() => {
    window.sidebarSockets = [];
    const Socket = window.WebSocket;
    window.WebSocket = class extends Socket {
        constructor(...args) { super(...args); sidebarSockets.push(this); }
    };
    window.sidebarReads = [];
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
        const response = await nativeFetch(...args);
        if (new URL(args[0], location.href).pathname === '/api/state') {
            const nativeJson = response.json.bind(response);
            response.json = async () => {
                const body = await nativeJson();
                // Record completion after the actual consumers' synchronous
                // projection and its paint, including an unchanged projection.
                requestAnimationFrame(() => requestAnimationFrame(() => sidebarReads.push(body._testRevision)));
                return body;
            };
        }
        return response;
    };
}"""


def activity(identity, project='', phase='working', **extra):
    return dict(activity_id=identity, project_id=project, chat_id=42 if project else 1,
                kind='managed_task' if project else 'direct_chat', phase=phase, **extra)


def model_wait(identity, project):
    # The producer's complete wait row: the sidebar admits waits exactly as the
    # chat card does, so a bare {state, task_attempt} would be dropped.
    return activity(identity, project, task_attempt=1, model_waits={'access': dict(
        wait_id='access', revision=1, task_attempt=1, state='waiting', reason='quota')})


def token_color(page, project, token):
    """Computed colour of a CSS token resolved inside the given project row."""
    return page.locator(f'[data-project-id="{project}"]').evaluate("""(el, token) => {
        const probe = document.createElement('span'); probe.style.color = 'var(' + token + ')';
        el.append(probe); const color = getComputedStyle(probe).color; probe.remove(); return color;}""", token)


def dot_color(page, project):
    """The ink of a row's mark: a dot's fill, or a ring's border (an unconfirmed ring is drawn in its border)."""
    return page.locator(f'[data-project-id="{project}"] .status-mark').evaluate("""el => {
        const s = getComputedStyle(el.firstElementChild);
        return s.backgroundColor === 'rgba(0, 0, 0, 0)' ? s.borderTopColor : s.backgroundColor; }""")


def row_color(page, project):
    return page.locator(f'[data-project-id="{project}"]').evaluate('el => getComputedStyle(el).color')


def served(body):
    """The body as the server sends it: each census row in its Project's room (by the Project's chat id), and the
    server's work projection of the rows. A body without its census carries no work fields either."""
    reply = copy.deepcopy(body)
    if 'active_chat_activities' in reply:
        rooms = {project['id']: project['chat_id'] for project in reply.get('projects') or ()}
        rows = [dict(row, chat_id=rooms.get(row.get('project_id'), row.get('chat_id', 1)))
                for row in reply['active_chat_activities']]
        reply['active_chat_activities'] = rows
        reply.update(work_fields(rows, complete=reply.get('active_chat_activities_complete', True)))
    return reply


def mount_sidebar(ui, width, theme, reduced, projects, census):
    """Open the real SPA over synthetic state replies; return the mutable body and its controls."""
    page = ui['page']
    page.set_viewport_size(dict(width=width, height=844))
    page.emulate_media(color_scheme=theme, reduced_motion='reduce' if reduced else 'no-preference')
    page.add_init_script('(' + OBSERVE + ')()')
    body = dict(sha='sidebar-fixture', supervisor_ready=True, active_chat_activities_complete=True,
                active_chat_activities=census, projects=projects, project_chat_ids=[p['chat_id'] for p in projects],
                _testRevision=1)
    mode, sockets, held = {'fault': ''}, [], []

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({'type': 'heartbeat'}))

    def state_response(route):
        if mode['fault'] == 'hold':
            held.append((route, served(body)))
        elif mode['fault'] == 'http':
            route.fulfill(status=503, json={'error': 'fixture failure'})
        elif mode['fault'] == 'network':
            route.abort()
        else:
            route.fulfill(json=served(body))

    origin = urlparse(ui['url']).netloc
    page.route('**/*', lambda r: r.fallback() if urlparse(r.request.url).netloc == origin else r.abort())
    page.route_web_socket('**/ws', connect)
    page.route('**/api/state', state_response)
    page.expose_function('sidebarHeldCount', lambda: len(held))
    page.goto(ui['url'] + '/', wait_until='domcontentloaded')
    page.wait_for_function('() => window.__ouroWs?.ws?.readyState === WebSocket.OPEN')
    page.wait_for_function('() => sidebarReads.includes(1)')
    page.locator(f'.nav-project-row[data-project-id="{projects[0]["id"]}"]').wait_for(state='attached')
    if width < 700:
        page.locator('#page-chat [data-mobile-nav-toggle]').click()
        page.wait_for_function("() => document.querySelector('#primary-sidebar').getBoundingClientRect().left >= 0")

    def refresh(**changes):
        body.update(changes)
        body['_testRevision'] += 1
        sockets[-1].send(json.dumps({'type': 'projects_changed'}))
        page.wait_for_function('rev => sidebarReads.includes(rev)', arg=body['_testRevision'])

    return dict(body=body, mode=mode, sockets=sockets, held=held, refresh=refresh)


@pytest.mark.parametrize('width,theme,reduced', [
    (1360, 'dark', False), (1360, 'light', True),
    (390, 'light', False), (390, 'dark', True),
])
def test_sidebar_activity_census_and_navigation(subscription_ui, width, theme, reduced):
    ui, page = subscription_ui, subscription_ui['page']
    projects = [dict(id='p-' + key, name=name, chat_id=42 + i, lifecycle='active', visible_revision=0)
                for i, (key, name) in enumerate([
                    ('work', 'Working room'), ('queue', 'Queued room'), ('wait', 'Waiting room'),
                    ('empty', 'Empty room'), ('delete', 'Deleting room')])]
    projects[0]['visible_revision'] = 4
    projects[-1]['lifecycle'] = 'deleting'
    work = activity('work', 'p-work')
    initial = [work, activity('queue', 'p-queue', 'queued'), model_wait('wait', 'p-wait'),
               activity('deleting', 'p-delete'), activity('main', phase='thinking')]
    mounted = mount_sidebar(ui, width, theme, reduced, projects, initial)
    body, mode, sockets, held, refresh = (mounted[key] for key in ('body', 'mode', 'sockets', 'held', 'refresh'))

    def marker(project):
        return page.locator(f'[data-project-id="{project}"] .status-mark')

    def observe(project, shape, hue='ink'):
        """The row's one mark (status_mark.js): its shape is what the work does, its hue whose move it is or new
        messages. Only moving dots animate; nothing is drawn at rest."""
        target = marker(project)
        page.wait_for_function("([selector, shape, hue]) => { const m = document.querySelector(selector);"
                               " return m?.dataset.shape === shape && m.dataset.hue === hue; }",
                               arg=[f'[data-project-id="{project}"] .status-mark', shape, hue])
        assert target.get_attribute('aria-hidden') == 'true'
        paint = target.evaluate("""el => ({gap:getComputedStyle(el).gap, hidden: el.hidden, dots:[...el.children].map(s => {
            const c=getComputedStyle(s); return {shown:c.display !== 'none',width:c.width,height:c.height,
                animation:c.animationName,duration:c.animationDuration,delay:c.animationDelay};})})""")
        assert paint['hidden'] is (shape == 'none') and len(paint['dots']) == 3, (project, paint)
        if shape == 'none':
            return
        assert paint['gap'] == '3px'
        # Work is three 4px dots (moving or still), a pause two 2x8 bars, a fact (an owner's move, new messages at
        # rest) one 7px dot and an unconfirmed room one 7px ring.
        shown = [dot for dot in paint['dots'] if dot['shown']]
        size = {'moving': ('4px', '4px'), 'still': ('4px', '4px'), 'pause': ('2px', '8px')}.get(shape, ('7px', '7px'))
        assert len(shown) == {'moving': 3, 'still': 3, 'pause': 2}.get(shape, 1), (project, shape, paint)
        for index, dot in enumerate(shown):
            assert (dot['width'], dot['height']) == size, (project, dot)
            assert dot['animation'] == ('status-mark-bounce' if shape == 'moving' and not reduced else 'none')
            if shape == 'moving' and not reduced:
                assert dot['duration'] == '1.15s'
                assert float(dot['delay'].removesuffix('s')) == pytest.approx(index * .15)

    def capture(suffix):
        setup_browser.capture(page, f'sidebar-{theme}-{width}-{reduced}-{suffix}')

    observe('p-work', 'moving', 'unread')
    observe('p-queue', 'still')
    # A model wait nobody has to act on continues by itself: the still dots, with its reason in the row's name.
    observe('p-wait', 'still')
    assert page.locator('[data-project-id="p-wait"]').get_attribute('title') == (
        'Waiting room · Waiting · Waiting for the model to become available')
    observe('p-empty', 'none')
    assert page.locator('[data-project-id="p-empty"]').is_enabled()
    assert page.locator('[data-project-id="p-delete"]').is_disabled()
    # A deleting row draws only `Deleting…`; its census fact stays in its name.
    assert page.locator('[data-project-id="p-delete"]').get_attribute('title') == 'Deleting room — Deleting… · Working'
    observe('p-delete', 'none')
    # New messages are the row's fact and its mark's crimson hue on the work it shows: no dot of their own.
    assert page.locator(WORK).get_attribute('data-unread') == ''
    assert page.locator('[data-project-id="p-queue"]').get_attribute('data-unread') is None
    assert page.locator('#nav-projects-list .nav-unread-dot, #nav-projects-list .nav-activity-marker').count() == 0
    # The Projects number counts unread messages (owner Q1 = B): the working room's four, not one room.
    assert page.locator('#nav-projects-count').inner_text() == '4'
    assert page.locator(WORK).get_attribute('aria-label') == 'Working room · Working · new messages'
    # Ink: amber is only what waits for the owner and crimson only new messages, so this wait and the queue take the
    # row's own foreground: the marks never out-shout the name, and every name has one ink, read or not.
    for project in ('p-queue', 'p-wait'):
        assert dot_color(page, project) == row_color(page, project)
    assert dot_color(page, 'p-work') == token_color(page, 'p-work', '--mark-unread')
    assert row_color(page, 'p-work') == row_color(page, 'p-queue')
    for project in ('p-work', 'p-queue', 'p-wait'):
        assert dot_color(page, project) != token_color(page, project, '--project')
    assert marker('p-queue').evaluate('el => getComputedStyle(el).opacity') == '0.4'
    assert marker('p-work').evaluate('el => getComputedStyle(el).opacity') == '1'
    header = page.locator('.nav-projects-facts .status-mark')
    # The header's mark stands in the rows' mark column, its count just before it, its chevron after its word.
    geometry = page.evaluate("""() => {
        const box = (el) => el.getBoundingClientRect();
        const mark = box(document.querySelector('.nav-projects-facts .status-mark'));
        const row = box(document.querySelector('[data-project-id="p-work"] .status-mark'));
        const count = box(document.getElementById('nav-projects-count'));
        const word = box(document.querySelector('#nav-projects-toggle .nav-row-label'));
        const chevron = box(document.querySelector('#nav-projects-toggle .nav-chevron'));
        const head = box(document.getElementById('nav-projects-toggle'));
        return {mark: [mark.left, mark.right], row: [row.left, row.right], gap: mark.left - count.right,
                chevron: chevron.left - word.right, middle: (mark.top + mark.bottom) / 2 - (head.top + head.bottom) / 2};
    }""")
    assert geometry['mark'] == pytest.approx(geometry['row'], abs=0.5), geometry
    assert 8 <= geometry['gap'] <= 16 and 0 <= geometry['chevron'] <= 12, geometry
    assert abs(geometry['middle']) <= 1, geometry
    # The header's one mark: the strongest Project work in the toggle's ink (Main's own work never moves it).
    assert (header.get_attribute('data-shape'), header.get_attribute('data-hue')) == ('moving', 'ink')
    capture('states')

    # The header's mark keeps its column in both states; collapse keeps only pinned rows; activity sorts nothing.
    order = page.locator('.nav-project-row').evaluate_all('els => els.map(el=>el.dataset.projectId)')
    page.locator('#nav-projects-toggle').click()
    assert page.locator('#nav-projects-list').is_hidden(), 'nothing pinned: no row stays, work or unread alike'
    assert page.locator('#nav-projects-show-all').count() == 0
    assert header.is_visible() and page.locator('#nav-projects-count').inner_text() == '4'
    capture('collapsed')
    page.locator('#nav-projects-toggle').click()

    # Keyboard opens the real portalled menu. Preserve the actual menu, focused
    # menu item, row and its mark through activity-only repaints.
    kebab = page.locator(WORK).locator('..').locator('.nav-project-kebab')
    kebab.focus()
    page.keyboard.press('Enter')
    menu = page.locator('body > .project-row-menu:not([data-leaving])')
    menu.wait_for()
    page.keyboard.press('End')
    page.evaluate("""() => { window.sidebarKept = {
        row:document.querySelector('[data-project-id="p-work"]'), menu:document.querySelector('.project-row-menu:not([data-leaving])'),
        focus:document.activeElement, marker:document.querySelector('[data-project-id="p-work"] .status-mark')}; }""")
    owner_wait = activity('work', 'p-work', required_question=dict(wait_for_answer=True, owner_wait_state='waiting'))
    refresh(active_chat_activities=[owner_wait])
    # The owner's question is the one amber fact: the 7px amber dot outranks the new messages; the word says it.
    observe('p-work', 'dot', 'owner')
    assert dot_color(page, 'p-work') == token_color(page, 'p-work', '--amber')
    assert page.locator(WORK).get_attribute('title') == 'Working room · Waiting for your answer · new messages'
    assert page.locator(WORK).get_attribute('data-unread') == '', 'the unread fact stands under the amber'
    mixed = [owner_wait, activity('independent', 'p-work', 'finalizing')]
    refresh(active_chat_activities=mixed)
    # Independent work goes on, but the owner's move stands alone in the mark; the row's name says both facts.
    observe('p-work', 'dot', 'owner')
    assert page.locator(WORK).get_attribute('aria-label').endswith(
        'Finalizing… · A question waits for your answer · new messages')
    assert page.evaluate("""() => sidebarKept.row === document.querySelector('[data-project-id="p-work"]')
        && sidebarKept.menu === document.querySelector('.project-row-menu:not([data-leaving])') && sidebarKept.focus === document.activeElement
        && sidebarKept.marker === document.querySelector('[data-project-id="p-work"] .status-mark')""")
    assert page.locator('.nav-project-row').evaluate_all('els => els.map(el=>el.dataset.projectId)') == order
    capture('mixed-menu')
    page.keyboard.press('Escape')
    assert menu.count() == 0
    assert kebab.evaluate('el=>el===document.activeElement')

    # Resumed owner wait is computation again. A partial omission retains the
    # old row as unknown; a current positive row remains independently moving.
    owner_wait['required_question']['owner_wait_state'] = 'resumed'
    refresh(active_chat_activities=[owner_wait])
    observe('p-work', 'moving', 'unread')
    refresh(active_chat_activities=[activity('positive', 'p-queue', 'thinking')], active_chat_activities_complete=False)
    observe('p-work', 'ring', 'unread')
    observe('p-queue', 'moving')
    assert 'Activity unconfirmed' in page.locator(WORK).get_attribute('title')
    refresh(active_chat_activities=[], active_chat_activities_complete=True, supervisor_ready=False)
    observe('p-work', 'ring', 'unread')
    observe('p-queue', 'ring')
    refresh(active_chat_activities=initial, supervisor_ready=True)
    observe('p-work', 'moving', 'unread')

    # Missing census, HTTP failure, and transport failure all stop retained
    # observations. Exercise the actual state readers, not an exported reducer.
    body.pop('active_chat_activities')
    refresh()
    observe('p-work', 'ring', 'unread')
    for fault in ['http', 'network']:
        refresh(active_chat_activities=initial)
        observe('p-work', 'moving', 'unread')
        mode['fault'] = fault
        sockets[-1].send(json.dumps({'type': 'projects_changed'}))
        observe('p-work', 'ring', 'unread')
        mode['fault'] = ''

    # One page-wide state read is in flight at a time: a forced refresh that
    # lands while an absence response is held starts no second read and
    # coalesces into one follow-up that begins once the held read settles. The
    # only thing that outruns a held read is the socket's own generation bump,
    # so make the held absence stale with a real close and reconnect and prove
    # it can neither clear the row nor hide the marker once it is released; the
    # follow-up, sequenced after it, is what carries the current census.
    refresh(active_chat_activities=initial)
    mode['fault'] = 'hold'
    body['active_chat_activities'] = []
    body['_testRevision'] += 1
    stale_revision = body['_testRevision']
    with page.expect_request(lambda r: urlparse(r.url).path == '/api/state'):
        sockets[-1].send(json.dumps({'type': 'projects_changed'}))
    page.wait_for_function('() => sidebarHeldCount().then(n => n > 0)')
    sockets[-1].send(json.dumps({'type': 'projects_changed'}))
    page.wait_for_timeout(300)
    assert page.evaluate('() => sidebarHeldCount()') == 1, 'a forced refresh behind a held read starts no second read'
    connections = len(sockets)
    sockets[-1].close()
    observe('p-work', 'ring', 'unread')
    page.wait_for_function('n => sidebarSockets.length > n && sidebarSockets.at(-1).readyState === WebSocket.OPEN', arg=connections)
    body['active_chat_activities'] = initial
    body['_testRevision'] += 1
    fresh_revision = body['_testRevision']
    stale_reads, held[:] = list(held), []
    for route, old in stale_reads:
        route.fulfill(json=old)
    # Every released read (the gated one and the socket's own SHA reads) consumes
    # its body, so this pins the gated read's settle, not merely the first body.
    page.wait_for_function('([rev, n]) => sidebarReads.filter(r => r === rev).length >= n',
                           arg=[stale_revision, len(stale_reads)])
    # The coalesced follow-up starts only now, behind the settled stale read,
    # and captures the census as it is now: it is held so the stale absence
    # stands alone when the row is judged.
    page.wait_for_function('() => sidebarHeldCount().then(n => n > 0)')
    page.evaluate(FRAMES)
    observe('p-work', 'ring', 'unread')
    assert marker('p-work').is_visible()
    mode['fault'] = ''
    for route, snapshot in held:
        route.fulfill(json=snapshot)
    held.clear()
    page.wait_for_function('rev => sidebarReads.includes(rev)', arg=fresh_revision)
    observe('p-work', 'moving', 'unread')

    # A real socket close and reconnect stop motion immediately and preserve
    # unknown until the new connection's held state responses actually arrive.
    mode['fault'] = 'hold'
    connections = len(sockets)
    sockets[-1].close()
    observe('p-work', 'ring', 'unread')
    page.wait_for_function('n => sidebarSockets.length > n && sidebarSockets.at(-1).readyState === WebSocket.OPEN', arg=connections)
    page.evaluate(FRAMES)
    observe('p-work', 'ring', 'unread')
    mode['fault'] = ''
    for route, snapshot in held:
        route.fulfill(json=snapshot)
    held.clear()
    refresh(active_chat_activities=initial)
    observe('p-work', 'moving', 'unread')
    # WebSocket error uses the same production disconnect handler.
    page.evaluate("sidebarSockets.at(-1).dispatchEvent(new Event('error'))")
    observe('p-work', 'ring', 'unread')
    page.wait_for_function('n => sidebarSockets.length > n && sidebarSockets.at(-1).readyState === WebSocket.OPEN', arg=connections + 1)
    refresh(active_chat_activities=[], active_chat_activities_complete=True, supervisor_ready=True)
    # At rest, a room with new messages is one crimson dot; a read one draws nothing, nor does the header.
    observe('p-work', 'dot', 'unread')
    observe('p-queue', 'none')
    assert page.locator('.nav-projects-facts .status-mark').is_hidden()
    assert page.locator('#nav-projects-count').inner_text() == '4'
    capture('complete-empty')

    # Merely observing state has not marked any project read. Keyboard opening
    # an empty room still navigates normally; the active room gets aria-current.
    assert not any(path == '/api/ui/preferences' and payload.get('project_seen_revision')
                   for path, payload in ui['posts'])
    empty = page.locator('[data-project-id="p-empty"]')
    empty.focus()
    page.keyboard.press('Enter')
    page.locator('#project-panel').wait_for(state='visible')
    assert empty.get_attribute('aria-current') == 'page'
    assert page.locator('#project-panel-title').inner_text() == 'Empty room'
    if width < 700:
        # The drawer stays painted for one beat while the room changes inside the card, then closes (owner 4A).
        rest(page)
        assert not page.locator('#primary-sidebar').evaluate("el=>el.classList.contains('open')")
    page.locator('#project-panel-close').click()
    page.locator('#project-panel').wait_for(state='hidden')
    assert page.locator('[data-nav-page="chat"]').get_attribute('aria-current') == 'page'


LONG_NAME = 'A deliberately long project name that must ellipsize before it reaches the rail'
RAIL = """() => {
    const rect = (el) => {
        if (!el) return null;
        const b = el.getBoundingClientRect();
        return {left: b.left, right: b.right, top: b.top, bottom: b.bottom, width: b.width, height: b.height};
    };
    return [...document.querySelectorAll('.nav-project-item')].map((item) => {
        const row = item.querySelector('.nav-project-row');
        const mark = row.querySelector('.status-mark');
        const label = row.querySelector('.nav-row-label');
        const style = getComputedStyle(mark);
        const dots = [...mark.children].filter((dot) => getComputedStyle(dot).display !== 'none');
        return {
            id: row.dataset.projectId, shape: mark.dataset.shape, hue: mark.dataset.hue,
            unread: row.hasAttribute('data-unread'), deleting: item.classList.contains('is-deleting'),
            active: row.classList.contains('active'), row: rect(row), label: rect(label),
            ellipsized: label.scrollWidth > label.clientWidth, marker: rect(mark),
            drawn: style.display !== 'none' && style.visibility !== 'hidden', opacity: style.opacity,
            // Where each dot lies in the column: a moving dot's wave lifts and scales it for a moment, so its place
            // is read with the wave suspended (restoring it restarts the wave in the page's one phase).
            dots: dots.map((dot) => {
                const wave = dot.style.animation;
                dot.style.animation = 'none';
                const left = rect(dot).left - rect(mark).left;
                dot.style.animation = wave;
                return left;
            }),
            // A hollow ring (unconfirmed) draws its ink in its border.
            dotColor: (({ backgroundColor: fill, borderTopColor: ring }) => (fill === 'rgba(0, 0, 0, 0)' ? ring : fill))(
                getComputedStyle(mark.firstElementChild)), rowColor: getComputedStyle(row).color,
            others: row.children.length,
            trailing: rect(item.lastElementChild), trailingKind: item.lastElementChild.className,
        };
    });
}"""
# Where each drawn shape's dots start in the 18px column: three 4px dots and two 3px gaps fill it, two 2px bars
# and one 7px dot or ring sit in its middle.
OFFSETS = {'moving': [0, 7, 14], 'still': [0, 7, 14], 'pause': [5.5, 10.5], 'ring': [5.5], 'dot': [5.5]}
# The quieter steps of the ink: still work 40 %, an unconfirmed ring 55 %, a pause 70 %; with new messages the
# crimson needs more.
OPACITY = {('still', 'ink'): '0.4', ('still', 'unread'): '0.6', ('ring', 'ink'): '0.55', ('ring', 'unread'): '0.6',
           ('pause', 'ink'): '0.7'}


def rail_rows(page):
    return {row['id']: row for row in page.evaluate(RAIL)}


def horizontal(rect):
    return rect and {key: rect[key] for key in ('left', 'right', 'width')}


def assert_rail(page, rows, states, *, lit=()):
    """Every drawn mark shares one 18px column, its shape and hue as the room and the row's unread fact say."""
    live = [row for row in rows.values() if not row['deleting']]
    marker = next(row['marker'] for row in live if row['drawn'])
    tokens = {hue: token_color(page, live[0]['id'], token) for hue, token in
              (('unread', '--mark-unread'), ('owner', '--amber'))}
    project = token_color(page, live[0]['id'], '--project')
    for row in live:
        assert (row['shape'], row['hue']) == states[row['id']], (row['id'], row['shape'], row['hue'])
        assert row['drawn'] == (row['shape'] != 'none'), row['id']
        assert row['unread'] == (row['hue'] == 'unread'), row['id']
        # One mark, nothing else after the name: no unread dot of its own.
        assert row['others'] == 2, row['id']
        # The label cell ends one gap before the mark column in every row,
        # so a row at rest (its mark removed from layout) still holds the slot.
        assert row['label']['right'] == pytest.approx(marker['left'] - 8, abs=0.01), (row['id'], row['label'], marker)
        assert row['trailingKind'] == 'nav-project-kebab'
        assert row['trailing']['left'] >= row['row']['right'] - 0.01, (row['id'], row['trailing'], row['row'])
        if not row['drawn']:
            continue
        for key in ('left', 'right', 'width'):
            assert row['marker'][key] == pytest.approx(marker[key], abs=0.01), (row['id'], key, row['marker'], marker)
        assert row['marker']['width'] == pytest.approx(18, abs=0.01)  # three 4px dots, two 3px gaps
        assert row['dots'] == pytest.approx(OFFSETS[row['shape']], abs=0.01), (row['id'], row['dots'])
        assert abs((row['marker']['top'] + row['marker']['bottom']) - (row['row']['top'] + row['row']['bottom'])) <= 1
        assert row['opacity'] == OPACITY.get((row['shape'], row['hue']), '1'), row['id']
        if row['hue'] in tokens:
            assert row['dotColor'] == tokens[row['hue']], (row['id'], row['dotColor'])
        else:
            # The row's own ink, hovered, lit or not (a chosen row's ink included); never a project colour of its own.
            assert row['dotColor'] == row['rowColor'], (row['id'], row['dotColor'], row['rowColor'])
            if row['id'] not in lit:
                assert row['dotColor'] != project, row['id']
    for row in rows.values():
        if not row['deleting']:
            continue
        assert not row['drawn'] and row['shape'] == 'none' and not row['unread'], row['id']
        assert row['trailingKind'] == 'nav-project-deleting-status'
        assert row['label']['right'] <= row['trailing']['left'] + 0.01, (row['id'], row['label'], row['trailing'])
    assert any(row['ellipsized'] for row in live)
    return {key: row['marker'] for key, row in rows.items()}


@pytest.mark.parametrize('width,theme,reduced', [
    (1360, 'dark', False), (1360, 'light', False), (390, 'dark', True),
])
def test_sidebar_activity_rail_geometry(subscription_ui, width, theme, reduced):
    ui, page = subscription_ui, subscription_ui['page']
    specs = [  # key, name, unread, deleting, census rows
        ('wu', 'Working unread', True, False, [activity('wu', 'p-wu')]),
        ('w', 'Working room', False, False, [activity('w', 'p-w')]),
        ('wl', LONG_NAME, True, False, [activity('wl', 'p-wl', 'thinking')]),
        ('qu', 'Queued unread', True, False, [activity('qu', 'p-qu', 'queued')]),
        ('q', 'Paused room', False, False, [activity('q', 'p-q', 'budget_paused')]),
        ('a', 'Waiting room', False, False, [model_wait('a', 'p-a')]),
        ('iu', 'Idle unread', True, False, []),
        ('i', 'Idle room', False, False, []),
        ('dw', 'Deleting working', False, True, [activity('dw', 'p-dw')]),
        ('d', 'Deleting room', False, True, []),
    ]
    projects = [dict(id='p-' + key, name=name, chat_id=42 + i, lifecycle='deleting' if deleting else 'active',
                     visible_revision=4 if unread else 0)
                for i, (key, name, unread, deleting, _) in enumerate(specs)]
    initial = [row for spec in specs for row in spec[-1]] + [activity('main', phase='thinking')]
    mounted = mount_sidebar(ui, width, theme, reduced, projects, initial)
    refresh = mounted['refresh']
    states = dict(wu=('moving', 'unread'), w=('moving', 'ink'), wl=('moving', 'unread'), qu=('still', 'unread'),
                  q=('pause', 'ink'), a=('still', 'ink'), iu=('dot', 'unread'), i=('none', 'ink'),
                  dw=('none', 'ink'), d=('none', 'ink'))
    states = {'p-' + key: value for key, value in states.items()}

    def capture(suffix):
        setup_browser.capture(page, f'sidebar-{theme}-{width}-{reduced}-rail-{suffix}')

    def same_rail(reference, current, ids=None):
        # Rows may legitimately re-order (unread rooms sort first); the column may not move.
        for key in ids or reference:
            if reference[key]['width'] and current[key]['width']:  # both drawn (a mark at rest has no box)
                assert horizontal(current[key]) == pytest.approx(horizontal(reference[key]), abs=0.01), key

    def settled(project, token):
        # Row ink transitions over 0.15s; measure only once it reached the token.
        page.wait_for_function("([project, color]) => getComputedStyle(document.querySelector("
                               "'[data-project-id=\"' + project + '\"]')).color === color",
                               arg=[project, token_color(page, project, token)])

    def shaped(project, shape):
        page.wait_for_function("([project, shape]) => document.querySelector("
                               "'[data-project-id=\"' + project + '\"] .status-mark')?.dataset.shape === shape",
                               arg=[project, shape])

    shaped('p-a', 'still')
    assert page.locator('[data-project-id="p-dw"]').get_attribute('title') == 'Deleting working — Deleting… · Working'
    reference = assert_rail(page, rail_rows(page), states)
    capture('default')

    # Hovering a row reveals its sibling kebab and brightens the title; its working dots follow the row's ink.
    plain = rail_rows(page)['p-w']['rowColor']
    page.locator('[data-project-id="p-w"]').hover()
    page.wait_for_function("() => getComputedStyle(document.querySelector('[data-project-id=\"p-w\"] ~ .nav-project-kebab')).opacity === '1'")
    settled('p-w', '--text-primary')
    hovered = rail_rows(page)
    assert hovered['p-w']['rowColor'] != plain and hovered['p-w']['dotColor'] == hovered['p-w']['rowColor']
    same_rail(reference, assert_rail(page, hovered, states))
    capture('hover')
    page.mouse.move(width / 2, 600)

    # Keyboard focus on a kebab and the open portalled menu leave every slot where it was.
    kebab = page.locator('[data-project-id="p-qu"] ~ .nav-project-kebab')
    kebab.focus()
    same_rail(reference, assert_rail(page, rail_rows(page), states))
    page.keyboard.press('Enter')
    page.locator('body > .project-row-menu:not([data-leaving])').wait_for()
    same_rail(reference, assert_rail(page, rail_rows(page), states))
    capture('menu')
    page.keyboard.press('Escape')
    page.locator('body > .project-row-menu:not([data-leaving])').wait_for(state='detached')

    # An incomplete census retains every observed room as an unconfirmed ring; new messages keep their crimson.
    refresh(active_chat_activities=[], active_chat_activities_complete=False)
    unknown = {key: (('ring', hue) if shape not in ('none', 'dot') and key not in ('p-dw', 'p-d') else (shape, hue))
               for key, (shape, hue) in states.items()}
    shaped('p-w', 'ring')
    same_rail(reference, assert_rail(page, rail_rows(page), unknown))
    capture('unknown')

    # Reading a room turns only its mark's hue back to the row's ink; the column does not move.
    refresh(active_chat_activities=initial, active_chat_activities_complete=True)
    shaped('p-w', 'moving')
    projects[0]['visible_revision'] = 0
    refresh(projects=projects)
    page.wait_for_function("() => !document.querySelector('[data-project-id=\"p-wu\"]').hasAttribute('data-unread')")
    read = rail_rows(page)
    states['p-wu'] = ('moving', 'ink')
    same_rail(reference, assert_rail(page, read, states))
    capture('read')

    if width >= 700:
        # The open room's title takes the project ink, and its working dots with it (the row's own ink); slots stay put.
        page.locator('[data-project-id="p-w"]').click()
        page.locator('#project-panel').wait_for(state='visible')
        page.mouse.move(width / 2, 600)
        settled('p-w', '--project')
        lit = rail_rows(page)
        assert lit['p-w']['active'] and lit['p-w']['rowColor'] == token_color(page, 'p-w', '--project')
        assert lit['p-w']['dotColor'] == lit['p-w']['rowColor']
        same_rail(reference, assert_rail(page, lit, states, lit=('p-w',)))
        capture('active')
        page.locator('#project-panel-close').click()
        page.locator('#project-panel').wait_for(state='hidden')

    # A complete empty census draws nothing but new messages, which are one crimson dot in the same column.
    refresh(active_chat_activities=[])
    shaped('p-w', 'none')
    empty = rail_rows(page)
    for key, row in empty.items():
        expected = ('dot', 'unread') if row['unread'] else ('none', 'ink')
        assert (row['shape'], row['hue']) == expected, key
        if row['drawn']:
            assert horizontal(row['marker']) == pytest.approx(horizontal(reference['p-w']), abs=0.01), key
    capture('empty')
