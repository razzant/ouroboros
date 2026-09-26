"""Real SPA census consumers, socket lifecycle, CSS and navigation preservation.

Uses the existing static-document fixture; never starts or bootstraps Ouroboros.
All runtime replies are synthetic. Run with either fixture browser engine.
"""
from __future__ import annotations

import copy
import json
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

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
    return page.locator(f'[data-project-id="{project}"] .nav-activity-marker').evaluate(
        'el => getComputedStyle(el.firstElementChild).backgroundColor')


def row_color(page, project):
    return page.locator(f'[data-project-id="{project}"]').evaluate('el => getComputedStyle(el).color')


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
            held.append((route, copy.deepcopy(body)))
        elif mode['fault'] == 'http':
            route.fulfill(status=503, json={'error': 'fixture failure'})
        elif mode['fault'] == 'network':
            route.abort()
        else:
            route.fulfill(json=copy.deepcopy(body))

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
        return page.locator(f'[data-project-id="{project}"] .nav-activity-marker')

    def observe(project, state, moving=False):
        target = marker(project)
        page.wait_for_function("([selector, state]) => document.querySelector(selector)?.dataset.state === state",
                               arg=[f'[data-project-id="{project}"] .nav-activity-marker', state])
        assert target.get_attribute('data-motion') == ('1' if moving else '0')
        paint = target.evaluate("""el => ({gap:getComputedStyle(el).gap, dots:[...el.children].map(s => {
            const c=getComputedStyle(s); return {width:c.width,height:c.height,animation:c.animationName,
                duration:c.animationDuration,delay:c.animationDelay};})})""")
        assert paint['gap'] == '3px'
        assert len(paint['dots']) == 3
        for index, dot in enumerate(paint['dots']):
            assert (dot['width'], dot['height']) == ('4px', '4px')
            assert dot['animation'] == ('typing-bounce' if moving and not reduced else 'none')
            if moving and not reduced:
                assert dot['duration'] == '1.4s'
                assert float(dot['delay'].removesuffix('s')) == pytest.approx(index * .2)

    def capture(suffix):
        setup_browser.capture(page, f'sidebar-{theme}-{width}-{reduced}-{suffix}')

    observe('p-work', 'working', True)
    observe('p-queue', 'queued')
    observe('p-wait', 'waiting')
    assert page.locator('[data-project-id="p-wait"]').get_attribute('title') == 'Waiting room · Waiting for access'
    assert marker('p-empty').is_hidden()
    assert page.locator('[data-project-id="p-empty"]').is_enabled()
    assert page.locator('[data-project-id="p-delete"]').is_disabled()
    assert page.locator('[data-project-id="p-delete"]').get_attribute('title') == 'Deleting room — Deleting… · Working'
    assert marker('p-delete').get_attribute('data-state') == 'working'
    assert marker('p-delete').is_hidden()
    assert page.locator(WORK + ' .nav-unread-dot').count() == 1
    assert page.locator('#nav-projects-count').inner_text() == '1'
    assert 'Unread' in page.locator(WORK).get_attribute('aria-label')
    # Ink: a wait is the one amber fact; working and queued dots take the row's own
    # foreground (never the saturated project token on a plain row), queued quieter.
    assert dot_color(page, 'p-wait') == token_color(page, 'p-wait', '--amber')
    for project in ('p-work', 'p-queue'):
        assert dot_color(page, project) == row_color(page, project)
        assert dot_color(page, project) != token_color(page, project, '--project')
    assert marker('p-queue').evaluate('el => getComputedStyle(el).opacity') == '0.55'
    assert marker('p-work').evaluate('el => getComputedStyle(el).opacity') == '1'
    geometry = page.locator('#nav-projects-activity').evaluate("""el => {
        const a=el.getBoundingClientRect(), b=el.previousElementSibling.getBoundingClientRect();
        return {gap:a.left-b.right, top:a.top, labelTop:b.top, height:b.height};}""")
    assert 0 < geometry['gap'] <= 10, geometry
    assert geometry['labelTop'] <= geometry['top'] <= geometry['labelTop'] + geometry['height']
    capture('states')

    # Aggregate is adjacent to Projects and survives collapse; activity does not sort rows.
    order = page.locator('.nav-project-row').evaluate_all('els => els.map(el=>el.dataset.projectId)')
    page.locator('#nav-projects-toggle').click()
    assert page.locator('#nav-projects-list').is_hidden()
    assert page.locator('#nav-projects-activity').is_visible()
    capture('collapsed')
    page.locator('#nav-projects-toggle').click()

    # Keyboard opens the real portalled menu. Preserve the actual menu, focused
    # menu item, row, unread node and marker through activity-only repaints.
    kebab = page.locator(WORK).locator('..').locator('.nav-project-kebab')
    kebab.focus()
    page.keyboard.press('Enter')
    menu = page.locator('body > .project-row-menu')
    menu.wait_for()
    page.keyboard.press('End')
    page.evaluate("""() => { window.sidebarKept = {
        row:document.querySelector('[data-project-id="p-work"]'), menu:document.querySelector('.project-row-menu'),
        focus:document.activeElement, unread:document.querySelector('[data-project-id="p-work"] .nav-unread-dot'),
        marker:document.querySelector('[data-project-id="p-work"] .nav-activity-marker')}; }""")
    owner_wait = activity('work', 'p-work', required_question=dict(wait_for_answer=True, owner_wait_state='waiting'))
    refresh(active_chat_activities=[owner_wait])
    observe('p-work', 'waiting')
    assert page.locator(WORK).get_attribute('title').endswith('Waiting for your answer')
    mixed = [owner_wait, activity('independent', 'p-work', 'finalizing')]
    refresh(active_chat_activities=mixed)
    observe('p-work', 'working', True)
    assert page.locator(WORK).get_attribute('aria-label').endswith('Finalizing · Waiting for your answer')
    assert page.evaluate("""() => sidebarKept.row === document.querySelector('[data-project-id="p-work"]')
        && sidebarKept.menu === document.querySelector('.project-row-menu') && sidebarKept.focus === document.activeElement
        && sidebarKept.unread.isConnected && sidebarKept.marker.isConnected""")
    assert page.locator('.nav-project-row').evaluate_all('els => els.map(el=>el.dataset.projectId)') == order
    capture('mixed-menu')
    page.keyboard.press('Escape')
    assert menu.count() == 0
    assert kebab.evaluate('el=>el===document.activeElement')

    # Resumed owner wait is computation again. A partial omission retains the
    # old row as unknown; a current positive row remains independently moving.
    owner_wait['required_question']['owner_wait_state'] = 'resumed'
    refresh(active_chat_activities=[owner_wait])
    observe('p-work', 'working', True)
    refresh(active_chat_activities=[activity('positive', 'p-queue', 'thinking')], active_chat_activities_complete=False)
    observe('p-work', 'unknown')
    observe('p-queue', 'working', True)
    assert 'Activity status unavailable' in page.locator(WORK).get_attribute('title')
    refresh(active_chat_activities=[], active_chat_activities_complete=True, supervisor_ready=False)
    observe('p-work', 'unknown')
    observe('p-queue', 'unknown')
    refresh(active_chat_activities=initial, supervisor_ready=True)
    observe('p-work', 'working', True)

    # Missing census, HTTP failure, and transport failure all stop retained
    # observations. Exercise the actual state readers, not an exported reducer.
    body.pop('active_chat_activities')
    refresh()
    observe('p-work', 'unknown')
    for fault in ['http', 'network']:
        refresh(active_chat_activities=initial)
        observe('p-work', 'working', True)
        mode['fault'] = fault
        sockets[-1].send(json.dumps({'type': 'projects_changed'}))
        observe('p-work', 'unknown')
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
    observe('p-work', 'unknown')
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
    observe('p-work', 'unknown')
    assert marker('p-work').is_visible()
    mode['fault'] = ''
    for route, snapshot in held:
        route.fulfill(json=snapshot)
    held.clear()
    page.wait_for_function('rev => sidebarReads.includes(rev)', arg=fresh_revision)
    observe('p-work', 'working', True)

    # A real socket close and reconnect stop motion immediately and preserve
    # unknown until the new connection's held state responses actually arrive.
    mode['fault'] = 'hold'
    connections = len(sockets)
    sockets[-1].close()
    observe('p-work', 'unknown')
    page.wait_for_function('n => sidebarSockets.length > n && sidebarSockets.at(-1).readyState === WebSocket.OPEN', arg=connections)
    page.evaluate(FRAMES)
    observe('p-work', 'unknown')
    mode['fault'] = ''
    for route, snapshot in held:
        route.fulfill(json=snapshot)
    held.clear()
    refresh(active_chat_activities=initial)
    observe('p-work', 'working', True)
    # WebSocket error uses the same production disconnect handler.
    page.evaluate("sidebarSockets.at(-1).dispatchEvent(new Event('error'))")
    observe('p-work', 'unknown')
    page.wait_for_function('n => sidebarSockets.length > n && sidebarSockets.at(-1).readyState === WebSocket.OPEN', arg=connections + 1)
    refresh(active_chat_activities=[], active_chat_activities_complete=True, supervisor_ready=True)
    assert marker('p-work').is_hidden()
    assert page.locator('#nav-projects-activity').is_hidden()
    assert page.locator(WORK + ' .nav-unread-dot').count() == 1
    assert page.locator('#nav-projects-count').inner_text() == '1'
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
        const marker = row.querySelector('.nav-activity-marker');
        const label = row.querySelector('.nav-row-label');
        const style = getComputedStyle(marker);
        return {
            id: row.dataset.projectId, state: marker.dataset.state, deleting: item.classList.contains('is-deleting'),
            active: row.classList.contains('active'), row: rect(row), label: rect(label),
            ellipsized: label.scrollWidth > label.clientWidth, marker: rect(marker),
            drawn: style.display !== 'none' && style.visibility !== 'hidden', opacity: style.opacity,
            dots: [...marker.children].map((dot) => rect(dot).left - rect(marker).left),
            dotColor: getComputedStyle(marker.firstElementChild).backgroundColor, rowColor: getComputedStyle(row).color,
            unread: rect(row.querySelector('.nav-unread-dot')),
            trailing: rect(item.lastElementChild), trailingKind: item.lastElementChild.className,
        };
    });
}"""


def rail_rows(page):
    return {row['id']: row for row in page.evaluate(RAIL)}


def horizontal(rect):
    return rect and {key: rect[key] for key in ('left', 'right', 'width')}


def assert_rail(page, rows, states, *, lit=()):
    """Every drawn marker shares one column; unread dots share another; nothing overlaps."""
    live = [row for row in rows.values() if not row['deleting']]
    marker = next(row['marker'] for row in live if row['drawn'])
    unread = next(row['unread'] for row in live if row['unread'])
    amber = token_color(page, live[0]['id'], '--amber')
    project = token_color(page, live[0]['id'], '--project')
    for row in live:
        assert row['state'] == states[row['id']], (row['id'], row['state'])
        assert row['drawn'] == (row['state'] != 'idle'), row['id']
        # The label cell ends one gap before the activity column in every row,
        # so an idle row (its marker removed from layout) still holds the slot.
        assert row['label']['right'] == pytest.approx(marker['left'] - 8, abs=0.01), (row['id'], row['label'], marker)
        assert row['trailingKind'] == 'nav-project-kebab'
        assert row['trailing']['left'] >= row['row']['right'] - 0.01, (row['id'], row['trailing'], row['row'])
        if row['unread']:
            for key in ('left', 'right', 'width'):
                assert row['unread'][key] == pytest.approx(unread[key], abs=0.01), (row['id'], key)
            assert row['unread']['left'] >= marker['right'] + 7.99
            assert row['unread']['right'] <= row['row']['right'] + 0.01
        if not row['drawn']:
            continue
        for key in ('left', 'right', 'width', 'height'):
            assert row['marker'][key] == pytest.approx(marker[key], abs=0.01), (row['id'], key, row['marker'], marker)
        assert row['marker']['width'] == pytest.approx(18, abs=0.01)  # three 4px dots, two 3px gaps
        assert row['dots'] == pytest.approx([0, 7, 14], abs=0.01)
        assert abs((row['marker']['top'] + row['marker']['bottom']) - (row['row']['top'] + row['row']['bottom'])) <= 1
        if row['state'] == 'waiting':
            assert row['dotColor'] == amber
        elif row['state'] != 'idle':
            assert row['dotColor'] == row['rowColor'], (row['id'], row['dotColor'], row['rowColor'])
            assert row['opacity'] == ('0.55' if row['state'] in ('queued', 'unknown') else '1'), row['id']
            if row['id'] not in lit:
                assert row['dotColor'] != project, row['id']
    for row in rows.values():
        if not row['deleting']:
            continue
        assert row['state'] == states[row['id']]
        assert not row['drawn'] and row['unread'] is None
        assert row['trailingKind'] == 'nav-project-deleting-status'
        assert row['label']['right'] <= row['trailing']['left'] + 0.01, (row['id'], row['label'], row['trailing'])
    assert any(row['ellipsized'] for row in live)
    return {key: (row['marker'], row['unread']) for key, row in rows.items()}


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
    states = dict(wu='working', w='working', wl='working', qu='queued', q='waiting', a='waiting',
                  iu='idle', i='idle', dw='working', d='idle')
    states = {'p-' + key: value for key, value in states.items()}

    def capture(suffix):
        setup_browser.capture(page, f'sidebar-{theme}-{width}-{reduced}-rail-{suffix}')

    def same_rail(reference, current, ids=None):
        # Rows may legitimately re-order (unread rooms sort first); the columns may not move.
        for key in ids or reference:
            for expected, actual in zip(reference[key], current[key]):
                if expected is None or actual is None:
                    assert expected == actual, key
                else:
                    assert horizontal(actual) == pytest.approx(horizontal(expected), abs=0.01), key

    def settled(project, token):
        # Row ink transitions over 0.15s; measure only once it reached the token.
        page.wait_for_function("([project, color]) => getComputedStyle(document.querySelector("
                               "'[data-project-id=\"' + project + '\"]')).color === color",
                               arg=[project, token_color(page, project, token)])

    page.wait_for_function("() => document.querySelector('[data-project-id=\"p-a\"] .nav-activity-marker')?.dataset.state === 'waiting'")
    assert page.locator('[data-project-id="p-dw"]').get_attribute('title') == 'Deleting working — Deleting… · Working'
    reference = assert_rail(page, rail_rows(page), states)
    capture('default')

    # Hovering a row reveals its sibling kebab and brightens the title; the dots follow the title ink.
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
    page.locator('body > .project-row-menu').wait_for()
    same_rail(reference, assert_rail(page, rail_rows(page), states))
    capture('menu')
    page.keyboard.press('Escape')
    page.locator('body > .project-row-menu').wait_for(state='detached')

    # An incomplete census retains every observed row as static unknown at the quieter step.
    refresh(active_chat_activities=[], active_chat_activities_complete=False)
    unknown = {key: ('unknown' if value != 'idle' else 'idle') for key, value in states.items()}
    page.wait_for_function("() => document.querySelector('[data-project-id=\"p-w\"] .nav-activity-marker')?.dataset.state === 'unknown'")
    same_rail(reference, assert_rail(page, rail_rows(page), unknown))
    capture('unknown')

    # Reading a room removes only its unread dot; the marker column does not move.
    refresh(active_chat_activities=initial, active_chat_activities_complete=True)
    page.wait_for_function("() => document.querySelector('[data-project-id=\"p-w\"] .nav-activity-marker')?.dataset.state === 'working'")
    projects[0]['visible_revision'] = 0
    refresh(projects=projects)
    page.locator('[data-project-id="p-wu"] .nav-unread-dot').wait_for(state='detached')
    read = rail_rows(page)
    assert read['p-wu']['unread'] is None
    same_rail(reference, assert_rail(page, read, states), ids=[key for key in reference if key != 'p-wu'])
    assert horizontal(read['p-wu']['marker']) == pytest.approx(horizontal(reference['p-wu'][0]), abs=0.01)
    capture('read')

    if width >= 700:
        # The open room's title takes the project ink and its dots follow it; slots stay put.
        page.locator('[data-project-id="p-w"]').click()
        page.locator('#project-panel').wait_for(state='visible')
        page.mouse.move(width / 2, 600)
        settled('p-w', '--project')
        lit = rail_rows(page)
        assert lit['p-w']['active'] and lit['p-w']['dotColor'] == lit['p-w']['rowColor'] == token_color(page, 'p-w', '--project')
        same_rail(reference, assert_rail(page, lit, states, lit=('p-w',)), ids=[key for key in reference if key != 'p-wu'])
        capture('active')
        page.locator('#project-panel-close').click()
        page.locator('#project-panel').wait_for(state='hidden')

    # A complete empty census hides every marker; the unread column still does not move.
    refresh(active_chat_activities=[])
    page.wait_for_function("() => document.querySelector('[data-project-id=\"p-w\"] .nav-activity-marker')?.hidden === true")
    empty = rail_rows(page)
    for key, row in empty.items():
        assert not row['drawn'] and row['state'] == 'idle', key
        if row['unread']:
            assert horizontal(row['unread']) == pytest.approx(horizontal(reference[key][1]), abs=0.01), key
    capture('empty')
