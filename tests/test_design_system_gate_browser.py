"""P11 integration gate: new shell geometry, all pages/tabs, and real touch scrolling.

No xfails: the L3 branch intentionally exposes L1 shell failures until integration.
Uses the existing static-server/API fixture; never starts the runtime or reads live data.
Synthetic CDP touch is browser evidence, not a claim about an iPhone Telegram WebView.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / 'web'
VIEWPORTS = [
    (360, 844, False), (390, 844, False), (390, 780, True),
    (719, 844, False), (720, 844, False), (721, 844, False),
    (1280, 860, False), (1280, 600, False), (844, 320, False),
]
PROJECTS = [dict(id=f'gate-{i}', name=f'Project {i}: a descriptive project name',
                 chat_id=100+i, lifecycle='active', visible_revision=0) for i in range(30)]
TG_STUB = """(() => {
    const listeners = new Map();
    const button = () => ({isVisible:false, show(){this.isVisible=true}, hide(){this.isVisible=false},
        onClick(fn){this.callback=fn}, offClick(){this.callback=null}});
    window.Telegram = {WebApp: {version:'9.1', platform:'ios', initData:'gate-fixture',
        colorScheme:'light', themeParams:{}, isExpanded:true,
        viewportHeight:innerHeight, viewportStableHeight:innerHeight,
        safeAreaInset:{top:0,bottom:0,left:0,right:0}, contentSafeAreaInset:{top:0,bottom:0,left:0,right:0},
        BackButton:button(), SettingsButton:button(), ready(){}, expand(){},
        isVersionAtLeast(){return true}, disableVerticalSwipes(){}, enableVerticalSwipes(){},
        setHeaderColor(){}, setBackgroundColor(){}, setBottomBarColor(){}, hideKeyboard(){},
        onEvent(name,fn){listeners.set(name,fn)}, offEvent(name){listeners.delete(name)},
        HapticFeedback:{impactOccurred(){}}, openLink(){}}};
})()"""

# What a person sees: a box with a size that nothing hides (an ancestor invisible or transparent, or a closed
# <details> other than at its summary: an engine may keep layout boxes for a closed <details>' content).
SHOWN = """e => {
        if (!e.getClientRects().length || !e.getBoundingClientRect().width ||
            !e.getBoundingClientRect().height || e.closest('[hidden], [inert]')) return false;
        for (let p=e; p; p=p.parentElement) {
            const s=getComputedStyle(p);
            if (s.visibility === 'hidden' || s.opacity === '0') return false;
            if (p.matches('details:not([open])') && !p.querySelector(':scope > summary')?.contains(e)) return false;
        }
        return true;
    }"""

# Deliberate wide content owns a local scroller. Never exempt a page scroll body
# merely because it happens to contain a table or pre somewhere below it.
MEASURE = """root => {
    const shown = """ + SHOWN + """;
    const name = e => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') +
        [...e.classList].slice(0,3).map(c=>'.'+c).join('');
    const local = e => e.matches('.app-tab-strip, pre, .md-table-wrap, .widget-table-wrap, ' +
        '.costs-tables-grid > div, .files-preview-content, .files-editor');
    const overflow = [], small = [], headers = [];
    const scope = document.querySelector(root);
    const doc = document.scrollingElement;
    // The 44px rule is a touch rule: it applies when the pointer is coarse, and it measures the HIT area, not the
    // box. A control passes when its centre ±21px (clipped to the viewport) lands on it or on something inside it
    // (an invisible hit extension counts); otherwise its box is reported. A control below the fold is brought into
    // view first, where a finger would meet it.
    const coarse = matchMedia('(pointer: coarse)').matches;
    const hits = (e, r) => {
        if (r.bottom <= 0 || r.top >= innerHeight || r.right <= 0 || r.left >= innerWidth) {
            e.scrollIntoView({block: 'center', inline: 'nearest'});
            r = e.getBoundingClientRect();
        }
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        const clampX = x => Math.min(Math.max(x, 0), innerWidth - 1);
        const clampY = y => Math.min(Math.max(y, 0), innerHeight - 1);
        return [[cx - 21, cy], [cx + 21, cy], [cx, cy - 21], [cx, cy + 21]].every(([x, y]) => {
            const t = document.elementFromPoint(clampX(x), clampY(y));
            return Boolean(t) && (t === e || e.contains(t));
        });
    };
    if (doc.scrollWidth > doc.clientWidth + 1)
        overflow.push({node:'document.scrollingElement',extra:doc.scrollWidth-doc.clientWidth});
    for (const e of [scope, ...scope.querySelectorAll('*')].filter(shown)) {
        const s = getComputedStyle(e), r = e.getBoundingClientRect();
        // A page body scrolls sideways only when it CAN scroll (overflow-x auto/scroll); a clipping container
        // (overflow: hidden) with a clipped child, such as a closed Effort strip, is not a page scroll.
        const pageBody = ['auto','scroll'].includes(s.overflowX) &&
            (e.matches('.page, .settings-scroll, .skills-scroll, .widgets-scroll, .files-list, .dashboard-panel') ||
             ['auto','scroll'].includes(s.overflowY));
        if (pageBody && !local(e) && e.scrollWidth > e.clientWidth + 1)
            overflow.push({node:name(e),extra:e.scrollWidth-e.clientWidth});
        if (local(e) && (r.left < -1 || r.right > innerWidth + 1))
            overflow.push({node:name(e),outside:[r.left,r.right]});
        if (coarse && e.matches('button, a[href], summary, select') &&
            (r.width < 43.9 || r.height < 43.9) && !hits(e, r))
            small.push({node:name(e),label:(e.getAttribute('aria-label') || e.textContent).trim().slice(0,65),
                width:Math.round(r.width*10)/10,height:Math.round(r.height*10)/10});
        if (e.matches('.app-page-header') && r.height > 56.1)
            headers.push({node:name(e),height:Math.round(r.height*10)/10});
    }
    return {overflow, small, headers};
}"""


@pytest.fixture(params=['webkit', 'chromium'])
def gate_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


def seed_page(ui, width, height, telegram=False, touch=None):
    # A narrow page is a phone (touch); a wide one is a desk unless the test drives a finger itself.
    page = ui['new_page'](viewport={'width': width, 'height': height}, has_touch=width < 720 if touch is None else touch)
    page.set_default_timeout(10_000)
    origin = urlparse(ui['url']).netloc
    page.route('**/*', lambda r: r.fallback() if urlparse(r.request.url).netloc == origin else r.abort())
    if telegram:
        page.add_init_script(TG_STUB)
        # Put the host flag on the real document before any app script runs.
        html = (WEB / 'index.html').read_text(encoding='utf-8').replace('<html', '<html data-ouroboros-host="telegram"', 1)
        page.route(ui['url'] + '/', lambda r: r.fulfill(content_type='text/html', body=html))
    payloads = {
        '/api/projects': {'projects': PROJECTS},
        '/api/state': {'supervisor_ready': True, 'active_chat_activities': [], 'projects': PROJECTS,
                       'project_chat_ids': [p['chat_id'] for p in PROJECTS]},
        '/api/files/list*': {'path': '.', 'parent_path': '.', 'entries': [
            {'name': 'project_notes_with_a_long_descriptive_name.md', 'path': 'notes.md', 'type': 'file', 'size': 50}]},
        '/api/extensions': {'skills': [dict(name='gate_skill', description='An installed skill with access details.',
            enabled=True, loaded=True, review_status='pass', review_gate='pass', source='self_authored',
            requested_keys=['EXAMPLE_SERVICE_TOKEN'], granted_keys=['EXAMPLE_SERVICE_TOKEN'], tools=['gate_tool'])],
            'ui_tabs': [dict(skill='gate_skill', tab_id='gate-widget', title='Gate widget', render={
                'kind':'declarative', 'schema_version':1, 'components':[
                    {'type':'metric','label':'Processed','value':'42'},
                    {'type':'callout','text':'A widget uses the same contained card on every screen.'}]} )], 'live': {}},
        '/api/tasks?queue_only=1': {'tasks': []}, '/api/schedules': {'schedules': []},
        '/api/cost-breakdown': {'total_calls': 4, 'accounting': {
            'settled_usd':25, 'confirmed_usd':25, 'estimated_usd':0, 'reserved_usd':0,
            'unresolved_upper_bound_usd':0, 'unknown_unmetered':0, 'limit_usd':100, 'cost_final':True},
            'by_model': {'long_model_identifier_' * 12: {'cost':25, 'calls':4}}},
    }
    payloads['/api/widgets'] = {'ui_tabs': payloads['/api/extensions'].pop('ui_tabs')}
    def response(body):
        return lambda route: route.fulfill(json=body)
    for path, body in payloads.items():
        page.route('**' + path, response(body))
    return page


def settle(page):
    page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')


# The shell at rest (shell_motion.js): no room still leaving and no animation or transition running that ends (the
# moving marks' wave loops for good). A place picked in the phone drawer changes inside the room card over a beat and
# the drawer's close, so what a test reads next is read once that motion is over.
AT_REST = """() => !document.querySelector('[data-leaving]') && document.getAnimations().every((a) =>
    a.playState !== 'running' || a.effect?.getComputedTiming?.().iterations === Infinity)"""


def rest(page, timeout=5000):
    page.wait_for_function(AT_REST, timeout=timeout)
    settle(page)


def nav_pages(page):
    # The one registry of places (web/modules/pages.js): a new page is gated here with no further edit.
    ids = page.evaluate("async () => (await import('/static/modules/pages.js')).PAGES.map(p => p.id)")
    rows = page.locator('[data-nav-page]').evaluate_all('rows => rows.map(e=>e.dataset.navPage)')
    assert sorted(ids) == sorted(rows), f'registry {ids} and rendered rows {rows} disagree'
    return ids


def open_nav(page, name):
    row = page.locator(f'[data-nav-page="{name}"]')
    toggle = page.locator('[data-mobile-nav-toggle]:visible').first
    drawer = bool(toggle.count())
    if drawer and toggle.get_attribute('aria-expanded') != 'true':
        toggle.click()
    row.click()
    page.locator(f'#page-{name}').wait_for(state='visible')
    rest(page)


def sidebar_metrics(page):
    side = page.locator('#primary-sidebar')
    toggle = page.locator('[data-mobile-nav-toggle]:visible').first
    if toggle.count() and toggle.get_attribute('aria-expanded') != 'true':
        toggle.click()
    settle(page)
    return side.evaluate("""side => [side, ...side.querySelectorAll('*')].filter(e => {
        const s = getComputedStyle(e);
        return e.getClientRects().length && ['auto','scroll'].includes(s.overflowY) &&
            e.scrollHeight > e.clientHeight + 1;
    }).map(e=>({node:e.id || e.className,client:e.clientHeight,scroll:e.scrollHeight}))""")


def save_evidence(tmp_path, name, facts):
    root = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{name}.json'
    path.write_text(json.dumps(facts, indent=2))
    return path


@pytest.mark.parametrize('width,height,telegram', VIEWPORTS,
    ids=[f'{w}x{h}' + ('-telegram' if tg else '') for w,h,tg in VIEWPORTS])
def test_every_page_and_settings_tab_meets_shell_contract(gate_ui, tmp_path, width, height, telegram):
    ui = gate_ui
    page = seed_page(ui, width, height, telegram)
    facts, failures = {}, {}
    try:
        page.goto(ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
        wanted = 'stack' if width < 720 else 'docked'
        actual = page.locator('html').get_attribute('data-layout')
        if actual != wanted:
            failures['layout'] = {'wanted': wanted, 'actual': actual}
        # The 44px rule keys on the pointer: the touch emulation of a narrow page must report a coarse one, or the
        # rule would silently skip (both engines do under has_touch).
        if width < 720:
            assert page.evaluate("matchMedia('(pointer: coarse)').matches"), 'a touch page reports a coarse pointer'
        if telegram:
            assert page.locator('html').get_attribute('data-ouroboros-host') == 'telegram'
            assert page.evaluate('Telegram.WebApp.platform') == 'ios'
        pages = nav_pages(page)
        assert pages and len(pages) == len(set(pages)), pages
        facts['pages'] = pages
        for name in pages:
            open_nav(page, name)
            root = f'#page-{name}'
            if name == 'settings':
                page.wait_for_selector('#available-subagents-editor .available-subagent-row', state='attached')
            if name == 'skills':
                page.wait_for_selector('.skills-card')
            if name == 'widgets':
                page.wait_for_selector('.widgets-card')
            # Settings is mandatory; Dashboard and Skills subviews use the same control.
            tabs = page.locator(root + ' .app-tab-strip [role="tab"]').all()
            for index in range(max(1, len(tabs))):
                tab_name = 'page'
                if tabs:
                    tabs[index].click()
                    tab_name = tabs[index].inner_text().strip()
                settle(page)
                key = f'{name}/{tab_name}'
                facts[key] = page.evaluate(MEASURE, root)
                if any(facts[key].values()):
                    failures[key] = facts[key]
        facts['sidebar'] = sidebar_metrics(page)
        if len(facts['sidebar']) != 1:
            failures['sidebar'] = facts['sidebar']
        # Also audit the open drawer, including its controls and overflow.
        facts['sidebar-controls'] = page.evaluate(MEASURE, '#primary-sidebar')
        if any(facts['sidebar-controls'].values()):
            failures['sidebar-controls'] = facts['sidebar-controls']
    finally:
        path = save_evidence(tmp_path, f'gate-{ui["engine"]}-{width}x{height}-{telegram}', facts)
        page.close()
    summary = {key: {kind: len(rows) for kind, rows in value.items()}
               if isinstance(value, dict) and 'small' in value else value for key,value in failures.items()}
    assert not failures, f'{ui["engine"]} {width}x{height}, evidence {path}:\n{json.dumps(summary, indent=2)}'


def test_main_header_keeps_one_row_whatever_width_its_words_set(gate_ui, tmp_path):
    """DESIGN "Page header": one row that never wraps. At the first docked width Main's bar is narrow (Restart and
    the budget wait in the status Shade since More ended, #2324) and this Mac's face left its commands about ten
    pixels; a wider face (Linux WebKit set them two rows high) moves nothing onto a second row and nothing out of the
    bar: the commands keep their place and the status yields, its word dropping whole while its mark stays. A word
    too wide for its place drops."""
    page = seed_page(gate_ui, 720, 844)
    facts = {}
    try:
        page.goto(gate_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
        # Every word in the bar set wider than any face the shell ships with would set it (a control does not
        # inherit letter-spacing, so each element takes it); then the status word alone, wider than any bar.
        for step, rule in (('wide-face', '#page-chat .chat-page-header, #page-chat .chat-page-header *'),
                           ('wide-word', '#page-chat .chat-page-header .rw-trigger .status-badge')):
            page.add_style_tag(content=rule + (' { letter-spacing: 0.3em; }' if step == 'wide-face'
                                               else ' { letter-spacing: 6em; }'))
            # The word is fitted when its text or the bar's width changes (room_work.js fitWord); a real face is
            # there at first paint, so the bar is made to fit it again: one pixel wider and back.
            for width in (721, 720):
                page.set_viewport_size({'width': width, 'height': 844})
                settle(page)
            facts[step] = page.locator('#page-chat .chat-page-header').evaluate("""bar => {
                const shown = """ + SHOWN + """;
                const box = bar.getBoundingClientRect();
                const within = (e) => { const r = e.getBoundingClientRect();
                    return r.left >= box.left - 1 && r.right <= box.right + 1 && r.top >= box.top - 1
                        && r.bottom <= box.bottom + 1; };
                const commands = [...bar.querySelectorAll('.app-page-actions :is(button, summary)')].filter(shown);
                const status = bar.querySelector('.rw-trigger'), word = status?.querySelector('.status-badge');
                return {height: box.height, commands: commands.map((node) => node.textContent.trim()),
                    outside: commands.filter((node) => !within(node)).map((node) => node.textContent.trim()),
                    mark: Boolean(status && shown(status) && within(status)), dropped: status?.dataset.word === 'off',
                    wordShown: Boolean(word && shown(word)),
                    wordClipped: Boolean(word && word.scrollWidth > word.clientWidth + 1)};
            }""")
        save_evidence(tmp_path, f'gate-{gate_ui["engine"]}-main-header-wide-words', facts)
    finally:
        page.close()
    for step, fact in facts.items():
        assert fact['height'] <= 56.1, (step, fact)
        assert fact['commands'] == ['Panic'] and not fact['outside'], (step, fact)
        # The word shows whole or not at all, and its mark stays in the bar either way.
        assert fact['mark'] and (fact['dropped'] != fact['wordShown']) and not fact['wordClipped'], (step, fact)
    assert facts['wide-word']['dropped'], facts


@pytest.mark.parametrize('width,height', [(390, 844), (844, 320)])
def test_projects_scroll_by_touch_without_a_second_sidebar_scroller(subscription_ui, tmp_path, width, height):
    # CDP is Chromium-specific. The geometry matrix above covers both engines.
    if os.environ.get('OUROBOROS_UI_BROWSER_ENGINE', 'chromium') != 'chromium':
        pytest.skip('CDP Input.dispatchTouchEvent requires Chromium')
    page = seed_page(subscription_ui, width, height, touch=True)
    try:
        page.goto(subscription_ui['url'] + '/')
        page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
        scrollers = sidebar_metrics(page)
        page.locator('#nav-projects-list [data-project-id]').first.scroll_into_view_if_needed()
        settle(page)
        position = page.locator('#nav-projects-list').evaluate("""list => {
            let e = list;
            while (e && !(e.scrollHeight > e.clientHeight+1 &&
                ['auto','scroll'].includes(getComputedStyle(e).overflowY))) e=e.parentElement;
            if (!e) throw Error('Projects has no scrolling pane');
            window.gateScroller=e;
            const r=e.getBoundingClientRect();
            return {x:r.left+r.width/2,y:Math.min(r.bottom,innerHeight)-24,travel:Math.min(140,r.height-48),before:e.scrollTop};
        }""")
        assert position['travel'] >= 24, position
        cdp = page.context.new_cdp_session(page)
        try:
            for i in range(9):
                cdp.send('Input.dispatchTouchEvent', {'type':'touchStart' if i == 0 else 'touchMove',
                    'touchPoints':[{'x':position['x'], 'y':position['y'] - position['travel']*i/8, 'id':1}]})
                page.wait_for_timeout(24)
            cdp.send('Input.dispatchTouchEvent', {'type':'touchEnd', 'touchPoints':[]})
            page.wait_for_timeout(200)
        finally:
            cdp.detach()
        after = page.evaluate('gateScroller.scrollTop')
        facts = {'scrollers':scrollers, 'gesture':position, 'after':after}
        save_evidence(tmp_path, f'gate-touch-{width}x{height}', facts)
        assert after > position['before'] + 8, facts
        assert len(scrollers) == 1, facts
    finally:
        page.close()


def test_content_regions_fit_with_touch_controls_and_local_table_scroll(gate_ui):
    """Independent green proof for L3's bodies while L1's shell is integrating."""
    page = seed_page(gate_ui, 360, 844)
    results = {}
    scopes = {'files': '.files-layout', 'skills': '.skills-scroll',
              'widgets': '.widgets-scroll', 'dashboard': '.dashboard-shell', 'settings': '.settings-shell'}
    try:
        page.goto(gate_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        for name, scope in scopes.items():
            open_nav(page, name)
            if name == 'skills':
                page.wait_for_selector('.skills-card')
            if name == 'widgets':
                page.wait_for_selector('.widgets-card')
            if name == 'dashboard':
                page.locator('[data-dashboard-tab="costs"]').click()
                page.wait_for_function("document.querySelector('#cost-top-model').textContent.length > 200")
                table = page.locator('#cost-by-model').locator('..')
                assert table.evaluate('e => e.scrollWidth > e.clientWidth'), 'fixture must scroll locally'
                table.evaluate('e => e.scrollLeft = e.scrollWidth')
                assert table.evaluate('e => e.scrollLeft > 0')
            if name == 'settings':
                page.wait_for_selector('#available-subagents-editor .available-subagent-row', state='attached')
                for tab in page.locator('[data-settings-tab]').all():
                    tab.click()
                    settle(page)
                    results['settings/' + tab.inner_text().strip()] = page.evaluate(MEASURE, scope)
                    assert tab.bounding_box()['height'] >= 44
            else:
                settle(page)
                results[name] = page.evaluate(MEASURE, scope)
        assert not {k:v for k,v in results.items() if any(v.values())}, json.dumps(results, indent=2)

        # Regression sensitivity: restoring the old ineffective stat-value rule
        # must make the same probe fail, although the table still scrolls locally.
        open_nav(page, 'dashboard')
        page.locator('[data-dashboard-tab="costs"]').click()
        page.add_style_tag(content='.costs-stats-grid .value { overflow-wrap: normal; }')
        settle(page)
        broken = page.evaluate(MEASURE, '.dashboard-shell')
        assert any(row['node'] == 'div.costs-scroll' for row in broken['overflow']), broken
    finally:
        page.close()
