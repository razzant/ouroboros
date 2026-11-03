"""Appearance through real shell documents on the selected Playwright engine."""
import json

import pytest

from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as _subscription_ui

subscription_ui = _subscription_ui  # re-exported pytest fixture (requested by name below)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def open_appearance(page):
    """Reach the only Appearance control the main document has: Settings."""
    page.locator('[data-nav-page="settings"]').click()
    page.wait_for_selector('#page-settings.active')
    page.locator('[data-settings-tab="appearance"]').click()
    page.wait_for_selector('[data-settings-panel="appearance"].active')
    return page.locator('[data-settings-panel="appearance"] [data-theme-control]')


def test_theme_choice_reload_and_main_surfaces(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    page.set_viewport_size({'width': 1100, 'height': 722})
    page.emulate_media(color_scheme='dark')
    page.route('**/api/chat/history*', lambda r: r.fulfill(content_type='application/json', body=json.dumps({'messages':[
        {'role':'user','text':'Please make the interface light and easy to read.','ts':'2026-09-18T10:00:00Z'},
        {'role':'assistant','text':'## Appearance\n\nLight and dark themes keep your conversation intact.\n\n```python\nprint("Hello, world!")\n```','ts':'2026-09-18T10:00:01Z'},
        {'role':'system','text':'This is a system notice.','ts':'2026-09-18T10:00:02Z'}], 'progress':[]})))
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    # The sidebar toggle is gone; Appearance lives in Settings and nowhere else.
    assert page.locator('[data-theme-toggle]').count() == 0
    assert page.locator('#primary-sidebar [data-theme-control]').count() == 0
    # A new client defaults to System, and this engine reports a dark OS.
    assert page.locator('html').get_attribute('data-theme-choice') == 'system'
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    capture(page, 'theme-dark-chat')
    page.locator('#chat-input').fill('Unsent draft stays here')

    control = open_appearance(page)
    assert control.get_attribute('role') == 'radiogroup'
    assert control.locator('[data-theme-choice]').evaluate_all(
        'nodes => nodes.map(n => n.dataset.themeChoice)') == ['light', 'dark', 'system']
    capture(page, 'theme-appearance-panel')
    # Keyboard is the accessible path: the group takes one tab stop and arrows move.
    control.locator('[data-theme-choice="system"]').focus()
    page.keyboard.press('Home')
    assert page.locator('html').get_attribute('data-theme') == 'light'
    assert page.locator('html').get_attribute('data-theme-choice') == 'light'
    assert control.locator('[data-theme-choice="light"]').get_attribute('aria-checked') == 'true'
    assert page.locator('#chat-input').input_value() == 'Unsent draft stays here'
    assert page.evaluate('getComputedStyle(document.body).backgroundColor') == 'rgb(255, 255, 255)'
    assert page.evaluate("getComputedStyle(document.documentElement).colorScheme") == 'light'
    capture(page, 'theme-light-settings')
    for name in ['files', 'skills', 'widgets', 'dashboard']:
        page.locator(f'[data-nav-page="{name}"]').click()
        page.wait_for_selector(f'#page-{name}.active')
        capture(page, f'theme-light-{name}')
    ui['errors'].watch(page)  # the reloads may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('#chat-input')
    assert page.locator('html').get_attribute('data-theme') == 'light'
    open_appearance(page).locator('[data-theme-choice="dark"]').click()
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    page.reload()
    page.wait_for_selector('#chat-input')
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    # System is reachable again and is a distinct stored state, not a synonym
    # for whatever it happens to resolve to right now.
    open_appearance(page).locator('[data-theme-choice="system"]').click()
    assert page.evaluate("localStorage.getItem('ouroboros.theme')") == 'system'
    assert page.locator('html').get_attribute('data-theme-choice') == 'system'
    page.emulate_media(color_scheme='light')
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    assert page.locator('html').get_attribute('data-theme') == 'light'
    page.emulate_media(color_scheme='dark')
    page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    assert not ui['posts'], 'Appearance must never write runtime settings'


def test_theme_onboarding_tristate(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    # Enough space for the real Accounts content: smaller windows may scroll
    # legitimately. The regression is a forced viewport-sized shell PLUS control.
    page.set_viewport_size({'width': 1360, 'height': 1080})
    page.add_init_script("localStorage.setItem('ouroboros.theme','light')")
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_selector('#root .wizard-shell')
    # A choice saved before System existed keeps its exact old meaning.
    assert page.locator('html').get_attribute('data-theme') == 'light'
    assert page.locator('html').get_attribute('data-theme-choice') == 'light'
    control = page.locator('.onboarding-appearance [data-theme-control]')
    assert control.locator('[data-theme-choice]').evaluate_all(
        'nodes => nodes.map(n => n.dataset.themeChoice)') == ['light', 'dark', 'system']
    capture(page, 'theme-light-onboarding')
    # The wizard preserves the saved Light choice and must not be taller than
    # a sufficiently large window merely because it carries the control.
    overflow = page.evaluate(
        'document.documentElement.scrollHeight - document.documentElement.clientHeight')
    assert overflow <= 1, f'onboarding overflows by {overflow}px'
    control.locator('[data-theme-choice="dark"]').click()
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    capture(page, 'theme-dark-onboarding')
    control.locator('[data-theme-choice="system"]').click()
    assert page.evaluate("localStorage.getItem('ouroboros.theme')") == 'system'


# Every frame gets a recording stand-in bridge, so a framed document that spoke for its
# parent's window would be caught by its own record, not just by a missing bridge.
_CAPTION_BRIDGE = """
    window.pywebview = {api: {set_native_appearance: (...args) => {
        const record = (window.top.__captionCalls = window.top.__captionCalls || []);
        record.push([location.pathname, ...args]);
        return Promise.resolve({ok: true, state: 'scheduled'});
    }}};
"""


def test_the_desktop_caption_hears_only_the_top_document(subscription_ui):
    """The real SPA tells its window's bridge the painted palette at boot, when the bridge
    announces itself and on each painted change; the setup document framed inside it never
    calls, even with a bridge of its own. The frame's choice still reaches the caption: it
    is stored, the top document repaints from the storage event and reports that."""
    ui = subscription_ui
    page = ui['page']
    page.emulate_media(color_scheme='dark')
    page.add_init_script(_CAPTION_BRIDGE)
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    page.evaluate("window.dispatchEvent(new CustomEvent('pywebviewready'))")
    open_appearance(page).locator('[data-theme-choice="light"]').click()
    page.evaluate("""() => new Promise((done) => {
        const frame = Object.assign(document.createElement('iframe'), {src: '/onboarding', id: 'caption-frame'});
        frame.addEventListener('load', () => done(), {once: true});
        document.body.append(frame);
    })""")
    frame = page.frame_locator('#caption-frame')
    frame.locator('.onboarding-appearance [data-theme-control] [data-theme-choice="dark"]').click()
    page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    page.evaluate("document.getElementById('caption-frame').contentWindow.dispatchEvent(new CustomEvent('pywebviewready'))")
    page.wait_for_function("window.__captionCalls.length >= 4")
    calls = page.evaluate('window.__captionCalls')
    assert {path for path, *_ in calls} == {'/'}, calls
    assert [(theme, sequence) for _path, theme, _page, sequence in calls] == [
        ('dark', 1), ('dark', 2), ('light', 3), ('dark', 4)]
    # The page names itself by when it began; the reload that replaces it began later.
    began = page.evaluate('performance.timeOrigin')
    assert {token for _path, _theme, token, _sequence in calls} == {began}
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('#chat-input')
    page.evaluate("window.dispatchEvent(new CustomEvent('pywebviewready'))")
    page.wait_for_function("window.__captionCalls && window.__captionCalls.length >= 2")
    reloaded = page.evaluate('window.__captionCalls')
    assert reloaded[0][2] == page.evaluate('performance.timeOrigin') > began, (began, reloaded)


def test_light_contrast_and_mounted_views_follow_the_theme(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    open_appearance(page).locator('[data-theme-choice="light"]').click()
    colors=page.evaluate("""() => {
        const c=getComputedStyle(document.documentElement);
        return ['--text-primary','--text-meta','--text-secondary','--status-ok-fg','--status-warn-fg','--status-error-fg'].map(k=>c.getPropertyValue(k).trim());
    }""")
    def luminance(color):
        rgb=[int(color[i:i+2],16)/255 for i in [1,3,5]]
        linear=[v/12.92 if v<=0.04045 else ((v+0.055)/1.055)**2.4 for v in rgb]
        return sum(a*b for a,b in zip(linear,[0.2126,0.7152,0.0722]))
    for color in colors:
        assert 1.05/(luminance(color)+0.05)>=4.5, color
    # Exercise the real markdown consumer. A diagram mounted under one palette
    # must REPAINT when the owner switches, without the source being re-fetched:
    # the node's rendered SVG has long since replaced the text it was built from.
    page.evaluate("""async () => {
        const {enhanceChatMarkdown}=await import('/static/modules/chat_markdown.js');
        const root=document.createElement('div');root.id='mounted-views';
        const node=document.createElement('div');node.className='md-mermaid';
        node.textContent='graph TD; A[Readable] --> B[Theme]';root.append(node);
        const chart=document.createElement('div');chart.className='md-chart';
        chart.textContent=JSON.stringify({type:'line',data:{labels:['a','b'],
            datasets:[{label:'series',data:[1,2]},{label:'authored',data:[2,1],borderColor:'#123456',
            backgroundColor:'#654321',ouroSeries:0}]}});root.append(chart);
        document.querySelector('#chat-messages').append(root);enhanceChatMarkdown(root);
    }""")
    page.locator('[data-nav-page="chat"]').click()
    page.wait_for_selector('#mounted-views svg')
    page.wait_for_selector('#mounted-views canvas')
    assert '#f3e8ec' in page.locator('#mounted-views').inner_html().lower()
    before = page.evaluate(
        "() => globalThis.Chart.getChart(document.querySelector('#mounted-views canvas')).data.datasets[0].data")
    # An author's chart keeps its own series colours: the host's default series never reach it,
    # even when its source names the host's mark (chat_markdown.js keeps only its own keys).
    authored_series = """() => {
        const c = globalThis.Chart.getChart(document.querySelector('#mounted-views canvas'));
        return [c.data.datasets.map((d) => [d.borderColor ?? null, d.backgroundColor ?? null, 'ouroSeries' in d]),
            [...new Set(c.getDatasetMeta(1).data.map((el) => `${el.options.borderColor} ${el.options.backgroundColor}`))]];
    }"""
    series_before = page.evaluate(authored_series)
    assert series_before == [[[None, None, False], ['#123456', '#654321', False]], ['#123456 #654321']], series_before
    open_appearance(page).locator('[data-theme-choice="dark"]').click()
    page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
    dark_primary = page.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--diagram-primary').trim().toLowerCase()")
    assert dark_primary.startswith('#') and dark_primary != '#f3e8ec', dark_primary
    page.wait_for_function(
        "(ink) => document.querySelector('#mounted-views').innerHTML.toLowerCase().includes(ink)", arg=dark_primary)
    after = page.evaluate("""() => {
        const c = globalThis.Chart.getChart(document.querySelector('#mounted-views canvas'));
        return { data: c.data.datasets[0].data, color: c.options.color };
    }""")
    assert after['data'] == before, 'a repaint must not rebuild the chart'
    assert after['color'] == page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--chart-text').trim()")
    assert page.evaluate(authored_series) == series_before


# Owns-a-listener accounting: which `ouro:theme-changed` subscriptions a widget
# mount registered and has not released yet (registration stack names widgets.js).
WIDGET_THEME_LISTENERS = """(() => {
    const owned = new Set();
    const add = EventTarget.prototype.addEventListener;
    const remove = EventTarget.prototype.removeEventListener;
    window.addEventListener = function (type, fn, options) {
        if (type === 'ouro:theme-changed' && /\\/modules\\/widgets\\.js/.test(new Error().stack || '')) owned.add(fn);
        return add.call(this, type, fn, options);
    };
    window.removeEventListener = function (type, fn, options) {
        if (type === 'ouro:theme-changed') owned.delete(fn);
        return remove.call(this, type, fn, options);
    };
    window.widgetThemeListeners = () => owned.size;
})()"""

WIDGET_CHART = """() => {
    const canvas = document.querySelector('#page-widgets canvas[data-widget-chart-key]');
    const chart = canvas && globalThis.Chart.getChart(canvas);
    if (!chart?.data?.datasets?.[0] || !chart.scales?.x || !chart.scales?.y) return null;
    const root = getComputedStyle(document.documentElement);
    const {x, y} = chart.scales;
    return {
        probe: chart === window.widgetChart, data: chart.data.datasets[0].data,
        color: chart.options.color, legend: chart.options.plugins.legend.labels.color,
        ticks: [x.options.ticks.color, y.options.ticks.color, y.options.title.color],
        grid: [x.options.grid.color, y.options.grid.color],
        text: root.getPropertyValue('--chart-text').trim(), line: root.getPropertyValue('--chart-grid').trim(),
        settled: !globalThis.Chart.animator?.running(chart),
    };
}"""

# The default series the owner approved (#474, 27.10), in order; a fifth series takes the first again.
APPROVED_SERIES = {'light': ['#c0284c', '#2c68d0', '#1e9c76', '#bd6a00'],
                   'dark': ['#ee6780', '#60a5fa', '#3fdca3', '#fbbf24']}

# A settled host chart's series as Chart.js draws them (the line and every point or bar), and its
# canvas against a fresh chart of the same data drawn in the approved colours: a colour left
# behind anywhere differs in pixels. Also counts which approved colours the canvas shows at all.
SERIES_RENDER = """async ({ selector, approved }) => {
    const canvas = document.querySelector(selector);
    const chart = canvas && globalThis.Chart.getChart(canvas);
    if (!chart || globalThis.Chart.animator.running(chart)) return null;
    const theme = document.documentElement.dataset.theme;
    const rgb = (hex) => [16, 8, 0].map((shift) => (parseInt(hex.slice(1), 16) >> shift) & 255);
    const want = chart.data.datasets.map((_, index) => {
        const hex = approved[theme][index % 4];
        return [hex, `rgba(${rgb(hex).join(', ')}, 0.22)`];
    });
    const drawn = chart.data.datasets.map((dataset, index) => {
        const meta = chart.getDatasetMeta(index);
        return { declared: [dataset.borderColor, dataset.backgroundColor], line: meta.dataset?.options.borderColor ?? null,
                 marks: [...new Set(meta.data.map((el) => `${el.options.borderColor} ${el.options.backgroundColor}`))] };
    });
    const { applyChartTheme } = await import('/static/modules/theme_palette.js');
    const box = canvas.parentElement.getBoundingClientRect();
    const twin = document.createElement('div');
    twin.style.cssText = `position:fixed;left:-20000px;top:0;width:${box.width}px;height:${box.height}px`;
    const twinCanvas = twin.appendChild(document.createElement('canvas'));
    document.body.append(twin);
    const data = JSON.parse(JSON.stringify(chart.data));
    data.datasets.forEach((dataset, index) => { [dataset.borderColor, dataset.backgroundColor] = want[index]; delete dataset.ouroSeries; });
    const options = { ...JSON.parse(canvas.dataset.widgetChartConfig).options, animation: false };
    const reference = new globalThis.Chart(twinCanvas, { type: chart.config.type, data, options });
    applyChartTheme(reference);
    const pixels = (c) => c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const live = pixels(canvas), fresh = pixels(twinCanvas);
    let differing = live.length === fresh.length ? 0 : -1;
    for (let i = 0; differing >= 0 && i < live.length; i += 4) {
        if ([0, 1, 2, 3].some((k) => Math.abs(live[i + k] - fresh[i + k]) > 8)) differing += 1;
    }
    reference.destroy();
    twin.remove();
    const shown = Object.fromEntries([...approved.light, ...approved.dark].map((hex) => [hex, 0]));
    for (let i = 0; i < live.length; i += 4) {
        if (live[i + 3] < 200) continue;
        for (const hex of Object.keys(shown)) {
            const [r, g, b] = rgb(hex);
            if (Math.abs(live[i] - r) + Math.abs(live[i + 1] - g) + Math.abs(live[i + 2] - b) <= 18) shown[hex] += 1;
        }
    }
    return { theme, type: chart.config.type, want, drawn, differing, shown };
}"""


def assert_series(page, selector):
    """The settled chart under ``selector`` draws exactly the approved series of the painted theme."""
    page.wait_for_function('(selector) => { const c = globalThis.Chart.getChart(document.querySelector(selector)); '
                           'return Boolean(c) && !globalThis.Chart.animator.running(c); }', arg=selector, polling=100)
    facts = page.evaluate(SERIES_RENDER, {'selector': selector, 'approved': APPROVED_SERIES})
    for (stroke, inner), drawn in zip(facts['want'], facts['drawn']):
        assert drawn['declared'] == [stroke, inner], facts
        assert drawn['line'] in (None, stroke), facts
        assert drawn['marks'] == [f'{stroke} {inner}'], facts
    assert facts['differing'] == 0, facts
    other = 'dark' if facts['theme'] == 'light' else 'light'
    assert not [hex for hex in APPROVED_SERIES[other] if facts['shown'][hex]], facts
    assert all(facts['shown'][stroke] for stroke, _ in facts['want']), facts
    return facts


def test_widget_chart_repaints_in_place_and_releases_its_theme_listener(subscription_ui):
    """A declarative Widgets chart through the real page: mount, data update with
    a fifth series, OS appearance switch while visible and again while new data
    animates, dispose on leave, remount after a switch. Its series follow the
    painted theme as drawn, not only as declared (#474)."""
    ui = subscription_ui
    page = ui['page']
    first, refreshed, later = [3, 5, 4], [6, 2, 7], [1, 8, 3]
    payloads = iter([refreshed, later])
    series = {'data': first, 'count': 3}

    def extension(route):
        if route.request.method == 'POST':
            series['data'], series['count'] = next(payloads), 5
        route.fulfill(content_type='application/json', body=json.dumps({'chart': {
            'labels': ['Mon', 'Tue', 'Wed'], 'datasets': [
                {'label': f'Requests {index + 1}', 'data': [value + index for value in series['data']]}
                for index in range(series['count'])]}}))

    page.route('**/api/widgets', lambda r: r.fulfill(content_type='application/json', body=json.dumps({'ui_tabs': [{
        'skill': 'theme_probe', 'tab_id': 'traffic', 'title': 'Traffic',
        'render': {'kind': 'declarative', 'schema_version': 1, 'components': [
            {'type': 'poll', 'id': 'load', 'label': 'Reload', 'route': 'series', 'target': 'live',
             'auto_start': True, 'max_ticks': 1},
            {'type': 'action', 'id': 'refresh', 'label': 'Refresh', 'route': 'series', 'method': 'POST',
             'target': 'live'},
            {'type': 'chart', 'id': 'traffic', 'label': 'Traffic', 'target': 'live', 'path': 'chart',
             'unit': 'req/s'},
        ]}}]})))
    page.route('**/api/extensions/theme_probe/**', extension)
    page.add_init_script(WIDGET_THEME_LISTENERS)
    page.emulate_media(color_scheme='dark')
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    assert page.locator('html').get_attribute('data-theme-choice') == 'system'
    assert page.evaluate('widgetThemeListeners()') == 0

    def chart_when(condition):
        page.wait_for_function(f'() => {{ const c = ({WIDGET_CHART})(); return Boolean(c && ({condition})); }}')
        return page.evaluate(WIDGET_CHART)

    def assert_themed(chart):
        assert chart['color'] == chart['legend'] == chart['text'], chart
        assert chart['ticks'] == [chart['text']] * 3, chart
        assert chart['grid'] == [chart['line']] * 2, chart

    # Mount: the chart is born in the palette that is painted right now.
    canvas = '#page-widgets canvas[data-widget-chart-key]'
    page.locator('[data-nav-page="widgets"]').click()
    page.wait_for_selector('#page-widgets.active')
    dark = chart_when(f'c.data.join() === "{",".join(map(str, first))}" && c.settled')
    assert_themed(dark)
    assert_series(page, canvas)
    assert page.evaluate('widgetThemeListeners()') == 1
    page.evaluate(f"window.widgetChart = globalThis.Chart.getChart(document.querySelector('{canvas}'))")
    capture(page, 'theme-dark-widget-chart')
    # The OS turns light while Widgets is on screen: repaint, not rebuild.
    page.emulate_media(color_scheme='light')
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    light = chart_when('c.color === c.text && c.settled')
    assert light['probe'], 'a theme switch must not replace the mounted chart'
    assert light['data'] == first, 'a theme switch must keep the plotted data'
    assert_themed(light)
    assert_series(page, canvas)
    assert (light['text'], light['line']) != (dark['text'], dark['line'])
    assert page.evaluate('widgetThemeListeners()') == 1
    # Update: a widget action feeds new data, five series now, into the SAME instance, in Light.
    page.locator('#page-widgets [data-widget-action]').click()
    updated = chart_when(f'c.data.join() === "{",".join(map(str, refreshed))}" && c.settled')
    assert updated['probe'], 'a data update must reuse the mounted chart'
    assert_themed(updated)
    assert len(assert_series(page, canvas)['want']) == 5
    capture(page, 'theme-light-widget-chart')
    # New data still animating when the OS turns dark again: the same instance lands in Dark.
    page.locator('#page-widgets [data-widget-action]').click()
    chart_when(f'c.data.join() === "{",".join(map(str, later))}" && !c.settled')
    page.emulate_media(color_scheme='dark')
    redark = chart_when(f'c.text === {json.dumps(dark["text"])} && c.settled')
    assert redark['probe'] and redark['data'] == later, redark
    assert_themed(redark)
    assert_series(page, canvas)

    # Dispose: leaving Widgets destroys the chart and releases its subscription,
    # so a later switch reaches nothing that belonged to the old mount.
    control = open_appearance(page)
    assert page.evaluate('widgetThemeListeners()') == 0
    assert page.evaluate('window.widgetChart.canvas === null'), 'leaving must destroy the chart'
    control.locator('[data-theme-choice="light"]').click()
    assert page.locator('html').get_attribute('data-theme') == 'light'
    assert page.evaluate('widgetThemeListeners()') == 0
    # Remount after the switch: a fresh instance, already in the new palette.
    page.locator('[data-nav-page="widgets"]').click()
    remounted = chart_when('!c.probe && c.settled')
    assert remounted['data'] == later
    assert (remounted['text'], remounted['line']) == (light['text'], light['line'])
    assert_themed(remounted)
    assert_series(page, canvas)
    assert page.evaluate('widgetThemeListeners()') == 1
    assert not [path for path, _ in ui['posts'] if path == '/api/settings']


CARD_CANVAS = '.chat-card[data-card-id="card-plan"] canvas[data-widget-chart-key]'
CARD_THEME_LISTENERS = WIDGET_THEME_LISTENERS.replace('widgets\\.js', 'card_components\\.js')
assert CARD_THEME_LISTENERS != WIDGET_THEME_LISTENERS


def test_living_card_chart_series_follow_the_theme_and_a_new_revision(subscription_ui):
    """A living card's bar chart through the real chat feed (chat_cards.js, card_components.js):
    born in the painted theme with its fifth series in the first colour, repainted in place when
    the OS switches while it is on screen, drawn again for a new revision with another series
    count, and the replaced mount's chart and theme listener released (#474)."""
    ui = subscription_ui
    page = ui['page']
    sockets, shown = [], {'revision': 1}
    sports = ['Running', 'Cycling', 'Swimming', 'Strength', 'Stretching']

    def body(revision):
        count = 5 if revision == 1 else 3
        return {'card_id': 'card-plan', 'kind': 'components', 'title': 'Training plan', 'chat_id': 1,
                'fallback_text': 'Training plan by week', 'revision': revision, 'pinned': False,
                'submissions_summary': [],
                'components': [{'type': 'chart', 'id': 'plan', 'label': 'Training plan', 'chart_type': 'bar',
                                'unit': 'min', 'path': 'plan'}],
                'data': {'plan': {'labels': ['Week 1', 'Week 2', 'Week 3', 'Week 4'], 'datasets': [
                    {'label': sport, 'data': [30 + 10 * index + revision, 40, 35 + index, 20]}
                    for index, sport in enumerate(sports[:count])]}}}

    card = {'card_id': 'card-plan', 'kind': 'components', 'title': 'Training plan'}
    page.route('**/api/chat/history*', lambda r: r.fulfill(content_type='application/json', body=json.dumps({
        'messages': [{'role': 'user', 'text': 'Plan my November training by week.', 'ts': '2026-10-27T10:00:00Z'},
                     {'role': 'assistant', 'msg_type': 'card', 'card': {**card, 'revision': 1},
                      'fallback_text': 'Training plan by week', 'text': 'Training plan by week',
                      'ts': '2026-10-27T10:00:01Z'}], 'progress': []})))
    page.route('**/api/cards/card-plan', lambda r: r.fulfill(content_type='application/json',
                                                             body=json.dumps(body(shown['revision']))))
    page.route_web_socket('**/ws', lambda ws: (sockets.append(ws), ws.send(json.dumps({'type': 'heartbeat'}))))
    page.add_init_script(CARD_THEME_LISTENERS)
    page.emulate_media(color_scheme='dark')
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    page.wait_for_selector(CARD_CANVAS)
    born = assert_series(page, CARD_CANVAS)
    assert born['type'] == 'bar' and len(born['want']) == 5 and born['want'][4] == born['want'][0], born
    assert page.evaluate('widgetThemeListeners()') == 1
    page.evaluate(f"window.cardChart = globalThis.Chart.getChart(document.querySelector('{CARD_CANVAS}'))")
    capture(page, 'theme-dark-card-chart')
    page.emulate_media(color_scheme='light')
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    assert_series(page, CARD_CANVAS)
    assert page.evaluate(f"globalThis.Chart.getChart(document.querySelector('{CARD_CANVAS}')) === window.cardChart"), \
        'a theme switch must repaint the card chart, not rebuild it'
    capture(page, 'theme-light-card-chart')
    # A new revision: the feed draws the card again, three series now, and the old mount lets go.
    shown['revision'] = 2
    sockets[-1].send(json.dumps({'type': 'card_state', 'card': {**card, 'revision': 2}, 'submissions_summary': [],
                                 'submissions_dropped': 0, 'ts': '2026-10-27T10:01:00Z'}))
    page.wait_for_function(f"() => {{ const c = globalThis.Chart.getChart(document.querySelector('{CARD_CANVAS}')); "
                           "return Boolean(c) && c !== window.cardChart && c.data.datasets.length === 3; }")
    assert page.evaluate('window.cardChart.canvas === null'), 'the replaced mount must destroy its chart'
    assert len(assert_series(page, CARD_CANVAS)['want']) == 3
    assert page.evaluate('widgetThemeListeners()') == 1
    page.emulate_media(color_scheme='dark')
    page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    assert_series(page, CARD_CANVAS)


def test_module_widget_theme_retains_state_when_hidden_and_releases_on_stop(subscription_ui):
    """A real module frame follows a pinned theme while retained off-page.

    The child owns a counter and registers an onTheme disposer. The parent
    must preserve that frame and state across navigation, then release the
    theme listener when the owner stops the widget.
    """
    ui = subscription_ui
    page = ui['page']
    page.emulate_media(color_scheme='dark')
    page.add_init_script("""(() => {
        const add = EventTarget.prototype.addEventListener;
        const remove = EventTarget.prototype.removeEventListener;
        window.__moduleThemeAdds = 0; window.__moduleThemeRemoves = 0;
        window.__moduleThemeListeners = new Set();
        EventTarget.prototype.addEventListener = function(type, fn, options) {
            if (type === 'ouro:theme-changed' && /widget_module\\.js/.test(new Error().stack || '')) {
                window.__moduleThemeAdds++; window.__moduleThemeListeners.add(fn);
            }
            return add.call(this, type, fn, options);
        };
        EventTarget.prototype.removeEventListener = function(type, fn, options) {
            if (type === 'ouro:theme-changed') {
                window.__moduleThemeRemoves++; window.__moduleThemeListeners.delete(fn);
            }
            return remove.call(this, type, fn, options);
        };
    })()""")
    source = """(() => {
        const root = document.getElementById('root'); let count = 0; let off = () => {};
        root.innerHTML = '<button id="b" type="button">add</button><span id="state"></span>';
        const paint = theme => { document.documentElement.dataset.theme = theme; root.dataset.theme = theme; root.querySelector('#state').textContent = theme + ':' + count; };
        off = OuroborosWidget.onTheme(paint);
        root.querySelector('#b').onclick = () => { count++; paint(document.documentElement.dataset.theme); };
        __ouroWidgetOnDispose(() => off());
    })();"""
    page.route('**/api/widgets', lambda r: r.fulfill(content_type='application/json', body=json.dumps({'ui_tabs': [{
        'skill': 'theme_probe', 'tab_id': 'module', 'title': 'Theme probe',
        'render': {'kind': 'module', 'entry': 'theme.js', 'height': 360, 'start': 'retain', 'appearance': 'host'},
    }]})))
    page.route('**/api/extensions/theme_probe/module/theme.js', lambda r: r.fulfill(content_type='text/javascript', body=source))
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    page.locator('[data-nav-page="widgets"]').click()
    page.wait_for_selector('#page-widgets.active')
    card = page.locator('[data-widget-key="theme_probe:module"]')
    card.wait_for(state='visible')
    frame_node = card.locator('iframe').element_handle()
    frame = frame_node.content_frame()
    frame.locator('#state').wait_for()
    assert frame.locator('#state').inner_text() == 'dark:0'
    frame.locator('#b').click()
    assert frame.locator('#state').inner_text() == 'dark:1'
    assert page.evaluate('window.__moduleThemeListeners.size') == 1
    page.evaluate("() => { document.querySelector('[data-widget-key=\\\"theme_probe:module\\\"] iframe').__themeProbe = true; }")
    open_appearance(page).locator('[data-theme-choice="light"]').click()
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    assert page.evaluate('window.__moduleThemeListeners.size') == 1
    assert page.evaluate("() => document.querySelector('[data-widget-key=\\\"theme_probe:module\\\"] iframe').__themeProbe === true")
    retained = card.locator('iframe').element_handle().content_frame()
    assert retained.locator('#state').inner_text() == 'light:1'
    page.locator('[data-nav-page="widgets"]').click()
    page.wait_for_selector('#page-widgets.active')
    card.locator('[data-widget-power]').click()
    page.wait_for_function('window.__moduleThemeListeners.size === 0')


def test_author_kit_native_scheme_is_opt_in(subscription_ui):
    """Loading the shared kit must not recolour unrelated native page controls."""
    ui = subscription_ui
    page = ui['page']
    css = page.request.get(ui['url'] + '/static/ui.css').text()
    page.set_content('<html><head><style>' + css + '</style></head><body>'
                     '<input id="outside"><div class="ouro-ui"><input id="inside"></div>'
                     '</body></html>')
    for theme in ('dark', 'light'):
        page.evaluate('(theme) => document.documentElement.dataset.theme = theme', theme)
        assert page.locator('#outside').evaluate('(e) => getComputedStyle(e).colorScheme') == 'normal'
        assert page.locator('#inside').evaluate('(e) => getComputedStyle(e).colorScheme') == theme
