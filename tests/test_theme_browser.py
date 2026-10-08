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
            datasets:[{label:'series',data:[1,2]}]}});root.append(chart);
        document.querySelector('#chat-messages').append(root);enhanceChatMarkdown(root);
    }""")
    page.locator('[data-nav-page="chat"]').click()
    page.wait_for_selector('#mounted-views svg')
    page.wait_for_selector('#mounted-views canvas')
    assert '#f3e8ec' in page.locator('#mounted-views').inner_html().lower()
    before = page.evaluate(
        "() => globalThis.Chart.getChart(document.querySelector('#mounted-views canvas')).data.datasets[0].data")
    open_appearance(page).locator('[data-theme-choice="dark"]').click()
    page.wait_for_function(
        "() => document.querySelector('#mounted-views').innerHTML.toLowerCase().includes('#25222c')")
    after = page.evaluate("""() => {
        const c = globalThis.Chart.getChart(document.querySelector('#mounted-views canvas'));
        return { data: c.data.datasets[0].data, color: c.options.color };
    }""")
    assert after['data'] == before, 'a repaint must not rebuild the chart'
    assert after['color'] == page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--chart-text').trim()")


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


def test_widget_chart_repaints_in_place_and_releases_its_theme_listener(subscription_ui):
    """A declarative Widgets chart through the real page: mount, data update,
    OS appearance switch while visible, dispose on leave, remount after a switch."""
    ui = subscription_ui
    page = ui['page']
    first, refreshed = [3, 5, 4], [6, 2, 7]
    series = {'data': first}

    def extension(route):
        if route.request.method == 'POST':
            series['data'] = refreshed
        route.fulfill(content_type='application/json', body=json.dumps({'chart': {
            'labels': ['Mon', 'Tue', 'Wed'], 'datasets': [{'label': 'Requests', 'data': series['data']}]}}))

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
    page.locator('[data-nav-page="widgets"]').click()
    page.wait_for_selector('#page-widgets.active')
    dark = chart_when(f'c.data.join() === "{",".join(map(str, first))}" && c.settled')
    assert_themed(dark)
    assert page.evaluate('widgetThemeListeners()') == 1
    page.evaluate("window.widgetChart = globalThis.Chart.getChart("
                  "document.querySelector('#page-widgets canvas[data-widget-chart-key]'))")
    # Update: a widget action feeds new data into the SAME instance.
    page.locator('#page-widgets [data-widget-action]').click()
    updated = chart_when(f'c.data.join() === "{",".join(map(str, refreshed))}" && c.settled')
    assert updated['probe'], 'a data update must reuse the mounted chart'
    assert_themed(updated)
    capture(page, 'theme-dark-widget-chart')
    # The OS turns light while Widgets is on screen: repaint, not rebuild.
    page.emulate_media(color_scheme='light')
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    light = chart_when('c.color === c.text && c.settled')
    assert light['probe'], 'a theme switch must not replace the mounted chart'
    assert light['data'] == refreshed, 'a theme switch must keep the plotted data'
    assert_themed(light)
    assert (light['text'], light['line']) != (dark['text'], dark['line'])
    assert page.evaluate('widgetThemeListeners()') == 1
    capture(page, 'theme-light-widget-chart')

    # Dispose: leaving Widgets destroys the chart and releases its subscription,
    # so a later switch reaches nothing that belonged to the old mount.
    control = open_appearance(page)
    assert page.evaluate('widgetThemeListeners()') == 0
    assert page.evaluate('window.widgetChart.canvas === null'), 'leaving must destroy the chart'
    control.locator('[data-theme-choice="dark"]').click()
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    assert page.evaluate('widgetThemeListeners()') == 0
    # Remount after the switch: a fresh instance, already in the new palette.
    page.locator('[data-nav-page="widgets"]').click()
    remounted = chart_when('!c.probe && c.settled')
    assert remounted['data'] == refreshed
    assert (remounted['text'], remounted['line']) == (dark['text'], dark['line'])
    assert_themed(remounted)
    assert page.evaluate('widgetThemeListeners()') == 1
    assert not [path for path, _ in ui['posts'] if path == '/api/settings']


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
