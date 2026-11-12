"""Settings grid tracks yield to their column, not to the viewport.

Two Settings grids used to pin pixel track minimums (`repeat(4, minmax(130px, 1fr))`
on the Available-subagents route, `minmax(180px, ...) minmax(260px, ...)` on a custom
secret row) and were rescued only by `@media (max-width: 760px)`. The Settings column
is the window minus the sidebar, so between 761px and roughly 980px those tracks are
wider than the column holding them and the page's scroll body scrolls sideways. The
class is engine-independent, so one Chromium viewport proves it; the desktop check
below keeps the narrow fit from being bought with the wide layout.
"""

from __future__ import annotations

import json

import pytest
from tests.test_ui_smoke_agents_panel import _AGENTS_PANEL_ROSTER
from tests import test_subscription_setup_browser as setup_browser
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_entry, open_settings

subscription_ui = setup_browser.subscription_ui
role_ui = roles.role_ui

pytest_plugins = ("tests.test_ui_smoke_playwright",)

# Wide enough that the ≤760px collapse does not apply, narrow enough that the
# Settings column (window − sidebar − padding) is well under 500px.
_NARROW = {"width": 780, "height": 680}
_DESKTOP = {"width": 1280, "height": 800}

# An owner-defined top-level secret: uppercase, neither an OUROBOROS_ nor a known
# settings key, so the settings read lists it in `_meta.custom_secret_keys` and the
# Accounts page renders one row for it (`[data-setting-row]`, web/modules/settings_accounts.js).
_CUSTOM_SECRET_KEY = "SLACK_WEBHOOK_URL"

_MEASURE_JS = """
    (selector) => {
        const scroll = document.querySelector('#page-settings [data-settings-page]:not([hidden])');
        const box = scroll.getBoundingClientRect();
        const nodes = [...document.querySelectorAll(selector)];
        const past = nodes.map((node) => ({
            name: node.tagName.toLowerCase() + '.' + (node.className || ''),
            over: Math.round((node.getBoundingClientRect().right - box.right) * 100) / 100,
        }));
        return {
            overflow: scroll.scrollWidth - scroll.clientWidth,
            clientWidth: scroll.clientWidth,
            measured: past.length,
            worst: past.length ? past.reduce((a, b) => (b.over > a.over ? b : a)) : null,
        };
    }
"""

_SHAPE_JS = """
    (selector) => {
        const route = document.querySelector(selector);
        const children = [...route.children];
        return {
            template: getComputedStyle(route).gridTemplateColumns,
            width: Math.round(route.getBoundingClientRect().width),
            children: children.length,
            rows: new Set(children.map((n) => Math.round(n.getBoundingClientRect().top))).size,
        };
    }
"""


@pytest.mark.ui_browser
def test_custom_secret_grid_tracks_never_scroll_the_body_sideways(direct_server_with_data):
    """The replacement key field and the new-key form fit the Settings column at both widths."""
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    path = server['data_dir'] / 'settings.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved[_CUSTOM_SECRET_KEY] = 'seeded-custom-secret-value'
    path.write_text(json.dumps(saved), encoding='utf-8')
    server['restart_server']()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            for viewport in (_NARROW, _DESKTOP):
                page = browser.new_page(viewport=viewport)
                page.goto(server['url'], wait_until='domcontentloaded')
                level = open_settings(page, 'accounts')
                row = level.locator(f'[data-setting-row="{_CUSTOM_SECRET_KEY}"]')
                row.locator('[data-secret-open]').click()
                for selector in (f'[data-setting-row="{_CUSTOM_SECRET_KEY}"] [data-secret-field] > *',
                                 f'[data-setting-row="{_CUSTOM_SECRET_KEY}"] [data-secret-remove]'):
                    result = page.evaluate(_MEASURE_JS, selector)
                    assert result['measured'] and result['overflow'] <= 1 and result['worst']['over'] <= 1, result
                open_settings(page, 'accounts/key/custom')
                result = page.evaluate(_MEASURE_JS, '.accounts-key-form > *')
                assert result['measured'] == 2 and result['overflow'] <= 1 and result['worst']['over'] <= 1, result
                shape = page.evaluate(_SHAPE_JS, '.accounts-key-form')
                if viewport == _DESKTOP:
                    assert len(shape['template'].split()) == 2 and shape['rows'] == 1, shape
                page.close()
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_action_results_reflow_before_their_text_is_squeezed(role_ui, theme):
    """An agent editor’s real Save row: a container can be narrow in a wide desktop window.

    The synthetic translation stresses the existing rendered controls, not a copy
    of their markup. Containment alone passed the old, unreadable two-column row.
    At 360/280px the result gets the row's full width (2px rounding tolerance),
    rather than the old 95/15px sliver. At 640px both still fit on the same line.
    """
    from playwright.sync_api import expect

    ui, seen = role_ui, {}
    roles.configure_mixed(ui)
    page = ui['page']
    page.set_viewport_size({'width': 1100, 'height': 750})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    form = roles.open_agent(ui, 'native')
    form.locator('[data-subagent-field="recommended_use"]').fill('An edited description.')
    button = page.locator('[data-editor-save]')
    row = button.locator('..')
    expect(button).to_be_visible()
    def refuse(route):
        # A refusal as the server words one (`saved: false`); an answer without it is an unknown outcome.
        route.fulfill(status=503, json={'error': 'Roster temporarily unavailable', 'saved': False})
    page.route('**/api/owner/safety-model', refuse)
    button.click()
    status = row.locator('.ui-result')
    expect(status).to_contain_text('Roster temporarily unavailable')
    status.evaluate("e => e.textContent = 'Der Modellkatalog ist momentan nicht erreichbar. Die gespeicherte Auswahl bleibt erhalten; bitte versuchen Sie die Aktualisierung erneut.'")
    button.evaluate("e => e.textContent = 'Modellkatalog erneut aktualisieren'")
    row.evaluate("e => { window.actionReadingRow = e; window.actionReadingButton = e.firstElementChild; }")
    for width in (640, 360, 280):
        row.evaluate("(e, w) => { e.style.width = w + 'px'; e.style.maxWidth = '100%'; }", width)
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        row.scroll_into_view_if_needed()
        seen[str(width)] = row.evaluate("""e => {
            const box = e.getBoundingClientRect(), action = e.firstElementChild.getBoundingClientRect();
            const text = e.querySelector('.ui-result').getBoundingClientRect();
            return {width: box.width, text: text.width, textTop: text.top,
                actionBottom: action.bottom, actionRight: action.right - box.right,
                overflow: e.scrollWidth - e.clientWidth};
        }""")
        setup_browser.capture(page, f"action-results-{theme}-{width}")
    where = json.dumps(seen, indent=2)
    for width in (360, 280):
        result = seen[str(width)]
        assert result["text"] >= result["width"] - 2, f"the long result needs its own readable line:\n{where}"
        assert result["textTop"] >= result["actionBottom"] - 1, f"the result wraps under its action:\n{where}"
    assert seen["640"]["textTop"] < seen["640"]["actionBottom"], f"keep the result beside its action when it fits:\n{where}"
    assert all(v["overflow"] <= 1 and v["actionRight"] <= 1 for v in seen.values()), where
    # Reflow preserves the actual node, focus and action; no reconstructed lookalike.
    button.focus()
    expect(button).to_be_focused()
    assert page.evaluate("window.actionReadingButton === document.querySelector('[data-editor-save]')")
    page.unroute('**/api/owner/safety-model', refuse)
    before = len(roles.roster_writes(ui))
    button.press('Enter')
    expect(page.locator('[data-editor-result]')).to_contain_text('Saved')
    assert len(roles.roster_writes(ui)) == before + 1


@pytest.mark.ui_browser
def test_agent_editor_grid_tracks_fit_the_page_at_both_widths(direct_server_with_data):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    path = server['data_dir'] / 'settings.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved['OUROBOROS_SUBAGENTS'] = json.dumps(_AGENTS_PANEL_ROSTER)
    path.write_text(json.dumps(saved), encoding='utf-8')
    server['restart_server']()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            for viewport in (_NARROW, _DESKTOP):
                page = browser.new_page(viewport=viewport)
                page.goto(server['url'], wait_until='domcontentloaded')
                open_settings(page, 'agents')
                form = open_entry(page, _AGENTS_PANEL_ROSTER['items'][0]['subagent_id'])
                form.locator('[data-subagent-field="route"]').wait_for(state='visible')
                controls = page.evaluate(_MEASURE_JS, '.available-subagent-route > *')
                selects = page.evaluate(_MEASURE_JS, '.available-subagent-route select')
                shape = page.evaluate(_SHAPE_JS, '.available-subagent-route')
                assert controls['measured'] == 5 and selects['measured'] >= 3
                assert controls['overflow'] <= 1 and controls['worst']['over'] <= 1, controls
                assert selects['worst']['over'] <= 1, selects
                if viewport == _DESKTOP:
                    assert len(shape['template'].split()) == 3 and shape['rows'] == 2, shape
                page.close()
        finally:
            browser.close()
