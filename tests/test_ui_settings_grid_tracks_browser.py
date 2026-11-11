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
import os
import pathlib

import pytest
from tests.test_ui_smoke_agents_panel import _AGENTS_PANEL_ROSTER
from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_entry, open_settings

subscription_ui = setup_browser.subscription_ui

pytest_plugins = ("tests.test_ui_smoke_playwright",)

# Wide enough that the ≤760px collapse does not apply, narrow enough that the
# Settings column (window − sidebar − padding) is well under 500px.
_NARROW = {"width": 780, "height": 680}
_DESKTOP = {"width": 1280, "height": 800}

# An owner-defined top-level secret: uppercase, neither an OUROBOROS_ nor a known
# settings key, so the settings read lists it in `_meta.custom_secret_keys` and the
# editor renders one `.settings-custom-secret-row` for it (ouroboros/gateway/settings.py).
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


def _open_settings(page, url: str, tab: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#page-chat", timeout=30_000)
    page.click('[data-nav-page="settings"]')
    page.wait_for_selector(".settings-shell", timeout=15_000)
    open_settings(page, "accounts" if tab == "secrets" else tab)


@pytest.mark.ui_browser
def test_custom_secret_grid_tracks_never_scroll_the_body_sideways(direct_server_with_data):
    """The custom-key grid retains its narrow fit and wide shape; its E3 page is still pending."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    data_dir = direct_server_with_data["data_dir"]
    settings_path = data_dir / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    saved["OUROBOROS_SUBAGENTS"] = json.dumps(_AGENTS_PANEL_ROSTER)
    saved[_CUSTOM_SECRET_KEY] = "seeded-custom-secret-value"
    settings_path.write_text(json.dumps(saved), encoding="utf-8")
    direct_server_with_data["restart_server"]()
    url = direct_server_with_data["url"]
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, object] = {}

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport=_NARROW)
                _open_settings(page, url, "secrets")
                page.wait_for_selector(".settings-custom-secret-row", timeout=20_000)
                seen["secrets@780"] = page.evaluate(_MEASURE_JS, ".settings-custom-secret-row > *")
                seen["secrets-remove@780"] = page.evaluate(
                    _MEASURE_JS, "[data-custom-secret-remove]")
                seen["secrets-row@780"] = page.evaluate(_SHAPE_JS, ".settings-custom-secret-row")
                page.screenshot(path=str(evidence_dir / "settings-secrets-780.png"))

                wide = browser.new_page(viewport=_DESKTOP)
                try:
                    _open_settings(wide, url, "secrets")
                    wide.wait_for_selector(".settings-custom-secret-row", timeout=20_000)
                    seen["secrets-row@1280"] = wide.evaluate(_SHAPE_JS, ".settings-custom-secret-row")
                finally:
                    wide.close()
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

    secrets, remove = seen["secrets@780"], seen["secrets-remove@780"]
    where = json.dumps(seen, indent=2, sort_keys=True)
    assert remove["measured"] == 1, f"the Remove button was not found: {where}"

    assert secrets["overflow"] <= 1, (
        f"Settings → Secrets scrolls sideways at 780px by {secrets['overflow']}px:\n{where}")
    assert remove["worst"]["over"] <= 1, (
        f"the custom secret row's Remove button escapes the scroll body:\n{where}")
    assert secrets["worst"]["over"] <= 1, (
        f"a custom secret field escapes the scroll body:\n{where}")



@pytest.mark.ui_browser
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_action_results_reflow_before_their_text_is_squeezed(subscription_ui, theme):
    """The real Models row: a container can be narrow in a wide desktop window.

    The synthetic translation stresses the existing rendered controls, not a copy
    of their markup. Containment alone passed the old, unreadable two-column row.
    At 360/280px the result gets the row's full width (2px rounding tolerance),
    rather than the old 95/15px sliver. At 640px both still fit on the same line.
    """
    from playwright.sync_api import expect

    ui, seen = subscription_ui, {}
    page = ui["page"]
    page.set_viewport_size({"width": 1100, "height": 750})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui["url"] + "/#settings")
    open_settings(page, 'models', 'models:catalogue')
    button = page.locator("#btn-refresh-model-catalog")
    row = button.locator("..")
    expect(button).to_be_visible()
    # Use the real refresh/failure transition before stressing its wording.
    ui["backend"]["catalog_status"] = 503
    ui["backend"]["catalog_response"] = {"error": "Catalog temporarily unavailable"}
    button.click()
    expect(button).to_be_enabled()
    # The shared action row (ui.css .ui-action-row): the trigger, then its result as the next sibling.
    status = row.locator(".ui-result")
    expect(status).not_to_be_empty()
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
    assert page.evaluate("window.actionReadingButton === document.querySelector('#btn-refresh-model-catalog')")
    ui["backend"].pop("catalog_status")
    ui["backend"].pop("catalog_response")
    before = sum('/api/model-catalog' in url for url in ui["reads"])
    button.press("Enter")
    expect(button).to_be_enabled()
    assert sum('/api/model-catalog' in url for url in ui["reads"]) > before


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
