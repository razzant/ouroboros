"""Real host-rendered widget scrollports, over the builtin shell without live account data."""
from __future__ import annotations

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import go_back, open_settings

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

SKILL = "build_diagnostics_fixture"
KEY = f"{SKILL}:build"
ROWS = [{"stage": f"Case {i + 1}", "elapsed": i + 12, "status": "Passed"} for i in range(35)]
LABELS = [f"Reference timing for configuration {i + 1}" for i in range(8)]
COMMAND = "python -m pytest " + "tests/system_e2e/test_ordinary_widget_placement.py " * 8
READONLY_LONG = "\n".join(
    f"Status line {i + 1} with enough words to wrap and wait past the edge." for i in range(40)
)


@pytest.fixture(params=["chromium", "webkit"])
def widget_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    return request.getfixturevalue("subscription_ui")


def _mask(locator):
    return locator.evaluate("el => getComputedStyle(el).maskImage")


def test_widget_scrollers_follow_scroll_repaint_stacking_and_board_height(widget_ui, tmp_path):
    page = widget_ui["page"]
    page.set_default_timeout(10_000)
    engine = page.context.browser.browser_type.name
    data = {"command": COMMAND, "rows": ROWS}
    page.route("**/api/ui/preferences", lambda route: route.fulfill(json={"widget_size": {KEY: {"w": 0, "h": 240}}}))
    page.route(f"**/api/extensions/{SKILL}/diagnostics", lambda route: route.fulfill(json=data))
    page.route("**/api/widgets", lambda route: route.fulfill(json={"ui_tabs": [
        {"key": KEY, "skill": SKILL, "tab_id": "build", "title": "Build diagnostics", "render": {
            "kind": "declarative", "components": [
                {"type": "action", "id": "read", "label": "Read diagnostics", "route": "diagnostics", "method": "GET"},
                {"type": "code", "id": "command", "label": "Re-run command", "path": "command"},
                {"type": "code", "id": "fit", "label": "Fitted", "text": "ok"},
                {"type": "table", "label": "Build stages", "path": "rows",
                 "columns": [{"label": label, "path": "stage"} for label in LABELS]},
                {"type": "json", "label": "Diagnostic data", "path": "rows"},
                {"type": "chart", "label": "Build timings", "labels": ["Latest build"],
                 "datasets": [{"label": label, "data": [20 + i]} for i, label in enumerate(LABELS)]},
            ],
        }},
    ]}))
    page.goto(widget_ui["url"] + "/#widgets")
    card = page.locator(f'[data-widget-key="{KEY}"]')
    card.get_by_role("button", name="Read diagnostics").click()
    card.locator(".widgets-card-body[data-widget-bounded][data-scroll-below]").wait_for()
    body = card.locator(".widgets-card-body")
    # The board keeps its crisp edges (DESIGN "Widgets board"): the card's own body never wears the shared mask,
    # however far the owner's height cuts it. The binding alone leaves the name the keyboard needs while it scrolls
    # (the bounded body is the one scroller the host gives a host-drawn card); the Show all line is the heights test's.
    assert _mask(body) == "none"
    assert body.get_attribute("tabindex") == "0" and body.get_attribute("aria-label") == "Build diagnostics"
    assert "scroll-fade-y" not in page.locator(".widgets-scroll").get_attribute("class")
    evidence = diagnostic_directory(tmp_path / "frames", "widget-scroll", publish=True)
    card.locator(".widget-code pre[data-scroll-end]").wait_for()
    page.screenshot(path=str(evidence / "bounded.png"), animations="disabled")

    tray = card.locator(".widget-code-tray").first
    code = tray.locator("pre")
    code.evaluate("el => window.__widgetCode = el")
    page.wait_for_function(
        "el => el.hasAttribute('data-scroll-end') && !el.hasAttribute('data-scroll-start')",
        arg=code.element_handle(),
    )
    paint = tray.evaluate(
        "el => ({fill: getComputedStyle(el).backgroundColor, radius: getComputedStyle(el).borderBottomLeftRadius})"
    )
    assert paint["fill"] == "rgba(0, 0, 0, 0.24)" and paint["radius"] != "0px", paint
    assert code.evaluate("el => getComputedStyle(el).backgroundColor") == "rgba(0, 0, 0, 0)"
    assert code.get_attribute("tabindex") == "0" and "linear-gradient" in _mask(code)
    heights = tray.evaluate(
        "el => ({tray: el.getBoundingClientRect().height, pre: el.firstElementChild.getBoundingClientRect().height})"
    )
    assert heights["tray"] == pytest.approx(heights["pre"], abs=0.5), heights
    code.evaluate("el => { el.scrollLeft = el.scrollWidth; }")
    page.wait_for_function(
        "el => el.hasAttribute('data-scroll-start') && !el.hasAttribute('data-scroll-end')",
        arg=code.element_handle(),
    )
    assert "linear-gradient" in _mask(code)
    assert tray.evaluate("el => getComputedStyle(el).backgroundColor") == "rgba(0, 0, 0, 0.24)"
    fitted = card.locator(".widget-code-tray pre").nth(1)
    page.wait_for_function(
        "el => !el.hasAttribute('data-scroll-end') && !el.hasAttribute('data-scroll-start')",
        arg=fitted.element_handle(),
    )
    assert _mask(fitted) == "none"

    card.locator(".widget-json summary").click()
    value = card.locator(".widget-json pre[data-scroll-below]")
    value.wait_for()
    assert "linear-gradient" in _mask(value)
    assert value.evaluate("el => getComputedStyle(el).backgroundColor") == "rgba(0, 0, 0, 0)"
    page.screenshot(path=str(evidence / f"code-tray-{engine}.png"), animations="disabled")

    data["command"] = "python -m pytest"
    card.get_by_role("button", name="Read diagnostics").click()
    page.wait_for_function("""() => {
        const code = document.querySelector('.widget-code pre');
        return code?.textContent === 'python -m pytest' && !code.hasAttribute('tabindex') && !code.hasAttribute('data-scroll-start');
    }""")
    assert code.evaluate("el => el === window.__widgetCode") and _mask(code) == "none"

    chart = card.locator(".widget-chart-data")
    chart.locator("summary").click()
    table = card.locator(".widget-table-wrap").first
    card.locator(":scope > .widgets-card-body > .widget-table-wrap[data-scroll-end]").wait_for()
    chart.locator(".widget-table-wrap[data-scroll-end]").wait_for()
    assert "linear-gradient" in _mask(table)
    table.locator("th").first.scroll_into_view_if_needed()
    page.screenshot(path=str(evidence / "wide.png"), animations="disabled")

    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_function("""() => [...document.querySelectorAll('.widget-table-wrap')].every(node =>
        getComputedStyle(node).overflowX === 'visible' && !node.hasAttribute('data-scroll-end') && !node.hasAttribute('tabindex'))""")
    assert _mask(table) == "none"
    table.scroll_into_view_if_needed()
    page.screenshot(path=str(evidence / "stacked.png"), animations="disabled")


def test_extension_settings_json_drops_its_scroll_binding_when_the_view_closes(widget_ui, tmp_path):
    page = widget_ui["page"]
    page.set_default_timeout(10_000)
    engine = page.context.browser.browser_type.name
    summary_text = {"value": READONLY_LONG}
    page.route(f"**/api/extensions/{SKILL}/status", lambda route: route.fulfill(
        json={"summary": summary_text["value"]},
    ))
    page.route("**/api/extensions", lambda route: route.fulfill(json={
        "skills": [{"name": SKILL, "enabled": True, "loaded": True, "type": "extension"}],
        "live": {"settings_sections": [{"skill": SKILL, "section_id": "report", "title": "Diagnostics",
                   "render": {"components": [
                       {"type": "form", "id": "status", "route": "status", "fields": [
                           {"name": "summary", "label": "Summary", "type": "textarea", "disabled": True},
                       ]},
                       {"type": "json", "label": "Diagnostic data", "value": ROWS},
                   ]}}]},
    }))
    page.goto(widget_ui["url"] + "/#settings")
    level = open_settings(page, f"skill/{SKILL}")
    chrome = level.locator(".ui-control-readonly")
    readonly = chrome.locator(".ui-readonly-body")
    readonly.wait_for()
    page.wait_for_function("el => el.hasAttribute('data-scroll-below')", arg=readonly.element_handle())
    paint = chrome.evaluate("""el => { const s = getComputedStyle(el);
        return {mask: s.maskImage, border: s.borderBottomWidth, fill: s.backgroundColor, radius: s.borderBottomLeftRadius}; }""")
    assert paint["mask"] == "none" and paint["border"] == "1px", paint
    assert paint["fill"] != "rgba(0, 0, 0, 0)" and paint["radius"] != "0px", paint
    assert "linear-gradient" in _mask(readonly)
    assert readonly.get_attribute("tabindex") == "0" and readonly.get_attribute("aria-label") == "Summary"
    assert readonly.get_attribute("role") is None, "the authored stop keeps the output's implicit role"
    sizes = readonly.evaluate("""el => ({over: el.scrollHeight - el.clientHeight,
        box: el.closest('.ui-control-readonly').getBoundingClientRect().height,
        em: parseFloat(getComputedStyle(el).fontSize)})""")
    assert sizes["over"] > 1 and sizes["box"] <= sizes["em"] * 18 + 1, sizes

    # Keyboard PageDown must scroll from the top (scrollTop == 0) before any programmatic jump.
    assert readonly.evaluate("el => el.scrollTop") == 0
    readonly.focus()
    page.keyboard.press("PageDown")
    page.wait_for_function("el => el.scrollTop > 0", arg=readonly.element_handle())
    assert "linear-gradient" in _mask(readonly), "the focused scrollport keeps its waiting-edge fade"
    assert chrome.evaluate("el => getComputedStyle(el).boxShadow") != "none", "keyboard focus wears the field's ring"

    readonly.evaluate("el => el.scrollTop = el.scrollHeight")
    page.wait_for_function("el => el.hasAttribute('data-scroll-above')", arg=readonly.element_handle())
    assert chrome.evaluate("el => getComputedStyle(el).maskImage") == "none"
    assert chrome.evaluate("el => getComputedStyle(el).borderBottomWidth") == "1px"
    frames = diagnostic_directory(tmp_path / "frames", "widget-scroll", publish=True)
    page.screenshot(path=str(frames / f"readonly-overflow-{engine}.png"))

    level.locator(".widget-json summary").click()
    value = level.locator(".widget-json pre[data-scroll-below]")
    value.wait_for()
    assert value.get_attribute("aria-label") == "Diagnostic data" and "linear-gradient" in _mask(value)
    readonly.evaluate("el => window.__formerReadonly = el")
    value.evaluate("el => window.__formerJson = el")
    summary_text["value"] = "Saved"
    go_back(page)
    page.wait_for_function("""() => !window.__formerJson.isConnected
        && !window.__formerJson.hasAttribute('data-scroll-below') && !window.__formerJson.hasAttribute('tabindex')
        && !window.__formerReadonly.isConnected && !window.__formerReadonly.hasAttribute('data-scroll-below')
        && !window.__formerReadonly.hasAttribute('data-scroll-above')
        && window.__formerReadonly.getAttribute('tabindex') === '0'""")

    # A fitted value: no edge facts and no mask, still the authored keyboard stop and the whole chrome.
    level = open_settings(page, f"skill/{SKILL}")
    fitted_chrome = level.locator(".ui-control-readonly")
    fitted = fitted_chrome.locator(".ui-readonly-body")
    fitted.wait_for()
    page.wait_for_function(
        "el => el.textContent === 'Saved' && !el.hasAttribute('data-scroll-below') && !el.hasAttribute('data-scroll-above')",
        arg=fitted.element_handle(),
    )
    assert _mask(fitted) == "none" and fitted.get_attribute("tabindex") == "0"
    assert fitted_chrome.evaluate("el => getComputedStyle(el).maskImage") == "none"
    assert fitted_chrome.evaluate("el => getComputedStyle(el).borderBottomWidth") == "1px"
