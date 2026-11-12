"""The model-concurrency cap is a Settings field: seen, saved, read back and applied by the real owner.

The REAL SPA against the REAL isolated server (``direct_server_with_data``: its own copy of this source, data
root, ports and mock provider): the field is edited in Settings › Models › More, saved through the
real gateway, read back from ``settings.json`` and from a fresh page load, and the runtime's own reader
(``model_concurrency._cap``) is pointed at that very document. 0 (no limit), 3 (the factory value) and 7, and an
earlier saved 5, must all round-trip; an unrelated save must leave a saved 0 alone. An install whose document
already holds a cap of 0 OR LESS (a hand-saved -1: the runtime has always read that as no queue) is seeded with the
server stopped: the field loads it as saved, the Settings save gate does not block, an unrelated save and a reload
keep it exactly, and the runtime reader still reads it as saved. Light and dark at the
owner's desktop window and at a phone width; overflow is measured, screenshots land in
``OUROBOROS_UI_EVIDENCE_DIR`` when it is set. Both engines run one after the other against ONE server (a server
boot is the costly, load-sensitive part). Every value is a synthetic fixture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.settings_browser_helpers import open_settings, commit_row, setting_row

pytest_plugins = ("tests.test_ui_smoke_playwright",)

KEY = "OUROBOROS_MODEL_MAX_CONCURRENCY"
FIELD = '[data-setting-row="OUROBOROS_MODEL_MAX_CONCURRENCY"] [data-setting-control]'
DESKTOP, PHONE = (1232, 880), (320, 640)

_MEASURE = r"""fieldSel => {
    const input = document.querySelector(fieldSel);
    const field = input.closest('[data-setting-row]');
    const section = field.closest('.settings-card');
    const help = field.querySelector('[data-setting-meta]');
    const label = field.querySelector('.ui-row-title');
    const box = el => el.getBoundingClientRect();
    return {
        pageOverflowX: document.documentElement.scrollWidth - innerWidth,
        fieldOverflowX: field.scrollWidth - field.clientWidth,
        sectionOverflowX: section.scrollWidth - section.clientWidth,
        helpOverflowX: help.scrollWidth - help.clientWidth,
        labelOverflowX: label.scrollWidth - label.clientWidth,
        inputInsideSection: box(input).left >= box(section).left && box(input).right <= box(section).right,
        helpInsideSection: box(help).left >= box(section).left && box(help).right <= box(section).right,
        inputMin: input.min, inputType: input.type, inputValue: input.value, inputValid: input.validity.valid,
        described: input.getAttribute('aria-describedby'),
        helpText: field.textContent.replace(/\s+/g, ' ').trim(), labelText: label.textContent.trim(),
    };
}"""


class _Run:
    """One engine's pass over the real server: its pages, saves and reads."""

    def __init__(self, browser, engine, server, evidence, runtime_cap, runtime_enabled):
        self.browser, self.engine, self.server = browser, engine, server
        self.url, self.document = server["url"], server["data_dir"] / "settings.json"
        self.evidence, self.runtime_cap, self.runtime_enabled = evidence, runtime_cap, runtime_enabled

    def stored(self):
        return json.loads(self.document.read_text(encoding="utf-8"))

    def seed_saved_cap(self, cap):
        """An existing install whose document already holds ``cap``: written with the server stopped (as the other
        host-UI cases seed their data root), then booted from, as after a hand edit."""
        self.server["stop_server"]()
        document = self.stored()
        document[KEY] = cap
        self.document.write_text(json.dumps(document), encoding="utf-8")
        self.server["start_server"]()
        assert self.stored()[KEY] == cap and isinstance(self.stored()[KEY], int), "a boot keeps the saved cap as it is"

    def open_settings(self, theme, size):
        page = self.browser.new_context(viewport={"width": size[0], "height": size[1]}, color_scheme=theme).new_page()
        page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
        return self.open_settings_in(page, size)

    def open_settings_in(self, page, size):
        from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
        page.goto(self.url, wait_until="domcontentloaded")
        page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
        self.load_settings(page, size)
        return page

    def load_settings(self, page, size):
        from playwright.sync_api import expect

        open_settings(page, 'models', KEY, timeout=30_000)
        expect(page.locator(FIELD)).to_be_visible()
        setting_row(page, KEY).locator('.settings-row-more > summary').click()

    def reload(self, page, size):
        page.reload(wait_until="domcontentloaded")
        page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
        self.load_settings(page, size)

    def save(self, page):
        result = commit_row(page, KEY, page.input_value(FIELD), timeout=20_000)
        assert result.startswith('Saved'), result

    def unrelated_save(self, page, value):
        open_settings(page, 'tools', 'OUROBOROS_TOOL_TIMEOUT_SEC')
        assert commit_row(page, 'OUROBOROS_TOOL_TIMEOUT_SEC', value).startswith('Saved')
        self.load_settings(page, DESKTOP)

    def shot(self, page, name):
        field = setting_row(page, KEY)
        field.scroll_into_view_if_needed()
        page.screenshot(path=str(self.evidence / f"concurrency-{name}-{self.engine}.png"))
        field.screenshot(path=str(self.evidence / f"concurrency-{name}-field-{self.engine}.png"))

    def measure(self, page):
        return page.evaluate(_MEASURE, FIELD)

    def exercise(self):
        from playwright.sync_api import expect

        # --- the owner's desktop window, light: the factory value, a saved 5, then 0 / 7 / 3 ---------------
        left_by_earlier_pass = self.stored().get(KEY)  # absent on an untouched install: the field shows the factory 3
        page = self.open_settings("light", DESKTOP)
        assert page.input_value(FIELD) == str(3 if left_by_earlier_pass is None else left_by_earlier_pass)
        self.shot(page, "light-1232-factory" if left_by_earlier_pass is None else "light-1232-opening")
        for saved in (5, 0, 7, 3):
            page.fill(FIELD, str(saved))
            self.save(page)
            assert self.stored()[KEY] == saved and isinstance(self.stored()[KEY], int), self.stored().get(KEY)
            assert self.runtime_cap() == saved, "the runtime's reader sees the saved cap at once"
            assert page.request.get(self.url + "/api/settings").json()[KEY] == saved
            self.reload(page, DESKTOP)
            assert page.input_value(FIELD) == str(saved), "a fresh load shows the actual saved value"
            self.shot(page, f"light-1232-saved-{saved}")
        # An unrelated save leaves a saved 0 (no limit) alone: it is a choice, never read back as "unset".
        page.fill(FIELD, "0")
        self.save(page)
        timeout = 601 if self.engine == 'chromium' else 602
        self.unrelated_save(page, timeout)
        assert self.stored()[KEY] == 0 and self.stored()['OUROBOROS_TOOL_TIMEOUT_SEC'] == timeout
        assert self.runtime_cap() == 0
        self.reload(page, DESKTOP)
        assert page.input_value(FIELD) == "0"
        # A non-integer is not sent: the row refuses it and the document keeps its 0.
        page.fill(FIELD, "2.5")
        page.locator(FIELD).blur()
        expect(setting_row(page, KEY).locator('[data-setting-result]')).to_contain_text('Not saved', timeout=20_000)
        assert self.stored()[KEY] == 0
        page.fill(FIELD, "7")
        self.save(page)
        self.shot(page, "light-1232-saved-7-final")
        facts = {"light-1232": self.measure(page)}
        page.context.close()

        # --- the wait the help names is this install's own value, never a copy of the default ---------------
        wide = self.browser.new_context(viewport={"width": DESKTOP[0], "height": DESKTOP[1]})
        page = wide.new_page()

        def serve_other_wait(route):
            if route.request.method != "GET":
                return route.fallback()
            response = route.fetch()  # the real document, with one number changed for this page load
            route.fulfill(response=response, json={**response.json(), "OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC": 90})

        page.route("**/api/settings", serve_other_wait)
        self.open_settings_in(page, DESKTOP)
        assert "waits up to 90 s" in setting_row(page, KEY).inner_text()
        wide.close()
        plain = self.open_settings("light", DESKTOP)
        assert "waits up to 180 s" in setting_row(plain, KEY).inner_text(), "an install that never set it shows the default"
        plain.context.close()

        # --- the same window, dark: the saved value, its words and fit ---------------------------------------
        page = self.open_settings("dark", DESKTOP)
        assert page.input_value(FIELD) == "7"
        self.shot(page, "dark-1232-saved-7")
        facts["dark-1232"] = self.measure(page)
        page.context.close()

        # --- a phone width in both themes: the existing grid stacks one column ------------------------------
        for theme in ("light", "dark"):
            page = self.open_settings(theme, PHONE)
            assert page.input_value(FIELD) == "7"
            self.shot(page, f"{theme}-320-saved-7")
            facts[f"{theme}-320"] = self.measure(page)
            page.context.close()

        # --- an install that ALREADY holds a cap of 0 or less: shown as saved, never blocking, never rewritten ---
        self.seed_saved_cap(-1)
        page = self.open_settings("light", DESKTOP)
        assert page.input_value(FIELD) == "-1", "a saved cap of 0 or less loads exactly as saved"
        assert page.eval_on_selector(FIELD, "el => el.validity.valid"), "and passes the field's own validity"
        self.shot(page, "light-1232-negative-loaded")
        facts["light-1232-negative"] = self.measure(page)
        timeout = 603 if self.engine == 'chromium' else 604
        self.unrelated_save(page, timeout)
        assert self.stored()[KEY] == -1 and isinstance(self.stored()[KEY], int), self.stored().get(KEY)
        assert self.stored()['OUROBOROS_TOOL_TIMEOUT_SEC'] == timeout
        assert page.request.get(self.url + "/api/settings").json()[KEY] == -1
        assert self.runtime_cap() == -1 and self.runtime_enabled() is False, "the runtime reader still reads it as saved"
        self.reload(page, DESKTOP)
        assert page.input_value(FIELD) == "-1", "an unrelated save and a reload keep it exactly"
        self.shot(page, "light-1232-negative-after-unrelated-save")
        page.context.close()
        page = self.open_settings("dark", DESKTOP)
        assert page.input_value(FIELD) == "-1"
        self.shot(page, "dark-1232-negative-loaded")
        facts["dark-1232-negative"] = self.measure(page)
        page.context.close()
        (self.evidence / f"concurrency-measure-{self.engine}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
        return facts


@pytest.mark.ui_browser
def test_concurrency_field(direct_server_with_data, monkeypatch):
    # A short name on purpose: pytest's tmp_path carries it into the server's socket paths (macOS caps them at 104 bytes).
    from playwright.sync_api import sync_playwright

    from ouroboros import config, model_concurrency
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    server = direct_server_with_data
    document = server["data_dir"] / "settings.json"
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(server["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "SETTINGS_PATH", document)  # the runtime's own reader reads this very document
    started_under_three = task_settings_snapshot({KEY: 3}, {KEY: "3"})

    def runtime_cap() -> int:
        with task_settings_scope(started_under_three):  # a task admitted under 3 follows the saved cap
            return model_concurrency._cap()

    def runtime_enabled() -> bool:
        with task_settings_scope(started_under_three):
            return model_concurrency.enabled()

    expected = {name.strip() for name in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(",") if name.strip()}
    measured = {}
    with sync_playwright() as pw:
        for engine in ("chromium", "webkit"):
            try:
                browser = getattr(pw, engine).launch()
            except Exception as exc:  # a missing engine is skipped unless this lane expects it
                if "Executable doesn't exist" in str(exc) and engine not in expected and "all" not in expected:
                    continue
                raise
            try:
                measured[engine] = _Run(browser, engine, server, evidence, runtime_cap, runtime_enabled).exercise()
            finally:
                browser.close()
    if not measured:
        pytest.skip("no browser engine is installed")

    for engine, windows in measured.items():
        for name, facts in windows.items():
            where = (engine, name, facts)
            assert facts["pageOverflowX"] <= 0 and facts["fieldOverflowX"] <= 1 and facts["sectionOverflowX"] <= 1, where
            assert facts["helpOverflowX"] <= 1 and facts["labelOverflowX"] <= 1, where
            assert facts["inputInsideSection"] and facts["helpInsideSection"], where
            assert facts['inputType'] == 'text' and facts['inputMin'] == '', where
            assert facts['inputValid'], where
            assert 'setting-OUROBOROS_MODEL_MAX_CONCURRENCY-meta' in facts['described'], where
        desktop = windows["light-1232"]
        assert desktop["labelText"] == "Requests at once per model"
        words = desktop["helpText"]
        assert "counted per model, key and server" in words and "0 or less sets no limit" in words
        assert "waits up to 180 s, then is refused and not sent" in words
        assert "Routes your subscriptions serve are not limited here" in words
