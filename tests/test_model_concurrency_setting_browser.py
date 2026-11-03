"""The model-concurrency cap is a Settings field: seen, saved, read back and applied by the real owner.

The REAL SPA against the REAL isolated server (``direct_server_with_data``: its own copy of this source, data
root, ports and mock provider): the field is edited in Settings › Advanced › Runtime Limits, saved through the
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

pytest_plugins = ("tests.test_ui_smoke_playwright",)

KEY = "OUROBOROS_MODEL_MAX_CONCURRENCY"
FIELD = "#s-model-max-concurrency"
IMMEDIATE = "Settings saved. Changes took effect immediately"
SECTION = "//section[@data-settings-panel='advanced']//div[contains(@class,'form-section')][.//h3[normalize-space()='Runtime Limits']]"
DESKTOP, PHONE = (1232, 880), (390, 844)

_MEASURE = """([fieldSel, sectionXPath]) => {
    const input = document.querySelector(fieldSel);
    const field = input.closest('.form-field');
    const section = document.evaluate(sectionXPath, document, null, 9, null).singleNodeValue;
    const help = document.getElementById('s-model-max-concurrency-help');
    const label = field.querySelector('label');
    const neighbourHelp = document.getElementById('s-evo-budget-help');
    const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom].map(Math.round); };
    const style = (el) => getComputedStyle(el);
    return {
        pageOverflowX: document.documentElement.scrollWidth - innerWidth,
        fieldOverflowX: field.scrollWidth - field.clientWidth,
        sectionOverflowX: section.scrollWidth - section.clientWidth,
        helpOverflowX: help.scrollWidth - help.clientWidth,
        labelOverflowX: label.scrollWidth - label.clientWidth,
        inputInsideSection: box(input)[0] >= box(section)[0] && box(input)[2] <= box(section)[2],
        helpInsideSection: box(help)[0] >= box(section)[0] && box(help)[2] <= box(section)[2],
        helpColor: style(help).color, helpSize: style(help).fontSize, neighbourHelpColor: style(neighbourHelp).color,
        labelSize: style(label).fontSize, labelWeight: style(label).fontWeight,
        labelLines: Math.round(label.getBoundingClientRect().height / parseFloat(style(label).lineHeight)),
        helpLines: Math.round(help.getBoundingClientRect().height / parseFloat(style(help).lineHeight)),
        inputHeight: Math.round(input.getBoundingClientRect().height),
        toolTimeoutHeight: Math.round(document.getElementById('s-tool-timeout').getBoundingClientRect().height),
        inputMin: input.min, inputType: input.type, inputValue: input.value, inputValid: input.validity.valid,
        described: input.getAttribute('aria-describedby'),
        helpText: help.textContent.replace(/\\s+/g, ' ').trim(), labelText: label.textContent.trim(),
        field: box(field), section: box(section), input: box(input),
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

        if size[0] < 720:
            page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
        page.click('[data-nav-page="settings"]')
        expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
        expect(page.locator("#settings-status")).to_have_text("Settings refreshed", timeout=30_000)
        page.click('[data-settings-tab="advanced"]')
        expect(page.locator(FIELD)).to_be_visible()

    def reload(self, page, size):
        page.reload(wait_until="domcontentloaded")
        page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
        self.load_settings(page, size)

    def save(self, page, expected=IMMEDIATE):
        from playwright.sync_api import expect

        page.click("#btn-save-settings")
        expect(page.locator("#settings-status")).to_contain_text(expected, timeout=20_000)

    def shot(self, page, name):
        """The window with the field in view, and the field's own box with its label and help."""
        field = page.locator(FIELD).locator("xpath=ancestor::div[contains(@class,'form-field')][1]")
        field.evaluate("el => el.scrollIntoView({block: 'center'})")
        page.screenshot(path=str(self.evidence / f"concurrency-{name}-{self.engine}.png"))
        page.locator(f"xpath={SECTION}").screenshot(path=str(self.evidence / f"concurrency-{name}-section-{self.engine}.png"))
        field.screenshot(path=str(self.evidence / f"concurrency-{name}-field-{self.engine}.png"))

    def measure(self, page):
        return page.evaluate(_MEASURE, [FIELD, SECTION])

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
        repo = f"owner/unrelated-save-{self.engine}"  # a change in each pass: an unchanged save says "No changes detected"
        page.fill("#s-gh-repo", repo)
        self.save(page)
        assert self.stored()[KEY] == 0 and self.stored()["GITHUB_REPO"] == repo
        assert self.runtime_cap() == 0
        self.reload(page, DESKTOP)
        assert page.input_value(FIELD) == "0"
        # A non-integer is not sent: native validation names the field and the document keeps its 0.
        page.fill(FIELD, "2.5")
        page.click("#btn-save-settings")
        expect(page.locator("#settings-status")).to_contain_text("Concurrent Requests", timeout=20_000)
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
        assert page.locator("#s-model-slot-wait").inner_text() == "90"
        wide.close()
        plain = self.open_settings("light", DESKTOP)
        assert plain.locator("#s-model-slot-wait").inner_text() == "180", "an install that never set it shows the default"
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
            self.shot(page, f"{theme}-390-saved-7")
            facts[f"{theme}-390"] = self.measure(page)
            page.context.close()

        # --- an install that ALREADY holds a cap of 0 or less: shown as saved, never blocking, never rewritten ---
        self.seed_saved_cap(-1)
        page = self.open_settings("light", DESKTOP)
        assert page.input_value(FIELD) == "-1", "a saved cap of 0 or less loads exactly as saved"
        assert page.eval_on_selector(FIELD, "el => el.validity.valid"), "and passes the field's own validity"
        self.shot(page, "light-1232-negative-loaded")
        facts["light-1232-negative"] = self.measure(page)
        repo = f"owner/negative-unrelated-{self.engine}"
        page.fill("#s-gh-repo", repo)
        self.save(page)  # the ordinary immediate-effect acknowledgement: the save gate did not block it
        assert self.stored()[KEY] == -1 and isinstance(self.stored()[KEY], int), self.stored().get(KEY)
        assert self.stored()["GITHUB_REPO"] == repo
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


@pytest.mark.serial
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
            # The neighbouring fields' own help ink and the one meta size; the same control height as its sibling.
            assert facts["helpColor"] == facts["neighbourHelpColor"] and facts["helpSize"] == "12px", where
            assert facts["labelSize"] == "12px" and facts["inputHeight"] == facts["toolTimeoutHeight"], where
            assert facts["inputType"] == "number" and facts["inputMin"] == "", where  # no minimum is declared
            assert facts["inputValid"], where  # whatever is saved, a negative one included, passes the save gate
            assert facts["described"].startswith("s-model-max-concurrency-help"), where
            assert facts["labelLines"] == 1, where
        desktop = windows["light-1232"]
        assert desktop["labelText"] == "Concurrent Requests per Model"
        words = desktop["helpText"]
        assert "counted per model, key and server" in words and "0 or less = no limit" in words
        assert "waits up to 180 s, then is refused and not sent" in words
        assert "Claudexor routes are not limited here" in words
