"""Settings → Agents acceptance (docs/DESIGN.md "List editors"): real UI, real browser.

Sibling of ``test_ui_smoke_playwright.py`` (which carries the shared server fixture and
sits at its byte gate); marker-gated the same way, runs in the same CI job.
"""

from __future__ import annotations

import json

import pytest

from tests.settings_browser_helpers import open_entry, open_settings, settings_page

pytest_plugins = ("tests.test_ui_smoke_playwright",)


_AGENTS_PANEL_ROSTER = {
    "enabled": True,
    "items": [
        {"subagent_id": "claude_builder", "recommended_use": "Main workhorse for code and design.",
         "route": {"kind": "agent_session", "target_id": "claude=claude-opus-5"}, "effort": "medium"},
        {"subagent_id": "codex_reviewer", "recommended_use": "Deep code review of diffs.",
         "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol-high"}},
        {"subagent_id": "api_scout", "recommended_use": "Fast independent research.",
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-luna"}, "effort": "high"},
    ],
}

def _open_agents(page, url: str):
    """Settings › Agents on the real server: the list, once its heads are drawn."""
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#page-chat", timeout=30_000)
    page.click('[data-nav-page="settings"]')
    page.wait_for_selector(".settings-shell", timeout=15_000)
    level = open_settings(page, "agents", timeout=20_000)
    level.locator("[data-entry-id]").first.wait_for(state="visible", timeout=20_000)
    return level


def _confirm(page, words: str) -> None:
    """Answer Yes in the shared confirm dialog, which must say `words`."""
    dialog = page.locator(".confirm-dialog")
    dialog.wait_for(state="visible", timeout=10_000)
    assert words in dialog.inner_text(), dialog.inner_text()
    dialog.locator("[data-confirm-ok]").click()
    page.wait_for_function("() => !document.querySelector('.confirm-dialog')", timeout=10_000)


def _agents_add_saves_and_the_list_reveals_it(page) -> str:
    """The add-and-reveal contract, run on whichever engine the caller launched: Add opens an unsaved agent's
    page, which invites and shows no error; a Save attempt names what is missing on the page and sends nothing;
    the fix clears it; saved, the list shows the new agent in view and gives it the caret."""
    level = settings_page(page, "agents")
    before = level.locator("[data-entry-id]").count()
    level.locator("[data-subagent-add]").click()
    editor = settings_page(page, "agents/new")
    form = editor.locator("[data-subagent-row]")
    form.wait_for(state="visible", timeout=10_000)
    assert editor.locator(".settings-page-title").inner_text().strip() == "New agent"
    line = form.locator("[data-subagents-validation]")
    assert line.is_hidden() and form.get_attribute("data-invalid") is None
    assert "Choose how this subagent runs" in form.locator("[data-subagent-meta]").inner_text()

    form.locator("[data-editor-save]").click()
    page.wait_for_function("() => !document.querySelector('#page-settings [data-settings-page=\"agents/new\"] "
                           "[data-subagents-validation]').hidden", timeout=5_000)
    assert line.inner_text().startswith("New agent needs a model or agent-session route.")
    assert form.get_attribute("data-invalid") is not None
    assert form.locator('[data-subagent-meta][data-tone="error"]').inner_text().startswith("New agent needs")

    form.locator('[data-subagent-field="model"]').fill("openai/gpt-5.6-sol-high")
    page.wait_for_function("() => document.querySelector('#page-settings [data-settings-page=\"agents/new\"] "
                           "[data-subagents-validation]').hidden", timeout=5_000)
    assert form.get_attribute("data-invalid") is None
    form.locator("[data-editor-save]").click()
    # This roster marks no Reviewer: the server judges the changed roster and the page asks first.
    _confirm(page, "No agent marked Reviewer would be on")
    level.locator("[data-entry-id]").nth(before).wait_for(state="visible", timeout=15_000)
    assert level.locator("[data-entry-id]").count() == before + 1
    page.wait_for_function(
        """() => {
            const heads = [...document.querySelectorAll('#page-settings [data-settings-page="agents"] [data-entry-id]')];
            const last = heads.at(-1), box = last.closest('[data-settings-page]').getBoundingClientRect();
            const r = last.getBoundingClientRect();
            return r.top >= box.top - 1 && r.bottom <= box.bottom + 1 && last.contains(document.activeElement);
        }""",
        timeout=10_000,
    )
    return level.locator("[data-entry-id]").nth(before).get_attribute("data-entry-id")


@pytest.mark.ui_browser
def test_ui_smoke_agents_panel_list_editor(direct_server_with_data):
    """Settings › Agents on the real server: the list holds one head per agent; Add opens an unsaved agent's
    page that invites; only a Save attempt turns the empty route into a line on the page and a tint, and the fix
    clears both; the save asks about the empty review pool because the server refuses it, then the list reveals
    the new agent; a Reviewer switch on an agent's page joins the pool at once (docs/DESIGN.md "List editors")."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    saved["OUROBOROS_SUBAGENTS"] = json.dumps(_AGENTS_PANEL_ROSTER)
    settings_path.write_text(json.dumps(saved), encoding="utf-8")
    direct_server_with_data["restart_server"]()
    url = direct_server_with_data["url"]

    def stored():
        return json.loads(json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_SUBAGENTS"])["items"]

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                level = _open_agents(page, url)
                # A list of heads, not open forms: every agent is one row, and none holds a field.
                assert [row.get_attribute("data-entry-id") for row in level.locator("[data-entry-id]").all()] == [
                    row["subagent_id"] for row in _AGENTS_PANEL_ROSTER["items"]]
                assert level.locator("[data-entry-id] textarea, [data-entry-id] select").count() == 0
                assert level.locator("[data-review-pool-empty]").is_visible(), "no Reviewer is marked: the list says so"

                added = _agents_add_saves_and_the_list_reveals_it(page)
                assert [row["subagent_id"] for row in stored()][-1] == added
                assert stored()[-1]["route"] == {"kind": "api_model", "target_id": "openai/gpt-5.6-sol-high"}

                # A Reviewer switch on an agent's page joins the pool at once: the list's count and the empty-pool
                # line follow, and with a reviewer in the pool nothing is asked.
                form = open_entry(page, "codex_reviewer").locator("[data-subagent-row]")
                form.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
                page.wait_for_function(
                    """() => { const input = document.querySelector(
                            '#page-settings [data-settings-page="agents/codex_reviewer"] [data-subagent-field="review_eligible"]');
                        return input && input.checked && input.getAttribute('aria-busy') !== 'true'; }""", timeout=15_000)
                assert page.locator(".confirm-dialog").count() == 0
                assert [row.get("review_eligible") for row in stored()] == [None, True, None, None]
                level = open_settings(page, "agents")
                page.wait_for_function(
                    """() => document.querySelector('#page-settings [data-agents-filter="reviewers"]')?.textContent.trim() === 'Reviewers 1'
                        && document.querySelector('#page-settings [data-review-pool-empty]').hidden""", timeout=15_000)
            finally:
                browser.close()

            # The desktop shell is WebKit: the add-and-reveal contract must hold there too.
            try:
                webkit = pw.webkit.launch()
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    webkit = None
                else:
                    raise
            if webkit is not None:
                try:
                    page = webkit.new_page(viewport={"width": 1440, "height": 900})
                    _open_agents(page, url)
                    level = settings_page(page, "agents")
                    before = level.locator("[data-entry-id]").count()
                    level.locator("[data-subagent-add]").click()
                    form = settings_page(page, "agents/new").locator("[data-subagent-row]")
                    form.locator('[data-subagent-field="model"]').fill("openai/gpt-5.6-terra")
                    form.locator("[data-editor-save]").click()
                    level.locator("[data-entry-id]").nth(before).wait_for(state="visible", timeout=15_000)
                    page.wait_for_function(
                        """() => [...document.querySelectorAll('#page-settings [data-settings-page="agents"] [data-entry-id]')]
                            .at(-1).contains(document.activeElement)""", timeout=10_000)
                finally:
                    webkit.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


def _wizard_step_until(page, predicate_js: str, forward: bool, limit: int = 8) -> None:
    """Walk the wizard with Next/Back until `predicate_js` holds. Steps that hold Continue
    until they have a value (a provider key, the main/light models) get placeholders — the
    subject here is Models' Available subagents and the summary's Finish, not those steps."""
    placeholders = {
        "#openrouter-key": "sk-or-placeholder-not-real",
        "#main-model": "openai/gpt-5.6-luna",
        "#light-model": "openai/gpt-5.6-luna",
    }
    for _ in range(limit):
        if page.locator("#onboarding-available-subagents").count():
            # Models keeps the roster in a disclosure; open it as an owner would.
            disclosure = page.locator("details.wizard-collapse").filter(
                has=page.locator("#onboarding-available-subagents"))
            if disclosure.count() and not disclosure.evaluate("node => node.open"):
                disclosure.locator(":scope > summary").click()
            # The roster settles asynchronously (saved roster or generated draft);
            # judge it only once it shows rows or its own failure line.
            page.wait_for_function(
                "() => document.querySelectorAll('#onboarding-available-subagents .available-subagent-row').length"
                " || !document.querySelector('#onboarding-available-subagents [data-subagents-validation]').hidden",
                timeout=20_000)
        if page.evaluate(predicate_js):
            return
        if forward and page.evaluate("() => Boolean(document.getElementById('next-btn')?.disabled)"):
            for selector, value in placeholders.items():
                # Account discovery can hold Continue while optional API fields stay collapsed.
                field = page.locator(selector)
                if field.is_visible() and not field.input_value():
                    field.fill(value)
        button = "#next-btn" if forward else "#back-btn"
        try:
            # A step may hold its button while it settles (a probe, a preview).
            page.wait_for_function(
                "(id) => !document.querySelector(id)?.disabled", arg=button, timeout=15_000)
        except Exception as exc:  # noqa: BLE001 - the step's own state is the useful message
            step = page.evaluate(
                "() => ({title: document.querySelector('.step-title')?.textContent,"
                " error: document.querySelector('.wizard-error')?.textContent,"
                " inputs: [...document.querySelectorAll('input:not([type=hidden])')].map((i) => i.id)})")
            raise AssertionError(f"wizard button {button} stayed disabled on {step}") from exc
        page.click(button)
        page.wait_for_timeout(300)
    seen = page.evaluate(
        "() => ({title: document.querySelector('.step-title')?.textContent,"
        " agents: (document.querySelector('#onboarding-available-subagents')?.innerText || '').slice(0, 400)})")
    raise AssertionError(f"wizard never reached: {predicate_js}; last seen {seen}")


_WIZARD_ON_AGENTS_JS = "() => document.querySelectorAll('#onboarding-available-subagents .available-subagent-row').length > 0"
_WIZARD_ON_SUMMARY_JS = "() => (document.querySelector('.step-title')?.textContent || '').startsWith('Review before launch')"


@pytest.mark.ui_browser
def test_ui_smoke_agents_panel_wizard_finish_judges_the_roster(direct_server_with_data):
    """First-run wizard (docs/ARCHITECTURE.md §2): an unrouted entry added on the Models step
    does not block Continue; Finish on the summary reports it and, back on Models, the card
    is already tinted and self-naming; the fix reconciles line and tint together; a roster
    marking no Reviewer is named next, until the owner ticks Save without reviewers, and
    then Finish passes the wizard's own checks and enters saving (the save's provider
    round-trip is not this test's subject)."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
    from tests.test_subscription_setup_browser import capture

    # A saved roster on an OpenRouter-shaped install: the Models step's preview validates
    # the model setup the way a first run does, and the shared fixture's mock-LLM model is
    # not a confirmed main model — so this test mirrors an owner's machine instead.
    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"):
        saved.pop(key, None)
    saved["OPENROUTER_API_KEY"] = "sk-or-v1-smoke-placeholder-not-real"
    roster = json.loads(json.dumps(_AGENTS_PANEL_ROSTER))
    roster['items'][0]['review_eligible'] = True  # the first Finish tests the missing route, then the empty pool
    saved["OUROBOROS_SUBAGENTS"] = json.dumps(roster)
    settings_path.write_text(json.dumps(saved), encoding="utf-8")
    direct_server_with_data["restart_server"]()
    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.goto(f"{url}/onboarding", wait_until="domcontentloaded")
                page.wait_for_selector("#next-btn", timeout=30_000)
                _wizard_step_until(page, _WIZARD_ON_AGENTS_JS, forward=True)
                # The step may still be generating its draft; Add waits for it to settle.
                page.wait_for_function(
                    "() => !document.querySelector('#onboarding-available-subagents [data-subagent-add]').disabled",
                    timeout=20_000)
                before = page.locator("#onboarding-available-subagents .available-subagent-row").count()
                page.click("#onboarding-available-subagents [data-subagent-add]")
                page.wait_for_function(
                    "(n) => document.querySelectorAll('#onboarding-available-subagents .available-subagent-row')"
                    ".length === n + 1", arg=before, timeout=5_000)
                assert page.locator(
                    "#onboarding-available-subagents .available-subagent-row[data-invalid]").count() == 0

                _wizard_step_until(page, _WIZARD_ON_SUMMARY_JS, forward=True)
                page.click("#next-btn")
                page.wait_for_function(
                    "(n) => (document.querySelector('.wizard-error')?.textContent || '')"
                    ".startsWith('Subagent ' + n + ' needs')", arg=before + 1, timeout=5_000)

                _wizard_step_until(page, _WIZARD_ON_AGENTS_JS, forward=False)
                tinted = page.locator("#onboarding-available-subagents .available-subagent-row[data-invalid]")
                assert tinted.count() == 1
                assert tinted.locator('[data-subagent-meta][data-tone="error"]').inner_text().startswith(
                    f"Subagent {before + 1} needs")
                assert not page.evaluate(
                    "() => document.querySelector('#onboarding-available-subagents [data-subagents-validation]').hidden")
                tinted.scroll_into_view_if_needed()
                capture(page, "wizard-roster-invalid")

                # A distinct fixture model fixes the missing route without creating a twin of another row.
                tinted.locator('[data-subagent-field="model"]').fill("owner/roster-validation-fixture")
                page.wait_for_function(
                    "() => document.querySelector('#onboarding-available-subagents [data-subagents-validation]').hidden"
                    " && !document.querySelector('#onboarding-available-subagents .available-subagent-row[data-invalid]')",
                    timeout=5_000)
                capture(page, "wizard-roster-repaired")

                # With every Reviewer box cleared, Finish names the empty pool until the owner
                # says Save without reviewers.
                boxes = page.locator('#onboarding-available-subagents [data-subagent-field="review_eligible"]')
                for index in range(boxes.count()):
                    boxes.nth(index).set_checked(False)
                _wizard_step_until(page, _WIZARD_ON_SUMMARY_JS, forward=True)
                page.click("#next-btn")
                page.wait_for_function(
                    "() => (document.querySelector('.wizard-error')?.textContent || '').includes('No row is marked Reviewer')",
                    timeout=5_000)
                _wizard_step_until(page, _WIZARD_ON_AGENTS_JS, forward=False)
                page.check('#onboarding-available-subagents [data-review-pool-allow-empty]')

                _wizard_step_until(page, _WIZARD_ON_SUMMARY_JS, forward=True)
                page.click("#next-btn")
                # The last Finish passes the wizard's own checks and hands the draft to
                # the save (which probes providers — with a placeholder key that round-trip
                # is not this test's subject): saved, or saving with no wizard error.
                try:
                    page.wait_for_function(
                        "() => (document.querySelector('.step-title')?.textContent || '').startsWith('Setup saved')"
                        " || (document.getElementById('next-btn')?.disabled"
                        "     && !(document.querySelector('.wizard-error')?.textContent || '').trim())",
                        timeout=10_000)
                except Exception as exc:  # noqa: BLE001 - the wizard's own error is the useful message
                    seen = page.evaluate(
                        "() => ({title: document.querySelector('.step-title')?.textContent,"
                        " error: document.querySelector('.wizard-error')?.textContent})")
                    raise AssertionError(f"the last Finish was refused: {seen}") from exc
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


_LIST_EDITOR_ENTRIES = {
    "OUROBOROS_MODEL_FALLBACKS": "openai/gpt-5.6-luna, openai/gpt-5.6-sol, openai/gpt-5.6-terra",
    "MCP_ENABLED": False,
    "MCP_SERVERS": [{"id": sid, "name": sid, "transport": "streamable_http", "enabled": False,
                     "url": f"https://{sid}.invalid/mcp",
                     **({"headers": {"X-A": "synthetic-a", "X-B": "synthetic-b"}} if sid == "alpha" else {})}
                    for sid in ("alpha", "beta", "gamma")],
    "REMOVE_FOCUS_ONE": "synthetic-one", "REMOVE_FOCUS_TWO": "synthetic-two",
}


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_ui_smoke_agents_remove_asks_first_and_hands_the_caret_on(direct_server_with_data, engine):
    """A keyboard Remove on an agent's page asks first, removes at once and returns to the list with the caret on
    the agent now in that place (the next, else the previous) and, once the list is empty, on Add: never on the
    page (docs/DESIGN.md "List editors")."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import expect, sync_playwright
    from tests.test_subscription_setup_browser import capture

    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    roster = json.loads(json.dumps(_AGENTS_PANEL_ROSTER))
    roster["items"][0]["review_eligible"] = True  # a reviewer stays until the list is empty: the pool is not the subject
    saved["OUROBOROS_SUBAGENTS"] = json.dumps(roster)
    settings_path.write_text(json.dumps(saved), encoding="utf-8")
    direct_server_with_data["restart_server"]()

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            level = _open_agents(page, direct_server_with_data["url"])

            def remove(agent_id, name):
                editor = open_entry(page, agent_id)
                editor.locator("[data-editor-remove]").focus()
                page.keyboard.press("Enter")
                dialog = page.locator(".confirm-dialog")
                dialog.wait_for(state="visible", timeout=10_000)
                assert f"Remove {name}?" in dialog.inner_text()
                dialog.locator("[data-confirm-ok]").click()
                level.wait_for(state="visible", timeout=15_000)
                expect(level.locator(f'[data-entry-id="{agent_id}"]')).to_have_count(0, timeout=15_000)

            def focused_agent():
                return page.evaluate("() => document.activeElement?.closest('[data-entry-id]')?.dataset.entryId || ''")

            remove("codex_reviewer", "gpt-5.6-sol-high")
            page.wait_for_function("() => document.activeElement?.closest('[data-entry-id]')?.dataset.entryId === 'api_scout'")
            remove("api_scout", "gpt-5.6-luna/high")
            page.wait_for_function("() => document.activeElement?.closest('[data-entry-id]')?.dataset.entryId === 'claude_builder'")
            assert focused_agent() == "claude_builder"
            remove("claude_builder", "claude-opus-5/medium")
            expect(level.locator("[data-subagent-add]:focus")).to_have_count(1, timeout=15_000)
            assert json.loads(json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_SUBAGENTS"])["items"] == []
            capture(page, f"agents-remove-focus-{engine}")
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("surface", ["fallbacks", "tools", "keys"])
def test_ui_smoke_list_editor_remove_hands_the_caret_on(direct_server_with_data, engine, surface):
    """A keyboard Remove in the other Settings list editors (fallbacks, tool servers and their headers, custom keys)
    hands the caret to the entry now in that place (the next, else the previous) and, once the list is empty, to
    the group's Add: never to the page (docs/DESIGN.md "List editors").

    The agents' own Remove is the test above; each other entity is reached through its page."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import expect, sync_playwright
    from tests.test_subscription_setup_browser import capture

    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    saved.update(_LIST_EDITOR_ENTRIES)
    if surface == "keys":
        saved["REMOVE_FOCUS_Z_LAST"] = "last-synthetic-secret"
    settings_path.write_text(json.dumps(saved), encoding="utf-8")
    direct_server_with_data["restart_server"]()

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.add_init_script("""window.removalFocus = []; document.addEventListener('focusin', event => {
                window.removalFocus.push(event.target.outerHTML.slice(0, 400));
            }, true);""")
            def delayed_settings(route):
                import time

                response = route.fetch()
                time.sleep(0.15)  # Transport delay, never an assertion wait.
                route.fulfill(response=response)

            page.route("**/api/settings", delayed_settings)
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat", timeout=30_000)
            page.click('[data-nav-page="settings"]')
            page.wait_for_selector(".settings-shell", timeout=15_000)

            def remove(buttons, index, target, value=None):
                buttons.nth(index).press("Enter")
                if surface == "fallbacks":
                    page.locator("[data-confirm-ok]").click()
                expect(page.locator(f"{target}:focus")).to_have_count(1)
                if value is not None:
                    expect(page.locator(":focus")).to_have_value(value)

            if surface == 'fallbacks':
                open_settings(page, 'models')
                fallbacks = page.locator('[data-fallback-remove]')
                refused = []

                def refuse_write(route):
                    if route.request.method != "POST":
                        route.fallback()
                        return
                    refused.append(route.request.post_data_json)
                    route.fulfill(status=409, json={"error": "List changed elsewhere", "code": "settings_changed_elsewhere"})

                page.route("**/api/settings", refuse_write)
                fallbacks.first.press("Enter")
                page.locator("[data-confirm-ok]").click()
                expect(page.locator('[data-entry-result]')).to_contain_text("Changed on another device")
                expect(fallbacks).to_have_count(3)
                expect(fallbacks.first).to_be_enabled()
                assert len(refused) == 1, "a refused removal is never retried behind the owner's back"
                page.unroute("**/api/settings", refuse_write)
                remove(fallbacks, 0, '[data-entry-open]')
                remove(fallbacks, 1, '[data-entry-open]')
                remove(fallbacks, 0, '[data-fallback-add]')

            elif surface == 'tools':
                open_settings(page, 'tools')
                editor = open_entry(page, 'alpha')
                editor.locator('[data-mcp-more] > summary').click()
                headers = page.locator('[data-mcp-header-remove]')
                remove(headers, 0, '[data-mcp-header-field="name"]', 'X-B')
                remove(headers, 0, '[data-mcp-header-add]')
                # Removing a saved server asks first and returns to its list.
                for identity, next_id in [('gamma', 'beta'), ('alpha', 'beta'), ('beta', '')]:
                    open_settings(page, 'tools')
                    open_entry(page, identity)
                    page.locator('[data-editor-remove]').focus()
                    page.keyboard.press('Enter')
                    page.locator('[data-confirm-ok]').click()
                    level = settings_page(page, 'tools')
                    level.wait_for(state='visible')
                    expect(level.locator(f'[data-entry-id="{identity}"]')).to_have_count(0)
                    if next_id:
                        expect(level.locator(f'[data-entry-id="{next_id}"] [data-entry-open]')).to_be_focused()
                    else:
                        expect(level.locator('[data-tools-add]')).to_be_focused()

            else:
                level = open_settings(page, 'accounts')
                for key, next_key in [('REMOVE_FOCUS_Z_LAST', 'REMOVE_FOCUS_TWO'),
                                      ('REMOVE_FOCUS_ONE', 'REMOVE_FOCUS_TWO'),
                                      ('REMOVE_FOCUS_TWO', '')]:
                    row = level.locator(f'[data-setting-row="{key}"]')
                    row.locator('[data-secret-open]').click()
                    row.locator('[data-secret-remove]').focus()
                    held = []

                    def hold_write_answer(route):
                        if route.request.method != "POST":
                            route.fallback()
                            return
                        held.append((route, route.fetch()))
                        page.evaluate("window.keyRemovalWriteHeld = true")

                    if not next_key:
                        page.route("**/api/settings", hold_write_answer)
                    page.keyboard.press('Enter')
                    page.locator('[data-confirm-ok]').click()
                    if not next_key:
                        page.wait_for_function("window.keyRemovalWriteHeld === true")
                        # A late write answer must not take the caret from another field the owner chose.
                        level.locator('[data-setting-row="GITHUB_REPO"] [data-setting-control]').focus()
                        page.unroute("**/api/settings", hold_write_answer)
                        route, response = held.pop()
                        route.fulfill(response=response)
                    # A custom key leaves the server's list when cleared; named skill keys stay.
                    expect(row).to_have_count(0)
                    target = (f'[data-setting-row="{next_key}"] [data-secret-open]' if next_key
                              else '[data-setting-row="GITHUB_REPO"] [data-setting-control]')
                    expect(level.locator(target)).to_be_focused()
            capture(page, f"list-editor-remove-focus-{surface}-{engine}")
        except AssertionError:
            print("Removal focus:", page.evaluate("""() => ({active: document.activeElement.outerHTML.slice(0, 500),
                history: window.removalFocus.slice(-12)})"""))
            raise
        finally:
            browser.close()
