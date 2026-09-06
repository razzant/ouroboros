"""Settings → Agents acceptance (docs/DESIGN.md "List editors"): real UI, real browser.

Sibling of ``test_ui_smoke_playwright.py`` (which carries the shared server fixture and
sits at its byte gate); marker-gated the same way, runs in the same CI job.
"""

from __future__ import annotations

import json

import pytest

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

_AGENTS_PANEL_VISIBLE_ROWS_JS = """
    (selector) => {
        // One pixel of slack: the list's own border sits on the scroll edge.
        const box = document.querySelector('.settings-scroll').getBoundingClientRect();
        return [...document.querySelectorAll(selector)]
            .map((row) => row.getBoundingClientRect())
            .filter((r) => r.top >= box.top - 1 && r.bottom <= box.bottom + 1).length;
    }
"""


def _open_agents_tab(page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#page-chat", timeout=30_000)
    page.click('[data-nav-page="settings"]')
    page.wait_for_selector(".settings-shell", timeout=15_000)
    page.click('[data-settings-tab="agents"]')
    page.wait_for_selector("#available-subagents-editor .available-subagent-row", timeout=20_000)
    # The list at the top of the scroll body: "three cards fit a laptop-height body" is a
    # claim about the cards, not about the section's heading and copy above them.
    page.evaluate("() => document.querySelector('.available-subagents-list').scrollIntoView({block: 'start'})")


def _agents_panel_add_reveals_the_new_card(page) -> None:
    """The shared add-and-reveal contract, run on whichever engine the caller launched."""
    page.click("[data-subagent-add]")
    page.wait_for_function(
        "() => document.querySelectorAll('.available-subagent-row').length === 4", timeout=5_000)
    # The appended card is fully inside the scroll body and its Description holds the caret;
    # the section-level error line stays hidden and no card is tinted before a save attempt.
    page.wait_for_function(
        """() => {
            const rows = [...document.querySelectorAll('.available-subagent-row')];
            const last = rows[rows.length - 1];
            const box = document.querySelector('.settings-scroll').getBoundingClientRect();
            const r = last.getBoundingClientRect();
            return r.top >= box.top - 1 && r.bottom <= box.bottom + 1
                && document.activeElement === last.querySelector('[data-subagent-field="recommended_use"]');
        }""",
        timeout=5_000,
    )
    assert page.evaluate("() => document.querySelector('[data-subagents-validation]').hidden") is True
    assert page.locator(".available-subagent-row[data-invalid]").count() == 0
    hint = page.locator(".available-subagent-row").last.locator("[data-subagent-meta]")
    assert "Choose how this subagent runs" in hint.inner_text()


def _agents_panel_typing_reads_draft(page) -> None:
    """A keystroke into a SAVED card (no structural repaint) turns every head status to
    Draft at once, patched in place — the caret stays in the field being typed into."""
    field = page.locator('.available-subagent-row [data-subagent-field="recommended_use"]').first
    field.click()
    page.keyboard.type(" ")
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-subagent-status]')]
            .every((el) => el.textContent.startsWith('Draft · '))
            && document.activeElement === document.querySelector(
                '.available-subagent-row [data-subagent-field="recommended_use"]')""",
        timeout=5_000,
    )


@pytest.mark.ui_browser
def test_ui_smoke_agents_panel_list_editor(direct_server_with_data):
    """Settings → Agents: three compact subagent cards fit a laptop-height body; typing turns
    the head status to Draft in place; Add reveals the appended card with the caret in it and
    no error; only a Save attempt (whichever validation aborts it) turns the empty route into
    a section-level line plus a tinted, self-naming card, and the fix typed into the card
    clears line, tint and footer together; a later Add is an invitation again; Review lanes'
    Add lives in the group head and reveals its new row (docs/DESIGN.md "List editors")."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    saved["OUROBOROS_SUBAGENTS"] = json.dumps(_AGENTS_PANEL_ROSTER)
    settings_path.write_text(json.dumps(saved), encoding="utf-8")
    direct_server_with_data["restart_server"]()
    url = direct_server_with_data["url"]

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                _open_agents_tab(page, url)
                assert page.evaluate(_AGENTS_PANEL_VISIBLE_ROWS_JS, ".available-subagent-row") >= 3
                _agents_panel_typing_reads_draft(page)
                _agents_panel_add_reveals_the_new_card(page)

                # Every Save click is an attempt, even one that another field's validation
                # aborts: with a malformed Every-N cadence the roster is still judged (line +
                # tint) while the footer names the cadence error. The segmented control's
                # hidden inputs are poked directly — the cadence row is not the point here.
                def set_cadence(mode, n):
                    page.evaluate(
                        "([mode, n]) => { document.getElementById('s-post-task-evolution-mode').value = mode;"
                        " document.getElementById('s-evo-cadence-n').value = n; }", [mode, n])

                def save_expecting(footer_prefix):
                    page.click("#btn-save-settings")
                    page.wait_for_function(
                        "(p) => document.getElementById('settings-status').textContent.startsWith(p)",
                        arg=footer_prefix, timeout=5_000)

                mode_before = page.evaluate("() => document.getElementById('s-post-task-evolution-mode').value")
                set_cadence("every_n", "x")
                save_expecting("Every-N cadence")
                assert not page.evaluate("() => document.querySelector('[data-subagents-validation]').hidden")
                assert page.locator(".available-subagent-row[data-invalid]").count() == 1
                set_cadence(mode_before, "")
                # With the cadence valid again, Save names the roster error in the footer too.
                save_expecting("Available subagents:")
                line = page.locator("[data-subagents-validation]").inner_text()
                assert line.startswith("Subagent 4 needs a model or agent-session route.")
                tinted = page.locator(".available-subagent-row[data-invalid]")
                assert tinted.count() == 1
                assert tinted.locator('[data-subagent-meta][data-tone="error"]').inner_text().startswith(
                    "Subagent 4 needs")

                # The NEXT added entry is still an invitation: the attempt judged the rows
                # that existed then, not every row forever — row 4 stays judged beside it.
                page.click("[data-subagent-add]")
                page.wait_for_function(
                    "() => document.querySelectorAll('.available-subagent-row').length === 5", timeout=5_000)
                fresh = page.locator(".available-subagent-row").last
                assert not fresh.evaluate("(el) => el.hasAttribute('data-invalid')")
                assert "Choose how this subagent runs" in fresh.locator("[data-subagent-meta]").inner_text()
                assert page.locator(".available-subagent-row[data-invalid]").count() == 1

                # A fix typed into the judged card clears the section line, the tint AND the
                # roster-owned footer message together; the unjudged fresh row keeps none of
                # them alive.
                tinted.locator('[data-subagent-field="model"]').fill("openai/gpt-5.6-luna")
                page.wait_for_function(
                    "() => document.querySelector('[data-subagents-validation]').hidden"
                    " && !document.querySelector('.available-subagent-row[data-invalid]')"
                    " && !document.getElementById('settings-status').textContent.startsWith('Available subagents:')",
                    timeout=5_000,
                )

                # A newer, unrelated footer message is not the roster's to clear: the cadence
                # error written by the next Save survives the roster fix that follows it.
                set_cadence("every_n", "x")
                save_expecting("Every-N cadence")
                assert page.locator(".available-subagent-row[data-invalid]").count() == 1
                fresh.locator('[data-subagent-field="model"]').fill("openai/gpt-5.6-luna")
                page.wait_for_function(
                    "() => document.querySelector('[data-subagents-validation]').hidden"
                    " && !document.querySelector('.available-subagent-row[data-invalid]')", timeout=5_000)
                assert page.locator("#settings-status").inner_text().startswith("Every-N cadence")
                set_cadence(mode_before, "")

                # Review lanes: the group's Add sits in its head and reveals the appended row.
                assert page.evaluate(
                    "() => Boolean(document.getElementById('btn-add-triad-slot').closest('.reviewer-slots-head'))")
                before = page.locator("#reviewer-triad-rows .reviewer-slot-row").count()
                page.click("#btn-add-triad-slot")
                page.wait_for_function(
                    """(before) => {
                        const rows = document.querySelectorAll('#reviewer-triad-rows .reviewer-slot-row');
                        if (rows.length !== before + 1) return false;
                        const last = rows[rows.length - 1];
                        const box = document.querySelector('.settings-scroll').getBoundingClientRect();
                        const r = last.getBoundingClientRect();
                        return r.top >= box.top - 1 && r.bottom <= box.bottom + 1
                            && document.activeElement === last.querySelector('[data-slot-route]');
                    }""",
                    arg=before,
                    timeout=5_000,
                )
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
                    _open_agents_tab(page, url)
                    _agents_panel_add_reveals_the_new_card(page)
                finally:
                    webkit.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


def _wizard_step_until(page, predicate_js: str, forward: bool, limit: int = 8) -> None:
    """Walk the wizard with Next/Back until `predicate_js` holds. Steps that hold Continue
    until they have a value (a provider key, the main/light models) get placeholders — the
    subject here is the Agents step and the summary's Finish, not those steps."""
    placeholders = {
        "#openrouter-key": "sk-or-placeholder-not-real",
        "#main-model": "openai/gpt-5.6-luna",
        "#light-model": "openai/gpt-5.6-luna",
    }
    for _ in range(limit):
        if page.locator("#onboarding-available-subagents").count():
            # The Agents step settles asynchronously (saved roster or generated draft);
            # judge it only once it shows rows or its own failure line.
            page.wait_for_function(
                "() => document.querySelectorAll('#onboarding-available-subagents .available-subagent-row').length"
                " || !document.querySelector('#onboarding-available-subagents [data-subagents-validation]').hidden",
                timeout=20_000)
        if page.evaluate(predicate_js):
            return
        if forward and page.evaluate("() => Boolean(document.getElementById('next-btn')?.disabled)"):
            for selector, value in placeholders.items():
                if page.locator(selector).count() and not page.input_value(selector):
                    page.fill(selector, value)
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
    """First-run wizard (docs/ARCHITECTURE.md §2): an unrouted entry added on the Agents step
    does not block Continue; Finish on the summary reports it and, back on Agents, the card
    is already tinted and self-naming; the fix reconciles line and tint together and the
    second Finish passes the wizard's own checks and enters saving (the save's provider
    round-trip is not this test's subject)."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    # A saved roster on an OpenRouter-shaped install: the Agents step's preview validates
    # the model setup the way a first run does, and the shared fixture's mock-LLM model is
    # not a confirmed main model — so this test mirrors an owner's machine instead.
    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"):
        saved.pop(key, None)
    saved["OPENROUTER_API_KEY"] = "sk-or-v1-smoke-placeholder-not-real"
    saved["OUROBOROS_SUBAGENTS"] = json.dumps(_AGENTS_PANEL_ROSTER)
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

                tinted.locator('[data-subagent-field="model"]').fill("openai/gpt-5.6-luna")
                page.wait_for_function(
                    "() => document.querySelector('#onboarding-available-subagents [data-subagents-validation]').hidden"
                    " && !document.querySelector('#onboarding-available-subagents .available-subagent-row[data-invalid]')",
                    timeout=5_000)

                _wizard_step_until(page, _WIZARD_ON_SUMMARY_JS, forward=True)
                page.click("#next-btn")
                # The second Finish passes the wizard's own checks and hands the draft to
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
                    raise AssertionError(f"second Finish was refused: {seen}") from exc
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
