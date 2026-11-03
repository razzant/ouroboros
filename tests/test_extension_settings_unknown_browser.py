"""Settings → Advanced: an extension form whose stored values nobody can read (#1552), in real engines.

A real server (``direct_server_with_data``: a candidate checkout over an isolated data root)
hosts four reviewed extensions whose settings route answers the hydration GET the way
installed skills do: an older POST-only skill (the real dispatcher's 405), a store that
fails (500), an explicit empty document (``{}``) and stored values. Only the first offers
Edit anyway, and only after it does Save send; that POST reaches the skill with every
field. Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

UNAVAILABLE = "Current values are unavailable. These are defaults. Sending this form may change saved settings."
EDITING = "You chose to edit without reading the saved values. Review every field before sending."
UNREADABLE = "Current values could not be read; Save is disabled so it does not overwrite them. Use Reload Settings to retry."
FIELDS = [
    {"name": "collection", "label": "Reminder collection", "type": "text", "default": "Default collection"},
    {"name": "hour", "label": "Daily reminder hour", "type": "number", "default": "9", "min": 0, "max": 23},
    {"name": "token", "label": "Optional token", "type": "password",
     "help": "Existing secret values are never shown in this form."},
    {"name": "description", "label": "Collection description", "type": "textarea"},
]
STORED = {"collection": "My ongoing research", "hour": "18",
          "description": "Articles to read and experiments to revisit this week."}
# name: (section title, route methods, what the GET answers)
SKILLS = {
    "legacy_reminders": ("Reading reminders", ("POST",), None),
    "offline_reminders": ("Reminders with an offline store", ("GET", "POST"), "fail"),
    "fresh_reminders": ("Reminders saved nowhere yet", ("GET", "POST"), {}),
    "stored_reminders": ("Reminders already saved", ("GET", "POST"), STORED),
}


def _write_extension(data_dir: pathlib.Path, name: str, title: str, methods, answer) -> None:
    """An exact-hash reviewed extension with one settings form on ``config/save``."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Synthetic reminders skill with a settings form.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["route", "widget"]
        ---
        # {title}
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        import json

        ANSWER = {json.dumps(answer)!r}


        async def config(request):
            if request.method == "GET":
                if ANSWER == '"fail"':
                    raise RuntimeError("reminder store offline")
                return json.loads(ANSWER)
            body = await request.json()
            return {{"message": "Received " + json.dumps(body, sort_keys=True)}}


        def register(api):
            api.register_route("config/save", config, methods={tuple(methods)!r})
            api.register_settings_section("config", {title!r}, schema={{"components": [{{
                "type": "form", "id": "reminders", "route": "config/save", "method": "POST",
                "submit_label": "Save", "fields": {FIELDS!r},
            }}]}})
        """), encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))


def _enable(page, name: str) -> None:
    toggled = page.evaluate("""async (skill) => {
        const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
        });
        return {status: response.status, body: await response.json()};
    }""", name)
    assert toggled["status"] == 200 and toggled["body"].get("enabled") is True, toggled


def _form(page, name: str):
    return page.locator(f'[data-extension-settings-form][data-skill="{name}"]')


def test_unread_extension_values_stay_locked_until_edit_anyway_in_chromium_and_webkit(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    for name, (title, methods, answer) in SKILLS.items():
        _write_extension(data_dir, name, title, methods, answer)
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    enabled = False
    try:
        with sync_playwright() as pw:
            for engine in ("chromium", "webkit"):
                browser = getattr(pw, engine).launch(headless=True)
                try:
                    # The owner's desktop window in both themes, then a phone-width card.
                    for theme, width, height in (("light", 1100, 750), ("dark", 1100, 750), ("light", 390, 844)):
                        context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme)
                        page = context.new_page()
                        errors = []
                        page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                        if not enabled:
                            for name in SKILLS:
                                _enable(page, name)
                            enabled = True
                        page.wait_for_selector('[data-nav-page="settings"]', state="attached", timeout=30_000)
                        # On a phone the sidebar is a closed drawer; the page, not the drawer, is under test.
                        page.locator('[data-nav-page="settings"]').evaluate("el => el.click()")
                        page.locator('[data-settings-tab="advanced"]').click()
                        legacy = _form(page, "legacy_reminders")
                        legacy.wait_for(state="visible", timeout=60_000)
                        for name in SKILLS:
                            _form(page, name).wait_for(state="visible", timeout=60_000)

                        # 405: the defaults are shown locked under the warning; only Edit anyway is live.
                        expect(legacy.locator("[data-extension-settings-read]")).to_have_text(UNAVAILABLE)
                        for field in ("collection", "hour", "token", "description"):
                            expect(legacy.locator(f'[name="{field}"]')).to_be_disabled()
                        expect(legacy.locator('[name="collection"]')).to_have_value("Default collection")
                        expect(legacy.locator('button[type="submit"]')).to_be_disabled()
                        expect(legacy.locator("[data-extension-settings-edit-anyway]")).to_be_enabled()

                        # A failed read stays blocked with no way around it.
                        offline = _form(page, "offline_reminders")
                        expect(offline.locator("[data-extension-settings-status]")).to_have_text(UNREADABLE)
                        expect(offline.locator('button[type="submit"]')).to_be_disabled()
                        assert offline.locator("[data-extension-settings-edit-anyway], [data-extension-settings-read]").count() == 0
                        # `{}` is a real empty document; stored values hydrate. Both save normally.
                        fresh, stored = _form(page, "fresh_reminders"), _form(page, "stored_reminders")
                        expect(fresh.locator('[name="collection"]')).to_have_value("Default collection")
                        expect(stored.locator('[name="collection"]')).to_have_value(STORED["collection"])
                        for ready in (fresh, stored):
                            expect(ready.locator('button[type="submit"]')).to_be_enabled()
                            assert ready.locator("[data-extension-settings-edit-anyway], [data-extension-settings-read]").count() == 0

                        geometry = page.evaluate("""() => {
                            const scroller = document.querySelector('.settings-scroll');
                            const legacy = document.querySelector('[data-skill="legacy_reminders"]');
                            const [save, anyway] = legacy.querySelectorAll('.settings-extension-actions .btn');
                            const a = save.getBoundingClientRect(), b = anyway.getBoundingClientRect();
                            return {overflow: scroller.scrollWidth - scroller.clientWidth,
                                    sameRow: Math.abs(a.top - b.top) < 1, height: Math.min(a.height, b.height)};
                        }""")
                        assert geometry["overflow"] <= 1, geometry
                        assert geometry["sameRow"] is (width > 600), geometry
                        assert geometry["height"] >= 24, geometry
                        legacy.locator("xpath=ancestor::article[1]").scroll_into_view_if_needed()
                        page.wait_for_timeout(200)
                        page.screenshot(path=str(evidence / f"1552-{engine}-{theme}-{width}-unknown.png"))
                        offline.locator("xpath=ancestor::article[1]").screenshot(
                            path=str(evidence / f"1552-{engine}-{theme}-{width}-failed.png"))

                        if (theme, width) == ("light", 1100):
                            legacy.locator("[data-extension-settings-edit-anyway]").click()
                            expect(legacy.locator("[data-extension-settings-read]")).to_have_text(EDITING)
                            expect(legacy.locator("[data-extension-settings-edit-anyway]")).to_be_disabled()
                            expect(legacy.locator('button[type="submit"]')).to_be_enabled()
                            legacy.locator('[name="collection"]').fill("My explicit choice")
                            page.screenshot(path=str(evidence / f"1552-{engine}-{theme}-{width}-edit-anyway.png"))
                            legacy.locator('button[type="submit"]').click()
                            status = legacy.locator("[data-extension-settings-status]")
                            expect(status).to_contain_text("Received ", timeout=30_000)
                            received = json.loads(status.inner_text().split("Received ", 1)[1])
                            assert received == {"collection": "My explicit choice", "hour": "9", "token": "",
                                                "description": ""}, received
                            page.screenshot(path=str(evidence / f"1552-{engine}-{theme}-{width}-sent.png"))
                        assert not errors, errors
                        context.close()
                finally:
                    browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
