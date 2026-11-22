"""WebKit guard: Settings never grows a horizontal scrollbar.

The owner's report was a thick red horizontal scrollbar across the bottom of
Settings -> Advanced in the packaged desktop shell (WKWebView), where skill
forms stood then; a skill's form is its own Settings page now
(``skill/<skill id>``). Blink's UA stylesheet clips a native ``<select>``
(``overflow: clip``); WebKit computes ``visible``, so a long selected option
paints past the control's own box, the page scroller grows,
and the scrollbar the app paints for it was UA-thick because the global
``::-webkit-scrollbar`` recipe sized only ``width``.

Chromium is deliberately NOT exercised here: the class cannot appear there, so
a second engine would only pin the engine that never had the bug.
"""

from __future__ import annotations

import os
import pathlib
import textwrap

import pytest

from tests.settings_browser_helpers import open_settings

pytest_plugins = ("tests.test_ui_smoke_playwright",)


# The real trigger from the owner's screenshot: a 65-character sentence as the
# selected option of an extension settings select.
LONG_OPTION_LABEL = "Full access (default) — raw owner commands incl. /panic, /restart"
SECTION_TITLE = "Owner command access"


def _write_long_option_settings_extension(data_dir: pathlib.Path) -> str:
    """Install an exact-hash reviewed extension whose settings form carries the
    long-label select, rendered by the production settings renderer.

    The field ORDER mirrors the real Telegram section and is load-bearing: the
    long select is the SECOND field, so it lands in the right-hand column of
    ``.form-grid.two``. A leak out of a left-column control is absorbed by the
    empty right half of the grid and never reaches the page scroller, so a
    left-column reproduction would be green on the unfixed stylesheet too.
    """
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_review_state,
    )

    name = "settings_overflow_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Settings select whose label is long enough to overflow.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Settings overflow smoke
            """
        ),
        encoding="utf-8",
    )
    plugin = textwrap.dedent(
        '''\
        async def save(request):
            body = await request.json()
            return {"message": f"Saved {body.get('access') or 'default'} access."}


        def register(api):
            api.register_route("save", save, methods=("POST",))
            api.register_settings_section(
                "config",
                "__SECTION_TITLE__",
                schema={
                    "components": [
                        {
                            "type": "form",
                            "id": "access-form",
                            "route": "save",
                            "method": "POST",
                            "submit_label": "Save access",
                            "fields": [
                                {
                                    "name": "language",
                                    "label": "Language",
                                    "type": "select",
                                    "default": "en",
                                    "options": [
                                        {"label": "English", "value": "en"},
                                        {"label": "Russian", "value": "ru"},
                                    ],
                                },
                                {
                                    "name": "command_mode",
                                    "label": "Command mode",
                                    "type": "select",
                                    "default": "full",
                                    "options": [
                                        {"label": "__LONG_LABEL__", "value": "full"},
                                        {"label": "Strict", "value": "strict"},
                                    ],
                                },
                            ],
                        }
                    ]
                },
            )
        '''
    )
    plugin = plugin.replace("__LONG_LABEL__", LONG_OPTION_LABEL)
    plugin = plugin.replace("__SECTION_TITLE__", SECTION_TITLE)
    (skill_dir / "plugin.py").write_text(plugin, encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(
        data_dir,
        name,
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    return name


@pytest.mark.ui_browser
def test_a_skill_settings_page_never_scrolls_horizontally_in_webkit(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_long_option_settings_extension(data_dir)
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    screenshot = evidence_dir / "settings-advanced-webkit-no-horizontal-overflow.png"

    try:
        with sync_playwright() as pw:
            browser = pw.webkit.launch(headless=True)
            # The narrow desktop shell window the owner reported from; the leak
            # grows as the window narrows, so this is the honest width to pin.
            page = browser.new_page(viewport={"width": 1000, "height": 680})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                toggled = page.evaluate(
                    """async (skill) => {
                        const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({enabled: true}),
                        });
                        return {status: response.status, body: await response.json()};
                    }""",
                    skill,
                )
                assert toggled["status"] == 200, toggled
                assert toggled["body"].get("enabled") is True, toggled

                page.wait_for_selector('[data-nav-page="settings"]', timeout=30_000)
                page.click('[data-nav-page="settings"]')
                level = open_settings(page, f"skill/{skill}", timeout=30_000)
                section = level.locator('.settings-extension-section').filter(
                    has_text=SECTION_TITLE
                )
                section.wait_for(state="visible", timeout=30_000)
                select = section.locator('select[name="command_mode"]')
                select.wait_for(state="visible", timeout=30_000)
                # The long option must actually be the selected one, or the
                # measurement below would prove nothing.
                assert select.input_value() == "full"
                assert LONG_OPTION_LABEL in select.inner_text()

                # The skill's page is the scroller that holds its form.
                overflow = level.evaluate("s => s.scrollWidth - s.clientWidth")
                widths = select.evaluate(
                    """(element) => [
                        element.getBoundingClientRect().width,
                        element.parentElement.getBoundingClientRect().width,
                    ]"""
                )
                # The page is an inner scroller, so a full-page shot would
                # frame its top and not the offending control.
                select.scroll_into_view_if_needed()
                page.wait_for_timeout(150)
                # Captured before the assertions so a red run leaves evidence too.
                page.screenshot(path=str(screenshot), full_page=True)

                assert overflow <= 1, (
                    "Settings scrolls horizontally: the skill's page is "
                    f"{overflow}px wider than its own box. A native select's "
                    "value is not clipped by WebKit, so a long option leaks "
                    "into the page scroller and the app paints the horizontal "
                    "bar the owner reported."
                )
                assert widths[0] <= widths[1] + 1, (
                    "the select is wider than its grid cell: "
                    f"{widths[0]}px control in a {widths[1]}px cell"
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
