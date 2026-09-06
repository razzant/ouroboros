"""Dashboard -> Updates update-letter acceptance (docs/ARCHITECTURE.md "Visual verification
policy"): the letter on the real consumer flow, real UI, real browser.

Sibling of ``test_ui_smoke_playwright.py`` (which carries the shared server fixture and
sits at its byte gate); marker-gated the same way, runs in the same CI job.
"""

from __future__ import annotations

import json

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
def test_ui_smoke_update_letter_renders_below_the_action(direct_server, tmp_path):
    """The update letter on the real Dashboard -> Updates flow at a phone width: a named
    section under the single action button and above Recovery, markdown sanitized (no
    script survives, emphasis does), the verdict untouched, and no horizontal page scroll."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    letter = {
        "state": "ready", "relation": "pending",
        "text": ("This update makes the **Updates panel** explain itself <script>window.__letter_xss = 1</script> "
                 "and carries a very long unbroken token " + "x" * 160 + " to prove wrapping.\n\n"
                 "```python\nprint('a fenced block the shared renderer would give a Copy button')\n```"),
        "author_version": "6.113.5", "target_version": "6.114.0",
        "written_at": "2026-09-03T20:10:10+00:00", "error_kind": "", "error_text": "",
        "key": {"base_sha": "a" * 40, "target_sha": "b" * 40, "update_channel": "stable", "target_ref": "managed/ouroboros"},
        "has_last_good": False,
    }
    status = {
        "managed": True, "check_ok": True, "available": True, "safe_to_apply": True,
        "current_version": "6.113.5", "current_short_sha": "abcd1234", "current_sha": "a" * 40,
        "latest_version": "6.114.0", "latest_short_sha": "ef567890", "latest_sha": "b" * 40,
        "checked_at": "2026-09-03T20:10:00+00:00", "warnings": [], "update_tx": {"active": False},
        "behind": 3, "ahead": 0, "dirty": False, "letter": letter,
    }
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        try:
            page.route(
                "**/api/update/status**",
                lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(status)),
            )
            page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector("#page-chat", timeout=30_000)
            page.click('[data-mobile-nav-toggle]')
            page.wait_for_selector('#primary-sidebar.open', timeout=5_000)
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            page.wait_for_selector("#updates-letter:not([hidden])", timeout=10_000)
            info = page.evaluate(
                """() => {
                    const section = document.querySelector('#updates-letter');
                    const label = document.querySelector('#updates-letter-label');
                    const body = document.querySelector('#updates-letter-body');
                    const top = (q) => document.querySelector(q).getBoundingClientRect().top;
                    return {
                        tag: label.tagName, label: label.textContent,
                        labelledBy: section.getAttribute('aria-labelledby'),
                        meta: document.querySelector('#updates-letter-meta').textContent,
                        strong: body.querySelectorAll('strong').length,
                        scripts: body.querySelectorAll('script').length,
                        xss: window.__letter_xss === 1,
                        buttons: section.querySelectorAll('button').length,
                        codeBlocks: section.querySelectorAll('.md-code-block').length,
                        headline: document.querySelector('#updates-summary').textContent,
                        action: document.querySelector('#btn-update-primary').textContent,
                        order: [top('.updates-action-row'), top('#updates-letter'), top('.updates-recovery')],
                        scrollWidth: document.documentElement.scrollWidth,
                        clientWidth: document.documentElement.clientWidth,
                    };
                }"""
            )
            # The rendered card is saved for the human who runs this lane: DEVELOPMENT's
            # visible-change rule is satisfied by INSPECTING it, and a stored file alone is
            # not that inspection — it is what makes the inspection possible from CI.
            shot = tmp_path / "updates_letter.png"
            page.screenshot(path=str(shot), full_page=True)
            assert shot.stat().st_size > 0
            print(f"update letter card rendered for inspection: {shot}")
        finally:
            browser.close()
    assert info["tag"] == "H4" and info["label"] == "What's new"
    assert info["labelledBy"] == "updates-letter-label"
    assert info["meta"].startswith("written by Ouroboros 6.113.5 about 6.114.0")
    assert info["strong"] == 1 and info["scripts"] == 0 and info["xss"] is False
    assert info["buttons"] == 0, "the letter is a fact, never an action — not even a Copy control"
    assert info["codeBlocks"] == 1, "the fenced block still renders; only its control is gone"
    assert info["headline"].startswith("Update available") and info["action"] == "Update to 6.114.0"
    assert info["order"][0] < info["order"][1] < info["order"][2], "action row, then letter, then Recovery"
    assert info["scrollWidth"] == info["clientWidth"], "no horizontal page scroll at a phone width"
