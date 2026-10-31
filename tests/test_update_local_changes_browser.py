"""The update pill and Updates panel name a checkout's own content, not its commit count (#1718).

The real SPA from the real candidate server, light and dark, wide and narrow; only the
status reply is synthetic, one per content state the server reports.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)

_BASE = {
    "managed": True, "check_ok": True, "available": True, "warnings": [], "update_tx": {"active": False},
    "official_repo_url": "https://github.com/razzant/ouroboros", "target_ref": "managed/ouroboros",
    "update_channel": "development", "current_version": "7.6.0", "latest_version": "7.6.1",
    "current_sha": "a" * 40, "current_short_sha": "aaaaaaaa", "latest_sha": "b" * 40,
    "latest_short_sha": "bbbbbbbb", "latest_message": "Official release", "behind": 1, "ahead": 45,
    "dirty": False, "dirty_count": 0,
}
STATES = {
    "absent": ({"local_changes": "absent", "local_changes_base": "c" * 40, "safe_to_apply": True}, "none", ""),
    "present": ({"ahead": 2, "local_changes": "present", "local_changes_base": "d" * 40, "safe_to_apply": False},
                "present since dddddddd", "This checkout changed since its last shared commit"),
    "unknown": ({"local_changes": "unknown", "local_changes_base": "", "local_changes_reason": "several_bases",
                 "safe_to_apply": False}, "unknown", "Could not compare this checkout"),
    "dirty": ({"ahead": 0, "local_changes": "absent", "local_changes_base": "a" * 40, "dirty": True,
               "dirty_count": 3, "safe_to_apply": False}, "none", "Uncommitted changes are stashed and restored"),
}


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_update_pill_and_panel_name_local_content_not_commit_counts(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(server["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            for theme, width in (("dark", 1280), ("light", 1280), ("light", 390)):
                for name, (fields, local_chip, hint) in STATES.items():
                    status = {**_BASE, **fields}
                    page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                    page.route("**/api/update/status*", lambda route: route.fulfill(json=status))
                    page.goto(server["url"], wait_until="domcontentloaded")
                    pill = page.locator("#update-pill")
                    page.wait_for_function("() => document.getElementById('update-pill')?.hidden === false",
                                           timeout=30_000)
                    assert pill.text_content() == "Update available"  # the identity is its title (owner Q6=A)
                    assert pill.get_attribute("title") == "Update 7.6.0 → 7.6.1"
                    assert ("has-local" in (pill.get_attribute("class") or "").split()) == (name != "absent")
                    if width < 980:
                        page.evaluate("document.getElementById('update-pill').click()")
                    else:
                        pill.click()
                    summary = page.locator("#updates-summary")
                    expect(summary).to_contain_text("Update available", timeout=30_000)
                    chips = page.locator("#updates-meta").inner_text()
                    commits = "1 incoming" if fields.get("ahead") == 0 else f"1 incoming / {status['ahead']} local"
                    assert f"Commits: {commits}" in chips and f"Local changes: {local_chip}" in chips, chips
                    assert ("Uncommitted: 3 file(s)" in chips) == (name == "dirty")
                    dot = page.locator("#updates-dot").get_attribute("data-tone")
                    hint_node = page.locator("#updates-hint")
                    if hint:
                        expect(hint_node).to_contain_text(hint)
                        assert dot == "warn", (name, dot)
                    else:
                        assert hint_node.is_hidden() and dot == "ok", (name, dot)
                    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                    page.locator("#updates-status-card").screenshot(
                        path=str(evidence / f"updates-{name}-{theme}-{width}-{engine}.png"))
                    if width >= 980:
                        page.locator("#update-pill").screenshot(
                            path=str(evidence / f"update-pill-{name}-{theme}-{engine}.png"))
                    if name == "dirty" and width >= 980:
                        # The apply confirmation names uncommitted files, the panel's own word; nothing applies.
                        page.route("**/api/update/preflight*", lambda route: route.fulfill(json={"merge_plan": {
                            "available": True, "kind": "clean", "base_sha": "a" * 40, "target_sha": "b" * 40,
                            "local_dirty_count": 3}}))
                        page.route("**/api/update/apply*", lambda route: errors.append("apply was sent"))
                        page.locator("#btn-update-primary").click()
                        dialog = page.locator(".confirm-dialog")
                        expect(dialog).to_contain_text("3 uncommitted file(s) are stashed and restored", timeout=30_000)
                        dialog.screenshot(path=str(evidence / f"updates-confirm-{name}-{theme}-{engine}.png"))
                        dialog.locator("[data-confirm-cancel]").last.click()
                        expect(dialog).to_have_count(0)
                    assert errors == [], errors
                    page.close()
        finally:
            browser.close()
