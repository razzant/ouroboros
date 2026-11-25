"""Introduce an Ouroboros in the real page on a keyless server (DESIGN "Circles and Add", Your Ouroboroses).

The owner invites from Circles: a name, a suggested rule, the one-time invitation with Copy. The other side's
introduction arrives under that invitation's token at the real ``POST /api/peer-links/introduce`` (as the accepting
installation's ``peer_links.accept_invite`` makes it), and the open sheet turns to the code check on its own; They
match starts the link here, and its row says the other side is listening with consciousness off, with the direction
in words. Nothing is wider than the window.
"""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request

import pytest

from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser]


def _introduce(url: str, invite: str) -> dict:
    """The accepting side's call under the one-time token, with its own (fictional) install and key hash."""
    body = invite[len("ouroboros-invite:1:"):]
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    request = urllib.request.Request(url + "/api/peer-links/introduce", method="POST", data=json.dumps({
        "link_id": payload["link_id"], "install_id": "install-tarel-test", "url": "http://127.0.0.1:9", "label": "Ouroboros",
        "key_sha256": hashlib.sha256(b"tarel-secret").hexdigest(), "may_start": True, "protocol": 1}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Ouroboros-Invite": payload["token"], "X-Ouroboros-Protocol": "1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [1100, 390])
def test_an_invitation_used_by_the_other_side_turns_to_the_code_check_and_connects(direct_server_with_data, engine, width):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": width, "height": 750})
        errors: list = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url + "/#circles")
            page.wait_for_selector("[data-own-block]", timeout=60_000)
            assert "None yet." in page.locator("[data-own-block]").inner_text()
            page.locator('[data-own-block] [data-own-act="introduce"]').click()
            page.locator("[data-intro-name]").fill("Tarel")
            assert page.locator("[data-intro-go]").is_disabled()
            assert page.locator("[data-intro-rule]").input_value().startswith("Tarel is my linked Ouroboros.")
            choice = page.get_by_role("group", name="Who can log in to Tarel’s computer?")
            choice.get_by_role("button", name="Other people too").click()
            assert page.locator("[data-intro-then]").inner_text() == "Tarel can write but cannot start work here."
            choice.get_by_role("button", name="Only me").press("Space")
            assert page.locator("[data-intro-then]").inner_text() == "Tarel can write and ask for work here."
            assert choice.get_by_role("button", name="Only me").evaluate("node => node === document.activeElement")
            with page.expect_request(lambda req: req.method == "POST" and req.url.endswith("/api/peer-links/invite")) as sent:
                page.locator("[data-intro-go]").click()
            assert sent.value.post_data_json["may_start"] is True
            token = page.locator("[data-intro-token]")
            token.wait_for(timeout=30_000)
            # One read-only line beside Copy, its value whole in the page: the ellipsis is drawn, never
            # in its text, and nothing is in a title.
            invite = token.inner_text().strip()
            assert invite.startswith("ouroboros-invite:1:") and "…" not in invite and token.get_attribute("title") is None
            assert token.evaluate("node => node.getBoundingClientRect().height < 64 && node.closest('.peer-invite-row')"
                                  ".querySelector('[data-intro-copy]') !== null"), "one line with Copy beside it"
            selected = token.evaluate("node => { getSelection().selectAllChildren(node); return getSelection().toString(); }")
            assert selected == invite and "\n" not in invite, "selecting the line takes the whole value, unbroken"
            assert page.locator("[data-intro-use]").inner_text() == "Not used yet"
            answer = _introduce(url, invite)
            # The open sheet reads only this local link until it appears, then asks for the code check.
            page.wait_for_selector("[data-intro-match]", timeout=30_000)
            assert page.locator(".peer-code").inner_text().replace(" ", "") == answer["code"]
            page.locator("[data-intro-match]").click()
            page.wait_for_selector(".peer-intro", state="detached", timeout=30_000)
            row = page.locator('.own-link[data-state="active"]')
            row.wait_for(timeout=30_000)
            facts = " ".join(row.locator("summary").inner_text().split())
            assert "Tarel" in facts and "Hears only · consciousness off" in facts and "Can ask for work here" in facts, facts
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 0
            assert not errors, errors
        finally:
            browser.close()
