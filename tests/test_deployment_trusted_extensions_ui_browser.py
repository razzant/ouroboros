"""The real SPA and server under a deployment with an execution domain, the same server unmanaged before it.

Unmanaged, both seeded extensions are the employee's ordinary cards. With ``execution`` and ``extensions.trusted``
naming only org_notes, its card says the organization delivered it and offers no removal or update, and its switch
turns it on and it runs; team_tools says it doesn't run here, a neutral fact with no review, grant, repair or update
offer, its switch locked, and the server refuses to turn it on before writing anything. No Docker command is run:
the domain names no dedicated container or engine, and nothing here starts work in it. Frames go to
``OUROBOROS_UI_EVIDENCE_DIR`` for a look.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import time
import urllib.request

import pytest

from tests.test_deployment_extensions_ui_browser import _post, _shot, _skills
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

NOTES, OTHER = "org_notes", "team_tools"  # _skills waits for the org_notes card


def _seed(data_dir: pathlib.Path) -> dict:
    """Two small reviewed extensions, saved Off; returns their pins."""
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state

    pins = {}
    for name, purpose in ((NOTES, "Organization notes, delivered by the administrator."),
                          (OTHER, "A team helper an employee installed here.")):
        folder = data_dir / "skills" / "external" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "SKILL.md").write_text(textwrap.dedent(f"""\
            ---
            name: {name}
            description: {purpose}
            version: 1.0.0
            type: extension
            entry: plugin.py
            plugin_api: "2.0"
            permissions: [tool]
            ---
            # {name}
            """), encoding="utf-8")
        (folder / "plugin.py").write_text(f"def register(api):\n    api.register_tool('note', lambda ctx: {name!r}, "
                                          "description='note', schema={})\n", encoding="utf-8")
        loaded = load_skill(folder, data_dir)
        save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
        save_enabled(data_dir, name, False, actor="owner_ui")
        pins[name] = {"name": name, "path": str(loaded.skill_dir), "content_hash": loaded.content_hash}
    return pins


def _listing(url: str) -> dict:
    with urllib.request.urlopen(url + "/api/extensions", timeout=60) as answer:
        return {row["name"]: row for row in json.loads(answer.read())["skills"]}


def _offer_fixture_update(page):
    """Only supply an available catalog version; the real server still supplies deployment/trust and run state."""
    def listing(route):
        response = route.fetch()
        body = response.json()
        for skill in body["skills"]:
            if skill["name"] in (NOTES, OTHER):
                skill["install"] = {"basis": "trusted", "base_known": True,
                                    "source": {"id": "fixture-catalog", "name": "Fixture catalog"},
                                    "update_available": {"digest": "f" * 64, "version": "1.1.0"}}
        route.fulfill(response=response, json=body)
    page.route("**/api/extensions", listing)


@pytest.mark.ui_browser
def test_delivered_and_untrusted_extensions_in_skills(direct_server_with_data, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    data_dir, url = fixture["data_dir"], fixture["url"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "frames")))
    evidence.mkdir(parents=True, exist_ok=True)
    fixture["stop_server"]()
    pins = _seed(data_dir)
    fixture["start_server"]()
    work, policy = tmp_path / "work", tmp_path / "admin" / "deployment-policy.json"
    work.mkdir()
    policy.parent.mkdir()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            # --- unmanaged: both are ordinary cards; the listing names no trust ---
            page = browser.new_page()
            _offer_fixture_update(page)
            _skills(page, url, "light", 1400)
            assert page.locator("#skills-update-all").is_visible()
            for name in (NOTES, OTHER):
                card = page.locator(f'.skills-card[data-skill="{name}"]')
                card.wait_for(state="visible", timeout=30_000)
                assert card.locator(".skills-toggle").is_enabled(), name
                assert card.locator("[data-delivered-by-organization]").count() == 0
                assert "Doesn’t run here" not in card.inner_text()
            _shot(page, evidence, "unmanaged-skills-light-1400")
            assert all("trust" not in row for row in _listing(url).values())
            page.close()

            # --- an execution domain trusting org_notes only ---
            fixture["stop_server"]()
            policy.write_text(json.dumps({"version": 1, "execution": {"executor": {
                "type": "docker_exec", "container_name": "ouro-work", "network": "host",
                "path_mappings": [{"host_path": str(work), "backend_path": str(work)}]}},
                "extensions": {"trusted": [pins[NOTES]]}}), encoding="utf-8")
            fixture["env"]["OUROBOROS_DEPLOYMENT_POLICY"] = str(policy)
            fixture["start_server"]()
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page()
                _offer_fixture_update(page)
                _skills(page, url, theme, width)
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                delivered = page.locator(f'.skills-card[data-skill="{NOTES}"]')
                other = page.locator(f'.skills-card[data-skill="{OTHER}"]')
                other.wait_for(state="visible", timeout=30_000)
                assert delivered.locator("[data-delivered-by-organization]").is_visible()
                assert page.locator("#skills-update-all").is_hidden(), "administered update is not offered in bulk either"
                assert delivered.locator('[data-skill-action="update"]').count() == 0
                assert delivered.locator(".skills-toggle").is_enabled(), "Off and On stay the employee's"
                assert delivered.locator('.skills-card-menu-dialog [data-skill-action="delete"]').count() == 0
                text = other.inner_text()
                assert "Doesn’t run here" in text and "isn’t among them" in text, text
                assert other.locator(".skills-toggle").is_disabled()
                assert other.locator("[data-needs-you]").count() == 0 and other.get_attribute("data-needs-you") is None
                for action in ("review", "rereview", "attest_review", "grant", "repair", "update", "turn_on"):
                    assert other.locator(f'[data-skill-action="{action}"]').count() == 0, action
                assert other.locator('.skills-card-menu-dialog [data-skill-action="delete"]').count() == 1
                assert page.evaluate("document.querySelector('#page-skills').scrollWidth <= innerWidth + 1")
                _shot(page, evidence, f"protected-skills-{theme}-{width}")
                _shot(page, evidence, f"protected-delivered-{theme}-{width}", delivered)
                _shot(page, evidence, f"protected-untrusted-{theme}-{width}", other)
                page.close()

            # --- the doors: the other is refused before any write; the delivered one turns on and runs ---
            enabled_other = data_dir / "state" / "skills" / OTHER / "enabled.json"
            saved = enabled_other.read_bytes()
            refused = _post(url, f"/api/skills/{OTHER}/toggle", {"enabled": True})
            assert (refused[0], refused[1].get("code")) == (403, "managed_by_deployment"), refused
            assert enabled_other.read_bytes() == saved
            page = browser.new_page()
            _skills(page, url, "light", 1400)
            delivered = page.locator(f'.skills-card[data-skill="{NOTES}"]')
            delivered.locator("label.ui-switch").click()
            deadline = time.monotonic() + 90
            while not (row := _listing(url)[NOTES]).get("live_loaded"):
                assert time.monotonic() < deadline, row
                time.sleep(0.5)
            listing = _listing(url)
            assert listing[NOTES]["trust"] == {"state": "trusted", "administered": True} and listing[NOTES]["dispatch_live"]
            assert (listing[OTHER]["trust"]["reason"], listing[OTHER]["live_reason"]) == ("not_delivered",
                                                                                           "deployment_untrusted")
            page.wait_for_function(f"() => document.querySelector('.skills-card[data-skill={NOTES}] .skills-toggle')"
                                   "?.checked === true", timeout=30_000)
            _shot(page, evidence, "protected-delivered-on-light-1400", delivered)
            (evidence / "protected-listing.json").write_text(json.dumps(
                {name: {key: row.get(key) for key in ("enabled", "live_loaded", "live_reason", "trust", "load_error")}
                 for name, row in listing.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
            page.close()
        finally:
            browser.close()
