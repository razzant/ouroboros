"""CI-only witness (not shipped): the real Skills page, Grant access, then Cancel, on a synthetic calendar skill.

The listing row comes from the product's own readers over a real synthetic skill folder (field saved, skill
reviewed and on, no grant). The page is the real `initSkills` mounted by `tests/_skills_page_harness.py`; a guard
registered after the harness sees every request first: a non-GET or a foreign host is recorded and aborted, never
answered. The frame is written only after every check before and after Cancel has passed.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests._skills_page_harness import URL, SkillsPage, skills_browser as _skills_browser

pytestmark = [pytest.mark.ui_browser]
skills_browser = _skills_browser

NAME, KEY, SAMPLE = "demo-calendar", "DEMO_CALENDAR_KEY", "synthetic-value-never-shown"
REPO = Path(__file__).resolve().parents[1]


def _synthetic_row(drive: Path) -> dict:
    from ouroboros.gateway.marketplace import _installed_skill_payload
    from ouroboros.skill_catalogue import skill_status
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state, summarize_skills

    folder = drive / "skills" / "external" / NAME
    (folder / "scripts").mkdir(parents=True)
    (drive / "state" / "skills" / NAME).mkdir(parents=True, exist_ok=True)
    (folder / "scripts" / "upcoming.py").write_text("print('No events in this example')\n", encoding="utf-8")
    (folder / "SKILL.md").write_text(
        f"---\nname: {NAME}\ndescription: Shows upcoming calendar events in this example.\nversion: '1.0.0'\n"
        f"type: script\nruntime: python3\nscripts:\n  - name: upcoming.py\nenv_from_settings: [{KEY}]\n"
        f"when_to_use: When the example calendar applies.\n---\nSynthetic skill; no calendar or account data.\n",
        encoding="utf-8")
    save_review_state(drive, NAME, SkillReviewState(status="clean", content_hash=find_skill(drive, NAME).content_hash))
    save_enabled(drive, NAME, True, actor="owner_ui")
    summary = next(row for row in summarize_skills(drive)["skills"] if row["name"] == NAME)
    row = {**summary, **_installed_skill_payload(find_skill(drive, NAME), drive)}
    grants = row["grants"]
    assert row["enabled"] and skill_status(summary)["ready"], skill_status(summary)
    assert not grants["blocked"] and grants["pending_keys"] == [KEY] and KEY not in grants["granted_keys"], grants
    assert SAMPLE not in json.dumps(row)
    return row


def test_grant_access_dialog_names_the_skill_and_field_and_cancel_changes_nothing(skills_browser, tmp_path, monkeypatch):
    from playwright.sync_api import expect
    from ouroboros import config

    frames = os.environ["OUROBOROS_UI_EVIDENCE_OUT"]
    node = os.environ["PLAYWRIGHT_NODEJS_PATH"]
    pinned = json.loads((REPO / "ouroboros" / "claudexor_runtime_pin.json").read_text(encoding="utf-8"))["release"]["node_version"]
    assert subprocess.check_output([node, "--version"], text=True, timeout=10).strip() == f"v{pinned}"

    drive = tmp_path / "data"
    drive.mkdir()
    settings = drive / "settings.json"
    settings.write_text(json.dumps({KEY: SAMPLE, "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": False}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: False)
    for key, value in {"OUROBOROS_DATA_DIR": str(drive), "OUROBOROS_SETTINGS_PATH": str(settings),
                       "OUROBOROS_SKILLS_REPO_PATH": "", "OUROBOROS_RUNTIME_MODE": "advanced",
                       "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false"}.items():
        monkeypatch.setenv(key, value)
    row = _synthetic_row(drive)
    written = settings.read_bytes()

    browser = skills_browser.chromium.launch(headless=True, chromium_sandbox=True)
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750}, device_scale_factor=1, color_scheme="light",
                                service_workers="block")
        errors, refused = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda event: errors.append(event.text) if event.type == "error" else None)
        facts = SkillsPage(page, extensions=[row])
        facts.answers["/api/mcp/status"] = {"enabled": False, "servers": []}

        def guard(route):  # registered after the harness, so it sees each request first
            request, url = route.request, urlsplit(route.request.url)
            if request.method != "GET" or url.netloc != urlsplit(URL).netloc:
                refused.append((request.method, url.netloc, url.path))
                return route.abort()
            if url.path == "/favicon.ico":
                return route.fulfill(status=204, body="")
            return route.fallback()

        page.route("**/*", guard)
        page.goto(URL, wait_until="domcontentloaded", timeout=15000)
        page.evaluate("() => { const r = document.documentElement; r.lang = 'en'; r.dataset.theme = 'light'; "
                      "r.classList.add('ouro-ui'); }")

        card = page.locator(f'article.skills-installed-card[data-skill="{NAME}"]')
        expect(card).to_be_visible(timeout=10000)
        note = card.locator(".skills-tray-note").filter(has_text=f"{KEY} is saved in Settings but not granted to this skill.")
        expect(note).to_be_visible(timeout=5000)
        switch = card.get_by_role("switch")
        expect(switch).to_be_checked()
        expect(switch).to_be_enabled()
        assert card.get_attribute("data-needs-you") is None, "a saved field alone does not need the owner"
        button = card.get_by_role("button", name="Grant access", exact=True).and_(
            card.locator('[data-skill-action="grant"]:visible'))
        expect(button).to_be_visible()

        assert settings.read_bytes() == written
        reads = len(facts.requests)
        button.click()
        dialog = page.get_by_role("dialog", name=f"Grant access to {NAME}", exact=True)
        expect(dialog).to_be_visible()
        assert "confirm-dialog" in dialog.get_attribute("class").split()
        body = dialog.locator(".ui-modal-body")
        expect(body).to_contain_text(KEY)
        assert body.inner_text().count(KEY) == 1
        assert not dialog.locator("input").count(), "the saved value is not asked for again"
        cancel = dialog.get_by_role("button", name="Cancel", exact=True)
        expect(cancel).to_be_visible()
        expect(dialog.get_by_role("button", name="Grant access", exact=True)).to_be_enabled()
        assert ("GET", "/api/extensions") in facts.requests[reads:], "Grant access reads the current row first"
        assert SAMPLE not in page.content(), "the saved value is never rendered"
        box = dialog.bounding_box()
        assert box and box["x"] >= 0 and box["y"] >= 0 and box["x"] + box["width"] <= 1100 and box["y"] + box["height"] <= 750, box
        frame = page.screenshot(animations="disabled", timeout=10000)
        assert struct.unpack(">II", frame[16:24]) == (1100, 750)

        cancel.click()
        expect(dialog).to_have_count(0)
        expect(button).to_be_enabled()
        expect(note).to_be_visible()
        expect(switch).to_be_checked()
        unexpected = [path for _, path in facts.requests if path.split("?")[0] not in facts.answers]
        assert not refused and not facts.posts, (refused, facts.posts)
        assert settings.read_bytes() == written, "Cancel and the page left Settings untouched"
        assert not unexpected and not errors, (unexpected, errors)
    finally:
        browser.close()
    Path(frames).mkdir(parents=True, exist_ok=True)
    (Path(frames) / "skills-grant-dialog-1100x750.png").write_bytes(frame)
