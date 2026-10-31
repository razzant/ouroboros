"""Focused browser proof for the selected skill-publish flow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import (
    direct_server_with_data as _direct_server_with_data,
)

direct_server_with_data = _direct_server_with_data
# The connected sources as the gateway lists them: the destination dialog offers the hub's shared catalog.
SOURCES = json.loads((Path(__file__).resolve().parents[1] / "web/tests/fixtures/skills_v4/skill_sources.json")
                     .read_text(encoding="utf-8"))


@pytest.mark.ui_browser
def test_ui_publish_stale_card_reaches_selected_preflight_and_task(
    direct_server_with_data,
):
    """A rendered card remains agent-repairable after its manifest disappears."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    data_dir = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    settings_path = data_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["GITHUB_TOKEN"] = "ui-smoke-github-token"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    skill_name = "publish-stale-card"
    skill_root = data_dir / "skills" / "external" / skill_name
    skill_root.mkdir(parents=True, exist_ok=True)
    manifest_path = skill_root / "SKILL.md"
    manifest_path.write_text(
        "---\n"
        f"name: {skill_name}\n"
        "type: instruction\n"
        "description: stale-card publish smoke\n"
        "version: 0.1.0\n"
        "---\n"
        f"# {skill_name}\n",
        encoding="utf-8",
    )
    direct_server_with_data["restart_server"]()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.route("**/api/skill-sources", lambda route: route.fulfill(json=SOURCES))
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                # Installed cards are usable before any source is read.
                page.click('[data-nav-page="skills"]')
                card = page.locator(f'.skills-card[data-skill="{skill_name}"]').first
                card.wait_for(state="visible", timeout=30_000)
                assert card.locator(".skills-submit-hub").get_attribute("data-submit-disabled") == "false"

                # The passive catalogue no longer contains this row, but the selected
                # preflight still knows the clicked leaf and can return needs_attention.
                manifest_path.unlink()
                card.locator("[data-skill-menu-trigger]").click()
                page.locator(f'.skills-card-menu-dialog:not([data-leaving])[open] .skills-submit-hub[data-skill="{skill_name}"]').click()
                # Where it goes is the owner's choice; the preflight checks the files for that destination.
                destinations = page.locator(".skills-dialog")
                destinations.wait_for(state="visible", timeout=30_000)
                assert destinations.locator('[aria-checked="true"] .chat-quiz-option-label').inner_text() == (
                    "Ouroboros Community — share with everyone")
                with page.expect_request(
                    f"**/api/skills/{skill_name}/publish-preflight",
                    timeout=30_000,
                ) as preflight_request:
                    destinations.locator("[data-dialog-ok]").click()
                destination = {"source_id": "ouroboros-community", "mode": "entry"}
                assert preflight_request.value.method == "POST"
                assert preflight_request.value.post_data_json == {"destination": destination}

                dialog = page.locator(".confirm-dialog")
                dialog.wait_for(state="visible", timeout=30_000)
                assert "Not ready yet" in (dialog.locator("#confirm-dialog-title").inner_text() or "")
                dialog.locator(".confirm-dialog-details summary").click()
                dialog_text = dialog.inner_text()
                assert "Needs attention" in dialog_text
                assert "snapshot_manifest_missing" in dialog_text
                with page.expect_response(
                    lambda response: response.request.method == "POST"
                    and response.url.rstrip("/").endswith("/api/tasks"),
                    timeout=30_000,
                ) as admission:
                    dialog.locator("[data-confirm-ok]").click()
                page.wait_for_function("() => !document.querySelector('.confirm-dialog')", timeout=30_000)

                response = admission.value
                assert response.status == 200, response.text()
                accepted = response.json()
                assert accepted["ok"] is True
                task_id = accepted["task_id"]
                payload = response.request.post_data_json
                assert payload["type"] == "skill_publish"
                assert payload["metadata"]["skill_publish_target"] == {
                    "skill": skill_name,
                    "repository": "razzant/OuroborosHub",
                    "destination": destination,
                }
                assert "workspace_root" not in payload
                assert "acceptance_claims" not in payload
                # Admission, live Main projection and history must agree on the
                # real identity. A fake HTTP success cannot prove this boundary.
                assert page.request.get(f"{url}/api/tasks/{task_id}").status == 200
                page.click('[data-nav-page="chat"]')
                visible_task = page.locator(f'#page-chat .chat-live-card[data-task-id="{task_id}"]')
                visible_task.wait_for(state="visible", timeout=30_000)
                page.reload(wait_until="domcontentloaded")
                visible_task.wait_for(state="visible", timeout=30_000)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            if "chromium" in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(","):
                raise
            pytest.skip(str(exc))
        raise
