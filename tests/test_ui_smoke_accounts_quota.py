"""Accounts and saved subagent cards distinguish incomplete quota from refusal."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.settings_browser_helpers import open_settings

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
def test_accounts_partial_quota_cards(direct_server_with_data):
    from playwright.sync_api import expect, sync_playwright

    fixture = json.loads((Path(__file__).resolve().parents[1]
                          / "web/tests/fixtures/quota_window_facts.json").read_text(encoding="utf-8"))
    examples = [row for row in fixture["cases"] if row["name"] in {
        "missing reset", "future reset", "explicit cooldown",
    }]
    profiles = ["missing-reset", "future-reset", "cooldown"]
    roster = {"enabled": True, "items": [{
        "subagent_id": profile, "recommended_use": f"Quota fixture: {example['name']}.",
        "route": {"kind": "agent_session", "target_id": "claude=fable",
                  "credential_profile_id": profile},
    } for profile, example in zip(profiles, examples)]}
    settings_path = direct_server_with_data["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["OUROBOROS_SUBAGENTS"] = json.dumps(roster)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    direct_server_with_data["restart_server"]()
    snapshot = {
        "daemon": {"state": "running", "engine_version": "3.9.8", "runtime": {"state": "ready"}},
        "harnesses": [{"id": "claude", "display_name": "Claude Code", "status": "ok",
                       "enabled": True, "models": [{"id": "fable"}]}],
        "profiles": {"profiles": [{
            "profile": {"harness_id": "claude", "profile_id": profile, "enabled": True},
            "status": {"verification": "passed", "verification_source": "vendor"},
        } for profile in profiles]},
        # An agent's card takes the engine's verdict for the scope it answered (#1981); the account row reads the
        # windows themselves. The engine names no verdict for a spent window with no reset, and a block it names
        # carries the instant it ends: only a dated, still-future block refuses (docs/PASSIVE_QUOTA_READ.md).
        "quota": [{"subject": {"harness": "claude", "subject_id": profile},
                   "freshness": "fresh", "constraints": [example["constraint"]],
                   **({"availability": {"state": state, "resets_at": example["resetsAt"]}} if state else {})}
                  for profile, example, state in zip(profiles, examples, ("", "exhausted", "cooldown"))],
        "availability_scope": {"model": "fable", "requestKind": "text", "basis": "quota_block"},
        "quota_absences": [], "reads": {"catalog": "ok", "accounts": "ok", "quota": "ok"},
        "unified_accounts": True,
    }
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(settings_path.parent.parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(f"Date.now = () => Date.parse({json.dumps(fixture['now'])});")
            requests = []

            def status(route):
                requests.append(route.request.method)
                route.fulfill(json=snapshot)

            page.route("**/api/claudexor/status**", status)
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.click('[data-nav-page="settings"]')
            page.wait_for_selector(".settings-shell")
            open_settings(page, "accounts/claude")
            account = page.locator('.harness-account-row[data-profile="missing-reset"]')
            account.wait_for(state="visible")
            assert "100% used · availability not proven" in account.inner_text()
            assert "harness-exhausted" not in (account.get_attribute("class") or "")
            for profile, hours in (("future-reset", "2h"), ("cooldown", "3h")):
                limited = page.locator(f'.harness-account-row[data-profile="{profile}"]')
                assert "harness-exhausted" in (limited.get_attribute("class") or "")
                assert f"Limit reached · resets in {hours}" in limited.inner_text()
            page.locator('[data-settings-page="accounts/claude"]').screenshot(path=str(evidence / "accounts-quota.png"))
            for profile, expected in zip(profiles, ("quota availability not proven", "limit reached", "limit reached")):
                level = open_settings(page, f"agents/{profile}")
                expect(level.locator("[data-subagent-status]")).to_contain_text(expected)
                level.screenshot(path=str(evidence / f"subagent-quota-{profile}.png"))
            # The same two verdicts with no valid reset are no live block: those cards fall back to Not checked
            # instead of refusing on a verdict nothing can date.
            for row in snapshot["quota"][1:]:
                row["availability"].pop("resets_at")
            page.reload(wait_until="domcontentloaded")
            page.click('[data-nav-page="settings"]')
            page.wait_for_selector(".settings-shell")
            for profile in profiles:
                level = open_settings(page, f"agents/{profile}")
                expect(level.locator("[data-subagent-status]")).to_contain_text("quota availability not proven")
                level.screenshot(path=str(evidence / f"subagent-quota-undated-{profile}.png"))
            assert requests and set(requests) == {"GET"}
        finally:
            browser.close()
