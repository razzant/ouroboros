"""#2574: the wizard's Test and model discovery keep a saved compatible key for one endpoint identity.

The real server, wizard, Provider Test and discovery proxy over an isolated root: the saved address spelled
another equivalent way keeps its key on both routes; another path is refused on both, and no request
carrying the key reaches it.
"""
from __future__ import annotations

import json

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_onboarding_legacy_endpoint_browser import _endpoint, _frame, _listed
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser]

SAVED_KEY = "sk-dedicated-pair-0123456789"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_wizard_test_and_discovery_share_one_endpoint_identity(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    ui = direct_server_with_data
    evidence = diagnostic_directory(tmp_path / "frames", f"endpoint-identity-{engine}", publish=True)
    report = {"candidate_identity": ui["candidate_identity"], "frames": {}}
    with _endpoint(["listed-model"]) as (url, calls), sync_playwright() as pw:
        saved_url = url.replace("127.0.0.1", "localhost")
        ui["stop_server"]()  # seed only while stopped
        path = ui["data_dir"] / "settings.json"
        settings = json.loads(path.read_text(encoding="utf-8"))
        settings.update({"OPENAI_COMPATIBLE_BASE_URL": saved_url, "OPENAI_COMPATIBLE_API_KEY": SAVED_KEY,
                         **dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"),
                                         "openai-compatible::listed-model")})
        path.write_text(json.dumps(settings), encoding="utf-8")
        ui["start_server"]()
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_context(viewport={"width": 980, "height": 780}, device_scale_factor=2).new_page()
        uncaught = []
        page.on("pageerror", lambda error: uncaught.append(str(error)))
        page.goto(ui["url"] + "/onboarding?builtin=1", wait_until="networkidle")
        row = page.locator('[data-provider-card="compatible"] [data-connection-row][data-builtin="1"]')
        base, result = row.locator('[data-connection-field="base_url"]'), row.locator('[data-connection-test-status]')
        expect(base).to_have_value(saved_url)
        for name, spelling, same in (("equivalent", saved_url.replace("localhost", "LOCALHOST") + "/", True),
                                     ("other-path", saved_url[:-len("/v1")] + "/v2", False)):
            base.fill(spelling)
            before = len(calls)
            with page.expect_response('**/api/providers/test') as tested:
                row.locator('[data-connection-test]').click()
            # The key stays on the server: only the edited address travels.
            assert tested.value.request.post_data_json == {
                "provider_id": "openai-compatible", "api_key_setting": "OPENAI_COMPATIBLE_API_KEY",
                "overrides": {"OPENAI_COMPATIBLE_BASE_URL": spelling}}
            expect(result).to_have_text('Works' if same else
                                        "Not ready — This server's address changed since its key was saved. "
                                        "Enter its key again to test the new address.")
            test_status = result.inner_text()
            row.scroll_into_view_if_needed()
            report["frames"][f"test-{name}"] = _frame(page, evidence, f"test-{name}", row)
            page.click('#next-btn')
            with page.expect_response('**/api/openai-compatible/models') as discovery:
                page.click('#load-compatible-models')
            assert discovery.value.request.post_data_json == {"baseUrl": spelling, "apiKey": "***set***",
                                                               "apiKeySetting": "OPENAI_COMPATIBLE_API_KEY"}
            assert discovery.value.status == (200 if same else 400), discovery.value.text()
            status = page.locator('#compatible-load-status')
            expect(status).to_contain_text('1 model listed' if same else "This server's address changed")
            report["frames"][f"discovery-{name}"] = _frame(page, evidence, f"discovery-{name}",
                                                           page.locator('#compatible-model-loader'))
            sent = calls[before:]
            report[name] = {"spelling": spelling, "test_status": test_status,
                            "discovery_status": status.inner_text(), "upstream": sent}
            if same:  # the saved key reached the saved server through both routes
                assert {(call["method"], call["path"]) for call in sent} >= {("POST", "/v1/chat/completions"), ("GET", "/v1/models")}
                assert {call["authorization"] for call in sent} == {f"Bearer {SAVED_KEY}"}
            else:  # nothing went to the other path, with the key or without it
                assert not [call for call in sent if call["path"].startswith("/v2")]
            page.click('#back-btn')
        report.update(listed=_listed(calls), uncaught_page_errors=uncaught)
        (evidence / "endpoint-identity-proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert all(not frame["overflow"] for frame in report["frames"].values())
        assert not uncaught
        browser.close()
