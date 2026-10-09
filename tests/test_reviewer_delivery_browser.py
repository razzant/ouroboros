"""Actual Settings renderer: a reviewer row's delivery survives Save and reload, and an
unrelated save through the real settings gateway writes no review lanes."""
import json
from urllib.parse import urlparse

import pytest
from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.mark.parametrize("width", [1360, 390])
def test_delivery_choice_saves_and_reloads(role_ui, width):
    ui = role_ui
    roles.configure_mixed(ui)
    roles.api_lane(ui, "openai")
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = roles.open_agents(ui)
    row = page.locator("[data-subagent-row]").nth(1)
    delivery = row.locator('[data-subagent-field="delivery"]')
    assert delivery.input_value() == "packet"
    delivery.select_option("native")
    row.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    roles.capture(page, f"reviewer-delivery-native-{width}")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    saved = roles.saved_catalog(ui)["items"][1]
    assert saved["review_eligible"] is True and "delivery" not in saved, "native is the default delivery"
    with page.expect_response("**/api/review-pool"):
        page.locator("#btn-reload-settings").click()
    delivery = page.locator("[data-subagent-row]").nth(1).locator('[data-subagent-field="delivery"]')
    assert delivery.input_value() == "native"
    delivery.select_option("packet")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    assert roles.saved_catalog(ui)["items"][1]["delivery"] == "packet"


@pytest.fixture
def settings_gateway(tmp_path, monkeypatch):
    """Real settings read/write and the review-pool read, isolated from runtime startup.
    A server error is a 500 response, as a served page would see it."""
    from types import SimpleNamespace
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as gateway

    data = tmp_path / "settings-data"
    data.mkdir()
    path = data / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    cfg.reset_runtime_mode_baseline_for_tests()

    def project(settings):
        environment = {}
        cfg.apply_settings_to_env(settings, environ=environment)
        for key, value in environment.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr(gateway, "_apply_settings_to_env", project)
    monkeypatch.setattr(gateway, "_start_supervisor_if_needed_for_request", lambda *a: False)
    monkeypatch.setattr(gateway, "_apply_settings_save_side_effects", lambda *a: [])
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: False)
    app = Starlette(routes=[
        Route("/api/settings", gateway.api_settings_get, methods=["GET"]),
        Route("/api/settings", gateway.api_settings_post, methods=["POST"]),
        Route("/api/review-pool", gateway.api_review_pool),
    ])
    app.state.drive_root = data
    app.state.repo_dir = tmp_path
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, path=path, project=project, cfg=cfg)
    cfg.reset_runtime_mode_baseline_for_tests()


_OWNER_ROW = {"subagent_id": "owner-row", "recommended_use": "Owner work.",
              "route": {"kind": "api_model", "target_id": "openai-compatible::glm-5.3"}}


def _serve_through(ui, gateway, writes):
    def endpoint(route):
        path = urlparse(route.request.url).path
        if route.request.method == "POST":
            writes.append(route.request.post_data_json)
            response = gateway.client.post(path, json=route.request.post_data_json)
        else:
            response = gateway.client.get(path)
        route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    ui["page"].route("**/api/settings", endpoint)
    ui["page"].route("**/api/review-pool", endpoint)
    # Keep discovery empty, and the generated proposal unavailable: the catalog on the
    # page is exactly the one the settings read showed.
    ui["fixture"]["status"].update(harnesses=[], profiles={}, model_sources=[])
    ui["fixture"]["catalog"].update(items=[], model_sources=[])
    ui["backend"]["preview_error"] = "No generated proposal in this test."


def _install(gateway, catalog=None):
    initial = {"OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_CONTEXT_MODE": "max",
               "OUROBOROS_MODEL": "openai-compatible::glm-5.3",
               "OPENAI_COMPATIBLE_BASE_URL": "https://llm.example/v1",
               "OPENAI_COMPATIBLE_API_KEY": "test-only-key"}
    if catalog is not None:
        initial["OUROBOROS_SUBAGENTS"] = json.dumps(catalog)
    gateway.path.write_text(json.dumps(initial), encoding="utf-8")
    gateway.project(gateway.cfg.load_settings())


def _stored(gateway):
    return json.loads(gateway.path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", ["fresh", "saved"])
def test_an_unrelated_save_re_posts_the_shown_catalog_and_writes_no_lanes(subscription_ui, settings_gateway, case):
    """Every Settings save re-posts the catalog: one saved before (no reviewer marked), or the
    unsaved candidate a fresh install is shown, whose factory review rows come marked (the
    read seam's never-configured cell). Re-posting either is no catalog change, so an unrelated
    save is never refused, not even for the saved catalog's empty pool, and no lanes key is written."""
    ui, gateway, writes = subscription_ui, settings_gateway, []
    _install(gateway, {"enabled": True, "items": [_OWNER_ROW]} if case == "saved" else None)
    shown = gateway.client.get("/api/settings").json()
    shown = (json.loads(shown["OUROBOROS_SUBAGENTS"]) if case == "saved"
             else shown["_meta"]["available_subagents"]["candidate"])
    marked = [row for row in shown["items"] if row.get("review_eligible")]
    if case == "saved":
        assert shown["items"] and not marked
    else:
        assert marked and all(row.get("minted_from") == "factory_default" for row in marked), marked
    before = gateway.path.read_bytes()
    _serve_through(ui, gateway, writes)
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="agents"]').click()
    page.wait_for_selector("[data-subagent-row]")
    assert page.locator("[data-review-pool-count]").inner_text() == f"Reviewers: {len(marked)}"
    assert page.locator("[data-review-pool-empty]").is_visible() is (not marked)
    assert gateway.path.read_bytes() == before, "both reads are passive"
    roles.capture(page, f"unrelated-save-{case}-before")
    page.locator('[data-settings-tab="behavior"]').click()
    with page.expect_response("**/api/settings") as response:
        page.locator("#btn-save-settings").click()
    assert response.value.status == 200, response.value.text()
    assert "not saved" not in page.locator("#settings-status").inner_text()
    assert "OUROBOROS_REVIEWER_SLOTS" not in writes[-1] and "allow_empty_review_pool" not in writes[-1]
    stored = _stored(gateway)
    assert "OUROBOROS_REVIEWER_SLOTS" not in stored
    assert [row["subagent_id"] for row in json.loads(stored["OUROBOROS_SUBAGENTS"])["items"]] == [
        row["subagent_id"] for row in shown["items"]]


def test_a_marked_row_round_trips_through_the_real_gateway(subscription_ui, settings_gateway):
    """The Reviewer box, the real save rules and the real pool read agree: the marked row is
    stored, the pool lists it, and its card states the review cost it can know."""
    from playwright.sync_api import expect

    ui, gateway, writes = subscription_ui, settings_gateway, []
    _install(gateway, {"enabled": True, "items": [_OWNER_ROW]})
    _serve_through(ui, gateway, writes)
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="agents"]').click()
    page.wait_for_selector("[data-subagent-row]")
    page.locator('[data-subagent-field="review_eligible"]').first.check()
    assert page.locator("[data-review-pool-count]").inner_text() == "Reviewers: 1"
    with page.expect_response("**/api/settings") as response:
        page.locator("#btn-save-settings").click()
    assert response.value.status == 200, response.value.text()
    assert json.loads(_stored(gateway)["OUROBOROS_SUBAGENTS"])["items"][0]["review_eligible"] is True
    pool = gateway.client.get("/api/review-pool").json()
    assert [row["subagent_id"] for row in pool["pool"]] == ["owner-row"] and not pool["config_error"]
    with page.expect_response("**/api/review-pool"):
        page.locator("#btn-reload-settings").click()
    facts = page.locator("[data-subagent-review-facts]").first
    # Save's own re-read and Reload each repaint the card before their pool read lands.
    expect(facts, "a custom endpoint has no known tariff; unknown is never zero").to_contain_text("cost unknown")
    expect(facts).to_contain_text("In the review pool")
    roles.capture(page, "reviewer-real-gateway-marked")
