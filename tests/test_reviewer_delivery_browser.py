"""Actual Settings renderer: a reviewer's delivery survives its page's Save and a reload, and through the real
settings gateway the Agents list reads passively, an unrelated row writes only its own key, and a Reviewer mark
round-trips."""
import json
from urllib.parse import urlparse

import pytest
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import commit_row, open_settings, press_editor_save, settings_page

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.mark.parametrize("width", [1360, 390])
def test_delivery_choice_saves_and_reloads(role_ui, width):
    ui = role_ui
    roles.configure_mixed(ui)
    roles.api_lane(ui, "openai")
    ui["page"].set_viewport_size({"width": width, "height": 900})
    row = roles.open_agent(ui, "direct")
    page = ui["page"]
    delivery = row.locator('[data-subagent-field="delivery"]')
    assert delivery.input_value() == "packet"
    delivery.select_option("native")
    delivery.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    roles.capture(page, f"reviewer-delivery-native-{width}")
    assert press_editor_save(page).startswith("Saved")
    saved = roles.saved_catalog(ui)["items"][1]
    assert saved["review_eligible"] is True and "delivery" not in saved, "native is the default delivery"
    # The page shows what is stored after its Save, and so does a fresh opening.
    assert row.locator('[data-subagent-field="delivery"]').input_value() == "native"
    delivery = roles.open_agent(ui, "direct").locator('[data-subagent-field="delivery"]')
    assert delivery.input_value() == "native"
    delivery.select_option("packet")
    assert press_editor_save(page).startswith("Saved")
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
    from ouroboros.gateway import safety_settings

    data = tmp_path / "settings-data"
    data.mkdir()
    path = data / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    cfg.reset_runtime_mode_baseline_for_tests()

    apply_settings_to_env = cfg.apply_settings_to_env

    def project(settings):
        environment = {}
        apply_settings_to_env(settings, environ=environment)
        for key, value in environment.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr(gateway, "_apply_settings_to_env", project)
    # The owner checker transaction projects through the config module itself.
    monkeypatch.setattr(cfg, "apply_settings_to_env", lambda settings, environ=None: (
        project(settings) if environ is None else apply_settings_to_env(settings, environ=environ)))
    monkeypatch.setattr(gateway, "_start_supervisor_if_needed_for_request", lambda *a: False)
    monkeypatch.setattr(gateway, "_apply_settings_save_side_effects", lambda *a: [])
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: False)
    app = Starlette(routes=[
        Route("/api/settings", gateway.api_settings_get, methods=["GET"]),
        Route("/api/settings", gateway.api_settings_post, methods=["POST"]),
        Route("/api/owner/safety-model", safety_settings.api_owner_safety_model, methods=["POST"]),
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
    ui["page"].route("**/api/owner/safety-model", endpoint)
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
def test_showing_the_list_is_passive_and_an_unrelated_row_writes_only_its_own_key(subscription_ui, settings_gateway, case):
    """The list shows a catalog saved before (no reviewer marked), or the unsaved candidate a fresh install is
    shown, whose factory review rows come marked (the read seam's never-configured cell). Showing it writes
    nothing; a row of the same page that is not the roster writes its one key, so the roster is neither authored
    nor judged by it (not even the saved catalog's empty pool), and no lanes key is written. The fresh catalog
    stays automatic (#1765); the saved one stays authored."""
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
    level = open_settings(page, "agents")
    level.locator("[data-entry-id]").first.wait_for(state="visible")
    assert [row.get_attribute("data-entry-id") for row in level.locator("[data-entry-id]").all()] == [
        row["subagent_id"] for row in shown["items"]]
    assert level.locator("[data-entry-id] .agent-chip", has_text="Reviewer").count() == len(marked)
    page.wait_for_function("marked => document.querySelector('#page-settings [data-review-pool-empty]')?.hidden === marked",
                           arg=bool(marked))
    assert gateway.path.read_bytes() == before and not writes, "both reads are passive"
    roles.capture(page, f"unrelated-save-{case}-before")
    level.locator("details[data-settings-fold] > summary").click()
    line = commit_row(page, "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", 4)
    assert line.startswith("Saved"), line
    assert writes == [{"OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT": 4}], "one key: the roster is not re-posted"
    stored = _stored(gateway)
    assert stored["OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"] == 4 and "OUROBOROS_REVIEWER_SLOTS" not in stored
    reread = gateway.client.get("/api/settings").json()
    assert reread["_meta"]["available_subagents"]["automatic"] is (case == "fresh")
    if case == "fresh":
        assert "OUROBOROS_SUBAGENTS" not in stored, "an unrelated row does not author the automatic catalog"
        catalog = reread["_meta"]["available_subagents"]["candidate"]
    else:
        catalog = json.loads(stored["OUROBOROS_SUBAGENTS"])
    assert [row["subagent_id"] for row in catalog["items"]] == [row["subagent_id"] for row in shown["items"]]


def test_a_marked_row_round_trips_through_the_real_gateway(subscription_ui, settings_gateway):
    """The Reviewer switch, the real save rules and the real pool read agree: the mark is written at once through
    the owner transaction, the pool lists the row, and its page states the review cost it can know."""
    from playwright.sync_api import expect

    ui, gateway, writes = subscription_ui, settings_gateway, []
    _install(gateway, {"enabled": True, "items": [_OWNER_ROW]})
    _serve_through(ui, gateway, writes)
    row = roles.open_agent(ui, "owner-row")
    page = ui["page"]
    row.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-subagent-field="review_eligible"]');
        return input && input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    assert len(writes) == 1 and list(writes[0]["settings"]) == ["OUROBOROS_SUBAGENTS"], "one owner transaction, the roster alone"
    assert json.loads(_stored(gateway)["OUROBOROS_SUBAGENTS"])["items"][0]["review_eligible"] is True
    pool = gateway.client.get("/api/review-pool").json()
    assert [entry["subagent_id"] for entry in pool["pool"]] == ["owner-row"] and not pool["config_error"]
    facts = settings_page(page, "agents/owner-row").locator("[data-subagent-review-facts]")
    # The write's own re-read repaints the page before its pool read lands.
    expect(facts, "a custom endpoint has no known tariff; unknown is never zero").to_have_text("cost unknown")
    expect(settings_page(page, "agents/owner-row").locator("[data-subagent-delivery-cost]")).to_have_text("cost unknown")
    open_settings(page, "agents")
    assert settings_page(page, "agents").locator('[data-agents-filter="reviewers"]').inner_text().strip() == "Reviewers 1"
    roles.capture(page, "reviewer-real-gateway-marked")
