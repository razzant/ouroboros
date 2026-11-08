"""Two named keys of one provider and no built-in key: Settings asks which key Main uses (#2969).

The page talks to the real handlers over a temp data dir (fixture keys, no network): ``GET`` and
``POST /api/settings`` and ``POST /api/owner/safety-model``, which Save writes first when Main
changes. The owner picks a key in Main's existing source select, the question goes away, Save
writes Main on that key through both handlers, and Light and Fallbacks the owner never set keep
following it. Saved or edited lanes, a blank and an unrelated Save stay exactly as the owner left them.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import pytest

from ouroboros.provider_models import CONNECTIONS_KEY
from tests import test_subscription_role_routes_browser as roles
from tests.test_connection_aware_defaults import HOME, WORK

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui
QUESTION = "Main is not set yet. Choose the key it uses: OpenAI · work or OpenAI · home."
CHOSEN = "openai@work::gpt-5.6-terra"
SLOTS = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")


@pytest.fixture
def real_settings(role_ui, tmp_path, monkeypatch):
    """``role_ui`` whose Settings reads and both Save writes reach the real gateway handlers.

    Every exchange is kept as ``(method, path, request body, status, response body)``."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    import server as srv
    from ouroboros import config as cfg
    from ouroboros.gateway import safety_settings, settings as gw

    environ = dict(os.environ)  # the handlers project what they save into this process's environment
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data_dir / "settings.json")
    monkeypatch.setattr(cfg, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(gw, "_apply_settings_save_side_effects", lambda *a, **kw: None)
    monkeypatch.setattr(gw, "_start_supervisor_if_needed_for_request", lambda *a, **kw: False)
    for key in (*cfg.SETTINGS_DEFAULTS, CONNECTIONS_KEY, "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    app = Starlette(routes=[
        Route("/api/settings", srv.api_settings_get, methods=["GET"]),
        Route("/api/settings", gw.api_settings_post, methods=["POST"]),
        Route("/api/owner/safety-model", safety_settings.api_owner_safety_model, methods=["POST"]),
    ])
    app.state.drive_root, app.state.repo_dir = data_dir, tmp_path
    exchanges = []

    def forward(route):
        request = route.request
        path = urlparse(request.url).path
        headers = {key: value for key, value in request.headers.items()
                   if key.lower() == "content-type" or key.lower().startswith("x-ouroboros-")}
        if request.method == "POST":
            response = client.post(path, content=request.post_data_buffer, headers=headers)
        else:
            response = client.get(path, headers=headers)
        exchanges.append((request.method, path, request.post_data_json if request.method == "POST" else None,
                          response.status_code, response.json()))
        route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    cfg.reset_runtime_mode_baseline_for_tests()
    try:
        with TestClient(app) as client:
            role_ui["page"].route("**/api/settings", forward)
            role_ui["page"].route("**/api/owner/safety-model", forward)
            yield {**role_ui, "data_dir": data_dir, "exchanges": exchanges, "client": client}
    finally:
        cfg.reset_runtime_mode_baseline_for_tests()
        os.environ.clear()
        os.environ.update(environ)


def _seed(ui, document):
    (ui["data_dir"] / "settings.json").write_text(json.dumps(document), encoding="utf-8")


def _disk(ui):
    return json.loads((ui["data_dir"] / "settings.json").read_text(encoding="utf-8"))


def _lanes(ui):
    """The model lanes the document on disk authors (the generic Save writes a whole document)."""
    return {key: value for key, value in _disk(ui).items() if key in SLOTS}


def _read(ui):
    return ui["client"].get("/api/settings").json()


def _open(ui, tab="models"):
    """Settings read afresh from the document on disk (a hash change alone would keep the loaded draft)."""
    page = ui["page"]
    if page.url.startswith(ui["url"]):
        ui["errors"].watch(page)  # the reload may cut off the earlier page's own reads
        page.reload()
    else:
        page.goto(ui["url"] + "/#settings")
    page.locator(f'[data-settings-tab="{tab}"]').click()
    return page


def _row(page, slot):
    return page.locator(f'[data-model-role-group="{slot}"]')


def _choose_main(page, source="api:openai@work", model="gpt-5.6-terra"):
    main = _row(page, "main")
    main.locator("select[data-model-role-source]").select_option(source)  # a new source clears the model field
    main.locator("[data-model-role-model]").fill(model)
    main.locator("h4").click()  # close the model suggestions
    return main


def _save(ui):
    """Press Save and wait for its writes and its re-read; returns this Save's POSTs."""
    page, before = ui["page"], len(ui["exchanges"])
    with page.expect_response(lambda r: r.url.endswith("/api/settings") and r.request.method == "POST"):
        page.locator("#btn-save-settings").click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled"
                           " && !/Saving/.test(document.querySelector('#settings-status')?.textContent || '')")
    return [exchange for exchange in ui["exchanges"][before:] if exchange[0] == "POST"]


def _follows(value, key="openai@work"):
    return bool(value) and all(entry.strip().startswith(f"{key}::") for entry in value.split(",") if entry.strip())


def test_the_owner_chooses_mains_key_in_the_existing_source_select(real_settings):
    from playwright.sync_api import expect

    ui = real_settings
    _seed(ui, {CONNECTIONS_KEY: [WORK, HOME]})
    page = _open(ui)
    main = _row(page, "main")
    status = main.locator("[data-model-role-status]")
    source = main.locator("select[data-model-role-source]")
    expect(status).to_have_text(QUESTION)
    expect(status).to_have_attribute("data-tone", "warn")
    assert source.locator('option[value="api:openai@work"]').count() == 1
    assert source.locator('option[value="api:openai@home"]').count() == 1
    for width in (1360, 390):  # the question fits a phone's width as well
        page.set_viewport_size({"width": width, "height": 900})
        for scheme in ("light", "dark"):
            page.emulate_media(color_scheme=scheme)
            main.scroll_into_view_if_needed()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            roles.capture(page, f"main-connection-choice-asked-{scheme}-{width}")
    page.set_viewport_size({"width": 1360, "height": 900})
    # Light, Safety and the rest say nothing of it: the question is Main's alone.
    assert page.locator("#settings-model-roles").inner_text().count("Choose the key") == 1

    _choose_main(page)
    expect(status).not_to_have_text(QUESTION)
    assert status.get_attribute("data-tone") is None
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        roles.capture(page, f"main-connection-choice-chosen-{scheme}")
    page.emulate_media(color_scheme="light")
    posts = _save(ui)
    assert [(path, status) for _m, path, _b, status, _r in posts] == [
        ("/api/owner/safety-model", 200), ("/api/settings", 200)], posts
    assert _lanes(ui) == {"OUROBOROS_MODEL": CHOSEN}

    after = _read(ui)
    assert after["_meta"]["main_connection_choice"] == []
    rows = json.loads(after["OUROBOROS_SUBAGENTS"])["items"]
    assert rows and all(row["route"]["target_id"].startswith("openai@work::") for row in rows)
    assert _follows(after["OUROBOROS_MODEL_LIGHT"]) and _follows(after["OUROBOROS_MODEL_FALLBACKS"])
    page.reload()
    page.locator('[data-settings-tab="models"]').click()
    expect(_row(page, "main").locator("[data-model-role-status]")).not_to_have_text(QUESTION)
    light = _row(page, "light")
    expect(light.locator("select[data-model-role-source]")).to_have_value("api:openai@work")
    expect(light.locator("[data-model-role-status]")).not_to_have_text("This source has no key in Accounts.")
    roles.capture(page, "main-connection-choice-saved-light")


SAFETY = "openai@home::gpt-5.6-luna"


@pytest.mark.parametrize("safety", ["inherited", "independent"])
def test_choosing_mains_key_leaves_unset_lanes_following_it_through_both_writes(real_settings, safety):
    """Save writes Main through the owner transaction first; the generic write after it must not
    turn the proposals the page showed before that choice into the owner's Light and Fallbacks."""
    from ouroboros.config import safety_configuration

    ui = real_settings
    document = {CONNECTIONS_KEY: [WORK, HOME], **({"OUROBOROS_MODEL_SAFETY": SAFETY} if safety == "independent" else {})}
    _seed(ui, document)
    page = _open(ui)
    _choose_main(page)
    posts = _save(ui)
    assert [status for *_rest, status, _r in posts] == [200, 200], posts
    assert "not saved" not in page.locator("#settings-status").inner_text()
    assert _lanes(ui) == {"OUROBOROS_MODEL": CHOSEN}, _disk(ui)
    disk = _disk(ui)
    assert disk.get("OUROBOROS_MODEL_SAFETY", "") == document.get("OUROBOROS_MODEL_SAFETY", "")
    after = _read(ui)
    assert _follows(after["OUROBOROS_MODEL_LIGHT"]) and _follows(after["OUROBOROS_MODEL_FALLBACKS"])
    checker = safety_configuration(disk)[0]
    assert checker == (SAFETY if safety == "independent" else after["OUROBOROS_MODEL_LIGHT"]), checker


def test_saved_and_edited_lanes_stay_exactly_as_the_owner_left_them(real_settings):
    ui = real_settings
    saved = {CONNECTIONS_KEY: [WORK, HOME], "OUROBOROS_MODEL_LIGHT": "openai@home::owner-light",
             "OUROBOROS_MODEL_FALLBACKS": ""}  # a saved Light on the other key and a saved blank chain
    _seed(ui, saved)
    page = _open(ui)
    _choose_main(page)
    assert [status for *_rest, status, _r in _save(ui)] == [200, 200]
    assert _lanes(ui) == {"OUROBOROS_MODEL_LIGHT": "openai@home::owner-light", "OUROBOROS_MODEL_FALLBACKS": "",
                          "OUROBOROS_MODEL": CHOSEN}, _lanes(ui)

    # An edit in the same Save is the owner's; the lane left alone still follows Main's key.
    _seed(ui, {CONNECTIONS_KEY: [WORK, HOME]})
    page = _open(ui)
    _choose_main(page)
    light = _row(page, "light")
    light.locator("select[data-model-role-source]").select_option("api:openai@home")
    light.locator("[data-model-role-model]").fill("owner-light")
    light.locator("h4").click()
    assert [status for *_rest, status, _r in _save(ui)] == [200, 200]
    assert _lanes(ui) == {"OUROBOROS_MODEL": CHOSEN, "OUROBOROS_MODEL_LIGHT": "openai@home::owner-light"}, _lanes(ui)
    assert _follows(_read(ui)["OUROBOROS_MODEL_FALLBACKS"])


@pytest.mark.parametrize("connections", [[WORK], [WORK, HOME]], ids=["sole-named", "two-named"])
def test_an_unrelated_save_authors_no_model_lane(real_settings, connections):
    ui = real_settings
    _seed(ui, {CONNECTIONS_KEY: connections})
    shown = {key: _read(ui)[key] for key in SLOTS}
    page = _open(ui, "advanced")
    page.fill("#s-tool-timeout", "601")
    posts = _save(ui)
    assert [(path, status) for _m, path, _b, status, _r in posts] == [("/api/settings", 200)], posts
    assert _lanes(ui) == {} and _disk(ui)["OUROBOROS_TOOL_TIMEOUT_SEC"] == 601, _disk(ui)
    after = _read(ui)
    assert {key: after[key] for key in SLOTS} == shown
    assert after["_meta"]["main_connection_choice"] == ([] if len(connections) == 1 else ["openai@work", "openai@home"])
