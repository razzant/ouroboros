"""Owner settings surfaces preserve ordinary provider and roster decisions."""
import copy
import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests._safety_fixtures import settings_install as install_fixture
from tests.test_safety_second_check import catalog

install = install_fixture


@pytest.fixture
def settings_client(install, monkeypatch):
    from ouroboros import config
    from ouroboros.gateway import safety_settings, settings

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")
    monkeypatch.setattr(settings, "_apply_settings_save_side_effects", lambda *a, **kw: None)
    monkeypatch.setattr(settings, "_start_supervisor_if_needed_for_request", lambda *a, **kw: False)
    install.write_text(json.dumps({"OPENROUTER_API_KEY": "fixture-router", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full"}))
    app = Starlette(routes=[
        Route("/api/settings", settings.api_settings_post, methods=["POST"]),
        Route("/api/owner/safety-model", safety_settings.api_owner_safety_model, methods=["POST"]),
    ])
    app.state.drive_root = app.state.repo_dir = install.parent
    with TestClient(app) as client:
        yield client, install


@pytest.mark.parametrize("mode", ["light", "advanced", "pro"])
@pytest.mark.parametrize("resend_shown", [False, True])
def test_provider_switch_reproposes_checker_but_explicit_edit_is_refused(settings_client, monkeypatch, mode, resend_shown):
    from ouroboros import config
    from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

    client, path = settings_client
    monkeypatch.setattr(config, "get_runtime_mode", lambda: mode)
    body = config.load_settings() if resend_shown else {}
    body.update(OPENROUTER_API_KEY="", OPENAI_API_KEY="fixture-openai")
    before = path.read_bytes()
    # Even explicitly choosing the new provider's proposal is an authored edit.
    denied = client.post("/api/settings", json={**body, "OUROBOROS_MODEL_LIGHT": OPENAI_DIRECT_DEFAULTS["light"]})
    assert denied.status_code == 400 and "owner-controlled" in denied.json()["error"]
    assert path.read_bytes() == before
    allowed = client.post("/api/settings", json=body)
    assert allowed.status_code == 200, allowed.json()
    stored = json.loads(path.read_text())
    assert stored["OPENAI_API_KEY"] == "fixture-openai" and not stored["OPENROUTER_API_KEY"]
    assert "OUROBOROS_MODEL_LIGHT" not in stored
    assert config.safety_configuration(stored)[0] == OPENAI_DIRECT_DEFAULTS["light"]
    events = [json.loads(line) for line in (path.parent / "logs" / "events.jsonl").read_text().splitlines()]
    event = next(row for row in events if row.get("action") == "safety_settings")
    assert event["checker"] == OPENAI_DIRECT_DEFAULTS["light"]
    # The proposal follows access back as well; provider spelling is not an owner choice.
    restored = client.post("/api/settings", json={"OPENAI_API_KEY": "", "OPENROUTER_API_KEY": "fixture-router"})
    assert restored.status_code == 200, restored.json()
    assert config.safety_configuration(json.loads(path.read_text()))[0] == config.safety_configuration(json.loads(before))[0]


def test_catalog_owner_endpoint_uses_same_roster_judgement(settings_client):
    client, path = settings_client
    valid = catalog(review_eligible=True)
    invalid = copy.deepcopy(valid)
    invalid["items"].append({**invalid["items"][0], "subagent_id": "twin", "rechecks_flags": False, "review_eligible": False})
    before = path.read_bytes()
    generic = client.post("/api/settings", json={"OUROBOROS_SUBAGENTS": invalid})
    owner = client.post("/api/owner/safety-model", json={"settings": {"OUROBOROS_SUBAGENTS": invalid}})
    assert generic.status_code == owner.status_code == 400
    assert "same engine" in generic.json()["error"]
    assert owner.json()["error"] == generic.json()["error"]
    assert path.read_bytes() == before
    accepted = client.post("/api/owner/safety-model", json={"settings": {"OUROBOROS_SUBAGENTS": valid}})
    assert accepted.status_code == 200 and accepted.json() == {"ok": True}
    stored = json.loads(json.loads(path.read_text())["OUROBOROS_SUBAGENTS"])
    assert stored["items"][0]["rechecks_flags"] is True


@pytest.mark.parametrize("unconfirmed", [None, False, "true"])
def test_empty_review_pool_needs_same_explicit_confirmation(settings_client, unconfirmed):
    client, path = settings_client
    empty = catalog(review_eligible=False)
    generic = client.post("/api/settings", json={"OUROBOROS_SUBAGENTS": empty})
    body = {"settings": {"OUROBOROS_SUBAGENTS": empty}, "allow_empty_review_pool": unconfirmed}
    before = path.read_bytes()
    owner = client.post("/api/owner/safety-model", json=body)
    assert owner.status_code == generic.status_code == 400
    assert owner.json()["error"] == generic.json()["error"]
    assert path.read_bytes() == before
    accepted = client.post("/api/owner/safety-model", json={**body, "allow_empty_review_pool": True})
    assert accepted.status_code == 200 and accepted.json() == {"ok": True}
    assert "allow_empty_review_pool" not in json.loads(path.read_text())
    from ouroboros.configured_subagents import parse_configured_subagents
    assert not parse_configured_subagents(json.loads(path.read_text())["OUROBOROS_SUBAGENTS"]).items[0].review_eligible
    # Reposting the unchanged catalog is not another empty-pool decision.
    repeated = client.post("/api/owner/safety-model", json={"settings": body["settings"]})
    assert repeated.status_code == 200 and repeated.json() == {"ok": True}


def test_operator_checker_is_not_replaced_by_generic_save_of_the_disk_default(settings_client, monkeypatch):
    from ouroboros import config

    client, path = settings_client
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "operator/checker")
    assert config.load_settings()["OUROBOROS_MODEL_LIGHT"] == "operator/checker"
    before = path.read_bytes()
    denied = client.post("/api/settings", json={"OUROBOROS_MODEL_LIGHT": config.SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]})
    assert denied.status_code == 400 and "owner-controlled" in denied.json()["error"]
    assert path.read_bytes() == before
    allowed = client.post("/api/settings", json={"OUROBOROS_MODEL_HEAVY": "unrelated"})
    assert allowed.status_code == 200, allowed.json()
    assert config.load_settings()["OUROBOROS_MODEL_LIGHT"] == "operator/checker"


def test_owner_roster_save_retires_legacy_lanes_only_with_catalog(settings_client):
    client, path = settings_client
    path.write_text(json.dumps({**json.loads(path.read_text()), "OUROBOROS_REVIEWER_SLOTS": "invalid legacy lanes"}))
    model_only = client.post("/api/owner/safety-model", json={"settings": {"OUROBOROS_MODEL_LIGHT": "fixture/checker"}})
    assert model_only.status_code == 200, model_only.json()
    assert json.loads(path.read_text())["OUROBOROS_REVIEWER_SLOTS"] == "invalid legacy lanes"
    roster = client.post("/api/owner/safety-model", json={"settings": {"OUROBOROS_SUBAGENTS": catalog(review_eligible=True)}})
    assert roster.status_code == 200 and roster.json() == {"ok": True}
    assert "OUROBOROS_REVIEWER_SLOTS" not in json.loads(path.read_text())


@pytest.mark.parametrize("remote", [False, True])
def test_owner_checker_save_preserves_startup_ready_route(settings_client, remote):
    client, path = settings_client
    path.write_text(json.dumps({
        "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full", "LOCAL_MODEL_SOURCE": "fixture",
        "OPENROUTER_API_KEY": "fixture-router" if remote else "",
        "USE_LOCAL_MAIN": True, "USE_LOCAL_LIGHT": False, "USE_LOCAL_FALLBACK": False,
    }))
    before = path.read_bytes()
    response = client.post("/api/owner/safety-model", json={"settings": {"USE_LOCAL_MAIN": False}})
    if remote:
        assert response.status_code == 200 and response.json() == {"ok": True}
        assert json.loads(path.read_text())["USE_LOCAL_MAIN"] is False
    else:
        assert response.status_code == 400
        assert response.json()["error"] == "Local-only setups must route at least one model to the local runtime."
        assert path.read_bytes() == before
    allowed = client.post("/api/owner/safety-model", json={"settings": {"USE_LOCAL_MAIN": False, "USE_LOCAL_LIGHT": True}})
    assert allowed.status_code == 200 and allowed.json() == {"ok": True}
    stored = json.loads(path.read_text())
    assert stored["USE_LOCAL_MAIN"] is False and stored["USE_LOCAL_LIGHT"] is True


def test_owner_checker_readiness_counts_an_environment_provider(settings_client, monkeypatch):
    from ouroboros import config
    from ouroboros.server_runtime import has_startup_ready_provider

    client, path = settings_client
    path.write_text(json.dumps({"OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full",
                               "LOCAL_MODEL_SOURCE": "fixture", "USE_LOCAL_MAIN": True,
                               "USE_LOCAL_LIGHT": False, "USE_LOCAL_FALLBACK": False}))
    body = {"settings": {"USE_LOCAL_MAIN": False}}
    before = path.read_bytes()
    assert client.post("/api/owner/safety-model", json=body).status_code == 400
    assert path.read_bytes() == before
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-environment-provider")
    assert has_startup_ready_provider({**config.load_settings(), **body["settings"]})
    allowed = client.post("/api/owner/safety-model", json=body)
    assert allowed.status_code == 200 and allowed.json() == {"ok": True}
    stored = json.loads(path.read_text())
    assert stored["USE_LOCAL_MAIN"] is False
    assert stored.get("OPENROUTER_API_KEY") != "fixture-environment-provider"
