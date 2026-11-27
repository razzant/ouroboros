"""Onboarding completion carries the starter skills and the two hub rows' trust in its one write; the setup offers
its Skills step only where that completion honors the ticks."""

from __future__ import annotations

import json

import pytest

from tests.test_onboarding_complete_endpoint import WIZARD_PAYLOAD, onboarding  # noqa: F401  (fixture)
from tests.test_onboarding_consciousness_offer import _client, _tree


def test_starter_choices_and_hub_trust_join_the_one_completion_write(onboarding):  # noqa: F811
    trust = {"ouroboros": "trusted", "ouroboros-community": "review"}
    response = onboarding.client.post("/api/onboarding/complete", json={
        **WIZARD_PAYLOAD, "starter_skills": ["builtin/telegram", "ouroboros/duckduckgo"],
        "skill_source_trust": trust})
    assert response.status_code == 200, response.text
    rows = {row["id"]: row for row in json.loads(onboarding.saved()["OUROBOROS_SKILL_SOURCES"])}
    assert rows["ouroboros"]["confirmed"] is True and rows["ouroboros-community"]["confirmed"] is True
    assert rows["ouroboros-community"]["trust"] == "review" and rows["builtin"]["trust"] == "trusted"
    assert [ids for _drive, ids in onboarding.calls["starter"]] == [["builtin/telegram", "ouroboros/duckduckgo"]]


@pytest.mark.parametrize("body, handed", [({"starter_skills": []}, []), ({}, None)], ids=["all-cleared", "no-choice"])
def test_cleared_ticks_are_an_explicit_empty_choice_and_no_choice_keeps_the_defaults(onboarding, body, handed):  # noqa: F811
    """Untick all on the Skills step sends ``[]``: the installer is handed nothing to install. A page that offers no
    choice sends no field: the installer resolves the recommended list itself (``after_onboarding(root, None)``)."""
    response = onboarding.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, **body})
    assert response.status_code == 200, response.text
    assert [ids for _drive, ids in onboarding.calls["starter"]] == [handed]


@pytest.mark.parametrize("source", ["cleared", "projection-only", "operator"])
def test_start_uses_current_endpoint_metadata_like_the_bootstrap(onboarding, monkeypatch, source):  # noqa: F811
    """The legacy address affects the voice default; the keyless address exercises first-run eligibility too."""
    from ouroboros import config
    from ouroboros.gateway import models
    from ouroboros.settings_integrity import PROJECTED_SETTINGS_ENV

    fields = {"OPENAI_BASE_URL": "https://legacy.example.invalid/v1",
              "OPENAI_COMPATIBLE_BASE_URL": "https://dedicated.example.invalid/v1"}
    operator = source == "operator"
    if operator:
        monkeypatch.delenv(PROJECTED_SETTINGS_ENV, raising=False)
    else:
        projected = {}
        config.apply_settings_to_env(fields, environ=projected)
        monkeypatch.setenv(PROJECTED_SETTINGS_ENV, projected[PROJECTED_SETTINGS_ENV])
    for key, value in fields.items():
        monkeypatch.setenv(key, value)
    if source == "cleared":
        onboarding.settings_path.write_text(json.dumps(dict.fromkeys(fields, "")), encoding="utf-8")

    catalog_reads = []

    async def fixture_catalog(_client, provider_id, _label, _key, base_url, **_options):
        catalog_reads.append(base_url)
        return [models._build_model_catalog_entry(provider_id, "Fixture", "fixture-model", "Fixture model")]

    monkeypatch.setattr(models, "_fetch_openai_compatible_model_catalog", fixture_catalog)
    before = onboarding.saved()
    with _client(onboarding.settings_path.parent) as client:
        setup = client.get("/api/onboarding/bootstrap").json()
        state = setup["bootstrap"]["initialState"]
        catalog = client.post("/api/onboarding/model-catalog", params={"provider_id": "openai-compatible"},
                              json={"OPENAI_COMPATIBLE_BASE_URL": state["compatibleBaseUrl"]})
    assert setup["ready"] is operator
    assert state["legacyCompatibleBaseUrl"] == (fields["OPENAI_BASE_URL"] if operator else "")
    assert state["compatibleBaseUrl"] == (fields["OPENAI_COMPATIBLE_BASE_URL"] if operator else "")
    assert catalog.status_code == 200 and bool(catalog.json()["items"]) is operator
    assert catalog_reads == ([fields["OPENAI_COMPATIBLE_BASE_URL"]] if operator else [])

    # A local draft, no voice/key input, and no explicit starter list while the hub is unread.
    draft = {"LOCAL_MODEL_SOURCE": "/models/fixture.gguf", "LOCAL_ROUTING_MODE": "all",
             "OUROBOROS_MODEL": "fixture-main", "OPENAI_COMPATIBLE_BASE_URL": state["compatibleBaseUrl"]}
    assert onboarding.client.post("/api/onboarding/subagents/preview", json=draft).status_code == 200
    assert onboarding.saved() == before
    finished = onboarding.client.post("/api/onboarding/complete", json=draft)
    assert finished.status_code == 200
    assert finished.json()["preset"]["reason"] == ("applied" if source == "projection-only" else "not_install_time")
    assert {key: onboarding.saved().get(key, "") for key in fields} == (fields if operator else dict.fromkeys(fields, ""))
    assert [ids for _drive, ids in onboarding.calls["starter"]] == [None]
    assert onboarding.calls["snapshot"] == 0


def test_untouched_starter_offer_and_saved_default_select_the_same_skills(onboarding, monkeypatch):  # noqa: F811
    from ouroboros import skill_source_catalog, skill_starter
    from ouroboros.gateway import onboarding as routes

    source = {"id": "distribution", "kind": "builtin", "trust": "review", "managed": True}
    voice = {"name": "fixture-voice", "digest": "synthetic", "needs_keys": ["OPENAI_API_KEY"],
             "permissions": ["realtime_provider"]}
    snapshot = {"entries": [voice], "official": {voice["name"]: {"digest": voice["digest"], "recommended": True}}}
    monkeypatch.setattr(skill_source_catalog, "snapshot_for", lambda *_a, **_kw: (snapshot, {"state": "ready"}))
    monkeypatch.setattr(skill_source_catalog, "refresh_source", lambda *_a, **_kw: pytest.fail("only a bundled source"))
    monkeypatch.setattr(skill_starter, "_install_one", lambda *_a: {"status": "installed"})
    monkeypatch.setattr(skill_starter, "schedule_starter", skill_starter.run_starter)
    monkeypatch.setattr(routes, "after_onboarding", skill_starter.after_onboarding)
    onboarding.settings_path.write_text(json.dumps({"OUROBOROS_SKILL_SOURCES": json.dumps([source]),
                                                    "OPENAI_API_KEY": "fixture-key-for-onboarding"}), encoding="utf-8")

    root = onboarding.settings_path.parent
    with _client(root) as client:
        shown = client.get("/api/onboarding/bootstrap").json()["bootstrap"]["initialState"]
        offer = client.get("/api/onboarding/starter-skills").json()
    [voice_item] = [item for group in offer["groups"] for item in group["items"]]
    assert voice_item["recommended"] and voice_item["default_with_keys"] == {"OPENAI_API_KEY": True}
    response = onboarding.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, "OPENAI_API_KEY": shown["openaiKey"]})
    assert response.status_code == 200, response.text
    assert skill_starter.read_marker(root)["requested"] == [voice_item["id"]] == ["distribution/fixture-voice"]
    assert onboarding.saved()["OPENAI_API_KEY"] == "fixture-key-for-onboarding"


@pytest.mark.parametrize("choices", [
    {"skill_source_trust": {"ouroboros": "always"}},
    {"skill_source_trust": {"no-such-source": "trusted"}},
    {"starter_skills": "builtin/telegram"},
])
def test_malformed_starter_choices_save_nothing_and_start_nothing(onboarding, choices):  # noqa: F811
    response = onboarding.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, **choices})
    assert response.status_code == 400, response.text
    assert not onboarding.settings_path.exists() and "starter" not in onboarding.calls


@pytest.mark.parametrize("state, offered", [("", True), ("fresh", True), ("done", False), ("pending", False),
                                             ("running", False), ("upgrade", False)])
def test_the_skills_step_is_offered_only_where_completion_honors_the_ticks(monkeypatch, tmp_path, state, offered):
    """A returning install (its starter list ran, or it upgraded) sees no Skills step, so no tick is shown that Finish
    would ignore; a new data root sees it between Consciousness and Review limits. Serving the page writes nothing."""
    from ouroboros.gateway import onboarding_host
    from ouroboros.skill_starter import marker_path

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    if state:
        marker_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        marker_path(tmp_path).write_text(json.dumps({"state": state}), encoding="utf-8")
    seeded = _tree(tmp_path)
    with _client(tmp_path) as client:
        bootstrap = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
        page = client.get("/onboarding")
    assert ("skills" in bootstrap["stepOrder"]) is offered, bootstrap["stepOrder"]
    if offered:
        assert bootstrap["stepOrder"].index("skills") == bootstrap["stepOrder"].index("budget") - 1
    assert page.status_code == 200 and ('"skills"' in page.text.split('"stepOrder"', 1)[1].split("]", 1)[0]) is offered
    assert _tree(tmp_path) == seeded, "serving the step classified the root or wrote a marker"
