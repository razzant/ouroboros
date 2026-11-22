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
