"""Onboarding completion carries the starter skills and the two hub rows' trust in its one write."""

from __future__ import annotations

import json

import pytest

from tests.test_onboarding_complete_endpoint import WIZARD_PAYLOAD, onboarding  # noqa: F401  (fixture)


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


@pytest.mark.parametrize("choices", [
    {"skill_source_trust": {"ouroboros": "always"}},
    {"skill_source_trust": {"no-such-source": "trusted"}},
    {"starter_skills": "builtin/telegram"},
])
def test_malformed_starter_choices_save_nothing_and_start_nothing(onboarding, choices):  # noqa: F811
    response = onboarding.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, **choices})
    assert response.status_code == 400, response.text
    assert not onboarding.settings_path.exists() and "starter" not in onboarding.calls
