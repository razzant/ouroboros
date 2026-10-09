"""The subscription wizard proposes and writes only keys the runtime honors (V-D4-12).

``compile_model_settings`` walks the model roles, and one role still names the retired
``OUROBOROS_MODEL_DEEP_SELF_REVIEW``. A key the wizard writes and the next load drops comes
back to the owner as a "retired key ... NOT honored" chat notice for a setting they never set.
"""

from ouroboros.settings_defaults import RETIRED_SETTING_KEYS
from ouroboros.subscription_install_presets import compile_model_settings
from tests.test_onboarding_complete_endpoint import (
    LIVE_SNAPSHOT,
    onboarding as onboarding,  # explicit fixture re-export, not another settings writer
)

MODEL = "claudexor::codex=default"
CATALOG = [{"value": MODEL, "is_default": True, "input_modalities": ["text", "image"]}]
RETIRED_DEEP_MODEL = "OUROBOROS_MODEL_DEEP_SELF_REVIEW"
LIVE_ROLE_KEYS = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_VISION",
                  "OUROBOROS_MODEL_CONSCIOUSNESS")


def test_the_zero_key_proposal_names_no_retired_key_and_keeps_every_live_role():
    proposed = compile_model_settings(CATALOG, {})

    assert RETIRED_DEEP_MODEL not in proposed
    assert not set(proposed) & set(RETIRED_SETTING_KEYS)
    assert {key: proposed[key] for key in LIVE_ROLE_KEYS} == dict.fromkeys(LIVE_ROLE_KEYS, MODEL)
    assert proposed["OUROBOROS_MODEL_FALLBACKS"] == ""
    assert "OUROBOROS_WEBSEARCH_MODEL" not in proposed


def test_the_wizard_preview_and_the_written_document_carry_no_retired_key(onboarding):
    onboarding.calls["snapshot_payload"] = {**LIVE_SNAPSHOT, "model_catalog": CATALOG}
    preview = onboarding.client.post("/api/onboarding/subagents/preview",
                                     json={"subscriptionsConnected": True, "OUROBOROS_MODEL": ""})
    assert preview.status_code == 200, preview.text
    proposed = preview.json()["model_settings"]

    completed = onboarding.client.post("/api/onboarding/complete", json={"subscriptionsConnected": True})
    assert completed.status_code == 200, completed.text
    saved = onboarding.saved()

    for document in (saved, proposed):
        assert RETIRED_DEEP_MODEL not in document
        assert not set(document) & set(RETIRED_SETTING_KEYS)
        assert {key: document[key] for key in LIVE_ROLE_KEYS} == dict.fromkeys(LIVE_ROLE_KEYS, MODEL)
