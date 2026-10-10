"""Model-source onboarding uses the same atomic transaction fixtures as API setup."""

import json

import pytest

from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY
from tests.test_onboarding_complete_endpoint import (
    LIVE_SNAPSHOT, WIZARD_PAYLOAD, _profile, _profile_account,
    onboarding as onboarding,  # explicit fixture re-export, not another settings writer
)

FACTORY_ROWS = [
    {"subagent_id": f"review-{n}", "recommended_use": "Factory reviewer.", "effort": effort,
     "route": {"kind": "api_model", "target_id": model}, "review_eligible": True, "minted_from": "factory_default"}
    for n, model, effort in ((1, "google/gemini-3.8-flash", "high"), (2, "openai/gpt-5.6-luna", "medium"),
                             (3, "deepseek/deepseek-v4-pro", "low"))
]
OWNER_DRAFT = {"enabled": True, "items": [{
    "subagent_id": "helper", "recommended_use": "Owner helper.",
    "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"},
}]}


@pytest.fixture
def pool_seams(monkeypatch):
    """The factory reviewer rows as a controlled double (fixed rows that are NOT the
    provider sequence, so the gateway is seen to carry the factory's output rather than
    its own), and package A's real empty-pool judge behind a call log."""
    from ouroboros import reviewer_slot_config, subscription_install_presets

    seen = {"factory": [], "judged": []}
    real_judge = reviewer_slot_config.review_pool_save_error

    def factory(doc):
        seen["factory"].append(dict(doc))
        return [json.loads(json.dumps(row)) for row in FACTORY_ROWS]

    def judge(raw, *, allow_empty):
        seen["judged"].append(raw)
        return real_judge(raw, allow_empty=allow_empty)

    monkeypatch.setattr(subscription_install_presets, "factory_review_rows", factory)
    monkeypatch.setattr(reviewer_slot_config, "review_pool_save_error", judge)
    return seen


def _ids(catalog):
    return [row["subagent_id"] for row in catalog["items"]]


def test_an_owner_draft_the_pool_rule_refuses_writes_nothing_until_confirmed(onboarding, pool_seams):
    """When package A's rule refuses the pool of the visible draft, the WHOLE onboarding
    write is refused; ``allow_empty_review_pool`` is the owner's confirmation and rides
    on the request only, never into the document."""
    from ouroboros.reviewer_slot_config import review_pool_save_error

    verdict = review_pool_save_error(json.dumps(OWNER_DRAFT), allow_empty=False)
    assert verdict, "an unmarked draft is the rule's refusal case"
    refused = onboarding.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, "OUROBOROS_SUBAGENTS": OWNER_DRAFT})
    assert refused.status_code == 400, refused.text
    assert refused.json()["code"] == "empty_review_pool" and refused.json()["saved"] is False
    assert refused.json()["error"] == verdict
    assert not onboarding.settings_path.exists()
    assert _ids(json.loads(pool_seams["judged"][-1])) == ["helper"]

    judged = len(pool_seams["judged"])
    confirmed = onboarding.client.post("/api/onboarding/complete", json={
        **WIZARD_PAYLOAD, "OUROBOROS_SUBAGENTS": OWNER_DRAFT, "allow_empty_review_pool": True})
    assert confirmed.status_code == 200, confirmed.text
    saved = onboarding.saved()
    assert _ids(json.loads(saved["OUROBOROS_SUBAGENTS"])) == ["helper"]
    assert not {key for key in saved if key.lower() == "allow_empty_review_pool"}
    assert not saved.get("OUROBOROS_REVIEWER_SLOTS")
    assert len(pool_seams["judged"]) == judged, "a confirmed write is not judged"


def test_a_generated_preview_proposes_the_factory_reviewers_and_an_owner_draft_is_never_topped_up(
        onboarding, pool_seams):
    response = onboarding.client.post("/api/onboarding/subagents/preview", json=WIZARD_PAYLOAD)
    assert response.status_code == 200, response.text
    proposed = response.json()
    assert "reviewer_slots" not in proposed
    assert [(row["subagent_id"], row["route"]["target_id"]) for row in proposed["available_subagents"]["items"][-3:]] == [
        (row["subagent_id"], row["route"]["target_id"]) for row in FACTORY_ROWS]
    (doc,) = pool_seams["factory"]
    assert doc["OPENROUTER_API_KEY"] and "OUROBOROS_SUBAGENTS" in doc, "the factory reads THIS document"
    assert not onboarding.settings_path.exists()

    owner = onboarding.client.post("/api/onboarding/subagents/preview",
                                   json={**WIZARD_PAYLOAD, "OUROBOROS_SUBAGENTS": OWNER_DRAFT})
    assert owner.status_code == 200, owner.text
    assert _ids(owner.json()["available_subagents"]) == ["helper"]
    assert len(pool_seams["factory"]) == 1


def test_a_generated_completion_saves_the_factory_reviewers_the_preview_proposed(onboarding, pool_seams):
    """A fresh install finishing without posting a catalog (an API or desktop caller) saves the
    reviewers the preview proposed, so it never lands with an empty pool; the receipt describes
    the saved bytes, so the catalog still reads as the onboarding default."""
    from ouroboros.configured_subagents import SUBAGENTS_RECEIPT_KEY, resolve_configured_subagents

    response = onboarding.client.post("/api/onboarding/complete", json=WIZARD_PAYLOAD)
    assert response.status_code == 200, response.text
    saved = onboarding.saved()
    catalog = json.loads(saved["OUROBOROS_SUBAGENTS"])
    assert _ids(catalog)[-3:] == [row["subagent_id"] for row in FACTORY_ROWS]
    assert pool_seams["judged"] == [saved["OUROBOROS_SUBAGENTS"]], "the pool rule judged the saved bytes"
    receipt = json.loads(saved[SUBAGENTS_RECEIPT_KEY])
    assert receipt["available_subagents"] == catalog
    assert receipt["review_pool"] == [row["subagent_id"] for row in catalog["items"] if row.get("review_eligible") is True]
    assert resolve_configured_subagents(saved).source == "onboarding_default"


def test_finishing_with_the_previewed_catalog_saves_exactly_its_reviewers(onboarding):
    """The wizard posts back the catalog the preview showed, marks included. The completion must
    save those reviewers once: re-marking an already-marked draft would mint a twin of every
    subscription reviewer, so each model would silently review every change twice."""
    preview = onboarding.client.post(
        "/api/onboarding/subagents/preview", json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})
    assert preview.status_code == 200, preview.text
    shown = preview.json()["available_subagents"]
    assert any(row.get("review_eligible") is True for row in shown["items"])
    response = onboarding.client.post("/api/onboarding/complete", json={
        **WIZARD_PAYLOAD, "subscriptionsConnected": True, "OUROBOROS_SUBAGENTS": shown})
    assert response.status_code == 200, response.text
    assert json.loads(onboarding.saved()["OUROBOROS_SUBAGENTS"]) == shown


def test_a_confirmed_empty_pool_gains_no_subscription_reviewers(onboarding):
    """'Save without reviewers' is the owner's answer: neither the preview nor the completion
    appends the connected subscriptions' reviewers to the confirmed draft."""
    body = {**WIZARD_PAYLOAD, "subscriptionsConnected": True, "OUROBOROS_SUBAGENTS": OWNER_DRAFT,
            "allow_empty_review_pool": True}
    preview = onboarding.client.post("/api/onboarding/subagents/preview", json=body)
    assert preview.status_code == 200, preview.text
    assert _ids(preview.json()["available_subagents"]) == ["helper"]
    response = onboarding.client.post("/api/onboarding/complete", json=body)
    assert response.status_code == 200, response.text
    assert _ids(json.loads(onboarding.saved()["OUROBOROS_SUBAGENTS"])) == ["helper"]


def test_factory_reviewers_top_up_only_a_catalog_nobody_marked(pool_seams):
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS
    from ouroboros.gateway.onboarding import with_factory_review_rows

    topped = with_factory_review_rows(OWNER_DRAFT, {"OUROBOROS_MODEL": "openai/gpt-5.6-sol"})
    assert _ids(topped) == ["helper", "review-1", "review-2", "review-3"]
    assert json.loads(pool_seams["factory"][-1]["OUROBOROS_SUBAGENTS"]) == OWNER_DRAFT
    marked = {**OWNER_DRAFT, "items": [{**OWNER_DRAFT["items"][0], "review_eligible": True}]}
    assert with_factory_review_rows(marked, {}) == marked and len(pool_seams["factory"]) == 1
    full = {**OWNER_DRAFT, "items": [{**OWNER_DRAFT["items"][0], "subagent_id": f"row-{n}"}
                                     for n in range(MAX_CONFIGURED_SUBAGENTS - 1)]}
    assert len(with_factory_review_rows(full, {})["items"]) == MAX_CONFIGURED_SUBAGENTS


def test_main_review_recovery_moves_only_marked_rows_onto_main_and_keeps_their_identity():
    """Finishing without agent defaults while subscriptions are connected: each row marked
    Reviewer keeps its id, switch, provenance and effort (a compound session effort becomes
    the row's), takes Main with Main's account and processing, and reads the work itself;
    an unmarked row is untouched."""
    from ouroboros.gateway.onboarding import review_rows_on_main

    main = "claudexor::codex=main"
    settings = {"OUROBOROS_MODEL": main, "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "account"}),
                "OUROBOROS_MODEL_PROCESSING_PREFERENCES": json.dumps({"main": "standard"})}
    helper = {"subagent_id": "helper", "recommended_use": "Helps.", "route": {"kind": "agent_session", "target_id": "codex"}}
    moved = review_rows_on_main({"enabled": False, "items": [
        {"subagent_id": "own-session", "recommended_use": "Reads.", "review_eligible": True, "access": "full",
         "route": {"kind": "agent_session", "target_id": "cursor=gpt-5.6-sol-xhigh"}},
        helper,
        {"subagent_id": "own-api", "recommended_use": "Checks.", "review_eligible": True, "enabled": False,
         "effort": "low", "minted_from": "review_lane", "delivery": "packet",
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-luna"}},
    ]}, settings)

    on_main = {"kind": "api_model", "target_id": main, "credential_profile_id": "account"}
    assert moved["enabled"] is False and moved["items"][1] == helper
    assert moved["items"][0] == {"subagent_id": "own-session", "recommended_use": "Reads.", "review_eligible": True,
                                 "route": on_main, "effort": "xhigh", "processing_preference": "standard"}
    assert moved["items"][2] == {"subagent_id": "own-api", "recommended_use": "Checks.", "enabled": False,
                                 "review_eligible": True, "minted_from": "review_lane",
                                 "route": on_main, "effort": "low", "processing_preference": "standard"}


@pytest.mark.parametrize("settings, message", [
    ({"OUROBOROS_MODEL": ""}, "Choose a Main model"),
    ({"OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "fixture-no-network",
      "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "pin"})}, "managed model source"),
])
def test_main_review_recovery_refuses_a_main_it_cannot_honour(settings, message):
    from ouroboros.gateway.onboarding import review_rows_on_main

    with pytest.raises(ValueError, match=message):
        review_rows_on_main({"enabled": True, "items": []}, settings)


def test_codex_only_preview_and_finish_share_models_agents_and_atomic_settings(onboarding, pool_seams):
    """One managed account supplies real model defaults and agent review, without API keys."""
    model = "claudexor::codex=provider-default"
    onboarding.calls["snapshot_payload"] = {
        **LIVE_SNAPSHOT,
        "harnesses": [LIVE_SNAPSHOT["harnesses"][1]],
        "profiles": {"harnessAccounts": [_profile_account("codex", "shared")],
                     "profiles": [_profile("codex", "shared")]},
        "model_catalog": [{"value": model, "is_default": True, "input_modalities": ["text", "image"],
                           "credential_profile_id": "shared", "max_context_window": 872000}],
    }
    draft = {"subscriptionsConnected": True, "OUROBOROS_MODEL": "", "TOTAL_BUDGET": 25.0}
    preview = onboarding.client.post("/api/onboarding/subagents/preview", json=draft)
    assert preview.status_code == 200, preview.text
    proposed = preview.json()
    assert proposed["model_settings"]["OUROBOROS_MODEL"] == model
    catalog = proposed["available_subagents"]
    assert any(row.get("review_eligible") is True for row in catalog["items"]), "the preview proposes reviewers"
    assert not onboarding.settings_path.exists()
    completed = onboarding.client.post("/api/onboarding/complete", json={
        **draft, **proposed["model_settings"], "OUROBOROS_SUBAGENTS": catalog,
        "OUROBOROS_MODEL_ACCOUNTS": {"main": "shared", "light": ""},
        "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 1000000},
    })
    assert completed.status_code == 200, completed.text
    saved = onboarding.saved()
    assert saved["OUROBOROS_MODEL"] == saved["OUROBOROS_MODEL_LIGHT"] == model
    assert not saved["OPENAI_API_KEY"] and not saved["OPENROUTER_API_KEY"]
    assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"])["main"] == "shared"
    assert json.loads(saved["OUROBOROS_MODEL_CONTEXT_WINDOWS"])["main"] == 1000000
    assert saved[ONBOARDING_COMPLETED_KEY]
    assert onboarding.calls["supervisor"] == 1
    assert json.loads(saved["OUROBOROS_SUBAGENTS"])["items"] == catalog["items"]
    assert not saved.get("OUROBOROS_REVIEWER_SLOTS")


def test_codex_only_finish_computes_omitted_quick_path_models(onboarding):
    onboarding.calls["snapshot_payload"] = {
        **LIVE_SNAPSHOT,
        "model_catalog": [{"value": "claudexor::codex=default", "is_default": True,
                           "input_modalities": ["text", "image"]}],
    }
    response = onboarding.client.post("/api/onboarding/complete", json={"subscriptionsConnected": True})
    assert response.status_code == 200, response.text
    assert onboarding.saved()["OUROBOROS_MODEL"] == "claudexor::codex=default"


def test_finish_preserves_visible_empty_inheritance_and_shipped_value(onboarding):
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    onboarding.calls["snapshot_payload"] = {
        **LIVE_SNAPSHOT,
        "model_catalog": [{"value": "claudexor::codex=default", "is_default": True}],
    }
    light = SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]
    response = onboarding.client.post("/api/onboarding/complete", json={
        "subscriptionsConnected": True, "OUROBOROS_MODEL": "",
        "OUROBOROS_MODEL_LIGHT": light, "OUROBOROS_MODEL_VISION": "",
        "OUROBOROS_MODEL_FALLBACKS": "",
    })
    assert response.status_code == 200, response.text
    saved = onboarding.saved()
    assert saved["OUROBOROS_MODEL_LIGHT"] == light
    assert saved["OUROBOROS_MODEL_VISION"] == ""
    assert saved["OUROBOROS_MODEL_FALLBACKS"] == ""


@pytest.mark.serial
def test_explicit_preset_recovery_previews_main_reviews_then_saves_visible_draft(onboarding, pool_seams):
    """A broken CLI inventory does not replace a working Main with keyless API defaults:
    every reviewer row the preview proposes runs on Main, and the visible draft is saved
    as shown, not recomputed."""
    from ouroboros.subscription_install_presets import PRESET_MARKER_KEY

    model = "claudexor::codex=chosen-model"
    onboarding.calls["snapshot_payload"] = RuntimeError("model inventory unavailable")
    draft = {
        "subscriptionsConnected": True, "skipSubscriptionPresets": True,
        "OUROBOROS_MODEL": model, "OUROBOROS_MODEL_LIGHT": "",
        "OUROBOROS_MODEL_FALLBACKS": "", "TOTAL_BUDGET": 25,
        "OUROBOROS_MODEL_ACCOUNTS": {"main": "chosen-account"},
        "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"main": "standard"},
    }
    preview = onboarding.client.post("/api/onboarding/subagents/preview", json=draft)
    assert preview.status_code == 200, preview.text
    assert onboarding.calls["snapshot"] == 0
    assert not onboarding.settings_path.exists()
    visible = preview.json()["available_subagents"]
    reviewers = [row for row in visible["items"] if row["subagent_id"] in {r["subagent_id"] for r in FACTORY_ROWS}]
    assert [row["effort"] for row in reviewers] == [row["effort"] for row in FACTORY_ROWS]
    assert all(row["route"] == {"kind": "api_model", "target_id": model, "credential_profile_id": "chosen-account"}
               and row["processing_preference"] == "standard" for row in reviewers)

    # A later owner edit to the displayed proposal is saved as shown, not recomputed.
    reviewers[0]["recommended_use"] = "Owner refinement."
    completed = onboarding.client.post("/api/onboarding/complete", json={**draft, "OUROBOROS_SUBAGENTS": visible})
    assert completed.status_code == 200, completed.text
    assert onboarding.calls["snapshot"] == 0
    saved = onboarding.saved()
    assert json.loads(saved["OUROBOROS_SUBAGENTS"])["items"] == visible["items"]
    assert not saved.get(PRESET_MARKER_KEY)
    assert saved[ONBOARDING_COMPLETED_KEY]
    assert onboarding.calls["supervisor"] == 1


@pytest.mark.serial
@pytest.mark.parametrize("model", ["", "openai/no-key-model"])
def test_main_review_recovery_does_not_invent_main_access(onboarding, model):
    response = onboarding.client.post("/api/onboarding/subagents/preview", json={
        "subscriptionsConnected": True, "skipSubscriptionPresets": True,
        "OUROBOROS_MODEL": model,
    })
    assert response.status_code == 400, response.text
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["snapshot"] == 0
