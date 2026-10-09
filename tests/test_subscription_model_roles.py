"""Role-owned subscription options; model identity never smuggles an account."""

import json

import pytest

from tests.test_llm_claudexor import MODEL, setup as subscription_transport  # noqa: F401
from tests.test_model_wait import live_wait as wait_fixture, _action_for

setup = subscription_transport  # The imported live-wait fixture depends on this name.
reviewer_wait = wait_fixture

from ouroboros.model_slots import (
    MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY,
    model_role_option, normalize_model_role_options,
)
from ouroboros.provider_models import (
    model_has_credentials_in_settings, parse_claudexor_model, provider_for_model,
    resolve_model_target,
)


def test_image_generation_slot_is_not_a_chat_model_role():
    from ouroboros.config import normalize_settings_raw

    saved = normalize_settings_raw({"OUROBOROS_MODEL_IMAGE": "gpt-image-2",
                                    MODEL_ACCOUNTS_KEY: {"main": "profile"}})
    assert saved["OUROBOROS_MODEL_IMAGE"] == "gpt-image-2"
    assert json.loads(saved[MODEL_ACCOUNTS_KEY]) == {"main": "profile"}
    # The shared browser editor must not offer image Account/Window/Processing:
    # a Settings POST carrying them is refused, not silently normalized.
    with pytest.raises(ValueError):
        normalize_model_role_options(MODEL_ACCOUNTS_KEY, {"image": "profile"})


def test_equal_model_names_keep_distinct_main_and_light_accounts():
    settings = {
        "OUROBOROS_MODEL": "claudexor::codex=one-model",
        "OUROBOROS_MODEL_LIGHT": "claudexor::codex=one-model",
        MODEL_ACCOUNTS_KEY: {"main": "personal", "light": "work"},
    }
    assert model_role_option(MODEL_ACCOUNTS_KEY, "main", settings=settings) == "personal"
    assert model_role_option(MODEL_ACCOUNTS_KEY, "light", settings=settings) == "work"
    assert model_role_option(MODEL_ACCOUNTS_KEY, "", settings=settings) == ""
    assert parse_claudexor_model(settings["OUROBOROS_MODEL"]) == ("codex", "one-model")
    assert resolve_model_target(settings["OUROBOROS_MODEL"]).credential_ref == ""


def test_fallback_options_preserve_order_and_explicit_auto():
    raw = {"main": "", "fallback": ["profile-B", "", "profile-A"]}
    parsed, encoded = normalize_model_role_options(MODEL_ACCOUNTS_KEY, raw)
    assert parsed == raw == json.loads(encoded)
    settings = {MODEL_ACCOUNTS_KEY: encoded}
    assert [model_role_option(MODEL_ACCOUNTS_KEY, f"fallback:{i}", settings=settings)
            for i in range(4)] == ["profile-B", "", "profile-A", ""]


def test_manual_context_above_catalog_is_an_assertion_not_an_adapter_cap():
    settings = {MODEL_CONTEXT_WINDOWS_KEY: {"main": 1_000_000, "light": 872_000, "fallback": [0]}}
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "main", settings=settings) == 1_000_000
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "vision", settings=settings) == 0
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "fallback:0", settings=settings) == 0


@pytest.mark.parametrize("key, raw", [
    (MODEL_ACCOUNTS_KEY, {"main": True}),
    (MODEL_ACCOUNTS_KEY, {"main": {"pin": "profile"}}),
    (MODEL_ACCOUNTS_KEY, {"fallback": "a,b"}),
    (MODEL_ACCOUNTS_KEY, {"unknown": "profile"}),
    (MODEL_ACCOUNTS_KEY, "not JSON"),
    (MODEL_CONTEXT_WINDOWS_KEY, {"main": -1}),
    (MODEL_CONTEXT_WINDOWS_KEY, {"main": True}),
    (MODEL_CONTEXT_WINDOWS_KEY, {"main": 1.2}),
    (MODEL_CONTEXT_WINDOWS_KEY, {"main": "1000000"}),
])
def test_malformed_options_cannot_silently_drop_a_pin_or_invent_capacity(key, raw):
    with pytest.raises(ValueError):
        normalize_model_role_options(key, raw)


def test_config_read_accepts_object_and_json_and_is_idempotent():
    from ouroboros.config import normalize_settings_raw

    raw = {MODEL_ACCOUNTS_KEY: {"main": "profile"}, MODEL_CONTEXT_WINDOWS_KEY: '{"light":872000}'}
    normalized = normalize_settings_raw(raw)
    assert normalized == normalize_settings_raw(normalized)
    assert json.loads(normalized[MODEL_ACCOUNTS_KEY]) == {"main": "profile"}
    assert json.loads(normalized[MODEL_CONTEXT_WINDOWS_KEY]) == {"light": 872000}


def test_managed_selection_does_not_need_or_fabricate_an_api_key():
    model = "claudexor::codex=model-A"
    assert provider_for_model(model) == "claudexor"
    assert model_has_credentials_in_settings(model, {})
    assert not model_has_credentials_in_settings("openai::model-A", {})
    assert resolve_model_target(model).provider_route == "claudexor"
    with pytest.raises(ValueError):
        parse_claudexor_model("claudexor::codex")


def test_persisted_wait_switch_changes_only_the_named_model_role():
    from ouroboros.model_slots import apply_model_role_override
    initial = {"OUROBOROS_MODEL": "claudexor::codex=same", "OUROBOROS_MODEL_LIGHT": "claudexor::codex=same",
               MODEL_ACCOUNTS_KEY: {"main": "first", "light": "second"}}
    saved = apply_model_role_override(initial, role="light", model="openai::other",
                                      credential_profile_id="", use_local=False)
    assert saved["OUROBOROS_MODEL"] == initial["OUROBOROS_MODEL"]
    assert saved["OUROBOROS_MODEL_LIGHT"] == "openai::other"
    assert json.loads(saved[MODEL_ACCOUNTS_KEY]) == {"main": "first", "light": ""}
    assert initial[MODEL_ACCOUNTS_KEY]["light"] == "second"


def test_managed_model_account_roundtrips_actor_and_reviewer_configuration():
    from ouroboros.configured_subagents import normalize_configured_subagents
    from ouroboros.reviewer_slot_config import parse_reviewer_slots
    route = {"kind": "api_model", "target_id": "claudexor::codex=same", "credential_profile_id": "named"}
    actor = {"subagent_id": "actor", "recommended_use": "Review", "route": route}
    parsed, encoded = normalize_configured_subagents({"enabled": True, "items": [actor]})
    assert parsed.items[0].route.credential_profile_id == "named"
    assert json.loads(encoded)['items'][0]['route'] == route
    slots = parse_reviewer_slots(json.dumps({
        group: [{"slot_id": group, "route": {"kind": "api_chat", "target_id": route['target_id'], "profile_id": "named"}}]
        for group in ("triad", "scope")
    }))
    assert slots.triad[0].profile_id == slots.scope[0].profile_id == "named"


@pytest.mark.parametrize("pin", ["", "named"])
def test_empty_subscription_model_is_rejected_before_actor_or_reviewer_serialization(pin):
    from ouroboros.configured_subagents import normalize_configured_subagents
    from ouroboros.reviewer_slot_config import parse_reviewer_slots

    target = "claudexor::codex="
    with pytest.raises(ValueError):
        normalize_configured_subagents({"enabled": True, "items": [{
            "subagent_id": "actor", "recommended_use": "Review", "route": {
                "kind": "api_model", "target_id": target, "credential_profile_id": pin}}]})
    with pytest.raises(ValueError):
        parse_reviewer_slots(json.dumps({group: [{"slot_id": group, "route": {
            "kind": "api_chat", "target_id": target, "profile_id": pin}}]
            for group in ("triad", "scope")}))


def test_persist_referenced_reviewer_keeps_other_roles_and_native_delivery():
    from ouroboros.model_slots import apply_model_role_override
    from ouroboros.reviewer_slot_config import parse_reviewer_slots, roster_env_override
    actor = {"subagent_id": "shared", "recommended_use": "Review", "route": {
        "kind": "api_model", "target_id": "claudexor::codex=old", "credential_profile_id": "old"}}
    original = {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [actor]}),
                "OUROBOROS_REVIEWER_SLOTS": json.dumps({group: [{"slot_id": group, "subagent_id": "shared"}]
                                                        for group in ("triad", "scope")})}
    saved = apply_model_role_override(original, role="reviewer:triad", model="claudexor::codex=new",
                                      credential_profile_id="new", use_local=False)
    roster = json.loads(saved['OUROBOROS_SUBAGENTS'])
    assert roster['items'][0] == actor
    with roster_env_override(saved['OUROBOROS_SUBAGENTS']):
        slots = parse_reviewer_slots(saved['OUROBOROS_REVIEWER_SLOTS'])
    assert slots.triad[0].native_retrieval and slots.scope[0].native_retrieval
    assert slots.triad[0].target_id == "claudexor::codex=new" and slots.triad[0].profile_id == "new"
    assert slots.scope[0].target_id == "claudexor::codex=old" and slots.scope[0].profile_id == "old"
    assert apply_model_role_override(saved, role="reviewer:triad", model="claudexor::codex=new",
                                      credential_profile_id="new", use_local=False) == saved


@pytest.mark.parametrize("roster", [None, "", '{"enabled":false,"items":[]}'])
def test_persist_default_reviewer_omits_untouched_deep_row_and_roster(roster):
    from ouroboros.model_slots import apply_model_role_override
    from ouroboros.subscription_install_presets import preview_api_reviewer_slots

    original = {"OUROBOROS_REVIEWER_SLOTS": "", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai::owner-deep"}
    if roster is not None:
        original["OUROBOROS_SUBAGENTS"] = roster
    before = json.loads(preview_api_reviewer_slots(original))
    identity = before["triad"][0]["slot_id"]
    saved = apply_model_role_override(original, role=f"reviewer:{identity}", model="claudexor::codex=new",
                                      credential_profile_id="new-pin", use_local=False)
    after = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
    expected = dict(before)
    expected.pop("deep_review")
    expected["triad"][0]["route"] = {"kind": "api_chat", "target_id": "claudexor::codex=new", "profile_id": "new-pin"}
    assert after == expected
    assert saved.get("OUROBOROS_SUBAGENTS") == original.get("OUROBOROS_SUBAGENTS")
    assert ("OUROBOROS_SUBAGENTS" in saved) is (roster is not None)
    assert saved["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] == "openai::owner-deep"
    assert original["OUROBOROS_REVIEWER_SLOTS"] == ""


def test_persist_default_deep_reviewer_authors_only_its_selected_assignment():
    from ouroboros.model_slots import apply_model_role_override
    from ouroboros.subscription_install_presets import preview_api_reviewer_slots

    original = {"OUROBOROS_REVIEWER_SLOTS": "", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai::owner-deep"}
    before = json.loads(preview_api_reviewer_slots(original))
    saved = apply_model_role_override(original, role="reviewer:deep_review_slot_1", model="claudexor::codex=review",
                                      credential_profile_id="review-pin", use_local=False)
    after = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
    before["deep_review"]["route"] = {"kind": "api_chat", "target_id": "claudexor::codex=review", "profile_id": "review-pin"}
    assert after == before
    assert "OUROBOROS_SUBAGENTS" not in saved


def test_default_reviewer_wait_persists_through_the_real_owner_writer_without_deep_materialization(reviewer_wait, monkeypatch):
    from ouroboros import config, model_wait
    from ouroboros.subscription_install_presets import preview_api_reviewer_slots

    root, _, _, controller, _, decide = reviewer_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    initial = {"OUROBOROS_REVIEWER_SLOTS": "", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai::owner-deep"}
    (root / "settings.json").write_text(json.dumps(initial))
    defaults = json.loads(preview_api_reviewer_slots(initial))
    slot = defaults["triad"][0]["slot_id"]
    row = {"wait_id": "reviewer-wait", "revision": 1, "task_attempt": 1,
           "state": "waiting", "role": f"reviewer:{slot}"}
    model_wait.mutate_wait(root, "task-one", row["wait_id"], lambda _: row)
    controller.waits[row["wait_id"]] = dict(row)
    body = _action_for({**row, "task_id": "task-one"}, "switch", model=MODEL,
                       credential_profile_id="reviewer-pin", use_local=False, persist_role=True)
    response = decide(body)
    assert response.status_code == 202 and json.loads(response.body)["saved"] is True
    saved = json.loads((root / "settings.json").read_text())
    panel = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
    assert panel["triad"][0]["route"]["profile_id"] == "reviewer-pin"
    assert panel["triad"][1:] == defaults["triad"][1:] and panel["scope"] == defaults["scope"]
    assert "deep_review" not in panel and saved["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] == initial["OUROBOROS_MODEL_DEEP_SELF_REVIEW"]
    assert not saved.get("OUROBOROS_SUBAGENTS")
    stamp = (root / "settings.json").stat().st_mtime_ns
    assert decide(body).status_code == 200
    assert (root / "settings.json").stat().st_mtime_ns == stamp


def test_public_and_summary_projection_drop_opaque_model_state_only():
    from ouroboros.anthropic_native_custody import public_custody_projection
    from ouroboros.context_compaction import _summary_projection
    value = {"role": "assistant", "content": "kept", "nativeContinuation": {"payload": "opaque"},
             "tool_calls": [{"id": "kept-tool"}]}
    for projection in (public_custody_projection, _summary_projection):
        result = projection(value)
        assert 'nativeContinuation' not in result
        assert result['content'] == 'kept' and result['tool_calls'] == value['tool_calls']
    assert 'nativeContinuation' in value
