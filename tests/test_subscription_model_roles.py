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
    from ouroboros.reviewer_slot_config import review_pool_rows
    route = {"kind": "api_model", "target_id": "claudexor::codex=same", "credential_profile_id": "named"}
    actor = {"subagent_id": "actor", "recommended_use": "Review", "route": route, "review_eligible": True}
    parsed, encoded = normalize_configured_subagents({"enabled": True, "items": [actor]})
    assert parsed.items[0].route.credential_profile_id == "named"
    assert json.loads(encoded)['items'][0]['route'] == route
    # The same row IS the reviewer row (the pool reads the catalog): the pin rides along.
    (row,) = review_pool_rows({"OUROBOROS_SUBAGENTS": encoded})
    assert (row.slot_id, row.profile_id) == ("actor", "named")


@pytest.mark.parametrize("pin", ["", "named"])
def test_empty_subscription_model_is_rejected_before_actor_or_reviewer_serialization(pin):
    from ouroboros.configured_subagents import normalize_configured_subagents
    from ouroboros.reviewer_slot_config import review_pool_rows

    target = "claudexor::codex="
    row = {"subagent_id": "actor", "recommended_use": "Review", "review_eligible": True,
           "route": {"kind": "api_model", "target_id": target, "credential_profile_id": pin}}
    with pytest.raises(ValueError):
        normalize_configured_subagents({"enabled": True, "items": [row]})
    with pytest.raises(ValueError):
        review_pool_rows({"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [row]})})


def test_a_reviewer_wait_persists_through_the_real_owner_writer_into_the_catalog_row(reviewer_wait, monkeypatch):
    """A review pool seat's wait card: ``persist_role`` writes the chosen model and
    pin into THAT catalog row through the real owner writer, leaves the other rows
    alone, authors no review lanes key, and a replay of the same decision writes
    nothing."""
    from ouroboros import config, model_wait

    root, _, _, controller, _, decide = reviewer_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    helper = {"subagent_id": "helper", "recommended_use": "Helps.", "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-luna"}}
    critic = {"subagent_id": "critic", "recommended_use": "Reviews.", "review_eligible": True, "effort": "low",
              "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"}}
    initial = {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [helper, critic]})}
    (root / "settings.json").write_text(json.dumps(initial))
    row = {"wait_id": "reviewer-wait", "revision": 1, "task_attempt": 1,
           "state": "waiting", "role": "reviewer:critic"}
    model_wait.mutate_wait(root, "task-one", row["wait_id"], lambda _: row)
    controller.waits[row["wait_id"]] = dict(row)
    body = _action_for({**row, "task_id": "task-one"}, "switch", model=MODEL,
                       credential_profile_id="reviewer-pin", use_local=False, persist_role=True)
    response = decide(body)
    assert response.status_code == 202 and json.loads(response.body)["saved"] is True
    saved = json.loads((root / "settings.json").read_text())
    assert "OUROBOROS_REVIEWER_SLOTS" not in saved
    items = json.loads(saved["OUROBOROS_SUBAGENTS"])["items"]
    assert items[0] == helper
    assert items[1]["route"] == {"kind": "api_model", "target_id": MODEL, "credential_profile_id": "reviewer-pin"}
    assert items[1]["review_eligible"] is True and items[1]["effort"] == "low"
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
