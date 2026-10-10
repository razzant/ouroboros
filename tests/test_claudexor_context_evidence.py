"""Context capacity belongs to an exact subscription account and role."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from ouroboros import capability_evidence as ce


MODEL = "claudexor::test-source=exact-model"


def _catalog(profile="account-a", fingerprint="identity-a", window=872_000):
    return {
        "source": "test-source", "credentialProfileId": profile,
        "accountFingerprint": fingerprint, "observedAt": ce.utc_now_iso(),
        "provenance": "exact raw transport catalog",
        "models": [{"id": "exact-model", "contextWindow": 272_000,
                    "maxContextWindow": window, "maxOutputTokens": None,
                    "effectiveContextWindow": 250_240, "compactionThreshold": 95}],
    }


@pytest.fixture
def catalog(monkeypatch):
    from ouroboros.llm import LLMClient

    current = {"value": _catalog(), "calls": []}

    def fetch(source, credential_profile_id=None, *, requested_model=None):
        current["calls"].append((source, credential_profile_id, requested_model))
        return copy.deepcopy(current["value"])

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(fetch), raising=False)
    monkeypatch.setattr(ce, "_generative_probe_window", lambda *_a, **_k: pytest.fail("generation probe"))
    return current


def _probe(root, **kwargs):
    return ce.probe(root, provider="claudexor", model=MODEL, **kwargs)


def _binding(evidence):
    return {"source": evidence.source_id, "model": "exact-model",
            "credentialProfileId": evidence.credential_profile_id,
            "accountFingerprint": evidence.account_fingerprint}


@pytest.mark.parametrize("changed", [
    {"source_id": "other-source"},
    {"credential_profile_id": "account-b"},
    {"account_fingerprint": "new-account-identity"},
])
def test_subscription_capacity_identity_includes_account_binding(changed):
    options = {
        "source_id": "test-source",
        "credential_profile_id": "account-a",
        "account_fingerprint": "account-identity-a",
    }
    route = {"provider": "claudexor", "model": "claudexor::test-source=exact-model"}
    first = ce.route_fingerprint(**route, options=options)
    assert first != ce.route_fingerprint(**route, options={**options, **changed})
    # A renewed bearer credential does not change the account's capacity identity.
    assert first == ce.route_fingerprint(**route, options=options,
                                         headers={"Authorization": "Bearer test-placeholder"})


def test_auto_uses_largest_advertised_capacity_and_retains_provenance(tmp_path, catalog):
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 872_000
    assert evidence.status == ce.STATUS_CONFIRMED
    assert evidence.source == ce.SOURCE_PROVIDER_METADATA
    assert evidence.credential_profile_id == "account-a"
    assert evidence.account_fingerprint == "identity-a"
    assert evidence.provenance == catalog["value"]["provenance"]
    assert evidence.ts == catalog["value"]["observedAt"]
    assert catalog["calls"] == [("test-source", None, "exact-model")]
    assert not ce.confirms_at_least(evidence)


def test_auto_capacity_uses_model_compatible_account_and_keeps_exact_binding(tmp_path, monkeypatch):
    from ouroboros.llm import LLMClient

    def catalog(source, credential_profile_id=None, *, requested_model=None):
        assert (source, credential_profile_id, requested_model) == ("test-source", None, "exact-model")
        return _catalog("account-b", "identity-b", 500_000)

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 500_000
    assert evidence.credential_profile_id == "account-b"
    assert evidence.account_fingerprint == "identity-b"
    wrong = _probe(tmp_path, allow_fetch=False, options={"source_id": "test-source",
                   "credential_profile_id": "account-a", "account_fingerprint": "identity-a"})
    assert wrong.window_tokens == 0


def test_auto_does_not_reuse_last_discovered_accounts_capacity(tmp_path, catalog):
    first = _probe(tmp_path)
    catalog["value"] = _catalog("account-b", "identity-b", 500_000)
    second = _probe(tmp_path)
    assert second.window_tokens == 500_000
    assert first.route_fp != second.route_fp
    assert second.credential_profile_id == "account-b"
    assert len(catalog["calls"]) == 2


def test_exact_observed_binding_reuses_only_its_cache(tmp_path, catalog):
    first = _probe(tmp_path)
    options = ce.model_account_options(MODEL, model_route=_binding(first))
    cached = _probe(tmp_path, options=options, allow_fetch=False)
    assert cached.to_json() == first.to_json()
    assert len(catalog["calls"]) == 1
    for change in ({"credential_profile_id": "account-b"},
                   {"account_fingerprint": "reauthorized-identity"}):
        other = _probe(tmp_path, options={**options, **change}, allow_fetch=False)
        assert other.window_tokens == 0
        assert not ce.is_known(other)


@pytest.mark.parametrize("change", [
    {"source": "wrong-source"}, {"credentialProfileId": "wrong-profile"},
    {"accountFingerprint": "wrong-identity"},
])
def test_mismatched_catalog_cannot_supply_requested_account_capacity(tmp_path, catalog, change):
    catalog["value"].update(change)
    evidence = _probe(tmp_path, options={"source_id": "test-source",
                      "credential_profile_id": "account-a", "account_fingerprint": "identity-a"})
    assert evidence.window_tokens == 0
    assert evidence.status == ce.STATUS_FAILED


@pytest.mark.parametrize("change", [
    {"accountFingerprint": None}, {"credentialProfileId": ""}, {"provenance": ""},
    {"observedAt": ""}, {"models": []},
    {"models": [{"id": "exact-model", "contextWindow": None, "maxContextWindow": None,
                 "effectiveContextWindow": 250_240}]},
])
def test_missing_capacity_or_binding_remains_unknown_without_generation(tmp_path, catalog, change):
    catalog["value"].update(change)
    evidence = _probe(tmp_path, allow_generative=True)
    assert evidence.window_tokens == 0
    assert not ce.is_known(evidence, require_fresh=True)
    assert not ce.confirms_at_least(evidence)


def test_catalog_observation_time_is_not_refreshed_by_local_cache(tmp_path, catalog):
    catalog["value"]["observedAt"] = "2020-01-01T00:00:00Z"
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 872_000
    assert evidence.stale
    assert not ce.is_known(evidence, require_fresh=True)


def test_observed_route_must_match_model_and_current_role_pin():
    observed = _binding(ce.CapabilityEvidence(
        872_000, "confirmed", "provider_metadata", "fp", source_id="test-source",
        credential_profile_id="account-a", account_fingerprint="identity-a"))
    settings = {"OUROBOROS_MODEL_ACCOUNTS": {"main": "account-b", "light": "account-a"}}
    main = ce.model_account_options(MODEL, role="main", settings=settings, model_route=observed)
    light = ce.model_account_options(MODEL, role="light", settings=settings, model_route=observed)
    assert main == {"source_id": "test-source", "credential_profile_id": "account-b"}
    assert light["account_fingerprint"] == "identity-a"
    changed = ce.model_account_options(MODEL + "-new", role="light", settings=settings,
                                       model_route=observed)
    assert "account_fingerprint" not in changed
    nameless = {**observed, "accountFingerprint": None}
    assert "account_fingerprint" not in ce.model_account_options(MODEL, model_route=nameless)


def test_role_manual_window_above_catalog_is_asserted_not_scope_ack(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.context_fit import resolve_context_fit_route

    settings = {"OUROBOROS_MODEL": MODEL,
                "OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a", "light": "account-b"},
                "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 1_200_000, "fallback": [900_000]}}
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    main_route, main = resolve_context_fit_route({"model": MODEL, "model_role": "main"}, allow_fetch=True)
    assert main.window_tokens == 1_200_000 and main.status == "asserted"
    assert main.source == "user_setting"
    assert ce.is_known(main)  # sizing assertion, not governance proof
    assert not ce.confirms_at_least(main)
    assert main_route["options"]["credential_profile_id"] == "account-a"
    assert ce.list_owner_acks(tmp_path) == []
    catalog["value"] = _catalog("account-b", "identity-b", 500_000)
    light_route, light = resolve_context_fit_route({"model": MODEL, "model_role": "light"}, allow_fetch=True)
    assert light.window_tokens == 500_000 and light.source == "provider_metadata"
    assert light_route["options"]["credential_profile_id"] == "account-b"
    assert light.route_fp != main.route_fp
    catalog["value"] = _catalog()
    _, fallback = resolve_context_fit_route({"model": MODEL, "model_role": "fallback:0"}, allow_fetch=True)
    assert fallback.window_tokens == 900_000 and fallback.source == "user_setting"


def test_manual_reviewer_sizing_does_not_change_scope_authority(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.reviewer_window import resolve_reviewer_window

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OUROBOROS_MODEL_CONTEXT_WINDOWS", json.dumps({"deep_review": 1_200_000}))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"deep_review": "account-a"}))
    window = resolve_reviewer_window(MODEL, model_role="deep_review", use_local=False)
    assert window.window_tokens == 872_000
    assert window.sizing_window() == 1_200_000
    assert window.sizing_source == "user_setting"
    assert window.model_route["credentialProfileId"] == "account-a"
    assert ce.list_owner_acks(tmp_path) == []


def test_main_input_reserve_and_account_capacity_remain_distinct(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.context_fit import measure_main_fit
    from tests.test_context_fit_v664 import _plan

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    evidence = _probe(tmp_path)
    plan = replace(_plan(), model=MODEL, provider="claudexor", route_fp=evidence.route_fp,
                   window_tokens=evidence.window_tokens, status=evidence.status,
                   model_route=_binding(evidence))
    measured = measure_main_fit(plan, plan.messages_for("max"), [], drive_root=tmp_path,
                                profile="owner_max", rendered_mode="max", round_id="r1").measurement
    assert measured.capacity_total_tokens == 872_000
    assert measured.response_reserve_tokens == 65_536
    assert 0 < measured.estimated_input_tokens < measured.response_reserve_tokens
    assert measured.target_total_tokens is None


def test_unknown_subscription_reviewer_keeps_existing_input_budget(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.reviewer_window import resolve_reviewer_window, window_scaled_reserves
    from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET, calibrated_input_token_limit
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    catalog["value"]["models"] = []
    window = resolve_reviewer_window(MODEL, use_local=False)
    assert window.window_tokens == 0
    assert window.sizing_window() == 0
    assert window_scaled_reserves(0, output_reserve=65_536, tokenizer_margin=50_000) == (65_536, 50_000)
    assert calibrated_input_token_limit(MODEL, context_window=0, output_reserve=65_536,
                                         tokenizer_margin=50_000, budget_cap=456_789) == 456_789
    caps = per_slot_input_token_limits([MODEL], output_reserve=65_536, tokenizer_margin=50_000)
    assert caps == {MODEL: REVIEW_PROMPT_TOKEN_BUDGET}


def test_unknown_subscription_native_episode_keeps_owner_ceiling(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.review_native_episode import review_native_transcript_bound

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "765432")
    catalog["value"]["models"] = []
    assert review_native_transcript_bound(MODEL, output_reserve=65_536, use_local=False) == 765432


def test_unknown_subscription_scope_stays_unknown_without_authority(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.tools.scope_window import scope_window, scope_window_provenance, window_provenance_phrase

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    catalog["value"]["models"] = []
    window = scope_window(MODEL)
    assert window.window_tokens == 0
    assert window_provenance_phrase(window.window_tokens, scope_window_provenance(window)) == "unknown window"


def test_model_only_ack_and_cache_cannot_certify_subscription_account(tmp_path, catalog):
    fp = ce.route_fingerprint(provider="claudexor", model=MODEL)
    with pytest.raises(ValueError, match="exact account binding"):
        ce.record_owner_ack(tmp_path, provider="claudexor", model=MODEL, window_tokens=1_200_000)
    # A legacy unbound record remains readable, but cannot certify an account.
    ce._store_evidence(tmp_path, "owner_acks", fp, {"route_fp": fp, "window_tokens": 1_200_000})
    ce._store_evidence(tmp_path, "probes", fp, ce.CapabilityEvidence(
        1_200_000, "confirmed", "provider_metadata", fp, MODEL, "claudexor",
        ts=ce.utc_now_iso(),
    ).to_json())
    cold = _probe(tmp_path, allow_fetch=False)
    assert cold.window_tokens == 0
    actual = _probe(tmp_path)
    assert actual.window_tokens == 872_000
    assert not ce.confirms_at_least(actual)


def test_exact_account_ack_retains_its_binding_and_expires_on_account_change(tmp_path, catalog):
    options = {"source_id": "test-source", "credential_profile_id": "account-a",
               "account_fingerprint": "identity-a"}
    ce.record_owner_ack(tmp_path, provider="claudexor", model=MODEL,
                        window_tokens=1_200_000, options=options)
    acknowledged = _probe(tmp_path, options=options, allow_fetch=False)
    assert ce.confirms_at_least(acknowledged)
    assert acknowledged.credential_profile_id == "account-a"
    assert acknowledged.account_fingerprint == "identity-a"
    replaced_account = _probe(tmp_path, options={**options, "account_fingerprint": "identity-b"},
                              allow_fetch=False)
    assert replaced_account.window_tokens == 0
    assert not ce.confirms_at_least(replaced_account)
