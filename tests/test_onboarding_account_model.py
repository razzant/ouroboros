"""#244: onboarding on the retired pre-unified account model, and its negatives.

The status read refuses a pre-unified roster with a typed reason; onboarding says
so instead of "no verified account", and no legacy native row or `native` routing
verdict vouches a seat. Same isolated endpoint fixture as the atomic transaction.
"""

from __future__ import annotations

from tests.test_onboarding_complete_endpoint import (
    LIVE_SNAPSHOT, WIZARD_PAYLOAD, _native_account, _pool, _profile, _routable_snapshot, _unified_snapshot,
    onboarding as onboarding,  # explicit fixture re-export, not another settings writer
)

_UNSUPPORTED_SNAPSHOT = {
    "daemon": {"state": "running", "engine_version": "3.5.2",
               "runtime": {"state": "update_staged", "target_version": "4.7.0", "staged_version": "4.7.0"}},
    "harnesses": LIVE_SNAPSHOT["harnesses"],
    "profiles": {},
    "reads": {"catalog": "ok", "accounts": "failed", "quota": "ok"},
    "facets": {"accounts": {"observed_at": None, "stale": False, "error": "account_model_unsupported"}},
}


def test_a_pre_unified_engine_is_its_own_typed_preset_refusal():
    from ouroboros.gateway.onboarding import verified_harness_discoveries

    discoveries, failure = verified_harness_discoveries(_UNSUPPORTED_SNAPSHOT)
    assert discoveries == ()
    assert failure is not None and failure.code == "account_model_unsupported"
    assert "3.5.2" in failure.detail and "newer Claudexor" in failure.detail
    # A roster held from an earlier answer never outranks the current refusal.
    held = {**_UNSUPPORTED_SNAPSHOT, "profiles": {
        "profiles": [_profile("claude", "claude-default")], "harnessAccounts": [],
        "accountPools": [_pool("claude", {"kind": "profile", "profileId": "claude-default"})]},
        "facets": {"accounts": {"observed_at": "2026-10-27T10:00:00Z", "stale": True,
                                "error": "account_model_unsupported"}}}
    assert verified_harness_discoveries(held)[1].code == "account_model_unsupported"


def test_a_pre_unified_engine_refuses_completion_with_the_skip_offer(onboarding):
    onboarding.calls["snapshot_payload"] = _UNSUPPORTED_SNAPSHOT

    response = onboarding.client.post(
        "/api/onboarding/complete", json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["code"] == "account_model_unsupported" and body["can_skip"] is True
    assert body["saved"] is False and not onboarding.settings_path.exists()


def test_legacy_native_rows_never_vouch_a_subscription_seat():
    """A legacy carrier riding beside the pools (or alone) is not read: a detected native
    login is not a configured seat, whatever the pool says this minute."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    hybrid = subscription_routable_harnesses(_unified_snapshot(
        [_pool("claude", {"kind": "none", "reason": "every window is spent"})],
        legacy_accounts=[_native_account("claude")]))
    raw_legacy = subscription_routable_harnesses(
        _routable_snapshot(None, legacy_accounts=[_native_account("claude")]))

    assert hybrid == ({}, {"claude": "every window is spent"})
    assert raw_legacy == ({}, {})


def test_a_native_kind_in_a_pool_is_an_unknown_routing_state():
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_unified_snapshot(
        [_pool("claude", {"kind": "native", "route": "local_session"})]))

    assert routable == {}
    assert "unknown routing state 'native'" in refused["claude"]


def test_an_api_key_pool_route_never_erases_a_configured_named_seat():
    """The API-key verdict describes an unpinned run right now; a configured subscription
    seat of the same harness stays configured (D-3)."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_unified_snapshot(
        [_pool("codex", {"kind": "api_key_route"})], [_profile("codex", "codex-team-seat")]))

    assert refused == {}
    assert routable["codex"].startswith("account 'codex-team-seat' (config_dir_login)")
    assert "API key" in routable["codex"]
