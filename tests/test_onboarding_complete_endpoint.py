"""POST /api/onboarding/complete — the ONE atomic onboarding transaction (D-8).

These tests use the REAL persist path against an isolated settings file, so
"nothing was saved" is proved by the file on disk rather than by a mock that was
not called.
"""

from __future__ import annotations

import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY
from ouroboros.subscription_install_presets import PRESET_MARKER_KEY
from ouroboros.configured_subagents import SUBAGENTS_RECEIPT_KEY, SUBAGENTS_SETTING

_PROVIDER_ENV = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_USER",
    "GIGACHAT_PASSWORD", "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_BASE_URL",
    "USE_LOCAL_MAIN", "USE_LOCAL_HEAVY", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
    "USE_LOCAL_CONSCIOUSNESS", "LOCAL_MODEL_SOURCE", PRESET_MARKER_KEY,
    "OUROBOROS_REVIEWER_SLOTS", SUBAGENTS_SETTING, "OUROBOROS_SUBAGENT_HARNESS",
    "OUROBOROS_SAFETY_MODE",
    ONBOARDING_COMPLETED_KEY,
)

# Shapes copied from the LIVE daemon (GET /v2/credential-profiles, GET
# /v2/agent-capabilities, 2026-08-09) through the same projection the accounts
# panel uses: a per-harness accounts row carrying the engine's own `next_up`
# routing pointer, and credential profiles carrying `credential_kind` +
# availability. A fake without `next_up` is a fake of an engine that does not
# exist.
def _native_account(harness, *, route="local_session", detected=True, enabled=True):
    return {
        "harness_id": harness,
        "native_credentials_enabled": enabled,
        "native_login_detected": detected,
        "identity": {"email": "owner@example.com", "plan": "claude_max"},
        "next_up": {"kind": "native", "route": route},
    }


def _profile_account(harness, profile_id):
    return {
        "harness_id": harness,
        "native_credentials_enabled": True,
        "native_login_detected": False,
        "identity": None,
        "next_up": {"kind": "profile", "profileId": profile_id},
    }


def _profile(harness, profile_id, *, kind="config_dir_login", enabled=True,
             availability="available", verification="passed"):
    return {
        "profile": {"profile_id": profile_id, "harness_id": harness,
                    "display_name": profile_id, "credential_kind": kind,
                    "enabled": enabled},
        "status": {"profile_id": profile_id, "harness_id": harness,
                   "availability": availability, "verification": verification,
                   "verification_source": "vendor"},
        "identity": {"email": "owner@example.com", "plan": "pro"},
    }


LIVE_SNAPSHOT = {
    "daemon": {"state": "running"},
    "harnesses": [
        {
            "id": "claude", "status": "ok", "enabled": True,
            "access_profiles_supported": ["readonly", "workspace_write"],
            "models": [{"id": "claude-opus-5"}, {"id": "claude-sonnet-5"},
                       {"id": "claude-fable-5"}, {"id": "claude-opus-4-6"}],
        },
        {
            "id": "codex", "status": "ok", "enabled": True,
            "models": [{"id": "gpt-5.6-sol"}, {"id": "gpt-5.6-terra"}, {"id": "gpt-5.5"}],
        },
    ],
    "profiles": {
        "harnessAccounts": [_native_account("claude"), _native_account("codex")],
        "profiles": [],
    },
}

WIZARD_PAYLOAD = {
    "OPENROUTER_API_KEY": "sk-or-v1-abcdefghijklmnop",
    "OUROBOROS_MODEL": "openai/gpt-5.6-luna",
    "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
    "OUROBOROS_RUNTIME_MODE": "advanced",
    "TOTAL_BUDGET": 25.0,
    "OUROBOROS_PER_TASK_COST_USD": 5.0,
}


class _Harness:
    """One isolated onboarding server plus the calls it made."""

    def __init__(self, client, settings_path, calls):
        self.client = client
        self.settings_path = settings_path
        self.calls = calls

    def saved(self) -> dict:
        if not self.settings_path.exists():
            return {}
        return json.loads(self.settings_path.read_text(encoding="utf-8"))


@pytest.fixture
def onboarding(monkeypatch, tmp_path):
    """Build the endpoint over an isolated settings file (real writes)."""
    import ouroboros.config as config
    import ouroboros.gateway.onboarding as gw_onboarding
    import ouroboros.gateway.settings as gw_settings

    for key in _PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)

    calls: dict = {"snapshot": 0, "env": [], "supervisor": 0, "side_effects": 0}

    def _snapshot():
        calls["snapshot"] += 1
        payload = calls.get("snapshot_payload", LIVE_SNAPSHOT)
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(gw_onboarding, "_read_harness_snapshot", _snapshot)
    monkeypatch.setattr(config, "apply_settings_to_env",
                        lambda settings: calls["env"].append(dict(settings)))

    def _supervisor(_request, _settings):
        calls["supervisor"] += 1
        return True

    def _side_effects(*_a, **_k):
        calls["side_effects"] += 1

    monkeypatch.setattr(gw_settings, "_start_supervisor_if_needed_for_request", _supervisor)
    monkeypatch.setattr(gw_settings, "_apply_settings_save_side_effects", _side_effects)

    app = Starlette(routes=[
        Route("/api/onboarding/subagents/preview",
              endpoint=gw_onboarding.api_onboarding_subagents_preview, methods=["POST"]),
        Route("/api/onboarding/complete",
              endpoint=gw_onboarding.api_onboarding_complete, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path / "repo"
    with TestClient(app) as client:
        yield _Harness(client, settings_path, calls)


def test_fresh_install_applies_the_preset_in_one_write(onboarding):
    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["preset"]["applied"] is True
    assert body["preset"]["reason"] == "applied"
    assert body["preset"]["harnesses"] == ["claude", "codex"]
    assert onboarding.calls["snapshot"] == 1  # exactly ONE daemon read

    saved = onboarding.saved()
    assert saved[PRESET_MARKER_KEY] == "3"
    assert saved["OUROBOROS_SUBAGENT_HARNESS"] == ""
    available = json.loads(saved[SUBAGENTS_SETTING])
    # 4=A: the advisory seat's session route matches no task actor, so the
    # preset mints a roster row for it and every reviewer slot is a
    # subagent_id REFERENCE into the roster the preset ships.
    roster = {row["subagent_id"]: row["route"]["target_id"] for row in available["items"]}
    assert [row["route"]["target_id"] for row in available["items"]] == [
        "claude=claude-opus-5", "codex=gpt-5.6-sol", "openai/gpt-5.6-luna",
        "claude=claude-sonnet-5",
    ]
    assert all("name" not in row for row in available["items"])  # retired (1=A)
    assert json.loads(saved[SUBAGENTS_RECEIPT_KEY])["available_subagents_fingerprint"]
    slots = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
    assert [roster[row["subagent_id"]] for row in slots["triad"]] == [
        "claude=claude-opus-5", "codex=gpt-5.6-sol"]
    assert roster[slots["scope"][0]["subagent_id"]] == "codex=gpt-5.6-sol"
    assert roster[slots["advisory"]["subagent_id"]] == "claude=claude-sonnet-5"
    # Everything else of the transaction landed in the SAME file.
    assert saved["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert saved["OPENROUTER_API_KEY"] == WIZARD_PAYLOAD["OPENROUTER_API_KEY"]
    assert saved["OUROBOROS_SAFETY_MODE"] == "light"  # fresh-install default
    # D-2: the API model slots are untouched by the preset.
    assert saved["OUROBOROS_MODEL"] == "openai/gpt-5.6-luna"
    assert onboarding.calls["supervisor"] == 1


def test_antigravity_install_succeeds_without_inventing_reviewer_seats(onboarding):
    onboarding.calls["snapshot_payload"] = {
        "daemon": {"state": "running"},
        "harnesses": [{
            "id": "agy", "status": "ok", "enabled": True,
            "models": [
                {"id": "gemini-3.8-flash-low"},
                {"id": "gemini-3.8-flash-medium"},
                {"id": "gemini-3.8-flash-high"},
            ],
        }],
        "profiles": {
            "harnessAccounts": [_profile_account("agy", "google-owner")],
            "profiles": [_profile("agy", "google-owner", kind="oauth_token")],
        },
    }

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 200, response.text
    saved = onboarding.saved()
    items = json.loads(saved[SUBAGENTS_SETTING])["items"]
    assert items[0]["route"] == {
        "kind": "agent_session",
        "target_id": "agy=gemini-3.8-flash-high",
        "credential_profile_id": "",
    }
    assert saved["OUROBOROS_REVIEWER_SLOTS"] == ""


def test_daemon_unavailable_persists_nothing_and_keeps_the_wizard_open(onboarding):
    onboarding.calls["snapshot_payload"] = {
        "daemon": {"state": "unreachable", "last_error": "connect_failed: boom"},
        "harnesses": [], "profiles": {},
    }

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["code"] == "daemon_unavailable"
    assert body["can_skip"] is True
    assert body["saved"] is False
    assert "could not be applied" in body["error"]
    # NOTHING was written: no settings file, no supervisor, no env projection.
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["supervisor"] == 0
    assert onboarding.calls["env"] == []


def test_snapshot_read_that_never_answers_is_a_typed_timeout_not_a_hang(onboarding, monkeypatch):
    """Issue #464: the owner pressed the final save and the button stayed on
    "Saving..." forever because the ONE Claudexor snapshot read blocked inside
    the owned-daemon manager. The read is bounded by the config SSOT; a read
    that outlives the bound answers the same skippable 503 the dead-engine
    case does, and nothing is written."""
    import threading
    import time

    import ouroboros.config as config
    import ouroboros.gateway.onboarding as gw_onboarding

    monkeypatch.setattr(config, "get_onboarding_snapshot_timeout_sec", lambda: 1)
    monkeypatch.setattr(gw_onboarding, "_snapshot_inflight", None)
    release = threading.Event()

    def _wedged_read():
        release.wait(30)  # never released before the bound; released at teardown
        return {"daemon": {"state": "running"}, "harnesses": [], "profiles": {}}

    monkeypatch.setattr(gw_onboarding, "_read_harness_snapshot", _wedged_read)
    started = time.monotonic()
    try:
        response = onboarding.client.post(
            "/api/onboarding/complete",
            json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
        )
    finally:
        release.set()
    elapsed = time.monotonic() - started
    assert elapsed < 10, f"completion blocked for {elapsed:.1f}s instead of timing out"
    assert response.status_code == 503, response.text
    body = response.json()
    assert body["code"] == "daemon_timeout"
    assert body["can_skip"] is True
    assert body["saved"] is False
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["supervisor"] == 0


def test_settings_document_held_past_its_bound_is_a_typed_busy_not_a_hang(onboarding, monkeypatch):
    """The second half of issue #464: a writer wedged inside the in-process
    settings-document lock must not hold the onboarding save forever. The
    lock acquisition is bounded by the config SSOT and answers 503
    ``settings_busy`` with nothing written."""
    import threading
    import time

    import ouroboros.config as config
    import ouroboros.gateway.owner_settings as owner_settings

    monkeypatch.setattr(config, "get_settings_document_lock_timeout_sec", lambda: 1)
    acquired = owner_settings._settings_document_lock.acquire(timeout=5)
    assert acquired
    started = time.monotonic()
    try:
        response = onboarding.client.post(
            "/api/onboarding/complete",
            json={**WIZARD_PAYLOAD, "subscriptionsConnected": False},
        )
    finally:
        owner_settings._settings_document_lock.release()
    elapsed = time.monotonic() - started
    assert elapsed < 10, f"completion blocked for {elapsed:.1f}s instead of timing out"
    assert response.status_code == 503, response.text
    body = response.json()
    assert body["code"] == "settings_busy"
    assert body["saved"] is False
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["supervisor"] == 0
    del threading


def test_a_persist_wedged_past_the_writer_bound_answers_the_typed_unknown(onboarding, monkeypatch):
    """Audit point 3: the INITIATING onboarding writer is capped by the same
    seam as every settings.py writer (``settings._run_settings_writer``). A
    persist wedged in its post-commit hot-reload answers 503
    ``settings_save_timeout`` with ``saved: null`` at twice the document-lock
    bound and the request RETURNS — the bytes are already on disk, which is
    exactly why neither ``true`` nor ``false`` would be honest — while the
    abandoned body still runs to completion in its thread."""
    import threading
    import time

    import ouroboros.config as config
    import ouroboros.gateway.settings as gw_settings

    monkeypatch.setattr(config, "get_settings_document_lock_timeout_sec", lambda: 0.2)
    release = threading.Event()
    finished = threading.Event()

    def _wedged(*_a, **_k):
        release.wait(10)
        finished.set()

    monkeypatch.setattr(gw_settings, "_apply_settings_save_side_effects", _wedged)
    started = time.monotonic()
    try:
        response = onboarding.client.post(
            "/api/onboarding/complete",
            json={**WIZARD_PAYLOAD, "subscriptionsConnected": False},
        )
        elapsed = time.monotonic() - started
        assert 2 * 0.2 <= elapsed < 5, elapsed
        assert not finished.is_set(), "the response must not wait for the wedged body"
        assert response.status_code == 503, response.text
        body = response.json()
        assert body["code"] == "settings_save_timeout"
        assert "saved" in body and body["saved"] is None
        assert "still running" in body["error"]
        assert onboarding.saved()[ONBOARDING_COMPLETED_KEY]
        assert onboarding.calls["supervisor"] == 1
    finally:
        release.set()
    assert finished.wait(10), "the abandoned body must still run to completion"


def test_retries_join_the_wedged_snapshot_read_instead_of_leaking_a_thread_each(onboarding, monkeypatch):
    """Adversarial finding: ``wait_for`` abandons the awaiting request, not the
    worker. On the loop's shared default executor every timed-out completion
    would consume one worker for good and, once exhausted, every other
    ``to_thread`` endpoint would queue behind the wedge. The read runs on its
    own single worker and retries JOIN the in-flight read: N attempts, ONE
    blocked thread, each answering the typed timeout within its bound."""
    import threading
    import time

    import ouroboros.config as config
    import ouroboros.gateway.onboarding as gw_onboarding

    monkeypatch.setattr(config, "get_onboarding_snapshot_timeout_sec", lambda: 1)
    monkeypatch.setattr(gw_onboarding, "_snapshot_inflight", None)
    release = threading.Event()
    started = []

    def _wedged_read():
        started.append(time.monotonic())
        release.wait(30)
        return {"daemon": {"state": "running"}, "harnesses": [], "profiles": {}}

    monkeypatch.setattr(gw_onboarding, "_read_harness_snapshot", _wedged_read)
    try:
        answers = []
        for _ in range(3):
            t0 = time.monotonic()
            response = onboarding.client.post(
                "/api/onboarding/complete",
                json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
            )
            answers.append((response.status_code, response.json()["code"], time.monotonic() - t0))
    finally:
        release.set()
    assert [a[:2] for a in answers] == [(503, "daemon_timeout")] * 3
    assert all(elapsed < 10 for _, _, elapsed in answers)
    assert len(started) == 1, "each retry must join the one in-flight read, not start another blocked thread"
    assert not onboarding.settings_path.exists()


def test_unresolvable_model_refuses_before_any_write(onboarding):
    onboarding.calls["snapshot_payload"] = {
        "daemon": {"state": "running"},
        "harnesses": [{"id": "claude", "status": "ok", "enabled": True,
                       "models": [{"id": "claude-sonnet-5"}]}],
        "profiles": {"harnessAccounts": [_native_account("claude")]},
    }

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 503, response.text
    assert response.json()["code"] == "model_not_in_discovery"
    assert not onboarding.settings_path.exists()


def test_a_signed_in_account_without_discovery_is_a_typed_failure(onboarding):
    onboarding.calls["snapshot_payload"] = {
        "daemon": {"state": "running"},
        "harnesses": [{"id": "claude", "status": "ok", "enabled": True,
                       "models": [], "models_error": "harness_unavailable"}],
        "profiles": {"harnessAccounts": [_native_account("claude")]},
    }

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "models_unavailable"
    assert not onboarding.settings_path.exists()


def test_skip_flag_completes_without_any_preset_key(onboarding):
    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True,
              "skipSubscriptionPresets": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["preset"] == {
        "applied": False, "reason": "skipped_by_owner", "harnesses": [], "receipt": {}}
    assert onboarding.calls["snapshot"] == 0  # the daemon is never even asked

    saved = onboarding.saved()
    assert saved  # onboarding DID complete
    assert not saved.get(PRESET_MARKER_KEY)
    assert not saved.get("OUROBOROS_REVIEWER_SLOTS")
    assert not saved.get("OUROBOROS_SUBAGENT_HARNESS")
    assert onboarding.calls["supervisor"] == 1


def test_no_subscription_declared_never_reads_the_daemon(onboarding):
    response = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["reason"] == "applied"
    assert onboarding.calls["snapshot"] == 0
    assert json.loads(onboarding.saved()[SUBAGENTS_SETTING])["items"]


def test_api_only_preview_is_read_only_and_never_reads_claudexor(onboarding):
    response = onboarding.client.post(
        "/api/onboarding/subagents/preview", json=dict(WIZARD_PAYLOAD),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "onboarding_default"
    assert body["available_subagents"]["enabled"] is True
    assert [row["code"] for row in body["diagnostics"]] == [
        "duplicate_api_routes_omitted",
    ]
    assert onboarding.calls["snapshot"] == 0
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["env"] == []
    assert onboarding.calls["supervisor"] == 0


def test_preview_does_not_persist_unrelated_context_mode_compat(onboarding):
    legacy_bytes = (
        b'{"OUROBOROS_CONTEXT_MODE":"low",'
        b'"OPENROUTER_API_KEY":"sk-or-v1-existing-owner-key"}\n'
    )
    onboarding.settings_path.write_bytes(legacy_bytes)

    response = onboarding.client.post(
        "/api/onboarding/subagents/preview", json=dict(WIZARD_PAYLOAD),
    )

    assert response.status_code == 200, response.text
    assert onboarding.settings_path.read_bytes() == legacy_bytes
    assert onboarding.calls["snapshot"] == 0
    assert onboarding.calls["env"] == []
    assert onboarding.calls["supervisor"] == 0


def test_local_only_preview_materializes_only_local_routes_without_daemon_read(onboarding):
    response = onboarding.client.post(
        "/api/onboarding/subagents/preview",
        json={
            **WIZARD_PAYLOAD,
            "OPENROUTER_API_KEY": "",
            "LOCAL_MODEL_SOURCE": "/models/owner.gguf",
            "LOCAL_ROUTING_MODE": "all",
            "OUROBOROS_MODEL": "owner-main",
            "OUROBOROS_MODEL_LIGHT": "owner-light",
        },
    )

    assert response.status_code == 200, response.text
    targets = [
        row["route"]["target_id"]
        for row in response.json()["available_subagents"]["items"]
    ]
    assert targets == ["owner-main (local)", "owner-light (local)"]
    assert onboarding.calls["snapshot"] == 0
    assert not onboarding.settings_path.exists()


def test_subscription_preview_reads_one_snapshot_and_still_persists_nothing(onboarding):
    response = onboarding.client.post(
        "/api/onboarding/subagents/preview",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 200, response.text
    assert onboarding.calls["snapshot"] == 1
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["supervisor"] == 0


def test_completion_validates_but_never_replaces_owner_edited_actor_rows(onboarding):
    owner = {
        "enabled": True,
        "items": [{
            "subagent_id": "owner-route",
            "name": "Owner route",
            "recommended_use": "Use exactly when the owner chooses this saved route.",
            "route": {"kind": "api_model", "target_id": "x-ai/grok-owner"},
            "effort": "high",
        }],
    }
    preview = onboarding.client.post(
        "/api/onboarding/subagents/preview",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True,
              SUBAGENTS_SETTING: owner},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["source"] == "configured"
    expected = {
        "enabled": True,
        "items": [{k: v for k, v in owner["items"][0].items() if k != "name"}],
    }
    assert preview.json()["available_subagents"] == expected
    assert not onboarding.settings_path.exists()

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True,
              SUBAGENTS_SETTING: owner},
    )
    assert response.status_code == 200, response.text
    assert json.loads(onboarding.saved()[SUBAGENTS_SETTING]) == expected
    assert response.json()["preset"]["receipt"]["source"] == "configured"
    assert onboarding.calls["snapshot"] == 2  # one independently verified snapshot per request


def test_malformed_owner_actor_draft_refuses_before_snapshot_or_write(onboarding):
    malformed = {
        "enabled": True,
        "items": [{
            "subagent_id": "dup",
            "name": "Bad",
            "recommended_use": "Bad",
            "route": {"kind": "api_model", "target_id": "x", "extra": True},
        }],
    }
    for path in ("/api/onboarding/subagents/preview", "/api/onboarding/complete"):
        response = onboarding.client.post(
            path,
            json={**WIZARD_PAYLOAD, "subscriptionsConnected": True,
                  SUBAGENTS_SETTING: malformed},
        )
        assert response.status_code == 400, response.text
        assert response.json()["code"] == "invalid_available_subagents"
    assert onboarding.calls["snapshot"] == 0
    assert not onboarding.settings_path.exists()


def test_skip_subscription_defaults_still_saves_an_explicit_owner_actor_draft(onboarding):
    owner = {
        "enabled": False,
        "items": [],
    }
    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True,
              "skipSubscriptionPresets": True, SUBAGENTS_SETTING: owner},
    )

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["reason"] == "configured_by_owner"
    saved = onboarding.saved()
    assert json.loads(saved[SUBAGENTS_SETTING]) == owner
    assert not saved.get(PRESET_MARKER_KEY)
    assert not saved.get("OUROBOROS_REVIEWER_SLOTS")
    assert onboarding.calls["snapshot"] == 0


def test_an_old_unconfigured_install_is_not_retro_presetted(onboarding):
    """The reviewer's probe: a long-lived install whose provider stopped working.

    "No startup-ready provider" is true for it too, so that predicate alone made
    the install-time window RE-OPEN and wrote the preset over reviewer/subagent
    choices the owner made themselves (D-4). A settings file that already exists
    is proof this is not a first run."""
    onboarding.settings_path.write_text(json.dumps({
        "OUROBOROS_MODEL": "openai/gpt-5.6-luna",
        "OUROBOROS_REVIEWER_SLOTS": '{"triad": [{"route": {"target_id": "mine"}}]}',
        "OUROBOROS_SUBAGENT_HARNESS": "claude=claude-sonnet-5",
        "OUROBOROS_SAFETY_MODE": "full",
    }), encoding="utf-8")

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["preset"] == {
        "applied": False, "reason": "not_install_time", "harnesses": [], "receipt": {}}
    assert onboarding.calls["snapshot"] == 0  # the daemon is not even asked
    saved = onboarding.saved()
    assert not saved.get(PRESET_MARKER_KEY)
    # The owner's OWN reviewer/subagent configuration survived untouched.
    assert saved["OUROBOROS_REVIEWER_SLOTS"] == '{"triad": [{"route": {"target_id": "mine"}}]}'
    assert saved["OUROBOROS_SUBAGENT_HARNESS"] == "claude=claude-sonnet-5"
    assert saved["OUROBOROS_SAFETY_MODE"] == "full"


def test_nonfresh_recovery_completion_still_saves_an_explicit_owner_draft(onboarding):
    onboarding.settings_path.write_text(json.dumps({
        "OUROBOROS_MODEL": "openai/gpt-5.6-luna",
        SUBAGENTS_SETTING: "",
        "OUROBOROS_REVIEWER_SLOTS": "owner-reviewer-bytes",
    }), encoding="utf-8")
    owner = {
        "enabled": True,
        "items": [{
            "subagent_id": "owner-route",
            "name": "Owner route",
            "recommended_use": "Use this exact recovery draft.",
            "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"},
        }],
    }

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={
            **WIZARD_PAYLOAD,
            "subscriptionsConnected": True,
            SUBAGENTS_SETTING: owner,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["reason"] == "configured_by_owner"
    assert response.json()["preset"]["applied"] is True
    assert onboarding.calls["snapshot"] == 0
    saved = onboarding.saved()
    expected = {
        "enabled": True,
        "items": [{k: v for k, v in owner["items"][0].items() if k != "name"}],
    }
    assert json.loads(saved[SUBAGENTS_SETTING]) == expected
    assert saved["OUROBOROS_REVIEWER_SLOTS"] == "owner-reviewer-bytes"
    assert not saved.get(PRESET_MARKER_KEY)


def test_every_completion_records_the_durable_onboarding_fact(onboarding):
    """Including one that connected nothing: absence of a provider must never be
    able to re-open the install-time window later."""
    response = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["reason"] == "applied"
    assert onboarding.saved()[ONBOARDING_COMPLETED_KEY]


def test_a_recorded_completion_closes_the_window_even_without_a_settings_file(onboarding):
    """The marker is the durable fact, checked before anything else: an install
    that once completed onboarding is not install-time again."""
    from ouroboros.gateway.onboarding import preset_eligible

    assert preset_eligible({}) is True
    assert preset_eligible({ONBOARDING_COMPLETED_KEY: "2026-08-09T00:00:00Z"}) is False
    assert preset_eligible({PRESET_MARKER_KEY: "1"}) is False


def test_an_environment_completion_fact_cannot_close_the_install_window(monkeypatch,
                                                                        onboarding):
    """FINDING 1 at the onboarding endpoint, exactly as probed.

    The completion fact and the preset marker are supposed to be authored by this
    endpoint alone, and the request-body merge skip enforced that for the BODY.
    The environment was still an authority: `load_settings` overlaid an
    environment timestamp, so on a genuinely FRESH install (no settings.json,
    connected subscriptions) the endpoint answered `not_install_time`, made ZERO
    daemon calls, and closed the onboarding window for good without ever
    installing the presets the owner connected accounts for."""
    monkeypatch.setenv(ONBOARDING_COMPLETED_KEY, "2020-01-01T00:00:00Z")
    monkeypatch.setenv(PRESET_MARKER_KEY, "1")
    assert not onboarding.settings_path.exists()

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preset"]["reason"] == "applied", body
    assert body["preset"]["applied"] is True
    assert onboarding.calls["snapshot"] == 1, "the daemon was never consulted"
    saved = onboarding.saved()
    assert saved[PRESET_MARKER_KEY] == "3"
    # The endpoint's own timestamp, not the environment's.
    assert saved[ONBOARDING_COMPLETED_KEY] != "2020-01-01T00:00:00Z"


def test_onboarding_validation_refusals_say_saved_false(onboarding):
    """FINDING 2 on this surface: an early validation refusal answered a bare
    `{"error": ...}`, so a client could not tell it apart from an old envelope."""
    malformed = onboarding.client.post("/api/onboarding/complete", json=["nope"])
    assert malformed.status_code == 400, malformed.text
    assert malformed.json()["saved"] is False, malformed.text

    no_provider = onboarding.client.post(
        "/api/onboarding/complete",
        json={"OUROBOROS_RUNTIME_MODE": "advanced", "subscriptionsConnected": True})
    assert no_provider.status_code == 400, no_provider.text
    assert no_provider.json()["saved"] is False, no_provider.text
    assert not onboarding.settings_path.exists()


def test_subscription_free_completion_still_materializes_actor_default(onboarding):
    """API/local-only first installs compile actors without reading Claudexor."""
    response = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))
    assert response.status_code == 200, response.text
    assert response.json()["preset"]["applied"] is True
    saved = onboarding.saved()
    assert saved[ONBOARDING_COMPLETED_KEY], "the completion fact IS unconditional"
    assert saved.get(PRESET_MARKER_KEY) == "3"
    assert json.loads(saved[SUBAGENTS_SETTING])["items"]


def test_generic_settings_save_cannot_author_or_clear_the_completion_fact():
    from ouroboros.gateway.settings import _merge_settings_payload

    merged = _merge_settings_payload({ONBOARDING_COMPLETED_KEY: "2026-08-09T00:00:00Z"},
                                     {ONBOARDING_COMPLETED_KEY: ""})
    assert merged[ONBOARDING_COMPLETED_KEY] == "2026-08-09T00:00:00Z"
    fresh = _merge_settings_payload({}, {ONBOARDING_COMPLETED_KEY: "2026-08-09T00:00:00Z"})
    assert not fresh.get(ONBOARDING_COMPLETED_KEY)


def test_a_contended_lock_writes_nothing_and_starts_nothing(onboarding):
    """FINDING 1 (the class): `_acquire_settings_lock` answers None on timeout.

    The write used to run the precondition and `atomic_write_json` ANYWAY, so a
    contended save was the one save that skipped the check it advertises. The
    lock is now a precondition of the write itself."""
    import os

    lock_path = onboarding.settings_path.with_name(onboarding.settings_path.name + ".lock")
    foreign_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        response = onboarding.client.post(
            "/api/onboarding/complete",
            json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})
    finally:
        os.close(foreign_fd)

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["code"] == "settings_locked"
    assert body["saved"] is False
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["env"] == []
    assert onboarding.calls["supervisor"] == 0
    assert lock_path.exists()  # the other holder's lock was neither taken nor removed


def test_the_precondition_never_runs_without_the_lock(monkeypatch, tmp_path):
    """The same class, at the seam every owner endpoint shares."""
    import os

    import ouroboros.config as config
    from ouroboros.gateway.owner_settings import SettingsLockUnavailable, _owner_write_settings

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    checked: list = []
    foreign_fd = os.open(str(tmp_path / "settings.json.lock"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with pytest.raises(SettingsLockUnavailable):
            _owner_write_settings({"TOTAL_BUDGET": 10.0},
                                  precondition=lambda: checked.append(1) or "")
    finally:
        os.close(foreign_fd)

    assert checked == []
    assert not settings_path.exists()


def test_a_post_commit_failure_reports_the_save_that_landed(monkeypatch, onboarding):
    """FINDING 4: the commit boundary. The bytes are on disk before the
    supervisor starts, so a supervisor failure is its own fact — reporting
    `saved=False` sent the owner back through a completed onboarding."""
    import ouroboros.gateway.settings as gw_settings

    def _boom(_request, _settings):
        raise RuntimeError("supervisor refused to start")

    monkeypatch.setattr(gw_settings, "_start_supervisor_if_needed_for_request", _boom)

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})

    assert response.status_code == 500, response.text
    body = response.json()
    assert body["saved"] is True
    assert body["status"] == "saved_with_post_commit_error"
    assert body["post_commit_failed"] == "supervisor start"
    assert "supervisor refused to start" in body["error"]
    # And the transaction really IS on disk, preset marker included.
    saved = onboarding.saved()
    assert saved[PRESET_MARKER_KEY] == "3"
    assert saved[ONBOARDING_COMPLETED_KEY]
    assert saved["OPENROUTER_API_KEY"] == WIZARD_PAYLOAD["OPENROUTER_API_KEY"]


def test_the_preset_save_does_not_stall_on_its_own_lock(onboarding):
    """FINDING 5: the precondition re-read used to take the settings lock a
    second time. It is not re-entrant, so every successful preset save burned
    the full 2s timeout before reading the file it was already holding."""
    import time

    started = time.monotonic()
    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})
    elapsed = time.monotonic() - started

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["applied"] is True
    assert elapsed < 1.0, f"the preset save waited {elapsed:.2f}s on a lock it held"


def test_a_desktop_shaped_completion_still_authors_light_on_a_fresh_install(onboarding):
    """D-8 closure proof. The desktop `save_wizard` bridge existed for exactly
    ONE reason: a genuinely fresh install had to author the new-install `light`
    safety coverage, and neither the shared validator nor the generic settings
    endpoint may lower safety. Removing that bridge is only honest if the same
    completion the desktop window now posts still lands `light`."""
    assert not onboarding.settings_path.exists()   # genuinely fresh

    response = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))

    assert response.status_code == 200, response.text
    assert onboarding.saved()["OUROBOROS_SAFETY_MODE"] == "light"


def test_a_completion_over_an_existing_install_can_never_lower_safety(onboarding):
    """The other half: the ratchet. With a settings file already on disk the
    install is not fresh, so completion authors nothing — and a payload that
    tries to carry the lowering itself gets no further, because
    OUROBOROS_SAFETY_MODE is not a wizard field at all."""
    onboarding.settings_path.write_text(
        json.dumps({"OUROBOROS_SAFETY_MODE": "full", "OPENROUTER_API_KEY": "sk-or-v1-existing"}),
        encoding="utf-8")

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "OUROBOROS_SAFETY_MODE": "light"})

    assert response.status_code == 200, response.text
    assert onboarding.saved()["OUROBOROS_SAFETY_MODE"] == "full"


def test_old_install_is_not_retro_presetted(onboarding):
    """An install that already has a provider is past install time — the missing
    marker is NOT permission to preset it (every pre-preset install lacks one)."""
    onboarding.settings_path.write_text(json.dumps({
        "OPENROUTER_API_KEY": "sk-or-v1-existing-install-key",
        "OUROBOROS_MODEL": "openai/gpt-5.6-luna",
        "OUROBOROS_SAFETY_MODE": "full",
    }), encoding="utf-8")

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["applied"] is False
    assert response.json()["preset"]["reason"] == "not_install_time"
    assert onboarding.calls["snapshot"] == 0
    saved = onboarding.saved()
    assert not saved.get(PRESET_MARKER_KEY)
    assert not saved.get("OUROBOROS_REVIEWER_SLOTS")
    # The fresh-install safety default is likewise not re-authored over it.
    assert saved["OUROBOROS_SAFETY_MODE"] == "full"


def test_a_presetted_install_is_never_presetted_twice(onboarding):
    first = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})
    assert first.json()["preset"]["applied"] is True

    second = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})

    assert second.status_code == 200, second.text
    assert second.json()["preset"]["reason"] == "not_install_time"
    assert onboarding.calls["snapshot"] == 1  # still only the FIRST read


def test_install_time_status_cannot_be_forged_from_the_payload(onboarding):
    """A browser boolean is a request, never an authority: neither the marker
    nor a claimed subscription can make a configured install install-time."""
    onboarding.settings_path.write_text(json.dumps({
        "OPENROUTER_API_KEY": "sk-or-v1-existing-install-key",
        "OUROBOROS_MODEL": "openai/gpt-5.6-luna",
    }), encoding="utf-8")

    response = onboarding.client.post("/api/onboarding/complete", json={
        **WIZARD_PAYLOAD,
        "subscriptionsConnected": True,
        PRESET_MARKER_KEY: "",              # try to clear the latch
        "OUROBOROS_SUBSCRIPTION_PRESET_VERSION ": "1",
        "OUROBOROS_REVIEWER_SLOTS": '{"triad": [], "scope": []}',
        "OUROBOROS_SAFETY_MODE": "off",
    })

    assert response.status_code == 200, response.text
    assert response.json()["preset"]["applied"] is False
    saved = onboarding.saved()
    assert not saved.get(PRESET_MARKER_KEY)
    # Neither the reviewer slots nor safety mode ride through the wizard payload:
    # the shared setup validator only copies the setup contract's own keys.
    assert not saved.get("OUROBOROS_REVIEWER_SLOTS")
    assert saved.get("OUROBOROS_SAFETY_MODE", "") != "off"


def test_subscription_alone_does_not_satisfy_the_launch_gate(onboarding):
    """D-1: at least one API key or a local model. A subscription amplifies.

    The shared setup validator is the first gate and refuses with its own
    provider-list wording; nothing is written and the daemon is never asked."""
    payload = {k: v for k, v in WIZARD_PAYLOAD.items() if k != "OPENROUTER_API_KEY"}

    response = onboarding.client.post(
        "/api/onboarding/complete", json={**payload, "subscriptionsConnected": True})

    assert response.status_code == 400, response.text
    assert "local model" in response.json()["error"]
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["snapshot"] == 0


def test_startup_gate_is_re_checked_after_normalization(monkeypatch, onboarding):
    """Defence in depth on the SAME invariant: even if a payload passed the
    shared validator, an install that would not be startup-ready is refused
    before anything is saved — a subscription never fills that gap."""
    import ouroboros.gateway.onboarding as gw_onboarding

    monkeypatch.setattr(
        gw_onboarding, "apply_runtime_provider_defaults",
        lambda settings: ({k: ("" if k == "OPENROUTER_API_KEY" else v)
                           for k, v in settings.items()}, False, []))

    response = onboarding.client.post(
        "/api/onboarding/complete", json={**WIZARD_PAYLOAD, "subscriptionsConnected": True})

    assert response.status_code == 400, response.text
    assert "API key or a local model" in response.json()["error"]
    assert not onboarding.settings_path.exists()
    assert onboarding.calls["snapshot"] == 0


def test_non_object_body_is_refused(onboarding):
    response = onboarding.client.post("/api/onboarding/complete", json=["nope"])

    assert response.status_code == 400
    assert not onboarding.settings_path.exists()


def test_running_process_keeps_its_boot_runtime_mode(onboarding):
    """The pending next-boot mode is persisted; the live env is not elevated —
    the same split the two-write flow had."""
    from ouroboros.config import get_runtime_mode

    response = onboarding.client.post(
        "/api/onboarding/complete", json={**WIZARD_PAYLOAD, "OUROBOROS_RUNTIME_MODE": "pro"})

    assert response.status_code == 200, response.text
    assert response.json()["runtime_mode"] == "pro"
    assert onboarding.saved()["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert onboarding.calls["env"][0]["OUROBOROS_RUNTIME_MODE"] == get_runtime_mode()


# ---------------------------------------------------------------------------
# Pure helpers (no server).
# ---------------------------------------------------------------------------


def _routable_snapshot(accounts, profiles=(), harnesses=None):
    return {
        "daemon": {"state": "running"},
        "harnesses": harnesses if harnesses is not None else [
            {"id": "claude", "status": "ok", "enabled": True, "models": [{"id": "claude-opus-5"}]},
            {"id": "codex", "status": "ok", "enabled": True, "models": [{"id": "gpt-5.6-sol"}]},
            {"id": "cursor", "status": "ok", "enabled": True,
             "models": [{"id": "cursor-grok-4.6-high"}]},
        ],
        "profiles": {"harnessAccounts": list(accounts), "profiles": list(profiles)},
    }


def test_only_daemon_routable_accounts_become_discoveries():
    """The engine's OWN answer decides: `next_up` says which credential an
    unpinned run would take, and only a subscription seat counts."""
    from ouroboros.gateway.onboarding import verified_harness_discoveries

    snapshot = _routable_snapshot(
        # claude: the live CLI session. codex: the engine would rotate onto a
        # verified subscription profile. cursor: signed in nowhere.
        [_native_account("claude"),
         _profile_account("codex", "koshak"),
         {"harness_id": "cursor", "native_credentials_enabled": True,
          "native_login_detected": False, "identity": None,
          "next_up": {"kind": "none", "reason": "the default credential is not ready"}}],
        [_profile("codex", "koshak")],
    )

    discoveries, failure = verified_harness_discoveries(snapshot)

    assert failure is None
    assert [d.harness_id for d in discoveries] == ["claude", "codex"]


def test_an_api_key_profile_is_never_counted_as_a_subscription():
    """The reviewer's probe: `credential_kind=api_key` with a PASSED
    verification. Presetting it would put agent_session reviewer rows on a lane
    that bills the owner's API key — exactly what D-3 forbids."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_routable_snapshot(
        [_profile_account("claude", "byok")],
        [_profile("claude", "byok", kind="api_key", availability="unavailable")],
    ))

    assert routable == {}
    assert "API credential" in refused["claude"]


def test_a_native_api_key_route_is_never_counted_as_a_subscription():
    """Same class on the OTHER seat: the default account of an API-key-only
    harness is a detected login too — the engine names its route `api_key`."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(
        _routable_snapshot([_native_account("codex", route="api_key")]))

    assert routable == {}
    assert "API key" in refused["codex"]


def test_a_disabled_or_unavailable_harness_is_refused_even_when_signed_in():
    """The other half of the reviewer's probe: the harness row itself. An
    engine that cannot run the harness cannot run a subscription session on it,
    however healthy the account looks."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_routable_snapshot(
        [_native_account("claude"), _native_account("codex")],
        harnesses=[
            {"id": "claude", "status": "ok", "enabled": False, "models": [{"id": "claude-opus-5"}]},
            {"id": "codex", "status": "unavailable", "enabled": True,
             "models": [{"id": "gpt-5.6-sol"}]},
        ],
    ))

    assert routable == {}
    assert refused == {"claude": "the engine has this harness disabled",
                       "codex": "the engine reports it unavailable"}


def test_an_unverified_or_disabled_profile_seat_is_refused():
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    unverified = subscription_routable_harnesses(_routable_snapshot(
        [_profile_account("cursor", "koshakcot-ultra")],
        [_profile("cursor", "koshakcot-ultra", availability="unavailable",
                  verification="not_run")]))
    disabled = subscription_routable_harnesses(_routable_snapshot(
        [_profile_account("cursor", "sol-validator")],
        [_profile("cursor", "sol-validator", enabled=False)]))
    missing = subscription_routable_harnesses(
        _routable_snapshot([_profile_account("cursor", "ghost")]))

    assert unverified[0] == {} and "unavailable" in unverified[1]["cursor"]
    assert disabled[0] == {} and "disabled" in disabled[1]["cursor"]
    assert missing[0] == {} and "does not list" in missing[1]["cursor"]


# The engine's own `next_up` refusal strings (Claudexor
# `orchestrator/credential-profiles.ts::nextUpIdentity`), copied verbatim.
_NOT_READY_NOW = ("the default credential is not ready; refresh Accounts or run "
                  "`claudexor doctor`")
_ROUTE_UNKNOWN_NOW = ("the default credential route is unknown; refresh Accounts or run "
                      "`claudexor doctor`")


def _no_capacity_now(harness, *, native_login):
    """A harness whose subscription IS configured and whose CURRENT routing answer
    is a refusal — the shape of a spent window at the moment of onboarding."""
    return {"harness_id": harness, "native_credentials_enabled": True,
            "native_login_detected": native_login,
            "identity": {"email": "owner@example.com", "plan": "claude_max"}
                        if native_login else None,
            "next_up": {"kind": "none",
                        "reason": _NOT_READY_NOW if native_login else _ROUTE_UNKNOWN_NOW}}


def test_a_routing_refusal_now_does_not_delete_a_subscription_from_the_preset():
    """FINDING 5. `next_up` is the daemon's answer to "who would an unpinned run
    take RIGHT NOW", computed from enabled profiles + default readiness + QUOTA
    (Claudexor INV-135), and the engine documents it as informational — it never
    gates routing. The preset is a once-only install-time decision (D-4) that
    never runs again, so deciding it on that answer meant an owner who connected
    Claude and Codex during an hour when the Claude window happened to be spent
    got a Codex-only preset PERMANENTLY, with no seam left to revisit it. D-3
    says an exhausted subscription row stays CONFIGURED and waits for capacity.

    Two shapes, both with the engine's verbatim reason strings: a signed-in
    default session the engine will not route this minute, and a harness whose
    only seat is a named subscription profile (the default `limit_action: fail`
    policy never names a profile in `next_up`, so this is the ordinary Cursor
    account shape)."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_routable_snapshot(
        [_no_capacity_now("claude", native_login=True),
         _no_capacity_now("cursor", native_login=False)],
        [_profile("cursor", "sol-validator")],
    ))

    assert refused == {}
    assert set(routable) == {"claude", "cursor"}
    # The evidence stays honest: the seat, and the engine's own reason beside it.
    assert routable["claude"] == f"signed-in default session; no capacity right now ({_NOT_READY_NOW})"
    assert routable["cursor"] == (
        f"account 'sol-validator' (config_dir_login); no capacity right now ({_ROUTE_UNKNOWN_NOW})")


def test_a_seat_with_no_capacity_still_reaches_the_compiled_preset_with_its_models():
    """The same finding one layer up: the harness must not merely be 'routable',
    it must survive into the compiled preset with its models resolved — which is
    what the owner's reviewer and subagent rows are actually written from."""
    from ouroboros.gateway.onboarding import verified_harness_discoveries

    snapshot = _routable_snapshot(
        [_no_capacity_now("claude", native_login=True), _native_account("codex")],
        harnesses=[
            {"id": "claude", "status": "ok", "enabled": True,
             "models": [{"id": "claude-opus-5"}, {"id": "claude-sonnet-5"},
                        {"id": "claude-fable-5"}, {"id": "claude-opus-4-6"}]},
            {"id": "codex", "status": "ok", "enabled": True,
             "models": [{"id": "gpt-5.6-sol"}, {"id": "gpt-5.6-terra"}, {"id": "gpt-5.5"}]},
        ],
    )

    discoveries, failure = verified_harness_discoveries(snapshot)

    assert failure is None
    assert [d.harness_id for d in discoveries] == ["claude", "codex"]
    assert "claude-opus-5" in dict((d.harness_id, d.model_ids) for d in discoveries)["claude"]


# ---------------------------------------------------------------------------
# The UNIFIED account model's wire (sprint plan §K.7 + frozen contract §L):
# `harnessAccounts` arrives EMPTY (every account a named registry row) and the
# routing verdict rides the additive `accountPools: [{harness_id, next_up}]`.
# ---------------------------------------------------------------------------


def _unified_snapshot(pools, profiles=(), harnesses=None, legacy_accounts=()):
    snapshot = _routable_snapshot(list(legacy_accounts), profiles, harnesses)
    snapshot["profiles"]["accountPools"] = list(pools)
    return snapshot


def _pool(harness, next_up):
    return {"harness_id": harness, "next_up": next_up}


def test_a_unified_engine_routes_through_account_pools_not_the_empty_legacy_key():
    """The dual-read's load-bearing half: on a unified engine the legacy
    `harnessAccounts` is `[]`, so the old accounts.get(harness) authority is
    absent — skipping there would silently drop EVERY unified harness from the
    preset. The pool row is the accounts authority on that wire."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_unified_snapshot(
        [_pool("claude", {"kind": "profile", "profileId": "claude-default"}),
         _pool("codex", {"kind": "api_key_route"})],
        [_profile("claude", "claude-default")],
    ))

    assert routable == {"claude": "account 'claude-default' (config_dir_login)"}
    # The pool union's explicit API-key verdict is a refusal, same wording
    # class as the legacy native api_key route.
    assert "API key" in refused["codex"]
    # No pool row and no legacy row: silent absence, exactly as before.
    assert "cursor" not in routable and "cursor" not in refused


def test_a_unified_refusal_still_keeps_a_configured_named_seat_in_the_preset():
    """D-3 through the new wire: a spent window at onboarding time must not
    delete a configured subscription from the once-only preset. On the unified
    wire the configured-seat fallback is the named-profile scan ALONE — there
    is no native fact to read."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_unified_snapshot(
        [_pool("claude", {"kind": "none", "reason": "every window is spent"}),
         _pool("cursor", {"kind": "none", "reason": "nothing routable"})],
        [_profile("claude", "claude-default")],
    ))

    assert routable == {
        "claude": "account 'claude-default' (config_dir_login); "
                  "no capacity right now (every window is spent)"}
    assert refused == {"cursor": "nothing routable"}


def test_an_unknown_pool_kind_is_fail_safe_and_the_seat_scan_still_answers():
    """Either wire may grow a spelling this reader predates (plan §E: degrade
    honestly). An unknown kind is never a subscription verdict; a configured
    named profile still counts through the seat scan, and a harness with none
    is refused with the unknown state named."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_unified_snapshot(
        [_pool("claude", {"kind": "quantum_pool"}),
         _pool("codex", {"kind": "quantum_pool"})],
        [_profile("claude", "claude-default")],
    ))

    assert set(routable) == {"claude"}
    assert "unknown routing state 'quantum_pool'" in routable["claude"]
    assert "unknown routing state 'quantum_pool'" in refused["codex"]


def test_when_both_wires_carry_a_verdict_the_pool_wins():
    """A transitional payload may carry both. Profiles own account facts, the
    pool owns routing facts — one authority, or the two could disagree about
    the same harness."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    routable, refused = subscription_routable_harnesses(_unified_snapshot(
        [_pool("codex", {"kind": "api_key_route"})],
        legacy_accounts=[_native_account("codex")],  # legacy says: healthy session
    ))

    assert routable == {}
    assert "API key" in refused["codex"]


def test_out_of_capacity_never_launders_an_api_key_or_an_unusable_seat():
    """The round-one finding must not regress through the new fallback: being out
    of capacity is not evidence of being a subscription. A harness whose only
    seats are API credentials, disabled or unverified stays refused."""
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    # The default login exists, but the harness's auth_preference puts an API key
    # ahead of it — durable, and exactly what D-3 forbids.
    api_route = subscription_routable_harnesses(
        _routable_snapshot([_native_account("codex", route="api_key")],
                           [_profile("codex", "koshak", kind="api_key")]))
    # No native login at all; the only named seats are unusable.
    only_bad_profiles = subscription_routable_harnesses(_routable_snapshot(
        [_profile_account("cursor", "ghost")],
        [_profile("cursor", "byok", kind="api_key"),
         _profile("cursor", "off", enabled=False),
         _profile("cursor", "new", verification="not_run")]))

    assert api_route[0] == {} and "API key" in api_route[1]["codex"]
    assert only_bad_profiles[0] == {} and "does not list" in only_bad_profiles[1]["cursor"]


def test_no_routable_account_is_a_typed_failure_that_names_the_reason():
    from ouroboros.gateway.onboarding import verified_harness_discoveries

    discoveries, failure = verified_harness_discoveries(_routable_snapshot(
        [_native_account("claude", detected=False)]))

    assert discoveries == ()
    assert failure is not None and failure.code == "no_verified_account"
    assert "no signed-in session" in failure.detail


def test_an_engine_with_no_accounts_authority_vouches_nothing():
    from ouroboros.gateway.onboarding import verified_harness_discoveries

    discoveries, failure = verified_harness_discoveries({
        "daemon": {"state": "running"}, "harnesses": [], "profiles": {}})

    assert discoveries == ()
    assert failure is not None and failure.code == "no_verified_account"


def test_generic_settings_save_cannot_author_or_clear_preset_facts():
    """Marker and receipt are endpoint-authored: Settings neither writes nor wipes."""
    from ouroboros.gateway.settings import _merge_settings_payload

    merged = _merge_settings_payload(
        {PRESET_MARKER_KEY: "1", SUBAGENTS_RECEIPT_KEY: "old", "TOTAL_BUDGET": 10.0},
        {PRESET_MARKER_KEY: "99", SUBAGENTS_RECEIPT_KEY: "new", "TOTAL_BUDGET": 20.0},
    )

    assert merged[PRESET_MARKER_KEY] == "1"
    assert merged[SUBAGENTS_RECEIPT_KEY] == "old"
    assert merged["TOTAL_BUDGET"] == 20.0

    fresh = _merge_settings_payload(
        {"TOTAL_BUDGET": 10.0},
        {PRESET_MARKER_KEY: "1", SUBAGENTS_RECEIPT_KEY: "new"},
    )
    assert not fresh.get(PRESET_MARKER_KEY)
    assert not fresh.get(SUBAGENTS_RECEIPT_KEY)


def test_get_onboarding_read_writes_nothing(monkeypatch, tmp_path):
    """A READ must never be the first author of settings.json (D-8)."""
    import ouroboros.config as config
    import ouroboros.gateway.settings as gw_settings

    for key in _PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)

    app = Starlette(routes=[Route("/api/onboarding", endpoint=gw_settings.api_onboarding)])
    with TestClient(app) as client:
        first = client.get("/api/onboarding")

    assert first.status_code == 200  # the blocking overlay
    assert not settings_path.exists(), "GET /api/onboarding created settings.json"

    # And with a settings file present, its bytes and mtime are untouched.
    settings_path.write_text(json.dumps({"OUROBOROS_MODEL": "openai/gpt-5.6-luna"}),
                             encoding="utf-8")
    before_bytes = settings_path.read_bytes()
    before_mtime = settings_path.stat().st_mtime_ns
    with TestClient(app) as client:
        second = client.get("/api/onboarding")

    assert second.status_code == 200
    assert settings_path.read_bytes() == before_bytes
    assert settings_path.stat().st_mtime_ns == before_mtime


# ---------------------------------------------------------------------------
# The freshness seam: completion derives the WHOLE document from an unlocked
# read and writes that whole dictionary back, so a concurrent owner write
# landing in between would be reverted key by key while the owner is told the
# save succeeded. These drive the real endpoint through the real persist path.
# ---------------------------------------------------------------------------

_WATCHED = "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"


def _write_concurrently(path, **values):
    """Another owner write, exactly as a second process would leave the file."""
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    path.write_text(json.dumps(current, indent=1), encoding="utf-8")


def test_a_concurrent_owner_write_is_refused_and_survives(onboarding, monkeypatch):
    """The defect this seam exists for: the owner flips a setting elsewhere while
    onboarding is being saved, and the save silently puts it back."""
    import ouroboros.gateway.onboarding as gw_onboarding

    onboarding.settings_path.write_text(
        json.dumps({"OPENROUTER_API_KEY": "sk-or-v1-existing", _WATCHED: "true"}, indent=1),
        encoding="utf-8")

    # Land the concurrent write in the window the fingerprint covers: after this
    # request read the document, before it takes the lock.
    real_prepared = gw_onboarding._prepared_settings

    def _prepared_then_interfere(body):
        result = real_prepared(body)
        _write_concurrently(onboarding.settings_path, **{_WATCHED: "false"})
        return result

    monkeypatch.setattr(gw_onboarding, "_prepared_settings", _prepared_then_interfere)

    response = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))

    assert response.status_code == 409
    body = response.json()
    assert body["saved"] is False
    assert body["code"] == "onboarding_state_changed"
    # The owner's change is still on disk, and onboarding did NOT complete.
    saved = json.loads(onboarding.settings_path.read_text(encoding="utf-8"))
    assert saved[_WATCHED] == "false"
    assert ONBOARDING_COMPLETED_KEY not in saved


def test_the_advised_retry_actually_succeeds(onboarding, monkeypatch):
    """The refusal tells the owner to finish again, so that must work — and it
    must carry the concurrent value through rather than reverting it."""
    import ouroboros.gateway.onboarding as gw_onboarding

    onboarding.settings_path.write_text(
        json.dumps({"OPENROUTER_API_KEY": "sk-or-v1-existing", _WATCHED: "true"}, indent=1),
        encoding="utf-8")

    real_prepared = gw_onboarding._prepared_settings
    interfered = {"done": False}

    def _interfere_once(body):
        result = real_prepared(body)
        if not interfered["done"]:
            interfered["done"] = True
            _write_concurrently(onboarding.settings_path, **{_WATCHED: "false"})
        return result

    monkeypatch.setattr(gw_onboarding, "_prepared_settings", _interfere_once)

    assert onboarding.client.post(
        "/api/onboarding/complete", json=dict(WIZARD_PAYLOAD)).status_code == 409
    second = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))

    assert second.status_code == 200
    saved = json.loads(onboarding.settings_path.read_text(encoding="utf-8"))
    assert saved[_WATCHED] == "false", "the retry must not revert the concurrent change either"
    assert ONBOARDING_COMPLETED_KEY in saved


def test_an_uncontended_save_is_never_refused(onboarding):
    """The check must not cost the ordinary install anything."""
    onboarding.settings_path.write_text(
        json.dumps({"OPENROUTER_API_KEY": "sk-or-v1-existing"}, indent=1), encoding="utf-8")

    response = onboarding.client.post("/api/onboarding/complete", json=dict(WIZARD_PAYLOAD))

    assert response.status_code == 200
    assert ONBOARDING_COMPLETED_KEY in json.loads(
        onboarding.settings_path.read_text(encoding="utf-8"))


def test_a_connected_agy_account_composes_with_core_reviewers(onboarding):
    # The REAL agy wire shape (Claudexor INV-135): a harness with no default
    # credential store reports its harness ROW "unavailable" structurally and
    # its next_up as kind="none" (the default credential is never ready), so
    # the routable verdict comes from the CONFIGURED-profile fallback over the
    # verified named profile. The old fixture's status:"ok" native agy account
    # cannot occur.
    onboarding.calls["snapshot_payload"] = {
        "daemon": {"state": "running"},
        "harnesses": [
            {"id": "claude", "status": "ok", "enabled": True,
             "models": [{"id": "claude-opus-5"}, {"id": "claude-sonnet-5"},
                        {"id": "claude-fable-5"}, {"id": "claude-opus-4-6"}]},
            {"id": "agy", "status": "unavailable", "enabled": True,
             "models": [{"id": "gemini-3.8-flash-high"}, {"id": "gemini-3.1-pro-high"}]},
        ],
        "profiles": {
            # next_up kind="none": the engine returns none whenever the
            # DEFAULT credential is not ready, which is agy's permanent state —
            # the routable verdict must come from the configured-profile
            # fallback, not from a profile-kind next_up.
            "harnessAccounts": [
                _native_account("claude"),
                {"harness_id": "agy", "native_credentials_enabled": True,
                 "native_login_detected": False, "identity": None,
                 "next_up": {"kind": "none", "reason": "no default credential store"}},
            ],
            "profiles": [_profile("agy", "work")],
        },
    }

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True},
    )

    assert response.status_code == 200, response.text
    saved = onboarding.saved()
    actor_targets = [
        row["route"]["target_id"]
        for row in json.loads(saved[SUBAGENTS_SETTING])["items"]
    ]
    assert "agy=gemini-3.8-flash-high" in actor_targets
    reviewer = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
    roster = {
        row["subagent_id"]: row["route"]["target_id"]
        for row in json.loads(saved[SUBAGENTS_SETTING])["items"]
    }
    triad_targets = [
        roster[row["subagent_id"]] if "subagent_id" in row
        else row["route"]["target_id"]
        for row in reviewer["triad"]
    ]
    assert all("agy=" not in target for target in triad_targets)
    assert onboarding.calls["supervisor"] == 1


def test_an_unavailable_harness_row_with_a_verified_named_profile_is_routable():
    # The runnability proof for a named profile is its own doctor probe, not
    # the harness row: agy's row is STRUCTURALLY "unavailable" (no default
    # credential store, Claudexor INV-135), and refusing on the row alone made
    # the Agy task-actor policy unreachable on the real wire shape. A
    # native default seat still requires a runnable row (its only signal).
    from ouroboros.gateway.onboarding import subscription_routable_harnesses

    snapshot = {
        "harnesses": [{"id": "agy", "status": "unavailable", "enabled": True}],
        "profiles": {
            "harnessAccounts": [_profile_account("agy", "work")],
            "profiles": [_profile("agy", "work")],
        },
    }
    routable, refused = subscription_routable_harnesses(snapshot)
    assert "agy" in routable, refused
    assert "no default credential store" in routable["agy"]

    # Same row, but only a signed-in DEFAULT session: still refused — the
    # harness row is that seat's only runnability signal.
    snapshot = {
        "harnesses": [{"id": "agy", "status": "unavailable", "enabled": True}],
        "profiles": {"harnessAccounts": [_native_account("agy")], "profiles": []},
    }
    routable, refused = subscription_routable_harnesses(snapshot)
    assert "agy" not in routable
    assert refused["agy"] == "the engine reports it unavailable"
