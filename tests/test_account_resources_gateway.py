"""Browser-to-owned-engine resource contract, without a daemon or provider call."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.claudexor_accounts import api_claudexor_status
from ouroboros.gateway.claudexor_quota import api_claudexor_account_reset, api_claudexor_quota_refresh
from ouroboros.gateways.claudexor import account_resource_capabilities

WIRE = json.loads((Path(__file__).resolve().parents[1]
                   / "web/tests/fixtures/account_resources.json").read_text(encoding="utf-8"))


@pytest.fixture
def engine(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateways import claudexor as wire

    calls, clients = [], []
    state = {"operations": deepcopy(WIRE["operations"]), "error": None}
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "owned")
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(status_dict=lambda: {"state": "running"}))

    def forbidden(*_args, **_kwargs):
        pytest.fail("resource reads and controls cannot start a daemon or inspect the operator home")

    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running", forbidden)
    monkeypatch.setattr(wire, "discover_daemon", forbidden)
    endpoint = wire.DaemonEndpoint("127.0.0.1", 1, "private-fixture-token")

    def discover(path):
        assert path == tmp_path / "owned"
        return endpoint

    monkeypatch.setattr(wire, "discover_daemon_at", discover)

    def respond(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, dict(request.url.params), body,
                      request.headers.get("Idempotency-Key")))
        assert request.headers["Authorization"] == "Bearer private-fixture-token"
        if request.url.path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": True, "protocolMajor": 3,
                                            "engine": {"version": wire.CLAUDEXOR_MIN_VERSION}})
        if request.url.path == "/v2/operations":
            if state.get("catalog_error") == "transport":
                raise httpx.ReadTimeout("catalog timeout", request=request)
            if state.get("catalog_error"):
                return httpx.Response(503, json={"code": "catalog_unavailable",
                    "message": "Operation catalog could not be read", "requiredActions": ["retry_catalog"]},
                    headers={"Retry-After": "7"})
            return httpx.Response(200, json={"operations": state["operations"]})
        if state["error"] and request.url.path.startswith("/v2/account-resets"):
            return httpx.Response(state["error"], json={"code": "idempotency_conflict",
                "message": "The key belongs to another request", "requiredActions": ["inspect_original_request"]},
                headers={"Retry-After": "3"})
        if request.url.path == "/v2/quota":
            result = deepcopy(WIRE["quota"])
            if request.url.params.get("view") != "resources":
                result.pop("resources")
            if state.get("malformed_resources"):
                result.pop("resources", None)
            return httpx.Response(200, json=result)
        if request.url.path.startswith("/v2/account-resets"):
            return httpx.Response(200, json=WIRE["receipt"])
        if request.url.path == "/v2/credential-profiles":
            return httpx.Response(200, json={"profiles": [{"profile": {
                "harness_id": "claude", "profile_id": "claude-default", "enabled": False}}], "harnessAccounts": []})
        if request.url.path in {"/v2/harnesses", "/v2/agent-capabilities"}:
            return httpx.Response(200, json={"harnesses": []})
        raise AssertionError(f"Unexpected engine call: {request.method} {request.url.path}")

    original = wire.ClaudexorGateway.__init__

    def initialize(gateway, actual):
        assert actual is endpoint
        original(gateway, actual)
        gateway._client.close()
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(respond),
                                      headers={"Authorization": "Bearer private-fixture-token"})
        clients.append(gateway._client)

    monkeypatch.setattr(wire.ClaudexorGateway, "__init__", initialize)
    app = Starlette(routes=[
        Route("/api/claudexor/status", api_claudexor_status),
        Route("/api/claudexor/quota/refresh", api_claudexor_quota_refresh, methods=["POST"]),
        Route("/api/claudexor/account-resets", api_claudexor_account_reset, methods=["POST"]),
        Route("/api/claudexor/account-resets/{operation_id}", api_claudexor_account_reset),
    ])
    with TestClient(app) as client:
        yield client, state, calls
    assert all(client.is_closed for client in clients)


def test_catalog_negotiates_each_operation_not_version_or_similar_query():
    assert all(account_resource_capabilities(WIRE["operations"]).values())
    wrong = deepcopy(WIRE["operations"][:2])
    wrong[0]["parameters"][0]["location"] = "body"
    wrong[1]["parameters"][0]["enum"] = ["accounts"]
    assert not any(account_resource_capabilities(wrong).values())


@pytest.mark.parametrize("legacy", [False, True])
def test_status_negotiates_rich_quota_once_and_preserves_legacy_accounts(engine, legacy):
    client, state, calls = engine
    if legacy:
        state["operations"] = []
    payload = client.get("/api/claudexor/status").json()
    quota_calls = [call for call in calls if call[1] == "/v2/quota"]
    assert quota_calls == [("GET", "/v2/quota", {} if legacy else {"view": "resources"}, None, None)]
    assert payload["quota"] == WIRE["quota"]["snapshots"]
    assert payload["profiles"]["profiles"][0]["profile"]["enabled"] is False
    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "ok"}
    assert payload["resource_capabilities_read"] == "ok"
    if legacy:
        assert "resources" not in payload
    else:
        assert payload["resources"] == WIRE["quota"]["resources"]
    assert "private-fixture-token" not in json.dumps(payload)


def test_status_and_passive_quota_each_send_only_their_own_selector(engine):
    """One resource-capable engine: full status negotiates resources, the passive view freshness."""
    client, _, calls = engine
    status = client.get("/api/claudexor/status").json()
    passive_start = len(calls)
    passive = client.get("/api/claudexor/status?view=quota").json()
    assert [params for _, path, params, *_ in calls if path == "/v2/quota"] == [
        {"view": "resources"}, {"view": "constraint_freshness"}]
    assert status["resources"] == WIRE["quota"]["resources"]
    assert passive["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "ok"}
    assert passive["quota"] == WIRE["quota"]["snapshots"]
    assert not {"resources", "resource_capabilities", "resource_capabilities_read"} & set(passive)
    assert sorted(path for _, path, *_ in calls[passive_start:]) == ["/v2/credential-profiles", "/v2/quota"]


def test_missing_rich_facet_is_failed_quota_not_fresh_empty(engine):
    client, state, _ = engine
    state["malformed_resources"] = True
    payload = client.get("/api/claudexor/status").json()
    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "failed"}


def test_disabled_named_default_refresh_forwards_exact_target_without_wake(engine):
    client, _, calls = engine
    target = WIRE["receipt"]["request"]["target"]
    response = client.post("/api/claudexor/quota/refresh", json={"target": target})
    assert response.status_code == 200
    assert response.json() == WIRE["quota"]
    assert calls[-1] == ("POST", "/v2/quota", {"view": "resources"}, {"target": target}, None)
    assert calls[0][1] == "/v2/handshake"


def test_old_engine_preserves_full_refresh_and_refuses_only_unsupported_target(engine):
    client, state, calls = engine
    state["operations"] = []
    assert client.post("/api/claudexor/quota/refresh").status_code == 200
    assert calls[-1] == ("POST", "/v2/quota", {}, {}, None)
    calls.clear()
    result = client.post("/api/claudexor/quota/refresh", json={"target": WIRE["receipt"]["request"]["target"]})
    assert result.status_code == 503 and result.json()["code"] == "account_resources_unsupported"
    assert not any(path == "/v2/quota" for _, path, *_ in calls)
    result = client.post("/api/claudexor/account-resets", json=WIRE["receipt"]["request"],
                         headers={"Idempotency-Key": "legacy"})
    assert result.status_code == 503 and result.json()["code"] == "account_resets_unsupported"
    result = client.get("/api/claudexor/account-resets/legacy")
    assert result.status_code == 503 and result.json()["code"] == "account_resets_unsupported"


@pytest.mark.parametrize("failure", ["http", "transport"])
def test_unread_operations_catalog_does_not_erase_successful_siblings(engine, failure):
    client, state, calls = engine
    state["catalog_error"] = failure
    payload = client.get("/api/claudexor/status").json()
    assert payload["resource_capabilities_read"] == "failed"
    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "ok"}
    assert payload["quota"] == WIRE["quota"]["snapshots"]
    assert payload["profiles"]["profiles"][0]["profile"]["profile_id"] == "claude-default"
    assert [call for call in calls if call[1] == "/v2/quota"] == [("GET", "/v2/quota", {}, None, None)]
    state["catalog_error"] = None
    payload = client.get("/api/claudexor/status").json()
    assert payload["resource_capabilities_read"] == "ok"
    assert payload["resources"] == WIRE["quota"]["resources"]


@pytest.mark.parametrize("failure", ["http", "transport"])
@pytest.mark.parametrize("action", ["full_refresh", "target_refresh", "reset", "inspect"])
def test_explicit_operations_preserve_catalog_failure_instead_of_claiming_unsupported(engine, failure, action):
    client, state, calls = engine
    state["catalog_error"] = failure
    if action.endswith("refresh"):
        result = client.post("/api/claudexor/quota/refresh", json=(
            {"target": WIRE["receipt"]["request"]["target"]} if action == "target_refresh" else {}))
    elif action == "reset":
        result = client.post("/api/claudexor/account-resets", json=WIRE["receipt"]["request"],
                             headers={"Idempotency-Key": "catalog-gap"})
    else:
        result = client.get("/api/claudexor/account-resets/original")
    assert result.status_code == 503
    if failure == "http":
        assert result.json()["code"] == "catalog_unavailable"
        assert result.json()["error"] == "Operation catalog could not be read"
        assert result.json()["required_actions"] == ["retry_catalog"]
        assert result.headers["Retry-After"] == "7"
    else:
        assert result.json()["code"] == "daemon_unreachable"
    assert [call[1] for call in calls] == ["/v2/handshake", "/v2/operations"]


def test_reset_replay_read_and_typed_refusal_keep_exact_contract(engine):
    client, state, calls = engine
    for _ in range(2):
        response = client.post("/api/claudexor/account-resets", json=WIRE["receipt"]["request"],
                               headers={"Idempotency-Key": "one-logical-operation"})
        assert response.json() == WIRE["receipt"]
        assert response.json()["outcome"] == "reset"
        assert response.json()["readback"]["state"] == "failed"
    reset_calls = [call for call in calls if call[1] == "/v2/account-resets"]
    assert reset_calls == [("POST", "/v2/account-resets", {}, WIRE["receipt"]["request"], "one-logical-operation")] * 2
    assert client.get("/api/claudexor/account-resets/reset-fixture").json() == WIRE["receipt"]
    assert calls[-1] == ("GET", "/v2/account-resets/reset-fixture", {}, None, None)
    state["error"] = 409
    result = client.post("/api/claudexor/account-resets", json=WIRE["receipt"]["request"], headers={"Idempotency-Key": "same"})
    assert result.status_code == 409
    assert result.json()["code"] == "idempotency_conflict"
    assert result.json()["required_actions"] == ["inspect_original_request"]
    assert result.headers["Retry-After"] == "3"


@pytest.mark.parametrize("body", [None, [], {"target": None}, {"target": {"harness": "claude", "profile_id": ""}},
                                       {"target": {"harness": "claude", "profile_id": False}}, {"unknown": 1}])
def test_refresh_invalid_body_never_becomes_full_refresh(engine, body):
    client, _, calls = engine
    response = client.post("/api/claudexor/quota/refresh", content=json.dumps(body))
    assert response.status_code == 400
    assert calls == []


def test_reset_requires_exact_body_and_key_before_transport(engine):
    client, _, calls = engine
    for body in ({}, {**WIRE["receipt"]["request"], "grant_id": None}, {**WIRE["receipt"]["request"], "confirmation": True}):
        assert client.post("/api/claudexor/account-resets", json=body, headers={"Idempotency-Key": "k"}).status_code == 400
    assert client.post("/api/claudexor/account-resets", json=WIRE["receipt"]["request"]).status_code == 400
    assert calls == []
