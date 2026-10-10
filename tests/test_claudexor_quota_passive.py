"""Quota consumers read cached quota and roster without status diagnostics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
import re
from threading import Barrier, Event
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import claudexor_daemon as owned
from ouroboros import claudexor_runtime as runtime
from ouroboros.gateway.claudexor_accounts import api_claudexor_status
from ouroboros.gateways import claudexor as wire


QUOTA_URL = "/api/claudexor/status?view=quota"
QUOTA_VIEW = {"view": "constraint_freshness"}
# The engine contract's recorded opt-in response: each constraint carries its own
# read-time freshness beside the conservative aggregate. Without that field it is
# exactly the legacy response of an engine that ignores the selector.
ENGINE_FRESHNESS_QUOTA = {
    "snapshots": [{
        "subject": {"harness": "claude", "credential_route": "vendor_native",
                    "plan_label": None, "subject_id": "fixture"},
        "constraints": [
            {"id": "five_hour", "label": "5h", "used_ratio": 0.8, "window_seconds": 18000,
             "resets_at": "2026-10-10T12:00:30.000Z", "cooldown_until": None, "freshness": "stale"},
            {"id": "weekly", "label": "Weekly", "used_ratio": 0.4, "window_seconds": 604800,
             "resets_at": "2026-10-11T12:00:00.000Z", "cooldown_until": None, "freshness": "fresh"},
        ],
        "source": "claude_oauth_usage",
        "observed_at": "2026-10-10T12:00:00.000Z",
        "freshness": "stale",
        "snapshot_id": "claude\x00vendor_native\x00fixture\x00claude_oauth_usage",
        "availability": {"state": "available", "blocking_constraints": [], "resets_at": None,
                         "model_scoped_exhaustions": []},
    }],
    "absences": [],
    "refreshed_at": None,
}


@pytest.fixture
def passive_engine(monkeypatch, tmp_path):
    """Real descriptor discovery and gateway transport, no operator daemon."""
    config_dir = tmp_path / "claudexor"
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    calls = []
    clients = []
    state = {
        "responses": {
            "/v2/credential-profiles": {"profiles": [], "harnessAccounts": []},
            "/v2/quota": {"snapshots": [], "absences": []},
        },
        "errors": {},
        "hooks": {},
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("quota view must not run status, handshake, diagnostics or mutation")

    monkeypatch.setattr(owned, "get_owned_daemon", forbidden)
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running", forbidden)
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "reconcile_rotation", forbidden)
    monkeypatch.setattr(runtime.ClaudexorRuntimeManager, "ensure", forbidden)
    monkeypatch.setattr(runtime.ClaudexorRuntimeManager, "status", forbidden)
    monkeypatch.setattr(owned.subprocess, "Popen", forbidden)
    monkeypatch.setattr(wire, "discover_daemon", forbidden)

    def respond(request):
        path = request.url.path
        calls.append((request.method, path, dict(request.url.params)))
        assert request.headers["Authorization"] == "Bearer fixture-passive-token"
        assert request.headers[wire.PROTOCOL_HEADER] == str(wire.CLAUDEXOR_PROTOCOL_MAJOR)
        # Hooks judge the request they intercept: roster and quota reads are
        # concurrent, so the last recorded call may belong to the other read.
        hook = state["hooks"].get(path)
        if hook is not None:
            hook(request)
        failure = state["errors"].get(path)
        if failure is not None:
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure[0], json=failure[1])
        if path == "/v2/handshake" and state.get("full_status"):
            return httpx.Response(200, json={
                "compatible": True,
                "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                "engine": {"version": wire.CLAUDEXOR_MIN_VERSION},
            })
        assert request.method == "GET", "passive consumer sent a mutation or handshake"
        assert path in state["responses"], "passive consumer requested a diagnostic dependency"
        return httpx.Response(200, json=deepcopy(state["responses"][path]))

    original_init = wire.ClaudexorGateway.__init__

    def initialize(gateway, endpoint):
        original_init(gateway, endpoint)
        headers = dict(gateway._client.headers)
        timeout = gateway._client.timeout
        gateway._client.close()
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            headers=headers, timeout=timeout, trust_env=False,
            transport=httpx.MockTransport(respond),
        )
        clients.append(gateway._client)

    monkeypatch.setattr(wire.ClaudexorGateway, "__init__", initialize)

    def provision():
        descriptor = config_dir / "daemon" / "control-api.json"
        descriptor.parent.mkdir(parents=True)
        token = config_dir / "daemon" / "token"
        token.write_text("fixture-passive-token", encoding="utf-8")
        descriptor.write_text(json.dumps({
            "host": "127.0.0.1", "port": 1, "tokenPath": str(token),
        }), encoding="utf-8")

    return SimpleNamespace(
        state=state, calls=calls, clients=clients, provision=provision, config_dir=config_dir,
    )


def _app():
    return Starlette(routes=[Route("/api/claudexor/status", api_claudexor_status)])


def _read(client, url=QUOTA_URL):
    response = client.get(url)
    assert response.status_code == 200
    payload = response.json()
    assert payload["view"] == "quota"
    assert type(payload["unified_accounts"]) is bool
    assert payload["reads"]["catalog"] == "not_read"
    assert "daemon" not in payload and "harnesses" not in payload
    assert set(payload["read_errors"]) <= {"discovery", "accounts", "quota"}
    for error in payload["read_errors"].values():
        assert set(error) <= {"code", "status_code"}
        assert re.fullmatch(r"[a-z0-9_]{1,64}", error["code"])
        if "status_code" in error:
            assert type(error["status_code"]) is int
    timings = payload["timings_ms"]
    assert {"discovery", "total"} <= set(timings) <= {"discovery", "accounts", "quota", "total"}
    assert all(type(value) is int and 0 <= value <= 2_147_483_647 for value in timings.values())
    assert timings["total"] >= max(timings.values())
    return payload


def test_quota_view_preserves_roster_and_one_quota_epoch_without_diagnostics(passive_engine):
    engine = passive_engine
    roster = {
        "profiles": [{
            "profile": {"profile_id": "account-a", "harness_id": "codex",
                        "enabled": False, "kind": "native_session"},
            "status": {"verification": "passed", "verification_source": "local_store"},
            "identity": {"email": "quota-test@example.invalid", "plan": "pro"},
        }],
        "harnessAccounts": [{"harness_id": "claude", "native_login_detected": True,
                             "identity": {"plan": "max"}}],
        "accountPools": [{"harness_id": "codex", "members": ["account-a"]}],
        "future_roster_fact": {"value": True},
    }
    snapshot = {
        "subject": {"harness": "codex", "subject_id": "account-a"},
        "freshness": "fresh", "observed_at": "2026-10-09T00:00:00Z", "constraints": [],
        "reset_credits": {"available": 2},
    }
    absence = {"subject": {"harness": "claude", "subject_id": "native"}, "reason": "poll_paced"}
    engine.state["responses"].update({
        "/v2/credential-profiles": roster,
        "/v2/quota": {"snapshots": [snapshot], "absences": [absence]},
    })
    # A rendezvous verifies the two required reads are concurrent, independent
    # of the wall-clock values in the response's diagnostic timing fields.
    rendezvous = Barrier(2, timeout=5)
    engine.state["hooks"] = dict.fromkeys(engine.state["responses"], lambda _request: rendezvous.wait())
    engine.provision()
    before = {path: path.read_bytes() for path in engine.config_dir.rglob("*") if path.is_file()}
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "ok"}
    assert payload["profiles"] == roster
    assert payload["unified_accounts"] is True
    assert payload["quota"] == [snapshot] and payload["quota_absences"] == [absence]
    assert payload["read_errors"] == {}
    assert set(payload["timings_ms"]) == {"discovery", "accounts", "quota", "total"}
    assert sorted(engine.calls) == [
        ("GET", "/v2/credential-profiles", {}), ("GET", "/v2/quota", QUOTA_VIEW),
    ]
    assert before == {path: path.read_bytes() for path in engine.config_dir.rglob("*") if path.is_file()}
    assert len(engine.clients) == 1 and engine.clients[0].is_closed
    assert "fixture-passive-token" not in json.dumps(payload)


def test_quota_view_known_empty_is_distinct_from_undiscovered(passive_engine):
    engine = passive_engine
    with TestClient(_app()) as client:
        missing = _read(client)
        assert missing["profiles"] == {} and missing["quota"] == []
        assert missing["unified_accounts"] is False
        assert missing["reads"] == dict.fromkeys(("catalog", "accounts", "quota"), "not_read")
        assert missing["read_errors"]["discovery"]["code"] == "daemon_not_discovered"
        assert set(missing["timings_ms"]) == {"discovery", "total"}
        assert not engine.config_dir.exists() and engine.calls == [] and engine.clients == []
        engine.provision()
        empty = _read(client)
    assert empty["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "ok"}
    assert empty["profiles"] == {"profiles": [], "harnessAccounts": []}
    assert empty["unified_accounts"] is False
    assert empty["quota"] == [] and empty["quota_absences"] == []
    assert empty["read_errors"] == {}
    assert all(client.is_closed for client in engine.clients)


@pytest.mark.parametrize("pools", [[], [{"harness_id": "codex", "members": ["account-a"]}]])
@pytest.mark.parametrize("quota_failed", [False, True])
def test_quota_view_unified_marker_comes_from_successful_roster_without_operations(
    passive_engine, pools, quota_failed,
):
    engine = passive_engine
    roster = {"profiles": [], "harnessAccounts": [], "accountPools": pools}
    engine.state["responses"]["/v2/credential-profiles"] = roster
    if quota_failed:
        engine.state["errors"]["/v2/quota"] = httpx.ReadTimeout("fixture quota unavailable")
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"]["accounts"] == "ok"
    assert payload["reads"]["quota"] == ("failed" if quota_failed else "ok")
    assert payload["profiles"] == roster and payload["unified_accounts"] is True
    assert sorted(engine.calls) == [
        ("GET", "/v2/credential-profiles", {}), ("GET", "/v2/quota", QUOTA_VIEW),
    ]
    assert all(client.is_closed for client in engine.clients)


@pytest.mark.parametrize("body", [
    None, [], {}, {"profiles": []}, {"harnessAccounts": []},
    {"profiles": None, "harnessAccounts": []},
    {"profiles": [], "harnessAccounts": {}},
    {"profiles": [None], "harnessAccounts": []},
    {"profiles": [{"profile": {"harness_id": "codex"}}], "harnessAccounts": []},
    {"profiles": [{"profile": {"harness_id": "codex", "profile_id": ""}}], "harnessAccounts": []},
    {"profiles": [{"profile": {"harness_id": "codex", "profile_id": "a", "enabled": "false"}}],
     "harnessAccounts": []},
    {"profiles": [], "harnessAccounts": [{}]},
    {"profiles": [], "harnessAccounts": [], "accountPools": None},
    {"profiles": [], "harnessAccounts": [], "accountPools": [None]},
    {"profiles": [{"profile": {"harness_id": "codex"}}], "harnessAccounts": [], "accountPools": []},
])
def test_quota_view_malformed_roster_does_not_claim_authoritative_empty(passive_engine, body):
    engine = passive_engine
    engine.state["responses"]["/v2/credential-profiles"] = body
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"] == {"catalog": "not_read", "accounts": "failed", "quota": "ok"}
    assert payload["profiles"] == {}
    assert payload["unified_accounts"] is False
    assert set(payload["read_errors"]) == {"accounts"}
    assert all(client.is_closed for client in engine.clients)


@pytest.mark.parametrize("body", [
    None, [], {}, {"absences": []}, {"snapshots": None},
    {"snapshots": [], "absences": None}, {"snapshots": [], "absences": {}},
    {"snapshots": [None]}, {"snapshots": [], "absences": [None]},
])
def test_quota_view_malformed_quota_preserves_successful_roster(passive_engine, body):
    engine = passive_engine
    engine.state["responses"]["/v2/quota"] = body
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "failed"}
    assert payload["profiles"] == {"profiles": [], "harnessAccounts": []}
    assert payload["quota"] == [] and payload["quota_absences"] == []
    assert set(payload["read_errors"]) == {"quota"}
    assert all(client.is_closed for client in engine.clients)


def test_quota_view_accepts_legacy_absence_omission(passive_engine):
    engine = passive_engine
    engine.state["responses"]["/v2/quota"] = {"snapshots": []}
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"]["quota"] == "ok" and payload["quota_absences"] == []
    assert payload["read_errors"] == {}


@pytest.mark.parametrize("facet,path", [
    ("accounts", "/v2/credential-profiles"), ("quota", "/v2/quota"),
])
@pytest.mark.parametrize("failure_type", ["refusal", "transport", "malformed_code"])
def test_quota_view_errors_are_bounded_and_preserve_successful_sibling(
    passive_engine, caplog, facet, path, failure_type,
):
    engine = passive_engine
    secret = "private-account-token-do-not-emit"
    caplog.set_level(logging.INFO, logger="ouroboros.gateway.claudexor_passive")
    if failure_type == "transport":
        failure = httpx.ReadTimeout(secret)
    else:
        failure = (503, {
            "code": "daemon_recovery_only" if failure_type == "refusal" else f"/{secret}/" * 100,
            "message": secret, "context": {"token": secret}, "requiredActions": [secret],
        })
    engine.state["errors"][path] = failure
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    other = "quota" if facet == "accounts" else "accounts"
    assert payload["reads"][facet] == "failed" and payload["reads"][other] == "ok"
    assert payload["unified_accounts"] is False
    assert set(payload["read_errors"]) == {facet}
    error = payload["read_errors"][facet]
    if failure_type == "refusal":
        assert error == {"code": "daemon_recovery_only", "status_code": 503}
    elif failure_type == "transport":
        assert error["code"] == "observation_read_timeout"
    assert secret not in json.dumps(payload) and secret not in caplog.text
    assert "fixture-passive-token" not in json.dumps(payload)
    records = [record for record in caplog.records if record.name == "ouroboros.gateway.claudexor_passive"]
    assert len(records) == 1
    assert records[0].args == (payload["reads"], payload["read_errors"], payload["timings_ms"])
    assert all(client.is_closed for client in engine.clients)


def test_quota_consumer_returns_while_full_status_catalog_is_still_blocked(passive_engine, monkeypatch):
    """The real route serves quota before the delayed diagnostic is released."""
    engine = passive_engine
    engine.state["full_status"] = True
    engine.state["responses"].update({
        "/v2/agent-capabilities": {"harnesses": []},
        "/v2/harnesses": {"harnesses": []},
        "/v2/operations": {"operations": []},
    })
    engine.state["responses"]["/v2/quota"] = {
        "snapshots": [{"subject": {"harness": "codex", "subject_id": "account-a"}, "constraints": []}],
        "absences": [],
    }
    catalog_entered = Event()
    release_catalog = Event()

    def delayed_catalog(_request):
        catalog_entered.set()
        assert release_catalog.wait(10), "test cleanup did not release catalog"

    engine.state["hooks"]["/v2/agent-capabilities"] = delayed_catalog
    status_reads = []

    def status_dict():
        status_reads.append(True)
        return {"state": "running"}

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(status_dict=status_dict))
    engine.provision()
    with TestClient(_app()) as client, ThreadPoolExecutor(max_workers=2) as consumers:
        full_status = consumers.submit(client.get, "/api/claudexor/status")
        try:
            assert catalog_entered.wait(5), "full status never reached delayed catalog control"
            passive = consumers.submit(_read, client)
            payload = passive.result(timeout=5)
            assert payload["quota"][0]["subject"]["subject_id"] == "account-a"
            assert not release_catalog.is_set() and not full_status.done()
            assert status_reads == [True], "passive reader invoked status_dict"
            assert sum(path == "/v2/handshake" for _, path, _ in engine.calls) == 1
            assert sum(path == "/v2/agent-capabilities" for _, path, _ in engine.calls) == 1
        finally:
            release_catalog.set()
        original = full_status.result(timeout=5)
    assert original.status_code == 200 and "view" not in original.json()
    assert original.json()["reads"] == dict.fromkeys(("catalog", "accounts", "quota"), "ok")
    # Only the passive reader opts in; the Accounts status keeps the legacy quota read.
    quota_params = [params for _, path, params in engine.calls if path == "/v2/quota"]
    assert sorted(quota_params, key=len) == [{}, QUOTA_VIEW]
    assert len(engine.clients) == 2 and all(client.is_closed for client in engine.clients)


def _without_constraint_freshness(body):
    legacy = deepcopy(body)
    for snapshot in legacy["snapshots"]:
        for constraint in snapshot["constraints"]:
            del constraint["freshness"]
    return legacy


def test_gateway_quota_selector_is_opt_in(passive_engine):
    engine = passive_engine
    engine.provision()
    with wire.ClaudexorGateway(wire.discover_daemon_at(engine.config_dir)) as gateway:
        assert gateway.quota_state() == engine.state["responses"]["/v2/quota"]
        assert gateway.quota_state(view="constraint_freshness") == engine.state["responses"]["/v2/quota"]
    assert engine.calls == [("GET", "/v2/quota", {}), ("GET", "/v2/quota", QUOTA_VIEW)]


def test_quota_view_preserves_engine_constraint_freshness_from_one_read(passive_engine):
    engine = passive_engine
    engine.state["responses"]["/v2/quota"] = ENGINE_FRESHNESS_QUOTA
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"]["quota"] == "ok" and payload["read_errors"] == {}
    assert payload["quota"] == ENGINE_FRESHNESS_QUOTA["snapshots"]
    assert payload["quota"][0]["freshness"] == "stale"
    assert [row["freshness"] for row in payload["quota"][0]["constraints"]] == ["stale", "fresh"]
    assert sorted(engine.calls) == [
        ("GET", "/v2/credential-profiles", {}), ("GET", "/v2/quota", QUOTA_VIEW),
    ]


def test_quota_view_selector_ignored_by_an_older_engine_keeps_the_legacy_envelope(passive_engine):
    engine = passive_engine
    legacy = _without_constraint_freshness(ENGINE_FRESHNESS_QUOTA)
    engine.state["responses"]["/v2/quota"] = legacy
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"]["quota"] == "ok" and payload["read_errors"] == {}
    assert payload["quota"] == legacy["snapshots"]
    # One selector read: no capability discovery, retry or fallback request.
    assert sorted(engine.calls) == [
        ("GET", "/v2/credential-profiles", {}), ("GET", "/v2/quota", QUOTA_VIEW),
    ]


def test_quota_view_repeated_refusal_stops_after_one_legacy_read(passive_engine):
    engine = passive_engine
    engine.state["errors"]["/v2/quota"] = (400, {"code": "invalid_query", "message": "view"})
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "failed"}
    assert payload["read_errors"] == {"quota": {"code": "read_refused", "status_code": 400}}
    assert [params for _, path, params in engine.calls if path == "/v2/quota"] == [QUOTA_VIEW, {}]


def test_quota_view_325_selector_refusal_uses_one_passive_legacy_response(passive_engine, monkeypatch):
    engine = passive_engine
    engine.state["responses"]["/v2/quota"] = _without_constraint_freshness(ENGINE_FRESHNESS_QUOTA)
    # Pin the interleaving where the concurrent roster read is recorded after the
    # selector request and before its refusal is decided.
    selector_recorded = Event()
    roster_recorded = Event()
    read_roster = wire.ClaudexorGateway.credential_profiles

    def roster_after_selector(gateway):
        assert selector_recorded.wait(5), "quota selector request never reached the engine"
        return read_roster(gateway)

    def refuse_selector_only(request):
        if dict(request.url.params) == QUOTA_VIEW:
            selector_recorded.set()
            assert roster_recorded.wait(5), "roster read did not overlap the selector request"
            engine.state["errors"]["/v2/quota"] = (400, {"code": "invalid_request", "message": "view must be resources"})
        else:
            engine.state["errors"].pop("/v2/quota", None)

    monkeypatch.setattr(wire.ClaudexorGateway, "credential_profiles", roster_after_selector)
    engine.state["hooks"].update({
        "/v2/credential-profiles": lambda _request: roster_recorded.set(),
        "/v2/quota": refuse_selector_only,
    })
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "ok"}
    assert payload["quota"][0]["freshness"] == "stale"
    assert all("freshness" not in c for c in payload["quota"][0]["constraints"])
    assert engine.calls == [
        ("GET", "/v2/quota", QUOTA_VIEW), ("GET", "/v2/credential-profiles", {}), ("GET", "/v2/quota", {}),
    ]
    assert [params for _, path, params in engine.calls if path == "/v2/quota"] == [QUOTA_VIEW, {}]
    assert all(method == "GET" for method, _, _ in engine.calls)


@pytest.mark.parametrize("change", [
    lambda snapshots: snapshots[0]["constraints"][1].pop("freshness"),
    lambda snapshots: snapshots[0]["constraints"][1].update(freshness="expired"),
    lambda snapshots: snapshots[0]["constraints"][1].update(freshness="Fresh"),
    lambda snapshots: snapshots[0]["constraints"][1].update(freshness=None),
    lambda snapshots: snapshots[0]["constraints"][1].update(freshness=["fresh"]),
    lambda snapshots: snapshots[0]["constraints"].append(None),
    lambda snapshots: snapshots.append({**deepcopy(snapshots[0]), "constraints": {}}),
], ids=["partial", "unknown-value", "case", "null", "list", "non-object-row", "non-list-group"])
def test_quota_view_incomplete_constraint_freshness_fails_only_the_quota_facet(passive_engine, change):
    engine = passive_engine
    body = deepcopy(ENGINE_FRESHNESS_QUOTA)
    change(body["snapshots"])
    engine.state["responses"]["/v2/quota"] = body
    engine.provision()
    with TestClient(_app()) as client:
        payload = _read(client)
    assert payload["reads"] == {"catalog": "not_read", "accounts": "ok", "quota": "failed"}
    assert payload["read_errors"] == {"quota": {"code": "malformed_response"}}
    assert payload["quota"] == [] and payload["quota_absences"] == []
    assert payload["profiles"] == {"profiles": [], "harnessAccounts": []}


def test_quota_view_dispatch_is_exact_and_does_not_change_default_status(monkeypatch):
    from ouroboros.gateway import claudexor_accounts

    calls = []

    def full_status(include_models):
        calls.append(include_models)
        return {"existing_full_status": True}

    monkeypatch.setattr(claudexor_accounts, "_status_payload", full_status)
    with TestClient(_app()) as client:
        for query in ("", "?include=models", "?view=other", "?view=quotas", "?view=Quota"):
            response = client.get("/api/claudexor/status" + query)
            assert response.status_code == 200 and response.json() == {"existing_full_status": True}
    assert calls == [False, True, False, False, False]
