"""Account status reads metadata; only launch preparation probes executables."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import claudexor_daemon as owned
from ouroboros import claudexor_runtime as runtime
from ouroboros.gateway.claudexor_accounts import api_claudexor_status
from ouroboros.gateway.models import api_model_catalog


@pytest.fixture
def metadata_engine(monkeypatch, tmp_path):
    """Real descriptor discovery and gateway HTTP, with no engine or vendor process."""
    from ouroboros.gateways import claudexor as wire

    config_dir = tmp_path / "claudexor"
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    calls = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("metadata must not start, install, reconcile or use the operator engine")

    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running", forbidden)
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "reconcile_rotation", forbidden)
    monkeypatch.setattr(runtime.ClaudexorRuntimeManager, "ensure", forbidden)
    monkeypatch.setattr(owned.subprocess, "Popen", forbidden)
    monkeypatch.setattr(wire, "discover_daemon", forbidden)
    state = {"reachable": True, "handshake_error": False, "accounts_view": False,
             "engine_version": wire.CLAUDEXOR_MIN_VERSION}

    def respond(request):
        calls.append((request.method, request.url.path, dict(request.url.params)))
        assert request.headers["Authorization"] == "Bearer fixture-owned-token"
        if not state["reachable"]:
            raise httpx.ConnectError("owned engine is stopped", request=request)
        if request.url.path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": not state["handshake_error"],
                "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                "engine": {"version": state["engine_version"]}})
        if request.url.path == "/v2/operations":
            return httpx.Response(200, json={"operations": [
                {"method": "GET", "path": path, "parameters": [
                    {"name": "view", "location": "query", "enum": ["accounts"]}]}
                for path in ("/v2/model-sources", "/v2/model-sources/:id/models", "/v2/harnesses/:id/models")
            ] if state["accounts_view"] else []})
        if request.url.path in state.get("responses", {}):
            answer = state["responses"][request.url.path]
            # A callable is a non-200 engine answer, e.g. a typed ControlProblem.
            return answer(request) if callable(answer) else httpx.Response(200, json=deepcopy(answer))
        if request.url.path == "/v2/model-sources":
            return httpx.Response(200, json={"sources": [{"id": "codex", "label": "Codex",
                "credentialHarness": "codex"}]})
        assert request.url.path == "/v2/model-sources/codex/models"
        return httpx.Response(200, json={"source": "codex", "credentialProfileId": "account-a",
            "accountFingerprint": "identity-a", "provenance": "fixture exact catalog",
            "observedAt": "2026-09-07T00:00:00Z", "models": [{"id": "exact-model",
                "contextWindow": 272000, "maxContextWindow": 872000}]})

    original_init = wire.ClaudexorGateway.__init__
    clients = []

    def initialize(gateway, endpoint):
        original_init(gateway, endpoint)
        gateway._client.close()
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1",
            headers={"Authorization": f"Bearer {endpoint.token}"},
            transport=httpx.MockTransport(respond))
        clients.append(gateway._client)

    monkeypatch.setattr(wire.ClaudexorGateway, "__init__", initialize)

    def provision():
        descriptor = config_dir / "daemon" / "control-api.json"
        descriptor.parent.mkdir(parents=True)
        token = config_dir / "daemon" / "token"
        token.write_text("fixture-owned-token", encoding="utf-8")
        descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": 1,
            "tokenPath": str(token)}), encoding="utf-8")

    return state, calls, clients, provision, config_dir


@pytest.mark.parametrize("provisioned", [False, True])
@pytest.mark.parametrize("query", ["", "?source_id=codex&credential_profile_id=account-a"])
def test_catalog_get_cold_or_stopped_never_starts_engine(metadata_engine, monkeypatch, query, provisioned):
    from ouroboros.gateway import models

    state, calls, clients, provision, config_dir = metadata_engine
    monkeypatch.setattr(models, "load_settings", lambda: {})
    if provisioned:
        provision()
        state["reachable"] = False
    before = sorted(str(path) for path in config_dir.rglob("*"))
    app = Starlette(routes=[Route("/api/model-catalog", api_model_catalog)])
    with TestClient(app) as client:
        payload = client.get("/api/model-catalog" + query).json()
    assert payload["items"] == [] and payload["model_sources"] == []
    if provisioned or query:
        assert payload["errors"][0]["code"] == (
            "daemon_unreachable" if provisioned else "daemon_not_discovered")
    else:
        assert payload["errors"] == [] and calls == []
    assert sorted(str(path) for path in config_dir.rglob("*")) == before
    assert all(client.is_closed for client in clients)


def test_warm_owned_metadata_preserves_models_profiles_and_closes(metadata_engine):
    from ouroboros.llm import LLMClient

    _, calls, clients, provision, _ = metadata_engine
    provision()
    app = Starlette(routes=[Route("/api/model-catalog", api_model_catalog)])
    with TestClient(app) as client:
        payload = client.get("/api/model-catalog?source_id=codex&credential_profile_id=account-a").json()
    assert payload["errors"] == []
    assert payload["items"][0]["value"] == "claudexor::codex=exact-model"
    assert payload["items"][0]["credential_profile_id"] == "account-a"
    assert LLMClient.claudexor_model_sources()["sources"][0]["id"] == "codex"
    catalog = LLMClient.claudexor_model_catalog("codex", "account-a", requested_model="exact-model")
    assert catalog["models"][0]["maxContextWindow"] == 872000
    assert calls[-1][2] == {"credentialProfileId": "account-a", "requestedModel": "exact-model"}
    assert sum(path == "/v2/handshake" for _, path, _ in calls) == 3
    assert len(clients) == 3 and all(client.is_closed for client in clients)


def test_account_catalog_view_reaches_raw_http_without_affecting_execution_discovery(metadata_engine):
    from ouroboros.llm import LLMClient

    state, calls, clients, provision, _ = metadata_engine
    state["accounts_view"] = True
    envelope = {"source": "codex", "partial": True, "accounts": [
        {"credentialProfileId": "account-a", "availability": "available", "problem": None,
         "catalog": {"source": "codex", "credentialProfileId": "account-a", "observedAt": None,
                     "provenance": "fixture", "models": [{"id": "exact-model"}]}},
        {"credentialProfileId": "account-b", "availability": "unknown", "catalog": None,
         "problem": {"code": "model_catalog_unavailable", "message": "Read failed"}},
    ]}
    state["responses"] = {"/v2/model-sources/codex/models": envelope}
    provision()
    app = Starlette(routes=[Route("/api/model-catalog", api_model_catalog)])
    with TestClient(app) as client:
        payload = client.get("/api/model-catalog?source_id=codex").json()
    assert payload["account_catalogs"] == [envelope]
    assert payload["items"][0]["observed_at"] is None and payload["partial"] is True
    assert ("GET", "/v2/model-sources", {"view": "accounts"}) in calls
    assert calls[-1] == ("GET", "/v2/model-sources/codex/models", {"view": "accounts"})
    LLMClient.claudexor_model_catalog("codex", "account-a", requested_model="exact-model")
    assert calls[-1][2] == {"credentialProfileId": "account-a", "requestedModel": "exact-model"}
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("account_view,wire_mode", [(False, "legacy"), (True, "stub"), (True, "http")])
def test_status_native_models_reuse_negotiated_account_catalog(metadata_engine, monkeypatch, account_view, wire_mode):
    from ouroboros.gateway.claudexor_accounts import _status_payload

    state, calls, clients, provision, _ = metadata_engine
    state["accounts_view"] = account_view
    records = [
        {"credentialProfileId": profile, "availability": availability, "problem": None,
         "catalog": {"harnessId": "codex", "credentialProfileId": profile, "source": "manifest",
                     "verifiedAgainst": "fixture-v1", "provenance": "manifest", "observedAt": None,
                     "models": [{"id": model, "processing": {"modes": modes, "eligible": None}}]}}
        for profile, availability, model, modes in (
            ("account-a", "available", "model-a", ["standard"]),
            ("account-b", "unavailable", "model-b", ["standard", "fast"]))
    ]
    records.append({"credentialProfileId": "account-c", "availability": "unknown", "catalog": None,
                    "problem": {"code": "model_catalog_unavailable", "message": "Could not read catalog"}})
    envelope = {"harnessId": "codex", "accounts": records, "partial": True}
    state["responses"] = {
        "/v2/agent-capabilities": {"harnesses": [{"id": "codex", "displayName": "Codex", "enabled": True}]},
        "/v2/harnesses": {"harnesses": []},
        "/v2/credential-profiles": {"profiles": [], "harnessAccounts": [], "accountPools": []},
        "/v2/quota": {"snapshots": [], "absences": []},
        "/v2/harnesses/codex/models": envelope if account_view else {"harnessId": "codex", "models": [{"id": "legacy"}]},
    }
    if wire_mode == "stub":
        from ouroboros.gateways.claudexor import ClaudexorGateway

        def catalog_stub(gateway, harness, *, view):
            assert harness == "codex" and view == "accounts"
            calls.append(("STUB", "/v2/harnesses/codex/models", {"view": view}))
            return deepcopy(envelope)

        monkeypatch.setattr(ClaudexorGateway, "harness_model_catalog", catalog_stub, raising=False)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(status_dict=lambda: {"state": "running"}))
    provision()
    payload = _status_payload(include_models=True)
    assert payload["reads"] == dict.fromkeys(("catalog", "accounts", "quota"), "ok")
    harness, = payload["harnesses"]
    if account_view:
        assert harness["model_catalog"] == envelope
        assert [(model["id"], model["credential_profile_id"], model["availability"]) for model in harness["models"]] == [
            ("model-a", "account-a", "available"), ("model-b", "account-b", "unavailable")]
        assert all(model["observed_at"] is None and model["catalog_source"] == "manifest" for model in harness["models"])
        assert harness["models"][1]["processing"]["modes"] == ["standard", "fast"]
    else:
        assert harness["models"] == [{"id": "legacy"}] and "model_catalog" not in harness
    assert sum(path == "/v2/operations" for _, path, _ in calls) == 1
    assert calls[-1] == ("STUB" if wire_mode == "stub" else "GET", "/v2/harnesses/codex/models",
                         {"view": "accounts"} if account_view else {})
    assert all(client.is_closed for client in clients)


STATUS_GOLDEN = Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures" / "status_quota_refused.json"


def test_status_golden_is_this_producers_answer_to_one_refused_read(metadata_engine, monkeypatch, tmp_path):
    """web/tests/fixtures/status_quota_refused.json, the payload the status store is judged
    against, is this producer's answer when the catalog and a unified roster land and only
    the quota read is refused. Engine HTTP is faked; the clock and the home path are fixed.
    After a deliberate producer change, rerun with OUROBOROS_WRITE_STATUS_GOLDEN=1 and
    review web/tests/claudexor_status_store.test.js against the rewritten file."""
    from ouroboros import config, utils
    from ouroboros.gateway.claudexor_accounts import _status_payload

    state, _, clients, provision, config_dir = metadata_engine
    state["engine_version"] = "4.7.0"
    state["responses"] = {
        "/v2/agent-capabilities": {"harnesses": [{
            "id": "codex", "displayName": "Codex CLI", "status": "ok", "enabled": True,
            "providerFamily": "openai", "accessProfilesSupported": ["readonly", "workspace_write"]}]},
        "/v2/harnesses": {"harnesses": []},
        "/v2/credential-profiles": {
            "profiles": [{"profile": {"harness_id": "codex", "profile_id": "koshak", "enabled": True},
                          "status": {"verification": "passed", "verification_source": "vendor"},
                          "identity": {"email": "owner@example.com", "plan": "pro"}}],
            "harnessAccounts": [],
            "accountPools": [{"harness_id": "codex", "next_up": {"kind": "profile", "profileId": "koshak"}}]},
        "/v2/quota": lambda _request: httpx.Response(
            503, json={"code": "quota_probe_failed", "message": "quota read died"}),
    }
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(status_dict=lambda: {
        "state": "running",
        "runtime": {"state": "ready", "version": "4.7.0", "target_version": "4.7.0", "source": "release"}}))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")  # no delegated run recorded
    monkeypatch.setattr(utils, "utc_now_iso", lambda: "2026-10-27T12:00:00+00:00")
    provision()
    payload = json.loads(json.dumps(_status_payload(include_models=False, join=False)))
    assert payload["config_dir"] == str(config_dir)
    payload["config_dir"] = "/tmp/claudexor"
    if os.environ.get("OUROBOROS_WRITE_STATUS_GOLDEN") == "1":
        STATUS_GOLDEN.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert json.loads(STATUS_GOLDEN.read_text(encoding="utf-8")) == payload, (
        f"{STATUS_GOLDEN.name} drifted from _status_payload; see this test's docstring")
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("operation", ["sources", "catalog", "capability"])
def test_cold_metadata_is_unknown_not_healthy(metadata_engine, tmp_path, operation):
    from ouroboros import capability_evidence as ce
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.llm import LLMClient

    _, calls, clients, _, config_dir = metadata_engine
    if operation == "capability":
        evidence = ce.probe(tmp_path / "evidence", provider="claudexor",
            model="claudexor::codex=exact-model", allow_fetch=True)
        assert evidence.status == ce.STATUS_FAILED and evidence.window_tokens == 0
    else:
        with pytest.raises(ClaudexorUnavailable) as caught:
            if operation == "sources":
                LLMClient.claudexor_model_sources()
            else:
                LLMClient.claudexor_model_catalog("codex")
        assert caught.value.code == "daemon_not_discovered"
    assert not config_dir.exists() and not calls and not clients


def test_read_gateway_closes_after_failed_handshake(metadata_engine):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    state, _, clients, provision, _ = metadata_engine
    provision()
    state["handshake_error"] = True
    with pytest.raises(ClaudexorUnavailable) as caught:
        owned.read_owned_gateway()
    assert caught.value.code == "protocol_incompatible"
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.parametrize("state", ["not_provisioned", "stale"])
def test_status_get_never_prepares_or_probes_a_launch(monkeypatch, tmp_path, state):
    """Installed metadata remains visible without even the first Node probe."""
    manager = runtime.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    metadata = {
        "version": pin.version, "build_sha": pin.build_sha,
        "node_version": pin.node_version, "archive_source": "cache",
    }
    monkeypatch.setattr(manager, "_managed_metadata", lambda: dict(metadata))
    monkeypatch.setattr(manager, "_install_in_progress", lambda: False)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    monkeypatch.setattr(runtime, "managed_runtime_root", lambda: tmp_path / "runtime")
    daemon = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(daemon, "_classify_liveness", lambda: (None, state, ""))
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: daemon)
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "claudexor")
    monkeypatch.setattr(owned, "verify_owned_home", lambda: "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("passive status must not prepare a command or start a process")

    monkeypatch.setattr(manager, "resolve_command", forbidden)
    monkeypatch.setattr(manager, "ensure", forbidden)
    monkeypatch.setattr(daemon, "ensure_running", forbidden)
    monkeypatch.setattr(owned.subprocess, "Popen", forbidden)
    app = Starlette(routes=[Route("/api/claudexor/status", api_claudexor_status)])
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/api/claudexor/status?include=models")
            assert response.status_code == 200
            payload = response.json()
            assert payload["daemon"]["state"] == state
            assert payload["daemon"]["runtime"]["state"] == "ready"
            assert payload["daemon"]["runtime"]["node_version"] == pin.node_version
            assert "binary" not in payload["daemon"]
            assert payload["reads"] == dict.fromkeys(("catalog", "accounts", "quota"), "not_read")


def test_launch_preparation_still_probes_the_exact_node(monkeypatch, tmp_path):
    manager = runtime.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)
    monkeypatch.setattr(manager, "_managed_metadata", lambda: {"version": pin.version})
    monkeypatch.setattr(runtime, "managed_runtime_root", lambda: tmp_path / "runtime")
    calls = []

    def resolve_node(requested_pin):
        assert requested_pin is pin
        calls.append("node")
        return str(tmp_path / "node")

    monkeypatch.setattr(manager, "_resolve_node", resolve_node)
    monkeypatch.setattr(manager, "_probe", lambda command, requested_pin: calls.append("engine"))
    assert manager.ensure()[0] == str(tmp_path / "node")
    assert calls == ["node", "engine"]
