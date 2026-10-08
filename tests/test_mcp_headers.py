"""Literal MCP HTTP headers: validation, wire projection, masks, restoration and Show.

Every value is a synthetic fixture. Settings GET/POST run through the real
handlers over the in-memory document of ``settings_client``; Show runs through
the mounted selected-secret route over a temporary settings file.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import mcp_client
from ouroboros.gateway.settings import (
    MCPSecretIdentityAmbiguous,
    _mask_mcp_servers_payload,
    _rehydrate_mcp_servers_payload,
)
from ouroboros.mcp_headers import (
    HEADER_VALUE_PLACEHOLDER as MASK,
    MCPHeaderPlaceholderUnmatched,
    mask_headers,
    restore_headers,
    validate_headers,
)
from tests.test_settings_secret_mask import settings_client  # noqa: F401
from tests.test_settings_secret_reveal import reveal_settings  # noqa: F401

BEARER = "Bearer synthetic-bearer-0001"
BASIC = "Basic c3ludGhldGljOnBhc3N3b3Jk"
CUSTOM = "synthetic-custom-key-0002"
QUOTED = 'synthetic "quoted" \\ value'


def http(**changes):
    return {"id": "Docs Server", "name": "Docs", "enabled": True, "transport": "streamable_http",
            "url": "https://docs.example/mcp", **changes}


def config(**changes):
    errors: list = []
    cfg = mcp_client.normalize_server_config(http(**changes), errors=errors)
    return cfg, errors


# --- validation and the wire ------------------------------------------------

@pytest.mark.parametrize("headers, wire", [
    ({"Authorization": BEARER, "X-Api-Key": CUSTOM}, {"Authorization": BEARER, "X-Api-Key": CUSTOM}),
    ({"Authorization": BASIC}, {"Authorization": BASIC}),
    ({"X-Tenant": "tenant-7", "X-Empty": ""}, {"X-Tenant": "tenant-7"}),
    ({}, {}),
    (None, {}),
])
def test_literal_headers_reach_the_wire_exactly_and_empty_values_are_not_sent(headers, wire):
    cfg, errors = config(headers=headers)
    assert not errors and cfg.wire_headers() == wire
    assert cfg.headers == (headers or {})  # an empty value stays configured


def test_legacy_pair_alone_and_beside_distinct_headers_is_unchanged():
    legacy, _ = config(auth_token="Bearer legacy-0003")
    assert legacy.wire_headers() == {"Authorization": "Bearer legacy-0003"} and legacy.headers == {}
    both, errors = config(auth_header="X-Legacy", auth_token="legacy-0004", headers={"Authorization": BEARER})
    assert not errors and both.wire_headers() == {"X-Legacy": "legacy-0004", "Authorization": BEARER}
    # An inactive legacy pair (no token) never collides; nor does an empty header value.
    assert config(auth_token="", headers={"Authorization": BEARER})[0] is not None
    assert config(auth_token="Bearer legacy", headers={"authorization": ""})[0].wire_headers() == {
        "Authorization": "Bearer legacy"}


def test_an_active_legacy_pair_that_sets_the_same_header_fails_explicitly():
    cfg, errors = config(auth_token="Bearer legacy-0005", headers={"AUTHORIZATION": BEARER})
    assert cfg is None and "also set by the legacy auth token" in errors[0]
    assert BEARER not in errors[0] and "legacy-0005" not in errors[0]


@pytest.mark.parametrize("headers, fragment", [
    ("Authorization: " + BEARER, "must be an object"),
    ([["Authorization", BEARER]], "must be an object"),
    ({"Authorization": 7}, "must be a string"),
    ({"Bad Name: " + BEARER: "x"}, "#1 name is not a single HTTP header token"),
    ({"X-A": "ok", "": "x"}, "#2 name"),
    ({"X-A": BEARER + "\r\nInjected: 1"}, "printable ASCII"),
    ({"X-A": "ends-with-newline\n"}, "printable ASCII"),
    ({"X-A\n": "value"}, "name is not a single HTTP header token"),
    ({"X-A": "tab\tvalue"}, "printable ASCII"),
    ({"X-A": "café"}, "printable ASCII"),
    ({"X-A": " " + CUSTOM}, "start or end with whitespace"),
    ({"X-Key": CUSTOM, "x-key": "other"}, "differ only by case"),
])
def test_malformed_headers_are_refused_without_echoing_values(headers, fragment):
    cfg, errors = config(headers=headers)
    assert cfg is None and fragment in errors[0]
    for secret in (BEARER, CUSTOM, "Injected"):
        assert secret not in errors[0]


def test_stdio_keeps_env_and_references_and_refuses_nonempty_headers():
    stdio = {"id": "local", "transport": "stdio", "command": "python3", "env": {"PORT": "8080"},
             "env_from_settings": {"TOKEN": "CUSTOM_MCP_KEY"}, "headers": {}}
    settings = {"CUSTOM_MCP_KEY": "synthetic-env-0006"}
    cfg = mcp_client.normalize_server_config(stdio, settings=settings)
    assert cfg is not None and cfg.env == {"PORT": "8080", "TOKEN": "synthetic-env-0006"} and cfg.headers == {}
    errors: list = []
    assert mcp_client.normalize_server_config({**stdio, "headers": {"X-A": "synthetic-stdio-header"}}, settings=settings,
                                              errors=errors) is None
    assert "unsupported by this transport: headers" in errors[0]


def test_values_stay_out_of_repr_and_every_diagnostic_echo_is_redacted():
    cfg, _ = config(headers={"Authorization": BEARER, "X-Quoted": QUOTED})
    assert BEARER not in repr(cfg) and QUOTED not in repr(cfg)
    echoed = f"401 for {BEARER}; body {json.dumps({'header': QUOTED})}; raw {QUOTED}"
    redacted = mcp_client._redact_error_text(echoed, cfg)
    assert BEARER not in redacted and QUOTED not in redacted
    assert json.dumps(QUOTED)[1:-1] not in redacted


def test_status_names_headers_and_a_malformed_saved_map_never_echoes_values():
    manager = mcp_client.MCPManager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [
        http(headers={"Authorization": BEARER, "X-Empty": ""}),
        http(id="broken", headers={"Authorization": {"nested": BEARER}}),
        http(id="string-map", headers="Authorization: " + BEARER),
    ]})
    status = manager.status_payload()
    text = json.dumps(status)
    assert BEARER not in text
    rows = {row["id"]: row for row in status["servers"]}
    assert rows["docs_server"]["header_names"] == ["Authorization", "X-Empty"]
    assert rows["broken"]["code"] == rows["string_map"]["code"] == "MCP_CONFIG_ERROR"


# --- passive masks and exact restoration --------------------------------------

def test_passive_projection_masks_every_value_including_malformed_maps():
    assert mask_headers({"Authorization": BEARER, "X-Empty": "", "X-Number": 7, "X-Nested": {"a": BEARER}}) == {
        "Authorization": MASK, "X-Empty": "", "X-Number": MASK, "X-Nested": MASK}
    for malformed in ("Authorization: " + BEARER, [BEARER], 7):
        assert mask_headers(malformed) == MASK
    assert mask_headers(None) is None
    projected = _mask_mcp_servers_payload([http(headers={"X-Api-Key": CUSTOM}), http(id="b", headers=[BEARER])])
    assert CUSTOM not in json.dumps(projected) and BEARER not in json.dumps(projected)


def test_exact_placeholder_restores_and_repeated_saves_are_stable():
    saved = [http(headers={"Authorization": BEARER, "X-Api-Key": CUSTOM, "X-Empty": ""})]
    before = copy.deepcopy(saved)
    first = _rehydrate_mcp_servers_payload(_mask_mcp_servers_payload(saved), saved)
    second = _rehydrate_mcp_servers_payload(_mask_mcp_servers_payload(first), first)
    assert first[0]["headers"] == second[0]["headers"] == saved[0]["headers"]
    assert saved == before


@pytest.mark.parametrize("lookalike", ["***", "Bearer s...", "***set", " ***set***"])
def test_only_the_exact_emitted_placeholder_is_treated_as_a_mask(lookalike):
    saved = http(headers={"Authorization": BEARER})
    assert restore_headers({"Authorization": lookalike}, saved, server_id="docs_server") == {
        "Authorization": lookalike}


def test_case_only_rename_keeps_the_value_under_the_new_spelling():
    saved = [http(headers={"X-Api-Key": CUSTOM})]
    incoming = _mask_mcp_servers_payload(saved)
    incoming[0]["headers"] = {"x-api-key": MASK}
    assert _rehydrate_mcp_servers_payload(incoming, saved)[0]["headers"] == {"x-api-key": CUSTOM}


@pytest.mark.parametrize("incoming_headers, saved_headers, reason", [
    ({"X-New-Name": MASK}, {"X-Api-Key": CUSTOM}, "not saved under that name"),
    ({"X-Api-Key": MASK}, {"X-Api-Key": CUSTOM, "x-api-key": "second"}, "several saved headers"),
    ({"X-Api-Key": MASK}, None, "not saved under that name"),
    (MASK, {"X-Api-Key": CUSTOM}, "no saved headers to keep"),
])
def test_an_unmatched_or_ambiguous_mask_refuses_with_a_reentry_instruction(incoming_headers, saved_headers, reason):
    saved = [http(headers=saved_headers)] if saved_headers is not None else [http()]
    incoming = [http(headers=incoming_headers)]
    with pytest.raises(MCPHeaderPlaceholderUnmatched) as caught:
        _rehydrate_mcp_servers_payload(incoming, saved)
    assert reason in str(caught.value) and "Nothing was saved" in str(caught.value)
    assert CUSTOM not in str(caught.value)


def test_a_new_server_id_or_shared_identity_never_receives_a_saved_header():
    saved = [http(headers={"X-Api-Key": CUSTOM})]
    renamed = _mask_mcp_servers_payload(saved)
    renamed[0]["id"] = "another-server"
    with pytest.raises(MCPHeaderPlaceholderUnmatched):
        _rehydrate_mcp_servers_payload(renamed, saved)
    with pytest.raises(MCPSecretIdentityAmbiguous):
        _rehydrate_mcp_servers_payload(_mask_mcp_servers_payload(saved) * 2, saved)


def test_a_whole_malformed_saved_map_round_trips_unchanged_until_removed():
    saved = [http(headers="Authorization: " + BEARER)]
    masked = _mask_mcp_servers_payload(saved)
    assert masked[0]["headers"] == MASK
    assert _rehydrate_mcp_servers_payload(masked, saved)[0]["headers"] == saved[0]["headers"]
    masked[0]["headers"] = {}
    assert _rehydrate_mcp_servers_payload(masked, saved)[0]["headers"] == {}


# --- the real Settings handlers -------------------------------------------------

@pytest.fixture
def mcp_settings(settings_client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(mcp_client, "refresh_all_background", lambda **kwargs: None)
    client, on_disk, saved = settings_client
    on_disk.update(MCP_ENABLED=False, MCP_SERVERS=[http(
        id="docs_server", auth_header="X-Legacy", auth_token="legacy-0007",
        headers={"Authorization": BEARER, "X-Api-Key": CUSTOM, "X-Empty": ""}, allowed_tools=["search"],
        future_option={"kept": True})])
    return SimpleNamespace(client=client, on_disk=on_disk, saved=saved)


def _served(case):
    response = case.client.get("/api/settings")
    assert response.status_code == 200
    for secret in (BEARER, CUSTOM, "legacy-0007"):
        assert secret not in response.text
    return response.json()["MCP_SERVERS"]


def test_settings_save_show_clear_rename_and_remove_through_the_real_handlers(mcp_settings):
    case = mcp_settings
    original = copy.deepcopy(case.on_disk["MCP_SERVERS"][0])
    for _ in range(2):  # repeated untouched saves keep every literal value
        response = case.client.post("/api/settings", json={"MCP_SERVERS": _served(case)})
        assert response.status_code == 200, response.text
        assert case.saved["MCP_SERVERS"][0] == original
    served = _served(case)[0]
    assert served["headers"] == {"Authorization": MASK, "X-Api-Key": MASK, "X-Empty": ""}
    served["headers"] = {"authorization": MASK, "X-Api-Key": "", "X-Added": "synthetic-added-0008"}
    response = case.client.post("/api/settings", json={"MCP_SERVERS": [served]})
    assert response.status_code == 200, response.text
    stored = case.saved["MCP_SERVERS"][0]
    assert stored["headers"] == {"authorization": BEARER, "X-Api-Key": "", "X-Added": "synthetic-added-0008"}
    assert {key: stored[key] for key in ("auth_header", "auth_token", "allowed_tools", "future_option")} == {
        "auth_header": "X-Legacy", "auth_token": "legacy-0007", "allowed_tools": ["search"],
        "future_option": {"kept": True}}
    served = _served(case)[0]
    del served["headers"]["X-Added"]
    assert case.client.post("/api/settings", json={"MCP_SERVERS": [served]}).status_code == 200
    assert case.saved["MCP_SERVERS"][0]["headers"] == {"authorization": BEARER, "X-Api-Key": ""}


def test_settings_save_refuses_a_masked_rename_and_writes_nothing(mcp_settings):
    case = mcp_settings
    served = _served(case)[0]
    served["headers"] = {"X-Renamed": MASK}
    case.saved.clear()
    response = case.client.post("/api/settings", json={"MCP_SERVERS": [served]})
    assert response.status_code == 409 and response.json()["code"] == "MCP_HEADER_REENTER"
    assert "X-Renamed" in response.text and BEARER not in response.text and not case.saved


@pytest.mark.serial  # the shared fixture's module is serial: it re-points the Settings file
def test_show_reads_one_exact_saved_header_value(reveal_settings):  # noqa: F811
    case = reveal_settings
    case.settings["MCP_SERVERS"] = [
        {"id": "first-server", "headers": {"Authorization": BEARER, "X-Bad": 7, "Dup": "a", "dup": "b"}},
    ]
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")
    before = case.path.read_bytes()
    with TestClient(case.app) as client:
        shown = client.post("/api/settings/secret", json={"mcp_server_id": "first-server", "header_name": "authorization"})
        assert shown.status_code == 200 and shown.json() == {"value": BEARER}
        assert shown.headers["cache-control"] == "no-store"
        assert client.post("/api/settings/secret", json={"mcp_server_id": "first-server",
                                                        "header_name": "X-Missing"}).status_code == 404
        for name in ("X-Bad", "dup"):
            refused = client.post("/api/settings/secret", json={"mcp_server_id": "first-server", "header_name": name})
            assert refused.status_code == 409 and refused.json()["code"] == "mcp_header_unreadable"
        for body in ({"mcp_server_id": "first-server", "header_name": ""}, {"key": "X", "header_name": "A"},
                     {"header_name": "Authorization"}):
            assert client.post("/api/settings/secret", json=body).status_code == 400
        assert BEARER not in client.get("/api/settings").text
    assert case.path.read_bytes() == before


# --- Test uses Save's restoration as its only path -------------------------------

@pytest.fixture
def test_endpoint(monkeypatch):
    from ouroboros.gateway import mcp

    saved = http(headers={"Authorization": BEARER, "X-Api-Key": CUSTOM}, auth_token="Bearer legacy-0009",
                 auth_header="X-Legacy")
    probes = []

    def probe(candidate, *, settings):
        probes.append(copy.deepcopy(candidate))
        return {"ok": True, "tool_count": 0}

    monkeypatch.setattr(mcp, "_ensure_configured", lambda: None)
    monkeypatch.setattr(mcp, "load_settings", lambda: {"MCP_SERVERS": [copy.deepcopy(saved)]})
    monkeypatch.setattr(mcp, "get_manager", lambda: SimpleNamespace(test_server=probe))
    with TestClient(Starlette(routes=[Route("/test", mcp.api_mcp_test, methods=["POST"])])) as client:
        yield client, saved, probes


def test_selected_test_restores_masks_exactly_as_save_would(test_endpoint):
    client, saved, probes = test_endpoint
    candidate = _mask_mcp_servers_payload([saved])[0]
    candidate["headers"]["X-Api-Key"] = "synthetic-edited-0010"
    response = client.post("/test", json={"server_id": saved["id"], "server": candidate})
    assert response.status_code == 200 and response.json()["ok"]
    assert probes[-1]["headers"] == {"Authorization": BEARER, "X-Api-Key": "synthetic-edited-0010"}
    assert probes[-1] == _rehydrate_mcp_servers_payload([candidate], [saved])[0]


@pytest.mark.parametrize("body", [
    {"server": {"id": "docs_server", "url": "https://docs.example/mcp", "headers": {"Authorization": MASK}}},
    {"server_id": "Docs Server", "server": {"id": "renamed", "url": "https://docs.example/mcp",
                                            "headers": {"Authorization": MASK}}},
    {"server_id": "Docs Server", "server": {"id": "Docs Server", "url": "https://docs.example/mcp",
                                            "headers": {"X-Other": MASK}}},
])
def test_test_refuses_a_mask_it_cannot_restore_without_probing(test_endpoint, body):
    client, _saved, probes = test_endpoint
    response = client.post("/test", json=body)
    assert response.status_code == 409 and response.json()["code"] == "MCP_HEADER_REENTER"
    assert not probes and BEARER not in response.text


@pytest.mark.parametrize("headers", [
    {"X-Key": "synthetic-edit\n"}, {"X-Key": 42}, {"X-Key": "one", "x-key": "two"},
    {"X-Legacy": "collision"},
])
def test_settings_rejects_invalid_authored_headers_without_writing(mcp_settings, headers):
    case = mcp_settings
    served = _served(case)
    served[0]["headers"] = headers
    case.saved.clear()
    response = case.client.post("/api/settings", json={"MCP_SERVERS": served})
    assert response.status_code == 400 and response.json()["code"] == "MCP_CONFIG_ERROR"
    assert not case.saved and "synthetic-edit" not in response.text


@pytest.mark.parametrize("headers", ["malformed-map-synthetic", {"X-Bad": {"value": CUSTOM}}, {"X-Bad": 9}])
def test_unrelated_save_keeps_malformed_existing_headers_and_null_rows(mcp_settings, headers):
    case = mcp_settings
    case.on_disk["MCP_SERVERS"] = [None, {"id": "old", "headers": headers, "future": {"yes": True}}]
    served = _served(case)
    assert served[0] is None and CUSTOM not in json.dumps(served)
    response = case.client.post("/api/settings", json={"MCP_SERVERS": served})
    assert response.status_code == 200, response.text
    assert case.saved["MCP_SERVERS"][0] is None
    assert case.saved["MCP_SERVERS"][1]["headers"] == headers
    assert case.saved["MCP_SERVERS"][1]["future"] == {"yes": True}


def test_invalid_config_diagnostic_masks_header_echoes_even_before_a_config_exists(caplog):
    secret = "syntheticsecret"
    cfg, errors = config(url=secret + "://example.test", headers={"X-Key": secret})
    assert cfg is None and secret not in json.dumps(errors) and secret not in caplog.text


@pytest.mark.parametrize("transport, sdk_name", [("streamable_http", "streamablehttp_client"), ("sse", "sse_client")])
def test_both_transport_factories_use_the_same_literal_wire_map(monkeypatch, transport, sdk_name):
    seen = []
    monkeypatch.setattr(mcp_client, sdk_name, lambda url, **kwargs: seen.append((url, kwargs)))
    cfg, errors = config(transport=transport, headers={"Authorization": BASIC, "X-Key": CUSTOM, "X-Clear": ""})
    assert not errors
    mcp_client._transport_factory(cfg)
    assert seen == [(cfg.url, {"headers": {"Authorization": BASIC, "X-Key": CUSTOM}})]


def test_renamed_test_candidate_cannot_inherit_the_selected_servers_legacy_token(test_endpoint):
    client, saved, probes = test_endpoint
    candidate = _mask_mcp_servers_payload([saved])[0]
    candidate.update(id="different", headers={})
    response = client.post("/test", json={"server_id": saved["id"], "server": candidate})
    assert response.status_code == 200 and probes[-1]["auth_token"] == ""


def test_settings_save_detects_a_new_legacy_collision_with_unchanged_headers(mcp_settings):
    case = mcp_settings
    served = _served(case)
    served[0]["auth_header"] = "authorization"
    case.saved.clear()
    response = case.client.post("/api/settings", json={"MCP_SERVERS": served})
    assert response.status_code == 400 and not case.saved
