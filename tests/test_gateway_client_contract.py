"""The published client contract: ``GET /api/contract`` is DERIVED from the router-verified
endpoint index, the WS envelope index and the contract TypedDicts, and every fact it
states is checked against its owner here — router, handler ingress, real producers and
consumers, the setup producer, the auth gate. A route without shapes is published as an
explicit gap (``typed: false``), never as an empty schema."""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import re
import textwrap
import typing

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway import client_contracts, endpoint_index
from ouroboros.gateway.contracts import HTTP_ENDPOINTS, WS_MESSAGE_TYPES
from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.schema import (
    _BODY_MEDIA,
    CONTRACT_MODULES,
    contract_document,
    contract_typed_dict,
    json_schema_for,
    validate_ingress,
)
from tests.test_gateway_parity import _js_typedef_fields

REPO = pathlib.Path(__file__).resolve().parents[1]
TWIN = REPO / "web" / "modules" / "client_contract_types.js"


def _all_shapes() -> dict:
    return {**endpoint_index.HTTP_SHAPES, **endpoint_index.AUTH_ENDPOINTS}


def _body_names(spec: str) -> list[str]:
    names = []
    for alternative in spec.split("|"):
        kind, _, item = alternative.partition("/")
        names.extend([item] if kind in _BODY_MEDIA and item else [] if kind in _BODY_MEDIA else [kind])
    return names


def test_every_client_envelope_has_a_field_identical_browser_twin():
    from ouroboros.gateway import history_contracts

    text = TWIN.read_text(encoding="utf-8")
    for module in (client_contracts, history_contracts):
        for name in module.__all__:
            cls = getattr(module, name)
            assert re.search(rf"@typedef \{{Object\}} {name}\b", text), f"client_contract_types.js lacks {name}"
            assert _js_typedef_fields(text, name) == set(typing.get_type_hints(cls, include_extras=True)), name
    index = (REPO / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")
    assert "./client_contract_types.js" in index.splitlines()[0]
    assert "import('./client_contract_types.js').ServedUiState" in index


def test_contract_exposes_calls_without_diagnostic_probe_frames():
    from ouroboros.gateway.contracts import ChatInbound

    retired = {"realtime_execute", "realtime_prepared", "realtime_authorized",
               "realtime_refused", "realtime_finished"}
    assert not retired.intersection(endpoint_index.WS_SHAPES)
    assert "execution_transport" not in typing.get_type_hints(ChatInbound)
    document = (REPO / "docs" / "CLIENT_CONTRACT.md").read_text(encoding="utf-8")
    assert not any(frame in document for frame in retired)
    assert "POST /api/call/open" in endpoint_index.HTTP_SHAPES
    assert "conversation_finished" in endpoint_index.WS_SHAPES
    assert not validate_ingress({"type": "chat", "content": "Hello"}, ChatInbound)


def test_shapes_bind_only_indexed_routes_with_valid_bodies_that_resolve():
    assert set(endpoint_index.HTTP_SHAPES) <= set(HTTP_ENDPOINTS)
    for token, shape in _all_shapes().items():
        assert shape.responses, token
        for group in shape.responses.split():
            statuses, _, body = group.partition(":")
            assert body and all(re.fullmatch(r"[1-5]\d\d", s) for s in statuses.split(",")), (token, group)
            for name in _body_names(body):
                assert contract_typed_dict(name), (token, name)
        for name in _body_names(shape.request) if shape.request else []:
            assert contract_typed_dict(name), (token, name)


def test_auth_endpoints_are_the_gate_owned_public_paths(gated_client):
    from ouroboros import server_auth

    paths = {token.split(" ", 1)[1] for token in endpoint_index.AUTH_ENDPOINTS}
    assert paths == {"/auth/login", "/auth/logout"} and paths <= server_auth._PUBLIC_HTTP_PATHS
    # The inner fixture has no auth routes: these responses prove gate ownership.
    assert gated_client.get("/auth/login").status_code == 200
    logout = gated_client.get("/auth/logout", follow_redirects=False)
    assert logout.status_code == 303 and "Max-Age=0" in logout.headers["set-cookie"]


def test_ws_shapes_cover_the_frozen_ws_index_and_name_the_right_frames():
    assert isinstance(WS_MESSAGE_TYPES, tuple) and isinstance(HTTP_ENDPOINTS, tuple)
    assert set(endpoint_index.WS_SHAPES) == set(WS_MESSAGE_TYPES)
    for ws_type, (inbound, outbound) in endpoint_index.WS_SHAPES.items():
        assert inbound or outbound, ws_type
        for name in (inbound, outbound):
            if name:
                hint = typing.get_type_hints(contract_typed_dict(name))["type"]
                assert typing.get_args(hint) == (ws_type,), (ws_type, name)


_CLIENT_LOCAL_EVENTS = {"open", "close", "message", "outbound_dropped", "outbound_sent"}
_DOM_EVENTS = {"change", "click", "dragstart", "submit"}


def test_every_ws_type_the_builtin_client_consumes_is_published():
    """Real consumers: the SPA's subscriptions. Client-local socket events are not frames."""
    from ouroboros.review import is_gated_js_module

    pattern = re.compile(r"""\b(?:onWs|on|listen|ws\??\.on\??\.?)\(\s*['"]([a-z0-9_.]+)['"]""")
    consumed = set()
    for path in [*(REPO / "web").glob("*.js"), *(REPO / "web" / "modules").glob("*.js")]:
        if is_gated_js_module(path.relative_to(REPO).as_posix()):  # first-party, not vendored/minified
            consumed |= set(pattern.findall(path.read_text(encoding="utf-8")))
    consumed |= {"photo", "video"}  # chat_media.js subscribes these two in a loop
    unpublished = consumed - set(WS_MESSAGE_TYPES) - _CLIENT_LOCAL_EVENTS - _DOM_EVENTS
    assert not unpublished, f"the builtin client consumes unpublished WS types: {sorted(unpublished)}"
    assert {"ui_language_changed", "ui_i18n_updated"} <= consumed


def test_every_literal_ws_broadcast_type_is_published():
    """Real producers: every literal frame handed to a broadcast seam names an indexed type."""
    sites = [REPO / "server.py", *sorted((REPO / "ouroboros").rglob("*.py")), *sorted((REPO / "supervisor").rglob("*.py"))]
    produced: dict[str, str] = {}
    for path in sites:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name not in {"broadcast_ws", "broadcast_ws_sync", "_broadcast", "_broadcast_fn"} or not node.args:
                continue
            frame = node.args[0]
            if not isinstance(frame, ast.Dict):
                continue
            for key, value in zip(frame.keys, frame.values):
                if isinstance(key, ast.Constant) and key.value == "type" and isinstance(value, ast.Constant):
                    produced[value.value] = f"{path.relative_to(REPO)}:{node.lineno}"
    assert {"ui_language_changed", "ui_i18n_updated", "update_status_ready"} <= set(produced)
    unpublished = {t: where for t, where in produced.items() if t not in WS_MESSAGE_TYPES}
    assert not unpublished, f"literal WS broadcasts of unpublished types: {unpublished}"


def _handler_ingress(endpoint) -> set[str]:
    """TypedDict names an endpoint (and its same-module helpers, one level) validates ingress against."""
    fn = getattr(endpoint, "__wrapped__", endpoint)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        module_tree = ast.parse(pathlib.Path(inspect.getsourcefile(fn)).read_text(encoding="utf-8"))
    except (OSError, TypeError):
        return set()
    helpers = {n.name: n for n in module_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found: set[str] = set()
    visited: set[str] = set()

    def scan(node, depth):
        for call in (c for c in ast.walk(node) if isinstance(c, ast.Call)):
            called = call.func.id if isinstance(call.func, ast.Name) else ""
            if called == "validate_ingress" and len(call.args) > 1 and isinstance(call.args[1], ast.Name):
                found.add(call.args[1].id)
            if depth < 1 and called in helpers and called not in visited:
                visited.add(called)
                scan(helpers[called], depth + 1)

    scan(tree, 0)
    return found


def test_request_bindings_agree_with_the_handlers_ingress_validation(tmp_path):
    checked = 0
    for route in collect_routes(data_dir=tmp_path):
        methods = sorted(m for m in (getattr(route, "methods", None) or ()) if m not in {"HEAD", "OPTIONS"})
        validated = _handler_ingress(route.endpoint) if methods else set()
        if not validated:
            continue
        checked += 1
        shapes = [endpoint_index.HTTP_SHAPES.get(f"{method} {route.path}") for method in methods]
        assert any(shapes), f"{route.path} validates {validated} but publishes no shapes"
        declared = {name for shape in shapes if shape for name in _body_names(shape.request)}
        assert declared & validated, (route.path, declared, validated)
    assert checked >= 6, "the ingress cross-check lost its subjects"


def test_core_paths_cover_every_routed_path_and_the_interface_is_only_root_and_static(tmp_path):
    def under(path, prefix):
        return path == prefix or path.startswith(prefix + "/")

    for route in collect_routes(data_dir=tmp_path):
        assert any(under(route.path, p) for p in endpoint_index.CORE_PATHS), route.path
    assert endpoint_index.INTERFACE_PATHS == ("/", "/static")
    assert under("/onboarding/static", "/onboarding")


def _contract_app(tmp_path) -> Starlette:
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    return app


def test_the_contract_is_served_derived_and_self_consistent(tmp_path):
    with TestClient(_contract_app(tmp_path)) as client:
        response = client.get("/api/contract")
        health = client.get("/api/health").json()
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    document = response.json()
    assert document == json.loads(json.dumps(contract_document()))
    assert not validate_ingress(document, client_contracts.GatewayContractResponse)
    assert document["abi_version"] == health["abi_version"] == "7.0"
    tokens = [f"{e['method']} {e['path']}" for e in document["http"]]
    expected = [t for t in HTTP_ENDPOINTS if not t.startswith("WS ")] + list(endpoint_index.AUTH_ENDPOINTS)
    assert tokens == expected
    for entry in document["http"]:
        assert not validate_ingress(entry, client_contracts.ContractHttpEntry), entry
        token = f"{entry['method']} {entry['path']}"
        assert entry["typed"] == (token in _all_shapes()), token
        assert entry["auth"] == ("public" if entry["path"] in {"/api/health", "/auth/login", "/auth/logout"} else "gate")
        for body in [entry.get("request"), *entry.get("responses", {}).values()]:
            for name in (body or {}).get("schemas", []):
                assert name in document["schemas"], (token, name)
    assert {m["type"] for m in document["ws"]["messages"]} == set(WS_MESSAGE_TYPES)
    for name, schema in document["schemas"].items():
        assert schema["type"] == "object" and "properties" in schema, name
    # Published facts the journeys rely on.
    by_token = {f"{e['method']} {e['path']}": e for e in document["http"]}
    assert by_token["GET /onboarding"]["query"] == ["builtin"]
    assert by_token["GET /auth/login"]["query"] == ["next", "builtin"]
    assert set(by_token["GET /auth/login"]["responses"]) == {"200"}
    assert by_token["GET /api/onboarding/bootstrap"]["responses"]["200"]["schemas"] == ["OnboardingBootstrapResponse"]
    assert by_token["GET /api/tasks/{task_id}/events"]["responses"]["200"] == {
        "content_type": "text/event-stream", "schemas": ["TaskEvent"]}
    assert by_token["POST /api/chat/upload"]["request"]["content_type"] == "multipart/form-data"
    assert by_token["POST /api/settings"]["responses"]["503"]["schemas"] == ["SettingsWriteRefusal"]
    assert by_token["GET /api/tasks/{task_id}"]["path_params"] == ["task_id"]
    assert by_token["GET /api/extensions/{skill}/module/{entry:path}"]["path_params"] == ["skill", "entry"]
    assert all(entry["typed"] for entry in document["http"]), "every indexed route publishes its shapes"
    assert by_token["POST /api/reset"]["responses"]["200"]["schemas"] == ["ResetResponse"]
    assert by_token["POST /api/claudexor/account-resets"]["headers"] == ["Idempotency-Key!"]
    assert "served_ui" in document["schemas"]["StateResponse"]["properties"]
    history = document["schemas"]["ChatHistoryResponse"]["properties"]["messages"]["items"]
    assert {"text", "role", "ts", "is_progress", "system_type", "task_id"} <= set(history["required"])


def test_contract_modules_name_typed_dicts_only():
    for module_name in CONTRACT_MODULES:
        module = __import__(module_name, fromlist=["__all__"])
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, "__required_keys__"):
                assert json_schema_for(obj)["type"] == "object", name


def test_the_setup_bootstrap_types_mirror_the_producer_rows():
    from ouroboros import settings_setup_contract as producer

    bootstrap = producer.build_setup_bootstrap({}, "web", fresh_install=True)
    assert set(bootstrap) == set(typing.get_type_hints(client_contracts.SetupBootstrap))
    contract = bootstrap["contract"]
    assert set(contract) == set(typing.get_type_hints(client_contracts.SetupContract))
    rows = {
        "steps": client_contracts.SetupStep, "providerFields": client_contracts.SetupProviderField,
        "modelSlots": client_contracts.SetupModelSlot, "reviewModes": client_contracts.SetupModeChoice,
        "runtimeModes": client_contracts.SetupModeChoice, "localRoutingModes": client_contracts.SetupLocalRoutingMode,
        "budgetFields": client_contracts.SetupBudgetField, "subscriptionFields": client_contracts.SetupSubscriptionField,
    }
    for key, cls in rows.items():
        for row in contract[key]:
            assert set(row) == set(typing.get_type_hints(cls)), key
    for profile in contract["providerProfiles"].values():
        assert set(profile) == set(typing.get_type_hints(client_contracts.SetupProviderProfile))
    for preset in bootstrap["localPresets"].values():
        assert set(preset) == set(typing.get_type_hints(client_contracts.SetupLocalPreset))
    fixed = set(typing.get_type_hints(client_contracts.SetupInitialState))
    keyed = {row["stateKey"] for name in ("providerFields", "budgetFields", "modelSlots") for row in contract[name]}
    assert set(bootstrap["initialState"]) == fixed | keyed
    assert not validate_ingress(json.loads(json.dumps(bootstrap)), client_contracts.SetupBootstrap)


def test_the_bootstrap_read_writes_nothing_and_equals_the_page_bootstrap(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "data" / "settings.json"
    settings_path.parent.mkdir()
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", settings_path.parent)
    with TestClient(_contract_app(tmp_path)) as client:
        response = client.get("/api/onboarding/bootstrap")
        page = client.get("/onboarding")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    body = response.json()
    assert not validate_ingress(body, client_contracts.OnboardingBootstrapResponse)
    assert body["ready"] is False and body["bootstrap"]["freshInstall"] is True
    embedded = re.search(r"window\.__OURO_ONBOARDING_BOOTSTRAP__ = (\{.*?\});\n", page.text, re.S)
    assert embedded and json.loads(embedded.group(1)) == body["bootstrap"]
    assert not settings_path.exists(), "a bootstrap read authored settings.json"


# --- the auth gate: anonymous refusal and a same-origin return ----------------------------

@pytest.fixture
def gated_client(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros import server_auth

    monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "gate-test-password")
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_auth, "_auth_secret_cache", None)
    inner = Starlette(routes=[Route("/api/ping", lambda request: __import__(
        "starlette.responses", fromlist=["JSONResponse"]).JSONResponse({"ok": True}))])
    gate = server_auth.NetworkAuthGate(inner)

    async def remote_peer(scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            scope = {**scope, "client": ("198.51.100.7", 40000)}
        await gate(scope, receive, send)

    return TestClient(remote_peer, base_url="http://testserver")


@pytest.mark.parametrize(("next_url", "expected"), [
    ("/", "/"),
    ("/?_ouro_reason=x#chat", "/?_ouro_reason=x#chat"),
    ("/projects/a?b=1", "/projects/a?b=1"),
    ("//evil.example", "/"),
    ("/\\evil.example", "/"),
    ("/\t/evil.example", "/"),
    ("/\n/evil.example", "/"),
    ("https://evil.example/", "/"),
    ("javascript:alert(1)", "/"),
    ("", "/"),
])
def test_the_login_return_stays_on_the_same_origin(next_url, expected):
    from ouroboros.server_auth import _sanitize_next_url

    assert _sanitize_next_url(next_url) == expected


def test_anonymous_requests_are_refused_and_login_returns_to_the_sanitized_page(gated_client):
    refused = gated_client.get("/api/ping")
    assert refused.status_code == 401 and refused.json() == {"error": "Authentication required."}
    wrong = gated_client.post("/auth/login", json={"password": "nope", "next": "/"})
    assert wrong.status_code == 401 and "set-cookie" not in wrong.headers
    ok = gated_client.post("/auth/login", json={"password": "gate-test-password", "next": "/\\evil.example"})
    assert ok.status_code == 200 and ok.json() == {"ok": True, "next": "/"}
    assert not validate_ingress(ok.json(), client_contracts.AuthLoginResponse)
    assert gated_client.get("/api/ping").status_code == 200  # the cookie jar now carries the session
    form = gated_client.post("/auth/login", data={"password": "gate-test-password", "next": "/ui?x=1"},
                             follow_redirects=False)
    assert form.status_code == 303 and form.headers["location"] == "/ui?x=1"


def test_an_anonymous_websocket_is_refused_before_accept(gated_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with gated_client.websocket_connect("/ws"):
            pass
