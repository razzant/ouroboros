"""What a private compatible connection may send while its program is unidentified (``llm_ollama_guard``).

Synthetic transports only: every server here is a function of the request.
"""
import asyncio
import hashlib
import json

import httpx
import pytest

from tests._ollama import MESSAGES, GatewayHTTP, LlamaCppHTTP, OllamaHTTP, install_http as install_http

GATEWAY = "http://gateway.lan:8080/v1"
MODEL = "openai-compatible::mock-model"


@pytest.fixture
def world(monkeypatch):
    """Runtime settings a test edits in place; ``send`` is one real chat through the public client."""
    from ouroboros import config, llm_routing
    from ouroboros.llm import LLMClient

    settings = {"OUROBOROS_MODEL": MODEL, "OPENAI_COMPATIBLE_BASE_URL": GATEWAY, "OPENAI_COMPATIBLE_API_KEY": "key-one"}
    monkeypatch.setattr(config, "runtime_settings", lambda: settings)
    monkeypatch.setattr(llm_routing, "runtime_setting", settings.get)

    def send(model=MODEL, *, asynchronous=False):
        client = LLMClient()
        if asynchronous:
            return asyncio.run(client.chat_async(MESSAGES, model=model))
        return client.chat(MESSAGES, model=model)

    send.settings = settings
    return send


def _store():
    from ouroboros import capability_evidence as ce
    return ce._read(ce.canonical_evidence_root())


def _see_ollama(install_http, base_url=GATEWAY, key="seen-key"):
    """Ollama answers one connection's own read at ``base_url``; the address keeps that."""
    from ouroboros import llm_ollama
    from tests._ollama import TARGET

    install_http(OllamaHTTP())
    assert llm_ollama.is_ollama({**TARGET, "base_url": base_url, "api_key": key}, before_send=True)


NO_INFORMATION = {
    "401": httpx.Response(401, json={"error": {"message": "key required"}}),
    "403": httpx.Response(403, text="Forbidden"),
    "404": httpx.Response(404, json={"detail": "Not Found"}),
    "405": httpx.Response(405, text="Method Not Allowed"),
    "501": httpx.Response(501, text="Not Implemented"),
    "redirect": httpx.Response(302, headers={"Location": "http://elsewhere.lan/api/version"}),
    "html": httpx.Response(200, text="<!doctype html><title>Gateway</title>", headers={"content-type": "text/html"}),
    "empty": httpx.Response(200, content=b""),
    "other-json": httpx.Response(200, json=["not", "a", "document"]),
    "error-json": httpx.Response(200, json={"error": "unknown path"}),
    "bad-request": httpx.Response(400, json={"error": {"message": "unsupported"}}),
}
WITHHELD = {
    "408": httpx.Response(408, text="Request Timeout"),
    "429": httpx.Response(429, json={"error": "slow down"}),
    "500": httpx.Response(500, text="boom"),
    "502": httpx.Response(502, text="Bad Gateway"),
    "503": httpx.Response(503, json={"error": "loading"}),
    "504": httpx.Response(504, text="Gateway Timeout"),
    "cut-short": httpx.Response(200, content=b'{"version": "0.4'),
    "connect": httpx.ConnectError,
    "timeout": httpx.ReadTimeout,
    "broken": httpx.RemoteProtocolError,
}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reply", sorted(NO_INFORMATION))
def test_a_read_that_states_nothing_needs_the_connections_consent(install_http, world, reply, asynchronous):
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.loop_llm_call import classify_llm_exception

    server = install_http(GatewayHTTP(NO_INFORMATION[reply]))
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        world(asynchronous=asynchronous)
    verdict = classify_llm_exception(refused.value)
    assert (refused.value.code, verdict.kind, verdict.retry_same_request) == ("ollama_guard_consent", "bad_request", False)
    assert "Continue via the compatible API without confirmation that history is preserved" in str(refused.value)
    assert "OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED" in str(refused.value) and "nothing was sent" in str(refused.value)
    assert server.chats == [], "no generation"
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    message, usage = world(asynchronous=asynchronous)
    assert message["content"] == "from the gateway" and server.chats == ["/v1/chat/completions"]
    assert "did not show which program" in usage["route_caution"] and "by your consent" in usage["route_caution"]
    assert "may shorten" in usage["route_caution"] and "route_notice" not in usage
    assert not any(path.startswith("/api/chat") for _, path in server.paths)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reply", sorted(WITHHELD))
def test_a_busy_unreachable_or_cut_short_read_withholds_the_send_whatever_was_allowed(
        install_http, world, reply, asynchronous):
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.llm_ollama import OllamaRouteUnsettledError

    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY  # consent decides nothing here
    server = install_http(GatewayHTTP(WITHHELD[reply]))
    for _ in range(3):  # asking again changes nothing
        with pytest.raises(OllamaRouteUnsettledError) as held:
            world(asynchronous=asynchronous)
        assert not isinstance(held.value, UnidentifiedRouteWithheld) and "not sent" in str(held.value)
    assert server.chats == [] and server.paths == [("GET", "/api/version")] * 3


def test_a_redirect_is_never_followed(install_http, world):
    asked = []

    def gateway(request):
        asked.append(str(request.url))
        if request.url.host != "gateway.lan":
            pytest.fail("the redirect target was asked")
        if request.method == "POST":
            return GatewayHTTP()(request)
        return httpx.Response(307, headers={"Location": f"http://elsewhere.lan{request.url.path}"})

    install_http(gateway)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    assert world()[0]["content"] == "from the gateway"
    assert all("gateway.lan" in url for url in asked)


def test_signs_of_ollama_without_its_model_metadata_withhold_the_send(install_http, world):
    """A version and a model list in Ollama's shape, the model's own metadata refused: neither native nor /v1."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.llm_ollama import OllamaRouteUnsettledError

    native, gateway = OllamaHTTP(), GatewayHTTP()

    def partial(request):
        if request.url.path == "/api/show":
            return httpx.Response(403, text="Forbidden")
        return (native if request.url.path.startswith("/api/") else gateway)(request)

    install_http(partial)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    with pytest.raises(OllamaRouteUnsettledError, match="Conflicting") as held:
        world()
    assert not isinstance(held.value, UnidentifiedRouteWithheld)
    assert native.posts == [] and gateway.chats == [] and "native_protocols" not in _store()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_saved_identity_never_replaces_this_sends_own_read(install_http, world, asynchronous):
    """Ollama was identified and saved; the same connection's next read is refused: no native send on the saved
    word, no /v1 on the consent, and the saved route is not retired by silence."""
    from ouroboros import llm_ollama
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    native = install_http(OllamaHTTP())
    assert world(asynchronous=asynchronous)[0]["content"] == "391"
    assert len(native.posts) == 1
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    silent = install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    for _ in range(2):
        with pytest.raises(UnidentifiedRouteWithheld) as held:
            world(asynchronous=asynchronous)
        assert held.value.code == "ollama_guard_held"
    assert silent.chats == [] and len(native.posts) == 1
    assert [row["protocol"] for row in _store()["native_protocols"].values()] == ["ollama"]
    # The reverse: a saved negative word does not keep /v1 once Ollama answers this send.
    llm_ollama._rejected.clear()
    native = install_http(OllamaHTTP())
    assert world(asynchronous=asynchronous)[0]["content"] == "391" and len(native.posts) == 1


@pytest.mark.parametrize("change", ["key", "query", "header", "model", "renamed", "userinfo", "spelling"])
def test_an_address_where_ollama_was_seen_holds_every_unidentified_connection_there(install_http, world, change):
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    _see_ollama(install_http, "http://Gateway.LAN:8080/v1")
    row = {"kind": "openai-compatible", "name": "other", "base_url": GATEWAY, "api_key": "key-two",
           "continue_unconfirmed": GATEWAY}
    model = "openai-compatible@other::mock-model"
    if change == "query":
        row["base_url"] = row["continue_unconfirmed"] = GATEWAY + "?tenant=b"
    elif change == "model":
        model = "openai-compatible@other::another-model"
    elif change == "renamed":
        row["name"], model = "recreated", "openai-compatible@recreated::mock-model"
    elif change == "userinfo":
        row["base_url"] = row["continue_unconfirmed"] = "http://someone:word@gateway.lan:8080/v1"
    elif change == "spelling":
        row["base_url"] = row["continue_unconfirmed"] = "http://gateway.lan.:8080/v1/"
    world.settings["OUROBOROS_CONNECTIONS"] = [row]
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    headers = []

    def refuses(request):
        headers.append(request.headers.get("authorization"))
        return httpx.Response(403, text="Forbidden")

    server = install_http(GatewayHTTP(refuses))
    for route in ([model, MODEL] if change == "key" else [model]):  # the built-in connection too, with its consent
        with pytest.raises(UnidentifiedRouteWithheld) as held:
            world(route)
        words = str(held.value)
        assert held.value.code == "ollama_guard_held" and not held.value.retryable
        assert "Ollama was seen at http://gateway.lan" in words and ":8080 before" in words and "not identified" in words
        assert "/api/version" in words and "another configured route" in words
        assert "is Ollama" not in words and "Ollama server" not in words and "this Ollama" not in words
        assert "someone" not in words and "tenant" not in words and "key-two" not in words
    assert server.chats == [] and all(headers), "the read carried the connection's own credential and was refused"


@pytest.mark.parametrize("elsewhere", ["http://gateway.lan:8081/v1", "http://gateway.lan:8080/team-b/v1",
                                       "https://gateway.lan:8080/v1", "http://other.lan:8080/v1"])
def test_another_origin_or_prefix_is_not_held(install_http, world, elsewhere):
    _see_ollama(install_http)
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL=elsewhere, OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=elsewhere)
    server = install_http(GatewayHTTP())
    assert world()[0]["content"] == "from the gateway" and len(server.chats) == 1


def test_a_connections_own_positive_read_decides_for_it_and_lifts_nothing_for_others(install_http, world):
    """Behind one address a key reaches Ollama, another llama.cpp, a third shows nothing."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    world.settings["OUROBOROS_CONNECTIONS"] = [
        {"kind": "openai-compatible", "name": name, "base_url": GATEWAY, "api_key": f"key-{name}", "continue_unconfirmed": GATEWAY}
        for name in ("native", "cpp", "silent")]
    native, cpp, silent = OllamaHTTP(), LlamaCppHTTP(), GatewayHTTP(httpx.Response(403, text="Forbidden"))

    def by_key(request):
        key = request.headers["authorization"].removeprefix("Bearer key-")
        return {"native": native, "cpp": cpp, "silent": silent}[key](request)

    install_http(by_key)
    assert world("openai-compatible@native::mock-model")[0]["content"] == "391"
    message, usage = world("openai-compatible@cpp::mock-model")
    assert message["content"] == "from llama.cpp" and "route_caution" not in usage, "an identified program is ordinary /v1"
    assert ("GET", "/props") in cpp.paths and ("POST", "/api/chat") not in cpp.paths
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world("openai-compatible@silent::mock-model")
    assert held.value.code == "ollama_guard_held" and silent.chats == []
    assert world("openai-compatible@native::mock-model")[0]["content"] == "391" and len(native.posts) == 2
    assert world("openai-compatible@cpp::mock-model")[0]["content"] == "from llama.cpp"


def test_an_identified_program_needs_no_consent_and_carries_no_caution(install_http, world):
    """A new connection whose server names itself by its own document is ordinary /v1; Settings learns its name."""
    from ouroboros.provider_connections import connections_meta

    server = install_http(LlamaCppHTTP())
    message, usage = world()
    assert message["content"] == "from llama.cpp" and "route_caution" not in usage
    assert [path for method, path in server.paths if method == "GET"] == ["/api/version", "/props"]
    [builtin] = [entry for entry in connections_meta(world.settings, world.settings) if entry["id"] == "openai-compatible"]
    assert builtin["unidentified"]["checked"]["type"] == "llama.cpp" and builtin["unidentified"]["admission"] == ""


def test_a_route_from_before_the_update_keeps_v1_with_a_caution_until_it_changes(install_http, world, tmp_path):
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama_guard import PRIOR_ROUTES, UnidentifiedRouteWithheld, record_prior_routes
    from ouroboros.provider_connections import compatible_addresses

    world.settings["OUROBOROS_CONNECTIONS"] = [
        {"kind": "openai-compatible", "name": "lab", "base_url": "http://lab.lan/v1", "api_key": "lab-key"},
        {"kind": "openai-compatible", "name": "off", "base_url": "http://off.lan/v1", "enabled": False}]
    routes = compatible_addresses(world.settings)
    assert routes == [("openai-compatible", GATEWAY), ("openai-compatible@lab", "http://lab.lan/v1")], "a disabled one served nothing"
    assert record_prior_routes(routes) == 2
    assert set(_store()[PRIOR_ROUTES]) == {"openai-compatible", "openai-compatible@lab"}
    server = install_http(GatewayHTTP())
    for route in (MODEL, "openai-compatible@lab::mock-model"):
        message, usage = world(route)
        assert message["content"] == "from the gateway"
        assert "on record as answering a request before this check existed" in usage["route_caution"] and "may shorten" in usage["route_caution"]
    assert len(server.chats) == 2
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = "http://gateway.lan:9090/v1"  # the address changed
    world.settings["OUROBOROS_CONNECTIONS"].append(  # a connection added later
        {"kind": "openai-compatible", "name": "new", "base_url": "http://lab.lan/v1", "api_key": "lab-key"})
    for route in (MODEL, "openai-compatible@new::mock-model"):
        with pytest.raises(UnidentifiedRouteWithheld) as refused:
            world(route)
        assert refused.value.code == "ollama_guard_consent"
    assert "continue_unconfirmed in the new row of OUROBOROS_CONNECTIONS" in str(refused.value)
    assert len(server.chats) == 2
    world.settings["OPENAI_COMPATIBLE_API_KEY"] = "rotated-key"  # a rotated key is the same route
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = GATEWAY
    assert world()[0]["content"] == "from the gateway"
    assert ce.canonical_evidence_root() != tmp_path


def test_consent_names_an_endpoint_and_is_revoked_by_clearing_it(install_http, world):
    """The consent covers the base URL it names: its case, a default port and one closing slash are that same
    endpoint; another port, the same origin without its /v1 and another query are others."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    server = install_http(GatewayHTTP())
    for consent, allowed in [("", False), ("true", False), ("http://gateway.lan:9090/v1", False), (GATEWAY, True),
                             ("http://gateway.lan:8080", False), (GATEWAY + "?tenant=b", False),
                             ("HTTP://GATEWAY.lan:8080/v1/", True), ("", False)]:
        world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = consent
        if allowed:
            assert "by your consent" in world()[1]["route_caution"]
        else:
            with pytest.raises(UnidentifiedRouteWithheld):
                world()
    assert len(server.chats) == 2
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    _see_ollama(install_http)
    server = install_http(GatewayHTTP())
    with pytest.raises(UnidentifiedRouteWithheld) as held:  # consent never lifts the address hold
        world()
    assert held.value.code == "ollama_guard_held" and server.chats == []


def test_an_endpoint_is_every_part_that_decides_where_a_request_goes():
    """Permission's name of a base URL: only what no server can tell apart is one endpoint. A hold's name of the
    same URLs is broader, and never the same digest."""
    from ouroboros.llm_ollama_guard import endpoint_id, scope_id

    base = "http://gateway.lan:8080/team/v1"
    same = ["HTTP://Gateway.LAN:8080/team/v1", base + "/", base + "#fragment", " " + base + " "]
    assert {endpoint_id(url) for url in same} == {endpoint_id(base)} and len(endpoint_id(base)) == 24
    assert endpoint_id("https://gateway.lan/v1") == endpoint_id("https://gateway.lan:443/v1")
    others = [base + "?", base + "?tenant=b", base + "?tenant=c", base + "?tenant=b/", "http://user:word@gateway.lan:8080/team/v1",
              "http://user:other@gateway.lan:8080/team/v1", "http://gateway.lan:8080/team", "http://gateway.lan:8080/team/v1//",
              "http://gateway.lan:8080/Team/v1", "http://gateway.lan.:8080/team/v1", "https://gateway.lan:8080/team/v1",
              "http://gateway.lan:8081/team/v1"]
    ids = [endpoint_id(url) for url in [base, *others]]
    assert all(ids) and len(set(ids)) == len(ids), "each is another destination"
    one_address = [base, base + "?tenant=b", "http://user:word@gateway.lan:8080/team/v1", "http://gateway.lan:8080/team",
                   "http://gateway.lan.:8080/team/v1"]
    assert {scope_id(url) for url in one_address} == {scope_id(base)} and endpoint_id(base) != scope_id(base)
    assert endpoint_id("http://127.1:8080/v1") != endpoint_id("http://127.0.0.1:8080/v1"), "a permission follows the spelling"
    assert scope_id("http://127.1:8080/v1") == scope_id("http://127.0.0.1:8080/v1"), "a hold follows the address"
    for nothing in ("", "gateway.lan/v1", "ftp://gateway.lan/v1", "http://", "http://[::1", None):
        assert endpoint_id(nothing) == "", nothing


@pytest.mark.parametrize("admitted_by", ["consent", "prior"])
@pytest.mark.parametrize("moved_by", ["query", "userinfo"])
def test_permission_for_one_endpoint_admits_no_other_behind_the_same_address(install_http, world, admitted_by, moved_by):
    """A connection admitted at an address is pointed at another tenant or account behind it, as a headless edit
    does. To a hold that is one address; the consent and the prior route stay with the endpoint they were for."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld, record_prior_routes, scope_id

    if admitted_by == "consent":
        world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    else:
        assert record_prior_routes([("openai-compatible", GATEWAY)]) == 1
    server = install_http(GatewayHTTP())
    assert world()[0]["content"] == "from the gateway"
    moved = {"query": GATEWAY + "?tenant=b", "userinfo": "http://tenant-b:word-b@gateway.lan:8080/v1"}[moved_by]
    assert scope_id(moved) == scope_id(GATEWAY)
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = moved
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        world()
    assert refused.value.code == "ollama_guard_consent" and len(server.chats) == 1
    assert "tenant" not in str(refused.value) and "word-b" not in str(refused.value)
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = GATEWAY
    assert world()[0]["content"] == "from the gateway" and len(server.chats) == 2, "the endpoint it was given for"


LAB = {"kind": "openai-compatible", "name": "lab", "base_url": "http://lab.lan/v1", "api_key": "lab-key"}
LAB_MODEL = "openai-compatible@lab::mock-model"
SAVED = {"OUROBOROS_MODEL": MODEL, "OUROBOROS_MODEL_LIGHT": LAB_MODEL, "OPENAI_COMPATIBLE_BASE_URL": GATEWAY,
         "OPENAI_COMPATIBLE_API_KEY": "key-one", "OUROBOROS_CONNECTIONS": [LAB]}


@pytest.fixture
def installed(install_http, monkeypatch, tmp_path):
    """An install's own data root: the settings document the real loader reads, under the launch environment
    a test gives it, beside its evidence store. Called, it saves the document as a body left it; the first
    time, in a data root an earlier body stamped (or a new install's)."""
    from ouroboros import capability_evidence as ce
    from ouroboros import config

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    for key in ("OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED",
                "OPENAI_BASE_URL", "OPENAI_API_KEY", "OUROBOROS_CONNECTIONS", "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT"):
        monkeypatch.delenv(key, raising=False)

    def save(document, *, earlier_body=True):
        (root / "settings.json").write_text(json.dumps(document), encoding="utf-8")
        stamps = root / "state" / "migrations.json"
        if earlier_body and not stamps.exists():
            stamps.write_text(json.dumps({"retired_sets_generation": 1}), encoding="utf-8")
        return root

    return save


def _answered(root, model=MODEL, base_url=GATEWAY, key="key-one", *, kept="observations"):
    """What a body since 22.10.2026 left once a server answered ``model`` at ``base_url`` under ``key``: the size it
    accepted, under that body's own name of the route. The recipe is restated, not imported: were today's to drift
    from it, the rows an earlier body really left would no longer be found, and these tests would say so."""
    name = hashlib.sha256(json.dumps({"provider": "openai-compatible", "model": model, "base_url": base_url, "credentials": {
        "api_key": key, "user": "", "password": "", "scope": ""}}, sort_keys=True).encode()).hexdigest()[:24]
    path = root / "state" / "capability_evidence.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    store.setdefault("main_windows", {})[name] = {"sequence": 1, kept: [{
        "accepted": 1200, "refused": 0, "kind": "input", "reserve": 0, "transmitted": 0, "tokens": 1200,
        "ts": "2026-11-01T00:00:00+00:00", "sequence": 1}]}
    path.write_text(json.dumps(store), encoding="utf-8")


def _left_serving(root):
    """The routes of ``SAVED`` as an earlier body left them: each answered by its server."""
    _answered(root)
    _answered(root, LAB_MODEL, LAB["base_url"], LAB["api_key"])
    return root


def _prior():
    from ouroboros.llm_ollama_guard import PRIOR_ROUTES
    return set(_store().get(PRIOR_ROUTES) or {})


def _send(monkeypatch, model=MODEL):
    """One chat through the public client, on the settings the install's loader reads now."""
    from ouroboros import config, llm_routing
    from ouroboros.llm import LLMClient

    settings = config.load_settings(strict=True)
    monkeypatch.setattr(config, "runtime_settings", lambda: settings)
    monkeypatch.setattr(llm_routing, "runtime_setting", settings.get)
    return LLMClient().chat(MESSAGES, model=model)


def test_the_first_start_of_this_check_keeps_the_routes_a_server_answered_before_once(installed):
    from ouroboros import startup_migrations as boot

    idle = "openai-compatible@idle::mock-model"
    root = _left_serving(installed({**SAVED, "OUROBOROS_MODEL_SAFETY": idle, "OUROBOROS_CONNECTIONS": [
        LAB, {"kind": "openai-compatible", "name": "idle", "base_url": "http://idle.lan/v1"},
        {"kind": "openai-compatible", "name": "off", "base_url": "http://off.lan/v1", "enabled": False}]}))
    _answered(root, "openai-compatible@off::mock-model", "http://off.lan/v1", "")
    assert boot.record_prior_compatible_routes(root.parent / "another-root", earlier_body=True) is None
    assert "prior_compatible_routes_generation" not in boot.watermarks(root)
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 2
    assert boot.watermarks(root)["prior_compatible_routes_generation"] == boot.PRIOR_COMPATIBLE_ROUTES_GENERATION
    assert _prior() == {"openai-compatible", "openai-compatible@lab"}, "one never answered and one switched off keep nothing"
    assert "lab.lan" not in json.dumps(_store()["compatible_prior_routes"]), "kept as the endpoint's id alone"
    _answered(root, idle, "http://idle.lan/v1", "")  # answered after the snapshot: this body's own send
    assert boot.record_prior_compatible_routes(root, earlier_body=True) is None, "judged once"
    assert _prior() == {"openai-compatible", "openai-compatible@lab"}


NEW = "http://new.lan:8080/v1"
GIVEN = {  # document, launch variables, the route no earlier body ever sent, and the routes the record does name
    "a replaced document": ({**SAVED, "OPENAI_COMPATIBLE_BASE_URL": NEW}, {}, MODEL, {"openai-compatible@lab"}),
    "launch variables": ({"OUROBOROS_MODEL": MODEL}, {"OPENAI_COMPATIBLE_BASE_URL": NEW, "OPENAI_COMPATIBLE_API_KEY": "key-one"},
                         MODEL, set()),
    "launch rows": ({"OUROBOROS_MODEL": "openai-compatible@fresh::mock-model"}, {"OUROBOROS_CONNECTIONS": json.dumps(
        [{"kind": "openai-compatible", "name": "fresh", "base_url": NEW}])}, "openai-compatible@fresh::mock-model", set()),
    "another key": ({**SAVED, "OPENAI_COMPATIBLE_API_KEY": "key-two"}, {}, MODEL, {"openai-compatible@lab"}),
    "another model": ({**SAVED, "OUROBOROS_MODEL": "openai-compatible::other-model"}, {}, "openai-compatible::other-model",
                      {"openai-compatible@lab"}),
    "a renamed connection": ({**SAVED, "OUROBOROS_MODEL_LIGHT": "openai-compatible@lab2::mock-model",
                              "OUROBOROS_CONNECTIONS": [{**LAB, "name": "lab2"}]}, {}, "openai-compatible@lab2::mock-model",
                             {"openai-compatible"}),
}


@pytest.mark.parametrize("given", sorted(GIVEN) + ["another settings path"])
def test_what_a_start_is_given_proves_no_earlier_route(installed, install_http, monkeypatch, given):
    """An earlier body ran in this data root and left its answers there. This start reads a route none of them is
    for: a document put in place of the saved one, one the launch points at, variables it supplies, or the saved
    connection under another key, model or name. It serves, and it is no prior route: a request is withheld with
    the consent that would allow it, while a route the record does name keeps its own, wherever it is configured."""
    from ouroboros import config
    from ouroboros import startup_migrations as boot
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.provider_connections import compatible_addresses

    document, environment, model, on_record = GIVEN.get(given) or GIVEN["a replaced document"]
    root = _left_serving(installed(SAVED))
    if given == "another settings path":
        elsewhere = root.parent / "elsewhere" / "settings.json"
        elsewhere.parent.mkdir()
        elsewhere.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr(config, "SETTINGS_PATH", elsewhere)
    else:
        installed(document)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    assert dict(compatible_addresses(config.load_settings(strict=True))), "what the start is given does serve"
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == len(on_record) and _prior() == on_record
    assert model.partition("::")[0] not in on_record
    server = install_http(GatewayHTTP())
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        _send(monkeypatch, model)
    assert refused.value.code == "ollama_guard_consent" and server.chats == []
    assert "this connection's base URL" in str(refused.value) or "continue_unconfirmed in the" in str(refused.value)


@pytest.mark.parametrize("configured_by", ["Settings", "legacy pair", "launch variables", "launch rows"])
def test_a_route_a_server_answered_before_is_prior_however_it_is_configured(installed, install_http, monkeypatch, configured_by):
    """The proof is the earlier body's record of the answer, so it holds alike for a route the saved document names,
    the legacy OpenAI pair and one only the launch supplies, where it is the same connection, model, address and key."""
    from ouroboros import startup_migrations as boot

    document, environment, model = {
        "Settings": ({key: value for key, value in SAVED.items() if key != "OUROBOROS_MODEL_LIGHT"}, {}, MODEL),
        "legacy pair": ({"OUROBOROS_MODEL": MODEL, "OPENAI_BASE_URL": GATEWAY, "OPENAI_API_KEY": "key-one"}, {}, MODEL),
        "launch variables": ({"OUROBOROS_MODEL": MODEL}, {"OPENAI_COMPATIBLE_BASE_URL": GATEWAY, "OPENAI_COMPATIBLE_API_KEY": "key-one"}, MODEL),
        "launch rows": ({"OUROBOROS_MODEL": LAB_MODEL}, {"OUROBOROS_CONNECTIONS": json.dumps([LAB])}, LAB_MODEL),
    }[configured_by]
    root = _left_serving(installed(document))
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 1, "a connection no model setting names keeps nothing"
    assert _prior() == {model.partition("::")[0]}
    server = install_http(GatewayHTTP())
    message, usage = _send(monkeypatch, model)
    assert message["content"] == "from the gateway" and len(server.chats) == 1
    assert "on record as answering a request before this check existed" in usage["route_caution"] and "may shorten" in usage["route_caution"]


@pytest.mark.parametrize("without", ["a readable document", "an earlier body", "an answer", "a store this check left untouched"])
def test_without_proof_no_route_is_prior(installed, monkeypatch, without):
    """The boot continues and nothing is taken for a prior route: the settings cannot be read, the data root carried no
    earlier body's watermark, the store holds a window read and no answer, or this check already wrote of a route
    there (a record may then be this body's own, sent under a consent). Judged once, whatever a later start finds."""
    from ouroboros import capability_evidence as ce
    from ouroboros import startup_migrations as boot
    from ouroboros.llm_ollama_guard import note_check

    root = installed(SAVED, earlier_body=without != "an earlier body")
    if without == "an answer":
        _answered(root, kept="reads")
    else:
        _left_serving(root)
    if without == "a readable document":
        (root / "settings.json").write_text('{"OPENAI_COMPATIBLE_BASE_URL": "http://gateway', encoding="utf-8")
    elif without == "a store this check left untouched":
        note_check({"provider": "openai-compatible", "connection": "", "base_url": "http://other.lan/v1"}, "unknown")
        assert ce._read(root)["compatible_checks"]
    report = boot.prepare_startup_state(root, initialize_journal=True)
    assert report["status"] and _prior() == set()
    assert boot.watermarks(root)["prior_compatible_routes_generation"] == boot.PRIOR_COMPATIBLE_ROUTES_GENERATION
    _left_serving(installed(SAVED))  # everything is there at a later start: judged once, never again
    boot.prepare_startup_state(root, initialize_journal=True)
    assert _prior() == set()


def test_the_startup_door_keeps_the_routes_an_earlier_body_left_while_they_serve(installed):
    """The real boot door, start after start: the routes of the first start stay while their connections serve
    where they did; one configured later is never added; one gone at a start is retired and stays so when a
    connection is made again under its name at the same address."""
    from ouroboros import startup_migrations as boot

    root = _left_serving(installed(SAVED))
    boot.prepare_startup_state(root, initialize_journal=True)
    assert _prior() == {"openai-compatible", "openai-compatible@lab"}
    later = {"kind": "openai-compatible", "name": "later", "base_url": "http://later.lan/v1"}
    installed({**SAVED, "OPENAI_COMPATIBLE_API_KEY": "rotated", "OUROBOROS_CONNECTIONS": [{**LAB, "api_key": "rotated"}, later]})
    boot.prepare_startup_state(root, initialize_journal=True)
    assert _prior() == {"openai-compatible", "openai-compatible@lab"}, "rotated keys serve where they did; later is no prior route"
    installed({**SAVED, "OUROBOROS_CONNECTIONS": [later]})  # lab removed between two starts
    boot.prepare_startup_state(root, initialize_journal=True)
    assert _prior() == {"openai-compatible"}
    installed(SAVED)  # and made again under its name, at its address
    boot.prepare_startup_state(root, initialize_journal=True)
    assert _prior() == {"openai-compatible"}, "a recreated connection inherits nothing"
    (root / "settings.json").write_text("{", encoding="utf-8")  # unknown settings retire nothing
    boot.prepare_startup_state(root, initialize_journal=True)
    assert _prior() == {"openai-compatible"}


def test_a_prior_route_lives_only_while_its_connection_serves_where_it_was_kept(install_http):
    from ouroboros.llm_ollama_guard import PRIOR_ROUTES, record_prior_routes, retire_prior_routes

    lab = "http://lab.lan/v1"
    routes = [("openai-compatible", GATEWAY), ("openai-compatible@lab", lab), ("openai-compatible@gone", "http://gone.lan/v1"),
              ("openai-compatible@moved", lab)]
    assert record_prior_routes(routes) == 4
    asked = []

    def serving():  # the built-in one respelled, lab's key rotated, one gone, one moved to another tenant, one new
        asked.append("serving")
        return [("openai-compatible", "HTTP://gateway.lan:8080/v1/"), ("openai-compatible@lab", lab),
                ("openai-compatible@moved", lab + "?tenant=b"), ("openai-compatible@new", lab)]

    # Before a write lands: only what it would bring back into force goes, and a write that brings none back deletes none.
    assert retire_prior_routes(serving, reviving=serving) == 0 and len(_store()[PRIOR_ROUTES]) == 4
    assert retire_prior_routes(serving, reviving=lambda: [routes[3]]) == 1, "moved back to where its route was kept"
    assert set(_store()[PRIOR_ROUTES]) == {"openai-compatible", "openai-compatible@lab", "openai-compatible@gone"}
    assert retire_prior_routes(serving) == 1 and set(_store()[PRIOR_ROUTES]) == {"openai-compatible", "openai-compatible@lab"}
    assert retire_prior_routes(serving) == 0, "nothing more to retire"
    assert retire_prior_routes(lambda: routes) == 0, "made again under their names, where they were"
    assert set(_store()[PRIOR_ROUTES]) == {"openai-compatible", "openai-compatible@lab"}, "a retired route never returns"
    assert retire_prior_routes(lambda: []) == 2 and not _store().get(PRIOR_ROUTES)
    asked.clear()
    assert retire_prior_routes(serving, reviving=serving) == 0 and asked == [], "with no route kept, nothing is read"


@pytest.mark.parametrize("edit", ["removed", "switched off", "moved"])
def test_a_settings_save_retires_the_prior_route_of_a_connection_it_removes_disables_or_moves(
        installed, install_http, monkeypatch, edit):
    """Through the owner's settings writer, as Settings and headless saves go: a rotated key keeps the route; the
    connection gone, off or pointed elsewhere loses it, and saving it back under its name at its address asks consent."""
    from ouroboros import startup_migrations as boot
    from ouroboros.gateway.owner_settings import _owner_write_settings
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    root = _left_serving(installed(SAVED))
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 2
    server = install_http(GatewayHTTP())

    def save(rows):
        _owner_write_settings({**SAVED, "OUROBOROS_CONNECTIONS": rows})

    save([{**LAB, "api_key": "rotated-key"}])
    assert _prior() == {"openai-compatible", "openai-compatible@lab"}
    assert "on record as answering a request before this check existed" in _send(monkeypatch, LAB_MODEL)[1]["route_caution"]
    save({"removed": [], "switched off": [{**LAB, "enabled": False}],
          "moved": [{**LAB, "base_url": LAB["base_url"] + "?tenant=b"}]}[edit])
    assert _prior() == {"openai-compatible"}
    save([LAB])
    assert _prior() == {"openai-compatible"}, "its name and address are back; its route is not"
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        _send(monkeypatch, LAB_MODEL)
    assert refused.value.code == "ollama_guard_consent" and len(server.chats) == 1
    assert "continue_unconfirmed in the lab row of OUROBOROS_CONNECTIONS" in str(refused.value)


@pytest.mark.parametrize("edit", ["removed", "switched off", "moved"])
@pytest.mark.parametrize("store", ["cannot save", "is busy"])
def test_a_save_never_brings_back_a_prior_route_whose_retirement_failed(installed, install_http, monkeypatch, edit, store):
    """The evidence store takes no write while a save removes, switches off or moves a connection: that save lands
    and its route stays behind, kept for an endpoint nothing serves. No later save revives it. One that would bring
    the connection back under its name at that endpoint is refused before a byte is written, while one that brings
    none back is held up by nothing; once the store answers, the connection is saved again and asks consent."""
    from ouroboros import capability_evidence as ce
    from ouroboros import startup_migrations as boot
    from ouroboros.gateway.owner_settings import SettingsPreconditionFailed, _owner_write_settings
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    root = _left_serving(installed(SAVED))
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 2
    gone = {"removed": [], "switched off": [{**LAB, "enabled": False}], "moved": [{**LAB, "base_url": "http://lab.lan:9/v1"}]}[edit]
    saved = lambda: json.loads((root / "settings.json").read_text(encoding="utf-8"))  # noqa: E731

    def busy(_root):
        raise TimeoutError("capability evidence store is busy")

    with monkeypatch.context() as unavailable:
        if store == "is busy":
            unavailable.setattr(ce, "_store_transaction", busy)
        else:
            unavailable.setattr(ce, "_save", lambda _root, _data: False)
        _owner_write_settings({**SAVED, "OUROBOROS_CONNECTIONS": gone})
        assert saved()["OUROBOROS_CONNECTIONS"] == gone, "the connection is gone, off or elsewhere, whatever the store did"
        assert _prior() == {"openai-compatible", "openai-compatible@lab"}, "its route could not be retired"
        _owner_write_settings({**SAVED, "OUROBOROS_CONNECTIONS": gone, "OUROBOROS_MODEL_VISION": MODEL})
        assert saved()["OUROBOROS_MODEL_VISION"] == MODEL, "a save that brings no connection back waits on nothing"
        with pytest.raises(SettingsPreconditionFailed, match="nothing was written"):
            _owner_write_settings(dict(SAVED))
        assert saved()["OUROBOROS_CONNECTIONS"] == gone, "refused before the bytes: the connection is not back"
    _owner_write_settings(dict(SAVED))
    assert saved()["OUROBOROS_CONNECTIONS"] == [LAB] and _prior() == {"openai-compatible"}
    server = install_http(GatewayHTTP())
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        _send(monkeypatch, LAB_MODEL)
    assert refused.value.code == "ollama_guard_consent" and server.chats == []
    assert "on record as answering a request before this check existed" in _send(monkeypatch)[1]["route_caution"], "the route that kept serving stays"


def test_a_save_that_clears_the_built_in_address_retires_its_route_whatever_the_environment_still_mirrors(installed, monkeypatch):
    """The server's environment still carries the address its last projection mirrored from the document. The save
    that clears it is judged on the document it wrote; an operator's own variable would keep the connection serving."""
    from ouroboros import settings_integrity
    from ouroboros import startup_migrations as boot
    from ouroboros.gateway.owner_settings import _owner_write_settings

    root = _left_serving(installed(SAVED))
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 2
    cleared = {**SAVED, "OPENAI_COMPATIBLE_BASE_URL": "", "OPENAI_COMPATIBLE_API_KEY": ""}
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", GATEWAY)  # an operator's own variable: it serves there still
    _owner_write_settings(dict(cleared))
    assert _prior() == {"openai-compatible", "openai-compatible@lab"}
    monkeypatch.setenv(settings_integrity.PROJECTED_SETTINGS_ENV,  # this tree's own mirror of the earlier document
                       json.dumps({"OPENAI_COMPATIBLE_BASE_URL": settings_integrity._value_digest(GATEWAY)}))
    _owner_write_settings(dict(cleared))
    assert _prior() == {"openai-compatible@lab"}


def _test_settings(**more):
    return {"OPENAI_COMPATIBLE_BASE_URL": "http://draft.lan/v1", "OPENAI_COMPATIBLE_API_KEY": "draft-key",
            "OUROBOROS_MODEL": MODEL, **more}


def test_settings_test_admits_nothing_and_keeps_only_the_ollama_observation(install_http, world, monkeypatch):
    """Test on an unsaved draft: the same conditions as a send, a synthetic generation, no saved admission; what it
    read of Ollama survives the process and holds another connection at that address."""
    import ouroboros.gateway.models as provider_api
    from ouroboros import llm_ollama
    from ouroboros.llm_ollama_guard import CHECKS, HOLDS, PRIOR_ROUTES, TEST_CAUTION, TOLD, UnidentifiedRouteWithheld

    gateway = install_http(GatewayHTTP())
    refused = provider_api._run_provider_test_with_settings("openai-compatible", _test_settings())
    assert not refused["ok"] and "Continue via the compatible API" in refused["error"] and gateway.chats == []
    drafted = _test_settings(OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED="http://draft.lan/v1")
    passed = provider_api._run_provider_test_with_settings("openai-compatible", drafted)
    assert passed == {"ok": True, "caution": TEST_CAUTION}, "a passed test says what it did not confirm"
    [sent] = [json.loads(request.content) for request in gateway.requests if request.method == "POST"]
    assert sent["messages"] == [{"role": "user", "content": "Reply OK"}], "only the synthetic prompt"
    assert not {HOLDS, PRIOR_ROUTES, CHECKS, TOLD, "native_protocols"} & set(_store()), "Test saved no admission and no readiness"
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = "http://draft.lan/v1"
    with pytest.raises(UnidentifiedRouteWithheld) as still:  # the passed Test allowed nothing
        world()
    assert still.value.code == "ollama_guard_consent" and len(gateway.chats) == 1

    native = install_http(OllamaHTTP())
    assert provider_api._run_provider_test_with_settings("openai-compatible", _test_settings()) == {"ok": True}
    [observed] = _store()[HOLDS].values()
    assert (observed["type"], observed["version"]) == ("ollama", "0.40.16") and observed["observed_at"] and observed["source"]
    assert "native_protocols" not in _store() and len(native.posts) == 1
    llm_ollama._rejected.clear()  # a new process holds no verdict in memory: only the stored row speaks
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = "http://draft.lan/v1"
    silent = install_http(GatewayHTTP(httpx.Response(401, text="no")))
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world()
    assert held.value.code == "ollama_guard_held" and silent.chats == []
    held_test = provider_api._run_provider_test_with_settings("openai-compatible", drafted)
    assert not held_test["ok"] and "Ollama was seen at http://draft.lan before" in held_test["error"] and silent.chats == []


def test_an_old_ollama_gets_no_generation_even_from_test_and_its_address_keeps_the_observation(install_http, world):
    import ouroboros.gateway.models as provider_api
    from ouroboros.llm_ollama_guard import HOLDS, UnidentifiedRouteWithheld
    from ouroboros.llm_ollama import OllamaUpgradeRequired

    native = OllamaHTTP()
    posts = []

    def old(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.5"})
        if request.method == "POST" and request.url.path != "/api/show":
            posts.append(request.url.path)
        return native(request)

    install_http(old)
    result = provider_api._run_provider_test_with_settings("openai-compatible", _test_settings(
        OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED="http://draft.lan/v1"))
    assert not result["ok"] and "0.12.5" in result["error"] and "Upgrade to 0.12.6" in result["error"]
    [observed] = _store()[HOLDS].values()
    assert (observed["type"], observed["version"], observed["source"]) == ("ollama", "0.12.5", "/api/version+/api/show")
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL="http://draft.lan/v1", OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED="http://draft.lan/v1")
    with pytest.raises(OllamaUpgradeRequired, match="0.12.5"):
        world()
    silent = install_http(GatewayHTTP(httpx.Response(404, text="gone")))
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world()
    assert held.value.code == "ollama_guard_held" and posts == [] and silent.chats == []


def _shown(usage):
    """The line the round shows after its call (``loop_model_call`` delivers the route's words), or None."""
    from ouroboros.loop_transport import emit_route_notice

    said = []
    emit_route_notice(usage, said.append)
    assert len(said) <= 1, said
    return said[0].removeprefix("🌐 ") if said else None


def _unstreamed_client(monkeypatch):
    """The loop's client, asked for one JSON answer: the synthetic gateway speaks no event stream."""
    from ouroboros.llm import LLMClient

    client = LLMClient()
    chat = client.chat
    monkeypatch.setattr(client, "chat", lambda **kwargs: chat(**{**kwargs, "stream": False}))
    return client


@pytest.mark.parametrize("outcome", ["answered", "failed"])
def test_the_chat_hears_of_an_unidentified_route_at_its_first_use_whatever_the_outcome(
        install_http, world, tmp_path, monkeypatch, outcome):
    """Told once: later rounds and later tasks (another process, after a restart) send without repeating it;
    Settings keeps it. Another admission of the same address is another first use, and so is the same
    connection at an address whose path prefix differs, the words being the same (they never show it)."""
    from ouroboros.llm_ollama_guard import TOLD, record_prior_routes
    from ouroboros.loop_llm_call import call_llm_with_retry

    client = _unstreamed_client(monkeypatch)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    chat = (lambda request: httpx.Response(400, json={"error": {"message": "bad sampler", "type": "invalid_request_error"}})) \
        if outcome == "failed" else None
    server = install_http(GatewayHTTP(chat=chat))

    def hear(usage):
        message, _cost = call_llm_with_retry(client, MESSAGES, MODEL, None, "low", 1, tmp_path / "logs", "root", 1,
                                             None, usage, initial_messages=MESSAGES)
        assert (message is None) == (outcome == "failed"), usage.get("_last_llm_error")
        return _shown(usage)

    def task(rounds=2):
        usage = {}  # a task's own usage: nothing of it outlives the task
        return [hear(usage) for _ in range(rounds)]

    first = task()
    assert first[1] is None and "did not show which program" in first[0] and "gateway.lan:8080" in first[0]
    assert "by your consent" in first[0] and "may shorten" in first[0]
    assert task() == [None, None] and len(server.chats) == 4, "said once, sent every time"
    [told] = _store()[TOLD].values()
    assert told["told_at"] and "gateway" not in json.dumps(_store()[TOLD]), "kept by digest: the words name an address"
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = ""
    assert record_prior_routes([("openai-compatible", GATEWAY)]) == 1
    again = task(rounds=1)
    assert "on record as answering a request before this check existed" in again[0] and task(rounds=1) == [None]

    def consent_to(prefix):  # the connection and the words stay; the prefix they never show and its consent change
        address = f"http://gateway.lan:8080/{prefix}/v1"
        world.settings.update(OPENAI_COMPATIBLE_BASE_URL=address, OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=address)

    heard = []
    for prefix in ("east", "west", "east", "west"):  # a prefix already heard is not told again
        consent_to(prefix)
        heard.append(task(rounds=1)[0])
    assert [bool(words) for words in heard] == [True, True, False, False] and heard[0] == heard[1]
    usage = {}  # one task whose route changes between its rounds hears of each
    for prefix in ("north", "south", "south"):
        consent_to(prefix)
        heard.append(hear(usage))
    assert [bool(words) for words in heard[4:]] == [True, True, False]
    stored, spoken = json.dumps(_store()[TOLD]), " ".join(words for words in heard if words)
    assert len(_store()[TOLD]) == 6 and "gateway" not in stored, "a mark per route (two admissions, four prefixes), kept by digest"
    assert not any(prefix in stored + spoken for prefix in ("east", "west", "north", "south")), "no prefix is shown or kept"
    assert not any(mark in spoken for mark in _store()[TOLD]), "and no mark is shown"


def test_a_withheld_send_tells_every_task_it_stops_why_once(install_http, world, tmp_path, monkeypatch):
    from ouroboros.loop_llm_call import call_llm_with_retry

    client = _unstreamed_client(monkeypatch)
    _see_ollama(install_http)
    server = install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    for _task in range(2):
        usage, shown = {}, []
        for _ in range(2):
            message, _cost = call_llm_with_retry(client, MESSAGES, MODEL, None, "low", 3, tmp_path / "logs", "root", 1,
                                                 None, usage, initial_messages=MESSAGES)
            shown.append(_shown(usage))
            assert message is None and usage["_last_llm_error_kind"] == "bad_request"
        assert shown[1] is None and "Ollama was seen at http://gateway.lan:8080 before" in shown[0]
        assert "Ollama server" not in shown[0] and "is Ollama" not in shown[0] and "another configured route" in shown[0]
    assert server.chats == [] and server.paths.count(("GET", "/api/version")) == 4, "a hold is not retried"


def test_nothing_secret_reaches_the_stored_identifiers_or_the_words(install_http, world, monkeypatch, tmp_path):
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama import OllamaRouteUnsettledError
    from ouroboros.llm_ollama_guard import CHECKS, HOLDS, PRIOR_ROUTES, UnidentifiedRouteWithheld, address_scope, record_prior_routes, scope_id
    from ouroboros.provider_connections import compatible_addresses

    secret_url = "http://user-name:pass-word@gateway.lan:8080/tenant-token-9f3/v1"
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL=secret_url, OPENAI_COMPATIBLE_API_KEY="bearer-secret",
                          OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=secret_url + "?api_key=query-secret&trace=on")
    assert address_scope(secret_url + "?api_key=query-secret&trace=on#frag") == "http://gateway.lan:8080/tenant-token-9f3"
    assert address_scope("https://Example.LAN./v1/") == "https://example.lan:443" and not address_scope("gateway.lan/v1")
    assert address_scope("http://[0:0:0:0:0:0:0:1]:11434/v1") == address_scope("http://[::1]:11434") == "http://[::1]:11434"
    assert scope_id(secret_url) == scope_id("http://gateway.lan:8080/tenant-token-9f3/v1") and len(scope_id(secret_url)) == 24
    assert record_prior_routes(compatible_addresses(world.settings)) == 1
    install_http(GatewayHTTP())
    usage = world()[1]
    native = OllamaHTTP()  # the same server behind the gateway's path prefix
    install_http(lambda request: native(httpx.Request(
        request.method, str(request.url).replace("/tenant-token-9f3", ""), headers=request.headers, content=request.content)))
    assert world()[0]["content"] == "391"
    native.status, native.reply = 400, {"error": "synthetic refusal"}  # the native chat's own error words
    with pytest.raises(Exception) as refused:
        world()
    chain, cause = [], refused.value
    while cause is not None and len(chain) < 8:
        chain.append(str(cause))
        cause = cause.__cause__ or cause.__context__
    words = [usage["route_caution"], "\n".join(chain)]
    assert "HTTP 400 from Ollama chat at http://gateway.lan:8080" in words[-1]
    for service in (httpx.Response(503, text="busy"), httpx.ConnectError):  # metadata reads that withhold the send
        install_http(GatewayHTTP(service))
        with pytest.raises(OllamaRouteUnsettledError) as unsettled:
            world()
        words.append(str(unsettled.value))
    install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world()
    words.append(str(held.value))
    stored = (ce.canonical_evidence_root() / "state" / "capability_evidence.json").read_text()
    kept = json.loads(stored)
    assert {HOLDS, PRIOR_ROUTES, CHECKS} <= set(kept)
    for namespace in (HOLDS, PRIOR_ROUTES, CHECKS):
        text = json.dumps(kept[namespace])
        for secret in ("user-name", "pass-word", "tenant-token", "query-secret", "bearer-secret", "gateway.lan", "trace"):
            assert secret not in text, (namespace, secret)
    for said in words:  # the server is named by its origin; its private path prefix stays out of every sentence
        assert "gateway.lan:8080" in said and "/api/version" not in said.replace(" /api/version", "")
        for secret in ("user-name", "pass-word", "tenant-token", "query-secret", "bearer-secret"):
            assert secret not in said, (secret, said)


def test_a_tenant_named_by_query_or_userinfo_is_kept_as_a_digest_alone(install_http):
    """The endpoint a prior route, a check and a told caution are kept for is a digest: a base URL whose userinfo
    and query carry a tenant and its secret writes none of their text, and no sentence says them."""
    from ouroboros import capability_evidence as ce
    from ouroboros import llm_ollama_guard as guard
    from ouroboros.llm_ollama import _address

    url = "http://tenant-user:tenant-word@gateway.lan:8080/team-path/v1?tenant=query-tenant&api_key=query-secret"
    target = {"provider": "openai-compatible", "connection": "lab", "base_url": url}
    assert guard.record_prior_routes([("openai-compatible@lab", url)]) == 1
    guard.note_check(target, "unknown")
    words, route = guard.unidentified(target, _address(target))
    assert guard.first_telling(route) and not guard.first_telling(route)
    with pytest.raises(guard.UnidentifiedRouteWithheld) as refused:  # the next tenant behind the same origin and prefix
        guard.unidentified({**target, "base_url": url.replace("query-tenant", "another-tenant")}, _address(target))
    kept = json.loads((ce.canonical_evidence_root() / "state" / "capability_evidence.json").read_text())
    rows = [kept[guard.PRIOR_ROUTES], kept[guard.CHECKS], kept[guard.TOLD]]
    assert rows[0]["openai-compatible@lab"]["endpoint"] == guard.endpoint_id(url) and len(rows[1]) == len(rows[2]) == 1
    for said in (json.dumps(rows), words, str(refused.value)):
        for secret in ("tenant-user", "tenant-word", "team-path", "query-tenant", "query-secret", "another-tenant"):
            assert secret not in said, (secret, said)
    assert "gateway.lan:8080" in words and "gateway.lan" not in json.dumps(rows)


def test_two_prefixes_on_one_origin_are_each_told_without_their_prefix(install_http):
    """The words show only the origin and the connection's name: each route, a connection at the whole address
    under its admission, is told once, and no prefix reaches the words or the mark, whether the words of two
    routes differ (two connections) or are the same (one connection, its prefix changed)."""
    from ouroboros.llm_ollama import _address
    from ouroboros.llm_ollama_guard import endpoint_id, first_telling, unidentified

    said, routes = [], []
    for name, prefix in (("east", "east-token"), ("west", "west-token"), ("east", "east-token-2")):
        url = f"http://gateway.lan:8080/{prefix}/v1"
        target = {"provider": "openai-compatible", "connection": name, "base_url": url,
                  "continue_unconfirmed": endpoint_id(url)}
        words, route = unidentified(target, _address(target))
        said.append(words)
        routes.append(route)
    assert said[0] != said[1] and said[0] == said[2] and all(
        "gateway.lan:8080" in words and "token" not in words for words in said)
    assert len(set(routes)) == 3 and [first_telling(route) for route in routes] == [True, True, True]
    assert not first_telling(routes[0]) and not first_telling(routes[2])


def test_settings_reads_what_is_stored_and_asks_the_network_nothing(install_http, world):
    from ouroboros.provider_connections import connections_meta

    world.settings["OUROBOROS_CONNECTIONS"] = [
        {"kind": "openai-compatible", "name": "cloud", "base_url": "https://api.example.com/v1", "api_key": "k"}]
    install_http(GatewayHTTP())
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    world()
    install_http(lambda request: pytest.fail(f"Settings asked {request.url}"))
    meta = {entry["id"]: entry for entry in connections_meta(world.settings, world.settings)}
    fact = meta["openai-compatible"]["unidentified"]
    assert fact["admission"] == "consent" and fact["checked"]["type"] == "unknown" and "ollama" not in fact
    assert "unidentified" not in meta["openai-compatible@cloud"], "a public address is never asked or held"
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = ""
    _see_ollama(install_http)
    install_http(lambda request: pytest.fail(f"Settings asked {request.url}"))
    fact = connections_meta(world.settings, world.settings)
    fact = {entry["id"]: entry for entry in fact}["openai-compatible"]["unidentified"]
    assert fact["admission"] == "" and fact["ollama"]["version"] == "0.40.16" and fact["ollama"]["observed_at"]


@pytest.mark.parametrize("held", [False, True])
def test_a_window_read_asks_no_generation_and_is_withheld_by_nothing(install_http, world, tmp_path, held):
    """A task's start reads its server's window with force (``response_limits.probe_context_route``). That read is no
    generation: a server that names no program, without consent or at an address where Ollama was seen, still
    states its window in its own listing, while a send there stays withheld."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    if held:
        _see_ollama(install_http)
    gateway = install_http(GatewayHTTP(service=lambda request: (
        httpx.Response(200, json={"data": [{"id": "mock-model", "max_model_len": 32768}]})
        if request.url.path == "/v1/models" else httpx.Response(404, json={"detail": "Not Found"}))))
    evidence = ce.probe(tmp_path, provider="openai-compatible", model=MODEL, base_url=GATEWAY, allow_fetch=True,
                        force=True, api_key="key-one")
    assert evidence.window_tokens == 32768 and gateway.chats == []
    assert [row["type"] for row in _store()["compatible_checks"].values()] == ["unknown"], "the read is this key's check"
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        world()
    assert refused.value.code == ("ollama_guard_held" if held else "ollama_guard_consent") and gateway.chats == []


@pytest.mark.parametrize("other", ["http://localhost:8080/v1", "http://127.0.0.2:8080/v1", "http://[::1]:8080/v1"])
def test_two_names_of_this_machine_are_two_origins(install_http, world, other):
    """Each may be another server: Ollama seen at one holds nothing at the other, and a consent names one."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld

    seen_at = "http://127.0.0.1:8080/v1"
    _see_ollama(install_http, seen_at)
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL=other, OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=seen_at)
    server = install_http(GatewayHTTP())
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        world()
    assert refused.value.code == "ollama_guard_consent" and server.chats == [], "not held, and not admitted either"
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = other
    assert world()[0]["content"] == "from the gateway" and len(server.chats) == 1


@pytest.mark.parametrize("spelled", ["http://10.1:8080/v1", "http://[::ffff:10.0.0.1]:8080/v1"])
def test_a_private_numeric_address_is_guarded_however_it_is_spelled(install_http, world, spelled):
    """A hold reads 10.1 as 10.0.0.1, and so does the question whether the server is private at all: the spelling
    never sends a request past the check, the consent or a hold kept for the dotted quad."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld, scope_id
    from ouroboros.server_window import private_host

    quad = "http://10.0.0.1:8080/v1"
    assert private_host(spelled) and scope_id(spelled) == scope_id(quad)
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = spelled
    server = install_http(GatewayHTTP())
    with pytest.raises(UnidentifiedRouteWithheld) as refused:
        world()
    assert refused.value.code == "ollama_guard_consent" and server.chats == []
    assert ("GET", "/api/version") in server.paths, "its program was asked before anything was sent"
    _see_ollama(install_http, quad)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = spelled
    silent = install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world()
    assert held.value.code == "ollama_guard_held" and silent.chats == []


def test_a_saved_consent_is_the_id_of_its_endpoint_never_the_url(install_http, world):
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld, consent_token, endpoint_id, scope_id
    from ouroboros.provider_connections import canonical_builtin_consent, normalize_connections

    secret_url = "http://user-name:pass-word@gateway.lan:8080/tenant-token-9f3/v1"
    keyed = secret_url + "?api_key=query-secret"
    flat = {"OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED": secret_url}
    canonical_builtin_consent(flat, {})
    assert flat["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"].startswith("http"), "a value this write did not name stays as written"
    canonical_builtin_consent(flat, flat)
    rows, error = normalize_connections([
        {"kind": "openai-compatible", "name": "lab", "base_url": secret_url, "continue_unconfirmed": secret_url + "#frag"},
        {"kind": "openai-compatible", "name": "keyed", "base_url": keyed, "continue_unconfirmed": keyed},
        {"kind": "openai-compatible", "name": "revoked", "base_url": secret_url, "continue_unconfirmed": ""},
        {"kind": "openai-compatible", "name": "worded", "base_url": secret_url, "continue_unconfirmed": "true"}])
    token = endpoint_id(secret_url)
    assert not error and len(token) == 24 and token != scope_id(secret_url), "never the id the address hold keeps"
    assert [row["continue_unconfirmed"] for row in rows] == [token, endpoint_id(keyed), "", ""], "a URL becomes its id, a word names none"
    assert endpoint_id(keyed) != token and scope_id(keyed) == scope_id(secret_url), "one address, two endpoints"
    assert flat == {"OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED": token} and consent_token(token) == token
    for secret in ("user-name", "pass-word", "tenant-token", "query-secret", "gateway.lan"):
        assert secret not in json.dumps([flat, [row["continue_unconfirmed"] for row in rows]])
    # The saved id admits the endpoint it names and no other; the id the hold keeps of that address admits nothing.
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL=secret_url, **flat)
    server = install_http(GatewayHTTP())
    assert "by your consent" in world()[1]["route_caution"]
    for other in ("http://gateway.lan:8080/another-tenant/v1", keyed, "http://gateway.lan:8080/tenant-token-9f3/v1",
                  secret_url.replace("pass-word", "another-word")):
        world.settings["OPENAI_COMPATIBLE_BASE_URL"] = other
        with pytest.raises(UnidentifiedRouteWithheld) as refused:
            world()
        assert refused.value.code == "ollama_guard_consent" and len(server.chats) == 1, other
    world.settings.update(OPENAI_COMPATIBLE_BASE_URL=secret_url, OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=scope_id(secret_url))
    with pytest.raises(UnidentifiedRouteWithheld):
        world()
    assert len(server.chats) == 1


def test_a_test_that_reached_v1_says_the_type_is_unconfirmed_when_its_chat_fails_too(install_http):
    """The chat's own 401 is its own error; the caution about the unnamed server is said beside it."""
    import ouroboros.gateway.models as provider_api

    gateway = install_http(GatewayHTTP(chat=lambda request: httpx.Response(
        401, json={"error": {"message": "bad key", "type": "invalid_api_key"}})))
    result = provider_api._run_provider_test_with_settings("openai-compatible", _test_settings(
        OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED="http://draft.lan/v1"))
    from ouroboros.llm_ollama_guard import TEST_CAUTION

    assert result == {"ok": False, "error": "Invalid key", "caution": TEST_CAUTION} and len(gateway.chats) == 1


@pytest.mark.parametrize("ticked", [True, False])
def test_the_built_in_keys_test_checks_the_consent_its_form_shows(install_http, monkeypatch, ticked):
    """The built-in pair's Test reads the draft's consent, ticked or cleared, over the saved one, and saves neither."""
    import ouroboros.gateway.models as provider_api

    address = "http://draft.lan/v1"
    saved = _test_settings(OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED="" if ticked else address)
    monkeypatch.setattr(provider_api, "load_settings", lambda: dict(saved))
    gateway = install_http(GatewayHTTP())
    result = provider_api._run_provider_test(
        "openai-compatible", {"OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED": address if ticked else ""}, "",
        "OPENAI_COMPATIBLE_API_KEY")
    assert result["ok"] is ticked and len(gateway.chats) == int(ticked)
    assert ("caution" in result) is ticked and ("Continue via the compatible API" in result.get("error", "")) is not ticked
    assert saved["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] == ("" if ticked else address)


def test_signs_of_ollama_beside_another_programs_document_still_withhold(install_http, world):
    """A version and a model list in Ollama's shape with llama.cpp's own /props: contradictory, so nothing is sent."""
    from ouroboros.llm_ollama import OllamaRouteUnsettledError

    native, cpp = OllamaHTTP(), LlamaCppHTTP()

    def both(request):
        if request.url.path == "/api/show":
            return httpx.Response(403, text="Forbidden")
        return (native if request.url.path.startswith("/api/") else cpp)(request)

    install_http(both)
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    with pytest.raises(OllamaRouteUnsettledError, match="Conflicting"):
        world()
    assert native.posts == [] and ("POST", "/v1/chat/completions") not in cpp.paths


def test_a_hold_outlives_a_restart_its_cache_and_other_writers(install_http, world):
    """The hold is a row of the stored file: a process that remembers nothing reads it, however old it is and
    whatever the store's other writers expire."""
    from ouroboros import capability_evidence as ce, llm_ollama
    from ouroboros.llm_ollama_guard import HOLDS, UnidentifiedRouteWithheld, scope_id
    from ouroboros.utils import json_snapshot

    _see_ollama(install_http)
    root = ce.canonical_evidence_root()
    path = root / "state" / "capability_evidence.json"
    kept = json.loads(path.read_text())
    assert set(kept[HOLDS]) == {scope_id(GATEWAY)}
    # Years later, in another process: no native route saved, every dated row long past any TTL.
    kept.pop("native_protocols")
    kept[HOLDS][scope_id(GATEWAY)]["observed_at"] = "2020-01-01T00:00:00+00:00"
    kept["probes"] = {"stale": {"status": "confirmed", "ts": "2020-01-01T00:00:00+00:00", "window_tokens": 1}}
    path.write_text(json.dumps(kept))
    llm_ollama._rejected.clear()
    assert json_snapshot(path)[HOLDS], "the reader takes the file, not a memory of it"
    assert ce._store_evidence(root, "image_input", "other-writer", {"observed_at": "2026-11-20T00:00:00+00:00"})
    stored = json.loads(path.read_text())
    assert "stale" not in stored["probes"] and set(stored[HOLDS]) == {scope_id(GATEWAY)}, "expiry is the probes' alone"
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    silent = install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world()
    assert held.value.code == "ollama_guard_held" and silent.chats == []


def test_settings_says_a_withheld_send_left_the_type_unconfirmed(install_http, world):
    """A refused send and a native route that lost its identification both leave 'not confirmed' for Settings."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.provider_connections import connections_meta

    def fact():
        return {entry["id"]: entry for entry in connections_meta(world.settings, world.settings)}["openai-compatible"]["unidentified"]

    install_http(GatewayHTTP())
    with pytest.raises(UnidentifiedRouteWithheld):
        world()
    assert fact()["checked"]["type"] == "unknown" and fact()["admission"] == "" and "ollama" not in fact()
    install_http(OllamaHTTP())
    assert world()[0]["content"] == "391" and fact()["checked"]["type"] == "ollama"
    install_http(GatewayHTTP(httpx.Response(403, text="Forbidden")))
    with pytest.raises(UnidentifiedRouteWithheld) as held:
        world()
    assert held.value.code == "ollama_guard_held"
    assert fact()["checked"]["type"] == "unknown" and fact()["ollama"]["version"] == "0.40.16"


def test_settings_reads_an_old_ollamas_needed_release_and_a_check_of_the_endpoint_it_read(install_http, world):
    """What Settings says of a connection comes from stored facts: an Ollama too old for its native chat carries
    the release to update to, from the transport's own threshold; a check is of the endpoint it read."""
    from ouroboros.llm_ollama import GUARD_SINCE
    from ouroboros.llm_ollama_guard import note_check, note_ollama
    from ouroboros.provider_connections import connections_meta

    target = {"provider": "openai-compatible", "connection": "", "base_url": GATEWAY}

    def fact():
        return {entry["id"]: entry for entry in connections_meta(world.settings, world.settings)}["openai-compatible"]["unidentified"]

    assert note_ollama(target, "0.12.5", "/api/version+/api/show")
    note_check(target, "ollama", version="0.12.5")
    assert fact()["ollama"]["update_to"] == fact()["checked"]["update_to"] == GUARD_SINCE == "0.12.6"
    assert (fact()["checked"]["type"], fact()["checked"]["version"]) == ("ollama", "0.12.5")
    assert note_ollama(target, "0.40.16", "/api/version+/api/ps+/api/show")
    note_check(target, "ollama", version="0.40.16")
    assert "update_to" not in fact()["ollama"] and "update_to" not in fact()["checked"]
    assert fact()["checked"]["version"] == "0.40.16", "a check is rewritten when its version changes"
    note_check(target, "llama.cpp")
    assert set(fact()["checked"]) == {"type", "observed_at"} and fact()["checked"]["type"] == "llama.cpp"
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = GATEWAY + "?tenant=b"
    assert "checked" not in fact() and fact()["ollama"]["version"] == "0.40.16", "another endpoint at the same address"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("meanwhile", ["ollama seen there", "another program named", "still unnamed", "consent withdrawn"])
def test_a_send_that_waited_for_its_slot_is_decided_by_its_check_after_the_wait(
        install_http, world, monkeypatch, tmp_path, asynchronous, meanwhile):
    """Admitted by consent, then queued for its model slot: the read made after the slot and the owner's consent as
    it stands then decide this send; the answer and the consent it had before the wait decide nothing."""
    from ouroboros import model_concurrency as slots, usage_accounting as ua
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld, note_ollama
    from tests._usage_store_testing import attempt_rows_in_start_order

    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = GATEWAY
    gateway, cpp = GatewayHTTP(), LlamaCppHTTP()
    queued = []
    install_http(lambda request: (cpp if queued and meanwhile == "another program named" else gateway)(request))
    take = slots._Waiter.step

    def wait_once(waiter):
        if not queued:
            queued.append(waiter)
            if meanwhile == "ollama seen there":  # another connection's read at this address, while this send waits
                assert note_ollama({"base_url": GATEWAY}, "0.40.16", "/api/version+/api/ps+/api/show")
            if meanwhile == "consent withdrawn":  # the owner clears the box while this send waits
                world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = ""
            return False
        return take(waiter)

    monkeypatch.setattr(slots._Waiter, "step", wait_once)
    monkeypatch.setattr(slots, "_queue_root", lambda: tmp_path / "slots")
    monkeypatch.setattr(slots, "_cap", lambda local: 1)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="queue", root_task_id="queue")):
        if meanwhile in ("ollama seen there", "consent withdrawn"):
            with pytest.raises(UnidentifiedRouteWithheld) as held:
                world(asynchronous=asynchronous)
            code = "ollama_guard_held" if meanwhile == "ollama seen there" else "ollama_guard_consent"
            assert held.value.code == code and gateway.chats == []
            [prepared] = attempt_rows_in_start_order(tmp_path)
            assert prepared["state"] == "released", "nothing was sent, and the prepared request is given back"
        else:
            message, usage = world(asynchronous=asynchronous)
            named = meanwhile == "another program named"
            assert message["content"] == ("from llama.cpp" if named else "from the gateway")
            assert ("route_caution" in usage) is not named, "the usage says what the last check found"
    assert queued and gateway.paths.count(("GET", "/api/version")) + cpp.paths.count(("GET", "/api/version")) == 2


def _refused_only(root, model=MODEL, base_url=GATEWAY, key="key-one"):
    """The earlier body's rows for one bound target that only ever REFUSED: ``_answered``'s recipe, accepted 0."""
    name = hashlib.sha256(json.dumps({"provider": "openai-compatible", "model": model, "base_url": base_url, "credentials": {
        "api_key": key, "user": "", "password": "", "scope": ""}}, sort_keys=True).encode()).hexdigest()[:24]
    path = root / "state" / "capability_evidence.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    store.setdefault("main_windows", {})[name] = {"sequence": 1, "observations": [{
        "accepted": 0, "refused": 1500, "kind": "total", "reserve": 0, "transmitted": 1500,
        "ts": "2026-11-01T00:00:00+00:00", "sequence": 1}]}
    path.write_text(json.dumps(store), encoding="utf-8")
    return root


def test_a_refusal_only_history_proves_no_prior_route(installed):
    """A refusal is no answer: the migration snapshot wants a positive accepted size in the earlier body's own
    record, so a route its server only ever refused asks the owner's consent like a new one."""
    from ouroboros import startup_migrations as boot

    root = _refused_only(installed(SAVED))
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 0
    assert _prior() == set()


def test_an_unreadable_store_defers_every_decision_that_needs_it(installed, install_http, monkeypatch):
    """An existing evidence store that cannot be parsed is unknown, never empty: the migration snapshot defers
    unstamped, a retirement raises, Settings keeps working with honest facts, and a send whose admission the
    store would decide is withheld — nothing is admitted from an imagined empty store, and no guard write
    'repairs' the file by overwriting it."""
    from ouroboros import capability_evidence as ce
    from ouroboros import startup_migrations as boot
    from ouroboros.llm_ollama import OllamaRouteUnsettledError
    from ouroboros.llm_ollama_guard import (EvidenceStoreUnavailable, connection_facts, retire_prior_routes,
                                            store_checked)

    root = _left_serving(installed(SAVED))
    path = ce._store_path(root)
    good = path.read_text(encoding="utf-8")
    path.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(EvidenceStoreUnavailable):
        store_checked(root)
    assert boot.record_prior_compatible_routes(root, earlier_body=True) is None
    assert "prior_compatible_routes_generation" not in boot.watermarks(root), "deferred, not judged"
    with pytest.raises(OSError):
        retire_prior_routes(lambda: [], root=root)
    install_http(GatewayHTTP())
    with pytest.raises(OllamaRouteUnsettledError) as withheld:
        _send(monkeypatch)
    assert "evidence store could not be read" in str(withheld.value)
    facts = connection_facts("openai-compatible", GATEWAY, GATEWAY)["unidentified"]
    assert facts["evidence"] == "unreadable" and facts["admission"] == "consent"
    assert path.read_text(encoding="utf-8") == "{corrupt", "no guard write overwrote the unreadable store"
    path.write_text(good, encoding="utf-8")
    assert boot.record_prior_compatible_routes(root, earlier_body=True) == 2, "the deferred snapshot returns with the store"


def test_equivalent_path_spellings_are_one_address_and_one_endpoint():
    """Holds and permissions derive the path the transport actually sends: httpx resolves dot segments before
    the request leaves (gateway/models.py documents it), so equivalent spellings of one prefix cannot acquire
    different holds, and distinct tenant prefixes stay distinct."""
    import httpx

    from ouroboros.llm_ollama_guard import _wire_path, address_scope, endpoint_id, scope_id

    assert address_scope("http://h.lan/a/../b") == address_scope("http://h.lan/b")
    assert scope_id("http://h.lan/x/y/../../v1") == scope_id("http://h.lan/v1")
    assert scope_id("http://h.lan/a/..") == scope_id("http://h.lan/")
    assert endpoint_id("http://h.lan/a/../v1") == endpoint_id("http://h.lan/v1")
    assert endpoint_id("http://h.lan/a") != endpoint_id("http://h.lan/b")
    for path in ("/a/../b", "/./b", "/a/b/..", "/..", "/a/./../c", "/x/../../y", "/a/b/.././c/"):
        assert _wire_path(path) == httpx.URL("http://h.lan" + path).raw_path.decode("ascii"), path


QUERY = "shard=b&shard=a&path=a%2fb%2Bc%3D&mode=x%20y&flag&empty="  # synthetic: order, a repeat, escapes, a bare and a blank name


def test_metadata_reads_keep_a_queried_base_verbatim():
    """The metadata read keeps the base URL's query intact and its path prefix, minus the client's own /v1:
    on a query-routed gateway the token that selects the tenant travels unmangled, so the read identifies the
    same tenant that then receives the history. Userinfo travels with the transport but never reaches words."""
    from ouroboros.llm_ollama import _url

    assert _url({"base_url": "http://h.lan/tenant?route=1"}, "/api/version") == "http://h.lan/tenant/api/version?route=1"
    assert _url({"base_url": "http://h.lan/v1?q="}, "/api/show") == "http://h.lan/api/show?q="
    assert _url({"base_url": "http://h.lan/v1"}, "/api/version") == "http://h.lan/api/version"
    assert _url({"base_url": "http://h.lan/v1/"}, "/api/ps") == "http://h.lan/api/ps"
    assert _url({"base_url": "http://h.lan/tenant/"}, "/api/version") == "http://h.lan/tenant/api/version"
    assert _url({"base_url": "http://u:p@h.lan:11434/v1?key=v"}, "/api/version") == "http://u:p@h.lan:11434/api/version?key=v"
    assert _url({"base_url": "http://h.lan:8080/tenant/v1?" + QUERY}, "/api/show") == (
        "http://h.lan:8080/tenant/api/show?" + QUERY), "order, repeats and escapes stay as written"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("keyless", [False, True])
def test_the_sdk_client_composes_each_path_before_the_base_query(asynchronous, keyless):
    """The other side of ``test_metadata_reads_keep_a_queried_base_verbatim``: the stock SDK concatenates each
    subpath after a base's raw path, query inside it, which would fold ``/chat/completions`` into the query's
    value: another tenant than the read permitting the send. The client the route builds keeps the query bytes
    as typed (order, repeats, escapes, bare and blank names) after the composed path, on its own requests and
    on a ``with_options`` copy, keyed and keyless alike; the keyless form still sends no credential."""
    from ouroboros.net_transport import openai_client_class
    from tests._ollama import _chat_completion

    base_url = "http://gateway.lan:8080/tenant/v1?" + QUERY
    target = {"provider": "openai-compatible", "base_url": base_url, "keyless": keyless, "api_key": "" if keyless else "k"}
    asked = []

    def gateway(request):
        asked.append(request)
        return _chat_completion("from the gateway")

    http_client = (httpx.AsyncClient if asynchronous else httpx.Client)(transport=httpx.MockTransport(gateway),
                                                                         trust_env=False)
    client = openai_client_class(target, async_client=asynchronous)(
        base_url=base_url, api_key=target["api_key"], http_client=http_client, max_retries=0)
    copy = client.with_options(timeout=5)
    ask = lambda sdk: sdk.chat.completions.create(model="mock-model", messages=MESSAGES)  # noqa: E731
    if asynchronous:
        async def both():
            try:
                return [await ask(sdk) for sdk in (client, copy)]
            finally:
                await client.close()
        replies = asyncio.run(both())
    else:
        try:
            replies = [ask(sdk) for sdk in (client, copy)]
        finally:
            client.close()
    assert [reply.choices[0].message.content for reply in replies] == ["from the gateway"] * 2
    assert [request.url.raw_path for request in asked] == [b"/tenant/v1/chat/completions?" + QUERY.encode()] * 2
    assert all(("authorization" in request.headers) is not keyless for request in asked)


def test_only_a_private_compatible_address_with_a_query_takes_the_query_form():
    """The route whose metadata read keeps the query verbatim is the one whose send composes before it; an
    address without a query, a query only inside a fragment, a public host and every other route keep the SDK's
    own class, keyed or keyless."""
    import openai

    from ouroboros.net_transport import openai_client_class

    private = "http://gateway.lan:8080/v1"
    for provider, base_url in (("openai-compatible", private), ("openai-compatible", private + "#?route=1"),
                               ("openai-compatible", "https://api.example.com/v1?route=1"),
                               ("openrouter", private + "?route=1"), ("deepseek", private + "?route=1")):
        for keyless in (False, True):
            target = {"provider": provider, "base_url": base_url, "keyless": keyless, "api_key": ""}
            classes = (openai_client_class(target), openai_client_class(target, async_client=True))
            assert (classes[0] is openai.OpenAI, classes[1] is openai.AsyncOpenAI) == (not keyless, not keyless), target
            assert not any(cls.__name__.startswith("Queried") for cls in classes), target
    queried = openai_client_class({"provider": "openai-compatible", "base_url": private + "?route=1"})
    assert issubclass(queried, openai.OpenAI) and queried.__name__ == "QueriedOpenAI"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("no_proxy", [False, True])
def test_the_permitted_chat_and_its_read_name_one_connection(install_http, world, no_proxy, asynchronous):
    """The read that permits a compatible send keeps the base's own prefix and query verbatim (``_url``); the
    client the route builds composes each path before the same query bytes, so on both transport lanes of both
    clients the metadata reads and the permitted chat name one connection: one origin, one prefix, one query
    with its order, repeats and escapes."""
    from ouroboros.llm import LLMClient

    address = "http://gateway.lan:8080/tenant/v1?" + QUERY  # a synthetic tenant prefix and route query
    world.settings["OPENAI_COMPATIBLE_BASE_URL"] = address
    world.settings["OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"] = address
    server = install_http(GatewayHTTP())
    if asynchronous:
        message, usage = asyncio.run(LLMClient().chat_async(MESSAGES, model=MODEL, no_proxy=no_proxy))
    else:
        message, usage = LLMClient().chat(MESSAGES, model=MODEL, no_proxy=no_proxy)
    assert message["content"] == "from the gateway" and "route_caution" in usage
    version = next(request.url for request in server.requests if request.url.path == "/tenant/api/version")
    chat = next(request.url for request in server.requests if request.method == "POST")
    assert str(version) == "http://gateway.lan:8080/tenant/api/version?" + QUERY
    assert str(chat) == "http://gateway.lan:8080/tenant/v1/chat/completions?" + QUERY
    assert (version.scheme, version.host, version.port) == (chat.scheme, chat.host, chat.port)
    assert all(request.url.query == QUERY.encode() for request in server.requests), \
        "every request of the call names the connection its decision was made about"


def test_consent_writes_withdraw_honestly_against_the_launch_environment(monkeypatch):
    """A grant is stored as its endpoint's id. A blank save can never outweigh a non-empty launch environment
    (config.settings_of_document), so where the launch holds the consent a withdrawal stores the explicit
    revocation word — honest in the interface and revocable from it — and every reader takes that word for no
    consent. Removal and rotation differ: a write that empties the address in force takes the stored consent
    with it; rotating the key under a standing address keeps it."""
    from ouroboros.llm_ollama_guard import CONSENT_REVOKED, consent_token, endpoint_id
    from ouroboros.provider_connections import canonical_builtin_consent

    key = "OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED"
    granted = {key: GATEWAY}
    canonical_builtin_consent(granted, {key: GATEWAY})
    assert granted[key] == endpoint_id(GATEWAY), "a grant is stored as its endpoint's id"

    monkeypatch.setenv(key, GATEWAY)  # the launch environment holds the consent
    blank = {key: ""}
    canonical_builtin_consent(blank, {key: ""})
    assert blank[key] == CONSENT_REVOKED, "a blank save stores the explicit revocation instead"
    assert consent_token(blank[key]) == "", "and every reader takes it for no consent"
    again = {key: CONSENT_REVOKED}
    canonical_builtin_consent(again, {key: CONSENT_REVOKED})
    assert again[key] == CONSENT_REVOKED, "the revocation travels unchanged through a later write"

    monkeypatch.delenv(key)
    plain = {key: ""}
    canonical_builtin_consent(plain, {key: ""})
    assert plain[key] == "", "without the environment a withdrawal stays a plain blank"

    removed = {key: endpoint_id(GATEWAY), "OPENAI_COMPATIBLE_BASE_URL": ""}
    canonical_builtin_consent(removed, {"OPENAI_COMPATIBLE_BASE_URL": ""})
    assert removed[key] == "", "a write that empties the address takes the stored consent with it"

    rotated = {key: endpoint_id(GATEWAY), "OPENAI_COMPATIBLE_BASE_URL": GATEWAY, "OPENAI_COMPATIBLE_API_KEY": "new-key"}
    canonical_builtin_consent(rotated, {"OPENAI_COMPATIBLE_API_KEY": "new-key"})
    assert rotated[key] == endpoint_id(GATEWAY), "rotating the key under the standing address keeps the consent"
