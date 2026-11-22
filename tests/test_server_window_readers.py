"""The window a local server states about itself, read from real and documented responses (P1, 15.11.2026).

llama.cpp responses are the stand's own (b12421, ``fixtures/server_windows/README.md``); vLLM, LM Studio and
Ollama shapes are reconstructed from their sources and docs. Every failure reads as unknown (0), never as a window.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros import server_window as sw

ROOT = pathlib.Path(__file__).parent / "fixtures" / "server_windows"


def _json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _item(name, model):
    return next(item for item in _json(name)["data"] if item["id"] == model)


def test_models_listings_name_the_servers_own_window_and_never_the_training_length():
    llama = _item("llama-b12421-v1-models.json", "qwen-stand")
    assert llama["meta"]["n_ctx_train"] == 262_144  # present, and never read
    assert sw.models_item_window(llama) == (80_000, sw.SOURCE_SERVER_METADATA)
    assert sw.models_item_window(_item("vllm-0.33-v1-models.json", "Qwen/Qwen3.5-9B")) == (80_000, sw.SOURCE_SERVER_METADATA)
    for name, model in (("lmstudio-0.4-v1-models.json", "qwen3.5-9b"), ("ollama-0.40-v1-models.json", "qwen3.5:9b")):
        assert sw.models_item_window(_item(name, model)) == (0, "")
    assert sw.models_item_window({"id": "x", "context_length": 200_000}) == (200_000, sw.SOURCE_PROVIDER_METADATA)


def test_root_readers_take_the_slot_window_and_the_loaded_instance_only():
    wanted = sw.wanted_ids("openai-compatible::qwen-stand")
    assert sw._llama_props(_json("llama-b12421-props.json"), wanted) == 80_000
    assert sw._llama_props(_json("llama-b12421-np1-80k-props.json"), wanted) == 80_128  # -c 80000 padded to 256
    assert sw._llama_props(_json("llama-b12421-np2-80k-props.json"), wanted) == 40_192  # --parallel 2 halves it
    assert sw._llama_props(_json("llama-b12421-props.json"), sw.wanted_ids("other-model")) == 0  # another alias
    lmstudio = _json("lmstudio-0.4-api-v0-models.json")
    assert sw._lmstudio_v0(lmstudio, sw.wanted_ids("qwen3.5-9b")) == 80_000  # loaded, not max_context_length
    assert sw._lmstudio_v0(lmstudio, sw.wanted_ids("gemma-4-12b")) == 0      # not loaded: unknown
    assert sw._ollama_ps(_json("ollama-0.40-api-ps.json"), sw.wanted_ids("openai-compatible::qwen3.5:9b")) == 32_768
    assert sw._ollama_ps(_json("ollama-0.40-api-ps-empty.json"), sw.wanted_ids("qwen3.5:9b")) == 0  # before a load


def test_a_program_is_named_only_by_its_own_document():
    """``PROGRAM_DOCUMENTS``: the stand's llama.cpp ``/props`` and LM Studio's v0 listing name their programs; no
    other recorded document, generic listing, version number or error body is taken for either."""
    shapes = {name: (path, shaped) for name, path, shaped in sw.PROGRAM_DOCUMENTS}
    assert {name: path for name, (path, _shaped) in shapes.items()} == {"llama.cpp": "/props", "LM Studio": "/api/v0/models"}
    own = {"llama-b12421-props.json": "llama.cpp", "llama-b12421-np1-80k-props.json": "llama.cpp",
           "llama-b12421-np2-80k-props.json": "llama.cpp", "lmstudio-0.4-api-v0-models.json": "LM Studio"}
    recorded = sorted(path.name for path in ROOT.glob("*.json"))
    assert set(own) < set(recorded)
    for name in recorded:
        named = {program for program, (_path, shaped) in shapes.items() if shaped(_json(name))}
        assert named == ({own[name]} if name in own else set()), name
    props = _json("llama-b12421-props.json")
    for nothing in ({}, [], None, {"data": []}, {"version": "0.40.16"}, {"total_slots": 1},
                    {**props, "total_slots": 0}, {**props, "default_generation_settings": {}},
                    {"error": "Unexpected endpoint or method. (GET /api/v0/models)"},
                    {"data": [{"id": "qwen", "state": "loaded"}]}, {"data": [{"id": "qwen", "max_context_length": 4096}]}):
        assert not any(shaped(nothing) for _path, shaped in shapes.values()), nothing


def _getter(routes, asked):
    def get(url, **kwargs):
        asked.append(url)
        if kwargs.get("timeout", 99) > sw.READ_TIMEOUT_SEC:
            raise AssertionError("readers keep a short timeout")
        status, payload = routes.get(url, (404, {}))
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(status_code=status, json=lambda: payload)
    return get


@pytest.mark.parametrize("routes,expected", [
    ({"http://127.0.0.1:8080/props": (200, "llama-b12421-props.json")}, (80_000, "llama.cpp /props")),
    ({"http://127.0.0.1:8080/api/v0/models": (200, "lmstudio-0.4-api-v0-models.json")}, (0, "")),  # other id
    ({"http://127.0.0.1:8080/api/ps": (200, "ollama-0.40-api-ps-empty.json")}, (0, "")),
    ({"http://127.0.0.1:8080/props": (200, TimeoutError("slow"))}, (0, "")),
    ({"http://127.0.0.1:8080/props": (500, {})}, (0, "")),
])
def test_root_reading_fails_closed(routes, expected):
    asked = []
    routes = {url: (status, _json(body) if isinstance(body, str) else body) for url, (status, body) in routes.items()}
    assert sw.read_root_window("http://127.0.0.1:8080/v1", "qwen-stand", get=_getter(routes, asked)) == expected
    assert all(url.startswith("http://127.0.0.1:8080/") and "/v1/" not in url for url in asked)


def test_root_readers_are_only_for_this_hosts_network():
    for url in ("http://127.0.0.1:18080/v1", "http://localhost:1234/v1", "http://192.168.1.20:11434/v1",
                "http://studio.local:1234/v1"):
        assert sw.private_host(url), url
    for url in ("https://api.example.com/v1", "https://8.8.8.8/v1", ""):
        assert not sw.private_host(url), url


class _Response:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _metadata(monkeypatch, tmp_path, routes, *, base_url="http://127.0.0.1:18080/v1", model="openai-compatible::qwen-stand"):
    import httpx

    asked = []

    def get(url, **kwargs):
        asked.append(url)
        status, payload = routes.get(url, (404, {}))
        if isinstance(payload, Exception):
            raise payload
        return _Response(status, payload)

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr("ouroboros.vision_routing.record_catalog_image_input", lambda *a, **k: None)
    reading = ce._openai_compatible_metadata_reading(model, base_url, True, "k", provider="openai-compatible",
                                                     evidence_root=tmp_path)
    return reading, asked


def test_the_stands_first_contact_read_now_knows_the_window(monkeypatch, tmp_path):
    """The stand's first request was this very GET (evidence2/00001); before P1 its meta.n_ctx read as unknown."""
    reading, asked = _metadata(monkeypatch, tmp_path, {
        "http://127.0.0.1:18080/v1/models": (200, _json("llama-b12421-v1-models.json"))})
    assert reading == (80_000, ce.SOURCE_SERVER_METADATA) and asked == ["http://127.0.0.1:18080/v1/models"]


def test_recognized_vllm_does_not_probe_other_engines(monkeypatch, tmp_path):
    reading, asked = _metadata(monkeypatch, tmp_path, {
        "http://127.0.0.1:18080/v1/models": (200, _json("vllm-0.33-v1-models.json"))},
        model="openai-compatible::Qwen/Qwen3.5-9B")
    assert reading == (80_000, ce.SOURCE_SERVER_METADATA)
    assert asked == ["http://127.0.0.1:18080/v1/models"]


def test_an_older_llama_listing_falls_back_to_props_and_a_cloud_gateway_is_not_asked(monkeypatch, tmp_path):
    bare = {"object": "list", "data": [{"id": "qwen-stand", "object": "model", "owned_by": "llamacpp"}]}
    reading, asked = _metadata(monkeypatch, tmp_path, {
        "http://127.0.0.1:18080/v1/models": (200, bare),
        "http://127.0.0.1:18080/props": (200, _json("llama-b12421-props.json"))})
    assert reading == (80_000, ce.SOURCE_SERVER_METADATA) and asked[-1] == "http://127.0.0.1:18080/props"
    reading, asked = _metadata(monkeypatch, tmp_path, {"https://api.example.com/v1/models": (200, bare)},
                               base_url="https://api.example.com/v1")
    assert reading == (0, "") and asked == ["https://api.example.com/v1/models"]
    reading, _asked = _metadata(monkeypatch, tmp_path, {"http://127.0.0.1:18080/v1/models": (200, ConnectionError())})
    assert reading == (0, "") and _asked == ["http://127.0.0.1:18080/v1/models"]


@pytest.mark.parametrize("error_kind", ["ConnectError", "ConnectTimeout"])
def test_root_chain_stops_after_the_first_connection_failure(error_kind):
    import httpx

    asked = []
    routes = {"http://127.0.0.1:8080/props": (200, getattr(httpx, error_kind)("unreachable"))}
    assert sw.read_root_window("http://127.0.0.1:8080/v1", "qwen-stand", get=_getter(routes, asked)) == (0, "")
    assert asked == ["http://127.0.0.1:8080/props"]


def test_a_servers_own_window_holds_its_reply_inside():
    """Owner-root note 15.11: n_ctx / max_model_len are one buffer for input and reply, as a refusal's total is."""
    from ouroboros.context_window import WindowLimit, WindowRecord

    total = WindowLimit(80_000, "total", ce.SOURCE_SERVER_METADATA)
    record = WindowRecord("fp", total=total, reserve=8_192, floor=4_096)
    assert record.admitted == 80_000 - 8_192
    catalog = WindowRecord("fp", total=WindowLimit(80_000, "total", ce.SOURCE_PROVIDER_METADATA), reserve=8_192, floor=4_096)
    assert catalog.admitted == 80_000 - 4_096  # a gateway catalog keeps its older reading


def test_server_reader_has_no_import_back_to_evidence_owners():
    import ast
    owners = {'capability_evidence', 'context_window'}
    tree = ast.parse(pathlib.Path(sw.__file__).read_text())
    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            dependencies.add(node.module or '')
            if node.module == 'ouroboros':
                dependencies.update('ouroboros.' + alias.name for alias in node.names)
    assert not dependencies & {'ouroboros.' + owner for owner in owners}


@pytest.mark.parametrize("prior_age_hours,known", [(0, True), (25, False)])
def test_a_failed_start_read_keeps_a_fresh_window_and_reads_once(monkeypatch, tmp_path, prior_age_hours, known):
    """A start forces its server read; a busy server then hides no confirmed read still inside its TTL."""
    from datetime import datetime, timedelta, timezone

    import httpx

    from ouroboros.context_window import resolve_window
    from ouroboros.response_limits import probe_context_route

    route = dict(provider="openai-compatible", model="openai-compatible::fixture", base_url="http://127.0.0.1:18080/v1",
                 use_local=False, options=None, _resolved_target=None)
    monkeypatch.setattr("ouroboros.vision_routing.record_catalog_image_input", lambda *a, **k: None)
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _Response(200, {"data": [{"id": "fixture", "meta": {"n_ctx": 80000}}]}))
    assert resolve_window(probe_context_route(tmp_path, route, task={"route_start_read": True}), 8192,
                          root=tmp_path).total.tokens == 80000
    data = ce._read(tmp_path)
    recorded_at = (datetime.now(timezone.utc) - timedelta(hours=prior_age_hours)).isoformat()
    for row in data["probes"].values():
        row["ts"] = recorded_at
    ce._save(tmp_path, data)
    asked = []

    def busy(url, **kw):
        asked.append(url)
        raise httpx.ReadTimeout("busy", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", busy)
    failed_at = datetime.now(timezone.utc)
    evidence = probe_context_route(tmp_path, route, task={"route_start_read": True})
    assert evidence.stale is not known and evidence.read == "failed"  # the record says its read failed
    record = resolve_window(evidence, 8192, root=tmp_path)
    assert (record.total.tokens if record.total else None) == (80000 if known else None)
    assert len(asked) == len(set(asked)) and asked  # the one forced read, no repeat
    if known:  # still sizing, as the stored statement of its own date, never as this start's live read
        assert "this read stated no window" in evidence.detail
        kind, observed_at = ce.window_evidence(evidence, record)
        assert kind == "recorded" and datetime.fromisoformat(observed_at) < failed_at

@pytest.mark.parametrize("host,private", [
    ("http://ollama:11434/v1", True), ("http://llama:8080/v1", True), ("http://ollama.lan/v1", True),
    ("http://gpu.internal:11434/v1", True), ("http://box.home.arpa/v1", True), ("http://nas.localdomain/v1", True),
    ("http://ollama.local/v1", True), ("http://[::1]:11434/v1", True), ("http://192.168.1.20:11434/v1", True),
    ("https://api.example.com/v1", False), ("https://ollama.com/v1", False), ("https://8.8.8.8/v1", False),
    ("https://lan.example.com/v1", False), ("https://internal.example.org/v1", False), ("", False),
])
def test_private_host_covers_single_label_and_private_suffix_names(host, private):
    from ouroboros.server_window import private_host

    assert private_host(host) is private


@pytest.mark.parametrize("host,address", [
    ("10.1", "10.0.0.1"), ("0xa.0.0.1", "10.0.0.1"), ("0300.0250.0.1", "192.168.0.1"), ("167772161", "10.0.0.1"),
    ("::ffff:10.0.0.1", "10.0.0.1"), ("8.8.8", "8.8.0.8"), ("134744072", "8.8.8.8"), ("::ffff:8.8.8.8", "8.8.8.8"),
    ("fe80::1", "fe80::1"), ("ollama", None), ("gateway.lan", None), ("", None),
])
def test_a_numeric_host_is_the_address_it_reaches_in_any_spelling(host, address):
    """One reading of a numeric host for the private-server question and for the address a hold names: the
    spellings the OS connects by are that address, a name is no address, and nothing is resolved."""
    import ipaddress

    from ouroboros.llm_ollama_guard import address_scope
    from ouroboros.server_window import host_address, private_host

    url = f"http://[{host}]:8080/v1" if ":" in host else f"http://{host}:8080/v1"
    if address is None:
        assert host_address(host) is None
        return
    reached = ipaddress.ip_address(address)
    assert host_address(host) == reached
    assert private_host(url) is (reached.is_loopback or reached.is_private or reached.is_link_local)
    quad = f"[{reached.compressed}]" if reached.version == 6 else str(reached)
    assert address_scope(url) == f"http://{quad}:8080"


def test_a_malformed_address_is_no_private_host():
    assert not sw.private_host("http://[::1/v1") and not sw.private_host("not a url")


def test_local_ready_carries_the_targets_custom_headers():
    import httpx
    seen = []
    def handler(request):
        seen.append(dict(request.headers))
        if request.headers.get('X-Local-Route') != 'required':
            return httpx.Response(403)
        bodies = {'/api/version': {'version': '0.40.16'}, '/api/show': {
            'model_info': {'general.architecture': 'qwen35'}, 'capabilities': ['completion']}}
        return httpx.Response(200, json=bodies.get(request.url.path, {}))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert sw.ollama_server_ready({'provider': 'openai-compatible', 'resolved_model': 'm',
            'base_url': 'http://localhost:8080/v1', 'api_key': 'test-only',
            'default_headers': {'X-Local-Route': 'required'}}, client)
    assert seen and all(row['authorization'] == 'Bearer test-only' for row in seen)


@pytest.mark.parametrize("architecture,lengths,maximum", [
    ("text", {"text": 131072, "vision": 262144}, 131072),
    ("text", {"vision": 262144}, 0),
    (None, {"text": 131072}, 131072),
    (None, {"text": 131072, "other": 131072}, 131072),
    (None, {"text": 131072, "vision": 262144}, 0),
])
def test_ollama_model_maximum_uses_the_declared_architecture(architecture, lengths, maximum):
    info = {f"{name}.context_length": size for name, size in lengths.items()}
    if architecture:
        info["general.architecture"] = architecture
    assert sw.ollama_model_maximum({"model_info": info}) == maximum
