"""Immutable metadata follows descriptor generation, never PID as store identity."""
import json
from types import SimpleNamespace

import httpx

from ouroboros.gateways import claudexor as wire, claudexor_metadata as metadata


def test_catalog_and_model_handshake_cache_follow_descriptor(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon as daemon

    home = tmp_path / "engine"
    (home / "daemon").mkdir(parents=True)
    token = home / "daemon/token"
    token.write_text("test-token")
    descriptor = home / "daemon/control-api.json"
    state, calls = {"generation": 1, "store": "store-a"}, []

    def publish():
        descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": 9999, "tokenPath": str(token),
                                          "pid": state["generation"]}))

    def answer(request):
        calls.append(request.url.path)
        if request.url.path == "/v2/handshake":
            # The engine names its store only to a client that opts in (older strict clients keep the old shape).
            opted_in = request.headers.get("X-Claudexor-Store-Identity") == "1"
            return httpx.Response(200, json={"compatible": True, "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                "engine": {"version": wire.CLAUDEXOR_MIN_VERSION, "sha": str(state["generation"])},
                **({"storeId": state["store"]} if opted_in else {}), "servingMode": "normal"})
        return httpx.Response(200, json={"operations": [{"id": str(state["generation"])}]})

    client = httpx.Client
    monkeypatch.setattr(wire.httpx, "Client", lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(answer)))
    monkeypatch.setattr(daemon, "attach_home", lambda: None)
    monkeypatch.setattr(daemon, "owned_config_dir", lambda: home)
    reconciled = []
    monkeypatch.setattr(daemon, "get_owned_daemon", lambda: SimpleNamespace(
        reconcile_rotation=lambda gateway: reconciled.append(gateway)))

    def ensure():
        gateway = wire.ClaudexorGateway(wire.discover_daemon_at(home))
        gateway.handshake()
        return gateway

    publish()
    for generation, store in ((1, "store-a"), (2, "store-a"), (3, "store-b")):
        state.update(generation=generation, store=store)
        if generation > 1:
            publish()
        for _round in range(3):
            with metadata.model_gateway(ensure) as gateway:
                assert gateway.operations() == [{"id": str(generation)}]
                assert gateway.store_identity == store
                # One reader cannot mutate the cached catalog for another.
                gateway.operations()[0]["id"] = "changed"
        assert calls.count("/v2/handshake") == generation
        assert calls.count("/v2/operations") == generation
    descriptor.unlink()
    assert gateway.store_identity == "", "a stale endpoint cannot certify a replacement's store"
    assert len(reconciled) == 6, "cached calls retain the owner's settings reconcile"
