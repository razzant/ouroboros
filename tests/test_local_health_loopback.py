"""Owned-loopback health reaches HTTP without ambient proxy discovery."""
from __future__ import annotations

import json
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from ouroboros.local_model import LocalModelManager



@pytest.fixture
def health_server(monkeypatch):
    seen = []
    response = {"status": 200, "body": {"data": [{"id": "fixture", "context_window": 8192}]}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            body = json.dumps(response["body"]).encode("utf-8")
            self.send_response(response["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    def no_reverse_dns(*_args):
        raise AssertionError("loopback fixture must not resolve hostnames")

    monkeypatch.setattr(socket, "getfqdn", no_reverse_dns)

    class LoopbackHTTPServer(ThreadingHTTPServer):
        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]

    server = LoopbackHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="health-http-fixture", daemon=True)
    thread.start()
    try:
        yield server.server_port, seen, response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize("worker", [False, True], ids=["server-process", "worker-process"])
def test_local_health_bypasses_proxy_discovery_and_reaches_real_http(health_server, monkeypatch, worker):
    port, seen, _response = health_server
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1" if worker else "")
    proxy_reads = []

    def ambient_proxy_reader(*args, **kwargs):
        proxy_reads.append(args[0])
        raise RuntimeError("proxy discovery must not run for owned loopback")

    monkeypatch.setattr(requests.sessions, "get_environ_proxies", ambient_proxy_reader)
    # Demonstrate that the trap really covers Requests' default environment path.
    with requests.Session() as generic:
        with pytest.raises(RuntimeError, match="proxy discovery must not run"):
            generic.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
    assert len(proxy_reads) == 1 and seen == []
    proxy_reads.clear()

    manager = LocalModelManager()
    manager._port = port
    assert manager.health_check() == {"ok": True, "model_name": "fixture", "context_length": 8192}
    assert seen == ["/v1/models"] and proxy_reads == []


def test_local_health_keeps_http_error_and_missing_endpoint_failures(health_server):
    port, seen, response = health_server
    manager = LocalModelManager()
    manager._port = port
    response["status"] = 503
    with pytest.raises(requests.HTTPError):
        manager.health_check()
    assert seen == ["/v1/models"]
    # A reserved but non-listening loopback socket cannot silently become healthy.
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        manager._port = unavailable.getsockname()[1]
        with pytest.raises(requests.ConnectionError):
            manager.health_check()
