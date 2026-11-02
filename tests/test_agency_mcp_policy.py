"""Cyber MCP endpoint policy preserves real transport and error facts."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ouroboros import config, mcp_client


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/", "http://metadata.google.internal/mcp",
    "http://metadata/mcp", "http://100.100.100.200/mcp", "http://[fe80::1]/mcp",
    "http://[::ffff:169.254.169.254]/mcp",
])
def test_cyber_metadata_is_admitted_without_changing_ordinary_policy(url, monkeypatch):
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    with pytest.raises(ValueError):
        mcp_client._validate_url(url)
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    assert mcp_client._validate_url(url) == url


@pytest.mark.parametrize("url", ["", "http:///mcp", "file:///tmp/mcp", "http://[invalid]/mcp", "http://host:bad/mcp"])
def test_cyber_malformed_or_unsupported_urls_remain_errors(url, monkeypatch):
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    assert mcp_client.normalize_server_config({"id": "demo", "url": url}) is None


def test_reconfigure_reconsiders_admission_when_effective_mode_changes(monkeypatch):
    settings = {"MCP_ENABLED": True, "MCP_SERVERS": [
        {"id": "metadata", "enabled": True, "url": "http://metadata/mcp"}]}
    manager = mcp_client.MCPManager()
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    assert manager.reconfigure(settings) and not manager.server_ids()
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    assert manager.reconfigure(settings) and manager.server_ids() == ["metadata"]
    assert not manager.reconfigure(settings)


def test_cyber_url_credentials_use_existing_diagnostic_masking(monkeypatch):
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    url = "https://agency-user:synthetic%3Apassword@example.com/mcp"
    cfg = mcp_client.normalize_server_config({"id": "auth", "url": url})
    assert cfg and cfg.url == url
    manager = mcp_client.MCPManager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [{"id": "auth", "url": url}]})
    assert manager.status_payload()["servers"][0]["url"] == "https://example.com/mcp"
    diagnostic = mcp_client._redact_error_text("agency-user synthetic:password " + url, cfg)
    assert "agency-user" not in diagnostic and "password" not in diagnostic
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    assert mcp_client.normalize_server_config({"id": "auth", "url": url}) is None


@pytest.mark.serial
@pytest.mark.parametrize("mode", ["pro", "cyber_pro"])
def test_live_http_mcp_discovery_call_and_remote_failure(tmp_path, monkeypatch, mode):
    pytest.importorskip("mcp")
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", mode)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = request["method"]
            if "id" not in request:
                self.send_response(202)
                self.end_headers()
                return
            if method == "initialize":
                result = {"protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}}, "serverInfo": {"name": "agency", "version": "1"}}
            elif method == "tools/list":
                result = {"tools": [{"name": "report", "inputSchema": {"type": "object"}}]}
            else:
                assert method == "tools/call"
                arguments = request["params"]["arguments"]
                calls.append(arguments)
                result = {"content": [{"type": "text", "text": json.dumps(arguments)}],
                          "isError": arguments.get("fail", False)}
            body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manager = mcp_client.MCPManager()
        origin = f"http://127.0.0.1:{server.server_address[1]}"
        url = origin + "/mcp" if mode == "pro" else origin.replace("http://", "http://agency-user:synthetic-password@") + "/mcp"
        manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 5, "MCP_SERVERS": [
            {"id": "agency", "enabled": True, "url": url, "allowed_tools": ["different_tool"]}]})
        assert manager.refresh_server("agency")["ok"]
        assert "synthetic-password" not in json.dumps(manager.status_payload())
        if mode == "pro":
            assert manager.list_tools_for_registry() == []
            assert manager._call_tool_result("mcp_agency__report", {}).status == "blocked"
            assert calls == []
            monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
        assert [item["name"] for item in manager.list_tools_for_registry()] == ["mcp_agency__report"]
        result = manager._call_tool_result("mcp_agency__report", {"content": "Policy words are report data"})
        assert result.status == "ok" and "Policy words are report data" in result.text
        failed = manager._call_tool_result("mcp_agency__report", {"fail": True})
        assert failed.status == "error" and failed.meta["mcp_is_error"] is True
        assert calls == [{"content": "Policy words are report data"}, {"fail": True}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
