"""Timeout projections distinguish clean unsent preparation from unknown effects."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.tool_result import TOOL_CODE_SPECS


@pytest.fixture
def registry(tmp_path, monkeypatch):
    mcp_client.reset_manager_for_tests()
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **kw: (True, ""))
    yield ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    mcp_client.reset_manager_for_tests()


def _configure(server, monkeypatch, *, timeout=1):
    from ouroboros import config
    manager = mcp_client.get_manager()
    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": timeout,
                "MCP_SERVERS": [{"id": "controlled", "enabled": True, **server}]}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    manager.reconfigure(settings)
    return manager


def _assert_unknown(text):
    assert "MCP_TOOL_TIMEOUT" in text
    assert "remote outcome is unknown" in text
    assert "side effects may already have happened" in text
    assert "remote cancellation is not confirmed" in text
    assert "server-specific status/read" in text
    assert "before retrying" in text


@pytest.mark.parametrize("timeout_phase", ["initialize", "after_effect"])
@pytest.mark.parametrize("projection", ["typed", "text"])
def test_timeout_disclosure_survives_registry_and_status_read(
    tmp_path, monkeypatch, registry, timeout_phase, projection,
):
    """Use real client wait/cancel and registry dispatch; only the peer is controlled."""
    calls = []
    effects = []
    closed_sessions, closed_transports = [], []
    first_session = True

    @asynccontextmanager
    async def transport(_cfg):
        try:
            yield None, None
        finally:
            closed_transports.append(True)

    class Session:
        def __init__(self, *_streams):
            nonlocal first_session
            self.initial_attempt = first_session
            first_session = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            closed_sessions.append(self.initial_attempt)

        async def initialize(self):
            if self.initial_attempt and timeout_phase == "initialize":
                await asyncio.Event().wait()

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            if name == "apply":
                effects.append(arguments["operation_id"])
                await asyncio.Event().wait()  # The response never arrives.
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(effects))], isError=False)

    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    manager = _configure({"url": "https://controlled.invalid/mcp"}, monkeypatch)

    async def list_tools(_cfg, _timeout):
        return [{"name": name, "input_schema": {}} for name in ("apply", "status")]

    manager._async_list_tools = list_tools
    assert manager.refresh_server("controlled")["ok"]
    args = {"operation_id": "op-667"}
    if projection == "typed":
        result = registry.execute_result("mcp_controlled__apply", args)
        assert (result.status, result.code) == ("timeout", "MCP_TIMEOUT")
        text = result.text
    else:
        text = registry.execute("mcp_controlled__apply", args)
    if timeout_phase == "initialize":
        assert "The tool call was not submitted" in text
        assert "session and transport have closed" in text
    else:
        _assert_unknown(text)
    if projection == "typed":
        assert result.meta.get("mcp_call_not_sent") is (timeout_phase == "initialize")
    assert "reconcile" in TOOL_CODE_SPECS["MCP_TIMEOUT"].recovery
    expected = ["op-667"] if timeout_phase == "after_effect" else []
    assert effects == expected
    assert len(calls) == len(expected)  # No implicit replay after timeout.
    assert closed_sessions == closed_transports == [True]  # Local close is not a remote cancel receipt.

    status = registry.execute_result("mcp_controlled__status", args)
    assert (status.status, status.code) == ("ok", "OK")
    assert status.text.endswith(json.dumps(expected))
    assert calls[-1] == ("status", args)
    assert effects == expected
    assert closed_sessions == [True, False] and closed_transports == [True, True]


def test_real_stdio_effect_before_timeout_is_reconciled_by_status_read(tmp_path, registry, monkeypatch):
    """A real SDK peer persists the effect but omits its reply, then serves status."""
    pytest.importorskip("mcp")
    effects = tmp_path / "effects.jsonl"
    calls = tmp_path / "calls.jsonl"
    script = tmp_path / "controlled_mcp_timeout.py"
    script.write_text('''import json, pathlib, sys
effects, calls = map(pathlib.Path, sys.argv[1:])
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "controlled", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                            for name in ("apply", "status")]}
    elif method == "tools/call":
        params = request["params"]
        with calls.open("a", encoding="utf-8") as out:
            out.write(json.dumps(params) + "\\n")
        if params["name"] == "apply":
            with effects.open("a", encoding="utf-8") as out:
                out.write(json.dumps(params["arguments"]) + "\\n")
            continue  # Effect committed; deliberately no response.
        rows = [json.loads(line) for line in effects.read_text().splitlines()] if effects.exists() else []
        result = {"content": [{"type": "text", "text": json.dumps(rows)}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''', encoding="utf-8")
    manager = _configure({"transport": "stdio", "command": sys.executable,
                          "args": [str(script), str(effects), str(calls)]}, monkeypatch, timeout=3)
    assert manager.refresh_server("controlled")["ok"]
    args = {"operation_id": "op-667-real"}
    timeout = registry.execute_result("mcp_controlled__apply", args)
    assert (timeout.status, timeout.code) == ("timeout", "MCP_TIMEOUT")
    _assert_unknown(timeout.text)
    assert [json.loads(line) for line in effects.read_text().splitlines()] == [args]
    assert [json.loads(line)["name"] for line in calls.read_text().splitlines()] == ["apply"]

    status = registry.execute_result("mcp_controlled__status", args)
    assert (status.status, status.code) == ("ok", "OK")
    assert status.text.endswith(json.dumps([args]))
    assert [json.loads(line)["name"] for line in calls.read_text().splitlines()] == ["apply", "status"]
    assert [json.loads(line) for line in effects.read_text().splitlines()] == [args]


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
