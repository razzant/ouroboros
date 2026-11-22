"""The MCP task-session harness: ``wire`` gives the process manager fixture servers and lists them; ``call`` runs
one of their tools through the registry's ordinary execution (``tests._invocation_harness.run``)."""

from __future__ import annotations

from ouroboros import mcp_client
from tests._invocation_harness import run


def wire(monkeypatch, registry, *servers, timeout=20):
    from ouroboros import config

    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": max(20, timeout), "MCP_SERVERS": list(servers)}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    manager = mcp_client.get_manager()
    manager.reconfigure(settings)
    for server in servers:
        assert manager.refresh_server(server["id"])["ok"]
    if settings["MCP_TOOL_TIMEOUT_SEC"] != timeout:
        settings["MCP_TOOL_TIMEOUT_SEC"] = timeout
        manager.reconfigure(settings)
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    return manager, settings


def call(registry, tool, args=None, *, server_id="fx", timeout=40):
    return run(registry, name=f"mcp_{server_id}__{tool}", args=args or {"n": 1}, timeout=timeout)["tool_result"]
