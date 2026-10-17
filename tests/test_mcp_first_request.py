"""A task's schema read never waits for an MCP listing (the first-request path).

A fresh process (a pooled worker) configures its manager from the saved settings on its first
schema read. That read — a live call's initial view or a text turn's initial schemas alike —
starts the background listing and answers with the rows listed so far, a server not listed yet
being an omission that says so; once listed, ``enable_tools`` and the next ``schemas()`` carry
its tools. A refresh someone asks for on purpose (Settings' Refresh, the settings-save refresh)
still lists every server as before. Counters, not clocks: the fake listing waits for the test.
"""
from __future__ import annotations

import threading

import pytest

from ouroboros import mcp_client
from ouroboros.live_call_route import initial_call_schemas
from ouroboros.tool_policy import initial_tool_schemas
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.tool_discovery import enable_tools

PING = {"name": "ping", "description": "Ping", "input_schema": {"type": "object", "properties": {}}}


def _settings() -> dict:
    return {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 60, "MCP_SERVERS": [
        {"id": "svc", "name": "Svc", "enabled": True, "transport": "streamable_http", "url": "https://e.example/mcp",
         "auth_header": "Authorization", "auth_token": "", "allowed_tools": []}]}


class _Listing:
    """A fake transport: every listing is counted and waits until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.finished: list = []

    async def list_tools(self, cfg, timeout):
        self.release.wait(10)  # a hang guard only; the counters decide
        self.finished.append(cfg.id)
        return [dict(PING)]


def _refresh_threads() -> list:
    return [thread for thread in threading.enumerate() if thread.name.startswith("mcp-refresh-")]


def _join_listings() -> None:
    for thread in _refresh_threads():
        thread.join(15)


@pytest.fixture
def cold_worker(tmp_path, monkeypatch):
    """A fresh process: its manager is configured from a saved settings file by the first schema read.

    The registry (and every tool module it imports) exists before ``load_settings`` is replaced:
    a module importing it by name must keep the real one for the tests after this one.
    """
    import ouroboros.config as config_mod

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config_mod, "load_settings", _settings)
    mcp_client.reset_manager_for_tests()
    listing = _Listing()
    mcp_client.get_manager()._async_list_tools = listing.list_tools
    caller, on_caller = threading.current_thread(), []
    original = mcp_client.MCPManager.refresh_server

    def counted(self, server_id, **kwargs):
        on_caller.append(threading.current_thread() is caller)
        return original(self, server_id, **kwargs)

    monkeypatch.setattr(mcp_client.MCPManager, "refresh_server", counted)
    yield registry, listing, on_caller
    listing.release.set()
    _join_listings()
    mcp_client.reset_manager_for_tests()


def _mcp_names(schemas) -> list:
    return [schema["function"]["name"] for schema in schemas if schema["function"]["name"].startswith("mcp_")]


def test_first_schema_reads_start_one_background_listing_and_wait_for_none(cold_worker):
    registry, listing, on_caller = cold_worker
    call_view = initial_call_schemas(registry)
    text_turn = initial_tool_schemas(registry, context_mode="max")

    assert listing.finished == [] and True not in on_caller  # neither read listed anything itself
    assert len(_refresh_threads()) == 1  # one listing started; the second read found it running
    assert _mcp_names(call_view) == [] and _mcp_names(text_turn) == []
    assert {"surface": "mcp", "reason": "server_no_tools",
            "servers": [{"id": "svc", "last_error": "not listed yet"}]} in registry.capability_omissions()


def test_tools_listed_in_the_background_reach_the_next_read_and_enable_tools(cold_worker):
    registry, listing, on_caller = cold_worker
    resident = list(initial_call_schemas(registry))
    assert listing.finished == []
    listing.release.set()
    _join_listings()

    assert listing.finished == ["svc"] and on_caller == [False]
    assert "registered late" in enable_tools(registry, registry._ctx, "mcp_svc__ping", resident)
    assert _mcp_names(resident) == ["mcp_svc__ping"]
    assert _mcp_names(registry.schemas()) == ["mcp_svc__ping"]
    assert not any(item.get("reason") == "server_no_tools" for item in registry.capability_omissions())
    assert _refresh_threads() == [] and listing.finished == ["svc"]  # a listed catalog starts no new listing


def test_a_refresh_asked_for_on_purpose_still_lists_every_server(cold_worker):
    registry, listing, on_caller = cold_worker
    registry.schemas()
    assert listing.finished == [] and len(_refresh_threads()) == 1
    listing.release.set()
    _join_listings()
    manager = mcp_client.get_manager()

    outcome = manager.refresh_server("svc")  # Settings' Refresh of one server (POST /api/mcp/refresh)
    assert outcome["ok"] and outcome["tool_count"] == 1 and listing.finished == ["svc", "svc"]
    assert manager.refresh_all()["refreshed"]["svc"]["ok"] and len(listing.finished) == 3
    manager.refresh_all_background(reason="settings")  # the settings-save refresh re-lists a listed server
    _join_listings()
    assert len(listing.finished) == 4 and on_caller == [False, True, True, False]
