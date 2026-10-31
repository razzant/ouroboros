"""#1328: one MCP server identity shared by every consumer.

``canonical_server_id`` used to be one ``_slugify`` pass. A long raw name whose
cut stem ends in ``_`` then stored ``x__<digest>`` while every advertised tool
name carried ``x_<digest>`` (``make_tool_name`` canonicalized again), so the
exact advertised name missed the manager map, Refresh/Test and discovery. The
id is now the fixed point. These tests drive the real registry, Settings and
Presence consumers over a fake transport; they also pin that two raw entries
claiming one id are refused rather than silently retargeted, and that a masked
secret never moves between servers.
"""

from __future__ import annotations

import copy
import itertools
import string
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
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import ToolResult
from tests.test_settings_secret_mask import settings_client  # noqa: F401

# The four recorded shapes (research probes): empty id + long name, a hand-written
# long id, and two ids already saved through Settings.
HISTORICAL_SHAPES = (
    ({"name": "My Company Internal Tools"}, "my_company_b762e68c1ef1"),
    ({"id": "github mcp server long name here"}, "github_mcp_ee30147733a4"),
    ({"id": "filesystem"}, "filesystem"),
    ({"id": "acme_corp_i_9ec0a0daa606"}, "acme_corp_i_9ec0a0daa606"),
)


def _single_pass(value: str) -> str:
    """The pre-#1328 canonicalization, kept here only to reproduce the defect."""
    return mcp_client._slugify(value, max_len=mcp_client._MAX_SERVER_SLUG)


def _server(**fields) -> dict:
    return {"enabled": True, "transport": "streamable_http", "url": "https://e.example/mcp", **fields}


class _Transport:
    def __init__(self, tools=("list_files",)):
        self.tools = tools
        self.list_calls: list = []
        self.call_calls: list = []

    async def list_tools(self, cfg, timeout):
        self.list_calls.append(cfg.id)
        return [{"name": name, "description": "fixture", "input_schema": {"type": "object"}} for name in self.tools]

    async def call_tool(self, cfg, name, arguments, timeout):
        self.call_calls.append((cfg.id, name, dict(arguments)))
        return ToolResult(status="ok", code="OK", text=f"echo({cfg.id}/{name})")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    mcp_client.reset_manager_for_tests()
    transport = _Transport()
    manager = mcp_client.get_manager()
    manager._async_list_tools, manager._async_call_tool = transport.list_tools, transport.call_tool
    safety: list = []

    def check_safety(name, args, **_kwargs):
        safety.append(name)
        return True, ""

    monkeypatch.setattr("ouroboros.safety.check_safety", check_safety)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, active_context_mode="max"))

    def configure(*servers):
        mcp_client.reconfigure_from_settings(
            {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 60, "MCP_SERVERS": list(servers)})
        for server_id in manager.server_ids():
            assert manager.refresh_server(server_id)["ok"]

    yield SimpleNamespace(registry=registry, transport=transport, safety=safety, manager=manager,
                          configure=configure)
    mcp_client.reset_manager_for_tests()


def test_canonical_server_id_is_a_fixed_point_that_keeps_every_wire_name():
    alphabet = string.ascii_lowercase
    for length, cut in itertools.product(range(1, 81), range(0, 26)):
        raw = "".join(alphabet[i % 26] for i in range(length))
        raw = raw[:cut] + "_" + raw[cut + 1:] if cut < length else raw
        for value in (raw, raw.upper(), raw.replace("_", " "), f"  {raw}!  "):
            server_id = mcp_client.canonical_server_id(value)
            if not server_id:  # "_" alone names no server
                continue
            assert mcp_client.canonical_server_id(server_id) == server_id
            assert "__" not in server_id and len(server_id) <= 24
            wire = mcp_client.make_tool_name(server_id, "list_files")
            assert mcp_client.parse_tool_name(wire)["server_slug"] == server_id
            # The advertised name is byte-identical to the one the old stored id produced.
            assert wire == mcp_client.make_tool_name(_single_pass(value), "list_files")


def test_the_pre_fix_single_pass_id_did_not_round_trip():
    stored = _single_pass("My Company Internal Tools")
    assert stored == "my_company__b762e68c1ef1"
    assert mcp_client.parse_tool_name(mcp_client.make_tool_name(stored, "t"))["server_slug"] != stored
    assert mcp_client.canonical_server_id("My Company Internal Tools") == "my_company_b762e68c1ef1"


@pytest.mark.parametrize(("fields", "server_id"), HISTORICAL_SHAPES)
def test_each_historical_shape_calls_exactly_and_misses_cleanly(world, fields, server_id):
    world.configure(_server(**fields))
    assert world.manager.server_ids() == [server_id]
    wire = f"mcp_{server_id}__list_files"
    assert world.registry.get_schema_by_name(wire) is not None
    assert f"echo({server_id}/list_files)" in world.registry.execute(wire, {"path": "."})
    assert world.transport.call_calls == [(server_id, "list_files", {"path": "."})]
    assert world.safety == [wire]
    listed = list(world.transport.list_calls)

    miss = world.registry.execute(f"mcp_{server_id}__missing_tool", {})
    assert "MCP_TOOL_NOT_FOUND" in miss
    assert len(world.transport.call_calls) == 1 and world.safety == [wire]
    assert world.transport.list_calls == listed  # a miss refreshes nothing
    assert world.manager.refresh_server(server_id)["ok"]


def test_discovery_and_empty_server_rows_carry_the_advertised_namespace(world):
    world.transport.tools = ()
    world.configure(_server(name="My Company Internal Tools"))
    empty = world.manager.enabled_servers_without_tools()
    assert [row["id"] for row in empty] == ["my_company_b762e68c1ef1"]
    assert mcp_client.parse_tool_name("mcp_my_company_b762e68c1ef1__x")["server_slug"] == empty[0]["id"]


def test_a_config_error_row_uses_the_loader_identity():
    errors: list = []
    assert mcp_client.parse_servers([{"slug": "Broken Server Slug", "url": "ftp://x"}], errors=errors) == []
    assert errors[0]["id"] == "broken_server_slug" and errors[0]["code"] == "MCP_CONFIG_ERROR"


def test_an_old_raw_name_and_its_new_explicit_slug_are_both_refused(world):
    # Before #1328 these were two distinct stored ids that advertised ONE wire prefix.
    old_raw = _server(name="My Company Internal Tools", url="https://a.example/mcp", auth_token="Bearer same")
    new_slug = _server(id="my_company_b762e68c1ef1", url="https://b.example/mcp", auth_token="Bearer same")
    world.configure(old_raw, new_slug, _server(id="unique"))
    assert world.manager.server_ids() == ["unique"]
    rows = {row["id"]: row for row in world.manager.status_payload()["servers"]}
    assert rows["my_company_b762e68c1ef1"]["code"] == "MCP_ID_AMBIGUOUS"
    refresh = world.manager.refresh_server("my_company_b762e68c1ef1")
    assert refresh["ok"] is False and refresh["code"] == "MCP_ID_AMBIGUOUS"
    fact = world.manager.resolve_tool_name("mcp_my_company_b762e68c1ef1__list_files")
    assert fact.status == "catalog_unavailable" and "MCP_ID_AMBIGUOUS" in fact.detail
    assert "echo(unique/list_files)" in world.registry.execute("mcp_unique__list_files", {})
    assert [call[0] for call in world.transport.call_calls] == ["unique"]
    assert set(world.transport.list_calls) == {"unique"}


@pytest.mark.parametrize("second_token", ["Bearer same-token", "Bearer other-token"])
def test_a_masked_secret_of_an_ambiguous_identity_refuses_the_save(second_token):
    current = [
        _server(name="My Company Internal Tools", url="https://a.example/mcp", auth_token="Bearer same-token"),
        _server(id="my_company_b762e68c1ef1", url="https://b.example/mcp", auth_token=second_token),
    ]
    with pytest.raises(MCPSecretIdentityAmbiguous) as caught:
        _rehydrate_mcp_servers_payload(_mask_mcp_servers_payload(current), current)
    assert "same-token" not in str(caught.value) and "other-token" not in str(caught.value)
    # A single edited row with a mask still has two saved claimants.
    with pytest.raises(MCPSecretIdentityAmbiguous):
        _rehydrate_mcp_servers_payload(_mask_mcp_servers_payload(current)[1:], current)


def test_two_incoming_rows_cannot_share_one_saved_secret():
    current = [_server(id="demo", auth_token="Bearer only-one")]
    duplicated = _mask_mcp_servers_payload(current) * 2
    with pytest.raises(MCPSecretIdentityAmbiguous):
        _rehydrate_mcp_servers_payload(duplicated, current)
    # Explicit new values (no mask) are the owner's own input and pass through.
    explicit = [dict(row, auth_token="Bearer typed") for row in duplicated]
    assert [row["auth_token"] for row in _rehydrate_mcp_servers_payload(explicit, current)] == ["Bearer typed"] * 2


def test_a_saved_server_without_an_explicit_id_keeps_its_masked_token():
    current = [_server(name="My Company Internal Tools", auth_token="Bearer name-only-secret")]
    restored = _rehydrate_mcp_servers_payload(_mask_mcp_servers_payload(current), current)
    assert restored[0]["auth_token"] == "Bearer name-only-secret"


def test_real_settings_save_refuses_the_ambiguous_mask_and_writes_nothing(settings_client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)
    client, on_disk, saved = settings_client
    stored = [
        _server(name="My Company Internal Tools", url="https://a.example/mcp", auth_token="Bearer a-secret"),
        _server(id="my_company_b762e68c1ef1", url="https://b.example/mcp", auth_token="Bearer b-secret"),
    ]
    on_disk.update(MCP_ENABLED=False, MCP_SERVERS=copy.deepcopy(stored))
    served = client.get("/api/settings").json()["MCP_SERVERS"]
    response = client.post("/api/settings", json={"MCP_SERVERS": served})
    assert response.status_code == 409
    assert response.json()["saved"] is False and response.json()["code"] == "MCP_ID_AMBIGUOUS_SECRET"
    assert saved == {} and on_disk["MCP_SERVERS"] == stored
    assert "a-secret" not in response.text and "b-secret" not in response.text


@pytest.fixture
def test_endpoint(monkeypatch):
    from ouroboros.gateway import mcp

    saved = [_server(name="My Company Internal Tools", auth_token="Bearer saved-secret")]
    seen: list = []
    monkeypatch.setattr(mcp, "_ensure_configured", lambda: None)
    monkeypatch.setattr(mcp, "load_settings", lambda: {"MCP_SERVERS": copy.deepcopy(saved)})
    monkeypatch.setattr(mcp, "get_manager", lambda: SimpleNamespace(
        test_server=lambda candidate, *, settings: seen.append(candidate) or {"ok": True, "tool_count": 0}))
    with TestClient(Starlette(routes=[Route("/test", mcp.api_mcp_test, methods=["POST"])])) as client:
        yield client, saved, seen


def test_saved_server_test_finds_an_id_less_entry_by_its_loader_identity(test_endpoint):
    client, saved, seen = test_endpoint
    response = client.post("/test", json={"server_id": "my_company_b762e68c1ef1"})
    assert response.status_code == 200 and response.json()["ok"]
    assert seen[-1]["auth_token"] == "Bearer saved-secret"


def test_saved_server_test_refuses_an_ambiguous_identity(test_endpoint):
    client, saved, seen = test_endpoint
    saved.append(_server(id="my_company_b762e68c1ef1", auth_token="Bearer other"))
    response = client.post("/test", json={"server_id": "my_company_b762e68c1ef1"})
    assert response.status_code == 409 and response.json()["code"] == "MCP_ID_AMBIGUOUS"
    assert seen == []


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    configure = mcp_client.reconfigure_from_settings
    def configured(settings):
        monkeypatch.setattr(config, "load_settings", lambda: settings)
        configure(settings)
    monkeypatch.setattr(mcp_client, "reconfigure_from_settings", configured)
