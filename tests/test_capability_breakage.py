"""Broken capabilities reach a work: once, dated, in its start notice, later as one tail row per change."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import capability_breakage as breakage
from ouroboros import config, extension_health, mcp_client
from ouroboros.tool_policy import format_capability_omissions
from ouroboros.tools.tool_result import ToolResult


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(mcp_client, "_manager", None)


def _manager(monkeypatch, *servers):
    value = {"MCP_ENABLED": True, "MCP_SERVERS": [
        {"id": server, "enabled": enabled, "url": f"https://{server}.example.test/mcp"} for server, enabled in servers]}
    config.SETTINGS_PATH.write_text(json.dumps(value), encoding="utf-8")
    from ouroboros.tools import skill_exec

    monkeypatch.setattr(config, "load_settings", lambda: value)
    monkeypatch.setattr(skill_exec, "load_settings", config.load_settings)
    manager = mcp_client.get_manager()
    manager.reconfigure(value, settings_mtime_ns=config.SETTINGS_PATH.stat().st_mtime_ns)

    async def list_tools(cfg, timeout):
        return [{"name": "ping", "input_schema": {"type": "object"}}]
    manager._async_list_tools = list_tools
    for server, enabled in servers:
        if enabled:
            assert manager.refresh_server(server)["published"]
    return manager


def _call(manager, server, outcome):
    async def call(cfg, name, arguments, timeout):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    manager._async_call_tool = call
    return manager._call_tool_result(f"mcp_{server}__ping", {})


def _work(tmp_path):
    return SimpleNamespace(drive_root=tmp_path, task_metadata={})


def test_the_start_notice_dates_a_broken_server_on_its_one_line_and_is_silent_when_all_is_well(monkeypatch, tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    manager = _manager(monkeypatch, ("demo", True), ("calm", True))
    assert breakage.start_notice_rows(_work(tmp_path), []) == []  # all is well: nothing to say
    _call(manager, "demo", TimeoutError("slow"))
    since = json.loads((tmp_path / "state" / "mcp" / "catalogs" / "server-demo.json").read_text(
        encoding="utf-8"))["latest_call"]["since"]
    [row] = breakage.start_notice_rows(_work(tmp_path), [])
    assert row == {"surface": "mcp", "reason": "capability_broken", "error": (
        f"demo: calls not answered since {since} (MCP_TIMEOUT; changed at {since} on ping)")}

    async def offline(cfg, timeout):
        raise TimeoutError("offline")
    manager._async_list_tools = offline
    manager.refresh_server("demo")  # the listing fails too: its provenance row carries both, still one line
    monkeypatch.setattr(mcp_client, "_manager", manager)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.schemas()
    lines = format_capability_omissions(breakage.start_notice_rows(_work(tmp_path), registry.capability_omissions()))
    demo = [line for line in lines if "demo" in line]
    assert len(demo) == 1 and "the latest listing at" in demo[0] and "broken: listing failing since " in demo[0]
    assert "calls not answered since " + since in demo[0] and not any("calm" in line for line in lines)


def test_mcp_configuration_facts_join_the_notice(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, ("demo", True), ("idle", False))
    assert breakage.start_notice_rows(_work(tmp_path), []) == [
        {"surface": "mcp", "reason": "configuration", "error": "disabled in Settings → Tools: idle"}]
    manager.reconfigure({"MCP_ENABLED": False, "MCP_SERVERS": [{"id": "demo", "enabled": True,
                                                                 "url": "https://demo.example.test/mcp"}]})
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)
    assert breakage.start_notice_rows(_work(tmp_path), []) == [{"surface": "mcp", "reason": "configuration", "error": (
        "MCP is off in Settings → Tools; its 1 configured servers are not offered")}]
    blocked = [{"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"}]
    assert breakage.start_notice_rows(_work(tmp_path), blocked) == blocked  # a surface out of reach is not read
    odd = SimpleNamespace(task_metadata=None, budget_drive_root=object())  # an unusable work context
    assert breakage.start_notice_rows(odd, blocked) == blocked


def test_a_skill_not_live_or_missing_a_permission_is_dated_and_a_live_one_is_not(monkeypatch, tmp_path):
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: SimpleNamespace(enabled=True))
    path = extension_health.health_path(tmp_path, "desk")
    path.write_text(json.dumps({"_schema_version": 1, "skill": "desk", "status": "broken", "last_observed": {
        "status": "broken", "reason": "companion_restart_exhausted:svc", "ts": "2026-11-21T10:00:00+00:00"},
        "companion_failures": {"svc": {"ts": "2026-11-21T09:00:00+00:00"}},
        "last_known_good": {"ts": "2026-11-20T08:00:00+00:00"}}), encoding="utf-8")
    [row] = breakage.start_notice_rows(_work(tmp_path), [])
    assert row == {"surface": "skill", "reason": "capability_broken", "error": (
        "desk: not live (companion_restart_exhausted:svc) since 2026-11-21T09:00:00+00:00; last live 2026-11-20T08:00:00+00:00")}
    told = breakage.start_notice_rows(_work(tmp_path), [])
    extension_health.record_extension_health(tmp_path, "desk", status=extension_health.LIVE)
    assert breakage.start_notice_rows(_work(tmp_path), []) == told  # a re-observation of the open failure is no change
    extension_health.record_extension_health(tmp_path, "desk", status=extension_health.LIVE, companions=[])
    assert breakage.start_notice_rows(_work(tmp_path), []) == []  # live, no failure open: nothing to say
    extension_health.record_permission_observation(tmp_path, "desk", refused={
        "permission": "accessibility", "code": "accessibility_untrusted", "repair": "owner"})
    since = extension_health.open_permission_refusals(
        extension_health.read_extension_health(tmp_path, "desk"))["accessibility"]["since"]
    assert breakage.start_notice_rows(_work(tmp_path), []) == [{"surface": "skill", "reason": "capability_broken", "error": (
        f"desk: permission accessibility unavailable since {since} (accessibility_untrusted; repair: owner)")}]
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: SimpleNamespace(enabled=False))
    assert breakage.start_notice_rows(_work(tmp_path), []) != []  # no state file changed: no new lookup
    (path.parent / "enabled.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")  # the owner's Off
    assert breakage.start_notice_rows(_work(tmp_path), []) == []  # an owner-disabled skill is not broken


def test_a_break_and_a_recovery_reach_a_running_work_as_one_tail_row_each_and_an_unchanged_round_adds_none(
        monkeypatch, tmp_path):
    manager = _manager(monkeypatch, ("demo", True))
    work = _work(tmp_path)
    assert breakage.start_notice_rows(work, []) == []
    ctx = SimpleNamespace(messages=[], tools=SimpleNamespace(_ctx=work))
    assert breakage.append_change_row(ctx) is False and ctx.messages == []
    _call(manager, "demo", TimeoutError("slow"))
    assert breakage.append_change_row(ctx) is True
    [row] = ctx.messages
    assert row["_host_context_kind"] == breakage.CHANGE_KIND and row["content"].startswith(breakage.CHANGE_HEADER)
    assert "\n- MCP server demo: calls not answered since " in row["content"]
    assert breakage.append_change_row(ctx) is False and len(ctx.messages) == 1  # the same facts: no row
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="pong"))
    assert breakage.append_change_row(ctx) is True
    assert "\n- MCP server demo: no longer reported broken (a call answered at " in ctx.messages[-1]["content"]
    assert breakage.append_change_row(ctx) is False and len(ctx.messages) == 2
    assert breakage.append_change_row(SimpleNamespace(messages=[], tools=SimpleNamespace(_ctx=_work(tmp_path)))) is False


def test_the_start_notice_row_stays_byte_identical_when_a_fold_rebuilds_the_tool_manifest(monkeypatch, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_discovery import update_resident_notice

    manager = _manager(monkeypatch, ("demo", True))
    _call(manager, "demo", TimeoutError("slow"))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    schemas = initial_tool_schemas(registry)
    messages = [{"role": "user", "content": "the task"}]
    schemas, _extra = loop_mod._setup_dynamic_tools(registry, schemas, messages)
    notices = [row for row in messages if "[CAPABILITY_OMISSION_MANIFEST]" in str(row.get("content"))]
    assert len(notices) == 1 and "demo: calls not answered since " in notices[0]["content"]
    before = json.dumps(notices[0], sort_keys=True)
    update_resident_notice(registry, schemas, messages)
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="pong"))  # the server recovers meanwhile
    update_resident_notice(registry, schemas, messages)  # what an applied fold does to the manifest row
    after = [row for row in messages if "[CAPABILITY_OMISSION_MANIFEST]" in str(row.get("content"))]
    assert [json.dumps(row, sort_keys=True) for row in after] == [before]


def test_a_skill_not_meant_to_run_is_silent_without_a_lookup_and_an_enabled_blocked_one_is_named(monkeypatch, tmp_path):
    def no_lookup(_root, name):
        raise AssertionError(f"looked up {name}")
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", no_lookup)

    def inactive(name, reason):
        extension_health.health_path(tmp_path, name).write_text(json.dumps({
            "_schema_version": 1, "skill": name, "status": "inactive",
            "last_observed": {"status": "inactive", "reason": reason, "ts": "2026-11-21T10:00:00+00:00"}}), encoding="utf-8")
    for name, reason in (("off", "disabled"), ("prose", "not_extension"), ("gone", "missing")):
        inactive(name, reason)
    assert breakage.start_notice_rows(_work(tmp_path), []) == []
    inactive("stale", "review_stale")
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: SimpleNamespace(enabled=True))
    assert breakage.start_notice_rows(_work(tmp_path), []) == [{"surface": "skill", "reason": "capability_broken",
                                                                 "error": "stale: enabled but not live (review_stale) since: unknown"}]
