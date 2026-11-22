"""Broken capabilities reach a work: once, dated, in its start notice, later as one tail row per change."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import capability_breakage as breakage
from ouroboros import config, extension_health, mcp_client
from ouroboros.tool_policy import format_capability_omissions
from ouroboros.tools.tool_result import ToolResult
from tests._live_world import host_world as host_world  # noqa: F401 -- a real call host for its tool round


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
    since = json.loads((tmp_path / "state" / "mcp" / "calls" / "server-demo.json").read_text(
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
        {"surface": "mcp", "reason": "configuration", "error": "disabled in Extensions → MCP servers: idle"}]
    manager.reconfigure({"MCP_ENABLED": False, "MCP_SERVERS": [{"id": "demo", "enabled": True,
                                                                 "url": "https://demo.example.test/mcp"}]})
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)
    assert breakage.start_notice_rows(_work(tmp_path), []) == [{"surface": "mcp", "reason": "configuration", "error": (
        "MCP is off in Extensions → MCP servers; its 1 configured servers are not offered")}]
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
        "desk: not live on the server (companion_restart_exhausted:svc) since 2026-11-21T09:00:00+00:00; last live "
        "2026-11-20T08:00:00+00:00")}
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
                                                                 "error": "stale: enabled but not live on the server (review_stale) since: unknown"}]


def _round(work):
    """One round of a work whose start recorded what it was told: the change row's text, or None."""
    ctx = SimpleNamespace(messages=[], tools=SimpleNamespace(_ctx=work))
    breakage.append_change_row(ctx)
    return ctx.messages[0]["content"] if ctx.messages else None


def _relist(manager, error=None):
    async def listing(cfg, timeout):
        if error:
            raise ConnectionError(error)
        return [{"name": "ping", "input_schema": {"type": "object"}}]
    manager._async_list_tools = listing
    manager.refresh_server("demo")


def test_a_failing_listing_does_not_hide_answered_calls_and_each_server_keeps_one_line(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, ("demo", True))
    _call(manager, "demo", TimeoutError("slow"))
    _relist(manager, "the listing timed out")
    work = _work(tmp_path)
    [row] = breakage.start_notice_rows(work, [])
    assert "listing failing since" in row["error"] and "calls not answered since" in row["error"]
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="pong"))  # the kept catalog still serves a call
    change = _round(work)
    assert "listing failing since" in change and "calls answered (changed at" in change  # one line, both facts
    assert _round(work) is None
    omissions = [{"surface": "mcp", "reason": "server_no_tools",
                  "servers": [{"id": "demo", "last_error": ""}, {"id": "calm", "last_error": ""}]}]
    lines = format_capability_omissions(breakage.start_notice_rows(_work(tmp_path), omissions))
    assert sum("demo" in line for line in lines) == 1 and "demo: no tools listed; broken: listing failing" in lines[1]


def test_a_worker_that_cannot_run_a_skill_is_named_and_a_server_failure_keeps_its_start_in_utc(monkeypatch, tmp_path):
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: SimpleNamespace(enabled=True))
    times = iter(["2026-11-22T09:00:00+00:00", "2026-11-22T09:30:00+00:00"])
    monkeypatch.setattr(extension_health, "utc_now_iso", lambda: next(times))
    for _seen in range(2):  # a load failure seen at 09:00 and again at 09:30 started at 09:00
        extension_health.record_extension_health(tmp_path, "desk", status=extension_health.BROKEN, reason="load_error")
    pad = extension_health.health_path(tmp_path, "pad")
    live = {"status": "live", "reason": "ready", "ts": "2026-11-22T08:00:00+00:00", "process": "server"}
    pad.write_text(json.dumps({"_schema_version": 1, "skill": "pad", "status": "live", "last_observed": live,
        "observations": {"server": live, "worker": {"status": "inactive", "reason": "deps_missing",
                                                    "ts": "2026-11-22T12:00:00+03:00", "process": "worker"}}}),
                   encoding="utf-8")
    assert [row["error"] for row in breakage.start_notice_rows(_work(tmp_path), [])] == [
        "desk: not live on the server (load_error) since 2026-11-22T09:00:00+00:00",
        "pad: not live in workers (deps_missing) since 2026-11-22T09:00:00+00:00"]  # a worker-only break, in UTC
    pad.write_text(json.dumps({"_schema_version": 1, "skill": "pad", "status": "live", "last_observed": live,
                               "observations": {"server": live}}), encoding="utf-8")
    assert [row["error"].split(":")[0] for row in breakage.start_notice_rows(_work(tmp_path), [])] == ["desk"]


def test_a_workers_break_shows_before_the_servers_first_observation_and_an_empty_record_stays_unknown(
        monkeypatch, tmp_path):
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: SimpleNamespace(enabled=True))

    def record(name, **observations):
        path = extension_health.health_path(tmp_path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_schema_version": 1, "skill": name, "status": "unknown",
                                    "observations": observations}), encoding="utf-8")

    record("desk", worker={"status": "broken", "reason": "load_error", "ts": "2026-11-22T09:00:00+00:00",
                           "process": "worker"})
    record("pad")  # no observation at all yet
    work = _work(tmp_path)
    assert [row["error"] for row in breakage.start_notice_rows(work, [])] == [
        "desk: not live in workers (load_error) since 2026-11-22T09:00:00+00:00"]
    work.__dict__["_capability_breakage_told"]["facts"]["skill:pad"] = "pad: told before"  # what it was told stays
    assert _round(work) is None  # pad is unknown, not repaired


def test_a_lost_skill_binding_and_an_invalid_settings_row_reach_a_running_work_and_so_does_their_repair(
        monkeypatch, tmp_path):
    import dataclasses

    manager = _manager(monkeypatch, ("demo", True))
    work = _work(tmp_path)
    assert breakage.start_notice_rows(work, []) == []
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)  # keep the edited state
    runtime = manager._servers["demo"]
    healthy = runtime.config
    runtime.config = dataclasses.replace(healthy, skill_error=(
        "MCP_SKILL_UNAVAILABLE: server 'demo' runs from skill 'desk', which cannot run it now (not installed)"))
    assert "- MCP server demo: its skill cannot run it now: MCP_SKILL_UNAVAILABLE" in _round(work)
    runtime.config = healthy
    assert "- MCP server demo: no longer reported broken (" in _round(work)
    manager._configuration_errors = [{"id": "files", "enabled": True, "code": "MCP_CONFIG_ERROR",
                                      "last_error": "secret ${MISSING_MCP_SECRET} is not set"}]
    assert "- MCP server files: its settings are invalid (MCP_CONFIG_ERROR): secret ${MISSING_MCP_SECRET}" in _round(work)
    manager._configuration_errors = [{**manager._configuration_errors[0], "enabled": False}]  # the owner's Off
    assert "- MCP server files: no longer reported broken (disabled in Extensions → MCP servers)" in _round(work)
    assert _round(work) is None


def test_an_invalid_settings_row_under_mcp_off_is_off_or_disabled_never_removed(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, ("demo", True))
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)  # keep the edited state
    work = _work(tmp_path)
    breakage.start_notice_rows(work, [])
    manager._configuration_errors = [{"id": "lost", "enabled": True, "code": "MCP_CONFIG_ERROR", "last_error": "no url"}]
    assert "- MCP server lost: its settings are invalid" in _round(work)
    manager._enabled = False  # the owner turns MCP off and that row with it
    manager._configuration_errors = [{**manager._configuration_errors[0], "enabled": False}]
    off = _round(work)
    assert "- MCP server lost: no longer reported broken (disabled in Extensions → MCP servers)" in off
    assert "removed from" not in off and "its 2 configured servers are not offered" in off


def test_an_unavailable_write_scope_stays_broken_through_an_answered_read_until_the_grant_widens(monkeypatch, tmp_path):
    from ouroboros import mcp_oauth

    manager = _manager(monkeypatch, ("demo", True))
    cfg = manager.server_config("demo")
    grant = {"tokens": {"access_token": "a", "scope": "files:read"},
             "sign_in": {"state": "signed_in", "since": "2026-11-22T08:00:00+00:00"}}
    mcp_oauth._save(cfg, grant)
    work = _work(tmp_path)
    assert breakage.start_notice_rows(work, []) == []
    mcp_oauth._save(cfg, {**grant, "required_scope": "files:read files:write",  # a write answered 403 insufficient_scope
                          "required_scope_since": "2026-11-22T10:00:00+00:00"})
    assert ("its sign-in lacks a scope a refused call needs (files:read files:write) since 2026-11-22T10:00:00+00:00"
            in _round(work))
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="read"))  # a read still answers...
    read = _round(work)
    assert "lacks a scope" in read and "calls answered" in read and "no longer reported broken" not in read  # ...not a repair
    mcp_oauth._save(cfg, {**grant, "tokens": {"access_token": "b", "scope": "files:read files:write"}})  # widened
    assert "- MCP server demo: no longer reported broken (a call answered at" in _round(work)
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="written"))
    assert _round(work) is None


def test_a_listing_failure_says_its_reason_and_only_a_changed_reason_reaches_a_running_work(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, ("demo", True))
    work = _work(tmp_path)
    assert breakage.start_notice_rows(work, []) == []
    _relist(manager, "[Errno 8] nodename nor servname provided")
    dns = _round(work)
    assert "listing failing since" in dns and "nodename nor servname provided" in dns
    _relist(manager, "401 Unauthorized: sign in again")
    auth = _round(work)
    assert "401 Unauthorized" in auth and "nodename" not in auth  # the reason changed: one row
    _relist(manager, "401 Unauthorized: sign in again")
    assert _round(work) is None  # the same outcome again: nothing
    _relist(manager)
    assert "- MCP server demo: no longer reported broken (listed at" in _round(work)


def test_an_unreadable_source_is_unknown_not_repaired_and_a_switched_off_or_removed_skill_says_so(monkeypatch, tmp_path):
    enabled = {"desk": True, "gone": True, "tool": True}
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: (
        SimpleNamespace(enabled=enabled[name]) if enabled[name] is not None else None))
    for name in enabled:
        extension_health.record_extension_health(tmp_path, name, status=extension_health.BROKEN, reason="load_error")
    work = _work(tmp_path)
    assert len(breakage.start_notice_rows(work, [])) == 3
    path = extension_health.health_path(tmp_path, "desk")
    good = path.read_bytes()
    path.write_text("{not json", encoding="utf-8")  # unreadable for a moment (a lock, a torn copy)
    assert _round(work) is None  # unknown, never "repaired"
    path.write_bytes(good)
    assert _round(work) is None  # readable again: the same break, nothing new
    enabled.update(desk=False, gone=None)  # the owner switched one off and uninstalled the other
    (path.parent / "enabled.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")
    (extension_health.health_path(tmp_path, "gone").parent / "uninstalled.json").write_text("{}", encoding="utf-8")
    extension_health.record_extension_health(tmp_path, "tool", status=extension_health.INACTIVE, reason="not_extension")
    change = _round(work)
    assert "- skill desk: no longer reported broken (switched off)" in change
    assert "- skill gone: no longer reported broken (uninstalled)" in change
    assert "- skill tool: no longer reported broken (no longer an extension)" in change
    manager = _manager(monkeypatch, ("demo", True))
    _call(manager, "demo", TimeoutError("slow"))
    mcp_work = _work(tmp_path)
    breakage.start_notice_rows(mcp_work, [])
    monkeypatch.setattr(breakage, "_mcp_facts", lambda: (_ for _ in ()).throw(RuntimeError("manager unreadable")))
    assert _round(mcp_work) is None  # a surface whose reader failed is unknown as a whole


@pytest.mark.parametrize("change", ["broke", "recovered", "unchanged"])
def test_a_resumed_work_hears_what_broke_or_recovered_while_it_was_parked(monkeypatch, tmp_path, change):
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from ouroboros.tools.registry import ToolRegistry

    manager = _manager(monkeypatch, ("demo", True))
    if change == "recovered":
        _call(manager, "demo", TimeoutError("slow"))

    def registry():
        made = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        made._ctx._execution_trace, made._ctx._accumulated_usage = {"tool_calls": [], "reasoning_notes": []}, {}
        return made

    parked, messages = registry(), [{"role": "user", "content": "the task"}]
    breakage.write_start_notice(parked, messages)
    state = json.loads(json.dumps(continuation_state(  # the work parks (owner wait, budget pause, checkpoint)
        parked._ctx, messages, parked._ctx._execution_trace, parked._ctx._accumulated_usage, 2, [], set())))
    if change == "broke":
        _call(manager, "demo", TimeoutError("slow"))
    elif change == "recovered":
        _call(manager, "demo", ToolResult(status="ok", code="OK", text="pong"))
    resumed, restored = registry(), [{"role": "user", "content": "the task"}]
    breakage.write_start_notice(resumed, restored)  # the loop's start runs before its restore
    restore_continuation_state(resumed, state, restored, resumed._ctx._execution_trace,
                               resumed._ctx._accumulated_usage, set())
    ctx = SimpleNamespace(messages=restored, tools=resumed)
    assert breakage.append_change_row(ctx) is (change != "unchanged")
    if change != "unchanged":
        expected = "calls not answered since" if change == "broke" else "no longer reported broken (a call answered at"
        assert expected in restored[-1]["content"]
    assert breakage.append_change_row(ctx) is False


def test_main_hears_at_its_next_turn_that_what_it_was_told_broke_recovered_and_another_work_does_not(
        monkeypatch, tmp_path):
    manager = _manager(monkeypatch, ("demo", True))
    _call(manager, "demo", TimeoutError("slow"))

    def turn(task, chat_id=1, depth=0, omissions=()):
        work = SimpleNamespace(drive_root=tmp_path, task_metadata={}, current_chat_id=chat_id, task_id=task,
                               is_direct_chat=True, task_depth=depth)
        return work, breakage.start_notice_rows(work, list(omissions))

    _first, notice = turn("main-1")
    assert "calls not answered since" in notice[0]["error"]
    blocked, notice = turn("main-offline", omissions=[{"surface": "mcp", "reason": "resource_blocked"}])
    assert notice == [{"surface": "mcp", "reason": "resource_blocked"}] and _round(blocked) is None  # unread: unknown
    assert "mcp:demo" in json.loads((tmp_path / "state" / "capability_breakage_main.json").read_text(encoding="utf-8"))[
        "facts"]  # and still told: the next turn that reads it settles it
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="pong"))  # repaired while Main was idle
    for other in (turn("room-1", chat_id=7)[0], turn("child-1", depth=1)[0]):
        assert _round(other) is None  # a room's work, or Main's child, was never told the break (and keeps Main's)
    second, notice = turn("main-2")
    assert notice == []  # its start says nothing is broken now...
    assert "- MCP server demo: no longer reported broken (a call answered at" in _round(second)  # ...its round, the repair
    assert _round(second) is None and _round(turn("main-3")[0]) is None


def test_the_owners_primary_chat_is_main_too_and_a_chat_that_is_neither_keeps_no_map(monkeypatch, tmp_path):
    from supervisor import schedule_notes

    monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda root: 7)  # the owner's first chat was, say, Telegram
    manager = _manager(monkeypatch, ("demo", True))
    _call(manager, "demo", TimeoutError("slow"))

    def wake(task, chat_id):
        work = SimpleNamespace(drive_root=tmp_path, task_metadata={}, current_chat_id=chat_id, task_id=task,
                               is_direct_chat=True, task_depth=0)
        breakage.start_notice_rows(work, [])
        return work

    wake("wake-1", 7)  # told the break
    _call(manager, "demo", ToolResult(status="ok", code="OK", text="pong"))
    assert _round(wake("room-1", 9)) is None  # neither the web chat nor the owner's: it inherits nothing
    assert "- MCP server demo: no longer reported broken (a call answered at" in _round(wake("wake-2", 7))


def test_selecting_a_works_tools_again_never_adds_a_second_typed_notice(monkeypatch, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros.context_budget import host_row_intact
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    manager = _manager(monkeypatch, ("demo", True))
    _call(manager, "demo", TimeoutError("slow"))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    messages = [{"role": "user", "content": "the task"}]
    schemas = initial_tool_schemas(registry)
    loop_mod._setup_dynamic_tools(registry, schemas, messages)
    told = registry._ctx._capability_breakage_told
    loop_mod._setup_dynamic_tools(registry, schemas, messages, context_mode="low")  # the mode is selected again
    notices = [row for row in messages if row.get("_host_context_kind") == breakage.START_KIND]
    assert len(notices) == 1 and host_row_intact(notices[0]) and messages[0] == {"role": "user", "content": "the task"}
    assert notices[0]["content"].startswith("[SYSTEM NOTICE]\n[CAPABILITY_OMISSION_MANIFEST]")
    assert registry._ctx._capability_breakage_told is told  # what the work was told stays its own


def test_a_changed_registry_omission_reaches_a_live_call_when_its_tools_are_selected_again_and_only_then(
        monkeypatch, tmp_path):
    import os

    from ouroboros import loop as loop_mod
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools import github
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh"))
    monkeypatch.setattr(github, "github_cli_configured", lambda: bool(os.environ.get("GITHUB_TOKEN")))
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-test-fixture")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "test-call"
    schemas = initial_tool_schemas(registry, context_mode="max")
    rows = [{"role": "user", "content": "call"}]
    loop_mod._setup_dynamic_tools(registry, schemas, rows)
    starts = [dict(row) for row in rows if row.get("_host_context_kind") == breakage.START_KIND]
    host = SimpleNamespace(tools=registry, ctx=registry._ctx, schemas=schemas, messages=rows, call_tools={},
                           turn=SimpleNamespace(feature={}), _context_changed=lambda: None)

    def changes():
        return [row["content"] for row in rows if row.get("_host_context_kind") == breakage.CHANGE_KIND]

    ConversationHost._reselect_tools(host, "low")
    assert changes() == []  # the same omissions: nothing to tell
    monkeypatch.delenv("GITHUB_TOKEN")  # the credential is revoked during the call
    ConversationHost._reselect_tools(host, "low")
    [changed] = changes()
    assert "submit_skill_to_hub" in changed and "missing_credential" in changed
    ConversationHost._reselect_tools(host, "max")
    assert len(changes()) == 1  # told once
    assert [row for row in rows if row.get("_host_context_kind") == breakage.START_KIND] == starts  # never rebuilt


def test_a_registry_omission_that_is_gone_is_said_to_be_gone_once():
    told = {"facts": {}, "registry": ["[CAPABILITY_OMISSION_MANIFEST]", "- a tool withheld at the start"]}
    registry = SimpleNamespace(_ctx=SimpleNamespace(**{breakage._TOLD: told}), capability_omissions=lambda: [])
    rows = []
    breakage.write_start_notice(registry, rows)
    [row] = rows
    assert row["content"].endswith("\nnone now")  # never a bare header
    breakage.write_start_notice(registry, rows)
    assert len(rows) == 1  # told once


def test_the_round_appends_the_change_row_before_its_facts(monkeypatch, tmp_path):
    from ouroboros import loop as loop_mod
    from ouroboros import loop_messages
    from ouroboros.loop_model_call import _dispatch_round_model

    manager = _manager(monkeypatch, ("demo", True))
    work = _work(tmp_path)
    breakage.start_notice_rows(work, [])
    _call(manager, "demo", TimeoutError("slow"))

    class Reached(Exception):
        pass
    kinds = []

    def facts(ctx, **_kw):
        kinds.append([row.get("_host_context_kind") for row in ctx.messages])
        raise Reached
    monkeypatch.setattr(loop_messages, "append_context_facts", facts)
    monkeypatch.setattr(loop_mod, "_retire_main_reasoning", lambda _ctx: None)
    ctx = SimpleNamespace(messages=[], tools=SimpleNamespace(_ctx=work))
    with pytest.raises(Reached):
        _dispatch_round_model(ctx, None, attempt_cap=None)
    assert kinds == [[breakage.CHANGE_KIND]]  # the change row stands before the round's facts


def test_a_live_call_hears_what_broke_with_its_next_tool_round_once_in_history_and_on_its_connection(
        host_world, monkeypatch):
    import asyncio

    host = host_world.host
    manager = _manager(monkeypatch, ("demo", True))  # configured and answering after the call was told its start

    def round_of(call_id):
        return host.tool_batch([{"id": call_id, "type": "function",
                                 "function": {"name": "list_available_tools", "arguments": "{}"}}])

    async def run():
        calm = await round_of("calm")
        _call(manager, "demo", TimeoutError("slow"))  # breaks while the call goes on
        broke = await round_of("broke")
        return calm, broke, await round_of("after")
    calm, broke, after = asyncio.run(run())
    told = [row for row in host.messages if row.get("_host_context_kind") == breakage.CHANGE_KIND]
    assert len(told) == 1 and "- MCP server demo: calls not answered since" in told[0]["content"]  # history, once
    sent = [row for row in broke if breakage.CHANGE_HEADER in str(row.get("content"))]
    assert [row["content"] for row in sent] == [told[0]["content"]]  # and the connection, with that round's results
    assert not any(breakage.CHANGE_HEADER in str(row.get("content")) for row in [*calm, *after])
