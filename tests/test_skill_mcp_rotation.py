"""Granted settings rotation: the normal lister publishes a new catalog for existing workers."""
from __future__ import annotations

import json
import queue
import types
import threading

import pytest

from ouroboros import config, mcp_client, skill_mcp


def _write(path, document):
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


@pytest.fixture()
def installed(tmp_path, monkeypatch):
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state, save_skill_grants

    data = tmp_path / "data"
    skill_dir = data / "skills" / "external" / "weather"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: weather\ndescription: d\nversion: 1.0.0\ntype: script\nruntime: node\nscripts: [setup.mjs]\n"
        "env_from_settings: [WEATHER_API_KEY]\n"
        'mcp_servers: [{"id": "weather", "entry": "scripts/server.mjs"}]\n---\nbody\n', encoding="utf-8")
    (skill_dir / "scripts" / "server.mjs").write_text("// server\n", encoding="utf-8")
    (skill_dir / "scripts" / "setup.mjs").write_text("// setup\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "no-repo"))
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr("ouroboros.tools.skill_exec.load_settings", config.load_settings)  # never a stale binding
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda runtime: ("/opt/bin/" + runtime, ""))
    loaded = load_skill(skill_dir, data)
    save_review_state(data, "weather", SkillReviewState(status="clean", content_hash=loaded.content_hash))
    row = {"id": "weather", "name": "Weather", "enabled": True, "transport": "stdio", "skill": "weather",
           "task_session": False, "browser_policy": False}
    # The key is saved first (a custom key is requestable once present), then granted, as the owner would.
    _write(data / "settings.json", {"MCP_ENABLED": True, "MCP_SERVERS": [row], "WEATHER_API_KEY": "key-one"})
    save_skill_grants(data, "weather", ["WEATHER_API_KEY"], content_hash=loaded.content_hash,
                      requested_keys=["WEATHER_API_KEY"])
    save_enabled(data, "weather", True, actor="owner_ui")
    return types.SimpleNamespace(data=data, row=row, skill_dir=skill_dir, revision=loaded.content_hash)


def _manager(listed):
    manager = mcp_client.MCPManager()

    async def list_tools(cfg, _timeout):
        listed.append(cfg)
        return [{"name": "forecast", "description": "forecast", "inputSchema": {"type": "object"}}]

    async def call_tool(cfg, name, _args, _timeout):
        return mcp_client.ToolResult(status="ok", code="OK", text="called")

    manager._async_list_tools, manager._async_call_tool = list_tools, call_tool
    return manager


@pytest.mark.parametrize("trigger", ["mtime", "settings_save", "worker_report"])
def test_granted_value_rotation_reaches_worker_without_manual_refresh(installed, monkeypatch, trigger):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.mcp_task_session import session_identity
    from ouroboros.gateway.settings import _apply_settings_save_side_effects
    from supervisor.events import dispatch_event

    listed, retired = [], []
    lister, worker = _manager(listed), _manager([])
    monkeypatch.setattr("ouroboros.mcp_task_session.retire_changed", lambda configs: retired.extend(configs))
    lister.become_lister()
    lister.configuration_changed(config.load_settings(), reason="startup")
    assert lister.discovery.wait(5)
    old_cfg = lister.server_config("weather")
    worker.reconfigure(config.load_settings(), settings_mtime_ns=config.SETTINGS_PATH.stat().st_mtime_ns)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(mcp_client, "get_manager", lambda: worker)
    registry = ToolRegistry(repo_dir=installed.data, drive_root=installed.data)
    registry._ctx.event_queue = queue.Queue()
    assert registry.execute_result("mcp_weather__forecast", {}).code == "OK"
    old = config.load_settings()
    _write(config.SETTINGS_PATH, {**old, "WEATHER_API_KEY": "key-two"})

    async def changed_tools(cfg, _timeout):
        listed.append(cfg)
        return [{"name": "forecast_v2", "description": "new credentials, different tools",
                 "inputSchema": {"type": "object"}}]

    lister._async_list_tools = changed_tools
    # A pre-Safety refusal still notifies the one lister; the worker never opens a discovery transport.
    assert registry.execute_result("mcp_weather__forecast", {}).code == "MCP_CATALOG_UNAVAILABLE"
    report = registry._ctx.event_queue.get_nowait()
    assert worker._async_list_tools is not lister._async_list_tools
    monkeypatch.setattr(mcp_client, "get_manager", lambda: lister)
    try:
        if trigger == "mtime":
            mcp_client.ensure_configured_from_settings()
        elif trigger == "settings_save":
            _apply_settings_save_side_effects(types.SimpleNamespace(), config.load_settings(), old, ["WEATHER_API_KEY"])
        else:
            dispatch_event(report, types.SimpleNamespace())
        assert lister.discovery.wait(5)
        assert len(listed) == 2
        new_cfg = lister.server_config("weather")
        assert session_identity(old_cfg) != session_identity(new_cfg)
        assert any(cfg.skill_settings == new_cfg.skill_settings for cfg in retired)
        monkeypatch.setattr(mcp_client, "get_manager", lambda: worker)
        assert registry.execute_result("mcp_weather__forecast_v2", {}).code == "OK"
        assert worker.server_config("weather").skill_settings == listed[-1].skill_settings
        assert worker.resolve_tool_name("mcp_weather__forecast").status != "callable"
    finally:
        assert lister.discovery.wait(5)


@pytest.mark.parametrize("gate", ["ready", "disabled", "mcp_disabled", "revoked", "unreviewed"])
def test_rotation_does_not_relist_other_servers_or_unready_binding(installed, monkeypatch, gate):
    from ouroboros.skill_loader import save_enabled, save_skill_grants, save_review_state, SkillReviewState

    plain = {"id": "manual", "enabled": True, "transport": "stdio", "command": "unused"}
    remote = {"id": "remote", "enabled": True, "url": "https://example.test/mcp"}
    value = {**config.load_settings(), "MCP_SERVERS": [installed.row, plain, remote]}
    _write(config.SETTINGS_PATH, value)
    calls = []
    lister = _manager(calls)
    lister.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: lister)
    lister.configuration_changed(value, reason="startup")
    assert lister.discovery.wait(5)
    assert sorted(cfg.id for cfg in calls) == ["manual", "remote", "weather"]
    calls.clear()
    if gate == "disabled":
        save_enabled(installed.data, "weather", False)
    elif gate == "mcp_disabled":
        value["MCP_ENABLED"] = False
    elif gate == "revoked":
        save_skill_grants(installed.data, "weather", [], content_hash=installed.revision,
                          requested_keys=[])
    elif gate == "unreviewed":
        save_review_state(installed.data, "weather", SkillReviewState(status="pending", content_hash=installed.revision))
    _write(config.SETTINGS_PATH, {**value, "WEATHER_API_KEY": "key-two"})
    try:
        mcp_client.ensure_configured_from_settings()
        assert lister.discovery.wait(5)
        assert [cfg.id for cfg in calls] == (["weather"] if gate == "ready" else [])
    finally:
        assert lister.discovery.wait(5)


def test_inflight_rotation_and_repeated_reads_list_each_identity_once(installed, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    lister = _manager(calls)
    lister.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: lister)
    lister.configuration_changed(config.load_settings(), reason="startup")
    assert lister.discovery.wait(5)

    async def blocked(cfg, _timeout):
        calls.append(cfg)
        if len(calls) == 2:
            entered.set()
            assert release.wait(5)
        return [{"name": "forecast", "inputSchema": {"type": "object"}}]

    lister._async_list_tools = blocked
    _write(config.SETTINGS_PATH, {**config.load_settings(), "WEATHER_API_KEY": "key-two"})
    try:
        mcp_client.ensure_configured_from_settings()
        assert entered.wait(5)
        for _ in range(5):
            mcp_client.ensure_configured_from_settings()
            skill_mcp.lifecycle_settled("weather")
        assert len(calls) == 2
        _write(config.SETTINGS_PATH, {**config.load_settings(), "WEATHER_API_KEY": "key-three"})
        mcp_client.ensure_configured_from_settings()
    finally:
        release.set()
        assert lister.discovery.wait(10)
    assert len(calls) == 3
    assert len({cfg.skill_settings for cfg in calls}) == 3
    from ouroboros.mcp_catalog import catalog_path, identity
    assert json.loads(catalog_path("weather").read_text())["identity"] == identity(calls[-1])
    for _ in range(3):
        mcp_client.ensure_configured_from_settings()
    assert lister.discovery.wait(5) and len(calls) == 3


def test_supervisor_intake_returns_while_owned_binding_is_blocked(installed, monkeypatch):
    from supervisor.events import dispatch_event

    calls = []
    lister = _manager(calls)
    lister.become_lister()
    lister.configuration_changed(config.load_settings(), reason="startup")
    assert lister.discovery.wait(5)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: lister)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    find = skill_mcp._find

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return find(*args)

    monkeypatch.setattr(skill_mcp, "_find", blocked)

    def dispatch():
        dispatch_event({"type": "mcp_configuration_changed", "reason": "skill_review"}, types.SimpleNamespace())
        returned.set()

    caller = threading.Thread(target=dispatch)
    caller.start()
    try:
        assert entered.wait(5)
        assert returned.wait(1), "supervisor event intake must not wait for binding IO"
        assert not lister.discovery.wait(0.01), "the work remains owned and joinable"
    finally:
        release.set()
        caller.join(5)
        assert not caller.is_alive()
        assert lister.discovery.wait(5)
    assert len(calls) == 1


def test_rotated_grant_restarts_live_task_session_with_new_value(installed, monkeypatch):
    import shutil
    from ouroboros import mcp_task_session
    from ouroboros.skill_loader import load_skill, save_review_state, save_skill_grants, SkillReviewState
    from tests.test_mcp_task_session import _registry, owner, call, body, alive, eventually, restarted

    registry, _, _ = _registry(installed.data, monkeypatch)
    log = installed.data / "session.jsonl"
    server = r"""
import { appendFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
let count = 0;
const reply = (id, result) => process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n');
createInterface({ input: process.stdin }).on('line', (line) => {
  const msg = JSON.parse(line);
  if (msg.id === undefined) return;
  if (msg.method === 'initialize') reply(msg.id, { protocolVersion: msg.params.protocolVersion,
    capabilities: { tools: {} }, serverInfo: { name: 'rotation', version: '1' } });
  else if (msg.method === 'tools/list') reply(msg.id, { tools: [{ name: 'bump',
    inputSchema: { type: 'object', properties: {} } }] });
  else if (msg.method === 'tools/call') {
    const value = { pid: process.pid, count: ++count, key: process.env.WEATHER_API_KEY };
    appendFileSync(LOG, JSON.stringify(value) + '\n');
    reply(msg.id, { content: [{ type: 'text', text: JSON.stringify(value) }] });
  } else reply(msg.id, {});
});
"""
    (installed.skill_dir / "scripts" / "server.mjs").write_text("const LOG = " + json.dumps(str(log)) + ";\n" + server)
    manifest = installed.skill_dir / "SKILL.md"
    manifest.write_text(manifest.read_text().replace(
        '"entry": "scripts/server.mjs"', '"entry": "scripts/server.mjs", "task_session": true'))
    revision = load_skill(installed.skill_dir, installed.data).content_hash
    save_review_state(installed.data, "weather", SkillReviewState(status="clean", content_hash=revision))
    save_skill_grants(installed.data, "weather", ["WEATHER_API_KEY"], content_hash=revision,
                      requested_keys=["WEATHER_API_KEY"])
    node = shutil.which("node")
    assert node, "this bounded stdio check requires the provided Node runtime"
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _runtime: (node, ""))
    manager = mcp_client.MCPManager()
    manager.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    try:
        manager.configuration_changed(config.load_settings(), reason="startup")
        assert manager.discovery.wait(20)
        assert not manager.server_config("weather").skill_error, manager.server_config("weather").skill_error
        with owner(registry):
            result = call(registry, "bump", server_id="weather")
            assert result.code == "OK", result.text
            first = body(result)
            assert first["count"] == 1 and alive(first["pid"])
            _write(config.SETTINGS_PATH, {**config.load_settings(), "WEATHER_API_KEY": "key-two"})
            mcp_client.ensure_configured_from_settings()
            assert manager.discovery.wait(20)
            assert eventually(lambda: not alive(first["pid"]))
            replaced = call(registry, "bump", server_id="weather")
            second = body(replaced)
            assert second["pid"] != first["pid"] and second["count"] == 1
            # MCP masks the granted secret in the result. The new key, rather than the old one, is masked.
            assert "key-two" not in replaced.text and "key-one" not in replaced.text
            assert restarted(replaced) == "its MCP configuration changed"
            assert [json.loads(row)["key"] for row in log.read_text().splitlines()] == ["key-one", "key-two"]
    finally:
        mcp_task_session.close_context_sessions(registry._ctx)
        assert manager.discovery.wait(20)


def test_fingerprint_tracks_clearing_declared_values_but_ignores_unrelated_settings(installed):
    before = config.load_settings()
    cleared = {key: value for key, value in before.items() if key != "WEATHER_API_KEY"}
    assert mcp_client.MCPManager._fingerprint(before) != mcp_client.MCPManager._fingerprint(cleared)
    assert mcp_client.MCPManager._fingerprint(before) == mcp_client.MCPManager._fingerprint({
        **before, "UNRELATED_SECRET": "not-delivered"})


@pytest.mark.parametrize("listing_error", [False, True])
def test_rotation_between_binding_and_transport_refuses_then_relists(installed, monkeypatch, listing_error):
    """Real bytes witness both the no-start boundary and the next credential-dependent catalog."""
    import shutil
    from ouroboros import mcp_catalog
    from ouroboros.skill_loader import load_skill, save_review_state, save_skill_grants, SkillReviewState

    log = installed.data / "starts.jsonl"
    server = r"""
import { appendFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
const key = process.env.WEATHER_API_KEY;
appendFileSync(LOG, JSON.stringify({key}) + '\n');
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
createInterface({input: process.stdin}).on('line', line => {
  const msg = JSON.parse(line);
  if (msg.id === undefined) return;
  const result = msg.method === 'initialize'
    ? {protocolVersion: msg.params.protocolVersion, capabilities: {tools: {}},
       serverInfo: {name: 'rotation', version: '1'}, instructions: 'Instructions ' + key}
    : {tools: [{name: key === 'key-one' ? 'forecast' : 'forecast_v2', description: 'Description ' + key,
       inputSchema: {type: 'object', properties: {value: {type: 'string', description: 'Schema ' + key}}}}]};
  if (FAIL && msg.method === 'tools/list' && key === 'key-two') {
    process.stderr.write('Diagnostic ' + key + '\n');
    send({jsonrpc: '2.0', id: msg.id, error: {code: -32603, message: 'Listing failed ' + key}});
  } else send({jsonrpc: '2.0', id: msg.id, result});
});
"""
    (installed.skill_dir / "scripts" / "server.mjs").write_text(
        "const LOG = " + json.dumps(str(log)) + "; const FAIL = " + json.dumps(listing_error) + ";\n" + server)
    revision = load_skill(installed.skill_dir, installed.data).content_hash
    save_review_state(installed.data, "weather", SkillReviewState(status="clean", content_hash=revision))
    save_skill_grants(installed.data, "weather", ["WEATHER_API_KEY"], content_hash=revision,
                      requested_keys=["WEATHER_API_KEY"])
    node = shutil.which("node")
    assert node, "this bounded stdio check requires the provided Node runtime"
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _: (node, ""))
    manager = mcp_client.MCPManager()
    manager.reconfigure(config.load_settings())
    first = manager.refresh_server("weather")
    assert first["ok"] and first["published"], first
    old_cfg = manager.server_config("weather")
    old_identity = mcp_catalog.identity(old_cfg)
    path = mcp_catalog.catalog_path("weather")
    initial_catalog = json.loads(path.read_text())
    assert "key-one" not in path.read_text()
    assert "***" in initial_catalog["instructions"]
    assert "***" in first["tools"][0]["description"]
    run = mcp_client._run_async
    barriers = []

    def rotate_before_start(factory, *args, **kwargs):
        # Same controlled interleaving as the retained original reproduction: binding has finished.
        barriers.append(manager.server_config("weather"))
        _write(config.SETTINGS_PATH, {**config.load_settings(), "WEATHER_API_KEY": "key-two"})
        return run(factory, *args, **kwargs)

    with monkeypatch.context() as barrier:
        barrier.setattr(mcp_client, "_run_async", rotate_before_start)
        refused = manager.refresh_server("weather")
    assert len(barriers) == 1 and mcp_catalog.identity(barriers[0]) == old_identity
    assert not refused["ok"] and "settings changed since it was bound" in refused["error"], refused
    assert [json.loads(line)["key"] for line in log.read_text().splitlines()] == ["key-one"]
    # A failed attempt may persist the old known catalog, never new-key output under its identity.
    failed_catalog = json.loads(path.read_text())
    assert failed_catalog["identity"] == old_identity
    assert failed_catalog["items"] == initial_catalog["items"]
    assert "key-two" not in path.read_text() + json.dumps(refused)
    assert mcp_catalog.identity(old_cfg) == old_identity and "key-two" not in old_cfg.secret_values

    # A separate normal refresh rebinds; no automatic replay of the refused start.
    next_result = manager.refresh_server("weather")
    new_cfg = manager.server_config("weather")
    assert mcp_catalog.identity(new_cfg) != old_identity
    assert json.loads(path.read_text())["identity"] == mcp_catalog.identity(new_cfg)
    assert next_result["published"]
    assert [json.loads(line)["key"] for line in log.read_text().splitlines()] == ["key-one", "key-two"]
    assert "key-two" not in path.read_text() + json.dumps(next_result)
    if listing_error:
        assert not next_result["ok"] and "Listing failed ***" in next_result["error"]
        assert not next_result["tools"]  # old identity's tool list is not usable with the new key
    else:
        assert next_result["ok"] and next_result["tools"][0]["name"] == "forecast_v2"
        assert "***" in next_result["tools"][0]["description"]
        assert "***" in json.loads(path.read_text())["instructions"]


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("rotated", ["key-two", ""])
def test_transport_rejects_changed_grants_before_launch_and_keeps_one_snapshot(installed, monkeypatch, managed, rotated):
    """All stdio consumers share this boundary, including managed work-copy preparation."""
    from ouroboros import execution_domain, work_domain, work_skill_payloads, mcp_catalog

    monkeypatch.setattr(work_domain, "managed", lambda: managed)
    monkeypatch.setattr(work_domain, "is_work", lambda _: True)
    monkeypatch.setattr(execution_domain, "current", lambda: None)
    cfg = skill_mcp.bind_config(mcp_client.normalize_server_config(installed.row))
    assert not cfg.skill_error
    identity = mcp_catalog.identity(cfg)
    copies, attached = [], []

    def copy_launch(_root, _skill, _revision, argv, env, _runtime):
        copies.append(dict(env))
        return argv, str(installed.skill_dir), env

    monkeypatch.setattr(work_skill_payloads, "launch", copy_launch)
    monkeypatch.setattr(execution_domain, "attached_launch", lambda *args: attached.append(args))
    monkeypatch.setattr(mcp_client, "_stdio_with_diagnostics", lambda params, *_: params)
    _write(config.SETTINGS_PATH, {**config.load_settings(), "WEATHER_API_KEY": rotated})
    with pytest.raises(work_domain.WorkDomainRefused, match="settings changed since it was bound") as refused:
        mcp_client._transport_factory(cfg)
    assert refused.value.mcp_unsent_joined is True
    assert not copies and not attached
    assert mcp_catalog.identity(cfg) == identity

    # Rebinding permits rotation/clearing. A later save cannot replace the checked launch snapshot.
    rebound = skill_mcp.bind_config(mcp_client.normalize_server_config(installed.row))
    assert not rebound.skill_error and mcp_catalog.identity(rebound) != identity
    skill_env = skill_mcp._skill_env
    reads = []

    def rotate_after_read(*args, **kwargs):
        snapshot = skill_env(*args, **kwargs)
        reads.append(snapshot)
        _write(config.SETTINGS_PATH, {**config.load_settings(), "WEATHER_API_KEY": "key-three"})
        return snapshot

    monkeypatch.setattr(skill_mcp, "_skill_env", rotate_after_read)
    params = mcp_client._transport_factory(rebound, {"OUROBOROS_BROWSER_GUARD": "test-guard"})
    assert len(reads) == 1 and len(attached) == 1 and len(copies) == int(managed)
    assert params.env.get("WEATHER_API_KEY", "") == rotated
    assert params.env["OUROBOROS_BROWSER_GUARD"] == "test-guard"
    assert "key-three" not in params.env.values()
